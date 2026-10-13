// Logits over the PhasorKV latent K cache, sm_120a: 32 tokens per warp.
//
// Same math and operand layouts as latent_logits.cu. One CTA = 128 tokens (4 warps x 32) of one
// sequence, for GH kv heads in turn. A warp covers its 32 tokens as two m-tiles and walks each kv
// head in two halves of the head dim; a half (8 n-tiles) holds complete RoPE pairs (rows in pair
// order), so every rebuild-weight fragment feeds two MMAs (one per m-tile) and the accumulator
// stays at 2 x 8 n-tiles. After a half: k = k * ga + alpha * kbar, rotate, and the bf16 MMA against
// the query heads accumulates that half's share of the logits. The fragments of the next 64-wide
// steps are staged in shared memory with cp.async (shared by the 4 warps); the next step's latent
// codes are prefetched into registers.
#include <cstdint>

#define HD 128
#define NT 16
#define NH 8              // n-tiles per half
#define REP 4
#ifndef MAXGH
#define MAXGH 8
#endif
#ifndef NBUF
#define NBUF 3
#endif

__device__ __forceinline__ void mma_fp4(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "r"(sa), "h"((uint16_t)0), "h"((uint16_t)0), "r"(sb), "h"((uint16_t)0), "h"((uint16_t)0));
}
__device__ __forceinline__ void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float bf(uint32_t bits16) { return __uint_as_float(bits16 << 16); }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

extern "C" __global__ void __launch_bounds__(128)
latent_logits(const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ ks,
              const uint16_t* __restrict__ al, const int* __restrict__ seqlen,
              const uint2* __restrict__ afrag, const uint32_t* __restrict__ sfrag,
              const int* __restrict__ head_i, const float* __restrict__ head_f, const float2* __restrict__ kbar,
              const uint16_t* __restrict__ cosp, const uint16_t* __restrict__ sinp, float* __restrict__ lg,
              int t_max, int pos0, int col0, float qk_scale, int gh,
              long long s_qb, long long s_kcb, long long s_kct, long long s_ksb, long long s_kst,
              long long s_alb, long long s_lb, long long s_lh) {
  __shared__ __align__(16) uint2 bsm[NBUF][NH * 32];     // half a step's B fragments (8 n-tiles)
  __shared__ __align__(16) uint32_t ssm[NBUF][NH * 32];
  __shared__ uint32_t qsm[MAXGH][REP][64];
  __shared__ float2 ksm[MAXGH][4][NT];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, tq = lane & 3;
  const int b = blockIdx.y, pg = blockIdx.z;
  const int T = seqlen[b];
  const int tbase = blockIdx.x * 128 + warp * 32;
  const float NEG = -__int_as_float(0x7f800000);
  if (blockIdx.x * 128 >= T) {
    for (int i = tid; i < 128 * gh * REP; i += 128) {
      int t = blockIdx.x * 128 + (i & 127), h = pg * gh * REP + (i >> 7);
      if (t < t_max) lg[b * s_lb + h * s_lh + col0 + t] = NEG;
    }
    return;
  }
  for (int i = tid; i < gh * REP * 64; i += 128) {
    const int gi = i >> 8, j = (i >> 6) & 3, pp = i & 63;
    const uint16_t* qh = q + b * s_qb + (long long)((pg * gh + gi) * REP + j) * HD;
    qsm[gi][j][pp] = (uint32_t)qh[pp] | ((uint32_t)qh[pp + 64] << 16);
  }
  for (int i = tid; i < gh * 64; i += 128) (&ksm[0][0][0])[i] = kbar[(long long)pg * gh * 64 + i];

  // the thread's four tokens: m-tile m, rows g / g + 8
  int tk[4];
  bool vk[4];
  float alk[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    tk[r] = tbase + (r >> 1) * 16 + g + (r & 1) * 8;
    vk[r] = tk[r] < T;
    alk[r] = vk[r] ? bf(al[b * s_alb + tk[r]]) : 0.f;
  }
  const uint8_t* kcr[4];
  const uint8_t* ksr[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    kcr[r] = kc + b * s_kcb + (long long)(vk[r] ? tk[r] : 0) * s_kct + tq * 4;
    ksr[r] = ks + b * s_ksb + (long long)(vk[r] ? tk[r] : 0) * s_kst;
  }

  // the flat sequence of (head, half, 64-wide step): fragments of one half-step = 2 KB + 1 KB
  auto nsteps = [&](int gi) { return max(1, (head_i[(pg * gh + gi) * 4 + 1] + 63) >> 6); };
  auto issue_b = [&](int buf, int gi, int hf, int c) {
    const int kvh = pg * gh + gi;
    const uint2* src = afrag + head_i[kvh * 4 + 2] + ((long long)c * NT + hf * NH) * 32;
    const uint32_t* ssrc = sfrag + head_i[kvh * 4 + 3] + ((long long)c * NT + hf * NH) * 32;
    cp_async16(&bsm[buf][tid * 2], src + tid * 2);
    if (tid < 64) cp_async16(&ssm[buf][tid * 4], ssrc + tid * 4);
  };
  auto load_a = [&](int gi, int c, uint32_t (*a)[4], uint32_t* sa) {
    const int kvh = pg * gh + gi;
    const int lat = head_i[kvh * 4], rk = head_i[kvh * 4 + 1];
    const bool lo = c * 64 < rk, hi = c * 64 + 32 < rk;
    const int byte0 = (lat >> 1) + c * 32, sbyte = (lat >> 4) + c * 4;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const int rA = 2 * m, rB = 2 * m + 1;
      a[m][0] = (vk[rA] && lo) ? *reinterpret_cast<const uint32_t*>(kcr[rA] + byte0) : 0u;
      a[m][1] = (vk[rB] && lo) ? *reinterpret_cast<const uint32_t*>(kcr[rB] + byte0) : 0u;
      a[m][2] = (vk[rA] && hi) ? *reinterpret_cast<const uint32_t*>(kcr[rA] + byte0 + 16) : 0u;
      a[m][3] = (vk[rB] && hi) ? *reinterpret_cast<const uint32_t*>(kcr[rB] + byte0 + 16) : 0u;
      sa[m] = 0u;
      const int rr = tq == 0 ? rA : rB;
      if (lo && tq < 2 && vk[rr]) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(ksr[rr] + sbyte);
        sa[m] = (uint32_t)p[0] | ((uint32_t)(hi ? p[1] : 0) << 16);
      }
    }
  };
  auto advance = [&](int& g_, int& h_, int& c_, int& n_) {
    if (++c_ >= n_) { c_ = 0; if (++h_ == 2) { h_ = 0; ++g_; if (g_ < gh) n_ = nsteps(g_); } }
  };

  float acc[2][NH][4];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int n = 0; n < NH; ++n) acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.f;
  float lo[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

  int pf_g = 0, pf_h = 0, pf_c = 0, pf_n = nsteps(0);
#pragma unroll
  for (int i = 0; i < NBUF - 1; ++i) {
    if (pf_g < gh) { issue_b(i, pf_g, pf_h, pf_c); advance(pf_g, pf_h, pf_c, pf_n); }
    asm volatile("cp.async.commit_group;\n");
  }
  int gi = 0, hf = 0, c = 0, ns = nsteps(0), it = 0;
  uint32_t an[2][4], san[2];
  load_a(0, 0, an, san);
  while (true) {
    uint32_t a[2][4], sa[2];
#pragma unroll
    for (int m = 0; m < 2; ++m) { a[m][0] = an[m][0]; a[m][1] = an[m][1]; a[m][2] = an[m][2]; a[m][3] = an[m][3]; sa[m] = san[m]; }
    asm volatile("cp.async.wait_group %0;\n" ::"n"(NBUF - 2));
    __syncthreads();
    if (pf_g < gh) { issue_b((it + NBUF - 1) % NBUF, pf_g, pf_h, pf_c); advance(pf_g, pf_h, pf_c, pf_n); }
    asm volatile("cp.async.commit_group;\n");
    int gn = gi, hn = hf, cn = c, nn = ns;
    advance(gn, hn, cn, nn);
    const bool more = gn < gh;
    if (more) load_a(gn, cn, an, san);
    const int buf = it % NBUF;
#pragma unroll
    for (int n = 0; n < NH; ++n) {
      const uint2 bb = bsm[buf][n * 32 + lane];
      const uint32_t sb = ssm[buf][n * 32 + lane];
      mma_fp4(acc[0][n], a[0][0], a[0][1], a[0][2], a[0][3], bb.x, bb.y, sa[0], sb);
      mma_fp4(acc[1][n], a[1][0], a[1][1], a[1][2], a[1][3], bb.x, bb.y, sa[1], sb);
    }
    if (c == ns - 1) {                           // half complete: rotate its pairs, add its logits share
      const int kvh = pg * gh + gi;
      const float ga = head_f[kvh];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        // cos / sin of this half's 8 pairs (p = 4 (8 hf + n) + tq) for rows A (g) and B (g + 8)
        uint32_t cs[2][2][4];                    // [row][cos / sin][pairs of n]
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const long long pos = pos0 + (vk[2 * m + r] ? tk[2 * m + r] : 0);
          const uint4 x = *reinterpret_cast<const uint4*>(cosp + pos * 64 + tq * 16 + hf * 8);
          const uint4 y = *reinterpret_cast<const uint4*>(sinp + pos * 64 + tq * 16 + hf * 8);
          cs[r][0][0] = x.x; cs[r][0][1] = x.y; cs[r][0][2] = x.z; cs[r][0][3] = x.w;
          cs[r][1][0] = y.x; cs[r][1][1] = y.y; cs[r][1][2] = y.z; cs[r][1][3] = y.w;
        }
        const float alA = alk[2 * m], alB = alk[2 * m + 1];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {         // k16 tiles of this half: n-tiles 2 kt, 2 kt + 1
          uint32_t ar[4];
#pragma unroll
          for (int s = 0; s < 2; ++s) {
            const int n = 2 * kt + s;
            const float2 kv = ksm[gi][tq][hf * NH + n];
            const float ca = bf((n & 1) ? cs[0][0][n >> 1] >> 16 : cs[0][0][n >> 1] & 0xFFFF);
            const float sA_ = bf((n & 1) ? cs[0][1][n >> 1] >> 16 : cs[0][1][n >> 1] & 0xFFFF);
            const float cb = bf((n & 1) ? cs[1][0][n >> 1] >> 16 : cs[1][0][n >> 1] & 0xFFFF);
            const float sB_ = bf((n & 1) ? cs[1][1][n >> 1] >> 16 : cs[1][1][n >> 1] & 0xFFFF);
            const float x0A = fmaf(acc[m][n][0], ga, alA * kv.x), x1A = fmaf(acc[m][n][1], ga, alA * kv.y);
            const float x0B = fmaf(acc[m][n][2], ga, alB * kv.x), x1B = fmaf(acc[m][n][3], ga, alB * kv.y);
            ar[2 * s] = pack_bf16(x0A * ca - x1A * sA_, x1A * ca + x0A * sA_);
            ar[2 * s + 1] = pack_bf16(x0B * cb - x1B * sB_, x1B * cb + x0B * sB_);
            acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.f;
          }
          // B: query head g, pairs p0 = 4 (8 hf + 2 kt) + tq and p0 + 4 (the k16 tile's two n-tiles)
          const int p0 = 32 * hf + 8 * kt + tq;
          const uint32_t b0 = g < REP ? qsm[gi][g][p0] : 0u, b1 = g < REP ? qsm[gi][g][p0 + 4] : 0u;
          mma_bf16(lo[m], ar[0], ar[1], ar[2], ar[3], b0, b1);
        }
      }
      if (hf == 1) {                             // head complete: store and reset the logits
        if (tq < 2) {
          const int h = kvh * REP + 2 * tq;
          float* o = lg + b * s_lb + (long long)h * s_lh + col0;
#pragma unroll
          for (int m = 0; m < 2; ++m) {
            const int tA = tk[2 * m], tB = tk[2 * m + 1];
            if (tA < t_max) { o[tA] = vk[2 * m] ? lo[m][0] * qk_scale : NEG; o[s_lh + tA] = vk[2 * m] ? lo[m][1] * qk_scale : NEG; }
            if (tB < t_max) { o[tB] = vk[2 * m + 1] ? lo[m][2] * qk_scale : NEG; o[s_lh + tB] = vk[2 * m + 1] ? lo[m][3] * qk_scale : NEG; }
          }
        }
#pragma unroll
        for (int m = 0; m < 2; ++m) lo[m][0] = lo[m][1] = lo[m][2] = lo[m][3] = 0.f;
      }
    }
    if (!more) break;
    gi = gn; hf = hn; c = cn; ns = nn; ++it;
  }
}
