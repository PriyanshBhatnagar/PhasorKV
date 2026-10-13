// Logits over the PhasorKV latent K cache, sm_120a -- 32 tokens per warp.
//
// latent_logits3.cu's numerics, with each warp covering 32 tokens (two m-tiles) so every rebuild-weight
// fragment, scale word and kbar word feeds two MMAs. A kv head is walked in two halves of 8 n-tiles
// (whole RoPE pairs, rows in pair order) to keep the accumulator at 2 x 8 n-tiles; after a half the
// rotation and the logits MMA consume it, and the logits of the head are stored after the second half.
//
// Same operand layouts as latent_logits.cu (see there). Differences:
//   * alpha_t * kbar enters the FP4 accumulator as a rank-1 bf16 MMA (alpha in k = 0, 1 of A; kbar / ga
//     as bf16 hi + lo in k = 0, 1 of B), so the epilogue is the ga scale, the rotation and the bf16 pack.
//   * the B-fragment scale words are stored once per (n-tile, row) and fetched with a broadcast load.
//   * the query (as RoPE-pair words) sits in shared memory; per-CTA state is small so 4 CTAs fit per SM.
//   * the logits' running max per (sequence, query head) is folded into mx with an atomic max (ordered
//     int encoding), for the PV kernel's softmax.
#include <cstdint>

#define HD 128
#define NT 16
#define REP 4
#ifndef MAXGH
#define MAXGH 8
#endif
#ifndef MINB
#define MINB 4
#endif
#ifndef NW
#define NW 4                  // warps per CTA (32 tokens each)
#endif

__device__ __forceinline__ void mma_fp4(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
  asm(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "r"(sa), "h"((uint16_t)0), "h"((uint16_t)0), "r"(sb), "h"((uint16_t)0), "h"((uint16_t)0));
}
__device__ __forceinline__ void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ int ord(float f) {          // order-preserving float -> int
  const int i = __float_as_int(f);
  return i >= 0 ? i : i ^ 0x7FFFFFFF;
}

extern "C" __global__ void __launch_bounds__(NW * 32, MINB)
latent_logits(const uint16_t* __restrict__ q,        // [B, n_q, HD] bf16 (RoPE applied)
              const uint8_t* __restrict__ kc,       // [B, Tmax, rk/2]
              const uint8_t* __restrict__ ks,       // [B, Tmax, rk/16]
              const uint16_t* __restrict__ al,      // [B, Tmax] bf16 alpha
              const int* __restrict__ seqlen,       // [B]
              const uint2* __restrict__ afrag,      // per kv head [steps][NT][32 lanes] (b0, b1)
              const uint32_t* __restrict__ sfrag,   // per kv head [steps][NT][8 rows] scale word
              const int* __restrict__ head_i,       // [n_kv][4]: latent offset, rank, afrag off, sfrag off
              const float* __restrict__ head_f,     // [n_kv] ga
              const uint32_t* __restrict__ kbw,     // [n_kv][NT][8]: bf16x2 (hi, lo) of kbar / ga
              const uint16_t* __restrict__ cosp,    // [P][64] bf16, column tq * 16 + nt = pair 4 nt + tq
              const uint16_t* __restrict__ sinp,
              float* __restrict__ lg,               // [B, n_q, width]: col0 + t
              int* __restrict__ mx,                 // [B, n_q] ordered-int running max (or null)
              int t_max, int pos0, int col0, float qk_scale, int gh,
              long long s_qb, long long s_kcb, long long s_kct, long long s_ksb, long long s_kst,
              long long s_alb, long long s_lb, long long s_lh) {
  __shared__ uint32_t qsm[MAXGH][REP][64];
  __shared__ int hsm[MAXGH][4];
  __shared__ float hfm[MAXGH];
  constexpr int TPC = 32 * NW;                    // tokens per CTA
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, tq = lane & 3;
  const int b = blockIdx.y, pg = blockIdx.z;
  const int T = seqlen[b];
  const float NEG = -__int_as_float(0x7f800000);
  if (blockIdx.x * TPC >= T) {
    for (int i = tid; i < TPC * gh * REP; i += 32 * NW) {
      int t = blockIdx.x * TPC + (i % TPC), h = pg * gh * REP + i / TPC;
      if (t < t_max) lg[b * s_lb + h * s_lh + col0 + t] = NEG;
    }
    return;
  }
  for (int i = tid; i < gh * REP * 64; i += 32 * NW) {
    const int gi = i >> 8, j = (i >> 6) & 3, pp = i & 63;
    const uint16_t* qh = q + b * s_qb + (long long)((pg * gh + gi) * REP + j) * HD;
    qsm[gi][j][pp] = (uint32_t)qh[pp] | ((uint32_t)qh[pp + 64] << 16);
  }
  if (tid < gh * 4) hsm[tid >> 2][tid & 3] = head_i[(pg * gh) * 4 + tid];
  if (tid < gh) hfm[tid] = head_f[pg * gh + tid];

  // the thread's four tokens: m-tile m in {0, 1}, rows g (r = 2m) and g + 8 (r = 2m + 1)
  int tk[4];
  bool vk[4];
  uint32_t ra[4];                                  // rank-1 A words: alpha at k = 0, 1 (lanes tq == 0)
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    tk[r] = blockIdx.x * TPC + warp * 32 + (r >> 1) * 16 + g + (r & 1) * 8;
    vk[r] = tk[r] < T;
    const uint32_t a = (tq == 0 && vk[r]) ? (uint32_t)al[b * s_alb + tk[r]] : 0u;
    ra[r] = a | (a << 16);
  }
  const uint8_t* kcr[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) kcr[r] = kc + b * s_kcb + (long long)(vk[r] ? tk[r] : 0) * s_kct + tq * 4;
  // scale rows: lanes tq == 0 serve rows g, tq == 1 rows g + 8 (one per m-tile)
  const uint8_t* ksr[2];
  bool sok[2];
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    const int r = 2 * m + (tq == 1 ? 1 : 0);
    sok[m] = tq < 2 && vk[r];
    ksr[m] = ks + b * s_ksb + (long long)(vk[r] ? tk[r] : 0) * s_kst;
  }
  __syncthreads();

  float lo[2][4];
#pragma unroll
  for (int m = 0; m < 2; ++m) lo[m][0] = lo[m][1] = lo[m][2] = lo[m][3] = 0.f;

  for (int gi = 0; gi < gh; ++gi) {
    const int kvh = pg * gh + gi;
    const int lat = hsm[gi][0], rk = hsm[gi][1], fo = hsm[gi][2], so = hsm[gi][3];
    const float ga = hfm[gi];
    const int nch = (rk + 63) >> 6;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      float acc[2][8][4];
      const uint32_t* kw = kbw + kvh * NT * 8 + hf * 64 + g;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const uint32_t kb = tq == 0 ? __ldg(kw + n * 8) : 0u;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.f;
          mma_bf16(acc[m][n], ra[2 * m], ra[2 * m + 1], 0u, 0u, kb, 0u);           // alpha * kbar / ga
        }
      }
      for (int c = 0; c < nch; ++c) {
        const int byte0 = (lat >> 1) + c * 32;
        const bool l_ok = c * 64 < rk, h_ok = c * 64 + 32 < rk;
        uint32_t a[2][4], sa[2];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          const int rA = 2 * m, rB = 2 * m + 1;
          a[m][0] = (vk[rA] && l_ok) ? *reinterpret_cast<const uint32_t*>(kcr[rA] + byte0) : 0u;
          a[m][1] = (vk[rB] && l_ok) ? *reinterpret_cast<const uint32_t*>(kcr[rB] + byte0) : 0u;
          a[m][2] = (vk[rA] && h_ok) ? *reinterpret_cast<const uint32_t*>(kcr[rA] + byte0 + 16) : 0u;
          a[m][3] = (vk[rB] && h_ok) ? *reinterpret_cast<const uint32_t*>(kcr[rB] + byte0 + 16) : 0u;
          sa[m] = 0u;
          if (sok[m] && l_ok) {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(ksr[m] + (lat >> 4) + c * 4);
            sa[m] = (uint32_t)p[0] | ((uint32_t)(h_ok ? p[1] : 0) << 16);
          }
        }
        const uint2* bf2 = afrag + fo + ((long long)c * NT + hf * 8) * 32 + lane;
        const uint32_t* sf = sfrag + so + ((long long)c * NT + hf * 8) * 8 + g;
        uint2 bb[8];
        uint32_t sb[8];
#pragma unroll
        for (int n = 0; n < 8; ++n) { bb[n] = __ldg(bf2 + n * 32); sb[n] = __ldg(sf + n * 8); }
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
          for (int m = 0; m < 2; ++m) mma_fp4(acc[m][n], a[m][0], a[m][1], a[m][2], a[m][3], bb[n].x, bb[n].y, sa[m], sb[n]);
      }
      // rotation + logits share of this half
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        uint32_t cw[2][4], sw[2][4];                // rows A / B: cos, sin of this half's 8 pairs (bf16 pairs)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const long long pos = pos0 + (vk[2 * m + r] ? tk[2 * m + r] : 0);
          const uint4 x = *reinterpret_cast<const uint4*>(cosp + pos * 64 + tq * 16 + hf * 8);
          const uint4 y = *reinterpret_cast<const uint4*>(sinp + pos * 64 + tq * 16 + hf * 8);
          cw[r][0] = x.x; cw[r][1] = x.y; cw[r][2] = x.z; cw[r][3] = x.w;
          sw[r][0] = y.x; sw[r][1] = y.y; sw[r][2] = y.z; sw[r][3] = y.w;
        }
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
          uint32_t ar[4];
#pragma unroll
          for (int s = 0; s < 2; ++s) {
            const int n = 2 * kt + s;
            const uint32_t msk = s ? 0xFFFF0000u : 0x0000FFFFu;
            const int sh = s ? 0 : 16;
            const float ca = __uint_as_float((cw[0][kt] & msk) << sh), sa_ = __uint_as_float((sw[0][kt] & msk) << sh);
            const float cb = __uint_as_float((cw[1][kt] & msk) << sh), sb_ = __uint_as_float((sw[1][kt] & msk) << sh);
            const float x0A = acc[m][n][0] * ga, x1A = acc[m][n][1] * ga, x0B = acc[m][n][2] * ga, x1B = acc[m][n][3] * ga;
            ar[2 * s] = pack_bf16(x0A * ca - x1A * sa_, x1A * ca + x0A * sa_);
            ar[2 * s + 1] = pack_bf16(x0B * cb - x1B * sb_, x1B * cb + x0B * sb_);
          }
          const int kg = 4 * hf + kt;
          const uint32_t b0 = g < REP ? qsm[gi][g][8 * kg + tq] : 0u, b1 = g < REP ? qsm[gi][g][8 * kg + 4 + tq] : 0u;
          mma_bf16(lo[m], ar[0], ar[1], ar[2], ar[3], b0, b1);
        }
      }
    }
    // the head's logits: lo[m] = (token row g: heads 2tq, 2tq + 1), (row g + 8: ...)
    if (tq < 2) {
      const int h = kvh * REP + 2 * tq;
      float* o = lg + b * s_lb + (long long)h * s_lh + col0;
      float m0 = NEG, m1 = NEG;
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int tA = tk[2 * m], tB = tk[2 * m + 1];
        const float l0 = vk[2 * m] ? lo[m][0] * qk_scale : NEG, l1 = vk[2 * m] ? lo[m][1] * qk_scale : NEG;
        const float l2 = vk[2 * m + 1] ? lo[m][2] * qk_scale : NEG, l3 = vk[2 * m + 1] ? lo[m][3] * qk_scale : NEG;
        if (tA < t_max) { o[tA] = l0; o[s_lh + tA] = l1; }
        if (tB < t_max) { o[tB] = l2; o[s_lh + tB] = l3; }
        m0 = fmaxf(m0, fmaxf(l0, l2));
        m1 = fmaxf(m1, fmaxf(l1, l3));
      }
      if (mx != nullptr) {
#pragma unroll
        for (int o_ = 4; o_ < 32; o_ <<= 1) {
          m0 = fmaxf(m0, __shfl_xor_sync(0x33333333u, m0, o_));
          m1 = fmaxf(m1, __shfl_xor_sync(0x33333333u, m1, o_));
        }
        if (g == 0) {
          int* mrow = mx + (long long)b * (gridDim.z * gh * REP);
          atomicMax(mrow + h, ord(m0));
          atomicMax(mrow + h + 1, ord(m1));
        }
      }
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) lo[m][0] = lo[m][1] = lo[m][2] = lo[m][3] = 0.f;
  }
}
