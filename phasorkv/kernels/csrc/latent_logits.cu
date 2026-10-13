// Logits over the PhasorKV latent K cache, sm_120a (Blackwell GeForce / RTX PRO).
//
// One CTA = 64 tokens (4 warps x 16) of one sequence, for GH kv heads in turn. The CTA walks the
// heads' 64-wide latent steps as one flat sequence: the rebuild-weight fragments of the next steps
// are staged in shared memory with cp.async (NBUF buffers, shared by the 4 warps) and the next
// step's latent codes are prefetched into registers while the current step's MMAs run; the query
// (as RoPE-pair words) and kbar sit in shared memory for the epilogues. Per warp and kv head:
//   1. k[16 x 128] = codes[16 x r] . A'^T: the block-scaled FP4 MMA (m16n8k64, NVFP4 x NVFP4, E4M3
//      scales per 16, fp32 accumulate), 16 n-tiles over the head dim per 64-wide latent step. The
//      latent codes are the A operand straight from the token-major cache (a byte holds latent
//      columns 2j, 2j+1, i.e. the MMA's nibble order) and a token's four block scales of a step are
//      one 32-bit word, the MMA's scale register. The rebuild weights A' (rows in RoPE-pair order
//      0, 64, 1, 65, ...) are pre-swizzled into B-fragment order.
//   2. the accumulator holds, per thread, the two dims of a RoPE pair side by side: k = k * ga +
//      alpha * kbar and the rotation run in registers.
//   3. rotated k (bf16) is the A operand of a bf16 MMA (m16n8k16) against the REP query heads of the
//      kv head (FlashAttention-2's accumulator-to-operand reuse): logits[16 x REP].
// Register layouts (measured on sm_120a): A a0 (row g, k 8tq..8tq+7), a1 (g+8, same), a2 / a3
// (+32); B b0 (col g, k 8tq..), b1 (+32); D (g, 2tq / 2tq+1), (g+8, ...); scale A: lane 4g row g,
// lane 4g+1 row g+8; scale B: lane 4g col g; byte j = 16-block j of the 64-wide step.
#include <cstdint>

#define HD 128            // head dim
#define NT 16             // n-tiles of 8 over the head dim
#define REP 4             // query heads per kv head

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

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {   // round to nearest even
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

#ifndef MAXGH
#define MAXGH 8
#endif

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

extern "C" __global__ void __launch_bounds__(128)
latent_logits(const uint16_t* __restrict__ q,        // [B, n_q, HD] bf16 (RoPE applied)
              const uint8_t* __restrict__ kc,       // [B, Tmax, rk/2] latent codes
              const uint8_t* __restrict__ ks,       // [B, Tmax, rk/16] E4M3 block scales
              const uint16_t* __restrict__ al,      // [B, Tmax] bf16 alpha
              const int* __restrict__ seqlen,       // [B]
              const uint2* __restrict__ afrag,      // per kv head [steps][NT][32 lanes] (b0, b1)
              const uint32_t* __restrict__ sfrag,   // per kv head [steps][NT][32 lanes] scale word
              const int* __restrict__ head_i,       // [n_kv][4]: latent offset, rank, afrag off, sfrag off
              const float* __restrict__ head_f,     // [n_kv] A' global scale
              const float2* __restrict__ kbar,      // [n_kv][4 tq][NT] (kbar[p], kbar[p + 64]), p = 4 nt + tq
              const uint16_t* __restrict__ cosp,    // [P][64] bf16, column tq * 16 + nt = pair 4 nt + tq
              const uint16_t* __restrict__ sinp,
              float* __restrict__ lg,               // [B, n_q, width]: col0 + t
              int t_max, int pos0, int col0, float qk_scale, int gh,
              long long s_qb, long long s_kcb, long long s_kct, long long s_ksb, long long s_kst,
              long long s_alb, long long s_lb, long long s_lh) {
#ifndef NBUF
#define NBUF 3
#endif
  __shared__ __align__(16) uint2 bsm[NBUF][NT * 32];    // B fragments of NBUF steps (prefetch NBUF - 1 ahead)
  __shared__ __align__(16) uint32_t ssm[NBUF][NT * 32]; // and their scale words
  __shared__ uint32_t qsm[MAXGH][8][64];                // per kv head: query heads (rows 4..7 zero) as pair words
  __shared__ float2 ksm[MAXGH][4][NT];                  // per kv head: kbar in thread order

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, tq = lane & 3;
  const int b = blockIdx.y, pg = blockIdx.z;
  const int T = seqlen[b];
  const int t0 = blockIdx.x * 64 + warp * 16;
  const int tA = t0 + g, tB = t0 + g + 8;
  const float NEG = -__int_as_float(0x7f800000);

  if (blockIdx.x * 64 >= T) {                   // tile past this sequence: -inf logits only
    for (int i = tid; i < 64 * gh * REP; i += 128) {
      int t = blockIdx.x * 64 + (i & 63), h = pg * gh * REP + (i >> 6);
      if (t < t_max) lg[b * s_lb + h * s_lh + col0 + t] = NEG;
    }
    return;
  }
  // query pair words and kbar of this CTA's kv heads
  for (int i = tid; i < gh * 8 * 64; i += 128) {
    const int gi = i >> 9, j = (i >> 6) & 7, pp = i & 63;
    uint32_t w = 0u;
    if (j < REP) {
      const uint16_t* qh = q + b * s_qb + (long long)((pg * gh + gi) * REP + j) * HD;
      w = (uint32_t)qh[pp] | ((uint32_t)qh[pp + 64] << 16);
    }
    qsm[gi][j][pp] = w;
  }
  for (int i = tid; i < gh * 64; i += 128) (&ksm[0][0][0])[i] = kbar[(long long)pg * gh * 64 + i];

  const bool vA = tA < T, vB = tB < T;
  const float alA = vA ? bf(al[b * s_alb + tA]) : 0.f;
  const float alB = vB ? bf(al[b * s_alb + tB]) : 0.f;

  // cos / sin of the thread's 16 RoPE pairs (p = 4 nt + tq) for its two tokens, bf16 pairs
  uint32_t cA[8], sA[8], cB[8], sB[8];
  {
    const uint4* pc = reinterpret_cast<const uint4*>(cosp + (long long)(pos0 + (vA ? tA : 0)) * 64 + tq * 16);
    const uint4* ps = reinterpret_cast<const uint4*>(sinp + (long long)(pos0 + (vA ? tA : 0)) * 64 + tq * 16);
    uint4 x = pc[0], y = pc[1];
    cA[0] = x.x; cA[1] = x.y; cA[2] = x.z; cA[3] = x.w; cA[4] = y.x; cA[5] = y.y; cA[6] = y.z; cA[7] = y.w;
    x = ps[0]; y = ps[1];
    sA[0] = x.x; sA[1] = x.y; sA[2] = x.z; sA[3] = x.w; sA[4] = y.x; sA[5] = y.y; sA[6] = y.z; sA[7] = y.w;
    pc = reinterpret_cast<const uint4*>(cosp + (long long)(pos0 + (vB ? tB : 0)) * 64 + tq * 16);
    ps = reinterpret_cast<const uint4*>(sinp + (long long)(pos0 + (vB ? tB : 0)) * 64 + tq * 16);
    x = pc[0]; y = pc[1];
    cB[0] = x.x; cB[1] = x.y; cB[2] = x.z; cB[3] = x.w; cB[4] = y.x; cB[5] = y.y; cB[6] = y.z; cB[7] = y.w;
    x = ps[0]; y = ps[1];
    sB[0] = x.x; sB[1] = x.y; sB[2] = x.z; sB[3] = x.w; sB[4] = y.x; sB[5] = y.y; sB[6] = y.z; sB[7] = y.w;
  }
  const uint8_t* kcA = kc + b * s_kcb + (long long)(vA ? tA : 0) * s_kct + tq * 4;
  const uint8_t* kcB = kc + b * s_kcb + (long long)(vB ? tB : 0) * s_kct + tq * 4;
  const uint8_t* ksA = ks + b * s_ksb + (long long)(vA ? tA : 0) * s_kst;
  const uint8_t* ksB = ks + b * s_ksb + (long long)(vB ? tB : 0) * s_kst;

  // a step = (head gi, 64-wide latent chunk c); heads with no latent take one all-zero step
  auto nsteps = [&](int gi) { return max(1, (head_i[(pg * gh + gi) * 4 + 1] + 63) >> 6); };
  auto issue_b = [&](int buf, int gi, int c) {   // cp.async one step's fragments: 4 KB + 2 KB
    const int kvh = pg * gh + gi;
    const uint2* src = afrag + head_i[kvh * 4 + 2] + (long long)c * NT * 32;
    const uint32_t* ssrc = sfrag + head_i[kvh * 4 + 3] + (long long)c * NT * 32;
    cp_async16(&bsm[buf][tid * 2], src + tid * 2);
    cp_async16(&bsm[buf][256 + tid * 2], src + 256 + tid * 2);
    cp_async16(&ssm[buf][tid * 4], ssrc + tid * 4);
  };
  auto load_a = [&](int gi, int c, uint32_t* a, uint32_t& sa) {
    const int kvh = pg * gh + gi;
    const int lat = head_i[kvh * 4], rk = head_i[kvh * 4 + 1];
    const bool lo = c * 64 < rk, hi = c * 64 + 32 < rk;
    const int byte0 = (lat >> 1) + c * 32;
    a[0] = (vA && lo) ? *reinterpret_cast<const uint32_t*>(kcA + byte0) : 0u;
    a[1] = (vB && lo) ? *reinterpret_cast<const uint32_t*>(kcB + byte0) : 0u;
    a[2] = (vA && hi) ? *reinterpret_cast<const uint32_t*>(kcA + byte0 + 16) : 0u;
    a[3] = (vB && hi) ? *reinterpret_cast<const uint32_t*>(kcB + byte0 + 16) : 0u;
    sa = 0u;
    const int sbyte = (lat >> 4) + c * 4;
    if (lo && ((tq == 0 && vA) || (tq == 1 && vB))) {
      const uint16_t* p = reinterpret_cast<const uint16_t*>((tq == 0 ? ksA : ksB) + sbyte);
      sa = (uint32_t)p[0] | ((uint32_t)(hi ? p[1] : 0) << 16);
    }
  };

  float acc[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
  // the flat step sequence, NBUF - 1 steps of fragments in flight
  int pf_g = 0, pf_c = 0, pf_ns = nsteps(0);     // next step to prefetch
  auto advance = [&](int& g_, int& c_, int& n_) {
    if (++c_ >= n_) { ++g_; c_ = 0; if (g_ < gh) n_ = nsteps(g_); }
  };
#pragma unroll
  for (int i = 0; i < NBUF - 1; ++i) {
    if (pf_g < gh) { issue_b(i, pf_g, pf_c); advance(pf_g, pf_c, pf_ns); }
    asm volatile("cp.async.commit_group;\n");
  }
  int gi = 0, c = 0, ns = nsteps(0), it = 0;
  uint32_t an[4], san;
  load_a(0, 0, an, san);
  while (true) {
    uint32_t a[4] = {an[0], an[1], an[2], an[3]}, sa = san;
    asm volatile("cp.async.wait_group %0;\n" ::"n"(NBUF - 2));
    __syncthreads();                             // this step's fragments landed; the slot refilled below is free
    if (pf_g < gh) { issue_b((it + NBUF - 1) % NBUF, pf_g, pf_c); advance(pf_g, pf_c, pf_ns); }
    asm volatile("cp.async.commit_group;\n");
    int gn = gi, cn = c, nn = ns;
    advance(gn, cn, nn);
    const bool more = gn < gh;
    if (more) load_a(gn, cn, an, san);           // next step's latent codes in flight
    const int buf = it % NBUF;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const uint2 bb = bsm[buf][nt * 32 + lane];
      mma_fp4(acc[nt], a[0], a[1], a[2], a[3], bb.x, bb.y, sa, ssm[buf][nt * 32 + lane]);
    }
    if (c == ns - 1) {                           // head complete: rotate, score, store
      const int kvh = pg * gh + gi;
      const float ga = head_f[kvh];
      float lo[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int kt = 0; kt < 8; ++kt) {
        uint32_t ar[4];
#pragma unroll
        for (int s = 0; s < 2; ++s) {
          const int nt = 2 * kt + s;
          const float2 kv = ksm[gi][tq][nt];
          const uint32_t cpA = cA[nt >> 1], spA = sA[nt >> 1], cpB = cB[nt >> 1], spB = sB[nt >> 1];
          const float ca = bf((nt & 1) ? cpA >> 16 : cpA & 0xFFFF), sa_ = bf((nt & 1) ? spA >> 16 : spA & 0xFFFF);
          const float cb = bf((nt & 1) ? cpB >> 16 : cpB & 0xFFFF), sb_ = bf((nt & 1) ? spB >> 16 : spB & 0xFFFF);
          const float x0A = fmaf(acc[nt][0], ga, alA * kv.x), x1A = fmaf(acc[nt][1], ga, alA * kv.y);
          const float x0B = fmaf(acc[nt][2], ga, alB * kv.x), x1B = fmaf(acc[nt][3], ga, alB * kv.y);
          ar[2 * s] = pack_bf16(x0A * ca - x1A * sa_, x1A * ca + x0A * sa_);
          ar[2 * s + 1] = pack_bf16(x0B * cb - x1B * sb_, x1B * cb + x0B * sb_);
          acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
        }
        mma_bf16(lo, ar[0], ar[1], ar[2], ar[3], qsm[gi][g][8 * kt + tq], qsm[gi][g][8 * kt + 4 + tq]);
      }
      if (tq < 2) {                              // lo: (token tA, heads 2tq, 2tq + 1), (token tB, ...)
        const int h = kvh * REP + 2 * tq;
        float* o = lg + b * s_lb + (long long)h * s_lh + col0;
        if (tA < t_max) { o[tA] = vA ? lo[0] * qk_scale : NEG; o[s_lh + tA] = vA ? lo[1] * qk_scale : NEG; }
        if (tB < t_max) { o[tB] = vB ? lo[2] * qk_scale : NEG; o[s_lh + tB] = vB ? lo[3] * qk_scale : NEG; }
      }
    }
    if (!more) break;
    gi = gn; c = cn; ns = nn; ++it;
  }
}
