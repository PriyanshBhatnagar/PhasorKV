// Logits over the PhasorKV latent K cache, sm_120a -- head halves in ping-pong.
//
// As latent_logits3.cu, but each kv head is processed as two halves of 8 n-tiles (64 output dims; with
// the rows in RoPE-pair order a half holds whole pairs). The two halves use separate accumulators and
// the code alternates them: the MMAs of the next half are issued before the epilogue (rotation, bf16
// pack, logits MMA) of the current one, so the tensor pipe and the ALUs overlap within a warp, and the
// per-thread state (8 n-tiles of accumulator and fragments) is half of latent_logits3.cu's.
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
#define MINB 3
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

extern "C" __global__ void __launch_bounds__(128, MINB)
latent_logits(const uint16_t* __restrict__ q,        // [B, n_q, HD] bf16 (RoPE applied)
              const uint8_t* __restrict__ kc,       // [B, Tmax, rk/2]
              const uint8_t* __restrict__ ks,       // [B, Tmax, rk/16]
              const uint16_t* __restrict__ al,      // [B, Tmax] bf16 alpha
              const int* __restrict__ seqlen,       // [B]
              const uint2* __restrict__ afrag,      // per kv head [steps][NT][32 lanes] (b0, b1)
              const uint32_t* __restrict__ sfrag,   // per kv head [steps][NT][8 rows] scale word
              const int* __restrict__ head_i,       // [n_kv][4]: latent offset, rank, afrag off, sfrag off
              const float* __restrict__ head_f,     // [n_kv] ga
              const uint32_t* __restrict__ kbw,     // [n_kv][NT][8]: bf16x2 (hi, lo) of kbar / ga, column 8 nt + g (RANK1)
              const float2* __restrict__ kbar2,     // [n_kv][4 tq][NT] (kbar[p], kbar[p + 64]), p = 4 nt + tq
              const uint16_t* __restrict__ cosp,    // [P][64] bf16, column tq * 16 + nt = pair 4 nt + tq
              const uint16_t* __restrict__ sinp,
              float* __restrict__ lg,               // [B, n_q, width]: col0 + t
              int* __restrict__ mx,                 // [B, n_q] ordered-int running max (or null)
              int t_max, int pos0, int col0, float qk_scale, int gh,
              long long s_qb, long long s_kcb, long long s_kct, long long s_ksb, long long s_kst,
              long long s_alb, long long s_lb, long long s_lh) {
  __shared__ uint32_t qsm[MAXGH][REP][64];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, tq = lane & 3;
  const int b = blockIdx.y, pg = blockIdx.z;
  const int T = seqlen[b];
  const int t0 = blockIdx.x * 64 + warp * 16;
  const int tA = t0 + g, tB = t0 + g + 8;
  const float NEG = -__int_as_float(0x7f800000);

  if (blockIdx.x * 64 >= T) {
    for (int i = tid; i < 64 * gh * REP; i += 128) {
      int t = blockIdx.x * 64 + (i & 63), h = pg * gh * REP + (i >> 6);
      if (t < t_max) lg[b * s_lb + h * s_lh + col0 + t] = NEG;
    }
    return;
  }
  for (int i = tid; i < gh * REP * 64; i += 128) {
    const int gi = i >> 8, j = (i >> 6) & 3, pp = i & 63;
    const uint16_t* qh = q + b * s_qb + (long long)((pg * gh + gi) * REP + j) * HD;
    qsm[gi][j][pp] = (uint32_t)qh[pp] | ((uint32_t)qh[pp + 64] << 16);
  }

  const bool vA = tA < T, vB = tB < T;
  // rank-1 A operand: alpha of the two tokens at k = 0 and k = 1 (lanes tq == 0); B carries kbar / ga
  // as bf16 hi + lo there, so the product keeps ~16 bits of kbar
  const uint32_t alA = (tq == 0 && vA) ? (uint32_t)al[b * s_alb + tA] : 0u;
  const uint32_t alB = (tq == 0 && vB) ? (uint32_t)al[b * s_alb + tB] : 0u;
  const uint32_t ra0 = alA | (alA << 16), ra1 = alB | (alB << 16);
  const float alAf = vA ? __uint_as_float((uint32_t)al[b * s_alb + tA] << 16) : 0.f;
  const float alBf = vB ? __uint_as_float((uint32_t)al[b * s_alb + tB] << 16) : 0.f;

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
  const uint8_t* ksR = ks + b * s_ksb + (long long)((tq == 0 ? vA : vB) ? (tq == 0 ? tA : tB) : 0) * s_kst;
  const bool sOK = (tq == 0 && vA) || (tq == 1 && vB);
  auto load_a = [&](int lat, int rk, int c, uint32_t* a, uint32_t& sa) {
    const int byte0 = (lat >> 1) + c * 32;
    const bool lo = c * 64 < rk, hi = c * 64 + 32 < rk;
    a[0] = (vA && lo) ? *reinterpret_cast<const uint32_t*>(kcA + byte0) : 0u;
    a[1] = (vB && lo) ? *reinterpret_cast<const uint32_t*>(kcB + byte0) : 0u;
    a[2] = (vA && hi) ? *reinterpret_cast<const uint32_t*>(kcA + byte0 + 16) : 0u;
    a[3] = (vB && hi) ? *reinterpret_cast<const uint32_t*>(kcB + byte0 + 16) : 0u;
    sa = 0u;
    if (sOK && lo) {
      const uint16_t* p = reinterpret_cast<const uint16_t*>(ksR + (lat >> 4) + c * 4);
      sa = (uint32_t)p[0] | ((uint32_t)(hi ? p[1] : 0) << 16);
    }
  };
  __syncthreads();

  // MMAs of one half (head gi, n-tiles 8 hf .. 8 hf + 7) into acc
  auto compute = [&](int gi, int hf, float (*acc)[4]) {
    const int kvh = pg * gh + gi;
    const int lat = head_i[kvh * 4], rk = head_i[kvh * 4 + 1];
    const int fo = head_i[kvh * 4 + 2], so = head_i[kvh * 4 + 3];
    const uint32_t* kw = kbw + kvh * NT * 8 + hf * 64 + g;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
      mma_bf16(acc[n], ra0, ra1, 0u, 0u, tq == 0 ? __ldg(kw + n * 8) : 0u, 0u);   // alpha * kbar / ga
    }
    const int nch = (rk + 63) >> 6;
    for (int c = 0; c < nch; ++c) {
      uint32_t ax[4], sa;
      load_a(lat, rk, c, ax, sa);
      const uint2* bf2 = afrag + fo + ((long long)c * NT + hf * 8) * 32 + lane;
      const uint32_t* sf = sfrag + so + ((long long)c * NT + hf * 8) * 8 + g;
      uint2 bb[8];
      uint32_t sb[8];
#pragma unroll
      for (int n = 0; n < 8; ++n) { bb[n] = __ldg(bf2 + n * 32); sb[n] = __ldg(sf + n * 8); }
#pragma unroll
      for (int n = 0; n < 8; ++n) mma_fp4(acc[n], ax[0], ax[1], ax[2], ax[3], bb[n].x, bb[n].y, sa, sb[n]);
    }
  };
  // rotation + logits share of one half; after the second half, store the head's logits
  float lo[4] = {0.f, 0.f, 0.f, 0.f};
  auto epilogue = [&](int gi, int hf, float (*acc)[4]) {
    const int kvh = pg * gh + gi;
    const float ga = head_f[kvh];
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      uint32_t ar[4];
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const int n = 2 * kt + s, nt = 8 * hf + n;
        const uint32_t cw = n & 1 ? 0xFFFF0000u : 0x0000FFFFu;
        const int sh = n & 1 ? 0 : 16;
        const int w = nt >> 1;
        const float ca = __uint_as_float((cA[w] & cw) << sh), sa_ = __uint_as_float((sA[w] & cw) << sh);
        const float cb = __uint_as_float((cB[w] & cw) << sh), sb_ = __uint_as_float((sB[w] & cw) << sh);
        const float x0A = acc[n][0] * ga, x1A = acc[n][1] * ga, x0B = acc[n][2] * ga, x1B = acc[n][3] * ga;
        ar[2 * s] = pack_bf16(x0A * ca - x1A * sa_, x1A * ca + x0A * sa_);
        ar[2 * s + 1] = pack_bf16(x0B * cb - x1B * sb_, x1B * cb + x0B * sb_);
      }
      const int kg = 4 * hf + kt;
      const uint32_t b0 = g < REP ? qsm[gi][g][8 * kg + tq] : 0u, b1 = g < REP ? qsm[gi][g][8 * kg + 4 + tq] : 0u;
      mma_bf16(lo, ar[0], ar[1], ar[2], ar[3], b0, b1);
    }
    if (hf == 1) {
      if (tq < 2) {                              // lo: (token tA, heads 2tq, 2tq + 1), (token tB, ...)
        const int h = kvh * REP + 2 * tq;
        float* o = lg + b * s_lb + (long long)h * s_lh + col0;
        const float l0 = vA ? lo[0] * qk_scale : NEG, l1 = vA ? lo[1] * qk_scale : NEG;
        const float l2 = vB ? lo[2] * qk_scale : NEG, l3 = vB ? lo[3] * qk_scale : NEG;
        if (tA < t_max) { o[tA] = l0; o[s_lh + tA] = l1; }
        if (tB < t_max) { o[tB] = l2; o[s_lh + tB] = l3; }
        if (mx != nullptr) {
          float m0 = fmaxf(l0, l2), m1 = fmaxf(l1, l3);
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
      lo[0] = lo[1] = lo[2] = lo[3] = 0.f;
    }
  };

  float acc0[8][4], acc1[8][4];
  compute(0, 0, acc0);
  for (int gi = 0; gi < gh; ++gi) {
    compute(gi, 1, acc1);                        // issued ahead of the first half's epilogue
    epilogue(gi, 0, acc0);
    if (gi + 1 < gh) compute(gi + 1, 0, acc0);   // ahead of the second half's epilogue
    epilogue(gi, 1, acc1);
  }
}
