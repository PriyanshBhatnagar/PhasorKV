// Logits over the PhasorKV latent K cache, sm_120a -- lean epilogue, high occupancy.
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
#define NW 4                  // warps per CTA (16 tokens each)
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
              const uint32_t* __restrict__ afrag4,  // WIDEB: per kv head [steps][NT/2][32 lanes][4] (two n-tiles' b0 b1)
              const uint32_t* __restrict__ sfrag4,  // WIDEB: per kv head [steps][NT/4][8 rows][4] (four n-tiles' scales)
              const int* __restrict__ head_i,       // [n_kv][4]: latent offset, rank, afrag off, sfrag off
              const float* __restrict__ head_f,     // [n_kv] ga
              const uint32_t* __restrict__ kbw,     // [n_kv][NT][8]: bf16x2 (hi, lo) of kbar / ga, column 8 nt + g (RANK1)
              const float2* __restrict__ kbar2,     // [n_kv][4 tq][NT] (kbar[p], kbar[p + 64]), p = 4 nt + tq
              const uint16_t* __restrict__ cosp,    // [P][64] bf16, column tq * 16 + nt = pair 4 nt + tq
              const uint16_t* __restrict__ sinp,
              float* __restrict__ lg,               // [B, n_q, width]: col0 + t
              int* __restrict__ mx,                 // [B, n_q] ordered-int running max (or null)
              int t_max, int pos0, int col0, float qk_scale, int gh, int rk_all,
              long long s_qb, long long s_kcb, long long s_kct, long long s_ksb, long long s_kst,
              long long s_alb, long long s_lb, long long s_lh) {
  __shared__ uint32_t qsm[MAXGH][REP][64];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, tq = lane & 3;
  const int b = blockIdx.y, pg = blockIdx.z;
  const int T = seqlen[b];
  const int t0 = blockIdx.x * (16 * NW) + warp * 16;
  const int tA = t0 + g, tB = t0 + g + 8;
  const float NEG = -__int_as_float(0x7f800000);

  if (blockIdx.x * (16 * NW) >= T) {
    for (int i = tid; i < 16 * NW * gh * REP; i += 32 * NW) {
      int t = blockIdx.x * (16 * NW) + (i % (16 * NW)), h = pg * gh * REP + i / (16 * NW);
      if (t < t_max) lg[b * s_lb + h * s_lh + col0 + t] = NEG;
    }
    return;
  }
#ifdef QPAIR
  {                                              // q given as pair words [B, n_q, 64]: 16-byte copies
    const uint4* src = reinterpret_cast<const uint4*>(q) + (long long)b * (s_qb / 8) + (long long)pg * gh * REP * 16;
    uint4* dst = reinterpret_cast<uint4*>(&qsm[0][0][0]);
    for (int i = tid; i < gh * REP * 16; i += 32 * NW) dst[i] = src[i];
  }
#else
  for (int i = tid; i < gh * REP * 64; i += 32 * NW) {
    const int gi = i >> 8, j = (i >> 6) & 3, pp = i & 63;
    const uint16_t* qh = q + b * s_qb + (long long)((pg * gh + gi) * REP + j) * HD;
    qsm[gi][j][pp] = (uint32_t)qh[pp] | ((uint32_t)qh[pp + 64] << 16);
  }
#endif

  const bool vA = tA < T, vB = tB < T;
  // rank-1 A operand: alpha of the two tokens at k = 0 and k = 1 (lanes tq == 0); B carries kbar / ga
  // as bf16 hi + lo there, so the product keeps ~16 bits of kbar
  const uint32_t alA = (tq == 0 && vA) ? (uint32_t)al[b * s_alb + tA] : 0u;
  const uint32_t alB = (tq == 0 && vB) ? (uint32_t)al[b * s_alb + tB] : 0u;
  const uint32_t ra0 = alA | (alA << 16), ra1 = alB | (alB << 16);
  const float alAf = vA ? __uint_as_float((uint32_t)al[b * s_alb + tA] << 16) : 0.f;
  const float alBf = vB ? __uint_as_float((uint32_t)al[b * s_alb + tB] << 16) : 0.f;

#ifdef COS_SMEM
  __shared__ __align__(16) uint32_t csm[NW * 16][2][32];     // per token row: cos (32 words), sin (32 words), table order
  for (int i = tid; i < NW * 16 * 16; i += 32 * NW) {        // 16 uint4 per row: 8 cos + 8 sin
    const int r = i >> 4, k = i & 15, t = blockIdx.x * (16 * NW) + r;
    const uint16_t* src = (k < 8 ? cosp : sinp) + (long long)(pos0 + (t < T ? t : 0)) * 64 + (k & 7) * 8;
    reinterpret_cast<uint4*>(&csm[r][k >> 3][0])[k & 7] = *reinterpret_cast<const uint4*>(src);
  }
#else
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
#endif
  const uint8_t* kcA = kc + b * s_kcb + (long long)(vA ? tA : 0) * s_kct + tq * 4;
  const uint8_t* kcB = kc + b * s_kcb + (long long)(vB ? tB : 0) * s_kct + tq * 4;
  const uint8_t* ksR = ks + b * s_ksb + (long long)((tq == 0 ? vA : vB) ? (tq == 0 ? tA : tB) : 0) * s_kst;
  const bool sOK = (tq == 0 && vA) || (tq == 1 && vB);
#ifdef A_SMEM
  // the tile's latent codes and scales, staged once (cp.async for the codes); the steps read them from
  // shared memory. Rows padded by 16 bytes to spread the banks.
  extern __shared__ __align__(16) uint8_t dsm[];
  const int kstr = rk_all / 2 + 16, sstr = ((rk_all / 16 + 3) & ~3) + 4;
  uint8_t* kcs = dsm;                                       // [NW * 16][kstr]
  uint8_t* kss = dsm + NW * 16 * kstr;                      // [NW * 16][sstr]
  {
    const int t_base = blockIdx.x * (16 * NW);
    const int cpr = rk_all / 32;                            // 16-byte chunks per row
    for (int i = tid; i < NW * 16 * cpr; i += 32 * NW) {
      const int r = i / cpr, o = (i - r * cpr) * 16;
      const int t = t_base + r;
      const uint8_t* src = kc + b * s_kcb + (long long)(t < T ? t : 0) * s_kct + o;
      unsigned d = (unsigned)__cvta_generic_to_shared(kcs + r * kstr + o);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(t < T ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;\n");
    const int spr = rk_all / 32;                            // 2-byte scale pairs per row
    for (int i = tid; i < NW * 16 * spr; i += 32 * NW) {
      const int r = i / spr, o = (i - r * spr) * 2;
      const int t = t_base + r;
      *reinterpret_cast<uint16_t*>(kss + r * sstr + o) =
          t < T ? *reinterpret_cast<const uint16_t*>(ks + b * s_ksb + (long long)t * s_kst + o) : (uint16_t)0;
    }
  }
  const uint8_t* kcAs = kcs + (warp * 16 + g) * kstr + tq * 4;
  const uint8_t* kcBs = kcs + (warp * 16 + g + 8) * kstr + tq * 4;
  const uint8_t* ksRs = kss + (warp * 16 + g + (tq == 1 ? 8 : 0)) * sstr;
#endif
  auto load_a = [&](int lat, int rk, int c, uint32_t* a, uint32_t& sa) {
    const int byte0 = (lat >> 1) + c * 32;
    const bool lo = c * 64 < rk, hi = c * 64 + 32 < rk;
#ifdef A_SMEM
    a[0] = lo ? *reinterpret_cast<const uint32_t*>(kcAs + byte0) : 0u;
    a[1] = lo ? *reinterpret_cast<const uint32_t*>(kcBs + byte0) : 0u;
    a[2] = hi ? *reinterpret_cast<const uint32_t*>(kcAs + byte0 + 16) : 0u;
    a[3] = hi ? *reinterpret_cast<const uint32_t*>(kcBs + byte0 + 16) : 0u;
    sa = 0u;
    if (tq < 2 && lo) {
      const uint16_t* p = reinterpret_cast<const uint16_t*>(ksRs + (lat >> 4) + c * 4);
      sa = (uint32_t)p[0] | ((uint32_t)(hi ? p[1] : 0) << 16);
    }
#else
    a[0] = (vA && lo) ? *reinterpret_cast<const uint32_t*>(kcA + byte0) : 0u;
    a[1] = (vB && lo) ? *reinterpret_cast<const uint32_t*>(kcB + byte0) : 0u;
    a[2] = (vA && hi) ? *reinterpret_cast<const uint32_t*>(kcA + byte0 + 16) : 0u;
    a[3] = (vB && hi) ? *reinterpret_cast<const uint32_t*>(kcB + byte0 + 16) : 0u;
    sa = 0u;
    if (sOK && lo) {
      const uint16_t* p = reinterpret_cast<const uint16_t*>(ksR + (lat >> 4) + c * 4);
      sa = (uint32_t)p[0] | ((uint32_t)(hi ? p[1] : 0) << 16);
    }
#endif
  };
  // head parameters in shared memory
  __shared__ int hsm[MAXGH][4];
  __shared__ float hfm[MAXGH];
  if (tid < gh * 4) hsm[tid >> 2][tid & 3] = head_i[(pg * gh) * 4 + tid];
  if (tid < gh) hfm[tid] = head_f[pg * gh + tid];
#ifdef PREFETCH
  uint32_t pa[4], psa;
  load_a(head_i[pg * gh * 4], head_i[pg * gh * 4 + 1], 0, pa, psa);
#endif
#ifdef A_SMEM
  asm volatile("cp.async.wait_group 0;\n");
#endif
  __syncthreads();

  for (int gi = 0; gi < gh; ++gi) {
    const int kvh = pg * gh + gi;
    const int lat = hsm[gi][0], rk = hsm[gi][1];
    const int fo = hsm[gi][2], so = hsm[gi][3];
    float acc[NT][4];
#ifdef RANK1
    {
      const uint32_t* kw = kbw + kvh * NT * 8 + g;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
#ifndef SKIP_RANK1
        mma_bf16(acc[nt], ra0, ra1, 0u, 0u, tq == 0 ? __ldg(kw + nt * 8) : 0u, 0u);   // alpha * kbar / ga (hi + lo)
#endif
      }
    }
#else
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
#endif
    const int nch = (rk + 63) >> 6;
    for (int c = 0; c < nch; ++c) {
#ifdef PREFETCH
      // this step's codes were fetched one step ahead (the previous step's MMAs, or the previous
      // head's epilogue); fetch the next one now
      const uint32_t a0 = pa[0], a1 = pa[1], a2 = pa[2], a3 = pa[3], sa = psa;
      if (c + 1 < nch) {
        load_a(lat, rk, c + 1, pa, psa);
      } else if (gi + 1 < gh) {
        const int nk = pg * gh + gi + 1;
        load_a(head_i[nk * 4], head_i[nk * 4 + 1], 0, pa, psa);
      }
#else
      uint32_t ax[4], sa;
#ifdef NO_ALOAD
      ax[0] = 0x22222222u ^ (c + lane); ax[1] = ax[0] ^ 1; ax[2] = ax[0] ^ 2; ax[3] = ax[0] ^ 3; sa = 0x38383838u;
#else
      load_a(lat, rk, c, ax, sa);
#endif
      const uint32_t a0 = ax[0], a1 = ax[1], a2 = ax[2], a3 = ax[3];
#endif
      const uint2* bf2 = afrag + fo + (long long)c * NT * 32 + lane;
      const uint32_t* sf = sfrag + so + (long long)c * NT * 8 + g;
      // all 16 fragments in flight before the first MMA (not one load round trip per MMA)
      uint2 bb[NT];
      uint32_t sb[NT];
#ifdef WIDEB
      {                                            // 16-byte loads: fragment pairs, scale quads
        const uint4* bw = reinterpret_cast<const uint4*>(afrag4) + fo / 2 + (long long)c * (NT / 2) * 32 + lane;
        const uint4* sw = reinterpret_cast<const uint4*>(sfrag4) + so / 4 + (long long)c * (NT / 4) * 8 + g;
#pragma unroll
        for (int k = 0; k < NT / 2; ++k) {
          const uint4 v = __ldg(bw + k * 32);
          bb[2 * k] = make_uint2(v.x, v.y); bb[2 * k + 1] = make_uint2(v.z, v.w);
        }
#pragma unroll
        for (int k = 0; k < NT / 4; ++k) {
          const uint4 v = __ldg(sw + k * 8);
          sb[4 * k] = v.x; sb[4 * k + 1] = v.y; sb[4 * k + 2] = v.z; sb[4 * k + 3] = v.w;
        }
      }
#else
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
#ifdef NO_BLOAD
        bb[nt] = make_uint2(0x22222222u ^ (nt + c), 0x22222222u ^ lane); sb[nt] = 0x38383838u;
#else
        bb[nt] = __ldg(bf2 + nt * 32); sb[nt] = __ldg(sf + nt * 8);
#endif
      }
#endif
#ifdef BWAVE
#pragma unroll
      for (int w8 = 0; w8 < 2; ++w8) {
        uint2 bw2[8];
        uint32_t sw2[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) { bw2[k] = __ldg(bf2 + (8 * w8 + k) * 32); sw2[k] = __ldg(sf + (8 * w8 + k) * 8); }
#pragma unroll
        for (int k = 0; k < 8; ++k) mma_fp4(acc[8 * w8 + k], a0, a1, a2, a3, bw2[k].x, bw2[k].y, sa, sw2[k]);
      }
      if (false)
#endif
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
#ifdef NO_OMMA
        acc[nt][0] += __uint_as_float((a0 ^ bb[nt].x ^ sa) & 0x3FFFFFFF); acc[nt][1] += __uint_as_float((a1 ^ bb[nt].y ^ sb[nt]) & 0x3FFFFFFF);
        acc[nt][2] += __uint_as_float((a2 ^ bb[nt].x) & 0x3FFFFFFF); acc[nt][3] += __uint_as_float((a3 ^ bb[nt].y) & 0x3FFFFFFF);
#else
        mma_fp4(acc[nt], a0, a1, a2, a3, bb[nt].x, bb[nt].y, sa, sb[nt]);
#endif
      }
    }
    // scale by ga (so the rotated keys round to bf16 exactly as the reference's do), rotate each RoPE
    // pair, pack to bf16 and score against the REP query heads
    const float ga = hfm[gi];
    const float2* kbf = kbar2 + (long long)kvh * 64 + tq * NT;
    float lo[4] = {0.f, 0.f, 0.f, 0.f};
#ifdef NO_EPI
    for (int nt = 0; nt < NT; ++nt) { lo[0] += acc[nt][0]; lo[1] += acc[nt][1]; lo[2] += acc[nt][2]; lo[3] += acc[nt][3]; }
    if (false)
#endif
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      uint32_t ar[4];
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const int nt = 2 * kt + s;
        const uint32_t cw = nt & 1 ? 0xFFFF0000u : 0x0000FFFFu;
        const int sh = nt & 1 ? 0 : 16;
#ifdef COS_SMEM
        const int rA = warp * 16 + g, rB = rA + 8;
        const uint32_t cwA = csm[rA][0][tq * 8 + kt], swA = csm[rA][1][tq * 8 + kt];
        const uint32_t cwB = csm[rB][0][tq * 8 + kt], swB = csm[rB][1][tq * 8 + kt];
        const float ca = __uint_as_float((cwA & cw) << sh), sa_ = __uint_as_float((swA & cw) << sh);
        const float cb = __uint_as_float((cwB & cw) << sh), sb_ = __uint_as_float((swB & cw) << sh);
#else
        const float ca = __uint_as_float((cA[kt] & cw) << sh), sa_ = __uint_as_float((sA[kt] & cw) << sh);
        const float cb = __uint_as_float((cB[kt] & cw) << sh), sb_ = __uint_as_float((sB[kt] & cw) << sh);
#endif
#ifdef RANK1
        const float x0A = acc[nt][0] * ga, x1A = acc[nt][1] * ga, x0B = acc[nt][2] * ga, x1B = acc[nt][3] * ga;
#else
        const float2 kv = __ldg(kbf + nt);                         // (kbar[p], kbar[p + 64]), p = 4 nt + tq
        const float x0A = fmaf(acc[nt][0], ga, alAf * kv.x), x1A = fmaf(acc[nt][1], ga, alAf * kv.y);
        const float x0B = fmaf(acc[nt][2], ga, alBf * kv.x), x1B = fmaf(acc[nt][3], ga, alBf * kv.y);
#endif
        ar[2 * s] = pack_bf16(x0A * ca - x1A * sa_, x1A * ca + x0A * sa_);
        ar[2 * s + 1] = pack_bf16(x0B * cb - x1B * sb_, x1B * cb + x0B * sb_);
      }
      const uint32_t b0 = g < REP ? qsm[gi][g][8 * kt + tq] : 0u, b1 = g < REP ? qsm[gi][g][8 * kt + 4 + tq] : 0u;
#ifdef SKIP_LMMA
      lo[0] += __uint_as_float(ar[0] ^ b0); lo[1] += __uint_as_float(ar[1] ^ b1);
      lo[2] += __uint_as_float(ar[2] ^ b0); lo[3] += __uint_as_float(ar[3] ^ b1);
#else
      mma_bf16(lo, ar[0], ar[1], ar[2], ar[3], b0, b1);
#endif
    }
    const float sc = qk_scale;
    if (tq < 2) {                                // lo: (token tA, heads 2tq, 2tq + 1), (token tB, ...)
      const int h = kvh * REP + 2 * tq;
      float* o = lg + b * s_lb + (long long)h * s_lh + col0;
      const float l0 = vA ? lo[0] * sc : NEG, l1 = vA ? lo[1] * sc : NEG;
      const float l2 = vB ? lo[2] * sc : NEG, l3 = vB ? lo[3] * sc : NEG;
      if (tA < t_max) { o[tA] = l0; o[s_lh + tA] = l1; }
      if (tB < t_max) { o[tB] = l2; o[s_lh + tB] = l3; }
      if (mx != nullptr) {                       // running max over the 8 rows of this quad column
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
  }
}
