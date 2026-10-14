// Absorbed P . V over the PhasorKV latent V cache with the rest tier packed to 2 bits, softmax fused.
// sm_120a.
//
// One CTA (16 warps) = one token segment of one sequence, all 32 query heads, all latent V columns:
//   z[b, h, c] += sum_t exp(l[b, h, t] - max) * s_tier(c)[t] * v[t, c]
//   lsum[b, h] += sum_t exp(l - max),   beta[b, h] += sum_t exp(l - max) * alpha[t]
// V per token: the top tier (3-bit levels in E2M1 nibbles; nb_t bytes, a multiple of 8: byte j holds
// columns j and nb_t + j) then the rest tier (2-bit codes sign | magnitude for +-0.5 / +-1.5; nq bytes,
// a multiple of 8: byte j holds columns j, nq + j, 2 nq + j, 3 nq + j in bits 0-1, 2-3, 4-5, 6-7).
// z is kept in the padded column space [top: 2 nb_t][rest: 4 nq]; the lift's weights carry the same
// padding. A thread gathers the byte of its column for its four tokens of an mma k16 step and turns
// each nibble / 2-bit field group into f16 B fragments with one byte_perm table lookup.
#include <cstdint>
#include <cuda_fp16.h>

#define NQ 32
#ifndef BT
#define BT 32
#endif
#define NWARP 16
#define MAXT 1                     // top byte tiles per warp   (16 warps: up to 16 x 8 bytes)
#define MAXR 2                     // rest byte tiles per warp  (16 warps: up to 32 x 8 bytes)
#ifndef NSTAGE
#define NSTAGE 3
#endif
#define PSTR (BT + 4)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void mma_f16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
#ifdef ACC16
  // GeForce Blackwell runs f16 MMA with an f16 accumulator at twice the rate of fp32 accumulation
  // (measured on the RTX 5090: 15.4 vs 30.8 clocks per m16n8k16): sum this k16 step (16 tokens) in
  // f16 from zero, then add it into the fp32 accumulator. The V operand carries 1/16 of its tier
  // scale (exact) and the sum is scaled back here, so 16 products of p <= 1 and |v| <= 6 s cannot
  // overflow f16 below s ~ 10^4.
  uint32_t h0, h1;
  asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%8};\n"
      : "=r"(h0), "=r"(h1) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0u));
  const float2 f0 = __half22float2(*reinterpret_cast<__half2*>(&h0)), f1 = __half22float2(*reinterpret_cast<__half2*>(&h1));
  d[0] = fmaf(f0.x, 16.f, d[0]); d[1] = fmaf(f0.y, 16.f, d[1]); d[2] = fmaf(f1.x, 16.f, d[2]); d[3] = fmaf(f1.y, 16.f, d[3]);
  return;
#endif
#ifdef NO_MMA
  d[0] += __uint_as_float(a[0] ^ b0); d[1] += __uint_as_float(a[1] ^ b1);
  return;
#endif
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t hmul2u(uint32_t a, uint32_t b) {
  __half2 r = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
// four small codes (one per byte, already masked to the table's index range) -> f16 pairs
// (byte 0, 1) and (byte 2, 3) through a byte table (fp16 high bytes; low bytes zero)
__device__ __forceinline__ void lut4(uint32_t x, uint32_t t_lo, uint32_t t_hi, uint32_t& p01, uint32_t& p23) {
#ifdef NO_DECODE
  p01 = x; p23 = x ^ t_lo;
  return;
#endif
  const uint32_t t = x | (x >> 4);
  const uint32_t sel = (t & 0xFFu) | ((t >> 8) & 0xFF00u);
  const uint32_t hb = __byte_perm(t_lo, t_hi, sel);
  p01 = __byte_perm(hb, 0u, 0x1404);
  p23 = __byte_perm(hb, 0u, 0x3424);
}

extern "C" __global__ void __launch_bounds__(NWARP * 32, 1)
latent_pv2(const float* __restrict__ lgt,       // [B, NQ, W] logits; latent token t at col0 + t
           const uint8_t* __restrict__ vc,      // [B, Tmax, v_row]
           const uint16_t* __restrict__ vs,     // [B, Tmax, 2] bf16 (top, rest) scales
           const uint16_t* __restrict__ al,     // [B, Tmax] bf16 alpha
           const int* __restrict__ seqlen,      // [B]
           const int* __restrict__ mx,          // [B, NQ] ordered-int max of the logits
           float* __restrict__ z,               // [B, NQ, zw]
           float* __restrict__ beta,            // [B, NQ]
           float* __restrict__ lsum,            // [B, NQ]
           int col0, int seg_len, int nb_t, int nq, int v_row, int zw, int ng, int ncopy, long long s_zc,
           long long s_pb, long long s_ph, long long s_vcb, long long s_vct, long long s_vsb, long long s_alb,
           long long s_zb) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int vstr = v_row + 16;
  uint8_t* vraw = smem;                                                   // [NSTAGE][BT][vstr]
  float* praw = reinterpret_cast<float*>(vraw + NSTAGE * BT * vstr);      // [NSTAGE][NQ][PSTR]
  uint32_t* sraw = reinterpret_cast<uint32_t*>(praw + NSTAGE * NQ * PSTR); // [NSTAGE][BT]
  float* araw = reinterpret_cast<float*>(sraw + NSTAGE * BT);             // [NSTAGE][BT]

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, tq = lane & 3;
  // grid (column group, segment, sequence): the ng CTAs of one segment split the byte tiles and run
  // side by side, so the V rows they all read come from L2 after the first
  const int gc = blockIdx.x, b = blockIdx.z;
  const int T = seqlen[b];
  const int l_beg = blockIdx.y * seg_len;
  const int l_end = min(l_beg + seg_len, T);
  if (l_beg >= T) return;

  auto issue = [&](int st, int t0) {
    uint8_t* vr = vraw + st * BT * vstr;
    const int cpr = v_row >> 4;
    for (int i = tid; i < BT * cpr; i += NWARP * 32) {
      const int t = i / cpr, o = (i - t * cpr) << 4;
      const bool ok = t0 + t < l_end;
      cp_async16(vr + t * vstr + o, vc + b * s_vcb + (long long)(ok ? t0 + t : 0) * s_vct + o, ok);
    }
    for (int i = tid; i < NQ * BT / 4; i += NWARP * 32) {
      const int h = i / (BT / 4), o = (i % (BT / 4)) << 2;
      const bool ok = t0 + o < l_end;
      cp_async16(praw + (st * NQ + h) * PSTR + o, lgt + b * s_pb + h * s_ph + col0 + (ok ? t0 + o : 0), ok);
    }
    if (tid >= 32 && tid < 32 + BT) {
      const int t = tid - 32;
      const bool ok = t0 + t < l_end;
      sraw[st * BT + t] = ok ? *reinterpret_cast<const uint32_t*>(vs + b * s_vsb + (long long)(t0 + t) * 2) : 0u;
      araw[st * BT + t] = ok ? __uint_as_float((uint32_t)al[b * s_alb + t0 + t] << 16) : 0.f;
    }
    asm volatile("cp.async.commit_group;\n");
  };

  // work split: rest byte tiles round-robin over warps; top byte tiles to the warps from the end
  const int nbt_t = nb_t >> 3, nbt_r = nq >> 3;
  float acc_t[2][MAXT][2][4];
  float acc_r[2][MAXR][4][4];
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int x = 0; x < MAXT; ++x)
#pragma unroll
      for (int s = 0; s < 2; ++s) acc_t[m][x][s][0] = acc_t[m][x][s][1] = acc_t[m][x][s][2] = acc_t[m][x][s][3] = 0.f;
#pragma unroll
    for (int x = 0; x < MAXR; ++x)
#pragma unroll
      for (int s = 0; s < 4; ++s) acc_r[m][x][s][0] = acc_r[m][x][s][1] = acc_r[m][x][s][2] = acc_r[m][x][s][3] = 0.f;
  }
  // softmax pass: thread -> (head ph, tokens pt0 .. pt0 + BT / 16 - 1)
  const int ph = tid >> 4, pt0 = (tid & 15) * (BT / 16);
  float pm;
  {
    const int i = mx[b * NQ + ph];
    pm = __int_as_float(i >= 0 ? i : i ^ 0x7FFFFFFF) * 1.4426950408889634f;
  }
  float bsum = 0.f, lacc = 0.f;

#pragma unroll
  for (int i = 0; i < NSTAGE - 1; ++i) issue(i, l_beg + i * BT);
  int it = 0;
  for (int t0 = l_beg; t0 < l_end; t0 += BT, ++it) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(NSTAGE - 2));
    __syncthreads();                             // stage landed; the slot refilled below is free
    issue((it + NSTAGE - 1) % NSTAGE, t0 + (NSTAGE - 1) * BT);
    const int st = it % NSTAGE;
    const int nval = l_end - t0;
    float* pr = praw + st * NQ * PSTR;
    {                                            // logits -> exp(l - max) in place; beta, normalizer
      float* row = pr + ph * PSTR;
      const float* ar = araw + st * BT;
#pragma unroll
      for (int j = 0; j < BT / 16; ++j) {
        const int t = pt0 + j;
#ifdef NO_SOFTMAX
        const float pv = t < nval ? row[t] : 0.f;
#else
        const float pv = t < nval ? exp2f(fmaf(row[t], 1.4426950408889634f, -pm)) : 0.f;
#endif
        row[t] = pv;
        lacc += pv;
        bsum += pv * ar[t];
      }
    }
    __syncthreads();
    const uint32_t* sr = sraw + st * BT;
#pragma unroll
    for (int ks = 0; ks < BT; ks += 16) {
      uint32_t a[2][4];
      auto pp = [&](int h, int t) {
        const float2 v = *reinterpret_cast<const float2*>(pr + h * PSTR + t);
        __half2 r = __floats2half2_rn(v.x, v.y);
        return *reinterpret_cast<uint32_t*>(&r);
      };
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int h = 16 * m + g;
        a[m][0] = pp(h, ks + 2 * tq);
        a[m][1] = pp(h + 8, ks + 2 * tq);
        a[m][2] = pp(h, ks + 2 * tq + 8);
        a[m][3] = pp(h + 8, ks + 2 * tq + 8);
      }
      const uint32_t s0 = sr[ks + 2 * tq], s1 = sr[ks + 2 * tq + 1], s8 = sr[ks + 2 * tq + 8], s9 = sr[ks + 2 * tq + 9];
      auto h2 = [](uint32_t u, uint32_t v) {     // bf16 tier scales -> f16 pair (ACC16: / 16 for f16-sum headroom)
#ifdef ACC16
        constexpr float k = 1.f / 16;
#else
        constexpr float k = 1.f;
#endif
        __half2 r = __floats2half2_rn(__uint_as_float(u << 16) * k, __uint_as_float(v << 16) * k);
        return *reinterpret_cast<uint32_t*>(&r);
      };
      const uint8_t* vr = vraw + (st * BT + ks) * vstr;
      auto gather = [&](int byte) {
#ifdef NO_GATHER
        return (uint32_t)(byte * 0x01010101u) ^ (uint32_t)ks;
#endif
        return (uint32_t)vr[(2 * tq) * vstr + byte] | ((uint32_t)vr[(2 * tq + 1) * vstr + byte] << 8) |
               ((uint32_t)vr[(2 * tq + 8) * vstr + byte] << 16) | ((uint32_t)vr[(2 * tq + 9) * vstr + byte] << 24);
      };
      {                                          // top tier: E2M1 nibbles (3-bit levels)
        const uint32_t sa = h2(s0, s1), sb = h2(s8, s9);
#pragma unroll
        for (int x = 0; x < MAXT; ++x) {
          const int xb = (ng * NWARP - 1 - (gc * NWARP + warp)) + x * ng * NWARP;
          if (xb < nbt_t) {
            const uint32_t w = gather((xb << 3) + g);
#pragma unroll
            for (int s = 0; s < 2; ++s) {
              const uint32_t f = (w >> (4 * s)) & 0x0F0F0F0Fu;
              uint32_t p01, p89;
              lut4(f & 0x07070707u, 0x3E3C3800u, 0x46444240u, p01, p89);
              p01 |= (f & 0x00000008u) << 12 | (f & 0x00000800u) << 20;      // signs of bytes 0, 1
              p89 |= (f & 0x00080000u) >> 4 | (f & 0x08000000u) << 4;        // signs of bytes 2, 3
              p01 = hmul2u(p01, sa);
              p89 = hmul2u(p89, sb);
              mma_f16(acc_t[0][x][s], a[0], p01, p89);
              mma_f16(acc_t[1][x][s], a[1], p01, p89);
            }
          }
        }
      }
      {                                          // rest tier: 2-bit fields (sign | magnitude)
        const uint32_t sa = h2(s0 >> 16, s1 >> 16), sb = h2(s8 >> 16, s9 >> 16);
#pragma unroll
        for (int x = 0; x < MAXR; ++x) {
          const int xb = (gc * NWARP + warp) + x * ng * NWARP;
          if (xb < nbt_r) {
            const uint32_t w = gather(nb_t + (xb << 3) + g);
#pragma unroll
            for (int s = 0; s < 4; ++s) {
              uint32_t p01, p89;
              lut4((w >> (2 * s)) & 0x03030303u, 0xBEB83E38u, 0u, p01, p89);
              p01 = hmul2u(p01, sa);
              p89 = hmul2u(p89, sb);
              mma_f16(acc_r[0][x][s], a[0], p01, p89);
              mma_f16(acc_r[1][x][s], a[1], p01, p89);
            }
          }
        }
      }
    }
  }
  asm volatile("cp.async.wait_group 0;\n");

#pragma unroll
  for (int o = 1; o < 16; o <<= 1) {             // 16 lanes per head
    bsum += __shfl_xor_sync(0xffffffffu, bsum, o);
    lacc += __shfl_xor_sync(0xffffffffu, lacc, o);
  }
  if ((tid & 15) == 0 && gc == 0) { atomicAdd(beta + b * NQ + ph, bsum); atomicAdd(lsum + b * NQ + ph, lacc); }
  float* zb = z + b * s_zb + (blockIdx.y % ncopy) * s_zc;    // segments spread over ncopy copies of z
  // float4 atomics: lanes tq, tq ^ 1 swap halves so the even lane owns 4 columns of row g and the odd
  // lane 4 columns of row g + 8 (twice fewer atomic operations than float2)
  auto add4 = [&](int col_pair0, const float* f) {     // f: the frag (g: f0 f1, g + 8: f2 f3), cols 2tq, 2tq + 1
    const bool odd = tq & 1;
    const float s0 = odd ? f[0] : f[2], s1 = odd ? f[1] : f[3];
    const float r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
    const int row = odd ? g + 8 : g;
    const float4 v = odd ? make_float4(r0, r1, f[2], f[3]) : make_float4(f[0], f[1], r0, r1);
#ifdef NO_ATOMIC
    *reinterpret_cast<float4*>(zb + (long long)row * zw + col_pair0 + 4 * (tq >> 1)) = v;
#else
    atomicAdd(reinterpret_cast<float4*>(zb + (long long)row * zw + col_pair0 + 4 * (tq >> 1)), v);
#endif
  };
#pragma unroll
  for (int x = 0; x < MAXT; ++x) {
    const int xb = (ng * NWARP - 1 - (gc * NWARP + warp)) + x * ng * NWARP;
    if (xb < nbt_t) {
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const int c0 = (xb << 3) + s * nb_t;
        add4(c0, acc_t[0][x][s]);
        add4(c0 + 16 * zw, acc_t[1][x][s]);
      }
    }
  }
#pragma unroll
  for (int x = 0; x < MAXR; ++x) {
    const int xb = (gc * NWARP + warp) + x * ng * NWARP;
    if (xb < nbt_r) {
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int c0 = 2 * nb_t + (xb << 3) + s * nq;
        add4(c0, acc_r[0][x][s]);
        add4(c0 + 16 * zw, acc_r[1][x][s]);
      }
    }
  }
}
