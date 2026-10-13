// Absorbed P . V over the PhasorKV latent V cache, sm_120a.
//
// One CTA (8 warps) = one token segment of one sequence, all 32 query heads, all latent V columns:
//   z[b, h, c] += sum_t p[b, h, t] * s_tier(c)[t] * v[t, c],   beta[b, h] += sum_t p[b, h, t] * alpha[t]
// The V cache is token-major; a tier of width w takes nb bytes (nb a multiple of 8, 2 nb >= w) and
// byte j holds the tier's columns j (low nibble) and nb + j (high nibble). z is kept in that padded
// column space ([top: 2 nb_t][rest: 2 nb_r]); the V lift's weights carry the same zero padding.
// Per 16-token stage, cp.async brings the codes, the probabilities, the tier scales and alpha into
// shared memory (4-stage pipeline, one barrier per stage). Each warp owns byte columns in groups of 8 ("byte tiles"): a
// thread gathers the byte of its column for its four tokens of the mma k16 step (2tq, 2tq + 1,
// 2tq + 8, 2tq + 9), decodes the four low and four high nibbles with one byte_perm table lookup each
// straight into the f16 B fragments of two n-tiles (low / high columns), scales them by the tokens'
// tier scales, and runs mma m16n8k16 against P (fp16, ldmatrix) for both 16-head halves.
#include <cstdint>
#include <cuda_fp16.h>

#define NQ 32
#ifndef BT
#define BT 16
#endif
#ifndef NWARP
#define NWARP 8
#endif
#define MAXBT ((72 + NWARP - 1) / NWARP)   // byte tiles per warp: up to 72 x 8 bytes = 1152 padded columns
#ifndef NSTAGE
#define NSTAGE 4
#endif
#define PSTR (BT + 4)             // fp32 row stride of the staged probabilities (bank spread)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;\n"); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n"); }

__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
  unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_f16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t hmul2u(uint32_t a, uint32_t b) {
  __half2 r = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
// four E2M1 codes (one per byte, low 4 bits) -> f16 pairs (byte 0, 1) and (byte 2, 3), exact
__device__ __forceinline__ void e2m1x4(uint32_t x, uint32_t& lo, uint32_t& hi) {
  const uint32_t m = x & 0x07070707u;
  const uint32_t t = m | (m >> 4);
  const uint32_t sel = (t & 0xFFu) | ((t >> 8) & 0xFF00u);
  uint32_t hb = __byte_perm(0x3E3C3800u, 0x46444240u, sel);          // fp16 high bytes of |v|
  hb |= (x & 0x08080808u) << 4;                                        // sign
  lo = __byte_perm(hb, 0u, 0x1404);
  hi = __byte_perm(hb, 0u, 0x3424);
}

extern "C" __global__ void __launch_bounds__(NWARP * 32, 1)
latent_pv(const float* __restrict__ p,          // [B, NQ, W] probabilities; latent token t at col0 + t
          const uint8_t* __restrict__ vc,       // [B, Tmax, v_row]
          const uint16_t* __restrict__ vs,      // [B, Tmax, 2] bf16 (top, rest) scales
          const uint16_t* __restrict__ al,      // [B, Tmax] bf16 alpha
          const int* __restrict__ seqlen,       // [B]
          float* __restrict__ z,                // [B, NQ, zw] (padded columns) accumulated atomically
          float* __restrict__ beta,             // [B, NQ]
          int col0, int seg_len, int nb_t, int nb_r, int v_row, int zw,
          long long s_pb, long long s_ph, long long s_vcb, long long s_vct, long long s_vsb, long long s_alb,
          long long s_zb) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int vstr = v_row + 16;                  // the four rows a thread reads sit on distinct banks
  uint8_t* vraw = smem;                                                   // [NSTAGE][BT][vstr]
  float* praw = reinterpret_cast<float*>(vraw + NSTAGE * BT * vstr);      // [NSTAGE][NQ][PSTR]
  uint32_t* sraw = reinterpret_cast<uint32_t*>(praw + NSTAGE * NQ * PSTR); // [NSTAGE][BT] bf16 (top, rest)
  float* araw = reinterpret_cast<float*>(sraw + NSTAGE * BT);             // [NSTAGE][BT]

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, tq = lane & 3;
  const int b = blockIdx.y;
  const int T = seqlen[b];
  const int l_beg = blockIdx.x * seg_len;
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
      cp_async16(praw + (st * NQ + h) * PSTR + o, p + b * s_pb + h * s_ph + col0 + (ok ? t0 + o : 0), ok);
    }
    if (tid >= 32 && tid < 32 + BT) {
      const int t = tid - 32;
      const bool ok = t0 + t < l_end;
      sraw[st * BT + t] = ok ? *reinterpret_cast<const uint32_t*>(vs + b * s_vsb + (long long)(t0 + t) * 2) : 0u;
      araw[st * BT + t] = ok ? __uint_as_float((uint32_t)al[b * s_alb + t0 + t] << 16) : 0.f;
    }
    cp_commit();
  };

  const int nbt_t = nb_t >> 3, nbt = (nb_t + nb_r) >> 3;
  const int per = (nbt + NWARP - 1) / NWARP;
  const int first = warp * per;
  const int nmy = max(0, min(per, nbt - first));
  float acc[2][MAXBT][2][4];                    // [head half][byte tile][low / high n-tile][frag]
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int x = 0; x < MAXBT; ++x)
#pragma unroll
      for (int s = 0; s < 2; ++s) acc[m][x][s][0] = acc[m][x][s][1] = acc[m][x][s][2] = acc[m][x][s][3] = 0.f;
  float bsum = 0.f;

#pragma unroll
  for (int i = 0; i < NSTAGE - 1; ++i) issue(i, l_beg + i * BT);
  int it = 0;
  for (int t0 = l_beg; t0 < l_end; t0 += BT, ++it) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(NSTAGE - 2));
    __syncthreads();                             // this stage landed; the slot refilled below is free
    issue((it + NSTAGE - 1) % NSTAGE, t0 + (NSTAGE - 1) * BT);
    const int st = it % NSTAGE;
    const int nval = l_end - t0;                 // tokens of this stage inside the segment
    const float* pr = praw + st * NQ * PSTR;
    const uint32_t* sr = sraw + st * BT;
    if (warp == 0) {                             // beta: lane = head
      const float* ar = araw + st * BT;
#pragma unroll
      for (int t = 0; t < BT; ++t) bsum += (t < nval) ? pr[lane * PSTR + t] * ar[t] : 0.f;
    }
    for (int ks = 0; ks < BT; ks += 16) {
    // P fragments (fp32 -> f16), masked past the segment
    uint32_t a[2][4];
    auto pp = [&](int h, int t) {
      const float2 v = *reinterpret_cast<const float2*>(pr + h * PSTR + t);
      __half2 r = __floats2half2_rn(t < nval ? v.x : 0.f, t + 1 < nval ? v.y : 0.f);
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
    auto h2 = [](uint32_t u, uint32_t v) {      // bf16 bits (low 16 of u, of v) -> f16 pair
      __half2 r = __floats2half2_rn(__uint_as_float(u << 16), __uint_as_float(v << 16));
      return *reinterpret_cast<uint32_t*>(&r);
    };
    const uint32_t st01 = h2(s0, s1), st89 = h2(s8, s9);
    const uint32_t sr01 = h2(s0 >> 16, s1 >> 16), sr89 = h2(s8 >> 16, s9 >> 16);
    const uint8_t* vr = vraw + (st * BT + ks) * vstr;
#pragma unroll
    for (int x = 0; x < MAXBT; ++x) {
      if (x < nmy) {
        const int xb = first + x;
        const bool top = xb < nbt_t;
        const int byte = (xb << 3) + g;
        const uint32_t w = (uint32_t)vr[(2 * tq) * vstr + byte] | ((uint32_t)vr[(2 * tq + 1) * vstr + byte] << 8) |
                           ((uint32_t)vr[(2 * tq + 8) * vstr + byte] << 16) | ((uint32_t)vr[(2 * tq + 9) * vstr + byte] << 24);
        uint32_t l01, l89, h01, h89;
        e2m1x4(w, l01, l89);
        e2m1x4(w >> 4, h01, h89);
        const uint32_t sa = top ? st01 : sr01, sb = top ? st89 : sr89;
        l01 = hmul2u(l01, sa); l89 = hmul2u(l89, sb); h01 = hmul2u(h01, sa); h89 = hmul2u(h89, sb);
        mma_f16(acc[0][x][0], a[0], l01, l89);
        mma_f16(acc[1][x][0], a[1], l01, l89);
        mma_f16(acc[0][x][1], a[0], h01, h89);
        mma_f16(acc[1][x][1], a[1], h01, h89);
      }
    }
    }
  }
  asm volatile("cp.async.wait_group 0;\n");

  if (warp == 0) atomicAdd(beta + b * NQ + lane, bsum);
#pragma unroll
  for (int x = 0; x < MAXBT; ++x) {
    if (x < nmy) {
      const int xb = first + x;
      const bool top = xb < nbt_t;
      const int nb = top ? nb_t : nb_r, base = top ? 0 : 2 * nb_t;
      const int j = ((top ? xb : xb - nbt_t) << 3) + 2 * tq;             // tier byte -> low column j
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const int col = base + j + s * nb;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          float* r0 = z + b * s_zb + (long long)(m * 16 + g) * zw + col;
          float* r1 = z + b * s_zb + (long long)(m * 16 + g + 8) * zw + col;
          atomicAdd(reinterpret_cast<float2*>(r0), make_float2(acc[m][x][s][0], acc[m][x][s][1]));
          atomicAdd(reinterpret_cast<float2*>(r1), make_float2(acc[m][x][s][2], acc[m][x][s][3]));
        }
      }
    }
  }
}
