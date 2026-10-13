"""Decode attention over the PhasorKV latent cache (Triton; native FP4 MMA on Blackwell sm_120).

Per layer and decode step:
  logits  one program per (tile of tokens, sequence, group of kv heads): rebuilds each kv head's
          pre-RoPE keys from the latent with the block-scaled FP4 MMA (NVFP4 latent codes x NVFP4
          rebuild weights, fp32 accumulate), adds alpha_t * kbar, rotates them with the model's
          bf16 cos / sin table and dots them with the head's query heads (elementwise: the rotated
          keys never have to move into another MMA operand layout).
  softmax over the exact tokens (attention sinks, recent window: bf16) and the latent tokens.
  PV      absorbed, as in STAR-KV-Serve's paged_quant: one program per (sequence, column tile,
          token segment), all query heads as MMA rows. Each V tier sits in its own byte range,
          so a tile never straddles two tiers and the per-token tier scale folds into the
          probabilities; codes go to fp16 with two integer ops. Segments add atomically.
  lift    o_h = A_v z_h + beta_h vbar + o_exact (beta_h = sum_t p_ht alpha_t).

The K rebuild is FP4 x FP4 rather than STAR-KV's fold of the query into U: with GQA the fold
costs n_q / n_kv times the rebuild's FLOPs, in half precision.

Per-head K (one latent per kv head) and joint / grouped K (one latent shared by several kv heads)
run the same kernels: each kv head is (latent column offset, rank, its rows of the rebuild weights).
"""
import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .. import quant
from .nvfp4 import (e2m1_decode, pack_nibbles, quantize_nv_block, quantize_nvfp4_weight,
                    quantize_per_token, unpack_nibbles)


@dataclass
class LayerPack:
    """One layer's static decode data, and the write-path projections."""
    n_q: int
    n_kv: int
    d: int
    rk: int                    # latent K width (all K latents side by side)
    rv: int                    # latent V width
    v_no: int                  # V top-tier width
    v_bits: tuple              # (top, rest) bits of the V per-token tiers
    head_i: torch.Tensor       # int32 [n_kv, 4]: latent column offset, rank, A code offset, A scale offset
    head_f: torch.Tensor       # fp32 [n_kv]: A global scale
    a_codes: torch.Tensor      # uint8 flat: per kv head [d, ceil64(rank) / 2], rows in RoPE-pair order
    a_scales: torch.Tensor     # e4m3 flat: per kv head [d, ceil64(rank) / 16], same rows
    kbar: torch.Tensor         # fp32 [n_kv * d]
    vbar: torch.Tensor         # fp32 [n_kv * d]
    A_v: torch.Tensor          # fp32 [n_kv, d, rv]
    inv_freq: torch.Tensor     # fp32 [d / 2]
    attn_scaling: float        # RoPE cos / sin multiplier (1 for llama3)
    B_k: torch.Tensor          # fp32 [rk, D]
    B_v: torch.Tensor          # fp32 [rv, D]
    mu_scaled: torch.Tensor    # fp32 [D]
    k_tiers: list              # [(lo, hi, fmt, fp32 second)] NV block tiers of the K latent
    a_hat: list                # per kv head: dequantized rebuild weights [d, rank] (reference)

    def __post_init__(self):
        h = self.d // 2
        self.pair = torch.stack([torch.arange(h), torch.arange(h) + h], -1).flatten()   # 0, h, 1, h+1, ...
        self.kbar_pair = self.kbar.view(self.n_kv, self.d)[:, self.pair.to(self.kbar.device)].contiguous()
        nb_t, nb_r = (8 * math.ceil(w / 16) for w in (self.v_no, self.rv - self.v_no))   # bytes per tier
        self.v_nb = (nb_t, nb_r)
        self.v_row = 16 * math.ceil((nb_t + nb_r) / 16)        # bytes per token, 16-aligned
        self._cs = None
        self.A_v_bf16 = self.A_v.to(torch.bfloat16).contiguous()
        # the CUDA PV kernel's z columns: [top tier, padded to 2 nb_t][rest tier, padded to 2 nb_r]
        self.zw = 2 * (nb_t + nb_r)
        A = torch.zeros(self.n_kv, self.d, self.zw, device=self.A_v.device)
        A[..., :self.v_no] = self.A_v[..., :self.v_no]
        A[..., 2 * nb_t:2 * nb_t + self.rv - self.v_no] = self.A_v[..., self.v_no:]
        self.A_v_pad = A.to(torch.bfloat16).contiguous()
        # "q2" layout (the fused path): top tier in nibbles (nb bytes), rest tier at 2 bits (nq bytes,
        # byte j = columns j, nq + j, 2 nq + j, 3 nq + j); z columns [top: 2 nb][rest: 4 nq]
        q_t, q_r = 8 * math.ceil(self.v_no / 16), 8 * math.ceil((self.rv - self.v_no) / 32)
        self.q2_nb = (q_t, q_r)
        self.q2_row = 16 * math.ceil((q_t + q_r) / 16)
        self.q2_zw = 2 * q_t + 4 * q_r
        A = torch.zeros(self.n_kv, self.d, self.q2_zw, device=self.A_v.device)
        A[..., :self.v_no] = self.A_v[..., :self.v_no]
        A[..., 2 * q_t:2 * q_t + self.rv - self.v_no] = self.A_v[..., self.v_no:]
        self.A_v_q2 = A.to(torch.bfloat16).contiguous()

    @property
    def k_rebuild_flops(self):
        """Per token: 2 * d * rank summed over kv heads (the FP4 MMA work)."""
        return int(2 * self.d * self.head_i[:, 1].sum())

    def bytes_per_token(self, v_layout="nib"):
        """Cache bytes per token: K codes (4-bit) + block scales, V codes + two bf16 scales, alpha."""
        return self.rk // 2 + self.rk // 16 + (self.q2_row if v_layout == "q2" else self.v_row) + 4 + 2

    def bytes_per_token_packed(self):
        """Same with NVINT3 / NVINT2 packed to 3 / 2 bits."""
        kbits = sum((hi - lo) * int(f[-1]) for lo, hi, f, _ in self.k_tiers)
        vbits = self.v_no * self.v_bits[0] + (self.rv - self.v_no) * self.v_bits[1]
        return (kbits + vbits) / 8 + self.rk // 16 + 4 + 2

    def cos_sin(self, n_pos):
        """(cos, sin) [n_pos, d/2] bf16 per RoPE pair, as the model forms them: fp32 angle, fp32
        cos / sin, times attention scaling, rounded to bf16."""
        if self._cs is None or self._cs[0].shape[0] < n_pos:
            n = max(n_pos, 2 * (0 if self._cs is None else self._cs[0].shape[0]))
            ang = torch.arange(n, device=self.inv_freq.device).float()[:, None] * self.inv_freq[None]
            self._cs = ((ang.cos() * self.attn_scaling).to(torch.bfloat16).contiguous(),
                        (ang.sin() * self.attn_scaling).to(torch.bfloat16).contiguous())
        return self._cs


def _pack_heads(n_kv, d, blocks, device):
    """blocks: [(kv heads, latent offset, rank, A_raw [len(heads) * d, rank])] -> per-head tables
    (rebuild-weight rows stored in RoPE-pair order 0, d/2, 1, d/2 + 1, ...)."""
    pair = torch.stack([torch.arange(d // 2), torch.arange(d // 2) + d // 2], -1).flatten().to(device)
    head_i = torch.zeros(n_kv, 4, dtype=torch.int32)
    head_f = torch.ones(n_kv, dtype=torch.float32)
    codes, scales, a_hat = [], [], [None] * n_kv
    co = so = 0
    for heads, off, r, A in blocks:
        if r == 0:
            continue
        cols = 64 * math.ceil(r / 64)
        c, s, g, Ah = quantize_nvfp4_weight(A.float().to(device), cols)
        for j, h in enumerate(heads):
            rows = slice(j * d, (j + 1) * d)
            head_i[h] = torch.tensor([off, r, co, so])
            head_f[h] = g
            codes.append(c[rows][pair].flatten())
            scales.append(s[rows][pair].flatten())
            a_hat[h] = Ah[rows]
            co += d * cols // 2
            so += d * cols // 16
    for h in range(n_kv):
        if a_hat[h] is None:                     # kv head without a latent: k = alpha * kbar
            a_hat[h] = torch.zeros(d, 0, device=device)
    cat = lambda xs, dt: torch.cat(xs) if xs else torch.zeros(1, dtype=dt, device=device)
    return (head_i.to(device), head_f.to(device), cat(codes, torch.uint8),
            cat(scales, torch.float8_e4m3fn), a_hat)


def pack_from_patch(patch, i, attn, n_q, rotary, device="cuda"):
    """LayerPack for layer i of a model patched by LatentPatch (kind TIERED, options .al.wq.ex,
    keep_meta=True), from the installed LatentProj modules, patch.kmeta[i] and the model's
    rotary embedding."""
    meta = patch.kmeta[i]
    kp, vp = attn.k_proj, attn.v_proj
    gk = meta["k_group"]
    inv_freq = rotary.inv_freq.float()
    d = inv_freq.numel() * 2
    n_kv = kp.A.shape[0] // d
    kt, kr, vt, vr = meta["tiers"]
    blocks, k_tiers = [], []
    for (a, no, r), kb, A in zip(meta["head_ranges"], meta["k_blocks"], meta["A_raw"]):
        blocks.append((list(range(kb * gk, (kb + 1) * gk)), a, r, A))
        var = meta["var_k"]
        for lo, hi, f in ((a, a + no, kt), (a + no, a + r, kr)):
            if hi > lo:
                k_tiers.append((lo, hi, f, quant.nv_static_second(var[lo:hi], f).to(device)))
    head_i, head_f, a_codes, a_scales, a_hat = _pack_heads(n_kv, d, blocks, device)
    rv = vp.B.shape[0]
    return LayerPack(n_q=n_q, n_kv=n_kv, d=d, rk=kp.B.shape[0], rv=rv, v_no=meta["v_no"],
                     v_bits=(int(vt[1]), int(vr[1])), head_i=head_i, head_f=head_f,
                     a_codes=a_codes, a_scales=a_scales,
                     kbar=kp.bias.float().to(device), vbar=vp.bias.float().to(device),
                     A_v=vp.A.float().to(device).view(n_kv, d, rv), inv_freq=inv_freq.to(device),
                     attn_scaling=float(getattr(rotary, "attention_scaling", 1.0)),
                     B_k=kp.B.float().to(device), B_v=vp.B.float().to(device),
                     mu_scaled=kp.mu_scaled.float().to(device), k_tiers=k_tiers, a_hat=a_hat)


def synthetic_pack(n_q, n_kv, d, D, k_blocks, rv, tiers=("nvint3", "nvint2", "i3t", "i2t"),
                   rope_theta=500000.0, device="cuda", seed=0):
    """A LayerPack with random factors and the given shapes, for tests and benchmarks.
    k_blocks: [(kv heads, rank)], e.g. per-head [([h], r_h) for h] or joint [(range(n_kv), r)]."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=gen)
    kt, kr, vt, vr = tiers
    blocks, k_tiers, off = [], [], 0
    for heads, r in k_blocks:
        heads = list(heads)
        decay = torch.exp(-torch.arange(r) / max(r, 1) * 2.0)           # a falling latent spectrum
        blocks.append((heads, off, r, rn(len(heads) * d, r) * decay / math.sqrt(max(r, 1))))
        no = max(16, 16 * round(0.2 * r / 16)) if r else 0
        for lo, hi, f in ((off, off + no, kt), (off + no, off + r, kr)):
            if hi > lo:
                k_tiers.append((lo, hi, f, quant.nv_static_second(decay[lo - off:hi - off] ** 2, f).to(device)))
        off += r
    rk = off
    head_i, head_f, a_codes, a_scales, a_hat = _pack_heads(n_kv, d, blocks, device)
    B_k = torch.zeros(rk, D)
    for heads, o, r, _ in blocks:
        B_k[o:o + r] = rn(r, D) / math.sqrt(D) * torch.exp(-torch.arange(r) / max(r, 1) * 2.0)[:, None]
    inv = 1.0 / rope_theta ** (torch.arange(0, d, 2).float() / d)
    return LayerPack(n_q=n_q, n_kv=n_kv, d=d, rk=rk, rv=rv, v_no=int(0.2 * rv),
                     v_bits=(int(vt[1]), int(vr[1])), head_i=head_i, head_f=head_f,
                     a_codes=a_codes, a_scales=a_scales,
                     kbar=(rn(n_kv * d) * 0.5).to(device), vbar=(rn(n_kv * d) * 0.1).to(device),
                     A_v=(rn(n_kv, d, rv) / math.sqrt(rv)).to(device), inv_freq=inv.to(device),
                     attn_scaling=1.0, B_k=B_k.to(device), B_v=(rn(rv, D) / math.sqrt(D)).to(device),
                     mu_scaled=(rn(D) / D).to(device), k_tiers=k_tiers, a_hat=a_hat)


# --------------------------------------------------------------------------------------------
# cache layout and write path (PyTorch): normalized hidden states -> cache entries
# --------------------------------------------------------------------------------------------
#   kc  uint8 [B, T, rk/2]     K codes; byte j holds latent columns 2j, 2j+1 (FP4 MMA operand order)
#   ks  e4m3  [B, T, rk/16]    K block-16 scales
#   vc  uint8 [B, T, v_row]    V codes, top tier then rest tier, each in its own byte range; in a
#                              tier of width w (nb = ceil(w/2) bytes) byte j holds columns j and nb + j
#   vs  bf16  [B, T, 2]        V per-token scales (top, rest)
#   al  bf16  [B, T]           alpha

def pack_v_codes(pack: LayerPack, codes):
    out = []
    for (lo, hi), nb in zip(((0, pack.v_no), (pack.v_no, pack.rv)), pack.v_nb):
        x = torch.nn.functional.pad(codes[..., lo:hi], (0, 2 * nb - (hi - lo)))
        out.append(x[..., :nb] | (x[..., nb:] << 4))
    v = torch.cat(out, -1)
    return torch.nn.functional.pad(v, (0, pack.v_row - v.shape[-1])).to(torch.uint8)


def unpack_v_codes(pack: LayerPack, vc):
    out, b0 = [], 0
    for (lo, hi), nb in zip(((0, pack.v_no), (pack.v_no, pack.rv)), pack.v_nb):
        x = vc[..., b0:b0 + nb]
        out.append(torch.cat([x & 15, x >> 4], -1)[..., :hi - lo])
        b0 += nb
    return torch.cat(out, -1)


def pack_v_q2(pack: LayerPack, codes):
    """E2M1 codes [..., rv] (top: 3-bit levels, rest: +-0.5 / +-1.5) -> q2 bytes [..., q2_row]."""
    q_t, q_r = pack.q2_nb
    top = torch.nn.functional.pad(codes[..., :pack.v_no], (0, 2 * q_t - pack.v_no))
    top = top[..., :q_t] | (top[..., q_t:] << 4)
    r = codes[..., pack.v_no:]
    f = ((r >> 3) << 1) | ((r >> 1) & 1)                        # 2-bit field: sign | magnitude
    f = torch.nn.functional.pad(f, (0, 4 * q_r - f.shape[-1]))
    rest = f[..., :q_r] | (f[..., q_r:2 * q_r] << 2) | (f[..., 2 * q_r:3 * q_r] << 4) | (f[..., 3 * q_r:] << 6)
    v = torch.cat([top, rest], -1)
    return torch.nn.functional.pad(v, (0, pack.q2_row - v.shape[-1])).to(torch.uint8)


def unpack_v_q2(pack: LayerPack, vc):
    q_t, q_r = pack.q2_nb
    top = vc[..., :q_t]
    top = torch.cat([top & 15, top >> 4], -1)[..., :pack.v_no]
    r = vc[..., q_t:q_t + q_r]
    f = torch.cat([(r >> (2 * i)) & 3 for i in range(4)], -1)[..., :pack.rv - pack.v_no]
    rest = (1 + 2 * (f & 1)) | ((f >> 1) << 3)                  # back to E2M1 codes
    return torch.cat([top, rest.to(torch.uint8)], -1)


def write(pack: LayerPack, x: torch.Tensor, v_layout="nib"):
    """x: [B, T, D] attention input (after the layer norm). -> dict of cache tensors for T tokens."""
    xf = x.float()
    ck = xf @ pack.B_k.T
    kcodes = torch.zeros(*ck.shape, dtype=torch.uint8, device=x.device)
    ks = torch.zeros(*ck.shape[:-1], pack.rk // 16, dtype=torch.float8_e4m3fn, device=x.device)
    for lo, hi, f, sec in pack.k_tiers:
        c, s, _ = quantize_nv_block(ck[..., lo:hi], f, sec)
        kcodes[..., lo:hi] = c
        ks[..., lo // 16:hi // 16] = s
    cv = xf @ pack.B_v.T
    ct, st, _ = quantize_per_token(cv[..., :pack.v_no], pack.v_bits[0])
    cr, sr, _ = quantize_per_token(cv[..., pack.v_no:], pack.v_bits[1])
    vcodes = torch.cat([ct, cr], -1)
    vc = pack_v_q2(pack, vcodes) if v_layout == "q2" else pack_v_codes(pack, vcodes)
    return dict(kc=pack_nibbles(kcodes), ks=ks, vc=vc, vs=torch.stack([st, sr], -1),
                al=(xf @ pack.mu_scaled).to(torch.bfloat16), v_layout=v_layout)


def random_cache(pack: LayerPack, B, T, device="cuda", v_layout="nib"):
    """Random bytes in the cache layout (any byte is a pair of valid E2M1 codes), for timing."""
    u8 = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, device=device)
    return dict(kc=u8(B, T, pack.rk // 2),
                ks=(torch.rand(B, T, pack.rk // 16, device=device) + 0.5).to(torch.float8_e4m3fn),
                vc=u8(B, T, pack.q2_row if v_layout == "q2" else pack.v_row), v_layout=v_layout,
                vs=(torch.rand(B, T, 2, device=device) + 0.5).to(torch.bfloat16),
                al=torch.ones(B, T, dtype=torch.bfloat16, device=device))


# --------------------------------------------------------------------------------------------
# decode kernels
# --------------------------------------------------------------------------------------------

@triton.jit
def _latent_logits_kernel(
        Q, KC, KS, AL, SEQ, AC, AS, HEAD_I, HEAD_F, KBAR, COS, SIN, LG,
        T_max, pos0, col0, qk_scale,
        s_qb, s_kcb, s_kct, s_ksb, s_kst, s_alb, s_lb, s_lh,
        REP: tl.constexpr, D: tl.constexpr, GH: tl.constexpr, BT: tl.constexpr):
    """Logits of one tile of BT tokens for GH kv heads: k = A . codes (FP4 x FP4 block-scaled
    MMA, the rank in 64-wide steps) + alpha * kbar, rotated, dotted with the REP query heads.
    LG[b, h, col0 + t] = q_h . k_t / sqrt(d); -inf past the sequence."""
    pt = tl.program_id(0)
    b = tl.program_id(1)
    pg = tl.program_id(2)
    H2: tl.constexpr = D // 2
    T = tl.load(SEQ + b)
    t = pt * BT + tl.arange(0, BT)
    tm = t < T
    if pt * BT >= T:                             # tile past this sequence: -inf logits only
        for gi in tl.static_range(GH):
            for j in tl.static_range(REP):
                h = (pg * GH + gi) * REP + j
                tl.store(LG + b * s_lb + h * s_lh + col0 + t, tl.full([BT], float("-inf"), tl.float32),
                         mask=t < T_max)
        return
    dd = tl.arange(0, D)
    d2 = tl.arange(0, H2)
    jb = tl.arange(0, 32)                        # bytes of one 64-wide latent step
    j4 = tl.arange(0, 4)                         # its four 16-blocks
    alpha = tl.load(AL + b * s_alb + t, mask=tm, other=0.0).to(tl.float32)
    for gi in tl.static_range(GH):
        g = pg * GH + gi
        lat = tl.load(HEAD_I + g * 4)
        rk = tl.load(HEAD_I + g * 4 + 1)
        aco = tl.load(HEAD_I + g * 4 + 2)
        aso = tl.load(HEAD_I + g * 4 + 3)
        nch = (rk + 63) // 64
        ra2 = nch * 32
        ra16 = nch * 4
        k = tl.zeros([BT, D], tl.float32)
        for c in range(0, nch):
            col = c * 64
            C = tl.load(KC + b * s_kcb + t[:, None] * s_kct + (lat + col) // 2 + jb[None, :],
                        mask=tm[:, None] & ((col + 2 * jb) < rk)[None, :], other=0)
            SC = tl.load(KS + b * s_ksb + t[:, None] * s_kst + (lat + col) // 16 + j4[None, :],
                         mask=tm[:, None] & ((col + 16 * j4) < rk)[None, :], other=0.0)
            A = tl.load(AC + aco + dd[:, None] * ra2 + c * 32 + jb[None, :])
            SA = tl.load(AS + aso + dd[:, None] * ra16 + c * 4 + j4[None, :])
            k = tl.dot_scaled(C, SC, "e2m1", tl.trans(A), SA, "e2m1", acc=k)
        k = k * tl.load(HEAD_F + g) + alpha[:, None] * tl.load(KBAR + g * D + dd)[None, :]
        x0, x1 = tl.split(tl.reshape(k, [BT, H2, 2]))               # pair j: dims j and j + d/2
        cos = tl.load(COS + (t + pos0)[:, None] * H2 + d2[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        sin = tl.load(SIN + (t + pos0)[:, None] * H2 + d2[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        r0 = (x0 * cos - x1 * sin).to(tl.bfloat16).to(tl.float32)
        r1 = (x1 * cos + x0 * sin).to(tl.bfloat16).to(tl.float32)
        for j in tl.static_range(REP):
            h = g * REP + j
            q0 = tl.load(Q + b * s_qb + h * D + d2).to(tl.float32)
            q1 = tl.load(Q + b * s_qb + h * D + H2 + d2).to(tl.float32)
            lg = tl.sum(r0 * q0[None, :] + r1 * q1[None, :], 1) * qk_scale
            tl.store(LG + b * s_lb + h * s_lh + col0 + t, tl.where(tm, lg, float("-inf")), mask=t < T_max)


@triton.jit
def _e2m1_f16(c):
    """4-bit E2M1 codes (uint8) -> fp16 values: the code's exponent / mantissa bits placed in an
    fp16 make E2M1 x 2^-14 (subnormal for the zero exponent), exact; times 2^14, exact."""
    bits = ((c & 7).to(tl.int16) << 9) | ((c & 8).to(tl.int16) << 12)
    return bits.to(tl.float16, bitcast=True) * 16384.0


@triton.jit
def _latent_pv_kernel(
        P, VC, VS, AL, SEQ, Z, BETA,
        seg_len, col0, nb_top, nb_rest, w_top, w_rest, nt_top,
        s_pb, s_ph, s_vcb, s_vct, s_vsb, s_vst, s_alb, s_zb, s_zh,
        N_Q: tl.constexpr, NB: tl.constexpr, BL: tl.constexpr):
    """z[b, h, :] += P[b, h, seg] . V[seg, tile] for one tile of NB bytes (2 NB columns) of one
    V tier; the tier's per-token scale is folded into P. Tile 0 also adds beta = P . alpha."""
    b = tl.program_id(0)
    pt = tl.program_id(1)
    sg = tl.program_id(2)
    T = tl.load(SEQ + b)
    l_beg = sg * seg_len
    l_end = tl.minimum(l_beg + seg_len, T)
    if l_beg >= T:
        return
    is_top = pt < nt_top
    tt = tl.where(is_top, pt, pt - nt_top)
    byte0 = tl.where(is_top, 0, nb_top)
    nb = tl.where(is_top, nb_top, nb_rest)
    w = tl.where(is_top, w_top, w_rest)
    c0 = tl.where(is_top, 0, w_top)
    si = tl.where(is_top, 0, 1)
    jb = tt * NB + tl.arange(0, NB)
    bm = jb < nb
    h = tl.arange(0, N_Q)
    acc_lo = tl.zeros([N_Q, NB], tl.float32)
    acc_hi = tl.zeros([N_Q, NB], tl.float32)
    beta = tl.zeros([N_Q], tl.float32)
    for l0 in range(l_beg, l_end, BL):
        l = l0 + tl.arange(0, BL)
        lm = l < l_end
        p = tl.load(P + b * s_pb + h[:, None] * s_ph + col0 + l[None, :], mask=lm[None, :], other=0.0)
        s = tl.load(VS + b * s_vsb + l * s_vst + si, mask=lm, other=0.0).to(tl.float32)
        ps = (p * s[None, :]).to(tl.float16)
        raw = tl.load(VC + b * s_vcb + l[:, None] * s_vct + byte0 + jb[None, :],
                      mask=lm[:, None] & bm[None, :], other=0)
        acc_lo = tl.dot(ps, _e2m1_f16(raw & 15), acc_lo)
        acc_hi = tl.dot(ps, _e2m1_f16(raw >> 4), acc_hi)
        al = tl.load(AL + b * s_alb + l, mask=lm, other=0.0).to(tl.float32)
        beta += tl.sum(p * al[None, :], 1)
    zb = Z + b * s_zb + h[:, None] * s_zh + c0
    tl.atomic_add(zb + jb[None, :], acc_lo, mask=bm[None, :], sem="relaxed")
    tl.atomic_add(zb + nb + jb[None, :], acc_hi, mask=(bm & (nb + jb < w))[None, :], sem="relaxed")
    if pt == 0:
        tl.atomic_add(BETA + b * N_Q + h, beta, sem="relaxed")


KCFG = dict(backend="cuda", bt_k=128, gh=None, warps_k=8, stages_k=2, nb=128, bl=32, warps_v=4, stages_v=2,
            v_programs=2048)


def latent_logits(pack: LayerPack, q, cache, seqlen, pos0, lg, col0, **cfg):
    """Writes lg[:, :, col0:col0 + T] (the latent tokens' logits, -inf past each sequence)."""
    c = {**KCFG, **cfg}
    B, d = q.shape[0], pack.d
    T = int(cache["kc"].shape[1])
    gh, n_t = c["gh"], math.ceil(T / c["bt_k"])           # (the Triton logits kernel)
    if gh is None:                   # kv heads per logits program: enough programs to fill the GPU
        n_sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        gh = pack.n_kv
        while gh > 1 and B * n_t * (pack.n_kv // gh) < 4 * n_sm:
            gh //= 2
    kc, ks, al = cache["kc"], cache["ks"], cache["al"]
    return _latent_logits_kernel[(n_t, B, pack.n_kv // gh)](
        q, kc, ks, al, seqlen, pack.a_codes, pack.a_scales, pack.head_i, pack.head_f, pack.kbar_pair,
        *pack.cos_sin(pos0 + T), lg, T, pos0, col0, 1.0 / math.sqrt(d),
        q.stride(0), kc.stride(0), kc.stride(1), ks.stride(0), ks.stride(1), al.stride(0),
        lg.stride(0), lg.stride(1),
        REP=pack.n_q // pack.n_kv, D=d, GH=gh, BT=c["bt_k"], num_warps=c["warps_k"], num_stages=c["stages_k"])


def latent_pv(pack: LayerPack, p, cache, seqlen, col0, z, beta, **cfg):
    """Adds the latent tokens' P . V (latent) into z [B, n_q, rv] and P . alpha into beta [B, n_q]."""
    c = {**KCFG, **cfg}
    B, T = p.shape[0], int(cache["vc"].shape[1])
    vc, vs, al = cache["vc"], cache["vs"], cache["al"]
    nb_t, nb_r = pack.v_nb
    NB, BL = c["nb"], c["bl"]
    nt_top = math.ceil(nb_t / NB)
    n_tiles = nt_top + math.ceil(nb_r / NB)
    n_seg = max(1, min(math.ceil(c["v_programs"] / (B * n_tiles)), math.ceil(T / BL)))
    seg = BL * math.ceil(math.ceil(T / n_seg) / BL)
    n_seg = math.ceil(T / seg)
    return _latent_pv_kernel[(B, n_tiles, n_seg)](
        p, vc, vs, al, seqlen, z, beta, seg, col0, nb_t, nb_r, pack.v_no, pack.rv - pack.v_no, nt_top,
        p.stride(0), p.stride(1), vc.stride(0), vc.stride(1), vs.stride(0), vs.stride(1), al.stride(0),
        z.stride(0), z.stride(1),
        N_Q=pack.n_q, NB=NB, BL=BL, num_warps=c["warps_v"], num_stages=c["stages_v"])


@triton.jit
def _exact_logits_kernel(Q, KE, LG, MX, E, E_pad, qk_scale, s_qb, s_kb, s_kg, s_ke, s_lb, s_lh,
                         REP: tl.constexpr, D: tl.constexpr, EP: tl.constexpr, QP: tl.constexpr,
                         N_Q: tl.constexpr, HAS_MX: tl.constexpr):
    """Logits of the exact (bf16, rotated) tokens of one kv head into LG[:, :, :E]; -inf up to E_pad.
    HAS_MX: fold each query head's max into MX (ordered-int encoding)."""
    b = tl.program_id(0)
    g = tl.program_id(1)
    jq = tl.arange(0, QP)
    qm = jq < REP
    dd = tl.arange(0, D)
    e = tl.arange(0, EP)
    q = tl.load(Q + b * s_qb + (g * REP + jq)[:, None] * D + dd[None, :], mask=qm[:, None], other=0.0)
    k = tl.load(KE + b * s_kb + g * s_kg + e[:, None] * s_ke + dd[None, :], mask=(e < E)[:, None], other=0.0)
    sc = tl.where((e < E)[None, :], tl.dot(q, tl.trans(k)) * qk_scale, float("-inf"))
    tl.store(LG + b * s_lb + (g * REP + jq)[:, None] * s_lh + e[None, :], sc, mask=qm[:, None] & (e < E_pad)[None, :])
    if HAS_MX:
        mi = tl.max(sc, 1).to(tl.int32, bitcast=True)
        mi = tl.where(mi >= 0, mi, mi ^ 0x7FFFFFFF)
        tl.atomic_max(MX + b * N_Q + g * REP + jq, mi, mask=qm)


@triton.jit
def _exact_pv_softmax_kernel(LG, MX, VE, O, LSUM, E, s_lb, s_lh, s_vb, s_vg, s_ve, s_ob,
                             N_Q: tl.constexpr, REP: tl.constexpr, D: tl.constexpr, EC: tl.constexpr,
                             QP: tl.constexpr):
    """o[b, h] += exp(l_exact - max) . V_exact and lsum[b, h] += sum exp(l_exact - max)."""
    b = tl.program_id(0)
    g = tl.program_id(1)
    jq = tl.arange(0, QP)
    qm = jq < REP
    dd = tl.arange(0, D)
    mi = tl.load(MX + b * N_Q + g * REP + jq, mask=qm, other=0)
    m = tl.where(mi >= 0, mi, mi ^ 0x7FFFFFFF).to(tl.float32, bitcast=True)
    acc = tl.zeros([QP, D], tl.float32)
    ls = tl.zeros([QP], tl.float32)
    for e0 in range(0, E, EC):
        e = e0 + tl.arange(0, EC)
        lg = tl.load(LG + b * s_lb + (g * REP + jq)[:, None] * s_lh + e[None, :],
                     mask=qm[:, None] & (e < E)[None, :], other=float("-inf"))
        pe = tl.exp2((lg - m[:, None]) * 1.4426950408889634)
        pe = tl.where(qm[:, None] & (e < E)[None, :], pe, 0.0)
        ls += tl.sum(pe, 1)
        ve = tl.load(VE + b * s_vb + g * s_vg + e[:, None] * s_ve + dd[None, :], mask=(e < E)[:, None], other=0.0)
        acc += tl.dot(pe.to(tl.bfloat16), ve)
    o = O + b * s_ob + (g * REP + jq)[:, None] * D + dd[None, :]
    tl.store(o, tl.load(o, mask=qm[:, None], other=0.0) + acc, mask=qm[:, None])
    tl.atomic_add(LSUM + b * N_Q + g * REP + jq, ls, mask=qm)


@triton.jit
def _lift_kernel(Z, BETA, AV, VBAR, O, B, rv, r_per,
                 s_zb, s_zh, s_ob,
                 N_Q: tl.constexpr, REP: tl.constexpr, D: tl.constexpr, BB: tl.constexpr, RC: tl.constexpr):
    """o[b, g * REP + j] += A_v[g] z[b, g * REP + j] over one slice of the latent rank, for a block of
    BB sequences (rows (b, j)); bf16 MMA, fp32 accumulate. Slice 0 adds beta * vbar."""
    g = tl.program_id(0)
    pb = tl.program_id(1)
    ps = tl.program_id(2)
    M: tl.constexpr = BB * REP
    i = tl.arange(0, M)
    bb = pb * BB + i // REP
    row = bb * s_zb + (g * REP + i % REP) * s_zh
    bm = bb < B
    dd = tl.arange(0, D)
    acc = tl.zeros([M, D], tl.float32)
    r_lo = ps * r_per
    r_hi = tl.minimum(rv, r_lo + r_per)
    for r0 in range(r_lo, r_hi, RC):
        r = r0 + tl.arange(0, RC)
        rm = r < r_hi
        z = tl.load(Z + row[:, None] + r[None, :], mask=bm[:, None] & rm[None, :], other=0.0)
        a = tl.load(AV + g * D * rv + dd[:, None] * rv + r[None, :], mask=rm[None, :], other=0.0)
        acc += tl.dot(z.to(tl.bfloat16), tl.trans(a))
    if ps == 0:
        beta = tl.load(BETA + bb * N_Q + g * REP + i % REP, mask=bm, other=0.0)
        acc += beta[:, None] * tl.load(VBAR + g * D + dd)[None, :]
    orow = bb * s_ob + (g * REP + i % REP) * D
    tl.atomic_add(O + orow[:, None] + dd[None, :], acc, mask=bm[:, None], sem="relaxed")


@triton.jit
def _exact_pv_kernel(P, VE, O, E, s_pb, s_ph, s_vb, s_vg, s_ve, s_ob,
                     REP: tl.constexpr, D: tl.constexpr, EC: tl.constexpr, QP: tl.constexpr):
    """o[b, g * REP + j] += p_exact . V_exact (bf16 MMA, fp32 accumulate); runs after the lift."""
    b = tl.program_id(0)
    g = tl.program_id(1)
    jq = tl.arange(0, QP)
    qm = jq < REP
    dd = tl.arange(0, D)
    acc = tl.zeros([QP, D], tl.float32)
    for e0 in range(0, E, EC):
        e = e0 + tl.arange(0, EC)
        pe = tl.load(P + b * s_pb + (g * REP + jq)[:, None] * s_ph + e[None, :], mask=qm[:, None] & (e < E)[None, :], other=0.0)
        ve = tl.load(VE + b * s_vb + g * s_vg + e[:, None] * s_ve + dd[None, :], mask=(e < E)[:, None], other=0.0)
        acc += tl.dot(pe.to(tl.bfloat16), ve)
    o = O + b * s_ob + (g * REP + jq)[:, None] * D + dd[None, :]
    tl.store(o, tl.load(o, mask=qm[:, None], other=0.0) + acc, mask=qm[:, None])


def latent_decode(pack: LayerPack, q, cache, seqlen, pos0, exact=None, **cfg):
    """q: [B, n_q, d] bf16 (RoPE applied). seqlen: int32 [B] latent tokens per sequence, at
    positions pos0 + t. exact: (k, v) bf16 [B, n_kv, E, d], k after RoPE (the sinks and the recent
    window). -> attention output [B, n_q, d] fp32."""
    B, dev, d = q.shape[0], q.device, pack.d
    T = int(cache["kc"].shape[1])
    rep = pack.n_q // pack.n_kv
    E = 0 if exact is None else exact[0].shape[2]
    E_pad = 16 * math.ceil(E / 16)                 # latent columns start 64-byte aligned
    qk = 1.0 / math.sqrt(d)
    W = E_pad + 16 * math.ceil(T / 16)              # rows 64-byte aligned for the PV kernel's copies
    lg = torch.empty(B, pack.n_q, W, device=dev)
    if W > E_pad + T:
        lg[..., E_pad + T:] = float("-inf")
    EP = triton.next_power_of_2(E_pad) if E else 0
    if E:
        ke = exact[0]
        _exact_logits_kernel[(B, pack.n_kv)](q, ke, lg, lg, E, E_pad, qk, q.stride(0), ke.stride(0), ke.stride(1),
                                             ke.stride(2), lg.stride(0), lg.stride(1), REP=rep, D=d, EP=EP, QP=16,
                                             N_Q=pack.n_q, HAS_MX=False)
    if cfg.get("backend", "cuda") == "cuda":
        from .latent_cuda import latent_logits_cuda
        hk = latent_logits_cuda(pack, q, cache, seqlen, pos0, lg, E_pad, gh=cfg.get("gh"))
    else:
        hk = latent_logits(pack, q, cache, seqlen, pos0, lg, E_pad, **cfg)
    p = torch.softmax(lg, -1)
    beta = torch.zeros(B, pack.n_q, device=dev)
    if cfg.get("pv_backend", "cuda") == "cuda":
        from .latent_cuda import latent_pv_cuda
        z, A_v, rv = torch.zeros(B, pack.n_q, pack.zw, device=dev), pack.A_v_pad, pack.zw
        hv = latent_pv_cuda(pack, p, cache, seqlen, E_pad, z, beta, n_seg=cfg.get("n_seg"))
    else:
        z, A_v, rv = torch.zeros(B, pack.n_q, pack.rv, device=dev), pack.A_v_bf16, pack.rv
        hv = latent_pv(pack, p, cache, seqlen, E_pad, z, beta, **cfg)
    latent_decode.last = (hk, hv)
    o = torch.zeros(B, pack.n_q, d, device=dev)
    BB = 16 if B >= 16 else (8 if B >= 8 else 4)
    n_b = math.ceil(B / BB)
    n_sm = torch.cuda.get_device_properties(dev).multi_processor_count
    n_s = max(1, min(math.ceil(rv / 64), math.ceil(n_sm / (pack.n_kv * n_b))))
    r_per = 64 * math.ceil(rv / n_s / 64)
    _lift_kernel[(pack.n_kv, n_b, math.ceil(rv / r_per))](
        z, beta, A_v, pack.vbar, o, B, rv, r_per, z.stride(0), z.stride(1), o.stride(0),
        N_Q=pack.n_q, REP=rep, D=d, BB=BB, RC=64, num_warps=4)
    if E:
        ve = exact[1]
        _exact_pv_kernel[(B, pack.n_kv)](p, ve, o, E, p.stride(0), p.stride(1), ve.stride(0), ve.stride(1),
                                         ve.stride(2), o.stride(0), REP=rep, D=d, EC=32, QP=16)
    return o


def latent_decode_fused(pack: LayerPack, q, cache, seqlen, pos0, exact=None, gh=None, n_seg=None):
    """latent_decode with the lean logits kernel and the softmax fused into the PV kernel: the logits'
    row max comes from the logits kernels (atomic), the PV kernel turns logits into exp(l - max) and
    accumulates the normalizer, which the last step divides out."""
    from .latent_cuda import latent_logits3_cuda, latent_pv_cuda, latent_pv2_cuda, ORD_NEG
    B, dev, d = q.shape[0], q.device, pack.d
    T = int(cache["kc"].shape[1])
    rep = pack.n_q // pack.n_kv
    E = 0 if exact is None else exact[0].shape[2]
    E_pad = 16 * math.ceil(E / 16)
    W = E_pad + 16 * math.ceil(T / 16)
    lg = torch.empty(B, pack.n_q, W, device=dev)
    if W > E_pad + T:
        lg[..., E_pad + T:] = float("-inf")
    mx = torch.full((B, pack.n_q), ORD_NEG, dtype=torch.int32, device=dev)
    if E:
        ke = exact[0]
        _exact_logits_kernel[(B, pack.n_kv)](q, ke, lg, mx, E, E_pad, 1.0 / math.sqrt(d), q.stride(0), ke.stride(0),
                                             ke.stride(1), ke.stride(2), lg.stride(0), lg.stride(1), REP=rep, D=d,
                                             EP=triton.next_power_of_2(E_pad), QP=16, N_Q=pack.n_q, HAS_MX=True)
    latent_logits3_cuda(pack, q, cache, seqlen, pos0, lg, E_pad, mx, gh=gh)
    q2 = cache.get("v_layout") == "q2"
    zw, A_v = (pack.q2_zw, pack.A_v_q2) if q2 else (pack.zw, pack.A_v_pad)
    z = torch.zeros(B, pack.n_q, zw, device=dev)
    beta = torch.zeros(B, pack.n_q, device=dev)
    lsum = torch.zeros(B, pack.n_q, device=dev)
    if q2:
        latent_pv2_cuda(pack, lg, cache, seqlen, E_pad, mx, z, beta, lsum, n_seg=n_seg)
    else:
        latent_pv_cuda(pack, lg, cache, seqlen, E_pad, z, beta, n_seg=n_seg, mx=mx, lsum=lsum)
    o = torch.zeros(B, pack.n_q, d, device=dev)
    BB = 16 if B >= 16 else (8 if B >= 8 else 4)
    n_b = math.ceil(B / BB)
    n_sm = torch.cuda.get_device_properties(dev).multi_processor_count
    n_s = max(1, min(math.ceil(zw / 64), math.ceil(n_sm / (pack.n_kv * n_b))))
    r_per = 64 * math.ceil(zw / n_s / 64)
    _lift_kernel[(pack.n_kv, n_b, math.ceil(zw / r_per))](
        z, beta, A_v, pack.vbar, o, B, zw, r_per, z.stride(0), z.stride(1), o.stride(0),
        N_Q=pack.n_q, REP=rep, D=d, BB=BB, RC=64, num_warps=4)
    if E:
        ve = exact[1]
        _exact_pv_softmax_kernel[(B, pack.n_kv)](lg, mx, ve, o, lsum, E, lg.stride(0), lg.stride(1), ve.stride(0),
                                                 ve.stride(1), ve.stride(2), o.stride(0), N_Q=pack.n_q, REP=rep,
                                                 D=d, EC=32, QP=16)
    return o / lsum[..., None]


# --------------------------------------------------------------------------------------------
# reference (PyTorch, fp32): dequantize the cache and attend
# --------------------------------------------------------------------------------------------

def rope(pack: LayerPack, x, pos):
    """HF rotate-half RoPE with the model's bf16 cos / sin; x [B, T, n_kv, d], fp32 arithmetic."""
    cos, sin = (x_[pos].float()[None, :, None] for x_ in pack.cos_sin(int(pos.max()) + 1))   # [1, T, 1, d/2]
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


def dequant_cache(pack: LayerPack, cache, T):
    """-> pre-RoPE K [B, T, n_kv, d], V [B, T, n_kv, d] (fp32) from the stored codes."""
    al = cache["al"][:, :T].float()
    ku = e2m1_decode(unpack_nibbles(cache["kc"][:, :T]))
    ku = ku * cache["ks"][:, :T].float().repeat_interleave(16, -1)
    K = []
    for g in range(pack.n_kv):
        off, r = (int(v) for v in pack.head_i[g, :2])
        K.append(ku[..., off:off + r] @ pack.a_hat[g].T + al[..., None] * pack.kbar[g * pack.d:(g + 1) * pack.d])
    unpack = unpack_v_q2 if cache.get("v_layout") == "q2" else unpack_v_codes
    vu = e2m1_decode(unpack(pack, cache["vc"][:, :T]))
    sc = torch.where(torch.arange(pack.rv, device=vu.device) < pack.v_no,
                     cache["vs"][:, :T, :1].float(), cache["vs"][:, :T, 1:].float())
    V = torch.einsum("btr,hdr->bthd", vu * sc, pack.A_v) + al[..., None, None] * pack.vbar.view(pack.n_kv, pack.d)
    return torch.stack(K, 2), V


def reference_decode(pack: LayerPack, q, cache, T, pos0, exact=None):
    """Attention of q [B, n_q, d] over the exact tokens and T latent tokens (positions pos0..)."""
    K, V = dequant_cache(pack, cache, T)
    K = rope(pack, K, torch.arange(T, device=q.device) + pos0).to(torch.bfloat16).float()
    if exact is not None:                                   # exact: [B, n_kv, E, d]
        K = torch.cat([exact[0].float().transpose(1, 2), K], 1)
        V = torch.cat([exact[1].float().transpose(1, 2), V], 1)
    rep = pack.n_q // pack.n_kv
    K, V = K.repeat_interleave(rep, 2), V.repeat_interleave(rep, 2)
    s = torch.einsum("bhd,bthd->bht", q.float(), K) / math.sqrt(pack.d)
    return torch.einsum("bht,bthd->bhd", s.softmax(-1), V)
