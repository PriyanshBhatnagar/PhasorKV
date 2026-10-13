"""Baseline decode attention over an uncompressed (bf16) or FP8 (E4M3, per-tensor scale) KV cache:
split-K flash decoding in Triton, the same structure as the latent kernel, so the comparison
isolates the cache format. One program per (sequence, token split, kv head) with that head's
query heads padded to 16 MMA rows."""
import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _dense_decode_kernel(Q, K, V, SEQ, OM, OL, OO, tok_per_split, qk_scale, k_scale, v_scale,
                         s_qb, s_kb, s_kt, s_kh, s_vb, s_vt, s_vh,
                         N_SPLIT: tl.constexpr, N_Q: tl.constexpr, REP: tl.constexpr, D: tl.constexpr,
                         HP: tl.constexpr, BLOCK_T: tl.constexpr):
    b = tl.program_id(0)
    sp = tl.program_id(1)
    g = tl.program_id(2)
    T = tl.load(SEQ + b)
    t_lo = sp * tok_per_split
    t_hi = tl.minimum(T, t_lo + tok_per_split)
    r = tl.arange(0, HP)
    rm = r < REP
    dd = tl.arange(0, D)
    q = tl.load(Q + b * s_qb + (g * REP + r[:, None]) * D + dd[None, :], mask=rm[:, None], other=0.0)
    m_i = tl.full([HP], float("-inf"), tl.float32)
    l_i = tl.zeros([HP], tl.float32)
    acc = tl.zeros([HP, D], tl.float32)
    for t0 in range(t_lo, t_hi, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)
        tm = t < t_hi
        k = tl.load(K + b * s_kb + t[:, None] * s_kt + g * s_kh + dd[None, :], mask=tm[:, None], other=0.0)
        k = (k.to(tl.float32) * k_scale).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k))
        s = tl.where(tm[None, :], s * qk_scale, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        corr = tl.exp2(m_i - m_new)
        l_i = l_i * corr + tl.sum(p, 1)
        v = tl.load(V + b * s_vb + t[:, None] * s_vt + g * s_vh + dd[None, :], mask=tm[:, None], other=0.0)
        v = (v.to(tl.float32) * v_scale).to(tl.bfloat16)
        acc = acc * corr[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = (b * N_SPLIT + sp) * N_Q + g * REP + r
    tl.store(OM + o, m_i, mask=rm)
    tl.store(OL + o, l_i, mask=rm)
    tl.store(OO + o[:, None] * D + dd[None, :], acc, mask=rm[:, None])


def dense_decode(q, k, v, seqlen, k_scale=1.0, v_scale=1.0, block_t=64, num_warps=4, num_stages=2,
                 n_split=None):
    """q [B, n_q, d] bf16; k, v [B, T, n_kv, d] bf16 or float8_e4m3fn (value = stored * scale).
    -> [B, n_q, d] fp32."""
    B, n_q, d = q.shape
    T, n_kv = k.shape[1], k.shape[2]
    if n_split is None:
        n_sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        n_split = max(1, min(math.ceil(2 * n_sm / (B * n_kv)), math.ceil(T / (4 * block_t))))
    per = block_t * math.ceil(math.ceil(T / n_split) / block_t)
    n_split = math.ceil(T / per)
    om = torch.empty(B, n_split, n_q, device=q.device)
    ol = torch.empty_like(om)
    oo = torch.empty(B, n_split, n_q, d, device=q.device)
    _dense_decode_kernel[(B, n_split, n_kv)](
        q, k, v, seqlen, om, ol, oo, per, LOG2E / math.sqrt(d), k_scale, v_scale,
        q.stride(0), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
        N_SPLIT=n_split, N_Q=n_q, REP=n_q // n_kv, D=d, HP=max(16, triton.next_power_of_2(n_q // n_kv)),
        BLOCK_T=block_t, num_warps=num_warps, num_stages=num_stages)
    M = om.amax(1, keepdim=True)
    w = torch.exp2(om - M)
    return (w[..., None] * oo).sum(1) / (w * ol).sum(1)[..., None]
