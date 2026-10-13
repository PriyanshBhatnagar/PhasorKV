"""CUDA (sm_120a) logits kernel for the latent K cache: host-side operand preparation and launch.

prepare(pack) re-lays a LayerPack's static data for csrc/latent_logits.cu: the NVFP4 rebuild
weights (rows already in RoPE-pair order) in mma B-fragment order, their scale words likewise,
kbar and the bf16 cos / sin table in the order each thread reads its 16 RoPE pairs (p = 4 nt + tq).
"""
import math
import os

import torch

from .cubin import CubinModule

_SRC = os.path.join(os.path.dirname(__file__), "csrc", "latent_logits.cu")         # 16 tokens / warp
_SRC2 = os.path.join(os.path.dirname(__file__), "csrc", "latent_logits2.cu")       # 32 tokens / warp
_SRC_PV = os.path.join(os.path.dirname(__file__), "csrc", "latent_pv.cu")
NT, LANES = 16, 32
PAIR_PERM = [4 * nt + tq for tq in range(4) for nt in range(NT)]      # column tq * 16 + nt -> pair


_MODS = {}


def module(defines=(), wide=False):
    key = (wide, *defines)
    if key not in _MODS:
        _MODS[key] = CubinModule(_SRC2 if wide else _SRC, flags=("-O3", "-std=c++17", *(f"-D{d}" for d in defines)))
    return _MODS[key]


def prepare(pack):
    """-> dict of device tensors for the kernel (cached on the pack)."""
    if getattr(pack, "_cuda", None) is not None:
        return pack._cuda
    d, dev = pack.d, pack.a_codes.device
    assert d == 128 and pack.n_q // pack.n_kv == 4, "kernel is specialized for head dim 128, 4 query heads per kv head"
    head = pack.head_i.clone().cpu()
    afr, sfr, fo, so = [], [], 0, 0
    for g in range(pack.n_kv):
        lat, r, co, sco = (int(v) for v in pack.head_i[g])
        if r == 0:
            continue
        nch = math.ceil(r / 64)
        codes = pack.a_codes[co:co + d * nch * 32].view(NT, 8, nch, 2, 4, 4)       # nt, g, c, half, tq, byte
        w = codes.permute(2, 0, 1, 4, 3, 5).contiguous().view(torch.int32)          # c, nt, g, tq, half
        afr.append(w.flatten())
        sc = pack.a_scales[sco:sco + d * nch * 4].view(torch.uint8).view(NT, 8, nch, 4)
        sw = sc.permute(2, 0, 1, 3).contiguous().view(torch.int32)                  # c, nt, g
        sfr.append(sw.expand(nch, NT, 8, 4).contiguous().flatten())                 # every tq of the quad
        head[g, 2], head[g, 3] = fo, so
        fo += nch * NT * LANES
        so += nch * NT * LANES
    kb = pack.kbar.view(pack.n_kv, d)
    p = torch.tensor(PAIR_PERM, device=kb.device).view(4, NT)
    kbar_t = torch.stack([kb[:, p], kb[:, p + d // 2]], -1).contiguous()           # [n_kv, tq, nt, 2]
    # heads without a latent read one all-zero step (their codes are masked to zero anyway)
    for g in range(pack.n_kv):
        if int(pack.head_i[g, 1]) == 0:
            head[g, 2], head[g, 3] = fo, so
    afr.append(torch.zeros(NT * LANES * 2, dtype=torch.int32, device=dev))
    sfr.append(torch.zeros(NT * LANES, dtype=torch.int32, device=dev))
    pack._cuda = dict(afrag=torch.cat(afr), sfrag=torch.cat(sfr),
                      head_i=head.to(dev).contiguous(), kbar=kbar_t.float(), cs=None)
    return pack._cuda


def cos_sin(pack, n_pos):
    c = prepare(pack)
    if c["cs"] is None or c["cs"][0].shape[0] < n_pos:
        cos, sin = pack.cos_sin(n_pos)
        perm = torch.tensor(PAIR_PERM, device=cos.device)
        c["cs"] = (cos[:, perm].contiguous(), sin[:, perm].contiguous())
    return c["cs"]


def latent_logits_cuda(pack, q, cache, seqlen, pos0, lg, col0, gh=None, defines=(), wide=False):
    """Same contract as latent_attn.latent_logits."""
    c = prepare(pack)
    B, T = q.shape[0], int(cache["kc"].shape[1])
    n_t = math.ceil(T / (128 if wide else 64))
    if gh is None:
        n_sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        gh = min(pack.n_kv, 8)
        while gh > 1 and B * n_t * (pack.n_kv // gh) < 4 * n_sm:
            gh //= 2
    cos, sin = cos_sin(pack, pos0 + T)
    kc, ks, al = cache["kc"], cache["ks"], cache["al"]
    i64 = lambda v: ("i64", int(v))
    module(defines, wide).launch("latent_logits", (n_t, B, pack.n_kv // gh), (128, 1, 1), [
        q, kc, ks, al, seqlen, c["afrag"], c["sfrag"], c["head_i"], pack.head_f, c["kbar"], cos, sin, lg,
        T, pos0, col0, 1.0 / math.sqrt(pack.d), gh,
        i64(q.stride(0)), i64(kc.stride(0)), i64(kc.stride(1)), i64(ks.stride(0)), i64(ks.stride(1)),
        i64(al.stride(0)), i64(lg.stride(0)), i64(lg.stride(1))])


PV_CFG = dict(BT=32, NWARP=8, NSTAGE=3)
_MOD_PV = {}


def module_pv(cfg):
    key = tuple(sorted(cfg.items()))
    if key not in _MOD_PV:
        _MOD_PV[key] = CubinModule(_SRC_PV, flags=("-O3", "-std=c++17", *(f"-D{k}={v}" for k, v in key)))
    return _MOD_PV[key]


def latent_pv_cuda(pack, p, cache, seqlen, col0, z, beta, n_seg=None, mx=None, lsum=None, **cfg):
    """latent_attn.latent_pv on the tensor cores; z is [B, n_q, pack.zw] (padded tier columns, see
    csrc/latent_pv.cu) and p's width must cover col0 + 16 * ceil(T / 16)."""
    B, T = p.shape[0], int(cache["vc"].shape[1])
    vc, vs, al = cache["vc"], cache["vs"], cache["al"]
    nb_t, nb_r = pack.v_nb
    assert pack.n_q == 32 and nb_t + nb_r <= 8 * 72 and z.shape[2] == pack.zw
    assert p.shape[2] >= col0 + 16 * math.ceil(T / 16)
    c = {**PV_CFG, **cfg}
    bt, nst = c["BT"], c["NSTAGE"]
    if n_seg is None:
        n_sm = torch.cuda.get_device_properties(p.device).multi_processor_count
        n_seg = max(1, min(math.ceil(n_sm / B), math.ceil(T / (4 * bt))))
    seg = bt * math.ceil(math.ceil(T / n_seg) / bt)
    n_seg = math.ceil(T / seg)
    v_row = vc.shape[2]
    smem = nst * (bt * (v_row + 16) + 32 * (bt + 4) * 4 + bt * 4 + bt * 4)
    i64 = lambda v: ("i64", int(v))
    module_pv(c).launch("latent_pv", (n_seg, B, 1), (32 * c["NWARP"], 1, 1), [
        p, vc, vs, al, seqlen, z, beta, mx if mx is not None else ("i64", 0),
        lsum if lsum is not None else ("i64", 0), col0, seg, nb_t, nb_r, v_row, pack.zw,
        i64(p.stride(0)), i64(p.stride(1)), i64(vc.stride(0)), i64(vc.stride(1)), i64(vs.stride(0)),
        i64(al.stride(0)), i64(z.stride(0))], smem=smem)


# ---- lean-epilogue logits kernel (csrc/latent_logits3.cu) ------------------------------------------
_SRC3 = os.path.join(os.path.dirname(__file__), "csrc", "latent_logits3.cu")
_MODS3 = {}


def prepare3(pack):
    """Operands of latent_logits3.cu: compact scale words [step][nt][8 rows] and the rank-1 kbar / ga
    B words [n_kv][nt][8] (bf16 in the low half), on top of prepare()."""
    c = prepare(pack)
    if "sfrag3" in c:
        return c
    d, dev = pack.d, pack.a_codes.device
    sfr, so, head = [], 0, c["head_i"].clone()
    for g in range(pack.n_kv):
        r, sco = int(pack.head_i[g, 1]), int(pack.head_i[g, 3])
        if r == 0:
            continue
        nch = math.ceil(r / 64)
        sc = pack.a_scales[sco:sco + d * nch * 4].view(torch.uint8).view(NT, 8, nch, 4)
        sfr.append(sc.permute(2, 0, 1, 3).contiguous().view(torch.int32).flatten())   # c, nt, g
        head[g, 3] = so
        so += nch * NT * 8
    for g in range(pack.n_kv):
        if int(pack.head_i[g, 1]) == 0:
            head[g, 3] = so
    sfr.append(torch.zeros(NT * 8, dtype=torch.int32, device=dev))
    kb = pack.kbar_pair / pack.head_f[:, None]                                      # [n_kv, 128] pair order
    hi = kb.to(torch.bfloat16)
    lo = (kb - hi.float()).to(torch.bfloat16)
    u16 = lambda x: x.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)
    c["kbw3"] = (u16(hi) | (u16(lo) << 16)).contiguous()                              # [n_kv, NT * 8]
    c["sfrag3"] = torch.cat(sfr)
    c["head3"] = head.contiguous()
    # WIDEB layouts: afrag [step][NT][32][2] -> [step][NT/2][32][4]; sfrag3 [step][NT][8] -> [step][NT/4][8][4]
    af = c["afrag"].view(-1, NT // 2, 2, LANES, 2).permute(0, 1, 3, 2, 4).contiguous()
    c["afrag4"] = af.flatten()
    sf = c["sfrag3"].view(-1, NT // 4, 4, 8).permute(0, 1, 3, 2).contiguous()
    c["sfrag4"] = sf.flatten()
    return c


def module3(minb=2, rank1=True, prefetch=False, defines=()):
    key = (minb, rank1, prefetch, *defines)
    if key not in _MODS3:
        _MODS3[key] = CubinModule(_SRC3, flags=("-O3", "-std=c++17", f"-DMINB={minb}",
                                                *(("-DRANK1",) if rank1 else ()), *(("-DPREFETCH",) if prefetch else ()),
                                                *(f"-D{d}" for d in defines)))
    return _MODS3[key]


ORD_NEG = -2 ** 31                       # ordered-int encoding below every float


def decode_ord(m):
    """int32 tensor of ordered-int maxima -> float32."""
    return torch.where(m >= 0, m, m ^ 0x7FFFFFFF).view(torch.float32)


def latent_logits3_cuda(pack, q, cache, seqlen, pos0, lg, col0, mx=None, gh=None, minb=2, rank1=True, prefetch=False,
                        defines=(), nw=4):
    """latent_logits_cuda with the lean epilogue; mx (int32 [B, n_q], ORD_NEG-initialized) receives
    the running max of the written logits."""
    c = prepare3(pack)
    B, T = q.shape[0], int(cache["kc"].shape[1])
    n_t = math.ceil(T / (16 * nw))
    if gh is None:
        n_sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        gh = min(pack.n_kv, 8)
        while gh > 1 and B * n_t * (pack.n_kv // gh) < 4 * n_sm:
            gh //= 2
    cos, sin = cos_sin(pack, pos0 + T)
    kc, ks, al = cache["kc"], cache["ks"], cache["al"]
    i64 = lambda v: ("i64", int(v))
    null = torch.empty(0, dtype=torch.int32, device=q.device)
    if "QPAIR" in defines:                           # pair words (q[p] | q[p + 64] << 16), [B, n_q, 64]
        qu = q.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        q = (qu[..., :64] | (qu[..., 64:] << 16)).contiguous()
    smem = 0
    if "A_SMEM" in defines:
        smem = nw * 16 * (pack.rk // 2 + 16) + nw * 16 * (((pack.rk // 16 + 3) & ~3) + 4)
    module3(minb, rank1, prefetch, (*defines, f"NW={nw}")).launch("latent_logits", (n_t, B, pack.n_kv // gh), (32 * nw, 1, 1), [
        q, kc, ks, al, seqlen, c["afrag"], c["sfrag3"], c["afrag4"], c["sfrag4"], c["head3"], pack.head_f, c["kbw3"],
        c["kbar"], cos, sin, lg, mx if mx is not None else ("i64", 0), T, pos0, col0, 1.0 / math.sqrt(pack.d), gh, pack.rk,
        i64(q.stride(0)), i64(kc.stride(0)), i64(kc.stride(1)), i64(ks.stride(0)), i64(ks.stride(1)),
        i64(al.stride(0)), i64(lg.stride(0)), i64(lg.stride(1))], smem=smem)


# ---- PV with the rest tier at 2 bits and the softmax fused (csrc/latent_pv2.cu) ---------------------
_SRC_PV2 = os.path.join(os.path.dirname(__file__), "csrc", "latent_pv2.cu")
_MOD_PV2 = {}


def latent_pv2_cuda(pack, lg, cache, seqlen, col0, mx, z, beta, lsum, n_seg=None, bt=64, nstage=2, defines=(), ng=1):
    """z: [B, n_q, zw] or [B, ncopy, n_q, zw] (segments spread over the copies; the caller sums them)."""
    """z [B, 32, pack.q2_zw] += exp(lg - max) . V (q2 cache), beta, lsum likewise."""
    B, T = lg.shape[0], int(cache["vc"].shape[1])
    vc, vs, al = cache["vc"], cache["vs"], cache["al"]
    q_t, q_r = pack.q2_nb
    assert pack.n_q == 32 and q_t <= 128 * ng and q_r <= 256 * ng and z.shape[-1] == pack.q2_zw
    ncopy, s_zc = (z.shape[1], z.stride(1)) if z.dim() == 4 else (1, 0)
    assert lg.shape[2] >= col0 + 16 * math.ceil(T / 16)
    key = (bt, nstage, *defines)
    if key not in _MOD_PV2:
        _MOD_PV2[key] = CubinModule(_SRC_PV2, flags=("-O3", "-std=c++17", f"-DBT={bt}", f"-DNSTAGE={nstage}",
                                                     *(f"-D{d}" for d in defines)))
    if n_seg is None:
        n_sm = torch.cuda.get_device_properties(lg.device).multi_processor_count
        n_seg = max(1, min(math.ceil(n_sm / (B * ng)), math.ceil(T / (4 * bt))))
    seg = bt * math.ceil(math.ceil(T / n_seg) / bt)
    n_seg = math.ceil(T / seg)
    v_row = vc.shape[2]
    smem = nstage * (bt * (v_row + 16) + 32 * (bt + 4) * 4 + bt * 4 + bt * 4)
    i64 = lambda v: ("i64", int(v))
    _MOD_PV2[key].launch("latent_pv2", (ng, n_seg, B), (512, 1, 1), [
        lg, vc, vs, al, seqlen, mx, z, beta, lsum, col0, seg, q_t, q_r, v_row, pack.q2_zw, ng, ncopy, ("i64", s_zc),
        i64(lg.stride(0)), i64(lg.stride(1)), i64(vc.stride(0)), i64(vc.stride(1)), i64(vs.stride(0)),
        i64(al.stride(0)), i64(z.stride(0))], smem=smem)


# ---- head halves in ping-pong (csrc/latent_logits4.cu) ------------------------------------------------
_SRC4 = os.path.join(os.path.dirname(__file__), "csrc", "latent_logits4.cu")
_MODS4 = {}


def latent_logits4_cuda(pack, q, cache, seqlen, pos0, lg, col0, mx=None, gh=None, minb=3):
    c = prepare3(pack)
    B, T = q.shape[0], int(cache["kc"].shape[1])
    n_t = math.ceil(T / 64)
    if gh is None:
        n_sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        gh = min(pack.n_kv, 8)
        while gh > 1 and B * n_t * (pack.n_kv // gh) < 4 * n_sm:
            gh //= 2
    if minb not in _MODS4:
        _MODS4[minb] = CubinModule(_SRC4, flags=("-O3", "-std=c++17", f"-DMINB={minb}"))
    cos, sin = cos_sin(pack, pos0 + T)
    kc, ks, al = cache["kc"], cache["ks"], cache["al"]
    i64 = lambda v: ("i64", int(v))
    _MODS4[minb].launch("latent_logits", (n_t, B, pack.n_kv // gh), (128, 1, 1), [
        q, kc, ks, al, seqlen, c["afrag"], c["sfrag3"], c["head3"], pack.head_f, c["kbw3"], c["kbar"], cos, sin, lg,
        mx if mx is not None else ("i64", 0), T, pos0, col0, 1.0 / math.sqrt(pack.d), gh,
        i64(q.stride(0)), i64(kc.stride(0)), i64(kc.stride(1)), i64(ks.stride(0)), i64(ks.stride(1)),
        i64(al.stride(0)), i64(lg.stride(0)), i64(lg.stride(1))])


# ---- 32 tokens per warp (csrc/latent_logits5.cu) -------------------------------------------------------
_SRC5 = os.path.join(os.path.dirname(__file__), "csrc", "latent_logits5.cu")
_MODS5 = {}


def latent_logits5_cuda(pack, q, cache, seqlen, pos0, lg, col0, mx=None, gh=None, minb=2, nw=4):
    c = prepare3(pack)
    B, T = q.shape[0], int(cache["kc"].shape[1])
    n_t = math.ceil(T / (32 * nw))
    if gh is None:
        n_sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        gh = min(pack.n_kv, 8)
        while gh > 1 and B * n_t * (pack.n_kv // gh) < 2 * n_sm:
            gh //= 2
    key = (minb, nw)
    if key not in _MODS5:
        _MODS5[key] = CubinModule(_SRC5, flags=("-O3", "-std=c++17", f"-DMINB={minb}", f"-DNW={nw}"))
    cos, sin = cos_sin(pack, pos0 + T)
    kc, ks, al = cache["kc"], cache["ks"], cache["al"]
    i64 = lambda v: ("i64", int(v))
    _MODS5[key].launch("latent_logits", (n_t, B, pack.n_kv // gh), (32 * nw, 1, 1), [
        q, kc, ks, al, seqlen, c["afrag"], c["sfrag3"], c["head3"], pack.head_f, c["kbw3"], cos, sin, lg,
        mx if mx is not None else ("i64", 0), T, pos0, col0, 1.0 / math.sqrt(pack.d), gh,
        i64(q.stride(0)), i64(kc.stride(0)), i64(kc.stride(1)), i64(ks.stride(0)), i64(ks.stride(1)),
        i64(al.stride(0)), i64(lg.stride(0)), i64(lg.stride(1))])
