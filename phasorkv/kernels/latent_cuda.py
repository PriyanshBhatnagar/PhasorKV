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


def latent_pv_cuda(pack, p, cache, seqlen, col0, z, beta, n_seg=None, **cfg):
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
        p, vc, vs, al, seqlen, z, beta, col0, seg, nb_t, nb_r, v_row, pack.zw,
        i64(p.stride(0)), i64(p.stride(1)), i64(vc.stride(0)), i64(vc.stride(1)), i64(vs.stride(0)),
        i64(al.stride(0)), i64(z.stride(0))], smem=smem)
