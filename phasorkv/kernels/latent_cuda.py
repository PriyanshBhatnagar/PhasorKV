"""CUDA (sm_120a) logits kernel for the latent K cache: host-side operand preparation and launch.

prepare(pack) re-lays a LayerPack's static data for csrc/latent_logits.cu: the NVFP4 rebuild
weights (rows already in RoPE-pair order) in mma B-fragment order, their scale words likewise,
kbar and the bf16 cos / sin table in the order each thread reads its 16 RoPE pairs (p = 4 nt + tq).
"""
import math
import os

import torch

from .cubin import CubinModule

_SRC = os.path.join(os.path.dirname(__file__), "csrc", "latent_logits.cu")
_MOD = None
NT, LANES = 16, 32
PAIR_PERM = [4 * nt + tq for tq in range(4) for nt in range(NT)]      # column tq * 16 + nt -> pair


def module():
    global _MOD
    if _MOD is None:
        _MOD = CubinModule(_SRC)
    return _MOD


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
            head[g, 2:] = 0
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
    z = lambda dt: torch.zeros(2, dtype=dt, device=dev)
    pack._cuda = dict(afrag=torch.cat(afr) if afr else z(torch.int32),
                      sfrag=torch.cat(sfr) if sfr else z(torch.int32),
                      head_i=head.to(dev).contiguous(), kbar=kbar_t.float(), cs=None)
    return pack._cuda


def cos_sin(pack, n_pos):
    c = prepare(pack)
    if c["cs"] is None or c["cs"][0].shape[0] < n_pos:
        cos, sin = pack.cos_sin(n_pos)
        perm = torch.tensor(PAIR_PERM, device=cos.device)
        c["cs"] = (cos[:, perm].contiguous(), sin[:, perm].contiguous())
    return c["cs"]


def latent_logits_cuda(pack, q, cache, seqlen, pos0, lg, col0, gh=None):
    """Same contract as latent_attn.latent_logits."""
    c = prepare(pack)
    B, T = q.shape[0], int(cache["kc"].shape[1])
    n_t = math.ceil(T / 64)
    if gh is None:
        n_sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        gh = pack.n_kv
        while gh > 1 and B * n_t * (pack.n_kv // gh) < 4 * n_sm:
            gh //= 2
    cos, sin = cos_sin(pack, pos0 + T)
    kc, ks, al = cache["kc"], cache["ks"], cache["al"]
    i64 = lambda v: ("i64", int(v))
    module().launch("latent_logits", (n_t, B, pack.n_kv // gh), (128, 1, 1), [
        q, kc, ks, al, seqlen, c["afrag"], c["sfrag"], c["head_i"], pack.head_f, c["kbar"], cos, sin, lg,
        T, pos0, col0, 1.0 / math.sqrt(pack.d), gh,
        i64(q.stride(0)), i64(kc.stride(0)), i64(kc.stride(1)), i64(ks.stride(0)), i64(ks.stride(1)),
        i64(al.stride(0)), i64(lg.stride(0)), i64(lg.stride(1))])
