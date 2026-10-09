"""KV-cache compression methods as per-layer patches, for fake-quant evaluation.

Latent methods (all on objective B's bases, see bases.py) replace k_proj/v_proj
by x -> A Q(B x): the cache holds the latent, Q is the quantizer, and k is still
rotated by RoPE afterwards (pre-RoPE latent, as STAR-KV caches). Units are 32
consecutive latent dims (one head's for K, the joint latent's for V); every
unit gets one format, chosen per layer by alloc.allocate on distortions
measured on calibration latents (prep_latent.py).

  lr       rank only, bf16 latent                        (B alone)
  lrnv     rank + uniform NVFP4, latent unrotated        (naive combination)
  palu     rank + Hadamard over the kept latent + NVFP4 per token (Palu-style)
  starq    rank + STAR-KV's quantizer: per head, leading 20% int4 / rest int3,
           one scale per token per block, block Hadamard per tier
  ours     menu {mxfp8, mxfp6, nvfp4, drop}: rank and precision from one
           allocation. Each K unit may also be rotated by a 32-wide Hadamard
           (inside the unit, so the energy order across units survives); the
           allocator decides per unit, since on a sorted latent the rotation
           helps some units and hurts others. V is quantized along tokens, the
           P.V reduction axis, so every tier maps to a block-scaled MMA
  ours_H   every K unit rotated;  ours_noH  none rotated
  ours_ext ours + {nvint3, nvint2} (formats a custom datapath could add)

Post-RoPE methods quantize the ordinary cache inside attention:
  fp8 (per token, E4M3), nvfp4, mxfp4, nvfp4H (Hadamard-128 per head, QuaRot
  style), kivi4 / kivi2 (asymmetric, group 32, K per channel, V per token).

Every method fake-quantizes all positions, prefill included, and no method
keeps a full-precision window of recent tokens.
"""
import math
import types

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from . import quant
from .alloc import allocate

UNIT = 32
MENU_ALL = ["bf16", "mxfp8", "mxfp6", "nvfp4", "mxfp4", "nvint4", "nvint3", "nvint2", "drop"]
MENUS = {
    "lr": ["bf16", "drop"],
    "lrnv": ["nvfp4", "drop"],
    "palu": ["nvfp4", "drop"],
    "starq": ["bf16", "drop"],              # cost overridden: 3.2 bits per kept dim
    "ours": ["mxfp8", "mxfp6", "nvfp4", "drop"],
    "ours_H": ["mxfp8", "mxfp6", "nvfp4", "drop"],
    "ours_noH": ["mxfp8", "mxfp6", "nvfp4", "drop"],
    "ours_ext": ["mxfp8", "mxfp6", "nvfp4", "nvint3", "nvint2", "drop"],
    # diagnostics: starq's ranks, a different quantizer
    "diag_bf16": ["bf16", "drop"], "diag_nvlat": ["bf16", "drop"], "diag_nvtok": ["bf16", "drop"],
    "diag_nvH": ["bf16", "drop"], "diag_nvtierH": ["bf16", "drop"], "diag_int4": ["bf16", "drop"],
    "diag_mx6": ["bf16", "drop"], "diag_mx6tierH": ["bf16", "drop"],
}
# rd family: rate-distortion allocation of rank + per-tier formats, one Hadamard per
# tier (contiguous kept units sharing a format), quantization noise discounted by
# gamma. Name: rd[x][.g<gamma>][.vl]@<compression>; rdx adds INT4/INT3 tiers.
MENUS["rd"] = ["mxfp8", "mxfp6", "nvfp4", "drop"]
MENUS["rdx"] = ["mxfp8", "mxfp6", "nvfp4", "nvint4", "nvint3", "drop"]
TIERED = {   # kind: (K top, K rest, V top, V rest)
    "starq": ("i4t", "i3t", "i4t", "i3t"),
    "tqA": ("nvint4", "nvint3", "i4t", "i3t"),
    "tqB": ("nvfp4", "nvint3", "i4t", "i3t"),
    "tqC": ("nvint4", "nvint3", "nvint4", "nvint3"),
    "tqD": ("mxfp6", "nvfp4", "i4t", "i3t"),
    "tqE": ("nvfp4", "nvfp4", "i4t", "i3t"),
    # lower precision: rest tier at 2 bits, or 2 bits everywhere
    "t42": ("nvint4", "nvint2", "i4t", "i2t"),
    "tqF": ("nvfp4", "nvint2", "i4t", "i2t"),          # native-FP4 4/2 (INT2 levels are FP4 values)
    "t22": ("nvint2", "nvint2", "i2t", "i2t"),
    # static per-channel scales: no per-token or per-group metadata at all
    "st43": ("s4", "s3", "s4", "s3"),
    "st42": ("s4", "s2", "s4", "s2"),
    # one per-token scale (bf16) times a static per-channel step: the token's RMS
    # carries the token-to-token variation, the offline step the per-channel shape
    "nt43g": ("n4", "n3", "n4", "n3"),     # one scale per token for all of K, one for V
    "nt43h": ("n4", "n3", "n4", "n3"),     # one per token per K head, one for V
    "nt42h": ("n4", "n2", "n4", "n2"),
    "nt42g": ("n4", "n2", "n4", "n2"),
}
TOKEN_SCALE = {"nt43g": "layer", "nt43h": "head", "nt42h": "head", "nt42g": "layer"}


def scale_overhead(kind, n_kv):
    """Per-token scale bits for (K, V) per layer: STAR-KV-style 't' tiers carry two bf16
    scales per head (K) and two for V; 'n' formats one per token (per head or per layer)."""
    if kind not in TIERED:
        return 0, 0
    kt, kr, vt, vr = TIERED[kind]
    if kind in TOKEN_SCALE:
        return (16 * n_kv if TOKEN_SCALE[kind] == "head" else 16), 16
    k = 32 * n_kv if (kt.endswith("t") or kr.endswith("t")) else 0
    v = 32 if (vt.endswith("t") or vr.endswith("t")) else 0
    return k, v
for _k in TIERED:
    MENUS.setdefault(_k, ["bf16", "drop"])


def tier_bits(fmt):
    return {"i4t": 4.0, "i3t": 3.0, "i2t": 2.0, "s4": 4.0, "s3": 3.0, "s2": 2.0,
            "n4": 4.0, "n3": 3.0, "n2": 2.0}.get(fmt) or quant.bits(fmt)


# Step of the MSE-optimal uniform quantizer for a unit Gaussian (Max, 1960), 2^b levels.
GAUSS_STEP = {4: 0.3352, 3: 0.5860, 2: 0.9957}


def static_quant(x, var, bits, widen=1.0):
    """Uniform midrise quantizer with a fixed step per channel, Delta_j = c_b * sigma_j.

    Objective B's latent has a known, token-independent variance per dimension (its
    eigenvalue; after the mean split it is zero-mean), so the step is set offline
    and folds into the reconstruction factors: the cache holds bare integer codes,
    with no per-token or per-group scales."""
    step = widen * GAUSS_STEP[bits] * var.clamp_min(1e-30).sqrt()
    L = 2 ** (bits - 1)
    return (torch.floor(x / step) + 0.5).clamp(-L + 0.5, L - 0.5) * step


def int2_per_token(x):
    """2-bit symmetric, levels +-{0.5, 1.5} x (amax / 1.5), one scale per vector."""
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-5) / 1.5
    return quant.round_to(x / s, "int2") * s


TIER_H = ("starq", "diag_nvtierH", "diag_int4", "diag_mx6tierH") + tuple(TIERED)     # STAR-KV's per-tier Hadamard
ADAPTIVE_ROT = ("ours", "ours_ext")
POST_ROPE_BITS = {"fp8": 8 + 16 / 128, "nvfp4": 4.5, "mxfp4": 4.25, "nvfp4H": 4.5,
                  "kivi4": 5.0, "kivi2": 3.0}
STARQ_OUT, STARQ_BITS = 0.2, (4, 3)


def k_rotations(n_kv, d, seed=1000):
    return [quant.block_diag_hadamard(d, UNIT, seed + 97 * h).float() for h in range(n_kv)]


def _pads_to_zero(c, lengths):
    if lengths is None:
        return c
    t = torch.arange(c.shape[1], device=c.device)
    return c * (t[None, :] < lengths.to(c.device)[:, None])[..., None]


def starq_blocks(r, gen, no=None):
    """STAR-KV's per-block Hadamard (fold_kv_hadamard) and tier split for one latent of width r
    (top tier = leading int(0.2 r) dims unless `no` is given)."""
    no = int(STARQ_OUT * r) if no is None else no
    T = torch.eye(r, dtype=torch.float64)
    for a, n in ((0, no), (no, r - no)):
        if n <= 1:
            continue
        m = 1 << (n.bit_length() - 1)
        H = quant.hadamard(m, seed=None)
        signs = torch.where(torch.rand(m, generator=gen) < 0.5, -1.0, 1.0).to(torch.float64)
        T[a:a + m, a:a + m] = signs[:, None] * H
    return T.T, no          # latent' = T^T latent (VS <- T^T VS, U <- U T)


def group_blocks(r, g, gen):
    """Like starq_blocks, but one random-sign Hadamard per scale group of g consecutive
    dims inside each tier (aligned with the block-g scales of NV formats); a trailing
    partial group rotates its largest power-of-two part."""
    no = int(STARQ_OUT * r)
    T = torch.eye(r, dtype=torch.float64)
    for a, n in ((0, no), (no, r - no)):
        for s in range(a, a + n, g):
            w = min(g, a + n - s)
            m = 1 << (w.bit_length() - 1)
            if m <= 1:
                continue
            signs = torch.where(torch.rand(m, generator=gen) < 0.5, -1.0, 1.0).to(torch.float64)
            T[s:s + m, s:s + m] = signs[:, None] * quant.hadamard(m, seed=None)
    return T.T, no


def tier_rotation(fmts, gen):
    """Block-diagonal rotation over the kept latent: one random-sign Hadamard per tier
    (a run of consecutive units with the same format), on its largest power-of-two
    leading part, identity on the rest."""
    widths = []
    for f in fmts:
        if widths and widths[-1][0] == f:
            widths[-1][1] += UNIT
        else:
            widths.append([f, UNIT])
    blocks = []
    for _, w in widths:
        T = torch.eye(w, dtype=torch.float64)
        m = 1 << (w.bit_length() - 1)
        signs = torch.where(torch.rand(m, generator=gen) < 0.5, -1.0, 1.0).to(torch.float64)
        T[:m, :m] = signs[:, None] * quant.hadamard(m, seed=None)
        blocks.append(T)
    return torch.block_diag(*blocks).float()


def _quant_units(c, lengths, groups):
    """Quantize the latent along itself, per token, unit by unit (per-unit second scale)."""
    c = _pads_to_zero(c, lengths)
    B, T, R = c.shape
    cu = c.view(B, T, R // UNIT, UNIT)
    out = torch.empty_like(cu)
    for f, units in groups.items():
        units = units.to(c.device)
        x = cu[:, :, units]
        second = None
        if quant.FORMATS.get(f, (0, 0, ""))[2] == "e4m3":
            vmax = quant.ELEM[quant.FORMATS[f][0]][1]
            second = (x.abs().amax(dim=(0, 1, 3)).clamp_min(1e-30) / (448 * vmax))[None, None, :, None, None]
        out[:, :, units] = quant.block_quant(x, f, -1, second) if f != "bf16" else x
    return out.view(B, T, R)


class LatentPatch:
    """Builds and installs one latent method on one layer."""

    def __init__(self, kind, frac, wV=1.0):
        self.label = kind
        self.gamma, self.v_axis = 1.0, "tok"
        if kind.split(".")[0] in ("rd", "rdx"):
            parts = kind.split(".")
            kind = parts[0]
            for o in parts[1:]:
                if o.startswith("g"):
                    self.gamma = float(o[1:])
                elif o == "vl":
                    self.v_axis = "lat"
        self.fixed_rank = self.global_sens = False
        self.plan = None
        self.prune = 0.0                      # fraction of tokens whose latent is dropped (alpha kept)
        self.window = 0                       # recent tokens a query always sees unpruned
        self.krot = "tier"                    # K rotation: "tier" (STAR-KV), int g (per scale group), "none"
        self.widen = 1.0                      # static step multiplier (clipping range) for s*/n* formats
        self.align = 0                        # K tier boundary rounded to a multiple of this (16: no
                                              #   scale block straddles two tiers, as the MMA layout needs)
        self.wq = False                       # K rebuild weights in NVFP4 (native FP4 x FP4 matmul)
        self.keep_first = 0                   # first tokens kept exact in bf16 (KVTC / AATC's sink rule)
        self.dyn_sink = False                 # also keep exact every token detected as a sink (alpha < 0.5)
        self.exempt_seen = [0, 0]             # [exempted, total] token-side counts for dyn_sink
        self._mean_mode = "alpha"             # from the prep: alpha / center / none (ablation bases)
        self.prune_seen = [0, 0]              # [pruned, eligible-or-not total] token-side counts
        if kind.split(".")[0] in TIERED:
            parts = kind.split(".")
            kind = parts[0]
            self.fixed_rank = "fr" in parts[1:]
            self.global_sens = "gs" in parts[1:]
            for o in parts[1:]:
                if o.startswith("p") and o[1:].isdigit():
                    self.prune = int(o[1:]) / 100
                if o.startswith("w") and o[1:].isdigit():
                    self.window = int(o[1:])
                if o.startswith("c") and o[1:].isdigit():
                    self.widen = int(o[1:]) / 10
                if o == "al":
                    self.align = 16
                if o == "wq":
                    self.wq = True
                if o.startswith("sink") and o[4:].isdigit():
                    self.keep_first = int(o[4:])
                if o == "dsink":
                    self.dyn_sink = True
                if o == "r0":
                    self.krot = "none"
                elif o.startswith("rg") and o[2:].isdigit():
                    self.krot = int(o[2:])
        if kind.startswith("dq-"):
            _, self.dq_fmt, self.dq_rot = kind.split("-")
            MENUS.setdefault(kind, ["bf16", "drop"])
        assert kind in MENUS, kind
        self.kind, self.frac, self.wV = kind, frac, wV
        self.layer_stats = {}

    @property
    def name(self):
        return f"{self.label}@{self.frac:g}"

    # -- allocation --------------------------------------------------------
    def _choose(self, prep, n_kv, d):
        """-> per-unit K formats ('fmt' or 'fmt|H' = rotated unit) and V formats."""
        base = MENUS[self.kind]
        fi = [MENU_ALL.index(f) for f in base]
        drop = MENU_ALL.index("drop")
        EK, EV = prep["DK"][:, drop].double().sum(), prep["DV"][:, drop].double().sum()
        if self.kind in ADAPTIVE_ROT:
            menu = base + [f + "|H" for f in base if f != "drop"]
            fh = [MENU_ALL.index(f) for f in base if f != "drop"]
            DK = torch.cat([prep["DK"][:, fi], prep["DK_H"][:, fh]], 1).double()
            DV = torch.cat([prep["DV"][:, fi].double(),
                            torch.full((prep["DV"].shape[0], len(fh)), float("inf"), dtype=torch.float64)], 1)
        else:
            menu = base
            DK = prep["DK_H" if self.kind == "ours_H" else "DK"][:, fi].double()
            DV = prep["DV"][:, fi].double()
        D = torch.cat([DK / EK, self.wV * DV / EV])
        if self.gamma != 1.0:
            noise = torch.tensor([f.split("|")[0] not in ("drop", "bf16") for f in menu])
            D[:, noise] = D[:, noise] * self.gamma
        if self.kind in TIERED and not self.fixed_rank:
            kt, kr, vt, vr = TIERED[self.kind]
            kd, vd = 0.2 * tier_bits(kt) + 0.8 * tier_bits(kr), 0.2 * tier_bits(vt) + 0.8 * tier_bits(vr)
        else:
            kd = vd = 3.2
        per_dim = torch.tensor([3.2 if (self.kind in TIERED or self.kind.startswith(("diag", "dq-"))) and f == "bf16"
                                else quant.bits(f.split("|")[0])
                                for f in menu], dtype=torch.float64)
        cost = per_dim[None, :].expand(D.shape[0], -1).clone() * UNIT
        if self.kind in TIERED:
            keep = [i for i, f in enumerate(menu) if f == "bf16"]
            cost[:DK.shape[0], keep] = kd * UNIT
            cost[DK.shape[0]:, keep] = vd * UNIT
        budget = (1 - self.frac) * 16 * 2 * n_kv * d
        if self.kind in TIERED:
            budget -= sum(scale_overhead(self.kind, n_kv))
        elif self.kind.startswith(("diag", "dq-")):
            budget -= 32 * (n_kv + 1)        # two bf16 scales per head (K) and for V, per token
        ch = allocate(D, cost, budget)
        nK = DK.shape[0]
        fK = [menu[i] for i in ch[:nK].tolist()]
        fV = [menu[i] for i in ch[nK:].tolist()]
        if self.kind == "ours_H":
            fK = [f if f == "drop" else f + "|H" for f in fK]
        if self.kind in ("palu", "lr", "lrnv") or self.kind in TIERED or self.kind.startswith(("diag", "dq-")):
            # rank-only selections: keep each head's (and V's) leading units
            per = d // UNIT
            kept = [sum(f != "drop" for f in fK[h * per:(h + 1) * per]) for h in range(n_kv)]
            keep_f = menu[0]
            fK = [keep_f if j < kept[h] else "drop" for h in range(n_kv) for j in range(per)]
            nv = sum(f != "drop" for f in fV)
            fV = [keep_f if j < nv else "drop" for j in range(len(fV))]
        return fK, fV

    def plan_global(self, tables, sens, n_kv, d):
        """Rank across all layers at once (the .gs variants).

        Objective B puts every layer's latent in its own error units: dropping a unit
        costs its energy e_u of that layer's logit (K) or output (V) error. sens[l][side]
        is the measured exchange rate from that error to final next-token loss
        (layer_sensitivity.py), so s * e_u is a unit's loss cost in common units, and
        keeping the units with the most s * e_u per bit, model-wide, minimises the
        total loss under an additive model. Layers then differ in rank and size."""
        assert self.kind in TIERED, "global allocation is implemented for the tiered kinds"
        drop = MENU_ALL.index("drop")
        kt, kr, vt, vr = TIERED[self.kind]
        kd = 0.2 * tier_bits(kt) + 0.8 * tier_bits(kr)
        vd = 0.2 * tier_bits(vt) + 0.8 * tier_bits(vr)
        overhead = sum(scale_overhead(self.kind, n_kv))
        pos = [v for l in sens for v in (sens[l]["k"], sens[l]["v"]) if v > 0]
        floor = 0.01 * sorted(pos)[len(pos) // 2]
        vals, costs, owner, budget = [], [], [], 0.0
        for l, tab in tables.items():
            for side, D, c in (("k", tab["DK"], kd), ("v", tab["DV"], vd)):
                e = D[:, drop].double()
                vals.append(max(sens[l][side], floor) * e)
                costs.append(torch.full_like(e, c * UNIT))
                owner += [(l, side, u) for u in range(len(e))]
            budget += (1 - self.frac) * 16 * 2 * n_kv * d - overhead
        vals, costs = torch.cat(vals), torch.cat(costs)
        order = torch.argsort(vals / costs, descending=True)
        keep = order[torch.cumsum(costs[order], 0) <= budget].tolist()
        per = d // UNIT
        kept = {l: {"k": [0] * n_kv, "v": 0} for l in tables}
        for idx in keep:
            l, side, u = owner[idx]
            if side == "k":
                kept[l]["k"][u // per] += 1
            else:
                kept[l]["v"] += 1
        self.plan = {}
        for l, tab in tables.items():
            fK = ["bf16" if j < kept[l]["k"][h] else "drop" for h in range(n_kv) for j in range(per)]
            fV = ["bf16" if j < kept[l]["v"] else "drop" for j in range(tab["DV"].shape[0])]
            self.plan[l] = (fK, fV)
        return self.plan

    # -- installation ------------------------------------------------------
    def patch(self, i, layer, prep, n_kv, d, n_q, device):
        fK, fV = self.plan[i] if self.plan is not None else self._choose(prep, n_kv, d)
        attn = layer.self_attn
        per = d // UNIT

        # K: latent columns of the kept units, head-major
        BK, AK = prep["BK"].to(device), prep["AK"].to(device)
        rots = [R.to(device) for R in k_rotations(n_kv, d)]
        k_cols, k_fmt, B_rows, A_cols, var_k = [], [], [], [], []
        head_ranges, gen = [], torch.Generator().manual_seed(1234)
        for h in range(n_kv):
            Bh, Ah = BK[h * d:(h + 1) * d], AK[h]
            units = [j for j in range(per) if fK[h * per + j] != "drop"]
            if not units:
                continue
            rot_units = [j for j in units if fK[h * per + j].endswith("|H")]
            if rot_units:
                R = torch.eye(d, device=device)
                for j in rot_units:
                    sl = slice(j * UNIT, (j + 1) * UNIT)
                    R[sl, sl] = rots[h][sl, sl]
                Bh, Ah = R @ Bh, Ah @ R.T
            idx = torch.cat([torch.arange(j * UNIT, (j + 1) * UNIT) for j in units]).to(device)
            Bk, Ak = Bh[idx], Ah[:, idx]
            r = len(idx)
            vk = prep["eK"][h].to(device)[idx]
            if self.kind == "palu":
                m = 1 << (r.bit_length() - 1)
                R = torch.eye(r, device=device)
                R[:m, :m] = quant.hadamard(m, seed=h).float().to(device)
                Bk, Ak = R @ Bk, Ak @ R.T
            if self.kind in ("rd", "rdx"):
                R = tier_rotation([fK[h * per + j] for j in units], gen).to(device)
                Bk, Ak = R @ Bk, Ak @ R.T
            if self.kind == "diag_nvH" or getattr(self, "dq_rot", "") == "blockH":
                R = torch.block_diag(*[rots[h][j * UNIT:(j + 1) * UNIT, j * UNIT:(j + 1) * UNIT] for j in units])
                Bk, Ak = R @ Bk, Ak @ R.T
            if self.kind in TIER_H or getattr(self, "dq_rot", "") == "tierH":
                if self.krot == "tier":
                    no = None
                    if self.align:
                        g = self.align
                        no = max(g, g * round(STARQ_OUT * r / g))
                    R, no = starq_blocks(r, gen, no)
                elif self.krot == "none":
                    R, no = torch.eye(r, dtype=torch.float64), int(STARQ_OUT * r)
                else:
                    R, no = group_blocks(r, self.krot, gen)
                R = R.float().to(device)
                Bk, Ak = R @ Bk, Ak @ R.T
                vk = (R * R) @ vk
                if self.wq:
                    # the matmul k = A . codes reduces over the latent axis, so NVFP4 blocks of 16
                    # run along it for the weights too (columns of A), one fp32 second scale per head
                    Ak = quant.block_quant(Ak, "nvfp4", -1)
                head_ranges.append((sum(len(c) for c in B_rows), no, r))
            else:
                head_ranges.append((sum(len(c) for c in B_rows), 0, r))
            out = torch.zeros(n_kv * d, r, device=device)
            out[h * d:(h + 1) * d] = Ak
            B_rows.append(Bk)
            A_cols.append(out)
            var_k.append(vk)
            k_fmt += [fK[h * per + j].split("|")[0] for j in units]
        Bk_all, Ak_all = torch.cat(B_rows), torch.cat(A_cols, dim=1)

        # V: kept channel units of the joint latent
        BV, AV = prep["BV"].to(device), prep["AV"].to(device)
        v_units = [j for j, f in enumerate(fV) if f != "drop"]
        vidx = torch.cat([torch.arange(j * UNIT, (j + 1) * UNIT) for j in v_units]).to(device)
        Bv, Av = BV[vidx], AV[:, vidx]
        v_fmt = [fV[j] for j in v_units]
        rv, v_no = len(vidx), 0
        vv = prep["eV"].to(device)[vidx]
        if self.kind == "palu":
            m = 1 << (rv.bit_length() - 1)
            R = torch.eye(rv, device=device)
            R[:m, :m] = quant.hadamard(m, seed=99).float().to(device)
            Bv, Av = R @ Bv, Av @ R.T
        if self.kind in ("rd", "rdx"):
            R = tier_rotation(v_fmt, torch.Generator().manual_seed(1234)).to(device)
            Bv, Av = R @ Bv, Av @ R.T
        if self.kind in TIER_H or getattr(self, "dq_rot", "") == "tierH":
            R, v_no = starq_blocks(rv, torch.Generator().manual_seed(1234))
            R = R.float().to(device)
            Bv, Av = R @ Bv, Av @ R.T
            vv = (R * R) @ vv

        self._var_k, self._var_v = torch.cat(var_k), vv
        kq, vq = self._quantizers(k_fmt, v_fmt, head_ranges, rv, v_no)
        old = (attn.k_proj, attn.v_proj)
        self._mean_mode = mm = prep.get("mean_mode", "alpha")
        mu = None if mm == "none" else prep["mu"].to(device)
        kb, vb = (None, None) if mm == "none" else (prep["kbar"].to(device), prep["vbar"].to(device))
        extra = dict(center=mm == "center", keep_first=self.keep_first,
                     dyn_sink=self.exempt_seen if self.dyn_sink else None)
        attn.k_proj = LatentProj(Bk_all, Ak_all, kq, mu, kb, self.prune, self.prune_seen, exact=old[0], **extra)
        attn.v_proj = LatentProj(Bv, Av, vq, mu, vb, self.prune, self.prune_seen, exact=old[1], **extra)
        windowed = self.prune > 0 and self.window > 0
        if windowed:
            attn._prune_window = self.window
            attn.forward = types.MethodType(windowed_prune_forward, attn)

        bits_k = self._bits(k_fmt, len(head_ranges), "k", [r for _, _, r in head_ranges],
                            [no for _, no, _ in head_ranges])
        bits_v = self._bits(v_fmt, 1, "v", [rv])
        self.layer_stats[i] = dict(bits_k=bits_k, bits_v=bits_v, rank_k=Bk_all.shape[0], rank_v=rv,
                                   rotated_k=sum(f.endswith("|H") for f in fK),
                                   fmt_k={f: k_fmt.count(f) for f in set(k_fmt)},
                                   fmt_v={f: v_fmt.count(f) for f in set(v_fmt)})

        def undo():
            attn.k_proj, attn.v_proj = old
            if windowed:
                del attn.forward
        return undo

    def _bits(self, fmts, n_heads_kept, side="k", head_widths=None, head_tops=None):
        # + 8: half of the per-token bf16 alpha that K and V share
        a = 8 if self._mean_mode == "alpha" else 0
        if self.kind in TIERED:
            kt, kr, vt, vr = TIERED[self.kind]
            top, rest = (kt, kr) if side == "k" else (vt, vr)
            tot = 0.0
            for i, r in enumerate(head_widths):
                no = head_tops[i] if head_tops is not None else int(STARQ_OUT * r)
                tot += no * tier_bits(top) + (r - no) * tier_bits(rest)
                tot += 16 * (top.endswith("t") and no > 0) + 16 * (rest.endswith("t") and r - no > 0)
            if self.kind in TOKEN_SCALE:
                per_head = side == "k" and TOKEN_SCALE[self.kind] == "head"
                tot += 16 * (len(head_widths) if per_head else 1)
            # a pruned token stores only its share of alpha
            return tot * (1 - self.prune) + a
        if self.kind.startswith("diag"):
            return 3.2 * UNIT * len(fmts) + 32 * n_heads_kept + a
        return sum(quant.bits(f) * UNIT for f in fmts) + a

    def _quantizers(self, k_fmt, v_fmt, head_ranges, rv, v_no):
        kind = self.kind
        if kind in ("lr", "diag_bf16"):
            return None, None
        if kind in ("diag_nvlat", "diag_nvH", "diag_nvtierH", "diag_mx6", "diag_mx6tierH") or kind.startswith("dq-"):
            fmt = "mxfp6" if kind.startswith("diag_mx6") else getattr(self, "dq_fmt", "nvfp4")

            def kq(c, lengths):
                out = torch.empty_like(c)
                for a, _, r in head_ranges:
                    out[..., a:a + r] = quant.block_quant(c[..., a:a + r], fmt, -1)
                return out
            return kq, lambda c, lengths: quant.block_quant(c, fmt, -1)
        if kind == "diag_int4":
            def kq(c, lengths):
                out = torch.empty_like(c)
                for a, no, r in head_ranges:
                    for lo, hi in ((a, a + no), (a + no, a + r)):
                        if hi > lo:
                            out[..., lo:hi] = quant.int_sym_per_token(c[..., lo:hi], 4)
                return out

            def vq(c, lengths):
                out = torch.empty_like(c)
                for lo, hi in ((0, v_no), (v_no, rv)):
                    if hi > lo:
                        out[..., lo:hi] = quant.int_sym_per_token(c[..., lo:hi], 4)
                return out
            return kq, vq
        if kind == "diag_nvtok":
            k_fmt, v_fmt = ["nvfp4"] * len(k_fmt), ["nvfp4"] * len(v_fmt)
        if kind in TIERED:
            kt, kr, vt, vr = TIERED[kind]

            var_k, var_v = self._var_k, self._var_v

            def token_scaled(x, var, fmts_spans):
                """One bf16 scale per token over x's columns: x = s_t * u, u ~ unit-variance per
                channel; u is quantized with the static step of each span's format."""
                s = ((x * x) / var.clamp_min(1e-30)).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
                s = s.to(torch.bfloat16).float()
                u = x / s
                out = torch.empty_like(x)
                for lo, hi, f in fmts_spans:
                    if hi > lo:
                        out[..., lo:hi] = static_quant(u[..., lo:hi], var[lo:hi], int(f[1]), self.widen) * s
                return out

            if kind in TOKEN_SCALE:
                per_head = TOKEN_SCALE[kind] == "head"

                def kq(c, lengths):
                    c = _pads_to_zero(c, lengths)
                    if not per_head:
                        spans = []
                        for a, no, r in head_ranges:
                            spans += [(a, a + no, kt), (a + no, a + r, kr)]
                        return token_scaled(c, var_k, spans)
                    out = torch.empty_like(c)
                    for a, no, r in head_ranges:
                        out[..., a:a + r] = token_scaled(c[..., a:a + r], var_k[a:a + r],
                                                         [(0, no, kt), (no, r, kr)])
                    return out

                def vq(c, lengths):
                    return token_scaled(_pads_to_zero(c, lengths), var_v, [(0, v_no, vt), (v_no, rv, vr)])
                return kq, vq

            def qt(x, fmt, var):
                if fmt in ("s4", "s3", "s2"):
                    return static_quant(x, var, int(fmt[1]))
                if fmt == "i2t":
                    return int2_per_token(x)
                if fmt in ("i4t", "i3t"):
                    return quant.int_sym_per_token(x, int(fmt[1]))
                return quant.block_quant(x, fmt, -1)

            def kq(c, lengths):
                c = _pads_to_zero(c, lengths)
                out = torch.empty_like(c)
                for a, no, r in head_ranges:
                    for lo, hi, f in ((a, a + no, kt), (a + no, a + r, kr)):
                        if hi > lo:
                            out[..., lo:hi] = qt(c[..., lo:hi], f, var_k[lo:hi])
                return out

            def vq(c, lengths):
                c = _pads_to_zero(c, lengths)
                out = torch.empty_like(c)
                for lo, hi, f in ((0, v_no, vt), (v_no, rv, vr)):
                    if hi > lo:
                        out[..., lo:hi] = qt(c[..., lo:hi], f, var_v[lo:hi])
                return out
            return kq, vq
        if kind == "palu":
            def kq(c, lengths):
                c = _pads_to_zero(c, lengths)
                out = torch.empty_like(c)
                for a, _, r in head_ranges:
                    out[..., a:a + r] = quant.block_quant(c[..., a:a + r], "nvfp4", -1)
                return out

            def vq(c, lengths):
                return quant.block_quant(_pads_to_zero(c, lengths), "nvfp4", -1)
            return kq, vq

        # lrnv / ours*: per-unit formats; K along the latent, V along tokens
        def grouped(fmts):
            g = {}
            for u, f in enumerate(fmts):
                g.setdefault(f, []).append(u)
            return {f: torch.tensor(u) for f, u in g.items()}
        gk, gv = grouped(k_fmt), grouped(v_fmt)

        def kq(c, lengths):
            c = _pads_to_zero(c, lengths)
            B, T, R = c.shape
            cu = c.view(B, T, R // UNIT, UNIT)
            out = torch.empty_like(cu)
            for f, units in gk.items():
                units = units.to(c.device)
                x = cu[:, :, units]
                second = None
                if quant.FORMATS.get(f, (0, 0, ""))[2] == "e4m3":
                    vmax = quant.ELEM[quant.FORMATS[f][0]][1]
                    second = (x.abs().amax(dim=(0, 1, 3)).clamp_min(1e-30) / (448 * vmax))[None, None, :, None, None]
                out[:, :, units] = quant.block_quant(x, f, -1, second) if f != "bf16" else x
            return out.view(B, T, R)

        if self.v_axis == "lat":
            return kq, (lambda c, lengths: _quant_units(c, lengths, gv))

        def vq(c, lengths):
            c = _pads_to_zero(c, lengths)
            B, T, R = c.shape
            out = torch.empty_like(c)
            for f, units in gv.items():
                cols = (units[:, None] * UNIT + torch.arange(UNIT)[None]).flatten().to(c.device)
                x = c[..., cols]
                second = None
                if quant.FORMATS.get(f, (0, 0, ""))[2] == "e4m3":
                    vmax = quant.ELEM[quant.FORMATS[f][0]][1]
                    s = x.abs().amax(dim=(0, 1)).view(-1, UNIT).amax(-1).clamp_min(1e-30) / (448 * vmax)
                    second = s.repeat_interleave(UNIT)[None, :, None, None]
                out[..., cols] = quant.block_quant(x, f, 1, second) if f != "bf16" else x
            return out
        return kq, vq


class ProbePatch(LatentPatch):
    """Sensitivity probe: layer `layer` only, `side` ('k' or 'v') truncated to the units a
    reference method keeps there (bf16 latent, no quantization); the other side and
    every other layer exact. `dropped` is the energy removed, in that layer's units."""

    def __init__(self, layer, side, ref_kind="starq", ref_frac=0.85):
        super().__init__("lr", 0.0)
        self.layer, self.side = layer, side
        self.ref = LatentPatch(ref_kind, ref_frac)
        self.label = f"probe{layer}{side}"
        self.dropped = 0.0

    @property
    def name(self):
        return self.label

    def patch(self, i, layer, prep, n_kv, d, n_q, device):
        if i != self.layer:
            return lambda: None
        return super().patch(i, layer, prep, n_kv, d, n_q, device)

    def _choose(self, prep, n_kv, d):
        fK, fV = self.ref._choose(prep, n_kv, d)
        drop = MENU_ALL.index("drop")
        fK = ["drop" if f == "drop" else "bf16" for f in fK]
        fV = ["drop" if f == "drop" else "bf16" for f in fV]
        if self.side == "k":
            self.dropped = sum(prep["DK"][u, drop].item() for u, f in enumerate(fK) if f == "drop")
            return fK, ["bf16"] * len(fV)
        self.dropped = sum(prep["DV"][u, drop].item() for u, f in enumerate(fV) if f == "drop")
        return ["bf16"] * len(fK), fV


def windowed_prune_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                           past_key_values=None, **kwargs):
    """Attention for token pruning with a recent window: a query at position m sees key t
    with its full latent when m - t < window and with its pruned version otherwise (what
    a cache that prunes tokens once they leave the recent window serves). Causal, eager,
    one sequence at a time."""
    B, T, _ = hidden_states.shape
    H, d, n_kv = self.config.num_attention_heads, self.head_dim, self.config.num_key_value_heads
    q = self.q_proj(hidden_states).view(B, T, H, d).transpose(1, 2)
    kp, kf = self.k_proj.both(hidden_states)
    vp, vf = self.v_proj.both(hidden_states)
    kp, kf, vp, vf = (y.view(B, T, n_kv, d).transpose(1, 2) for y in (kp, kf, vp, vf))
    cos, sin = position_embeddings
    q, kp = apply_rotary_pos_emb(q, kp, cos, sin)
    kf = apply_rotary_pos_emb(kf, kf, cos, sin)[0]
    rep = H // n_kv
    kp, kf, vp, vf = (y.repeat_interleave(rep, dim=1) for y in (kp, kf, vp, vf))
    i = torch.arange(T, device=q.device)
    dist = i[:, None] - i[None, :]
    far, causal = dist >= self._prune_window, dist >= 0
    out = torch.empty_like(q)
    for b in range(B):
        Sp = (q[b] @ kp[b].transpose(-1, -2)).float() * self.scaling
        Sf = (q[b] @ kf[b].transpose(-1, -2)).float() * self.scaling
        P = torch.softmax(torch.where(far, Sp, Sf).masked_fill(~causal, float("-inf")), -1)
        out[b] = ((P * far) @ vp[b].float() + (P * ~far) @ vf[b].float()).to(q.dtype)
    out = out.transpose(1, 2).reshape(B, T, -1)
    return self.o_proj(out), None


class LatentProj(torch.nn.Module):
    """x -> A Q(B x) + alpha * W mu, alpha = x . mu / |mu|^2 cached in bf16 (B is
    already folded with the projection that removes the mean direction).
    `lengths` is set per batch by the runner (None for full windows).

    prune: drop the whole latent of this fraction of tokens and keep only alpha, so
    the token's key / value becomes alpha * W mu (an "average" token that still takes
    part in attention). Candidates are ordinary tokens (0.5 < alpha < 1.5; attention
    sinks have alpha ~ 0 and are never pruned), lowest latent energy first; by
    objective B, ||c_t||^2 is exactly the attention error that dropping it causes.
    The decision is per token at write time; the threshold is the per-sequence
    quantile here (a deployment would calibrate it per layer offline)."""

    def __init__(self, B, A, quantize, mu=None, bias=None, prune=0.0, seen=None,
                 center=False, exact=None, keep_first=0, dyn_sink=None):
        super().__init__()
        self.B, self.A, self.quantize, self.bias = B, A, quantize, bias
        self.mu_scaled = None if mu is None else mu / mu.dot(mu)
        self.lengths = None
        self.prune, self.seen = prune, seen
        # ablation bases: center = fixed mean (c = B (x - mu), out = A c + W mu, alpha = 1 for
        # every token); keep_first = that many leading tokens bypass the latent (exact projection)
        self.mu, self.center, self.exact, self.keep_first = mu, center, exact, keep_first
        # dyn_sink: counter list; tokens with alpha < 0.5 (sinks, wherever they are) bypass the
        # latent, as outlier-token tracing keeps detected sinks in full precision
        self.dyn_sink = dyn_sink

    def _prune(self, c, alpha):
        Bn, T, _ = c.shape
        valid = torch.ones(Bn, T, dtype=torch.bool, device=c.device)
        if self.lengths is not None:
            valid = torch.arange(T, device=c.device)[None] < self.lengths.to(c.device)[:, None]
        energy = (c * c).sum(-1)
        ok = valid & (alpha > 0.5) & (alpha < 1.5)
        energy = torch.where(ok, energy, torch.full_like(energy, float("inf")))
        n_drop = (self.prune * valid.sum(-1)).long()                    # per sequence
        order = energy.argsort(-1)
        rank = torch.empty_like(order)
        rank.scatter_(-1, order, torch.arange(T, device=c.device).expand(Bn, T))
        drop = (rank < n_drop[:, None]) & ok
        if self.seen is not None:
            self.seen[0] += int(drop.sum())
            self.seen[1] += int(valid.sum())
        return c.masked_fill(drop[..., None], 0.0)

    def both(self, x):
        """(pruned, full) outputs: the same quantized latent, with and without the pruned
        tokens' latents zeroed, for attention that keeps a recent window unpruned."""
        xf = x.float()
        c = xf @ self.B.T
        alpha = (xf @ self.mu_scaled).to(torch.bfloat16).float()
        cq = self.quantize(c, self.lengths) if self.quantize is not None else c
        cp = self._prune(c, alpha)
        cp = cq.masked_fill((cp == 0).all(-1, keepdim=True) & (c != 0).any(-1, keepdim=True), 0.0)
        mean = alpha[..., None] * self.bias
        return (cp @ self.A.T + mean).to(x.dtype), (cq @ self.A.T + mean).to(x.dtype)

    def forward(self, x):
        xf = x.float()
        c = (xf - self.mu if self.center else xf) @ self.B.T
        if self.prune > 0 and self.mu_scaled is not None:
            c = self._prune(c, (xf @ self.mu_scaled).to(torch.bfloat16).float())
        if self.quantize is not None:
            c = self.quantize(c, self.lengths)
        out = c @ self.A.T
        if self.bias is not None:
            if self.center:
                out = out + self.bias
            else:
                alpha = (xf @ self.mu_scaled).to(torch.bfloat16).float()
                out = out + alpha[..., None] * self.bias
        if self.keep_first:
            n = min(self.keep_first, x.shape[1])
            out[:, :n] = self.exact(x[:, :n]).float()
        if self.dyn_sink is not None:
            sink = (xf @ self.mu_scaled) < 0.5
            if self.keep_first:
                sink[:, :self.keep_first] = False
            valid = torch.ones_like(sink) if self.lengths is None else \
                torch.arange(x.shape[1], device=x.device)[None] < self.lengths.to(x.device)[:, None]
            self.dyn_sink[0] += int((sink & valid).sum())
            self.dyn_sink[1] += int(valid.sum())
            if sink.any():
                out[sink] = self.exact(x[sink]).float()
        return out.to(x.dtype)


# ---------------------------------------------------------------------------
# post-RoPE baselines
# ---------------------------------------------------------------------------

def _fill_pads_last(x, lengths, axis):
    """Replace pad tokens by the last valid token, so token-axis groups keep their range."""
    if lengths is None:
        return x
    T = x.shape[axis]
    t = torch.arange(T, device=x.device)
    idx = torch.minimum(t[None, :], (lengths.to(x.device) - 1)[:, None])    # [B, T]
    shape = [1] * x.dim()
    shape[0], shape[axis] = x.shape[0], T
    idx = idx.view(shape).expand_as(x)
    return x.gather(axis, idx)


class PostRopePatch:
    def __init__(self, kind):
        assert kind in POST_ROPE_BITS, kind
        self.kind = kind
        self.layer_stats = {}

    @property
    def name(self):
        return self.kind

    def patch(self, i, layer, prep, n_kv, d, n_q, device):
        kind = self.kind
        H = quant.hadamard(d, seed=7).float().to(device) if kind == "nvfp4H" else None

        def per_head_nv(x):
            second = (x.abs().amax(dim=(0, 2, 3)).clamp_min(1e-30) / (448 * 6.0))[None, :, None, None, None]
            return quant.block_quant(x, "nvfp4", -1, second)

        def fq(k, v, lengths):           # [B, n_kv, T, d]
            k, v = k.float(), v.float()
            if kind == "fp8":
                kq, vq = quant.fp8_per_token(k), quant.fp8_per_token(v)
            elif kind == "nvfp4":
                kq, vq = per_head_nv(k), per_head_nv(v)
            elif kind == "mxfp4":
                kq, vq = quant.block_quant(k, "mxfp4", -1), quant.block_quant(v, "mxfp4", -1)
            elif kind == "nvfp4H":
                kq, vq = per_head_nv(k @ H.T) @ H, per_head_nv(v @ H.T) @ H
            else:
                b = 4 if kind == "kivi4" else 2
                kq = quant.int_asym(_fill_pads_last(k, lengths, 2), b, 32, axis=2)
                vq = quant.int_asym(v, b, 32, axis=-1)
            return kq, vq
        layer.self_attn.kv_fq = fq
        bits = POST_ROPE_BITS[kind] * n_kv * d
        self.layer_stats[i] = dict(bits_k=bits, bits_v=bits)

        def undo():
            layer.self_attn.kv_fq = None
        return undo


class NoPatch:
    """Uncompressed reference."""
    name = "bf16"

    def __init__(self):
        self.layer_stats = {}

    def patch(self, i, layer, prep, n_kv, d, n_q, device):
        self.layer_stats[i] = dict(bits_k=16 * n_kv * d, bits_v=16 * n_kv * d)
        return lambda: None


def compression(patch, n_layers, n_kv, d) -> float:
    full = n_layers * 2 * 16 * n_kv * d
    used = sum(s["bits_k"] + s["bits_v"] for s in patch.layer_stats.values())
    used += (n_layers - len(patch.layer_stats)) * 2 * 16 * n_kv * d
    return 1 - used / full
