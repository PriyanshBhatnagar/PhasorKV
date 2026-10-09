"""Fake quantizers for the KV cache (fp32 in, fp32 out).

Block formats follow the OCP MX spec and NVIDIA's NVFP4:

  mxfp8   E4M3 elements, block 32, E8M0 (power-of-two) scale     8.25 bits
  mxfp6   E2M3 elements, block 32, E8M0 scale                     6.25 bits
  mxfp4   E2M1 elements, block 32, E8M0 scale                     4.25 bits
  nvfp4   E2M1 elements, block 16, E4M3 scale x fp32 second level 4.5 bits
  nvint4  symmetric int4, block 16, E4M3 scale (Ada INT4 MMA)      4.5 bits
  nvint3  symmetric int3, block 16, E4M3 scale (custom hardware)  3.5 bits
  nvint2  symmetric int2, block 16, E4M3 scale (custom hardware)  2.5 bits

Every block format quantizes along one axis of a tensor; `block_quant` takes the
axis so the same format can run along the latent (per token, the K reconstruction
GEMM's reduction axis) or along tokens (per channel, the P.V reduction axis).
`second` is the per-tensor (or per-tier) fp32 scale NVFP4 pairs with its E4M3
block scales; it only sets the E4M3 range, so it is computed from the data.
"""
import math
import os

import torch


def _grid(vals):
    g = torch.tensor(sorted(set(vals)), dtype=torch.float32)
    return g


FP4_E2M1 = _grid([0, 0.5, 1, 1.5, 2, 3, 4, 6])
FP6_E2M3 = _grid([m / 8 for m in range(8)] +
                 [(1 + m / 8) * 2 ** (e - 1) for e in range(1, 4) for m in range(8)])
INT4 = _grid(list(range(8)))          # symmetric: +-{0..7}
INT3 = _grid([0, 1, 2, 3])            # symmetric: +-{0..3}
INT2 = _grid([0.5, 1.5])              # symmetric, no zero: +-{0.5, 1.5}

ELEM = {  # name -> (grid or 'e4m3', max magnitude, element bits)
    "e4m3": ("e4m3", 448.0, 8),
    "e2m3": (FP6_E2M3, 7.5, 6),
    "e2m1": (FP4_E2M1, 6.0, 4),
    "int4": (INT4, 7.0, 4),
    "int3": (INT3, 3.0, 3),
    "int2": (INT2, 1.5, 2),
}

FORMATS = {  # name -> (element, block, scale kind)
    "mxfp8": ("e4m3", 32, "e8m0"),
    "mxfp6": ("e2m3", 32, "e8m0"),
    "mxfp4": ("e2m1", 32, "e8m0"),
    "nvfp4": ("e2m1", 16, "e4m3"),
    "nvint4": ("int4", 16, "e4m3"),
    "nvint3": ("int3", 16, "e4m3"),
    "nvint2": ("int2", 16, "e4m3"),
}


SCALE_UP = os.environ.get("PHASORKV_SCALE_UP", "0") == "1"   # E4M3 block scales: round up (no clipping) or to nearest


def bits(fmt: str) -> float:
    """Stored bits per element, scales included."""
    if fmt == "bf16":
        return 16.0
    if fmt == "drop":
        return 0.0
    elem, block, _ = FORMATS[fmt]
    return ELEM[elem][2] + 8.0 / block


def round_to(y: torch.Tensor, elem: str) -> torch.Tensor:
    """Round |y| to the element grid (saturating), keeping the sign."""
    grid, vmax, _ = ELEM[elem]
    if grid == "e4m3":
        return y.clamp(-vmax, vmax).to(torch.float8_e4m3fn).float()
    g = grid.to(y.device)
    mid = (g[1:] + g[:-1]) / 2
    a = g[torch.bucketize(y.abs().contiguous(), mid)]
    return torch.sign(y) * a if g[0] == 0 else torch.where(y < 0, -a, a)


def cast_e4m3(x: torch.Tensor, up: bool = False) -> torch.Tensor:
    """To E4M3 (non-negative scales). up=True rounds up instead of to nearest, so a
    block's largest element is never clipped by a scale that was rounded down."""
    y = x.clamp(0, 448.0).to(torch.float8_e4m3fn).float()
    if up:
        e = torch.floor(torch.log2(y.clamp_min(2.0 ** -9)))
        ulp = torch.exp2(torch.clamp(e, min=-6) - 3)              # subnormals: 2^-9
        y = torch.where(y < x, (y + ulp).clamp(max=448.0), y)
    return y


def block_quant(x: torch.Tensor, fmt: str, axis: int = -1, second=None) -> torch.Tensor:
    """Quantize-dequantize x in blocks along `axis` (padded with zeros to a whole block)."""
    if fmt == "bf16":
        return x
    if fmt == "drop":
        return torch.zeros_like(x)
    elem, block, kind = FORMATS[fmt]
    vmax = ELEM[elem][1]
    xt = x.movedim(axis, -1)
    n = xt.shape[-1]
    pad = (-n) % block
    if pad:
        xt = torch.nn.functional.pad(xt, (0, pad))
    xb = xt.reshape(*xt.shape[:-1], -1, block)
    amax = xb.abs().amax(-1, keepdim=True)
    if kind == "e8m0":
        emax = math.floor(math.log2(vmax))
        scale = torch.exp2(torch.floor(torch.log2(amax.clamp_min(2.0 ** -126))) - emax)
    else:
        if second is None:
            second = amax.amax().clamp_min(1e-30) / (448.0 * vmax)
        scale = cast_e4m3(amax / vmax / second, up=SCALE_UP) * second
    q = torch.where(scale > 0, round_to(xb / scale.clamp_min(1e-30), elem) * scale, torch.zeros_like(xb))
    q = q.reshape(xt.shape)[..., :n]
    return q.movedim(-1, axis)


def fp8_per_token(x: torch.Tensor) -> torch.Tensor:
    """E4M3 with one fp16 scale per vector along the last axis (8 + 16/len bits)."""
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 448.0
    return round_to(x / s, "e4m3") * s


def int_asym(x: torch.Tensor, nbits: int, group: int, axis: int) -> torch.Tensor:
    """KIVI-style asymmetric min-max integer quantization, one fp16 scale and zero per group."""
    xt = x.movedim(axis, -1)
    n = xt.shape[-1]
    pad = (-n) % group
    if pad:      # repeat the last element so padding never widens a group's range
        xt = torch.cat([xt, xt[..., -1:].expand(*xt.shape[:-1], pad)], -1)
    xg = xt.reshape(*xt.shape[:-1], -1, group)
    lo, hi = xg.amin(-1, keepdim=True), xg.amax(-1, keepdim=True)
    s = ((hi - lo) / (2 ** nbits - 1)).clamp_min(1e-30)
    q = torch.round((xg - lo) / s).clamp(0, 2 ** nbits - 1) * s + lo
    return q.reshape(xt.shape)[..., :n].movedim(-1, axis)


def int_sym_per_token(x: torch.Tensor, nbits: int) -> torch.Tensor:
    """STAR-KV's quantizer: symmetric, one scale per vector (amax / qmax), clamp to the int range."""
    qmax = 2 ** (nbits - 1) - 1
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-5) / qmax
    return torch.round(x / s).clamp(-(2 ** (nbits - 1)), qmax) * s


def hadamard(n: int, seed: int | None = 0, dtype=torch.float64) -> torch.Tensor:
    """n x n orthonormal Sylvester Hadamard (n a power of two), optionally with random row signs."""
    assert n & (n - 1) == 0, n
    H = torch.ones(1, 1, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    H = H / math.sqrt(n)
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        H = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).to(dtype)[:, None] * H
    return H


def block_diag_hadamard(n: int, block: int, seed: int = 0, dtype=torch.float64) -> torch.Tensor:
    """n x n block-diagonal Hadamard; a trailing partial block keeps its largest power of two."""
    T = torch.eye(n, dtype=dtype)
    for i, a in enumerate(range(0, n, block)):
        w = min(block, n - a)
        m = 1 << (w.bit_length() - 1)
        if m > 1:
            T[a:a + m, a:a + m] = hadamard(m, seed + i, dtype)
    return T
