"""Bit-exact NVFP4-style packing for the latent KV cache (the write path, in PyTorch).

Every value the cache stores is an E2M1 (FP4) code: NVINT3 levels {0, +-1, +-2, +-3} and NVINT2
levels {+-0.5, +-1.5} are exact E2M1 values, so the K rebuild runs on the native block-scaled FP4
tensor-core MMA. Codes pack two per byte (low nibble first). This first version stores every
code in 4 bits; packing NVINT3 / NVINT2 into 3 / 2 bits is a storage-only change on top.

Numerics match phasorkv.quant / kvmethods with the .ex.wq options: K tiers are block-16 along
the latent with an E4M3 scale per block, relative to a static fp32 per-tier scale that is folded
into the NVFP4 rebuild weights; V tiers carry one bf16 scale per token per tier (STAR-KV style),
which the decode kernel applies in registers.
"""
import torch

from .. import quant

E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def e2m1_encode(v: torch.Tensor) -> torch.Tensor:
    """Exact E2M1 values -> 4-bit codes (sign bit 3, magnitude index bits 0-2)."""
    mags = E2M1_VALUES[:8].to(v.device)
    mag = v.abs()
    idx = torch.bucketize(mag, (mags[1:] + mags[:-1]) / 2)
    assert torch.equal(mags[idx], mag), "value is not an exact E2M1 level"
    return (idx | ((v < 0).to(idx.dtype) << 3)).to(torch.uint8)


def e2m1_decode(codes: torch.Tensor) -> torch.Tensor:
    return E2M1_VALUES.to(codes.device)[codes.long()]


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    assert codes.shape[-1] % 2 == 0
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack([packed & 15, packed >> 4], -1).reshape(*packed.shape[:-1], -1)


def quantize_nv_block(x: torch.Tensor, fmt: str, second: torch.Tensor):
    """Block-16 NV quantization of x[..., n] (n a multiple of 16) along the last axis.
    -> (E2M1 codes uint8 [..., n], E4M3 scales [..., n/16], x_hat) with x_hat = value(code) *
    scale * second, the same rounding as quant.block_quant(x, fmt, -1, second)."""
    elem, block, kind = quant.FORMATS[fmt]
    assert block == 16 and kind == "e4m3" and x.shape[-1] % 16 == 0
    vmax = quant.ELEM[elem][1]
    xb = x.reshape(*x.shape[:-1], -1, 16)
    amax = xb.abs().amax(-1, keepdim=True)
    s = quant.cast_e4m3(amax / vmax / second, up=quant.SCALE_UP)
    scale = s * second
    v = torch.where(scale > 0, quant.round_to(xb / scale.clamp_min(1e-30), elem), torch.zeros_like(xb))
    return e2m1_encode(v.reshape(x.shape)), s.squeeze(-1).to(torch.float8_e4m3fn), (v * scale).reshape(x.shape)


def quantize_per_token(x: torch.Tensor, nbits: int):
    """STAR-KV per-token tier with a bf16 scale: -> (E2M1 codes [..., n], bf16 scale [...], x_hat)."""
    if nbits == 2:
        s = (x.abs().amax(-1, keepdim=True).clamp_min(1e-5) / 1.5).to(torch.bfloat16).float()
        v = quant.round_to(x / s, "int2")
    else:
        qmax = 2 ** (nbits - 1) - 1
        s = (x.abs().amax(-1, keepdim=True).clamp_min(1e-5) / qmax).to(torch.bfloat16).float()
        v = torch.round(x / s).clamp(-(2 ** (nbits - 1)), qmax)
    return e2m1_encode(v), s.squeeze(-1).to(torch.bfloat16), v * s


def quantize_nvfp4_weight(A: torch.Tensor, cols: int):
    """Rebuild weights in NVFP4 along the reduction (latent) axis, as LatentPatch(.wq) does, with
    zero columns padded up to `cols` (a whole number of 64-wide MMA steps).
    -> (packed codes uint8 [rows, cols/2], E4M3 scales [rows, cols/16], fp32 global scale, A_hat)."""
    second = (A.abs().amax().clamp_min(1e-30) / (448.0 * 6.0)).reshape(1)
    Ap = torch.nn.functional.pad(A, (0, cols - A.shape[1]))
    codes, s, A_hat = quantize_nv_block(Ap, "nvfp4", second)
    return pack_nibbles(codes), s, second, A_hat[:, :A.shape[1]]
