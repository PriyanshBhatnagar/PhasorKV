"""RoPE written as complex phasors.

HF Llama rotates the pair (j, j + d/2) of every head by theta_j * t (the
"rotate_half" layout). Writing that pair as z_j = x[j] + i x[j + d/2], RoPE is
just z_j -> z_j * e^{i theta_j t}: one phase per frequency, identical for every
head. That shared phase is what lets a latent mix heads within a frequency and
still commute with RoPE.
"""
import torch


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """[..., d] real (rotate_half layout) -> [..., d/2] complex."""
    h = x.shape[-1] // 2
    return torch.complex(x[..., :h], x[..., h:])


def to_real(z: torch.Tensor) -> torch.Tensor:
    """Inverse of to_complex."""
    return torch.cat([z.real, z.imag], dim=-1)


def rope_phasors(rotary_emb, positions: torch.Tensor, dtype=torch.float64) -> torch.Tensor:
    """e^{i theta_j t} * attention_scaling as complex [len(positions), d/2].

    Built from the module's own inv_freq, so every static scaling HF supports
    (linear, llama3, yarn's attention factor) is carried over exactly.
    """
    inv_freq = rotary_emb.inv_freq.to(device=positions.device, dtype=dtype)
    ang = positions.to(dtype)[:, None] * inv_freq[None, :]
    scale = float(getattr(rotary_emb, "attention_scaling", 1.0))
    return torch.polar(torch.full_like(ang, scale), ang)


def apply_rope(x: torch.Tensor, ph: torch.Tensor) -> torch.Tensor:
    """Rotate real [..., T, d] by phasors [T, d/2]; same result as HF apply_rotary_pos_emb."""
    z = to_complex(x)
    return to_real(z * ph.to(z.dtype))


def weight_to_freq(W: torch.Tensor, n_kv: int, d: int) -> torch.Tensor:
    """k_proj weight [n_kv*d, D] -> complex [d/2, n_kv, D].

    Entry [j, g] is the complex row producing pair j of kv head g, so
    M[j] @ x gives that frequency's coordinate in every head at once.
    """
    Wh = W.reshape(n_kv, d, W.shape[1])
    return torch.complex(Wh[:, : d // 2], Wh[:, d // 2 :]).transpose(0, 1)


def freq_to_weight(M: torch.Tensor) -> torch.Tensor:
    """Inverse of weight_to_freq: complex [d/2, n_kv, D] -> real [n_kv*d, D]."""
    h, n_kv, D = M.shape
    Mt = M.transpose(0, 1)
    return torch.cat([Mt.real, Mt.imag], dim=1).reshape(n_kv * 2 * h, D)
