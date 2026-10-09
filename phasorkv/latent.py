"""The phasor key cache: a cross-head latent per RoPE frequency, scored without
rebuilding keys.

For frequency j, kv head g's key coordinate is approximated as
    z_{g,j}(t) = sum_k A_j[g, k] c_{j,k}(t),     c_j(t) = B_j x_t.
RoPE multiplies z_{g,j} by e^{i theta_j t}, a scalar shared by every head, so it
can be applied to the latent instead: the cache stores c_{j,k}(t) e^{i theta_j t},
rotated once when the token is written. For a query at position m,

    score = sum_j Re( conj(zq_j e^{i theta_j m}) z_{g,j}(t) e^{i theta_j t} )
          = sum_k Re( conj(q'_k) c_rot_k(t) ),   q'_k = conj(A[g, k]) zq_{j(k)} e^{i theta_{j(k)} m},

a plain real dot product between [Re q', Im q'] and [Re c_rot, Im c_rot]. The
query is absorbed once per step (cost independent of context length) and the
cache is one latent shared by all heads, as in MLA, with RoPE kept exact.

Latent dims are sorted globally by the energy they carry (Sigma sits on the
cache side), which is the order a prefix-truncated or tiered-precision cache
will want.
"""
import torch

from .factorize import Factorization
from .rope import to_complex


class PhasorKey:
    def __init__(self, fac: Factorization, ranks: list[int], n_q: int):
        assert fac.absorbed, "PhasorKey needs an absorbed layout ('freq' or 'bandN')"
        h = fac.d // 2
        A_cols, B_rows, freq, energy = [], [], [], []
        self.A_s = self.B_s = None
        for b, r in zip(fac.blocks, ranks):
            Ur = b.U[:, :r]
            if b.cost == 1:                              # slow band: real, no RoPE
                if r:
                    self.A_s = (Ur / b.sq[:, None]).reshape(fac.n_kv, -1, r)   # [n_kv, 2*n_slow, r]
                    self.B_s = Ur.T @ b.Mw                                    # [r, D]
                continue
            if r == 0:
                continue
            A_cols.append(Ur / b.sq[:, None])            # A_j = diag(1/sq) U_r
            B_rows.append(Ur.conj().T @ b.Mw)            # B_j = U_r^H diag(sq) M_j
            freq.append(torch.full((r,), b.where, device=Ur.device))
            energy.append(b.s2[:r])
        order = torch.argsort(torch.cat(energy), descending=True)
        self.A = torch.cat(A_cols, dim=1)[:, order]      # [n_kv, R] complex
        self.B = torch.cat(B_rows, dim=0)[order]         # [R, D] complex
        self.freq = torch.cat(freq)[order]               # [R] frequency of each latent dim
        self.energy = torch.cat(energy)[order]
        self.slow = fac.nope.nonzero().flatten().to(self.B.device)
        self.n_q, self.n_kv, self.h = n_q, fac.n_kv, h
        self.R = self.B.shape[0]
        self.R_s = 0 if self.B_s is None else self.B_s.shape[0]

    @property
    def cache_width(self) -> int:
        """Cached reals per token (shared by all heads)."""
        return 2 * self.R + self.R_s

    def write(self, x: torch.Tensor, ph: torch.Tensor) -> torch.Tensor:
        """Tokens x [T, D] at phasors ph [T, d/2] -> cache rows [T, 2R + R_s]."""
        rd = torch.float64 if x.dtype == torch.float64 else torch.float32
        x = x.to(rd)
        c = torch.complex(x @ self.B.real.to(rd).T, x @ self.B.imag.to(rd).T)
        c = c * ph[:, self.freq].to(c.dtype)
        parts = [c.real, c.imag]
        if self.R_s:
            parts.append(x @ self.B_s.to(rd).T)
        return torch.cat(parts, dim=-1)

    def absorb(self, q: torch.Tensor, ph: torch.Tensor) -> torch.Tensor:
        """Pre-RoPE queries q [H, Tq, d] at phasors ph [Tq, d/2] -> [H, Tq, 2R + R_s]."""
        zq = to_complex(q)
        zq = zq * ph.to(zq.dtype)
        g = torch.arange(self.n_q, device=q.device) // (self.n_q // self.n_kv)
        Ah = self.A[g].conj().to(zq.dtype)               # [H, R]
        qp = Ah[:, None, :] * zq[:, :, self.freq]        # [H, Tq, R]
        parts = [qp.real, qp.imag]
        if self.R_s:
            qs = torch.cat([q[..., self.slow], q[..., self.slow + self.h]], dim=-1)  # unrotated
            parts.append(torch.einsum("htr,hrk->htk", qs, self.A_s[g].to(q.dtype)))
        return torch.cat(parts, dim=-1)

    @staticmethod
    def scores(q_abs: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        """Unscaled logits [H, Tq, Tk] = q_abs @ cache^T."""
        return q_abs @ cache.T.to(q_abs.dtype)
