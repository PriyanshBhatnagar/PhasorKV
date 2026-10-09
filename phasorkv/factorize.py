"""Low-rank factorizations of k_proj under one metric, so they can be compared.

Every method splits the key outputs into blocks and approximates each block
M_b (rows of W_k) by a rank-r_b matrix. All of them minimise the same
weighted error

    sum_b || diag(sq_b) (M_b - M_hat_b) Cx^{1/2} ||_F^2,

which is the expected logit error when sq_b^2 = lambda, the per-(kv head,
frequency) query energy, and Cx = E[x x^T] of the k_proj input. Averaging
q^T R_delta e over relative positions delta cancels every cross-frequency term,
leaving exactly this diagonal weighting. The optimum for each block is a
projection onto the top eigenvectors of the weighted Gram matrix

    G_b = diag(sq) M_b Cx M_b^H diag(sq),   M_hat_b = diag(1/sq) U_r U_r^H diag(sq) M_b,

and the budget is split across blocks by water-filling the eigenvalues.

Structures
  head      one block per kv head (STAR-KV's K layout)
  groupN    blocks of N consecutive kv heads (Palu's G-LRD layout)
  joint     one block for the whole layer
  freq      one complex block per RoPE frequency, spanning every kv head.
            The only layout whose latent commutes with RoPE, so the query can
            be absorbed and keys are never rebuilt (see latent.PhasorKey).
  bandN     freq for all but the N slowest frequencies; those form one real
            block across heads and frequencies that skips RoPE (exact up to a
            phase error of at most theta_j * distance, tiny for slow pairs).

Weightings
  plain     Cx = I, sq = 1: SVD of the weights alone (STAR-KV's initialisation)
  x         Cx from calibration, sq = 1
  xq        Cx and the query energy: the logit-error objective above
"""
from dataclasses import dataclass

import torch

from .rope import weight_to_freq, freq_to_weight

STRUCTURES = ("head", "group2", "group4", "group8", "joint", "freq", "bandN")
WEIGHTINGS = ("plain", "x", "xq")


def kv_query_energy(q2: torch.Tensor, n_kv: int) -> torch.Tensor:
    """E|q_{h,j}|^2 per query head [H, d/2] -> lambda per kv head [n_kv, d/2].

    A kv head serves n_rep query heads (HF repeat_kv order), and its logit
    error is felt by each, so their energies add. The 1/2 is E[cos^2] from the
    average over relative position.
    """
    H, h = q2.shape
    return 0.5 * q2.reshape(n_kv, H // n_kv, h).sum(1)


@dataclass
class Block:
    U: torch.Tensor    # eigenvectors, columns sorted by decreasing eigenvalue
    s2: torch.Tensor   # eigenvalues (weighted energy each component removes)
    sq: torch.Tensor   # row weights sqrt(lambda)
    Mw: torch.Tensor   # diag(sq) @ M_b
    where: object      # complex block: its frequency j; real block: row indices into W
    cost: int          # cache reals per component (2 complex, 1 real)


class Factorization:
    """One method's decomposition of one layer's k_proj."""

    def __init__(self, W, Cx, q2, n_kv, d, structure, weighting, device="cuda"):
        assert weighting in WEIGHTINGS, weighting
        self.structure, self.weighting = structure, weighting
        self.n_kv, self.d = n_kv, d
        h = d // 2
        W = W.to(device=device, dtype=torch.float64)
        self.D = D = W.shape[1]
        C = None if weighting == "plain" else Cx.to(device=device, dtype=torch.float64)
        lam = kv_query_energy(q2.to(device=device, dtype=torch.float64), n_kv)
        if weighting != "xq":
            lam = torch.ones_like(lam)
        lam = lam.clamp_min(lam.max() * 1e-12)
        row_lam = torch.cat([lam, lam], dim=1).reshape(-1)          # per row of W
        self.blocks = []

        # bandN: the N slowest frequencies (HF orders inv_freq fastest first) form
        # one real block across heads and frequencies that is scored without RoPE;
        # every other frequency keeps its exact complex block.
        n_slow = int(structure[4:]) if structure.startswith("band") else 0
        self.absorbed = structure == "freq" or n_slow > 0
        self.nope = torch.zeros(h, dtype=torch.bool)
        if self.absorbed:
            fast = torch.arange(h - n_slow)
            M = weight_to_freq(W, n_kv, d)[fast]                    # [F, n_kv, D]
            sq = lam.T[fast].sqrt()                                 # [F, n_kv]
            Mw = sq[..., None] * M
            MC = Mw if C is None else torch.complex(Mw.real @ C, Mw.imag @ C)
            G = MC @ Mw.conj().transpose(-1, -2)
            self._add(G, sq, Mw, fast.tolist(), cost=2)
            if n_slow:
                self.nope[h - n_slow:] = True
                js = torch.arange(h - n_slow, h, device=device)
                g = torch.arange(n_kv, device=device)[:, None] * d
                rows = torch.cat([g + js, g + js + h], dim=1).reshape(-1)
                self._add_real([rows], W, row_lam, C)
        else:
            g = {"head": 1, "joint": n_kv}.get(structure) or int(structure[5:])
            assert n_kv % g == 0, f"{n_kv} kv heads do not split into groups of {g}"
            rows = torch.arange(n_kv * d, device=device).reshape(n_kv // g, g * d)
            self._add_real(list(rows), W, row_lam, C)

    def _add_real(self, row_sets, W, row_lam, C):
        rows = torch.stack(row_sets)
        sq = row_lam[rows].sqrt()
        Mw = sq[..., None] * W[rows]
        MC = Mw if C is None else Mw @ C
        self._add(MC @ Mw.transpose(-1, -2), sq, Mw, list(rows), cost=1)

    def _add(self, G, sq, Mw, where, cost):
        G = 0.5 * (G + G.conj().transpose(-1, -2))
        # Trace-normalised and on the CPU: cuSOLVER's batched eigh fails to
        # converge on some of these (energies span many decades); LAPACK does not.
        scale = G.diagonal(dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-300)
        evals, evecs = torch.linalg.eigh((G / scale[:, None, None]).cpu())
        evals = evals.to(G.device) * scale[:, None]
        evecs = evecs.to(G.device)
        evals, evecs = evals.flip(-1).clamp_min(0), evecs.flip(-1)
        self.blocks += [Block(evecs[i], evals[i], sq[i], Mw[i], where[i], cost)
                        for i in range(G.shape[0])]

    # ------------------------------------------------------------------
    @property
    def full_size(self) -> int:
        """Cache reals per token per layer of the uncompressed K."""
        return self.n_kv * self.d

    def ranks(self, budget: int) -> list[int]:
        """Water-filling: keep the components with the most energy per cached real."""
        s2 = torch.cat([b.s2 for b in self.blocks])
        cost = torch.cat([torch.full_like(b.s2, b.cost) for b in self.blocks])
        owner = torch.cat([torch.full((len(b.s2),), i, device=s2.device)
                           for i, b in enumerate(self.blocks)])
        order = torch.argsort(s2 / cost, descending=True)
        keep = order[torch.cumsum(cost[order], 0) <= budget]
        return torch.bincount(owner[keep], minlength=len(self.blocks)).tolist()

    def residual(self, ranks) -> float:
        """Weighted error the method itself predicts (comparable within a weighting)."""
        return sum(b.s2[r:].sum().item() for b, r in zip(self.blocks, ranks))

    def reconstruct(self, ranks) -> torch.Tensor:
        """W_hat as a dense real [n_kv*d, D] matrix (float64)."""
        h = self.d // 2
        W_hat = torch.zeros(self.n_kv * self.d, self.D, dtype=torch.float64,
                            device=self.blocks[0].Mw.device)
        heads = torch.arange(self.n_kv, device=W_hat.device) * self.d
        for b, r in zip(self.blocks, ranks):
            Ur = b.U[:, :r]
            M_hat = (Ur @ (Ur.conj().T @ b.Mw)) / b.sq[:, None]
            if b.cost == 2:                     # frequency j of every kv head
                W_hat[heads + b.where] = M_hat.real
                W_hat[heads + b.where + h] = M_hat.imag
            else:
                W_hat[b.where] = M_hat
        return W_hat

    def decode_flops(self, budget: int, n_q: int) -> int:
        """K-path FLOPs per cached token per decode step, for one layer.

        Reconstructing layouts rebuild every head they cover from the block's
        latent before RoPE (2*d per latent dim per head) and then take the
        usual dot products; absorbed layouts (freq, band) score the latent
        directly (2 per latent real per query head).
        """
        if self.absorbed:
            return 2 * n_q * budget
        g = len(self.blocks[0].where) // self.d          # kv heads per block
        return 2 * self.d * g * budget + 2 * n_q * self.d
