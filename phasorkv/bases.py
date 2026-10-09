"""Full-rank latent bases for K and V from objective B.

Both bases are orthogonal changes of coordinates in which the error that matters
is plain Euclidean: a latent error delta costs ||delta||^2 of query-weighted
logit error (K) or of attention-output error after W_o (V). So quantization
noise can be measured, and bits allocated, directly on the latent, and the
latent's per-dimension energy (sorted, Sigma on the cache side) says how much
each dimension is worth.

  K  per kv head:  c = U^T diag(sqrt(lambda)) W_k^h x,  k = diag(1/sqrt(lambda)) U c
  V  all heads:    c = U^T L^{1/2} W_v x,               v = L^{-1/2} U c,
     L = blockdiag_g( sum_{h in g} W_o^h^T W_o^h ), so ||L^{1/2} dv||^2 = ||W_o dv||^2
     for the heads sharing kv head g. Right weighting is E[x x^T] for both.
"""
import torch

from .factorize import Factorization


def k_basis(Wk, Cx, q2, n_kv, d, device="cuda"):
    """Returns B [n_kv*d, D] (latent = B x, head-major, sorted within head), A [n_kv, d, d], energy [n_kv, d]."""
    fac = Factorization(Wk, Cx, q2, n_kv, d, "head", "xq", device=device)
    B = torch.cat([b.U.T @ b.Mw for b in fac.blocks])                 # [n_kv*d, D]
    A = torch.stack([b.U / b.sq[:, None] for b in fac.blocks])       # [n_kv, d, d]
    energy = torch.stack([b.s2 for b in fac.blocks])                 # [n_kv, d]
    return B, A, energy


def _psd_pow(M, p, rel_floor=1e-6):
    e, V = torch.linalg.eigh(M.cpu())          # LAPACK; see factorize._add
    e = e.clamp_min(e.max() * rel_floor)
    return ((V * e ** p) @ V.T).to(M.device)


def v_basis(Wv, Wo, Cx, n_kv, d, n_q, device="cuda"):
    """Returns B [n_kv*d, D] (sorted globally), A [n_kv*d, n_kv*d], energy [n_kv*d]."""
    Wv = Wv.to(device=device, dtype=torch.float64)
    Wo = Wo.to(device=device, dtype=torch.float64)
    C = Cx.to(device=device, dtype=torch.float64)
    n_rep = n_q // n_kv
    Lh, Lih = [], []
    for g in range(n_kv):
        L = sum(Wo[:, h * d:(h + 1) * d].T @ Wo[:, h * d:(h + 1) * d]
                for h in range(g * n_rep, (g + 1) * n_rep))
        Lh.append(_psd_pow(L, 0.5))
        Lih.append(_psd_pow(L, -0.5))
    Lh, Lih = torch.block_diag(*Lh), torch.block_diag(*Lih)
    Mw = Lh @ Wv
    G = Mw @ C @ Mw.T
    G = 0.5 * (G + G.T)
    scale = G.diagonal().sum()
    e, U = torch.linalg.eigh((G / scale).cpu())
    e, U = (e.flip(0).clamp_min(0) * scale.cpu()).to(device), U.flip(1).to(device)
    return U.T @ Mw, Lih @ U, e
