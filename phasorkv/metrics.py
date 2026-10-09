"""Attention-fidelity metrics for an approximate key projection.

Everything is measured on real activations at their real positions with the
model's exact RoPE, so the relative-position averaging the factorization
assumes is checked here, not assumed.
"""
import torch

from .rope import apply_rope


class LayerReference:
    """True queries, keys, values and attention for one layer on held-out windows."""

    def __init__(self, xs, Wq, Wk, Wv, ph, qpos, n_kv, d):
        """xs: list of [T, D] k_proj inputs; ph: phasors [T, d/2]; qpos: query rows."""
        self.Wk, self.n_kv, self.d = Wk, n_kv, d
        self.qpos, self.ph = qpos, ph
        H = Wq.shape[0] // d
        self.n_rep = H // n_kv
        T = xs[0].shape[0]
        self.mask = torch.arange(T, device=qpos.device)[None, :] <= qpos[:, None]   # [Q, T]
        self.scale = d ** -0.5
        self.items = []
        for x in xs:
            q_pre = (x @ Wq.T).reshape(T, H, d).transpose(0, 1)[:, qpos]                # [H, Q, d]
            v = self._expand(x @ Wv.T)                                                  # [H, T, d]
            S = self._logits(q_pre, x @ Wk.T, ph)
            lp = torch.log_softmax(S.masked_fill(~self.mask, float("-inf")), -1)
            self.items.append(dict(x=x, q_pre=q_pre, v=v, S=S, lp=lp, O=lp.exp() @ v))

    def _expand(self, y):
        T = y.shape[0]
        y = y.reshape(T, self.n_kv, self.d).transpose(0, 1)
        return y.repeat_interleave(self.n_rep, dim=0)

    def _logits(self, q_pre, k, ph):
        T = k.shape[0]
        q = apply_rope(q_pre, ph[self.qpos])
        k = apply_rope(k.reshape(T, self.n_kv, self.d).transpose(0, 1), ph)
        return (q @ k.repeat_interleave(self.n_rep, dim=0).transpose(1, 2)) * self.scale

    def _center(self, X):
        m = self.mask
        mean = (X * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True)
        return (X - mean) * m

    @torch.no_grad()
    def score(self, W_hat: torch.Tensor, nope=None) -> dict:
        """kl: mean KL(attn || attn_hat) per query row and head.
        logit: relative error of the row-centred logits (softmax ignores row offsets).
        out: relative error of the attention output softmax(S) V.

        nope: frequencies the compressed path scores without RoPE (bandN's slow
        block); the reference always uses exact RoPE, so that error is counted."""
        ph = self.ph
        if nope is not None and nope.any():
            ph = ph.clone()
            ph[:, nope.to(ph.device)] = ph[0, 0].abs().to(ph.dtype)    # no rotation, same attention scaling
        acc = dict(kl=0.0, rows=0, logit_num=0.0, logit_den=0.0, out_num=0.0, out_den=0.0)
        for it in self.items:
            Sh = self._logits(it["q_pre"], it["x"] @ W_hat.T, ph)
            lph = torch.log_softmax(Sh.masked_fill(~self.mask, float("-inf")), -1)
            kl = (it["lp"].exp() * (it["lp"] - lph)).masked_fill(~self.mask, 0).sum(-1)
            acc["kl"] += kl.sum().item()
            acc["rows"] += kl.numel()
            acc["logit_num"] += (self._center(Sh - it["S"]) ** 2).sum().item()
            acc["logit_den"] += (self._center(it["S"]) ** 2).sum().item()
            acc["out_num"] += ((lph.exp() @ it["v"] - it["O"]) ** 2).sum().item()
            acc["out_den"] += (it["O"] ** 2).sum().item()
        return dict(kl=acc["kl"] / acc["rows"],
                    logit=acc["logit_num"] / acc["logit_den"],
                    out=acc["out_num"] / acc["out_den"])


def proxy_error(W, W_hat, Cx, lam_rows) -> float:
    """The factorization objective, normalised: sum_r lam_r e_r Cx e_r^T / same for W."""
    E = (W - W_hat).double()
    num = (lam_rows * ((E @ Cx) * E).sum(1)).sum()
    den = (lam_rows * ((W.double() @ Cx) * W.double()).sum(1)).sum()
    return (num / den).item()
