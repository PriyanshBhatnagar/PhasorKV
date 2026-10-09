"""Why the attention-aware basis and the mean split need each other: per-token evidence.

For every layer, on held-out text, per token t:
  alpha_t   share of the mean direction (x_t = alpha_t mu + P x_t)
  mass_t    attention the token receives (mean over heads and over the last --queries positions)
  err_t[v]  post-RoPE logit error of its compressed key under basis variant v,
            mean over those queries and heads of (q_t'^T R (k_hat_t - k_t))^2 / d
with every variant compressed by the same method (default starq@0.85: STAR-KV tiers, 85%).
Per layer it also records how strongly queries read the mean key direction compared
with an average key direction (rho), which decides whether a query-weighted basis keeps it.

Writes results/<tag>/synergy.pt (per-token arrays for --keep-layers, summaries for all).

  python scripts/synergy_analysis.py --model unsloth/Meta-Llama-3.1-8B-Instruct --tag llama31_8b_inst
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phasorkv.engine import load, head_dims, Layerwise, token_stream, consecutive_windows
from phasorkv.kvmethods import LatentPatch

VARIANTS = ["diag-wo-alpha", "diag-wo-center", "diag-wo-none", "pca-pca-alpha", "pca-pca-center", "pca-pca-none"]


def rope(x, cos, sin):
    """HF rotate-half RoPE on [W, T, heads, d] with cos/sin [T, d]."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    rot = torch.cat([-x2, x1], -1)
    return x * cos[None, :, None] + rot * sin[None, :, None]


def main():
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--method", default="starq@0.85")
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--queries", type=int, default=256, help="last positions used as queries")
    p.add_argument("--keep-layers", default="2,8,16,24,30")
    p.add_argument("--data", default="c4", help="wikitext2 or c4 (held-out validation/test text)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    model, tok = load(args.model)
    n_q, n_kv, d = head_dims(model)
    rep = n_q // n_kv
    run = Layerwise(model, args.device, batch=args.windows)
    split = "validation" if args.data == "c4" else "test"
    win = consecutive_windows(token_stream(args.data, split, tok), args.windows, args.seqlen)
    h = run.embed(win)
    cos, sin = run.position_embeddings(args.seqlen, torch.float32)
    cos, sin = cos[0].float(), sin[0].float()
    keep = {int(l) for l in args.keep_layers.split(",")}
    kind, frac = args.method.split("@")
    T, Q = args.seqlen, args.queries
    out = dict(tokens=win, layers={}, summary={}, method=args.method, variants=VARIANTS)

    for i, layer in enumerate(run.layers):
        layer.to(run.device)
        attn = layer.self_attn
        hb = h.to(run.device)
        x = layer.input_layernorm(hb)                                       # [W, T, D] bf16
        xf = x.float()
        preps = {v: torch.load(os.path.join(root, f"latent_{v}", f"L{i}.pt")) for v in VARIANTS}
        mu = preps[VARIANTS[0]]["mu"].to(run.device)
        alpha = (xf @ mu) / mu.dot(mu)                                      # [W, T]
        W = xf.shape[0]
        q = rope(attn.q_proj(x).float().view(W, T, n_q, d), cos, sin)[:, T - Q:]       # [W, Q, H, d]
        k = attn.k_proj(x).float().view(W, T, n_kv, d)
        kr = rope(k, cos, sin)
        # attention received (causal), mean over heads and the query positions
        qh = q.permute(0, 2, 1, 3)                                          # [W, H, Q, d]
        kh = kr.permute(0, 2, 1, 3).repeat_interleave(rep, 1)               # [W, H, T, d]
        logits = qh @ kh.transpose(-1, -2) / d ** 0.5                       # [W, H, Q, T]
        causal = torch.arange(T, device=run.device)[None] <= torch.arange(T - Q, T, device=run.device)[:, None]
        p_att = torch.softmax(logits.masked_fill(~causal, float("-inf")), -1)
        mass = p_att.mean(1).mean(1)                                        # [W, T]
        del logits, p_att
        errs = {}
        for v in VARIANTS:
            patch = LatentPatch(kind, float(frac))
            undo = patch.patch(i, layer, preps[v], n_kv, d, n_q, run.device)
            k_hat = attn.k_proj(x).float().view(W, T, n_kv, d)
            undo()
            e = rope(k_hat - k, cos, sin).permute(0, 2, 1, 3).repeat_interleave(rep, 1)   # [W, H, T, d]
            de = (qh @ e.transpose(-1, -2)) ** 2 / d                        # [W, H, Q, T]
            errs[v] = (de * causal).sum(2).mean(1) / causal.sum(0).clamp_min(1)          # [W, T]
            del e, de
        # how much queries read the mean key relative to an average key direction, per kv head
        kbar = preps[VARIANTS[0]]["kbar"].to(run.device).view(n_kv, d)
        qq = attn.q_proj(x).float().view(-1, n_kv, rep, d)
        Sq = torch.einsum("ngra,ngrb->gab", qq, qq) / qq.shape[0]          # pre-RoPE second moment
        lam = 0.5 * (Sq.diagonal(dim1=-2, dim2=-1)[:, : d // 2] + Sq.diagonal(dim1=-2, dim2=-1)[:, d // 2:])
        lam = torch.cat([lam, lam], -1)                                     # RoPE-averaged weight per dim
        kk = k.reshape(-1, n_kv, d)
        rho = ((kbar ** 2 * lam).sum(-1) / (kbar ** 2).sum(-1)) / ((kk ** 2 * lam).sum(-1).mean(0) / (kk ** 2).sum(-1).mean(0))
        a2 = (alpha ** 2).mean()
        share_u = (a2 * (kbar ** 2).sum() / (kk ** 2).sum(-1).sum(-1).mean()).item()
        share_w = (a2 * (kbar ** 2 * lam).sum() / (kk ** 2 * lam).sum(-1).sum(-1).mean()).item()
        late = alpha[:, 4:]
        summ = dict(rho=rho.cpu(), share_unweighted=share_u, share_weighted=share_w,
                    alpha_var=alpha.var().item(), sink_frac=(late < 0.5).float().mean().item(),
                    err={v: errs[v].mean().item() for v in VARIANTS},
                    err_attn={v: (errs[v] * mass).sum(-1).mean().item() for v in VARIANTS})
        out["summary"][i] = summ
        if i in keep:
            out["layers"][i] = dict(alpha=alpha.cpu(), mass=mass.cpu(), err={v: e.cpu() for v, e in errs.items()})
        print(f"layer {i:2d} rho {rho.mean():.3f} mean-share u {share_u:.3f} w {share_w:.3f} "
              f"sinks {100 * summ['sink_frac']:.2f}% | attn-weighted err " +
              " ".join(f"{v}={summ['err_attn'][v]:.2e}" for v in VARIANTS), flush=True)
        h = run.run_layer(layer, h)
        layer.cpu()
        del preps, x, xf, q, k, kr, qh, kh
        torch.cuda.empty_cache()
    torch.save(out, os.path.join(root, f"synergy_{args.data}.pt"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
