"""Which write-time per-token signal predicts the attention a token will receive?

Candidates, all available when a token is written to the cache (no attention scores):
  sink        -alpha_t                 (alpha ~ 0: attention-sink-like)
  dev         |alpha_t - 1|
  energy      ||c_t||^2                latent energy in the attention-aware basis (~ k^T Lambda k)
  meanq       E[q]^T R k_t / sqrt(d)   mean pre-RoPE query, RoPE-averaged over distance (slow pairs survive)
  expected    meanq + energy / (2 d)   Gaussian-query expected logit, the two combined
  keynorm     -||k_t||                 (low key norm attracts attention; a known heuristic)
  random      control
Target: attention mass received from the last --queries positions (mean over heads). Tokens the
cache keeps exact anyway (the first --skip and the last --queries) are excluded. Per layer:
Spearman correlation, and the share of attention captured by the top 1 / 3 / 5 / 10% of tokens.

  python scripts/token_saliency.py --model unsloth/Meta-Llama-3.1-8B-Instruct --tag llama31_8b_inst
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from phasorkv.engine import load, head_dims, Layerwise, token_stream, consecutive_windows
from synergy_analysis import rope

TOPS = (0.01, 0.03, 0.05, 0.10)


def rankdata(x):
    r = torch.empty_like(x)
    r[x.argsort()] = torch.arange(len(x), dtype=x.dtype, device=x.device)
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


def main():
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--data", default="c4")
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--queries", type=int, default=256)
    p.add_argument("--skip", type=int, default=4)
    p.add_argument("--variant", default="diag-wo-alpha")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    model, tok = load(args.model)
    n_q, n_kv, d = head_dims(model)
    rep, h = n_q // n_kv, d // 2
    run = Layerwise(model, args.device, batch=1)
    split = "validation" if args.data == "c4" else "test"
    win = consecutive_windows(token_stream(args.data, split, tok), args.windows, args.seqlen)
    hs = run.embed(win)
    T, Q = args.seqlen, args.queries
    cos, sin = run.position_embeddings(T, torch.float32)
    cos, sin = cos[0].float(), sin[0].float()
    # RoPE-averaged weight of the mean-query term per pair: E_delta cos(theta_j delta) over the distances seen
    inv = run.rotary.inv_freq.float().to(run.device)
    dist = torch.arange(Q, T, device=run.device).float()
    avg_cos = torch.cos(inv[None] * dist[:, None]).mean(0)                         # [h]
    keep = torch.arange(T, device=run.device)
    keep = (keep >= args.skip) & (keep < T - Q)
    out = dict(model=args.model, tops=TOPS, layers={})
    for i, layer in enumerate(run.layers):
        layer.to(run.device)
        attn = layer.self_attn
        prep = torch.load(os.path.join(root, f"latent_{args.variant}", f"L{i}.pt"))
        mu, BK = prep["mu"].to(run.device), prep["BK"].to(run.device)
        acc = {}
        for w in range(hs.shape[0]):
            x = layer.input_layernorm(hs[w:w + 1].to(run.device)).float()               # [1, T, D]
            alpha = (x[0] @ mu) / mu.dot(mu)
            c = (x[0] @ BK.T).view(T, n_kv, d)                                          # latent per kv head
            qraw = attn.q_proj(x.to(hs.dtype)).float().view(T, n_kv, rep, d)
            k = attn.k_proj(x.to(hs.dtype)).float().view(T, n_kv, d)
            qm = qraw.mean(0)                                                            # mean pre-RoPE query [n_kv, rep, d]
            wgt = torch.cat([avg_cos, avg_cos])                                          # RoPE-averaged
            meanq = torch.einsum("grd,tgd->tgr", qm * wgt, k).mean(-1) / d ** 0.5        # [T, n_kv]
            energy = (c * c).sum(-1)                                                     # [T, n_kv]
            q = rope(qraw[T - Q:].reshape(1, Q, n_q, d), cos[T - Q:], sin[T - Q:])[0].permute(1, 0, 2)
            kr = rope(k[None], cos, sin)[0].permute(1, 0, 2).repeat_interleave(rep, 0)
            logits = q @ kr.transpose(-1, -2) / d ** 0.5                                 # [H, Q, T]
            causal = torch.arange(T, device=run.device)[None] <= torch.arange(T - Q, T, device=run.device)[:, None]
            prob = torch.softmax(logits.masked_fill(~causal, float("-inf")), -1)
            mass = prob.view(n_kv, rep, Q, T).mean(1).mean(1).T                          # [T, n_kv]
            sig = {"sink": -alpha[:, None].expand(T, n_kv), "dev": (alpha - 1).abs()[:, None].expand(T, n_kv),
                   "energy": energy, "meanq": meanq, "expected": meanq + energy / (2 * d),
                   "keynorm": -k.norm(dim=-1), "random": torch.rand(T, n_kv, device=run.device)}
            for g in range(n_kv):
                m = mass[keep, g]
                tot = m.sum()
                for name, s in sig.items():
                    sv = s[keep, g]
                    order = sv.argsort(descending=True)
                    r = acc.setdefault(name, {"rho": [], **{f"top{t}": [] for t in TOPS}})
                    r["rho"].append(spearman(sv, m))
                    for t in TOPS:
                        r[f"top{t}"].append(float(m[order[:max(1, int(t * len(m)))]].sum() / tot))
        res = {n: {k: sum(v) / len(v) for k, v in r.items()} for n, r in acc.items()}
        out["layers"][i] = res
        print(f"layer {i:2d} attention captured by top 3% / 10% (rho): " + "  ".join(
            f"{n}={res[n]['top0.03']:.2f}/{res[n]['top0.1']:.2f}({res[n]['rho']:+.2f})" for n in res), flush=True)
        hs = run.run_layer(layer, hs)
        layer.cpu()
        torch.cuda.empty_cache()
    L = list(out["layers"].values())
    out["mean"] = {n: {k: sum(l[n][k] for l in L) / len(L) for k in L[0][n]} for n in L[0]}
    print("\nmean over layers:")
    for n, r in out["mean"].items():
        print(f"  {n:9s} rho {r['rho']:+.3f}  " + "  ".join(f"top{int(t * 100)}% {r[f'top{t}']:.3f}" for t in TOPS))
    json.dump(out, open(os.path.join(root, f"token_saliency_{args.data}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
