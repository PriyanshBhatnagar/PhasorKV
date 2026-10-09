"""Attention-distance profile per layer and kv head, for the distance-aware key weighting.

The expected logit error of a key error e, for a query q at relative distance delta, is
E[(q^T R_delta e)^2] = e^T M e with M = sum_delta p(delta) R_delta^T Sigma_q R_delta.
p(delta) = 1 at 0 gives the full query covariance (KQ-SVD, SAKI); uniform over a long
range gives the per-pair diagonal (RoPE-averaged objective B). This script measures the
p(delta) the model actually uses: attention probability mass by distance, from the last
--queries positions of long windows of generic text, excluding the first --skip keys
(attention sinks the cache keeps exact). Saves results/<tag>/attn_dist.pt:
layers[i] = [n_kv, seqlen] mass per distance (each row sums to 1).

  python scripts/attn_distance.py --model unsloth/Meta-Llama-3.1-8B-Instruct --tag llama31_8b_inst
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from phasorkv.engine import load, head_dims, Layerwise, token_stream, random_windows
from synergy_analysis import rope


def main():
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--seqlen", type=int, default=16384)
    p.add_argument("--queries", type=int, default=512)
    p.add_argument("--skip", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    model, tok = load(args.model)
    n_q, n_kv, d = head_dims(model)
    rep = n_q // n_kv
    run = Layerwise(model, args.device, batch=1)
    T, Q = args.seqlen, args.queries
    win = random_windows(token_stream("wikitext2", "train", tok), args.windows, T, seed=3)
    h = run.embed(win)
    cos, sin = run.position_embeddings(T, torch.float32)
    cos, sin = cos[0].float(), sin[0].float()
    qpos = torch.arange(T - Q, T, device=run.device)
    dist = qpos[:, None] - torch.arange(T, device=run.device)[None]                       # [Q, T]
    valid = (dist >= 0) & (torch.arange(T, device=run.device)[None] >= args.skip)
    idx = dist.clamp_min(0)
    out = dict(seqlen=T, queries=Q, skip=args.skip, layers={})
    for i, layer in enumerate(run.layers):
        layer.to(run.device)
        attn = layer.self_attn
        hist = torch.zeros(n_kv, T, device=run.device, dtype=torch.float64)
        for _, hb in run.batches(h):
            x = layer.input_layernorm(hb)
            q = rope(attn.q_proj(x[:, T - Q:]).float().view(1, Q, n_q, d), cos[T - Q:], sin[T - Q:])
            k = rope(attn.k_proj(x).float().view(1, T, n_kv, d), cos, sin)
            for g in range(n_kv):
                qg = q[0, :, g * rep:(g + 1) * rep].permute(1, 0, 2)                         # [rep, Q, d]
                logits = qg @ k[0, :, g].T / d ** 0.5                                        # [rep, Q, T]
                prob = torch.softmax(logits.masked_fill(dist[None] < 0, float("-inf")), -1)
                prob = (prob * valid).sum(0)                                                 # [Q, T], sinks dropped
                hist[g].scatter_add_(0, idx[valid], prob[valid].double())
        out["layers"][i] = (hist / hist.sum(-1, keepdim=True)).float().cpu()
        near = out["layers"][i][:, :512].sum(-1).mean()
        print(f"layer {i:2d}: mass within 512 tokens {near:.2f}, within 64 {out['layers'][i][:, :64].sum(-1).mean():.2f}", flush=True)
        h = run.run_layer(layer, h)
        layer.cpu()
        torch.cuda.empty_cache()
    torch.save(out, os.path.join(root, "attn_dist.pt"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
