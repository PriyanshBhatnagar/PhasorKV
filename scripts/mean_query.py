"""RoPE-averaged mean query per layer and kv head, for the token-hybrid anchor score.

score_t = sum_g w_g . k_{t,g} (z-scored per head, then averaged), with
w_g = sum_{h in g} E[q_h] * avgcos / sqrt(d): the mean pre-RoPE query, each RoPE pair weighted
by cos(theta_j delta) averaged over distances 1..--horizon, so fast pairs (which average out over
distance) drop and slow pairs (which keep a consistent sign) stay. In the latent this is one dot
product per token: w . (A c_t + alpha_t W mu). Writes results/<tag>/meanq.pt = {layer: [n_kv, d]}.

  python scripts/mean_query.py --model unsloth/Meta-Llama-3.1-8B --tag llama31_8b
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phasorkv.engine import load, head_dims, Layerwise, token_stream, random_windows


def main():
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--horizon", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    root = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    model, tok = load(args.model)
    n_q, n_kv, d = head_dims(model)
    rep = n_q // n_kv
    run = Layerwise(model, args.device, batch=4)
    h = run.embed(random_windows(token_stream("wikitext2", "train", tok), args.windows, args.seqlen))
    inv = run.rotary.inv_freq.float().to(run.device)
    delta = torch.arange(1, args.horizon + 1, device=run.device).float()
    avgcos = torch.cos(inv[None] * delta[:, None]).mean(0)
    wgt = torch.cat([avgcos, avgcos])
    out = {}
    for i, layer in enumerate(run.layers):
        layer.to(run.device)
        qs, n = torch.zeros(n_q, d, device=run.device, dtype=torch.float64), 0
        for _, hb in run.batches(h):
            q = layer.self_attn.q_proj(layer.input_layernorm(hb)).float().reshape(-1, n_q, d)
            qs += q.sum(0).double()
            n += q.shape[0]
        qm = (qs / n).float().view(n_kv, rep, d)
        out[i] = (qm * wgt).sum(1).cpu() / d ** 0.5
        h = run.run_layer(layer, h)
        layer.cpu()
    torch.save(out, os.path.join(root, "meanq.pt"))
    print("wrote", os.path.join(root, "meanq.pt"), flush=True)


if __name__ == "__main__":
    main()
