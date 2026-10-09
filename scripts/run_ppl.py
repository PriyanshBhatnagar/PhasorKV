"""Perplexity with k_proj replaced by each layout's training-free factorization.

Replacing W_k by W_hat gives exactly the logits the compressed cache produces
(the phasor path scores the rotated latent, and that equals W_hat + RoPE to
machine precision), so this measures what each cache would deliver. V is left
uncompressed to isolate K. All budgets of a method share one layerwise sweep.

  python scripts/run_ppl.py --model lmsys/longchat-7b-v1.5-32k --tag longchat7b
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phasorkv.engine import load, head_dims, Layerwise, token_stream, consecutive_windows
from phasorkv.factorize import Factorization


def main():
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True, help="reads results/<tag>/stats.pt from run_spectrum.py")
    p.add_argument("--methods", default="head:plain,head:xq,group4:xq,joint:xq,freq:x,freq:xq")
    p.add_argument("--budgets", default="0.125,0.1875,0.25,0.375,0.5",
                   help="K cache kept, as a fraction of n_kv*head_dim")
    p.add_argument("--skip-layers", default="0,1,2,31", help="left uncompressed (STAR-KV's default)")
    p.add_argument("--datasets", default="wikitext2,c4")
    p.add_argument("--windows", type=int, default=40, help="per dataset; 0 = all")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    stats = {s["layer"]: s for s in torch.load(os.path.join(out_dir, "stats.pt"))["layers"]}
    dev = torch.device(args.device)
    model, tok = load(args.model)
    H, n_kv, d = head_dims(model)
    full = n_kv * d
    run = Layerwise(model, args.device, batch=args.batch)
    skip = {int(i) for i in args.skip_layers.split(",") if i}

    sets = {}
    for name in args.datasets.split(","):
        split = "validation" if name == "c4" else "test"
        ids = token_stream(name, split, tok)
        n = args.windows or ids.numel() // args.seqlen
        sets[name] = consecutive_windows(ids, n, args.seqlen)
    windows = torch.cat(list(sets.values()))
    bounds, i0 = {}, 0
    for name, w in sets.items():
        bounds[name] = slice(i0, i0 + w.shape[0])
        i0 += w.shape[0]
    h0 = run.embed(windows)

    out_path = os.path.join(out_dir, "ppl.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    fracs = [float(f) for f in args.budgets.split(",")]
    plan = [("baseline", [None])] + [(m, fracs) for m in args.methods.split(",")]

    for method, fl in plan:
        if method != "baseline":
            s, w = method.split(":")
            g = {"head": 1, "joint": n_kv, "freq": 1}.get(s) or (1 if s.startswith("band") else int(s[5:]))
            if n_kv % g:
                continue
        todo = [f for f in fl if f"{method}@{f}" not in results]
        if not todo:
            continue
        t0 = time.time()
        hs = {f: h0.clone() for f in todo}
        for i, layer in enumerate(run.layers):
            layer.to(dev)
            k = layer.self_attn.k_proj
            if method == "baseline" or i in skip:
                for f in todo:
                    hs[f] = run.run_layer(layer, hs[f])
            else:
                W = k.weight.data.clone()
                fac = Factorization(W, stats[i]["Cx"], stats[i]["q2"], n_kv, d, s, w, device=dev)
                for f in todo:
                    b = 2 * round(f * full / 2)
                    k.weight.data.copy_(fac.reconstruct(fac.ranks(b)).to(W.dtype))
                    hs[f] = run.run_layer(layer, hs[f], nope=fac.nope)
                k.weight.data.copy_(W)
                del fac
            layer.cpu()
        for f in todo:
            row = {}
            for name, sl in bounds.items():
                nll, cnt = run.nll(hs[f][sl], windows[sl])
                row[name] = math.exp(nll / cnt)
            results[f"{method}@{f}"] = dict(method=method, frac=f, **row)
            print(f"{method:12s} K kept {f if f else 1:6.4f}  " +
                  "  ".join(f"{n}={v:.3f}" for n, v in row.items()), flush=True)
        print(f"  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(results, open(out_path, "w"), indent=1)
        del hs
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
