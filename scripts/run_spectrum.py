"""Experiment 1: how much does each K layout lose at a given cache size?

For every layer: collect calibration statistics, factorize k_proj with each
layout x weighting, and measure attention fidelity on held-out C4 windows at a
range of cache budgets. Inputs to every layer are the exact model's, so errors
are per layer, not compounded (run_ppl.py measures the compounded effect).

  python scripts/run_spectrum.py --model lmsys/longchat-7b-v1.5-32k --tag longchat7b
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phasorkv.engine import load, head_dims, Layerwise, token_stream, random_windows, c4_heldout
from phasorkv.factorize import Factorization, kv_query_energy
from phasorkv.metrics import LayerReference, proxy_error
from phasorkv.rope import rope_phasors, to_complex

DEFAULT_METHODS = ("head:plain,head:x,head:xq,group2:xq,group4:xq,group8:xq,joint:xq,"
                   "freq:plain,freq:x,freq:xq")


def main():
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True, help="results/<tag>/")
    p.add_argument("--methods", default=DEFAULT_METHODS,
                   help="comma-separated structure:weighting; structures that do not divide "
                        "the kv heads are skipped")
    p.add_argument("--budgets", default="0.0625,0.125,0.1875,0.25,0.375,0.5",
                   help="K cache kept, as a fraction of n_kv*head_dim")
    p.add_argument("--calib", default="wikitext2")
    p.add_argument("--calib-windows", type=int, default=32)
    p.add_argument("--eval-windows", type=int, default=4)
    p.add_argument("--n-queries", type=int, default=256, help="query rows scored per window")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--layers", default="", help="comma-separated subset (default: all)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--stats-only", action="store_true", help="only the calibration statistics (stats.pt)")
    args = p.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(args.device)

    model, tok = load(args.model)
    H, n_kv, d = head_dims(model)
    D = model.config.hidden_size
    run = Layerwise(model, args.device, batch=args.batch)
    only = {int(i) for i in args.layers.split(",") if i}

    calib = random_windows(token_stream(args.calib, "train", tok), args.calib_windows, args.seqlen)
    held = c4_heldout(tok, args.eval_windows, args.seqlen)
    h_cal, h_eval = run.embed(calib), run.embed(held)
    ph = rope_phasors(run.rotary, torch.arange(args.seqlen, device=dev))
    qpos = torch.linspace(args.seqlen // 16, args.seqlen - 1, args.n_queries, device=dev).long()

    full = n_kv * d
    budgets = [2 * round(float(f) * full / 2) for f in args.budgets.split(",")]
    methods = []
    for m in args.methods.split(","):
        s, w = m.split(":")
        g = {"head": 1, "joint": n_kv, "freq": 1}.get(s) or (1 if s.startswith("band") else int(s[5:]))
        if n_kv % g == 0 and not (s.startswith("group") and g >= n_kv):
            methods.append((s, w))

    print(f"{args.model}: {len(run.layers)} layers, {H} q heads, {n_kv} kv heads, d={d}; "
          f"budgets {budgets} of {full}; methods {[f'{s}:{w}' for s, w in methods]}", flush=True)

    stats, layers_out = [], []
    for i, layer in enumerate(run.layers):
        t0 = time.time()
        layer.to(dev)
        attn = layer.self_attn
        if not only or i in only:
            # calibration statistics of this layer's k_proj input and queries
            Cx = torch.zeros(D, D, dtype=torch.float64, device=dev)
            q2 = torch.zeros(H, d // 2, dtype=torch.float64, device=dev)
            n = 0
            with torch.no_grad():
                for _, hb in run.batches(h_cal):
                    x = layer.input_layernorm(hb).reshape(-1, D)
                    xf = x.float()
                    Cx += (xf.T @ xf).double()
                    q = attn.q_proj(x).float().reshape(-1, H, d)
                    q2 += (to_complex(q).abs() ** 2).sum(0).double()
                    n += xf.shape[0]
            Cx /= n
            q2 /= n
            stats.append(dict(layer=i, Cx=Cx.float().cpu(), q2=q2.float().cpu(), n_tokens=n))
        if args.stats_only:
            print(f"stats layer {i}", flush=True)
            h_cal = run.run_layer(layer, h_cal)
            layer.cpu()
            continue
        if not only or i in only:

            Wq, Wk, Wv = (m.weight.float() for m in (attn.q_proj, attn.k_proj, attn.v_proj))
            with torch.no_grad():
                xs = [x for _, hb in run.batches(h_eval) for x in layer.input_layernorm(hb).float()]
            ref = LayerReference(xs, Wq, Wk, Wv, ph, qpos, n_kv, d)
            lam = kv_query_energy(q2, n_kv)
            lam_rows = torch.cat([lam, lam], dim=1).reshape(-1)

            rec = dict(layer=i, methods={})
            for s, w in methods:
                fac = Factorization(Wk, Cx, q2, n_kv, d, s, w, device=dev)
                rows = []
                for b in budgets:
                    ranks = fac.ranks(b)
                    W_hat = fac.reconstruct(ranks)
                    m = ref.score(W_hat.float(), nope=fac.nope)
                    m.update(budget=b, frac=b / full,
                             proxy=proxy_error(Wk, W_hat, Cx, lam_rows),
                             flops=fac.decode_flops(b, H), flops_dense=2 * H * d,
                             ranks=ranks)
                    rows.append(m)
                rec["methods"][f"{s}:{w}"] = rows
                del fac
            layers_out.append(rec)
            del ref, xs
            torch.cuda.empty_cache()

            line = "  ".join(f"{k}={v[budgets.index(full // 4) if full // 4 in budgets else 0]['kl']:.2e}"
                             for k, v in rec["methods"].items())
            print(f"layer {i:2d} ({time.time() - t0:5.1f}s) KL@25%: {line}", flush=True)

        h_cal = run.run_layer(layer, h_cal)
        h_eval = run.run_layer(layer, h_eval)
        layer.cpu()

    with open(os.path.join(out_dir, "spectrum.json"), "w") as f:
        json.dump(dict(model=args.model, H=H, n_kv=n_kv, d=d, budgets=budgets, full=full,
                       config=vars(args), layers=layers_out), f)
    if not only:
        torch.save(dict(model=args.model, calib=args.calib, calib_windows=args.calib_windows,
                        seqlen=args.seqlen, layers=stats), os.path.join(out_dir, "stats.pt"))
    print(f"wrote {out_dir}/spectrum.json", flush=True)


if __name__ == "__main__":
    main()
