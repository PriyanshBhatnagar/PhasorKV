"""Per-layer latent shapes of a model-wide (.gs) tiered method, for the decode kernel benchmarks.

From the unit tables (latent_<variant>/L{i}.pt) and the layer sensitivities, runs the same global
allocation as run_quant.py and writes, per layer, the K latents (kv heads, rank), the V rank and
its top-tier width. No model or GPU needed.

  python scripts/kernel_shapes.py --tag llama31_8b_inst --latent diagj-wo-alpha \
      --methods t32.gs.sink4.al.wq.ex@0.925 --n-kv 8 --d 128
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phasorkv.kvmethods import LatentPatch, UNIT, STARQ_OUT, kgrid_v_no


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--latent", required=True)
    p.add_argument("--methods", required=True)
    p.add_argument("--n-kv", type=int, default=8)
    p.add_argument("--d", type=int, default=128)
    args = p.parse_args()
    root = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    lat_dir = os.path.join(root, f"latent_{args.latent}")
    n_layers = len([f for f in os.listdir(lat_dir) if f.startswith("L") and f.endswith(".pt")])
    tables = {}
    for i in range(n_layers):
        t = torch.load(os.path.join(lat_dir, f"L{i}.pt"), mmap=True)
        tables[i] = {"DK": t["DK"].clone(), "DV": t["DV"].clone(), "k_group": t.get("k_group", 1)}
    own = os.path.join(root, f"layer_sens_{args.latent}.json")
    sens = json.load(open(own if os.path.exists(own) else os.path.join(root, "layer_sens.json")))["layers"]
    sens = {int(l): v for l, v in sens.items()}
    out_path = os.path.join(root, "kernel_shapes.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for m in args.methods.split(","):
        kind, frac = m.split("@")
        pch = LatentPatch(kind, float(frac))
        assert pch.global_sens, "model-wide (.gs) methods only"
        plan = pch.plan_global(tables, sens, args.n_kv, args.d)
        gk = tables[0]["k_group"]
        per, nkb = gk * args.d // UNIT, args.n_kv // gk
        layers = {}
        for i in range(n_layers):
            if i not in plan:
                layers[i] = None                                 # left exact (.sk)
                continue
            fK, fV = plan[i]
            ranks = [UNIT * sum(f != "drop" for f in fK[h * per:(h + 1) * per]) for h in range(nkb)]
            rv = UNIT * sum(f != "drop" for f in fV)
            layers[i] = dict(k_blocks=[[list(range(h * gk, (h + 1) * gk)), r] for h, r in enumerate(ranks)],
                             rv=rv, v_no=kgrid_v_no(rv) if pch.kg_v else int(STARQ_OUT * rv))
        out[f"{args.latent}/{m}"] = dict(k_group=gk, layers=layers)
        rk = sum(sum(r for _, r in l["k_blocks"]) for l in layers.values() if l) / n_layers
        rv = sum(l["rv"] for l in layers.values() if l) / n_layers
        print(f"{args.latent}/{m}: mean rank K {rk:.0f}, V {rv:.0f}", flush=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
