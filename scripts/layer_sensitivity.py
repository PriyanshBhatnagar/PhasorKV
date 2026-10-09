"""Per-layer exchange rate from attention error to final loss, for cross-layer allocation.

For every layer and side (K, V): compress only that side of that layer to what a
reference method keeps there (truncation only, bf16 latent), and measure the KL
divergence of the model's next-token distribution from the exact model's on
calibration text. KL is the clean second-order loss change (a change in NLL is KL
plus first-order noise that averages to zero but swamps one-layer effects).
s = KL / dropped energy is how much final loss one unit of that layer's latent
energy is worth; LatentPatch.plan_global ranks units model-wide by s * energy per bit.

  python scripts/layer_sensitivity.py --model lmsys/longchat-7b-v1.5-32k --tag longchat7b
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phasorkv.engine import load, token_stream, random_windows
from phasorkv.kvmethods import NoPatch, ProbePatch
from phasorkv.runner import Runner, make_batches


def main():
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--ref", default="starq", help="method whose per-layer choice is probed")
    p.add_argument("--frac", type=float, default=0.9, help="probe strength: the reference's per-layer choice at this compression")
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    model, tok = load(args.model)
    lat = os.path.join(root, "latent")
    runner = Runner(model, args.device, prep_loader=lambda i: torch.load(os.path.join(lat, f"L{i}.pt")))
    L = model.config.num_hidden_layers
    # calibration text, disjoint from the windows the bases were fitted on (seed 0/1)
    w = random_windows(token_stream("wikitext2", "train", tok), args.windows, args.seqlen, seed=2)
    seqs, spans = list(w), [(1, args.seqlen)] * len(w)
    probes = [ProbePatch(l, s, args.ref, args.frac) for l in range(L) for s in ("k", "v")]
    t0 = time.time()
    kl = runner.score_kl(make_batches(seqs, spans, 4 * args.seqlen), [NoPatch()] + probes, ref="bf16")
    res = {}
    for pr in probes:
        r = res.setdefault(pr.layer, {})
        r[pr.side] = kl[pr.name] / max(pr.dropped, 1e-30)
        r[pr.side + "_kl"], r[pr.side + "_energy"] = kl[pr.name], pr.dropped
    json.dump(dict(ref=args.ref, frac=args.frac, metric="kl", layers=res),
              open(os.path.join(root, "layer_sens.json"), "w"), indent=1)
    print(f"probes done ({time.time() - t0:.0f}s)")
    for l in range(L):
        r = res[l]
        print(f"layer {l:2d}: K KL {r['k_kl']:.2e} (s {r['k']:.2e})   V KL {r['v_kl']:.2e} (s {r['v']:.2e})")


if __name__ == "__main__":
    main()
