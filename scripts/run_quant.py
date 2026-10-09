"""Fake-quant accuracy of KV compression methods: perplexity and zero-shot tasks.

Methods (see phasorkv/kvmethods.py): bf16, the post-RoPE baselines
(fp8, nvfp4, mxfp4, nvfp4H, kivi4, kivi2) and latent methods written
kind@compression, e.g. ours@0.9 = 90% smaller K+V cache, scales included.

  python scripts/run_quant.py --model lmsys/longchat-7b-v1.5-32k --tag longchat7b \\
      --methods bf16,nvfp4,ours@0.85,ours@0.9 --ppl --tasks

Results accumulate in results/<tag>/quant_ppl.json and quant_tasks.json;
methods already there are skipped.
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phasorkv.engine import load, token_stream, consecutive_windows
from phasorkv.kvmethods import LatentPatch, PostRopePatch, NoPatch, POST_ROPE_BITS, compression
from phasorkv.runner import Runner, make_batches
from phasorkv.tasks import TASKS, REPORT, build_requests, accuracy


def make_patch(name):
    if name == "bf16":
        return NoPatch()
    if name in POST_ROPE_BITS:
        return PostRopePatch(name)
    kind, frac = name.split("@")
    return LatentPatch(kind, float(frac))


def summarize(patch, n_layers, n_kv, d):
    st = patch.layer_stats
    out = dict(compression=compression(patch, n_layers, n_kv, d))
    an = getattr(patch, "anchor_seen", None)
    if getattr(patch, "hi_kind", None) and an and an[1]:
        out["anchor_tokens"] = an[0] / an[1]
    ex = getattr(patch, "exempt_seen", None)
    if getattr(patch, "dyn_sink", False) and ex and ex[1]:
        out["exempt_tokens"] = ex[0] / ex[1]       # K and V side counts: same tokens, so a ratio
    seen = getattr(patch, "prune_seen", None)
    if getattr(patch, "prune", 0) and seen and seen[1]:
        out["pruned_tokens"] = seen[0] / seen[1]
    if st and "rank_k" in next(iter(st.values())):
        full = n_kv * d
        out["k_bits_share"] = sum(s["bits_k"] for s in st.values()) / sum(s["bits_k"] + s["bits_v"] for s in st.values())
        out["rank_k"] = sum(s["rank_k"] for s in st.values()) / len(st) / full
        out["rank_v"] = sum(s["rank_v"] for s in st.values()) / len(st) / full
        fm = {}
        for s in st.values():
            for side in ("fmt_k", "fmt_v"):
                for f, c in s[side].items():
                    fm[f"{side[-1]}:{f}"] = fm.get(f"{side[-1]}:{f}", 0) + c
        out["formats"] = fm
    return out


def main():
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--methods", required=True)
    p.add_argument("--ppl", action="store_true")
    p.add_argument("--ppl-windows", type=int, default=32, help="per dataset (wikitext2, c4)")
    p.add_argument("--ppl-data", default="wikitext2,c4",
                   help="datasets; others than the default come from alpha_scenarios.py and go to *_scen.json")
    p.add_argument("--tasks", action="store_true")
    p.add_argument("--task-limit", type=int, default=1000, help="questions per task")
    p.add_argument("--group", type=int, default=4, help="methods per layerwise sweep (perplexity)")
    p.add_argument("--task-group", type=int, default=1, help="methods per sweep for tasks (~11 GB RAM each)")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--latent", default="", help="ablation basis variant (results/<tag>/latent_<variant>, "
                   "see ablation_prep.py); results go to ablation_<ppl|tasks>.json as <variant>/<method>")
    args = p.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    model, tok = load(args.model)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    lat_dir = os.path.join(root, f"latent_{args.latent}" if args.latent else "latent")
    prefix = f"{args.latent}/" if args.latent else ""
    runner = Runner(model, args.device, prep_loader=lambda i: torch.load(os.path.join(lat_dir, f"L{i}.pt")))
    n_kv, d = runner.n_kv, runner.d
    names = [m for m in args.methods.split(",") if m]
    cache = {}

    def global_inputs():
        """Per-layer unit tables and measured sensitivities, for the .gs (model-wide) methods."""
        if not cache:
            for i in range(n_layers):
                t = torch.load(os.path.join(lat_dir, f"L{i}.pt"), mmap=True)
                cache.setdefault("tables", {})[i] = {"DK": t["DK"].clone(), "DV": t["DV"].clone()}
            sens = json.load(open(os.path.join(root, "layer_sens.json")))["layers"]
            cache["sens"] = {int(l): v for l, v in sens.items()}
        return cache["tables"], cache["sens"]

    def run(kind, seqs, spans, max_tokens, finish, group):
        path = os.path.join(root, f"ablation_{kind}.json" if args.latent else f"quant_{kind}.json")
        res = json.load(open(path)) if os.path.exists(path) else {}
        todo = [m for m in names if prefix + m not in res]
        batches = make_batches(seqs, spans, max_tokens=max_tokens, pad_id=tok.pad_token_id or 0)
        for g in range(0, len(todo), group):
            patches = [make_patch(m) for m in todo[g:g + group]]
            for pch in patches:
                if getattr(pch, "hi_kind", None) and pch.sal_mode == "sq":
                    pch.meanq = torch.load(os.path.join(root, "meanq.pt"))
                if getattr(pch, "global_sens", False):
                    pch.plan_global(*global_inputs(), n_kv, d)
            t0 = time.time()
            out = runner.score(batches, patches)
            if args.latent:      # another run may have added variants to the shared file meanwhile
                res = {**(json.load(open(path)) if os.path.exists(path) else {}), **res}
            for pch in patches:
                lp, cnt = out[pch.name]
                res[prefix + pch.name] = dict(**finish(lp, cnt), **summarize(pch, n_layers, n_kv, d))
                r = res[prefix + pch.name]
                print(f"[{kind}] {prefix + pch.name:14s} comp {r['compression']:.3f}  " +
                      "  ".join(f"{k}={v:.4f}" for k, v in r.items() if isinstance(v, float) and k != "compression"),
                      flush=True)
            print(f"  ({time.time() - t0:.0f}s for {len(patches)})", flush=True)
            json.dump(res, open(path, "w"), indent=1)

    if args.ppl:
        seqs, spans, bounds = [], [], {}
        data = args.ppl_data.split(",")
        extra = {}
        if data != ["wikitext2", "c4"]:
            from alpha_scenarios import scenarios
            extra = scenarios(tok, args.ppl_windows, args.seqlen)
        for name in data:
            split = "validation" if name == "c4" else "test"
            w = extra[name] if name in extra else \
                consecutive_windows(token_stream(name, split, tok), args.ppl_windows, args.seqlen)
            bounds[name] = (len(seqs), len(seqs) + len(w))
            seqs += list(w)
            spans += [(1, args.seqlen)] * len(w)

        def finish_ppl(lp, cnt):
            return {n: math.exp(-lp[a:b].sum().item() / cnt[a:b].sum().item()) for n, (a, b) in bounds.items()}
        run("ppl" if data == ["wikitext2", "c4"] else "scen", seqs, spans, 4 * args.seqlen, finish_ppl, args.group)

    if args.tasks:
        seqs, spans, meta = build_requests(tok, TASKS, args.task_limit)

        def finish_tasks(lp, cnt):
            acc = accuracy(meta, lp)
            out = {t: acc[t][REPORT[t]] for t in TASKS}
            out["avg"] = sum(out.values()) / len(TASKS)
            out["detail"] = acc
            return out
        run("tasks", seqs, spans, 16384, finish_tasks, args.task_group)


if __name__ == "__main__":
    main()
