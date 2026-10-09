"""Where the mean split matters: per-token alpha on different kinds of text.

alpha_t = x_t . mu / |mu|^2 (mu: the layer's calibration mean input). Ordinary
tokens sit near 1; attention sinks near 0. Fixed-mean centering reconstructs every
token as if alpha were 1, so any token far from 1 lands outside the fitted
subspace; the KVTC / AATC rule exempts only the first 4 positions. This counts,
per scenario, the tokens after position 4 whose alpha is far from 1.

  python scripts/alpha_scenarios.py --model unsloth/Meta-Llama-3.1-8B --tag llama31_8b
"""
import argparse
import json
import os
import sys

import torch
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phasorkv.engine import load, Layerwise, token_stream, consecutive_windows

C4 = dict(path="allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
          revision="607bd4c8450a42878aa9ddc051a65a055450ef87", split="validation")


def scenarios(tok, n, T):
    out = {"wikitext2": consecutive_windows(token_stream("wikitext2", "test", tok), n, T),
           "c4": consecutive_windows(token_stream("c4", "validation", tok), n, T)}
    docs = load_dataset(C4["path"], data_files=C4["data_files"], revision=C4["revision"],
                        split=C4["split"])[2000:2600]["text"]
    bos = tok.bos_token_id
    # packed documents / RAG prompts: every document starts with BOS
    ids = [t for d in docs for t in [bos] + tok(d, add_special_tokens=False).input_ids]
    out["c4_packed_bos"] = consecutive_windows(torch.tensor(ids), n, T)
    # many short documents, each with BOS (dense chunk boundaries, ~100 tokens apart)
    short = [t for d in docs for t in [bos] + tok(d, add_special_tokens=False).input_ids[:100]]
    out["c4_chunks100_bos"] = consecutive_windows(torch.tensor(short), n, T)
    try:
        he = load_dataset("openai/openai_humaneval", split="test")
        code = "\n\n".join(p + s for p, s in zip(he["prompt"], he["canonical_solution"]))
        out["code_humaneval"] = consecutive_windows(tok(code, return_tensors="pt").input_ids[0], n, T)
    except Exception as e:                       # offline or renamed dataset: skip the scenario
        print(f"code scenario skipped: {e}")
    return out


def main():
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    root = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    model, tok = load(args.model)
    run = Layerwise(model, args.device, batch=4)
    sc = scenarios(tok, args.windows, args.seqlen)
    mus = [torch.load(os.path.join(root, "latent", f"L{i}.pt"), mmap=True)["mu"] for i in range(len(run.layers))]
    hs = {k: run.embed(w) for k, w in sc.items()}
    res = {k: [] for k in sc}
    examples = {}
    for i, layer in enumerate(run.layers):
        layer.to(run.device)
        mu = mus[i].to(run.device)
        for k in sc:
            a = torch.cat([(layer.input_layernorm(hb).float() @ mu) / mu.dot(mu) for _, hb in run.batches(hs[k])])
            late = a[:, 4:]
            res[k].append(dict(sink=(late < 0.5).float().mean().item(),
                               partial=(((late - 1).abs() >= 0.2) & ((late - 1).abs() <= 0.5)).float().mean().item(),
                               far=((late - 1).abs() > 0.5).float().mean().item()))
            if i == len(run.layers) // 2:
                pos = (a[0, 4:] < 0.5).nonzero().flatten()[:12] + 4
                examples[k] = [(int(t), tok.decode(sc[k][0, t])) for t in pos]
            hs[k] = run.run_layer(layer, hs[k])
        layer.cpu()
    summary = {}
    for k, per in res.items():
        L = per[1:]                              # layer 0's input is the raw embedding
        summary[k] = {m: dict(mean=sum(r[m] for r in L) / len(L), max=max(r[m] for r in L)) for m in ("sink", "partial", "far")}
        print(f"{k:18s} after pos 4: sink-like (alpha<0.5) {100 * summary[k]['sink']['mean']:.2f}% "
              f"(max layer {100 * summary[k]['sink']['max']:.2f}%), |alpha-1|>0.5 {100 * summary[k]['far']['mean']:.2f}%, "
              f"partial 0.2-0.5 {100 * summary[k]['partial']['mean']:.2f}%")
        print(f"   mid-layer sink-like positions in window 0: {examples[k]}")
    json.dump(dict(summary=summary, per_layer=res, examples=examples),
              open(os.path.join(root, "alpha_scenarios.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
