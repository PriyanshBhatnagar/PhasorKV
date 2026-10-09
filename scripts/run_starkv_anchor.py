"""Trained STAR-KV as an anchor, on the same perplexity windows and task questions.

Loads the STAR-KV checkpoint (every weight is KD fine-tuned) into the base
architecture with dense K/V = U @ VS on the compressed layers, which is exactly
what the bf16 low-rank cache computes, and evaluates:

  starkv            the trained model, bf16 latent cache
  starkv+q3.2       plus STAR-KV's own fake quantization (fold_kv_hadamard +
                    leading 20% int4 / rest int3, per token), replicated here

  python scripts/run_starkv_anchor.py --weights ../starkv_developer/fused_weights.pt --ppl --tasks
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phasorkv import quant
from phasorkv.engine import load, token_stream, consecutive_windows
from phasorkv.kvmethods import LatentProj, starq_blocks, STARQ_BITS
from phasorkv.runner import Runner, make_batches
from phasorkv.tasks import TASKS, REPORT, build_requests, accuracy

MODEL = "lmsys/longchat-7b-v1.5-32k"


class StarKVPatch:
    def __init__(self, sd, quantize):
        self.sd, self.quantize = sd, quantize
        self.name = "starkv+q3.2" if quantize else "starkv"
        self.layer_stats = {}

    def patch(self, i, layer, prep, n_kv, d, n_q, device):
        pre = f"model.layers.{i}.self_attn."
        if pre + "k_proj.VS.weight" not in self.sd:
            self.layer_stats[i] = dict(bits_k=16 * n_kv * d, bits_v=16 * n_kv * d)
            return lambda: None
        ranks = self.sd[pre + "k_proj.head_ranks"].tolist()
        Rv = self.sd[pre + "v_proj.VS.weight"].shape[0]
        if not self.quantize:
            self.layer_stats[i] = dict(bits_k=16 * sum(ranks), bits_v=16 * Rv)
            return lambda: None
        attn = layer.self_attn
        out = {}
        for side, rlist in (("k", ranks), ("v", [Rv])):
            VS = self.sd[pre + f"{side}_proj.VS.weight"].to(device, torch.float32)
            U = self.sd[pre + f"{side}_proj.U.weight"].to(device, torch.float32)
            gen = torch.Generator().manual_seed(1234)
            blocks, a = [], 0
            for r in rlist:
                T, no = starq_blocks(r, gen)
                T = T.float().to(device)
                VS[a:a + r] = T @ VS[a:a + r]
                U[:, a:a + r] = U[:, a:a + r] @ T.T
                blocks.append((a, no, r))
                a += r

            def q(c, lengths, blocks=blocks):
                o = torch.empty_like(c)
                for a, no, r in blocks:
                    for lo, hi, b in ((a, a + no, STARQ_BITS[0]), (a + no, a + r, STARQ_BITS[1])):
                        if hi > lo:
                            o[..., lo:hi] = quant.int_sym_per_token(c[..., lo:hi], b)
                return o
            out[side] = LatentProj(VS, U, q)
            bits = sum(4 * int(0.2 * r) + 3 * (r - int(0.2 * r)) + 32 for r in rlist)
            self.layer_stats.setdefault(i, {})[f"bits_{side}"] = bits
        old = (attn.k_proj, attn.v_proj)
        attn.k_proj, attn.v_proj = out["k"], out["v"]

        def undo():
            attn.k_proj, attn.v_proj = old
        return undo


def main():
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--tag", default="longchat7b")
    p.add_argument("--ppl", action="store_true")
    p.add_argument("--ppl-windows", type=int, default=32)
    p.add_argument("--tasks", action="store_true")
    p.add_argument("--task-limit", type=int, default=1000)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    model, tok = load(MODEL)
    sd = torch.load(args.weights, map_location="cpu", mmap=True, weights_only=False)
    own = model.state_dict()
    with torch.no_grad():
        for k, v in own.items():
            if k in sd:
                v.copy_(sd[k])
            else:            # k_proj / v_proj of a compressed layer: the dense product
                base = k.rsplit(".", 1)[0]
                v.copy_((sd[base + ".U.weight"].float() @ sd[base + ".VS.weight"].float()).to(v.dtype))
    n_layers, n_kv = model.config.num_hidden_layers, model.config.num_key_value_heads
    runner = Runner(model, args.device)
    d = runner.d
    patches = lambda: [StarKVPatch(sd, False), StarKVPatch(sd, True)]

    def comp(pch):
        full = n_layers * 2 * 16 * n_kv * d
        return 1 - sum(s["bits_k"] + s["bits_v"] for s in pch.layer_stats.values()) / full

    def record(kind, out, finish, pchs):
        path = os.path.join(root, f"quant_{kind}.json")
        res = json.load(open(path)) if os.path.exists(path) else {}
        for pch in pchs:
            lp, cnt = out[pch.name]
            res[pch.name] = dict(**finish(lp, cnt), compression=comp(pch), trained=True)
            print(f"[{kind}] {pch.name}: comp {res[pch.name]['compression']:.3f} " +
                  "  ".join(f"{k}={v:.4f}" for k, v in res[pch.name].items() if isinstance(v, float)), flush=True)
        json.dump(res, open(path, "w"), indent=1)

    if args.ppl:
        seqs, spans, bounds = [], [], {}
        for name in ("wikitext2", "c4"):
            split = "validation" if name == "c4" else "test"
            w = consecutive_windows(token_stream(name, split, tok), args.ppl_windows, args.seqlen)
            bounds[name] = (len(seqs), len(seqs) + len(w))
            seqs += list(w)
            spans += [(1, args.seqlen)] * len(w)
        pchs = patches()
        out = runner.score(make_batches(seqs, spans, 4 * args.seqlen), pchs)
        record("ppl", out, lambda lp, cnt: {n: math.exp(-lp[a:b].sum().item() / cnt[a:b].sum().item())
                                            for n, (a, b) in bounds.items()}, pchs)
    if args.tasks:
        seqs, spans, meta = build_requests(tok, TASKS, args.task_limit)
        batches = make_batches(seqs, spans, 16384, pad_id=tok.pad_token_id or 0)
        for pch in patches():
            out = runner.score(batches, [pch])

            def fin(lp, cnt):
                acc = accuracy(meta, lp)
                o = {t: acc[t][REPORT[t]] for t in TASKS}
                o["avg"] = sum(o.values()) / len(TASKS)
                o["detail"] = acc
                return o
            record("tasks", out, fin, [pch])


if __name__ == "__main__":
    main()
