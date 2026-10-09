"""Long-context diagnostic: attention error of each basis on real task queries.

On N LongBench prompts (built exactly as run_longbench.py does), the last --queries
positions (the question and answer prompt) are the queries. Per layer and basis
variant, with every variant compressed the same way (default starq.sink4@0.85):
  err_far / err_mid / err_near   attention-weighted logit error of those queries on
                                 context keys > 4096, 512-4096 and < 512 tokens back
  attn_far ...                   the attention mass in each distance band (bf16)
and, per kv head and RoPE pair, the query energy of these test queries next to the
calibration estimate (q2 in stats.pt), to see whether retrieval queries read
directions the calibration text rarely does.

  python scripts/longctx_diag.py --model unsloth/Meta-Llama-3.1-8B-Instruct --tag llama31_8b_inst \\
      --task 2wikimqa --n 20
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from phasorkv.engine import load, head_dims, Layerwise
from phasorkv.factorize import kv_query_energy
from phasorkv.kvmethods import LatentPatch
from run_longbench import longbench_files, encode
from synergy_analysis import rope

VARIANTS = ["diag-wo-alpha", "pca-pca-alpha", "pca-pca-center", "diag-wo-center", "full-wo-none"]
BANDS = {"near": (0, 512), "mid": (512, 4096), "far": (4096, 10 ** 9)}


def main():
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--task", default="2wikimqa")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--queries", type=int, default=64)
    p.add_argument("--method", default="starq.sink4@0.85")
    p.add_argument("--max-length", type=int, default=31500)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    variants = args.variants.split(",")

    root = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    cache = os.path.join(os.path.dirname(__file__), "..", "results", "longbench_data")
    prompts, _, _ = longbench_files(cache)
    model, tok = load(args.model)
    n_q, n_kv, d = head_dims(model)
    rep = n_q // n_kv
    run = Layerwise(model, args.device, batch=1)
    data = [json.loads(l) for l in open(os.path.join(cache, "data", f"{args.task}.jsonl"))][:args.n]
    seqs = [torch.tensor(encode(tok, prompts[args.task].format(**o), args.task, args)) for o in data]
    hs = [run.embed(s[None]) for s in seqs]
    stats = {s["layer"]: s for s in torch.load(os.path.join(root, "stats.pt"))["layers"]}
    kind, frac = args.method.split("@")
    Q = args.queries
    out = dict(task=args.task, n=args.n, variants=variants, method=args.method, layers={})

    for i, layer in enumerate(run.layers):
        layer.to(run.device)
        attn = layer.self_attn
        preps = {v: torch.load(os.path.join(root, f"latent_{v}", f"L{i}.pt")) for v in variants}
        acc = {v: {b: 0.0 for b in BANDS} for v in variants}
        mass = {b: 0.0 for b in BANDS}
        qe = torch.zeros(n_q, d // 2, device=run.device, dtype=torch.float64)
        nq = 0
        for s, h in zip(seqs, hs):
            T = h.shape[1]
            hb = h.to(run.device)
            x = layer.input_layernorm(hb)
            cos, sin = run.position_embeddings(T, torch.float32)
            cos, sin = cos[0].float(), sin[0].float()
            qraw = attn.q_proj(x[:, T - Q:]).float().view(1, Q, n_q, d)
            qe += (qraw[..., : d // 2] ** 2 + qraw[..., d // 2:] ** 2).sum(1)[0].double()
            nq += Q
            q = rope(qraw, cos[T - Q:], sin[T - Q:]).permute(0, 2, 1, 3)                      # [1, H, Q, d]
            k = attn.k_proj(x).float().view(1, T, n_kv, d)
            kh = rope(k, cos, sin).permute(0, 2, 1, 3).repeat_interleave(rep, 1)              # [1, H, T, d]
            qpos = torch.arange(T - Q, T, device=run.device)
            dist = qpos[:, None] - torch.arange(T, device=run.device)[None]                    # [Q, T]
            causal = dist >= 0
            prob = torch.softmax((q @ kh.transpose(-1, -2) / d ** 0.5).masked_fill(~causal, float("-inf")), -1)
            band = {b: causal & (dist >= lo) & (dist < hi) for b, (lo, hi) in BANDS.items()}
            for b, m in band.items():
                mass[b] += float((prob * m).sum() / (n_q * Q))
            for v in variants:
                patch = LatentPatch(kind, float(frac))
                undo = patch.patch(i, layer, preps[v], n_kv, d, n_q, run.device)
                k_hat = attn.k_proj(x).float().view(1, T, n_kv, d)
                undo()
                e = rope(k_hat - k, cos, sin).permute(0, 2, 1, 3).repeat_interleave(rep, 1)
                de = (q @ e.transpose(-1, -2)) ** 2 / d                                         # [1, H, Q, T]
                for b, m in band.items():
                    acc[v][b] += float((de * prob * m).sum() / (n_q * Q))                       # attention-weighted
                del e, de, k_hat
            del x, q, k, kh, prob
        lam_test = kv_query_energy((qe / nq).float(), n_kv)                                      # [n_kv, d/2]
        lam_cal = kv_query_energy(stats[i]["q2"].to(run.device).float(), n_kv)
        rel = lambda L: L / L.sum(-1, keepdim=True)
        shift = 0.5 * (rel(lam_test) - rel(lam_cal)).abs().sum(-1)                               # total variation per head
        out["layers"][i] = dict(err={v: {b: acc[v][b] / len(seqs) for b in BANDS} for v in variants},
                                mass={b: mass[b] / len(seqs) for b in BANDS},
                                lam_test=lam_test.cpu(), lam_cal=lam_cal.cpu(), shift_tv=shift.cpu())
        L = out["layers"][i]
        print(f"layer {i:2d} attn mass near/mid/far {L['mass']['near']:.2f}/{L['mass']['mid']:.2f}/{L['mass']['far']:.2f} "
              f"| query-energy shift (TV) {shift.mean():.3f} | far err " +
              " ".join(f"{v}={L['err'][v]['far']:.2e}" for v in variants), flush=True)
        hs = [run.run_layer(layer, h) for h in hs]
        layer.cpu()
        del preps
        torch.cuda.empty_cache()
    torch.save(out, os.path.join(root, f"longctx_diag_{args.task}.pt"))
    tot = {v: {b: sum(out["layers"][l]["err"][v][b] for l in out["layers"]) for b in BANDS} for v in variants}
    print("\nsum over layers, attention-weighted logit error by distance band:")
    for v in variants:
        print(f"  {v:16s} " + "  ".join(f"{b}={tot[v][b]:.3e}" for b in BANDS))
    print("done", flush=True)


if __name__ == "__main__":
    main()
