"""Decode-attention benchmark: PhasorKV latent kernels (per-head K vs joint K) against bf16 and FP8
KV caches, over all layers of the model, per decode step.

Shapes per layer come from kernel_shapes.py (the real allocation of each method); factors and
cache contents are random (timing does not depend on them). Every configuration is timed as one
CUDA graph over all layers (attention only: the kernel, the split combine with the exact sink and
recent-window tokens, and the V rebuild), on caches far larger than the L2. The dense baselines
time as many layers as fit and scale to the full depth.

  python scripts/bench_decode.py --tag llama31_8b_inst --lens 4096,16384,65536 --batches 1,8,32
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phasorkv.kernels import latent_attn as la
from phasorkv.kernels.dense_attn import dense_decode

N_Q, N_KV, D_HEAD, D_MODEL, N_LAYERS = 32, 8, 128, 4096, 32
EXACT = 4 + 128                       # attention sinks + recent window, bf16


def graph_time(fn, iters=20, warmup=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    del g
    return a.elapsed_time(b) / iters


def bench_latent(packs, B, T, dev, kw):
    """-> ms per step for all layers through latent_decode_fused (logits kernel, fused-softmax PV over
    the 2-bit-rest V cache, exact sink / recent-window tokens, V lift, normalization)."""
    caches = [la.random_cache(p, B, T, dev, v_layout="q2") for p in packs]
    q = torch.randn(B, N_Q, D_HEAD, device=dev).to(torch.bfloat16)
    seqlen = torch.full((B,), T, dtype=torch.int32, device=dev)
    ex = [(torch.randn(B, N_KV, EXACT, D_HEAD, device=dev).to(torch.bfloat16),
           torch.randn(B, N_KV, EXACT, D_HEAD, device=dev).to(torch.bfloat16)) for _ in packs]

    def full():
        for p, c, e in zip(packs, caches, ex):
            la.latent_decode_fused(p, q, c, seqlen, 4, e, **kw)
    t_full = graph_time(full)
    del caches, ex
    torch.cuda.empty_cache()
    return t_full, t_full


def bench_dense(B, T, dev, fp8, budget):
    per_layer = 2 * B * T * N_KV * D_HEAD * (1 if fp8 else 2)
    L = int(min(N_LAYERS, budget // per_layer))
    if L < 1 or L * per_layer < 4 * 128 * 2 ** 20 and L < N_LAYERS:
        return None
    dt = torch.float8_e4m3fn if fp8 else torch.bfloat16
    kv = [(torch.randn(B, T, N_KV, D_HEAD, device=dev).to(dt), torch.randn(B, T, N_KV, D_HEAD, device=dev).to(dt))
          for _ in range(L)]
    q = torch.randn(B, N_Q, D_HEAD, device=dev).to(torch.bfloat16)
    seqlen = torch.full((B,), T, dtype=torch.int32, device=dev)

    def run():
        for k, v in kv:
            dense_decode(q, k, v, seqlen, 1.0, 1.0)
    t = graph_time(run) * N_LAYERS / L
    del kv
    torch.cuda.empty_cache()
    return t


def bench_sdpa(B, T, dev, budget):
    per_layer = 2 * B * T * N_KV * D_HEAD * 2
    L = int(min(N_LAYERS, budget // per_layer))
    if L < 1:
        return None
    kv = [(torch.randn(B, N_KV, T, D_HEAD, device=dev).to(torch.bfloat16),
           torch.randn(B, N_KV, T, D_HEAD, device=dev).to(torch.bfloat16)) for _ in range(L)]
    q = torch.randn(B, N_Q, 1, D_HEAD, device=dev).to(torch.bfloat16)

    def run():
        for k, v in kv:
            torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    try:
        t = graph_time(run) * N_LAYERS / L
    except Exception as e:                                         # backend without GQA / graphs
        print("  sdpa failed:", type(e).__name__, str(e)[:120], flush=True)
        t = None
    del kv
    torch.cuda.empty_cache()
    return t


def main():
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--configs", default="", help="kernel_shapes.json keys (default: all)")
    p.add_argument("--lens", default="4096,16384,32768,65536")
    p.add_argument("--batches", default="1,8,32")
    p.add_argument("--budget-gb", type=float, default=70.0, help="cache memory the benchmark may use")
    p.add_argument("--kcfg", default="", help="kernel options, e.g. bt_k=64,nb=128 (default: KCFG)")
    p.add_argument("--no-dense", action="store_true")
    p.add_argument("--out", default="bench_decode.json")
    args = p.parse_args()
    dev = "cuda"
    root = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    shapes = json.load(open(os.path.join(root, "kernel_shapes.json")))
    names = [c for c in args.configs.split(",") if c] or list(shapes)
    budget = args.budget_gb * 2 ** 30
    kw = eval(f"dict({args.kcfg})")
    out_path = os.path.join(root, args.out)
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    lens = [int(x) for x in args.lens.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    print(torch.cuda.get_device_name(), "kernel", {**la.KCFG, **kw}, flush=True)

    for name in names:
        layers = shapes[name]["layers"]
        packs = [la.synthetic_pack(N_Q, N_KV, D_HEAD, D_MODEL, [(h, r) for h, r in l["k_blocks"]], l["rv"],
                                   device=dev, seed=i, v_no=l.get("v_no"))
                 for i, l in sorted(((int(i), l) for i, l in layers.items() if l))]
        bpt = sum(p.bytes_per_token("q2") for p in packs)
        bpt_packed = sum(p.bytes_per_token_packed() for p in packs)
        flops = sum(p.k_rebuild_flops for p in packs)
        for p_ in packs:                                      # the benchmark never runs the write path
            p_.B_k = p_.B_v = None
        print(f"{name}: {bpt / 1024:.1f} KiB/token (stored: 4-bit K, 2-bit V rest tier), {bpt_packed / 1024:.1f} KiB/token fully packed, "
              f"K rebuild {flops / 1e6:.2f} MFLOP/token", flush=True)
        for T in lens:
            for B in batches:
                key = f"{name}|T{T}|B{B}|" + ",".join(f"{k}={v}" for k, v in {**la.KCFG, **kw}.items())
                if key in res:
                    continue
                if B * T * bpt > budget:
                    print(f"  T={T} B={B}: over budget", flush=True)
                    continue
                t0 = time.time()
                t_full, t_kern = bench_latent(packs, B, T, dev, kw)
                gbs = B * T * bpt / (t_kern * 1e-3) / 1e9
                res[key] = dict(ms=t_full, kernel_ms=t_kern, gbs=gbs, tflops=B * T * flops / (t_kern * 1e-3) / 1e12)
                print(f"  T={T:6d} B={B:3d}: {t_full:8.3f} ms/step ({t_kern:.3f} kernels)  {gbs:6.0f} GB/s  "
                      f"{res[key]['tflops']:6.1f} TFLOP/s K rebuild  [{time.time() - t0:.0f}s]", flush=True)
                json.dump(res, open(out_path, "w"), indent=1)

    if not args.no_dense:
        for T in lens:
            for B in batches:
                for kind in ("bf16", "fp8", "sdpa-bf16"):
                    key = f"{kind}|T{T}|B{B}"
                    if key in res:
                        continue
                    t = bench_sdpa(B, T, dev, budget) if kind == "sdpa-bf16" else \
                        bench_dense(B, T, dev, kind == "fp8", budget)
                    if t is None:
                        continue
                    bpt = N_LAYERS * 2 * N_KV * D_HEAD * (1 if kind == "fp8" else 2)
                    res[key] = dict(ms=t, gbs=B * T * bpt / (t * 1e-3) / 1e9)
                    print(f"  {kind:9s} T={T:6d} B={B:3d}: {t:8.3f} ms/step  {res[key]['gbs']:6.0f} GB/s", flush=True)
                    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
