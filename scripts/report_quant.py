"""Tables, breaking points and plots for the fake-quant study.

Breaking point of a method = the highest K+V compression at which its WikiText-2
perplexity stays within t of the uncompressed model, interpolated (in log PPL)
between the sweep points that bracket it.

  python scripts/report_quant.py --tags longchat7b,llama31_8b
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESH = (0.01, 0.03, 0.05, 0.10)
LABELS = {
    "lr": "B low-rank, bf16 latent",
    "starq": "B + STAR-KV quantizer (int4/int3 tiers)",
    "tqA": "B + NV tiers for K (nvint4/nvint3), STAR-KV V",
    "rd": "B + MSE rate-distortion MX/NV tiers (as first proposed)",
    "palu": "B + Hadamard + NVFP4 (Palu-style)",
    "lrnv": "B + NVFP4, no rotation",
}
COLORS = {"lr": "#999999", "starq": "#d1495b", "tqA": "#30638e", "rd": "#edae49",
          "palu": "#00798c", "lrnv": "#8f6420"}


def families(res):
    fam = {}
    for name, r in res.items():
        if "@" in name:
            k = name.split("@")[0]
            fam.setdefault(k, []).append((r["compression"], r))
    return {k: sorted(v, key=lambda x: x[0]) for k, v in fam.items()}


def breaking_point(points, base, t, key="wikitext2"):
    thr = base * (1 + t)
    prev = None
    for c, r in points:
        p = r[key]
        if p > thr:
            if prev is None:
                return f"<{c:.3f}"
            c0, p0 = prev
            f = (math.log(thr) - math.log(p0)) / (math.log(p) - math.log(p0))
            return f"{c0 + f * (c - c0):.3f}"
        prev = (c, p)
    return f">{prev[0]:.3f}" if prev else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="longchat7b,llama31_8b")
    args = ap.parse_args()
    root = os.path.join(os.path.dirname(__file__), "..", "results")
    for tag in args.tags.split(","):
        pp = os.path.join(root, tag, "quant_ppl.json")
        if not os.path.exists(pp):
            continue
        ppl = json.load(open(pp))
        tp = os.path.join(root, tag, "quant_tasks.json")
        tasks = json.load(open(tp)) if os.path.exists(tp) else {}
        base = ppl["bf16"]
        out = [f"# {tag}\n",
               f"bf16: WikiText-2 {base['wikitext2']:.3f}, C4 {base['c4']:.3f}"
               + (f", task avg {tasks['bf16']['avg']*100:.2f}" if "bf16" in tasks else "") + "\n",
               "## Breaking points (max K+V compression within t of bf16 WikiText-2 PPL)\n",
               "| method | " + " | ".join(f"+{int(t*100)}%" for t in THRESH) + " |",
               "|---" * (len(THRESH) + 1) + "|"]
        fam = families(ppl)
        for k, pts in fam.items():
            if k.startswith(("diag", "dq-", "ours")):
                continue
            out.append(f"| {LABELS.get(k, k)} | " + " | ".join(breaking_point(pts, base["wikitext2"], t) for t in THRESH) + " |")
        out += ["", "Single-point methods (post-RoPE quantization of the ordinary cache; STAR-KV trained):", "",
                "| method | compression | WikiText-2 | C4 | task avg |", "|---|---|---|---|---|"]
        for name, r in sorted(ppl.items(), key=lambda kv: kv[1]["compression"]):
            if "@" in name:
                continue
            ta = tasks.get(name, {}).get("avg")
            out.append(f"| {name} | {r['compression']:.3f} | {r['wikitext2']:.3f} | {r['c4']:.3f} | "
                       + (f"{ta*100:.2f}" if ta is not None else "-") + " |")
        out += ["", "## All sweep points", "", "| method | compression | WikiText-2 | C4 | K share of bits | K rank | V rank | task avg |",
                "|---|---|---|---|---|---|---|---|"]
        for k, pts in fam.items():
            for c, r in pts:
                name = next(n for n, rr in ppl.items() if rr is r)
                ta = tasks.get(name, {}).get("avg")
                out.append(f"| {name} | {c:.3f} | {r['wikitext2']:.3f} | {r['c4']:.3f} | {r.get('k_bits_share', 0):.2f} | "
                           f"{r.get('rank_k', 0):.2f} | {r.get('rank_v', 0):.2f} | " + (f"{ta*100:.2f}" if ta is not None else "-") + " |")
        if tasks:
            ts = [t for t in ("piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "openbookqa")]
            out += ["", "## Zero-shot tasks (acc for piqa/arc_e/winogrande, acc_norm for arc_c/hellaswag/obqa)", "",
                    "| method | compression | " + " | ".join(ts) + " | avg |", "|---" * (len(ts) + 3) + "|"]
            for name, r in sorted(tasks.items(), key=lambda kv: kv[1].get("compression", 0)):
                out.append(f"| {name} | {r.get('compression', 0):.3f} | " + " | ".join(f"{r[t]*100:.1f}" for t in ts)
                           + f" | {r['avg']*100:.2f} |")
        open(os.path.join(root, tag, "quant_summary.md"), "w").write("\n".join(out) + "\n")
        print("\n".join(out))

        fig, axes = plt.subplots(1, 2 if tasks else 1, figsize=(14 if tasks else 7, 5), squeeze=False)
        ax = axes[0][0]
        for k, pts in fam.items():
            if k not in LABELS:
                continue
            ax.plot([c for c, _ in pts], [r["wikitext2"] for _, r in pts], "-o", ms=3, color=COLORS[k], label=LABELS[k])
        for name, r in ppl.items():
            if "@" not in name and name != "bf16":
                ax.scatter(r["compression"], r["wikitext2"], marker="x" if not r.get("trained") else "*", s=60,
                           color="black" if not r.get("trained") else "#6a4c93", zorder=5)
                ax.annotate(name, (r["compression"], r["wikitext2"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(base["wikitext2"], color="black", lw=0.8, ls=":")
        for t in (0.01, 0.05):
            ax.axhline(base["wikitext2"] * (1 + t), color="gray", lw=0.6, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("K+V cache compression")
        ax.set_ylabel("WikiText-2 PPL")
        ax.set_title(f"{tag}: perplexity vs compression (training-free)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
        if tasks:
            ax = axes[0][1]
            for k in LABELS:
                pts = sorted((r["compression"], r["avg"]) for n, r in tasks.items() if n.split("@")[0] == k and "@" in n)
                if pts:
                    ax.plot(*zip(*pts), "-o", ms=3, color=COLORS[k], label=LABELS[k])
            for name, r in tasks.items():
                if "@" not in name:
                    ax.scatter(r["compression"], r["avg"], marker="x" if not r.get("trained") else "*", s=60,
                               color="black" if not r.get("trained") else "#6a4c93", zorder=5)
                    ax.annotate(name, (r["compression"], r["avg"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.axhline(tasks["bf16"]["avg"], color="black", lw=0.8, ls=":")
            ax.axhline(tasks["bf16"]["avg"] - 0.01, color="gray", lw=0.6, ls="--")
            ax.set_xlabel("K+V cache compression")
            ax.set_ylabel("zero-shot average")
            ax.set_title("6-task zero-shot average")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(root, tag, "quant.png"), dpi=130)


if __name__ == "__main__":
    main()
