"""Plots and a summary table from results/<tag>/spectrum.json.

  python scripts/plot_spectrum.py --tag longchat7b
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {  # structure -> colour; weighting -> line style
    "head": "#d1495b", "group2": "#edae49", "group4": "#c98b2c", "group8": "#8f6420",
    "joint": "#00798c", "freq": "#30638e",
    "band8": "#6a4c93", "band16": "#8e6cc0", "band24": "#b39ddb",
}
DASH = {"plain": ":", "x": "--", "xq": "-"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    root = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    R = json.load(open(os.path.join(root, "spectrum.json")))
    methods = list(R["layers"][0]["methods"])
    fracs = [r["frac"] for r in R["layers"][0]["methods"][methods[0]]]

    def gather(metric):
        """[method] -> [layer, budget] array."""
        return {m: np.array([[r[metric] for r in L["methods"][m]] for L in R["layers"]]) for m in methods}

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, metric, title in zip(axes[:3], ["kl", "logit", "out"],
                                 ["attention KL", "centred logit rel. MSE", "attention output rel. MSE"]):
        data = gather(metric)
        for m in methods:
            s, w = m.split(":")
            ax.plot(fracs, np.exp(np.log(data[m]).mean(0)), DASH[w], color=STYLE[s], marker="o", ms=3, label=m)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("K cache kept (fraction)")
        ax.set_title(f"{title} (geo-mean over layers)")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)

    ax = axes[3]
    kl, fl = gather("kl"), gather("flops")
    dense = R["layers"][0]["methods"][methods[0]][0]["flops_dense"]
    for m in methods:
        s, w = m.split(":")
        if w != "xq":
            continue
        ax.plot(fl[m][0] / dense, np.exp(np.log(kl[m]).mean(0)), "-", color=STYLE[s], marker="o", ms=3, label=m)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K-path decode FLOPs per cached token / dense attention")
    ax.set_title("accuracy vs compute (xq weighting)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.suptitle(f"{R['model']}  ({R['n_kv']} kv heads, head_dim {R['d']})")
    fig.tight_layout()
    fig.savefig(os.path.join(root, "spectrum.png"), dpi=130)

    # per-layer KL at 25% for the key methods
    fig, ax = plt.subplots(figsize=(10, 4))
    j = int(np.argmin([abs(f - 0.25) for f in fracs]))
    for m in methods:
        s, w = m.split(":")
        if w == "xq" or m == "head:plain":
            ax.plot([L["layer"] for L in R["layers"]], kl[m][:, j], DASH[w], color=STYLE[s], marker=".", label=m)
    ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_ylabel(f"attention KL at {fracs[j]:.0%} K kept")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "spectrum_layers.png"), dpi=130)

    # summary table
    lines = [f"# {R['model']}\n", "Geo-mean over layers of attention KL (lower is better)\n",
             "| method | " + " | ".join(f"{f:.4g}" for f in fracs) + " | K FLOPs/token @25% vs dense |",
             "|---" * (len(fracs) + 2) + "|"]
    for m in methods:
        geo = np.exp(np.log(kl[m]).mean(0))
        lines.append(f"| {m} | " + " | ".join(f"{v:.2e}" for v in geo) + f" | {fl[m][0, j] / dense:.1f}x |")
    open(os.path.join(root, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
