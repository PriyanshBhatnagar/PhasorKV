"""Figures for why the attention-aware basis and the mean split work together.

  fig_anchor_error   per-token logit error vs alpha_t, one panel per mean handling
                     (attention-aware basis), with the fitted (alpha - a)^2 law
  fig_synergy_bars   attention-weighted logit error, basis x mean handling, per model
  fig_token_groups   where attention goes and which tokens each method gets wrong
  fig_ppl_grid       perplexity, basis x mean handling (first 4 tokens exact everywhere)

  python scripts/plot_synergy.py --tags llama31_8b_inst,longchat7b
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
# color = mean handling (validated categorical slots 1-3, all pairs); basis = marker / fill
MEAN = {"none": ("no mean handling", "#1baf7a"), "center": ("fixed-mean centering", "#eb6834"),
        "alpha": ("mean split (ours)", "#2a78d6")}
BASIS = {"pca": "PCA basis", "diag": "attention-aware basis"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"llama31_8b_inst": "Llama-3.1-8B-Instruct", "llama31_8b": "Llama-3.1-8B", "longchat7b": "LongChat-7B-32K"}


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def variant(basis, mean):
    return f"{basis}-{'wo' if basis == 'diag' else 'pca'}-{mean}"


def save(fig, name, out):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_anchor_error(S, model, out, layer, suffix=""):
    D = S["layers"][layer]
    a, m = D["alpha"].flatten(), D["mass"].flatten()
    T = D["alpha"].shape[1]
    first = torch.zeros_like(D["alpha"], dtype=torch.bool)
    first[:, 0] = True
    first = first.flatten()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    base = D["err"][variant("diag", "alpha")].flatten()
    for ax, mean in zip(axes, ("none", "center", "alpha")):
        label, col = MEAN[mean]
        e = D["err"][variant("diag", mean)].flatten()
        ax.scatter(a[~first], e[~first], s=3, color=col, alpha=0.25, lw=0, rasterized=True)
        ax.scatter(a[first], e[first], s=40, color=col, edgecolor=INK, lw=0.8, zorder=3)
        anchor = {"none": torch.zeros_like(a), "center": torch.ones_like(a), "alpha": a}[mean]
        if mean != "alpha":
            # least-squares fit of the excess error to c (alpha - a)^2
            x = (a - anchor) ** 2
            c = ((e - base) * x).sum() / (x * x).sum()
            g = torch.linspace(float(a.min()), float(a.max()), 200)
            ga = torch.zeros_like(g) if mean == "none" else torch.ones_like(g)
            ax.plot(g, c * (g - ga) ** 2 + base.median(), color=INK, lw=1.2, ls="--")
            ax.text(0.97, 0.93, f"fit: {c:.1f}·(α−{'0' if mean == 'none' else '1'})²", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color=INK)
        else:
            ax.text(0.97, 0.93, "anchor = α: no mean term left", transform=ax.transAxes, ha="right",
                    va="top", fontsize=8, color=INK)
        share = float(m[first].sum() / m.sum())
        ax.annotate(f"first token\n({100 * share:.0f}% of attention)", (float(a[first].mean()), float(e[first].mean())),
                    xytext=(12, 4), textcoords="offset points", fontsize=7.5, color=INK2)
        ax.set_yscale("log")
        ax.set_title(label, fontsize=10, color=INK)
        ax.set_xlabel("α_t (token's share of the mean direction)", fontsize=8.5, color=INK2)
        style(ax)
    axes[0].set_ylabel("per-token logit error", fontsize=8.5, color=INK2)
    fig.suptitle(f"{model}, layer {layer}: each anchor fails where α_t is far from it", fontsize=10.5, color=INK, y=1.02)
    save(fig, "fig_anchor_error" + suffix, out)


def attn_weighted(S, v, layers):
    return sum(S["summary"][l]["err_attn"][v] for l in layers) / len(layers)


def fig_synergy_bars(SS, out):
    fig, axes = plt.subplots(1, len(SS), figsize=(4.6 * len(SS), 3.3), squeeze=False)
    for ax, (model, S) in zip(axes[0], SS.items()):
        layers = [l for l in S["summary"] if l > 0]
        ours = attn_weighted(S, variant("diag", "alpha"), layers)
        w = 0.26
        for bi, basis in enumerate(("pca", "diag")):
            for mi, mean in enumerate(("none", "center", "alpha")):
                val = attn_weighted(S, variant(basis, mean), layers)
                x = bi + (mi - 1) * (w + 0.02)
                ax.bar(x, val, width=w, color=MEAN[mean][1], edgecolor="white", lw=2)
                ax.text(x, val * 1.15, f"{val / ours:.0f}×" if val / ours >= 1.95 else f"{val / ours:.1f}×",
                        ha="center", fontsize=7.5, color=INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([BASIS["pca"], BASIS["diag"]], fontsize=9, color=INK)
        ax.set_yscale("log")
        ax.set_title(model, fontsize=10, color=INK)
        style(ax)
    axes[0][0].set_ylabel("attention-weighted logit error\n(mean over layers; × = relative to ours)", fontsize=8.5, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MEAN[k][1]) for k in ("none", "center", "alpha")]
    fig.legend(handles, [MEAN[k][0] for k in ("none", "center", "alpha")], loc="upper center", ncol=3,
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.08))
    save(fig, "fig_synergy_bars", out)


def fig_token_groups(S, model, out, suffix=""):
    groups = ("first token", "mid-seq. sinks\n(α < 0.5)", "partial\n(|α−1| ≥ 0.2)", "ordinary\n(|α−1| < 0.2)")
    tok_share = {g: [] for g in groups}
    att_share = {g: [] for g in groups}
    err = {v: {g: [] for g in groups} for v in (variant("pca", "alpha"), variant("diag", "center"),
                                                 variant("diag", "none"), variant("diag", "alpha"))}
    for l, D in S["layers"].items():
        a, m = D["alpha"], D["mass"]
        first = torch.zeros_like(a, dtype=torch.bool)
        first[:, 0] = True
        masks = {groups[0]: first, groups[1]: (a < 0.5) & ~first,
                 groups[2]: ((a - 1).abs() >= 0.2) & (a >= 0.5) & ~first, groups[3]: ((a - 1).abs() < 0.2) & ~first}
        for g, k in masks.items():
            tok_share[g].append(float(k.float().mean()))
            att_share[g].append(float(m[k].sum() / m.sum()))
            for v in err:
                if k.any():
                    err[v][g].append(float(D["err"][v][k].mean()))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.3), gridspec_kw=dict(width_ratios=[1, 1.35]))
    x = range(len(groups))
    a1.bar([i - 0.2 for i in x], [100 * mean(tok_share[g]) for g in groups], width=0.38, color="#86b6ef", label="share of tokens")
    a1.bar([i + 0.2 for i in x], [100 * mean(att_share[g]) for g in groups], width=0.38, color="#1c5cab", label="share of attention")
    a1.set_xticks(list(x))
    a1.set_xticklabels(groups, fontsize=7.5, color=INK)
    a1.text(0 - 0.2, 1.0, f"{100 * mean(tok_share[groups[0]]):.2f}%", ha="center", fontsize=7, color=INK2)
    a1.set_ylabel("%", fontsize=8.5, color=INK2)
    a1.set_title("Tokens vs where attention goes", fontsize=10, color=INK)
    a1.legend(frameon=False, fontsize=8)
    style(a1)
    show = [(variant("diag", "none"), MEAN["none"][1], "attention-aware, no mean handling", "o"),
            (variant("diag", "center"), MEAN["center"][1], "attention-aware, centering", "o"),
            (variant("pca", "alpha"), MEAN["alpha"][1], "PCA, mean split", "s"),
            (variant("diag", "alpha"), MEAN["alpha"][1], "attention-aware, mean split (ours)", "o")]
    for j, (v, col, lab, mk) in enumerate(show):
        ys = [mean(err[v][g]) for g in groups]
        a2.plot([i + (j - 1.5) * 0.08 for i in x], ys, mk, color=col, ms=7,
                mfc=col if "PCA" not in lab else "white", mew=1.6, label=lab)
    a2.set_yscale("log")
    a2.set_xticks(list(x))
    a2.set_xticklabels(groups, fontsize=7.5, color=INK)
    a2.set_ylabel("mean per-token logit error", fontsize=8.5, color=INK2)
    a2.set_title("Which tokens each combination gets wrong", fontsize=10, color=INK)
    a2.legend(frameon=False, fontsize=7.5, loc="upper right")
    style(a2)
    fig.suptitle(model, fontsize=10.5, color=INK, y=1.02)
    save(fig, "fig_token_groups" + suffix, out)


def fig_ppl_grid(out):
    rows = []
    for tag in ("llama31_8b", "longchat7b"):
        p = os.path.join(ROOT, tag, "ablation_ppl.json")
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        cell = lambda b, mn: R.get(f"{variant(b, mn)}/starq.sink4@0.85")
        if all(cell(b, mn) for b in ("pca", "diag") for mn in ("none", "center", "alpha")):
            rows.append((NAMES[tag], {(b, mn): cell(b, mn)["wikitext2"] for b in ("pca", "diag") for mn in ("none", "center", "alpha")}))
    if not rows:
        print("ppl grid incomplete, skipped")
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(4.4 * len(rows), 3.2), squeeze=False)
    for ax, (model, C) in zip(axes[0], rows):
        xs = range(3)
        for b, ls, mk in (("pca", "--", "s"), ("diag", "-", "o")):
            ys = [C[(b, mn)] for mn in ("none", "center", "alpha")]
            ax.plot(xs, ys, ls=ls, color=INK2, lw=1.2, zorder=1)
            for x, mn, y in zip(xs, ("none", "center", "alpha"), ys):
                ax.plot(x, y, mk, color=MEAN[mn][1], ms=8, mfc=MEAN[mn][1] if b == "diag" else "white", mew=1.8, zorder=2)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.2f}", (x, y), xytext=(9 if b == "diag" else -9, 0), textcoords="offset points",
                            ha="left" if b == "diag" else "right", va="center", fontsize=7, color=INK2)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(["no mean\nhandling", "fixed-mean\ncentering", "mean split\n(ours)"], fontsize=8, color=INK)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(model, fontsize=10, color=INK)
        style(ax)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=INK2, ls="-", marker="o", mfc=INK2, ms=6),
                Line2D([], [], color=INK2, ls="--", marker="s", mfc="white", ms=6)],
               [BASIS["diag"], BASIS["pca"]], loc="upper center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.07))
    axes[0][0].set_ylabel("WikiText-2 perplexity at 85%\n(first 4 tokens exact for all)", fontsize=8.5, color=INK2)
    save(fig, "fig_ppl_grid", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", default="llama31_8b_inst,longchat7b")
    p.add_argument("--data", default="c4")
    p.add_argument("--layer", type=int, default=16)
    args = p.parse_args()
    out = os.path.join(ROOT, "figures")
    os.makedirs(out, exist_ok=True)
    SS = {}
    for tag in args.tags.split(","):
        f = os.path.join(ROOT, tag, f"synergy_{args.data}.pt")
        if os.path.exists(f):
            SS[NAMES.get(tag, tag)] = torch.load(f)
    for tag, (model, S) in zip([t for t in args.tags.split(",") if NAMES.get(t, t) in SS], SS.items()):
        fig_anchor_error(S, model, out, args.layer, "_" + tag)
        fig_token_groups(S, model, out, "_" + tag)
    fig_synergy_bars(SS, out)
    fig_ppl_grid(out)
    print("figures in", out)


if __name__ == "__main__":
    main()
