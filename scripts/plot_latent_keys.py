"""3D maps of key magnitudes over (token, channel): raw pre-RoPE keys vs our latent keys.

For one kv head in a few layers: |k| in the model's own channels, and |c| in the attention-aware
latent without the mean split and with it (the latent our cache stores, channels sorted by
eigenvalue). Plus a summary: per-channel RMS (the spectrum the bit tiers follow) and per-token
RMS (the token-to-token scale the per-token / block scales absorb). The first 4 tokens (attention
sinks, kept exact by the cache) are left out of the surfaces so the scale stays readable.

  python scripts/plot_latent_keys.py --model unsloth/Meta-Llama-3.1-8B --tag llama31_8b --layers 2,16,28
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phasorkv.engine import load, head_dims, Layerwise, token_stream, consecutive_windows

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]                # sequential ramp, light -> dark
CMAP = LinearSegmentedColormap.from_list("seqblue", BLUE)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"raw": ("raw pre-RoPE key (model channels)", "#52514e"),
          "none": ("latent, no mean split", "#1baf7a"),
          "alpha": ("latent, mean split (ours, stored)", "#2a78d6")}


def capture(args):
    model, tok = load(args.model)
    n_q, n_kv, d = head_dims(model)
    run = Layerwise(model, args.device, batch=1)
    win = consecutive_windows(token_stream("wikitext2", "test", tok), 1, args.tokens)
    h = run.embed(win)
    layers = [int(l) for l in args.layers.split(",")]
    g, out = args.head, {}
    for i, layer in enumerate(run.layers):
        layer.to(run.device)
        if i in layers:
            x = layer.input_layernorm(h.to(run.device)).float()[0]                       # [T, D]
            k = layer.self_attn.k_proj(x.to(h.dtype)).float().view(-1, n_kv, d)[:, g]
            rec = {"raw": k.cpu()}
            for name, v in (("none", "diag-wo-none"), ("alpha", "diag-wo-alpha")):
                BK = torch.load(os.path.join(ROOT, args.tag, f"latent_{v}", f"L{i}.pt"))["BK"].to(run.device)
                rec[name] = (x @ BK[g * d:(g + 1) * d].T).cpu()
            out[i] = rec
        h = run.run_layer(layer, h)
        layer.cpu()
        if i >= max(layers):
            break
    return out


def surface(ax, Z, title, zlabel):
    T, C = Z.shape
    X, Y = torch.meshgrid(torch.arange(T), torch.arange(C), indexing="ij")
    ax.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), cmap=CMAP, linewidth=0, antialiased=False,
                    rcount=min(T, 160), ccount=C)
    ax.set_title(title, fontsize=9.5, color=INK, pad=2)
    ax.set_xlabel("token", fontsize=8, color=INK2, labelpad=-2)
    ax.set_ylabel("channel", fontsize=8, color=INK2, labelpad=-2)
    ax.set_zlabel(zlabel, fontsize=8, color=INK2, labelpad=-2)
    ax.tick_params(labelsize=6.5, colors=INK2, pad=-2)
    ax.view_init(elev=28, azim=-58)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor("white")
        a.pane.set_edgecolor(GRID)


def plot(data, args):
    out = os.path.join(ROOT, "figures")
    os.makedirs(out, exist_ok=True)
    s0, s1 = 4, 4 + args.show
    for i, rec in data.items():
        fig = plt.figure(figsize=(14, 4.6))
        for j, name in enumerate(("raw", "none", "alpha")):
            ax = fig.add_subplot(1, 3, j + 1, projection="3d")
            surface(ax, rec[name][s0:s1].abs(), SERIES[name][0], "|value|")
        fig.suptitle(f"Layer {i}, kv head {args.head}: key magnitude over tokens {s0}-{s1 - 1} x 128 channels",
                     fontsize=11, color=INK)
        fig.subplots_adjust(left=0.01, right=0.99, wspace=0.02, top=0.9, bottom=0.02)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out, f"fig_latent3d_L{i}.{ext}"), dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(2, len(data), figsize=(4.3 * len(data), 6.4), squeeze=False)
    for col, (i, rec) in enumerate(data.items()):
        a1, a2 = axes[0][col], axes[1][col]
        for name in ("raw", "none", "alpha"):
            label, color = SERIES[name]
            v = rec[name][s0:]
            ch = v.pow(2).mean(0).sqrt()
            a1.plot(ch.numpy(), color=color, lw=1.6, label=label)
            if name == "raw":
                a1.plot(ch.sort(descending=True).values.numpy(), color=color, lw=1.2, ls="--",
                        label="raw, channels sorted by RMS")
            tok_rms = v.pow(2).mean(1).sqrt()
            a2.plot(tok_rms.numpy(), color=color, lw=0.9, label=label)
        a1.set_yscale("log")
        a1.set_title(f"layer {i}: per-channel RMS", fontsize=9.5, color=INK)
        a1.set_xlabel("channel (latent: sorted by eigenvalue)", fontsize=8, color=INK2)
        a2.set_yscale("log")
        a2.set_title(f"layer {i}: per-token RMS", fontsize=9.5, color=INK)
        a2.set_xlabel("token", fontsize=8, color=INK2)
        for a in (a1, a2):
            a.grid(axis="y", color=GRID, lw=0.7)
            a.set_axisbelow(True)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            a.tick_params(labelsize=7.5, colors=INK2)
    axes[0][0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_latent_spectrum.{ext}"), dpi=170)
    plt.close(fig)
    return out


def main():
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--layers", default="2,16,28")
    p.add_argument("--head", type=int, default=0)
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--show", type=int, default=256, help="tokens drawn in the 3D surfaces")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cache = os.path.join(ROOT, args.tag, f"latent_keys_h{args.head}.pt")
    data = torch.load(cache) if os.path.exists(cache) else capture(args)
    torch.save(data, cache)
    print("figures in", plot(data, args))


if __name__ == "__main__":
    main()
