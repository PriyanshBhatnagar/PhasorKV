"""Per-layer latent bases (objective B) and per-unit quantization distortions.

For every layer: the full-rank K and V bases from bases.py, and, on calibration
latents from the exact model, the mean squared error each 32-dim unit takes
under every format in kvmethods.MENU_ALL (K along the latent, with and without
the block Hadamard; V along tokens). Because the bases make latent MSE equal to
the logit / output error, these tables are what the allocator trades off.

  python scripts/prep_latent.py --model lmsys/longchat-7b-v1.5-32k --tag longchat7b
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phasorkv import quant
from phasorkv.bases import k_basis, v_basis
from phasorkv.engine import load, head_dims, Layerwise, token_stream, random_windows
from phasorkv.kvmethods import MENU_ALL, UNIT, k_rotations


def unit_distortion(c, fmt, axis, chunk=2048):
    """Mean squared error per 32-dim unit of quantizing c in format fmt.

    axis=-1: c is [N, n_units, UNIT], quantized along the unit, per token (K).
    axis=1:  c is [W, T, n_units*UNIT], quantized along tokens, per channel (V).
    NVFP4's second-level scale is per unit over all of c; the rest is chunked."""
    n_units = c.shape[1] if axis == -1 else c.shape[-1] // UNIT
    count = c.shape[0] if axis == -1 else c.shape[0] * c.shape[1]
    if fmt == "bf16":
        return torch.zeros(n_units, device=c.device)
    sq = lambda e: e.sum(-1).sum(0) if axis == -1 else e.sum(dim=(0, 1)).view(-1, UNIT).sum(-1)
    if fmt == "drop":
        return sum(sq(x ** 2) for x in c.split(chunk if axis == -1 else 1)) / count
    elem, _, kind = quant.FORMATS[fmt]
    vmax = quant.ELEM[elem][1]
    second = None
    if kind == "e4m3":
        if axis == -1:
            second = (c.abs().amax(dim=(0, 2)) / (448 * vmax)).clamp_min(1e-30)[None, :, None, None]
        else:
            s = c.abs().amax(dim=(0, 1)).view(-1, UNIT).amax(-1).clamp_min(1e-30) / (448 * vmax)
            second = s.repeat_interleave(UNIT)[None, :, None, None]
    acc = torch.zeros(n_units, device=c.device)
    for x in c.split(chunk if axis == -1 else 1):
        acc += sq((quant.block_quant(x, fmt, axis, second) - x) ** 2)
    return acc / count


def main():
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True, help="reads results/<tag>/stats.pt, writes results/<tag>/latent/")
    p.add_argument("--calib-windows", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    out_dir = os.path.join(root, "latent")
    os.makedirs(out_dir, exist_ok=True)
    stats = {s["layer"]: s for s in torch.load(os.path.join(root, "stats.pt"))["layers"]}
    dev = torch.device(args.device)
    model, tok = load(args.model)
    n_q, n_kv, d = head_dims(model)
    run = Layerwise(model, args.device, batch=4)
    calib = random_windows(token_stream("wikitext2", "train", tok), args.calib_windows, args.seqlen, seed=1)
    h = run.embed(calib)
    rots = [R.to(dev) for R in k_rotations(n_kv, d)]

    for i, layer in enumerate(run.layers):
        t0 = time.time()
        layer.to(dev)
        attn = layer.self_attn
        Wk, Wv, Wo = (m.weight.float() for m in (attn.k_proj, attn.v_proj, attn.o_proj))
        with torch.no_grad():
            xs = torch.cat([layer.input_layernorm(hb).float() for _, hb in run.batches(h)])   # [W, T, D]
        # Split off the mean direction: x = alpha_t * mu + P x, P = I - mu mu^T / |mu|^2.
        # The mean input gives every ordinary token nearly the same key / value
        # (95%+ of the K latent's energy) and carries no information; a per-token
        # scalar alpha (bf16, shared by K and V) carries it instead, which also
        # keeps attention sinks exact (their alpha is ~0, where subtracting the
        # mean would leave them as -mu, far outside the fitted subspace). The
        # basis is fitted to P Cx P and folded with P, so B x = B P x.
        mu = xs.reshape(-1, xs.shape[-1]).double().mean(0)
        mh = mu / mu.norm()
        P = torch.eye(mu.numel(), device=dev, dtype=torch.float64) - torch.outer(mh, mh)
        Sx = P @ stats[i]["Cx"].to(dev).double() @ P
        BK, AK, eK = k_basis(Wk, Sx, stats[i]["q2"], n_kv, d, device=dev)
        BV, AV, eV = v_basis(Wv, Wo, Sx, n_kv, d, n_q, device=dev)
        BK, BV = (BK @ P).float(), (BV @ P).float()
        AK, AV, eK, eV = AK.float(), AV.float(), eK.float(), eV.float()
        mu = mu.float()
        del P, Sx
        torch.cuda.empty_cache()
        # exactness at full rank: W = A B + (W mu) mu^T / |mu|^2
        mean_part = lambda W: torch.outer(W @ mu, mu) / mu.dot(mu)
        WK_hat = torch.cat([AK[g] @ BK[g * d:(g + 1) * d] for g in range(n_kv)]) + mean_part(Wk)
        ek = ((WK_hat - Wk).norm() / Wk.norm()).item()
        ev = ((AV @ BV + mean_part(Wv) - Wv).norm() / Wv.norm()).item()
        del WK_hat
        cK = (xs.reshape(-1, xs.shape[-1]) @ BK.T)                                           # [N, n_kv*d]
        cKH = torch.cat([cK[:, g * d:(g + 1) * d] @ rots[g].T for g in range(n_kv)], dim=1)
        cV = xs @ BV.T                                                                        # [W, T, R]
        uK = lambda c: c.view(c.shape[0], -1, UNIT)
        DK = torch.stack([unit_distortion(uK(cK), f, -1) for f in MENU_ALL], 1)
        DK_H = torch.stack([unit_distortion(uK(cKH), f, -1) for f in MENU_ALL], 1)
        DV = torch.stack([unit_distortion(cV, f, 1) for f in MENU_ALL], 1)
        torch.save(dict(mu=mu.cpu(), kbar=(Wk @ mu).cpu(), vbar=(Wv @ mu).cpu(),
                        BK=BK.cpu(), AK=AK.cpu(), eK=eK.float().cpu(), BV=BV.cpu(), AV=AV.cpu(),
                        eV=eV.float().cpu(), DK=DK.cpu(), DK_H=DK_H.cpu(), DV=DV.cpu(), menu=MENU_ALL),
                   os.path.join(out_dir, f"L{i}.pt"))
        rel = lambda D, f: (D[:, MENU_ALL.index(f)].sum() / D[:, MENU_ALL.index("drop")].sum()).item()
        print(f"layer {i:2d} ({time.time() - t0:4.1f}s) basis err K {ek:.1e} V {ev:.1e} | "
              f"rel MSE nvfp4 K {rel(DK, 'nvfp4'):.4f} K+H {rel(DK_H, 'nvfp4'):.4f} V(tok) {rel(DV, 'nvfp4'):.4f} | "
              f"mxfp6 K+H {rel(DK_H, 'mxfp6'):.5f} V {rel(DV, 'mxfp6'):.5f}", flush=True)
        del xs, cK, cKH, cV
        h = run.run_layer(layer, h)
        layer.cpu()
        torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
