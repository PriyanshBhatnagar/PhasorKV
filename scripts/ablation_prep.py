"""Latent bases for the ablation: which part of the method buys the accuracy.

A variant is <K weighting>-<V weighting>-<mean handling>:

  K weighting   wsvd  SVD of the weights alone (STAR-KV / Palu initialisation)
                pca   key covariance (data-aware, unweighted: Eigen Attention, KVTC)
                full  full pre-RoPE query second moment per kv head, pooled over its
                      query heads (KQ-SVD's objective; SAKI's per head)
                diag  ours: query energy per RoPE pair, averaged over relative position
  V weighting   wsvd, pca as above; wo = W_o Gram (ours, and KQ-SVD's V)
  mean          none    no centering, basis on the raw second moment
                center  fixed mean: c = B (x - mu), k = A c + W mu (SAKI, KVTC)
                alpha   ours: x = alpha_t mu + P x, alpha cached per token

Every variant is exact at full rank; only what rank truncation keeps differs.
Writes results/<tag>/latent_<variant>/L<i>.pt in prep_latent.py's format, with
only the bf16 and drop columns of the distortion tables (the rank-only methods
the ablation runs need nothing else).

  python scripts/ablation_prep.py --model unsloth/Meta-Llama-3.1-8B --tag llama31_8b \\
      --variants diag-wo-alpha,diag-wo-center,full-wo-alpha
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phasorkv.bases import _psd_pow
from phasorkv.engine import load, head_dims, Layerwise, token_stream, random_windows
from phasorkv.factorize import kv_query_energy
from phasorkv.kvmethods import MENU_ALL, UNIT

K_W, V_W, MEANS = ("wsvd", "pca", "full", "diag"), ("wsvd", "pca", "wo"), ("none", "center", "alpha")


def query_moments(model, run, tok, n_q, n_kv, d, path, windows=32, seqlen=2048):
    """E[q q^T] of pre-RoPE queries per kv head (summed over its query heads), on the
    calibration windows run_spectrum.py used for Cx and q2."""
    if os.path.exists(path):
        return torch.load(path)
    D = model.config.hidden_size
    h = run.embed(random_windows(token_stream("wikitext2", "train", tok), windows, seqlen))
    out = []
    for i, layer in enumerate(run.layers):
        layer.to(run.device)
        S = torch.zeros(n_kv, d, d, dtype=torch.float64, device=run.device)
        n = 0
        for _, hb in run.batches(h):
            x = layer.self_attn.q_proj(layer.input_layernorm(hb).reshape(-1, D)).float()
            q = x.reshape(-1, n_kv, n_q // n_kv, d)
            S += torch.einsum("ngra,ngrb->gab", q, q).double()
            n += q.shape[0]
        out.append((S / n).float().cpu())
        h = run.run_layer(layer, h)
        layer.cpu()
        print(f"query moments layer {i}", flush=True)
    torch.save(out, path)
    return out


def _eig_desc(G):
    G = 0.5 * (G + G.transpose(-1, -2))
    scale = G.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-300)
    e, U = torch.linalg.eigh((G / scale[..., None, None]).cpu())    # LAPACK, as in factorize._add
    e, U = e.to(G.device) * scale[..., None], U.to(G.device)
    return e.flip(-1).clamp_min(0), U.flip(-1)


def k_basis(Wk, C, kw, q2, Sq, n_kv, d):
    """-> B [n_kv*d, D], A [n_kv, d, d], energy [n_kv, d] (sorted within head)."""
    B, A, E = [], [], []
    lam = kv_query_energy(q2.double(), n_kv)
    lam = lam.clamp_min(lam.max() * 1e-12)
    for g in range(n_kv):
        M = Wk[g * d:(g + 1) * d]
        if kw == "diag":
            sq = torch.cat([lam[g], lam[g]]).sqrt()
            Qh, Qih = torch.diag(sq), torch.diag(1 / sq)
        elif kw == "full":
            Qh, Qih = _psd_pow(Sq[g], 0.5), _psd_pow(Sq[g], -0.5)
        else:
            Qh = Qih = torch.eye(d, dtype=Wk.dtype, device=Wk.device)
        Mw = Qh @ M
        e, U = _eig_desc(Mw @ C @ Mw.T)
        B.append(U.T @ Mw)
        A.append(Qih @ U)
        E.append(e)
    return torch.cat(B), torch.stack(A), torch.stack(E)


def v_basis(Wv, Wo, C, vw, n_kv, d, n_q):
    n_rep = n_q // n_kv
    if vw == "wo":
        Lh, Lih = [], []
        for g in range(n_kv):
            L = sum(Wo[:, h * d:(h + 1) * d].T @ Wo[:, h * d:(h + 1) * d]
                    for h in range(g * n_rep, (g + 1) * n_rep))
            Lh.append(_psd_pow(L, 0.5))
            Lih.append(_psd_pow(L, -0.5))
        Lh, Lih = torch.block_diag(*Lh), torch.block_diag(*Lih)
    else:
        Lh = Lih = torch.eye(n_kv * d, dtype=Wv.dtype, device=Wv.device)
    Mw = Lh @ Wv
    e, U = _eig_desc(Mw @ C @ Mw.T)
    return U.T @ Mw, Lih @ U, e


def drop_table(c, axis):
    """[n_units, len(MENU_ALL)]: zero for bf16, mean squared latent for drop, NaN otherwise."""
    if axis == -1:
        e = (c.view(c.shape[0], -1, UNIT) ** 2).sum(-1).mean(0)
    else:
        e = (c.reshape(-1, c.shape[-1]) ** 2).mean(0).view(-1, UNIT).sum(-1)
    t = torch.full((e.numel(), len(MENU_ALL)), float("nan"), device=c.device)
    t[:, MENU_ALL.index("bf16")] = 0
    t[:, MENU_ALL.index("drop")] = e
    return t


def main():
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--variants", required=True)
    p.add_argument("--calib-windows", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    variants = [v.split("-") for v in args.variants.split(",")]
    for kw, vw, mm in variants:
        assert kw in K_W and vw in V_W and mm in MEANS, (kw, vw, mm)
    root = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    stats = {s["layer"]: s for s in torch.load(os.path.join(root, "stats.pt"))["layers"]}
    dev = torch.device(args.device)
    model, tok = load(args.model)
    n_q, n_kv, d = head_dims(model)
    run = Layerwise(model, args.device, batch=4)
    Sq_all = None
    if any(kw == "full" for kw, _, _ in variants):
        Sq_all = query_moments(model, run, tok, n_q, n_kv, d, os.path.join(root, "stats_q.pt"))
    for kw, vw, mm in variants:
        os.makedirs(os.path.join(root, f"latent_{kw}-{vw}-{mm}"), exist_ok=True)
    # the same calibration latents prep_latent.py measures its tables on
    h = run.embed(random_windows(token_stream("wikitext2", "train", tok), args.calib_windows, args.seqlen, seed=1))

    for i, layer in enumerate(run.layers):
        t0 = time.time()
        layer.to(dev)
        attn = layer.self_attn
        Wk, Wv, Wo = (m.weight.to(dev, torch.float64) for m in (attn.k_proj, attn.v_proj, attn.o_proj))
        xs = torch.cat([layer.input_layernorm(hb).float() for _, hb in run.batches(h)])    # [W, T, D]
        mu = xs.reshape(-1, xs.shape[-1]).double().mean(0)
        Cx = stats[i]["Cx"].to(dev).double()
        mh = mu / mu.norm()
        P = torch.eye(mu.numel(), device=dev, dtype=torch.float64) - torch.outer(mh, mh)
        C_of = {"none": Cx, "center": Cx - torch.outer(mu, mu), "alpha": P @ Cx @ P}
        Sq = None if Sq_all is None else Sq_all[i].to(dev).double()
        if Sq is not None and i == 0:
            # Sq and q2 come from the same calibration pass: their pair energies must agree
            hd = d // 2
            dg = Sq.diagonal(dim1=-2, dim2=-1)
            lam_sq = 0.5 * (dg[:, :hd] + dg[:, hd:])
            lam_q2 = kv_query_energy(stats[i]["q2"].to(dev).double(), n_kv)
            print(f"  check Sq vs q2: rel diff {((lam_sq - lam_q2).norm() / lam_q2.norm()).item():.1e}")
        msg = []
        for kw, vw, mm in variants:
            C = C_of[mm]
            Ck = (P if mm == "alpha" else torch.eye(Cx.shape[0], device=dev, dtype=torch.float64)) \
                if kw == "wsvd" else C
            Cv = (P if mm == "alpha" else torch.eye(Cx.shape[0], device=dev, dtype=torch.float64)) \
                if vw == "wsvd" else C
            BK, AK, eK = k_basis(Wk, Ck, kw, stats[i]["q2"].to(dev), Sq, n_kv, d)
            BV, AV, eV = v_basis(Wv, Wo, Cv, vw, n_kv, d, n_q)
            if mm == "alpha":
                BK, BV = BK @ P, BV @ P
            # exact at full rank in every variant (center: A B (x - mu) + W mu = W x needs A B = W)
            mean_part = (lambda W: torch.outer(W @ mu, mu) / mu.dot(mu)) if mm == "alpha" else (lambda W: 0)
            WK_hat = torch.cat([AK[g] @ BK[g * d:(g + 1) * d] for g in range(n_kv)]) + mean_part(Wk)
            ek = ((WK_hat - Wk).norm() / Wk.norm()).item()
            ev = ((AV @ BV + mean_part(Wv) - Wv).norm() / Wv.norm()).item()
            BK, AK, BV, AV = BK.float(), AK.float(), BV.float(), AV.float()
            xin = xs - mu.float() if mm == "center" else xs
            cK = xin.reshape(-1, xin.shape[-1]) @ BK.T
            cV = xin @ BV.T
            DK, DV = drop_table(cK, -1), drop_table(cV, 1)
            torch.save(dict(mean_mode=mm, mu=mu.float().cpu(), kbar=(Wk @ mu).float().cpu(),
                            vbar=(Wv @ mu).float().cpu(), BK=BK.cpu(), AK=AK.cpu(), eK=eK.float().cpu(),
                            BV=BV.cpu(), AV=AV.cpu(), eV=eV.float().cpu(), DK=DK.cpu(), DK_H=DK.cpu(),
                            DV=DV.cpu(), menu=MENU_ALL),
                       os.path.join(root, f"latent_{kw}-{vw}-{mm}", f"L{i}.pt"))
            msg.append(f"{kw}-{vw}-{mm} {max(ek, ev):.0e}")
            del cK, cV
        print(f"layer {i:2d} ({time.time() - t0:4.1f}s) full-rank err: " + ", ".join(msg), flush=True)
        del xs
        h = run.run_layer(layer, h)
        layer.cpu()
        torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
