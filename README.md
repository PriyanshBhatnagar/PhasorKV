# PhasorKV: a RoPE-exact, cross-head latent for the K cache

Working repo for the follow-up to STAR-KV. This first piece implements **idea A**:
compress keys across heads *one RoPE frequency at a time*, so the cached latent
commutes with RoPE, the query can be absorbed, and keys are never rebuilt.

## The idea

HF Llama rotates the pair `(j, j + d/2)` of every head by the same angle
`theta_j * t`. Written as a complex number `z_j = x[j] + i x[j + d/2]`, RoPE is
`z_j -> z_j * e^{i theta_j t}`: one phase per frequency, shared by every head
(a *phasor*, hence the name).

- **Structure.** Stack pair `j` of all kv heads into a complex matrix
  `M_j` (`n_kv x d_model`) and factor `M_j ~ A_j B_j`. The latent
  `c_j(t) = B_j x_t` only picks up the scalar phase `e^{i theta_j t}`, so the
  cache stores it *already rotated*. The linear maps that commute with RoPE are
  exactly these: never mix frequencies, complex-linear within one.
- **Decode.** Absorb the query once per step,
  `q'_k = conj(A[g, k]) * zq_{j(k)} * e^{i theta m}`; every logit is then a real
  dot product with the cached `[Re c_rot, Im c_rot]`. RoPE stays exact and one
  latent is shared by all heads (MLA-shaped). K-path cost per cached token is
  `2 * H_q * width` instead of STAR-KV's `2 * head_dim * width` for the rebuild.
- **Objective.** Each block minimises the query-weighted logit error
  `|| diag(sqrt(lambda)) (M - M_hat) Cx^{1/2} ||_F^2`, where `Cx = E[x x^T]` and
  `lambda` is the query energy per (kv head, frequency). Averaging over relative
  position cancels the cross-frequency terms, so this is the expected logit MSE,
  and its optimum is closed form: a weighted (complex) SVD per block, with
  ranks water-filled across blocks.

`phasorkv/factorize.py` puts every layout under that one objective so they can
be compared at equal cache size:

| structure | blocks | latent commutes with RoPE |
|---|---|---|
| `head` | one per kv head (STAR-KV's K) | no, keys rebuilt per step |
| `groupN` | N consecutive kv heads (Palu's G-LRD) | no |
| `joint` | the whole layer | no |
| `freq` | one complex block per frequency, all kv heads | **yes**, absorbed |
| `bandN` | `freq`, except the N slowest frequencies share one real block scored without RoPE | absorbed; exact up to a phase error of at most `theta_j * distance` on the slow band |

| weighting | meaning |
|---|---|
| `plain` | SVD of the weights alone (STAR-KV's initialisation) |
| `x` | input covariance only |
| `xq` | input covariance and query energy (the logit objective) |

## Repository structure

```
├── phasorkv/
│   ├── rope.py        # RoPE as complex phasors; k_proj <-> per-frequency complex rows
│   ├── factorize.py   # all layouts x weightings, weighted-Gram eigh, water-filling, decode FLOPs
│   ├── latent.py      # PhasorKey: rotated-latent cache write, query absorption, scoring
│   ├── metrics.py     # attention KL / logit / output error on real activations and positions
│   ├── engine.py      # layer-at-a-time execution (model on CPU, one layer on the GPU), data
│   ├── bases.py       # objective-B latent bases for K (per head) and V (joint, W_o-weighted)
│   ├── quant.py       # MXFP8/6/4, NVFP4, NV-int block formats, FP8, KIVI-style, Hadamard
│   ├── alloc.py       # Lagrangian rate-distortion allocation of formats (incl. drop)
│   ├── kvmethods.py   # every KV method as a per-layer patch (latent and post-RoPE)
│   ├── runner.py      # layerwise scoring of many sequences under many methods
│   └── tasks.py       # zero-shot tasks (lm-eval prompts and scoring)
├── scripts/
│   ├── run_spectrum.py  # experiment 1: per-layer fidelity vs cache size, every layout
│   ├── plot_spectrum.py # plots + summary table
│   ├── run_ppl.py       # perplexity with k_proj replaced by each layout (training-free)
│   ├── prep_latent.py   # per-layer bases + per-unit quantization distortions
│   ├── run_quant.py     # fake-quant perplexity and zero-shot for any set of methods
│   ├── run_starkv_anchor.py  # the trained STAR-KV checkpoint, bf16 and 3.2-bit
│   ├── sweep_all.sh     # the full sweep for one model
│   └── report_quant.py  # tables, breaking points, plots
└── results/<tag>/       # spectrum.json, stats.pt (calibration), ppl.json, plots
```

`PhasorKey`'s scores match "rebuild keys with `W_hat`, then HF's own
`apply_rotary_pos_emb`" to 2e-15 relative error at positions up to 30k, so
`run_ppl.py` evaluates each cache by substituting `W_hat` into `k_proj`.

## Setup

```
conda create -n PhasorKV python=3.12
conda activate PhasorKV
pip install -r requirements.txt
```

Everything runs one decoder layer at a time, so a 24 GB card shared with other
jobs is enough for 8B models (peak ~2 GB of GPU memory).

## Usage

```
# experiment 1: per-layer attention fidelity vs K cache size, all layouts
python scripts/run_spectrum.py --model lmsys/longchat-7b-v1.5-32k --tag longchat7b
python scripts/plot_spectrum.py --tag longchat7b

# perplexity with K compressed (V untouched), training-free; reuses results/<tag>/stats.pt
python scripts/run_ppl.py --model lmsys/longchat-7b-v1.5-32k --tag longchat7b
```

Calibration is 32 random 2048-token windows of WikiText-2 train. Fidelity is
measured on 4 held-out C4 validation windows (documents 2000-4000, disjoint
from the PPL slice), 256 query rows each, against all causal keys. Inputs to
each layer come from the exact model, so `run_spectrum.py` reports per-layer
error; `run_ppl.py` reports the compounded effect.

If the HF token in `~/.cache/huggingface/token` is invalid, set
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1`; the GQA model used here is
`unsloth/Meta-Llama-3.1-8B` (the same weights as `meta-llama/Llama-3.1-8B`,
ungated).

## Results (training-free, K only, V uncompressed)

All numbers are training-free: no KD and no fine-tuning. `head:plain` is
STAR-KV's *initialisation*, not trained STAR-KV.

### Attention fidelity per layer (`run_spectrum.py`)

Geo-mean over 32 layers of attention KL at 25% of the K cache kept, and K-path
decode FLOPs per cached token relative to dense attention.

| layout | longchat-7b (MHA, 32 kv heads) | Llama-3.1-8B (GQA, 8 kv heads) | K FLOPs vs dense (MHA / GQA) |
|---|---|---|---|
| head:plain | 0.369 | 0.921 | 33x / 9x |
| head:xq | 0.037 | 0.083 | 33x / 9x |
| group4:xq | 0.020 | 0.040 | 129x / 33x |
| joint:xq | 0.0047 | 0.025 | 1025x / 65x |
| **freq:xq** | 0.068 | 0.137 | **8x / 2x** |
| **band16:xq** | - | 0.089 (0.101 at 32k context) | **- / 2x** |

Full curves: `results/<tag>/summary.md` and `spectrum.png`.

### Perplexity, 40 windows x 2048 tokens (`run_ppl.py`)

Calibration is WikiText-2 train, which flatters WikiText-2 relative to C4.
Cells are WikiText-2 / C4.

longchat-7b, baseline 8.08 / 10.65:

| layout | 12.5% K | 25% K | 50% K |
|---|---|---|---|
| head:plain | 508 / 483 | 664 / 369 | 34.8 / 42.7 |
| head:xq | 11.87 / 17.64 | 8.79 / 12.78 | 8.22 / 11.27 |
| group4:xq | 8.70 / 12.54 | 8.20 / 11.33 | 8.14 / 10.85 |
| joint:xq | 8.05 / 11.00 | 8.06 / 10.77 | 8.08 / 10.64 |
| freq:xq | 16.22 / 21.72 | 10.33 / 15.28 | 8.41 / 11.83 |

Llama-3.1-8B, baseline 6.01 / 9.50:

| layout | 12.5% K | 25% K | 50% K |
|---|---|---|---|
| head:plain | 62.6 / 76.4 | 33.2 / 42.4 | 26.4 / 41.0 |
| head:xq | 15.50 / 24.90 | 8.71 / 15.43 | 6.62 / 11.17 |
| group4:xq | 8.93 / 15.42 | 6.95 / 11.89 | 6.17 / 10.03 |
| joint:xq | 7.66 / 13.17 | 6.48 / 10.97 | 6.09 / 9.79 |
| freq:xq | 38.07 / 42.83 | 12.79 / 18.59 | 7.06 / 11.99 |
| band8:xq | 31.08 / 35.37 | 10.98 / 16.98 | 6.88 / 11.72 |
| band16:xq | 20.53 / 26.60 | 8.98 / 14.85 | 6.68 / 11.32 |

### What this says

1. **The objective matters most.** Query-weighted (`xq`) factorizations beat
   weight SVD by about 10x in attention KL on both models. In perplexity, that
   is the difference between a broken model (664 PPL) and a usable one (8.8)
   at the same rank.
2. **Pure `freq` (exact RoPE, absorbed query) loses about 1.6-1.9x in KL to
   per-head at equal bytes,** with MHA and GQA alike, while using about 4x
   fewer K FLOPs. A latent that commutes with RoPE can never mix frequencies,
   because the RoPE-invariant subspaces are the per-frequency ones. Mixing
   across frequencies within a head turns out to be worth more than sharing
   across heads within a frequency. At equal *compute*, `freq` is the best
   layout by a wide margin (`spectrum.png`, right panel).
3. **Llama-3.1's slow frequencies close most of the gap.** With rope_theta
   5e5 plus llama3 scaling, 20 of 64 frequencies turn less than 0.5 rad over
   32k tokens, and that is where the K energy sits.
   - `band16` lets those dims mix across heads *and* frequencies, scored
     without RoPE. It matches per-head in KL (0.089 vs 0.083; 0.101 vs 0.096
     at 32k context) and in perplexity, at 2x instead of 9x dense K FLOPs.
   - The band must fit the context: `band24` collapses at 32k (KL 0.19).
   - longchat (theta 1e4) has no such band.
4. **The joint layout is near-lossless even training-free**: longchat stays at
   8.05 PPL with 12.5% of K. It needs per-step key reconstruction across all
   heads, so it is an accuracy ceiling, not a deployable layout.

## Quantization study (fake quant, K+V, training-free)

On top of objective B, this compares low-rank + quantized latents with strong
quantizers for the ordinary cache, measuring perplexity and six zero-shot tasks.
Each method is swept over compression to find where it breaks.

### Pipeline

1. **Bases** (`phasorkv/bases.py`, `scripts/prep_latent.py`).
   - **K:** per kv head, query-weighted, whitened by the input covariance.
   - **V:** all heads jointly, weighted by how its error reaches the output
     through `W_o`.
   - **Why these bases:** in both, latent MSE equals the logit (K) or
     attention-output (V) error, and the latent is sorted by energy.
2. **Mean-direction split.** Write `x = alpha_t * mu + P x`, with
   `P = I - mu mu^T / |mu|^2`.
   - **The problem:** the mean input gives every token nearly the same key,
     which is 95-97% of the K latent's energy. That one dimension dominates
     both the allocation and each head's first quantization block.
   - **The fix:** a per-token bf16 scalar `alpha` (shared by K and V)
     carries the mean, and the cached latent only holds `P x`.
   - **Why not subtract the mean:** attention-sink tokens (position 0 and
     some punctuation tokens) have `alpha ~ 0`. Subtracting the mean leaves
     them as `-mu`, outside the fitted subspace, and their keys come back
     ~12x off.
3. **Allocation** (`phasorkv/alloc.py`). Per layer, per 32-dim unit: a
   Lagrangian rate-distortion choice of format, including `drop`, on
   distortions measured on calibration latents.
4. **Methods** (`phasorkv/kvmethods.py`). All latent methods share steps 1-2
   and differ only in how rank and bits are chosen and how the latent is
   quantized:

| method | rank / bits | quantizer |
|---|---|---|
| `lr` | rank only | bf16 latent |
| `starq` | rank by energy | STAR-KV's: leading 20% int4, rest int3, one scale per token per tier, Hadamard per tier |
| `tqA` | rank by energy | K: NVINT4 / NVINT3 tiers (block-16 E4M3 scales, block-scaled-MMA friendly), V: STAR-KV's |
| `rd` | MSE rate-distortion over {MXFP8, MXFP6, NVFP4, drop} | 32-wide block Hadamard on K, V quantized along tokens (the first proposal) |
| `palu` | rank by energy | Hadamard over the kept latent + NVFP4 per token |
| `lrnv` | rank by energy | NVFP4 per token, no rotation |

Baselines quantize the ordinary post-RoPE cache:
- `fp8`: E4M3 per token.
- `nvfp4`, `mxfp4`: NVFP4 and MXFP4.
- `nvfp4H`: Hadamard-128 per head + NVFP4.
- `kivi4`, `kivi2`: asymmetric INT, group 32; K per channel, V per token.

`run_starkv_anchor.py` evaluates the trained STAR-KV checkpoint (64%) as is
and with its own 3.2-bit quantization.

### Conventions

- **What counts:** compression covers K+V over all 32 layers, scales and the
  per-token alpha included.
- **Perplexity:** 32 x 2048 windows each of WikiText-2 test and C4 validation.
- **Tasks:** first 1000 questions of piqa, arc_easy, arc_challenge,
  hellaswag and winogrande, and all 500 of openbookqa.
  - Prompts and scoring follow lm-eval-harness.
  - Metric: acc for piqa, arc_e and winogrande; acc_norm for the rest.
- **Fake quantization:** every position is quantized, prefill included, and
  no recent-token window is kept in full precision.
- **Reproduce:** `bash scripts/sweep_all.sh <model> <tag> <device>`, then
  `python scripts/report_quant.py`.

### Results

#### longchat-7b (32 kv heads)

Baseline: bf16 PPL 8.295, 6-task average 59.88.

| method | compression | WikiText-2 (Δ) | task avg (Δ) |
|---|---|---|---|
| fp8 | 49% | 8.31 (+0.1%) | 60.00 (+0.1) |
| kivi4 | 69% | 8.31 (+0.2%) | 59.82 (-0.1) |
| nvfp4 | 72% | 8.42 (+1.5%) | 59.58 (-0.3) |
| mxfp4 | 73% | 8.99 (+8.4%) | 58.95 (-0.9) |
| kivi2 | 81% | 8.90 (+7.3%) | 56.02 (-3.9) |
| STAR-KV trained, bf16 latent | 64% | 9.02 (+8.7%) | 54.68 (-5.2) |
| STAR-KV trained + its 3.2-bit quant | 82% | 9.16 (+10.5%) | 54.43 (-5.5) |
| **B + STAR-KV quantizer (`starq`)** | **85%** | **8.37 (+0.9%)** | **58.25 (-1.6)** |
| `starq` | 90% | 9.06 (+9.2%) | 57.25 (-2.6) |
| `tqA` | 85% / 90% | 8.40 (+1.3%) / 8.97 (+8.1%) | 58.18 / 56.72 |
| `palu` | 80% / 85% | 8.30 (+0.0%) / 8.62 (+4.0%) | - / 57.75 |
| `rd` (first proposal) | 90% | 10.12 (+22%) | 54.92 (-5.0) |
| `lr` (no quantization) | 75% | 11.19 (+35%) | 50.73 (-9.2) |

#### Llama-3.1-8B (8 kv heads)

Baseline: bf16 PPL 6.424, 6-task average 67.40.

| method | compression | WikiText-2 (Δ) | task avg (Δ) |
|---|---|---|---|
| kivi4 | 69% | 6.45 (+0.5%) | 67.48 (+0.1) |
| nvfp4 | 72% | 6.58 (+2.5%) | 66.75 (-0.7) |
| nvfp4H | 72% | 6.52 (+1.5%) | 67.27 (-0.1) |
| kivi2 | 81% | 7.89 (+23%) | 60.78 (-6.6) |
| `starq` | 80% | 6.65 (+3.6%) | - |
| `starq` / `tqA` | 85% | 7.54 / 7.51 (+17%) | 62.48 / 62.40 (-4.9) |
| `starq` / `tqA` | 90% | 10.75 / 10.58 (+66%) | 51.75 / 52.90 |
| `rd` | 90% | 14.09 | 47.48 |

#### Breaking points

Training-free, max K+V compression within +x of bf16 WikiText-2 PPL;
`<` means the method's lowest sweep point already exceeds it. Full tables:
`results/<tag>/quant_summary.md`; plots: `quant.png`.

| method | longchat +1% | +3% | +5% | +10% | Llama +5% | +10% |
|---|---|---|---|---|---|---|
| `starq` | <79% | 86.6% | 87.9% | 90.2% | 80.6% | 82.4% |
| `tqA` | 80.3% | 86.5% | 88.0% | 90.5% | 80.6% | 82.5% |
| `palu` | 81.2% | 83.8% | 85.5% | 87.4% | <80% | <80% |
| `rd` | <85% | <85% | <85% | 86.2% | <85% | <85% |
| `lr` | <50% | <50% | 50.2% | 58.4% | <50% | <50% |

#### Findings

1. **B plus the mean-direction split is what makes low-rank KV work without
   training.** On longchat it beats *trained* STAR-KV on both perplexity and
   tasks, at every compression we tested. At 85% it is within 1% of bf16
   perplexity.
2. **Without the split, the same quantizer at 90% is far worse:** 9.9 vs 8.1
   PPL on a 2-window check.
3. **The quantizer matters much less than rank.**
   - At a fixed rank, any format with 4 or more bits is lossless.
   - STAR-KV's tiered INT quantizer (Hadamard per tier, one scale per token)
     is already about as good as fine-grained NV/MX block formats at equal
     bits.
   - `tqA` keeps K in NVINT tiers, which map to block-scaled matmuls, at no
     accuracy cost.
4. **The first design (`rd`) loses, for three reasons.**
   - **Latent MSE misprices quantization noise against truncation.** Noise
     averages over attended tokens; truncation does not. So MSE-optimal
     allocation buys precision when it should buy rank.
   - **A 32-wide block Hadamard hurts FP4 on a sorted latent.** One Hadamard
     per tier helps.
   - **Quantizing V along tokens fails at 3 bits.** That is the axis
     block-scaled P·V needs; at INT3 it gives 11.6 PPL vs 8.5 along the
     latent. Per-token-per-tier V scales (foldable into P, as STAR-KV's
     kernel does) avoid it.
5. **GQA breaks much earlier.** Llama-3.1 has 8 kv heads, so there is little
   redundancy to remove: the +5% breaking point is 81%, against 88% on
   longchat. Below ~75%, plain NVFP4 or KIVI-4 on the ordinary cache is the
   better choice there.

### Cross-layer allocation (`.gs` methods)

The uniform runs above give every layer the same bit budget. In `.gs` methods,
budget moves across layers.

How the plan is built:

1. **One exchange rate per layer and side.** Objective B already puts each
   layer's latent in its own error units: dropping a unit costs its energy.
   For each layer and side (K, V), `scripts/layer_sensitivity.py` compresses
   only that side of that layer and measures the KL divergence of the final
   next-token distribution from the exact model's.
   - The result, `s = KL / dropped energy`, is that layer's rate from
     attention error to final loss.
   - That's 64 probes on 8 calibration windows, ~7-10 min per model.
   - KL is needed because a single layer's change in NLL (about ±0.001) is
     below the noise.
2. **One global ranking.** `LatentPatch.plan_global` ranks every 32-dim unit
   of every layer by `s x energy` per bit. It keeps the best units until the
   model-wide budget is spent.
3. **Where the budget goes.** On Llama-3.1, layers 3-5 stay at ~80-82%
   compression and layers 23-27 go to ~90%. On longchat the plan stays nearly
   flat (84-89%).

Results:

| model | method | 80% | 85% | 90% |
|---|---|---|---|---|
| longchat (bf16 8.30 / 59.88) | `starq` uniform | 8.43 | 8.37 / 58.25 | 9.06 / 57.25 |
| | `starq.gs` | - | 8.42 / 58.75 | 8.88 / 57.32 |
| | `tqA.gs` | - | 8.37 / 58.78 | 8.77 / - |
| Llama-3.1 (bf16 6.42 / 67.40) | `starq` uniform | 6.65 | 7.54 / 62.48 | 10.75 / 51.75 |
| | `starq.gs` | 6.66 | 7.11 / 64.23 | 9.13 / 55.33 |
| | `tqA.gs` | 6.62 | 7.04 / 63.83 | 8.88 / - |

Cells are WikiText-2 PPL / 6-task average.

Breaking points move out:
- **longchat:** +1% from ~80% to 85.1%, +5% from 88.0% to 89.2% (`tqA.gs`).
- **Llama:** +5% from 80.6% to 82.5%, +10% from 82.4% to 85.1%.

Reproduce: `bash scripts/sweep_gs.sh <model> <tag> <device>`.

### Lower precision (`t42`, `t22`) and token pruning (`.pNN.wNN`)

Two configurations tested at full scale:
- **`t42`:** the top tier at 4 bits and the rest at 2 bits (K NVINT4/NVINT2, V per-token int4/int2).
- **`t22`:** 2 bits everywhere. A 2-window check gives 11.25 PPL at 90% vs 8.01 for `t42`, so it was not run at full scale.

Results, WikiText-2 PPL (task average where measured):

| model | method | 85% | 87.5% | 90% | 92.5% | 95% |
|---|---|---|---|---|---|---|
| longchat | `starq.gs` / `tqA.gs` (4/3 tiers) | 8.42 / **8.37** (58.8) | **8.54** | 8.88 / **8.77** (57.3) | 9.79 | - |
| longchat | `t42.gs` (4/2 tiers) | 8.93 (57.1) | 8.68 | 8.91 (56.7) | **9.32** | **10.72** |
| Llama-3.1 | `starq.gs` / `tqA.gs` | 7.11 / **7.04** (64.2) | 7.74 | 9.13 / 8.88 (55.3) | 14.70 | - |
| Llama-3.1 | `t42.gs` | 7.71 (62.8) | **7.60** | **8.01** (60.1) | **9.81** | 19.83 |

Cheaper bits only pay once rank is the bottleneck:
- **At 85%,** 4/3 tiers already keep about 70% of the rank, so 2-bit noise costs more than the extra rank buys.
- **From 90% (Llama) and 92.5% (longchat),** 4/2 tiers win clearly. On Llama at 90%, PPL drops from 8.88 to 8.01 and the task average rises from 55.3 to 60.1.

Choose the tier configuration by target compression.

**Token pruning.** For an ordinary token (`0.5 < alpha < 1.5`; sinks are never pruned), drop the whole latent and keep only `alpha`. Its key and value become `alpha * W mu`, and the lowest-energy tokens go first. It loses to cutting rank at the same compression, even when a recent window keeps nearby tokens unpruned (2-window check, bf16 7.49):

| ~88% | PPL | ~90-91% | PPL |
|---|---|---|---|
| rank only (`starq.gs@0.875`) | **7.68** | rank only (`starq.gs@0.9`) | **8.08** |
| 85% + prune 20%, window 128 | 8.38 | 85% + prune 40%, window 128 | 10.24 |
| 85% + prune 20%, window 512 | 7.90 | | |

Without a window it is far worse (14.06 at 20%). A token's latent energy does not predict whether future queries will need it. Dropping one token's entire residual is a large, coherent error. Cutting rank spreads small errors across all tokens, in the directions queries weight least.

### Scale granularity: how much per-token metadata the latent needs

With the mean split and objective B, the latent has no outliers within a token, so the rotation choice barely matters. Per-group 16 vs per-tier vs none differ by about 0.5-1% PPL. That raises the question of how many scales are needed. Results at 85% with layer-adaptive allocation (`*.gs`), WikiText-2 / C4 PPL:

| scales | stored per token per layer (longchat) | longchat (8.30 / 10.76) | Llama-3.1 (6.42 / 9.55) |
|---|---|---|---|
| static per channel (`st43`, folded into the factors) | 0 | 20.82 / 17.55 | 8.80 / 18.13 |
| one per token for K + one for V (`nt43g`) | 32 bits | 8.73 / 10.99 | 7.59 / 11.27 |
| two per head (K) + two (V), STAR-KV tiers (`starq`) | 1,056 bits | 8.42 / 10.93 | 7.11 / 11.45 |
| NV block-16 for K, STAR-KV V (`tqA`) | 0.5 bit per K value + 32 bits | **8.37 / 10.92** | **7.04 / 11.29** |

What the scales are doing:
- **Static scales fail.** The latent varies a lot in magnitude from token to token, even though each token is internally well conditioned.
- **One scale per token keeps C4 intact but costs 4-8% on WikiText-2.** WikiText's formatting tokens make a few heads large, and one shared scale crushes the others.
- **Widening the clipping range does not help.** Multiplying the Gaussian-optimal step by 1.3, 1.6 or 2.0 gives 7.63, 7.81 and 9.82, against 7.59 (2-window check).

How cheap each scale is to apply:
- **A per-token scale** passes through the rebuild and RoPE unchanged. It multiplies the K score once per token per head and folds into P for V. Store `alpha / s` so the mean term shares the same factor.
- **NV block scales** are applied inside the matmul by block-scaled tensor cores on Blackwell.

Which to use:
- **Blackwell:** `tqA`, which gives the best accuracy with the scales handled in hardware.
- **Ada or a custom datapath without block scaling:** `nt43g`, with one scale per token and the per-channel step folded into the factors.

### Native Blackwell FP4 path (go/no-go)

This fake-quantizes exactly what a native FP4 block-scaled matmul would compute for the K rebuild `k = A . codes`:
- **K codes:** NVFP4 (top tier) and NVINT3 or NVINT2 (rest), with E4M3 scales per 16 along the latent, which is the matmul's reduction axis. INT3 {0, ±1, ±2, ±3} and INT2 {±0.5, ±1.5} are exact FP4 (E2M1) values, so the codes run natively once widened to 4 bits.
- **Tier boundaries:** multiples of 16 (`.al`), so no scale block mixes two tiers.
- **Rebuild weights `A`:** NVFP4, scales per 16 along the same axis (`.wq`).
- **Everything else:** RoPE, q.k, the mean term and all other weights stay in bf16. V keeps `tqA`'s per-token tiers.

Layer-adaptive budgets throughout. Cells are WikiText-2 PPL / 6-task average.

| model | compression | best software format | native: codes only | **native: codes + NVFP4 weights** |
|---|---|---|---|---|
| longchat (8.30 / 59.88) | 85% (4/3 tiers) | 8.37 / 58.78 (`tqA`) | 8.39 | **8.47 / 58.10** |
| | 90% (4/2 tiers) | 8.91 / 56.73 (`t42`) | - | **8.97 / 55.53** |
| | 92.5% (4/2) | 9.32 (`t42`) | - | **9.40** |
| Llama-3.1 (6.42 / 67.40) | 85% (4/3) | 7.04 / 63.83 (`tqA`) | 7.05 | **7.10 / 63.70** |
| | 90% (4/2) | 8.01 / 60.10 (`t42`) | - | **8.07 / 59.65** |
| | 92.5% (4/2) | 9.81 (`t42`) | - | **9.98** |

What the native constraints cost:
- **The native code layout alone:** +0.1-0.3% PPL.
- **NVFP4 rebuild weights:** another ~0.7-1%.
- **In total:** about 1-2% PPL and 0.1-1.2 task points against the best software formats. That is within the go/no-go threshold.

Reproduce: `bash scripts/sweep_native.sh <model> <tag> <device>`.
