#!/bin/bash
# Ablation: which part of the method buys the accuracy (bases from ablation_prep.py).
# Same quantizer (STAR-KV tiers) and per-layer budget for every basis, so the basis
# weighting and the mean handling are the only differences.
#   bash scripts/sweep_ablation.sh <model> <tag> <device>
set -e
M=$1; T=$2; DEV=$3
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
cd "$(dirname "$0")/.."
VARIANTS="diag-wo-alpha wsvd-wsvd-none pca-pca-none diag-wo-none diag-wo-center wsvd-wsvd-alpha pca-pca-alpha
          full-wo-alpha diag-pca-alpha pca-wo-alpha full-wo-center pca-pca-center"
run() { conda run --no-capture-output -n PhasorKV python scripts/run_quant.py --model $M --tag $T --device $DEV "$@"; }
[ "$PHASE" = tasks ] || for v in $VARIANTS; do
  m="starq@0.85,lr@0.75"
  [[ $v == *-center ]] && m="$m,starq.sink4@0.85"
  run --latent $v --methods $m --ppl --group 3
done
# task sweeps hold ~27 GB of RAM: one model at a time (LOCK shared by both models' sweeps)
# tasks skip what perplexity already shows broken: fixed-mean centering without the
# sink exemption (PPL in the hundreds) and the second weight-SVD row
for v in ${TASK_VARIANTS:-diag-wo-alpha wsvd-wsvd-none pca-pca-none diag-wo-none diag-wo-center pca-pca-alpha
          full-wo-alpha diag-pca-alpha pca-wo-alpha full-wo-center pca-pca-center}; do
  m="starq@0.85"
  [[ $v == *-center ]] && m="starq.sink4@0.85"
  if [ -n "$LOCK" ]; then
    flock "$LOCK" conda run --no-capture-output -n PhasorKV python scripts/run_quant.py \
      --model $M --tag $T --device $DEV --latent $v --methods $m --tasks
  else
    run --latent $v --methods $m --tasks
  fi
done
echo ABLATION_DONE
