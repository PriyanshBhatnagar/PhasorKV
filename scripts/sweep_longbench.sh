#!/bin/bash
# LongBench for one model, from scratch: calibration statistics, the bases the methods
# need, layer sensitivities (.gs), then the methods. Steps already done are skipped.
#   bash scripts/sweep_longbench.sh <model> <tag> [samples per task, 0 = all]
set -e
M=$1; T=$2; N=${3:-0}
cd "$(dirname "$0")/.."
R=results/$T
METHODS=${METHODS:-bf16,diag-wo-alpha:starq.sink4@0.85,diag-wo-alpha:starq.gs.sink4@0.85,pca-pca-center:starq.sink4@0.85,full-wo-alpha:starq.sink4@0.85}
[ -f $R/stats.pt ] || python scripts/run_spectrum.py --model $M --tag $T --stats-only
[ -f $R/latent_full-wo-alpha/L0.pt ] || \
  python scripts/ablation_prep.py --model $M --tag $T --variants diag-wo-alpha,pca-pca-center,full-wo-alpha
[ -e $R/latent ] || ln -s latent_diag-wo-alpha $R/latent
[ -f $R/layer_sens.json ] || python scripts/layer_sensitivity.py --model $M --tag $T
python scripts/run_longbench.py --model $M --tag $T --limit $N --methods $METHODS
echo LONGBENCH_DONE
