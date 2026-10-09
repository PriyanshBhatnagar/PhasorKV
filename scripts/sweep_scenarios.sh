#!/bin/bash
# Mean handling on text with sinks after the first 4 tokens (alpha_scenarios.py):
# ours (alpha) vs fixed-mean + first-4 exemption (KVTC / AATC) vs + every detected sink.
#   bash scripts/sweep_scenarios.sh <model> <tag> <device>
set -e
M=$1; T=$2; DEV=$3
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
cd "$(dirname "$0")/.."
D=wikitext2,c4,c4_packed_bos,c4_chunks100_bos,code_humaneval
run() { conda run --no-capture-output -n PhasorKV python scripts/run_quant.py --model $M --tag $T --device $DEV --ppl --ppl-data $D "$@"; }
run --latent diag-wo-alpha --methods bf16,starq@0.85 --group 2
run --latent diag-wo-center --methods starq.sink4@0.85,starq.sink4.dsink@0.85 --group 2
run --latent pca-pca-center --methods starq.sink4@0.85 --group 1
echo SCEN_DONE
