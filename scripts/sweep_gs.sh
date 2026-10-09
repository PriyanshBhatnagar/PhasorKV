#!/bin/bash
# Cross-layer (sensitivity-weighted) allocation: probe each layer, then evaluate the
# model-wide plans next to the uniform ones already in quant_*.json.
#   bash scripts/sweep_gs.sh <model> <tag> <device>
set -e
MODEL=$1; TAG=$2; DEV=$3
PY=/home/prbhatnagar/.conda/envs/PhasorKV/bin/python
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
$PY scripts/layer_sensitivity.py --model $MODEL --tag $TAG --device $DEV
$PY scripts/run_quant.py --model $MODEL --tag $TAG --device $DEV --ppl --group 4 \
    --methods starq.gs@0.85,tqA.gs@0.85,starq.gs@0.875,starq.gs@0.9,tqA.gs@0.9,starq.gs@0.925
$PY scripts/run_quant.py --model $MODEL --tag $TAG --device $DEV --tasks --task-group 1 \
    --methods starq.gs@0.85,tqA.gs@0.85,starq.gs@0.9
echo GS_DONE
