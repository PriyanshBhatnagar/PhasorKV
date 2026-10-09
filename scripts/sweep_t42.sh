#!/bin/bash
# 4-bit top tier + 2-bit rest (t42): uniform and layer-adaptive, perplexity then tasks.
#   bash scripts/sweep_t42.sh <model> <tag> <device>
set -e
MODEL=$1; TAG=$2; DEV=$3
PY=/home/prbhatnagar/.conda/envs/PhasorKV/bin/python
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
$PY scripts/run_quant.py --model $MODEL --tag $TAG --device $DEV --ppl --group 4 \
    --methods t42@0.85,t42@0.9,t42@0.925,t42.gs@0.85,t42.gs@0.875,t42.gs@0.9,t42.gs@0.925,t42.gs@0.95
$PY scripts/run_quant.py --model $MODEL --tag $TAG --device $DEV --tasks --task-group 1 \
    --methods t42.gs@0.85,t42.gs@0.9
echo T42_DONE
