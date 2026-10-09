#!/bin/bash
# Fake quantization of exactly what a native Blackwell FP4 rebuild would compute:
# K codes NVFP4 / NVINT3 (or NVINT2) with 16-aligned tiers, K rebuild weights NVFP4.
#   bash scripts/sweep_native.sh <model> <tag> <device>
set -e
MODEL=$1; TAG=$2; DEV=$3
PY=/home/prbhatnagar/.conda/envs/PhasorKV/bin/python
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
$PY scripts/run_quant.py --model $MODEL --tag $TAG --device $DEV --ppl --group 3 \
    --methods tqB.gs.al@0.85,tqB.gs.al.wq@0.85,tqB.gs.al.wq@0.875,tqB.gs.al.wq@0.9,tqF.gs.al.wq@0.9,tqF.gs.al.wq@0.925
$PY scripts/run_quant.py --model $MODEL --tag $TAG --device $DEV --tasks --task-group 1 \
    --methods tqB.gs.al.wq@0.85,tqF.gs.al.wq@0.9
echo NATIVE_DONE
