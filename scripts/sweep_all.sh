#!/bin/bash
# Full fake-quant sweep for one model: perplexity over every method's compression
# range, then zero-shot tasks at the main operating points.
#   bash scripts/sweep_all.sh <model> <tag> <device>
set -e
MODEL=$1; TAG=$2; DEV=$3
PY=/home/prbhatnagar/.conda/envs/PhasorKV/bin/python
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."

POST="bf16,fp8,nvfp4,mxfp4,nvfp4H,kivi4,kivi2"
LR="lr@0.5,lr@0.6,lr@0.7,lr@0.75,lr@0.8"
STARQ="starq@0.75,starq@0.8,starq@0.85,starq@0.875,starq@0.9,starq@0.925,starq@0.95"
TQA="tqA@0.8,tqA@0.85,tqA@0.875,tqA@0.9,tqA@0.925,tqA@0.95"
OTHER="rd@0.85,rd@0.9,palu@0.8,palu@0.85,palu@0.9,lrnv@0.85,lrnv@0.9"
$PY scripts/run_quant.py --model $MODEL --tag $TAG --device $DEV --ppl --group 4 \
    --methods $POST,$LR,$STARQ,$TQA,$OTHER

TASKM="bf16,fp8,nvfp4,mxfp4,nvfp4H,kivi4,kivi2,lr@0.75,starq@0.85,starq@0.9,starq@0.925,tqA@0.85,tqA@0.9,tqA@0.925,rd@0.9,palu@0.85"
$PY scripts/run_quant.py --model $MODEL --tag $TAG --device $DEV --tasks --task-group 1 --methods $TASKM
echo SWEEP_DONE
