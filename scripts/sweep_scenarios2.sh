#!/bin/bash
# Follow-up: ours at the memory the sink exemption actually uses, and alpha + exemption.
#   bash scripts/sweep_scenarios2.sh <model> <tag> <device>
set -e
M=$1; T=$2; DEV=$3
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
cd "$(dirname "$0")/.."
D=wikitext2,c4,c4_packed_bos,c4_chunks100_bos,code_humaneval
conda run --no-capture-output -n PhasorKV python scripts/run_quant.py --model $M --tag $T --device $DEV --ppl \
  --ppl-data $D --latent diag-wo-alpha --methods starq.dsink@0.85,starq@0.84,starq@0.824 --group 3
echo SCEN2_DONE
