#!/usr/bin/env bash
# Serial local homing runs (GPU 1). Run from onpolicy/custom/fish/ via:
#   nohup /home/${USER:-$(whoami)}/miniforge3/bin/mamba run --name mfrefactor \
#       bash scripts/run_local_homing.sh > logs/local_homing_TIMESTAMP.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

echo "=== [1/2] homing dynamic seeds 1-10 ==="
CUDA_VISIBLE_DEVICES=1 SEEDS="1 2 3 4 5 6 7 8 9 10" bash scripts/run_homing.sh

echo "=== [2/2] homing frac seeds 1-10 ==="
CUDA_VISIBLE_DEVICES=1 SEEDS="1 2 3 4 5 6 7 8 9 10" SENSING_MODEL_TYPE=frac bash scripts/run_homing.sh

echo "=== [3/3] foraging dynamic FO=0.0 eat-cooldown=10 seeds 1-2 ==="
CUDA_VISIBLE_DEVICES=1 SEEDS="1 2" FOOD_ORIENTATION_DRIFT=0.0 EAT_COOLDOWN_RATE=0.1 bash scripts/run_full.sh

echo "=== All local homing runs complete ==="
