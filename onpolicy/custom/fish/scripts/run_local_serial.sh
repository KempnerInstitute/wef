#!/usr/bin/env bash
# Serial local production runs. Run from onpolicy/custom/fish/ via:
#   nohup /home/${USER:-$(whoami)}/miniforge3/bin/mamba run --name mfrefactor \
#       bash scripts/run_local_serial.sh > logs/local_serial_TIMESTAMP.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

echo "=== [1/3] foraging dynamic FO=0.1 seeds 1-2 ==="
CUDA_VISIBLE_DEVICES=0 SEEDS="1 2" bash scripts/run_full.sh

echo "=== [2/3] foraging dynamic FO=0.0 seeds 1-2 ==="
CUDA_VISIBLE_DEVICES=0 SEEDS="1 2" FOOD_ORIENTATION_DRIFT=0.0 bash scripts/run_full.sh

echo "=== [3/3] foraging frac FO=0.0 seeds 1-2 ==="
CUDA_VISIBLE_DEVICES=0 SEEDS="1 2" FOOD_ORIENTATION_DRIFT=0.0 SENSING_MODEL_TYPE=frac bash scripts/run_full.sh

echo "=== All local foraging runs complete ==="
