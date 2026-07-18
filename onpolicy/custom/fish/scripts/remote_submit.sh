#!/usr/bin/env bash
# Submit a job to FASRC SLURM via SSH tunnel.
# Run from local machine after: bash scripts/fasrc_connect_shs.sh
#
# Usage:
#   bash scripts/fasrc_submit.sh full
#   SEEDS="3 4 5" bash scripts/fasrc_submit.sh full
#   bash scripts/fasrc_submit.sh smoke
#   RUN_DIR=/n/.../run bash scripts/fasrc_submit.sh eval
#   RUN_DIRS="/n/.../r1 /n/.../r2" bash scripts/fasrc_submit.sh eval
#
# Presets: smoke | short | full | eval | homing

set -uo pipefail

PRESET="${1:-full}"
SEEDS="${SEEDS:-}"
RUN_DIR="${RUN_DIR:-}"
RUN_DIRS="${RUN_DIRS:-}"
FOOD_ORIENTATION_DRIFT="${FOOD_ORIENTATION_DRIFT:-}"
SENSING_MODEL_TYPE="${SENSING_MODEL_TYPE:-}"
KNOLLEN_MODE="${KNOLLEN_MODE:-}"
KNOLLEN_PROCESSING="${KNOLLEN_PROCESSING:-}"
ELECTRIC_BACKEND="${ELECTRIC_BACKEND:-}"
EAT_COOLDOWN_RATE="${EAT_COOLDOWN_RATE:-}"
GROUP="${GROUP:-}"
HOMING2="${HOMING2:-}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
FISH_DIR="${FISH_DIR:-~/mfrefactor/onpolicy/custom/fish}"

if ! ssh -O check fasrc 2>/dev/null; then
    echo "ERROR: No active FASRC tunnel. Run: bash scripts/fasrc_connect_shs.sh"
    exit 1
fi

echo "=== Syncing repo on FASRC ==="
ssh fasrc "cd ~/mfrefactor && git pull --ff-only"

echo "=== Submitting '${PRESET}' job(s) (SEEDS='${SEEDS:-default}') ==="

ssh fasrc bash --login -c "'
    cd ${FISH_DIR}
    SEEDS=\"${SEEDS}\" RUN_DIR=\"${RUN_DIR}\" RUN_DIRS=\"${RUN_DIRS}\" \
        FOOD_ORIENTATION_DRIFT=\"${FOOD_ORIENTATION_DRIFT}\" \
        SENSING_MODEL_TYPE=\"${SENSING_MODEL_TYPE}\" \
        KNOLLEN_MODE=\"${KNOLLEN_MODE}\" \
        KNOLLEN_PROCESSING=\"${KNOLLEN_PROCESSING}\" \
        ELECTRIC_BACKEND=\"${ELECTRIC_BACKEND}\" \
        EAT_COOLDOWN_RATE=\"${EAT_COOLDOWN_RATE}\" \
        GROUP=\"${GROUP}\" \
        HOMING2=\"${HOMING2}\" \
        NUM_TRAIN_STEPS=\"${NUM_TRAIN_STEPS}\" \
        SLURM_PARTITION=\"${SLURM_PARTITION}\" \
        bash scripts/slurm_submit.sh ${PRESET}
'"

echo ""
echo "Monitor with:"
echo "  ssh fasrc \"squeue -u ${USER:-$(whoami)}\""
echo "  ssh fasrc \"squeue -u ${USER:-$(whoami)} --format='%.18i %.9P %.30j %.8T %.10M %.6D %R'\""
