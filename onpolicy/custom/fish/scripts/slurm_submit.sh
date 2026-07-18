#!/usr/bin/env bash
# Single SLURM submission wrapper. One job per seed for train presets;
# one job (processing all dirs) for the eval preset.
#
# Usage (from onpolicy/custom/fish/):
#   SEEDS="1 2 3" bash scripts/slurm_submit.sh full
#   bash scripts/slurm_submit.sh smoke
#   RUN_DIR=/path/to/run bash scripts/slurm_submit.sh eval
#   RUN_DIRS="/r1 /r2" bash scripts/slurm_submit.sh eval
#   SEEDS="1 2" bash scripts/slurm_submit.sh homing
#
# Presets: smoke | short | full | eval | homing

set -uo pipefail

# Always pull latest code before submitting
git -C "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" pull

PRESET="${1:-full}"
CONDA_ENV="${CONDA_ENV:-mfrefactor}"
CLUSTER_STORAGE_DIR="/n/holylfs06/LABS/krajan_lab/Lab/$(whoami)/marl_fish_storage"
RESULTS_PARENT_DIR="${RESULTS_PARENT_DIR:-${CLUSTER_STORAGE_DIR}/}"
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
SLURM_PARTITION="${SLURM_PARTITION:-kempner}"

case "$PRESET" in
  smoke)
    SCRIPT="scripts/run_smoke.sh"
    NCPUS=4;  TIMELIMIT="0-01:00:00"; MEM="32G";  REQUEUE=0
    SEEDS="${SEEDS:-1}"
    ;;
  short)
    SCRIPT="scripts/run_short.sh"
    NCPUS=8;  TIMELIMIT="0-08:00:00"; MEM="64G";  REQUEUE=0
    SEEDS="${SEEDS:-1}"
    ;;
  full)
    SCRIPT="scripts/run_full.sh"
    NCPUS=16; TIMELIMIT="1-12:00:00"; MEM="256G"; REQUEUE=0
    SEEDS="${SEEDS:-1 2}"
    ;;
  eval)
    SCRIPT="scripts/run_eval.sh"
    NCPUS=8;  TIMELIMIT="0-12:00:00"; MEM="64G";  REQUEUE=1
    if [[ -z "$RUN_DIR" && -z "$RUN_DIRS" ]]; then
        echo "ERROR: eval preset requires RUN_DIR=/path or RUN_DIRS=\"/r1 /r2\"."
        exit 1
    fi
    ;;
  homing)
    SCRIPT="scripts/run_homing.sh"
    NCPUS=24; TIMELIMIT="0-12:00:00"; MEM="128G"; REQUEUE=1
    SEEDS="${SEEDS:-1 2}"
    ;;
  *)
    echo "ERROR: Unknown preset '${PRESET}'. Valid: smoke | short | full | eval | homing"
    exit 1
    ;;
esac

JOB_NAME="${PRESET}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./logs/${JOB_NAME}"
mkdir -p "$LOG_DIR"

echo "Preset    : ${PRESET}"
echo "Script    : ${SCRIPT}"
echo "Resources : ${NCPUS} CPUs  ${TIMELIMIT}  ${MEM} RAM"
echo "Logs      : ${LOG_DIR}/"

make_and_submit() {
    local seed="${1:-}"
    local requeue_line=""
    [[ "$REQUEUE" -eq 1 ]] && requeue_line="#SBATCH --requeue"

    local tmp
    tmp=$(mktemp "submit_${PRESET}_XXXXXX.sh")

    cat > "$tmp" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --cpus-per-task=${NCPUS}
#SBATCH --time=${TIMELIMIT}
#SBATCH --mem=${MEM}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=kempner_krajan_lab
#SBATCH --gpus-per-node=1
#SBATCH --constraint=cc8.0
#SBATCH --exclude=holygpu8a[19102,19604,19605,19606]
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mail-user=satpreet_singh@hms.harvard.edu
#SBATCH --mail-type=END,FAIL
${requeue_line}

export LD_LIBRARY_PATH=/n/home02/${USER:-$(whoami)}/miniforge3/envs/${CONDA_ENV}/lib/python3.10/site-packages/nvidia/cudnn/lib:\${LD_LIBRARY_PATH:-}
export RESULTS_PARENT_DIR="${RESULTS_PARENT_DIR}"
export PYTHONUNBUFFERED=1
EOF

    [[ -n "$seed"                   ]] && echo "export SEEDS=\"${seed}\"" >> "$tmp"
    [[ -n "$RUN_DIR"                ]] && echo "export RUN_DIR=\"${RUN_DIR}\"" >> "$tmp"
    [[ -n "$RUN_DIRS"               ]] && echo "export RUN_DIRS=\"${RUN_DIRS}\"" >> "$tmp"
    [[ -n "$FOOD_ORIENTATION_DRIFT" ]] && echo "export FOOD_ORIENTATION_DRIFT=\"${FOOD_ORIENTATION_DRIFT}\"" >> "$tmp"
    [[ -n "$SENSING_MODEL_TYPE"     ]] && echo "export SENSING_MODEL_TYPE=\"${SENSING_MODEL_TYPE}\"" >> "$tmp"
    [[ -n "$KNOLLEN_MODE"           ]] && echo "export KNOLLEN_MODE=\"${KNOLLEN_MODE}\"" >> "$tmp"
    [[ -n "$KNOLLEN_PROCESSING"     ]] && echo "export KNOLLEN_PROCESSING=\"${KNOLLEN_PROCESSING}\"" >> "$tmp"
    [[ -n "$ELECTRIC_BACKEND"       ]] && echo "export ELECTRIC_BACKEND=\"${ELECTRIC_BACKEND}\"" >> "$tmp"
    [[ -n "$EAT_COOLDOWN_RATE"      ]] && echo "export EAT_COOLDOWN_RATE=\"${EAT_COOLDOWN_RATE}\"" >> "$tmp"
    [[ -n "$GROUP"                  ]] && echo "export GROUP=\"${GROUP}\"" >> "$tmp"
    [[ -n "$HOMING2"                ]] && echo "export HOMING2=\"${HOMING2}\"" >> "$tmp"
    [[ -n "$NUM_TRAIN_STEPS"        ]] && echo "export NUM_TRAIN_STEPS=\"${NUM_TRAIN_STEPS}\"" >> "$tmp"

    cat >> "$tmp" << BODY
start=\$SECONDS
source /n/home02/${USER:-$(whoami)}/miniforge3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
bash ${SCRIPT}
elapsed=\$((SECONDS - start))
printf -v t '%02d:%02d:%02d' \$((elapsed/3600)) \$(((elapsed%3600)/60)) \$((elapsed%60))
echo "Elapsed: \${t}"
echo "Done."
BODY

    local job_id
    job_id=$(sbatch --parsable "$tmp")
    echo "  Submitted job ${job_id}${seed:+ (SEEDS=${seed})}"
    echo "  Log: ${LOG_DIR}/${JOB_NAME}_${job_id}.out"
    rm "$tmp"
}

if [[ "$PRESET" == "eval" ]]; then
    make_and_submit ""
else
    for SEED in $SEEDS; do
        make_and_submit "$SEED"
    done
fi
