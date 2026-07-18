#!/usr/bin/env bash
# Local overnight: 2 parallel tracks, 1 seed each per combo.
#   Track 1 (GPU 0): all dynamic combos — 4 foraging + 4 homing, interleaved
#   Track 2 (GPU 1): all frac combos   — 4 foraging + 4 homing, interleaved
# Interleaving keeps both GPUs busy until the end (homing ~3× faster than foraging).
#
# Usage (from onpolicy/custom/fish/):
#   bash scripts/submit_overnight_local.sh
#   nohup bash scripts/submit_overnight_local.sh > logs/overnight_driver.log 2>&1 &

set -uo pipefail

MAMBA=/home/${USER:-$(whoami)}/miniforge3/bin/mamba
CUDNN_LIB=/home/${USER:-$(whoami)}/miniforge3/envs/mfrefactor/lib/python3.10/site-packages/nvidia/cudnn/lib
RESULTS_PARENT_DIR="${RESULTS_PARENT_DIR:-./}"
FISH_DIR="$(cd "$(dirname "$0")/.." && pwd)"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${FISH_DIR}/logs/local_overnight_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=== Local overnight starting: ${TIMESTAMP} ==="
echo "Working dir : ${FISH_DIR}"
echo "Logs        : ${LOG_DIR}/"

# gpu  script          group  sensing  fo  kp
run_job() {
    local gpu="$1" script="$2" group="$3" sensing="$4" fo="$5" kp="$6"
    local log="${LOG_DIR}/${group}.log"
    local tag; [[ "$script" == *homing* ]] && tag="HOM" || tag="FOR"
    echo "[${tag}] $(date +%H:%M:%S) Starting ${group}  (GPU ${gpu})"
    (
        cd "$FISH_DIR"
        export CUDA_VISIBLE_DEVICES="$gpu"
        export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
        export RESULTS_PARENT_DIR="$RESULTS_PARENT_DIR"
        export GROUP="$group" SEEDS=1
        export SENSING_MODEL_TYPE="$sensing"
        export FOOD_ORIENTATION_DRIFT="$fo"
        export KNOLLEN_PROCESSING="$kp"
        "$MAMBA" run --name mfrefactor bash "scripts/$script"
    ) >> "$log" 2>&1
    echo "[${tag}] $(date +%H:%M:%S) Done    ${group}  (GPU ${gpu})"
}

dynamic_track() {
    run_job 0 run_full.sh   20260611_Dyn_F00_Kb_For dynamic 0.0 binarize
    run_job 0 run_homing.sh 20260611_Dyn_F00_Kb_Hom dynamic 0.0 binarize
    run_job 0 run_full.sh   20260611_Dyn_F00_Kl_For dynamic 0.0 log
    run_job 0 run_homing.sh 20260611_Dyn_F00_Kl_Hom dynamic 0.0 log
    run_job 0 run_full.sh   20260611_Dyn_F01_Kb_For dynamic 0.1 binarize
    run_job 0 run_homing.sh 20260611_Dyn_F01_Kb_Hom dynamic 0.1 binarize
    run_job 0 run_full.sh   20260611_Dyn_F01_Kl_For dynamic 0.1 log
    run_job 0 run_homing.sh 20260611_Dyn_F01_Kl_Hom dynamic 0.1 log
    echo "[GPU 0] All dynamic combos complete."
}

frac_track() {
    run_job 1 run_full.sh   20260611_Frc_F00_Kb_For frac 0.0 binarize
    run_job 1 run_homing.sh 20260611_Frc_F00_Kb_Hom frac 0.0 binarize
    run_job 1 run_full.sh   20260611_Frc_F00_Kl_For frac 0.0 log
    run_job 1 run_homing.sh 20260611_Frc_F00_Kl_Hom frac 0.0 log
    run_job 1 run_full.sh   20260611_Frc_F01_Kb_For frac 0.1 binarize
    run_job 1 run_homing.sh 20260611_Frc_F01_Kb_Hom frac 0.1 binarize
    run_job 1 run_full.sh   20260611_Frc_F01_Kl_For frac 0.1 log
    run_job 1 run_homing.sh 20260611_Frc_F01_Kl_Hom frac 0.1 log
    echo "[GPU 1] All frac combos complete."
}

dynamic_track &
DYN_PID=$!
echo "Dynamic track PID: ${DYN_PID}  (GPU 0)"

frac_track &
FRC_PID=$!
echo "Frac track PID:    ${FRC_PID}  (GPU 1)"

echo ""
echo "Monitor with:"
echo "  tail -f ${LOG_DIR}/<group>.log"
echo "  ps -p ${DYN_PID},${FRC_PID}"

wait $DYN_PID; echo "=== Dynamic track done ==="
wait $FRC_PID; echo "=== Frac track done ==="
echo "=== All local overnight runs complete ==="
