#!/usr/bin/env bash
# Submit overnight jobs to FASRC SLURM: 16 combos × 5 seeds = 80 jobs.
# Staggered: all combos get seed 1 before any combo gets seed 2.
#
# Combos (2×2×2×2):
#   sensing:    dynamic | frac
#   FO:         0.0 | 0.1   (food_orientation_drift; no-op for homing)
#   knollen:    binarize (Kb) | log (Kl)
#   task:       foraging (For, run_full.sh) | homing (Hom, run_homing.sh)
#
# Usage:
#   bash scripts/submit_overnight_cluster.sh
#   SEEDS="1 2 3" bash scripts/submit_overnight_cluster.sh

set -uo pipefail

SEEDS="${SEEDS:-1 2 3 4 5}"

if ! ssh -O check fasrc 2>/dev/null; then
    echo "ERROR: No active FASRC tunnel. Run: bash scripts/fasrc_connect_shs.sh"
    exit 1
fi

echo "=== Syncing repo on FASRC ==="
ssh fasrc "cd ~/mfrefactor && git pull --ff-only"

echo "=== Submitting 16 combos × seeds (${SEEDS}) — staggered ==="

# NOTE: heredoc is unquoted so local ${SEEDS} expands here.
# Remote variables use \$ to defer expansion to the cluster shell.
ssh fasrc bash --login << ENDSSH
set -uo pipefail
cd \$HOME/mfrefactor/onpolicy/custom/fish

for SEED in ${SEEDS}; do
    echo ""
    echo "===== Seed \${SEED} ====="
    # Odd seeds → kempner, even seeds → kempner_requeue (roughly even load)
    if (( \${SEED} % 2 == 1 )); then PART="kempner"; else PART="kempner_requeue"; fi

    # dynamic, FO=0.0, binarize
    SEEDS=\$SEED SLURM_PARTITION=\$PART GROUP=20260611_Dyn_F00_Kb_For SENSING_MODEL_TYPE=dynamic FOOD_ORIENTATION_DRIFT=0.0 KNOLLEN_PROCESSING=binarize bash scripts/slurm_submit.sh full
    SEEDS=\$SEED SLURM_PARTITION=\$PART GROUP=20260611_Dyn_F00_Kb_Hom SENSING_MODEL_TYPE=dynamic FOOD_ORIENTATION_DRIFT=0.0 KNOLLEN_PROCESSING=binarize bash scripts/slurm_submit.sh homing

    # dynamic, FO=0.0, log
    SEEDS=\$SEED SLURM_PARTITION=\$PART GROUP=20260611_Dyn_F00_Kl_For SENSING_MODEL_TYPE=dynamic FOOD_ORIENTATION_DRIFT=0.0 KNOLLEN_PROCESSING=log bash scripts/slurm_submit.sh full
    SEEDS=\$SEED SLURM_PARTITION=\$PART GROUP=20260611_Dyn_F00_Kl_Hom SENSING_MODEL_TYPE=dynamic FOOD_ORIENTATION_DRIFT=0.0 KNOLLEN_PROCESSING=log bash scripts/slurm_submit.sh homing

    # dynamic, FO=0.1, binarize
    SEEDS=\$SEED SLURM_PARTITION=\$PART GROUP=20260611_Dyn_F01_Kb_For SENSING_MODEL_TYPE=dynamic FOOD_ORIENTATION_DRIFT=0.1 KNOLLEN_PROCESSING=binarize bash scripts/slurm_submit.sh full
    SEEDS=\$SEED SLURM_PARTITION=\$PART GROUP=20260611_Dyn_F01_Kb_Hom SENSING_MODEL_TYPE=dynamic FOOD_ORIENTATION_DRIFT=0.1 KNOLLEN_PROCESSING=binarize bash scripts/slurm_submit.sh homing

    # dynamic, FO=0.1, log
    SEEDS=\$SEED SLURM_PARTITION=\$PART GROUP=20260611_Dyn_F01_Kl_For SENSING_MODEL_TYPE=dynamic FOOD_ORIENTATION_DRIFT=0.1 KNOLLEN_PROCESSING=log bash scripts/slurm_submit.sh full
    SEEDS=\$SEED SLURM_PARTITION=\$PART GROUP=20260611_Dyn_F01_Kl_Hom SENSING_MODEL_TYPE=dynamic FOOD_ORIENTATION_DRIFT=0.1 KNOLLEN_PROCESSING=log bash scripts/slurm_submit.sh homing

    # frac, FO=0.0, binarize
    SEEDS=\$SEED SLURM_PARTITION=\$PART GROUP=20260611_Frc_F00_Kb_For SENSING_MODEL_TYPE=frac FOOD_ORIENTATION_DRIFT=0.0 KNOLLEN_PROCESSING=binarize bash scripts/slurm_submit.sh full
    SEEDS=\$SEED SLURM_PARTITION=\$PART GROUP=20260611_Frc_F00_Kb_Hom SENSING_MODEL_TYPE=frac FOOD_ORIENTATION_DRIFT=0.0 KNOLLEN_PROCESSING=binarize bash scripts/slurm_submit.sh homing

    # frac, FO=0.0, log
    SEEDS=\$SEED SLURM_PARTITION=\$PART GROUP=20260611_Frc_F00_Kl_For SENSING_MODEL_TYPE=frac FOOD_ORIENTATION_DRIFT=0.0 KNOLLEN_PROCESSING=log bash scripts/slurm_submit.sh full
    SEEDS=\$SEED SLURM_PARTITION=\$PART GROUP=20260611_Frc_F00_Kl_Hom SENSING_MODEL_TYPE=frac FOOD_ORIENTATION_DRIFT=0.0 KNOLLEN_PROCESSING=log bash scripts/slurm_submit.sh homing

    # frac, FO=0.1, binarize
    SEEDS=\$SEED SLURM_PARTITION=\$PART GROUP=20260611_Frc_F01_Kb_For SENSING_MODEL_TYPE=frac FOOD_ORIENTATION_DRIFT=0.1 KNOLLEN_PROCESSING=binarize bash scripts/slurm_submit.sh full
    SEEDS=\$SEED SLURM_PARTITION=\$PART GROUP=20260611_Frc_F01_Kb_Hom SENSING_MODEL_TYPE=frac FOOD_ORIENTATION_DRIFT=0.1 KNOLLEN_PROCESSING=binarize bash scripts/slurm_submit.sh homing

    # frac, FO=0.1, log
    SEEDS=\$SEED SLURM_PARTITION=\$PART GROUP=20260611_Frc_F01_Kl_For SENSING_MODEL_TYPE=frac FOOD_ORIENTATION_DRIFT=0.1 KNOLLEN_PROCESSING=log bash scripts/slurm_submit.sh full
    SEEDS=\$SEED SLURM_PARTITION=\$PART GROUP=20260611_Frc_F01_Kl_Hom SENSING_MODEL_TYPE=frac FOOD_ORIENTATION_DRIFT=0.1 KNOLLEN_PROCESSING=log bash scripts/slurm_submit.sh homing

done
echo ""
echo "=== All submissions done. Check with: squeue -u ${USER:-$(whoami)} ==="
ENDSSH
