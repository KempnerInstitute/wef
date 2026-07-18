#!/usr/bin/env bash
#SBATCH --job-name=regen_latex
#SBATCH --partition=kempner
#SBATCH --account=kempner_krajan_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=./logs/regen_latex_%j.out

set -uo pipefail
FISH_DIR=~/mfrefactor/onpolicy/custom/fish
STORAGE=/n/holylfs06/LABS/krajan_lab/Lab/${USER:-$(whoami)}/marl_fish_storage/results
MAMBA=/n/home02/${USER:-$(whoami)}/miniforge3/bin/mamba
OK=0; FAIL=0; SKIP=0

cd "$FISH_DIR"

for GROUP in ConsNoise20260609dynamicT5MFO0.1 ConsNoise20260609dynamicT5MFO0.0 ConsNoise20260610fracT5MFO0.1; do
  FULL=$(ls -d $STORAGE/${GROUP}* 2>/dev/null | head -1)
  [[ -z "$FULL" ]] && continue
  for SEED_DIR in "$FULL"/*/; do
    RUN_DIR=$(ls -d "$SEED_DIR"*/ 2>/dev/null | head -1)
    [[ -z "$RUN_DIR" ]] && continue
    DONE=$(find "$RUN_DIR" -name '.analysis_done_general' 2>/dev/null | wc -l)
    if [[ $DONE -lt 2 ]]; then SKIP=$((SKIP+1)); continue; fi
    echo "=== $(basename $SEED_DIR) ==="
    $MAMBA run -n mfrefactor \
        python3 analysis_interventions.py \
        --evals_dir "$RUN_DIR/evals" \
        2>&1 | tail -3
    $MAMBA run -n mfrefactor \
        python3 copy_to_figs.py "$RUN_DIR" > /dev/null 2>&1
    bash scripts/regenerate_latex.sh "$RUN_DIR" > /dev/null 2>&1 \
        && OK=$((OK+1)) || { echo "  latex FAILED"; FAIL=$((FAIL+1)); }
  done
done
echo ""
echo "Done: OK=$OK  FAIL=$FAIL  SKIP=$SKIP"
