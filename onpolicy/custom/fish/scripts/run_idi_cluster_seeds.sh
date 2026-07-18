#!/usr/bin/env bash
# Re-run analysis_idi + copy_to_figs + regenerate_latex for all 20 cluster seeds.
# Run from onpolicy/custom/fish/
set -euo pipefail

FISH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_BASE=~/cluster_lab/satsingh/marl_fish_storage/results
GROUP=20260611_foraging/20260611_Dyn_F00_Kb_For
SPEC=2fish_m1a1k1_uniform_wide
MAMBA=/home/$USER/miniforge3/bin/mamba
N_PARALLEL=5

echo "=== IDI cluster seeds batch ==="
echo "FISH_DIR: $FISH_DIR"
echo "Parallel: $N_PARALLEL"
echo

process_seed() {
    local ts_dir="$1"
    local seed_name
    seed_name=$(basename "$(dirname "$ts_dir")")
    local log_file="$ts_dir/idi_regen.log"

    echo "[$seed_name] starting → $ts_dir"

    {
        # Remove done marker so analysis re-runs
        rm -f "$ts_dir/evals/$SPEC/analyses/idi/.analysis_done_idi"

        # Run IDI analysis
        cd "$FISH_DIR"
        $MAMBA run --name mfrefactor \
            python analysis_idi.py \
            --spec_dir "$ts_dir/evals/$SPEC"

        # Copy outputs to figs/
        $MAMBA run --name mfrefactor \
            python copy_to_figs.py "$ts_dir" --force

        # Regenerate figseed.pdf
        bash "$FISH_DIR/scripts/regenerate_latex.sh" "$ts_dir"

        echo "[$seed_name] DONE"
    } > "$log_file" 2>&1 && echo "[$seed_name] SUCCESS" || echo "[$seed_name] FAILED (see $log_file)"
}

export -f process_seed
export FISH_DIR CLUSTER_BASE GROUP SPEC MAMBA

# Collect all seed ts_dirs
SEED_DIRS=()
for seed_dir in "$CLUSTER_BASE/$GROUP"/*/; do
    ts_dir=$(ls -d "$seed_dir"*/ 2>/dev/null | head -1)
    [ -z "$ts_dir" ] && continue
    ts_dir="${ts_dir%/}"
    SEED_DIRS+=("$ts_dir")
done

echo "Found ${#SEED_DIRS[@]} seeds"
echo

# Run N_PARALLEL at a time
job_count=0
pids=()
names=()

for ts_dir in "${SEED_DIRS[@]}"; do
    process_seed "$ts_dir" &
    pids+=($!)
    names+=("$(basename "$(dirname "$ts_dir")")")
    job_count=$((job_count + 1))
    if (( job_count % N_PARALLEL == 0 )); then
        for i in "${!pids[@]}"; do
            wait "${pids[$i]}" && echo "  [batch] ${names[$i]} done" || echo "  [batch] ${names[$i]} FAILED"
        done
        pids=()
        names=()
    fi
done

# Wait for any remaining jobs
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" && echo "  [batch] ${names[$i]} done" || echo "  [batch] ${names[$i]} FAILED"
done

echo
echo "=== All seeds processed ==="
