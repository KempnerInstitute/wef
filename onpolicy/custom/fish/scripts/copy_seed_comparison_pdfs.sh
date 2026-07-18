#!/usr/bin/env bash
# Copy per-seed comparison PDFs from cluster mount to manuscript/manual/seeds/rank{1,2,3}/.
# Seeds: Rank 1 = Seed 1, Rank 2 = Seed 18, Rank 3 = Seed 9  (20260611_Dyn_F00_Kb_For group)
# Run from anywhere; paths are relative to this script's location.

set -euo pipefail

SRC_BASE=~/cluster_lab/satsingh/marl_fish_storage/results/20260611_foraging/20260611_Dyn_F00_Kb_For
DST_BASE="$(cd "$(dirname "$0")/.." && pwd)/manuscript/manual/seeds"

declare -A SEED_DIRS=(
  [1]="20260611_Dyn_F00_Kb_ForSeed1"
  [2]="20260611_Dyn_F00_Kb_ForSeed18"
  [3]="20260611_Dyn_F00_Kb_ForSeed9"
)

for rank in 1 2 3; do
  seed_dir="$SRC_BASE/${SEED_DIRS[$rank]}"
  run_dir=$(ls -d "$seed_dir"/*/ 2>/dev/null | sort | tail -1)
  [[ -z "$run_dir" ]] && { echo "ERROR: no run dir for rank $rank ($seed_dir)"; continue; }

  dst="$DST_BASE/rank${rank}"
  mkdir -p "$dst"
  echo "Rank $rank  (${SEED_DIRS[$rank]}) -> $dst"

  # IDI histogram — prefer _12ms if present
  copied=0
  for f in idi_histogram_12ms.pdf idi_histogram.pdf; do
    src="$run_dir/evals/2fish_m1a1k1_uniform_wide/analyses/idi/$f"
    if [[ -f "$src" ]]; then
      cp "$src" "$dst/idi_histogram.pdf"
      echo "  idi_histogram.pdf  <- $f"
      copied=1; break
    fi
  done
  [[ $copied -eq 0 ]] && echo "  WARNING: idi_histogram not found"

  # IDI powerlaw
  f="$run_dir/evals/2fish_m1a1k1_uniform_wide/analyses/idi/idi_powerlaw.pdf"
  [[ -f "$f" ]] && cp "$f" "$dst/" && echo "  idi_powerlaw.pdf" || echo "  WARNING: idi_powerlaw not found"

  # size vs food (Fig 3M)
  f="$run_dir/evals/m1a1k1_patchy_square/analyses/general/size_vs_food.pdf"
  [[ -f "$f" ]] && cp "$f" "$dst/" && echo "  size_vs_food.pdf" || echo "  WARNING: size_vs_food not found"

  # biter/bitten size by role boxplot (Fig 5M)
  f="$run_dir/evals/2fish_m1a1k1_uniform_wide/analyses/pairwise/biter_size_by_role_boxplot.pdf"
  [[ -f "$f" ]] && cp "$f" "$dst/" && echo "  biter_size_by_role_boxplot.pdf" || echo "  WARNING: biter_size_by_role_boxplot not found"

  # chaser/chased size by role boxplot (Fig 5K; alt for Col 3)
  f="$run_dir/evals/2fish_m1a1k1_uniform_wide/analyses/pairwise/chaser_size_by_role_boxplot.pdf"
  [[ -f "$f" ]] && cp "$f" "$dst/" && echo "  chaser_size_by_role_boxplot.pdf" || echo "  WARNING: chaser_size_by_role_boxplot not found"

  # RNN effective dimensionality (Fig 6A)
  f="$run_dir/evals/m1a1k1_patchy_square/analyses/rnn_dim/rnn_dim_pca_per_episode_cumvar.pdf"
  [[ -f "$f" ]] && cp "$f" "$dst/" && echo "  rnn_dim_pca_per_episode_cumvar.pdf" || echo "  WARNING: rnn_dim not found"
done
echo "Done."
