#!/usr/bin/env bash
# Copy seed_ranking PDFs from cluster mount to manuscript/manual/seeds/.
# Run from onpolicy/custom/fish/.

SRC=~/cluster_lab/satsingh/marl_fish_storage/results/20260611_foraging/20260611_Dyn_F00_Kb_For
DST="$(dirname "$0")/../manuscript/manual/seeds"

mkdir -p "$DST"
cp "$SRC"/seed_ranking_*.pdf "$DST"/
echo "Copied to $DST/"
ls "$DST"/seed_ranking_*.pdf
