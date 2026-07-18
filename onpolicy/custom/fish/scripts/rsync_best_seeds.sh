#!/usr/bin/env bash
# Rsync best-seed run dirs from cluster mount to ~/srv/marl/${USER:-$(whoami)}/marl_fish/NEW/
# Each run dir is placed as NEW/{seed_dir_name}/{timestamp}/

set -euo pipefail

DEST=~/srv/marl/${USER:-$(whoami)}/marl_fish/NEW

# Best-seed run dirs (one per group, from find_best_seeds.py)
RUN_DIRS=(
  "/home/${USER:-$(whoami)}/cluster_lab/${USER:-$(whoami)}/marl_fish_storage/results/ConsNoise20260609dynamicT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU/ConsNoise20260610dynamicT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRUSeed16/20260610_024044"
  "/home/${USER:-$(whoami)}/cluster_lab/${USER:-$(whoami)}/marl_fish_storage/results/ConsNoise20260609dynamicT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU/ConsNoise20260609dynamicT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRUSeed7/20260609_193244"
  "/home/${USER:-$(whoami)}/cluster_lab/${USER:-$(whoami)}/marl_fish_storage/results/ConsNoise20260610dynamicT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU/ConsNoise20260610dynamicT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRUSeed1/20260610_192043"
  "/home/${USER:-$(whoami)}/cluster_lab/${USER:-$(whoami)}/marl_fish_storage/results/ConsNoise20260610dynamicT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU/ConsNoise20260610dynamicT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRUSeed2/20260610_192031"
  "/home/${USER:-$(whoami)}/cluster_lab/${USER:-$(whoami)}/marl_fish_storage/results/ConsNoise20260610fracT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU/ConsNoise20260610fracT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRUSeed1/20260610_192130"
  "/home/${USER:-$(whoami)}/cluster_lab/${USER:-$(whoami)}/marl_fish_storage/results/ConsNoise20260610fracT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU/ConsNoise20260610fracT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRUSeed14/20260610_062019"
  "/home/${USER:-$(whoami)}/cluster_lab/${USER:-$(whoami)}/marl_fish_storage/results/ConsNoise20260611dynamicT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU/ConsNoise20260611dynamicT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRUSeed6/20260611_001608"
  "/home/${USER:-$(whoami)}/cluster_lab/${USER:-$(whoami)}/marl_fish_storage/results/ConsNoise20260611dynamicT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K2M1GRUBEnumba/ConsNoise20260611dynamicT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K2M1GRUBEnumbaSeed3/20260611_105017"
  "/home/${USER:-$(whoami)}/cluster_lab/${USER:-$(whoami)}/marl_fish_storage/results/ConsNoise20260611dynamicT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU/ConsNoise20260611dynamicT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRUSeed10/20260611_000817"
  "/home/${USER:-$(whoami)}/cluster_lab/${USER:-$(whoami)}/marl_fish_storage/results/ConsNoise20260611fracT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU/ConsNoise20260611fracT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRUSeed6/20260611_050252"
  "/home/${USER:-$(whoami)}/cluster_lab/${USER:-$(whoami)}/marl_fish_storage/results/Homing20260609ConsNoise0.5dynamicT2MFood0.0/Homing20260609ConsNoise0.5dynamicT2MFood0.0Seed20/20260609_102145"
)

mkdir -p "$DEST"

TOTAL=${#RUN_DIRS[@]}
i=0
for RUN_DIR in "${RUN_DIRS[@]}"; do
  i=$((i + 1))
  SEED_DIR_NAME=$(basename "$(dirname "$RUN_DIR")")
  TIMESTAMP=$(basename "$RUN_DIR")
  DEST_PATH="${DEST}/${SEED_DIR_NAME}/${TIMESTAMP}"
  echo ""
  echo "[$i/$TOTAL] $SEED_DIR_NAME / $TIMESTAMP"
  echo "  src : $RUN_DIR/"
  echo "  dest: $DEST_PATH/"
  mkdir -p "$DEST_PATH"
  rsync -av --progress --stats "$RUN_DIR/" "$DEST_PATH/" 2>&1
  echo "  [DONE $i/$TOTAL]"
done

echo ""
echo "All $TOTAL rsyncs complete."
