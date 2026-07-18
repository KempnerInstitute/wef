"""
Find best seed per group on cluster storage, print run dirs for rsync.
Uses utils_seeds.py / utils_seeds_homing.py ranking logic.
"""
import sys
import os
import getpass
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "notebooks"))

from pathlib import Path
import numpy as np
import utils_seeds as us
import utils_seeds_homing as ush

_USER = getpass.getuser()
RESULTS_DIR = Path(f"/home/{_USER}/cluster_lab/{_USER}/marl_fish_storage/results")
MAIN_EVAL   = "m1a1k1_patchy_square"

FORAGING_GROUPS = [
    "ConsNoise20260609dynamicT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU",
    "ConsNoise20260609dynamicT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU",
    "ConsNoise20260610dynamicT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU",
    "ConsNoise20260610dynamicT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU",
    "ConsNoise20260610fracT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU",
    "ConsNoise20260610fracT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU",
    "ConsNoise20260611dynamicT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU",
    "ConsNoise20260611dynamicT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K2M1GRUBEnumba",
    "ConsNoise20260611dynamicT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU",
    "ConsNoise20260611fracT5MFO0.0FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU",
    # overnight 20260611 runs (short group names)
    "20260611_Dyn_F00_Kb_For",
    "20260611_Dyn_F00_Kl_For",
    "20260611_Dyn_F01_Kb_For",
    "20260611_Dyn_F01_Kl_For",
    "20260611_Frc_F00_Kb_For",
    "20260611_Frc_F00_Kl_For",
    "20260611_Frc_F01_Kb_For",
    "20260611_Frc_F01_Kl_For",
]

HOMING_GROUPS = [
    "Homing20260609ConsNoise0.5dynamicT2MFood0.0",
    "20260611_Dyn_F00_Kb_Hom",
    "20260611_Dyn_F00_Kl_Hom",
    "20260611_Dyn_F01_Kb_Hom",
    "20260611_Dyn_F01_Kl_Hom",
    "20260611_Frc_F00_Kb_Hom",
    "20260611_Frc_F00_Kl_Hom",
    "20260611_Frc_F01_Kb_Hom",
    "20260611_Frc_F01_Kl_Hom",
]

best_run_dirs = []

print("=" * 80)
print("FORAGING GROUPS")
print("=" * 80)
for group in FORAGING_GROUPS:
    try:
        run_dirs = us.discover_run_dirs(RESULTS_DIR, group)
        if not run_dirs:
            print(f"\n{group}\n  [NO RUN DIRS]")
            continue
        df = us.load_ranking_df(run_dirs, main_eval=MAIN_EVAL, last_n=5)
        # Filter out seeds with NaN food_mean before ranking (avoids NaN-rank-0 bug)
        df_valid = df[df["food_mean"].notna()].copy()
        if df_valid.empty:
            print(f"\n{group}\n  [ALL food_mean NaN — skipping]")
            continue
        df_valid = us.rank_seeds(df_valid)
        ranked = df_valid.sort_values("combined_rank")
        best = ranked.iloc[0]
        best_dir = best["run_dir"]
        seed = int(best["seed"])
        food = best.get("food_mean", np.nan)
        slope = best.get("size_food_slope", np.nan)
        r2 = best.get("size_food_r2", np.nan)
        crank = int(best["combined_rank"])
        print(f"\n{group}")
        print(f"  n_seeds={len(df)}  best=Seed{seed}  food={food:.2f}  slope={slope:.3f}  R²={r2:.3f}  combined_rank={crank}")
        print(f"  -> {best_dir}")
        best_run_dirs.append(best_dir)
    except Exception as e:
        print(f"\n{group}\n  [ERROR] {e}")

print("\n" + "=" * 80)
print("HOMING GROUPS")
print("=" * 80)
for homing_group in HOMING_GROUPS:
    try:
        run_dirs = us.discover_run_dirs(RESULTS_DIR, homing_group)
        if not run_dirs:
            print(f"\n{homing_group}\n  [NO RUN DIRS]")
            continue
        df_h = ush.load_homing_ranking_df(run_dirs, last_n=5)
        if "success_percent" in df_h.columns and df_h["success_percent"].notna().any():
            best_h = df_h.sort_values("success_percent", ascending=False).iloc[0]
            metric_str = f"success={best_h['success_percent']:.1f}%"
        else:
            best_h = df_h.sort_values("train_final_reward", ascending=False).iloc[0]
            metric_str = f"train_reward={best_h['train_final_reward']:.3f}"
        seed_h = int(best_h["seed"])
        best_dir_h = best_h["run_dir"]
        print(f"\n{homing_group}")
        print(f"  n_seeds={len(df_h)}  best=Seed{seed_h}  {metric_str}")
        print(f"  -> {best_dir_h}")
        best_run_dirs.append(best_dir_h)
    except Exception as e:
        print(f"\n{homing_group}\n  [ERROR] {e}")

print("\n" + "=" * 80)
print("RSYNC SOURCES (one per group)")
print("=" * 80)
for d in best_run_dirs:
    print(d)
