import sys, os; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pandas as pd
import glob
import argparse
import utils_report as ru
import utils_homing as uh

parser = argparse.ArgumentParser()
parser.add_argument("base_path", nargs="?", default="./results/degenHoming/")
args = parser.parse_args()

BASE_PATH = args.base_path
RUN_FOLDERS = None # If None, will auto-discover run folders in BASE_PATH. Otherwise, specify a list of folder names to use.

base_path_abs = os.path.abspath(BASE_PATH)
if RUN_FOLDERS is None:
    RUN_FOLDERS = sorted([
        name for name in os.listdir(base_path_abs)
        if os.path.isdir(os.path.join(base_path_abs, name))
    ])

RUN_PATHS = [os.path.join(base_path_abs, run) for run in RUN_FOLDERS]
print(f"Using BASE_PATH: {base_path_abs}")
print(f"Discovered {len(RUN_FOLDERS)} run folders")

import re
pattern = "**/outputs/*_agg_flattened.pkl"
all_fpkls = glob.glob(os.path.join(BASE_PATH, "**", "*.pkl"), recursive=True)
subset_fpkls = [f for f in all_fpkls if glob.fnmatch.fnmatch(f, pattern)]

dfs = []
for fpkl_file in subset_fpkls:
    dff, OUTPUTS_FOLDER, pkl_str = ru.load_flat_pkl_file(fpkl_file, task='homing')
    eligible_episodes, success_percent = uh.get_eligible_homing_episode_ids(dff)

    csv_base_name = os.path.basename(fpkl_file)
    m = re.search(r"(\d{8}_\d{6})", csv_base_name)
    run_timestamp_id = m.group(1) if m else None

    dfs.append(pd.DataFrame({
        "run_timestamp_id": [run_timestamp_id],
        "success_percent": [success_percent],
        # "df": dff
    }))
    print(f"Found FPKL file: {csv_base_name} with shape {dff.shape}")

if not dfs:
    print("No matching FPKLs found.")
else:
    df = pd.concat(dfs, ignore_index=True)
    # print(df.columns)

    if "arena_type" not in df.columns:
        print("arena_type column not found; cannot group by arena_type.")
    else:
        numeric_cols = df.select_dtypes(include="number").columns
        df_summary = (
            df.groupby(["run_timestamp_id", "arena_type"])[numeric_cols]
            .agg(["mean", "std"])
            .sort_index()
        )

# Print the success_percent column sorted by mean success_percent in descending order
print(df.sort_values("success_percent", ascending=False))
