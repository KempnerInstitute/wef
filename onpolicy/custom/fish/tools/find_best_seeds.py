#!/usr/bin/env python3
"""
compare_csvs_best.py

This script reads several CSV files containing summary statistics over multiple runs,
filters on seeds that are above the bottom quartile in food eaten for each of the arena types,
finds the top 5 CSV files based on the aggregated food eaten metric (sum over arenas),
and then generates a plot that visualizes the spread of all summary statistics for these top seeds.

Usage:
    python compare_csvs_best.py <parent_dir>

CSV Format Assumption:
    - The CSV files are tab-delimited.
    - The first row contains column names (e.g. "All", "UniformArena", "PatchyArena", "FeederArena", etc.).
    - The first column contains statistic names (e.g. "food_eaten", "num_agents", etc.).
    - The row for "food_eaten" is used to compute performance.
"""

import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

def read_csv_file(file_path, sep='\t'):
    """
    Reads a CSV file into a DataFrame using the first column as the index.
    """
    try:
        df = pd.read_csv(file_path, sep=sep, index_col=0)
        df = df.apply(pd.to_numeric, errors='coerce')
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        raise

def find_stats_files(root_folder):
    """
    Recursively search for CSV files that match the pattern *_m1a1k1cs1sd0_stats.csv
    in the provided directory.
    """
    root = Path(root_folder)
    return list(root.rglob('*_m1a1k1cs1sd0degen_stats.csv'))


def find_eligible_files(performance_dict, arena_names, desired_k, tolerance=1):
    """
    Find eligible files by binary search over percentile thresholds to get close to desired_k seeds.
    
    Args:
        performance_dict: dict mapping filename to performance Series.
        arena_names: list of arenas to consider.
        desired_k: target number of eligible seeds.
        tolerance: allowed difference between found and desired number.
    
    Returns:
        eligible_files: dict of eligible seeds
        thresholds: dict of per-arena thresholds
    """
    lo = 0
    hi = 100
    best_eligible = {}
    best_percentile = None

    max_iterations = 20
    for _ in range(max_iterations):
        mid = (lo + hi) / 2

        # Compute thresholds at current percentile
        thresholds = {}
        for arena in arena_names:
            values = [perf[arena] for perf in performance_dict.values()]
            thresholds[arena] = np.percentile(values, mid)

        # Find eligible seeds
        eligible = {}
        for fname, perf in performance_dict.items():
            if all(perf[arena] >= thresholds[arena] for arena in arena_names):
                eligible[fname] = perf

        n_eligible = len(eligible)
        print(f"At percentile {mid:.2f}: {n_eligible} seeds eligible (target {desired_k})")

        # Check if close enough
        if abs(n_eligible - desired_k) <= tolerance:
            return eligible, mid

        # Save best so far (closest to k)
        if best_percentile is None or abs(n_eligible - desired_k) < abs(len(best_eligible) - desired_k):
            best_eligible = eligible
            best_percentile = mid

        # Adjust search range
        if n_eligible > desired_k:
            lo = mid
        else:
            hi = mid

    print(f"Binary search did not find exact match within {max_iterations} iterations.")
    print(f"Using closest found at percentile {best_percentile:.2f} with {len(best_eligible)} seeds.")
    
    # Recompute thresholds for best_percentile
    thresholds = {}
    for arena in arena_names:
        values = [perf[arena] for perf in performance_dict.values()]
        thresholds[arena] = np.percentile(values, best_percentile)

    return best_eligible, best_percentile



def main():
    parser = argparse.ArgumentParser(
        description="Find the top-performing CSV files based on food eaten and generate summary plots."
    )
    parser.add_argument("parent_dir", help="Parent directory containing subdirectories with CSV files.")
    args = parser.parse_args()

    # Find CSV files matching the expected pattern.
    files = find_stats_files(args.parent_dir)

    if not files:
        print("No CSV files matching the pattern were found in the given directory.")
        return

    print("Found CSV files:")
    for file in files:
        print(file)

    # arena_names = ["All", "UniformArena", "PatchyArena", "FeederArena"]
    arena_names = ["All", "UniformArena", "PatchyArena"]

    # Read all CSV files and store the full DataFrame in a dictionary.
    all_dataframes = {}
    performance_dict = {}  # Maps file name (as string) to food_eaten Series for the specified arenas.
    for file in files:
        try:
            df = read_csv_file(file, sep='\t')
            all_dataframes[str(file)] = df  # Save full dataframe for later plotting.
            if "food_eaten" not in df.index:
                print(f"Skipping file {file} because 'food_eaten' is not present.")
                continue
            perf = df.loc["food_eaten", arena_names]
            # Skip if any arena value is missing.
            if perf.isnull().any():
                print(f"Skipping file {file} due to missing values in 'food_eaten'.")
                continue
            performance_dict[str(file)] = perf
        except Exception as e:
            print(f"Skipping file {file} due to error: {e}")

    if not performance_dict:
        print("No valid CSV files with food_eaten data to process. Exiting.")
        return

    # # Compute the 25th percentile for each arena across all files.
    # thresholds = {}
    # for arena in arena_names:
    #     values = [perf[arena] for perf in performance_dict.values()]
    #     threshold_value = np.percentile(values, 50)
    #     thresholds[arena] = threshold_value

    # # Print out the 25th percentiles.
    # print("\n=== 25th Percentile for food_eaten per Arena ===")
    # for arena, threshold in thresholds.items():
    #     print(f"{arena}: {threshold}")

    desired_k = 5  # or whatever number you want
    eligible_files, best_percentile = find_eligible_files(performance_dict, arena_names, desired_k=desired_k)
    print(f"\nFound {len(eligible_files)} eligible files at percentile {best_percentile}%.")

    # # Filter CSV files: only include files that are above the threshold in every arena.
    # eligible_files = {}
    # for fname, perf in performance_dict.items():
    #     if all(perf[arena] > thresholds[arena] for arena in arena_names):
    #         eligible_files[fname] = perf

    if not eligible_files:
        print("\nNo CSV files passed the filter (all arenas above bottom quartile). Exiting.")
        return

    # Compute performance sum for each eligible file.
    performance_sums = {fname: perf.sum() for fname, perf in eligible_files.items()}

    # Rank files based on the overall sum (higher is better).
    sorted_files = sorted(performance_sums.items(), key=lambda x: x[1], reverse=True)

    # Select the top 5 CSV files.
    top_n = 5
    top_files = sorted_files[:top_n]

    print("\n=== Top Performing CSV Files ===")
    for fname, total in top_files:
        perf = eligible_files[fname]
        print(f"\nFile: {fname}")
        for arena in arena_names:
            print(f"  {arena}: {perf[arena]}")
        print(f"  Total food_eaten (sum): {total}")

    # ---------------------------------------------------------------------
    # Plotting for the Top 5 Seeds
    # ---------------------------------------------------------------------
    # Create a dictionary of DataFrames for only the top seeds.
    top_dataframes = {}
    for fname, _ in top_files:
        if fname in all_dataframes:
            top_dataframes[fname] = all_dataframes[fname]
    
    # Verify that we have at least one dataframe to plot.
    if not top_dataframes:
        print("No data available for plotting the top seeds. Exiting plotting.")
        return

    # Use the first dataframe to extract the list of statistics (rows) and conditions (columns).
    first_df = next(iter(top_dataframes.values()))
    statistics = list(first_df.index)
    conditions = list(first_df.columns)

    n_stats = len(statistics)
    n_conditions = len(conditions)

    # Create a figure with a grid of subplots.
    fig, axes = plt.subplots(n_stats, n_conditions,
                             figsize=(n_conditions * 3, n_stats * 3),
                             squeeze=False)

    top_filenames = list(top_dataframes.keys())
    color_map = cm.get_cmap('tab10', len(top_filenames))  # Or 'Set1', 'Dark2', etc.
    file_colors = {fname: color_map(i) for i, fname in enumerate(top_filenames)}


    # For each statistic (row) and each condition (column), gather the values across the top CSV files.
    for i, stat in enumerate(statistics):
        for j, cond in enumerate(conditions):
            values = []
            for df in top_dataframes.values():
                if stat in df.index and cond in df.columns:
                    values.append(df.loc[stat, cond])
            ax = axes[i][j]
            # Plot a boxplot on a fixed x location (we use position 1).
            ax.boxplot(values, positions=[1], widths=0.3)

            for fname_idx, (fname, df) in enumerate(top_dataframes.items()):
                if stat in df.index and cond in df.columns:
                    y = df.loc[stat, cond]
                    x = 1 + np.random.normal(loc=0, scale=0.05)  # jitter
                    ax.scatter(x, y, color=file_colors[fname], label=fname if (i == 0 and j == 0) else "", zorder=3)
            # Overlay individual data points with a little horizontal jitter.
            # jitter = np.random.normal(loc=1, scale=0.05, size=len(values))
            # ax.scatter(jitter, values, color='blue', zorder=3)
            ax.set_xlim(0.5, 1.5)
            ax.set_xticks([])

            # Optionally, add grid lines for better readability.
            ax.grid(True, linestyle='--', alpha=0.5)

            # Label the y-axis with the statistic name on the leftmost column.
            if j == 0:
                ax.set_ylabel(stat, fontsize=10)
            # Set the title to the condition on the top row.
            if i == 0:
                ax.set_title(cond, fontsize=10)
    
    fig.suptitle("Spread of Summary Statistics Across Top 5 CSV Files", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = Path(args.parent_dir) / f"summary_statistics_comparison_top_5_percentile_{best_percentile}.pdf"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved as {save_path}")

if __name__ == "__main__":
    main()
