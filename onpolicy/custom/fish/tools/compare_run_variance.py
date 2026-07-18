#!/usr/bin/env python3
"""
This script reads several CSV files containing summary statistics over multiple runs,
concatenates them, and computes the aggregated mean and variance for each statistic
across the CSVs.

Usage:
    python compare_run_variance.py parent_directory
    
Assumption:
    parent_directory contains CSVs with the suffix "_m1a1k1cs1sd0_stats.csv" in some subdirectories.

Output:
    command line stats, big figure with boxplots and jittered points saved in parent_directory
"""

import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def read_csv_file(file_path, sep='\t'):
    """
    Reads a CSV file into a DataFrame using the first column as the index.
    """
    try:
        df = pd.read_csv(file_path, sep=sep, index_col=0)
        # Convert all columns to numeric, non-convertible entries become NaN.
        df = df.apply(pd.to_numeric, errors='coerce')
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Compute the aggregated mean and variance of summary statistics across CSV files."
    )
    parser.add_argument("parent_dir", help="Parent directory containing subdirs containing CSV files")
    
    args = parser.parse_args()


    def find_stats_files(root_folder):
        root = Path(root_folder)
        # return list(root.rglob('*_m1a1k1cs1sd0degen_stats.csv'))
        return list(root.rglob('*_m1a1k1cs1sd0uniformdiverse_stats.csv'))

    # Example usage
    args.files = find_stats_files(args.parent_dir)

    for file in args.files:
        print(file)

    
    # Read CSV files into a dictionary {filename: DataFrame}
    dataframes = {}
    for file in args.files:
        try:
            df = pd.read_csv(file, sep='\t', index_col=0)
            dataframes[file] = df
        except Exception:
            print(f"Skipping file {file} due to a read error.")

    if not dataframes:
        print("No valid CSV files to process. Exiting.")
        return

    # Concatenate the DataFrames into one DataFrame with a new "file" level.
    # The MultiIndex will have levels: file and statistic (the original index)
    concatenated = pd.concat(dataframes.values(), keys=dataframes.keys(), names=["file", "statistic"])

    # Now, for each statistic (i.e. for each row) we compute the mean and variance across the CSVs.
    # We use groupby on the second level of the index (the statistic name).
    aggregated_mean = concatenated.groupby(level="statistic").mean()
    aggregated_variance = concatenated.groupby(level="statistic").var()

    print("=== Aggregated Mean of Each Statistic (across CSV files) ===")
    print(aggregated_mean)
    print("\n=== Aggregated Variance of Each Statistic (across CSV files) ===")
    print(aggregated_variance)

    first_df = next(iter(dataframes.values()))
    statistics = list(first_df.index)
    conditions = list(first_df.columns)

    n_stats = len(statistics)
    n_conditions = len(conditions)

    # Create a figure with a grid of subplots.
    fig, axes = plt.subplots(n_stats, n_conditions,
                             figsize=(n_conditions * 3, n_stats * 3),
                             squeeze=False)

    # # For each statistic (row) and each condition (column), gather the values across CSV files
    # for i, stat in enumerate(statistics):
    #     for j, cond in enumerate(conditions):
    #         # Collect the values for this statistic and condition over all CSVs.
    #         values = []
    #         for df in dataframes.values():
    #             # Ensure the CSV contains the statistic and condition column.
    #             if stat in df.index and cond in df.columns:
    #                 values.append(df.loc[stat, cond])
    #         ax = axes[i][j]
    #         # Plot a boxplot on a fixed x location (we use position 1).
    #         ax.boxplot(values, positions=[1], widths=0.3)
    #         # Overlay individual data points with a little horizontal jitter.
    #         jitter = np.random.normal(loc=1, scale=0.05, size=len(values))
    #         ax.scatter(jitter, values, color='blue', zorder=3)
    #         ax.set_xlim(0.5, 1.5)
    #         ax.set_xticks([])

    #         # Optionally, you can add grid lines for better readability.
    #         ax.grid(True, linestyle='--', alpha=0.5)

    #         # If leftmost column, label the y-axis with the statistic name.
    #         if j == 0:
    #             ax.set_ylabel(stat, fontsize=10)
    #         # If top row, set the title to the condition.
    #         if i == 0:
    #             ax.set_title(cond, fontsize=10)
    # Assign a fixed color and jitter to each experiment (CSV) consistently across plots.
    exp_names = list(dataframes.keys())
    num_exps = len(exp_names)
    colors = plt.cm.tab10.colors  # Can use other color maps if more than 10 exps.
    color_map = {exp: colors[i % len(colors)] for i, exp in enumerate(exp_names)}
    np.random.seed(2024)  # Fixed seed for reproducible jitters (or any integer)
    jitters = np.random.normal(loc=1, scale=0.07, size=num_exps)
    jitter_map = {exp: jitters[i] for i, exp in enumerate(exp_names)}

    for i, stat in enumerate(statistics):
        for j, cond in enumerate(conditions):
            values = []
            scatter_x = []
            scatter_color = []
            for exp in exp_names:
                df = dataframes[exp]
                if stat in df.index and cond in df.columns:
                    values.append(df.loc[stat, cond])
                    scatter_x.append(jitter_map[exp])
                    scatter_color.append(color_map[exp])
            ax = axes[i][j]
            # Plot boxplot
            ax.boxplot(values, positions=[1], widths=0.3, patch_artist=True,
                    boxprops=dict(facecolor='white', alpha=0.5))
            # Plot scatter with consistent jitter and color
            ax.scatter(scatter_x, values, color=scatter_color, zorder=3, edgecolor='k', linewidth=0.5)
            ax.set_xlim(0.5, 1.5)
            ax.set_xticks([])
            ax.grid(True, linestyle='--', alpha=0.5)
            if j == 0:
                ax.set_ylabel(stat, fontsize=10)
            if i == 0:
                ax.set_title(cond, fontsize=10)

    # Add a legend to the first subplot, outside the figure for clarity.
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[exp], label=os.path.basename(str(exp)), markersize=6)
        for exp in exp_names
    ]
    axes[0][0].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title='CSV File')
    
    fig.suptitle("Spread of Summary Statistics Across CSV Files", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(Path(args.parent_dir) / "degeneracy_summary_statistics_comparison.pdf", dpi=300)

if __name__ == "__main__":
    main()
