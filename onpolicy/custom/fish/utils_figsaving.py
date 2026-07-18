import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json

# --- Utility functions ---
def _save_data(data, path_base):
    """
    Save `data` intelligently to files based on a common base path:
      - pd.DataFrame -> CSV at `<path_base>.csv`
      - list or dict -> JSON at `<path_base>.json`
      - otherwise    -> pickle at `<path_base>`
    Returns the actual path used.
    """
    if isinstance(data, pd.DataFrame):
        out_path = f"{path_base}.csv"
        data.to_csv(out_path, index=False)
    # elif isinstance(data, (list, dict)):
    #     out_path = f"{path_base}.json"
    #     with open(out_path, 'w') as f:
    #         json.dump(data, f, indent=2)
    else:
        out_path = f"{path_base}.pkl"
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
    return out_path


def _load_data(path_base):
    """
    Load data from files matching `path_base` (excluding .png), trying CSV, JSON, then pickle.
    If all fail, raises FileNotFoundError.
    """
    for ext in ['.csv', '.json', '.pkl']:
        try:
            candidate = f"{path_base}{ext}"
            if os.path.exists(candidate):
                if candidate.endswith('.csv'):
                    return pd.read_csv(candidate)
                elif candidate.endswith('.json'):
                    with open(candidate) as f:
                        return json.load(f)
                elif candidate.endswith('.pkl'):
                    with open(candidate, 'rb') as f:
                        return pickle.load(f)
        except Exception:
            continue
    raise FileNotFoundError(f"No data file found for base '{path_base}'")



def _make_fig(nrows=1, ncols=1, height_multiplier=1, width_multiplier=1, edge=2.5):
    """
    Create a figure with subplots sized in 2.5×2.5 block units.
    """
    width = edge * width_multiplier * ncols
    height = edge * height_multiplier * nrows
    figsize = (width, height)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes


def save_fig(fig, base_path, dpi=300, formats=(".pdf",), **kwargs):
    """Save figure. base_path must NOT include extension. Default is PDF only."""
    for ext in formats:
        fig.savefig(f"{base_path}{ext}", dpi=dpi, bbox_inches="tight", **kwargs)
