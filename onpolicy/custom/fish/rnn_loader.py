"""
Shared iterator for ep{k}_rnn.npy files in a Recorder raw/ directory.

ep{k}_rnn.npy  shape: (T, E, A, 1, H)
  T = timesteps, E = n_rollout_threads, A = num_agents,
  1 = recurrent layer (GRU), H = hidden_size (512)

Usage
-----
    from rnn_loader import iter_rnn_episodes, load_rnn_episode

    for k, rnn_arr, dff_ep in iter_rnn_episodes(raw_dir, dff):
        # rnn_arr: (T, E, A, H) — layer dim squeezed
        # dff_ep:  dff[dff.episode_index == k]
        ...

    rnn = load_rnn_episode(raw_dir, k)   # (T, E, A, H)
"""

import re
import warnings
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd


def _sorted_rnn_files(raw_dir: str | Path):
    """Return list of (k, path) for ep{k}_rnn.npy, sorted numerically by k."""
    raw_dir = Path(raw_dir)
    files = []
    for p in raw_dir.glob("ep*_rnn.npy"):
        m = re.match(r"ep(\d+)_rnn\.npy$", p.name)
        if m:
            files.append((int(m.group(1)), p))
    return sorted(files, key=lambda x: x[0])


def load_rnn_episode(raw_dir: str | Path, k: int) -> np.ndarray:
    """Load ep{k}_rnn.npy and return (T, E, A, H) float32 array."""
    path = Path(raw_dir) / f"ep{k}_rnn.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    rnn = np.load(path)           # (T, E, A, 1, H)
    return rnn.squeeze(axis=3)    # (T, E, A, H)


def iter_rnn_episodes(
    raw_dir: str | Path,
    dff: Optional[pd.DataFrame] = None,
) -> Iterator[tuple[int, np.ndarray, Optional[pd.DataFrame]]]:
    """
    Yield (k, rnn_arr, dff_ep) for each ep{k}_rnn.npy, sorted by k.

    Parameters
    ----------
    raw_dir : path to the eval raw/ directory
    dff     : optional per_env_ep_agent_step DataFrame; if given, yield the
              slice where episode_index == k as dff_ep, else yield None.

    Yields
    ------
    k       : int — episode index
    rnn_arr : ndarray (T, E, A, H) float32
    dff_ep  : DataFrame slice or None
    """
    files = _sorted_rnn_files(raw_dir)
    if not files:
        warnings.warn(f"No ep*_rnn.npy files found in {raw_dir}")
        return

    if dff is not None:
        ep_indices_in_dff = set(dff["episode_index"].unique())
        for k, path in files:
            if k not in ep_indices_in_dff:
                warnings.warn(
                    f"ep{k}_rnn.npy found but episode_index={k} missing from dff"
                )

    for k, path in files:
        rnn_arr = np.load(path).squeeze(axis=3)   # (T, E, A, H)
        dff_ep = dff[dff["episode_index"] == k].copy() if dff is not None else None
        yield k, rnn_arr, dff_ep
