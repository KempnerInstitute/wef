"""Event-time data extraction and caching for analysis_*.py scripts.

Pure data layer behind the event-time *plotters* in analysis_style.py:
converts step-level DataFrames into per-episode event-time arrays (seconds)
and reads/writes the derived/event_times.pkl cache.  No plotting here.
"""
import os
import pickle

import numpy as np
import pandas as pd

from cfg import SIM_FPS


def _event_times_from_step_df(df, event_col):
    """Return one sorted time-array (seconds) per episode for events in event_col."""
    events = df[df[event_col].astype(bool)][["env_id", "episode_index", "time_step"]]
    if len(events) == 0:
        return []
    ep_starts = df.groupby(["env_id", "episode_index"])["time_step"].min()
    result = []
    for (env_id, ep_idx), grp in events.groupby(["env_id", "episode_index"]):
        ts = np.sort(grp["time_step"].values)
        start = ep_starts.loc[(env_id, ep_idx)]
        result.append((ts - start) / SIM_FPS)
    return result


def consumption_times_from_step_df(df):
    """One sorted time-array per episode (eating_event)."""
    return _event_times_from_step_df(df, "eating_event")


def biting_times_from_step_df(df):
    """One sorted time-array per episode (was_bitten events)."""
    if "was_bitten" not in df.columns:
        return []
    return _event_times_from_step_df(df, "was_bitten")


def _load_event_times_cache(spec_dir):
    """Return cached (consumption_times, biting_times) from event_times.pkl, or None if absent."""
    cache = os.path.join(spec_dir, "derived", "event_times.pkl")
    if not os.path.exists(cache):
        return None
    with open(cache, "rb") as f:
        data = pickle.load(f)
    return data.get("consumption_times", []), data.get("biting_times", [])


def _build_and_cache_event_times(spec_dir):
    """Read step pkl once for both event types, write event_times.pkl, return (ct, bt)."""
    path = os.path.join(spec_dir, "derived", "per_env_ep_agent_step.pkl")
    if not os.path.exists(path):
        return [], []
    keep = ["env_id", "episode_index", "time_step", "eating_event", "was_bitten"]
    df_full = pd.read_pickle(path)
    df = df_full[[c for c in keep if c in df_full.columns]]
    del df_full
    ct = consumption_times_from_step_df(df)
    bt = biting_times_from_step_df(df) if "was_bitten" in df.columns else []
    data = {"consumption_times": ct, "biting_times": bt}
    cache = os.path.join(spec_dir, "derived", "event_times.pkl")
    try:
        with open(cache, "wb") as f:
            pickle.dump(data, f, protocol=4)
    except OSError:
        pass  # read-only filesystem — skip caching silently
    return ct, bt


def load_consumption_times(spec_dir):
    """Return episode consumption-time arrays, using event_times.pkl cache when available."""
    cached = _load_event_times_cache(spec_dir)
    if cached is not None:
        return cached[0]
    ct, _ = _build_and_cache_event_times(spec_dir)
    return ct


def load_biting_times(spec_dir):
    """Return episode biting-time arrays, using event_times.pkl cache when available."""
    cached = _load_event_times_cache(spec_dir)
    if cached is not None:
        return cached[1]
    _, bt = _build_and_cache_event_times(spec_dir)
    return bt
