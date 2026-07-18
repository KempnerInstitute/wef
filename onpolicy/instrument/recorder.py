"""
Recorder: per-episode bundle writer replacing Observer.

Writes a 6-file bundle per save() call:
  {stem}_behavior.pkl   — flat DataFrame, one row per (env_id, ep, t, agent_id)
  {stem}_arena.pkl      — flat DataFrame, one row per (env_id, ep, t)
  {stem}_episodes.json  — list of dicts, one per (env_id, ep)
  {stem}_obs.npy        — (n_timesteps, n_envs, n_agents, obs_dim) array
  {stem}_rnn.npy        — (n_timesteps, n_envs, n_agents, ...) array
  {stem}_attn.npy       — (n_timesteps, n_envs, n_agents, obs_dim) array

Fields are passed as **kwargs so the schema is fully extensible: adding or
removing a key at the call site automatically adds/drops a column in the
resulting DataFrame without touching Recorder itself.
"""
import csv
import json
import os

import numpy as np
import pandas as pd


class Recorder:
    def __init__(self):
        self.behavior = []
        self.arena    = []
        self.episodes = []
        self.obs      = []   # list of (n_envs, n_agents, obs_dim) arrays, one per step
        self.rnn      = []   # list of (n_envs, n_agents, ...) arrays, one per step
        self.attn     = []   # list of (n_envs, n_agents, obs_dim) arrays, one per step

    # ------------------------------------------------------------------
    # Recording methods
    # ------------------------------------------------------------------

    def record_behavior(self, env_id, ep, t, agent_id, **fields):
        self.behavior.append({
            "env_id": env_id, "episode_index": ep, "time_step": t,
            "agent_id": agent_id, **fields,
        })

    def record_arena(self, env_id, ep, t, **fields):
        self.arena.append({"env_id": env_id, "episode_index": ep, "time_step": t, **fields})

    def record_episode(self, env_id, ep, **fields):
        self.episodes.append({"env_id": env_id, "episode_index": ep, **fields})

    def record_obs(self, array):
        """Append one timestep's observations; array shape (n_envs, n_agents, obs_dim)."""
        self.obs.append(np.asarray(array))

    def record_rnn(self, array):
        """Append one timestep's RNN states; array shape (n_envs, n_agents, ...)."""
        self.rnn.append(np.asarray(array))

    def record_attn(self, array):
        """Append one timestep's attention mask; array shape (n_envs, n_agents, obs_dim)."""
        self.attn.append(np.asarray(array))

    # ------------------------------------------------------------------
    # Save & reset
    # ------------------------------------------------------------------

    def save(self, stem, summary_csv=None):
        """Write the 5-file bundle, optionally append to summary_csv, then reset."""
        if self.behavior:
            pd.DataFrame(self.behavior).to_pickle(f"{stem}_behavior.pkl")

        if self.arena:
            pd.DataFrame(self.arena).to_pickle(f"{stem}_arena.pkl")

        if self.episodes:
            with open(f"{stem}_episodes.json", "w") as f:
                json.dump(_to_serializable(self.episodes), f)

        if self.obs:
            np.save(f"{stem}_obs.npy", np.stack(self.obs, axis=0))

        if self.rnn:
            np.save(f"{stem}_rnn.npy", np.stack(self.rnn, axis=0))

        if self.attn:
            np.save(f"{stem}_attn.npy", np.stack(self.attn, axis=0))

        if summary_csv is not None and self.behavior:
            _append_summary_csv(self.behavior, self.episodes, summary_csv)

        self.behavior = []
        self.arena    = []
        self.episodes = []
        self.obs      = []
        self.rnn      = []
        self.attn     = []


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _append_summary_csv(behavior, episodes, summary_csv):
    df = pd.DataFrame(behavior)
    ep_meta = {(e["env_id"], e["episode_index"]): e for e in episodes}
    exists = os.path.exists(summary_csv)

    fieldnames = ["episode_index", "env_id", "seed", "eval_run_name",
                  "total_food_eaten", "episode_steps", "mean_cumulative_reward"]

    rows = []
    for (env_id, ep), grp in df.groupby(["env_id", "episode_index"]):
        meta = ep_meta.get((env_id, ep), {})
        total_food  = int(grp["food_eaten_count"].sum()) if "food_eaten_count" in grp.columns else 0
        episode_steps = int(grp["time_step"].nunique())
        mean_reward = float(grp["rewards"].mean()) if "rewards" in grp.columns else float("nan")
        rows.append({
            "episode_index": ep,
            "env_id": env_id,
            "seed": meta.get("seed"),
            "eval_run_name": meta.get("eval_run_name"),
            "total_food_eaten": total_food,
            "episode_steps": episode_steps,
            "mean_cumulative_reward": mean_reward,
        })

    with open(summary_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
