"""
Helpful notes:

Code Flow:
1. Parse evaluation-specific command-line arguments `eval_[...]`.
2. Load saved JSONs for training arguments (`all_args`) and environment settings (`env_args`).
3. Override training arguments with evaluation-time options via `update_all_args()`.
4. Reconstruct the environment and agent configuration using `make_env`.
5. Instantiate runner and run `.render()`.
6. Log the configuration used in a CSV for easy reference.

To Add a new eval argument:
1. Add the `--eval_<argname>` to the argparsing.
2. In `update_all_args()`, check if `eval_args.eval_<argname>` is not None and apply it to `all_args`.
"""

import itertools
import os
import argparse
import json
import glob
from pathlib import Path
from datetime import datetime
import ast

import numpy as np

from fish_runner_shared import MAFishRunner as SharedRunner
from fish_runner_separated import MAFishRunner as SeparatedRunner
from train_fish import (
    make_env,
    setup_device,
)  # maybe these should go in a utilities file
import cfg
import random
import torch

def int_or_none(value):
    if value.lower() == "none":
        return None
    return int(value)


def float_or_none(value):
    if value.lower() == "none":
        return None
    return float(value)


def str_or_none(value):
    if value.lower() == "none":
        return None
    return str(value)


def tuple_of_ints(value):
    """
    Safely parse a Python tuple literal or a comma‐separated string of ints.
    Examples:
      "(70,70)"  → (70, 70)
      "70,70"    → (70, 70)
    """
    if value.lower() == "none":
        return None
    try:
        # first try a real Python literal
        t = ast.literal_eval(value)
        if isinstance(t, tuple) and all(isinstance(x, int) for x in t):
            return t
    except (ValueError, SyntaxError):
        pass
    # fallback: split on commas
    parts = value.strip("()[] ").split(",")
    return tuple(int(p) for p in parts)


def list_of_ints(value):
    """
    Safely parse a Python list literal or a comma-separated string of ints.

    Examples:
      "[70, 70]"  → [70, 70]
      "70, 70"    → [70, 70]
    """
    if value.lower() == "none":
        return None

    if isinstance(value, list):
        return [int(v) for v in value]

    try:
        # Try to parse as a Python literal
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [int(v) for v in parsed]
    except (ValueError, SyntaxError):
        pass  # Fall back to comma-splitting

    # Parse as a comma-separated string
    value = value.strip("()[] ")
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def read_args_from_file(log_dir, arg_str="all_args", return_dict=False):
    args_filename = str(log_dir / f"{arg_str}.json")

    with open(args_filename, "r") as f:
        args_dict = json.load(f)

    if return_dict:
        return args_dict

    all_args = argparse.Namespace()
    all_args.__dict__.update(args_dict)
    return all_args


def get_config_vals_of_interest(all_args):
    """
    Generally, these will be the flags (largely from train_fish.py) that
    we might want to vary between train and test.
    Typically, no need to add flags that would break model if changed
    (e.g. things that change obs_dim, like the number of rays)

    No need to be comprehensive since metadata is stored in the pkl file,
    this is just for quick reference and easy comparison generation.
    """
    episode_config = {
        "mormyromast_mode": all_args.mormyromast_mode,
        "ampullary_mode": all_args.ampullary_mode,
        "knollen_mode": all_args.knollen_mode,
        "knollen_processing": getattr(all_args, "knollen_processing", "binarize"),
        "sensing_model_type": getattr(all_args, "sensing_model_type", "dynamic"),
        "ampullary_intrinsic_only": getattr(all_args, "ampullary_intrinsic_only", True),
        "ampullary_ema": getattr(all_args, "ampullary_ema", False),
        "ampullary_alpha": getattr(all_args, "ampullary_alpha", None),
        "noise_frac_morm": getattr(all_args, "noise_frac_morm", 0.05),
        "noise_frac_amp": getattr(all_args, "noise_frac_amp", 0.05),
        "noise_frac_amp_cons_eod": getattr(all_args, "noise_frac_amp_cons_eod", 0.5),
        "noise_frac_knollen": getattr(all_args, "noise_frac_knollen", 0.05),
        "allow_aggression": all_args.allow_aggression,
        "collective_sensing_mode": all_args.collective_sensing_mode,
        "dist_perturbation": all_args.dist_perturbation,
        "agent_size_mode": all_args.agent_size_mode,
        "pfeeder": all_args.pfeeder,
        "prandom": all_args.prandom,
        "urandom": all_args.urandom,
        "prob_n_patch": all_args.prob_n_patch,
        "base_food_multiplier": all_args.base_food_multiplier,
        "eval_seed": all_args.seed,
    }
    # NOTE add more values at end as needed
    return episode_config


def log_exp_info(random_id, run_dir, all_args, log_dir):
    """
    Log the experiment info to a quick-reference csv file that associates
    pkl ids with the config values used for that run.
    """
    episode_config = get_config_vals_of_interest(all_args)
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = run_dir.split("/")[-1]

    exp_info_filename = log_dir / f"{run_dir_name}_exp_info.csv"
    if not os.path.exists(exp_info_filename):
        with open(exp_info_filename, "w") as f:
            f.write("pkl_id\t")
            f.write("run_timestamp\t")
            f.write("to_process\t")
            f.write("\t".join(episode_config.keys()))
            f.write("\n")
    with open(exp_info_filename, "a") as f:
        f.write(f"{random_id}\t")
        f.write(f"{current_time_str}\t")
        f.write("1\t")
        f.write("\t".join([str(val) for val in episode_config.values()]))
        f.write("\n")


def get_old_cfg_args(env_args):
    """
    Get the values from the env_args file that are also in the cfg file
    to ensure compatibility between env params and trained model.
    """
    all_env_params = {}
    agent_params = {
        k: v for k, v in env_args["agent_env_args"].items() if k in cfg.AGENT_PARAMS
    }
    env_params = {
        k: v for k, v in env_args["multi_agent_env_args"].items() if k in cfg.ENV_PARAMS
    }
    reward_params = env_args["multi_agent_env_args"]["reward_params"]
    object_types = env_args["multi_agent_env_args"]["OBJECT_TYPES"]

    # convert to numpy
    np_array_vars = ["monopole_positions_ego", "monopole_charges"]
    for var in np_array_vars:
        if var in agent_params:
            agent_params[var] = np.array(agent_params[var])

    all_env_params["AGENT_PARAMS"] = agent_params
    all_env_params["ENV_PARAMS"] = env_params
    all_env_params["REWARDS"] = reward_params
    all_env_params["OBJECT_TYPES"] = object_types

    if "FISH_CONSTANTS" in env_args["multi_agent_env_args"]:
        all_env_params["FISH_CONSTANTS"] = env_args["multi_agent_env_args"][
            "FISH_CONSTANTS"
        ]

    return all_env_params


def update_all_args(all_args, eval_args):
    """
    Update all_args with the values from eval_args (if not None).
    """
    all_args.save_vids  = eval_args.save_vids
    all_args.save_rnn   = eval_args.save_rnn
    all_args.save_obs   = eval_args.save_obs
    all_args.save_attn  = eval_args.save_attn
    all_args.episode_length = eval_args.eval_episode_length
    all_args.max_episode_length = eval_args.eval_episode_length
    all_args.render_episodes = eval_args.eval_render_episodes  # = num_eval_rollouts: controls how many rollouts are recorded as pkls
    all_args.n_eval_rollout_threads = eval_args.n_rollout_threads
    all_args.n_rollout_threads = eval_args.n_rollout_threads
    all_args.num_vids_to_save = eval_args.num_vids_to_save
    all_args.num_render_envs = eval_args.num_render_envs

    if eval_args.eval_collective_sensing_mode is not None:
        all_args.collective_sensing_mode = eval_args.eval_collective_sensing_mode
    if eval_args.eval_food_replenish is not None:
        all_args.food_replenish = eval_args.eval_food_replenish
    if eval_args.eval_knollen_mode is not None:
        all_args.knollen_mode = eval_args.eval_knollen_mode
    if eval_args.eval_dist_perturbation is not None:
        all_args.dist_perturbation = eval_args.eval_dist_perturbation
    if eval_args.eval_allow_aggression is not None:
        # handle case where model was trained with biting but we want to evaluate without
        if not eval_args.eval_allow_aggression and all_args.allow_aggression:
            all_args.enable_bite_action = True
        all_args.allow_aggression = eval_args.eval_allow_aggression
    if eval_args.eval_prandom is not None:
        all_args.prandom = eval_args.eval_prandom
    if eval_args.eval_urandom is not None:
        all_args.urandom = eval_args.eval_urandom
    if eval_args.eval_pfeeder is not None:
        all_args.pfeeder = eval_args.eval_pfeeder
    if eval_args.eval_prob_n_patch is not None:
        all_args.prob_n_patch = eval_args.eval_prob_n_patch
    if eval_args.eval_agent_size_mode is not None:
        all_args.agent_size_mode = eval_args.eval_agent_size_mode
    if eval_args.eval_ampullary_mode is not None:
        all_args.ampullary_mode = eval_args.eval_ampullary_mode
    if eval_args.eval_food_drift is not None:
        all_args.food_drift = eval_args.eval_food_drift
    if eval_args.eval_food_drag is not None:
        all_args.food_drag = eval_args.eval_food_drag
    if eval_args.eval_food_orientation_drift is not None:
        all_args.food_orientation_drift = eval_args.eval_food_orientation_drift
    if eval_args.eval_run_name is not None:
        all_args.run_name = eval_args.eval_run_name
    if eval_args.eval_mormyromast_mode is not None:
        all_args.mormyromast_mode = eval_args.eval_mormyromast_mode
    if eval_args.eval_active_agent_ids is not None:
        all_args.active_agent_ids = eval_args.eval_active_agent_ids

    if eval_args.eval_seed is not None:
        all_args.seed = eval_args.eval_seed

    # cfg overrides (these vars are set using cfg.py and require a different approach)
    if eval_args.eval_indiv_sensing_radius is not None:
        all_args.cfg_override["AGENT_PARAMS"][
            "indiv_sensing_radius"
        ] = eval_args.eval_indiv_sensing_radius
    if eval_args.eval_arena_size_max is not None:
        all_args.cfg_override["ENV_PARAMS"][
            "arena_size_max_cm"
        ] = eval_args.eval_arena_size_max
    if eval_args.eval_arena_size_min is not None:
        all_args.cfg_override["ENV_PARAMS"][
            "arena_size_min_cm"
        ] = eval_args.eval_arena_size_min

    if eval_args.eval_base_food_multiplier is not None:
        all_args.base_food_multiplier = eval_args.eval_base_food_multiplier

    if eval_args.task is not None:
        all_args.task = eval_args.task

    if eval_args.eval_mute_k is not None:
        all_args.mute_k = eval_args.eval_mute_k
    if eval_args.eval_num_active_agents is not None:
        all_args.num_active_agents = eval_args.eval_num_active_agents
    if eval_args.eval_rw_eod_rate is not None:
        all_args.rw_eod_rate = eval_args.eval_rw_eod_rate
    if eval_args.eval_rw_freeze is not None:
        all_args.rw_freeze = eval_args.eval_rw_freeze
    if eval_args.eval_agent_size_sampling_mode is not None:
        all_args.agent_size_sampling_mode = eval_args.eval_agent_size_sampling_mode
    if eval_args.eval_noise_frac_morm is not None:
        all_args.noise_frac_morm = eval_args.eval_noise_frac_morm
    if eval_args.eval_noise_frac_amp is not None:
        all_args.noise_frac_amp = eval_args.eval_noise_frac_amp
    if eval_args.eval_noise_frac_amp_cons_eod is not None:
        all_args.noise_frac_amp_cons_eod = eval_args.eval_noise_frac_amp_cons_eod
    if eval_args.eval_noise_frac_knollen is not None:
        all_args.noise_frac_knollen = eval_args.eval_noise_frac_knollen
    if eval_args.eval_ampullary_ema is not None:
        all_args.ampullary_ema = bool(eval_args.eval_ampullary_ema)
    if eval_args.eval_ampullary_alpha is not None:
        all_args.ampullary_alpha = eval_args.eval_ampullary_alpha
    if eval_args.eval_fixed_num_patches is not None:
        all_args.fixed_num_patches = eval_args.eval_fixed_num_patches
        if all_args.task not in ("n_patch", "n_patch_fixed"):
            print(
                "[WARN] fixed_num_patches is set, but task is not 'n_patch'. "
                "Setting task to 'n_patch' automatically."
            )
            all_args.task = "n_patch"

    # override all args' ifi with video fps from cfg
    all_args.ifi = 1.0 / cfg.ENV_PARAMS["fps_video"]

    return all_args

def set_system_wide_seeds(all_args):
    random.seed(all_args.seed)
    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(all_args.seed)

def add_grid_to_all_args(all_args):
    if all_args.task != "2f1p_grid":
        return all_args

    size_A = size_B = [0.1, 0.3, 0.5, 0.7, 0.9]
    start_radius_cm_B = [20, 40, 60]
    start_theta_B = [np.pi/4 * i for i in range(8)]  # 8 thetas around start radius circle: 0, 45, ..., 315 degrees
    outer_product_iterator = itertools.product(size_A, size_B, start_radius_cm_B, start_theta_B)
    grid = list(outer_product_iterator)
    
    # split list into all_args.num_eval_rollout_threads sublists to be used for each eval episode
    grid = [grid[i::all_args.n_eval_rollout_threads] for i in range(all_args.n_eval_rollout_threads)]

    # all_args is argparse.Namespace; store under its backing dict so the
    # downstream env code can read key "2f1p_grid" from vars(all_args).
    all_args.__dict__["2f1p_grid"] = grid
    return all_args



def get_runner(eval_args, episode_configs=None):
    run_dir = Path(eval_args.run_dir)
    log_dir = run_dir / "logs"

    all_args = read_args_from_file(log_dir)
    env_args = read_args_from_file(log_dir, "env_args", return_dict=True)
    old_cfg_args = get_old_cfg_args(env_args)
    all_args.cfg_override = old_cfg_args
    update_all_args(all_args, eval_args)

    if episode_configs is not None:
        n = all_args.n_rollout_threads
        task_key = all_args.task  # store under task name so MAEFish can find it
        all_args.__dict__[task_key] = [episode_configs[i::n] for i in range(n)]
        # Each thread gets ceil(len/n) episodes; render them all.
        all_args.render_episodes = len(all_args.__dict__[task_key][0])
    elif all_args.task == "2f1p_grid":
        all_args = add_grid_to_all_args(all_args)

    set_system_wide_seeds(all_args)
    envs = make_env(all_args, eval=True)
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": all_args.num_agents,
        "device": setup_device(all_args),
        "run_dir": run_dir,
        "write_args_to_file": False,
        "output_subdir": getattr(eval_args, "output_subdir", "outputs"),
    }

    Runner = SharedRunner if all_args.share_policy else SeparatedRunner
    if isinstance(Runner, SeparatedRunner):
        raise NotImplementedError

    runner = Runner(config)
    return runner, all_args, log_dir


def main(eval_args, episode_configs=None):
    """
    Run evaluation for a single configuration. (Run multiple times for multiple configurations.)

    Argument priority:
        1. eval_args passed into script (highest priority; overrides if not None)
        2. all_args from file
        3. env_args from file (used to override the current cfg file)
        4. values in current cfg.py file (might not match with cfg file used for training
           but helps with backwards compatibility if some attributes are added later)

    episode_configs: optional flat list of per-episode init params. If provided,
        eval_fish distributes them across n_rollout_threads. Currently used by
        2f1p_grid; None means the env initializes stochastically each episode.
    """
    runner, all_args, log_dir = get_runner(eval_args, episode_configs=episode_configs)
    runner.render()

    log_exp_info(runner.all_args.run_name, eval_args.run_dir, all_args, log_dir)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        type=str,
        help="Experiment directory, where logs/, models/ and outputs/ are subdirectories",
    )
    parser.add_argument(
        "--eval_episode_length", type=int, default=50, help="Evaluation episode length"
    )
    parser.add_argument(
        "--save_vids",
        action="store_true",
        default=False,
        help="Whether to save GIFs for all eval episodes",
    )
    parser.add_argument(
        "--save_rnn",
        action="store_true",
        default=False,
        help="Write ep{k}_rnn.npy per episode; only needed for rnn_* analyses",
    )
    parser.add_argument(
        "--save_obs",
        action="store_true",
        default=False,
        help="Write ep{k}_obs.npy per episode; only needed for decoding analyses",
    )
    parser.add_argument(
        "--save_attn",
        action="store_true",
        default=False,
        help="Write ep{k}_attn.npy per episode; only needed for attention analysis",
    )
    parser.add_argument(
        "--n_rollout_threads", type=int, default=1, help="Number of rollout threads"
    )
    parser.add_argument(
        "--eval_render_episodes",
        type=int,
        default=1,
        help="Number of eval rollouts to record as pkls (and optionally as video). Misleading name: implies num_eval_rollouts.",
    )
    parser.add_argument(
        "--eval_seed",
        type=int_or_none,
        default=None,
        help="Seed for evaluation. If None, will use the seed from the training run.",
    )
    parser.add_argument("--eval_collective_sensing_mode", type=int_or_none, default=None)
    parser.add_argument("--eval_food_replenish", type=float_or_none, default=None)  # TODO deprecate
    parser.add_argument("--eval_dist_perturbation", type=float_or_none, default=None)
    parser.add_argument(
        "--eval_allow_aggression",
        type=int_or_none,
        default=None,
        help="Controls whether biting is allowed during evaluation. "
        "If the model was trained with aggression but is evaluated "
        "without aggression, enable_bite_action is set to False to "
        "preserve observation space but prevent biting.",
    )  # can pass in 0, 1, or "None"
    parser.add_argument("--eval_pfeeder", type=float_or_none, default=None)
    parser.add_argument("--eval_prandom", type=float_or_none, default=None)
    parser.add_argument("--eval_urandom", type=float_or_none, default=None)
    parser.add_argument("--eval_prob_n_patch", type=float_or_none, default=None)
    parser.add_argument("--eval_agent_size_mode", type=str_or_none, default=None)
    parser.add_argument("--eval_mormyromast_mode", type=int_or_none, default=None)
    parser.add_argument("--eval_ampullary_mode", type=int_or_none, default=None)
    parser.add_argument("--eval_knollen_mode", type=int_or_none, default=None)
    parser.add_argument("--eval_indiv_sensing_radius", type=int_or_none, default=None)  # TODO deprecate
    parser.add_argument("--eval_food_drift", type=float_or_none, default=None)
    parser.add_argument("--eval_food_drag", type=float_or_none, default=None)
    parser.add_argument("--eval_food_orientation_drift", type=float_or_none, default=None)
    parser.add_argument("--eval_base_food_multiplier", type=float_or_none, default=None)
    parser.add_argument("--task", type=str, default="foraging")
    parser.add_argument(
        "--eval_mute_k", type=int_or_none, default=None
    )  # Note first-k agents not random-k are muted
    parser.add_argument(
        "--eval_num_active_agents",
        type=int_or_none,
        default=None,
        help="Sets the number of active agents during evaluation. If None, defaults to the training configuration.",
    )
    parser.add_argument(
        "--eval_run_name",
        type=str,
        default=None,
        help="Name of the eval run, ideally a short unique identifier",
    )
    parser.add_argument("--num_vids_to_save", type=int, default=1)
    parser.add_argument("--num_render_envs", type=int, default=1,
        help="Number of envs to render video for (default 1; clamped to n_rollout_threads)")
    parser.add_argument("--eval_rw_eod_rate", type=float_or_none, default=None)
    parser.add_argument("--eval_rw_freeze", type=int_or_none, default=None,
        help="Whether to freeze the RW movement (not EOD) during evaluation (0 for no freeze, 1 for freeze)") 
    parser.add_argument(
        "--eval_arena_size_max",
        type=tuple_of_ints,
        default=None,
        help="Override ENV_PARAMS['arena_size_max'],"
        "pass as tuple, e.g. 70,70 or '(70,70)'",
    )
    parser.add_argument(
        "--eval_arena_size_min",
        type=tuple_of_ints,
        default=None,
        help="Override ENV_PARAMS['arena_size_min'],"
        "pass as tuple, e.g. 70,70 or '(70,70)'",
    )
    parser.add_argument(
        "--eval_active_agent_ids",
        type=list_of_ints,
        default=None,
        help="List of agent ids to evaluate, e.g. 0,1 or [0,1]",
    )
    parser.add_argument(
        "--eval_fixed_num_patches",
        type=int_or_none,
        default=None,
        help="If set, uses NPatchArena to fix the number of patches (default: 1 when unset)."
    )
    parser.add_argument(
        "--eval_agent_size_sampling_mode",
        type=str,
        default=None,
        choices=["uniform", "grid"],
        help="Method for sampling agent size: uniform random or from a fixed grid of values.",
    )
    parser.add_argument("--eval_noise_frac_morm", type=float_or_none, default=None)
    parser.add_argument("--eval_noise_frac_amp", type=float_or_none, default=None)
    parser.add_argument("--eval_noise_frac_amp_cons_eod", type=float_or_none, default=None)
    parser.add_argument("--eval_noise_frac_knollen", type=float_or_none, default=None)
    parser.add_argument("--eval_ampullary_ema", type=int_or_none, default=None,
        help="Whether to apply EMA to ampullary readings during evaluation (0 for no EMA, 1 for EMA)")
    parser.add_argument("--eval_ampullary_alpha", type=float_or_none, default=None,
        help="Alpha value for EMA of ampullary readings during evaluation (ignored if eval_ampullary_ema is not set or is 0)")


    args = parser.parse_args(args)
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
