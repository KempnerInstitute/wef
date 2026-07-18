import sys
import os
import socket
from datetime import datetime
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
import cProfile
import pstats

# from MAFish import MultiAgentFishEnv
from MAEFish import MultiAgentFishEnv
from fish_runner_shared import MAFishRunner as SharedRunner
from fish_runner_separated import MAFishRunner as SeparatedRunner
from cfg import ENV_PARAMS
from pprint import pprint
import io
from utils_logging import log_on_done


def make_env(all_args, eval=False):
    def get_env_fn(rank):
        def init_env():
            env_seed = all_args.seed * 50000 + rank * 10000 if eval else all_args.seed + rank * 1000
            env_args = vars(all_args).copy()
            env_args["env_rank"] = rank
            if not eval:
                env = MultiAgentFishEnv(
                    env_args,
                    seed=env_seed,
                    done_callback=log_on_done,
                    is_eval=False,
                )
            else:
                env = MultiAgentFishEnv(
                    env_args,
                    seed=env_seed,
                    is_eval=True
                    )

            env.seed(env_seed)
            if all_args.homing_mode:
                env.homing_mode = True

            return env

        return init_env

    num_threads = (
        all_args.n_eval_rollout_threads if eval else all_args.n_rollout_threads
    )
    print(f"Making {num_threads} {'Eval' if eval else 'Train'} environments...")
    if num_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    return SubprocVecEnv([get_env_fn(i) for i in range(num_threads)])


def parse_args(args, parser):
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--num_patches", type=int, default=10)
    parser.add_argument("--render_mode", type=str, default=None)
    parser.add_argument("--shared_reward", action="store_true", default=False)
    parser.add_argument(
        "--penalize_effort_over_frac",
        type=float,
        default=1.0,
        help="Apply effort penalty to movement/turning over this fraction of command range (0.0-1.0).",
    )
    # 0: can only use self-EOD, 1: can use self-EOD and cons-EOD, 2: can only use cons-EOD
    parser.add_argument("--collective_sensing_mode", type=int, default=1)
    # Food/Arena params
    parser.add_argument("--food_radius", type=float, default=0.1)
    parser.add_argument("--base_food_multiplier", type=float, default=1.0)
    parser.add_argument("--rnn_type", type=str, default="Vanilla", choices=["Vanilla", "GRU"])

    # Non-food
    parser.add_argument(
        "--knollen_mode",
        type=int,
        default=1,
        help="0: knollen off, 1: knollen on",
    )
    parser.add_argument(
        "--knollen_processing",
        type=str,
        default="binarize",
        choices=["binarize", "log"],
        help="knollen signal processing when knollen_mode=1: 'binarize' (binary detection) or 'log' (sign-log continuous)",
    )
    parser.add_argument("--max_food_eaten_per_step", type=int, default=1)
    parser.add_argument(
        "--allow_aggression",
        type=int,
        default=0,
        help="0: aggression/biting off, 1: aggression/biting on",
    )
    # parser.add_argument("--feedback_action", action="store_true", default=False)
    parser.add_argument("--feedback_displacement", action="store_true", default=False)
    parser.add_argument(
        "--pfeeder", type=float, default=0
    )  # proportion of patchy-feeder arena
    parser.add_argument(
        "--prandom", type=float, default=2/3
    )  # proportion of patchy-random arena
    parser.add_argument(
        "--urandom", type=float, default=1/3
    )  # proportion of uniform-random arena
    # proportion of n-patch arena... different naming convention for clarity
    parser.add_argument("--prob_n_patch", type=float, default=0)

    parser.add_argument("--dist_perturbation", type=float, default=0.0)
    parser.add_argument("--save_vid", action="store_true", default=False)
    parser.add_argument(
        "--agent_size_mode", type=str, default=None
    )  # None, "hierarchy", "random"
    parser.add_argument("--use_bite_cooldown", action="store_true", default=False)
    parser.add_argument("--eat_cooldown_rate", type=float, default=None)
    parser.add_argument("--food_drift", type=float, default=0.0)
    parser.add_argument("--food_drag", type=float, default=0.0)
    parser.add_argument("--food_orientation_drift", type=float, default=0.0)
    parser.add_argument(
        "--ampullary_mode",
        type=int,
        default=1,
        help="0: ampullary observations off, 1: ampullary observations on",
    )
    parser.add_argument("--backwards", action="store_true", default=False)
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="timestamp; if None, use current time",
    )
    parser.add_argument("--indiv_sensing_radius", type=float, default=10.0)
    parser.add_argument("--food_sensing_radius", type=float, default=4.0)
    parser.add_argument("--max_food_sensing_radius", type=float, default=15.)
    parser.add_argument("--knollen_binarize_threshold", type=float, default=None)
    parser.add_argument("--max_episode_length", type=int, default=1200)
    parser.add_argument("--homing_distance", type=float, default=4.0)
    parser.add_argument("--required_homing_steps", type=int, default=10)
    parser.add_argument(
        "--homing_mode",
        action="store_true",
        default=False,
        help="If True, the fish agents will be in homing mode",
    )
    parser.add_argument(
        "--always_freeze_agent0",
        action="store_true",
        default=False,
        help="Freeze agent 0 (target) during training as well as eval; gives it zero reward.",
    )
    parser.add_argument(
        "--mormyromast_mode",
        type=int,
        default=1,
        help="0: mormyromast observations off, 1: mormyromast observations on",
    )
    parser.add_argument(
        "--sensing_model_type",
        type=str,
        default="dynamic",
        choices=["frac", "dynamic"],
        help="frac: fractionated sensing, dynamic: self-EOD/cons-EOD baseline subtraction",
    )
    parser.add_argument(
        "--morm_selfimage_mode",
        type=int,
        default=1,
        help="0: self-image observations off, 1: self-image observations on",
    )
    parser.add_argument(
        "--morm_consimage_mode",
        type=int,
        default=1,
        help="0: cons-image observations off, 1: cons-image observations on",
    )
    parser.add_argument(
        "--knollen_metadata_mode",
        type=str,
        default="relative",
        choices=["absolute", "relative"],
        help="How other agent size is encoded in knollen metadata",
    )
    parser.add_argument(
        "--train_sensor_dropout_p",
        type=float,
        default=0.0,
        help="Prob of ablating sensors during training (default: 0.0, i.e., no ablation)",
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default=None,
        choices=["x", "hx", "x+hx", None],
        help="Attention input mode over observations: x, hx, x+hx, or None to disable.",
    )
    parser.add_argument(
        "--attn_use_softmax",
        action="store_true",
        default=False,
        help="Use softmax for attention gating instead of sigmoid",
    )
    parser.add_argument(
        "--num_vids_to_save",
        type=int,
        default=1,
        help="Number of vids to save during training",
    )
    parser.add_argument(
        "--num_render_envs",
        type=int,
        default=1,
        help="Number of envs to render video for (default 1; clamped to n_rollout_threads)",
    )
    parser.add_argument(
        "--save_svg_snapshots",
        action="store_true",
        default=False,
        help="If True, save SVG snapshots during evaluation",
    )
    parser.add_argument("--train_food_scaling_min", type=float, default=0.25)
    parser.add_argument("--train_food_scaling_max", type=float, default=1.0)
    parser.add_argument(
        "--train_food_scaling_type",
        type=str,
        default="log_uniform",
        choices=["uniform", "log_uniform"],
        help="Type of food scaling during training: linear or log",
    )
    parser.add_argument(
        "--motion_order",
        type=int,
        default=1,
        choices=[1, 2],
        help="1st or 2nd order dynamics for the fish agents",
    )
    parser.add_argument("--results_parent_dir", type=str, default=None)
    parser.add_argument(
        "--mute_k",
        type=int,
        default=0,
        help="Number of agents to silence: randomly selected during training, first-K during evaluation.",
    )
    parser.add_argument(
        "--num_active_agents",
        type=int,
        default=None,
        help="Maximum number of agents that can be active during evaluation (default: None, i.e., no limit).",
    )
    parser.add_argument(
        "--p_init_closeby",
        type=float,
        default=0.0,
        help="Probability of initializing agents close to each other (default: 0.0, i.e., no closeby initialization).",
    )
    parser.add_argument(
        "--ampullary_ema",
        action="store_true",
        default=False,
        help="Use EMA filtering on ampullary sensors.",
    )
    parser.add_argument(
        "--ampullary_intrinsic_only",
        action="store_true",
        default=True,
        help="If True, ampullary sensors only sense intrinsic charges, not EODs/induced charges",
    )
    parser.add_argument(
        "--auxs",
        nargs="*",
        choices=["eods", "spis", "observations", "energy", "attentions", "center_fields"],
        help="Auxiliary subplots to render (omit to keep current default).",
        default=None,   # <-- None means USE FUNCTION DEFAULTS
    )
    parser.add_argument(
        "--fixed_num_patches",
        type=int,
        default=None,
        help="Number of patches for fixed-patch environments (default: 1 when unset).",
    )
    parser.add_argument("--multiplier_linear", type=float, default=1.0,
                        help="Multiplier for linear dynamics (e.g., speed) of the fish agents.")
    parser.add_argument("--multiplier_angular", type=float, default=1.0,
                        help="Multiplier for angular dynamics (e.g., turning rate) of the fish agents.")
    parser.add_argument(
        "--size_speed_exponent",
        type=float,
        default=1.0,
        help="Exponent for size-based movement scaling applied as (1 + agent_size) ** exponent.",
    )
    parser.add_argument(
        "--noise_frac_amp",
        type=float,
        default=0.05,
        help="Multiplicative uniform noise fraction for ampullary sensors",
    )
    parser.add_argument(
        "--noise_frac_amp_cons_eod",
        type=float,
        default=0.5,
        help="Multiplicative uniform ampullary noise fraction when any conspecific EOD is emitting",
    )
    parser.add_argument(
        "--amp_cons_eod_sign_only",
        action="store_true",
        default=False,
        help="When cons-EOD fires, replace amp reading with sign(amp) instead of adding multiplicative noise",
    )
    parser.add_argument(
        "--noise_frac_morm",
        type=float,
        default=0.05,
        help="Multiplicative uniform noise fraction for mormyromast sensors",
    )
    parser.add_argument(
        "--noise_frac_knollen",
        type=float,
        default=0.05,
        help="Multiplicative uniform noise fraction for knollen sensors",
    )
    # prob_eating
    parser.add_argument(
        "--probabilistic_eating",
        action="store_true",
        default=False,
        help="If True, eating food is probabilistic based on velocity in 2nd order motion",
    )
    parser.add_argument(
        "--use_curriculum",
        action="store_true",
        default=False,
        help="If True, use a curriculum during training",
    )
    parser.add_argument(
        "--curriculum_early_end_frac",
        type=float,
        default=0.9,
        help="Fraction of training after which to stop the curriculum (default: 0.9)",
    )
    parser.add_argument(
        "--r_timestep",
        type=float,
        default=None,
        help="Reward for timestep penalty (per step)",
    )
    parser.add_argument(
        "--r_eat",
        "--eat_reward",
        dest="r_eat",
        type=float,
        default=None,
        help="Reward for eating a unit of food",
    )
    parser.add_argument(
        "--r_proximity_shaping",
        type=float,
        default=None,
        help="Reward for getting closer to food",
    )
    parser.add_argument(
        "--r_bitten",
        "--bitten_reward",
        dest="r_bitten",
        type=float,
        default=None,
        help="Reward for being bitten (scaled by size difference when agent_size_mode=True)",
    )
    parser.add_argument(
        "--r_bite",
        type=float,
        default=None,
        help="Reward for biting (discourage excessive biting actions)",
    )
    parser.add_argument(
        "--r_collision",
        type=float,
        default=None,
        help="Reward/penalty for collisions",
    )
    parser.add_argument(
        "--r_effort_over",
        type=float,
        default=None,
        help="Reward for movement/turning effort over threshold",
    )
    parser.add_argument(
        "--r_homing",
        type=float,
        default=None,
        help="Reward for homing success",
    )
    parser.add_argument(
        "--r_homing_shaping",
        type=float,
        default=None,
        help="Shaping reward for moving toward homing target",
    )
    parser.add_argument(
        "--r_homing_time_penalty",
        type=float,
        default=None,
        help="Penalty for homing time",
    )
    parser.add_argument(
        "--r_wall_proximity",
        type=float,
        default=None,
        help="Penalty for being close to a wall",
    )
    parser.add_argument(
        "--log_tb",
        action="store_true",
        default=False,
        help="Enable TensorBoard logging (disabled by default).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Enable experiment profiling (disabled by default).",
    )
    parser.add_argument(
        "--agent_size_sampling_mode",
        type=str,
        default="uniform",
        choices=["uniform", "grid"],
        help="Method for sampling agent size: uniform random or from a fixed grid of values.",
    )


    all_args = parser.parse_known_args(args)[0]

    # Override num_agents if in homing mode
    if all_args.homing_mode:
        print("Homing mode detected: Setting num_agents to 2")
        all_args.num_agents = 2

    if all_args.max_food_sensing_radius < 0:
        all_args.max_food_sensing_radius = None

    # override all args ifi with the cfg's fps
    all_args.ifi = 1.0 / ENV_PARAMS["fps_video"]

    return all_args


def update_cfg_args(all_args):
    all_args.cfg_override = {
        "AGENT_PARAMS": {},
        "ENV_PARAMS": {},
        "REWARDS": {},
        "OBJECT_TYPES": {},
    }
    if all_args.indiv_sensing_radius is not None:
        all_args.cfg_override["AGENT_PARAMS"][
            "indiv_sensing_radius"
        ] = all_args.indiv_sensing_radius
    if all_args.food_sensing_radius is not None:
        all_args.cfg_override["AGENT_PARAMS"][
            "food_sensing_radius"
        ] = all_args.food_sensing_radius
    if all_args.knollen_binarize_threshold is not None:
        all_args.cfg_override["AGENT_PARAMS"][
            "knollen_binarize_threshold"
        ] = all_args.knollen_binarize_threshold
    if all_args.r_timestep is not None:
        all_args.cfg_override["REWARDS"]["timestep"] = all_args.r_timestep
    if all_args.r_eat is not None:
        all_args.cfg_override["REWARDS"]["eat"] = all_args.r_eat
    if all_args.r_proximity_shaping is not None:
        all_args.cfg_override["REWARDS"]["proximity_shaping"] = (
            all_args.r_proximity_shaping
        )
    if all_args.r_bitten is not None:
        all_args.cfg_override["REWARDS"]["bitten"] = all_args.r_bitten
    if all_args.r_bite is not None:
        all_args.cfg_override["REWARDS"]["bite"] = all_args.r_bite
    if all_args.r_collision is not None:
        all_args.cfg_override["REWARDS"]["collision"] = all_args.r_collision
    if all_args.r_effort_over is not None:
        all_args.cfg_override["REWARDS"]["effort_over"] = all_args.r_effort_over
    if all_args.r_homing is not None:
        all_args.cfg_override["REWARDS"]["homing"] = all_args.r_homing
    if all_args.r_homing_shaping is not None:
        all_args.cfg_override["REWARDS"]["homing_shaping"] = (
            all_args.r_homing_shaping
        )
    if all_args.r_homing_time_penalty is not None:
        all_args.cfg_override["REWARDS"]["homing_time_penalty"] = (
            all_args.r_homing_time_penalty
        )
    if all_args.r_wall_proximity is not None:
        all_args.cfg_override["REWARDS"]["wall_proximity"] = (
            all_args.r_wall_proximity
        )
    return all_args


def setup_algorithm(all_args):
    if all_args.algorithm_name == "rmappo":
        print("Using rmappo, setting use_recurrent_policy to True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print(
            "Using mappo, setting use_recurrent_policy & use_naive_recurrent_policy to False"
        )
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("Using ippo, setting use_centralized_V to False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError


def setup_device(all_args):
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device:", device)
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    # elif all_args.cuda and torch.mps.is_available():
    elif all_args.cuda and hasattr(torch, "mps") and torch.mps.is_available():
        device = torch.device("mps")
        print("Using device:", device)
        torch.set_num_threads(all_args.n_training_threads)
    else:
        device = torch.device("cpu")
        print("Using device:", device, "Threads:", all_args.n_training_threads)
        torch.set_num_threads(all_args.n_training_threads)
    return device


def setup_run_dir(all_args, results_parent_dir=None):
    if results_parent_dir is None:
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        run_dir = (
            Path(CURRENT_DIR)
            / "results"
            / all_args.experiment_name
        )
    else:
        run_dir = (
            Path(results_parent_dir)
            / "results"
            / all_args.experiment_name
        )
    print("run_dir:", run_dir)
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    curr_run = all_args.timestamp

    run_dir = run_dir / curr_run  # append
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    all_args.run_dir = str(run_dir)
    return run_dir


def main(args):
    import time

    start_time = time.time()
    parser = get_config()
    all_args = parse_args(args, parser)

    # OVERRIDES
    all_args.env_name = "MultiAgentFishEnv"
    all_args.algorithm_name = ["rmappo", "ippo"][0]
    all_args.feedback_action = True
    all_args.share_policy = True

    if all_args.timestamp is None:
        all_args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_args = update_cfg_args(all_args)

    print("shared_reward", all_args.shared_reward)
    print("all_args:", all_args)

    # Set up the algorithm etc.
    setup_algorithm(all_args)
    device = setup_device(all_args)
    run_dir = setup_run_dir(all_args, results_parent_dir=all_args.results_parent_dir)

    setproctitle.setproctitle(
        f"{all_args.experiment_name}@{all_args.user_name}"
    )

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # Setup the environment
    envs = make_env(all_args, eval=False)
    eval_envs = make_env(all_args, eval=True) if all_args.use_eval else None
    num_agents = all_args.num_agents

    _agent0 = envs.envs[0].agent_objects[0]
    _obs_dim = envs.observation_space[0].shape[0]
    _share_obs_dim = envs.share_observation_space[0].shape[0]
    _act_dim = envs.action_space[0].shape[0]
    print(
        f"Dims — obs: {_obs_dim} (morm={_agent0.num_mormyromast_sensors_virtual},"
        f" amp={_agent0.num_ampullary_sensors},"
        f" knollen={_agent0.num_knollen_sensors}x{num_agents-1}={_agent0.num_knollen_sensors*(num_agents-1)}),"
        f" share_obs: {_share_obs_dim}, act: {_act_dim}"
    )

    # S3: store sensor slice boundaries so downstream code doesn't need hardcoded index arithmetic
    all_args.obs_dim_morm = _agent0.num_mormyromast_sensors_virtual
    all_args.obs_dim_amp = _agent0.num_ampullary_sensors
    all_args.obs_dim_knollen = _agent0.num_knollen_sensors * (num_agents - 1)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    Runner = SharedRunner if all_args.share_policy else SeparatedRunner
    runner = Runner(config)

    # Create a profiler object
    if all_args.profile:
        pr = cProfile.Profile()
        pr.enable()

    runner.run()

    if all_args.profile:
        pr.disable()
        profile_filename = runner.log_dir + "/train_fish.prof"
        pstats.Stats(pr).sort_stats("cumtime").dump_stats(profile_filename)
        print(f"View profile using: python -m pstats -s cumtime {profile_filename}")
    

    print("Done training, now rendering....")
    # Only switch to homing mode if eval_homing is True
    if all_args.homing_mode:
        print("Switching to homing mode for evaluation...")
        for remote in envs.remotes:
            # TODO: state why this is explicitly needed (doesn't just use the eval_envs?)
            remote.send(("set_attr", ("is_eval", True)))
        _ = [remote.recv() for remote in envs.remotes]
    runner.render()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if runner.writter is not None:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()

    end_time = time.time()
    print(
        f"------------Total training time: {end_time - start_time:.2f} seconds------------"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
