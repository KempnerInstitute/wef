import sys, os; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np

from MAEFish import MultiAgentFishEnv

def _make_args():
    return {
        "arena_size_min": (50, 50),
        "arena_size_max": (50, 50),
        "num_agents": 3,
        "energy_food": 20,
        "render_mode": None,
        "food_drift": 0.01,
        "food_drag": 0.2,
        "collective_sensing_mode": 1,
        "timestamp": "seed_test",
        "max_food_eaten_per_step": 1,
        "feedback_action": True,
        "feedback_displacement": True,
        "pfeeder": 0,
        "prandom": 1,
        "urandom": 0,
        "prob_n_patch": 0,
        "save_vid": False,
        "allow_aggression": 1,
        "knollen_mode": 1,
        "ampullary_mode": 1,
        "mormyromast_mode": 1,
        "morm_selfimage_mode": 1,
        "morm_consimage_mode": 1,
        "sensing_model_type": "frac",
        "backwards": True,
        "max_episode_length": 120,
        "homing_distance": 4.0,
        "required_homing_steps": 20,
        "homing_mode": False,
        "is_eval": False,
        "rw_eod_rate": 0.0,
        "rw_freeze": 0,
        "task": "foraging",
        "p_init_closeby": 0.0,
        "ampullary_ema": True,
        "max_food_sensing_radius": 15,
        "base_food_multiplier": 1.0,
        "auxs": None,
        "multiplier_linear": 2.0,
        "multiplier_angular": 3.0,
    }


ARENA_CASES = {
    "uniform": {"pfeeder": 0, "prandom": 0, "urandom": 1, "prob_n_patch": 0},
    "patchy": {"pfeeder": 0, "prandom": 1, "urandom": 0, "prob_n_patch": 0},
    "npatch": {"pfeeder": 0, "prandom": 0, "urandom": 0, "prob_n_patch": 1},
    "feeder": {"pfeeder": 1, "prandom": 0, "urandom": 0, "prob_n_patch": 0},
}

def _rollout_snapshot(seed, arena_case, steps=100, extra_args=None):
    args = _make_args()
    args.update(ARENA_CASES[arena_case])
    if extra_args:
        args.update(extra_args)

    env = MultiAgentFishEnv(args, seed=seed)
    env.seed(seed)
    env.reset()
    actions = np.zeros((env.num_agents, 4), dtype=float) # ZERO ACTIONS
    snapshot = []

    for _ in range(steps):
        num_to_eat = min(1, env.arena.num_food_pellets)
        if num_to_eat > 0:
            env.arena.eat_food(*range(num_to_eat))

        snapshot.append(
            (
                env.arena.food_positions.copy(),
                env.arena.food_orientation_angles.copy(),
                np.stack([agent.position.copy() for agent in env.agent_objects]),
                np.array([agent.orientation for agent in env.agent_objects]),
            )
        )
        env.step(actions)
    return snapshot


def _snapshots_equal(a, b):
    if len(a) != len(b):
        return False
    for ai, bi in zip(a, b):
        for a_arr, b_arr in zip(ai, bi):
            if a_arr.shape != b_arr.shape:
                return False
            if not np.allclose(a_arr, b_arr, equal_nan=True):
                return False
    return True


def test_arena_reproducible_for_same_seed(arena_case):
    snapshot_a = _rollout_snapshot(seed=1234, arena_case=arena_case)
    snapshot_b = _rollout_snapshot(seed=1234, arena_case=arena_case)
    assert _snapshots_equal(snapshot_a, snapshot_b)
    print(f">[{arena_case}] snapshots are equal for the same seed.")


def test_arena_changes_for_different_seed(arena_case):
    snapshot_a = _rollout_snapshot(seed=1234, arena_case=arena_case)
    snapshot_b = _rollout_snapshot(seed=1235, arena_case=arena_case)
    assert not _snapshots_equal(snapshot_a, snapshot_b)
    print(f">[{arena_case}] snapshots are different for different seeds.")


if __name__ == "__main__":
    for arena_case in ARENA_CASES:
        test_arena_reproducible_for_same_seed(arena_case)
        test_arena_changes_for_different_seed(arena_case)

    # Rollout-level env tests for arena types reachable via MultiAgentFishEnv.
    env_rollout_cases = {
        "onepatch": {"task": "2f1p"},
        "square_quadrant": {"task": "fede01"},
        "nofood": {"task": "homing", "homing_mode": True},
    }
    for arena_case, extra_args in env_rollout_cases.items():
        snapshot_a = _rollout_snapshot(
            seed=1234,
            arena_case="uniform",
            extra_args=extra_args,
        )
        snapshot_b = _rollout_snapshot(
            seed=1234,
            arena_case="uniform",
            extra_args=extra_args,
        )
        assert _snapshots_equal(snapshot_a, snapshot_b)
        print(f">[{arena_case}] env snapshots are equal for the same seed.")

        snapshot_c = _rollout_snapshot(
            seed=1234,
            arena_case="uniform",
            extra_args=extra_args,
        )
        snapshot_d = _rollout_snapshot(
            seed=1235,
            arena_case="uniform",
            extra_args=extra_args,
        )
        assert not _snapshots_equal(snapshot_c, snapshot_d)
        print(f">[{arena_case}] env snapshots are different for different seeds.")
    print("All tests passed.")
