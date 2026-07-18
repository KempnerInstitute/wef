"""Centralized registry of all fish eval conditions.

EVAL_REGISTRY is a dict[str, EvalSpec] keyed by spec_key (= eval output folder name).
Pass spec keys directly to pipeline.py via --specs.

Spec families (for reference):
  m1a1k1_patchy_diverse, m1a1k1_uniform_diverse
  m1a1k1_patchy_square, m1a1k1_uniform_square, m1a0k1_uniform_square
  m{M}a{A}k{K}_patchy_square        — sensor ablations (patchy arena, 4 fish)
  1fish_m{M}a{A}k{K}_patchy_square  — sensor ablations (patchy arena, fish 0 only)
  food{F}_m{M}a{A}k{K}_patchy_square — sensor ablations × food multiplier
  small_cs{0,1,2}               — collective-sensing sweep, small arena
  m1a1k1_1patch
  2fish_m1a1k1_uniform_square, 2fish_m1a1k1_uniform_wide
  nfish{1,2,3,4}_m1a1k{0,1}_patchy_square  — nfish sweep (k1 baseline + k0 knollen-off)
  2f1p_{AltB,AeqB,AgtB,control_a,control_b}
  2f1p_k0_{AltB,AeqB,AgtB,control_a,control_b}  — same with knollen_mode=0
  1rw1f1p_grid  — robotic-waggle sweep (single spec, episode_configs)
  2f1p_grid
"""

# TODO: render_episodes should not really be in the spec; what we need is total_episodes (= render_episodes * n_rollout_threads) 
# Above assumes n_rollout_threads=10, which is OK for now

import argparse
import itertools
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvalSpec:
    spec_key: str
    analyses: List[str] = field(default_factory=list)
    task: str = "foraging"
    seed: int = 0
    episode_length: int = 1600
    render_episodes: int = 3  # misleading name: controls num eval rollouts recorded as pkls, not just videos
    n_rollout_threads: int = 10
    save_vids: bool = False
    num_vids_to_save: int = 1
    mormyromast_mode: int = 1
    ampullary_mode: int = 1
    knollen_mode: int = 1
    collective_sensing_mode: int = 1
    arena_size_min: Optional[Tuple[int, int]] = None   # None = use training cfg value
    arena_size_max: Optional[Tuple[int, int]] = None
    base_food_multiplier: float = 1.0
    pfeeder: float = 0.0
    prandom: float = 1.0
    urandom: float = 0.0
    prob_n_patch: float = 0.0
    fixed_num_patches: Optional[int] = None
    agent_size_mode: Optional[str] = None  # None preserves training default; set explicitly when override is needed
    active_agent_ids: Optional[Tuple[int, ...]] = None
    mute_k: Optional[int] = None
    dist_perturbation: float = 0.0
    rw_eod_rate: Optional[float] = None
    rw_freeze: Optional[int] = None
    episode_configs: Optional[List[Any]] = None   # for grid tasks; render_episodes inferred in eval_fish
    allow_aggression: Optional[int] = None  # None preserves training default; homing uses 0
    save_rnn:  bool = False  # write ep{k}_rnn.npy; only needed for rnn_* analyses
    save_obs:  bool = False  # write ep{k}_obs.npy; only needed for decoding analyses
    save_attn: bool = False  # write ep{k}_attn.npy; only needed for attention analysis
    flatten_split: bool = False  # keep flat/features pkls split per env; never consolidate into agg_flat.pkl
    analysis_kwargs: Dict[str, Dict] = field(default_factory=dict)  # per-analysis extra kwargs for pipeline dispatch

    def to_eval_args(self, run_dir: str) -> argparse.Namespace:
        """Converts spec to the argparse.Namespace that eval_fish.main() expects."""
        return argparse.Namespace(
            run_dir=run_dir,
            task=self.task,
            eval_run_name=self.spec_key,
            eval_seed=self.seed,
            eval_episode_length=self.episode_length,
            eval_render_episodes=self.render_episodes,
            n_rollout_threads=self.n_rollout_threads,
            num_vids_to_save=self.num_vids_to_save,
            num_render_envs=1,
            save_vids=self.save_vids,
            eval_mormyromast_mode=self.mormyromast_mode,
            eval_ampullary_mode=self.ampullary_mode,
            eval_knollen_mode=self.knollen_mode,
            eval_collective_sensing_mode=self.collective_sensing_mode,
            eval_base_food_multiplier=self.base_food_multiplier,
            eval_pfeeder=self.pfeeder,
            eval_prandom=self.prandom,
            eval_urandom=self.urandom,
            eval_prob_n_patch=self.prob_n_patch,
            eval_arena_size_min=self.arena_size_min,
            eval_arena_size_max=self.arena_size_max,
            eval_fixed_num_patches=self.fixed_num_patches,
            eval_agent_size_mode=self.agent_size_mode,
            eval_active_agent_ids=(
                list(self.active_agent_ids)
                if self.active_agent_ids is not None
                else None
            ),
            eval_mute_k=self.mute_k,
            eval_dist_perturbation=self.dist_perturbation,
            eval_rw_eod_rate=self.rw_eod_rate,
            eval_rw_freeze=self.rw_freeze,
            # not yet surfaced as EvalSpec fields — pass None to use training defaults
            eval_food_replenish=None,
            eval_allow_aggression=self.allow_aggression,
            eval_food_drift=None,
            eval_food_drag=None,
            eval_food_orientation_drift=None,
            eval_indiv_sensing_radius=None,
            eval_num_active_agents=None,
            eval_agent_size_sampling_mode=None,
            eval_noise_frac_morm=None,
            eval_noise_frac_amp=None,
            eval_noise_frac_amp_cons_eod=None,
            eval_noise_frac_knollen=None,
            eval_ampullary_ema=None,
            eval_ampullary_alpha=None,
            save_rnn=self.save_rnn,
            save_obs=self.save_obs,
            save_attn=self.save_attn,
        )


# ---------------------------------------------------------------------------
# Helper functions for multi-spec groups
# ---------------------------------------------------------------------------

def _2f1p_grid_configs() -> List[Tuple[float, float, int, float]]:
    """Cartesian product of (size_A, size_B, radius_cm, theta_rad) — 600 entries."""
    sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
    radii = [20, 40, 60]
    thetas = [math.pi / 4 * i for i in range(8)]
    return list(itertools.product(sizes, sizes, radii, thetas))


def _sensor_sweep_patchy_ablation_specs() -> List[EvalSpec]:
    """5 ablation specs for patchy arena: all combos with ≥2 sensors, excluding (1,1,1),
    plus explicit (0,0,1) and (0,0,0)."""
    common = dict(
        task="foraging",
        render_episodes=3,
        collective_sensing_mode=1,
        prandom=1.0,
        urandom=0.0,
        arena_size_min=(70, 70),
        arena_size_max=(70, 70),
    )
    specs = []

    # combos with ≥2 active sensors, skipping (1,1,1) which belongs to m1a1k1_patchy_square
    for morm, amp, knollen in itertools.product([0, 1], repeat=3):
        if sum((morm, amp, knollen)) < 2 or (morm, amp, knollen) == (1, 1, 1):
            continue
        specs.append(EvalSpec(
            spec_key=f"m{morm}a{amp}k{knollen}_patchy_square",
            mormyromast_mode=morm, ampullary_mode=amp, knollen_mode=knollen,
            **common,
        ))

    # single-sensor and no-sensor entries
    for morm, amp, knollen in [(0, 0, 1), (0, 0, 0)]:
        specs.append(EvalSpec(
            spec_key=f"m{morm}a{amp}k{knollen}_patchy_square",
            mormyromast_mode=morm, ampullary_mode=amp, knollen_mode=knollen,
            **common,
        ))

    return specs


def _sensor_sweep_patchy_1fish_specs() -> List[EvalSpec]:
    """6 specs: patchy arena, fish 0 only. ≥2 active sensors plus K-only and All-off.
    M-only (1,0,0) and A-only (0,1,0) are excluded."""
    _skip = {(1, 0, 0), (0, 1, 0)}  # M-only, A-only
    common = dict(
        task="foraging",
        render_episodes=3,
        collective_sensing_mode=1,
        prandom=1.0,
        urandom=0.0,
        arena_size_min=(70, 70),
        arena_size_max=(70, 70),
        active_agent_ids=(0,),
    )
    specs = []
    for morm, amp, knollen in itertools.product([0, 1], repeat=3):
        if (morm, amp, knollen) in _skip:
            continue
        specs.append(EvalSpec(
            spec_key=f"1fish_m{morm}a{amp}k{knollen}_patchy_square",
            mormyromast_mode=morm, ampullary_mode=amp, knollen_mode=knollen,
            **common,
        ))
    return specs


def _sensor_sweep_patchy_food_specs() -> List[EvalSpec]:
    """8 specs: patchy arena × food_multiplier ∈ {0.5, 0.25} × ≥2 sensor combos."""
    common = dict(
        task="foraging",
        render_episodes=3,
        collective_sensing_mode=1,
        prandom=1.0,
        urandom=0.0,
        arena_size_min=(70, 70),
        arena_size_max=(70, 70),
    )
    specs = []
    for food_mult in [0.5, 0.25]:
        food_tag = str(food_mult).replace(".", "")
        for morm, amp, knollen in itertools.product([0, 1], repeat=3):
            if sum((morm, amp, knollen)) < 2:
                continue
            specs.append(EvalSpec(
                spec_key=f"food{food_tag}_m{morm}a{amp}k{knollen}_patchy_square",
                mormyromast_mode=morm, ampullary_mode=amp, knollen_mode=knollen,
                base_food_multiplier=food_mult,
                **common,
            ))
    return specs


def _nfish_sweep_specs(knollen_mode: int = 1) -> List[EvalSpec]:
    """4 specs: nfish ∈ {1,2,3,4}, patchy arena. knollen_mode=1 (k1) or 0 (k0)."""
    id_sets = [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3)]
    spec_keys = [
        f"nfish{n}_m1a1k{knollen_mode}_patchy_square"
        for n in [1, 2, 3, 4]
    ]
    return [
        EvalSpec(
            spec_key=spec_key,
            task="foraging",
            render_episodes=1,          # 1 × n_rollout_threads(10) = 10 episodes
            mormyromast_mode=1, ampullary_mode=1, knollen_mode=knollen_mode,
            collective_sensing_mode=1,
            mute_k=0,
            prandom=1.0, urandom=0.0,
            arena_size_min=(70, 70), arena_size_max=(70, 70),
            active_agent_ids=id_set,
            save_rnn=True,
        )
        for spec_key, id_set in zip(spec_keys, id_sets)
    ]


# Deterministic bot heading for the 1rw1f1p grid task (+y / up). The RL-fish
# start-position grid is laid out relative to this; MAEFish.reset() pins the
# bot (agent 0) to the same value.
BOT_ORIENTATION = np.pi / 2


def _1rw1f1p_episode_configs() -> List[Tuple]:
    """300 configs: N=10 trials × (3 size pairs × 5 eod rates × 2 freeze values).

    Each config: (size_A, size_B, eod_rate, freeze, rl_x, rl_y, trial_id).
    The same N pre-generated RL-fish start positions are reused for every
    condition so that starting location is controlled across the eod_rate /
    size / freeze sweep.  Bot (A) is always placed at centre (75, 75).

    Positions form a deterministic polar grid: N/2 evenly-spaced angles crossed
    with two radii (25 and 50 cm from the bot), giving full angular coverage of
    both hemispheres at two distances.  Angles are measured *relative to the
    bot's heading* (BOT_ORIENTATION, +y / up), so theta=0 is directly ahead of
    the bot.  The bot's orientation is pinned to the same value in
    MAEFish.reset() for the 1rw1f1p task.
    """
    size_pairs  = [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]
    eod_rates   = [0.0, 0.25, 0.5, 0.75, 1.0]
    freeze_vals = [0, 1]
    N = 10

    center = 75.0
    radii  = [25.0, 50.0]
    n_theta = N // len(radii)  # N/2 evenly-spaced angles
    # theta is relative to the bot's heading; offset into world frame.
    thetas = BOT_ORIENTATION + np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    rl_xs, rl_ys = [], []
    for r in radii:
        for th in thetas:
            rl_xs.append(center + r * np.cos(th))
            rl_ys.append(center + r * np.sin(th))
    rl_xs, rl_ys = np.array(rl_xs), np.array(rl_ys)

    configs = []
    for trial_id in range(N):
        for (sA, sB) in size_pairs:
            for eod in eod_rates:
                for freeze in freeze_vals:
                    configs.append((sA, sB, eod, freeze,
                                    float(rl_xs[trial_id]),
                                    float(rl_ys[trial_id]),
                                    trial_id))
    return configs


def _homing_episode_configs(
    n: int = 100,
    arena_size: float = 70.0,
    wall_margin: float = 5.0,
    min_dist: float = 20.0,
    seed: int = 42,
) -> List[Tuple]:
    """Pre-calculated homing episode starting positions.

    Arena is fixed to arena_size × arena_size.  Each config is an 8-tuple:
        (arena_w, arena_h, target_x, target_y, target_ori, homer_x, homer_y, homer_ori)

    The arena size is included so MAEFish can override the per-episode random
    arena draw and guarantee that positions are within bounds.

    Both agents are placed at least wall_margin cm from every wall.
    The homing agent starts at least min_dist cm from the target.
    """
    rng  = np.random.default_rng(seed)
    lo   = wall_margin
    hi   = arena_size - wall_margin
    configs = []
    while len(configs) < n:
        tx  = rng.uniform(lo, hi)
        ty  = rng.uniform(lo, hi)
        to  = rng.uniform(-np.pi, np.pi)
        # sample homer until far enough from target
        for _ in range(10_000):
            hx = rng.uniform(lo, hi)
            hy = rng.uniform(lo, hi)
            if np.hypot(hx - tx, hy - ty) >= min_dist:
                break
        ho = rng.uniform(-np.pi, np.pi)
        configs.append((arena_size, arena_size, tx, ty, to, hx, hy, ho))
    return configs


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EVAL_REGISTRY: Dict[str, EvalSpec] = {}

# --- Foraging groups ---

# EVAL_REGISTRY["m1a1k1_patchy_diverse"] = EvalSpec(
#     spec_key="m1a1k1_patchy_diverse",
#     task="foraging",
#     render_episodes=3,
#     pfeeder=0.0, prandom=1.0, urandom=0.0, prob_n_patch=0.0,
#     arena_size_min=(40, 40), arena_size_max=(300, 300),
#     agent_size_mode="random",
#     save_vids=True, num_vids_to_save=1,
# )

# EVAL_REGISTRY["m1a1k1_uniform_diverse"] = EvalSpec(
#     spec_key="m1a1k1_uniform_diverse",
#     task="foraging",
#     render_episodes=3,
#     pfeeder=0.0, prandom=0.0, urandom=1.0, prob_n_patch=0.0,
#     arena_size_min=(40, 40), arena_size_max=(300, 300),
#     agent_size_mode="random",
#     save_vids=True, num_vids_to_save=1,
# )

# m1a1k1 full-sensor patchy — separate from the ablation sweep so it can
# get the richer report set (predict_action, eod) without polluting the sweep.
EVAL_REGISTRY["m1a1k1_patchy_square"] = EvalSpec(
    spec_key="m1a1k1_patchy_square",
    task="foraging",
    render_episodes=3,
    mormyromast_mode=1, ampullary_mode=1, knollen_mode=1,
    collective_sensing_mode=1,
    prandom=1.0, urandom=0.0,
    arena_size_min=(70, 70), arena_size_max=(70, 70),
    save_rnn=True,
)

# Sensor ablations for patchy arena (all combos except m1a1k1 → see patchy_full)
EVAL_REGISTRY.update({s.spec_key: s for s in _sensor_sweep_patchy_ablation_specs()})
EVAL_REGISTRY.update({s.spec_key: s for s in _sensor_sweep_patchy_1fish_specs()})
EVAL_REGISTRY.update({s.spec_key: s for s in _sensor_sweep_patchy_food_specs()})

# m1a1k1 full-sensor uniform — richer reports (attention, rnn, diagnostics)
EVAL_REGISTRY["m1a1k1_uniform_square"] = EvalSpec(
    spec_key="m1a1k1_uniform_square",
    task="foraging",
    render_episodes=3,
    mormyromast_mode=1, ampullary_mode=1, knollen_mode=1,
    collective_sensing_mode=1,
    prandom=0.0, urandom=1.0,
    arena_size_min=(70, 70), arena_size_max=(70, 70),
    save_rnn=True, save_attn=True,
)

# Single ablation for uniform arena (m1a0k1 only)
EVAL_REGISTRY["m1a0k1_uniform_square"] = EvalSpec(
    spec_key="m1a0k1_uniform_square",
    task="foraging",
    render_episodes=3,
    mormyromast_mode=1, ampullary_mode=0, knollen_mode=1,
    collective_sensing_mode=1,
    prandom=0.0, urandom=1.0,
    arena_size_min=(70, 70), arena_size_max=(70, 70),
)

# Collective-sensing sweep at small arena size
for spec in [
    EvalSpec(
        spec_key="small_cs0",
        task="foraging", render_episodes=3,
        mormyromast_mode=1, ampullary_mode=1, knollen_mode=1,
        collective_sensing_mode=0,
        prandom=0.0, urandom=1.0,
        arena_size_min=(40, 40), arena_size_max=(40, 40),
    ),
    EvalSpec(
        spec_key="small_cs1",
        task="foraging", render_episodes=3,
        mormyromast_mode=1, ampullary_mode=1, knollen_mode=1,
        collective_sensing_mode=1,
        prandom=0.0, urandom=1.0,
        arena_size_min=(40, 40), arena_size_max=(40, 40),
    ),
    EvalSpec(
        spec_key="small_cs2",
        task="foraging", render_episodes=3,
        mormyromast_mode=1, ampullary_mode=1, knollen_mode=1,
        collective_sensing_mode=2,
        prandom=0.0, urandom=1.0,
        arena_size_min=(40, 40), arena_size_max=(40, 40),
    ),
]:
    EVAL_REGISTRY[spec.spec_key] = spec

EVAL_REGISTRY["m1a1k1_1patch"] = EvalSpec(
    spec_key="m1a1k1_1patch",
    task="n_patch",
    render_episodes=3,
    base_food_multiplier=2.0,
    fixed_num_patches=1,
)


def _npatch_iso_specs() -> List[EvalSpec]:
    """9 specs for the iso-total-food patch sweep (n_patches ∈ {1,2,3,4,5}).

    iso series — n × food_mult = 1.0 (total food constant at ~26.5 items):
      iso_p1_m1, iso_p2_m05, iso_p3_m0333, iso_p4_m025, iso_p5_m02

    free series — food_mult = 1.0 (total food scales with n_patches):
      iso_p1_m1 (shared anchor), free_p2_m1, free_p3_m1, free_p4_m1, free_p5_m1

    Uses task="n_patch_fixed" so base_food_multiplier is correctly applied to
    NPatchArena per-patch density (not number-of-patches).
    """
    _base = dict(
        task="n_patch_fixed",
        render_episodes=3,
        n_rollout_threads=10,
        mormyromast_mode=1, ampullary_mode=1, knollen_mode=1,
        collective_sensing_mode=1,
        arena_size_min=(70, 70), arena_size_max=(70, 70),
    )
    iso_pairs = [
        (1, 1.000, "iso_p1_m1"),
        (2, 0.500, "iso_p2_m05"),
        (3, 0.333, "iso_p3_m0333"),
        (4, 0.250, "iso_p4_m025"),
        (5, 0.200, "iso_p5_m02"),
    ]
    free_pairs = [
        (2, 1.0, "free_p2_m1"),
        (3, 1.0, "free_p3_m1"),
        (4, 1.0, "free_p4_m1"),
        (5, 1.0, "free_p5_m1"),
    ]
    specs = []
    for n, m, key in iso_pairs + free_pairs:
        specs.append(EvalSpec(
            spec_key=key,
            fixed_num_patches=n,
            base_food_multiplier=m,
            **_base,
        ))
    return specs


EVAL_REGISTRY.update({s.spec_key: s for s in _npatch_iso_specs()})

EVAL_REGISTRY["2fish_m1a1k1_uniform_square"] = EvalSpec(
    spec_key="2fish_m1a1k1_uniform_square",
    task="foraging",
    render_episodes=10,
    mormyromast_mode=1, ampullary_mode=1, knollen_mode=1,
    collective_sensing_mode=1,
    prandom=0.0, urandom=1.0,
    mute_k=0,
    arena_size_min=(70, 70), arena_size_max=(70, 70),
    active_agent_ids=(0, 1),
    save_rnn=True, save_obs=True,
    analysis_kwargs={
        "rnn_plsc": {"knollen_range_cm": None},   # 2-range: in_range / out_range
        "rnn_dim":  {"two_fish_mode": True, "knollen_range_cm": None},
    },
)

EVAL_REGISTRY["2fish_1mute_m1a1k1_uniform_square"] = EvalSpec(
    spec_key="2fish_1mute_m1a1k1_uniform_square",
    task="foraging",
    render_episodes=10,
    mormyromast_mode=1, ampullary_mode=1, knollen_mode=1,
    collective_sensing_mode=1,
    prandom=0.0, urandom=1.0,
    mute_k=1,
    arena_size_min=(70, 70), arena_size_max=(70, 70),
    active_agent_ids=(0, 1),
)

EVAL_REGISTRY["2fish_m1a1k1_uniform_wide"] = EvalSpec(
    spec_key="2fish_m1a1k1_uniform_wide",
    task="wide",
    render_episodes=10,
    mormyromast_mode=1, ampullary_mode=1, knollen_mode=1,
    collective_sensing_mode=1,
    prandom=0.0, urandom=1.0,
    mute_k=0,
    arena_size_min=(160, 40), arena_size_max=(160, 40),
    active_agent_ids=(0, 1),
    save_rnn=True, save_obs=True,
    analysis_kwargs={
        "rnn_plsc": {"knollen_range_cm": 100.0},   # 3-range: within_morm / morm_to_knollen / beyond_knollen
        "rnn_dim":  {"two_fish_mode": True, "knollen_range_cm": 100.0},
    },
)

EVAL_REGISTRY.update({s.spec_key: s for s in _nfish_sweep_specs(knollen_mode=1)})
EVAL_REGISTRY.update({s.spec_key: s for s in _nfish_sweep_specs(knollen_mode=0)})


def _nfish_scaled_food_specs(knollen_mode: int = 1) -> List[EvalSpec]:
    """4 specs: nfish ∈ {1,2,3,4}, patchy arena, food multiplier = N.

    Food density scales linearly with group size so per-fish food supply is
    constant across N.  Comparing these to the standard nfish sweep isolates
    food-competition costs from collective-sensing effects.
    """
    id_sets = [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3)]
    return [
        EvalSpec(
            spec_key=f"nfish{n}_m1a1k{knollen_mode}_patchy_square_scaled",
            task="foraging",
            render_episodes=1,
            mormyromast_mode=1, ampullary_mode=1, knollen_mode=knollen_mode,
            collective_sensing_mode=1,
            mute_k=0,
            prandom=1.0, urandom=0.0,
            arena_size_min=(70, 70), arena_size_max=(70, 70),
            active_agent_ids=id_set,
            base_food_multiplier=float(n),
            save_rnn=False,
        )
        for n, id_set in zip([1, 2, 3, 4], id_sets)
    ]


EVAL_REGISTRY.update({s.spec_key: s for s in _nfish_scaled_food_specs(knollen_mode=1)})
EVAL_REGISTRY.update({s.spec_key: s for s in _nfish_scaled_food_specs(knollen_mode=0)})


def _nfish4_density_specs() -> List[EvalSpec]:
    """6 specs: N=4 fish × 3 arena sizes × k1 only.

    Arena areas: 50×50 (dense, ~51% of training area), 70×70 (standard),
    99×99 (wide, ~200% of training area).  Food multiplier scales with area
    so per-fish food supply is held constant across densities; only fish
    proximity (crowding) varies.
    """
    _TRAIN_AREA = 70.0 * 70.0
    configs = [
        ("dense", (50, 50)),
        ("wide",  (99, 99)),
    ]
    specs = []
    for knollen_mode in [0, 1]:
        for label, (sz, _) in [(l, (s, s)) for l, s in [("dense", 50), ("wide", 99)]]:
            arena = (sz, sz)
            food_mult = round((sz * sz) / _TRAIN_AREA, 3)
            for k in [0, 1]:
                key = f"nfish4_m1a1k{k}_patchy_{label}"
                specs.append(EvalSpec(
                    spec_key=key,
                    task="foraging",
                    render_episodes=1,
                    mormyromast_mode=1, ampullary_mode=1, knollen_mode=k,
                    collective_sensing_mode=1,
                    mute_k=0,
                    prandom=1.0, urandom=0.0,
                    arena_size_min=arena, arena_size_max=arena,
                    active_agent_ids=(0, 1, 2, 3),
                    base_food_multiplier=food_mult,
                    save_rnn=False,
                ))
    # deduplicate (the double-loop above creates duplicates)
    seen = {}
    for s in specs:
        seen[s.spec_key] = s
    return list(seen.values())


EVAL_REGISTRY.update({s.spec_key: s for s in _nfish4_density_specs()})

# --- 2f1p groups ---

# Main conditions (AltB, AeqB, AgtB) and controls (a-only, b-only) in one group.
for spec in [
    EvalSpec(spec_key="2f1p_AltB",     task="2f1p", agent_size_mode="AltB", render_episodes=10, dist_perturbation=0.0),
    EvalSpec(spec_key="2f1p_AeqB",     task="2f1p", agent_size_mode="AeqB", render_episodes=10, dist_perturbation=0.0),
    EvalSpec(spec_key="2f1p_AgtB",     task="2f1p", agent_size_mode="AgtB", render_episodes=10, dist_perturbation=0.0),
    EvalSpec(spec_key="2f1p_control_a", task="2f1p_control_a", agent_size_mode="AeqB", render_episodes=10, dist_perturbation=0.0),
    EvalSpec(spec_key="2f1p_control_b", task="2f1p_control_b", agent_size_mode="AeqB", render_episodes=10, dist_perturbation=0.0),
]:
    EVAL_REGISTRY[spec.spec_key] = spec

# k0 versions of the 2f1p conditions (knollen off)
for spec in [
    EvalSpec(spec_key="2f1p_k0_AltB",      task="2f1p", agent_size_mode="AltB",  knollen_mode=0, render_episodes=10, dist_perturbation=0.0),
    EvalSpec(spec_key="2f1p_k0_AeqB",      task="2f1p", agent_size_mode="AeqB",  knollen_mode=0, render_episodes=10, dist_perturbation=0.0),
    EvalSpec(spec_key="2f1p_k0_AgtB",      task="2f1p", agent_size_mode="AgtB",  knollen_mode=0, render_episodes=10, dist_perturbation=0.0),
    EvalSpec(spec_key="2f1p_k0_control_a", task="2f1p_control_a", agent_size_mode="AeqB", knollen_mode=0, render_episodes=10, dist_perturbation=0.0),
    EvalSpec(spec_key="2f1p_k0_control_b", task="2f1p_control_b", agent_size_mode="AeqB", knollen_mode=0, render_episodes=10, dist_perturbation=0.0),
]:
    EVAL_REGISTRY[spec.spec_key] = spec

# 300 episodes: 10 trials × (3 size pairs × 5 eod rates × 2 freeze).
# render_episodes is inferred from len(episode_configs) / n_rollout_threads in eval_fish.
EVAL_REGISTRY["1rw1f1p_grid"] = EvalSpec(
    spec_key="1rw1f1p_grid",
    task="1rw1f1p",
    render_episodes=10,
    episode_configs=_1rw1f1p_episode_configs(),
    flatten_split=True,
)

# Grid sweep over (size_A, size_B, start_radius, start_theta) — 600 episodes.
# render_episodes is inferred from len(episode_configs) / n_rollout_threads in eval_fish.
EVAL_REGISTRY["2f1p_grid"] = EvalSpec(
    spec_key="2f1p_grid",
    task="2f1p_grid",
    agent_size_mode="random",
    episode_configs=_2f1p_grid_configs(),
)

# --- Homing group ---
# Homing layout: agent_id=0 is the stationary target; agent_id=1 is the homing agent.
# Episodes are variable-length: successful ones terminate early
# (MAEFish homing_success_counter >= required_homing_steps), timed-out ones run
# to episode_length. analysis_homing.py loads raw/agg_flat.pkl directly and
# applies its own agent filtering — it does NOT use the features/summaries pkls.
EVAL_REGISTRY["homing"] = EvalSpec(
    spec_key="homing",
    task="homing",
    seed=1,
    episode_length=500,
    # render_episodes inferred from len(episode_configs) / n_rollout_threads
    pfeeder=0.0, prandom=0.0, urandom=1.0,
    base_food_multiplier=0.0,
    agent_size_mode="AeqB",
    allow_aggression=0,
    save_vids=True, num_vids_to_save=1,
    save_rnn=True,
    episode_configs=_homing_episode_configs(),
)


# ---------------------------------------------------------------------------
# Analyses registry — single source of truth for spec → analyses mapping.
# Edit here to add/remove analyses for any spec; do not set analyses on EvalSpec.
#
# Resolution order:
#   1. SPEC_ANALYSES  — explicit per-spec entry (highest priority)
#   2. _FAMILY_RULES  — prefix-matched family defaults
#   3. DEFAULT_ANALYSES — fallback for any unmatched spec
# ---------------------------------------------------------------------------

DEFAULT_ANALYSES: List[str] = ["behavior"]

# Prefix-based family defaults — first matching prefix wins.
_FAMILY_RULES: List[Tuple[str, List[str]]] = [
    ("1fish_",        ["behavior"]),
    ("nfish",         ["behavior", "rnn_dim"]),
    ("1rw1f1p_",      ["1f1rw1p"]),
    ("2f1p_k0_",      []),
    ("2f1p_",         []),
]

# Per-spec overrides for named specs with non-default analyses.
SPEC_ANALYSES: Dict[str, List[str]] = {
    "homing":                  ["homing"],  # analysis_homing.py does its own loading/filtering
    "m1a1k1_patchy_square":        ["general", "behavior", "predict_action", "eod", "idi", "rnn_dim", "pairwise", "biting_network", "food_distribution", "rollout_diagnostics"],
    "m1a1k1_uniform_square":       ["general", "behavior", "attention", "predict_action", "eod", "idi", "rnn_dim", "rollout_diagnostics", "biting_network"],
    "m1a0k1_uniform_square":       ["behavior"],
    "small_cs0":                   [],
    "small_cs1":                   [],
    "small_cs2":                   [],
    "m1a1k1_1patch":               [],
    # iso/free patch sweep — analysis is done at multi-spec level by analysis_food_grid_iso
    **{s.spec_key: [] for s in _npatch_iso_specs()},
    "2fish_m1a1k1_uniform_square": ["eod", "idi", "2fish", "pairwise", "decoding", "rnn_psd", "rnn_plsc", "rnn_dim"],
    "2fish_m1a1k1_uniform_wide":   ["eod", "idi", "2fish", "pairwise", "decoding", "rnn_timescales", "rnn_dim", "rnn_plsc"],
    "2f1p_grid":                   ["2f1p_grid"],
}

# Apply analyses to every registered spec.
for _key, _spec in EVAL_REGISTRY.items():
    if _key in SPEC_ANALYSES:
        _spec.analyses = SPEC_ANALYSES[_key]
        continue
    for _prefix, _analyses in _FAMILY_RULES:
        if _key.startswith(_prefix):
            _spec.analyses = _analyses
            break
    else:
        _spec.analyses = DEFAULT_ANALYSES


# ---------------------------------------------------------------------------
# Spec groups — named collections of spec_keys for --group in pipeline.py.
# Groups can overlap; pipeline deduplicates before running.
# ---------------------------------------------------------------------------

SPEC_GROUPS: Dict[str, List[str]] = {
    # Core foraging specs (primary manuscript figures 3 & 4 base)
    "core": [
        "m1a1k1_patchy_square",
        "m1a1k1_uniform_square",
    ],
    # Sensor-ablation specs (patchy 4-fish)
    "ablations": [
        "m0a1k1_patchy_square", "m1a0k1_patchy_square", "m1a1k0_patchy_square",
        "m0a0k1_patchy_square", "m0a0k0_patchy_square",
    ],
    # Single-fish sensor ablations (patchy arena, fish 0 only)
    "ablations_1fish": [
        "1fish_m0a1k1_patchy_square", "1fish_m1a0k1_patchy_square",
        "1fish_m1a1k0_patchy_square", "1fish_m1a1k1_patchy_square",
    ],
    # Food-multiplier sweep (patchy arena, ≥2 active sensors)
    "ablations_food": [
        "food05_m0a1k1_patchy_square", "food05_m1a0k1_patchy_square",
        "food05_m1a1k0_patchy_square", "food05_m1a1k1_patchy_square",
        "food025_m0a1k1_patchy_square", "food025_m1a0k1_patchy_square",
        "food025_m1a1k0_patchy_square", "food025_m1a1k1_patchy_square",
    ],
    # Collective-sensing sweep
    "cs": ["small_cs0", "small_cs1", "small_cs2"],
    # Two-fish dyadic assays (2f1p conditions + controls)
    "2f1p":    [k for k in EVAL_REGISTRY if k.startswith("2f1p_") and not k.startswith("2f1p_k0_") and k != "2f1p_grid"],
    "2f1p_k0": [k for k in EVAL_REGISTRY if k.startswith("2f1p_k0_")],
    # Robotic-waggle sweep
    "rw": ["1rw1f1p_grid"],
    # Two-fish square/wide
    "2fish": ["2fish_m1a1k1_uniform_square", "2fish_m1a1k1_uniform_wide"],
    # N-fish scaling
    "nfish": [k for k in EVAL_REGISTRY if k.startswith("nfish")],
    # Homing task
    "homing": ["homing"],
    # Iso-total-food patch sweep (9 specs; analysis via analysis_food_grid_iso multi-spec)
    "npatch_iso": [s.spec_key for s in _npatch_iso_specs()],
    # All specs referenced in the manuscript (excluding missing homing/all-muted)
    "manuscript": [
        "m1a1k1_patchy_square",
        "m1a1k1_uniform_square",
        "m0a1k1_patchy_square", 
        "m0a0k1_patchy_square",
        "m1a0k1_patchy_square", 
        "m1a1k0_patchy_square", 
        "2f1p_AltB", "2f1p_AeqB", "2f1p_AgtB", "2f1p_control_a", "2f1p_control_b",        
        "1fish_m1a1k1_patchy_square",
        "food05_m1a1k1_patchy_square",
        "m1a1k1_1patch",
        "small_cs0", "small_cs1", "small_cs2",
        "2fish_m1a1k1_uniform_square",
        "nfish1_m1a1k1_patchy_square", "nfish2_m1a1k1_patchy_square",
        "nfish3_m1a1k1_patchy_square", "nfish4_m1a1k1_patchy_square",
        "nfish1_m1a1k0_patchy_square", "nfish2_m1a1k0_patchy_square",
        "nfish3_m1a1k0_patchy_square", "nfish4_m1a1k0_patchy_square",
    ],
}
