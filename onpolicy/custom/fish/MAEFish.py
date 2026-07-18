from pathlib import Path
import os
import sys
import time
import math
import numpy as np
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import tqdm
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from datetime import datetime
import random
import arena as ar
import cProfile
import pstats
import cfg
import electric
import movement
import utils_electric as ue
from utils_general import sigmoid, sample_skewed_uniform
from cfg import m_to_cm, cm_to_m
from sensing import (
    FracRandModel,
    DynamicBaselineModel,
    SensingParams,
    SensorModesEpisode,
    DEFAULT_AMPULLARY_ALPHA,
)
from utils_general import polar_to_cartesian
from renderer import FishRenderer

OBJECT_TYPES = cfg.OBJECT_TYPES
ENV_PARAMS = cfg.ENV_PARAMS
AGENT_PARAMS = cfg.AGENT_PARAMS
REWARDS = cfg.REWARDS
COLORS = cfg.COLORS
FISH_CONSTANTS = cfg.FISH_CONSTANTS
SCALABLE_FOOD_PARAMS = cfg.SCALABLE_FOOD_PARAMS
class EFishAgent:
    def __init__(
        self,
        num_agents,
        arena_size,
        agent_id,
        motion_order=1,
        max_food_eaten_per_step=1,
        allow_aggression=0,
        use_bite_cooldown=True,
        agent_size_min=0,
        agent_size_max=1,
        backwards=False,
        enable_bite_action=True,
        seed=None,
        active=True,
        ampullary_ema=False,
        multiplier_linear=1.0,
        multiplier_angular=1.0,
        size_speed_exponent=1.0,
        probabilistic_eating=False,
        agent_size_sampling_mode="uniform",
        eat_cooldown_rate=None,
    ):
        self.__dict__.update(AGENT_PARAMS)
        self.eating_radius = self.eating_radius_cm
        self.biting_radius = self.biting_radius_cm
        self.body_radius = self.body_radius_cm
        self.collective_sensing_radius = self.collective_sensing_radius_cm
        self.morm_agent_detection_range = self.morm_agent_detection_range_m
        self.morm_food_detection_range = self.morm_food_detection_range_m
        self.amp_food_detection_range = self.amp_food_detection_range_m
        self.amp_fish_detection_range = self.amp_fish_detection_range_m
        self.knollen_agent_detection_range = self.knollen_agent_detection_range_m
        self.active = active
        self.num_agents = num_agents
        self.arena_size = arena_size
        self.agent_id = agent_id
        self.muted = False
        self.probabilistic_eating = probabilistic_eating

        self.orientation = 0.0
        self.last_orientation = 0.0
        self.position = np.array([self.arena_size[0] / 2, self.arena_size[1] / 2])

        self.agent_size_min = agent_size_min
        self.agent_size_max = agent_size_max
        self.agent_size_sampling_mode = agent_size_sampling_mode

        self.motion_order = motion_order

        # NOTE that these have already been *dt in cfg.py so actually (velocity --> step/turn, acceleration --> velocity)
        self.max_linear_velocity_baseline = AGENT_PARAMS["max_linear_velocity_baseline"] # cm/timestep
        self.min_linear_velocity_baseline = AGENT_PARAMS["min_linear_velocity_baseline"] # cm/timestep
        self.max_linear_acceleration_baseline = AGENT_PARAMS["max_linear_acceleration_baseline"] # cm/timestep^2
        self.max_angular_velocity_baseline = AGENT_PARAMS["max_angular_velocity_baseline"] # rad/timestep
        self.min_angular_velocity_baseline = AGENT_PARAMS["min_angular_velocity_baseline"] # rad/timestep
        self.max_angular_acceleration_baseline = AGENT_PARAMS["max_angular_acceleration_baseline"] # rad/timestep^2

        # Apply Movement Multipliers [GLOBAL]
        self.multiplier_linear = multiplier_linear
        self.multiplier_angular = multiplier_angular
        self.size_speed_exponent = size_speed_exponent

        # For per-step bookkeeping of how much we exceed the original max angular velocity

        # Electric sensing geometry and baselines
        # init_agent_electric(self)

        # Set up agent properties
        self.max_food_eaten_per_step = max_food_eaten_per_step
        self.has_nearby_emit_eod_ids = []
        self.allow_aggression = allow_aggression
        self.use_bite_cooldown = use_bite_cooldown
        self.bite_cooldown_counter = 0.0
        self.bite_cooldown_rate = AGENT_PARAMS["bite_cooldown_rate"]
        self.eat_cooldown_counter = 0.0
        self.eat_cooldown_rate = eat_cooldown_rate if eat_cooldown_rate is not None else AGENT_PARAMS["eat_cooldown_rate"]
        self.bite_cooldown_history = []
        self.backwards = backwards
        self.enable_bite_action = enable_bite_action
        self.previous_agent_distance = None
        self.ampullary_alpha = DEFAULT_AMPULLARY_ALPHA if ampullary_ema else 1
        self.reset(seed=seed)  # Additional initializations are in reset()

    def reset(self, seed=None):
        # update np_random
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        elif not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()

        if self.agent_size_sampling_mode == "uniform":
            self.agent_size = self.np_random.uniform(
                self.agent_size_min, self.agent_size_max
            )
        elif self.agent_size_sampling_mode == "grid":
            num_grid_options = AGENT_PARAMS.get("agent_size_grid_num_options", 6)
            grid_values = np.linspace(self.agent_size_min, self.agent_size_max, num_grid_options)
            self.agent_size = self.np_random.choice(grid_values)
        else:
            raise ValueError(
                f"Unsupported agent_size_sampling_mode: {self.agent_size_sampling_mode}"
            )
        movement.reset_agent_movement_parameters_to_baselines(self) # sets params like max/min linear/angular velocity/acceleration to cfg baseline values
        # Now scale by global multipliers AND size-based scaling
        movement.apply_global_movement_multipliers(self, self.multiplier_linear, self.multiplier_angular)
        movement.apply_size_multipliers(
            self,
            self.agent_size,
            size_speed_exponent=self.size_speed_exponent,
        )

        self.position = self.np_random.uniform([0, 0], self.arena_size)
        if not self.active:
            self.position = np.array(
                [-9999, -9999]
            )  # Inactive, placed well outside arena
        self.orientation = self.np_random.uniform(-np.pi, np.pi)
        self.last_orientation = self.orientation
        self.energy = self.np_random.uniform(self.energy_minimum, self.energy_maximum)
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.trajectory = [self.position.copy()]
        self.eod_history = [True]  # hard-code starting with an observation
        self.energy_history = [self.energy]
        self.move_forward = 0
        self.turn_angle = 0
        self.cumulative_reward = 0
        self.previous_food_distance = None
        self.previous_agent_distance = None
        self.emit_eod = (
            True if self.active else False
        )  # hard-code starting with an observation
        self.bite_action = False
        self.has_nearby = False  # hard-code starting with large-radius observation
        self.nearby_emitting = False
        self.knollen_sensed_agents = []
        self.curr_food_consumed = []
        self.has_nearby_emit_eod_ids = []
        self.was_bitten = False
        self.was_bitten_by_agent_id = None
        self.bite_other_fish = False
        self.bite_history = [False]
        self.was_bitten_history = [False]
        self.last_action = [0] * 4 if self.enable_bite_action else [0] * 3
        self.collided = False
        self.bite_cooldown_counter = 0.0
        self.bite_cooldown_history = [0.0]
        self.eat_cooldown_counter = 0.0
        self.ampullary_sensed_objects = []
        # self.mormyromast_observations = np.zeros(self.num_mormyromast_sensors_real)
        self.mormyromast_observations_virtual = np.zeros(
            self.num_mormyromast_sensors_virtual
        )
        self.knollen_observations = np.zeros(self.num_knollen_sensors * (self.num_agents - 1))
        # self.knollen_readings = {
        #     i: np.zeros(self.num_knollen_sensors)
        #     for i in range(self.num_agents)
        #     if i != self.agent_id
        # }
        self.knollen_metadata = {
            i: {"size": -1} for i in range(self.num_agents) if i != self.agent_id
        }
        self.ampullary_observations = np.zeros(self.num_ampullary_sensors)
        self.ampullary_filtered_observations = np.zeros(
            self.num_ampullary_sensors
        )  # Add ExpSmoothing
        # (
        #     self.mormyromast_cd,
        #     self.ampullary_cd,
        #     self.ampullary_intrinsic_baseline,
        #     self.center_mormyromast_cd,
        #     self.center_ampullary_cd,
        #     self.center_ampullary_intrinsic_baseline,
        self.center_field = {}  # Raw electric field gradients at fish center
        self.center_field["mormyromast"] = np.zeros(2)
        self.center_field["ampullary"] = np.zeros(2)
        self.center_field["knollen"] = np.zeros(
            (self.num_agents, 2)
        )  # gradient generated by EOD of agent_i on this agent
        self.displacement_ground = np.zeros(2)
        self.displacement_ego = np.zeros(2)

    def _actor_to_command(self, action):
        # NOTE move_forward and turn_angle are *commands*, not velocities.
        # NOTE misnomer: self.move_forward and self.turn_angle are more appropriately called move_command and turn_command
        # Actor network produces  outputs in (-inf, inf)
        # convert to commands within VALID ranges [0,1], [-1,1], True/False
        # Range of move_forward depends on backwards capability
        if self.enable_bite_action:
            self.move_forward, self.turn_angle, self.emit_eod, self.bite_action = action
        else:
            self.move_forward, self.turn_angle, self.emit_eod = action
            self.bite_action = False

        # Bite becomes True/False
        if self.allow_aggression == 1:
            if self.use_bite_cooldown and self.bite_cooldown_counter > 0:
                self.bite_action = False
            else:
                self.bite_action = (
                    sigmoid(self.bite_action) > 0.5
                )  # Treat as binary (0: no bite, 1: bite)
                if self.bite_action:
                    self.bite_cooldown_counter = 1.0
        else:  # No aggression
            self.bite_action = False

        # Scale movement commands
        if self.motion_order == 1:
            activation_fn = np.tanh if self.backwards else sigmoid
            self.move_forward = activation_fn(self.move_forward) # in [0,1] (only forward) or [-1,1] (backwards enabled)
            self.turn_angle = np.tanh(self.turn_angle) # in [-1,1]
        elif self.motion_order == 2:
            self.move_forward = np.tanh(self.move_forward) # in [-1,1]
            self.turn_angle = np.tanh(self.turn_angle) # in [-1,1]

        # EOD becomes True/False
        self.emit_eod = self.emit_eod > 0 # True/False
        self.emit_eod = False if self.muted else self.emit_eod # True/False


    def step(self, action, arena, agent_objects):
        self.was_bitten = False
        self.was_bitten_by_agent_id = None
        self.bite_other_fish = False
        self.curr_food_consumed = []

        self._actor_to_command(action)
        self.last_action = (
            [self.move_forward, self.turn_angle, self.emit_eod, self.bite_action]
            if self.enable_bite_action
            else [self.move_forward, self.turn_angle, self.emit_eod]
        )

        # Act
        if not self.bite_action and self.eat_cooldown_counter == 0:
            ate_food = self.check_and_consume_food(arena)
            if ate_food:
                self.eat_cooldown_counter = 1.0

        movement.apply_movement_step(self, arena, agent_objects)

        # Update eat/bite cooldown counter
        if getattr(self, "eat_cooldown_counter", 0) > 0:
            self.eat_cooldown_counter = max(0.0, self.eat_cooldown_counter - self.eat_cooldown_rate)
        if self.use_bite_cooldown:
            self.bite_cooldown_counter = max(
                0.0, self.bite_cooldown_counter - self.bite_cooldown_rate
            )
            self.bite_cooldown_history.append(self.bite_cooldown_counter)

        # Update histories
        self.trajectory.append(self.position.copy())
        self.trajectory = self.trajectory[-5:]
        self.eod_history.append(self.emit_eod)
        self.energy_history.append(self.energy)
        self.bite_history.append(self.bite_action)
        self.was_bitten_history.append(False)  # will be updated post-step


    def check_and_bite_nearest(self, agent_objects):
        # Check if on cooldown
        # if self.use_bite_cooldown and self.bite_cooldown_counter > 0:
        #     return False  # Can't bite while on cooldown

        # Find nearest agent within the eating radius
        nearest_agent = None
        min_distance = float("inf")
        for other_agent in agent_objects:
            if other_agent != self:  # Don't check itself
                pos_diff = other_agent.position - self.position
                agent_dist = np.linalg.norm(pos_diff)
                if (
                    self.asym_eating
                    and agent_dist < self.biting_radius
                    and agent_dist < min_distance
                ):
                    angle_to_agent = np.arctan2(pos_diff[1], pos_diff[0])
                    angle_diff = (angle_to_agent - self.orientation + np.pi) % (
                        2 * np.pi
                    ) - np.pi
                    if abs(angle_diff) <= self.eating_angle / 2:
                        nearest_agent = other_agent
                        min_distance = agent_dist
        # If a nearby agent is found, apply bite effect and reset cooldown
        if nearest_agent:
            nearest_agent.was_bitten = True
            nearest_agent.was_bitten_by_agent_id = self.agent_id
            nearest_agent.was_bitten_history[-1] = True
            # if self.use_bite_cooldown:
            #     self.bite_cooldown_counter = 1.0  # Reset counter after biting
            #     print(f'agent {self.agent_id} cooldown: {self.bite_cooldown_counter} in check_and_bite_nearest')
            return True  # Indicates that biting occurred
        return False  # No biting occurred

    def check_and_consume_food(self, arena):
        food_positions = arena.food_positions
        if food_positions.size == 0:
            return False

        distances = cdist([self.position], food_positions)[0]
        within_eating_radius = distances < self.eating_radius
        indices = np.where(within_eating_radius)[0]

        if self.asym_eating and len(indices) > 0:
            relative_positions = (
                food_positions[indices] - self.position
            )  # shape: (N, 2)
            angles_to_food = np.arctan2(
                relative_positions[:, 1], relative_positions[:, 0]
            )
            angle_diffs = (angles_to_food - self.orientation + np.pi) % (
                2 * np.pi
            ) - np.pi
            within_angle = np.abs(angle_diffs) <= self.eating_angle / 2
            indices = indices[within_angle]

        if len(indices) > 0:  # Check again in case angle filtering removed all food
            sorted_indices = indices[np.argsort(distances[indices])]
            max_food_to_eat = (
                min(self.max_food_eaten_per_step, len(sorted_indices))
                if self.max_food_eaten_per_step is not None
                else len(sorted_indices)
            )

            to_consume = sorted_indices[:max_food_to_eat]

            if self.probabilistic_eating and self.motion_order == 2:
                # filter out food probabilistically if moving too fast
                speed = np.linalg.norm(self.linear_velocity)
                max_speed = self.max_linear_velocity # set by movement.reset_agent_movement_parameters_to_baselines()
                # if velocity is 0, prob_eat = 1.0; if velocity is max_speed, prob_eat = 0.0
                prob_eat = max(0.0, 1.0 - speed / max_speed)
                to_consume = [
                    idx
                    for idx in to_consume
                    if self.np_random.uniform(0.0, 1.0) < prob_eat
                ]

            arena.eat_food(*to_consume)
            self.curr_food_consumed.extend(to_consume)
            self.energy += self.energy_food * len(to_consume)
            return True
        return False

    def _calculate_relative_positions(self, object_positions):
        relative_positions = object_positions - self.position
        distances = np.linalg.norm(relative_positions, axis=1)
        return relative_positions, distances

    def _get_angles_from_relative_positions(self, relative_positions):
        angles = (
            np.arctan2(relative_positions[:, 1], relative_positions[:, 0])
            - self.orientation
        )
        return (angles + np.pi) % (2 * np.pi) - np.pi

    def _update_sensors(self, sensor_array, max_range, sector_angle, sensed_objects):
        for distance, angle in sensed_objects:
            normalized_proximity = 1 - (distance / max_range)
            sensor_index = int((angle + np.pi) // sector_angle) % len(sensor_array)
            sensor_array[sensor_index] = max(
                sensor_array[sensor_index], normalized_proximity
            )
        return sensor_array

    def next_observation(
        self,
        feedback_action=True,
        enable_bite_action=True,
        use_bite_cooldown=True,
        agent_size_mode=None,
        feedback_displacement=False,
    ):
        # If actions are being fed back as observations
        action_observations = (
            self.last_action if feedback_action else [0] * self.num_actions
        )
        bounded_action_observations = np.tanh(action_observations)  # NOTE: In [-1, 1] due to tanh

        # second-order model only -- self-motion features
        vel_obs = [] # default for first-order model
        if self.motion_order == 2:
            lin_norm = self.linear_velocity / self.max_linear_velocity
            ang_norm = self.angular_velocity / self.max_angular_velocity
            vel_obs = [lin_norm, ang_norm]


        # flatten self.knollen_readings
        # knollen_observations = np.concatenate(
        #     [self.knollen_readings[i] for i in self.knollen_readings.keys()]
        # )
        # flatten self.knollen_metadata
        _meta_lists = [list(self.knollen_metadata[i].values()) for i in self.knollen_metadata.keys()]
        knollen_metadatas = np.concatenate(_meta_lists) if _meta_lists else np.array([])  # data is already plural, I know

        displacement_ego_obs = []
        if feedback_displacement:
            denom = self.max_linear_velocity if self.max_linear_velocity > 0 else 1.0
            displacement_ego_obs = np.clip(
                self.displacement_ego / denom, -1.0, 1.0
            ).flatten()

        observation = np.concatenate(
            [
                # self.position / self.arena_size,  # Normalized agent position
                # self.energy / self.energy_maximum,  # Normalized energy level
                np.array(self.mormyromast_observations_virtual).flatten(),
                np.array(self.ampullary_observations).flatten(),
                np.array(self.knollen_observations).flatten(),
                np.array(knollen_metadatas).flatten(),
                np.array(bounded_action_observations).flatten(),
                [0.0], # Skip fatigue for now
                (
                    [self.was_bitten] if enable_bite_action else []
                ),  # Don't include if no aggression
                [self.agent_size] if agent_size_mode is not None else [],
                (
                    [self.bite_cooldown_counter]
                    if (enable_bite_action and use_bite_cooldown)
                    else []
                ),
                displacement_ego_obs,  # Lateral-line like feedback; normalized to [-1,1]
                ([self.eat_cooldown_counter]),
                vel_obs, # second order model only
            ]
        )
        return observation


class BoundedRandomWalkerEFishAgent(EFishAgent):
    """
    An EFishAgent wrapper that makes bounded random walks and emits EODs according to a Poisson process.
    """

    def __init__(
        self,
        num_agents,
        arena_size,
        agent_id,
        bounding_radius,
        freeze,
        eod_rate=0.1,
        seed=None,
        **kwargs,
    ):
        # This agent assumes FIRST-ORDER motion semantics.
        # If used with motion_order == 2, commands will be interpreted as accelerations,
        # altering the random-walk statistics.
        self.center = np.array(arena_size) / 2  # Start at the center of the arena
        self.freeze = freeze
        self.motion_order = 1
        self.bounding_radius = bounding_radius
        self.eod_rate = eod_rate
        super().__init__(
            num_agents=num_agents,
            arena_size=arena_size,
            agent_id=agent_id,
            seed=seed,
            **kwargs,
        )
        # print("self.enable_bite_action", self.enable_bite_action)
        # print("allow_aggression", kwargs["allow_aggression"])

    def reset(self, seed=None):
        super().reset(seed)
        self.motion_order = 1
        self.position = self.center = np.array(self.arena_size) / 2
        self.trajectory = [self.position.copy()]
        if self.freeze:
            self.max_linear_velocity = 0.0
            self.max_angular_velocity = 0.0
        return None  # EFishAgent.reset() already did everything else

    def get_action(self):
        """Sample one bounded‐walk action triple (forward, turn, emit)."""
        rand_turn = self.np_random.uniform(-1.0, 1.0)
        rand_forward = self.np_random.uniform(0.0, 1.0)

        theta = self.orientation + rand_turn * self.max_angular_velocity
        delta = np.array([np.cos(theta), np.sin(theta)]) * rand_forward * self.max_linear_velocity
        cand_pos = self.position + delta

        within_bounds = np.linalg.norm(cand_pos - self.center) <= self.bounding_radius
        normalized_distance = (
            np.linalg.norm(cand_pos - self.center) / self.bounding_radius
        )

        # Make a random walk step with probability p_random
        # with probability (1-p), face back toward center
        p_random = 1 - normalized_distance
        if within_bounds and self.np_random.random() < p_random:
            forward, turn = rand_forward, rand_turn
            # print(f"d:{normalized_distance}, random walk (forward, turn):", forward, turn)
        else:
            # face back toward center
            vec_to_center = self.center - self.position
            desired_heading = math.atan2(vec_to_center[1], vec_to_center[0])
            # heading error in [-pi,pi]
            error = (desired_heading - self.orientation + np.pi) % (2 * np.pi) - np.pi
            # scale into [-1,1] for your action space
            # turn = float(np.clip(error / self.max_angular_velocity, -1.0, 1.0))
            turn = np.sign(error)  # make more pronounced
            # forward = 0 # Never exit the bounds
            # Gate forward on alignment with center: only move forward when facing
            # roughly toward center (cos(error) > 0), so the agent doesn't drift
            # further away while turning. When within_bounds the original p_random
            # >= 0 so the factor is 1.0 (backward-compatible).
            fwd_scale = p_random if within_bounds else abs(p_random) * max(0.0, float(np.cos(error)))
            forward = self.np_random.uniform(0.0, fwd_scale)
            # print(f"d:{normalized_distance}, turn back (forward, turn):", forward, turn)

        emit = (
            1.0 if self.np_random.random() < self.eod_rate else -1.0
        )  # -1 because passed into sigmoid in super().step()
        if self.enable_bite_action:
            actor_output = np.array([forward, turn, emit, -1.0], dtype=np.float32)
        else:
            actor_output = np.array([forward, turn, emit], dtype=np.float32)
        return actor_output

    def step(self, _action, arena, agent_objects):
        # _action ignores the neural-network output (uses hard-coded random walk)
        # print("Random walker location:", self.position, "center:", self.center)
        sampled_actor_output = self.get_action()
        super().step(sampled_actor_output, arena, agent_objects)

        # push us back to the nearest point on the circle if needed
        # offset = self.position - self.center
        # dist = np.linalg.norm(offset)
        # if dist > self.radius:
        #     self.position = self.center + offset * (self.radius / dist)


class MultiAgentFishEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", None],
        "name": "MultiAgentFishEnv_v1",
    }

    def __init__(
        self,
        all_args,
        reset_callback=None,
        reward_callback=None,
        observation_callback=None,
        info_callback=None,
        done_callback=None,
        post_step_callback=None,
        is_eval=False,
        seed=None,
    ):
        self.sensing_type = "electric"  # electric or rays
        self.all_args = all_args

        # Add homing mode flag and parameters
        self.homing_mode = all_args.get("homing_mode", False)
        self.always_freeze_agent0 = all_args.get("always_freeze_agent0", False)
        self.is_eval = is_eval
        self.homing_distance = all_args.get("homing_distance", 4.0)
        self.required_homing_steps = all_args.get("required_homing_steps", 10)
        self.homing_success_counter = 0
        self.pfeeder = all_args.get("pfeeder", 0)
        self.prandom = all_args.get("prandom", 0)
        self.urandom = all_args.get("urandom", 1)
        self.prob_n_patch = all_args.get("prob_n_patch", 0)
        self.feedback_action = all_args.get("feedback_action", True)
        self.feedback_displacement = all_args.get("feedback_displacement", True)
        self.multiplier_linear = all_args.get("multiplier_linear", 1.0)
        self.multiplier_angular = all_args.get("multiplier_angular", 1.0)
        self.size_speed_exponent = all_args.get("size_speed_exponent", 1.0)
        penalize_effort_over_frac = all_args.get("penalize_effort_over_frac", 1.0)
        if penalize_effort_over_frac is None:
            penalize_effort_over_frac = 1.0
        self.penalize_effort_over_frac = min(
            1.0, max(0.0, float(penalize_effort_over_frac))
        )
        self.use_curriculum = all_args.get("use_curriculum", False)
        self.curriculum_early_end_frac = all_args.get("curriculum_early_end_frac", 0.9)
        self.current_episode = 0

        self.max_food_sensing_radius = all_args.get("max_food_sensing_radius", None)
        if self.max_food_sensing_radius is not None:
            self.max_food_sensing_radius = max(
                self.max_food_sensing_radius,
                all_args.get("amp_food_detection_range_m", 0.0),
                all_args.get("morm_food_detection_range_m", 0.0),
            )

        if self.all_args.get("cfg_override", None) is not None:
            ENV_PARAMS.update(self.all_args["cfg_override"]["ENV_PARAMS"])
            AGENT_PARAMS.update(self.all_args["cfg_override"]["AGENT_PARAMS"])
            REWARDS.update(self.all_args["cfg_override"]["REWARDS"])
            OBJECT_TYPES.update(self.all_args["cfg_override"]["OBJECT_TYPES"])

        self.__dict__.update(ENV_PARAMS)
        self.arena_size_min = self.arena_size_min_cm
        self.arena_size_max = self.arena_size_max_cm

        # Tasks: foraging, homing, 2f1p (2-fish, 1 patch)
        self.task = all_args.get("task", "foraging")
        if self.homing_mode:
            self.task = "homing"  # TODO: bring into task arg
        print(
            f"[{self.task}]: Feedback action: {self.feedback_action}, displacement: {self.feedback_displacement}, "
            f"linearX: {self.multiplier_linear}, angularX: {self.multiplier_angular}, "
            f"sizeSpeedExp: {self.size_speed_exponent}"
        )

        # Task-specific overrides
        self.num_agents = all_args["num_agents"]  # default
        self.num_active_agents = self.num_agents  # default
        if self.task in ["homing"]:
            self.num_agents = 2
            self.active_agent_ids = [0, 1]
            self.num_active_agents = 2
            self.arena_size_min = ENV_PARAMS.get("homing_arena_size_min_cm", (20, 20))
            self.arena_size_max = ENV_PARAMS.get("homing_arena_size_max_cm", (90, 90))
            env_rank = all_args.get("env_rank", 0)
            raw_list = all_args.get("homing", []) if self.is_eval else []
            if raw_list:
                self.task_params_list = raw_list[env_rank] if env_rank < len(raw_list) else []
                # Base seed for this env: matches eval_fish.py `env_seed = all_args.seed * 50000 + rank * 10000`
                self._homing_eval_base_seed = all_args.get("seed", 1) * 50000 + env_rank * 10000
        elif self.task in ["2f1p", "2f1p_restricted"]:
            self.num_agents = all_args[
                "num_agents"
            ]  # Keep same number of agents as training (for eval to work)
            self.active_agent_ids = [0, 1]
            self.num_active_agents = 2
        elif self.task in ["2f1p_control_a"]:
            self.num_agents = all_args["num_agents"]
            self.active_agent_ids = [0]
            self.num_active_agents = 1
        elif self.task in ["2f1p_control_b"]:
            self.num_agents = all_args["num_agents"]
            self.active_agent_ids = [1]
            self.num_active_agents = 1
            # self.agent_size_ranges TODO
            # self.agent_size_preset = all_args.get("agent_size_preset", "A=B")
            # if self.agent_size_preset == "A>B":
            #     self.agent_size_ranges = [(0.75, 0.75), (0.25, 0.25)]
            # elif self.agent_size_preset == "A<B":
            #     self.agent_size_ranges = [(0.25, 0.25), (0.75, 0.75)]
            # else:  # "A=B"
            #     self.agent_size_ranges = [(0.5, 0.5), (0.5, 0.5)]
        elif self.task in ["2f1p_grid"]:
            # Keep same number of agents as training (for eval to work),
            # while activating the canonical two fish for this task family.
            self.num_agents = all_args["num_agents"]
            self.active_agent_ids = [0, 1]
            self.num_active_agents = 2
            self.task_params_list = all_args.get("2f1p_grid", [])
            if len(self.task_params_list) == 0:
                print("Warning: 2f1p_grid task has empty task_params_list")
            env_rank = all_args.get("env_rank", 0)
            if len(self.task_params_list) > env_rank:
                # Pick the task params corresponding to this env's rank
                self.task_params_list = self.task_params_list[env_rank]
            
        elif self.task in ["1rw1f1p"]:
            self.num_agents = all_args["num_agents"]  # Same as training
            self.active_agent_ids = [0, 1]
            self.num_active_agents = 2
            raw_list = all_args.get("1rw1f1p", None) or []
            env_rank = all_args.get("env_rank", 0)
            self.task_params_list = raw_list[env_rank] if len(raw_list) > env_rank else (raw_list if raw_list else None)
        else:
            self.num_agents = all_args["num_agents"]
            self.num_active_agents = all_args.get("num_active_agents", self.num_agents)
            if self.num_active_agents is None:
                self.num_active_agents = self.num_agents

            self.active_agent_ids = all_args.get(
                "active_agent_ids", list(range(self.num_active_agents))
            )

            # NOTE: I've noticed num_active_agents and active_agent_ids are slightly dependent on one another
            # num_active_agents generates a list of active agent IDs up to num if not specified explicitly
            # active_agent_ids might be specified at eval time, so num_active_agents depends on them
            self.num_active_agents = all_args.get("num_active_agents", self.num_agents)
            if self.num_active_agents is None:
                self.num_active_agents = len(self.active_agent_ids)
            assert self.num_active_agents == len(
                self.active_agent_ids
            ), "Number of active agents must match the list of active agent IDs."

            # if active agent ids are passed in explicitly,
            # override the default number of active agents
            self.num_active_agents = len(self.active_agent_ids)
            assert (
                self.num_active_agents <= self.num_agents
            ), "num_active_agents cannot be greater than num_agents"

        self.shared_reward = all_args.get("shared_reward", False)
        self.collective_sensing_mode = all_args.get("collective_sensing_mode", 1)
        self.expt_name = None  # TODO: why not being picked up?
        self.video_save_dir = (
            f"{all_args['run_dir']}/outputs" if "run_dir" in all_args else "./"
        )
        self.save_vid = all_args.get("save_vid", False)
        self.timestamp = all_args["timestamp"]
        self.sensing_model_type = all_args.get("sensing_model_type", "dynamic")
        self.knollen_mode = all_args.get("knollen_mode", 1)
        self.knollen_processing = all_args.get("knollen_processing", "binarize")
        self.ampullary_mode = all_args.get("ampullary_mode", 1)
        self.mormyromast_mode = all_args.get("mormyromast_mode", 1)
        self.knollen_metadata_mode = all_args.get("knollen_metadata_mode", "relative")
        self.morm_selfimage_mode = all_args.get("morm_selfimage_mode", 1)
        self.morm_consimage_mode = all_args.get("morm_consimage_mode", 1)
        self.ampullary_intrinsic_only = all_args.get(
            "ampullary_intrinsic_only", True
        )
        self.train_sensor_dropout_p = all_args.get(
            "train_sensor_dropout_p", 0.0
        )  # ablate sensors with this prob per episode
        self.train_food_scaling_min = all_args.get("train_food_scaling_min", 1.0)
        self.train_food_scaling_max = all_args.get("train_food_scaling_max", 1.0)
        self.train_food_scaling_type = all_args.get(
            "train_food_scaling_type", "log_uniform"
        )
        self.dist_perturbation = all_args.get("dist_perturbation", 0.0)
        self.food_orientation_drift = all_args.get("food_orientation_drift", 0.0)
        self.noise_frac_amp = all_args.get("noise_frac_amp", 0.05)
        self.noise_frac_amp_cons_eod = all_args.get("noise_frac_amp_cons_eod", 0.5)
        self.noise_frac_morm = all_args.get("noise_frac_morm", 0.05)
        self.noise_frac_knollen = all_args.get("noise_frac_knollen", 0.05)
        self.noise_frac_knollen_metadata = all_args.get("noise_frac_knollen_metadata", 0.05)
        self.max_food_eaten_per_step = all_args.get("max_food_eaten_per_step", 1)
        self.agent_size_mode = all_args.get("agent_size_mode", None)
        self.backwards = all_args.get("backwards", False)
        self.allow_aggression = all_args.get(
            "allow_aggression", 1
        )  # TODO: One of allow_aggression and enable_bite_action seems redundant
        self.enable_bite_action = all_args.get(
            "enable_bite_action", self.allow_aggression
        )
        self.use_bite_cooldown = all_args.get("use_bite_cooldown", True)
        self.eat_cooldown_rate = all_args.get("eat_cooldown_rate", None)
        self.probabilistic_eating = all_args.get("probabilistic_eating", False)

        self.mute_k = min(
            max(all_args.get("mute_k", 0), 0), len(self.active_agent_ids)
        )  # Mute K>=0 out of num_agents
        self.muted_agent_ids = []

        self.agent_size_ranges = self._get_agent_size_ranges()
        self.agent_size_sampling_mode = all_args.get("agent_size_sampling_mode", "uniform")
        self.arena_size = all_args.get(
            "arena_size", (200, 200)
        )  # temporary for initialization
        self.motion_order = all_args.get("motion_order", 1)
        assert self.motion_order in [1, 2], "motion_order must be 1 or 2"

        if "num_env_steps" in all_args and "max_episode_length" in all_args and "n_rollout_threads" in all_args:
            self.total_num_episodes = (
                all_args["num_env_steps"]
                // all_args["max_episode_length"]
                // all_args["n_rollout_threads"]
            )
            self.total_num_episodes = max(1, self.total_num_episodes)
        else:
            self.total_num_episodes = None


        self.agent_objects = []
        for agent_id in range(self.num_agents):
            common_kwargs = dict(
                num_agents=self.num_agents,
                arena_size=self.arena_size,
                agent_id=agent_id,
                motion_order=self.motion_order,
                # seed=self.np_random.integers(1e9),
                max_food_eaten_per_step=self.max_food_eaten_per_step,
                allow_aggression=self.allow_aggression,
                use_bite_cooldown=self.use_bite_cooldown,
                agent_size_min=self.agent_size_ranges[agent_id][0],
                agent_size_max=self.agent_size_ranges[agent_id][1],
                backwards=self.backwards,
                enable_bite_action=self.enable_bite_action,
                active=True if agent_id in self.active_agent_ids else False,
                multiplier_linear=self.multiplier_linear,
                multiplier_angular=self.multiplier_angular,
                size_speed_exponent=self.size_speed_exponent,
                probabilistic_eating=self.probabilistic_eating,
                agent_size_sampling_mode=self.agent_size_sampling_mode,
                ampullary_ema=all_args.get("ampullary_ema", False),
                eat_cooldown_rate=self.eat_cooldown_rate,
            )
            if self.task == "1rw1f1p" and agent_id == 0:
                # Use BoundedRandomWalkerEFishAgent for the random walker
                brw_args = common_kwargs.copy()
                brw_args["arena_size"] = ENV_PARAMS["1p_arena_size_cm"]
                # print("brw_args:", brw_args)
                agent = BoundedRandomWalkerEFishAgent(
                    **brw_args,
                    bounding_radius=ENV_PARAMS["1p_d_mean_cm"] / 2.0
                    + AGENT_PARAMS["morm_food_detection_range_m"]
                    * m_to_cm
                    / 2.0,  # Bounded walk radius
                    freeze=all_args.get("rw_freeze", 0),  # Whether the random walker is frozen in place
                    eod_rate=all_args.get("rw_eod_rate", 0.1),  # EOD emission rate
                )
            else:
                # Use EFishAgent for other agents
                agent = EFishAgent(**common_kwargs)
            self.agent_objects.append(agent)

        self.active_agent_objects = [
            agent
            for agent in self.agent_objects
            if agent.agent_id in self.active_agent_ids
        ]

        # Set args that are fixed for eval time
        if self.is_eval:
            self.all_args['p_init_closeby'] = 0.0  # disable closeby init at eval time

        # self.active_electric_measurements = np.zeros(
        #     (self.num_agents, self.agent_objects[0].num_mormyromast_sensors)
        # )

        # Define possible arenas, arena-specific kwargs and their probabilities,
        self.arena_mapping = {
            ar.UniformArena.__name__: ar.UniformArena,
            ar.PatchyArena.__name__: ar.PatchyArena,
            ar.FeederArena.__name__: ar.FeederArena,
            ar.NPatchArena.__name__: ar.NPatchArena,
            ar.NoFoodArena.__name__: ar.NoFoodArena,
        }
        sum_proportions = self.pfeeder + self.prandom + self.urandom + self.prob_n_patch
        self.arena_generators = [
            (ar.FeederArena.__name__, self.pfeeder / sum_proportions),
            (ar.PatchyArena.__name__, self.prandom / sum_proportions),
            (ar.UniformArena.__name__, self.urandom / sum_proportions),
            (ar.NPatchArena.__name__, self.prob_n_patch / sum_proportions),
        ]
        self.fixed_num_patches = all_args.get("fixed_num_patches", 1)
        if self.fixed_num_patches is None:
            self.fixed_num_patches = 1 # None can be passed in sometimes 

        self.render_mode = all_args.get("render_mode", None)
        self.renderer = FishRenderer(self)
        self.food_intrinsic_dipole_moment = cfg.ENV_PARAMS[
            "food_intrinsic_dipole_moment"
        ]

        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.post_step_callback = post_step_callback

        self.action_space = []
        self.observation_space = []
        for agent in range(self.num_agents):
            self.num_actions = 3  # move, turn, eod
            if self.enable_bite_action:
                self.num_actions += 1  # add biting to actions

            self.action_space.append(
                spaces.Box(
                    low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
                )
            )
            if self.observation_callback is not None:
                obs_dim = len(self.observation_callback(agent, self))
            else:
                # Use sensing-model-specific num_morm_sets so obs_space matches actual obs
                if self.sensing_model_type == 'frac':
                    _num_morm_sets = cfg.SENSING_PARAMS_FRAC["num_morm_sets"]
                else:
                    _num_morm_sets = cfg.SENSING_PARAMS_DYNAMIC["num_morm_sets"]
                _num_morm_virtual = self.agent_objects[0].num_rays * _num_morm_sets
                obs_dim = (
                    _num_morm_virtual  # voltages
                    + self.agent_objects[0].num_ampullary_sensors  # voltages
                    + (
                        self.agent_objects[0].num_knollen_sensors
                        * (self.num_agents - 1)
                    )  # [-1,0,1]
                    + (self.num_agents - 1)  # knollen metadatas [size]
                    + (self.num_actions if self.feedback_action else 0)
                    + 1  # fatigue
                    + (1 if self.enable_bite_action else 0)  # bitten
                    + (1 if self.agent_size_mode is not None else 0)  # agent self size
                    + (
                        1 if self.enable_bite_action and self.use_bite_cooldown else 0
                    )  # bite cooldown
                    + (
                        2 if self.feedback_displacement else 0
                    )  # last displacement ego (backwards compatible)
                    + (1)  # eat cooldown
                    + (2 if self.motion_order == 2 else 0) # velocity obs (second-order model only)
                )  # Agent position + rays + knollens + internal
            self.observation_space.append(
                # spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
                spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            )

        self.share_observation_space = [
            spaces.Box(
                # low=0.0, high=1.0, shape=(obs_dim * self.num_agents,), dtype=np.float32
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim * self.num_agents,),
                dtype=np.float32,
            )
            for _ in range(self.num_agents)
        ]

        self.reward_params = REWARDS
        self.FISH_CONSTANTS = FISH_CONSTANTS
        self.OBJECT_TYPES = OBJECT_TYPES

        # Multiply scalable food params according to base_food_multiplier
        self.base_food_multiplier = all_args.get("base_food_multiplier", 1.0)
        for param in SCALABLE_FOOD_PARAMS:
            if hasattr(self, param):
                original_value = getattr(self, param)
                if original_value is None:
                    continue
                scaled_value = original_value * self.base_food_multiplier
                setattr(self, param, scaled_value)
            else:
                print(f"Warning: {param} not found in environment attributes.")

        # Define arena-specific kwargs mapping
        self.arena_kwargs_mapping = {
            ar.UniformArena.__name__: {
                "reset_food_density": self.uniform_reset_food_density,
                "max_food_density": self.uniform_max_food_density,
                "step_food_density": self.uniform_step_food_density,
            },
            ar.PatchyArena.__name__: {
                "reset_food_density": self.patchy_reset_food_density,
                "max_food_density": self.patchy_max_food_density,
                "step_food_density": self.step_food_density,
                "reset_patch_density": self.reset_patch_density,
            },
            ar.FeederArena.__name__: {
                "reset_food_density": self.feeder_reset_food_density,
                "max_food_density": self.feeder_max_food_density,
                "step_food_density": self.step_food_density,
            },
            ar.NPatchArena.__name__: {
                "reset_food_density": self.npatch_reset_food_density,
                "max_food_density": self.npatch_max_food_density,
                "step_food_density": self.step_food_density,
                "fixed_num_patches": self.fixed_num_patches,
                "placement_radius_frac": all_args.get("placement_radius_frac", 0.75),
            },
        }

        if not hasattr(self, "arena_size_min"):
            self.arena_size_min = (
                self.arena_size_min[0] / 3,
                self.arena_size_min[1] / 3,
            )
        if not hasattr(self, "arena_size_bias_alpha"):
            self.arena_size_bias_alpha = 1

        self.common_arena_kwargs = {
            "step_food_decay": self.step_food_decay,
            "stockpile_density": self.stockpile_density,
            "drift": self.food_drift,
            "drag": self.food_drag,
            "orientation_drift": self.food_orientation_drift,
        }

        # TODO review and maybe refactor/clean up places where this is repeated
        self.sensing_params = SensingParams(
            knollen_metadata_mode=self.knollen_metadata_mode,
            collective_sensing_mode=self.collective_sensing_mode,
            ampullary_intrinsic_only=self.ampullary_intrinsic_only,
            morm_selfimage_mode=self.morm_selfimage_mode,
            morm_consimage_mode=self.morm_consimage_mode,
            noise_frac_morm=self.noise_frac_morm,
            noise_frac_amp=self.noise_frac_amp,
            noise_frac_amp_cons_eod=self.noise_frac_amp_cons_eod,
            noise_frac_knollen=self.noise_frac_knollen,
            noise_frac_knollen_metadata=self.noise_frac_knollen_metadata,
            ampullary_ema=all_args.get("ampullary_ema", False),
            amp_cons_eod_sign_only=all_args.get("amp_cons_eod_sign_only", False),
            max_food_sensing_radius=self.max_food_sensing_radius,
        )

        self.sensing_model = self._get_sensing_model()

        self.reset(seed=seed)

    def seed(self, seed):
        super().seed(seed)
        self.env_seed = seed  # used in render for tracking env seeds
        # update np_random
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        elif not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()

    def _get_sensing_model(self):
        if self.sensing_model_type == 'frac':
            return FracRandModel(self.sensing_params, self)
        elif self.sensing_model_type == 'dynamic':
            return DynamicBaselineModel(self.sensing_params, self)
        else:
            raise ValueError(f"Unknown sensing_model_type: {self.sensing_model_type}")

    def _apply_time_normalized_curriculum(self, debug=False):
        assert self.total_num_episodes is not None, "total_num_episodes must be specified for time_normalized_step curriculum."
        effective_total = self.total_num_episodes * self.curriculum_early_end_frac
        progress = (self.current_episode - 1) / effective_total
        progress = np.clip(progress, 0.0, 1.0)

        for agent in self.agent_objects:
            agent.eating_angle = agent.eating_angle_start + progress * (agent.eating_angle_end - agent.eating_angle_start)

        if debug:
            print(f"Curriculum progress: {progress:.4f}, eating_angle agent 0: {self.agent_objects[0].eating_angle:.4f}")

    def _apply_final_curriculum_values(self):
        for agent in self.agent_objects:
            agent.eating_angle = agent.eating_angle_end

    def _reset_arena_size(self):
        arena_size_width = sample_skewed_uniform(
            alpha=self.arena_size_bias_alpha,
            low=self.arena_size_min[0],
            high=self.arena_size_max[0],
            rng=self.np_random,
        )
        arena_size_height = sample_skewed_uniform(
            alpha=self.arena_size_bias_alpha,
            low=self.arena_size_min[1],
            high=self.arena_size_max[1],
            rng=self.np_random,
        )
        self.arena_size = (arena_size_width, arena_size_height)

    def resample_food_scaling_factor(self):
        # TODO: @sjohnsonyu to figure out how to log this properly
        if self.train_food_scaling_type == "uniform":
            # storing on object for ease of logging
            self.curr_food_scaling_factor = self.np_random.uniform(
                self.train_food_scaling_min, self.train_food_scaling_max
            )
        elif self.train_food_scaling_type == "log_uniform":
            self.curr_food_scaling_factor = np.exp(
                self.np_random.uniform(
                    np.log(self.train_food_scaling_min),
                    np.log(self.train_food_scaling_max),
                )
            )

    def reset(self, seed=None):
        # Reset internal variables
        if seed is not None or not hasattr(self, "np_random"):
            self.seed(seed)

        self.step_count = 0
        self.curr_time = 0
        self.homing_success_counter = 0
        if not hasattr(self, "current_episode") or self.current_episode is None:
            self.current_episode = 0

        if hasattr(self, "task_params_list") and self.task_params_list is not None:
            if hasattr(self, "_homing_eval_base_seed") and seed is not None:
                # Map eval episode seed → config index.
                # fish_runner_shared render() uses: episode_seed = base_seed + 197 * episode + 1
                # Mid-episode auto-resets (done=True) call reset(seed=None) and are skipped here.
                seed_offset = seed - self._homing_eval_base_seed - 1
                if seed_offset >= 0 and seed_offset % 197 == 0:
                    episode_idx = (seed_offset // 197) % len(self.task_params_list)
                    task_params = self.task_params_list[episode_idx]
                else:
                    task_params = None  # init reset or unrecognized seed → don't override positions
            else:
                # Non-homing tasks: legacy current_episode indexing
                episode_idx = int(self.current_episode) % len(self.task_params_list)
                task_params = self.task_params_list[episode_idx]
        else:
            task_params = None

        self._reset_arena_size()
        # Homing eval: fix arena size and agent positions from pre-calculated configs.
        # task_params: (arena_w, arena_h, target_x, target_y, target_ori, homer_x, homer_y, homer_ori)
        if self.is_eval and (self.task == "homing" or self.homing_mode) and task_params is not None:
            self.arena_size = (task_params[0], task_params[1])
        arena_names, probabilities = zip(*self.arena_generators)
        selected_arena_name = self.np_random.choice(arena_names, p=probabilities)

        # At train time, randomize food density and apply curriculum
        if not self.is_eval:
            if self.use_curriculum:
                self._apply_time_normalized_curriculum()

            episode_arena_kwargs_mapping = self.arena_kwargs_mapping[
                selected_arena_name
            ].copy()
            # Sample scaling factor independently for each attribute
            for key in [
                "reset_food_density",
                "max_food_density",
                "step_food_density",
                "reset_patch_density", # Also change the number of patches for PatchyArena
                ]:
                if key in episode_arena_kwargs_mapping:
                    self.resample_food_scaling_factor()
                    episode_arena_kwargs_mapping[key] *= self.curr_food_scaling_factor
        else:
            if self.use_curriculum:
                self._apply_final_curriculum_values()
            episode_arena_kwargs_mapping = self.arena_kwargs_mapping[
                selected_arena_name
            ]

        ## Setup Arena
        # Combine common kwargs with arena-specific kwargs
        arena_kwargs = {
            "min_arena_size": self.arena_size,
            "max_arena_size": self.arena_size,
            **self.common_arena_kwargs,
            **episode_arena_kwargs_mapping,
        }

        # Add any additional arena-specific parameters
        if selected_arena_name == ar.PatchyArena.__name__:
            arena_kwargs.update(
                {
                    "patch_d_mean": 12,
                    "patch_d_var": 3,
                    # "patch_d_mean": 8, # DEBUG: denser patches
                    # "patch_d_var": 2,
                    "reset_patch_density": self.reset_patch_density,
                    "step_patch_density": self.step_patch_density,
                    "max_patch_density": self.max_patch_density,
                }
            )
        elif selected_arena_name == ar.FeederArena.__name__:
            arena_kwargs.update(
                {
                    "patch_d_mean": 12,
                    "patch_d_var": 3,
                    "patches_per_edge": self.np_random.choice([1, 2]),
                }
            )
        elif selected_arena_name == ar.NPatchArena.__name__:
            arena_kwargs.update({"patch_radius": 6})

        if self.task == "homing" or self.homing_mode:
            self.arena = ar.NoFoodArena(
                min_arena_size=self.arena_size,
                max_arena_size=self.arena_size,
                # the rest are irrelevant to NoFoodArena but set to zeros for clarity
                reset_food_density=0.0,
                step_food_density=0.0,
                step_food_decay=0.0,
                max_food_density=0.0,
                stockpile_density=0.0,
                drift=0.0,
                drag=0.0,
                orientation_drift=0.0,
            )
        else:
            self.arena = self.arena_mapping[selected_arena_name](**arena_kwargs)

        # Override if in specific tasks
        if self.task in ("2f1p", "2f1p_control_a", "2f1p_control_b", "1rw1f1p", "2f1p_restricted", "2f1p_grid"):
            self.arena_size = ENV_PARAMS["1p_arena_size_cm"]  # fixed for all 1p exps
            self.arena = ar.OnePatchArena(
                min_arena_size=self.arena_size,
                max_arena_size=self.arena_size,
                reset_food_density=0.1,
                step_food_density=0.01,
                step_food_decay=0.0,
                max_food_density=0.3,
                stockpile_density=float("inf"),
                drift=0.0,
                drag=0.0,
                patch_d_mean=ENV_PARAMS["1p_d_mean_cm"],  # tune these as needed
                patch_d_var=0,
            )

        for i, agent in enumerate(self.agent_objects):
            # if seed is None:
            #    agent_seed = None
            # else:
            #    agent_seed = seed * 100003 + i * 10007
            agent_seed = self.np_random.integers(10e8)
            agent.arena_size = self.arena_size
            # Patch bot params before reset so freeze takes effect on velocities
            if self.task == "1rw1f1p" and task_params is not None and i == 0:
                agent.freeze = bool(task_params[3])
                agent.eod_rate = task_params[2]
            agent.reset(seed=agent_seed)

            # In p_init_closeby fraction of the time, place agents close to each other
            if self.all_args.get("p_init_closeby", 0.0) > 0.0:
                if self.np_random.uniform(0, 1) < self.all_args["p_init_closeby"]:
                    self._init_agents_in_corner(
                        self.agent_objects,
                        self.arena_size,
                        corner=None,  # Picks a random corner
                    )

            if (
                ("2f1p" in self.task or "1rw1f1p" in self.task)
                and i < 2
            ):
                if i == 0:
                    # NOTE: check to see if B is placed in the same location during the control.
                    # Calling this function anyways to keep use of rng consistent
                    new_position = self._get_position_within_annulus(
                        0, 0
                    )  # setting to the center for now!
                    if self.task in ("2f1p", "1rw1f1p", "2f1p_control_a", "2f1p_restricted", "2f1p_grid"):
                        agent.position = new_position
                        agent.trajectory = [new_position]

                    if self.task == "2f1p_grid":
                        agent.agent_size = task_params[0]  # task params (sizeA, sizeB, radius, theta)
                        agent.orientation = np.pi / 2  # face upwards for grid task
                    elif self.task == "1rw1f1p" and task_params is not None:
                        agent.agent_size = task_params[0]  # bot size
                        # Pin bot heading (+y / up); RL-fish start grid in
                        # eval_registry is laid out relative to this.
                        agent.orientation = np.pi / 2

                elif i == 1:
                    if self.task == "2f1p_restricted":
                        radii = [30, 60, 90]
                        radius = self.np_random.choice(radii)
                        new_position = self._get_position_within_annulus(radius, radius)
                    elif self.task in ("2f1p", "2f1p_control_b", "1rw1f1p"):
                        new_position = self._get_position_within_annulus(20, 100)
                        if self.task == "1rw1f1p" and task_params is not None:
                            agent.agent_size = task_params[1]  # RL fish size
                            if len(task_params) >= 6:
                                # Use pre-specified position (overrides random sample above).
                                new_position = np.array([task_params[4], task_params[5]])
                                self._rw_B_pos = new_position.copy()
                                self._rw_trial_id = int(task_params[6]) if len(task_params) >= 7 else None
                            else:
                                self._rw_B_pos = None
                                self._rw_trial_id = None
                    elif self.task == "2f1p_grid":
                        assert task_params is not None, "task_params must be provided for 2f1p_grid"
                        agent.agent_size = task_params[1]  # task params (sizeA, sizeB, radius, theta)
                        radius = task_params[2]
                        theta = task_params[3]
                        new_position = np.array([
                            self.arena_size[0] / 2 + radius * math.cos(theta),
                            self.arena_size[1] / 2 + radius * math.sin(theta),
                        ])

                    if self.task in ("2f1p", "1rw1f1p", "2f1p_control_b", "2f1p_restricted", "2f1p_grid"):
                        agent.position = new_position
                        agent.trajectory = [new_position]
                        # orient agent to face the center
                        center = np.array(self.arena_size, dtype=float) / 2.0
                        vec_to_center = center - new_position
                        heading = math.atan2(vec_to_center[1], vec_to_center[0])
                        agent.orientation = heading
                        if self.task == "1rw1f1p":
                            self._rw_B_orientation = float(heading)


        # Homing eval: apply pre-calculated episode positions when available (set via episode_configs).
        if self.is_eval and (self.task == "homing" or self.homing_mode) and task_params is not None:
            # task_params: (arena_w, arena_h, target_x, target_y, target_ori, homer_x, homer_y, homer_ori)
            self.agent_objects[0].position = np.array([task_params[2], task_params[3]])
            self.agent_objects[0].orientation = task_params[4]
            self.agent_objects[0].trajectory = [self.agent_objects[0].position.copy()]
            self.agent_objects[1].position = np.array([task_params[5], task_params[6]])
            self.agent_objects[1].orientation = task_params[7]
            self.agent_objects[1].trajectory = [self.agent_objects[1].position.copy()]

        # Override for fede01 task needs to be unaffected by above resets()
        if self.task == "fede01":
            center_location = "bottom_right"
            self.arena_size = (60, 60)
            self.arena = ar.SquareQuadrantArena(
                num_radial_patches=4,
                patch_radius=7.5,
                center_location=center_location,
                min_arena_size=self.arena_size,
                max_arena_size=self.arena_size,
                reset_food_density=0.01,
                step_food_density=0.001,
                step_food_decay=0.0,
                max_food_density=0.05,
                stockpile_density=1.0,
                orientation_drift=self.food_orientation_drift,
            )

            for agent in self.agent_objects:
                agent.arena_size = self.arena_size
                agent.reset(seed=self.np_random.integers(10e8))
            self._init_agents_in_corner(
                self.agent_objects, self.arena_size, corner=center_location
            )
        elif self.task == "n_patch":
            self.arena_size = (70, 70)
            self.arena = ar.NPatchArena(
                fixed_num_patches=self.fixed_num_patches,
                placement_radius_frac=0.75,
                min_arena_size=self.arena_size,
                max_arena_size=self.arena_size,
                patch_radius=7.5,
                orientation_drift=self.food_orientation_drift,
            )
            for agent in self.agent_objects:
                agent.arena_size = self.arena_size
                agent.reset(seed=self.np_random.integers(10e8))
        elif self.task == "n_patch_fixed":
            # Like n_patch but wires npatch food params through to NPatchArena AND
            # applies base_food_multiplier to per-patch density (not number-of-patches).
            # npatch_reset/max_food_density are excluded from SCALABLE_FOOD_PARAMS by
            # design (that list scales patch count, not per-patch density), so we apply
            # the multiplier explicitly here for controlled food-density experiments.
            self.arena_size = (70, 70)
            self.arena = ar.NPatchArena(
                fixed_num_patches=self.fixed_num_patches,
                placement_radius_frac=0.75,
                min_arena_size=self.arena_size,
                max_arena_size=self.arena_size,
                patch_radius=7.5,
                reset_food_density=self.npatch_reset_food_density * self.base_food_multiplier,
                step_food_density=0.0,
                step_food_decay=0.0,
                max_food_density=self.npatch_max_food_density * self.base_food_multiplier,
                stockpile_density=float("inf"),
                orientation_drift=self.food_orientation_drift,
            )
            for agent in self.agent_objects:
                agent.arena_size = self.arena_size
                agent.reset(seed=self.np_random.integers(10e8))
        elif self.task == "wide":
            # initialize agents at opposite ends of the 160, 40 arena, facing towards each other. agents shouldn't be at the walls but a little ways away
            self.arena_size = (160, 40)
            self.arena = ar.UniformArena(
                min_arena_size=self.arena_size,
                max_arena_size=self.arena_size,
                reset_food_density=self.uniform_reset_food_density,
                step_food_density=self.uniform_step_food_density,
                step_food_decay=0.0,
                max_food_density=self.uniform_max_food_density,
                stockpile_density=1.0,
                orientation_drift=self.food_orientation_drift,
            )
            for i, agent in enumerate(self.agent_objects):
                agent.arena_size = self.arena_size
                agent.reset(seed=self.np_random.integers(10e8))

                offset = 20
                if i == 0:
                    agent.position = np.array([offset, self.arena_size[1] / 2])
                    agent.orientation = 0.0
                elif i == 1:
                    agent.position = np.array(
                        [self.arena_size[0] - offset, self.arena_size[1] / 2]
                    )
                    agent.orientation = np.pi
                agent.trajectory = [agent.position.copy()]

        arena_seed = self.np_random.integers(10e8)
        self.arena.reset(seed=arena_seed)

        if self.homing_mode and (self.is_eval or self.always_freeze_agent0):
            self.agent_0_initial_position = self.agent_objects[0].position.copy()
            self.agent_objects[0].emit_eod = True  # Ensure EOD is always emitted

        knollen_mode = self.knollen_mode
        ampullary_mode = self.ampullary_mode
        mormyromast_mode = self.mormyromast_mode
        # ablate each agents M/A/K sensors if train_sensor_dropout_p > 0.0 (only one sensor type per episode)
        if (not self.is_eval) and (self.train_sensor_dropout_p > 0.0):
            # Draw one random number to decide *whether* to drop any sensor
            if self.np_random.uniform(0, 1) < self.train_sensor_dropout_p:
                # Choose exactly one sensor type to drop, uniformly at random
                sensor_to_drop = self.np_random.choice(["knollen", "ampullary", "mormyromast"])

                if sensor_to_drop == "knollen":
                    knollen_mode = 0
                elif sensor_to_drop == "ampullary":
                    ampullary_mode = 0
                elif sensor_to_drop == "mormyromast":
                    mormyromast_mode = 0

        self.sensor_modes_episode = SensorModesEpisode(
            knollen_mode=knollen_mode,
            knollen_processing=self.knollen_processing,
            ampullary_mode=ampullary_mode,
            mormyromast_mode=mormyromast_mode,
        )
        self.sensing_model.set_sensor_modes_episode(self.sensor_modes_episode)

        if self.mute_k > 0:
            if self.is_eval:
                # Deterministic muting during eval
                self.muted_agent_ids = sorted(self.active_agent_ids)[: self.mute_k]
            else:
                # Randomly mute up to k agents (if there are enough)
                num_to_mute = min(self.mute_k, len(self.active_agent_ids))
                self.muted_agent_ids = list(
                    self.np_random.choice(
                        self.active_agent_ids, size=num_to_mute, replace=False
                    )
                )
            for agent in self.agent_objects:
                agent.muted = (
                    agent.agent_id in self.muted_agent_ids
                )  # Init agent objects
            # print("Muted agent IDs:", self.muted_agent_ids)

        if self.reset_callback is not None:
            self.reset_callback(self)
        # need to update agent observations after reset
        # all_agents_sense(self)
        fish_info, food_info = self.get_fish_and_food_info()
        observation_tuple = self.sensing_model.sense_step(
            fish_info,
            food_info,
            self.active_agent_ids,
            self.arena_size,
            self.np_random
        )
        self.store_observations_on_agents(*observation_tuple)

        # observations = [self._next_observation(agent) for agent in self.agent_objects]
        observations = [
            agent.next_observation(
                feedback_action=self.feedback_action,
                enable_bite_action=self.enable_bite_action,
                use_bite_cooldown=self.use_bite_cooldown,
                agent_size_mode=self.agent_size_mode,
                feedback_displacement=self.feedback_displacement,
            )
            for agent in self.agent_objects
        ]
        # Increment the episode counter *after* selecting & applying task params
        # so the next reset() call will use the next index.
        self.current_episode += 1
        return observations

    def set_homing_mode(self):
        self.homing_mode = True

    def store_observations_on_agents(
            self,
            ampullary_obs,
            morm_obs,
            knollen_obs,
            center_fields,
            knollen_metadata,
        ):
        for i, agent in enumerate(self.agent_objects):
            agent.ampullary_observations = ampullary_obs[i]
            agent.mormyromast_observations_virtual = morm_obs[i]
            agent.knollen_observations = knollen_obs[i]
            agent.center_field.update({k: v[i] for k, v in center_fields.items()})
            agent.knollen_metadata = knollen_metadata[i]  # TODO double-check this is as expected

    def get_fish_and_food_info(self):
        fish_info = {}
        fish_info["fish_positions_cm"] = np.array([
            agent.position.copy() for agent in self.agent_objects
        ])
        fish_info["fish_orientations"] = np.array([
            agent.orientation for agent in self.agent_objects
        ])
        fish_info["fish_eods"] = np.array([agent.emit_eod for agent in self.agent_objects])
        fish_info["agent_sizes"] = np.array([agent.agent_size for agent in self.agent_objects])

        food_info = {}
        food_info["food_positions_cm"] = self.arena.food_positions.copy()
        num_food = food_info["food_positions_cm"].shape[0]
        try:
            food_orientations = self.arena.food_orientation_angles.copy()
        except AttributeError:
            food_orientations = np.zeros((num_food,), dtype=float)
        if food_orientations.shape[0] != num_food:
            food_orientations = np.zeros((num_food,), dtype=float)
        food_info["food_orientations"] = food_orientations
        food_info["food_radius"] = self.arena.food_radius
        return fish_info, food_info

    def step(self, actions):
        self.step_count += 1
        observations = []
        rewards = []
        terminations = []
        infos = [{} for _ in range(self.num_agents)]

        # handles food/patch growth and decay
        self.arena.step()

        # if self.is_eval:
        #! wrong as we have two sets of activation functions for backwards on and off!!!
        #     print("agent 0 fixed but emitting EOD")
        #     actions[0] = [0, 0, 1, 0] if self.allow_aggression else [0, 0, 1] #hacky way to ensure agent 0 is always emitting EOD
        #     # print("actions", actions)

        # First, process agent movement and eating
        for agent, action in zip(self.agent_objects, actions):
            # Homing
            if self.homing_mode and (self.is_eval or self.always_freeze_agent0) and agent.agent_id == 0:
                # For agent 0 in homing mode: (a hacky way to ensure agent 0 is always emitting EOD)
                # - Keep position and orientation fixed
                # - Force EOD on
                # - Update trajectory with current position
                # - Handle other necessary updates
                agent.emit_eod = True
                agent.trajectory.append(agent.position.copy())
                agent.trajectory = agent.trajectory[-5:]  # Keep only last 5 positions
                agent.eod_history.append(agent.emit_eod)
                agent.energy_history.append(agent.energy)
                agent.bite_history.append(False)
                agent.was_bitten_history.append(False)
                if agent.use_bite_cooldown:
                    agent.bite_cooldown_counter = max(
                        0.0, agent.bite_cooldown_counter - agent.bite_cooldown_rate
                    )
                    agent.bite_cooldown_history.append(agent.bite_cooldown_counter)
                continue
            # All tasks
            if agent.agent_id in self.active_agent_ids:
                agent.step(action, self.arena, self.agent_objects)

        # Next, process agent sensing and agent-agent interactions
        emit_eods = [agent.emit_eod for agent in self.agent_objects]
        if self.homing_mode and (self.is_eval or self.always_freeze_agent0):
            emit_eods[0] = True  # Agent 0 always emits EOD
            # emit_eods = [True, False]  # Agent 0 always emits, Agent 1 muted

        agents_emitting_eods = [
            agent for agent, emit_eod in zip(self.agent_objects, emit_eods) if emit_eod
        ]
        emitting_agent_positions = [agent.position for agent in agents_emitting_eods]
        # all_agents_sense(self)
        fish_info, food_info = self.get_fish_and_food_info()
        (
            ampullary_obs,
            morm_obs,
            knollen_obs,
            center_fields,
            knollen_metadata
        ) = self.sensing_model.sense_step(
            fish_info,
            food_info,
            self.active_agent_ids,
            self.arena_size,
            self.np_random
        )
        self.store_observations_on_agents(ampullary_obs, morm_obs, knollen_obs, center_fields, knollen_metadata)

        for agent in self.agent_objects:
            if self.allow_aggression and agent.bite_action:
                agent.bite_other_fish = agent.check_and_bite_nearest(self.agent_objects)

            if self.observation_callback is not None:
                observation = self.observation_callback(agent, self)

            agent.has_nearby = self._has_agents_nearby(agent)
            agent.nearby_emitting = self._has_emitting_agents_nearby(
                agent, agents_emitting_eods
            )
            self._update_nearby_emitting_agents_ids(agent, agents_emitting_eods)

            observation = agent.next_observation(
                feedback_action=self.feedback_action,
                enable_bite_action=self.enable_bite_action,
                use_bite_cooldown=self.use_bite_cooldown,
                agent_size_mode=self.agent_size_mode,
                feedback_displacement=self.feedback_displacement,
            )
            observations.append(observation)

        # Finally, process all agents to completion
        for agent in self.agent_objects:
            if self.reward_callback is not None:
                reward = self.reward_callback(agent, self)
            if self.homing_mode:
                if agent.agent_id in self.active_agent_ids:
                    reward = self._calculate_homing_reward(agent, self.arena)
                else:
                    reward = 0
                    agent.last_reward_components = {
                        "homing": 0.0,
                        "homing_shaping": 0.0,
                        "homing_time_penalty": 0.0,
                    }
            else:
                if agent.agent_id in self.active_agent_ids:
                    reward = self._calculate_foraging_reward(agent, self.arena)
                else:
                    reward = 0
                    agent.last_reward_components = {
                        "timestep": 0.0,
                        "food": 0.0,
                        "proximity_shaping": 0.0,
                        "bitten": 0.0,
                        "bite": 0.0,
                        "collision": 0.0,
                        "wall_proximity": 0.0,
                        "effort_over": 0.0,
                    }
            agent.cumulative_reward += reward
            rewards.append([reward])

            # done = agent.energy <= 0
            done = False  # NOTE manual override
            terminations.append(done)

            if self.info_callback is not None:
                info_agent = self.info_callback(agent, self)
                infos[agent.agent_id] = info_agent

            # Log per-step data
            infos[agent.agent_id].update(
                {
                    "reward_components": getattr(
                        agent, "last_reward_components", {}
                    ).copy(),
                    "position": agent.position.copy(),
                    "orientation": agent.orientation,
                    "energy": agent.energy,
                    "has_nearby": agent.has_nearby,
                    "nearby_emitting": agent.nearby_emitting,
                    "has_nearby_emit_eod_ids": agent.has_nearby_emit_eod_ids,
                    "was_bitten": agent.was_bitten,
                    "was_bitten_by_agent_id": agent.was_bitten_by_agent_id,
                    "bite_other_fish": agent.bite_other_fish,
                    "agent_size": agent.agent_size,
                    "collided": agent.collided,
                    # 'food_positions': self.arena.food_positions.copy()
                    "center_field": {
                        k: v.copy() for k, v in agent.center_field.items()
                    },
                    "angular_velocity": agent.angular_velocity,
                    "linear_velocity": agent.linear_velocity,
                    "active": agent.active,
                    "displacement_ground": agent.displacement_ground.copy(),
                    "displacement_ego": agent.displacement_ego.copy(),
                    "food_eaten_count": len(agent.curr_food_consumed),
                    "actions_postactivation": agent.last_action,
                    "emit_eod": agent.emit_eod,
                    "curr_time": self.curr_time,
                }
            )

            # Log per-episode data
            if agent.agent_id == 0:
                infos[agent.agent_id].update(
                    {
                        "food_positions": self.arena.food_positions.copy(),
                        "active_agent_ids": self.active_agent_ids,
                        "muted_agent_ids": self.muted_agent_ids,
                    }
                )
                if (
                    self.step_count == 1
                ):  # step_count==1 because updated at beginning of step
                    # print("Step 0, saving arena info")
                    infos[agent.agent_id].update(
                        {
                            "arena_type": self.arena.__class__.__name__,
                            "patch_kwargs": self.arena.patch_kwargs,
                            "arena_size": self.arena_size,
                        }
                    )
                    if self.task == "1rw1f1p" and agent.agent_id == 0:
                        bot = self.agent_objects[0]
                        entry = {
                            "rw_eod_A":   bot.eod_rate,
                            "rw_freeze_A": int(bot.freeze),
                            "rw_size_A":  bot.agent_size,
                            "rw_size_B":  self.agent_objects[1].agent_size,
                        }
                        B_pos = getattr(self, "_rw_B_pos", None)
                        if B_pos is not None:
                            entry["rw_B_x"] = float(B_pos[0])
                            entry["rw_B_y"] = float(B_pos[1])
                        trial_id = getattr(self, "_rw_trial_id", None)
                        if trial_id is not None:
                            entry["rw_trial_id"] = trial_id
                        B_orientation = getattr(self, "_rw_B_orientation", None)
                        if B_orientation is not None:
                            entry["rw_B_orientation"] = B_orientation
                        infos[0].update(entry)

        # End episode if run out of food
        ran_out_of_time = self.curr_time >= self.all_args["max_episode_length"]
        ran_out_of_food = len(self.arena.food_positions) == 0
        homed_for_enough_time = (
            self.homing_success_counter >= self.required_homing_steps
        )

        if ran_out_of_time:
            terminations = [True for _ in terminations]
            if self.done_callback is not None:
                self.done_callback(self)

            

        if self.task == "homing" and homed_for_enough_time:
            terminations = [True for _ in self.agent_objects]
            print(
                "At time step:",
                self.curr_time,
                "\tHoming success counter:",
                self.homing_success_counter,
            )

        if self.shared_reward:
            reward = np.sum(rewards)
            rewards = [[reward]] * self.num_agents

        if self.post_step_callback is not None:
            self.post_step_callback(self)

        self.curr_time += 1

        return observations, rewards, terminations, infos


    def _init_agents_in_corner(
        self, agents, arena_size, corner=None, spacing=(AGENT_PARAMS["body_radius_cm"] * 3)
    ):
        if corner is None:
            corner = self.np_random.choice(
                ["bottom_left", "bottom_right", "top_left", "top_right"]
            )
        assert corner in [
            "bottom_left",
            "bottom_right",
            "top_left",
            "top_right",
        ], f"Invalid corner specified {corner}"
        base_pos = {
            "bottom_left": np.array([5.0, 5.0]),
            "bottom_right": np.array([arena_size[0] - 5.0, 5.0]),
            "top_left": np.array([5.0, arena_size[1] - 5.0]),
            "top_right": np.array([arena_size[0] - 5.0, arena_size[1] - 5.0]),
        }[corner]

        # TODO: maybe add more heading options
        heading = {
            "bottom_left": np.pi / 2,  # Upward facing
            "bottom_right": np.pi / 2,  # Upward facing
            "top_left": -np.pi / 2,  # Downward facing
            "top_right": -np.pi / 2,  # Downward facing
        }[corner]

        for i, agent in enumerate(agents):
            offset = np.array([i % 2, i // 2]) * spacing
            agent.position = base_pos + offset
            agent.orientation = heading
            agent.trajectory = [agent.position.copy()]

    def _get_position_within_annulus(self, inner_radius, outer_radius):
        center = np.array(self.arena_size, dtype=float) / 2.0
        theta = self.np_random.uniform(0, 2 * np.pi)
        u = self.np_random.uniform()
        r = np.sqrt(u * (outer_radius**2 - inner_radius**2) + inner_radius**2)
        position = center + np.array([r * np.cos(theta), r * np.sin(theta)])
        return position

    def _has_emitting_agents_nearby(self, agent, agents_emitting_eods):
        other_emitting_agents_positions = np.array(
            [
                other_agent.position
                for other_agent in agents_emitting_eods
                if other_agent != agent
            ]
        )
        if len(other_emitting_agents_positions) == 0:
            return False
        agent_distances = cdist([agent.position], other_emitting_agents_positions)[0]
        return np.any(
            agent_distances <= (agent.morm_agent_detection_range * m_to_cm)
        )  #

    def _update_nearby_emitting_agents_ids(self, agent, agents_emitting_eods):
        agent.has_nearby_emit_eod_ids = []  # Reset list
        other_emitting_agents = [
            other_agent for other_agent in agents_emitting_eods if other_agent != agent
        ]

        if len(other_emitting_agents) == 0:
            return False

        for other_agent in other_emitting_agents:
            distance = np.linalg.norm(agent.position - other_agent.position)
            if distance <= (agent.morm_agent_detection_range * m_to_cm):
                agent.has_nearby_emit_eod_ids.append(other_agent.agent_id)

        return len(agent.has_nearby_emit_eod_ids) > 0

    def _has_agents_nearby(self, agent):
        others = [o.position for o in self.agent_objects if o != agent]
        if not others:
            return False
        agent_distances = cdist([agent.position], others)[0]
        return np.any(agent_distances <= (agent.morm_agent_detection_range * m_to_cm))

    def _calculate_effort_over_penalty(self, agent):
        if self.penalize_effort_over_frac >= 1.0:
            return 0.0
        move_over = max(0.0, abs(agent.move_forward) - self.penalize_effort_over_frac)
        turn_over = max(0.0, abs(agent.turn_angle) - self.penalize_effort_over_frac)
        if move_over == 0.0 and turn_over == 0.0:
            return 0.0
        return REWARDS["effort_over"] * (move_over + turn_over)

    def _calculate_homing_reward(self, agent, arena):
        if agent.agent_id == 0 and (self.is_eval or self.always_freeze_agent0):  # Target agent
            agent.last_reward_components = {
                "homing": 0.0,
                "homing_shaping": 0.0,
                "homing_time_penalty": 0.0,
            }
            return 0  # No reward for stationary agent

        reward = 0
        components = {
            "homing": 0.0,
            "homing_shaping": 0.0,
            "homing_time_penalty": 0.0,
        }

        # Calculate distances to other agents
        other_agents = [other for other in self.agent_objects if other != agent]
        distances = [
            np.linalg.norm(agent.position - other.position) for other in other_agents
        ]
        nearest_agent_distance = min(distances) if distances else float("inf")

        # Binary reward for being within homing distance
        if nearest_agent_distance <= self.homing_distance:
            reward += REWARDS["homing"]
            components["homing"] += REWARDS["homing"]
            self.homing_success_counter += 1
        else:
            self.homing_success_counter = 0

        # Shaping: Minor reward for getting closer to other agents
        if agent.previous_agent_distance is not None:
            shaping = (
                REWARDS["homing_shaping"]
                * (agent.previous_agent_distance - nearest_agent_distance)
                / np.sum(arena.arena_size)
            )
            reward += shaping
            components["homing_shaping"] += shaping
        agent.previous_agent_distance = nearest_agent_distance

        # small time penalty
        reward += REWARDS["homing_time_penalty"]
        components["homing_time_penalty"] += REWARDS["homing_time_penalty"]

        agent.last_reward_components = components
        return reward

    def _calculate_foraging_reward(self, agent, arena):
        reward = 0
        components = {
            "timestep": 0.0,
            "food": 0.0,
            "proximity_shaping": 0.0,
            "bitten": 0.0,
            "bite": 0.0,
            "collision": 0.0,
            "wall_proximity": 0.0,
            "effort_over": 0.0,
        }
        reward += REWARDS["timestep"]  # Small negative reward each timestep to encourage efficiency
        components["timestep"] += REWARDS["timestep"]
        # Food/eating related
        food_positions = self.arena.food_positions
        if food_positions.size == 0:  # No food remaining
            pass # reward already set to 0 above
        else:
            # Main reward
            food_reward = REWARDS["eat"] * len(agent.curr_food_consumed)
            reward += food_reward
            components["food"] += food_reward

            # Shaping: Minor reward for getting closer to food
            food_distances = cdist([agent.position], food_positions)[0]
            nearest_food_distance = np.min(food_distances)
            if agent.previous_food_distance is not None:
                shaping = (
                    REWARDS["proximity_shaping"]
                    * (agent.previous_food_distance - nearest_food_distance)
                    / np.sum(arena.arena_size)
                )
                reward += shaping
                components["proximity_shaping"] += shaping
            agent.previous_food_distance = nearest_food_distance

        # Bitten penalty
        if agent.was_bitten:
            if self.agent_size_mode is None:
                reward += REWARDS["bitten"]
                components["bitten"] += REWARDS["bitten"]
            else:
                # REWARDS["bitten"] * (1 + size_difference)
                # REWARDS["bitten"] is a negative value
                # size_difference: negative (up to -1) if biting agent smaller, positive if bigger (up to +1)
                # (1 + size_difference) scales from 0 (bitten by much smaller) to 2 (bitten by much bigger)
                biting_agent_size = self.agent_objects[
                    agent.was_bitten_by_agent_id
                ].agent_size
                size_difference = biting_agent_size - agent.agent_size
                bitten_reward = REWARDS["bitten"] * (1 + size_difference)
                reward += bitten_reward
                components["bitten"] += bitten_reward

        bite_reward = REWARDS["bite"] if agent.bite_other_fish else 0
        reward += bite_reward  # temporary cost for biting
        components["bite"] += bite_reward

        # Internal variables
        # reward += -1 if agent.energy <= self.energy_minimum else 0
        # Add collision penalty
        if agent.collided:
            reward += REWARDS["collision"]  # Same penalty as MAForageRays
            components["collision"] += REWARDS["collision"]

        # Negative reward for coming too close to an arena boundary
        distance_to_wall_r = min(
            agent.position[0],  # Left wall
            arena.arena_size[0] - agent.position[0],  # Right wall
            agent.position[1],  # Bottom wall
            arena.arena_size[1] - agent.position[1],  # Top wall
        )
        distance_to_wall_r = max(0.5, distance_to_wall_r)  # can't go below
        wall_threshold = 5  # cm
        if distance_to_wall_r < wall_threshold:
            wall_reward = REWARDS["wall_proximity"] * (
                1 - distance_to_wall_r / wall_threshold
            )
            reward += wall_reward
            components["wall_proximity"] += wall_reward

        effort_penalty = self._calculate_effort_over_penalty(agent)
        reward += effort_penalty
        components["effort_over"] += effort_penalty

        agent.last_reward_components = components
        return reward

    def _get_agent_size_ranges(self):
        if self.agent_size_mode in (None, "random"):
            agent_sizes = [(0, 1)] * self.num_agents
        elif self.agent_size_mode == "hierarchy":
            # Divide the range [0, 1] into non-overlapping segments based on the number of agents
            interval_size = 1.0 / self.num_agents
            # Assign intervals in descending order so agent 0 is biggest
            agent_sizes = [
                (1 - (i + 1) * interval_size, 1 - i * interval_size)
                for i in range(self.num_agents)
            ]
        elif self.agent_size_mode == "egalitarian" or self.agent_size_mode == "AeqB":
            # set all agents to 0.5
            agent_sizes = [(0.5, 0.5)] * self.num_agents
        elif self.agent_size_mode == "boss":
            # set agent 0 to 1, all others to 0
            agent_sizes = [(1, 1)] + [(0, 0)] * (self.num_agents - 1)
        elif self.agent_size_mode == "AltB":
            # NOTE: if running 2f1p with a non-None agent_size_mode, model must have
            # been trained using some agent_size_mode (with aggression activated).
            # Otherwise, there will be an observation dimension mismatch.
            agent_sizes = (
                [(0.25, 0.25)] + [(0.75, 0.75)] + [(0, 0)] * (self.num_agents - 2)
            )
        elif self.agent_size_mode == "AgtB":
            agent_sizes = (
                [(0.75, 0.75)] + [(0.25, 0.25)] + [(0, 0)] * (self.num_agents - 2)
            )
        elif self.agent_size_mode == "fixed":
            agent_sizes = [(1, 1)] + [(0, 0)] + [(0.5, 0.5)] * (self.num_agents - 2)
        elif self.agent_size_mode == "A5ltB":
            agent_sizes = [(0.5, 0.5)] + [(1.0, 1.0)] + [(0, 0)] * (self.num_agents - 2)
        elif self.agent_size_mode == "A5gtB":
            agent_sizes = [(0.5, 0.5)] + [(0, 0)] + [(0, 0)] * (self.num_agents - 2)
        return agent_sizes

## --------------------- Visualization  ---------------------------- ##

    def render(self, mode=None, auxs=None, attentions=None):
        if auxs is None:
            auxs = ["eods", "spis"]
        return self.renderer.render(mode=mode, auxs=auxs, attentions=attentions)

    def save_svg_snapshot(self, filepath, env_index=None):
        return self.renderer.save_svg_snapshot(filepath, env_index=env_index)

    def close(self):
        return self.renderer.close()


## --------------------- TESTS  ---------------------------- ##
class TestTurnsAgent:
    def __init__(self, max_angular_velocity, allow_aggression, agent_id):
        self.max_angular_velocity = max_angular_velocity
        self.turn_direction = 1
        # self.steps_in_current_direction = agent_id % 4
        self.steps_in_current_direction = 0
        self.allow_aggression = allow_aggression
        self.agent_id = agent_id

    def get_action(self):
        if self.steps_in_current_direction >= 8:
            self.turn_direction *= -1
            self.steps_in_current_direction = 0
        self.steps_in_current_direction += 1

        move_forward = 1
        turn_angle = self.turn_direction * self.max_angular_velocity
        eod_action = 1 if self.steps_in_current_direction == (self.agent_id + 1) else -1
        # eod_action = -1  # disable EOD, for debugging ampullary only
        bite_action = (
            True if self.steps_in_current_direction % 5 == 0 else False
        )  # Bite every 5 steps for testing

        if self.allow_aggression:
            return np.array([move_forward, turn_angle, eod_action, bite_action])
        else:
            return np.array([move_forward, turn_angle, eod_action])


def get_brw_agent(all_args, agent_id=0):
    common_kwargs = dict(
        num_agents=all_args["num_agents"],
        arena_size=all_args["arena_size_min"],
        agent_id=agent_id,
        max_food_eaten_per_step=all_args["max_food_eaten_per_step"],
        allow_aggression=all_args["allow_aggression"],
        agent_size_min=0,
        agent_size_max=1,
        backwards=all_args["backwards"],
        enable_bite_action=all_args["allow_aggression"],
        active=True,
    )
    # Use BoundedRandomWalkerEFishAgent for the random walker
    brw_args = common_kwargs.copy()
    brw_args["arena_size"] = ENV_PARAMS["1p_arena_size_cm"]
    # print("brw_args:", brw_args)
    agent = BoundedRandomWalkerEFishAgent(
        **brw_args,
        bounding_radius=ENV_PARAMS["1p_d_mean_cm"] / 2.0,  # Bounded walk radius
        eod_rate=all_args["rw_eod_rate"],
        seed=1,
        freeze=all_args.get("rw_freeze", 0)
    )
    return agent


def without_render(all_args, NUM_STEPS=1000, observe=False, seed=42):

    # Without render/observe profiling
    start_time = time.time()

    env = MultiAgentFishEnv(all_args, seed=seed)
    hardcoded_agents = [
        TestTurnsAgent(
            max_angular_velocity=1, allow_aggression=all_args["allow_aggression"], agent_id=i
        )
        for i in range(env.num_agents)
    ]
    if all_args["task"] == "1rw1f1p":
        hardcoded_agents[0] = get_brw_agent(all_args)
        print("Using BoundedRandomWalkerEFishAgent for agent 0")

    if observe:
        from onpolicy.instrument.observer import Observer
        observer = Observer()  # Initialize the observer

    for step in tqdm.tqdm(range(NUM_STEPS)):
        actions = np.array(
            [agent.get_action() for agent in hardcoded_agents], dtype=object
        )  # Produces VisibleDeprecationWarning
        observations, rewards, terminations, infos = env.step(actions)

        if observe:
            observer.record(
                episode_index=0,
                time_step=step,
                data={
                    "agent_id": list(range(env.num_agents)),
                    "actions": actions,
                    "observations": observations,
                    "rnn_states": env.np_random.standard_normal(64).tolist(),
                    "rewards": rewards,
                    "infos": infos,
                },
            )
    end_time = time.time()
    print(
        f"Without render/observer time for {NUM_STEPS} steps: {end_time - start_time} seconds"
    )
    if observe:
        outfile = "observer_data_raw.pkl"
        observer.save(outfile)
        print("Saved", outfile)


def with_render(all_args, observe=False, NUM_STEPS=25, seed=None):
    # With render/observe profiling
    start_time = time.time()
    env = MultiAgentFishEnv(all_args, seed=seed)
    env.fps_video = 5
    hardcoded_agents = [
        TestTurnsAgent(
            max_angular_velocity=1, allow_aggression=all_args["allow_aggression"], agent_id=i
        )
        for i in range(env.num_agents)
    ]
    if all_args["task"] == "1rw1f1p":
        hardcoded_agents[0] = get_brw_agent(all_args)
        print("Using BoundedRandomWalkerEFishAgent for agent 0")

    if observe:
        from onpolicy.instrument.observer import Observer

        observer = Observer()  # Initialize the observer

    for step in tqdm.tqdm(range(NUM_STEPS)):
        actions = [agent.get_action() for agent in hardcoded_agents]
        # print([a.shape for a in actions])  # Debugging: print action shapes
        actions = np.array(actions)
        observations, rewards, terminations, infos = env.step(actions)

        if observe:
            observer.record(
                episode_index=0,
                time_step=step,
                data={
                    "agent_id": list(range(env.num_agents)),
                    "actions": actions,
                    "observations": observations,
                    "rnn_states": env.np_random.standard_normal(64).tolist(),
                    "rewards": rewards,
                    "infos": infos,
                },
            )

        if all_args["render_mode"] is not None:
            env.render(mode=all_args["render_mode"])
        if np.all(terminations):
            break

    outfile = env.close()

    if observe:
        outfile = outfile.replace(".mp4", ".pkl")
        observer.save(outfile)
        print("Saved", outfile)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


def profile_code():
    profiler = cProfile.Profile()
    profiler.enable()
    without_render(all_args, NUM_STEPS=200, observe=False)  # or with_render(all_args)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats()

def profile_loop(all_args, num_steps=200, sort="cumtime", out_dir="./"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prof_path = Path(out_dir) / f"profile_MAEFish_{all_args.get('task','task')}_{num_steps}steps_{stamp}.out"

    pr = cProfile.Profile()
    pr.enable()

    without_render(all_args, NUM_STEPS=num_steps, observe=False, seed=42)
    # or: with_render(all_args, NUM_STEPS=40, observe=False, seed=1234)

    pr.disable()
    pr.dump_stats(str(prof_path))

    stats = pstats.Stats(pr).strip_dirs().sort_stats(sort)
    stats.print_stats(50)          # top 50 functions
    stats.print_callers(20)        # who calls the hot stuff
    stats.print_callees(20)        # what hot stuff calls

    print(f"Wrote: {prof_path}")
    return prof_path


def plot_arena_by_density():
    np.random.seed(1234)
    random.seed(1234)
    for scale in cfg.FOOD_DENSITY_PRESETS.keys():
        food_density_params = cfg.FOOD_DENSITY_PRESETS[scale]
        for key, value in food_density_params.items():
            try:
                all_args[key] = value
            except Exception as e:
                print(e)
        # without_render(all_args, NUM_STEPS=200, observe=False)
        all_args.update({"timestamp": scale})
        with_render(all_args, NUM_STEPS=5, seed=1234)


if __name__ == "__main__":
    all_args = {
        # "num_agents": 2,
        "arena_size_min": (50, 150),
        "arena_size_max": (50, 150),
        "num_agents": 3,
        # "agent_size_mode": "hierarchy",
        # "arena_size_min": (100, 100),
        # "arena_size_max": (100, 100),
        "energy_food": 20,
        # "render_mode": None,
        "render_mode": "rgb_array",
        "food_drift": 0.01,
        "food_drag": 0.2,
        "collective_sensing_mode": 1,  # "simple_extension",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "max_food_eaten_per_step": 1,
        "feedback_action": True,  # Enable action feedback
        "feedback_displacement": True,  # Enable displacement_ego feedback for testing
        "pfeeder": 0,
        "prandom": 1,
        "urandom": 0,
        "prob_n_patch": 0,
        "save_vid": True,
        "allow_aggression": 1,  # TODO: re-enable
        "knollen_mode": 1,
        "ampullary_mode": 1,
        "mormyromast_mode": 1,
        "morm_selfimage_mode": 1,
        "morm_consimage_mode": 1,
        "sensing_model_type": "dynamic",
        "backwards": True,
        "max_episode_length": 1200,
        "homing_distance": 4.0,  # Distance threshold for successful homing
        "required_homing_steps": 20,  # Steps needed for episode completion
        "homing_mode": False,
        "is_eval": False,
        "rw_eod_rate": 0.2,  # EOD rate for the random walker agent
        "rw_freeze": 1,  # freeze random walker movement (but not EOD)
        "task": ["foraging", "homing", "2f1p", "1rw1f1p", "fede01", "n_patch", "wide", "2f1p_restricted"][
            0
        ],
        "p_init_closeby": 0.0,
        "ampullary_intrinsic_only": True,
        "ampullary_ema": False,
        "max_food_sensing_radius": 15,
        "base_food_multiplier": 1.0,  # multiplier for food abundance
        # "auxs": [
        #     "eods",
        #     # "spis",
        #     # "observations",
        #     # "attentions",  # Must be called via train/eval (because needs a trained network)
        # ],
        "auxs": None,
        "multiplier_linear": 2.0,
        "multiplier_angular": 3.0,
    }

    # for scale in [0.5, 1, 2]:
    for scale in [1]:
        seed=1
        np.random.seed(seed)
        random.seed(seed)
        prandom = all_args["prandom"]
        urandom = all_args["urandom"]

        all_args.update({"timestamp": scale})
        all_args.update({"base_food_multiplier": scale})

        all_args.update({"timestamp": f"{scale}_p{prandom}u{urandom}"})
        print("task:", all_args["task"])
        with_render(all_args, NUM_STEPS=100, seed=1234)
        # without_render(all_args, NUM_STEPS=200, observe=False, seed=42)
        # without_render(all_args, NUM_STEPS=100, observe=True, seed=42)
    # profile_code()
    # profile_loop(all_args, num_steps=500, sort="tottime")
