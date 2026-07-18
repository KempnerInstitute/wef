import sys, os; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import arena
import electric as electric
import cfg
from cfg import AGENT_PARAMS, ENV_PARAMS, ELECTRIC_CONSTANTS, cm_to_m, m_to_cm
from matplotlib import contour
from MAEFish import EFishAgent


#### ------------------------- Collective Sensing Calibration ------------------------- ####
class SimpleEFishAgent:
    def __init__(
        self,
        position,
        orientation,
        body_radius_m=None,
        monopole_charges=None,
        monopole_positions_ego=None,
        mormyromast_angles=None,
    ):

        self.position = np.asarray(position)
        self.orientation = orientation

        self.body_radius_m = body_radius_m or AGENT_PARAMS["body_radius_cm"] * cm_to_m
        self.monopole_charges = monopole_charges or AGENT_PARAMS["monopole_charges"]

        if monopole_positions_ego is None:
            d = self.body_radius_m
            self.monopole_positions_ego_m = np.array([[d / 2, 0], [-d / 2, 0]])
        else:
            self.monopole_positions_ego_m = monopole_positions_ego

        self.monopole_positions_ego = self.monopole_positions_ego_m * m_to_cm

        if mormyromast_angles is None:
            num_rays = AGENT_PARAMS["num_rays"]
            self.mormyromast_angles = np.linspace(
                -np.pi, np.pi, num_rays, endpoint=False
            )
        else:
            self.mormyromast_angles = mormyromast_angles

        self.num_mormyromast_sensors = len(self.mormyromast_angles)
        self.mormyromast_positions_ego_m = np.stack(
            [
                self.body_radius_m
                * np.column_stack(
                    (np.cos(self.mormyromast_angles), np.sin(self.mormyromast_angles))
                )
            ]
        )[0]
        self.mormyromast_positions_ego = self.mormyromast_positions_ego_m * m_to_cm
        self.mormyromast_normals_ego = np.column_stack(
            (np.cos(self.mormyromast_angles), np.sin(self.mormyromast_angles))
        )

    def get_world_monopole_positions(self):
        theta = self.orientation
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return (R @ self.monopole_positions_ego_m.T).T * m_to_cm + self.position


class RealSimpleEFishAgent(EFishAgent):
    def __init__(self, position, orientation, arena_size=(100, 100)):
        super().__init__(
            num_agents=0,
            arena_size=arena_size,
            max_step=ENV_PARAMS["max_step"],
            max_turn=ENV_PARAMS["max_turn"],
            agent_id=0,
        )
        self.position = np.asarray(position)
        self.orientation = orientation
        (
            self.mormyromast_cd,
            self.ampullary_cd,
            self.ampullary_intrinsic_baseline,
            self.center_mormyromast_cd,
            self.center_ampullary_cd,
            self.center_ampullary_intrinsic_baseline,
        ) = self._calculate_corollary_discharge()
        self.body_radius_m = AGENT_PARAMS["body_radius_cm"] * cm_to_m
        self.monopole_charges = AGENT_PARAMS["monopole_charges"]
        # NOTE: to be consistent with SimpleEFishAgent, need to set asym_rays to False in cfg
        # this must be done in cfg since it sets up the mormyromast angles in init.
        # don't do it here.

    def get_world_monopole_positions(self):
        theta = self.orientation
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        print("Monopole positions in ego frame:", self.monopole_positions_ego)
        print(
            "Monopole positions in world frame:",
            (R @ self.monopole_positions_ego.T).T + self.position,
        )
        return (R @ self.monopole_positions_ego.T).T + self.position

def angular_residual(readings, sensor_angles):
    """
    readings: (N_sensors,) current mormyromast (dot-with-normal) readings
    sensor_angles: (N_sensors,) angles of those sensors (ego frame)
    Returns: baseline_hat, residual
    """
    import numpy as np
    X = np.column_stack([np.ones_like(sensor_angles),
                         np.cos(sensor_angles), np.sin(sensor_angles)])  # [1, cosθ, sinθ]
    beta, *_ = np.linalg.lstsq(X, readings, rcond=None)
    baseline_hat = X @ beta
    residual = readings - baseline_hat
    return baseline_hat, residual

def angular_residual_log(readings, sensor_angles, eps=None):
    """
    Fit the angular baseline in log10(|reading|) space, then return multiplicative residuals.

    readings: (N_sensors,) signed mormyromast readings (dot-with-normal)
    sensor_angles: (N_sensors,) angles at which sensors are placed (radians)
    eps: optional floor for |reading| before log; if None, picked robustly from data

    Returns
    -------
    baseline_hat_log : (N_sensors,) fitted baseline in log10(|reading|)
    residual_log     : (N_sensors,) log residual = log10(|reading|) - baseline_hat_log
    ratio            : (N_sensors,) multiplicative factor = 10**residual_log
                       ( <1 => dip, >1 => bump relative to local angular baseline )
    rel_change       : (N_sensors,) ratio - 1  (negative = dip, positive = bump)
    """
    import numpy as np

    # Robust epsilon so zeros/very small magnitudes don't blow up log
    abs_read = np.abs(readings)
    if eps is None:
        # Use a small fraction of a low percentile of nonzero magnitudes
        nz = abs_read[abs_read > 0]
        if nz.size == 0:
            eps = 1e-12
        else:
            eps = max(1e-12, 0.1 * np.percentile(nz, 1))
    log_mag = np.log10(np.clip(abs_read, eps, None))

    # First-harmonic fit: b0 + b1 cosθ + b2 sinθ
    X = np.column_stack([np.ones_like(sensor_angles),
                         np.cos(sensor_angles), np.sin(sensor_angles)])
    beta, *_ = np.linalg.lstsq(X, log_mag, rcond=None)
    baseline_hat_log = X @ beta

    residual_log = log_mag - baseline_hat_log          # additive in log-space
    ratio = np.power(10.0, residual_log)               # multiplicative factor
    rel_change = ratio - 1.0                           # scale-invariant dip/bump

    return baseline_hat_log, residual_log, ratio, rel_change


def make_axis_aligned_food_positions(center, buffer_cm=12, num_points=50):
    """
    Return positions along horizontal and vertical axes through 'center'.
    """
    xs = np.linspace(center[0] - buffer_cm, center[0] + buffer_cm, num_points)
    ys = np.linspace(center[1] - buffer_cm, center[1] + buffer_cm, num_points)

    # Horizontal axis (y fixed at center[1])
    food_x = np.stack([xs, np.full_like(xs, center[1])], axis=1)

    # Vertical axis (x fixed at center[0])
    food_y = np.stack([np.full_like(ys, center[0]), ys], axis=1)

    # Concatenate
    food_pos_axis = np.vstack([food_x, food_y])
    return food_pos_axis


def run_self_image_experiment(
    fish_orientation=np.pi / 2,
    num_points=50,
    do_self_induced=True,
    arena_size=(100, 100),
    buffer_cm=10,
    fish_charge=5.56e-10,
    do_baseline_subtraction=True,
    criterion='min',
    no_food=False,
):
    # What is the max sensor reading due to self EOD with food at various locations?

    center = np.array(arena_size) / 2
    fish = SimpleEFishAgent(center, fish_orientation)

    food_x = np.linspace(center[0] - buffer_cm, center[0] + buffer_cm, num_points)
    food_y = np.linspace(center[1] - buffer_cm, center[1] + buffer_cm, num_points)
    food_pos_grid = np.array(np.meshgrid(food_x, food_y)).T.reshape(-1, 2)
    # food_pos_grid = make_axis_aligned_food_positions(center, buffer_cm=12, num_points=50)


    fish.monopole_charges = np.array(
        [fish_charge, -fish_charge]
    )  # Two monopoles with opposite charges
    # Baseline (no food)
    if do_baseline_subtraction:
        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            fish_positions=np.array([fish.position]),
            fish_orientations=np.array([fish.orientation]),
            fish_monopole_positions=fish.monopole_positions_ego[None],
            fish_monopole_charges=fish.monopole_charges[None],
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=np.array([True]),
            fish_sensor_positions=fish.mormyromast_positions_ego[None],
            fish_sensor_normals=fish.mormyromast_normals_ego[None],
            fish_sensor_type_mask=None,
            conductor_positions=np.zeros((0, 2)),
            conductor_contrasts=np.array([]),
            conductor_radii=np.array([]),
            arena_size=arena_size,
            arena_clipping_distance=None,
            do_induced=do_self_induced,
            fish_contrasts=(
                np.array([ELECTRIC_CONSTANTS["fish_contrast"]]) if do_self_induced else None
            ),
            fish_radii=(
                np.array([fish.body_radius_m * m_to_cm]) if do_self_induced else None
            ),
            do_reflection=False,
            return_raw_field=False,
        )
        baseline = sensor_measurements[0]
    else:
        baseline = np.zeros(fish.num_mormyromast_sensors)

    food_contrast = ELECTRIC_CONSTANTS["food_contrast"]
    food_radius_cm = ENV_PARAMS["food_radius_cm"]
    readings_self = []
    all_readings = []

    for food_pos in food_pos_grid:
        if no_food:
            conductor_positions=np.zeros((0, 2))
            conductor_contrasts=np.array([])
            conductor_radii=np.array([])
        else:
            conductor_positions=np.array([food_pos])
            conductor_contrasts=np.array([food_contrast])
            conductor_radii=np.array([food_radius_cm])


        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            fish_positions=np.array([fish.position]),
            fish_orientations=np.array([fish.orientation]),
            fish_monopole_positions=fish.monopole_positions_ego[None],
            fish_monopole_charges=fish.monopole_charges[None],
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=np.array([True]),
            fish_sensor_positions=fish.mormyromast_positions_ego[None],
            fish_sensor_normals=fish.mormyromast_normals_ego[None],
            fish_sensor_type_mask=None,
            conductor_positions=conductor_positions,
            conductor_contrasts=conductor_contrasts,
            conductor_radii=conductor_radii,
            arena_size=arena_size,
            arena_clipping_distance=None,
            do_induced=True,
            fish_contrasts=(
                np.array([ELECTRIC_CONSTANTS["fish_contrast"]])
                if do_self_induced
                else None
            ),
            fish_radii=(
                np.array([fish.body_radius_m * m_to_cm]) if do_self_induced else None
            ),
            do_reflection=False,
            return_raw_field=False,
        )
        reading = sensor_measurements[0]

        delta = reading - baseline
        if criterion == 'max':
            val = np.max(np.abs(delta))
        elif criterion == 'mean':
            val = np.mean(np.abs(delta))
        elif criterion == 'median':
            val = np.median(np.abs(delta))
        elif criterion == 'min':
            val = np.min(np.abs(delta))
        elif criterion == 'angular_residual':
            baseline_hat, residual, _, _ = angular_residual_log(reading, fish.mormyromast_angles)
            val = np.max(np.abs(10**residual))
        elif criterion == "axis":
            val = np.max(np.abs(delta[0]))

        readings_self.append([food_pos[0], food_pos[1], val])
        all_reading = np.concatenate([food_pos, reading])
        all_readings.append(all_reading)

    return np.array(readings_self), fish, np.array(all_readings), baseline




def run_self_image_experiment_with_without_food(
    fish_orientation=np.pi / 2,
    num_points=50,
    do_self_induced=True,
    arena_size=(100, 100),
    buffer_cm=10,
    fish_charge=5.56e-10,
):
    # What is the max sensor reading due to self EOD with food at various locations?

    center = np.array(arena_size) / 2
    fish = SimpleEFishAgent(center, fish_orientation)

    food_x = np.linspace(center[0] - buffer_cm, center[0] + buffer_cm, num_points)
    food_y = np.linspace(center[1] - buffer_cm, center[1] + buffer_cm, num_points)
    food_pos_grid = np.array(np.meshgrid(food_x, food_y)).T.reshape(-1, 2)

    fish.monopole_charges = np.array(
        [fish_charge, -fish_charge]
    )  # Two monopoles with opposite charges
    food_contrast = ELECTRIC_CONSTANTS["food_contrast"]
    food_radius_cm = ENV_PARAMS["food_radius_cm"]
    min_readings_self = []

    for food_pos in food_pos_grid:
        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            fish_positions=np.array([fish.position]),
            fish_orientations=np.array([fish.orientation]),
            fish_monopole_positions=fish.monopole_positions_ego[None],
            fish_monopole_charges=fish.monopole_charges[None],
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=np.array([True]),
            fish_sensor_positions=fish.mormyromast_positions_ego[None],
            fish_sensor_normals=fish.mormyromast_normals_ego[None],
            fish_sensor_type_mask=None,
            conductor_positions=np.zeros((0, 2)),
            conductor_contrasts=np.array([]),
            conductor_radii=np.array([]),
            arena_size=arena_size,
            arena_clipping_distance=None,
            do_induced=do_self_induced,
            fish_contrasts=(
                np.array([ELECTRIC_CONSTANTS["fish_contrast"]])
                if do_self_induced
                else None
            ),
            fish_radii=(
                np.array([fish.body_radius_m * m_to_cm]) if do_self_induced else None
            ),
            do_reflection=False,
            return_raw_field=False,
        )
        no_food_baseline = sensor_measurements[0]

        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            fish_positions=np.array([fish.position]),
            fish_orientations=np.array([fish.orientation]),
            fish_monopole_positions=fish.monopole_positions_ego[None],
            fish_monopole_charges=fish.monopole_charges[None],
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=np.array([True]),
            fish_sensor_positions=fish.mormyromast_positions_ego[None],
            fish_sensor_normals=fish.mormyromast_normals_ego[None],
            fish_sensor_type_mask=None,
            conductor_positions=np.array([food_pos]),
            conductor_contrasts=np.array([food_contrast]),
            conductor_radii=np.array(
                [food_radius_cm]
            ),  # multiplying by 4.5 seems to help fit
            arena_size=arena_size,
            arena_clipping_distance=None,
            do_induced=True,
            fish_contrasts=(
                np.array([ELECTRIC_CONSTANTS["fish_contrast"]])
                if do_self_induced
                else None
            ),
            fish_radii=(
                np.array([fish.body_radius_m * m_to_cm]) if do_self_induced else None
            ),
            do_reflection=False,
            return_raw_field=False,
        )
        reading = sensor_measurements[0]

        delta = reading - no_food_baseline
        min_val = np.min(np.abs(delta))
        min_readings_self.append([food_pos[0], food_pos[1], min_val])

    return np.array(min_readings_self), fish


def run_sense_food_ampullary(
    fish_orientation=np.pi / 2,
    num_points=50,
    arena_size=(100, 100),
    buffer_cm=10,
):
    # What is the max sensor reading due to self EOD with food at various locations?

    center = np.array(arena_size) / 2
    fish = SimpleEFishAgent(center, fish_orientation)

    food_x = np.linspace(center[0] - buffer_cm, center[0] + buffer_cm, num_points)
    food_y = np.linspace(center[1] - buffer_cm, center[1] + buffer_cm, num_points)
    food_pos_grid = np.array(np.meshgrid(food_x, food_y)).T.reshape(-1, 2)

    fish_intrinsic_dipole_moments = np.tile(cfg.AGENT_PARAMS["fish_intrinsic_dipole_moment"], (1, 1))
    # fish_intrinsic_dipole_moments = np.tile(np.array([0,0]), (1, 1)) # DEBUG
    fish.monopole_charges = np.array(
        [0, 0]
    )  # Two monopoles with opposite charges

    conductor_intrinsic_dipole_moments = np.full(
                (1, 2), cfg.ENV_PARAMS["food_intrinsic_dipole_moment"]
            )
    # Baseline (no food)
    # sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
    #     fish_positions=np.array([fish.position]),
    #     fish_orientations=np.array([fish.orientation]),
    #     fish_monopole_positions=fish.monopole_positions_ego[None],
    #     fish_monopole_charges=fish.monopole_charges[None],
    #     fish_dipole_positions=np.zeros((1, 0, 2)),
    #     fish_dipole_moments=np.zeros((1, 0, 2)),
    #     fish_eods=np.array([False]),
    #     fish_sensor_positions=fish.mormyromast_positions_ego[None],
    #     fish_sensor_normals=fish.mormyromast_normals_ego[None],
    #     fish_sensor_type_mask=None,
    #     fish_intrinsic_dipole_moments=fish_intrinsic_dipole_moments,
    #     # conductor_intrinsic_dipole_moments=conductor_intrinsic_dipole_moments, None for baseline!
    #     conductor_positions=np.zeros((0, 2)),
    #     conductor_contrasts=np.array([]),
    #     conductor_radii=np.array([]),
    #     arena_size=arena_size,
    #     arena_clipping_distance=None,
    #     do_induced=False,  # would be True but no EODs
    #     do_intrinsics=True,
    #     fish_contrasts=None,
    #     fish_radii=None,
    #     do_reflection=False,
    #     return_raw_field=False,
    # )
    # baseline = sensor_measurements[0]

    food_contrast = ELECTRIC_CONSTANTS["food_contrast"]
    food_radius_cm = ENV_PARAMS["food_radius_cm"]
    max_readings_self = []

    for food_pos in food_pos_grid:
        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            fish_positions=np.array([fish.position]),
            fish_orientations=np.array([fish.orientation]),
            fish_monopole_positions=fish.monopole_positions_ego[None],
            fish_monopole_charges=fish.monopole_charges[None],
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=np.array([False]),
            fish_sensor_positions=fish.mormyromast_positions_ego[None],
            fish_sensor_normals=fish.mormyromast_normals_ego[None],
            fish_sensor_type_mask=None,
            fish_intrinsic_dipole_moments=fish_intrinsic_dipole_moments,
            conductor_intrinsic_dipole_moments=conductor_intrinsic_dipole_moments,
            conductor_positions=np.array([food_pos]),
            conductor_contrasts=np.array([food_contrast]),
            conductor_radii=np.array(
                [food_radius_cm]
            ),  # multiplying by 4.5 seems to help fit
            arena_size=arena_size,
            arena_clipping_distance=None,
            do_induced=False,
            do_intrinsics=True,
            fish_contrasts=None,
            fish_radii=None,
            do_reflection=False,
            return_raw_field=False,
        )
        reading = sensor_measurements[0]


        # delta = reading - baseline
        # print("delta", delta)
        # max_val = np.min(np.abs(reading))
        max_val = np.max(np.abs(reading))
        max_readings_self.append([food_pos[0], food_pos[1], max_val])

    return np.array(max_readings_self), fish


def run_self_image_active_sense_other_fish_experiment(
    fish_orientation: float = np.pi / 2,
    other_fish_orientation: float = np.pi / 2,
    num_points: int = 50,
    buffer: float = 25,  # cm buffer around fish body
    fish_charge: float = 1.5e-7,
    do_self_induced=True,
    do_baseline_subtraction=False,
    criterion='min',
):
    arena_size = (100, 100)
    center = np.array(arena_size) / 2
    fish = SimpleEFishAgent(center.copy(), fish_orientation)
    fish.monopole_charges = np.array([fish_charge, -fish_charge])

    xs = np.linspace(center[0] - buffer, center[0] + buffer, num_points)
    ys = np.linspace(center[1] - buffer, center[1] + buffer, num_points)
    other_fish_grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)

    if do_baseline_subtraction:
        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            fish_positions=np.array([fish.position]),
            fish_orientations=np.array([fish.orientation]),
            fish_monopole_positions=fish.monopole_positions_ego[None],
            fish_monopole_charges=fish.monopole_charges[None],
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=np.array([True]),
            fish_sensor_positions=fish.mormyromast_positions_ego[None],
            fish_sensor_normals=fish.mormyromast_normals_ego[None],
            fish_sensor_type_mask=None,
            conductor_positions=np.zeros((0, 2)),
            conductor_contrasts=np.array([]),
            conductor_radii=np.array([]),
            fish_contrasts=(
                np.array([cfg.ELECTRIC_CONSTANTS["fish_contrast"]])
                if do_self_induced
                else None
            ),
            fish_radii=np.array([AGENT_PARAMS["body_radius_cm"]]) if do_self_induced else None,
            arena_size=arena_size,
            arena_clipping_distance=None,
            do_induced=True,
            do_reflection=False,
            return_raw_field=False,
            max_charge_allowed=fish_charge,
        )
        baseline = sensor_measurements[0]
    else:
        baseline = np.zeros(fish.num_mormyromast_sensors)

    readings = []
    for pos in other_fish_grid:
        other_agent = SimpleEFishAgent(pos.copy(), other_fish_orientation)
        positions = np.vstack([fish.position, other_agent.position])
        orientations = np.array([fish.orientation, other_agent.orientation])
        eods = np.array([True, False])
        contrasts = np.array(
            [
                ELECTRIC_CONSTANTS["fish_contrast"] if do_self_induced else 0,
                ELECTRIC_CONSTANTS["fish_contrast"],
            ]
        )
        radii = np.array(
            [
                fish.body_radius_m * m_to_cm,
                fish.body_radius_m * m_to_cm,
            ]
        )

        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            fish_positions=positions,
            fish_orientations=orientations,
            fish_monopole_positions=fish.monopole_positions_ego[None],
            fish_monopole_charges=fish.monopole_charges[None],
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=eods,
            fish_sensor_positions=fish.mormyromast_positions_ego[None],
            fish_sensor_normals=fish.mormyromast_normals_ego[None],
            fish_sensor_type_mask=None,
            conductor_positions=np.zeros((0, 2)),
            conductor_contrasts=np.array([]),
            conductor_radii=np.array([]),
            fish_contrasts=contrasts,
            fish_radii=radii,
            arena_size=arena_size,
            arena_clipping_distance=None,
            do_induced=True,
            do_reflection=False,
            return_raw_field=False,
            max_charge_allowed=fish_charge,
        )
        field = sensor_measurements[0]

        delta = field - baseline
        if criterion == 'max':
            val = np.max(np.abs(delta))
        elif criterion == 'mean':
            val = np.mean(np.abs(delta))
        elif criterion == 'min':
            val = np.min(np.abs(delta))

        readings.append([pos[0], pos[1], val])

    return np.array(readings), fish


def run_sense_other_fish_ampullary(
    fish_orientation: float = np.pi / 2,
    other_fish_orientation: float = np.pi / 2,
    num_points: int = 50,
    buffer: float = 25,  # cm buffer around fish body
    fish_charge: float = 1.5e-7,
):
    arena_size = (100, 100)
    center = np.array(arena_size) / 2
    fish = SimpleEFishAgent(center.copy(), fish_orientation)
    fish.monopole_charges = np.array([0, 0])

    xs = np.linspace(center[0] - buffer, center[0] + buffer, num_points)
    ys = np.linspace(center[1] - buffer, center[1] + buffer, num_points)
    other_fish_grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)

    fish_intrinsic_dipole_moments = np.tile(
        cfg.AGENT_PARAMS["fish_intrinsic_dipole_moment"], (2, 1)
    )

    # sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
    #     fish_positions=np.array([fish.position]),
    #     fish_orientations=np.array([fish.orientation]),
    #     fish_monopole_positions=fish.monopole_positions_ego[None],
    #     fish_monopole_charges=fish.monopole_charges[None],
    #     fish_dipole_positions=np.zeros((1, 0, 2)),
    #     fish_dipole_moments=np.zeros((1, 0, 2)),
    #     fish_eods=np.array([False]),
    #     fish_sensor_positions=fish.mormyromast_positions_ego[None],
    #     fish_sensor_normals=fish.mormyromast_normals_ego[None],
    #     fish_intrinsic_dipole_moments=fish_intrinsic_dipole_moments[0][None],
    #     fish_sensor_type_mask=None,
    #     conductor_positions=np.zeros((0, 2)),
    #     conductor_contrasts=np.array([]),
    #     conductor_radii=np.array([]),
    #     fish_contrasts=None,
    #     fish_radii=None,
    #     arena_size=arena_size,
    #     arena_clipping_distance=None,
    #     do_induced=False,
    #     do_intrinsics=True,  # ampullary!
    #     do_reflection=False,
    #     return_raw_field=False,
    #     max_charge_allowed=fish_charge,
    # )
    # baseline = sensor_measurements[0]

    readings = []
    for pos in other_fish_grid:
        other_agent = SimpleEFishAgent(pos.copy(), other_fish_orientation)
        positions = np.vstack([fish.position, other_agent.position])
        orientations = np.array([fish.orientation, other_agent.orientation])
        eods = np.array([False, False])
        contrasts = np.array(
            [
                ELECTRIC_CONSTANTS["fish_contrast"],  # not relevant bc no EOD
                ELECTRIC_CONSTANTS["fish_contrast"],
            ]
        )
        radii = np.array(
            [
                fish.body_radius_m * m_to_cm,
                fish.body_radius_m * m_to_cm,
            ]
        )

        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            fish_positions=positions,
            fish_orientations=orientations,
            fish_monopole_positions=fish.monopole_positions_ego[None],
            fish_monopole_charges=fish.monopole_charges[None],
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=eods,
            fish_sensor_positions=fish.mormyromast_positions_ego[None],
            fish_sensor_normals=fish.mormyromast_normals_ego[None],
            fish_sensor_type_mask=None,
            fish_intrinsic_dipole_moments=fish_intrinsic_dipole_moments,  # this affects ampullary
            conductor_positions=np.zeros((0, 2)),
            conductor_contrasts=np.array([]),
            conductor_radii=np.array([]),
            fish_contrasts=contrasts,
            fish_radii=radii,
            arena_size=arena_size,
            arena_clipping_distance=None,
            do_induced=False,
            do_intrinsics=True,  # ampullary!
            do_reflection=False,
            return_raw_field=False,
            max_charge_allowed=fish_charge,
        )
        field = sensor_measurements[0]

        # delta = field - baseline
        delta = field
        max_delta = np.min(np.abs(delta))
        readings.append([pos[0], pos[1], max_delta])

    return np.array(readings), fish


def run_direct_eod_sensing_experiment_no_food(
    num_points=50,
    do_self_induced=False,
    do_induced=False,
    buffer_cm=50,
    arena_size=(200, 200),
    fish_charge=5.56e-10,
    subtract_array_mean=False,
    dynamic_gain_modulation=False,
    criterion='max',
):
    # Fish 1 emits at different locations, Fish 2 senses at center
    center = np.array(arena_size) / 2
    fish2_orientation = np.pi / 2
    fish2 = SimpleEFishAgent(center, fish2_orientation)
    fish2.monopole_charges = np.array([fish_charge, -fish_charge])

    fish1_x = np.linspace(center[0] - buffer_cm, center[0] + buffer_cm, num_points)
    fish1_y = np.linspace(center[1] - buffer_cm, center[1] + buffer_cm, num_points)
    fish1_pos_grid = np.array(np.meshgrid(fish1_x, fish1_y)).T.reshape(-1, 2)

    fish_eods = np.array([True, False])  # Fish 1 emits, Fish 2 senses without emitting
    fish_contrasts = np.array(
        [
            ELECTRIC_CONSTANTS["fish_contrast"],
            ELECTRIC_CONSTANTS["fish_contrast"],
        ]
    )
    fish_radii = np.array(
        [
            fish2.body_radius_m * m_to_cm,  # should be same as Fish 2's radius
            fish2.body_radius_m * m_to_cm,  # Fish 2's radius
        ]
    )
    if not do_self_induced:
        fish_contrasts = fish_contrasts * (~fish_eods).astype(float)

    readings = []
    for pos in fish1_pos_grid:
        fish1_orientation = np.pi / 2
        # fish1_orientation = 0
        # Have fish face fish2
        # vec_to_fish2 = fish2.position - pos
        # fish1_orientation = np.arctan2(vec_to_fish2[1], vec_to_fish2[0])
        fish1 = SimpleEFishAgent(pos, fish1_orientation)
        fish1.monopole_charges = np.array([fish_charge, -fish_charge])
        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            fish_positions=np.array([fish1.position, fish2.position]),
            fish_orientations=np.array([fish1.orientation, fish2.orientation]),
            fish_monopole_positions=np.stack(
                [fish1.monopole_positions_ego, fish2.monopole_positions_ego]
            ),
            fish_monopole_charges=np.stack(
                [fish1.monopole_charges, fish2.monopole_charges]
            ),
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=fish_eods,
            fish_sensor_positions=np.stack(
                [fish1.mormyromast_positions_ego, fish2.mormyromast_positions_ego]
            ),
            fish_sensor_normals=np.stack(
                [fish1.mormyromast_normals_ego, fish2.mormyromast_normals_ego]
            ),
            fish_sensor_type_mask=None,
            conductor_positions=np.zeros((0, 2)),  # No food
            conductor_contrasts=np.array([]),  # No food
            conductor_radii=np.array([]),  # No food
            arena_size=arena_size,
            arena_clipping_distance=None,
            do_induced=do_induced,
            fish_contrasts=fish_contrasts,
            fish_radii=fish_radii,
            do_reflection=False,
            return_raw_field=False,
            max_charge_allowed=fish_charge,
        )
        result = sensor_measurements[1]  # fish2's sensors

        if subtract_array_mean:
            result = result - np.mean(np.abs(result))

        if dynamic_gain_modulation:
            result = (result - result.min()) / result.max()

        if criterion == 'max':
            processed_result = np.max(np.abs(result))
        elif criterion == 'mean':
            processed_result = np.mean(np.abs(result))
        elif criterion == 'min':
            processed_result = np.min(np.abs(result))

        readings.append([pos[0], pos[1], processed_result])

    return np.array(readings), fish2


def run_direct_eod_sensing_experiment_no_food_subtract_cons_baseline(
    num_points=50,
    do_self_induced=False,
    do_induced=True,
    buffer_cm=50,
    arena_size=(200, 200),
    fish_charge=5.56e-10,
):
    # Fish 1 emits at different locations, Fish 2 senses at center
    center = np.array(arena_size) / 2
    fish2_orientation = np.pi / 2
    fish2 = SimpleEFishAgent(center, fish2_orientation)
    fish2.monopole_charges = np.array([fish_charge, -fish_charge])

    fish1_x = np.linspace(center[0] - buffer_cm, center[0] + buffer_cm, num_points)
    fish1_y = np.linspace(center[1] - buffer_cm, center[1] + buffer_cm, num_points)
    fish1_pos_grid = np.array(np.meshgrid(fish1_x, fish1_y)).T.reshape(-1, 2)

    fish_eods = np.array([True, False])  # Fish 1 emits, Fish 2 senses without emitting
    fish_contrasts = np.array(
        [
            ELECTRIC_CONSTANTS["fish_contrast"],
            ELECTRIC_CONSTANTS["fish_contrast"],
        ]
    )
    fish_radii = np.array(
        [
            fish2.body_radius_m * m_to_cm,  # should be same as Fish 2's radius
            fish2.body_radius_m * m_to_cm,  # Fish 2's radius
        ]
    )
    if not do_self_induced:
        fish_contrasts = fish_contrasts * (~fish_eods).astype(float)

    readings = []
    for pos in fish1_pos_grid:
        fish1_orientation = np.pi / 2
        # fish1_orientation = 0
        # Have fish face fish2
        # vec_to_fish2 = fish2.position - pos
        # fish1_orientation = np.arctan2(vec_to_fish2[1], vec_to_fish2[0])
        fish1 = SimpleEFishAgent(pos, fish1_orientation)
        fish1.monopole_charges = np.array([fish_charge, -fish_charge])

        # Get baseline reading "with no food" (no food this time, but other fish present)
        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            fish_positions=np.array([fish1.position, fish2.position]),
            fish_orientations=np.array([fish1.orientation, fish2.orientation]),
            fish_monopole_positions=np.stack(
                [fish1.monopole_positions_ego, fish2.monopole_positions_ego]
            ),
            fish_monopole_charges=np.stack(
                [fish1.monopole_charges, fish2.monopole_charges]
            ),
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=fish_eods,
            fish_sensor_positions=np.stack(
                [fish1.mormyromast_positions_ego, fish2.mormyromast_positions_ego]
            ),
            fish_sensor_normals=np.stack(
                [fish1.mormyromast_normals_ego, fish2.mormyromast_normals_ego]
            ),
            conductor_positions=np.zeros((0, 2)),
            conductor_contrasts=np.array([]),
            conductor_radii=np.array([]),
            arena_size=arena_size,
            do_induced=True,
            fish_contrasts=fish_contrasts,
            fish_radii=fish_radii,
            fish_sensor_type_mask=None,
            arena_clipping_distance=None,
            do_reflection=False,
            return_raw_field=False,
            max_charge_allowed=fish_charge,
        )
        baseline = sensor_measurements[1]


        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            fish_positions=np.array([fish1.position, fish2.position]),
            fish_orientations=np.array([fish1.orientation, fish2.orientation]),
            fish_monopole_positions=np.stack(
                [fish1.monopole_positions_ego, fish2.monopole_positions_ego]
            ),
            fish_monopole_charges=np.stack(
                [fish1.monopole_charges, fish2.monopole_charges]
            ),
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=fish_eods,
            fish_sensor_positions=np.stack(
                [fish1.mormyromast_positions_ego, fish2.mormyromast_positions_ego]
            ),
            fish_sensor_normals=np.stack(
                [fish1.mormyromast_normals_ego, fish2.mormyromast_normals_ego]
            ),
            fish_sensor_type_mask=None,
            conductor_positions=np.zeros((0, 2)),  # No food
            conductor_contrasts=np.array([]),  # No food
            conductor_radii=np.array([]),  # No food
            arena_size=arena_size,
            arena_clipping_distance=None,
            do_induced=do_induced,
            fish_contrasts=fish_contrasts,
            fish_radii=fish_radii,
            do_reflection=False,
            return_raw_field=False,
            max_charge_allowed=fish_charge,
        )
        result = sensor_measurements[1]  # fish2's sensors

        result = result - baseline
        readings.append([pos[0], pos[1], np.max(np.abs(result))])

    return np.array(readings), fish2


def run_direct_eod_sensing_experiment_no_food_ampullary(
    num_points=50,
    buffer_cm=50,
    arena_size=(200, 200),
    fish_charge=5.56e-10,
    subtract_array_mean=False,
    dynamic_gain_modulation=False,
):
    # Fish 1 emits at different locations, Fish 2 senses at center
    center = np.array(arena_size) / 2
    fish2_orientation = np.pi / 2
    fish2 = SimpleEFishAgent(center, fish2_orientation)
    fish2.monopole_charges = np.array([fish_charge, -fish_charge])

    fish1_x = np.linspace(center[0] - buffer_cm, center[0] + buffer_cm, num_points)
    fish1_y = np.linspace(center[1] - buffer_cm, center[1] + buffer_cm, num_points)
    fish1_pos_grid = np.array(np.meshgrid(fish1_x, fish1_y)).T.reshape(-1, 2)

    fish_eods = np.array([True, False])  # Fish 1 emits, Fish 2 senses without emitting
    fish_contrasts = np.array(
        [
            ELECTRIC_CONSTANTS["fish_contrast"],
            ELECTRIC_CONSTANTS["fish_contrast"],
        ]
    )
    fish_radii = np.array(
        [
            fish2.body_radius_m * m_to_cm,  # should be same as Fish 2's radius
            fish2.body_radius_m * m_to_cm,  # Fish 2's radius
        ]
    )

    fish_intrinsic_dipole_moments = np.tile(
        cfg.AGENT_PARAMS["fish_intrinsic_dipole_moment"], (2, 1)
    )

    readings = []
    for pos in fish1_pos_grid:
        fish1_orientation = np.pi / 2
        fish1 = SimpleEFishAgent(pos, fish1_orientation)
        fish1.monopole_charges = np.array([fish_charge, -fish_charge])

        # baseline with only fish2 (no fish1, no fish1 EOD)
        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            fish_positions=np.array([fish2.position]),
            fish_orientations=np.array([fish2.orientation]),
            fish_monopole_positions=fish2.monopole_positions_ego[None],
            fish_monopole_charges=fish2.monopole_charges[None],
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=np.array([False]),  # Fish 2 does not emit
            fish_sensor_positions=fish2.mormyromast_positions_ego[None],
            fish_sensor_normals=fish2.mormyromast_normals_ego[None],
            fish_sensor_type_mask=None,
            conductor_positions=np.zeros((0, 2)),  # No food
            conductor_contrasts=np.array([]),  # No food
            conductor_radii=np.array([]),  # No food
            fish_intrinsic_dipole_moments=fish_intrinsic_dipole_moments[0][None],  # this affects ampullary
            arena_size=arena_size,
            arena_clipping_distance=None,
            do_induced=False,  # no EODs, so irrelevant
            do_intrinsics=True,
            do_reflection=False,
            return_raw_field=False,
        )
        ampullary_cd = sensor_measurements[0]

        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            fish_positions=np.array([fish1.position, fish2.position]),
            fish_orientations=np.array([fish1.orientation, fish2.orientation]),
            fish_monopole_positions=np.stack(
                [fish1.monopole_positions_ego, fish2.monopole_positions_ego]
            ),
            fish_monopole_charges=np.stack(
                [fish1.monopole_charges, fish2.monopole_charges]
            ),
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=fish_eods,
            fish_sensor_positions=np.stack(
                [fish1.mormyromast_positions_ego, fish2.mormyromast_positions_ego]
            ),
            fish_sensor_normals=np.stack(
                [fish1.mormyromast_normals_ego, fish2.mormyromast_normals_ego]
            ),
            fish_sensor_type_mask=None,
            fish_intrinsic_dipole_moments=fish_intrinsic_dipole_moments,  # this affects ampullary
            conductor_positions=np.zeros((0, 2)),  # No food
            conductor_contrasts=np.array([]),  # No food
            conductor_radii=np.array([]),  # No food
            arena_size=arena_size,
            arena_clipping_distance=None,
            do_induced=True,  # fish1 can induce charge on fish2
            do_intrinsics=True,
            fish_contrasts=fish_contrasts,
            fish_radii=fish_radii,
            do_reflection=False,
            return_raw_field=False,
            max_charge_allowed=fish_charge,
        )
        result = sensor_measurements[1]  # fish2's sensors

        result = result - ampullary_cd  # Subtract baseline reading

        readings.append([pos[0], pos[1], np.max(np.abs(result))])

    return np.array(readings), fish2


def run_direct_eod_sensing_experiment_no_food_fish2_grid(
    num_points=50,
    do_induced=False,
    buffer_cm=150,
    arena_size=(300, 300),
    fish_charge=5.56e-10,
    subtract_array_mean=False,
):
    # Fish 1 is fixed at position arena_size/2 with orientation np.pi/2,
    # Fish 2 is swept across a grid, max abs sensor reading at each location on Fish 2 is plotted

    # Fish 1 is fixed
    fish1_position = np.array(arena_size) / 2
    fish1_orientation = np.pi / 2
    fish1 = SimpleEFishAgent(fish1_position, fish1_orientation)
    fish1.monopole_charges = np.array([fish_charge, -fish_charge])

    # Sweep Fish 2 across the grid
    fish2_x = np.linspace(
        fish1_position[0] - buffer_cm, fish1_position[0] + buffer_cm, num_points
    )
    fish2_y = np.linspace(
        fish1_position[1] - buffer_cm, fish1_position[1] + buffer_cm, num_points
    )
    fish2_pos_grid = np.array(np.meshgrid(fish2_x, fish2_y)).T.reshape(-1, 2)

    readings = []
    for pos in fish2_pos_grid:
        fish2_orientation = np.pi / 2  # Can also face fish1 or be varied
        fish2 = SimpleEFishAgent(pos, fish2_orientation)

        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            fish_positions=np.array([fish1.position, fish2.position]),
            fish_orientations=np.array([fish1.orientation, fish2.orientation]),
            fish_monopole_positions=np.stack(
                [fish1.monopole_positions_ego, fish2.monopole_positions_ego]
            ),
            fish_monopole_charges=np.stack(
                [fish1.monopole_charges, fish2.monopole_charges]
            ),
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=np.array([True, False]),  # Fish 1 emits, Fish 2 senses
            fish_sensor_positions=np.stack(
                [fish1.mormyromast_positions_ego, fish2.mormyromast_positions_ego]
            ),
            fish_sensor_normals=np.stack(
                [fish1.mormyromast_normals_ego, fish2.mormyromast_normals_ego]
            ),
            fish_sensor_type_mask=None,
            conductor_positions=np.zeros((0, 2)),  # No food
            conductor_contrasts=np.array([]),
            conductor_radii=np.array([]),
            arena_size=arena_size,
            arena_clipping_distance=None,
            do_induced=do_induced,
            fish_contrasts=np.array(
                [
                    ELECTRIC_CONSTANTS["fish_contrast"],
                    ELECTRIC_CONSTANTS["fish_contrast"] if do_induced else 0,
                ]
            ),
            fish_radii=(
                np.array([fish1.body_radius_m * m_to_cm, fish2.body_radius_m * m_to_cm])
                if do_induced
                else None
            ),
            do_reflection=False,
            return_raw_field=False,
            max_charge_allowed=fish_charge,
        )
        result = sensor_measurements[1]  # fish2's sensors

        if subtract_array_mean:
            result = result - np.mean(result)

        readings.append([pos[0], pos[1], np.max(np.abs(result))])

    return np.array(readings), fish1


def run_direct_eod_sensing_experiment_with_food(
    num_points=50,
    do_self_induced=False,
    buffer_cm=20,
    dist_fish=15,
    arena_size=(100, 100),
    fish_config=1,
    dynamic_baseline=None,
    fish_charge=5.56e-10,
    criterion='max',
):
    """
    Simulates Fish 2 sensing a food object moving around
    which is the EOD emitter.

    Parameters:
    - num_points: Number of grid points for the food's movement.
    - do_induced: Whether to include induced dipoles.
    - buffer_cm: Buffer distance for the food grid.
    - arena_size: Size of the simulation arena.
    - fish_config: An integer (1, 2, or 3) specifying the fish configuration.
    """
    center = np.array(arena_size) / 2

    # Set up fish positions and orientations based on the configuration
    if fish_config == 1:
        # Fish 1 (facing left) is to the upper right of Fish 2 (facing up)
        fish1_pos = center + np.array([dist_fish / np.sqrt(2), dist_fish / np.sqrt(2)])
        fish1_orientation = np.pi  # Facing left
        fish2_pos = center
        fish2_orientation = np.pi / 2  # Facing up
    elif fish_config == 2:
        # Fish 1 (facing up) is above Fish 2 (facing up)
        fish1_pos = center + np.array([0, dist_fish])
        fish1_orientation = np.pi / 2  # Facing up
        fish2_pos = center
        fish2_orientation = np.pi / 2  # Facing up
    elif fish_config == 3:
        # Fish 1 (facing up) is below Fish 2 (facing up)
        fish1_pos = center - np.array([0, dist_fish])
        fish1_orientation = np.pi / 2  # Facing up
        fish2_pos = center
        fish2_orientation = np.pi / 2  # Facing up
    else:
        raise ValueError("Invalid fish_config. Must be 1, 2, or 3.")

    fish1 = SimpleEFishAgent(fish1_pos, fish1_orientation)
    fish2 = SimpleEFishAgent(fish2_pos, fish2_orientation)
    fish1.monopole_charges = np.array([fish_charge, -fish_charge])
    fish2.monopole_charges = np.array(
        [fish_charge, -fish_charge]
    )  # Not used since fish2 is not emitting

    # Define the food's movement grid
    food_x = np.linspace(center[0] - buffer_cm, center[0] + buffer_cm, num_points)
    food_y = np.linspace(center[1] - buffer_cm, center[1] + buffer_cm, num_points)
    food_pos_grid = np.array(np.meshgrid(food_x, food_y)).T.reshape(-1, 2)

    food_contrast = ELECTRIC_CONSTANTS["food_contrast"]
    food_radius_cm = ENV_PARAMS["food_radius_cm"]

    readings = []

    fish_eods = np.array([True, False])  # Fish 1 emits, Fish 2 senses without emitting
    fish_contrasts = np.array(
        [
            ELECTRIC_CONSTANTS["fish_contrast"],
            ELECTRIC_CONSTANTS["fish_contrast"],
        ]
    )
    fish_radii = np.array(
        [
            fish1.body_radius_m * m_to_cm,  # Fish 1's radius
            fish2.body_radius_m * m_to_cm,  # Fish 2's radius
        ]
    )
    if not do_self_induced:
        fish_contrasts = fish_contrasts * (~fish_eods).astype(float)

    # Get baseline reading with no food
    # OK to do this once because distance between fish not changing in this "test"
    if dynamic_baseline == "no_food":
        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            # baseline = electric.fish_forward(
            fish_positions=np.array([fish1.position, fish2.position]),
            fish_orientations=np.array([fish1.orientation, fish2.orientation]),
            fish_monopole_positions=np.stack(
                [fish1.monopole_positions_ego, fish2.monopole_positions_ego]
            ),
            fish_monopole_charges=np.stack(
                [fish1.monopole_charges, fish2.monopole_charges]
            ),
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=fish_eods,
            fish_sensor_positions=np.stack(
                [fish1.mormyromast_positions_ego, fish2.mormyromast_positions_ego]
            ),
            fish_sensor_normals=np.stack(
                [fish1.mormyromast_normals_ego, fish2.mormyromast_normals_ego]
            ),
            conductor_positions=np.zeros((0, 2)),
            conductor_contrasts=np.array([]),
            conductor_radii=np.array([]),
            arena_size=arena_size,
            do_induced=True,  # Always true for food sensing
            fish_contrasts=fish_contrasts,
            fish_radii=fish_radii,
            fish_sensor_type_mask=None,
            arena_clipping_distance=None,
            do_reflection=False,
            return_raw_field=False,
            max_charge_allowed=fish_charge,
        )
        baseline = sensor_measurements[1]  # fish2
        # Alternatively, could have used cons_baseline[1] if using return_cons_baseline=True

    # Loop through each food position
    for food_pos in food_pos_grid:
        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            fish_positions=np.array([fish1.position, fish2.position]),
            fish_orientations=np.array([fish1.orientation, fish2.orientation]),
            fish_monopole_positions=np.stack(
                [fish1.monopole_positions_ego, fish2.monopole_positions_ego]
            ),
            fish_monopole_charges=np.stack(
                [fish1.monopole_charges, fish2.monopole_charges]
            ),
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=fish_eods,
            fish_sensor_positions=np.stack(
                [fish1.mormyromast_positions_ego, fish2.mormyromast_positions_ego]
            ),
            fish_sensor_normals=np.stack(
                [fish1.mormyromast_normals_ego, fish2.mormyromast_normals_ego]
            ),
            conductor_positions=np.array([food_pos]),
            conductor_contrasts=np.array([food_contrast]),
            conductor_radii=np.array([food_radius_cm]),
            arena_size=arena_size,
            do_induced=True,  # Always true for food sensing
            fish_contrasts=fish_contrasts,
            fish_radii=fish_radii,
            fish_sensor_type_mask=None,
            arena_clipping_distance=None,
            do_reflection=False,
            return_raw_field=False,
            max_charge_allowed=fish_charge,
        )
        result = sensor_measurements[1]  # fish2's sensors

        if criterion == 'max':
            fn = np.max
        elif criterion == 'mean':
            fn = np.mean
        elif criterion == 'min':
            fn = np.min

        # Calculate the change due to the food
        if dynamic_baseline is None:
            val = fn(np.abs(result))

        if dynamic_baseline == "no_food":
            delta = result - baseline
            val = fn(np.abs(delta))

        if dynamic_baseline == "mean":
            result_signs, result_magnitudes = np.sign(result), np.abs(result)
            print(
                "Mean, Median, Max, Min",
                np.mean(result_magnitudes),
                np.median(result_magnitudes),
                np.max(result_magnitudes),
                np.min(result_magnitudes),
            )
            result_magnitudes = result_magnitudes - np.mean(result_magnitudes)
            result = result_signs * np.abs(result_magnitudes)
            val = fn(np.abs(result))

        if dynamic_baseline == "min_max":
            result_signs, result_magnitudes = np.sign(result), np.abs(result)
            result_magnitudes = (result_magnitudes - result_magnitudes.min()) / (
                result_magnitudes.max() - result_magnitudes.min()
            )
            print(
                "Mean, Median, Max, Min",
                np.mean(result_magnitudes),
                np.median(result_magnitudes),
                fn(result_magnitudes),
                np.min(result_magnitudes),
            )
            result = result_signs * np.abs(result_magnitudes)
            val = fn(np.abs(result))

        if dynamic_baseline == "spatial":
            from scipy.ndimage import gaussian_filter1d

            sigma_value = 5
            baseline = gaussian_filter1d(result, sigma=sigma_value, mode="wrap")
            delta = result - baseline
            val = np.max(np.abs(delta))

        readings.append([food_pos[0], food_pos[1], val])

    return np.array(readings), fish1, fish2


def run_direct_eod_sensing_experiment_fixed_midpoint_food_vary_fish(
    num_points=50,
    do_self_induced=False,
    buffer_cm=20,
    dist_fish=15,
    arena_size=(100, 100),
    fish_config=1,
    dynamic_baseline="fixed_food",
    fish_charge=5.56e-10,
):
    """
    Fix food at the midpoint between the two fish's *initial* positions, use that
    reading as a baseline, and vary the non-centered fish's (Fish 1) position.

    dynamic_baseline options:
      - None:        use raw magnitudes at Fish 2
      - "no_food":   subtract baseline taken with no food
      - "fixed_food": subtract baseline taken with food fixed at initial midpoint
      - "mean":      mean-center magnitudes (sign preserved)
      - "min_max":   min-max scale magnitudes (sign preserved)
      - "spatial":   subtract a spatially smoothed (1D Gaussian) version of result
    """
    center = np.array(arena_size) / 2

    # --- Fixed fish arrangement (no orientation changes inside the loop) ---
    if fish_config == 1:
        fish1_pos = center + np.array([dist_fish / np.sqrt(2), dist_fish / np.sqrt(2)])
        fish1_orientation = np.pi       # left
        fish2_pos = center
        fish2_orientation = np.pi / 2   # up
    elif fish_config == 2:
        fish1_pos = center + np.array([0, dist_fish])
        fish1_orientation = np.pi / 2   # up
        fish2_pos = center
        fish2_orientation = np.pi / 2   # up
    elif fish_config == 3:
        fish1_pos = center - np.array([0, dist_fish])
        fish1_orientation = np.pi / 2   # up
        fish2_pos = center
        fish2_orientation = np.pi / 2   # up
    else:
        raise ValueError("Invalid fish_config. Must be 1, 2, or 3.")

    fish1_init = SimpleEFishAgent(fish1_pos, fish1_orientation)
    fish2 = SimpleEFishAgent(fish2_pos, fish2_orientation)

    fish1_init.monopole_charges = np.array([fish_charge, -fish_charge])
    fish2.monopole_charges = np.array([fish_charge, -fish_charge])  # fish2 is not emitting

    # Grid for varying Fish 1's position
    grid_x = np.linspace(center[0] - buffer_cm, center[0] + buffer_cm, num_points)
    grid_y = np.linspace(center[1] - buffer_cm, center[1] + buffer_cm, num_points)
    fish1_pos_grid = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)

    food_contrast = ELECTRIC_CONSTANTS["food_contrast"]
    food_radius_cm = ENV_PARAMS["food_radius_cm"]

    # EOD / induction parameters
    fish_eods = np.array([True, False])  # Fish 1 emits, Fish 2 senses
    fish_contrasts = np.array([
        ELECTRIC_CONSTANTS["fish_contrast"],
        ELECTRIC_CONSTANTS["fish_contrast"],
    ])
    fish_radii = np.array([
        fish1_init.body_radius_m * m_to_cm,
        fish2.body_radius_m * m_to_cm,
    ])
    if not do_self_induced:
        fish_contrasts = fish_contrasts * (~fish_eods).astype(float)

    # --- Baselines ---
    baseline_no_food = None
    baseline_fixed_food = None

    # Fixed food position = midpoint between the fish's INITIAL positions
    fixed_food_pos = (fish1_init.position + fish2.position) / 2

    if dynamic_baseline in ("no_food", "fixed_food"):
        # Prepare common args
        common_kwargs = dict(
            fish_positions=np.array([fish1_init.position, fish2.position]),
            fish_orientations=np.array([fish1_init.orientation, fish2.orientation]),
            fish_monopole_positions=np.stack([
                fish1_init.monopole_positions_ego, fish2.monopole_positions_ego
            ]),
            fish_monopole_charges=np.stack([
                fish1_init.monopole_charges, fish2.monopole_charges
            ]),
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=fish_eods,
            fish_sensor_positions=np.stack([
                fish1_init.mormyromast_positions_ego, fish2.mormyromast_positions_ego
            ]),
            fish_sensor_normals=np.stack([
                fish1_init.mormyromast_normals_ego, fish2.mormyromast_normals_ego
            ]),
            arena_size=arena_size,
            do_induced=True,  # food sensing uses induced dipoles
            fish_contrasts=fish_contrasts,
            fish_radii=fish_radii,
            fish_sensor_type_mask=None,
            arena_clipping_distance=None,
            do_reflection=False,
            return_raw_field=False,
            max_charge_allowed=fish_charge,
        )

        if dynamic_baseline == "no_food":
            sm, _, _ = electric.fish_forward(
                conductor_positions=np.zeros((0, 2)),
                conductor_contrasts=np.array([]),
                conductor_radii=np.array([]),
                **common_kwargs,
            )
            baseline_no_food = sm[1]  # Fish 2 sensors

        if dynamic_baseline == "fixed_food":
            sm, _, _ = electric.fish_forward(
                conductor_positions=np.array([fixed_food_pos]),
                conductor_contrasts=np.array([food_contrast]),
                conductor_radii=np.array([food_radius_cm]),
                **common_kwargs,
            )
            baseline_fixed_food = sm[1]  # Fish 2 sensors

    # --- Sweep Fish 1 positions (Fish 2 + food stay fixed) ---
    readings = []
    for pos in fish1_pos_grid:
        fish1 = SimpleEFishAgent(pos, fish1_init.orientation)  # same orientation
        fish1.monopole_charges = np.array([fish_charge, -fish_charge])

        sm, _, _ = electric.fish_forward(
            fish_positions=np.array([fish1.position, fish2.position]),
            fish_orientations=np.array([fish1.orientation, fish2.orientation]),
            fish_monopole_positions=np.stack([
                fish1.monopole_positions_ego, fish2.monopole_positions_ego
            ]),
            fish_monopole_charges=np.stack([
                fish1.monopole_charges, fish2.monopole_charges
            ]),
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=fish_eods,
            fish_sensor_positions=np.stack([
                fish1.mormyromast_positions_ego, fish2.mormyromast_positions_ego
            ]),
            fish_sensor_normals=np.stack([
                fish1.mormyromast_normals_ego, fish2.mormyromast_normals_ego
            ]),
            conductor_positions=np.array([fixed_food_pos]),  # FIXED food
            conductor_contrasts=np.array([food_contrast]),
            conductor_radii=np.array([food_radius_cm]),
            arena_size=arena_size,
            do_induced=True,  # induced dipoles on for food sensing
            fish_contrasts=fish_contrasts,
            fish_radii=fish_radii,
            fish_sensor_type_mask=None,
            arena_clipping_distance=None,
            do_reflection=False,
            return_raw_field=False,
            max_charge_allowed=fish_charge,
        )
        result = sm[1]  # Fish 2 sensors

        # --- Post-processing / baseline subtraction ---
        if dynamic_baseline is None:
            max_val = np.max(np.abs(result))

        elif dynamic_baseline == "no_food":
            delta = result - baseline_no_food
            max_val = np.max(np.abs(delta))

        elif dynamic_baseline == "fixed_food":
            delta = result - baseline_fixed_food
            max_val = np.max(np.abs(delta))

        elif dynamic_baseline == "mean":
            signs, mags = np.sign(result), np.abs(result)
            mags = mags - np.mean(mags)
            max_val = np.max(np.abs(signs * np.abs(mags)))

        elif dynamic_baseline == "min_max":
            signs, mags = np.sign(result), np.abs(result)
            rng = (mags.max() - mags.min())
            mags = (mags - mags.min()) / (rng if rng != 0 else 1.0)
            max_val = np.max(np.abs(signs * np.abs(mags)))

        elif dynamic_baseline == "spatial":
            from scipy.ndimage import gaussian_filter1d
            sigma_value = 5
            baseline_spatial = gaussian_filter1d(result, sigma=sigma_value, mode="wrap")
            delta = result - baseline_spatial
            max_val = np.max(np.abs(delta))

        else:
            raise ValueError(f"Unknown dynamic_baseline option: {dynamic_baseline}")

        readings.append([pos[0], pos[1], max_val])

    # Return initial fish1 (for reference), not the last loop instance
    return np.array(readings), fish1_init, fish2


def virtual_sense_food(
    fish_orientation=np.pi / 2,
    num_points=50,
    do_self_induced=True,
    arena_size=(100, 100),
    buffer_cm=10,
    fish_charge=5.56e-10,
    criterion='min',
    no_food=False,
    virtual_baseline_multiplier=1.0,
    sensor_idx=None
):
    # What is the max sensor reading due to self EOD with food at various locations?

    center = np.array(arena_size) / 2
    fish = SimpleEFishAgent(center, fish_orientation)

    food_x = np.linspace(center[0] - buffer_cm, center[0] + buffer_cm, num_points)
    food_y = np.linspace(center[1] - buffer_cm, center[1] + buffer_cm, num_points)
    food_pos_grid = np.array(np.meshgrid(food_x, food_y)).T.reshape(-1, 2)
    # food_pos_grid = make_axis_aligned_food_positions(center, buffer_cm=12, num_points=50)

    fish.monopole_charges = np.array(
        [fish_charge, -fish_charge]
    )  # Two monopoles with opposite charges

    # Baseline (no food)
    sensor_measurements, center_field_dict, _ = electric.fish_forward(
        fish_positions=np.array([fish.position]),
        fish_orientations=np.array([fish.orientation]),
        fish_monopole_positions=fish.monopole_positions_ego[None],
        fish_monopole_charges=fish.monopole_charges[None],
        fish_dipole_positions=np.zeros((1, 0, 2)),
        fish_dipole_moments=np.zeros((1, 0, 2)),
        fish_eods=np.array([True]),
        fish_sensor_positions=fish.mormyromast_positions_ego[None],
        fish_sensor_normals=fish.mormyromast_normals_ego[None],
        fish_sensor_type_mask=None,
        conductor_positions=np.zeros((0, 2)),
        conductor_contrasts=np.array([]),
        conductor_radii=np.array([]),
        arena_size=arena_size,
        arena_clipping_distance=None,
        do_induced=do_self_induced,
        fish_contrasts=(
            np.array([ELECTRIC_CONSTANTS["fish_contrast"]]) if do_self_induced else None
        ),
        fish_radii=(
            np.array([fish.body_radius_m * m_to_cm]) if do_self_induced else None
        ),
        do_reflection=False,
        return_raw_field=False,
    )
    mormyromast_cd = sensor_measurements[0]

    food_contrast = ELECTRIC_CONSTANTS["food_contrast"]
    food_radius_cm = ENV_PARAMS["food_radius_cm"]
    readings_self = []
    all_readings = []

    for food_pos in food_pos_grid:
        if no_food:
            conductor_positions=np.zeros((0, 2))
            conductor_contrasts=np.array([])
            conductor_radii=np.array([])
        else:
            conductor_positions=np.array([food_pos])
            conductor_contrasts=np.array([food_contrast])
            conductor_radii=np.array([food_radius_cm])


        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            fish_positions=np.array([fish.position]),
            fish_orientations=np.array([fish.orientation]),
            fish_monopole_positions=fish.monopole_positions_ego[None],
            fish_monopole_charges=fish.monopole_charges[None],
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=np.array([True]),
            fish_sensor_positions=fish.mormyromast_positions_ego[None],
            fish_sensor_normals=fish.mormyromast_normals_ego[None],
            fish_sensor_type_mask=None,
            conductor_positions=conductor_positions,
            conductor_contrasts=conductor_contrasts,
            conductor_radii=conductor_radii,
            arena_size=arena_size,
            arena_clipping_distance=None,
            do_induced=True,
            fish_contrasts=(
                np.array([ELECTRIC_CONSTANTS["fish_contrast"]])
                if do_self_induced
                else None
            ),
            fish_radii=(
                np.array([fish.body_radius_m * m_to_cm]) if do_self_induced else None
            ),
            do_reflection=False,
            return_raw_field=False,
        )
        reading = sensor_measurements[0]

        delta = reading - mormyromast_cd * virtual_baseline_multiplier

        if sensor_idx is not None:
            val = np.abs(delta[sensor_idx])
        else:
            if criterion == 'max':
                val = np.max(np.abs(delta))
            elif criterion == 'mean':
                val = np.mean(np.abs(delta))
            elif criterion == 'median':
                val = np.median(np.abs(delta))
            elif criterion == 'min':
                val = np.min(np.abs(delta))
            elif criterion == 'angular_residual':
                baseline_hat, residual, _, _ = angular_residual_log(reading, fish.mormyromast_angles)
                val = np.max(np.abs(10**residual))
            elif criterion == "axis":
                val = np.max(np.abs(delta[0]))

        readings_self.append([food_pos[0], food_pos[1], val])
        all_reading = np.concatenate([food_pos, reading])
        all_readings.append(all_reading)

    return np.array(readings_self), fish, np.array(all_readings)
# agent.mormyromast_observations_real = ampullary_and_mormyromast_measurements[agent_id][agent.mormyromast_indices_real]
# stacked = np.tile(agent.mormyromast_observations_real[None, :], (agent.num_morm_sets, 1))
# stacked -= agent.mormyromast_cd * agent.mormyromast_baselines[:, None]



def virtual_sense_agent_food(
    num_points=50,
    do_self_induced=False,
    buffer_cm=20,
    dist_fish=15,
    arena_size=(100, 100),
    fish_config=1,
    dynamic_baseline=None,
    fish_charge=5.56e-10,
    criterion='max',
    virtual_baseline_multiplier=1.0,
    sensor_idx=None,
    sample_mode="grid", # "grid" (for 2d) or "line" (for 1d)
    fish_food_buffer_cm=1.25,
    no_food=False,
):
    """
    Simulates Fish 2 sensing a food object moving around
    which is the EOD emitter.

    Parameters:
    - num_points: Number of grid points for the food's movement.
    - do_induced: Whether to include induced dipoles.
    - buffer_cm: Buffer distance for the food grid.
    - arena_size: Size of the simulation arena.
    - fish_config: An integer (1, 2, or 3) specifying the fish configuration.
    """
    center = np.array(arena_size) / 2

    # Set up fish positions and orientations based on the configuration
    if fish_config == 1:
        # Fish 1 (facing left) is to the upper right of Fish 2 (facing up)
        fish1_pos = center + np.array([dist_fish / np.sqrt(2), dist_fish / np.sqrt(2)])
        fish1_orientation = np.pi  # Facing left
        fish2_pos = center
        fish2_orientation = np.pi / 2  # Facing up
    elif fish_config == 2:
        # Fish 1 (facing DOWN) is above Fish 2 (facing up)
        fish1_pos = center + np.array([0, dist_fish])
        fish1_orientation = -np.pi / 2  # Facing DOWN
        fish2_pos = center
        fish2_orientation = np.pi / 2  # Facing up
    elif fish_config == 3:
        # Fish 1 (facing up) is below Fish 2 (facing up)
        fish1_pos = center - np.array([0, dist_fish])
        fish1_orientation = np.pi / 2  # Facing up
        fish2_pos = center
        fish2_orientation = np.pi / 2  # Facing up
    else:
        raise ValueError("Invalid fish_config. Must be 1, 2, or 3.")

    fish1 = SimpleEFishAgent(fish1_pos, fish1_orientation)
    fish2 = SimpleEFishAgent(fish2_pos, fish2_orientation)
    fish1.monopole_charges = np.array([fish_charge, -fish_charge])
    fish2.monopole_charges = np.array(
        [fish_charge, -fish_charge]
    )  # Not used since fish2 is not emitting

    # Define the food's movement grid
    # food_x = np.linspace(center[0] - buffer_cm, center[0] + buffer_cm, num_points)
    # food_y = np.linspace(center[1] - buffer_cm, center[1] + buffer_cm, num_points)
    # food_pos_grid = np.array(np.meshgrid(food_x, food_y)).T.reshape(-1, 2)
    # ------- Sampling pattern -------
    if sample_mode == "grid":
        # Define the food's movement grid (original behavior)
        food_x = np.linspace(center[0] - buffer_cm, center[0] + buffer_cm, num_points)
        food_y = np.linspace(center[1] - buffer_cm, center[1] + buffer_cm, num_points)
        food_pos_grid = np.array(np.meshgrid(food_x, food_y)).T.reshape(-1, 2)
    elif sample_mode == "line":
        # Sample ONLY along the line from Fish2 -> Fish1 (with optional extension)
        vec = fish1.position - fish2.position
        L = np.linalg.norm(vec)

        if L == 0:
            raise ValueError("Fish 1 and Fish 2 positions coincide; cannot define a line.")
        u = vec / L # unit vector along the line
        s_vals = np.linspace(fish_food_buffer_cm, L - fish_food_buffer_cm, num_points)
        food_pos_grid = np.stack([fish2.position + s * u for s in s_vals], axis=0)
        print(f"Sampling line from {food_pos_grid[0]} to {food_pos_grid[-1]}, length {np.linalg.norm(food_pos_grid[-1]-food_pos_grid[0])} cm, num_points={num_points}")
        # for pos in food_pos_grid:
        #     print(pos)
    else:
        raise ValueError("sample_mode must be 'grid' or 'line'.")


    food_contrast = ELECTRIC_CONSTANTS["food_contrast"]
    food_radius_cm = ENV_PARAMS["food_radius_cm"]

    readings = []

    fish_eods = np.array([True, False])  # Fish 1 emits, Fish 2 senses without emitting
    fish_contrasts = np.array(
        [
            ELECTRIC_CONSTANTS["fish_contrast"],
            ELECTRIC_CONSTANTS["fish_contrast"],
        ]
    )
    fish_radii = np.array(
        [
            fish1.body_radius_m * m_to_cm,  # Fish 1's radius
            fish2.body_radius_m * m_to_cm,  # Fish 2's radius
        ]
    )
    if not do_self_induced:
        fish_contrasts = fish_contrasts * (~fish_eods).astype(float)

    sensor_measurements, center_field_dict, _ = electric.fish_forward(
        fish_positions=np.array([fish2.position]),
        fish_orientations=np.array([fish2.orientation]),
        fish_monopole_positions=fish2.monopole_positions_ego[None],
        fish_monopole_charges=fish2.monopole_charges[None],
        fish_dipole_positions=np.zeros((1, 0, 2)),
        fish_dipole_moments=np.zeros((1, 0, 2)),
        fish_eods=np.array([True]),
        fish_sensor_positions=fish2.mormyromast_positions_ego[None],
        fish_sensor_normals=fish2.mormyromast_normals_ego[None],
        fish_sensor_type_mask=None,
        conductor_positions=np.zeros((0, 2)),
        conductor_contrasts=np.array([]),
        conductor_radii=np.array([]),
        arena_size=arena_size,
        arena_clipping_distance=None,
        do_induced=do_self_induced,
        fish_contrasts=(
            np.array([ELECTRIC_CONSTANTS["fish_contrast"]]) if do_self_induced else None
        ),
        fish_radii=(
            np.array([fish2.body_radius_m * m_to_cm]) if do_self_induced else None
        ),
        do_reflection=False,
        return_raw_field=False,
    )
    mormyromast_cd = sensor_measurements[0]

    # Loop through each food position
    for food_pos in food_pos_grid:
        sensor_measurements, center_field_dict, cons_baseline = electric.fish_forward(
            fish_positions=np.array([fish1.position, fish2.position]),
            fish_orientations=np.array([fish1.orientation, fish2.orientation]),
            fish_monopole_positions=np.stack(
                [fish1.monopole_positions_ego, fish2.monopole_positions_ego]
            ),
            fish_monopole_charges=np.stack(
                [fish1.monopole_charges, fish2.monopole_charges]
            ),
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=fish_eods,
            fish_sensor_positions=np.stack(
                [fish1.mormyromast_positions_ego, fish2.mormyromast_positions_ego]
            ),
            fish_sensor_normals=np.stack(
                [fish1.mormyromast_normals_ego, fish2.mormyromast_normals_ego]
            ),
            conductor_positions=np.array([food_pos]) if not no_food else np.zeros((0,2)),
            conductor_contrasts=np.array([food_contrast]) if not no_food else np.array([]),
            conductor_radii=np.array([food_radius_cm]) if not no_food else np.array([]),
            arena_size=arena_size,
            do_induced=True,  # Always true for food sensing
            fish_contrasts=fish_contrasts,
            fish_radii=fish_radii,
            fish_sensor_type_mask=None,
            arena_clipping_distance=None,
            do_reflection=False,
            return_raw_field=False,
            max_charge_allowed=fish_charge,
        )
        reading = sensor_measurements[1]  # fish2's sensors

        if criterion == 'max':
            fn = np.max
        elif criterion == 'mean':
            fn = np.mean
        elif criterion == 'min':
            fn = np.min

        delta = reading - (mormyromast_cd * virtual_baseline_multiplier)
        # delta = reading 

        if sensor_idx is not None:
            val = np.abs(delta[sensor_idx])
        else:
            val = fn(np.abs(delta))

        readings.append([food_pos[0], food_pos[1], val])

    return np.array(readings), fish1, fish2



# def compute_contour_distance_ranges(contour_set, fish_center):
#     """
#     Given a Matplotlib QuadContourSet and the (x,y) of a fish,
#     compute for each contour level the minimum and maximum
#     Euclidean distances from the fish center to the contour lines.

#     Parameters
#     ----------
#     contour_set : matplotlib.contour.QuadContourSet
#         The contour object returned by ax.contour(...)
#     fish_center : tuple of float (x_center, y_center)
#         The 2D coordinates of the fish.

#     Returns
#     -------
#     dict[level_value, (min_distance, max_distance)]
#     """
#     x_center, y_center = fish_center
#     distance_ranges_by_level = {}

#     # contour_set.levels is a list of the contour values
#     # contour_set.allsegs[level_index] is a list of segments for that level
#     for level_index, contour_value in enumerate(contour_set.levels):
#         segment_list = contour_set.allsegs[level_index]
#         if not segment_list:
#             continue  # no line segments for this level

#         # concatenate distances for every point in every segment
#         all_distances = np.hstack([
#             np.hypot(segment[:, 0] - x_center,
#                      segment[:, 1] - y_center)
#             for segment in segment_list
#             if segment.size > 0
#         ])
#         if all_distances.size > 0:
#             distance_ranges_by_level[contour_value] = (
#                 float(all_distances.min()),
#                 float(all_distances.max())
#             )

#     return distance_ranges_by_level


def compute_contour_distance_ranges(contour_set, fish_center):
    x0, y0 = fish_center
    ranges_by_level = {}

    for lvl_idx, lvl in enumerate(contour_set.levels):
        segs = contour_set.allsegs[lvl_idx]
        per_seg = []
        for seg in segs:
            if len(seg) == 0:
                continue
            d = np.hypot(seg[:, 0] - x0, seg[:, 1] - y0)
            per_seg.append((round(float(d.min()), 2), round(float(d.max()), 2)))
        if per_seg:
            ranges_by_level[lvl] = per_seg

    return ranges_by_level


def plot_image_fish2(
    readings,
    fish1=None,
    fish2=None,
    type="cons",
    log_scale=True,
    vmin=None,
    vmax=None,
    contour_levels=None,
    cmap="hot",
    criterion="max",
):
    """
    Visualize a 2D grid of max sensor readings for Fish 2.

    Parameters:
    - readings: np.ndarray of shape (N, 3): [x, y, value]
    - fish1, fish2: SimpleEFishAgent instances
    - type: 'cons' or 'self'
    - log_scale: Apply log10 transform to values
    - vmin, vmax: Value limits for color scaling
    - contour_levels: Optional list of contour levels to overlay
    - cmap: Matplotlib colormap
    """
    # Infer grid
    num_x = len(np.unique(readings[:, 0]))
    num_y = len(np.unique(readings[:, 1]))
    grid_x = readings[:, 0].reshape((num_y, num_x))
    grid_y = readings[:, 1].reshape((num_y, num_x))
    raw_vals = readings[:, 2].reshape((num_y, num_x))

    # Apply scaling
    if log_scale:
        with np.errstate(divide="ignore"):
            grid_z = np.log10(np.clip(raw_vals, a_min=1e-10, a_max=None))
        colorbar_label = f"log10({criterion.capitalize()} Sensor Reading) (V/m)"
    else:
        grid_z = raw_vals
        colorbar_label = f"{criterion.capitalize()} Sensor Reading (V/m)"

    # Plot main image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(
        f"{criterion.capitalize()} Collective Sensing Reading on Fish 2"
        if type == "cons"
        else f"{criterion.capitalize()} Self Sensing Reading on Fish 2"
    )
    cs = ax.contourf(grid_x, grid_y, grid_z, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label(colorbar_label)

    # Optional contour overlays
    if contour_levels is not None and fish2:
        contour_lines = ax.contour(
            grid_x,
            grid_y,
            grid_z,
            levels=contour_levels,
            colors="white",
            linewidths=0.8,
        )
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")
        ranges = compute_contour_distance_ranges(contour_lines, fish2.position)
        print(f"Contour distance ranges from fish at {fish2.position}:")
        for level, dists in ranges.items():
            print(f"  Level {level}: min/max distances = {dists} cm")

    # Fish 2 circle and dipoles
    if fish2:
        ax.add_patch(
            plt.Circle(
                fish2.position,
                fish2.body_radius_m * m_to_cm,
                color="lightblue",
                label="Fish 2",
            )
        )
        for pos, charge in zip(
            fish2.get_world_monopole_positions(), fish2.monopole_charges
        ):
            ax.plot(*pos, "o", color="red" if charge > 0 else "blue", markersize=6)

    # Fish 1 circle and dipoles (if cons-image)
    if type == "cons" and fish1:
        ax.add_patch(
            plt.Circle(
                fish1.position,
                fish1.body_radius_m * m_to_cm,
                color="lightgray",
                label="Fish 1",
            )
        )
        for pos, charge in zip(
            fish1.get_world_monopole_positions(), fish1.monopole_charges
        ):
            ax.plot(*pos, "o", color="red" if charge > 0 else "blue", markersize=6)

    ax.set_xlabel("X Position (cm)")
    ax.set_ylabel("Y Position (cm)")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize="small", framealpha=1.0)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"readings_{type}_sensing_other_agent.png", dpi=300)
    # print(f"Saved max readings plot to readings_{type}_sensing_other_agent.png")

    # Plot histogram
    try:
        plt.figure(figsize=(6, 3))
        hist_vals = np.log10(readings[:, 2]) if log_scale else readings[:, 2]
        plt.hist(hist_vals, bins=50, color="blue", alpha=0.7)
        hist_label = f"log10({criterion.capitalize()} Sensor Reading)" if log_scale else "Max Sensor Reading"
        plt.title(
            f"Histogram of {criterion.capitalize()} {'Cons-' if type == 'cons' else 'Self-'}Sensing Readings"
        )
        plt.xlabel(hist_label + " (V/m)")
        plt.ylabel("Frequency")
        plt.yscale("log" if log_scale else "linear")
        plt.grid()
        plt.tight_layout()
        plt.show()

        print(
            f"{type.capitalize()} sensing: colormap range = [{grid_z.min():.2e}, {grid_z.max():.2e}]"
        )
    except Exception as e:
        print(f"Failed to plot histogram: {e}")


def plot_line_fish2(
    readings,
    no_food_readings=None,
    fish1=None,
    fish2=None,
    type="cons",
    log_scale=True,
    criterion="max",
    annotate_agents=True,
    fish_food_buffer_cm=1.25,
    shade_difference=True,
):
    """
    Visualize line-sampled readings for Fish 2:
    - readings: np.ndarray (N,3) with columns [x, y, value]  (WITH food)
    - no_food_readings: same shape (baseline WITHOUT food); if provided, it will be overlaid
    - Projects (x,y) onto the line from Fish2 -> Fish1, plotting value vs. arc-length 's'.
    """
    if fish2 is None or fish1 is None:
        raise ValueError("plot_line_fish2 requires both fish1 and fish2 to be provided.")

    # Unit vector along Fish2 -> Fish1
    vec = fish1.position - fish2.position
    L = float(np.linalg.norm(vec))
    if L == 0:
        raise ValueError("Fish 1 and Fish 2 positions coincide; cannot define a line.")
    u = vec / L

    def project_and_sort(arr):
        XY = arr[:, :2]
        vals = arr[:, 2]
        s = np.dot((XY - fish2.position), u)         # signed arc-length from Fish 2
        idx = np.argsort(s)
        return s[idx], vals[idx]

    # With-food series (primary)
    s_food, v_food = project_and_sort(readings)

    # Prepare labels/scales
    y_label = f"{criterion.capitalize()} Sensor Reading (V/m)"
    title_base = f"{criterion.capitalize()} {'Cons' if type=='cons' else 'Self'}-Sensing on Fish 2"
    if no_food_readings is not None:
        title_base += "  (with baseline overlay)"

    plt.figure(figsize=(9, 3.6))
    line_food, = plt.plot(s_food, v_food, lw=2, marker='o', markersize=3, alpha=0.9, label="With food")

    # Optional: overlay baseline (no food)
    if no_food_readings is not None:
        s_base, v_base = project_and_sort(no_food_readings)
        # Draw the baseline as a dashed line; independent grid okay for overlay
        line_base, = plt.plot(s_base, v_base, lw=2, linestyle='-', alpha=0.9, label="No food (baseline)")

    # Axes cosmetics
    plt.title(title_base)
    plt.xlabel("Arc-length along line from Fish 2 → Fish 1 (cm)")
    plt.ylabel(y_label)
    if log_scale:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", frameon=True)

    # Annotations for agent positions \ buffers (after plotting to use final y-lims)
    if annotate_agents:
        y_top = plt.ylim()[1]
        # Fish 2 at s=0
        plt.axvline(0.0, linestyle="--", alpha=0.6)
        plt.text(0.0, y_top, "Fish 2", ha="left", va="top")
        plt.axvline(0.0 + fish_food_buffer_cm, linestyle="--", alpha=0.6)
        plt.text(0.0 + fish_food_buffer_cm, y_top, "F2+Buffer", ha="left", va="top")
        # Fish 1 at s=L
        plt.axvline(L, linestyle="--", alpha=0.6)
        plt.text(L, y_top, "Fish 1", ha="right", va="top")
        plt.axvline(L - fish_food_buffer_cm, linestyle="--", alpha=0.6)
        plt.text(L - fish_food_buffer_cm, y_top, "F1+Buffer", ha="right", va="top")

    plt.tight_layout()
    plt.show()



def get_vmin_vmax(readings_self, readings_cons, log_scale):
    all_vals = np.concatenate([readings_self[:, 2], readings_cons[:, 2]])
    if log_scale:
        with np.errstate(divide="ignore"):
            all_vals = np.log10(np.clip(all_vals, 1e-10, None))
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)
    print(f"Global min/max values for colormap: vmin = {vmin:.2e}, vmax = {vmax:.2e}")
    return vmin, vmax


def plot_sensor_polar_log(
    sensor_values,
    sensor_angles,
    sensor_min,
    sensor_max,
    *,
    theta_offset=np.pi/2,
    base=10.0,
    label='',
    label_pos="sensor +",
    label_neg="sensor -",
    color_pos="blue",
    color_neg="red",
    hide_ticks=False,
):
    """
    Plot a sensor on a polar axis with a LOG radial scale.
    - sensor_values: 1D array of readings (can be signed; sign shown via +/– color)
    - sensor_angles: 1D array of angles in radians, same shape as sensor_values
    - sensor_min, sensor_max: positive axis limits for the radial (log) scale
    """

    # Create a fresh polar axis
    fig, polar_ax = plt.subplots(subplot_kw={"projection": "polar"})

    v = np.asarray(sensor_values, dtype=float)
    th = np.asarray(sensor_angles, dtype=float)
    if v.shape != th.shape:
        raise ValueError("sensor_values and sensor_angles must have the same shape.")

    # Validate radial limits (log scale needs strictly positive min)
    if not np.isfinite(sensor_min) or not np.isfinite(sensor_max):
        raise ValueError("sensor_min and sensor_max must be finite.")
    if sensor_min <= 0:
        raise ValueError("sensor_min must be > 0 for log scaling.")
    if sensor_max <= sensor_min:
        raise ValueError("sensor_max must be greater than sensor_min.")

    # Use absolute magnitude for radius; clamp to axis limits
    mag = np.clip(np.abs(v), sensor_min, sensor_max)

    # Configure polar axis
    polar_ax.set_rscale("log")                          # log radial scale
    polar_ax.set_rlim(sensor_min, sensor_max)           # impose axis limits
    # Pretty ticks (optional): match the chosen base
    try:
        polar_ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=base))
        polar_ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=base, subs=np.arange(2, int(base))))
    except Exception:
        pass  # fallback gracefully if base handling differs by Matplotlib version

    if theta_offset is not None:
        polar_ax.set_theta_offset(theta_offset)         # rotate so 0 rad points "up"

    polar_ax.plot(th, mag, color="red", alpha=1)  # plot all in gray as background
    polar_ax.set_title(label, va='bottom', fontsize=10)  # title for the plot
    # Split by sign for styling
    # pos_mask = v >= 0
    # neg_mask = ~pos_mask

    # if np.any(pos_mask):
    #     polar_ax.plot(th[pos_mask], mag[pos_mask], label=label_pos, color=color_pos)
    # if np.any(neg_mask):
    #     polar_ax.plot(th[neg_mask], mag[neg_mask], label=label_neg, color=color_neg)

    if hide_ticks:
        polar_ax.set_xticks([])
        polar_ax.set_yticks([])
        polar_ax.grid(False)
    else:
        polar_ax.grid(True)


# get an index of a sensor reading at 52.5, 52.5
# get the point closest to (52.5, 52.5)
def get_closest_reading_index(readings, point):
    distances = np.linalg.norm(readings[:, :2] - point, axis=1)
    return np.argmin(distances)
#### Factionated Mormyromast Sensing ####

def _self_eod_readings(arena_size=(100, 100), theta=np.pi/2):
    """Mormyromast sensor vector for a single fish emitting; no food, no induced/self-charge."""
    pos = np.array(arena_size) / 2.0
    fish = SimpleEFishAgent(position=pos, orientation=theta)

    sensor_measurements, _, _ = electric.fish_forward(
        fish_positions=np.array([fish.position]),
        fish_orientations=np.array([fish.orientation]),
        fish_monopole_positions=fish.monopole_positions_ego[None, ...],
        fish_monopole_charges=fish.monopole_charges[None, ...],
        fish_dipole_positions=np.zeros((1, 0, 2)),
        fish_dipole_moments=np.zeros((1, 0, 2)),
        fish_eods=np.array([True]),
        fish_sensor_positions=fish.mormyromast_positions_ego[None, ...],
        fish_sensor_normals=fish.mormyromast_normals_ego[None, ...],
        fish_sensor_type_mask=None,
        conductor_positions=np.zeros((0, 2)),     # no food
        conductor_contrasts=np.array([]),
        conductor_radii=np.array([]),
        arena_size=arena_size,
        arena_clipping_distance=None,
        do_induced=False,     # match morm CD logic: no induced self-charge
        do_intrinsics=False,  # morm doesn't use intrinsic dipoles
        do_reflection=False,
        return_raw_field=False,
    )
    return sensor_measurements[0]  # shape: (num_morm_sensors,)

def _cons_eod_readings(dist_cm, arena_size=(100, 100), theta=np.pi/2):
    """Mormyromast sensor vector at the receiver when the conspecific emits at a fixed distance."""
    center = np.array(arena_size) / 2.0
    emit = SimpleEFishAgent(position=center + np.array([0.0, dist_cm]), orientation=-theta) # above and facing recv (downwards)
    recv = SimpleEFishAgent(position=center, orientation=theta) # facing up 

    fish_positions = np.stack([emit.position, recv.position])
    fish_orientations = np.array([emit.orientation, recv.orientation])
    fish_eods = np.array([True, False])  # emitter on, receiver off

    # Allow induced charge on the receiver, but not on the emitting fish
    fish_contrasts = np.array([ELECTRIC_CONSTANTS["fish_contrast"],
                               ELECTRIC_CONSTANTS["fish_contrast"]], dtype=float)
    fish_contrasts = fish_contrasts * (~fish_eods).astype(float)

    fish_radii = np.array([emit.body_radius_m, recv.body_radius_m]) * m_to_cm  # cm

    sensor_positions = np.stack([emit.mormyromast_positions_ego,
                                 recv.mormyromast_positions_ego])
    sensor_normals = np.stack([emit.mormyromast_normals_ego,
                               recv.mormyromast_normals_ego])

    sensor_measurements, _, _ = electric.fish_forward(
        fish_positions=fish_positions,
        fish_orientations=fish_orientations,
        fish_monopole_positions=np.stack([emit.monopole_positions_ego,
                                          recv.monopole_positions_ego]),
        fish_monopole_charges=np.stack([emit.monopole_charges,
                                        recv.monopole_charges]),
        fish_dipole_positions=np.zeros((1, 0, 2)),
        fish_dipole_moments=np.zeros((1, 0, 2)),
        fish_eods=fish_eods,
        fish_sensor_positions=sensor_positions,
        fish_sensor_normals=sensor_normals,
        fish_sensor_type_mask=None,
        conductor_positions=np.zeros((0, 2)),
        conductor_contrasts=np.array([]),
        conductor_radii=np.array([]),
        arena_size=arena_size,
        arena_clipping_distance=None,
        do_induced=True,          # cons EOD can induce on receiver
        fish_contrasts=fish_contrasts,
        fish_radii=fish_radii,
        do_reflection=False,
        return_raw_field=False,
    )
    return sensor_measurements[1]  # receiver's morm array


def plot_morm_histograms(distances_cm=(3,5,7,9),
                         arena_size=(100,100),
                         bins=60,
                         mode="abs",   # "abs" | "signed" | "logabs"
                         percentiles=(50, 90, 95, 99)):
    """
    Make histograms of mormyromast sensor amplitudes:
      (1) self-EOD (no food)
      (2) cons-EOD at specified distances (cm) on-axis
    Prints percentile stats for quick baseline selection.
    """

    def _transform(x):
        x = np.asarray(x)
        if mode == "abs":
            return np.abs(x)
        elif mode == "signed":
            return x
        elif mode == "logabs":
            return np.log10(np.maximum(np.abs(x), 1e-25))
        else:
            raise ValueError("mode must be one of {'abs','signed','logabs'}")

    # --- Collect data
    self_vec = _transform(_self_eod_readings(arena_size=arena_size))
    cons_map = {d: _transform(_cons_eod_readings(d, arena_size=arena_size))
                for d in distances_cm}

    # --- Plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), constrained_layout=True)

    # Left: self-EOD
    axes[0].hist(self_vec, bins=bins, density=True, alpha=0.8)
    axes[0].set_title(f"Self-EOD (no food) — morm sensors [{mode}]")
    axes[0].set_xlabel("sensor amplitude")
    axes[0].set_ylabel("density")

    # Right: cons-EOD overlays
    for d, v in cons_map.items():
        axes[1].hist(v, bins=bins, density=True, alpha=0.5, label=f"{d} cm")
    axes[1].set_title(f"Cons-EOD at distance — morm sensors [{mode}]")
    axes[1].set_xlabel("sensor amplitude")
    axes[1].legend(title="distance")

    plt.show()

    # --- Print helpful percentiles
    def pct_str(x):
        return ", ".join([f"p{p}={np.percentile(x, p):.3e}" for p in percentiles])

    print("[Self-EOD] percentiles:", pct_str(self_vec))
    for d, v in cons_map.items():
        print(f"[Cons-EOD @ {d:>2} cm] percentiles:", pct_str(v))

    return self_vec, cons_map
