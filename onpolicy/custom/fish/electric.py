import numpy as np
from numpy.testing import assert_allclose
import timeit

import matplotlib.pyplot as plt
import utils_electric as ue
from tqdm import tqdm
from cfg import ELECTRIC_CONSTANTS

k_coulomb = ELECTRIC_CONSTANTS["k_coulomb"]  # coulomb's constant
epsilon_0 = ELECTRIC_CONSTANTS["epsilon_0"]  # Vacuum permittivity (F/m)
REFLECTION_SCALE = ELECTRIC_CONSTANTS["reflection_scale"]  # Reflection scale factor
FLIP_ON_REFLECTION = ELECTRIC_CONSTANTS[
    "flip_on_reflection"
]  # Whether to flip monopole/dipole charges on reflection
CM_TO_M = 0.01  # Conversion factor from centimeters to meters

# --- Switches ---
CHECK_SHAPES = False  # flip off for speed once stable

# --- shape check utils ---
def _shape(x):
    return None if x is None else tuple(x.shape)

def _require(cond, msg):
    if CHECK_SHAPES and (not cond):
        raise ValueError(msg)

def require_shape(x, expected, name="array"):
    """expected: tuple like (N,2) where N can be int or None for 'any'."""
    if not CHECK_SHAPES:
        return
    _require(x is not None, f"{name} is None, expected shape {expected}")
    sx = x.shape
    _require(len(sx) == len(expected),
             f"{name} ndim={len(sx)}; expected {len(expected)}; got shape {sx}")
    for i, (got, exp) in enumerate(zip(sx, expected)):
        if exp is None:
            continue
        _require(got == exp, f"{name} shape mismatch at dim {i}: expected {expected}, got {sx}")

def require_shape_in(x, allowed, name="array"):
    """allowed: iterable of exact shapes, e.g. [(f,p,2), (1,p,2)]"""
    if not CHECK_SHAPES:
        return
    sx = _shape(x)
    _require(sx in allowed, f"{name} shape {sx} not in allowed {allowed}")

def require_xy(x, name="xy"):
    """Require last dimension is 2."""
    if not CHECK_SHAPES:
        return
    _require(x is not None, f"{name} is None")
    _require(x.ndim >= 1 and x.shape[-1] == 2, f"{name} must have last dim 2, got shape {x.shape}")

# --- physics utils ---
def measure_electric_field_original(
    measurement_positions_cm,  # [num-positions, x, y] in cm
    monopole_positions_cm=None,  # [num-monopoles, x, y] in cm
    monopole_charges=None,  # [num-monopoles] in Coulombs
    dipole_positions_cm=None,  # [num-dipoles, x, y] in cm
    dipole_moments=None,  # [num-dipoles]
    measurement_noise=None, # Deprecated for speed  # Volts/m OR Newtons/Coulomb
    eps_m=1e-5,  # m
):
    require_xy(measurement_positions_cm, "measurement_positions_cm")
    if monopole_positions_cm is not None:
        # axes: (m,p,2)
        p = monopole_positions_cm.shape[0]
        require_xy(monopole_positions_cm, "monopole_positions_cm")
        require_shape(monopole_positions_cm, (p, 2), "monopole_positions_cm")
        require_shape(monopole_charges, (p,), "monopole_charges")
    if dipole_positions_cm is not None:
        # axes: (m,d,2)
        d = dipole_positions_cm.shape[0]
        require_xy(dipole_positions_cm, "dipole_positions_cm")
        require_shape(dipole_positions_cm, (d, 2), "dipole_positions_cm")
        require_shape(dipole_moments, (d, 2), "dipole_moments")

    # units
    measurement_positions = measurement_positions_cm * CM_TO_M

    # initialize measurement gradients
    m = measurement_positions.shape[0]
    measured_gradients = np.zeros((m, 2))

    # monopoles
    if monopole_positions_cm is not None:

        # convert to meters
        monopole_positions = monopole_positions_cm * CM_TO_M

        # compute the gradients
        monopole_offsets = (
            measurement_positions[:, None, :] - monopole_positions[None, :, :]
        )
        monopole_distances = np.linalg.norm(monopole_offsets, axis=-1) + eps_m
        measured_gradients += np.sum(
            k_coulomb
            * monopole_charges[None, :, None]
            * monopole_offsets
            / (monopole_distances[:, :, None] ** 3),
            axis=1,
        )

    # dipoles
    if dipole_positions_cm is not None:

        # convert to meters
        dipole_positions = dipole_positions_cm * CM_TO_M

        # compute the gradients
        dipole_offsets = (
            measurement_positions[:, None, :] - dipole_positions[None, :, :]
        )
        dipole_distances = np.linalg.norm(dipole_offsets, axis=-1)[:, :, None] + eps_m
        moment_dot_offset = np.sum(
            dipole_moments[None, :, :] * dipole_offsets, axis=-1
        )[:, :, None]
        measured_gradients += k_coulomb * np.sum(
            3 * moment_dot_offset * dipole_offsets / dipole_distances**5
            - dipole_moments[None, :, :] / dipole_distances**3,
            axis=1,
        )

    # Deprecated for speed
    # add noise
    # measured_gradients += np.random.normal(size=(m, 2)) * measurement_noise

    return measured_gradients  # [num-positions, 2]

# Dispatch: backend selected at import time via cfg.py — no run-time switching.
# ELECTRIC_CONSTANTS["electric_backend"]: "original" (default) | "numba"
# Note: "fast" (vectorised NumPy) is intentionally excluded — dtype inheritance
# from measurement_positions causes silent float32 downcast that breaks trained networks.
_BACKEND = ELECTRIC_CONSTANTS.get("electric_backend", "original")

if _BACKEND == "original":
    measure_electric_field = measure_electric_field_original
elif _BACKEND == "numba":
    from electric_accelerated import measure_electric_field_numba, warmup as _warmup
    measure_electric_field = measure_electric_field_numba
    print("[electric] numba backend active — warming up...", flush=True)
    _warmup()
    print("[electric] warmup done.", flush=True)
else:
    raise ValueError(f"[electric] unknown electric_backend={_BACKEND!r}; expected 'original' or 'numba'")

def induce_dipoles(
    monopole_positions_cm,
    monopole_charges,
    dipole_positions_cm,
    dipole_moments,
    conductor_positions_cm,
    conductor_contrasts,
    conductor_radii_cm,
    **kwargs,
):

    initial_conductor_gradients = measure_electric_field(
        conductor_positions_cm,
        monopole_positions_cm,
        monopole_charges,
        dipole_positions_cm,
        dipole_moments,
        **kwargs,
    )

    # convert to meters
    conductor_radii = conductor_radii_cm * CM_TO_M

    # original version Aaron hacked from Chen paper
    """
    # compute the conductor moments
    conductor_moments = (
        conductor_contrasts[:, None] *
        conductor_radii[:, None]**3 *
        initial_conductor_gradients
    )
    """

    # better version from Claude
    # TODO: Citation needed!
    conductor_volumes = 4 / 3 * np.pi * conductor_radii**3
    conductor_moments = (
        3
        * epsilon_0
        * conductor_volumes[:, None]
        * conductor_contrasts[:, None]
        * initial_conductor_gradients
    )

    return conductor_moments


def clip_conductor_moments(
    conductor_moments,
    conductor_radii,
    max_charge_allowed=None,
):
    if max_charge_allowed is None:
        return conductor_moments
    magnitudes = np.linalg.norm(conductor_moments, axis=1)  # (N,)
    max_moment_magnitudes = max_charge_allowed * conductor_radii  # (N,)
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = max_moment_magnitudes / magnitudes  # (N,)
    factors = np.minimum(1.0, factors)  # don’t up‐scale small vectors
    factors = np.nan_to_num(factors, nan=1.0, posinf=1.0)  # for zero magnitudes

    conductor_moments = conductor_moments * factors[:, None]  # (N, d)
    return conductor_moments


def project_field(
    sensor_gradients,
    sensor_normals,
):
    # shape checking
    require_xy(sensor_gradients, "sensor_gradients")
    require_xy(sensor_normals, "sensor_normals")
    _require(sensor_gradients.shape == sensor_normals.shape,
             f"sensor_gradients shape {sensor_gradients.shape} must match sensor_normals shape {sensor_normals.shape}")

    # the sensor measurement is the dot product of the electrical gradient
    # with the sensor normal, clipped by the threshold values and rescaled
    # between 0 and 1
    sensor_measurements = np.sum(sensor_gradients * sensor_normals, axis=-1)
    return sensor_measurements


def rotate_and_translate(
    points_cm,
    orientations_rad,
    translations_cm=0.0,
):
    m = orientations_rad.shape[0]

    c = np.cos(orientations_rad)
    s = np.sin(orientations_rad)
    R = np.stack([np.stack([c, s], axis=-1), np.stack([-s, c], axis=-1)], axis=-1)
    tx = np.matmul(R, points_cm[..., None])[..., 0] + translations_cm
    return tx


def to_world_sources(
    fish_positions_cm,
    fish_orientations_rad,
    fish_monopole_positions_cm,
    fish_monopole_charges,
    fish_dipole_positions_cm,
    fish_dipole_moments,
    fish_eods,
):
    f = fish_positions_cm.shape[0]
    p = fish_monopole_positions_cm.shape[1]
    d = fish_dipole_positions_cm.shape[1]

    # check shapes
    require_shape(fish_positions_cm, (f, 2), "fish_positions_cm")
    require_shape(fish_orientations_rad, (f,), "fish_orientations_rad")
    require_shape_in(fish_monopole_positions_cm, [(f, p, 2), (1, p, 2)], "fish_monopole_positions_cm")
    require_shape_in(fish_monopole_charges,      [(f, p),   (1, p)],   "fish_monopole_charges")
    require_shape_in(fish_dipole_positions_cm, [(f, d, 2), (1, d, 2)], "fish_dipole_positions_cm")
    require_shape_in(fish_dipole_moments,      [(f, d, 2), (1, d, 2)], "fish_dipole_moments")
    require_shape(fish_eods, (f,), "fish_eods")

    # build the list of all field generators from the fish
    scene_global_monopole_positions = rotate_and_translate(
        fish_monopole_positions_cm,
        fish_orientations_rad[:, None, ...],
        fish_positions_cm[:, None, ...],
    ).reshape(-1, 2)
    scene_monopole_charges = (
        fish_monopole_charges * fish_eods.reshape((f, 1))
    ).reshape(
        -1,
    )
    scene_global_dipole_positions = rotate_and_translate(
        fish_dipole_positions_cm,
        fish_orientations_rad[:, None, ...],
        fish_positions_cm[:, None, ...],
    ).reshape(-1, 2)
    scene_global_dipole_moments = rotate_and_translate(
        fish_dipole_moments * fish_eods.reshape(f, 1, 1),
        fish_orientations_rad[:, None, ...],
    ).reshape(-1, 2)

    return (
        scene_global_monopole_positions,
        scene_monopole_charges,
        scene_global_dipole_positions,
        scene_global_dipole_moments,
    )


def to_world_sensors(
    fish_positions_cm,
    fish_orientations_rad,
    fish_sensor_positions_cm,
    fish_sensor_normals,
):
    f = fish_positions_cm.shape[0]
    s = fish_sensor_positions_cm.shape[1]

    # check shapes
    require_shape(fish_positions_cm, (f, 2), "fish_positions_cm")
    require_shape(fish_orientations_rad, (f,), "fish_orientations_rad")
    require_shape_in(fish_sensor_positions_cm, [(f, s, 2), (1, s, 2)], "fish_sensor_positions_cm")
    require_shape_in(fish_sensor_normals,      [(f, s, 2), (1, s, 2)], "fish_sensor_normals")

    fish_sensor_global_positions = rotate_and_translate(
        fish_sensor_positions_cm,
        fish_orientations_rad[:, None, ...],
        fish_positions_cm[:, None, ...],
    ).reshape(f * s, 2)
    fish_sensor_global_normals = rotate_and_translate(
        fish_sensor_normals,
        fish_orientations_rad[:, None, ...],
    ).reshape(f * s, 2)

    return fish_sensor_global_positions, fish_sensor_global_normals


# TODO: vectorize below
# TODO: check if iterative application within one num_reflection step is even required
def reflect_sources_iterative(
    monopole_positions_cm,
    monopole_charges,
    dipole_positions_cm,
    dipole_moments,
    arena_size_cm,
    num_reflections=1,  # >1 reflects previous reflections which may be unnecessary
    reflection_scale=REFLECTION_SCALE,
    flip_on_reflection=FLIP_ON_REFLECTION,
):
    """
    Reflect monopoles and dipoles across arena boundaries using method of images.

    Args:
        monopole_positions: (N,2) array of monopole positions
        monopole_charges: (N,) array of monopole charges
        dipole_positions: (M,2) array of dipole positions
        dipole_moments: (M,2) array of dipole moments
        arena_size: (width, height) of arena
        num_reflections: Number of reflections to compute

    Returns:
        Reflected positions and charges/moments including original ones
    """
    width, height = arena_size_cm

    # Initialize lists to store all reflections
    all_monopole_positions = [monopole_positions_cm]
    all_monopole_charges = [monopole_charges]
    all_dipole_positions = [dipole_positions_cm]
    all_dipole_moments = [dipole_moments]

    # Generate reflections
    for n in range(num_reflections):
        # Current positions to reflect
        curr_monopole_positions = all_monopole_positions[-1]
        curr_monopole_charges = all_monopole_charges[-1]
        curr_dipole_positions = all_dipole_positions[-1]
        curr_dipole_moments = all_dipole_moments[-1]

        # Reflect across each wall
        # Left wall (x=0)
        left_monopole_pos = curr_monopole_positions.copy()
        left_monopole_pos[:, 0] = -left_monopole_pos[:, 0]
        left_monopole_charges = curr_monopole_charges.copy()  # No need to flip
        if flip_on_reflection:
            left_monopole_charges = -left_monopole_charges

        left_dipole_pos = curr_dipole_positions.copy()
        left_dipole_pos[:, 0] = -left_dipole_pos[:, 0]
        left_dipole_moments = curr_dipole_moments.copy()
        if flip_on_reflection:
            left_dipole_moments[:, 0] = -left_dipole_moments[:, 0]

        # Right wall (x=width)
        right_monopole_pos = curr_monopole_positions.copy()
        right_monopole_pos[:, 0] = 2 * width - right_monopole_pos[:, 0]
        right_monopole_charges = curr_monopole_charges.copy()  # No need to flip
        if flip_on_reflection:
            right_monopole_charges = -right_monopole_charges

        right_dipole_pos = curr_dipole_positions.copy()
        right_dipole_pos[:, 0] = 2 * width - right_dipole_pos[:, 0]
        right_dipole_moments = curr_dipole_moments.copy()
        if flip_on_reflection:
            right_dipole_moments[:, 0] = -right_dipole_moments[:, 0]

        # Bottom wall (y=0)
        bottom_monopole_pos = curr_monopole_positions.copy()
        bottom_monopole_pos[:, 1] = -bottom_monopole_pos[:, 1]
        bottom_monopole_charges = curr_monopole_charges.copy()  # No need to flip
        if flip_on_reflection:
            bottom_monopole_charges = -bottom_monopole_charges

        bottom_dipole_pos = curr_dipole_positions.copy()
        bottom_dipole_pos[:, 1] = -bottom_dipole_pos[:, 1]
        bottom_dipole_moments = curr_dipole_moments.copy()
        if flip_on_reflection:
            bottom_dipole_moments[:, 1] = -bottom_dipole_moments[:, 1]

        # Top wall (y=height)
        top_monopole_pos = curr_monopole_positions.copy()
        top_monopole_pos[:, 1] = 2 * height - top_monopole_pos[:, 1]
        top_monopole_charges = curr_monopole_charges.copy()  # No need to flip
        if flip_on_reflection:
            top_monopole_charges = -top_monopole_charges

        top_dipole_pos = curr_dipole_positions.copy()
        top_dipole_pos[:, 1] = 2 * height - top_dipole_pos[:, 1]
        top_dipole_moments = curr_dipole_moments.copy()
        if flip_on_reflection:
            top_dipole_moments[:, 1] = -top_dipole_moments[:, 1]

        # Add all reflections
        all_monopole_positions.extend(
            [
                left_monopole_pos,
                right_monopole_pos,
                bottom_monopole_pos,
                top_monopole_pos,
            ]
        )
        all_monopole_charges.extend(
            [
                left_monopole_charges * reflection_scale,
                right_monopole_charges * reflection_scale,
                bottom_monopole_charges * reflection_scale,
                top_monopole_charges * reflection_scale,
            ]
        )
        all_dipole_positions.extend(
            [left_dipole_pos, right_dipole_pos, bottom_dipole_pos, top_dipole_pos]
        )
        all_dipole_moments.extend(
            [
                left_dipole_moments * reflection_scale,
                right_dipole_moments * reflection_scale,
                bottom_dipole_moments * reflection_scale,
                top_dipole_moments * reflection_scale,
            ]
        )

    # Concatenate all reflections
    reflected_monopole_positions = np.concatenate(all_monopole_positions, axis=0)
    reflected_monopole_charges = np.concatenate(all_monopole_charges, axis=0)
    reflected_dipole_positions = np.concatenate(all_dipole_positions, axis=0)
    reflected_dipole_moments = np.concatenate(all_dipole_moments, axis=0)

    return (
        reflected_monopole_positions,
        reflected_monopole_charges,
        reflected_dipole_positions,
        reflected_dipole_moments,
    )


def reflect_sources_vectorized(
    monopole_positions_cm,
    monopole_charges,
    dipole_positions_cm,
    dipole_moments,
    arena_size_cm,
    reflection_scale=1.0,
    flip_on_reflection=FLIP_ON_REFLECTION,
    num_reflections=1,  # compatibility with original function
):
    """
    Computes a single order of reflections for monopoles and dipoles using the method of images.
    This version is fully vectorized for efficiency.

    Args:
        monopole_positions: (N,2) array of monopole positions
        monopole_charges: (N,) array of monopole charges
        dipole_positions: (M,2) array of dipole positions
        dipole_moments: (M,2) array of dipole moments
        arena_size: (width, height) of arena
        reflection_scale: Scaling factor for the charge/moment of reflected images.
        flip_on_reflection: If True, flips the charge/moment of reflected images. (conducting walls)

    Returns:
        Reflected positions and charges/moments (NOT including original ones)
    """
    width, height = arena_size_cm

    # 1. Monopoles
    # Prepare positions for reflection. The original positions are repeated 4 times.
    reflected_monopole_positions = np.tile(monopole_positions_cm, (4, 1))

    # Create the reflection transformations
    # Wall normals: Left, Right, Bottom, Top
    # Reflection formula: p' = (2*wall_pos - p)
    # The transformations for each wall can be stacked.
    # Note: The original positions are already included.
    # We apply the transformation to the tiled positions.
    # x-reflections (left and right)
    reflected_monopole_positions[: 2 * len(monopole_positions_cm), 0] = (
        2 * np.array([0, width]).repeat(len(monopole_positions_cm))
        - reflected_monopole_positions[: 2 * len(monopole_positions_cm), 0]
    )

    # y-reflections (bottom and top)
    reflected_monopole_positions[2 * len(monopole_positions_cm) :, 1] = (
        2 * np.array([0, height]).repeat(len(monopole_positions_cm))
        - reflected_monopole_positions[2 * len(monopole_positions_cm) :, 1]
    )

    # Prepare charges
    monopole_charge_scale = -1.0 if flip_on_reflection else 1.0
    reflected_monopole_charges = (
        np.tile(monopole_charges, 4) * monopole_charge_scale * reflection_scale
    )

    # 2. Dipoles
    # Prepare positions for reflection
    reflected_dipole_positions = np.tile(dipole_positions_cm, (4, 1))

    reflected_dipole_positions[: 2 * len(dipole_positions_cm), 0] = (
        2 * np.array([0, width]).repeat(len(dipole_positions_cm))
        - reflected_dipole_positions[: 2 * len(dipole_positions_cm), 0]
    )

    reflected_dipole_positions[2 * len(dipole_positions_cm) :, 1] = (
        2 * np.array([0, height]).repeat(len(dipole_positions_cm))
        - reflected_dipole_positions[2 * len(dipole_positions_cm) :, 1]
    )

    # Prepare moments for reflection
    reflected_dipole_moments = np.tile(dipole_moments, (4, 1))

    # Flip the component normal to the wall
    if flip_on_reflection:
        # x-component for left/right walls
        reflected_dipole_moments[: 2 * len(dipole_moments), 0] *= -1
        # y-component for bottom/top walls
        reflected_dipole_moments[2 * len(dipole_moments) :, 1] *= -1

    reflected_dipole_moments *= reflection_scale

    return (
        reflected_monopole_positions,
        reflected_monopole_charges,
        reflected_dipole_positions,
        reflected_dipole_moments,
    )


def reflect_sources(**kwargs):
    """
    This function is a wrapper to call the vectorized version.
    """
    # return reflect_sources_iterative(**kwargs) # vectorized only faster when n_poles > 10000
    return reflect_sources_vectorized(**kwargs)  # probably faster when multiple envs


def fish_forward(
    # fish position and orientation in the tank
    fish_positions_cm,  # [f, 2] Fish positions in cm
    fish_orientations_rad,  # [f,] Fish orientations (radians).
    # the charge objects for each fish (in fish local coordinates)
    fish_monopole_positions_cm,  # [f, p, 2] or [1, p, 2] Monopole positions (egocentric) in cm
    fish_monopole_charges,  # [f, p] or [1, p] Monopole charges.
    fish_dipole_positions_cm,  # [f, d, 2] or [1, d, 2] Dipole positions (egocentric) in cm.
    fish_dipole_moments,  # [f, d, 2] or [1, d, 2] Dipole moments.
    # whether the fish is currently pulsing
    fish_eods,  # [f,] Fish EOD states (1=emitting, 0=not emitting).
    # the fish sensors (in fish local coordinates)
    fish_sensor_positions_cm,  # [f, s, 2] or [1, s, 2] Sensor positions (egocentric) in cm.
    fish_sensor_normals,  # [f, s, 2] or [1, s, 2] Sensor normals.
    fish_sensor_type_mask,  # [s,] or None Sensor type mask (0=mormyromast, 1=ampullary). if None, include all sensors (treat all as same)
    # conductors (typically food)
    conductor_positions_cm,  # [c, 2] Conductor positions in cm.
    conductor_contrasts,  # [c,] or [1,] Conductor contrasts.
    conductor_radii_cm,  # [c,] or [1,] Conductor radii in cm.
    # arena
    arena_size_cm,  # (width, height) Arena dimensions in (cm, cm).
    arena_clipping_distance_cm,  # Distance for arena reflection clipping in cm.
    # misc
    measurement_noise=0.0,  # Measurement noise standard deviation.
    eps_m=1e-5,  # Small value for singularity prevention.
    do_induced=True,  # Include induced charges (True/False).
    do_reflection=True,  # Include arena reflections (True/False).
    do_intrinsics=False,  # Include intrinsic charges (True/False).
    conductor_intrinsic_dipole_moments=None,  # [c, 2] Conductor intrinsic dipole moments
    fish_contrasts=None,  # [f,] or [1,] Fish contrasts.
    fish_radii_cm=None,  # [f,] or [1,] Fish radii in cm.
    fish_intrinsic_dipole_moments=None,  # [f, 2] or [1, 2] Fish intrinsic dipole moments
    return_raw_field=False,  # Return raw field vectors from fish centers
    max_charge_allowed=None,  # Maximum charge allowed for monopoles (None for no limit).
    return_cons_baseline=False,  # Return baseline for cons-EOD without food/non-emitting fish distortions
    ampullary_intrinsic_only=False,  # If True, ampullary sensors sense ONLY intrinsic dipoles
):
    """Calculates electric fish sensor measurements."""
    # check shapes
    f = fish_positions_cm.shape[0]
    p = fish_monopole_positions_cm.shape[1]
    d = fish_dipole_positions_cm.shape[1]
    s = fish_sensor_positions_cm.shape[1]

    # Track intrinsic dipoles separately
    intrinsic_dipole_positions = None
    intrinsic_dipole_moments = None

    if do_induced:
        c = conductor_positions_cm.shape[0]
        require_shape(conductor_positions_cm, (c, 2), "conductor_positions_cm")
        require_shape_in(conductor_contrasts, [(c,), (1,)], "conductor_contrasts")
        require_shape_in(conductor_radii_cm,  [(c,), (1,)], "conductor_radii_cm")

    if do_intrinsics:
        if conductor_intrinsic_dipole_moments is not None:
            c = conductor_positions_cm.shape[0]
            require_shape(conductor_intrinsic_dipole_moments, (c, 2), "conductor_intrinsic_dipole_moments")
        if fish_intrinsic_dipole_moments is not None:
            require_shape(fish_intrinsic_dipole_moments, (f, 2), "fish_intrinsic_dipole_moments")

    # Pre-work: map the fish sensors into world coordinates
    (
        fish_sensor_global_positions,
        fish_sensor_global_normals,
    ) = to_world_sensors(
        fish_positions_cm,
        fish_orientations_rad,
        fish_sensor_positions_cm,
        fish_sensor_normals,
    )

    # Actual calculations begin
    # Get scene fields for active/EODs
    (
        scene_global_monopole_positions,
        scene_monopole_charges,
        scene_global_dipole_positions,
        scene_global_dipole_moments,
    ) = to_world_sources(
        fish_positions_cm,
        fish_orientations_rad,
        fish_monopole_positions_cm,
        fish_monopole_charges,
        fish_dipole_positions_cm,
        fish_dipole_moments,
        fish_eods,
    )

    # reflect the field generators off the walls
    if do_reflection:
        (
            scene_global_monopole_positions,
            scene_monopole_charges,
            scene_global_dipole_positions,
            scene_global_dipole_moments,
        ) = reflect_sources(
            monopole_positions_cm=scene_global_monopole_positions,
            monopole_charges=scene_monopole_charges,
            dipole_positions_cm=scene_global_dipole_positions,
            dipole_moments=scene_global_dipole_moments,
            arena_size_cm=arena_size_cm,
            num_reflections=1,
        )

    # Get induced charges on food
    # Save a copy of below for return_cons_baseline (no food, and ideally no emitting fish either)
    if return_cons_baseline:
        baseline_dipole_positions = scene_global_dipole_positions.copy()
        baseline_dipole_moments = scene_global_dipole_moments.copy()

        if (
            do_induced
            and fish_contrasts is not None
            and fish_radii_cm is not None
            and fish_contrasts.sum() != 0  # when self-induction disabled for some
        ):
            fish_moments = induce_dipoles(
                scene_global_monopole_positions,
                scene_monopole_charges,
                baseline_dipole_positions,
                baseline_dipole_moments,
                fish_positions_cm,
                fish_contrasts,
                fish_radii_cm,
                measurement_noise=0.0,
                eps_m=eps_m,
            )
            fish_moments = clip_conductor_moments(
                fish_moments,
                fish_radii_cm,
                max_charge_allowed=max_charge_allowed,
            )
            baseline_dipole_positions = np.concatenate(
                (baseline_dipole_positions, fish_positions_cm), axis=0
            )
            baseline_dipole_moments = np.concatenate(
                (baseline_dipole_moments, fish_moments), axis=0
            )

        # Reflect induced fish dipoles (no food)
        if do_reflection:
            _, _, baseline_dipole_positions, baseline_dipole_moments = reflect_sources(
                monopole_positions_cm=np.zeros((0, 2)),
                monopole_charges=np.array([]),
                dipole_positions_cm=baseline_dipole_positions,
                dipole_moments=baseline_dipole_moments,
                arena_size_cm=arena_size_cm,
                num_reflections=1,
            )

        # Measure the electric field for the baseline scenario
        baseline_gradients = measure_electric_field(
            fish_sensor_global_positions,
            scene_global_monopole_positions,
            scene_monopole_charges,
            baseline_dipole_positions,
            baseline_dipole_moments,
            measurement_noise,
            eps_m,
        )

        baseline_sensor_global_normals = fish_sensor_global_normals
        baseline_readings = project_field(
            baseline_gradients,
            baseline_sensor_global_normals,
        ).reshape(f, s)
        # print("baseline_readings.shape", baseline_readings.shape)

    # create the conductor induced dipoles
    if do_induced:
        food_moments = induce_dipoles(
            scene_global_monopole_positions,
            scene_monopole_charges,
            scene_global_dipole_positions,
            scene_global_dipole_moments,
            conductor_positions_cm,
            conductor_contrasts,
            conductor_radii_cm,
            measurement_noise=0.0,
            eps_m=eps_m,
        )

        # Doing this later because don't want to use induced fish dipoles in prey induction
        if fish_contrasts is not None and fish_radii_cm is not None:
            # Could be done in the same pass as the conductors below
            # but included here for clarity
            fish_moments = induce_dipoles(
                scene_global_monopole_positions,
                scene_monopole_charges,
                scene_global_dipole_positions,
                scene_global_dipole_moments,
                fish_positions_cm,
                fish_contrasts,
                fish_radii_cm,
                measurement_noise=0.0,
                eps_m=eps_m,
            )
            fish_moments = clip_conductor_moments(
                fish_moments,
                fish_radii_cm,
                max_charge_allowed=max_charge_allowed,
            )

            # Append the fish induced-dipole positions & moments
            scene_global_dipole_positions = np.concatenate(
                (scene_global_dipole_positions, fish_positions_cm),
                axis=0,
            )
            scene_global_dipole_moments = np.concatenate(
                (scene_global_dipole_moments, fish_moments),
                axis=0,
            )

        # add the induced dipole positions and moments to the list of scene dipoles
        scene_global_dipole_positions = np.concatenate(
            (
                scene_global_dipole_positions,
                conductor_positions_cm,
            ),
            axis=0,
        )
        # print("scene_monopole_charges", scene_monopole_charges)
        # print("scene_global_dipole_moments", scene_global_dipole_moments)
        # print("conductor_moments", conductor_moments)
        scene_global_dipole_moments = np.concatenate(
            (
                scene_global_dipole_moments,
                food_moments,
            ),
            axis=0,
        )

    # reflect the food + fish induced dipoles off the walls
    if do_reflection:
        (_, _, scene_global_dipole_positions, scene_global_dipole_moments) = (
            reflect_sources(
                monopole_positions_cm=np.zeros((0, 2)),
                monopole_charges=np.array([]),  # No monopoles to reflect
                dipole_positions_cm=scene_global_dipole_positions,
                dipole_moments=scene_global_dipole_moments,
                arena_size_cm=arena_size_cm,
                num_reflections=1,
            )
        )


    fish_eod_monopole_positions = scene_global_monopole_positions
    fish_eod_monopole_charges = scene_monopole_charges
    if do_intrinsics:  # Only applies to Ampullary (not M and K)
        # Build list of intrinsic-only dipoles to route to ampullary if flag is set
        intrinsic_dipole_positions = []
        intrinsic_dipole_moments = []

        # Food/prey aka conductor dipoles
        if conductor_intrinsic_dipole_moments is not None:  # Check if provided
            intrinsic_dipole_positions.append(conductor_positions_cm)
            intrinsic_dipole_moments.append(conductor_intrinsic_dipole_moments)

        # Fish intrinsic dipoles
        if fish_intrinsic_dipole_moments is not None:
            intrinsic_dipole_positions.append(fish_positions_cm)
            intrinsic_dipole_moments.append(fish_intrinsic_dipole_moments)

        if len(intrinsic_dipole_positions) > 0:
            intrinsic_dipole_positions = np.concatenate(
                intrinsic_dipole_positions, axis=0
            )
            intrinsic_dipole_moments = np.concatenate(
                intrinsic_dipole_moments, axis=0
            )

            # Reflect intrinsic dipoles as well
            if do_reflection:
                (
                    _,
                    _,
                    intrinsic_dipole_positions,
                    intrinsic_dipole_moments,
                ) = reflect_sources(
                    monopole_positions_cm=np.zeros((0, 2)),
                    monopole_charges=np.array([]),
                    dipole_positions_cm=intrinsic_dipole_positions,
                    dipole_moments=intrinsic_dipole_moments,
                    arena_size_cm=arena_size_cm,
                    num_reflections=1,
                )

            # Append intrinsics to the scene so other sensors can still see them
            scene_global_dipole_positions = np.concatenate(
                (scene_global_dipole_positions, intrinsic_dipole_positions), axis=0
            )
            scene_global_dipole_moments = np.concatenate(
                (scene_global_dipole_moments, intrinsic_dipole_moments), axis=0
            )

        else:  # no intrinsics in scene
            intrinsic_dipole_positions = None
            intrinsic_dipole_moments = None


    if fish_sensor_type_mask is None:
        # measure the electric field at the sensor locations
        # sense everything without filtering
        fish_sensor_and_fish_center_positions = np.concatenate(
            (fish_sensor_global_positions, fish_positions_cm), axis=0
        )

        fish_sensor_and_fish_center_gradients = measure_electric_field(
            # fish_sensor_gradients = measure_electric_field(
            # fish_sensor_global_positions,
            fish_sensor_and_fish_center_positions,
            scene_global_monopole_positions,
            scene_monopole_charges,
            scene_global_dipole_positions,
            scene_global_dipole_moments,
            measurement_noise,
            eps_m,
        )
        fish_sensor_gradients = fish_sensor_and_fish_center_gradients[
            : fish_sensor_global_positions.shape[0]
        ]
        fish_centers_gradients = fish_sensor_and_fish_center_gradients[
            fish_sensor_global_positions.shape[0] :
        ]

    else:
        # Split sensing by sensor type: ampullary vs mormyromast
        all_agents_sensor_mask = np.tile(fish_sensor_type_mask, f)
        mormyromast_sensor_positions = fish_sensor_global_positions[
            all_agents_sensor_mask == 0
        ]
        ampullary_sensor_positions = fish_sensor_global_positions[
            all_agents_sensor_mask == 1
        ]
        # Also measure field at the fish centers
        mormyromast_and_fish_center_positions = np.concatenate(
            (mormyromast_sensor_positions, fish_positions_cm), axis=0
        )
        ampullary_and_fish_center_positions = np.concatenate(
            (ampullary_sensor_positions, fish_positions_cm), axis=0
        )

        # measure electric field of mormyromasts
        # mormyromasts sense fish monopoles and induced dipoles
        # but not intrinsic conductor charges

        # mormyromast_sensor_gradients = measure_electric_field(
        mormyromast_sensor_and_fish_center_gradients = measure_electric_field(
            # mormyromast_sensor_positions,
            mormyromast_and_fish_center_positions,
            fish_eod_monopole_positions,
            fish_eod_monopole_charges,
            scene_global_dipole_positions,
            scene_global_dipole_moments,
            measurement_noise,
            eps_m,
        )
        mormyromast_sensor_gradients = mormyromast_sensor_and_fish_center_gradients[
            : mormyromast_sensor_positions.shape[0]
        ]

        # measure electric field for ampullary
        # if ampullary_intrinsic_only: use ONLY intrinsic dipoles
        if ampullary_intrinsic_only:
            amp_monopole_positions = np.zeros((0, 2))
            amp_monopole_charges = np.array([])
            amp_dipole_positions = intrinsic_dipole_positions
            amp_dipole_moments = intrinsic_dipole_moments
        else:
            # ampullary senses everything
            amp_monopole_positions = scene_global_monopole_positions
            amp_monopole_charges = scene_monopole_charges
            amp_dipole_positions = scene_global_dipole_positions
            amp_dipole_moments = scene_global_dipole_moments

        ampullary_sensor_and_fish_center_gradients = measure_electric_field(
            ampullary_and_fish_center_positions,
            amp_monopole_positions,
            amp_monopole_charges,
            amp_dipole_positions,
            amp_dipole_moments,
            measurement_noise,
            eps_m,
        )  # [num-ampullary-positions, 2]
        ampullary_sensor_gradients = ampullary_sensor_and_fish_center_gradients[
            : ampullary_sensor_positions.shape[0]
        ]

        # Collect the sensor gradients
        fish_sensor_gradients = np.zeros_like(fish_sensor_global_positions)
        fish_sensor_gradients[all_agents_sensor_mask == 0] = (
            mormyromast_sensor_gradients
        )
        fish_sensor_gradients[all_agents_sensor_mask == 1] = ampullary_sensor_gradients

        # Collect the fish centers gradients
        fish_centers_mormyromast_gradients = (
            mormyromast_sensor_and_fish_center_gradients[
                mormyromast_sensor_positions.shape[0] :
            ]
        )  # [num-fish, 2]
        fish_centers_ampullary_gradients = ampullary_sensor_and_fish_center_gradients[
            ampullary_sensor_positions.shape[0] :
        ]  # [num-fish, 2]

    # #knollen mask
    # knollen_sensor_positions = fish_sensor_global_positions[all_agents_sensor_mask == 1]
    # knollen_sensor_gradients = measure_electric_field( # knollen sensors only sense monopoles
    #     knollen_sensor_positions,
    #     monopole_positions=fish_eod_monopole_positions,
    #     monopole_charges=fish_eod_monopole_charges,
    #     dipole_positions=None,
    #     dipole_moments=None,
    #     measurement_noise=measurement_noise,
    #     eps=eps,
    #     units=units,
    # ) # [num-knollen-positions, 2]

    # other_sensor_positions = fish_sensor_global_positions[all_agents_sensor_mask == 0]

    # other_sensor_gradients = measure_electric_field(
    #     other_sensor_positions,
    #     scene_global_monopole_positions,
    #     scene_monopole_charges,
    #     scene_global_dipole_positions,
    #     scene_global_dipole_moments,
    #     measurement_noise,
    #     eps,
    #     units,
    # ) # [num-other-positions, 2]

    # fish_sensor_gradients = np.zeros_like(fish_sensor_global_positions)
    # fish_sensor_gradients[all_agents_sensor_mask == 1] = knollen_sensor_gradients
    # fish_sensor_gradients[all_agents_sensor_mask == 0] = other_sensor_gradients

    # Project the gradients onto the sensor normals
    # apply the sensor directionality and thresholding
    fish_sensor_measurements = project_field(
        fish_sensor_gradients,
        fish_sensor_global_normals,
    )

    # expand return for additional return_cons_baseline element
    returned = [fish_sensor_measurements.reshape(f, s)]
    if return_raw_field:
        # also return the raw field vectors from fish centers
        fish_center_gradients_dict = {}
        if fish_sensor_type_mask is None:
            fish_center_gradients_dict["all"] = fish_centers_gradients  # [num-fish, 2]
        else:
            fish_center_gradients_dict["mormyromast"] = (
                fish_centers_mormyromast_gradients  # [num-fish, 2]
            )
            fish_center_gradients_dict["ampullary"] = (
                fish_centers_ampullary_gradients  # [num-fish, 2]
            )
        # return fish_sensor_measurements.reshape(f, s), fish_center_gradients_dict
        returned.append(fish_center_gradients_dict)
    else:
        returned.append(None)

    if return_cons_baseline:
        returned.append(baseline_readings)
    else:
        returned.append(None)

    if len(returned) == 1:
        return returned[0]
    return tuple(returned)


def measure_electric_potential(
    measurement_positions_cm,  # [num-positions, x, y] in cm
    monopole_positions_cm=None,  # [num-monopoles, x, y] in cm
    monopole_charges=None,  # [num-monopoles] in Coulombs
    dipole_positions_cm=None,  # [num-dipoles, x, y] in cm
    dipole_moments=None,  # [num-dipoles, x, y]
    measurement_noise=0.0,  # Volts
    eps_m=1e-5,  # m
):
    """
    Compute the electric potential at specified positions due to monopoles and dipoles.
    """
    require_xy(measurement_positions_cm, "measurement_positions_cm")
    if monopole_positions_cm is not None:
        require_xy(monopole_positions_cm, "monopole_positions_cm")
    if dipole_positions_cm is not None:
        require_xy(dipole_positions_cm, "dipole_positions_cm")
        require_xy(dipole_moments, "dipole_moments")


    # Convert positions to meters
    measurement_positions = measurement_positions_cm * CM_TO_M

    # Initialize potential
    m = measurement_positions.shape[0]
    electric_potential = np.zeros(m)

    # Contribution from monopoles
    if monopole_positions_cm is not None:
        p = monopole_positions_cm.shape[0]
        monopole_positions = monopole_positions_cm * CM_TO_M
        monopole_offsets = (
            measurement_positions[:, None, :] - monopole_positions[None, :, :]
        )  # [m, p, 2]
        monopole_distances = np.linalg.norm(monopole_offsets, axis=-1) + eps_m  # [m, p]
        electric_potential += np.sum(
            k_coulomb * monopole_charges[None, :] / monopole_distances, axis=1
        )  # [m]

    # Contribution from dipoles
    if dipole_positions_cm is not None:
        d = dipole_positions_cm.shape[0]
        dipole_positions = dipole_positions_cm * CM_TO_M
        dipole_offsets = (
            measurement_positions[:, None, :] - dipole_positions[None, :, :]
        )  # [m, d, 2]
        dipole_distances = (
            np.linalg.norm(dipole_offsets, axis=-1)[:, :, None] + eps_m
        )  # [m, d, 1]
        dipole_unit_offsets = dipole_offsets / dipole_distances  # [m, d, 2]
        dipole_potential = np.sum(
            k_coulomb
            * np.sum(dipole_moments[None, :, :] * dipole_unit_offsets, axis=-1)
            / dipole_distances[:, :, 0] ** 2,
            axis=1,
        )  # [m]
        electric_potential += dipole_potential

    # Add measurement noise
    electric_potential += np.random.normal(scale=measurement_noise, size=m)

    return electric_potential

## --------------------------------------- TESTS --------------------------------------- ##

def check_reflections_original_and_vectorized():
    monopole_positions = np.array([[2.5, 3.5], [8.0, 1.0]])
    monopole_charges = np.array([1.0, -0.5])
    dipole_positions = np.array([[4.0, 6.0], [1.0, 9.0]])
    dipole_moments = np.array([[0.2, -0.3], [0.5, 0.1]])
    arena_size_cm = (10, 10)
    reflection_scale = 1.0

    for flip_on_reflection in [True, False]:
        print(f"Testing with flip_on_reflection={flip_on_reflection}...")

        # Run the non-vectorized function
        result_original = reflect_sources(
            monopole_positions_cm=monopole_positions,
            monopole_charges=monopole_charges,
            dipole_positions_cm=dipole_positions,
            dipole_moments=dipole_moments,
            arena_size_cm=arena_size_cm,
            num_reflections=1,
            reflection_scale=reflection_scale,
            flip_on_reflection=flip_on_reflection,
        )

        # Run the vectorized function
        result_vectorized = reflect_sources_vectorized(
            monopole_positions_cm=monopole_positions,
            monopole_charges=monopole_charges,
            dipole_positions_cm=dipole_positions,
            dipole_moments=dipole_moments,
            arena_size_cm=arena_size_cm,
            reflection_scale=reflection_scale,
            flip_on_reflection=flip_on_reflection,
        )

        # --- Corrected comparison logic ---

        # Unpack the tuples for easier comparison
        (
            orig_monopole_pos,
            orig_monopole_charges,
            orig_dipole_pos,
            orig_dipole_moments,
        ) = result_original
        (
            vec_monopole_pos,
            vec_monopole_charges,
            vec_dipole_pos,
            vec_dipole_moments,
        ) = result_vectorized

        # The order of reflected images might be different, so we sort them
        # to ensure a valid comparison.
        def sort_results(positions, charges_or_moments):
            if positions.size == 0:
                return positions, charges_or_moments

            # Use lexsort to sort by x and then by y
            indices = np.lexsort((positions[:, 1], positions[:, 0]))
            return positions[indices], charges_or_moments[indices]

        # Sort and compare monopoles
        sorted_orig_monopole_pos, sorted_orig_monopole_charges = sort_results(
            orig_monopole_pos, orig_monopole_charges
        )
        sorted_vec_monopole_pos, sorted_vec_monopole_charges = sort_results(
            vec_monopole_pos, vec_monopole_charges
        )

        assert_allclose(sorted_orig_monopole_pos, sorted_vec_monopole_pos, atol=1e-9)
        assert_allclose(
            sorted_orig_monopole_charges, sorted_vec_monopole_charges, atol=1e-9
        )

        # Sort and compare dipoles
        sorted_orig_dipole_pos, sorted_orig_dipole_moments = sort_results(
            orig_dipole_pos, orig_dipole_moments
        )
        sorted_vec_dipole_pos, sorted_vec_dipole_moments = sort_results(
            vec_dipole_pos, vec_dipole_moments
        )

        assert_allclose(sorted_orig_dipole_pos, sorted_vec_dipole_pos, atol=1e-9)
        assert_allclose(
            sorted_orig_dipole_moments, sorted_vec_dipole_moments, atol=1e-9
        )

    print("All tests passed!")


def run_reflection_benchmark_simple(num_sources=100, num_runs=1000):
    print(
        f"Running simple benchmark with {num_sources} monopoles and {num_sources} dipoles, over {num_runs} runs."
    )

    arena_size_cm = (100, 100)
    width, height = arena_size_cm
    monopole_positions = np.random.uniform(
        low=0, high=[width, height], size=(num_sources, 2)
    )
    monopole_charges = np.random.uniform(low=-1, high=1, size=(num_sources,))
    dipole_positions = np.random.uniform(
        low=0, high=[width, height], size=(num_sources, 2)
    )
    dipole_moments = np.random.uniform(low=-1, high=1, size=(num_sources, 2))

    def call_original():
        reflect_sources_iterative(
            monopole_positions_cm=monopole_positions,
            monopole_charges=monopole_charges,
            dipole_positions_cm=dipole_positions,
            dipole_moments=dipole_moments,
            arena_size_cm=arena_size_cm,
            num_reflections=1,
            flip_on_reflection=True,
        )

    def call_vectorized():
        reflect_sources_vectorized(
            monopole_positions_cm=monopole_positions,
            monopole_charges=monopole_charges,
            dipole_positions_cm=dipole_positions,
            dipole_moments=dipole_moments,
            arena_size_cm=arena_size_cm,
            flip_on_reflection=True,
        )

    time_original = timeit.timeit(call_original, number=num_runs)
    time_vectorized = timeit.timeit(call_vectorized, number=num_runs)
    speed_up = time_original / time_vectorized
    print(f"Original function total time: {time_original:.4f} seconds")
    print(f"Vectorized function total time: {time_vectorized:.4f} seconds")
    print(f"Average speedup: Vectorized is {speed_up:.2f}x faster.")

def _expected_single_reflection(monopole_pos, monopole_q, dipole_pos, dipole_m, arena_size_cm,
                                reflection_scale=1.0, flip_on_reflection=False):
    """
    Build expected outputs for reflect_sources for ONE reflection order.
    This matches the vectorized implementation: returns ONLY reflected images (no originals),
    in the order: left, right, bottom, top.
    """
    width, height = arena_size_cm

    # --- monopoles ---
    # left: x -> -x
    left_mpos = monopole_pos.copy()
    left_mpos[:, 0] = -left_mpos[:, 0]
    # right: x -> 2W - x
    right_mpos = monopole_pos.copy()
    right_mpos[:, 0] = 2.0 * width - right_mpos[:, 0]
    # bottom: y -> -y
    bottom_mpos = monopole_pos.copy()
    bottom_mpos[:, 1] = -bottom_mpos[:, 1]
    # top: y -> 2H - y
    top_mpos = monopole_pos.copy()
    top_mpos[:, 1] = 2.0 * height - top_mpos[:, 1]

    exp_mpos = np.concatenate([left_mpos, right_mpos, bottom_mpos, top_mpos], axis=0)

    q_scale = (-1.0 if flip_on_reflection else 1.0) * reflection_scale
    exp_mq = np.tile(monopole_q, 4) * q_scale

    # --- dipoles ---
    left_dpos = dipole_pos.copy()
    left_dpos[:, 0] = -left_dpos[:, 0]
    right_dpos = dipole_pos.copy()
    right_dpos[:, 0] = 2.0 * width - right_dpos[:, 0]
    bottom_dpos = dipole_pos.copy()
    bottom_dpos[:, 1] = -bottom_dpos[:, 1]
    top_dpos = dipole_pos.copy()
    top_dpos[:, 1] = 2.0 * height - top_dpos[:, 1]

    exp_dpos = np.concatenate([left_dpos, right_dpos, bottom_dpos, top_dpos], axis=0)

    exp_dm = np.concatenate(
        [dipole_m.copy(), dipole_m.copy(), dipole_m.copy(), dipole_m.copy()],
        axis=0
    )

    if flip_on_reflection:
        # left/right flip x-component (normal to vertical walls)
        n = dipole_m.shape[0]
        exp_dm[:2 * n, 0] *= -1.0
        # bottom/top flip y-component (normal to horizontal walls)
        exp_dm[2 * n:, 1] *= -1.0

    exp_dm *= reflection_scale

    return exp_mpos, exp_mq, exp_dpos, exp_dm


def test_reflection_matches_closed_form_flip_false():
    monopole_pos = np.array([[2.5, 3.5], [8.0, 1.0]], dtype=np.float64)
    monopole_q = np.array([1.0, -0.5], dtype=np.float64)
    dipole_pos = np.array([[4.0, 6.0], [1.0, 9.0]], dtype=np.float64)
    dipole_m = np.array([[0.2, -0.3], [0.5, 0.1]], dtype=np.float64)

    arena_size_cm = (10.0, 10.0)
    reflection_scale = 0.37
    flip_on_reflection = False

    got = reflect_sources(
        monopole_positions_cm=monopole_pos,
        monopole_charges=monopole_q,
        dipole_positions_cm=dipole_pos,
        dipole_moments=dipole_m,
        arena_size_cm=arena_size_cm,
        reflection_scale=reflection_scale,
        flip_on_reflection=flip_on_reflection,
        num_reflections=1,
    )
    exp = _expected_single_reflection(
        monopole_pos, monopole_q, dipole_pos, dipole_m,
        arena_size_cm,
        reflection_scale=reflection_scale,
        flip_on_reflection=flip_on_reflection,
    )

    for g, e in zip(got, exp):
        assert_allclose(g, e, rtol=0.0, atol=0.0)


def test_reflection_matches_closed_form_flip_true():
    monopole_pos = np.array([[2.5, 3.5], [8.0, 1.0]], dtype=np.float64)
    monopole_q = np.array([1.0, -0.5], dtype=np.float64)
    dipole_pos = np.array([[4.0, 6.0], [1.0, 9.0]], dtype=np.float64)
    dipole_m = np.array([[0.2, -0.3], [0.5, 0.1]], dtype=np.float64)

    arena_size_cm = (10.0, 10.0)
    reflection_scale = 1.0
    flip_on_reflection = True

    got = reflect_sources(
        monopole_positions_cm=monopole_pos,
        monopole_charges=monopole_q,
        dipole_positions_cm=dipole_pos,
        dipole_moments=dipole_m,
        arena_size_cm=arena_size_cm,
        reflection_scale=reflection_scale,
        flip_on_reflection=flip_on_reflection,
        num_reflections=1,
    )
    exp = _expected_single_reflection(
        monopole_pos, monopole_q, dipole_pos, dipole_m,
        arena_size_cm,
        reflection_scale=reflection_scale,
        flip_on_reflection=flip_on_reflection,
    )

    for g, e in zip(got, exp):
        assert_allclose(g, e, rtol=0.0, atol=0.0)


def test_reflection_shapes_and_counts():
    rng = np.random.RandomState(0)
    n_m = 7
    n_d = 11
    monopole_pos = rng.uniform(low=-5.0, high=15.0, size=(n_m, 2)).astype(np.float64)
    monopole_q = rng.uniform(low=-1.0, high=1.0, size=(n_m,)).astype(np.float64)
    dipole_pos = rng.uniform(low=-5.0, high=15.0, size=(n_d, 2)).astype(np.float64)
    dipole_m = rng.uniform(low=-1.0, high=1.0, size=(n_d, 2)).astype(np.float64)

    arena_size_cm = (10.0, 10.0)

    mpos_r, mq_r, dpos_r, dm_r = reflect_sources(
        monopole_positions_cm=monopole_pos,
        monopole_charges=monopole_q,
        dipole_positions_cm=dipole_pos,
        dipole_moments=dipole_m,
        arena_size_cm=arena_size_cm,
        reflection_scale=1.0,
        flip_on_reflection=False,
        num_reflections=1,
    )

    assert mpos_r.shape == (4 * n_m, 2)
    assert mq_r.shape == (4 * n_m,)
    assert dpos_r.shape == (4 * n_d, 2)
    assert dm_r.shape == (4 * n_d, 2)


if __name__ == "__main__":
    test_reflection_shapes_and_counts()
    test_reflection_matches_closed_form_flip_false()
    test_reflection_matches_closed_form_flip_true()

    # check_reflections_original_and_vectorized()
    # NOTE: vectorized slower for small num_sources, but might parallelize better
    print("Running reflection benchmark...")
    run_reflection_benchmark_simple(num_sources=500, num_runs=100)

    print("\nFor electric field measurement backend tests, run electric_accelerated.py")
