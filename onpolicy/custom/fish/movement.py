import numpy as np
import math

def reset_agent_movement_parameters_to_baselines(agent):
    agent.max_linear_velocity = agent.max_linear_velocity_baseline
    agent.min_linear_velocity = agent.min_linear_velocity_baseline
    agent.max_linear_acceleration = agent.max_linear_acceleration_baseline
    agent.max_angular_velocity = agent.max_angular_velocity_baseline
    agent.min_angular_velocity = agent.min_angular_velocity_baseline
    agent.max_angular_acceleration = agent.max_angular_acceleration_baseline

def apply_global_movement_multipliers(
    agent,
    linear_multiplier=1.0,
    angular_multiplier=1.0,
    linear_acc_multiplier=None,
    angular_acc_multiplier=None,
):
    agent.max_linear_velocity *= linear_multiplier
    agent.min_linear_velocity *= linear_multiplier
    agent.max_linear_acceleration *= linear_acc_multiplier if linear_acc_multiplier is not None else linear_multiplier
    agent.max_angular_velocity *= angular_multiplier
    agent.min_angular_velocity *= angular_multiplier
    agent.max_angular_acceleration *= angular_acc_multiplier if angular_acc_multiplier is not None else angular_multiplier

def apply_size_multipliers(agent, 
                           agent_size, 
                           size_speed_exponent=1.0,
                           ):
    # IMPORTANT:
    # Size-based multipliers should be applied ONCE per episode (typically at reset).
    # Do NOT call this every timestep or size effects will compound.
    size_multiplier = (1 + agent_size) ** size_speed_exponent
    agent.max_linear_velocity *= size_multiplier
    agent.min_linear_velocity *= size_multiplier
    agent.max_linear_acceleration *= size_multiplier
    agent.max_angular_velocity *= size_multiplier
    agent.min_angular_velocity *= size_multiplier
    agent.max_angular_acceleration *= size_multiplier

def _wrap_angle(theta):
    # stable wrap to [-pi, pi]
    return math.atan2(math.sin(theta), math.cos(theta))


def _propose_next_state_first_order(agent):
    """
    First-order: commands act like direct velocity commands.
    Returns a motion delta tuple:
        (proposed_position, proposed_orientation, proposed_linear_velocity, proposed_angular_velocity)
    Note: Does NOT mutate agent pose/velocities.
    """
    # Default commands
    move_cmd = agent.move_forward
    turn_cmd = agent.turn_angle

    # Eating freeze: treat as zero commands for this step (do not overwrite action vars)
    if getattr(agent, "eat_cooldown_counter", 0) > 0:
        move_cmd = 0.0
        turn_cmd = 0.0

    angular_velocity = turn_cmd * agent.max_angular_velocity
    proposed_orientation = _wrap_angle(agent.orientation + angular_velocity)

    linear_velocity = move_cmd * agent.max_linear_velocity
    heading = np.array([math.cos(proposed_orientation), math.sin(proposed_orientation)], dtype=float)
    proposed_position = agent.position + heading * linear_velocity

    return proposed_position, proposed_orientation, linear_velocity, angular_velocity


def _propose_next_state_second_order(agent):
    """
    Second-order: commands act like accelerations integrated into velocities (with drag).
    Returns a motion delta tuple:
        (proposed_position, proposed_orientation, proposed_linear_velocity, proposed_angular_velocity)
    Note: Does NOT mutate agent pose/velocities.
    If backwards == False, enforce non-negative forward speed here.
    First-order relies on actor-side squashing instead.
    """
    move_cmd = agent.move_forward
    turn_cmd = agent.turn_angle

    # Integrate acceleration
    linear_velocity = agent.linear_velocity + move_cmd * agent.max_linear_acceleration
    angular_velocity = agent.angular_velocity + turn_cmd * agent.max_angular_acceleration

    # Clamp velocities
    linear_velocity = np.clip(linear_velocity, agent.min_linear_velocity, agent.max_linear_velocity)
    if not getattr(agent, "backwards", True):
        linear_velocity = max(0.0, float(linear_velocity))

    # Drag
    linear_velocity *= agent.linear_drag_factor
    angular_velocity *= agent.angular_drag_factor

    # Clamp velocities
    angular_velocity = np.clip(angular_velocity, agent.min_angular_velocity, agent.max_angular_velocity)

    # Eating freeze overrides motion for this step (but we still decrement cooldown in commit step)
    if getattr(agent, "eat_cooldown_counter", 0) > 0:
        linear_velocity = 0.0
        angular_velocity = 0.0

    proposed_orientation = _wrap_angle(agent.orientation + angular_velocity)
    heading = np.array([math.cos(proposed_orientation), math.sin(proposed_orientation)], dtype=float)
    proposed_position = agent.position + heading * linear_velocity

    return proposed_position, proposed_orientation, float(linear_velocity), float(angular_velocity)


def _snapshot_pose(agent):
    last_position = agent.position.copy()
    last_orientation = agent.orientation
    agent.last_orientation = last_orientation
    return last_position, last_orientation

def _propose_next_state(agent):
    if agent.motion_order == 1:
        return _propose_next_state_first_order(agent)
    if agent.motion_order == 2:
        return _propose_next_state_second_order(agent)
    raise ValueError(f"Unknown motion_order: {agent.motion_order}")

def _check_collision_with_agents(proposed_position, agent, agent_objects):
    # NOTE: For simplicity collision check is order-dependent
    # We compare this agent's *proposed* position against other agents'
    # CURRENT positions (not their proposed ones).
    # This introduces mild update-order asymmetry but avoids centralized resolution.

    for other in agent_objects:
        if other is agent:
            continue
        if np.linalg.norm(proposed_position - other.position) < (agent.body_radius + other.body_radius):
            return True
    return False

def _commit_pose(agent, proposed_position, proposed_orientation, collided):
    # Orientation is always updated, even if position update is blocked by collision.
    # This allows agents to "turn in place" when blocked.
    agent.orientation = proposed_orientation
    if not collided:
        agent.position = proposed_position
    agent.collided = collided

def _enforce_arena_bounds(agent, arena):
    arena_w, arena_h = agent.arena_size

    # cache pre-clip for wall-hit detection
    x0 = float(agent.position[0])
    y0 = float(agent.position[1])

    x1 = float(np.clip(x0, agent.body_radius, arena_w - agent.body_radius))
    y1 = float(np.clip(y0, agent.body_radius, arena_h - agent.body_radius))

    hit_wall = (x1 != x0) or (y1 != y0)

    agent.position[0] = x1
    agent.position[1] = y1

    return hit_wall


def _commit_velocities_if_needed(agent, proposed_linear_velocity, proposed_angular_velocity, stop_motion):
    if agent.motion_order == 2:
        if stop_motion:
            # On collision or wall contact in second-order dynamics,
            # we zero stateful velocities to avoid integrating momentum through obstacles.
            agent.linear_velocity = 0.0
            agent.angular_velocity = 0.0
        else:
            agent.linear_velocity = proposed_linear_velocity
            agent.angular_velocity = proposed_angular_velocity

def _update_displacements(agent, last_position, last_orientation):
    displacement_ground = agent.position - last_position
    agent.displacement_ground = displacement_ground

    c = math.cos(-last_orientation)
    s = math.sin(-last_orientation)
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    agent.displacement_ego = R @ displacement_ground

def apply_movement_step(agent, arena, agent_objects):
    last_position, last_orientation = _snapshot_pose(agent)

    proposed_position, proposed_orientation, v_lin, v_ang = _propose_next_state(agent)

    collided = _check_collision_with_agents(proposed_position, agent, agent_objects)

    _commit_pose(agent, proposed_position, proposed_orientation, collided)

    hit_wall = _enforce_arena_bounds(agent, arena)

    stop_motion = collided or hit_wall

    _commit_velocities_if_needed(agent, v_lin, v_ang, stop_motion)

    _update_displacements(agent, last_position, last_orientation)
