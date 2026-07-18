import numpy as np
from cfg import AGENT_PARAMS


def get_obs_partitions(metadata: dict) -> dict:
    """
    Build obs-vector index partitions from saved run metadata.

    Takes the raw metadata dict (as stored in eval pkls), not a DataFrame.
    Raises KeyError if required sensor-count keys are absent — no silent
    fallback to cfg, so misconfigured runs fail loudly.

    Parameters
    ----------
    metadata : dict
        Dict with 'all_args' and 'agent_args' sub-dicts, as stored in the
        eval pkl metadata column.

    Returns
    -------
    dict with np.ndarray index arrays for each obs component:
        mormyromast, mormyromast_self, mormyromast_cons
        ampullary
        knollen, knollen_metadata
        actions          (only if feedback_action)
        fatigue
        bitten           (only if enable_bite_action)
        agent_size       (only if agent_size_mode is set)
        bite_cooldown    (only if enable_bite_action and use_bite_cooldown)
        feedback_displacement  (only if feedback_displacement)
        obs_dim          — int total obs length (for validation)
    """
    all_args   = metadata['all_args']
    agent_args = metadata['agent_args']

    num_morm_virtual = agent_args['num_mormyromast_sensors_virtual']
    num_morm_real    = agent_args['num_mormyromast_sensors_real']
    num_amp          = agent_args['num_ampullary_sensors']
    num_kn           = agent_args['num_knollen_sensors']
    num_agents       = all_args.get('num_agents') or agent_args['num_agents']

    feedback_action       = all_args['feedback_action']
    enable_bite           = all_args.get('enable_bite_action', all_args.get('allow_aggression', False))
    use_bite_cooldown     = all_args.get('use_bite_cooldown', False)
    agent_size_mode       = all_args.get('agent_size_mode')
    feedback_displacement = all_args.get('feedback_displacement', False)
    num_actions           = 3 + int(enable_bite)

    idx = 0
    p = {}

    p['mormyromast']      = np.arange(idx, idx + num_morm_virtual)
    p['mormyromast_self'] = np.arange(idx, idx + num_morm_real)
    p['mormyromast_cons'] = np.arange(idx + num_morm_real, idx + num_morm_virtual)
    idx += num_morm_virtual

    p['ampullary'] = np.arange(idx, idx + num_amp)
    idx += num_amp

    num_kn_total = num_kn * (num_agents - 1)
    p['knollen'] = np.arange(idx, idx + num_kn_total)
    idx += num_kn_total

    p['knollen_metadata'] = np.arange(idx, idx + (num_agents - 1))
    idx += num_agents - 1

    if feedback_action:
        p['actions'] = np.arange(idx, idx + num_actions)
        idx += num_actions

    p['fatigue'] = np.arange(idx, idx + 1)
    idx += 1

    if enable_bite:
        p['bitten'] = np.arange(idx, idx + 1)
        idx += 1

    if agent_size_mode is not None:
        p['agent_size'] = np.arange(idx, idx + 1)
        idx += 1

    if enable_bite and use_bite_cooldown:
        p['bite_cooldown'] = np.arange(idx, idx + 1)
        idx += 1

    if feedback_displacement:
        p['feedback_displacement'] = np.arange(idx, idx + 2)
        idx += 2

    p['obs_dim'] = idx
    return p


def get_obs_partitions_from_df(dff) -> dict:
    """Convenience wrapper: extracts metadata from first row of dff, then calls get_obs_partitions."""
    return get_obs_partitions(dff['metadata'].iloc[0])


# DEPRECATED: reads from cfg instead of saved metadata — use get_obs_partitions instead.
def get_sensor_indices_from_cfg(
    num_mormyromast=None, num_ampullary=None, num_knollen=None, num_agents=4,
    model="fracrand",
):
    """
    Compute index ranges for mormyromast, ampullary, and knollen sensors.
    """
    if num_mormyromast is None:
        num_mormyromast = AGENT_PARAMS["num_rays"]
        if model == "fracrand":
            num_mormyromast = AGENT_PARAMS["num_morm_sets"] * AGENT_PARAMS["num_rays"]
    if num_ampullary is None:
        num_ampullary = AGENT_PARAMS["num_ampullary_sensors"]
    if num_knollen is None:
        num_knollen = AGENT_PARAMS["num_knollen_sensors"] * (num_agents-1)

    mormyromast_indices = np.arange(0, num_mormyromast).astype(int)
    ampullary_indices = (len(mormyromast_indices) + np.arange(0, num_ampullary)).astype(
        int
    )
    knollen_indices = (
        len(mormyromast_indices)
        + len(ampullary_indices)
        + np.arange(0, num_knollen).astype(int)
    )

    return mormyromast_indices, ampullary_indices, knollen_indices

def compute_sensor_boundaries(morm=None, amp=None, kn=None):
    bounds = []
    for arr in [morm, amp, kn]:
        if arr is not None and len(arr) > 0:
            bounds.append((min(arr), max(arr)))
    bounds = sorted(bounds, key=lambda x: x[0])

    # Return the max index of each group → boundary after that index
    cutpoints = [b[1] for b in bounds[:-1]]  # exclude last group end
    return cutpoints  # list of integer boundaries
