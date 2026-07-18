import os
import sys
import numpy as np
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import matplotlib.patheffects as path_effects
import matplotlib.collections as mcollections
import imageio
import cfg
from cfg import m_to_cm, ENV_PARAMS, COLORS


class FishRenderer:
    def __init__(self, env):
        self.env = env
        self.fig = None
        self.axes = None
        self.ax1 = None
        self.aux_axes = None
        self.quiver_scale = None
        self.attentions = None
        self.fps_video = None
        self.video_writer = None
        self.video_writer_path = None

    # ------------------------------------------------------------------ #
    # Axis helpers
    # ------------------------------------------------------------------ #

    def _resize_with_buffer(self, buffer_area):
        self.ax1.set_xlim(0 - buffer_area, self.env.arena_size[0] + buffer_area)
        self.ax1.set_ylim(0 - buffer_area, self.env.arena_size[1] + buffer_area)

    def _format_time_based_ax(self, ax, window=50):
        if self.env.curr_time <= window:
            ax.set_xlim(0, window)
        else:
            ax.set_xlim(self.env.curr_time - window, self.env.curr_time)

    def _remove_child_polar_axes(self, ax):
        if hasattr(ax, "polar_axes"):
            for polar_ax in ax.polar_axes:
                polar_ax.remove()
            ax.polar_axes = []

    # ------------------------------------------------------------------ #
    # Aux subplot plotters
    # ------------------------------------------------------------------ #

    def _plot_energy_levels(self, ax):
        ax.clear()
        for agent_id, agent in enumerate(self.env.agent_objects):
            if agent_id not in self.env.active_agent_ids:
                continue
            energy_history = np.array(agent.energy_history)[-50:]
            time_steps = np.arange(max(0, self.env.curr_time - 49), self.env.curr_time + 1)
            ax.plot(
                time_steps,
                energy_history,
                label=f"Agent {agent_id}",
                color=f"C{agent_id}",
            )
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Energy Level")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title("Energy Levels")
        self._format_time_based_ax(ax)

    def _plot_eods_and_biting(self, ax, plot_bites=False):
        ax.clear()
        num_agents = len(self.env.agent_objects)
        for agent_id, agent in enumerate(self.env.agent_objects):
            if agent_id not in self.env.active_agent_ids:
                continue
            eod_history = agent.eod_history
            eod_data = np.array(eod_history[-50:])
            time_steps = np.arange(max(0, self.env.curr_time - 49), self.env.curr_time + 1)
            eod_idxs = np.where(eod_data >= 0.5)[0]
            eod_time_steps = time_steps[eod_idxs]
            ax.scatter(
                eod_time_steps,
                np.full(len(eod_time_steps), num_agents - 1 - agent_id),
                alpha=1,
                c=f"C{agent_id}",
                s=200,
                marker="|",
            )

            was_bitten_history = np.array(agent.was_bitten_history)[-50:]
            was_bitten_idxs = np.where(was_bitten_history)[0]
            was_bitten_time_steps = time_steps[was_bitten_idxs]
            ax.scatter(
                was_bitten_time_steps,
                np.full(len(was_bitten_time_steps), num_agents - 1 - agent_id),
                alpha=1,
                c=COLORS["agent_bitten"],
                s=20,
                marker="o",
            )
            bite_history = np.array(agent.bite_history)[-50:]
            bite_idxs = np.where(bite_history)[0]
            bite_time_steps = time_steps[bite_idxs]
            if plot_bites:
                ax.scatter(
                    bite_time_steps,
                    np.full(len(bite_time_steps), num_agents - 1 - agent_id),
                    alpha=0.5,
                    c=COLORS["agent_biting"],
                    s=5,
                    marker="o",
                )

        ax.set_yticks(range(num_agents))
        ax.set_yticklabels([f"Agent {i}" for i in reversed(range(num_agents))])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title("EODs")
        self._format_time_based_ax(ax)

    def _plot_spis(self, ax, plot_bites=False):
        ax.clear()
        num_agents = len(self.env.agent_objects)
        window = 200

        for agent_id, agent in enumerate(self.env.agent_objects):
            if agent_id not in self.env.active_agent_ids:
                continue

            eod_hist = np.array(agent.eod_history[-window:])
            time_steps = np.arange(
                max(0, self.env.curr_time - window + 1), self.env.curr_time + 1
            )
            fired = np.where(eod_hist >= 0.5)[0]
            times = time_steps[fired]

            if len(times) < 2:
                continue

            spis = np.diff(times) * 12
            spi_times = times[1:]

            ax.scatter(
                spi_times,
                spis,
                c=f"C{agent_id}",
                alpha=0.8,
                edgecolors="none",
                label=f"Agent {agent_id}",
            )
            ax.set_ylim(0, 200)

            if plot_bites:
                bite_hist = np.array(agent.bite_history[-window:])
                bite_idx = np.where(bite_hist)[0]
                bite_times = time_steps[bite_idx]
                ax.scatter(
                    bite_times,
                    np.zeros_like(bite_times),
                    c=COLORS["agent_biting"],
                    s=20,
                    marker="v",
                    alpha=0.6,
                )

        ax.set_xlabel("Time step")
        ax.set_ylabel("Inter-pulse interval (IPI) [ms]")
        ax.set_title("IPIs by agent")
        ax.legend(loc="upper right", ncol=2, fontsize="small")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._format_time_based_ax(ax, window=window)

    def _plot_observations(self, ax):
        max_radius = 1
        max_radius = max_radius * 1.25
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        self._remove_child_polar_axes(ax)

        num_agents = len(self.env.agent_objects)
        left, bottom, width, height = ax.get_position().bounds
        polar_width = (width / num_agents) * 0.9
        polar_height = height * 0.9

        for agent_id in range(num_agents):
            if agent_id not in self.env.active_agent_ids:
                continue
            polar_ax = self.fig.add_axes(
                [
                    left + agent_id * (width / num_agents) + (width / num_agents) * 0.05,
                    bottom + height * 0.05,
                    polar_width,
                    polar_height,
                ],
                polar=True,
            )
            ax.polar_axes.append(polar_ax)
            polar_ax.set_theta_offset(np.pi / 2)

            mormyromast_obs = np.array(
                self.env.agent_objects[agent_id].mormyromast_observations_virtual
            )[:self.env.sensing_model.agent_electrics.num_mormyromast_sensors_real]
            mormyromast_angles = self.env.sensing_model.agent_electrics.mormyromast_angles

            positive_mask_mor = mormyromast_obs >= 0
            negative_mask_mor = ~positive_mask_mor

            polar_ax.plot(
                mormyromast_angles[positive_mask_mor],
                mormyromast_obs[positive_mask_mor],
                label="mormyr +",
                color="blue",
            )
            polar_ax.plot(
                mormyromast_angles[negative_mask_mor],
                np.abs(mormyromast_obs[negative_mask_mor]),
                label="mormyr -",
                color="red",
            )

            num_agents = len(self.env.agent_objects)
            obs_offset = max_radius / 15
            knollen_angles = self.env.sensing_model.agent_electrics.knollen_angles
            num_knollen_sensors = self.env.sensing_model.agent_electrics.num_knollen_sensors_real
            knollen_arc_width = np.pi / num_knollen_sensors
            emitting_agent_ids = [i for i in range(self.env.num_agents) if i != agent_id]

            for index, emitting_agent_id in enumerate(emitting_agent_ids):
                emitting_agent_knollen_obs = np.array(
                    self.env.agent_objects[agent_id].knollen_observations
                )[index * num_knollen_sensors : (index + 1) * num_knollen_sensors]

                for i in range(num_knollen_sensors):
                    angle = (
                        knollen_angles[i]
                        - (knollen_arc_width / 2)
                        + (emitting_agent_id * knollen_arc_width / num_agents)
                    )
                    magnitude = emitting_agent_knollen_obs[i]

                    if abs(magnitude) < cfg.AGENT_PARAMS["knollen_sensor_min"]:
                        color = "black"
                    elif magnitude < 0:
                        color = "red"
                    else:
                        color = "blue"

                    start_point = max_radius - 2 * obs_offset
                    end_point = max_radius - obs_offset
                    polar_ax.plot(
                        [angle, angle],
                        [start_point, end_point],
                        color=color,
                        linewidth=1,
                    )

            ampullary_obs = np.array(
                self.env.agent_objects[agent_id].ampullary_observations
            )
            ampullary_angles = self.env.sensing_model.agent_electrics.ampullary_angles

            non_negative_mask_amp = ampullary_obs >= 0
            negative_mask_amp = ~non_negative_mask_amp
            positive_mask_amp = ampullary_obs > 0

            if np.any(positive_mask_amp):
                polar_ax.plot(
                    ampullary_angles[positive_mask_amp],
                    ampullary_obs[positive_mask_amp],
                    label="ampull +",
                    color="green",
                    linestyle="dotted",
                )
            if np.any(negative_mask_amp):
                polar_ax.plot(
                    ampullary_angles[negative_mask_amp],
                    np.abs(ampullary_obs[negative_mask_amp]),
                    label="ampull -",
                    color="orange",
                    linestyle="dotted",
                )

            polar_ax.set_ylim(0, max_radius)

            if agent_id != 0:
                polar_ax.set_xticklabels([])
                polar_ax.set_yticklabels([])

            if agent_id == len(self.env.agent_objects) - 1:
                legend_elements = [
                    Line2D([0], [0], color="blue", label="mormyr +"),
                    Line2D([0], [0], color="red", label="mormyr -"),
                    Line2D([0], [0], marker="o", color="blue", label="knoll +", linestyle=""),
                    Line2D([0], [0], marker="o", color="red", label="knoll -", linestyle=""),
                    Line2D([0], [0], color="green", label="ampull +"),
                    Line2D([0], [0], color="orange", label="ampull -"),
                ]
                polar_ax.legend(
                    handles=legend_elements,
                    fontsize=6,
                    loc="lower center",
                    ncol=2,
                    bbox_to_anchor=(0.5, -0.2),
                )
            else:
                polar_ax.legend().remove()

    # ------------------------------------------------------------------ #
    # Main arena plotters
    # ------------------------------------------------------------------ #

    def _plot_electric_field(self, ax):
        grid_size = 4.0
        x = np.arange(0, self.env.arena_size[0], grid_size)
        y = np.arange(0, self.env.arena_size[1], grid_size)
        X, Y = np.meshgrid(x, y)
        xy = np.stack((X.reshape(-1), Y.reshape(-1)), axis=-1)

        import utils_electric as ue
        measured_gradients = self.env.sensing_model.sense_field_at_positions(
            positions_cm=xy,
            use_reflections=self.env.sensing_model.sensing_params.do_reflection,
        )
        measured_gradients = ue.clip_gradients(measured_gradients, percentile=90)

        gradient_magnitudes = np.sqrt(np.sum(measured_gradients**2, axis=1))
        quiver_scale = np.percentile(gradient_magnitudes, 95)
        if self.quiver_scale is None or quiver_scale > self.quiver_scale:
            self.quiver_scale = quiver_scale

        mask = gradient_magnitudes > 0
        if not np.all(gradient_magnitudes == 0):
            ax.quiver(
                X.reshape(-1)[mask],
                Y.reshape(-1)[mask],
                measured_gradients[..., 0][mask],
                measured_gradients[..., 1][mask],
                color="dimgray",
                alpha=0.5,
                pivot="mid",
            )

    def _plot_arena_trajectory(self, ax, include_title=True):
        energy_values = [
            f"{agent.energy:.1f}" for agent in self.env.agent_objects if agent.active
        ]
        cumulative_rewards = [
            f"{agent.cumulative_reward:.1f}"
            for agent in self.env.agent_objects
            if agent.active
        ]
        agent_sizes = [
            f"{agent.agent_size:.1f}" for agent in self.env.agent_objects if agent.active
        ]

        self.fps_video = cfg.ENV_PARAMS["fps_video"]
        video_fps_multiplier = round(self.fps_video / ENV_PARAMS["fps_sim"], 2)
        if include_title:
            ax.set_title(
                f"Step: {self.env.step_count}\nVideo {video_fps_multiplier}x speed (sim fps: {ENV_PARAMS['fps_sim']})\nEnergy: {energy_values}\nCumu Reward: {cumulative_rewards}\nAgent Size: {agent_sizes}",
                fontsize=9,
            )

        # All food pellets as one PatchCollection — avoids 256 individual add_patch calls.
        food_radius = self.env.arena.food_radius
        food_patches = [
            plt.Circle(fp, radius=food_radius)
            for fp in self.env.arena.food_positions
        ]
        if food_patches:
            food_col = mcollections.PatchCollection(
                food_patches,
                facecolor=COLORS["food"],
                edgecolor=COLORS["food"],
                match_original=False,
            )
            ax.add_collection(food_col)

        for agent_id, agent in enumerate(self.env.agent_objects):
            if agent_id not in self.env.active_agent_ids:
                continue

            trajectory = np.array(agent.trajectory)
            if len(trajectory) > 0:
                ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    f"C{agent_id}-",
                    label=None,
                    alpha=0.8,
                )
                agent_color = (
                    COLORS["agent_bitten"] if agent.was_bitten else f"C{agent_id}"
                )
                if self.env.homing_mode:
                    other_agents = [
                        other for other in self.env.agent_objects if other != agent
                    ]
                    distances = [
                        np.linalg.norm(agent.position - other.position)
                        for other in other_agents
                    ]
                    is_homing = (
                        min(distances) <= self.env.homing_distance if distances else False
                    )
                else:
                    is_homing = False
                agent_color = "red" if is_homing else f"C{agent_id}"

                min_linewidth, max_linewidth = 0.5, 4.0
                agent_edge_width = min_linewidth + agent.agent_size * (max_linewidth - min_linewidth)

                agent_circle = plt.Circle(
                    agent.position,
                    radius=agent.body_radius,
                    edgecolor=agent_color,
                    facecolor=agent_color,
                )
                ax.add_patch(agent_circle)
                ax.text(
                    agent.position[0],
                    agent.position[1],
                    str(agent.agent_id),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    path_effects=[
                        path_effects.withStroke(linewidth=1, foreground="black")
                    ],
                )
                eating_color = (
                    COLORS["eating_other"]
                    if agent.bite_other_fish
                    else COLORS["eating_food"]
                )
                eating_circle = plt.Circle(
                    agent.position,
                    radius=agent.eating_radius,
                    edgecolor=eating_color,
                    facecolor="none",
                    linewidth=agent_edge_width,
                )
                ax.add_patch(eating_circle)
                food_sensing_circle = plt.Circle(
                    agent.position,
                    radius=agent.morm_food_detection_range * m_to_cm,
                    edgecolor="lightgray",
                    facecolor="none",
                )
                ax.add_patch(food_sensing_circle)

            if agent.asym_eating:
                left_angle = agent.orientation - agent.eating_angle / 2
                right_angle = agent.orientation + agent.eating_angle / 2

                arc = plt.matplotlib.patches.Arc(
                    agent.position,
                    width=2 * agent.eating_radius,
                    height=2 * agent.eating_radius,
                    angle=0,
                    theta1=np.degrees(left_angle),
                    theta2=np.degrees(right_angle),
                    edgecolor=COLORS["eating_cone"],
                    linestyle="--",
                    linewidth=1,
                )
                ax.add_patch(arc)

                left_end = agent.position + agent.eating_radius * np.array(
                    [np.cos(left_angle), np.sin(left_angle)]
                )
                right_end = agent.position + agent.eating_radius * np.array(
                    [np.cos(right_angle), np.sin(right_angle)]
                )
                ax.plot(
                    [agent.position[0], left_end[0]],
                    [agent.position[1], left_end[1]],
                    color=COLORS["eating_cone"],
                    linestyle="--",
                    linewidth=1,
                )
                ax.plot(
                    [agent.position[0], right_end[0]],
                    [agent.position[1], right_end[1]],
                    color=COLORS["eating_cone"],
                    linestyle="--",
                    linewidth=1,
                )

            if self.env.task == "1rw1f1p" and agent.agent_id == 0:
                bounded_walk_radius = agent.bounding_radius
                bounded_walk_circle = plt.Circle(
                    np.array(self.env.arena_size) / 2,
                    radius=bounded_walk_radius,
                    edgecolor=COLORS["bounded_walk_circ"],
                    facecolor="none",
                )
                ax.add_patch(bounded_walk_circle)

    def _plot_knollen_sensing(self, ax):
        for agent in self.env.agent_objects:
            for sensed_agent_pos in agent.knollen_sensed_agents:
                ax.plot(
                    [agent.position[0], sensed_agent_pos[0]],
                    [agent.position[1], sensed_agent_pos[1]],
                    linestyle="dashed",
                    color=COLORS["knollen"],
                    linewidth=1,
                    alpha=0.8,
                )

    def _plot_ampullary_sensing(self, ax):
        for agent in self.env.agent_objects:
            for sensed_object_pos in agent.ampullary_sensed_objects:
                ax.plot(
                    [agent.position[0], sensed_object_pos[0]],
                    [agent.position[1], sensed_object_pos[1]],
                    linestyle="dashed",
                    color=COLORS["ampullary"],
                    linewidth=1,
                    alpha=0.8,
                )

    def _plot_center_fields(self, ax, which=None):
        if which is None:
            which = ["knollen", "ampullary", "mormyromast"]
        for agent in self.env.agent_objects:
            for field_type in which:
                if field_type == "knollen":
                    for other_agent_id in range(self.env.num_agents):
                        if other_agent_id != agent.agent_id:
                            field_vector = agent.center_field[field_type][other_agent_id]
                            if np.linalg.norm(field_vector) > 0:
                                normalized_vector = field_vector / np.linalg.norm(field_vector)
                                ax.quiver(
                                    agent.position[0],
                                    agent.position[1],
                                    normalized_vector[0],
                                    normalized_vector[1],
                                    color=COLORS.get(f"{field_type}_direction", "black"),
                                    scale=5,
                                    width=0.005,
                                    linestyle="--",
                                )
                else:
                    field_vector = agent.center_field[field_type]
                    color = (
                        COLORS.get(f"{field_type}_direction", "blue")
                        if field_type == "mormyromast"
                        else "green"
                    )
                    if np.linalg.norm(field_vector) > 0:
                        normalized_vector = field_vector / np.linalg.norm(field_vector)
                        ax.quiver(
                            agent.position[0],
                            agent.position[1],
                            normalized_vector[0],
                            normalized_vector[1],
                            color=color,
                            scale=5,
                            width=0.005,
                        )

    # not currently in use
    def _plot_attention_allocation(self, ax):
        ax.clear()
        num_agents = len(self.env.agent_objects)
        timesteps = np.arange(max(0, self.env.curr_time - 49), self.env.curr_time + 1)

        for agent_id in range(num_agents):
            if not hasattr(self.env.agent_objects[agent_id], "attention_history"):
                continue

            data = np.array(self.env.agent_objects[agent_id].attention_history[-50:])
            bottoms = np.zeros(len(data))

            labels = ["mormyromast", "ampullary", "knollen"]
            colors = ["blue", "green", "purple"]

            for i in range(data.shape[1]):
                ax.bar(
                    timesteps,
                    data[:, i],
                    bottom=bottoms,
                    label=(labels[i] if agent_id == 0 else None),
                    color=colors[i],
                    alpha=0.7,
                )
                bottoms += data[:, i]

        ax.set_ylim(0, 1.0)
        ax.set_title("Sensor Attention Allocation")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Attention")
        ax.legend(loc="upper right")
        self._format_time_based_ax(ax)

    def _plot_attentions(self, ax, max_radius=1, sensor_band_radius=0.25):
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        self._remove_child_polar_axes(ax)

        num_agents = len(self.env.agent_objects)
        left, bottom, width, height = ax.get_position().bounds
        polar_width = (width / num_agents) * 0.9
        polar_height = height * 0.9

        for agent_id in range(num_agents):
            if agent_id not in self.env.active_agent_ids:
                continue

            polar_ax = self.fig.add_axes(
                [
                    left + agent_id * (width / num_agents) + (width / num_agents) * 0.05,
                    bottom + height * 0.05,
                    polar_width,
                    polar_height,
                ],
                polar=True,
            )
            ax.polar_axes.append(polar_ax)
            polar_ax.set_theta_offset(np.pi / 2)

            morm_att = self.attentions[
                agent_id,
                self.env.sensing_model.agent_electrics.mormyromast_indices_real,
            ]
            morm_att_closed = np.append(morm_att, morm_att[0])
            morm_angles = self.env.sensing_model.agent_electrics.mormyromast_angles
            morm_angles_closed = np.append(morm_angles, morm_angles[0])
            polar_ax.plot(
                morm_angles_closed,
                morm_att_closed + sensor_band_radius * 2,
                label="mormyr att",
                color="blue",
            )

            amp_att = self.attentions[
                agent_id, self.env.sensing_model.agent_electrics.ampullary_indices
            ]
            amp_att_closed = np.append(amp_att, amp_att[0])
            amp_angles = self.env.sensing_model.agent_electrics.ampullary_angles
            amp_angles_closed = np.append(amp_angles, amp_angles[0])
            polar_ax.plot(
                amp_angles_closed,
                amp_att_closed + sensor_band_radius,
                label="ampullary att",
                color="orange",
            )

            knollen_angles = self.env.sensing_model.agent_electrics.knollen_angles
            num_knollen_sensors = self.env.sensing_model.agent_electrics.num_knollen_sensors_real
            knollen_att = self.attentions[
                agent_id, self.env.sensing_model.agent_electrics.knollen_indices
            ]
            knollen_att_by_agent = np.reshape(
                knollen_att, (self.env.num_agents - 1, num_knollen_sensors)
            )
            agent_colors = [f"C{i}" for i in range(self.env.num_agents) if i != agent_id]
            knollen_angles_closed = np.append(knollen_angles, knollen_angles[0])
            for emitting_agent_id in range(self.env.num_agents - 1):
                knollen_att_closed = np.append(
                    knollen_att_by_agent[emitting_agent_id],
                    knollen_att_by_agent[emitting_agent_id][0],
                )
                polar_ax.plot(
                    knollen_angles_closed,
                    knollen_att_closed + sensor_band_radius * 3,
                    label="agg knollen att",
                    color=agent_colors[emitting_agent_id],
                    alpha=0.75,
                )

            polar_ax.set_ylim(0, max_radius)
            if agent_id != 0:
                polar_ax.set_xticklabels([])
                polar_ax.set_yticklabels([])

            if agent_id == num_agents - 1:
                legend_elements = [
                    Line2D([0], [0], color="blue", label="mormyr att"),
                    Line2D([0], [0], color="green", label="knollen att"),
                    Line2D([0], [0], color="orange", linestyle="dotted", label="ampullary att"),
                ]
                polar_ax.legend(
                    handles=legend_elements,
                    fontsize=6,
                    loc="lower center",
                    ncol=2,
                    bbox_to_anchor=(0.5, -0.2),
                )
            else:
                polar_ax.legend().remove()

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def render(self, mode=None, auxs=None, attentions=None):
        if auxs is None:
            auxs = ["eods", "spis"]

        env = self.env
        if getattr(env, "all_args", None) is not None and "auxs" in env.all_args:
            args_auxs = env.all_args["auxs"]
            if args_auxs is not None:
                auxs = args_auxs

        buffer_area = 0
        self.attentions = attentions

        aux_plot_funcs = {
            "eods": self._plot_eods_and_biting,
            "spis": self._plot_spis,
            "energy": self._plot_energy_levels,
            "observations": self._plot_observations,
            "attentions": self._plot_attentions,
        }
        num_aux_subplots = sum(aux in auxs for aux in aux_plot_funcs.keys())
        total_subplots = 1 + num_aux_subplots

        if self.fig is None or len(self.axes) != total_subplots:
            if self.fig is not None:
                plt.close(self.fig)
            self.fig, axes = plt.subplots(
                total_subplots,
                1,
                figsize=ENV_PARAMS["render_figsize"],
                dpi=ENV_PARAMS.get("render_dpi", 100),
                gridspec_kw={"height_ratios": [4] + [1] * num_aux_subplots},
            )
            self.axes = np.atleast_1d(axes)
            self.ax1 = self.axes[0]
            self.aux_axes = self.axes[1:]
            self._resize_with_buffer(buffer_area)

        self.ax1.clear()
        # Arena limits are fixed; set them before any patches so matplotlib's
        # _update_patch_limits (Bézier extrema → polynomial.roots → eigvals)
        # is skipped entirely for every patch added.
        self.ax1.set_aspect("equal", "box")
        self._resize_with_buffer(buffer_area)
        self.ax1.set_autoscale_on(False)
        # Suppress tick-object creation: ax.clear() resets the locator so we
        # re-apply NullLocator each frame (saves ~58 Tick.__init__ calls/frame).
        import matplotlib.ticker as mticker
        self.ax1.xaxis.set_major_locator(mticker.NullLocator())
        self.ax1.yaxis.set_major_locator(mticker.NullLocator())
        # Temporarily disable _update_patch_limits at the instance level.
        # The class method runs iter_bezier/axis_aligned_extrema per patch
        # even though we've already fixed the limits above.
        self.ax1._update_patch_limits = lambda _p: None
        self._plot_electric_field(self.ax1)
        self._plot_arena_trajectory(self.ax1)
        del self.ax1._update_patch_limits  # restore class method

        if "center_fields" in auxs:
            which = ["knollen"]
            which += ["ampullary"] if env.ampullary_mode == 1 else []
            which += ["mormyromast"] if env.mormyromast_mode == 1 else []
            self._plot_center_fields(self.ax1, which=which)

        aux_index = 0
        for aux_name in aux_plot_funcs:
            if aux_name in auxs:
                ax = self.aux_axes[aux_index]
                aux_plot_funcs[aux_name](ax)
                aux_index += 1

        for ax in self.aux_axes[aux_index:]:
            self._remove_child_polar_axes(ax)

        self._resize_with_buffer(buffer_area)

        if mode == "rgb_array":
            self.fig.canvas.draw()
            w, h = self.fig.canvas.get_width_height()
            frame = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
            if env.save_vid:
                if self.video_writer is None:
                    os.makedirs(env.video_save_dir, exist_ok=True)
                    random_digits = env.np_random.choice(10, size=8)
                    random_string = "".join(str(d) for d in random_digits)
                    filename = f"{env.video_save_dir}/MAEFish_{env.timestamp}_{random_string}.mp4"
                    self.video_writer_path = filename
                    self.video_writer = imageio.get_writer(
                        filename, format="ffmpeg", mode="I", fps=self.fps_video
                    )
                    print(f"Opened video writer at {filename}")
                self.video_writer.append_data(frame)
            return frame

    def save_svg_snapshot(self, filepath, env_index=None):
        env = self.env
        fig, ax = plt.subplots(1, 1, figsize=ENV_PARAMS.get("render_figsize", (6, 6)))
        ax.set_aspect("equal", "box")
        ax.set_xlim(0, env.arena_size[0])
        ax.set_ylim(0, env.arena_size[1])
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")

        homing_agent_emit_eod = env.agent_objects[1].emit_eod
        env.agent_objects[1].emit_eod = 0
        self._plot_electric_field(ax)
        env.agent_objects[1].emit_eod = homing_agent_emit_eod

        self._plot_arena_trajectory(ax, include_title=False)

        which = ["knollen"]
        if env.ampullary_mode == 1:
            which.append("ampullary")
        if env.mormyromast_mode == 1:
            which.append("mormyromast")
        self._plot_center_fields(ax, which=which)

        if env_index is not None:
            base, ext = os.path.splitext(filepath)
            filepath = f"{base}_env{env_index}{ext}"

        plt.savefig(filepath)
        plt.close(fig)
        print(f"Saved SVG snapshot to {filepath}")
        return filepath

    def close(self):
        if self.video_writer is not None:
            self.video_writer.close()
            print(f"Saved video to {self.video_writer_path}")
            path = self.video_writer_path
            self.video_writer = None
        else:
            path = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
            self.ax1 = None
            self.aux_axes = None
        return path
