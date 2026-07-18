import sys, os; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib.pyplot as plt
import numpy as np

from electric import (
    measure_electric_field,
    measure_electric_potential,
)
from electric_scene import ElectricScene, ElectricSourceSet, to_world_sensors
import utils_electric as ue

# ----------------------------------------------------------------------
# Visualization helpers (ElectricScene-first)
# ----------------------------------------------------------------------
def combine_scene_sources(scene, include_reflections=True, source_names=None):
    """Combine scene sources into a single ElectricSourceSet."""
    combined = ElectricSourceSet.empty()
    names = list(scene.source_sets.keys()) if source_names is None else source_names

    for name in names:
        if name in scene.source_sets:
            combined = combined.concat(scene.source_sets[name])
        if include_reflections and name in scene.reflected_source_sets:
            combined = combined.concat(scene.reflected_source_sets[name])
    return combined


def render_scene(
    fish_eods,
    fish_positions_cm,
    fish_orientations_rad,
    fish_monopole_positions_cm,
    fish_monopole_charges,
    fish_dipole_positions_cm,
    fish_dipole_moments,
    fish_sensor_positions_cm,
    fish_sensor_normals,
    fish_radius_cm,
    conductor_positions_cm,
    conductor_radii_cm,
    conductor_contrasts,
    arena_size_cm,
    grid_size_cm,
    measurement_noise=0.0,
    eps_m=1e-5,
    outfile="electric_scene.png",
    title_text="Electric Field Simulation",
    visualize="quiver",  # quiver, streamplot, contour, contour_streamplot
    fish_contrasts=None,
    fish_radii_cm=None,
    conductor_intrinsic_dipole_moments=None,
    fish_intrinsic_dipole_moments=None,
    clip_percentile=90,
):
    """
    ElectricScene-native version of electric.debug_render().
    Reflects sources before inducing dipoles (matching electric.py visuals).
    """
    scene = ElectricScene(arena_size_cm=arena_size_cm, eps_m=eps_m)
    scene.update(
        fish_positions_cm,
        fish_orientations_rad,
        fish_eods,
        fish_monopole_positions_cm,
        fish_monopole_charges,
        fish_dipole_positions_cm,
        fish_dipole_moments,
        conductor_positions_cm,
        conductor_contrasts,
        conductor_radii_cm,
        fish_contrasts=fish_contrasts,
        fish_radii=fish_radii_cm,
        conductor_intrinsic_dipole_moments_ego=conductor_intrinsic_dipole_moments,
        fish_intrinsic_dipole_moments_ego=fish_intrinsic_dipole_moments,
        induce_with_reflections=True,
    )

    ax = plt.gca()
    ax.set_aspect("equal", "box")
    ax.clear()

    # Draw fish bodies
    f = fish_positions_cm.shape[0]
    for i in range(f):
        fish_circle = plt.Circle(
            fish_positions_cm[i],
            radius=fish_radius_cm,
            edgecolor="black",
            facecolor="white",
        )
        ax.add_patch(fish_circle)

    # Visualize monopoles (pre-reflection, like electric.debug_render)
    eod_sources = scene.source_sets["eod"]
    p = eod_sources.monopole_pos.shape[0]
    for i in range(p):
        color = "blue" if eod_sources.monopole_q[i] < 0 else "red"
        monopole_circle = plt.Circle(
            eod_sources.monopole_pos[i],
            radius=0.5,
            edgecolor="black",
            facecolor=color,
        )
        ax.add_patch(monopole_circle)

    # Sensors (optional)
    have_sensors = (
        fish_sensor_positions_cm is not None
        and fish_sensor_normals is not None
    )
    sensor_measurements = None
    if have_sensors:
        (
            scene_global_sensor_positions,
            scene_global_sensor_normals,
        ) = to_world_sensors(
            fish_positions_cm,
            fish_orientations_rad,
            fish_sensor_positions_cm,
            fish_sensor_normals,
        )
        sensor_gradients = scene.get_field_at_positions(
            ["eod", "induced_food", "induced_fish", "intrinsic"],
            scene_global_sensor_positions,
            use_reflections=True,
            measurement_noise=measurement_noise,
        )
        sensor_measurements = scene.project_field_on_normals(
            sensor_gradients,
            scene_global_sensor_normals,
        )
        s = scene_global_sensor_positions.shape[0]
        for i in range(s):
            x0 = scene_global_sensor_positions[i]
            x1 = x0 + scene_global_sensor_normals[i] * 5
            m = sensor_measurements[i]
            color = [0, 0, -m] if m < 0 else [m, 0, 0]
            ax.plot([x0[0], x1[0]], [x0[1], x1[1]], color=color, linewidth=2, alpha=0.5)

    # Conductors
    if conductor_positions_cm is not None and conductor_positions_cm.shape[0] > 0:
        c = conductor_positions_cm.shape[0]
        for i in range(c):
            conductor_circle = plt.Circle(
                conductor_positions_cm[i],
                radius=conductor_radii_cm[i],
                edgecolor="black",
                facecolor="green",
            )
            ax.add_patch(conductor_circle)

    # Field sampling grid
    x = np.arange(0, arena_size_cm[0], grid_size_cm)
    y = np.arange(0, arena_size_cm[1], grid_size_cm)
    X, Y = np.meshgrid(x, y)
    xy = np.stack((X.reshape(-1), Y.reshape(-1)), axis=-1)

    scene_all_reflected = combine_scene_sources(scene, include_reflections=True)
    measured_gradients = measure_electric_field(
        xy,
        monopole_positions_cm=scene_all_reflected.monopole_pos,
        monopole_charges=scene_all_reflected.monopole_q,
        dipole_positions_cm=scene_all_reflected.dipole_pos,
        dipole_moments=scene_all_reflected.dipole_p,
        measurement_noise=0,
        eps_m=eps_m,
    )
    measured_gradients = ue.clip_gradients(
        measured_gradients, percentile=clip_percentile
    )

    if visualize == "quiver":
        plt.quiver(
            X.reshape(-1),
            Y.reshape(-1),
            measured_gradients[..., 0],
            measured_gradients[..., 1],
            color="green",
            pivot="mid",
        )
    elif visualize == "streamplot":
        plt.streamplot(
            x,
            y,
            measured_gradients[..., 0].reshape(X.shape),
            measured_gradients[..., 1].reshape(X.shape),
            density=1.5,
            color="lightgray",
            linewidth=0.5,
        )
    elif visualize == "contour":
        electric_potential = measure_electric_potential(
            xy,
            monopole_positions_cm=scene_all_reflected.monopole_pos,
            monopole_charges=scene_all_reflected.monopole_q,
            dipole_positions_cm=scene_all_reflected.dipole_pos,
            dipole_moments=scene_all_reflected.dipole_p,
            measurement_noise=0,
            eps_m=eps_m,
        ).reshape(X.shape)
        scale_factor = np.percentile(np.abs(electric_potential), 95)
        electric_potential_scaled = np.sign(electric_potential) * np.log1p(
            np.abs(electric_potential) / scale_factor
        )
        contour_levels = np.linspace(
            np.min(electric_potential_scaled), np.max(electric_potential_scaled), 50
        )
        plt.contourf(
            X,
            Y,
            electric_potential_scaled,
            levels=contour_levels,
            cmap="coolwarm",
            alpha=0.5,
        )
        plt.colorbar(label="Scaled Electric Potential")
    elif visualize == "contour_streamplot":
        electric_potential = measure_electric_potential(
            xy,
            monopole_positions_cm=scene_all_reflected.monopole_pos,
            monopole_charges=scene_all_reflected.monopole_q,
            dipole_positions_cm=scene_all_reflected.dipole_pos,
            dipole_moments=scene_all_reflected.dipole_p,
            measurement_noise=0,
            eps_m=eps_m,
        ).reshape(X.shape)
        scale_factor = np.percentile(np.abs(electric_potential), 95)
        electric_potential_scaled = np.sign(electric_potential) * np.log1p(
            np.abs(electric_potential) / scale_factor
        )
        contour_levels = np.linspace(
            np.min(electric_potential_scaled), np.max(electric_potential_scaled), 50
        )
        plt.contourf(
            X,
            Y,
            electric_potential_scaled,
            levels=contour_levels,
            cmap="coolwarm",
            alpha=0.5,
        )
        plt.streamplot(
            x,
            y,
            measured_gradients[..., 0].reshape(X.shape),
            measured_gradients[..., 1].reshape(X.shape),
            color="green",
            density=1.0,
            linewidth=0.5,
        )

    ax.set_xlim(0, arena_size_cm[0])
    ax.set_ylim(0, arena_size_cm[1])
    ax.set_yticks([])
    ax.set_xticks([])

    charges_str = "[" + ", ".join([f"{c:.1e}" for c in scene_all_reflected.monopole_q]) + "]"

    if sensor_measurements is not None:
        s_per_fish = fish_sensor_positions_cm.shape[1]
        measurements_str = "\n  ".join([
            f"Fish {i}: [{', '.join([f'{m:.1e}' for m in sensor_measurements[i * s_per_fish:(i + 1) * s_per_fish]])}]"
            for i in range(f)
        ])
    else:
        measurements_str = "No sensor measurements"

    title = f"{title_text}\nCharges: {charges_str}\nMeasurements: {measurements_str}"
    ax.set_title(title, pad=5, fontsize=9)

    if outfile is not None:
        plt.savefig(outfile)
        print(f"Saved {outfile}")
    return ax


def test_render_scene(figtype="svg"):
    """ElectricScene equivalent of electric.manuscript_figure()."""
    fish_positions_cm = np.array([[20, 20]])
    fish_orientations_rad = np.array([0.0])

    fish_monopole_positions_cm = np.array([[1, 0], [-1, 0]])[None, ...]
    fish_monopole_charges      = np.array([1e-15, -1e-15])[None, ...]
    fish_dipole_positions_cm   = np.zeros((1, 0, 2))
    fish_dipole_moments        = np.zeros((1, 0, 2))
    fish_eods                  = np.array([1])

    fish_sensor_positions_cm = None
    fish_sensor_normals   = None

    conductor_positions_cm = np.array([[30, 30]])
    conductor_contrasts = np.array([-0.95]) # emphasize effect for figure
    conductor_radii_cm  = np.array([1.2])

    arena_size_cm = (60, 60)
    arena_clipping_distance_cm = 10

    fish_radius_cm = 2.0
    for viz_type in ["quiver", "streamplot", "contour", "contour_streamplot"]:
        if figtype == "png":
            fname = f"electric_with_conductors_scene_{viz_type}.png"
        else:
            fname = f"electric_with_conductors_scene_{viz_type}.svg"
        # print(f"Saving: {fname}")
        render_scene(
            fish_eods,
            fish_positions_cm,
            fish_orientations_rad,
            fish_monopole_positions_cm,
            fish_monopole_charges,
            fish_dipole_positions_cm,
            fish_dipole_moments,
            fish_sensor_positions_cm,
            fish_sensor_normals,
            fish_radius_cm,
            conductor_positions_cm,
            conductor_radii_cm,
            conductor_contrasts,
            arena_size_cm,
            grid_size_cm=1.5,
            measurement_noise=0.0,
            eps_m=1e-5,
            outfile=fname,
            title_text="Electric Field Simulation",
            visualize=viz_type,
            fish_contrasts=None,
            fish_radii_cm=None,
            conductor_intrinsic_dipole_moments=None,
            fish_intrinsic_dipole_moments=None,
            clip_percentile=90,
        )


def test_render_manuscript_three_conditions(figtype="svg"):
    """
    ElectricScene-native version of manuscript_figure_three_conditions().
    Renders the same four panels using the scene interface.
    """

    def make_grid(xmin, xmax, ymin, ymax, n=160):
        x = np.linspace(xmin, xmax, n)
        y = np.linspace(ymin, ymax, n)
        X, Y = np.meshgrid(x, y)
        xy = np.stack((X.ravel(), Y.ravel()), axis=-1)
        return x, y, X, Y, xy

    def plot_panel(
        label,
        scene: ElectricScene,
        grid_limits,
        fish_positions_cm,
        fish_orientations_rad,
        fish_monopole_positions_cm,
        fish_monopole_charges,
        fish_dipole_positions_cm,
        fish_dipole_moments,
        fish_eods,
        conductor_positions_cm=None,
        conductor_contrasts=None,
        conductor_radii_cm=None,
        fish_contrasts=None,
        fish_radii_cm=None,
        wall_x=None,
        use_reflections=False,
    ):
        xmin, xmax, ymin, ymax = grid_limits
        x, y, X, Y, xy = make_grid(xmin, xmax, ymin, ymax, n=160)

        scene.update(
            fish_positions_cm,
            fish_orientations_rad,
            fish_eods,
            fish_monopole_positions_cm,
            fish_monopole_charges,
            fish_dipole_positions_cm,
            fish_dipole_moments,
            conductor_positions_cm,
            conductor_contrasts,
            conductor_radii_cm,
            fish_contrasts=fish_contrasts,
            fish_radii=fish_radii_cm,
            induce_with_reflections=use_reflections,
        )

        grads = scene.get_field_at_positions(
            ["eod", "induced_food", "induced_fish", "intrinsic"],
            xy,
            use_reflections=use_reflections,
            measurement_noise=0.0,
        )
        Ex = grads[:, 0].reshape(X.shape)
        Ey = grads[:, 1].reshape(X.shape)

        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        ax.set_aspect("equal", "box")

        ax.streamplot(
            x,
            y,
            Ex,
            Ey,
            color="0.5",
            density=1.25,
            linewidth=0.3,
        )

        if conductor_positions_cm is not None and conductor_radii_cm is not None:
            for pos, r in zip(conductor_positions_cm, conductor_radii_cm):
                circ = plt.Circle(
                    (pos[0], pos[1]),
                    radius=r,
                    edgecolor="black",
                    facecolor="#4caf50",
                    zorder=3,
                )
                ax.add_patch(circ)

        if wall_x is not None:
            ax.plot(
                [wall_x, wall_x],
                [ymin, ymax],
                color="black",
                linewidth=1.5,
                linestyle="-",
                zorder=4,
            )

        if fish_positions_cm is not None:
            for cx, cy in fish_positions_cm:
                body = plt.Circle(
                    (cx, cy),
                    radius=0.3,
                    edgecolor="black",
                    facecolor="white",
                    zorder=4,
                )
                ax.add_patch(body)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="black",
            linestyle="--",
            linewidth=1.0,
            zorder=5,
        )
        ax.add_patch(rect)

        outname = f"manuscript_{label}.{figtype}"
        fig.savefig(outname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {outname}")

    # ---------- shared physical setup ----------
    d_cm = 1.0
    q = 1e-15

    mono_pos_local_cm = np.array([[0.0, +d_cm], [0.0, -d_cm]])[None, ...]
    mono_q_local = np.array([+q, -q])[None, ...]
    dipole_pos_local = np.zeros((1, 0, 2))
    dipole_mom_local = np.zeros((1, 0, 2))
    fish_orientations = np.array([0.0])

    # Condition 1: isolated fish
    scene = ElectricScene(arena_size_cm=(12, 12))
    plot_panel(
        label="cond1_isolated",
        scene=scene,
        grid_limits=(-6, 6, -6, 6),
        fish_positions_cm=np.array([[0.0, 0.0]]),
        fish_orientations_rad=fish_orientations,
        fish_monopole_positions_cm=mono_pos_local_cm,
        fish_monopole_charges=mono_q_local,
        fish_dipole_positions_cm=dipole_pos_local,
        fish_dipole_moments=dipole_mom_local,
        fish_eods=np.array([1]),
        conductor_positions_cm=None,
        conductor_contrasts=None,
        conductor_radii_cm=None,
        use_reflections=False,
    )

    # Condition 2: isolated fish + nearby conductor
    conductor_positions_cm = np.array([[3.0, 1.0]])
    conductor_contrasts = np.array([-0.95])
    conductor_radii_cm = np.array([1.0])
    scene = ElectricScene(arena_size_cm=(12, 12))
    plot_panel(
        label="cond2_conductor",
        scene=scene,
        grid_limits=(-6, 6, -6, 6),
        fish_positions_cm=np.array([[0.0, 0.0]]),
        fish_orientations_rad=fish_orientations,
        fish_monopole_positions_cm=mono_pos_local_cm,
        fish_monopole_charges=mono_q_local,
        fish_dipole_positions_cm=dipole_pos_local,
        fish_dipole_moments=dipole_mom_local,
        fish_eods=np.array([1]),
        conductor_positions_cm=conductor_positions_cm,
        conductor_contrasts=conductor_contrasts,
        conductor_radii_cm=conductor_radii_cm,
        use_reflections=False,
    )

    # Condition 3: fish near non-conducting vertical wall (image charges included explicitly)
    wall_x = 0.0
    a_cm = 2.5
    fish_positions_cm = np.array([[-a_cm, 0.0], [+a_cm, 0.0]])
    fish_orientations_rad = np.array([0.0, 0.0])
    fish_eods = np.array([1, 1])
    scene = ElectricScene(arena_size_cm=(12, 12))
    plot_panel(
        label="cond3_wall",
        scene=scene,
        grid_limits=(-6, 6, -6, 6),
        fish_positions_cm=fish_positions_cm,
        fish_orientations_rad=fish_orientations_rad,
        fish_monopole_positions_cm=mono_pos_local_cm,
        fish_monopole_charges=mono_q_local,
        fish_dipole_positions_cm=dipole_pos_local,
        fish_dipole_moments=dipole_mom_local,
        fish_eods=fish_eods,
        conductor_positions_cm=None,
        conductor_contrasts=None,
        conductor_radii_cm=None,
        wall_x=wall_x,
        use_reflections=False,
    )

    # Condition 4: fish centered in a box with reflected field (non-conducting walls)
    arena_half = 3.0
    arena_size_cm = (2 * arena_half, 2 * arena_half)
    scene = ElectricScene(arena_size_cm=arena_size_cm)
    plot_panel(
        label="cond4_boxwall",
        scene=scene,
        grid_limits=(-arena_half, arena_half, -arena_half, arena_half),
        fish_positions_cm=np.array([[0.0, 0.0]]),
        fish_orientations_rad=fish_orientations,
        fish_monopole_positions_cm=mono_pos_local_cm,
        fish_monopole_charges=mono_q_local,
        fish_dipole_positions_cm=dipole_pos_local,
        fish_dipole_moments=dipole_mom_local,
        fish_eods=np.array([1]),
        conductor_positions_cm=None,
        conductor_contrasts=None,
        conductor_radii_cm=None,
        wall_x=None,
        use_reflections=True,
    )

def test_food_intrinsic_on_axis_field(
    q_food=1e-15,
    distances_cm=np.arange(1, 11, dtype=float),
    receiver_pos_cm=np.array([0.0, 0.0], dtype=float),
    eps_m=1e-5,
    outfile="food_intrinsic_on_axis_field.png",
):
    """
    Plot the on-axis electric field at a receiver fish (at receiver_pos_cm),
    produced by a *food intrinsic monopole charge* placed on the +x axis at
    distances 1..10 cm from the receiver.

    Plots:
      - signed on-axis component: E · u  (u points from receiver -> food)
      - field magnitude: ||E||

    Saves to `outfile`.
    """
    distances_cm = np.asarray(distances_cm, dtype=float).ravel()
    assert np.all(distances_cm > 0), "Distances must be > 0"

    # Unit axis direction: receiver -> food (here always +x)
    u_axis = np.array([1.0, 0.0], dtype=float)

    E_on_axis = []
    E_mag = []

    # Evaluate field at receiver position, due to a single monopole at food position
    receiver_points_cm = receiver_pos_cm[None, :]  # (1,2)

    for d in distances_cm:
        food_pos_cm = receiver_pos_cm + np.array([d, 0.0], dtype=float)

        grads = measure_electric_field(
            receiver_points_cm,
            monopole_positions_cm=food_pos_cm[None, :],   # (1,2)
            monopole_charges=np.array([q_food], dtype=float),  # (1,)
            dipole_positions_cm=np.zeros((0, 2), dtype=float),
            dipole_moments=np.zeros((0, 2), dtype=float),
            measurement_noise=0.0,
            eps_m=eps_m,
        )
        E = grads[0]  # (2,)

        E_on_axis.append(float(np.dot(E, u_axis)))
        E_mag.append(float(np.linalg.norm(E)))

    E_on_axis = np.array(E_on_axis, dtype=float)
    E_mag = np.array(E_mag, dtype=float)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # Signed on-axis component (can be negative depending on charge sign)
    ax.plot(distances_cm, E_on_axis, marker="o", label="E_on_axis = E · u (signed)")

    # Magnitude (use log scale because ~1/r^2)
    ax2 = ax.twinx()
    ax2.plot(distances_cm, E_mag, marker="s", linestyle="--", label="||E||")
    ax2.set_yscale("log")

    ax.set_xlabel("Food distance from receiver (cm)")
    ax.set_ylabel("On-axis field component (arb units)")
    ax2.set_ylabel("Field magnitude ||E|| (arb units, log scale)")

    title = (
        "On-axis field at receiver due to intrinsic food monopole\n"
        f"q_food={q_food:.2e}, eps_m={eps_m:.1e}  (food on +x axis)"
    )
    ax.set_title(title)

    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    test_render_scene(figtype="png")
    test_render_manuscript_three_conditions(figtype="png")
    test_food_intrinsic_on_axis_field(outfile="food_intrinsic_on_axis_field.png")
    

