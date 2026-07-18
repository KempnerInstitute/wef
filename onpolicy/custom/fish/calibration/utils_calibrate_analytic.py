import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, _here)                          # calibration/ for utils_calibrate
sys.path.insert(2, os.path.join(_here, '..'))      # fish/ for cfg, electric, MAEFish
import utils_calibrate as uc
import numpy as np
import matplotlib.pyplot as plt
from electric import fish_forward
from cfg import ELECTRIC_CONSTANTS, AGENT_PARAMS, ENV_PARAMS
from MAEFish import EFishAgent
import pandas as pd
import math

# Constants from your code
k_coulomb = ELECTRIC_CONSTANTS["k_coulomb"]  # Coulomb's constant
epsilon_0 = ELECTRIC_CONSTANTS["epsilon_0"]  # Vacuum permittivity (F/m)
mVcm_to_Vm = ELECTRIC_CONSTANTS["mVcm_to_Vm"]
cm_to_m = 0.01


def dipole_moment(E, r):
    # axial formula: E = (1/(2*pi*eps0)) * p / r^3
    return E * 2 * math.pi * epsilon_0 * r**3

def monopole_charge(p, d):
    # p = q * d
    return p / d

def axial_field(p, r):
    # axial field: E = (1/(2*pi*eps0)) * p / r^3
    return (1 / (2 * math.pi * epsilon_0)) * p / (r**3)


# Axial field calculations
def monopole_field_on_axis(y, y_i, q, epsilon0=8.854e-12):
    """On-axis field Ey at y due to a monopole q at y_i."""
    dy = y - y_i
    return (q * dy) / (4.0 * np.pi * epsilon0 * np.abs(dy) ** 3)


def dipole_field_on_axis(delta_y, p, epsilon0=8.854e-12):
    """On-axis field Ey at separation delta_y due to a dipole moment p (aligned with +y)."""
    return (p * np.sign(delta_y)) / (2.0 * np.pi * epsilon0 * np.abs(delta_y) ** 3)


def induced_dipole_on_sphere(E_external, R, chi, epsilon0=8.854e-12):
    """p_ind = 4π ε0 R^3 χ E_external (aligned with the external field, i.e., along +y here)."""
    return 4.0 * np.pi * epsilon0 * (R**3) * chi * E_external


# -----------------------------
# Geometry helpers (all on y-axis)
# -----------------------------


def self_eod_field_at_sensor(q, d, epsilon0=8.854e-12):
    """
    Receiver at y_s = d/2. Self EOD is ±q at y=±d/4.
    Exact on-axis (two-monopole) expression.
    """
    ys = d / 2.0
    return monopole_field_on_axis(ys, d / 4.0, +q, epsilon0) + monopole_field_on_axis(
        ys, -d / 4.0, -q, epsilon0
    )


def cons_eod_field_at_sensor(q, d, x_cons, epsilon0=8.854e-12):
    """
    Receiver sensor at y_s = d/2. Cons fish centered at x_cons with EOD at x_cons±d/4.
    Returns Ey from those two monopoles.
    """
    ys = d / 2.0
    return monopole_field_on_axis(
        ys, x_cons + d / 4.0, +q, epsilon0
    ) + monopole_field_on_axis(ys, x_cons - d / 4.0, -q, epsilon0)


def self_external_field_at_food(q, d, x_food, epsilon0=8.854e-12):
    """
    Field (Ey) at food location y = x_food due to own EOD ±q at ±d/4.
    """
    return monopole_field_on_axis(
        x_food, d / 4.0, +q, epsilon0
    ) + monopole_field_on_axis(x_food, -d / 4.0, -q, epsilon0)


def cons_external_field_at_food(q, d, x_cons, x_food, epsilon0=8.854e-12):
    """
    Field (Ey) at food y = x_food due to cons EOD ±q at x_cons±d/4.
    """
    return monopole_field_on_axis(
        x_food, x_cons + d / 4.0, +q, epsilon0
    ) + monopole_field_on_axis(x_food, x_cons - d / 4.0, -q, epsilon0)


def induced_field_at_sensor_from_food(p_ind, d, x_food, epsilon0=8.854e-12):
    """
    Field at the receiver sensor at y_s = d/2, from a food-centered dipole p_ind at y = x_food.
    On-axis dipole expression.
    """
    delta_y = (d / 2.0) - x_food
    return dipole_field_on_axis(delta_y, p_ind, epsilon0)


# -----------------------------
# High-level composites
# -----------------------------


def self_image_components(
    q, d, R, chi, x_food, epsilon0=8.854e-12, subtract_baseline=True
):
    """
    Self-image case:
      1) Self field at sensor (baseline)
      2) External self field at food
      3) Induced dipole on food
      4) Induced field at sensor
      5) Total sensed (with or without baseline subtraction)
    Returns a dict of components.
    """
    E_self_sensor = self_eod_field_at_sensor(q, d, epsilon0)  # baseline
    E_ext_food = self_external_field_at_food(q, d, x_food, epsilon0)  # drives induction
    p_ind = induced_dipole_on_sphere(E_ext_food, R, chi, epsilon0)  # induced dipole
    E_ind_sensor = induced_field_at_sensor_from_food(p_ind, d, x_food, epsilon0)

    if subtract_baseline:
        E_sensed = E_ind_sensor
    else:
        E_sensed = E_self_sensor + E_ind_sensor

    return {
        "E_self_EOD_at_sensor": E_self_sensor,
        "E_external_at_food": E_ext_food,
        "p_induced": p_ind,
        "E_induced_at_sensor": E_ind_sensor,
        "E_sensed": E_sensed,
    }


def cons_image_components(
    q, d, R, chi, x_food, x_cons, epsilon0=8.854e-12, subtract_baseline=True
):
    """
    Cons-image case:
      1) Cons EOD field at sensor (cons-baseline)
      2) Cons EOD field at food
      3) Induced dipole on food (from cons field)
      4) Induced field at sensor
      5) Total sensed (with or without cons-baseline subtraction)
    Returns a dict of components.
    """
    E_cons_sensor = cons_eod_field_at_sensor(q, d, x_cons, epsilon0)  # cons-baseline
    E_ext_food = cons_external_field_at_food(
        q, d, x_cons, x_food, epsilon0
    )  # drives induction
    p_ind = induced_dipole_on_sphere(E_ext_food, R, chi, epsilon0)  # induced dipole
    E_ind_sensor = induced_field_at_sensor_from_food(p_ind, d, x_food, epsilon0)

    if subtract_baseline:
        E_sensed = E_ind_sensor
    else:
        E_sensed = E_cons_sensor + E_ind_sensor

    return {
        "E_cons_EOD_at_sensor": E_cons_sensor,
        "E_external_at_food": E_ext_food,
        "p_induced": p_ind,
        "E_induced_at_sensor": E_ind_sensor,
        "E_sensed": E_sensed,
    }


# -----------------------------
# Far-field (d << x) approximations
# -----------------------------


def self_image_far_field(
    q, d, R, chi, x_food, epsilon0=8.854e-12, subtract_baseline=True
):
    """
    Far-field self-image (d << x_food). Use dipole p = q d / 2.
    Ey_self at sensor ~ 2q/(π ε0 d^2)
    Ey_ext at food   ~ p/(2π ε0 x_food^3)
    p_ind ~ 2 R^3 χ p / x_food^3
    Ey_ind at sensor ~ (R^3 χ / (π ε0)) * p / x_food^6
    """
    p = q * d / 2.0
    E_self_sensor = 2.0 * q / (np.pi * epsilon0 * d**2)
    E_ind_sensor = (R**3 * chi * p) / (np.pi * epsilon0 * x_food**6)
    if subtract_baseline:
        E_sensed = -E_ind_sensor  # induced field opposes
    else:
        E_sensed = E_self_sensor - E_ind_sensor
    return {
        "E_self_EOD_at_sensor": E_self_sensor,
        "E_induced_at_sensor": E_ind_sensor,
        "E_sensed": E_sensed,
    }


def cons_image_far_field(
    q, d, R, chi, x_food, x_cons, epsilon0=8.854e-12, subtract_baseline=True
):
    """
    Far-field cons-image (d << x_food, d << x_cons). p = q d / 2.
    Ey_cons baseline at sensor ~ p/(2π ε0 x_cons^3)
    Ey_ind at sensor ~ (R^3 χ / (π ε0)) * p / x_food^6
    """
    p = q * d / 2.0
    E_cons_sensor = p / (2.0 * np.pi * epsilon0 * x_cons**3)
    E_ind_sensor = (R**3 * chi * p) / (np.pi * epsilon0 * x_food**6)
    if subtract_baseline:
        E_sensed = -E_ind_sensor  # induced field opposes
    else:
        E_sensed = E_cons_sensor - E_ind_sensor
    return {
        "E_cons_EOD_at_sensor": E_cons_sensor,
        "E_induced_at_sensor": E_ind_sensor,
        "E_sensed": E_sensed,
    }


def receiver_induced_field(q, x_cons, r, d, chi_r=-0.5, epsilon0=8.854e-12):
    # External field at receiver center (y=0) from cons EOD
    E_ext_recv = monopole_field_on_axis(
        0.0, x_cons + d / 4.0, +q, epsilon0
    ) + monopole_field_on_axis(0.0, x_cons - d / 4.0, -q, epsilon0)

    # Induced dipole on receiver
    p_ind_recv = induced_dipole_on_sphere(E_ext_recv, r, chi_r, epsilon0)

    # Field from induced dipole at sensor (y=d/2)
    delta_y = d / 2.0
    return dipole_field_on_axis(delta_y, p_ind_recv, epsilon0)

def get_frac_rand_grid(grid_style, mag_min, mag_max, n, p=1):
    if grid_style == "linear_in_logspace":
        grid = np.exp(np.linspace(np.log(mag_max), np.log(mag_min), n))
    elif grid_style == "linear":
        grid = np.linspace(mag_max, mag_min, n)
    elif grid_style == "log":
        grid = np.logspace(np.log10(mag_max), np.log10(mag_min), n)
    elif grid_style == "linear_in_distance":
        # distance bounds implied by |E| ∝ r^-p
        r_min = mag_max ** (-1.0 / p)
        r_max = mag_min ** (-1.0 / p)
        r = np.linspace(r_min, r_max, n)
        grid = r**-p
    else:
        raise ValueError(f"Unknown grid_style: {grid_style}")
    return grid


def compute_induced_to_self_ratio_vs_xvals(x_vals = np.linspace(0.0125, 0.05, 20),  # 1.25 cm to 5 cm
                                            epsilon0 = 8.854e-12,
                                            q = 1.11e-15,  # C (from Chen'2005)
                                            d = 0.02,  # m (body diameter)
                                            R = 0.0025,  # m (food radius)
                                            chi = -0.5,  # conductor contrast
                                           # Sweep object distances (must be > d/2 to be beyond the sensor on +y).
                                           method="exact", 
                                          ):
    rows = []
    for x in x_vals:
        readings = self_image_components(q, d, R, chi, x, epsilon0, subtract_baseline=False)
        if method != "exact":
            readings = self_image_far_field(q, d, R, chi, x, epsilon0, subtract_baseline=False)

        E_self = np.abs(readings["E_self_EOD_at_sensor"])
        E_ind = np.abs(readings["E_induced_at_sensor"])
        ratio = E_ind / E_self
        rows.append(
            {
                "x_obj_m": x,
                "x_obj_cm": x * 100.0,
                "E_self_at_sensor (V/m)": E_self,
                "E_ind_at_sensor (V/m)": E_ind,
                "E_ind/E_self": ratio,
            }
        )
    df = pd.DataFrame(rows)
    return df

def plot_morm_induced_to_self_ratio_vs_food_dist(df):
    # Plot E_ind/E_self vs x_obj on log-log axes for both exact and far-field
    plt.figure(figsize=(5, 3))
    plt.plot(df["x_obj_cm"], df["E_ind/E_self"], marker="o")
    plt.yscale("log")
    plt.xlabel("Food object distance x_obj (cm)")
    plt.ylabel("E_ind / E_self")
    plt.title("Induced-to-Self Field Ratio vs Food Distance")
    plt.axhline(y=0.25, color="gray", linestyle="--", label="25% ratio")
    plt.axhline(y=1e-6, color="gray", linestyle="--", label="1e-6 ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig("morm_induced_to_self_ratio_vs_food_dist.pdf")
    plt.show()
    # print(df["E_ind/E_self"].describe())

def plot_morm_induced_abs_vs_food_dist(df):
    # Plot induced distortion magnitude
    plt.figure(figsize=(5, 3))
    plt.plot(df["x_obj_cm"], df["E_ind_at_sensor (V/m)"], marker="o")
    plt.yscale("log")
    plt.xlabel("Object distance x_obj (cm)")
    plt.ylabel("E_ind_at_sensor (V/m)")
    plt.title(
        f"Induced Field vs Food Object Distance \nE_self: {df['E_self_at_sensor (V/m)'].mean():.2e} V/m"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("morm_induced_abs_vs_food_dist.pdf")
    plt.show()
    # print(df["E_ind_at_sensor (V/m)"].describe())

def plot_morm_sensed_vs_food_dist(df):
    # Plot actual field sensed at the sensor
    E_sensed = df["E_self_at_sensor (V/m)"] - df["E_ind_at_sensor (V/m)"]
    plt.figure(figsize=(5, 3))
    plt.plot(df["x_obj_cm"], E_sensed, marker="o")
    plt.axhline(
        df["E_self_at_sensor (V/m)"].mean() * 0.75,
        color="gray",
        linestyle="--",
        label="E_self_EOD - 25%",
    )
    plt.axhline(
        df["E_self_at_sensor (V/m)"].mean() * 1.25,
        color="gray",
        linestyle="--",
        label="E_self_EOD + 25%",
    )
    # plt.yscale("log")
    plt.xlabel("Food Object distance x_obj (cm)")
    plt.ylabel("E_sensed (V/m)")
    plt.title(f"Sensed Field vs Food Object Distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig("morm_sensed_vs_food_dist.pdf")
    plt.show()

# def plot_cons_field_vs_distance(
#     x_cons_vals, 
#     E_cons_vals, 
#     E_self_mean, 
#     plot_relative_to_self=False, 
#     plot_cons_spans=True,
#     x_cons_baselines=None,             # optional: list/array of meters
#     E_cons_baselines=None,             # optional: list/array of V/m matching x_cons_baselines
#     yscale="log",
#     savepath="morm_cons_field_vs_distance.png",
# ):
#     """
#     Plot |E_cons_EOD_at_sensor| vs conspecific distance.
#     If plot_relative_to_self=True, plot values relative to E_self_mean (unitless).
#     Optionally draw +/-25% spans around selected baseline distances.
#     """
#     x_cm = np.asarray(x_cons_vals, dtype=float).ravel() * 100.0
#     E = np.asarray(E_cons_vals, dtype=float).ravel()

#     # Choose y-data and horizontal guides based on mode
#     if plot_relative_to_self:
#         if not np.isfinite(E_self_mean) or E_self_mean == 0:
#             raise ValueError("E_self_mean must be finite and non-zero for relative plotting.")
#         y = np.abs(E) / float(E_self_mean)
#         guide_levels = [1.25, 1.0, 0.75]
#         y_label = r"$|E_{\mathrm{cons}}| / E_{\mathrm{self}}$)"
#         title = "Cons-EOD at sensor (relative to Self-EOD)"
#     else:
#         y = np.abs(E)
#         guide_levels = [E_self_mean * 1.25, E_self_mean, E_self_mean * 0.75]
#         y_label = r"$|E_{\mathrm{cons}}|$ (V/m)"
#         title = "Cons-EOD field at sensor vs distance"

#     plt.figure()
#     plt.plot(x_cm, y, marker="o", label="|E_cons_EOD|")

#     # Horizontal guides
#     if plot_relative_to_self:
#         # plt.axhline(guide_levels[0], color="gray", linestyle="--", label="E_self_EOD × 1.25")
#         plt.axhline(guide_levels[1], color="gray", linestyle="-.",  label="E_self_EOD × 1.00")
#         # plt.axhline(guide_levels[2], color="gray", linestyle=":",   label="E_self_EOD × 0.75")
#     else:
#         # plt.axhline(guide_levels[0], color="gray", linestyle="--", label="E_self_EOD + 25%")
#         plt.axhline(guide_levels[1], color="gray", linestyle="-.",  label="E_self_EOD")
#         # plt.axhline(guide_levels[2], color="gray", linestyle=":",   label="E_self_EOD - 25%")

#     # Optional spans at selected baseline distances
#     if plot_cons_spans and (x_cons_baselines is not None) and (E_cons_baselines is not None):
#         xb = np.asarray(x_cons_baselines, dtype=float).ravel()
#         Eb = np.asarray(E_cons_baselines, dtype=float).ravel()
#         if xb.size != Eb.size:
#             raise ValueError("x_cons_baselines and E_cons_baselines must have the same length.")

#         xb_cm = xb * 100.0
#         # transform baselines to plotting space (absolute or relative)
#         if plot_relative_to_self:
#             Eb_plot = np.abs(Eb) / float(E_self_mean)
#             plus  = Eb_plot * 1.25
#             minus = Eb_plot * 0.75
#         else:
#             Eb_plot = np.abs(Eb)
#             plus  = Eb_plot * 1.25
#             minus = Eb_plot * 0.75

#         for i, (xc_cm, base, hi, lo) in enumerate(zip(xb_cm, Eb_plot, plus, minus)):
#             c = f"C{(i % 10)}"
#             # Add labels only on the first for legend cleanliness
#             # lbl_plus = f"x_cons={xc_cm:.0f} cm + 25%" if i == 0 and not plot_relative_to_self else (f"x_cons={xc_cm:.0f} cm × 1.25" if i == 0 else None)
#             lbl_mid  = f"x_cons={xc_cm:.0f} cm" if i == 0 else None
#             # lbl_min  = f"x_cons={xc_cm:.0f} cm - 25%" if i == 0 and not plot_relative_to_self else (f"x_cons={xc_cm:.0f} cm × 0.75" if i == 0 else None)
#             # plt.axhline(hi,   color=c, linestyle="--", label=lbl_plus)
#             plt.axhline(base, color=c, linestyle="-",  label=lbl_mid)
#             # plt.axhline(lo,   color=c, linestyle=":",  label=lbl_min)

#     plt.xlabel("Conspecific distance x_cons (cm)")
#     plt.ylabel(y_label)
#     plt.title(title)
#     if yscale:
#         plt.yscale(yscale)
#     plt.grid(True, which="both", axis="both", alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(savepath)
#     plt.show()

def plot_cons_field_vs_distance(
    x_cons_vals, 
    E_cons_vals, 
    E_self_mean, 
    plot_relative_to_self=False, 
    plot_cons_spans=True,
    x_cons_baselines=None,             # optional: list/array of meters
    E_cons_baselines=None,             # optional: list/array of V/m matching x_cons_baselines
    yscale="log",
    savepath="morm_cons_field_vs_distance.png",
):
    """
    Plot |E_cons_EOD_at_sensor| vs conspecific distance.
    If plot_relative_to_self=True, plot values relative to E_self_mean (unitless).
    For selected baselines, draw green 'L' guides:
      vertical from x-axis (plot bottom) up to the curve value at x_baseline,
      then horizontal left to the y-axis.
    """
    x_cm = np.asarray(x_cons_vals, dtype=float).ravel() * 100.0
    E = np.asarray(E_cons_vals, dtype=float).ravel()

    # y-data & labels
    if plot_relative_to_self:
        if not np.isfinite(E_self_mean) or E_self_mean == 0:
            raise ValueError("E_self_mean must be finite and non-zero for relative plotting.")
        y = np.abs(E) / float(E_self_mean)
        y_label = r"$|E_{\mathrm{cons}}| / E_{\mathrm{self}}$"
        title = "Cons-EOD at sensor (relative to Self-EOD)"
        # Only the middle guide (E_self) if you want to keep it; comment out to remove
        middle_level = 1.0
    else:
        y = np.abs(E)
        y_label = r"$|E_{\mathrm{cons}}|$ (V/m)"
        title = "Cons-EOD field at sensor vs distance"
        middle_level = float(E_self_mean)

    fig, ax = plt.subplots()
    ax.plot(x_cm, y, marker="o", label="|E_cons_EOD|")

    # Show ONLY the middle horizontal guide if desired; comment these 2 lines to remove entirely
    ax.axhline(middle_level, color="gray", linestyle="-.", label="E_self_EOD")

    ax.set_xlabel("Conspecific distance x_cons (cm)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if yscale:
        ax.set_yscale(yscale)

    # Optional green L-shaped spans for selected baseline distances (ONLY the middle value)
    if plot_cons_spans and (x_cons_baselines is not None) and (E_cons_baselines is not None):
        xb = np.asarray(x_cons_baselines, dtype=float).ravel()
        Eb = np.asarray(E_cons_baselines, dtype=float).ravel()
        if xb.size != Eb.size:
            raise ValueError("x_cons_baselines and E_cons_baselines must have the same length.")

        xb_cm = xb * 100.0
        # transform baselines to plotting space (absolute or relative)
        Eb_plot = np.abs(Eb) / float(E_self_mean) if plot_relative_to_self else np.abs(Eb)

        # After yscale is set and data plotted, get current bottom (x-axis in plot coords)
        ymin, ymax = ax.get_ylim()
        # ensure the L vertical starts from the visible bottom
        ystart = ymin

        span_color = "green"
        for xc_cm, base in zip(xb_cm, Eb_plot):
            # vertical: from bottom up to the curve value at this x
            ax.plot([xc_cm, xc_cm], [ystart, base], color=span_color, linewidth=1.3) # TODO: Add a label?
            # horizontal: from y-axis (x=0) to the vertical line at xc
            ax.plot([0.0, xc_cm], [base, base], color=span_color, linewidth=1.3)

    ax.grid(True, which="both", axis="both", alpha=0.3)
    ax.legend()
    plt.xlim(left=0)
    fig.tight_layout()
    fig.savefig(savepath)
    plt.show()
