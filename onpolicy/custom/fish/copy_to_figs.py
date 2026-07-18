"""Copy per-eval PDF outputs to manuscript figure directories.

Destinations use a two-level path: figs/{fig}/{subfolder}/{original_filename}.
The `fig` field on each Entry encodes both the figure number and optional
subfolder, e.g. "fig3/patchy" → figs/fig3/patchy/{dst}.

Usage:
    python copy_to_figs.py <run_dir> [--dry-run] [--force] [--figs FIG [FIG ...]]

Output:
    run_dir/figs/fig3/patchy/{file}.pdf
    run_dir/figs/fig3/2fishwide/{file}.pdf
    run_dir/figs/fig4/sensor/{file}.pdf   (etc.)
    run_dir/figs/fig5/2f1p/{file}.pdf     (etc.)
    run_dir/figs/fig6/nfish/{file}.pdf    (etc.)
"""

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Entry:
    fig: str   # destination subdir under figs/, e.g. "fig3/patchy" or "fig4/sensor"
    src: str   # path relative to RUN_DIR; may include glob wildcards
    dst: str   # filename under figs/{fig}/
    note: str = ""  # non-empty → MISSING warning, skip copy attempt


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE MAP  — matches paths referenced in manuscript/figseed.tex
# ─────────────────────────────────────────────────────────────────────────────

ENTRIES: list[Entry] = [

    # ── FIG 3 / patchy — m1a1k1_patchy_square ────────────────────────────────

    Entry(fig="fig3/patchy",
          src="evals/m1a1k1_patchy_square/analyses/behavior/ethogram_sample.pdf",
          dst="ethogram_sample.pdf"),

    Entry(fig="fig3/patchy",
          src="evals/m1a1k1_patchy_square/analyses/behavior/trajectory_sample.pdf",
          dst="trajectory_sample.pdf"),

    Entry(fig="fig3/patchy",
          src="evals/m1a1k1_patchy_square/analyses/biting_network/biting_heatmap.pdf",
          dst="biting_heatmap.pdf"),

    Entry(fig="fig3/patchy",
          src="evals/m1a1k1_patchy_square/analyses/biting_network/win_ratio_vs_size.pdf",
          dst="win_ratio_vs_size.pdf"),

    # bitten-network (victim-centric) — same merged analysis_bite_network module
    Entry(fig="fig3/patchy",
          src="evals/m1a1k1_patchy_square/analyses/bitten_network/bitten_heatmap.pdf",
          dst="bitten_heatmap.pdf"),

    Entry(fig="fig3/patchy",
          src="evals/m1a1k1_patchy_square/analyses/bitten_network/loss_ratio_vs_size.pdf",
          dst="loss_ratio_vs_size.pdf"),

    Entry(fig="fig3/patchy",
          src="evals/m1a1k1_patchy_square/analyses/bitten_network/bitten_vs_food.pdf",
          dst="bitten_vs_food.pdf"),

    Entry(fig="fig3/patchy",
          src="evals/m1a1k1_patchy_square/analyses/bitten_network/bitten_rank_stability.pdf",
          dst="bitten_rank_stability.pdf"),

    Entry(fig="fig3/patchy",
          src="evals/m1a1k1_patchy_square/analyses/general/food_rank_vs_size_rank.pdf",
          dst="food_rank_vs_size_rank.pdf"),

    Entry(fig="fig3/patchy",
          src="evals/m1a1k1_patchy_square/analyses/general/size_vs_eod_rate.pdf",
          dst="size_vs_eod_rate.pdf"),

    Entry(fig="fig3/patchy",
          src="evals/m1a1k1_patchy_square/analyses/eod/eod_vs_size_advantage.pdf",
          dst="eod_vs_size_advantage.pdf"),

    Entry(fig="fig3/patchy",
          src="evals/m1a1k1_patchy_square/analyses/general/size_vs_food.pdf",
          dst="size_vs_food.pdf"),

    Entry(fig="fig3/patchy",
          src="evals/m1a1k1_patchy_square/analyses/general/food_ineq_vs_size_ineq_theil.pdf",
          dst="food_ineq_vs_size_ineq_theil.pdf"),

    Entry(fig="fig3/patchy",
          src="evals/m1a1k1_patchy_square/analyses/general/size_vs_biting.pdf",
          dst="size_vs_biting.pdf"),

    # ── FIG 3 / 2fishwide — 2fish_m1a1k1_uniform_wide ────────────────────────

    Entry(fig="fig3/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/idi/idi_histogram_12ms.pdf",
          dst="idi_histogram_12ms.pdf"),

    Entry(fig="fig3/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/idi/idi_gebhardt.pdf",
          dst="idi_gebhardt.pdf"),

    Entry(fig="fig3/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/idi/idi_powerlaw.pdf",
          dst="idi_powerlaw.pdf"),

    Entry(fig="fig3/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/idi/idi_by_agent_size.pdf",
          dst="idi_by_agent_size.pdf"),

    Entry(fig="fig3/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/eod/eod_vs_dist_agent_adaptive.pdf",
          dst="eod_vs_dist_agent_adaptive.pdf"),

    Entry(fig="fig3/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/eod/eod_vs_dist_food_adaptive.pdf",
          dst="eod_vs_dist_food_adaptive.pdf"),

    Entry(fig="fig3/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/eod/eod_vs_dist_wall_adaptive.pdf",
          dst="eod_vs_dist_wall_adaptive.pdf"),

    Entry(fig="fig3/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/eod/peri_eating_adaptive.pdf",
          dst="peri_eating_adaptive.pdf"),

    Entry(fig="fig3/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/eod/peri_biting_adaptive.pdf",
          dst="peri_biting_adaptive.pdf"),

    Entry(fig="fig3/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/eod/peri_bitten_adaptive.pdf",
          dst="peri_bitten_adaptive.pdf"),

    # missing / deferred
    Entry(fig="fig3/patchy", src="", dst="perm_importance_eod.pdf",
          note="not in pipeline (predict_action analysis missing)"),
    Entry(fig="fig3/patchy", src="", dst="perm_importance_moveturn.pdf",
          note="not in pipeline (predict_action analysis missing)"),

    # ── FIG 4 / sensor — sensor ablation intervention ─────────────────────────

    Entry(fig="fig4/sensor_1fish",
          src="multi_eval/interventions/sensor_1fish/food_eaten_dunnett.pdf",
          dst="food_eaten_dunnett.pdf"),

    Entry(fig="fig4/sensor",
          src="multi_eval/interventions/sensor/food_eaten_dunnett.pdf",
          dst="food_eaten_dunnett.pdf"),

    Entry(fig="fig4/sensor",
          src="multi_eval/interventions/sensor/time_to_consumption_timecourse_linear.pdf",
          dst="time_to_consumption_timecourse_linear.pdf"),

    Entry(fig="fig4/sensor",
          src="multi_eval/interventions/sensor/num_biting_events_dunnett.pdf",
          dst="num_biting_events_dunnett.pdf"),

    Entry(fig="fig4/sensor",
          src="multi_eval/interventions/sensor/mean_nn_distance_cm_dunnett.pdf",
          dst="mean_nn_distance_cm_dunnett.pdf"),

    Entry(fig="fig4/sensor",
          src="multi_eval/interventions/sensor/p_emit_eod_dunnett.pdf",
          dst="p_emit_eod_dunnett.pdf"),

    Entry(fig="fig4/sensor",
          src="multi_eval/interventions/sensor/food_eaten_theil_dunnett.pdf",
          dst="food_eaten_theil_dunnett.pdf"),

    Entry(fig="fig4/sensor",
          src="multi_eval/interventions/sensor/polarization_dunnett.pdf",
          dst="polarization_dunnett.pdf"),

    # ── FIG 4 / muting ────────────────────────────────────────────────────────

    Entry(fig="fig4/muting",
          src="multi_eval/interventions/muting/food_eaten_dunnett.pdf",
          dst="food_eaten_dunnett.pdf"),

    Entry(fig="fig4/muting",
          src="multi_eval/interventions/muting/num_biting_events_dunnett.pdf",
          dst="num_biting_events_dunnett.pdf"),

    Entry(fig="fig4/muting",
          src="multi_eval/interventions/muting/food_eaten_theil_dunnett.pdf",
          dst="food_eaten_theil_dunnett.pdf"),

    Entry(fig="fig4/muting",
          src="multi_eval/interventions/muting/mean_nn_distance_cm_dunnett.pdf",
          dst="mean_nn_distance_cm_dunnett.pdf"),

    # per-trial Δ profile across muted-sensor conditions (referenced in FIG 4)
    Entry(fig="fig4/muting",
          src="multi_eval/interventions/muting/muting_delta_profile.pdf",
          dst="muting_delta_profile.pdf"),

    # ── FIG 4 / collective_sensing ────────────────────────────────────────────

    Entry(fig="fig4/collective_sensing",
          src="multi_eval/interventions/collective_sensing/food_eaten_dunnett.pdf",
          dst="food_eaten_dunnett.pdf"),

    Entry(fig="fig4/collective_sensing",
          src="multi_eval/interventions/collective_sensing/p_emit_eod_dunnett.pdf",
          dst="p_emit_eod_dunnett.pdf"),

    Entry(fig="fig4/collective_sensing",
          src="multi_eval/interventions/collective_sensing/mean_nn_distance_cm_dunnett.pdf",
          dst="mean_nn_distance_cm_dunnett.pdf"),

    Entry(fig="fig4/collective_sensing",
          src="multi_eval/interventions/collective_sensing/time_to_bitten_linear.pdf",
          dst="time_to_bitten_linear.pdf"),

    # ── FIG 4 — competition (food_grid_iso multi-spec analysis) ──────────────

    Entry(fig="fig4/food_grid_iso",
          src="multi_eval/food_grid_iso/comparison/competition.pdf",
          dst="competition.pdf"),

    Entry(fig="fig4/food_grid_iso",
          src="multi_eval/food_grid_iso/comparison/food_eaten_theil.pdf",
          dst="food_eaten_theil.pdf"),

    # ── FIG 4 — food_abundance extras ────────────────────────────────────────

    Entry(fig="fig4/food_abundance",
          src="multi_eval/interventions/food_abundance/food_eaten_dunnett.pdf",
          dst="food_eaten_dunnett.pdf"),

    Entry(fig="fig4/food_abundance",
          src="multi_eval/interventions/food_abundance/time_to_bitten_linear.pdf",
          dst="time_to_bitten_linear.pdf"),

    Entry(fig="fig4/food_abundance",
          src="multi_eval/interventions/food_abundance/p_near_food_dunnett.pdf",
          dst="p_near_food_dunnett.pdf"),

    Entry(fig="fig4/num_patches",
          src="multi_eval/interventions/num_patches/food_eaten_theil_dunnett.pdf",
          dst="food_eaten_theil_dunnett.pdf"),

    Entry(fig="fig4/num_patches",
          src="multi_eval/interventions/num_patches/mean_nn_distance_cm_dunnett.pdf",
          dst="mean_nn_distance_cm_dunnett.pdf"),

    Entry(fig="fig4/uniform_amp",
          src="multi_eval/interventions/uniform_amp/food_eaten_theil_dunnett.pdf",
          dst="food_eaten_theil_dunnett.pdf"),

    # ── FIG 4 EXTRAS — additional interesting intervention panels ─────────────
    # Added so they land in figs/ for selection; not all referenced in figseed.tex yet.

    # food_grid_iso — competition / resource-geometry story (2 strongest panels
    # were previously missed by the copy map)
    Entry(fig="fig4/food_grid_iso",
          src="multi_eval/food_grid_iso/comparison/food_per_fish.pdf",
          dst="food_per_fish.pdf"),
    Entry(fig="fig4/food_grid_iso",
          src="multi_eval/food_grid_iso/comparison/food_ratio_iso_vs_free.pdf",
          dst="food_ratio_iso_vs_free.pdf"),
    Entry(fig="fig4/food_grid_iso",
          src="multi_eval/food_grid_iso/comparison/num_biting_events.pdf",
          dst="num_biting_events.pdf"),
    Entry(fig="fig4/food_grid_iso",
          src="multi_eval/food_grid_iso/comparison/p_emit_eod.pdf",
          dst="p_emit_eod.pdf"),
    Entry(fig="fig4/food_grid_iso",
          src="multi_eval/food_grid_iso/comparison/mean_nn_distance_cm.pdf",
          dst="mean_nn_distance_cm.pdf"),

    # ── FIG 4 / patchy_vs_uniform — resource-geometry comparison ─────────────
    # (analysis_patchy_vs_uniform multi-spec; referenced in FIG 4 row 4)
    Entry(fig="fig4/patchy_vs_uniform",
          src="multi_eval/patchy_vs_uniform/food_eaten.pdf",
          dst="food_eaten.pdf"),
    Entry(fig="fig4/patchy_vs_uniform",
          src="multi_eval/patchy_vs_uniform/food_eaten_theil.pdf",
          dst="food_eaten_theil.pdf"),
    Entry(fig="fig4/patchy_vs_uniform",
          src="multi_eval/patchy_vs_uniform/mean_nn_distance_cm.pdf",
          dst="mean_nn_distance_cm.pdf"),
    Entry(fig="fig4/patchy_vs_uniform",
          src="multi_eval/patchy_vs_uniform/p_emit_eod.pdf",
          dst="p_emit_eod.pdf"),
    Entry(fig="fig4/patchy_vs_uniform",
          src="multi_eval/patchy_vs_uniform/num_biting_events.pdf",
          dst="num_biting_events.pdf"),

    # collective_sensing — interactions collapse with self-image-only (mechanism)
    Entry(fig="fig4/collective_sensing",
          src="multi_eval/interventions/collective_sensing/num_interactions_dunnett.pdf",
          dst="num_interactions_dunnett.pdf"),

    # sensor — per-trial Δ ablation profiles (dense multi-metric supplement)
    Entry(fig="fig4/sensor",
          src="multi_eval/interventions/sensor/ablation_profile_nomorm.pdf",
          dst="ablation_profile_nomorm.pdf"),
    Entry(fig="fig4/sensor",
          src="multi_eval/interventions/sensor/ablation_profile_noamp.pdf",
          dst="ablation_profile_noamp.pdf"),
    Entry(fig="fig4/sensor",
          src="multi_eval/interventions/sensor/ablation_profile_noknollen.pdf",
          dst="ablation_profile_noknollen.pdf"),
    Entry(fig="fig4/sensor",
          src="multi_eval/interventions/sensor/ablation_profile_k_only.pdf",
          dst="ablation_profile_k_only.pdf"),
    Entry(fig="fig4/sensor",
          src="multi_eval/interventions/sensor/ablation_profile_all_off.pdf",
          dst="ablation_profile_all_off.pdf"),

    # sensor_1fish — full social-control panel set (only food_eaten was copied above)
    Entry(fig="fig4/sensor_1fish",
          src="multi_eval/interventions/sensor_1fish/food_eaten_theil_dunnett.pdf",
          dst="food_eaten_theil_dunnett.pdf"),
    Entry(fig="fig4/sensor_1fish",
          src="multi_eval/interventions/sensor_1fish/mean_nn_distance_cm_dunnett.pdf",
          dst="mean_nn_distance_cm_dunnett.pdf"),
    Entry(fig="fig4/sensor_1fish",
          src="multi_eval/interventions/sensor_1fish/p_emit_eod_dunnett.pdf",
          dst="p_emit_eod_dunnett.pdf"),
    Entry(fig="fig4/sensor_1fish",
          src="multi_eval/interventions/sensor_1fish/num_biting_events_dunnett.pdf",
          dst="num_biting_events_dunnett.pdf"),
    Entry(fig="fig4/sensor_1fish",
          src="multi_eval/interventions/sensor_1fish/polarization_dunnett.pdf",
          dst="polarization_dunnett.pdf"),

    # food_abundance — inequality panel (food/p_near/time already copied above)
    Entry(fig="fig4/food_abundance",
          src="multi_eval/interventions/food_abundance/food_eaten_theil_dunnett.pdf",
          dst="food_eaten_theil_dunnett.pdf"),

    # num_patches — food_eaten panel (theil/spacing already copied above)
    Entry(fig="fig4/num_patches",
          src="multi_eval/interventions/num_patches/food_eaten_dunnett.pdf",
          dst="food_eaten_dunnett.pdf"),

    # uniform_amp — full amp-ablation set (only theil was copied above)
    Entry(fig="fig4/uniform_amp",
          src="multi_eval/interventions/uniform_amp/food_eaten_dunnett.pdf",
          dst="food_eaten_dunnett.pdf"),
    Entry(fig="fig4/uniform_amp",
          src="multi_eval/interventions/uniform_amp/mean_nn_distance_cm_dunnett.pdf",
          dst="mean_nn_distance_cm_dunnett.pdf"),
    Entry(fig="fig4/uniform_amp",
          src="multi_eval/interventions/uniform_amp/p_emit_eod_dunnett.pdf",
          dst="p_emit_eod_dunnett.pdf"),
    Entry(fig="fig4/uniform_amp",
          src="multi_eval/interventions/uniform_amp/num_biting_events_dunnett.pdf",
          dst="num_biting_events_dunnett.pdf"),
    Entry(fig="fig4/uniform_amp",
          src="multi_eval/interventions/uniform_amp/polarization_dunnett.pdf",
          dst="polarization_dunnett.pdf"),

    # sensor_food05 — half-food robustness sweep (ampullary effect persists)
    Entry(fig="fig4/sensor_food05",
          src="multi_eval/interventions/sensor_food05/food_eaten_dunnett.pdf",
          dst="food_eaten_dunnett.pdf"),
    Entry(fig="fig4/sensor_food05",
          src="multi_eval/interventions/sensor_food05/mean_nn_distance_cm_dunnett.pdf",
          dst="mean_nn_distance_cm_dunnett.pdf"),
    Entry(fig="fig4/sensor_food05",
          src="multi_eval/interventions/sensor_food05/num_biting_events_dunnett.pdf",
          dst="num_biting_events_dunnett.pdf"),
    Entry(fig="fig4/sensor_food05",
          src="multi_eval/interventions/sensor_food05/p_emit_eod_dunnett.pdf",
          dst="p_emit_eod_dunnett.pdf"),
    Entry(fig="fig4/sensor_food05",
          src="multi_eval/interventions/sensor_food05/polarization_dunnett.pdf",
          dst="polarization_dunnett.pdf"),

    # sensor_food025 — quarter-food robustness sweep (ampullary effect persists)
    Entry(fig="fig4/sensor_food025",
          src="multi_eval/interventions/sensor_food025/food_eaten_dunnett.pdf",
          dst="food_eaten_dunnett.pdf"),
    Entry(fig="fig4/sensor_food025",
          src="multi_eval/interventions/sensor_food025/mean_nn_distance_cm_dunnett.pdf",
          dst="mean_nn_distance_cm_dunnett.pdf"),
    Entry(fig="fig4/sensor_food025",
          src="multi_eval/interventions/sensor_food025/num_biting_events_dunnett.pdf",
          dst="num_biting_events_dunnett.pdf"),
    Entry(fig="fig4/sensor_food025",
          src="multi_eval/interventions/sensor_food025/p_emit_eod_dunnett.pdf",
          dst="p_emit_eod_dunnett.pdf"),
    Entry(fig="fig4/sensor_food025",
          src="multi_eval/interventions/sensor_food025/polarization_dunnett.pdf",
          dst="polarization_dunnett.pdf"),

    # ── FIG 5 / 2f1p — two-fish one-piece foraging ───────────────────────────

    Entry(fig="fig5/2f1p",
          src="multi_eval/2f1p/b_food_social_vs_alone.pdf",
          dst="b_food_social_vs_alone.pdf"),

    Entry(fig="fig5/2f1p",
          src="multi_eval/2f1p/a_food_social_vs_alone.pdf",
          dst="a_food_social_vs_alone.pdf"),

    Entry(fig="fig5/2f1p",
          src="multi_eval/2f1p/b_food_pct_social_vs_alone.pdf",
          dst="b_food_pct_social_vs_alone.pdf"),

    Entry(fig="fig5/2f1p",
          src="multi_eval/2f1p/a_food_pct_social_vs_alone.pdf",
          dst="a_food_pct_social_vs_alone.pdf"),

    Entry(fig="fig5/2f1p",
          src="multi_eval/2f1p/time_to_consumption_timecourse_linear.pdf",
          dst="time_to_consumption_timecourse_linear.pdf"),

    Entry(fig="fig5/2f1p",
          src="multi_eval/2f1p/survival_b_main.pdf",
          dst="survival_b_main.pdf"),

    Entry(fig="fig5/2f1p",
          src="multi_eval/2f1p/time_to_consumption_B_incl_zeros_timecourse_linear.pdf",
          dst="time_to_consumption_B_incl_zeros_timecourse_linear.pdf"),

    Entry(fig="fig5/2f1p",
          src="multi_eval/2f1p/time_to_consumption_A_incl_zeros_timecourse_linear.pdf",
          dst="time_to_consumption_A_incl_zeros_timecourse_linear.pdf"),

    Entry(fig="fig5/2f1p",
          src="multi_eval/2f1p/b_eats_pct.pdf",
          dst="b_eats_pct.pdf"),

    Entry(fig="fig5/2f1p",
          src="multi_eval/2f1p/nn_distance_by_condition.pdf",
          dst="nn_distance_by_condition.pdf"),

    Entry(fig="fig5/2f1p",
          src="multi_eval/2f1p/peod_by_role_condition.pdf",
          dst="peod_by_role_condition.pdf"),

    Entry(fig="fig5/2f1p",
          src="multi_eval/2f1p/time_to_first_food_b_eaters_stats_main.pdf",
          dst="time_to_first_food_b_eaters_stats_main.pdf"),

    Entry(fig="fig5/2f1p",
          src="multi_eval/2f1p/trajectories/2f1p_AeqB/trajectory_env*.pdf",
          dst="trajectory_AeqB_sample.pdf"),

    # ── FIG 5 / 1rw1f1p — robot-walker one-fish one-piece ───────────────────

    Entry(fig="fig5/1rw1f1p",
          src="evals/1rw1f1p_grid/analyses/1f1rw1p/grouped_by_size_all_rates.pdf",
          dst="grouped_by_size_all_rates.pdf"),

    Entry(fig="fig5/1rw1f1p",
          src="evals/1rw1f1p_grid/analyses/1f1rw1p/grouped_by_rate_aggregated.pdf",
          dst="grouped_by_rate_aggregated.pdf"),

    Entry(fig="fig5/1rw1f1p",
          src="evals/1rw1f1p_grid/analyses/1f1rw1p/frozen_vs_moving.pdf",
          dst="frozen_vs_moving.pdf"),

    Entry(fig="fig5/1rw1f1p",
          src="evals/1rw1f1p_grid/analyses/1f1rw1p/b_eats_pct_heatmap.pdf",
          dst="b_eats_pct_heatmap.pdf"),

    Entry(fig="fig5/1rw1f1p",
          src="evals/1rw1f1p_grid/analyses/1f1rw1p/grouped_by_rate_frozen.pdf",
          dst="grouped_by_rate_frozen.pdf"),

    Entry(fig="fig5/1rw1f1p",
          src="evals/1rw1f1p_grid/analyses/1f1rw1p/peod_vs_bot_rate.pdf",
          dst="peod_vs_bot_rate.pdf"),

    Entry(fig="fig5/1rw1f1p",
          src="evals/1rw1f1p_grid/analyses/1f1rw1p/nn_dist_vs_bot_rate.pdf",
          dst="nn_dist_vs_bot_rate.pdf"),

    Entry(fig="fig5/1rw1f1p",
          src="evals/1rw1f1p_grid/analyses/1f1rw1p/food_vs_eod_rate.pdf",
          dst="food_vs_eod_rate.pdf"),

    Entry(fig="fig5/1rw1f1p",
          src="evals/1rw1f1p_grid/analyses/1f1rw1p/time_to_consumption_linear.pdf",
          dst="time_to_consumption_linear.pdf"),

    # ── FIG 5 / pairwise — 2fish_m1a1k1_uniform_square ───────────────────────

    Entry(fig="fig5/pairwise",
          src="evals/2fish_m1a1k1_uniform_square/analyses/pairwise/chaser_size_by_role_boxplot.pdf",
          dst="chaser_size_by_role_boxplot.pdf"),

    Entry(fig="fig5/pairwise",
          src="evals/2fish_m1a1k1_uniform_square/analyses/pairwise/chaser_size_by_role_hist2d.pdf",
          dst="chaser_size_by_role_hist2d.pdf"),

    Entry(fig="fig5/pairwise",
          src="evals/2fish_m1a1k1_uniform_square/analyses/pairwise/biter_size_advantage.pdf",
          dst="biter_size_advantage.pdf"),

    Entry(fig="fig5/pairwise",
          src="evals/2fish_m1a1k1_uniform_square/analyses/pairwise/biter_size_by_role_boxplot.pdf",
          dst="biter_size_by_role_boxplot.pdf"),

    Entry(fig="fig5/pairwise",
          src="evals/2fish_m1a1k1_uniform_square/analyses/pairwise/biting_timing_in_interaction.pdf",
          dst="biting_timing_in_interaction.pdf"),

    Entry(fig="fig5/pairwise",
          src="evals/2fish_m1a1k1_uniform_square/analyses/pairwise/biting_rate_by_interaction_class.pdf",
          dst="biting_rate_by_interaction_class.pdf"),

    Entry(fig="fig5/pairwise",
          src="evals/2fish_m1a1k1_uniform_square/analyses/pairwise/confrontation_size_hist2d.pdf",
          dst="confrontation_size_hist2d.pdf"),

    Entry(fig="fig5/pairwise",
          src="evals/2fish_m1a1k1_uniform_square/analyses/pairwise/interaction_classes_by_distance.pdf",
          dst="interaction_classes_by_distance.pdf"),

    # new % size heatmaps (matching bite_network style) — for manual review
    Entry(fig="fig5/pairwise",
          src="evals/2fish_m1a1k1_uniform_square/analyses/pairwise/confrontation_size_heatmap.pdf",
          dst="confrontation_size_heatmap.pdf"),
    Entry(fig="fig5/pairwise",
          src="evals/2fish_m1a1k1_uniform_square/analyses/pairwise/chaser_size_by_role_heatmap.pdf",
          dst="chaser_size_by_role_heatmap.pdf"),
    Entry(fig="fig5/pairwise",
          src="evals/2fish_m1a1k1_uniform_square/analyses/pairwise/eod_by_interaction_class.pdf",
          dst="eod_by_interaction_class.pdf"),

    # ── FIG 6 / patchy — patchy_square RNN analyses ──────────────────────────

    Entry(fig="fig6/patchy",
          src="evals/m1a1k1_patchy_square/analyses/rnn_dim/rnn_dim_pca_per_episode_cumvar.pdf",
          dst="rnn_dim_pca_per_episode_cumvar.pdf"),

    # ── FIG 6 / nfish — collective-dim analyses ───────────────────────────────

    Entry(fig="fig6/nfish",
          src="multi_eval/nfish/comparison_deff_boxplot.pdf",
          dst="comparison_deff_boxplot.pdf"),

    # mean ± SEM alternative to comparison_deff_boxplot
    Entry(fig="fig6/nfish",
          src="multi_eval/nfish/comparison_deff_mean_sem.pdf",
          dst="comparison_deff_mean_sem.pdf"),

    Entry(fig="fig6/nfish",
          src="multi_eval/nfish/k1_deff_unit_subsampling.pdf",
          dst="k1_deff_unit_subsampling.pdf"),

    # referenced from FIG 4 block in figseed.tex (shared across figs)
    Entry(fig="fig6/nfish",
          src="multi_eval/nfish/comparison_interagent_dist_boxplot.pdf",
          dst="comparison_interagent_dist_boxplot.pdf"),

    # mean ± SEM alternative to comparison_interagent_dist_boxplot
    Entry(fig="fig6/nfish",
          src="multi_eval/nfish/comparison_interagent_dist_mean_sem.pdf",
          dst="comparison_interagent_dist_mean_sem.pdf"),

    Entry(fig="fig6/nfish",
          src="multi_eval/nfish/comparison_D_eff_vs_nagents.pdf",
          dst="comparison_D_eff_vs_nagents.pdf"),

    Entry(fig="fig6/nfish",
          src="multi_eval/nfish/k1_interagent_dist_boxplot.pdf",
          dst="k1_interagent_dist_boxplot.pdf"),

    # mean ± SD variants (alongside the mean ± SEM / boxplot versions above)
    Entry(fig="fig6/nfish",
          src="multi_eval/nfish/comparison_deff_mean_sd.pdf",
          dst="comparison_deff_mean_sd.pdf"),

    Entry(fig="fig6/nfish",
          src="multi_eval/nfish/comparison_interagent_dist_mean_sd.pdf",
          dst="comparison_interagent_dist_mean_sd.pdf"),

    # per-nagent cumulative-variance D_eff curves
    Entry(fig="fig6/nfish",
          src="multi_eval/nfish/comparison_cumvar_nagents1_D_eff.pdf",
          dst="comparison_cumvar_nagents1_D_eff.pdf"),
    Entry(fig="fig6/nfish",
          src="multi_eval/nfish/comparison_cumvar_nagents2_D_eff.pdf",
          dst="comparison_cumvar_nagents2_D_eff.pdf"),
    Entry(fig="fig6/nfish",
          src="multi_eval/nfish/comparison_cumvar_nagents3_D_eff.pdf",
          dst="comparison_cumvar_nagents3_D_eff.pdf"),
    Entry(fig="fig6/nfish",
          src="multi_eval/nfish/comparison_cumvar_nagents4_D_eff.pdf",
          dst="comparison_cumvar_nagents4_D_eff.pdf"),

    # ── FIG 6 / 2fishwide — uniform_wide RNN analyses ────────────────────────

    Entry(fig="fig6/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/rnn_decoding/rnn_decoding_split.pdf",
          dst="rnn_decoding_split.pdf"),

    Entry(fig="fig6/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/rnn_decoding/rnn_decoding_combined.pdf",
          dst="rnn_decoding_combined.pdf"),

    Entry(fig="fig6/2fishwide",
          src="evals/2fish_m1a1k1_uniform_square/analyses/rnn_psd/rnn_psd_power_ts_vs_dist_agent.pdf",
          dst="rnn_psd_power_ts_vs_dist_agent.pdf"),

    Entry(fig="fig6/2fishwide",
          src="evals/2fish_m1a1k1_uniform_square/analyses/rnn_psd/rnn_psd_power_ts_vs_nearby.pdf",
          dst="rnn_psd_power_ts_vs_nearby.pdf"),

    Entry(fig="fig6/2fishwide",
          src="evals/2fish_m1a1k1_uniform_square/analyses/rnn_psd/rnn_psd_power_ts_vs_dominant.pdf",
          dst="rnn_psd_power_ts_vs_dominant.pdf"),

    Entry(fig="fig6/2fishwide",
          src="evals/2fish_m1a1k1_uniform_square/analyses/rnn_psd/rnn_psd_peri_eating.pdf",
          dst="rnn_psd_peri_eating.pdf"),

    Entry(fig="fig6/2fishwide",
          src="evals/2fish_m1a1k1_uniform_square/analyses/rnn_psd/rnn_psd_peri_was_bitten.pdf",
          dst="rnn_psd_peri_was_bitten.pdf"),

    # ── FIG 6 / 2fish — uniform_square RNN analyses ──────────────────────────

    Entry(fig="fig6/2fishsquare",
          src="evals/2fish_m1a1k1_uniform_square/analyses/rnn_plsc/rnn_plsc_plsc1_by_range.pdf",
          dst="rnn_plsc_plsc1_by_range.pdf"),

    Entry(fig="fig6/2fishsquare",
          src="evals/2fish_m1a1k1_uniform_square/analyses/rnn_plsc/rnn_plsc_plsc_sig_dims.pdf",
          dst="rnn_plsc_plsc_sig_dims.pdf"),

    Entry(fig="fig6/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/rnn_plsc/rnn_plsc_plsc1_by_range.pdf",
          dst="rnn_plsc_plsc1_by_range.pdf"),

    Entry(fig="fig6/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/rnn_dim/rnn_dim_deff_by_range.pdf",
          dst="rnn_dim_deff_by_range.pdf"),

    Entry(fig="fig6/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/rnn_plsc/rnn_plsc_plsc_sig_dims.pdf",
          dst="rnn_plsc_plsc_sig_dims.pdf"),

    # new D_eff-by-condition plots (rnn_dim) + rnn_plsc D_eff-by-range — for manual review
    Entry(fig="fig6/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/rnn_dim/rnn_dim_deff_by_is_dominant.pdf",
          dst="rnn_dim_deff_by_is_dominant.pdf"),
    Entry(fig="fig6/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/rnn_dim/rnn_dim_deff_by_has_nearby.pdf",
          dst="rnn_dim_deff_by_has_nearby.pdf"),
    Entry(fig="fig6/2fishwide",
          src="evals/2fish_m1a1k1_uniform_wide/analyses/rnn_dim/rnn_dim_deff_by_was_bitten.pdf",
          dst="rnn_dim_deff_by_was_bitten.pdf"),
    Entry(fig="fig6/2fishsquare",
          src="evals/2fish_m1a1k1_uniform_square/analyses/rnn_dim/rnn_dim_deff_by_range.pdf",
          dst="rnn_dim_deff_by_range.pdf"),
    Entry(fig="fig6/2fishsquare",
          src="evals/2fish_m1a1k1_uniform_square/analyses/rnn_dim/rnn_dim_deff_by_is_dominant.pdf",
          dst="rnn_dim_deff_by_is_dominant.pdf"),
    Entry(fig="fig6/2fishsquare",
          src="evals/2fish_m1a1k1_uniform_square/analyses/rnn_dim/rnn_dim_deff_by_has_nearby.pdf",
          dst="rnn_dim_deff_by_has_nearby.pdf"),
    Entry(fig="fig6/2fishsquare",
          src="evals/2fish_m1a1k1_uniform_square/analyses/rnn_dim/rnn_dim_deff_by_was_bitten.pdf",
          dst="rnn_dim_deff_by_was_bitten.pdf"),
    # homing decoding — not yet in pipeline
    Entry(fig="fig6", src="", dst="homing_knollen_decoding_perf.pdf",
          note="not in pipeline (homing decoding analysis missing)"),
]


# ─────────────────────────────────────────────────────────────────────────────

def _resolve_src(run_dir: Path, src_glob: str) -> Path | None:
    """Return first alphabetical match for src_glob under run_dir, or None."""
    matches = sorted(run_dir.glob(src_glob))
    return matches[0] if matches else None


def copy_figures(
    run_dir: Path,
    *,
    figs_filter: list[str] | None,
    dry_run: bool,
    force: bool,
) -> tuple[int, int, int, dict]:
    """Copy all mapped figures.  Returns (copied, missing, skipped, report_data)."""
    copied = missing = skipped = 0
    report: dict = {"copied": [], "missing": [], "skipped": []}

    for e in ENTRIES:
        # figs_filter matches the top-level fig number, e.g. "fig3" matches "fig3/patchy"
        if figs_filter and not any(e.fig == f or e.fig.startswith(f + "/") for f in figs_filter):
            continue

        dst_dir = run_dir / "figs" / e.fig
        dst = dst_dir / e.dst
        tag = f"[{e.fig}]"

        # Entries with a note are known gaps — warn and skip
        if e.note:
            print(f"  MISSING {tag} {e.dst}  <- {e.note}")
            missing += 1
            report["missing"].append({"fig": e.fig, "dst": e.dst, "reason": e.note})
            continue

        src = _resolve_src(run_dir, e.src)
        if src is None:
            reason = f"no match for: {e.src}"
            print(f"  MISSING {tag} {e.dst}  <- {reason}")
            missing += 1
            report["missing"].append({"fig": e.fig, "dst": e.dst, "reason": reason})
            continue

        if dst.exists() and not force:
            skipped += 1
            report["skipped"].append({"fig": e.fig, "dst": e.dst})
            continue

        src_rel = str(src.relative_to(run_dir))
        if dry_run:
            print(f"  DRY     {tag} {src_rel}  ->  figs/{e.fig}/{e.dst}")
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  COPY    {tag} {src_rel}  ->  figs/{e.fig}/{e.dst}")
        copied += 1
        report["copied"].append({"fig": e.fig, "src": src_rel, "dst": e.dst})

    return copied, missing, skipped, report


def copy_train_supplement(
    run_dir: Path,
    *,
    dry_run: bool,
    force: bool,
) -> tuple[int, int]:
    """Copy logs/*.pdf → supp/train/.  Returns (copied, skipped)."""
    copied = skipped = 0
    dst_dir = run_dir / "figs" / "supp" / "train"
    for src in sorted((run_dir / "logs").glob("*.pdf")):
        dst = dst_dir / src.name
        if dst.exists() and not force:
            skipped += 1
            continue
        if dry_run:
            print(f"  DRY     [supp/train] logs/{src.name}  ->  supp/train/{src.name}")
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  COPY    [supp/train] logs/{src.name}  ->  supp/train/{src.name}")
        copied += 1
    return copied, skipped


def copy_trajectory_tree(
    run_dir: Path,
    *,
    dry_run: bool,
    force: bool,
) -> tuple[int, int]:
    """Copy multi_eval/2f1p/trajectories/**/*.pdf → figs/fig5/2f1p/trajectories/, preserving subdirs."""
    src_root = run_dir / "multi_eval" / "2f1p" / "trajectories"
    dst_root = run_dir / "figs" / "fig5" / "2f1p" / "trajectories"
    copied = skipped = 0
    if not src_root.is_dir():
        return copied, skipped
    for src in sorted(src_root.rglob("*.pdf")):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        if dst.exists() and not force:
            skipped += 1
            continue
        tag = f"fig5/2f1p/trajectories/{rel}"
        if dry_run:
            print(f"  DRY     [{tag}]")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  COPY    [fig5/2f1p/trajectories] {rel}")
        copied += 1
    return copied, skipped


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_dir", help="Path to a timestamped training run directory")
    p.add_argument("--dry-run", action="store_true", help="Show what would be copied")
    p.add_argument("--force", action="store_true", help="Overwrite existing files")
    p.add_argument(
        "--figs", nargs="+", metavar="FIG",
        help="Limit to these figure numbers, e.g. --figs fig3 fig6",
    )
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        sys.exit(f"ERROR: run_dir not found: {run_dir}")

    print(f"run_dir : {run_dir}")
    print(f"output  : {run_dir}/figs/")
    if args.dry_run:
        print("(dry run -- nothing will be written)\n")

    copied, missing, skipped, report_data = copy_figures(
        run_dir,
        figs_filter=args.figs,
        dry_run=args.dry_run,
        force=args.force,
    )

    t_copied, t_skipped = copy_train_supplement(
        run_dir, dry_run=args.dry_run, force=args.force
    )
    tr_copied, tr_skipped = copy_trajectory_tree(
        run_dir, dry_run=args.dry_run, force=args.force
    )

    label = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{label}Done.  copied={copied}  missing={missing}  skipped={skipped}"
          f"  supp/train: copied={t_copied}  skipped={t_skipped}"
          f"  trajectories: copied={tr_copied}  skipped={tr_skipped}")

    if not args.dry_run:
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "summary": {"copied": copied, "missing": missing, "skipped": skipped},
            **report_data,
        }
        report_path = run_dir / "figs" / "copy_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        print(f"Report  : {report_path}")


if __name__ == "__main__":
    main()
