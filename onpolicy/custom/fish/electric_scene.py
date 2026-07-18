from typing import Dict, Optional
from dataclasses import dataclass, field

import numpy as np
from electric import (
    measure_electric_field,
    measure_electric_potential,
    reflect_sources,
    induce_dipoles,
    to_world_sources,
    to_world_sensors,
    clip_conductor_moments,
    rotate_and_translate,
)


@dataclass
class ElectricSourceSet:
    """A bundle of electric sources. Empty arrays allowed."""
    monopole_pos: np.ndarray  # (N,2)
    monopole_q:   np.ndarray  # (N,)
    dipole_pos:   np.ndarray  # (M,2)
    dipole_p:     np.ndarray  # (M,2)

    @staticmethod
    def empty():
        return ElectricSourceSet(
            monopole_pos=np.zeros((0, 2), dtype=float),
            monopole_q=np.zeros((0,), dtype=float),
            dipole_pos=np.zeros((0, 2), dtype=float),
            dipole_p=np.zeros((0, 2), dtype=float),
        )

    def concat(self, other: "ElectricSourceSet") -> "ElectricSourceSet":
        return ElectricSourceSet(
            monopole_pos=np.concatenate([self.monopole_pos, other.monopole_pos], axis=0),
            monopole_q=np.concatenate([self.monopole_q, other.monopole_q], axis=0),
            dipole_pos=np.concatenate([self.dipole_pos, other.dipole_pos], axis=0),
            dipole_p=np.concatenate([self.dipole_p, other.dipole_p], axis=0),
        )


@dataclass
class ElectricScene:
    """
    Physics-only electric scene for one timestep. *Update every timestep*.

    Responsibilities:
      - Build and cache named ElectricSourceSet objects
      - Build reflected versions of all source sets
      - Answer electric field queries at arbitrary locations

    Non-responsibilities:
      - Sensors
      - Agent logic
      - Calibration / baselines / noise
    """

    arena_size_cm: tuple  # (width_cm, height_cm)
    eps_m: float = 1e-5
    max_charge_allowed: Optional[float] = None

    # populated by update()
    source_sets: Dict[str, ElectricSourceSet] = field(default_factory=dict)
    reflected_source_sets: Dict[str, ElectricSourceSet] = field(default_factory=dict)


    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def update(
        self,
        # fish pose
        fish_positions,            # (F,2)
        fish_orientations,         # (F,)
        fish_eods,                 # (F,)

        # fish EOD sources (ego)
        fish_monopole_positions,   # (F,P,2) or (1,P,2)
        fish_monopole_charges,     # (F,P) or (1,P)
        fish_dipole_positions,     # (F,D,2) or (1,D,2)
        fish_dipole_moments,       # (F,D,2) or (1,D,2)

        # conductors (food)
        conductor_positions,       # (C,2)
        conductor_contrasts,       # (C,) or (1,)
        conductor_radii,           # (C,) or (1,)

        # induction on fish bodies  # TODO sometimes we zero out EOD'ing fish -- make sure this is handled in call in all_agents_sense equivalent
        fish_contrasts=None,       # (F,) or None
        fish_radii=None,           # (F,) or None

        # intrinsic dipoles (in local coords)
        conductor_intrinsic_dipole_moments_ego=None,  # (C,2) or None
        conductor_orientations=None,                    # (C,) or None
        fish_intrinsic_dipole_moments_ego=None,       # (F,2) or None

        # intrinsic dipoles (in world coords)  # TODO move the world coord transformations
        # conductor_intrinsic_dipole_moments=None,  # (C,2) or None
        # fish_intrinsic_dipole_moments=None,       # (F,2) or None

        build_cons_baseline=False,
        induce_with_reflections=False,            # if True, use reflected EOD sources when inducing dipoles
    ):
        """Build all source sets for the current timestep."""

        self.source_sets.clear()
        self.reflected_source_sets.clear()

        # --- EOD sources (fish only, EOD-masked)
        eod_sources = self._build_eod_sources(
            fish_positions,
            fish_orientations,
            fish_monopole_positions,
            fish_monopole_charges,
            fish_dipole_positions,
            fish_dipole_moments,
            fish_eods,
        )
        self.source_sets["eod"] = eod_sources

        # Optionally fold reflected images into the induction calculations (matches electric.py viz)
        induction_sources = (
            self._reflect_set_with_originals(eod_sources) if induce_with_reflections else eod_sources
        )

        # --- Induced dipoles on food
        induced_food = self._build_induced_food_dipoles(
            induction_sources,
            conductor_positions,
            conductor_contrasts,
            conductor_radii,
        )
        self.source_sets["induced_food"] = induced_food

        # --- Induced dipoles on fish bodies
        induced_fish = self._build_induced_fish_dipoles(
            induction_sources,
            fish_positions,
            fish_contrasts,
            fish_radii,
        )
        self.source_sets["induced_fish"] = induced_fish

        # --- Intrinsic dipoles
        intrinsic = self._build_intrinsic_dipoles(
            fish_positions,
            fish_orientations,
            fish_intrinsic_dipole_moments_ego,
            conductor_positions,
            conductor_orientations,
            conductor_intrinsic_dipole_moments_ego,
        )
        self.source_sets["intrinsic"] = intrinsic

        # --- Optional conspecific baseline
        if build_cons_baseline:
            self.source_sets["cons_baseline"] = eod_sources.concat(induced_fish)

        # --- Reflections for *all* sets
        self._build_reflections()

        return self


    def get_field_at_positions(
        self,
        source_names,              # str or list[str]
        positions_world,           # (K,2)
        use_reflections=True,
        measurement_noise=0.0,
    ):
        """Return electric field vectors (K,2)."""

        if isinstance(source_names, str):
            source_names = [source_names]

        combined = ElectricSourceSet.empty()
        source_sets_to_use = [self.source_sets]
        if use_reflections:
            source_sets_to_use.append(self.reflected_source_sets)

        for source_set in source_sets_to_use:
            for name in source_names:
                if name not in source_set:
                    raise ValueError(f"'{name}' not found in ElectricScene source set.")
                combined = combined.concat(source_set[name])

        return measure_electric_field(
            positions_world,
            combined.monopole_pos,
            combined.monopole_q,
            combined.dipole_pos,
            combined.dipole_p,
            measurement_noise=measurement_noise,
            eps_m=self.eps_m,
        )

    def project_field_on_normals(self, field_vectors, normals):
        """Signed scalar projection dot(E, n)."""
        return np.sum(field_vectors * normals, axis=-1)


    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_eod_sources(
        self,
        fish_positions,
        fish_orientations,
        fish_monopole_positions,
        fish_monopole_charges,
        fish_dipole_positions,
        fish_dipole_moments,
        fish_eods,
    ) -> ElectricSourceSet:
        """
        Uses scene_field_generators() to transform fish-local sources into world coords,
        and applies fish_eods as a multiplicative mask (handled inside scene_field_generators).
        """
        (
            fish_eod_monopole_positions_world,
            fish_eod_monopole_charges,
            fish_eod_dipole_positions_world,
            fish_eod_dipole_moments_world,
        ) = to_world_sources(
            fish_positions,
            fish_orientations,
            fish_monopole_positions,
            fish_monopole_charges,
            fish_dipole_positions,
            fish_dipole_moments,
            fish_eods,
        )
        return ElectricSourceSet(
            monopole_pos=fish_eod_monopole_positions_world,
            monopole_q=fish_eod_monopole_charges,
            dipole_pos=fish_eod_dipole_positions_world,
            dipole_p=fish_eod_dipole_moments_world,
        )

    def _build_induced_food_dipoles(
        self,
        eod_sources: ElectricSourceSet,
        conductor_positions,
        conductor_contrasts,
        conductor_radii,
    ) -> ElectricSourceSet:
        """
        Computes induced dipole moments on conductors (food) due to the EOD sources.
        Returns empty if no conductors.
        """
        if conductor_positions is None:
            return ElectricSourceSet.empty()
        if conductor_positions.shape[0] == 0:
            return ElectricSourceSet.empty()

        induced_food_dipole_moments = induce_dipoles(
            monopole_positions_cm=eod_sources.monopole_pos,
            monopole_charges=eod_sources.monopole_q,
            dipole_positions_cm=eod_sources.dipole_pos,
            dipole_moments=eod_sources.dipole_p,
            conductor_positions_cm=conductor_positions,
            conductor_contrasts=conductor_contrasts,
            conductor_radii_cm=conductor_radii,
            measurement_noise=0.0,
            eps_m=self.eps_m,
        )
        return ElectricSourceSet(
            monopole_pos=np.zeros((0, 2), dtype=float),
            monopole_q=np.zeros((0,), dtype=float),
            dipole_pos=conductor_positions,
            dipole_p=induced_food_dipole_moments,
        )

    def _build_induced_fish_dipoles(
        self,
        eod_sources: ElectricSourceSet,
        fish_positions,
        fish_contrasts,
        fish_radii,
    ) -> ElectricSourceSet:
        """
        Computes induced dipole moments on fish bodies due to the EOD sources.
        Returns empty if fish_contrasts/fish_radii not provided or all contrasts are zero.
        Clips moments if max_charge_allowed is set.
        """
        if fish_contrasts is None or fish_radii is None:
            return ElectricSourceSet.empty()

        fish_contrasts_arr = np.asarray(fish_contrasts)
        if not np.any(fish_contrasts_arr != 0):
            return ElectricSourceSet.empty()

        induced_fish_dipole_moments = induce_dipoles(
            monopole_positions_cm=eod_sources.monopole_pos,
            monopole_charges=eod_sources.monopole_q,
            dipole_positions_cm=eod_sources.dipole_pos,
            dipole_moments=eod_sources.dipole_p,
            conductor_positions_cm=fish_positions,
            conductor_contrasts=fish_contrasts,
            conductor_radii_cm=fish_radii,
            measurement_noise=0.0,
            eps_m=self.eps_m,
        )
        induced_fish_dipole_moments = clip_conductor_moments(
            induced_fish_dipole_moments,
            fish_radii,
            max_charge_allowed=self.max_charge_allowed,
        )
        return ElectricSourceSet(
            monopole_pos=np.zeros((0, 2), dtype=float),
            monopole_q=np.zeros((0,), dtype=float),
            dipole_pos=fish_positions,
            dipole_p=induced_fish_dipole_moments,
        )

    def _build_intrinsic_dipoles(
        self,
        fish_positions,
        fish_orientations,
        fish_intrinsic_dipole_moments_ego,  # need to transform to world coords using orientations
        conductor_positions,
        conductor_orientations,
        conductor_intrinsic_dipole_moments_ego,  # need to transform to world coords
    ) -> ElectricSourceSet:
        """
        Bundles intrinsic dipoles (fish + conductors) into a single dipole-only source set.
        Assumes the base dipole moments are given in local coords, and transforms them to world coords.
        Returns empty if none provided.
        """
        dipole_pos_list = []
        dipole_p_list = []

        if (
            conductor_intrinsic_dipole_moments_ego is not None
            and conductor_positions is not None
            and conductor_positions.shape[0] > 0
        ):
            conductor_intrinsic_dipole_moments = rotate_and_translate(
                conductor_intrinsic_dipole_moments_ego,
                conductor_orientations,
            )
            dipole_pos_list.append(conductor_positions)
            dipole_p_list.append(conductor_intrinsic_dipole_moments)

        if fish_intrinsic_dipole_moments_ego is not None:
            fish_intrinsic_dipole_moments = rotate_and_translate(
                fish_intrinsic_dipole_moments_ego,
                fish_orientations,
            )
            dipole_pos_list.append(fish_positions)
            dipole_p_list.append(fish_intrinsic_dipole_moments)

        if len(dipole_pos_list) == 0:
            return ElectricSourceSet.empty()

        return ElectricSourceSet(
            monopole_pos=np.zeros((0, 2), dtype=float),
            monopole_q=np.zeros((0,), dtype=float),
            dipole_pos=np.concatenate(dipole_pos_list, axis=0),
            dipole_p=np.concatenate(dipole_p_list, axis=0),
        )

    def _build_reflections(self) -> None:
        """
        Builds reflected variants for every source set currently in self.source_sets,
        containing *only* the first-order reflections.
        Stored with same name in self.reflected_source_sets.
        """
        for name, sources in self.source_sets.items():
            (
                reflected_monopole_positions,
                reflected_monopole_charges,
                reflected_dipole_positions,
                reflected_dipole_moments,
            ) = reflect_sources(
                monopole_positions_cm=sources.monopole_pos,
                monopole_charges=sources.monopole_q,
                dipole_positions_cm=sources.dipole_pos,
                dipole_moments=sources.dipole_p,
                arena_size_cm=self.arena_size_cm,
                num_reflections=1,
            )

            self.reflected_source_sets[name] = ElectricSourceSet(
                monopole_pos=reflected_monopole_positions,
                monopole_q=reflected_monopole_charges,
                dipole_pos=reflected_dipole_positions,
                dipole_p=reflected_dipole_moments,
            )

    def _reflect_set_with_originals(self, sources: ElectricSourceSet) -> ElectricSourceSet:
        """Return the reflected copy of a source set (includes originals)."""
        (
            reflected_monopole_positions,
            reflected_monopole_charges,
            reflected_dipole_positions,
            reflected_dipole_moments,
        ) = reflect_sources(
            monopole_positions_cm=sources.monopole_pos,
            monopole_charges=sources.monopole_q,
            dipole_positions_cm=sources.dipole_pos,
            dipole_moments=sources.dipole_p,
            arena_size_cm=self.arena_size_cm,
            num_reflections=1,
        )
        return ElectricSourceSet(
        monopole_pos=np.concatenate([sources.monopole_pos, reflected_monopole_positions], axis=0),
        monopole_q=np.concatenate([sources.monopole_q, reflected_monopole_charges], axis=0),
        dipole_pos=np.concatenate([sources.dipole_pos, reflected_dipole_positions], axis=0),
        dipole_p=np.concatenate([sources.dipole_p, reflected_dipole_moments], axis=0),
    )


