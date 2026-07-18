# To generate the figures in the manuscript, run the following command:
# python arena.py # for uniform, patchy, and feeder arenas
# python arena.py --patches_per_edge=10 --reset_food_density=10 --stockpile_density=1.0 # for lattice
# TODO: Aaron, WTF
import math
import io
import argparse
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from PIL import Image
import imageio

import tqdm

from cfg import ENV_PARAMS


class FoodPellet:
    pellets_produced = 0

    def __init__(self, position, parent, drift=0.0, drag=0.2, orientation_angle=None):
        self.position = position
        self.velocity = np.zeros_like(self.position)
        self.parent = parent
        self.drift = drift
        self.drag = drag
        self.global_index = self.__class__.pellets_produced
        self.__class__.pellets_produced += 1

        if orientation_angle is None:
            # Backwards-compatible default
            orientation_angle = 0.0
        self.orientation_angle = float(orientation_angle)


    def __hash__(self):
        return hash(self.global_index)

    def remove(self):
        self.parent.remove_food_pellet(self)

    # def step(self, a):
    #    self.velocity = (1. - self.drag) * (
    #    #    self.velocity + np.random.normal(size=2) * self.drift)
    #        self.velocity + a * self.drift)
    #    self.position = self.position + self.velocity
    #    self.position = np.clip(self.position, [0,0], self.parent.arena_size)


class FoodPatch:
    def __init__(
        self,
        position,
        shape,
        dimensions,
        arena_size,
        reset_food_density,
        step_food_density,
        step_food_decay,
        max_food_density=float("inf"),
        stockpile_density=float("inf"),
        drift=0.0,
        drag=0.2,
        orientation_drift=0.0,
    ):
        self.position = position
        self.shape = shape
        self.dimensions = dimensions
        self.arena_size = arena_size
        self.reset_food_density = reset_food_density
        self.step_food_density = step_food_density
        self.step_food_decay = step_food_decay
        self.max_food_density = max_food_density
        self.stockpile_density = stockpile_density
        self.drift = drift
        self.drag = drag
        self.orientation_drift = orientation_drift
        self.food_pellets = set()

    def sample_positions(self, n):
        if self.shape == "circle":
            r = self.dimensions[0]
            positions = self.np_random.uniform(low=[-r, -r], high=[r, r], size=(n, 2))

            # resample points outside the circle
            n2 = positions[:, 0] ** 2 + positions[:, 1] ** 2
            noncircles = n2 > r**2
            num_noncircles = np.sum(noncircles)
            if num_noncircles:
                resampled_positions = self.sample_positions(num_noncircles)
                positions[noncircles] = resampled_positions
            positions[~noncircles] = positions[~noncircles] + self.position

        elif self.shape == "rectangle":
            positions = (
                self.np_random.uniform(
                    low=[self.dimensions[0], self.dimensions[1]],
                    high=[self.dimensions[2], self.dimensions[3]],
                    size=(n, 2),
                )
                + self.position
            )
        else:
            raise NotImplementedError

        # resample out of bounds
        oob = (
            (positions[:, 0] < 0)
            | (positions[:, 0] >= self.arena_size[0])
            | (positions[:, 1] < 0)
            | (positions[:, 1] >= self.arena_size[1])
        )
        oob_indices = np.where(oob)[0]
        n_oob = oob_indices.shape[0]

        if n_oob:
            resampled_positions = self.sample_positions(n_oob)
            positions[oob_indices] = resampled_positions

        return positions

    @property
    def area(self):
        if self.shape == "circle":
            return np.pi * self.dimensions[0] ** 2
        elif self.shape == "rectangle":
            a = self.dimensions[2] - self.dimensions[0]
            b = self.dimensions[3] - self.dimensions[1]
            return a * b

    @property
    def max_food_in_area(self):
        return int(round(self.max_food_density * self.area))

    def __len__(self):
        return len(self.food_pellets)

    def grow_food_pellets(self, n):
        n = min(n, self.max_food_in_area - len(self), self.stockpile)
        if n <= 0:
            return

        positions = self.sample_positions(n)
        orientation_angles = self.np_random.uniform(0.0, 2.0 * np.pi, size=n)
        for position, orientation_angle in zip(positions, orientation_angles):
            self.food_pellets.add(
                FoodPellet(
                    position,
                    self,
                    drift=self.drift,
                    drag=self.drag,
                    orientation_angle=orientation_angle,
                )
            )
        self.stockpile -= n

    def remove_food_pellet(self, food_pellet):
        self.food_pellets.remove(food_pellet)

    def reset(self, seed=None):

        # setup seeded random number generator
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        elif not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()

        self.food_pellets = set()
        a = self.area
        if self.stockpile_density == float("inf"):
            self.stockpile = float("inf")
        else:
            self.stockpile = int(round(self.stockpile_density * a))
        n = self.np_random.poisson(self.reset_food_density * a)
        self.grow_food_pellets(n)

    def list_food_pellets(self):
        return list(self.food_pellets)

    def step(self):
        n_grow = self.np_random.poisson(self.step_food_density * self.area)
        self.grow_food_pellets(n_grow)

        decay = self.np_random.random(len(self)) < self.step_food_decay
        decay_indices = np.where(decay)[0]
        pellet_list = self.list_food_pellets()
        for i in decay_indices:
            self.remove_food_pellet(pellet_list[i])

        # for pellet in pellet_list:
        #    pellet.step()


class Arena:
    def __init__(
        self,
        min_arena_size,
        max_arena_size,
        reset_food_density=0.01,
        step_food_density=0.001,
        step_food_decay=0.0,
        max_food_density=0.05,
        stockpile_density=float("inf"),
        drift=0.01,
        drag=0.2,
        orientation_drift=0.0,
    ):
        self.min_arena_size = min_arena_size
        self.max_arena_size = max_arena_size
        self.reset_food_density = reset_food_density
        self.step_food_density = step_food_density
        self.step_food_decay = step_food_decay
        self.max_food_density = max_food_density
        self.stockpile_density = stockpile_density
        self.food_radius = ENV_PARAMS["food_radius_cm"] # cm
        self.drift = drift
        self.drag = drag
        self.orientation_drift = orientation_drift

    @property
    def area(self):
        return self.arena_size[0] * self.arena_size[1]

    @property
    def num_patches(self):
        return len(self.patches)

    @property
    def food_pellets(self):
        return sum((patch.list_food_pellets() for patch in self.patches), [])

    @property
    def num_food_pellets(self):
        return len(self.food_pellets)

    @property
    def food_positions(self):
        pellets = self.food_pellets
        return np.stack([p.position for p in pellets]) if pellets else np.zeros((0, 2))

    @property
    def food_orientation_angles(self):
        """Returns (N,) array of orientation angles (radians) for each food pellet."""
        pellets = self.food_pellets
        return (
            np.array([p.orientation_angle for p in pellets])
            if pellets
            else np.zeros((0,), dtype=float)
        )

    @property
    def food_velocities(self):
        pellets = self.food_pellets
        return np.stack([p.velocity for p in pellets]) if pellets else np.zeros((0, 2))

    def remove_food_pellet(self, index):
        self.food_pellets[index].remove()

    def eat_food(self, *indices):
        for index in sorted(indices, reverse=True):
            try:
                self.remove_food_pellet(index)
            except Exception as e:
                import warnings
                warnings.warn(f"eat_food: failed to remove pellet at index {index}: {e}")

    def prune_unused_patches(self):
        # self.patches = [
        #     patch for patch in self.patches if len(patch) or patch.stockpile
        # ]

        # keep empty patches
        self.patches = [patch for patch in self.patches if patch.stockpile]

    def reset(self, seed=None):
        # Reset the global pellet counter to ensure consistent indexing/hashing
        # NOTE: Makes arena_randomness_test pass
        FoodPellet.pellets_produced = 0

        # setup seeded random number generator
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        elif not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()

        self.arena_size = self.np_random.uniform(
            low=self.min_arena_size,
            high=self.max_arena_size,
        )
        self.reset_num_patches()
        for patch in self.patches:
            patch_seed = self.np_random.integers(10e8)
            patch.reset(seed=patch_seed)
        self.prune_unused_patches()

    def step(self):
        self.step_num_patches()
        self.prune_unused_patches()
        for patch in self.patches:
            patch.step()

        food_pellets = self.food_pellets
        if not food_pellets:
            return
        food_positions = np.stack([p.position for p in food_pellets])
        food_velocities = np.stack([p.velocity for p in food_pellets])
        food_accelerations = self.np_random.standard_normal(
            size=(len(food_pellets), 2)
        )
        food_velocities *= 1.0 - self.drag
        food_velocities += food_accelerations * self.drift
        food_positions += food_velocities
        food_positions = np.clip(food_positions, [0, 0], self.arena_size)

        if self.orientation_drift:
            orientation_shifts = self.np_random.normal(
                0.0,
                self.orientation_drift,
                size=len(food_pellets),
            )
        else:
            orientation_shifts = np.zeros(len(food_pellets))

        for pellet, p, v, dtheta in zip(
            food_pellets, food_positions, food_velocities, orientation_shifts
        ):
            pellet.position = p
            pellet.velocity = v
            pellet.orientation_angle = (pellet.orientation_angle + dtheta) % (2.0 * np.pi)

    @property
    def patch_kwargs(self):
        return {
            "arena_size": self.arena_size,
            "reset_food_density": self.reset_food_density,
            "step_food_density": self.step_food_density,
            "step_food_decay": self.step_food_decay,
            "max_food_density": self.max_food_density,
            "stockpile_density": self.stockpile_density,
            "drift": self.drift,
            "drag": self.drag,
            "orientation_drift": self.orientation_drift,
        }

    def render(self, path=None, show=False, **kwargs):
        fig, ax = plt.subplots()
        ax.set_xlim(-5, self.arena_size[0] + 5)
        ax.set_ylim(-5, self.arena_size[1] + 5)
        ax.set_aspect("equal")

        arena_boundary = plt.Rectangle(
            (0, 0),
            self.arena_size[0],
            self.arena_size[1],
            edgecolor="gray",
            fill=False,
            linewidth=1,
        )
        ax.add_patch(arena_boundary)

        for patch in self.patches:
            if patch.shape == "circle":
                patch_circle = plt.Circle(
                    patch.position,
                    patch.dimensions[0],
                    color="lightgreen",
                    alpha=0.2,
                )
                ax.add_patch(patch_circle)

        ax.plot(
            self.food_positions[:, 0],
            self.food_positions[:, 1],
            "go",
            markersize=2 if not "markersize" in kwargs else kwargs["markersize"],
        )

        if "manuscript" in kwargs and kwargs["manuscript"]:
            plt.xlim(0, self.arena_size[0])
            plt.ylim(0, self.arena_size[1])
            plt.xticks([])
            plt.yticks([])

        if path:
            plt.savefig(path)
            print(f"Saved arena visualization to {path}")
        if show:
            plt.show()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        frame = np.array(image)
        buf.close()
        plt.close()

        return frame

    def reset_num_patches(self):
        self.patches = []

    def step_num_patches(self):
        pass


class PatchyArena(Arena):
    def __init__(
        self,
        *args,
        patch_d_mean,
        patch_d_var,
        reset_patch_density,
        step_patch_density=0.0,
        max_patch_density=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs) # Needs reset_patch_density to be defined
        self.patch_d_mean = patch_d_mean
        self.patch_d_var = patch_d_var
        self.reset_patch_density = reset_patch_density
        self.step_patch_density = step_patch_density
        self.max_patch_density = max_patch_density  # None --> unlimited

    def add_new_patches(self, n):
        # Enforce maximum number of patches
        if self.max_patches is not None:
            n = min(n, self.max_patches - len(self.patches))
        if n <= 0:
            return []

        new_patches = []
        for i in range(n):
            position = self.np_random.uniform(
                low=(0, 0),
                high=self.arena_size,
            )
            radius = self.np_random.normal(
                loc=self.patch_d_mean / 2, scale=self.patch_d_var / 2
            )
            if radius < 1:
                print(f"Warning: patch radius {radius} is below the minimum threshold, resetting to 1")
            radius = np.clip(radius, 1, min(self.arena_size) / 2) # Safeguard 
            new_patches.append(
                FoodPatch(position, "circle", [radius], **self.patch_kwargs)
            )
        self.patches.extend(new_patches)
        return new_patches

    def reset_num_patches(self):
        self.patches = []
        # n = self.np_random.poisson(self.reset_patch_density * self.area)
        # n = max(1, n)
        n = max(math.ceil(self.reset_patch_density * self.area), 1)
        self.add_new_patches(n)

    def step_num_patches(self):
        n = self.np_random.poisson(self.step_patch_density * self.area)
        new_patches = self.add_new_patches(n)
        for patch in new_patches:
            patch.reset()

    @property
    def max_patches(self):
        """
        Absolute cap on the number of patches in the arena, computed from
        max_patch_density * area. Returns None if unlimited.
        """
        if self.max_patch_density is None:
            return None
        return max(1, int(round(self.max_patch_density * self.area)))


class UniformArena(Arena):
    def reset_num_patches(self):
        self.patches = [
            FoodPatch(
                position=[0, 0],
                shape="rectangle",
                dimensions=[0, 0, *self.arena_size],
                **self.patch_kwargs,
            )
        ]

    def prune_unused_patches(self):
        pass


class LatticeArena(Arena):
    def __init__(
        self,
        *args,
        patches_per_edge,
        tightness=0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patches_per_edge = patches_per_edge
        self.tightness = tightness

    def reset_num_patches(self):
        self.patches = []
        patch_width = self.arena_size[0] / self.patches_per_edge
        patch_height = self.arena_size[1] / self.patches_per_edge

        for ix in range(self.patches_per_edge):
            for iy in range(self.patches_per_edge):
                position_x = (ix + 0.5) * patch_width
                position_y = (iy + 0.5) * patch_height
                patch_dimensions = [0, 0, self.tightness, self.tightness]

                self.patches.append(
                    FoodPatch(
                        position=[position_x, position_y],
                        shape="rectangle",
                        dimensions=patch_dimensions,
                        **self.patch_kwargs,
                    )
                )

    @property
    def patch_kwargs(self):
        return {
            "arena_size": self.arena_size,
            "reset_food_density": self.reset_food_density / self.tightness**2,
            "step_food_density": self.step_food_density / self.tightness**2,
            "step_food_decay": self.step_food_decay,
            "max_food_density": 1.0 / self.tightness**2,
            "stockpile_density": self.stockpile_density / self.tightness**2,
            "drift": self.drift,
            "drag": self.drag,
            "orientation_drift": self.orientation_drift,
        }

    def step_num_patches(self):
        # This could be left empty if no dynamic changes are desired in the patches.
        pass

# NOTE: Mostly deprecated in favor of NPatchArena
class FeederArena(Arena):
    def __init__(
        self,
        *args,
        patch_d_mean,
        patch_d_var,
        patches_per_edge,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patch_d_mean = patch_d_mean
        self.patch_d_var = patch_d_var
        self.patches_per_edge = patches_per_edge

    def prune_unused_patches(self):
        pass

    def reset_num_patches(self):
        self.patches = []
        min_edge_length = min(self.arena_size)
        width, height = self.arena_size

        x_positions = np.linspace(
            0.5 * width / self.patches_per_edge,
            width - 0.5 * width / self.patches_per_edge,
            self.patches_per_edge,
        )
        y_positions = np.linspace(
            0.5 * height / self.patches_per_edge,
            height - 0.5 * height / self.patches_per_edge,
            self.patches_per_edge,
        )

        centers = np.concatenate(
            [
                np.stack([x_positions, np.zeros_like(x_positions)], axis=1),
                np.stack([np.full_like(y_positions, width), y_positions], axis=1),
                np.stack([x_positions, np.full_like(x_positions, height)], axis=1),
                np.stack([np.zeros_like(y_positions), y_positions], axis=1),
            ]
        )

        for center in centers:
            radius = np.clip(
                # self.np_random.standard_normal(loc=self.patch_d_mean / 2, scale=self.patch_d_var / 2),
                self.np_random.standard_normal() * self.patch_d_var / 2.0
                + self.patch_d_mean / 2.0,
                0.1,
                min_edge_length / 2,
            )
            self.patches.append(
                FoodPatch(
                    position=center,
                    shape="circle",
                    dimensions=[radius],
                    **self.patch_kwargs,
                )
            )


class OnePatchArena(Arena):
    def __init__(
        self,
        *args,
        patch_d_mean,
        patch_d_var,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patch_d_mean = patch_d_mean
        self.patch_d_var = patch_d_var

    def reset_num_patches(self):
        center = np.array(self.arena_size) / 2
        radius = self.np_random.normal(
            loc=self.patch_d_mean / 2, scale=self.patch_d_var / 2
        )
        radius = np.clip(radius, 0.1, min(self.arena_size) / 2)
        self.patches = [
            FoodPatch(
                position=center,
                shape="circle",
                dimensions=[radius],
                **self.patch_kwargs,
            )
        ]

    def prune_unused_patches(self):
        pass

class SquareQuadrantArena(Arena):
    def __init__(self, *args,
                 num_radial_patches=4,
                 patch_radius=7.5,
                 center_location="bottom_right",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_radial_patches = num_radial_patches
        self.patch_radius = patch_radius
        self.center_location = center_location

    def reset_num_patches(self):
        self.patches = []
        w, h = self.arena_size

        center_map = {
            "bottom_left": np.array([0.0 + self.patch_radius, 0.0 + self.patch_radius]),
            "bottom_right": np.array([w - self.patch_radius, 0.0 + self.patch_radius]),
            "top_left": np.array([0.0 + self.patch_radius, h - self.patch_radius]),
            "top_right": np.array([w - self.patch_radius, h - self.patch_radius]),
        }
        angle_ranges = {
            "bottom_left": (0.0, np.pi / 2),
            "bottom_right": (np.pi / 2, np.pi),
            "top_left": (-np.pi / 2, 0.0),
            "top_right": (np.pi, 3 * np.pi / 2),
        }

        center = center_map[self.center_location]
        theta_start, theta_end = angle_ranges[self.center_location]
        max_radius = min(w, h) / 1.2  # radius of circular arc

        thetas = np.linspace(theta_start, theta_end, self.num_radial_patches)
        for theta in thetas:
            offset = np.array([np.cos(theta), np.sin(theta)]) * max_radius
            position = center + offset
            position = np.clip(position, 
                               [self.patch_radius, self.patch_radius], 
                               [w - self.patch_radius, h - self.patch_radius])  # ensure inside bounds

            self.patches.append(
                FoodPatch(
                    position=position,
                    shape="circle",
                    dimensions=[self.patch_radius],
                    **self.patch_kwargs,
                )
            )


class NPatchArena(Arena):
    def __init__(self, *args,
                 fixed_num_patches=1,
                 placement_radius_frac=1.0,
                 patch_radius=7.5,
                 **kwargs):
        """
        placement_radius_frac: float in [0,1], fraction of half‐width/half‐height
            to use for the ellipse on which to place patch centers.
        patch_radius: radius of each patch
        """
        super().__init__(*args, **kwargs)
        self.fixed_num_patches = fixed_num_patches
        self.placement_radius_frac = placement_radius_frac
        self.patch_radius = patch_radius


    def reset_num_patches(self):
        self.patches = []
        arena_width, arena_height = self.arena_size
        center = np.array([arena_width, arena_height]) / 2.0
        half_width  = arena_width  / 2.0
        half_height = arena_height / 2.0

        # Radii of the placement ellipse along x and y
        placement_radius_x = half_width  * self.placement_radius_frac
        placement_radius_y = half_height * self.placement_radius_frac

        if self.fixed_num_patches == 1:
            # now, single patch in the center
            # but could also follow the paradigm of the other n-patch arenas
            self.patches.append(
                FoodPatch(
                    position=center,
                    shape="circle",
                    dimensions=[self.patch_radius],
                    **self.patch_kwargs,
                )
            )
        else:
            # random starting angle so layout isn’t always aligned
            initial_angle = self.np_random.uniform(0, 2 * np.pi)
            for i in range(self.fixed_num_patches):
                # evenly spaced around the ellipse
                angle = initial_angle + 2 * np.pi * i / self.fixed_num_patches
                offset_x = placement_radius_x * np.cos(angle)
                offset_y = placement_radius_y * np.sin(angle)
                position = center + np.array([offset_x, offset_y])

                self.patches.append(
                    FoodPatch(
                        position=position,
                        shape="circle",
                        dimensions=[self.patch_radius],
                        **self.patch_kwargs,
                    )
                )

class NoFoodArena(Arena):
    """
    An empty arena with no food patches/pellets for homing experiments 
    """
    def reset_num_patches(self):
        # No patches -> no food
        self.patches = []

    def step_num_patches(self):
        # Nothing to add/remove over time
        pass

    def prune_unused_patches(self):
        # Nothing to prune
        pass

# ----------------------------------------- 

def demo_animate(args, kwargs):
    if args.arena_style == "uniform":
        arena = UniformArena(**kwargs)
    elif args.arena_style == "patchy":
        arena = PatchyArena(
            patch_d_mean=args.patch_d_mean,
            patch_d_var=args.patch_d_var,
            reset_patch_density=args.reset_patch_density,
            step_patch_density=args.step_patch_density,
            max_patch_density=args.max_patch_density,
            **kwargs,
        )
    elif args.arena_style == "feeder":
        arena = FeederArena(
            patch_d_mean=args.patch_d_mean,
            patch_d_var=args.patch_d_var,
            patches_per_edge=args.patches_per_edge,
            **kwargs,
        )
    elif args.arena_style == "onepatch":
        arena = OnePatchArena(
            patch_d_mean=args.patch_d_mean,
            patch_d_var=args.patch_d_var,
            **kwargs,
        )
    elif args.arena_style == "lattice":
        arena = LatticeArena(
            patches_per_edge=args.patches_per_edge,
            **kwargs,
        )
    elif args.arena_style == "nofood":
        arena = NoFoodArena(**kwargs)
    else:
        raise ValueError(f"Unknown arena style: {args.arena_style}")

    frames = []
    arena.reset()
    for i in tqdm.tqdm(range(args.steps)):
        frames.append(arena.render())
        arena.step()

    frames.append(arena.render())

    filename = "./arena.mp4"
    imageio.mimsave(filename, frames, fps=83)


def demo_figures(args, kwargs):
    kwargs["min_arena_size"] = (30, 30)
    kwargs["max_arena_size"] = (30, 30)
    args.reset_patch_density = 0.01
    args.patch_d_mean = 8
    args.patch_d_var = 1
    args.reset_food_density = 0.2

    for arena_type in [
        "uniform",
        "patchy",
        "feeder",
        "onepatch",
        # "lattice",  # Note hacky overrides in this section
        "square_quadrant", 
        "npatch",
        "nofood",
    ]:
        if arena_type == "patchy":
            arena = PatchyArena(
                patch_d_mean=args.patch_d_mean,
                patch_d_var=args.patch_d_var,
                reset_patch_density=args.reset_patch_density,
                step_patch_density=args.step_patch_density,
                max_patch_density=args.max_patch_density,
                **kwargs,
            )
        if arena_type == "feeder":
            arena = FeederArena(
                patch_d_mean=args.patch_d_mean,
                patch_d_var=args.patch_d_var,
                patches_per_edge=args.patches_per_edge,
                **kwargs,
            )
        if arena_type == "onepatch":
            arena = OnePatchArena(
                patch_d_mean=args.patch_d_mean,
                patch_d_var=args.patch_d_var,
                **kwargs,
            )
        if arena_type == "lattice":
            args.patches_per_edge = 10
            args.reset_food_density = 10
            args.stockpile_density = 1.0
            # print(args)
            arena = LatticeArena(
                patches_per_edge=args.patches_per_edge,
                **kwargs,
            )
        if arena_type == "uniform":
            arena = UniformArena(**kwargs)
        if arena_type == "square_quadrant":
            kwargs["min_arena_size"] = (60, 60)
            kwargs["max_arena_size"] = (60, 60)
            for center_location in [
                "bottom_left",
                "bottom_right",
                "top_left",
                "top_right",
                ]:
                arena = SquareQuadrantArena(
                    num_radial_patches=4,
                    patch_radius=7.5,
                    center_location=center_location,
                    **kwargs,
                )
                arena.reset()
                kwargs_render = {
                    "path": f"./arena_{arena_type}_{center_location}.png",
                    "show": False,
                    "manuscript": True,
                    "markersize": 2,  # default: 2,
                }
                frame = arena.render(**kwargs_render)
        if arena_type == "npatch":
            arena = NPatchArena(
                fixed_num_patches=args.fixed_num_patches,
                placement_radius_frac=0.75,
                patch_radius=args.patch_d_mean / 2,
                **kwargs,
            )
        if arena_type == "nofood":
            arena = NoFoodArena(**kwargs)

        arena.reset()
        kwargs_render = {
            "path": f"./arena_{arena_type}.png",
            # "path": f"./arena_{arena_type}.svg",
            "show": False,
            "manuscript": True,
            "markersize": 4,  # default: 2,
        }
        frame = arena.render(**kwargs_render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena_style", type=str, default="uniform")
    parser.add_argument("--reset_food_density", type=float, default=0.05)
    parser.add_argument("--step_food_density", type=float, default=0.0)
    parser.add_argument("--step_food_decay", type=float, default=0.000)
    parser.add_argument("--max_food_density", type=float, default=0.25)
    parser.add_argument("--stockpile_density", type=float, default=float("inf"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--patch_d_mean", type=float, default=10)
    parser.add_argument("--patch_d_var", type=float, default=2)
    parser.add_argument("--reset_patch_density", type=float, default=0.001)
    parser.add_argument("--step_patch_density", type=float, default=0.00000)
    parser.add_argument("--max_patch_density", type=float, default=None)
    parser.add_argument("--patches_per_edge", type=int, default=1)
    parser.add_argument("--fixed_num_patches", type=int, default=1)
    args = parser.parse_args()

    min_arena_size = (60, 60)
    max_arena_size = (60, 60)
    kwargs = {
        "min_arena_size": min_arena_size,
        "max_arena_size": max_arena_size,
        "reset_food_density": args.reset_food_density,
        "step_food_density": args.step_food_density,
        "step_food_decay": args.step_food_decay,
        "max_food_density": args.max_food_density,
        "stockpile_density": args.stockpile_density,
        "drift": 0.01,
        "drag": 0.2,
    }

    # demo_animate(args, kwargs)
    demo_figures(args, kwargs)
