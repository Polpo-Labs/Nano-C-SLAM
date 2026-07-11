"""
A simple 2D range sensor (R3).

This is a generic, configurable range sensor: it fires a set of beams at fixed
body-frame bearings and, for each, reports the distance to the nearest wall (via
world.cast_ray). It stands in for "some depth sensor" so we can build and see a
map. The paper's specific 4x ToF / 8x8-to-row model is layered on later (R5) by
just choosing the right bearings and reduction -- the ray-casting stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.types import Pose2D
from .world import World, cast_ray


@dataclass
class RangeReading:
    """One sweep of the sensor: aligned arrays, one entry per beam.

    bearings : beam angles in the robot body frame (radians).
    ranges   : measured distance per beam (metres); equals max_range on a no-hit.
    valid    : True where a wall was actually hit within range (else False).
    """

    bearings: np.ndarray
    ranges: np.ndarray
    valid: np.ndarray


@dataclass
class RangeSensor:
    """A set of range beams at fixed body-frame bearings.

    bearings    : the beam directions (radians, body frame).
    max_range   : maximum sensing distance (metres). 4 m matches the ToF sensor.
    noise_sigma : std-dev of Gaussian range noise added to valid hits (metres).
    """

    bearings: np.ndarray
    max_range: float = 4.0
    noise_sigma: float = 0.0

    @classmethod
    def ring(cls, n_beams: int = 48, max_range: float = 4.0, noise_sigma: float = 0.0) -> "RangeSensor":
        """A generic 360 deg ring of `n_beams` equally spaced beams."""
        bearings = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
        return cls(bearings, max_range, noise_sigma)

    def measure(self, pose: Pose2D, world: World, rng: np.random.Generator | None = None) -> RangeReading:
        """Sense the world from `pose`, one distance per beam.

        Each beam is cast in the world frame at (pose heading + beam bearing).
        Hits within range are optionally corrupted with Gaussian noise (imperfect
        ranging); no-hit beams are marked invalid and clamped to max_range.
        """
        ranges = np.empty(len(self.bearings))
        valid = np.empty(len(self.bearings), dtype=bool)

        for k, bearing in enumerate(self.bearings):
            distance = cast_ray((pose.x, pose.y), pose.psi + bearing, world.walls, self.max_range)
            if np.isinf(distance):
                ranges[k], valid[k] = self.max_range, False
            else:
                if rng is not None and self.noise_sigma > 0.0:
                    distance += rng.normal(0.0, self.noise_sigma)
                ranges[k], valid[k] = distance, True

        return RangeReading(self.bearings, ranges, valid)
