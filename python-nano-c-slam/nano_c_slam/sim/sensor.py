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

from ..core.types import NUM_SENSORS, PIXELS_PER_SENSOR, DepthFrame, Pose2D
from .world import World, cast_ray

# Mounting directions of the four ToF sensors in the body frame, in the row
# order used by core.types.DepthFrame: front, back, left, right.
SENSOR_CENTERS = np.array([0.0, np.pi, np.pi / 2, -np.pi / 2])


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


@dataclass
class QuadToFSensor:
    """The paper's sensor: four ToF sensors (front/back/left/right).

    Each sensor sees a 45 deg field of view sampled by PIXELS_PER_SENSOR beams,
    so together they cover 180 deg *instantaneously* -- but with 45 deg gaps
    between the four sensors. The drone fills those gaps by rotating while it
    acquires a scan (see slam.scan).

    Note on the 8x8 -> 8 reduction: the real VL53 returns an 8x8 depth matrix and
    the paper takes the median pixel per column to get one 8-value row. In our 2D
    world there is no vertical dimension, so each column is already a single
    range -- the reduction is a no-op and we cast one ray per column directly.

    max_range   : maximum sensing distance (4 m, as in the paper).
    noise_sigma : std-dev of Gaussian range noise on valid hits (metres).
    fov         : per-sensor field of view (45 deg).
    """

    max_range: float = 4.0
    noise_sigma: float = 0.0
    fov: float = np.pi / 4

    @property
    def bearings(self) -> np.ndarray:
        """(NUM_SENSORS, PIXELS_PER_SENSOR) beam bearings in the body frame."""
        offsets = np.linspace(-self.fov / 2, self.fov / 2, PIXELS_PER_SENSOR)
        return SENSOR_CENTERS[:, None] + offsets[None, :]

    def measure(self, pose: Pose2D, world: World, rng: np.random.Generator | None = None) -> DepthFrame:
        """Sense the world from `pose`, returning a 4x8 DepthFrame.

        Same ray-casting as RangeSensor, just laid out in the paper's (sensor,
        pixel) grid. No-hit pixels are marked invalid and clamped to max_range.
        """
        bearings = self.bearings
        ranges = np.full((NUM_SENSORS, PIXELS_PER_SENSOR), self.max_range)
        valid = np.zeros((NUM_SENSORS, PIXELS_PER_SENSOR), dtype=bool)

        for s in range(NUM_SENSORS):
            for p in range(PIXELS_PER_SENSOR):
                distance = cast_ray((pose.x, pose.y), pose.psi + bearings[s, p], world.walls, self.max_range)
                if not np.isinf(distance):
                    if rng is not None and self.noise_sigma > 0.0:
                        distance += rng.normal(0.0, self.noise_sigma)
                    ranges[s, p], valid[s, p] = distance, True

        return DepthFrame(ranges, valid)
