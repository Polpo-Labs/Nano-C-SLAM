"""
Turning range readings into a point-cloud map (R3).

A range reading is distances at known bearings *in the robot's body frame*. To
put those hits on the map we place each one at its (range, bearing) location in
the body frame and transform it to the world using the robot's pose. Crucially,
we use the robot's *believed* pose (odometry or the optimized estimate) -- so any
pose error smears the map. That is exactly why correcting the poses (R2) sharpens
the map.

All the transforming is delegated to core.geometry.transform_points (DRY).
"""

from __future__ import annotations

import numpy as np

from ..core.geometry import transform_points
from ..core.types import DepthFrame, Pose2D
from ..sim.sensor import RangeReading


def beams_to_local_points(bearings: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Turn (bearing, range) pairs into body-frame (x, y) points.

    Each hit sits at (r*cos b, r*sin b) in the robot body frame. This is the one
    place that conversion lives -- reused for both generic range readings and the
    paper's 4x8 depth frames (DRY).
    """
    return np.column_stack([ranges * np.cos(bearings), ranges * np.sin(bearings)])


def reading_local_points(reading: RangeReading) -> np.ndarray:
    """Body-frame points of a range reading's valid beams."""
    return beams_to_local_points(reading.bearings[reading.valid], reading.ranges[reading.valid])


def depth_local_points(depth: DepthFrame, bearings: np.ndarray) -> np.ndarray:
    """Body-frame points of a depth frame's valid pixels (bearings from the sensor)."""
    return beams_to_local_points(bearings[depth.valid], depth.ranges[depth.valid])


def project_reading(pose: Pose2D, reading: RangeReading) -> np.ndarray:
    """Project the valid beams of one reading into world-frame points (K, 2).

    Only beams that actually hit a wall are mapped; no-hit beams carry no
    information about where something is, so they are dropped.
    """
    return transform_points(pose, reading_local_points(reading))


def build_point_cloud(poses: list[Pose2D], readings: list[RangeReading]) -> np.ndarray:
    """Build one combined map by projecting each reading from its matching pose.

    `poses` and `readings` are aligned (same index = same time step). Returns an
    (N, 2) array of all mapped points stacked together.
    """
    clouds = [project_reading(pose, reading) for pose, reading in zip(poses, readings)]
    return np.vstack(clouds) if clouds else np.empty((0, 2))
