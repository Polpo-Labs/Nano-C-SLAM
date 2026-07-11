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
from ..core.types import Pose2D
from ..sim.sensor import RangeReading


def project_reading(pose: Pose2D, reading: RangeReading) -> np.ndarray:
    """Project the valid beams of one reading into world-frame points (K, 2).

    Only beams that actually hit a wall are mapped; no-hit beams carry no
    information about where something is, so they are dropped.
    """
    b = reading.bearings[reading.valid]
    r = reading.ranges[reading.valid]
    # Each hit sits at (r*cos, r*sin) in the body frame, then moves to the world.
    local_points = np.column_stack([r * np.cos(b), r * np.sin(b)])
    return transform_points(pose, local_points)


def build_point_cloud(poses: list[Pose2D], readings: list[RangeReading]) -> np.ndarray:
    """Build one combined map by projecting each reading from its matching pose.

    `poses` and `readings` are aligned (same index = same time step). Returns an
    (N, 2) array of all mapped points stacked together.
    """
    clouds = [project_reading(pose, reading) for pose, reading in zip(poses, readings)]
    return np.vstack(clouds) if clouds else np.empty((0, 2))
