"""
Accuracy metrics for evaluating a mapping run (R9).

Two measures, matching the paper's evaluation:
  * mean_position_error -- how far the estimated poses are from ground truth
    (an ATE-style trajectory error). We compare directly because every drone is
    anchored at its known take-off pose, so no extra alignment is needed.
  * mapping_rmse -- how well the mapped points sit on the real walls: for each
    point we take the distance to the nearest wall segment and RMS them (the
    paper projects each point to the closest line and RMSEs the residuals).

Pure measurement, no side effects.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Pose2D


def mean_position_error(estimate: list[Pose2D], truth: list[Pose2D]) -> float:
    """Average straight-line position error between two aligned pose lists (m)."""
    return float(np.mean([np.hypot(e.x - t.x, e.y - t.y) for e, t in zip(estimate, truth)]))


def _point_segment_distance(points: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """Distance from each of `points` (N, 2) to one wall segment [x1, y1, x2, y2]."""
    a = segment[:2]
    ab = segment[2:] - a
    length2 = float(ab @ ab)
    if length2 == 0.0:                      # degenerate segment = a point
        return np.linalg.norm(points - a, axis=1)
    # Project each point onto the segment, clamped to its endpoints.
    t = np.clip((points - a) @ ab / length2, 0.0, 1.0)
    projection = a + t[:, None] * ab
    return np.linalg.norm(points - projection, axis=1)


def mapping_rmse(cloud: np.ndarray, walls: np.ndarray) -> float:
    """RMS distance from each mapped point to the nearest wall (metres).

    A perfect map (every point exactly on a wall) scores 0; drift and noise push
    it up. Matches the paper's absolute mapping error.
    """
    if len(cloud) == 0 or len(walls) == 0:
        return 0.0
    # (num_walls, num_points) distances, then nearest wall per point.
    per_wall = np.array([_point_segment_distance(cloud, w) for w in walls])
    nearest = per_wall.min(axis=0)
    return float(np.sqrt(np.mean(nearest ** 2)))
