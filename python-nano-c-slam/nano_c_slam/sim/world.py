"""
The simulated 2D world: walls, and ray-casting against them (R3).

The environment is just a set of wall *segments*. That is all a 2D range sensor
needs: to measure a beam we shoot a ray from the sensor and find the nearest
wall it hits. Keeping the world this simple (line segments, one ray-cast
function) keeps it easy to build test scenes and to reason about what the sensor
"should" return.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class World:
    """A 2D environment described by wall segments.

    walls : (M, 4) array, each row [x1, y1, x2, y2] -- a wall from (x1,y1) to
            (x2,y2). Everything the sensor can see is one of these segments.
    """

    walls: np.ndarray

    @classmethod
    def from_segments(cls, segments: list[tuple[float, float, float, float]]) -> "World":
        """Build a world from a list of (x1, y1, x2, y2) segments."""
        return cls(np.asarray(segments, dtype=float).reshape(-1, 4))

    @classmethod
    def rectangle(cls, xmin: float, ymin: float, xmax: float, ymax: float) -> "World":
        """Build a closed rectangular room (its four walls)."""
        return cls.from_segments(
            [
                (xmin, ymin, xmax, ymin),  # bottom
                (xmax, ymin, xmax, ymax),  # right
                (xmax, ymax, xmin, ymax),  # top
                (xmin, ymax, xmin, ymin),  # left
            ]
        )

    def with_segments(self, segments: list[tuple[float, float, float, float]]) -> "World":
        """Return a copy of this world with extra wall segments added (e.g. obstacles)."""
        extra = np.asarray(segments, dtype=float).reshape(-1, 4)
        return World(np.vstack([self.walls, extra]))


def cast_ray(
    origin: tuple[float, float], angle: float, walls: np.ndarray, max_range: float
) -> float:
    """Distance from `origin` along `angle` to the nearest wall, or inf if none.

    Casts the ray against every wall segment at once (vectorized) and returns the
    smallest positive hit distance that is within `max_range`. Returns +inf when
    nothing is hit in range -- the caller decides how to report a no-hit beam.

    Ray:      origin + t * (cos a, sin a),  t >= 0
    Segment:  A + u * (B - A),              u in [0, 1]
    We solve the 2x2 system for (t, u) per segment; a hit needs t > 0 and
    0 <= u <= 1. Since the direction is a unit vector, t is the distance.
    """
    if walls.size == 0:
        return float("inf")

    ox, oy = origin
    dx, dy = np.cos(angle), np.sin(angle)

    ax, ay = walls[:, 0], walls[:, 1]
    bx, by = walls[:, 2], walls[:, 3]
    ex, ey = bx - ax, by - ay      # segment direction vectors
    rx, ry = ax - ox, ay - oy      # from ray origin to segment start

    # Determinant of the 2x2 system; ~0 means ray and segment are parallel.
    det = ex * dy - ey * dx
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ex * ry - ey * rx) / det   # distance along the ray
        u = (dx * ry - dy * rx) / det   # position along the segment

    hit = (np.abs(det) > 1e-12) & (t > 1e-9) & (u >= 0.0) & (u <= 1.0)
    distances = np.where(hit, t, np.inf)
    nearest = float(distances.min())
    return nearest if nearest <= max_range else float("inf")
