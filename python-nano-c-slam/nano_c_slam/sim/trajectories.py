"""
Reusable ground-truth trajectories for demos and tests.

Keeping trajectory generators here (instead of copy-pasting them into every
experiment) means one definition, reused everywhere -- and they double as clean,
known-answer reference paths for the SLAM tests.
"""

from __future__ import annotations

import numpy as np

# A motion step is (dx, dy, dpsi) in the robot body frame -- see sim.motion.


def square_steps(
    side: float = 2.0, laps: int = 1, increments: int = 20
) -> list[tuple[float, float, float]]:
    """Motion steps that walk a `side`-metre square, `laps` times.

    Each side is split into `increments` small forward steps (for a smooth path
    and many odometry steps), followed by a 90 deg in-place turn. Driving this
    with exact motion returns exactly to the start after every lap, so the pose
    at index `steps_per_lap(increments) * L` coincides with the start -- handy as
    a built-in loop closure for testing.
    """
    forward = (side / increments, 0.0, 0.0)
    turn = (0.0, 0.0, np.pi / 2)
    one_side = [forward] * increments + [turn]
    return one_side * 4 * laps


def steps_per_lap(increments: int = 20) -> int:
    """Number of motion steps in one full square lap (4 sides + 4 turns)."""
    return 4 * (increments + 1)
