"""
Unit tests for the R1 motion model.

These pin the fundamental behaviours we build everything else on: a step moves
the pose the way we expect, a closed loop of true steps returns to the start
(no drift without noise), and noisy odometry actually accumulates drift.
"""

import numpy as np
import pytest

from nano_c_slam.core.geometry import wrap_angle
from nano_c_slam.core.types import Pose2D
from nano_c_slam.sim.motion import noisy_step, step


def total_drift(a: Pose2D, b: Pose2D) -> float:
    """Straight-line distance between two poses' positions (metres)."""
    return float(np.hypot(a.x - b.x, a.y - b.y))


def square_loop_steps(side: float = 1.0) -> list[tuple[float, float, float]]:
    """The (dx, dy, dpsi) steps of a unit square: forward, turn 90 deg, x4.

    Driving this from any start pose and integrating the true steps must return
    exactly to the start pose -- a clean, drift-free reference trajectory.
    """
    forward = (side, 0.0, 0.0)
    turn = (0.0, 0.0, np.pi / 2)
    return [forward, turn] * 4


def test_zero_step_is_identity():
    pose = Pose2D(1.0, 2.0, 0.5)
    assert step(pose, 0.0, 0.0, 0.0) == pose


def test_forward_step_moves_along_heading():
    # Facing +x at the origin, a forward step of 2 m lands at (2, 0).
    moved = step(Pose2D(0.0, 0.0, 0.0), 2.0, 0.0, 0.0)
    assert moved.x == pytest.approx(2.0)
    assert moved.y == pytest.approx(0.0)
    # Facing +90 deg (+y), the same forward step lands at (0, 2).
    moved = step(Pose2D(0.0, 0.0, np.pi / 2), 2.0, 0.0, 0.0)
    assert moved.x == pytest.approx(0.0, abs=1e-9)
    assert moved.y == pytest.approx(2.0)


def test_closed_loop_returns_to_start_without_noise():
    start = Pose2D(0.0, 0.0, 0.0)
    pose = start
    for dx, dy, dpsi in square_loop_steps():
        pose = step(pose, dx, dy, dpsi)
    assert total_drift(pose, start) == pytest.approx(0.0, abs=1e-9)
    assert wrap_angle(pose.psi - start.psi) == pytest.approx(0.0, abs=1e-9)


def test_noisy_step_with_zero_sigma_equals_step():
    # DRY guarantee: with no noise, noisy_step must behave exactly like step.
    rng = np.random.default_rng(0)
    pose = Pose2D(0.3, -0.7, 1.1)
    assert noisy_step(pose, 0.5, 0.1, 0.2, rng) == step(pose, 0.5, 0.1, 0.2)


def test_noise_accumulates_drift():
    # Integrating the same closed loop with noisy odometry must NOT return to the
    # start -- that residual gap is the drift SLAM will later remove.
    rng = np.random.default_rng(42)
    start = Pose2D(0.0, 0.0, 0.0)
    pose = start
    for dx, dy, dpsi in square_loop_steps():
        pose = noisy_step(pose, dx, dy, dpsi, rng, trans_sigma=0.02, rot_sigma=0.02)
    assert total_drift(pose, start) > 0.01
