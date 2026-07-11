"""
Unit tests for the SE(2) geometry foundation.

Everything downstream (ICP, pose-graph edges, the external-pose loop-closure
trick) trusts these functions, so we pin their core algebraic properties here.
Run with:  ../.venv/Scripts/python.exe -m pytest
"""

import numpy as np
import pytest

from nano_c_slam.core.geometry import (
    compose,
    invert,
    matrix_to_pose,
    pose_to_matrix,
    relative,
    wrap_angle,
)
from nano_c_slam.core.types import Pose2D


def assert_pose_close(a: Pose2D, b: Pose2D, tol: float = 1e-9) -> None:
    """Assert two poses are equal, comparing headings modulo 2*pi."""
    assert a.x == pytest.approx(b.x, abs=tol)
    assert a.y == pytest.approx(b.y, abs=tol)
    # Compare headings via their wrapped difference so +pi and -pi count equal.
    assert wrap_angle(a.psi - b.psi) == pytest.approx(0.0, abs=tol)


def test_wrap_angle_folds_into_range():
    # The result must always land in [-pi, pi), and must stay equivalent to the
    # input modulo 2*pi (that equivalence is what actually matters downstream).
    for angle in (3 * np.pi, -3 * np.pi, 0.5, 10.0, -10.0):
        wrapped = wrap_angle(angle)
        assert -np.pi <= wrapped < np.pi
        assert wrap_angle(wrapped - angle) == pytest.approx(0.0, abs=1e-9)


def test_matrix_round_trip():
    # pose -> matrix -> pose must return the original pose.
    pose = Pose2D(1.2, -3.4, 0.9)
    assert_pose_close(matrix_to_pose(pose_to_matrix(pose)), pose)


def test_invert_is_true_inverse():
    # Composing a pose with its inverse must give the identity pose.
    pose = Pose2D(2.0, -1.0, 1.3)
    assert_pose_close(compose(pose, invert(pose)), Pose2D(0.0, 0.0, 0.0))
    assert_pose_close(compose(invert(pose), pose), Pose2D(0.0, 0.0, 0.0))


def test_relative_then_compose_reconstructs_target():
    # If z = relative(a, b) is b seen from a, then a composed with z gives b back.
    # This is the identity the pose-graph optimizer relies on.
    a = Pose2D(1.0, 2.0, 0.3)
    b = Pose2D(-0.5, 4.0, -1.1)
    z = relative(a, b)
    assert_pose_close(compose(a, z), b)


def test_relative_of_pose_with_itself_is_identity():
    pose = Pose2D(5.0, 5.0, 2.0)
    assert_pose_close(relative(pose, pose), Pose2D(0.0, 0.0, 0.0))
