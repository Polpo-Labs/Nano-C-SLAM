"""
Unit tests for the R4 ICP scan-matcher.

We test the ICP *math* on a scattered point cloud: random points have unique
nearest neighbours, so there is no "sliding along a straight edge" ambiguity and
a correct ICP recovers the exact transform. (That edge-sliding effect is real --
it is why the paper uses corner-rich 360-degree scans + rejection filters -- but
it is a property of the geometry, not of the solver, so we don't test it here.)
The experiment exercises ICP on real room scans with an odometry initial guess.
"""

import numpy as np
import pytest

from nano_c_slam.core.geometry import compose, invert, transform_points, wrap_angle
from nano_c_slam.core.types import Pose2D
from nano_c_slam.slam.icp import _best_rigid_transform, icp


def scattered_cloud() -> np.ndarray:
    """120 random points in a 4x4 area -- distinctive, no correspondence ambiguity."""
    rng = np.random.default_rng(0)
    return rng.uniform(-2.0, 2.0, size=(120, 2))


def assert_pose_close(a: Pose2D, b: Pose2D, tol: float = 1e-2) -> None:
    assert a.x == pytest.approx(b.x, abs=tol)
    assert a.y == pytest.approx(b.y, abs=tol)
    assert wrap_angle(a.psi - b.psi) == pytest.approx(0.0, abs=tol)


def test_best_rigid_transform_recovers_known_transform():
    # With exact correspondences (same order), the SVD fit must be exact.
    cloud = scattered_cloud()
    move = Pose2D(0.4, -0.3, np.radians(25))
    moved = transform_points(move, cloud)
    R, t = _best_rigid_transform(cloud, moved)
    recovered = Pose2D(t[0], t[1], np.arctan2(R[1, 0], R[0, 0]))
    assert_pose_close(recovered, move, tol=1e-6)


def test_icp_recovers_transform_from_identity():
    # target = original cloud; source = cloud moved by `move`. ICP(source, target)
    # should map source back onto target, i.e. return inv(move), error -> 0.
    cloud = scattered_cloud()
    move = Pose2D(0.15, -0.1, np.radians(8))
    source = transform_points(move, cloud)
    transform, aligned, error, iters = icp(source, cloud)
    assert_pose_close(transform, invert(move))
    assert error == pytest.approx(0.0, abs=1e-6)
    # Applying the original move then the found transform returns to identity.
    assert_pose_close(compose(move, transform), Pose2D(0.0, 0.0, 0.0))


def test_icp_refines_a_rough_initial_guess():
    # The realistic loop-closure case: a larger displacement, but seeded with a
    # rough odometry-style guess (off by ~8 deg and a few cm). ICP refines it.
    cloud = scattered_cloud()
    move = Pose2D(0.5, -0.4, np.radians(30))
    source = transform_points(move, cloud)
    truth = invert(move)
    rough_init = Pose2D(truth.x + 0.08, truth.y - 0.06, truth.psi + np.radians(8))
    transform, _, error, _ = icp(source, cloud, init=rough_init)
    assert_pose_close(transform, truth)
    assert error == pytest.approx(0.0, abs=1e-6)


def test_icp_identity_for_same_cloud():
    cloud = scattered_cloud()
    transform, _, error, _ = icp(cloud, cloud)
    assert_pose_close(transform, Pose2D(0.0, 0.0, 0.0))
    assert error == pytest.approx(0.0, abs=1e-9)
