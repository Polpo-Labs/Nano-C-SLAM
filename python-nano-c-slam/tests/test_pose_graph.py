"""
Unit tests for the R2 generic pose-graph optimizer.

We pin the behaviours SLAM relies on: an anchored pose stays fixed, a single
constraint pulls a free pose to the exact solution, a graph already satisfied by
its edges is left unchanged, and -- the key one -- a loop closure reduces the
drift of a noisy trajectory.
"""

import numpy as np
import pytest

from nano_c_slam.core.geometry import relative
from nano_c_slam.core.types import EdgeType, Pose2D
from nano_c_slam.slam.pose_graph import PoseGraph
from nano_c_slam.sim.motion import integrate
from nano_c_slam.sim.trajectories import square_steps, steps_per_lap


def mean_position_error(estimate: list[Pose2D], truth: list[Pose2D]) -> float:
    """Average straight-line position error between two equal-length paths (m)."""
    return float(np.mean([np.hypot(e.x - t.x, e.y - t.y) for e, t in zip(estimate, truth)]))


def test_single_constraint_moves_free_pose_to_solution():
    # Pose 1 starts wrong at (2,0,0); the edge says it should sit 1 m ahead of
    # the anchored pose 0. Optimization must move it exactly to (1,0,0).
    graph = PoseGraph([Pose2D(0.0, 0.0, 0.0), Pose2D(2.0, 0.0, 0.0)])
    graph.add_edge(0, 1, Pose2D(1.0, 0.0, 0.0))
    graph.optimize(fixed_ids=(0,))
    assert graph.poses[1].x == pytest.approx(1.0, abs=1e-6)
    assert graph.poses[1].y == pytest.approx(0.0, abs=1e-6)


def test_anchor_pose_never_moves():
    graph = PoseGraph([Pose2D(0.0, 0.0, 0.0), Pose2D(2.0, 0.0, 0.0)])
    graph.add_edge(0, 1, Pose2D(1.0, 0.0, 0.0))
    graph.optimize(fixed_ids=(0,))
    assert graph.poses[0] == Pose2D(0.0, 0.0, 0.0)


def test_graph_consistent_with_edges_is_unchanged():
    # If every odometry edge already matches the initial estimate (edges built
    # from the same poses), there is nothing to fix -- poses must stay put.
    rng = np.random.default_rng(0)
    odom = integrate(Pose2D(0.0, 0.0, 0.0), square_steps(laps=1, increments=10),
                     rng=rng, trans_sigma=0.01, rot_sigma=0.01)
    graph = PoseGraph(odom)
    for i in range(len(odom) - 1):
        graph.add_edge(i, i + 1, relative(odom[i], odom[i + 1]))
    graph.optimize(fixed_ids=(0,))
    for optimized, original in zip(graph.poses, odom):
        assert optimized.x == pytest.approx(original.x, abs=1e-6)
        assert optimized.y == pytest.approx(original.y, abs=1e-6)


def test_loop_closure_reduces_drift():
    start = Pose2D(0.0, 0.0, 0.0)
    steps = square_steps(side=1.0, laps=1, increments=10)
    truth = integrate(start, steps)                                   # exact path
    rng = np.random.default_rng(0)
    odom = integrate(start, steps, rng=rng, trans_sigma=0.01, rot_sigma=0.01)

    graph = PoseGraph(odom)
    # Odometry edges = the (noisy) relative motions the robot actually reported.
    for i in range(len(steps)):
        graph.add_edge(i, i + 1, relative(odom[i], odom[i + 1]))
    # Loop closure: the last pose really coincides with the first (truth knows it).
    last = len(steps)
    graph.add_edge(0, last, relative(truth[0], truth[last]),
                   information=np.eye(3) * 10.0, edge_type=EdgeType.INTRA_LC)

    before = mean_position_error(odom, truth)
    graph.optimize(fixed_ids=(0,))
    after = mean_position_error(graph.poses, truth)

    assert after < before          # the correction helped
    assert steps_per_lap(10) == last  # sanity: index bookkeeping is right
