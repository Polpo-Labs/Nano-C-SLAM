"""
Unit tests for the R6 collaborative pieces: the mission simulator, the
loop-closure derivation (ICP + rejection filter), and the decoupled optimizer
that aligns one drone to a fixed reference drone.
"""

import numpy as np
import pytest

from nano_c_slam.core.geometry import relative
from nano_c_slam.core.types import Edge, EdgeType, Pose2D
from nano_c_slam.slam.collaborative import optimize_against_reference
from nano_c_slam.slam.loop_closure import anchors_within, derive_closure
from nano_c_slam.slam.scan import ScanRecord
from nano_c_slam.sim.mission import simulate_mission
from nano_c_slam.sim.sensor import QuadToFSensor
from nano_c_slam.sim.trajectories import square_mission
from nano_c_slam.sim.world import World


def small_mission():
    world = World.rectangle(-1.5, -1.5, 3.5, 3.5).with_segments([(2.6, 0.4, 2.6, 1.6)])
    sensor = QuadToFSensor(max_range=8.0, noise_sigma=0.005)
    steps, anchors = square_mission(side=2.0, laps=1, increments=10, scan_frames=20)
    rng = np.random.default_rng(0)
    result = simulate_mission(0, Pose2D(0, 0, 0), steps, anchors, 20, world, sensor, rng)
    return world, sensor, result


def test_mission_produces_a_scan_per_corner():
    _, _, result = small_mission()
    # One lap = four corner scans; each scan aggregates 20 frames x 32 beams.
    assert len(result.scans) == 4
    assert all(scan.cloud.shape[0] <= 640 and scan.cloud.shape[0] > 400 for scan in result.scans)
    # Odometry has drifted from truth by the end (there is something to correct).
    assert np.hypot(result.odom[-1].x - result.truth[-1].x,
                    result.odom[-1].y - result.truth[-1].y) > 0.0


def test_derive_closure_matches_revisited_scan_and_rejects_garbage():
    _, _, result = small_mission()
    # The last corner returns to the start corner, so scan 0 and the start-corner
    # scan of the next visit overlap. Here we match scan 0 against itself-like:
    # re-derive using two scans of the same physical corner is covered in the
    # experiment; for the unit test we check the accept/reject logic directly.
    good_z, good_err = derive_closure(result.scans[0], result.scans[0])
    assert good_z is not None and good_err == pytest.approx(0.0, abs=1e-6)

    # Garbage target (random cloud) must be rejected by the error filter.
    garbage = result.scans[0]
    garbage_copy = type(garbage)(garbage.drone_id, garbage.anchor_id, garbage.anchor_pose,
                                 np.random.default_rng(1).uniform(-5, 5, size=(600, 2)))
    z, err = derive_closure(result.scans[0], garbage_copy)
    assert z is None and err > 0.10


def test_anchors_within_distance():
    a = Pose2D(0.0, 0.0, 0.0)
    b = Pose2D(0.5, 0.0, 1.0)
    far = Pose2D(3.0, 0.0, 0.0)
    empty = np.empty((0, 2))
    assert anchors_within(ScanRecord(0, 0, a, empty), ScanRecord(0, 1, b, empty))
    assert not anchors_within(ScanRecord(0, 0, a, empty), ScanRecord(1, 0, far, empty))


def test_optimize_against_reference_aligns_to_fixed_anchor():
    # Drone 1 believes it is at (5,0) but an inter-drone closure to a fixed
    # reference pose says it should coincide with reference pose R at (1,0).
    own = [Pose2D(0.0, 0.0, 0.0), Pose2D(5.0, 0.0, 0.0)]
    own_edges = [Edge(0, 1, relative(own[0], own[1]), np.eye(3), EdgeType.ODOMETRY)]
    reference = {7: Pose2D(1.0, 0.0, 0.0)}
    # measurement: own pose 1 seen from reference 7 -> identity (they coincide).
    inter = [(1, 7, Pose2D(0.0, 0.0, 0.0), np.eye(3) * 10.0)]
    corrected = optimize_against_reference(own, own_edges, reference, inter)
    assert len(corrected) == 2                       # reference dropped from result
    assert corrected[0].x == pytest.approx(0.0, abs=1e-9)  # own anchor stayed fixed
    assert corrected[0].y == pytest.approx(0.0, abs=1e-9)
    # Own pose 1 pulled toward the reference at (1,0) (from its wrong (5,0)).
    assert corrected[1].x < 5.0 and corrected[1].x == pytest.approx(1.0, abs=0.6)
