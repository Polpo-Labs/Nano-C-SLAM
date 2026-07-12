"""
Unit tests for the R7 communication layer.

We check the medium's accounting, the message sizes (paper's data model), and --
most importantly -- that the message-driven distributed run produces the same
corrected poses as the direct R6-style computation, and that it reduces drone 1's
drift by aligning it to drone 0.
"""

import numpy as np
import pytest

from nano_c_slam.comms.messages import POSE_BYTES, SCAN_BYTES, PoseUpdateMessage, ScanMessage
from nano_c_slam.comms.medium import Medium
from nano_c_slam.comms.swarm import run_swarm
from nano_c_slam.core.types import Pose2D
from nano_c_slam.sim.mission import simulate_mission
from nano_c_slam.sim.sensor import QuadToFSensor
from nano_c_slam.sim.trajectories import square_mission
from nano_c_slam.sim.world import World


def two_drone_missions():
    """Two small overlapping missions (drone 1 starts one square right of drone 0)."""
    world = World.rectangle(-1.5, -1.5, 5.5, 3.5).with_segments(
        [(1.0, 3.5, 1.0, 2.7), (3.8, -1.5, 3.8, -0.7)]
    )
    sensor = QuadToFSensor(max_range=12.0, noise_sigma=0.01)
    steps, anchors = square_mission(side=2.0, laps=1, increments=10, scan_frames=20)
    m0 = simulate_mission(0, Pose2D(0, 0, 0), steps, anchors, 20, world, sensor, np.random.default_rng(1))
    m1 = simulate_mission(1, Pose2D(2, 0, 0), steps, anchors, 20, world, sensor, np.random.default_rng(2))
    return m0, m1


def mean_position_error(estimate, truth):
    return float(np.mean([np.hypot(e.x - t.x, e.y - t.y) for e, t in zip(estimate, truth)]))


def test_message_sizes_follow_paper_model():
    assert PoseUpdateMessage(0, 0, Pose2D(0, 0, 0)).size_bytes == POSE_BYTES == 12
    assert ScanMessage(0, 0, Pose2D(0, 0, 0), np.empty((0, 2))).size_bytes == SCAN_BYTES


def test_medium_accounting():
    medium = Medium([0, 1, 2])
    medium.broadcast(0, [1, 2], PoseUpdateMessage(0, 5, Pose2D(0, 0, 0)))  # 2 messages
    medium.send(2, ScanMessage(0, 5, Pose2D(0, 0, 0), np.empty((0, 2))))    # 1 message
    assert medium.messages_sent == 3
    assert medium.bytes_sent == 2 * POSE_BYTES + SCAN_BYTES
    # Delivery hands over and clears the inbox.
    assert len(medium.deliver(2)) == 2
    assert medium.deliver(2) == []


def test_swarm_message_counts():
    m0, m1 = two_drone_missions()
    _, medium = run_swarm([m0, m1])
    s0 = len(m0.scans)
    # Drone 0 sends each scan and each pose-update to drone 1; drone 1 sends to no
    # higher drone. So total messages = 2 * (number of drone-0 scans).
    assert medium.messages_sent == 2 * s0
    assert medium.bytes_sent == s0 * SCAN_BYTES + s0 * POSE_BYTES


def test_distributed_reduces_drone1_drift():
    m0, m1 = two_drone_missions()
    agents, _ = run_swarm([m0, m1])
    before = mean_position_error(m1.odom, m1.truth)
    after = mean_position_error(agents[1].poses, m1.truth)
    assert after < before  # aligning to drone 0 improved drone 1's estimate


def test_distributed_matches_direct_computation():
    # The swarm result must equal computing the two drones' corrections directly
    # with the same library functions (this pins the message plumbing).
    from nano_c_slam.comms.agent import DroneAgent

    m0, m1 = two_drone_missions()
    agents, _ = run_swarm([m0, m1])

    # Direct: drone 0 optimizes alone; drone 1 optimizes with drone 0's corrected
    # scans handed to it directly (no medium).
    direct0 = DroneAgent(0, m0.odom)
    for s in m0.scans:
        direct0.acquire(s)
    direct0.optimize()

    direct1 = DroneAgent(1, m1.odom)
    for s in m1.scans:
        direct1.acquire(s)
    for s in m0.scans:
        direct1.receive(ScanMessage(0, s.anchor_id, s.anchor_pose, s.cloud))
    for s in m0.scans:  # corrected poses from drone 0
        direct1.receive(PoseUpdateMessage(0, s.anchor_id, direct0.poses[s.anchor_id]))
    direct1.optimize()

    for a, b in zip(agents[1].poses, direct1.poses):
        assert a.x == pytest.approx(b.x, abs=1e-9)
        assert a.y == pytest.approx(b.y, abs=1e-9)
