"""
Unit tests for the R8 exploration policy.

We drive the controller with synthetic depth frames (fast, exact) to pin the
Cruise/Spinning transitions, and run a short closed-loop mission to confirm it
explores, acquires corner scans, and returns a consistent MissionResult.
"""

import numpy as np

from nano_c_slam.core.types import NUM_SENSORS, PIXELS_PER_SENSOR, DepthFrame, Pose2D
from nano_c_slam.exploration.controller import (
    FRONT,
    ExploreState,
    WallFollowController,
    min_distance,
)
from nano_c_slam.sim.explore import explore_mission
from nano_c_slam.sim.sensor import QuadToFSensor
from nano_c_slam.sim.world import World


def frame(front=4.0, back=4.0, left=4.0, right=4.0) -> DepthFrame:
    """A depth frame with each sensor reporting a flat distance (all pixels valid)."""
    ranges = np.empty((NUM_SENSORS, PIXELS_PER_SENSOR))
    ranges[:] = np.array([front, back, left, right])[:, None]
    return DepthFrame(ranges, np.ones((NUM_SENSORS, PIXELS_PER_SENSOR), dtype=bool))


def test_min_distance_and_no_hit():
    assert min_distance(frame(front=1.0), FRONT) == 1.0
    empty = DepthFrame(np.full((NUM_SENSORS, PIXELS_PER_SENSOR), 4.0),
                       np.zeros((NUM_SENSORS, PIXELS_PER_SENSOR), dtype=bool))
    assert np.isinf(min_distance(empty, FRONT))


def test_cruise_when_front_clear():
    c = WallFollowController()
    vx, vy, omega = c.command(frame(front=3.0, right=0.6))
    assert vx > 0 and omega == 0.0
    assert c.state == ExploreState.CRUISE


def test_spin_when_front_blocked():
    c = WallFollowController()
    vx, vy, omega = c.command(frame(front=0.3))
    assert vx == 0.0 and omega != 0.0
    assert c.state == ExploreState.SPINNING


def test_resume_cruise_after_clearing():
    c = WallFollowController()
    c.command(frame(front=0.3))                       # -> spinning
    assert c.state == ExploreState.SPINNING
    c.command(frame(front=0.5))                       # still blocked -> keep spinning
    assert c.state == ExploreState.SPINNING
    c.command(frame(front=2.0, right=0.6))            # clear -> cruise
    assert c.state == ExploreState.CRUISE


def test_explore_mission_is_consistent_and_scans():
    world = World.rectangle(0, 0, 3, 3)
    sensor = QuadToFSensor(max_range=4.0)
    # Start facing a nearby wall (0.5 m < stop_distance) so the drone spins right
    # away and records a corner scan within the run.
    mission = explore_mission(0, Pose2D(2.5, 1.0, 0.0), WallFollowController(),
                              world, sensor, np.random.default_rng(0), n_steps=80)
    assert len(mission.truth) == len(mission.odom) == 81
    assert len(mission.scans) >= 1
    first = mission.scans[0]
    assert first.cloud.ndim == 2 and first.cloud.shape[1] == 2 and len(first.cloud) > 0
