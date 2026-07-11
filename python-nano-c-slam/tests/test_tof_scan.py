"""
Unit tests for the R5 paper-style sensing: the 4x8 quad-ToF depth frame and the
20-frame rotating scan.

Key things we pin: the depth frame has the right shape and reads a known box
correctly; a single frame has 45 deg gaps between the four sensors; rotating
while acquiring a scan fills those gaps; and two scans of the same spot match
(ICP ~ identity).
"""

import numpy as np
import pytest

from nano_c_slam.core.types import (
    NUM_SENSORS,
    PIXELS_PER_SENSOR,
    AugmentedPose,
    Pose2D,
)
from nano_c_slam.slam.icp import icp
from nano_c_slam.slam.mapping import depth_local_points
from nano_c_slam.slam.scan import build_scan, scan_point_cloud
from nano_c_slam.sim.sensor import QuadToFSensor
from nano_c_slam.sim.world import World


def largest_angular_gap(points: np.ndarray) -> float:
    """Biggest empty angular wedge (radians) in a set of points seen from origin.

    Small = the points surround the origin evenly; large = there is a blind arc.
    """
    angles = np.sort(np.arctan2(points[:, 1], points[:, 0]))
    wrapped = np.concatenate([angles, [angles[0] + 2 * np.pi]])
    return float(np.diff(wrapped).max())


def test_depth_frame_shape_and_box_distances():
    # Robot at the centre of a 4x4 box (walls at +/-2). Every beam hits a wall.
    world = World.rectangle(-2.0, -2.0, 2.0, 2.0)
    sensor = QuadToFSensor(max_range=8.0)
    depth = sensor.measure(Pose2D(0.0, 0.0, 0.0), world)

    assert depth.ranges.shape == (NUM_SENSORS, PIXELS_PER_SENSOR)
    assert depth.valid.all()
    # The front sensor (row 0) looks along +x at the wall 2 m away; a beam angled
    # by `offset` from centre travels 2/cos(offset) to reach that wall.
    offsets = np.linspace(-sensor.fov / 2, sensor.fov / 2, PIXELS_PER_SENSOR)
    np.testing.assert_allclose(depth.ranges[0], 2.0 / np.cos(offsets), atol=1e-9)


def test_single_frame_has_sensor_gaps_that_rotation_fills():
    world = World.rectangle(-2.0, -2.0, 2.0, 2.0)
    sensor = QuadToFSensor(max_range=8.0)

    # One frame: four 45 deg arcs with 45 deg gaps -> a sizeable angular gap.
    single = depth_local_points(sensor.measure(Pose2D(0, 0, 0), world), sensor.bearings)
    single_gap = largest_angular_gap(single)
    assert single_gap > np.radians(40)  # the ~45 deg blind wedge between sensors

    # A scan: 20 frames while rotating 90 deg in place -> gaps get swept over.
    frames = [
        AugmentedPose(k, float(k), Pose2D(0.0, 0.0, psi), sensor.measure(Pose2D(0.0, 0.0, psi), world))
        for k, psi in enumerate(np.linspace(0.0, np.pi / 2, 20))
    ]
    cloud = scan_point_cloud(build_scan(0, frames), sensor.bearings)
    scan_gap = largest_angular_gap(cloud)
    assert scan_gap < single_gap          # rotation improved coverage
    assert scan_gap < np.radians(15)      # essentially a full 360 deg sweep


def test_two_scans_of_same_place_match_with_icp():
    # Build the same rotating scan twice from identical true poses; the aggregated
    # clouds should be (near) identical, so ICP returns ~identity with ~0 error.
    world = World.rectangle(-2.0, -2.0, 2.0, 2.0).with_segments([(1.0, 1.5, 1.8, 1.5)])
    sensor = QuadToFSensor(max_range=8.0)

    def make_scan():
        frames = [
            AugmentedPose(k, float(k), Pose2D(0.0, 0.0, psi), sensor.measure(Pose2D(0.0, 0.0, psi), world))
            for k, psi in enumerate(np.linspace(0.0, np.pi / 2, 20))
        ]
        return scan_point_cloud(build_scan(0, frames), sensor.bearings)

    transform, _, error, _ = icp(make_scan(), make_scan())
    assert error == pytest.approx(0.0, abs=1e-9)
    assert transform.x == pytest.approx(0.0, abs=1e-6)
    assert transform.y == pytest.approx(0.0, abs=1e-6)
