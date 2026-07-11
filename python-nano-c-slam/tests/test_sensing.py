"""
Unit tests for the R3 sensing + mapping pieces: ray casting, the range sensor,
and projecting readings into the map. We use tiny hand-checkable scenes so the
expected distances/points are obvious.
"""

import numpy as np
import pytest

from nano_c_slam.core.types import Pose2D
from nano_c_slam.sim.sensor import RangeSensor
from nano_c_slam.sim.world import World, cast_ray
from nano_c_slam.slam.mapping import build_point_cloud, project_reading


def test_cast_ray_hits_wall_at_known_distance():
    # A vertical wall at x = 1; a ray from the origin pointing +x hits it at 1 m.
    wall = World.from_segments([(1.0, -1.0, 1.0, 1.0)]).walls
    assert cast_ray((0.0, 0.0), 0.0, wall, max_range=4.0) == pytest.approx(1.0)


def test_cast_ray_misses_when_wall_is_behind_or_out_of_range():
    wall = World.from_segments([(1.0, -1.0, 1.0, 1.0)]).walls
    # Pointing -x: the wall is behind the ray -> no hit.
    assert np.isinf(cast_ray((0.0, 0.0), np.pi, wall, max_range=4.0))
    # Wall within geometry but beyond max_range -> reported as no hit.
    far = World.from_segments([(5.0, -1.0, 5.0, 1.0)]).walls
    assert np.isinf(cast_ray((0.0, 0.0), 0.0, far, max_range=3.0))


def test_sensor_measures_box_walls():
    # Robot at the centre of a 2x2 room (walls at x,y = +/-1). The four axis-
    # aligned beams should each read 1 m and be valid.
    world = World.rectangle(-1.0, -1.0, 1.0, 1.0)
    sensor = RangeSensor(bearings=np.array([0.0, np.pi / 2, np.pi, -np.pi / 2]), max_range=4.0)
    reading = sensor.measure(Pose2D(0.0, 0.0, 0.0), world)
    assert reading.valid.all()
    np.testing.assert_allclose(reading.ranges, [1.0, 1.0, 1.0, 1.0], atol=1e-9)


def test_no_hit_beam_is_invalid_and_clamped():
    # A single wall only in front (+x); the beam pointing -x hits nothing.
    world = World.from_segments([(1.0, -1.0, 1.0, 1.0)])
    sensor = RangeSensor(bearings=np.array([0.0, np.pi]), max_range=4.0)
    reading = sensor.measure(Pose2D(0.0, 0.0, 0.0), world)
    assert reading.valid[0] and not reading.valid[1]
    assert reading.ranges[1] == pytest.approx(4.0)  # clamped to max_range


def test_project_reading_places_points_in_world():
    # Robot at (1,1) facing +x; one valid beam straight ahead at 2 m -> (3,1).
    world = World.from_segments([(3.0, 0.0, 3.0, 2.0)])
    sensor = RangeSensor(bearings=np.array([0.0]), max_range=4.0)
    reading = sensor.measure(Pose2D(1.0, 1.0, 0.0), world)
    points = project_reading(Pose2D(1.0, 1.0, 0.0), reading)
    np.testing.assert_allclose(points, [[3.0, 1.0]], atol=1e-9)


def test_build_point_cloud_stacks_all_readings():
    world = World.rectangle(-1.0, -1.0, 1.0, 1.0)
    sensor = RangeSensor.ring(n_beams=8, max_range=4.0)
    poses = [Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 0.0, 0.0)]
    readings = [sensor.measure(p, world) for p in poses]
    cloud = build_point_cloud(poses, readings)
    # Two identical readings of 8 valid beams -> 16 mapped points, all on walls.
    assert cloud.shape == (16, 2)
    assert np.all(np.abs(cloud) <= 1.0 + 1e-9)
