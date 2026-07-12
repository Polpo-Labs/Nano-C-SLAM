"""
Unit tests for the R9 accuracy metrics.
"""

import numpy as np
import pytest

from nano_c_slam.analysis.metrics import mapping_rmse, mean_position_error
from nano_c_slam.core.types import Pose2D
from nano_c_slam.sim.world import World


def test_mean_position_error_zero_for_identical():
    path = [Pose2D(0, 0, 0), Pose2D(1, 1, 0)]
    assert mean_position_error(path, path) == pytest.approx(0.0)


def test_mean_position_error_known_offset():
    est = [Pose2D(0, 0, 0), Pose2D(3, 4, 0)]     # second pose is 5 m off
    truth = [Pose2D(0, 0, 0), Pose2D(0, 0, 0)]
    assert mean_position_error(est, truth) == pytest.approx((0 + 5) / 2)


def test_mapping_rmse_zero_for_points_on_walls():
    world = World.rectangle(0, 0, 2, 2)
    on_walls = np.array([[0.0, 1.0], [2.0, 1.0], [1.0, 0.0], [1.0, 2.0]])  # midpoints
    assert mapping_rmse(on_walls, world.walls) == pytest.approx(0.0, abs=1e-9)


def test_mapping_rmse_measures_offset():
    world = World.rectangle(0, 0, 2, 2)
    # Two points sitting 0.1 m inside the left wall (x=0) -> each 0.1 m off.
    points = np.array([[0.1, 0.5], [0.1, 1.5]])
    assert mapping_rmse(points, world.walls) == pytest.approx(0.1, abs=1e-9)
