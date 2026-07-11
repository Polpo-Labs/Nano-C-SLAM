"""
Flying a drone mission in simulation (R6).

Ties the pieces together for one drone: integrate its ground-truth and (noisy)
odometry paths, and at each corner acquire a 20-frame rotating scan. The sensor
always senses the *real* world from the *true* pose, but scans are aggregated
using the drone's *estimated* (odometry) poses -- just like the real system,
which only knows its estimate.

Returns everything a collaborative-SLAM experiment needs: the true path (for
evaluation only), the odometry path (the drone's belief), and the scan records.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.types import AugmentedPose, Pose2D
from ..slam.scan import ScanRecord, build_scan, scan_point_cloud
from ..sim.sensor import QuadToFSensor
from ..sim.world import World
from .motion import integrate


@dataclass
class MissionResult:
    """Outcome of flying one drone: true path, believed path, and its scans."""

    truth: list[Pose2D]
    odom: list[Pose2D]
    scans: list[ScanRecord]


def simulate_mission(
    drone_id: int,
    start: Pose2D,
    steps: list[tuple[float, float, float]],
    scan_anchors: list[int],
    scan_frames: int,
    world: World,
    sensor: QuadToFSensor,
    rng: np.random.Generator,
    trans_sigma: float = 0.01,
    rot_sigma: float = 0.01,
) -> MissionResult:
    """Fly the mission and collect a rotating scan at each anchor index."""
    truth = integrate(start, steps)
    odom = integrate(start, steps, rng=rng, trans_sigma=trans_sigma, rot_sigma=rot_sigma)

    scans = []
    for anchor in scan_anchors:
        # Build the 20-frame scan: sense from the true poses, tag with odom poses.
        frames = [
            AugmentedPose(k, float(k), odom[k], sensor.measure(truth[k], world, rng=rng))
            for k in range(anchor, anchor + scan_frames)
        ]
        cloud = scan_point_cloud(build_scan(drone_id, frames), sensor.bearings)
        scans.append(ScanRecord(drone_id, anchor, odom[anchor], cloud))

    return MissionResult(truth, odom, scans)
