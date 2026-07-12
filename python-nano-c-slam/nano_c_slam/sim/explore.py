"""
Closed-loop autonomous exploration (R8).

Where sim.mission flies a *scripted* path, this flies a drone under a reactive
controller: at each tick we sense, ask the controller for a body command, and
integrate the motion (true + noisy odometry). We also watch the controller's
state to catch the paper's corner-triggered scans: a spin lasting at least
`scan_frames` steps is a rotation-in-place at a corner, so we record the 20
frames leading up to the moment it stops spinning as a scan.

It returns the same MissionResult as sim.mission, so an explored mission plugs
straight into the collaborative / communication pipeline (R6-R7).
"""

from __future__ import annotations

from ..core.types import AugmentedPose, Pose2D
from ..exploration.controller import ExploreState
from ..slam.scan import ScanRecord, build_scan, scan_point_cloud
from .mission import MissionResult
from .motion import noisy_step, step
from .sensor import QuadToFSensor
from .world import World

DEFAULT_DT = 1.0 / 7.5  # seconds per pose (the paper's 7.5 Hz acquisition rate)


def explore_mission(
    drone_id: int,
    start: Pose2D,
    controller,
    world: World,
    sensor: QuadToFSensor,
    rng,
    n_steps: int,
    scan_frames: int = 20,
    dt: float = DEFAULT_DT,
    trans_sigma: float = 0.01,
    rot_sigma: float = 0.01,
) -> MissionResult:
    """Fly `n_steps` of autonomous exploration and collect corner scans."""
    true_pose = odom_pose = start
    truth = [start]
    odom = [start]
    frames: list[AugmentedPose] = []

    scan_anchors: list[int] = []
    spin_start: int | None = None
    prev_state = controller.state

    for k in range(n_steps):
        depth = sensor.measure(true_pose, world, rng=rng)
        frames.append(AugmentedPose(k, k * dt, odom_pose, depth))

        vx, vy, omega = controller.command(depth)

        # Track spin start/end to detect a corner scan (a long rotation in place).
        if controller.state == ExploreState.SPINNING and prev_state != ExploreState.SPINNING:
            spin_start = k
        elif controller.state == ExploreState.CRUISE and prev_state == ExploreState.SPINNING:
            if spin_start is not None and (k - spin_start) >= scan_frames:
                scan_anchors.append(k - scan_frames)  # the 20 frames ending the spin
        prev_state = controller.state

        body_step = (vx * dt, vy * dt, omega * dt)
        true_pose = step(true_pose, *body_step)
        odom_pose = noisy_step(odom_pose, *body_step, rng, trans_sigma, rot_sigma)
        truth.append(true_pose)
        odom.append(odom_pose)

    # Final frame so frames align with the pose lists (index == pose index).
    frames.append(AugmentedPose(n_steps, n_steps * dt, odom_pose, sensor.measure(true_pose, world, rng=rng)))

    # Build a rotating scan from the frames at each detected corner.
    scans = []
    for anchor in scan_anchors:
        cloud = scan_point_cloud(build_scan(drone_id, frames[anchor:anchor + scan_frames]), sensor.bearings)
        scans.append(ScanRecord(drone_id, anchor, odom[anchor], cloud))

    return MissionResult(truth, odom, scans)
