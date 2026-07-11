"""
Scans: the paper's unit of loop closure (R5).

A single depth frame from the four ToF sensors covers only 180 deg, with 45 deg
gaps between the sensors. So the drone aggregates SCAN_LENGTH (20) consecutive
frames acquired *while rotating*: the rotation sweeps the beams through the gaps,
and the combined frames give a ~360 deg local view of a place -- a "scan".

To combine the frames into one point cloud we express every frame's hits in the
frame of the scan's anchor (its first pose). Each later frame is offset from the
anchor by the little bit the drone moved/rotated during acquisition, which we
read straight from the (estimated) poses. The result is the point cloud ICP uses
for scan-matching.

Reuses core.geometry and slam.mapping (DRY).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.geometry import relative, transform_points
from ..core.types import AugmentedPose, Pose2D, Scan
from .mapping import depth_local_points


@dataclass
class ScanRecord:
    """An aggregated scan ready for matching: its cloud plus where it was taken.

    drone_id    : which drone acquired it (needed for inter- vs intra-matching).
    anchor_id   : pose id of the scan's anchor within that drone's graph.
    anchor_pose : the drone's *estimated* anchor pose (for proximity + ICP init).
    cloud       : the (N, 2) aggregated scan points, in the anchor's body frame.
    """

    drone_id: int
    anchor_id: int
    anchor_pose: Pose2D
    cloud: np.ndarray


def build_scan(drone_id: int, frames: list[AugmentedPose]) -> Scan:
    """Group consecutive augmented poses into a Scan anchored at the first one."""
    return Scan(drone_id=drone_id, anchor_id=frames[0].pose_id, frames=list(frames))


def scan_point_cloud(scan: Scan, bearings: np.ndarray) -> np.ndarray:
    """Aggregate a scan's frames into one point cloud, in the anchor's frame.

    Every frame's hits start in that frame's own body frame; we move them into
    the anchor frame using the relative pose between the anchor and that frame,
    so the rotation during acquisition lines the frames up into a full sweep.
    """
    anchor_pose = scan.frames[0].pose
    clouds = []
    for frame in scan.frames:
        local = depth_local_points(frame.depth, bearings)
        # Pose of this frame as seen from the anchor -> where its points belong.
        offset = relative(anchor_pose, frame.pose)
        clouds.append(transform_points(offset, local))
    return np.vstack(clouds) if clouds else np.empty((0, 2))
