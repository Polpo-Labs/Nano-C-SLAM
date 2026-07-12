"""
Messages drones exchange over the radio (R7).

For collaborative SLAM a drone only needs to send two things to the others:
  - the scans it acquires (so others can loop-close against them), and
  - the updated poses of those scans after it optimizes (so others can refresh
    their inter-drone constraints).

Each message carries a `size_bytes` so the medium can tally data volume for the
scalability study (R9). The sizes follow the paper's data model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.types import SCAN_LENGTH, Pose2D

# Data-volume model from the paper (used only for communication accounting):
POSE_BYTES = 12                                   # a pose = 3 float values
AUGMENTED_POSE_BYTES = 84                          # pose + 32 int16 depths + id + timestamp
SCAN_BYTES = SCAN_LENGTH * AUGMENTED_POSE_BYTES    # a scan = 20 augmented poses (~1.7 KB)


@dataclass
class ScanMessage:
    """A drone broadcasting one acquired scan to higher-ID drones (Stage 3)."""

    drone_id: int
    anchor_id: int
    anchor_pose: Pose2D
    cloud: np.ndarray
    size_bytes: int = SCAN_BYTES


@dataclass
class PoseUpdateMessage:
    """The updated value of a scan's anchor pose, sent after PGO (Stage 5)."""

    drone_id: int
    anchor_id: int
    new_pose: Pose2D
    size_bytes: int = POSE_BYTES
