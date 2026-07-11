"""
Core data structures that flow through the Nano-C-SLAM pipeline.

These are deliberately *plain data* (dataclasses): they carry values and do no
heavy logic, so any component can create/inspect them in isolation during
tests. The numbers here mirror the paper's data model (Section VI):

  - 4 ToF depth sensors, mounted front / back / left / right.
  - Each 8x8 sensor is reduced to a single row of 8 range values (median per
    column), giving 4 x 8 = 32 range measurements per depth frame.
  - A pose is (x, y, heading); a pose + its depth frame is an "augmented pose".
  - A "scan" is 20 consecutive augmented poses aggregated at a texture-rich
    spot, used for scan-matching / loop closure.

Geometry operations on Pose2D live in `geometry.py` (kept separate so this
module stays dependency-free data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# --- Sensor layout constants (match the hardware in the paper) ---------------

NUM_SENSORS: int = 4            # ToF sensors: front, back, left, right
PIXELS_PER_SENSOR: int = 8     # columns kept after reducing each 8x8 frame to a row
RANGES_PER_FRAME: int = NUM_SENSORS * PIXELS_PER_SENSOR  # = 32 range values

SCAN_LENGTH: int = 20          # augmented poses aggregated into one scan


@dataclass
class Pose2D:
    """A 2D rigid-body pose in the world frame: position + heading.

    x, y   : position in metres.
    psi    : heading angle in radians (see geometry.wrap_angle for range).
    """

    x: float
    y: float
    psi: float

    def as_array(self) -> np.ndarray:
        """Return the pose as a length-3 numpy array [x, y, psi] for math ops."""
        return np.array([self.x, self.y, self.psi], dtype=float)


@dataclass
class DepthFrame:
    """One synchronized reading from all four ToF sensors, already reduced to 2D.

    ranges : (NUM_SENSORS, PIXELS_PER_SENSOR) array of distances in metres, i.e.
             one 8-value row per sensor. Row order is front, back, left, right.
    valid  : same-shape boolean mask; False marks a pixel the sensor flagged as
             noisy or out of range (the VL53 chips report this per pixel).
    """

    ranges: np.ndarray
    valid: np.ndarray


@dataclass
class AugmentedPose:
    """A pose paired with the depth frame captured at that pose (Section VI).

    pose_id   : unique, monotonically increasing index within a drone's graph.
    timestamp : acquisition time in seconds.
    pose      : the (x, y, psi) estimate at acquisition time.
    depth     : the 32 range measurements captured there.
    """

    pose_id: int
    timestamp: float
    pose: Pose2D
    depth: DepthFrame


@dataclass
class Scan:
    """A group of SCAN_LENGTH consecutive augmented poses used for matching.

    A scan gives a ~360 deg local view of a texture-rich spot (the drone rotates
    while acquiring it) and is the unit that ICP aligns for loop closure.

    drone_id : which drone captured the scan (needed for inter-drone matching).
    anchor_id: pose_id of the first augmented pose -- the pose the resulting
               loop-closure transform is applied to.
    frames   : the SCAN_LENGTH augmented poses making up the scan.
    """

    drone_id: int
    anchor_id: int
    frames: list[AugmentedPose] = field(default_factory=list)


class EdgeType(Enum):
    """What kind of constraint a pose-graph edge represents."""

    ODOMETRY = "odometry"      # between consecutive poses, from the state estimator
    INTRA_LC = "intra_lc"      # loop closure between two scans of the same drone
    INTER_LC = "inter_lc"      # loop closure between scans of two different drones


@dataclass
class Edge:
    """A constraint in the pose graph: a relative measurement between two poses.

    from_id, to_id : pose ids the edge connects.
    measurement    : the observed relative pose z_{i,j} (`to` seen from `from`).
    information     : 3x3 weight matrix (inverse covariance) -- how much the
                      optimizer trusts this edge, per Eq. 1 of the paper.
    edge_type      : odometry / intra- / inter-drone loop closure.
    """

    from_id: int
    to_id: int
    measurement: Pose2D
    information: np.ndarray
    edge_type: EdgeType
