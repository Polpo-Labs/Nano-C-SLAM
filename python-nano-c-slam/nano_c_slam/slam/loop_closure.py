"""
Deriving loop-closure constraints from scans (R6).

A loop closure is proposed when two scans were taken close together (they might
be the same place). We scan-match them with ICP, seeded by the relative pose of
their anchors, and keep the result only if the alignment is good. This works the
same whether the two scans belong to one drone (intra) or two drones (inter).

The acceptance test converges toward the paper: it rejects a match whose mean
point-to-point error after ICP exceeds ~10 cm (the paper's threshold for "the
scans are too noisy or not the same place"). The paper additionally rejects
matches with a large relative rotation -- a filter we can add later.
"""

from __future__ import annotations

import numpy as np

from ..core.geometry import relative
from ..core.types import Pose2D
from .icp import icp
from .scan import ScanRecord

# Paper's rejection threshold: mean post-ICP point distance above this = bad match.
MAX_MATCH_ERROR = 0.10  # metres
# Only consider scans whose anchors are at least this close as the same place.
MATCH_DISTANCE = 1.0  # metres


def anchors_within(a: ScanRecord, b: ScanRecord, distance: float = MATCH_DISTANCE) -> bool:
    """True if two scans' estimated anchor positions are within `distance`."""
    return float(np.hypot(a.anchor_pose.x - b.anchor_pose.x, a.anchor_pose.y - b.anchor_pose.y)) < distance


def derive_closure(source: ScanRecord, target: ScanRecord, max_error: float = MAX_MATCH_ERROR):
    """Scan-match `source` onto `target`; return (measurement, error) or (None, error).

    The returned measurement is the relative pose of the source anchor as seen
    from the target anchor -- i.e. the loop-closure edge from target to source.
    ICP is seeded with the anchors' current relative estimate. Returns None for
    the measurement if the alignment error is above `max_error`.
    """
    init = relative(target.anchor_pose, source.anchor_pose)
    z, _, error, _ = icp(source.cloud, target.cloud, init=init)
    if error > max_error:
        return None, error
    return z, error
