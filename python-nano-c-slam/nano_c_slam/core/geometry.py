"""
SE(2) geometry: the 2D rigid-body math shared across the whole project.

A rigid-body pose in the plane is (x, y, psi): a 2D position plus a heading
angle in radians. We represent every pose/transform as a 3x3 homogeneous
matrix, because then "chain two transforms" is just a matrix product and
"invert a transform" is just a matrix inverse -- no hand-rolled trig scattered
around the codebase.

This single module is the *only* place that knows how a pose maps to a matrix.
ICP alignment, pose-graph edge errors, and the external-pose loop-closure trick
(Eq. 2-4 of the paper) all reuse these functions instead of re-deriving them.

Convention: `pose_to_matrix(a) @ pose_to_matrix(b)` composes "apply b in a's
local frame", i.e. it is the world pose you get by starting at pose `a` and then
moving by the relative transform `b`.
"""

from __future__ import annotations

import numpy as np

from .types import Pose2D


def wrap_angle(angle: float) -> float:
    """Wrap an angle (radians) into the half-open range [-pi, pi).

    Headings and heading *differences* must be normalized before use, otherwise
    e.g. a +179 deg and a -179 deg heading look 358 deg apart instead of 2 deg.
    The exact boundary (which side pi lands on) is irrelevant downstream because
    angles are only ever compared modulo 2*pi.
    """
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def pose_to_matrix(pose: Pose2D) -> np.ndarray:
    """Convert a pose (x, y, psi) to its 3x3 homogeneous transform matrix.

    The matrix is [[cos, -sin, x], [sin, cos, y], [0, 0, 1]]: a rotation by the
    heading `psi` followed by a translation to (x, y).
    """
    c, s = np.cos(pose.psi), np.sin(pose.psi)
    return np.array(
        [
            [c, -s, pose.x],
            [s, c, pose.y],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def matrix_to_pose(matrix: np.ndarray) -> Pose2D:
    """Convert a 3x3 homogeneous transform back to a pose (x, y, psi).

    The heading is recovered from the rotation block with atan2 so it stays a
    proper angle, and is wrapped into (-pi, pi].
    """
    x = float(matrix[0, 2])
    y = float(matrix[1, 2])
    psi = wrap_angle(float(np.arctan2(matrix[1, 0], matrix[0, 0])))
    return Pose2D(x, y, psi)


def invert(pose: Pose2D) -> Pose2D:
    """Return the inverse transform of a pose (undoes it)."""
    return matrix_to_pose(np.linalg.inv(pose_to_matrix(pose)))


def compose(a: Pose2D, b: Pose2D) -> Pose2D:
    """Chain two transforms: start at `a`, then apply `b` in a's local frame.

    Equivalent to the matrix product A @ B. Used e.g. to integrate an odometry
    step onto the current pose.
    """
    return matrix_to_pose(pose_to_matrix(a) @ pose_to_matrix(b))


def relative(a: Pose2D, b: Pose2D) -> Pose2D:
    """Pose of `b` expressed in the frame of `a` (i.e. inv(A) @ B).

    This is exactly the relative measurement z_{a,b} that a pose-graph edge
    stores, and the prediction the optimizer compares against. Reused by both
    odometry-edge construction and ICP-based loop-closure edges.
    """
    return matrix_to_pose(np.linalg.inv(pose_to_matrix(a)) @ pose_to_matrix(b))
