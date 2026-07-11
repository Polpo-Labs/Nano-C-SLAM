"""
Generic point-to-point ICP scan-matching (R4).

ICP (Iterative Closest Point) finds the rigid transform (rotation + translation)
that best overlays one point cloud (`source`) onto another (`target`). It is how
we *discover* a loop closure: when a robot revisits a place, aligning the new
scan with the old one yields the relative pose between the two -- a constraint we
feed to the pose graph.

The algorithm, at its simplest (which is where we start):
  1. For every source point, find its nearest target point (the "correspondence").
  2. Compute the best-fit rigid transform between those matched pairs (SVD/Kabsch).
  3. Apply it to the source and repeat, until the alignment stops improving.

Correspondences are unknown up front, so a rough initial guess (e.g. from
odometry) helps ICP start close and avoid bad local minima. We keep this generic;
converging to the paper's specifics (their 2D point-to-point variant, rejection
filters, 640-point scans) comes later.

SE(2) conversions are reused from core.geometry (DRY).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..core.geometry import matrix_to_pose, pose_to_matrix
from ..core.types import Pose2D


def _best_rigid_transform(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid transform (R, t) mapping matched points P onto Q.

    Classic Kabsch/Umeyama solution: centre both clouds, take the SVD of their
    cross-covariance, and read the rotation off it (with a reflection guard so we
    always get a proper rotation, never a mirror).
    """
    mu_p, mu_q = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - mu_p, Q - mu_q
    covariance = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(covariance)
    V = Vt.T
    # If det(V U^T) is negative the naive solution is a reflection; flip it.
    reflection = np.sign(np.linalg.det(V @ U.T))
    R = V @ np.diag([1.0, reflection]) @ U.T
    t = mu_q - R @ mu_p
    return R, t


def _homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pack a rotation R (2x2) and translation t (2,) into a 3x3 transform."""
    M = np.eye(3)
    M[:2, :2] = R
    M[:2, 2] = t
    return M


def icp(
    source: np.ndarray,
    target: np.ndarray,
    init: Pose2D | None = None,
    max_iterations: int = 50,
    tol: float = 1e-6,
    max_correspondence_dist: float = np.inf,
) -> tuple[Pose2D, np.ndarray, float, int]:
    """Align `source` onto `target` and return the transform between them.

    source, target : (N, 2) / (M, 2) point clouds.
    init           : optional initial guess for the transform (e.g. from odometry).
    max_correspondence_dist : ignore matched pairs farther apart than this (a
                     simple outlier reject; inf = keep all).

    Returns (transform, aligned_source, error, iterations):
      transform      : Pose2D mapping the ORIGINAL source points onto target.
      aligned_source : the source points after applying `transform`.
      error          : mean nearest-neighbour distance at the final alignment.
      iterations     : how many ICP steps ran.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    # Accumulated transform (as a 3x3 matrix), seeded with the initial guess.
    total = pose_to_matrix(init) if init is not None else np.eye(3)
    P = (total[:2, :2] @ source.T).T + total[:2, 2]

    tree = cKDTree(target)  # target is fixed, so index it once
    error = float("inf")
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        distances, idx = tree.query(P)
        keep = distances < max_correspondence_dist
        if keep.sum() < 3:  # too few matches to fit a transform reliably
            break

        R, t = _best_rigid_transform(P[keep], target[idx[keep]])
        P = (R @ P.T).T + t
        total = _homogeneous(R, t) @ total

        new_error = float(distances[keep].mean())
        if abs(error - new_error) < tol:  # alignment stopped improving
            break
        error = new_error

    # Report the mean correspondence distance at the final alignment.
    final_distances, _ = tree.query(P)
    return matrix_to_pose(total), P, float(final_distances.mean()), iterations
