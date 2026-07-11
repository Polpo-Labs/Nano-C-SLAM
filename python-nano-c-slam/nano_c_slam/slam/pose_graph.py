"""
Generic pose-graph optimization (R2) -- the fundamental SLAM 'solution'.

The idea: we have noisy pose estimates (from drifting odometry) and a set of
constraints between poses (edges). Each edge says "pose j, seen from pose i,
should look like this relative measurement z". Odometry edges are satisfied by
the initial guess, but a *loop-closure* edge (e.g. "you are back where you
started") disagrees with the drifted estimate. Optimization nudges all the poses
to best satisfy every constraint at once, spreading the loop-closure correction
back along the trajectory -- which straightens out the drift.

We solve it with plain Gauss-Newton least squares:

    minimize  sum_edges  e(x_i, x_j)^T  Omega  e(x_i, x_j)

where e is the difference between the measured and predicted relative pose, and
Omega is the edge's information (inverse-covariance) weight.

This is deliberately generic and simple:
  * We parameterize each pose directly by (x, y, psi) and use *numerical*
    Jacobians (finite differences). They are trivially correct -- no error-prone
    hand-derived SE(2) Jacobians -- and plenty fast for the graph sizes we test.
  * We use a small dense linear solve. Both choices are things we can later
    converge toward the paper's efficient method (analytic Jacobians, sparse
    Cholesky, hierarchical splitting) once this generic version is trusted.

All the SE(2) math is reused from core.geometry (DRY).
"""

from __future__ import annotations

import numpy as np

from ..core.geometry import relative, wrap_angle
from ..core.types import Edge, EdgeType, Pose2D

# Small step used for the finite-difference Jacobians.
_EPS = 1e-6
# Weight forcing an "anchored" pose to stay put (added to its H diagonal block).
_ANCHOR_WEIGHT = 1e12


def _pose_error(xi: Pose2D, xj: Pose2D, z: Pose2D) -> np.ndarray:
    """Residual (3-vector) between the measured and predicted relative pose.

    predicted = pose of j seen from i = x_i^-1 . x_j  (core.geometry.relative).
    The error is that prediction expressed relative to the measurement z, i.e.
    z^-1 . predicted, which is the zero pose exactly when they agree.
    """
    predicted = relative(xi, xj)
    residual = relative(z, predicted)
    return np.array([residual.x, residual.y, residual.psi])


def _with_component(pose: Pose2D, k: int, delta: float) -> Pose2D:
    """Return a copy of `pose` with its k-th parameter (x=0, y=1, psi=2) nudged."""
    values = [pose.x, pose.y, pose.psi]
    values[k] += delta
    return Pose2D(*values)


def _edge_jacobians(
    xi: Pose2D, xj: Pose2D, z: Pose2D
) -> tuple[np.ndarray, np.ndarray]:
    """Numerical Jacobians of the edge error w.r.t. pose i (A) and pose j (B).

    Column k of A is how the 3-element error changes when parameter k of pose i
    moves, estimated by a central finite difference (and likewise B for pose j).
    """
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    for k in range(3):
        A[:, k] = (
            _pose_error(_with_component(xi, k, _EPS), xj, z)
            - _pose_error(_with_component(xi, k, -_EPS), xj, z)
        ) / (2 * _EPS)
        B[:, k] = (
            _pose_error(xi, _with_component(xj, k, _EPS), z)
            - _pose_error(xi, _with_component(xj, k, -_EPS), z)
        ) / (2 * _EPS)
    return A, B


def optimize_poses(
    poses: list[Pose2D],
    edges: list[Edge],
    fixed_ids: tuple[int, ...] = (0,),
    max_iterations: int = 50,
    tol: float = 1e-9,
) -> list[Pose2D]:
    """Optimize a pose graph and return the corrected poses (input left untouched).

    poses      : current estimate, indexed by pose id (0..n-1).
    edges      : constraints between poses (odometry and loop closures).
    fixed_ids  : poses to anchor (never move) -- gauge freedom must be pinned, and
                 the paper anchors each drone's first pose. Defaults to pose 0.
    max_iterations / tol : Gauss-Newton stopping criteria (on the update size).
    """
    x = list(poses)  # working copy -- we never mutate the caller's list
    n = len(x)
    fixed = set(fixed_ids)

    for _ in range(max_iterations):
        # H (approx. Hessian) and b (gradient) of the least-squares problem.
        H = np.zeros((3 * n, 3 * n))
        b = np.zeros(3 * n)

        for edge in edges:
            i, j = edge.from_id, edge.to_id
            e = _pose_error(x[i], x[j], edge.measurement)
            A, B = _edge_jacobians(x[i], x[j], edge.measurement)
            omega = edge.information

            bi, bj = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
            # Accumulate this edge's contribution into the normal equations.
            H[bi, bi] += A.T @ omega @ A
            H[bi, bj] += A.T @ omega @ B
            H[bj, bi] += B.T @ omega @ A
            H[bj, bj] += B.T @ omega @ B
            b[bi] += A.T @ omega @ e
            b[bj] += B.T @ omega @ e

        # Anchor fixed poses: a huge weight on their block makes their update ~0.
        for k in fixed:
            H[3 * k : 3 * k + 3, 3 * k : 3 * k + 3] += _ANCHOR_WEIGHT * np.eye(3)

        # Gauss-Newton step: solve H . delta = -b.
        delta = np.linalg.solve(H, -b)

        # Apply the increment to every pose (heading wrapped to stay a valid angle).
        for k in range(n):
            d = delta[3 * k : 3 * k + 3]
            x[k] = Pose2D(x[k].x + d[0], x[k].y + d[1], wrap_angle(x[k].psi + d[2]))

        if np.max(np.abs(delta)) < tol:
            break

    return x


class PoseGraph:
    """Thin, readable container around `optimize_poses`.

    Holds the poses (indexed by id) and the edges, and lets experiments build a
    graph with `add_edge(...)` then `optimize()`. All the math lives in
    `optimize_poses`, so the algorithm stays testable without this class.
    """

    def __init__(self, poses: list[Pose2D]) -> None:
        self.poses = list(poses)
        self.edges: list[Edge] = []

    def add_edge(
        self,
        from_id: int,
        to_id: int,
        measurement: Pose2D,
        information: np.ndarray | None = None,
        edge_type: EdgeType = EdgeType.ODOMETRY,
    ) -> None:
        """Add one constraint. `information` defaults to the identity (equal trust)."""
        if information is None:
            information = np.eye(3)
        self.edges.append(Edge(from_id, to_id, measurement, information, edge_type))

    def optimize(self, fixed_ids: tuple[int, ...] = (0,), **kwargs) -> list[Pose2D]:
        """Optimize in place and return the corrected poses."""
        self.poses = optimize_poses(self.poses, self.edges, fixed_ids, **kwargs)
        return self.poses
