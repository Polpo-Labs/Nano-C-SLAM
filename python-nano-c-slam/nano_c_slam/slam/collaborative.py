"""
Decoupled collaborative optimization (R6).

The paper's multi-drone scheme is *decoupled and cascaded*: each drone optimizes
only its own poses, using the poses of lower-ID drones as FIXED anchors. Drone 0
optimizes first (alone); then drone 1 aligns its map to drone 0; then drone 2 to
drones 0-1; and so on. Because the external poses never move, each drone stays
globally consistent without a joint optimization.

Here we implement the mathematically-equivalent "fix the external nodes" version:
we drop the referenced external poses into the drone's own graph as extra, fixed
nodes and add the inter-drone loop closures as edges to them. (The paper's Eq.
2-4 "virtual node" trick avoids adding the nodes at all, for efficiency -- a
convergence step we can take later; it gives the same optimum.)

Reuses slam.pose_graph.optimize_poses (DRY).
"""

from __future__ import annotations

import numpy as np

from ..core.types import Edge, EdgeType, Pose2D
from .pose_graph import optimize_poses


def optimize_against_reference(
    own_poses: list[Pose2D],
    own_edges: list[Edge],
    reference_poses: dict[int, Pose2D],
    inter_closures: list[tuple[int, int, Pose2D, np.ndarray]],
    own_fixed: tuple[int, ...] = (0,),
) -> list[Pose2D]:
    """Optimize a drone's own poses while holding external reference poses fixed.

    own_poses      : the drone's current pose estimates (ids 0..n-1).
    own_edges      : constraints among own poses (odometry + intra-drone closures).
    reference_poses: {ref_id -> fixed pose} from a lower-ID drone (already optimized).
    inter_closures : list of (own_id, ref_id, measurement, information); each is an
                     inter-drone loop-closure edge from the fixed reference pose
                     `ref_id` to the drone's own pose `own_id`.
    own_fixed      : the drone's own anchored pose(s) (its take-off pose).

    Returns the corrected own poses (the reference poses are dropped from the
    result -- they only served as anchors).
    """
    n = len(own_poses)
    # Give each referenced external pose an index just past the own poses.
    used_refs = list(dict.fromkeys(ref_id for _, ref_id, _, _ in inter_closures))
    ref_index = {ref_id: n + i for i, ref_id in enumerate(used_refs)}

    combined = list(own_poses) + [reference_poses[ref_id] for ref_id in used_refs]
    edges = list(own_edges) + [
        Edge(ref_index[ref_id], own_id, measurement, information, EdgeType.INTER_LC)
        for own_id, ref_id, measurement, information in inter_closures
    ]
    # Anchor the drone's own take-off pose AND every external reference pose.
    fixed = tuple(set(own_fixed) | set(ref_index.values()))

    corrected = optimize_poses(combined, edges, fixed_ids=fixed)
    return corrected[:n]
