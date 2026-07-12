"""
A drone as a distributed collaborative-SLAM node (R7).

Each agent knows only its own mission (its odometry estimate and the scans it
acquired) plus whatever arrives over the medium from lower-ID drones. It:
  - buffers external scans it receives,
  - refreshes their anchor poses when pose-updates arrive,
  - scan-matches its own scans against its own (intra) and against external ones
    (inter) to form loop closures, and
  - optimizes its own graph with the external poses held FIXED (the decoupled
    cascaded scheme from R6) -- now driven entirely by messages, no god-view.

Reuses the R4-R6 building blocks (ICP loop closures, decoupled optimization);
this class is just the message-driven coordination around them.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Pose2D
from ..slam.collaborative import optimize_against_reference
from ..slam.loop_closure import anchors_within, derive_closure
from ..slam.pose_graph import PoseGraph
from ..slam.scan import ScanRecord
from .messages import PoseUpdateMessage, ScanMessage

# Trust placed on loop-closure edges relative to odometry (same as R6).
LC_INFORMATION = np.eye(3) * 10.0


class DroneAgent:
    """One drone's distributed collaborative-SLAM state and behaviour."""

    def __init__(self, drone_id: int, odom_poses: list[Pose2D]) -> None:
        self.drone_id = drone_id
        self.poses = list(odom_poses)              # current estimate (starts = odometry)
        self.own_scans: list[ScanRecord] = []      # scans this drone acquired
        self.external_scans: list[ScanRecord] = [] # scans received from lower-ID drones

    # --- Acquiring and broadcasting own data ---

    def acquire(self, scan: ScanRecord) -> None:
        """Record a scan this drone captured."""
        self.own_scans.append(scan)

    def scan_messages(self) -> list[ScanMessage]:
        """One broadcast per own scan (Stage 3)."""
        return [ScanMessage(self.drone_id, s.anchor_id, s.anchor_pose, s.cloud) for s in self.own_scans]

    def pose_update_messages(self) -> list[PoseUpdateMessage]:
        """The updated anchor pose of each own scan after optimizing (Stage 5)."""
        return [PoseUpdateMessage(self.drone_id, s.anchor_id, self.poses[s.anchor_id]) for s in self.own_scans]

    # --- Receiving others' data ---

    def receive(self, message) -> None:
        """Buffer an external scan, or refresh a buffered scan's anchor pose."""
        if isinstance(message, ScanMessage):
            self.external_scans.append(
                ScanRecord(message.drone_id, message.anchor_id, message.anchor_pose, message.cloud)
            )
        elif isinstance(message, PoseUpdateMessage):
            for ext in self.external_scans:
                if ext.drone_id == message.drone_id and ext.anchor_id == message.anchor_id:
                    ext.anchor_pose = message.new_pose

    # --- Optimization ---

    def _own_graph(self) -> PoseGraph:
        """This drone's graph: odometry chain + its own intra-drone loop closures."""
        graph = PoseGraph(self.poses)
        graph.add_odometry_chain()
        for i, scan_i in enumerate(self.own_scans):
            for scan_j in self.own_scans[i + 1:]:
                if not anchors_within(scan_i, scan_j):
                    continue
                z, _ = derive_closure(scan_j, scan_i)  # z = pose j seen from i
                if z is not None:
                    graph.add_edge(scan_i.anchor_id, scan_j.anchor_id, z, LC_INFORMATION)
        return graph

    def optimize(self) -> list[Pose2D]:
        """Correct own poses, aligning to lower-ID drones via external scans."""
        graph = self._own_graph()

        # Inter-drone loop closures against buffered external scans. External
        # references are keyed by (drone_id, anchor_id) so several drones' scans
        # never collide -- optimize_against_reference treats the key opaquely.
        closures = []
        references: dict[tuple[int, int], Pose2D] = {}
        for own in self.own_scans:
            for ext in self.external_scans:
                if not anchors_within(own, ext):
                    continue
                z, _ = derive_closure(own, ext)  # z = own anchor seen from ext anchor
                if z is not None:
                    key = (ext.drone_id, ext.anchor_id)
                    closures.append((own.anchor_id, key, z, LC_INFORMATION))
                    references[key] = ext.anchor_pose

        if closures:
            self.poses = optimize_against_reference(graph.poses, graph.edges, references, closures)
        else:
            self.poses = graph.optimize(fixed_ids=(0,))
        return self.poses
