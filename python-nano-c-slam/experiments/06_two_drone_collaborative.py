"""
R6 demo: two drones map a room together (inter-drone loop closure).

Drone 0 maps the left half, drone 1 the right half; their squares share the two
corners at x=2, so each drone scans places the other also scanned. Pipeline:

  1. Each drone flies its mission and collects rotating scans (R5).
  2. Each drone finds its own (intra) loop closures by scan-matching (R4).
  3. Cascaded optimization: drone 0 optimizes alone; then drone 1 aligns to
     drone 0 via inter-drone loop closures, with drone 0's poses held fixed
     (the paper's decoupled scheme).

Outputs:
  * a before/after map (raw odometry vs. collaboratively optimized), and
  * a GIF of the two drones flying with their odometry trails drifting.

Run:  ../.venv/Scripts/python.exe experiments/06_two_drone_collaborative.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from nano_c_slam.core.geometry import transform_points
from nano_c_slam.core.types import Pose2D
from nano_c_slam.slam.collaborative import optimize_against_reference
from nano_c_slam.slam.loop_closure import anchors_within, derive_closure
from nano_c_slam.slam.pose_graph import PoseGraph
from nano_c_slam.slam.scan import ScanRecord
from nano_c_slam.sim.mission import simulate_mission
from nano_c_slam.sim.sensor import QuadToFSensor
from nano_c_slam.sim.trajectories import square_mission
from nano_c_slam.sim.world import World
from nano_c_slam.viz.animation import save_gif

DRONE_COLORS = ["tab:blue", "tab:orange"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def own_graph(mission) -> PoseGraph:
    """Build a drone's own pose graph: odometry chain + intra-drone loop closures."""
    graph = PoseGraph(mission.odom)
    graph.add_odometry_chain()
    for i, scan_i in enumerate(mission.scans):
        for scan_j in mission.scans[i + 1:]:
            if not anchors_within(scan_i, scan_j):
                continue
            z, error = derive_closure(scan_j, scan_i)  # z = pose j seen from i
            if z is not None:
                graph.add_edge(scan_i.anchor_id, scan_j.anchor_id, z, np.eye(3) * 10.0)
    return graph


def inter_closures(own_scans, ref_scans_corrected):
    """Inter-drone loop closures: match own scans to a reference drone's scans.

    Returns (closures, reference_poses) for optimize_against_reference.
    """
    closures = []
    reference_poses = {}
    for own in own_scans:
        for ref in ref_scans_corrected:
            if not anchors_within(own, ref):
                continue
            z, error = derive_closure(own, ref)  # z = own anchor seen from ref anchor
            if z is not None:
                closures.append((own.anchor_id, ref.anchor_id, z, np.eye(3) * 10.0))
                reference_poses[ref.anchor_id] = ref.anchor_pose
    return closures, reference_poses


def corrected_scans(scans, corrected_poses):
    """Copies of scan records whose anchor pose is replaced by the optimized one."""
    return [ScanRecord(s.drone_id, s.anchor_id, corrected_poses[s.anchor_id], s.cloud) for s in scans]


def scan_map(scans, poses_by_id):
    """World-frame point cloud from scans placed at the given anchor poses."""
    clouds = [transform_points(poses_by_id[s.anchor_id], s.cloud) for s in scans]
    return np.vstack(clouds) if clouds else np.empty((0, 2))


def main() -> None:
    world = World.rectangle(-1.5, -1.5, 5.5, 3.5).with_segments(
        [(1.0, 3.5, 1.0, 2.7), (3.8, -1.5, 3.8, -0.7)]  # asymmetric features for ICP
    )
    sensor = QuadToFSensor(max_range=12.0, noise_sigma=0.01)
    steps, anchors = square_mission(side=2.0, laps=2, increments=15, scan_frames=20)

    # Two drones with known take-off poses; drone 1 starts one square to the right.
    m0 = simulate_mission(0, Pose2D(0, 0, 0), steps, anchors, 20, world, sensor, np.random.default_rng(1))
    m1 = simulate_mission(1, Pose2D(2, 0, 0), steps, anchors, 20, world, sensor, np.random.default_rng(2))

    # --- Cascaded optimization: drone 0 first, then drone 1 aligns to it ---
    corrected0 = own_graph(m0).optimize(fixed_ids=(0,))
    poses0_by_id = {s.anchor_id: corrected0[s.anchor_id] for s in m0.scans}

    g1 = own_graph(m1)
    closures, reference_poses = inter_closures(m1.scans, corrected_scans(m0.scans, poses0_by_id))
    corrected1 = optimize_against_reference(g1.poses, g1.edges, reference_poses, closures)
    print(f"inter-drone loop closures found: {len(closures)}")

    # --- Static before/after map ---
    _plot_before_after(world, m0, m1, corrected0, corrected1)
    # --- Runtime GIF ---
    _animate(world, m0, m1)


def _plot_before_after(world, m0, m1, corrected0, corrected1):
    odom0_by_id = {s.anchor_id: s.anchor_pose for s in m0.scans}
    odom1_by_id = {s.anchor_id: s.anchor_pose for s in m1.scans}
    corr0_by_id = {s.anchor_id: corrected0[s.anchor_id] for s in m0.scans}
    corr1_by_id = {s.anchor_id: corrected1[s.anchor_id] for s in m1.scans}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, title, p0, p1 in [
        (ax1, "Raw odometry (maps misaligned)", odom0_by_id, odom1_by_id),
        (ax2, "After collaborative optimization", corr0_by_id, corr1_by_id),
    ]:
        for x1, y1, x2, y2 in world.walls:
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.5)
        for mission, poses, color in [(m0, p0, DRONE_COLORS[0]), (m1, p1, DRONE_COLORS[1])]:
            cloud = scan_map(mission.scans, poses)
            ax.scatter(cloud[:, 0], cloud[:, 1], s=2, alpha=0.2, color=color)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.scatter([], [], color=DRONE_COLORS[0], label="drone 0")
    ax1.scatter([], [], color=DRONE_COLORS[1], label="drone 1")
    ax1.legend(loc="upper right", fontsize=8)
    fig.suptitle("R6: two drones build one merged map via inter-drone loop closure")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "06_two_drone_map.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved map -> {out}")


def _animate(world, m0, m1):
    """A runtime GIF: both drones (squares) flying, odometry trails drifting."""
    missions = [m0, m1]
    n_poses = len(m0.truth)
    n_frames = 90
    stride = max(1, n_poses // n_frames)

    fig, ax = plt.subplots(figsize=(7, 6))

    def update(frame):
        ax.clear()
        t = min(frame * stride, n_poses - 1)
        for x1, y1, x2, y2 in world.walls:
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.5)
        for mission, color in zip(missions, DRONE_COLORS):
            odom = mission.odom
            ax.plot([p.x for p in odom[: t + 1]], [p.y for p in odom[: t + 1]],
                    color=color, linewidth=1, alpha=0.8)
            ax.plot(mission.truth[t].x, mission.truth[t].y, marker="s",
                    color=color, markersize=12)
        ax.set_aspect("equal")
        ax.set_xlim(-2, 6)
        ax.set_ylim(-2, 4)
        ax.set_title(f"R6 runtime: two drones mapping (t={t})")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "06_two_drone_runtime.gif")
    save_gif(fig, update, n_frames, out, fps=15)
    print(f"saved gif -> {out}")


if __name__ == "__main__":
    main()
