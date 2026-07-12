"""
R8 demo: one drone explores a room on its own and maps it.

No scripted path -- a reactive wall-following controller (Cruise/Spinning) drives
the drone from its ToF readings. It traces the room's perimeter, acquires a
rotating scan at each corner (the corner-triggered scans deferred from R5), and
on the second lap revisits those corners, giving loop closures that correct the
drift (SLAM via the single-drone DroneAgent path).

Outputs:
  * a before/after map (raw odometry vs. SLAM-corrected), with trajectories, and
  * a GIF of the drone exploring and its live map building up.

Run:  ../.venv/Scripts/python.exe experiments/08_autonomous_exploration.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from nano_c_slam.comms.agent import DroneAgent
from nano_c_slam.core.geometry import transform_points
from nano_c_slam.core.types import Pose2D
from nano_c_slam.exploration.controller import WallFollowController
from nano_c_slam.sim.explore import explore_mission
from nano_c_slam.sim.sensor import QuadToFSensor
from nano_c_slam.sim.world import World
from nano_c_slam.viz.animation import save_gif

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def scan_map(scans, poses):
    clouds = [transform_points(poses[s.anchor_id], s.cloud) for s in scans]
    return np.vstack(clouds) if clouds else np.empty((0, 2))


def draw_walls(ax, world):
    for x1, y1, x2, y2 in world.walls:
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.5)


def main() -> None:
    world = World.rectangle(0, 0, 5, 4).with_segments([(3.0, 4.0, 3.0, 3.0)])
    sensor = QuadToFSensor(max_range=4.0, noise_sigma=0.005)

    mission = explore_mission(0, Pose2D(1, 1, 0), WallFollowController(),
                              world, sensor, np.random.default_rng(0), n_steps=1200)

    # Single-drone SLAM: build the graph and optimize (reuse the agent path).
    agent = DroneAgent(0, mission.odom)
    for scan in mission.scans:
        agent.acquire(scan)
    corrected = agent.optimize()

    def err(est):
        return float(np.mean([np.hypot(e.x - t.x, e.y - t.y) for e, t in zip(est, mission.truth)]))
    print(f"scans acquired: {len(mission.scans)}   "
          f"mean pos error  odom {err(mission.odom):.3f} m  ->  corrected {err(corrected):.3f} m")

    # --- Static before/after map with trajectories ---
    odom_by_id = {s.anchor_id: s.anchor_pose for s in mission.scans}
    corr_by_id = {s.anchor_id: corrected[s.anchor_id] for s in mission.scans}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    for ax, title, path, poses in [
        (ax1, "Raw odometry", mission.odom, odom_by_id),
        (ax2, "After autonomous SLAM", corrected, corr_by_id),
    ]:
        draw_walls(ax, world)
        cloud = scan_map(mission.scans, poses)
        ax.scatter(cloud[:, 0], cloud[:, 1], s=3, alpha=0.25, color="tab:green")
        ax.plot([p.x for p in path], [p.y for p in path], color="tab:blue", linewidth=1, label="trajectory")
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    fig.suptitle("R8: one drone autonomously explores and maps a room")
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUT_DIR, "08_autonomous_exploration.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    _animate(world, mission)


def _animate(world, mission):
    """GIF of the drone exploring with its live (odometry) map building up."""
    n_poses = len(mission.truth)
    n_frames = 90
    stride = max(1, n_poses // n_frames)
    revealed = [(s.anchor_id + 20, transform_points(s.anchor_pose, s.cloud)[::3]) for s in mission.scans]

    fig, ax = plt.subplots(figsize=(7, 6))

    def update(frame):
        ax.clear()
        t = min(frame * stride, n_poses - 1)
        draw_walls(ax, world)
        shown = [pts for reveal_t, pts in revealed if reveal_t <= t]
        if shown:
            cloud = np.vstack(shown)
            ax.scatter(cloud[:, 0], cloud[:, 1], s=2, alpha=0.2, color="tab:green")
        ax.plot([p.x for p in mission.odom[: t + 1]], [p.y for p in mission.odom[: t + 1]],
                color="tab:blue", linewidth=1, alpha=0.8)
        ax.plot(mission.odom[t].x, mission.odom[t].y, marker="s", color="tab:blue", markersize=12)
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 4.5)
        ax.set_title(f"R8: autonomous exploration (t={t})")

    out = os.path.join(OUT_DIR, "08_autonomous_exploration.gif")
    save_gif(fig, update, n_frames, out, fps=15)
    print(f"saved -> {os.path.join(OUT_DIR, '08_autonomous_exploration.png')} and the GIF")


if __name__ == "__main__":
    main()
