"""
R7 demo: the same collaborative map, but built by distributed message passing.

R6 computed the two-drone map with god-view access to both drones' scans. Here
each drone is a DroneAgent that knows only its own data plus what it receives
over the Medium: drone 0 broadcasts its scans and (after optimizing) its updated
poses; drone 1 buffers them, scan-matches, and aligns itself to drone 0. The
result is identical to R6 (see tests/test_comms.py) -- but now it is produced by
communication, and we can measure how much data crossed the radio. That byte/
message tally is exactly what the scalability study (R9) will extrapolate to
hundreds of drones.

Run:  ../.venv/Scripts/python.exe experiments/07_distributed_comms.py
Saves a plot to experiments/output/07_distributed_comms.png.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from nano_c_slam.comms.swarm import run_swarm
from nano_c_slam.core.geometry import transform_points
from nano_c_slam.core.types import Pose2D
from nano_c_slam.sim.mission import simulate_mission
from nano_c_slam.sim.sensor import QuadToFSensor
from nano_c_slam.sim.trajectories import square_mission
from nano_c_slam.sim.world import World

DRONE_COLORS = ["tab:blue", "tab:orange"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def scan_map(scans, poses):
    """World-frame point cloud from scans placed at the given poses (by pose id)."""
    clouds = [transform_points(poses[s.anchor_id], s.cloud) for s in scans]
    return np.vstack(clouds) if clouds else np.empty((0, 2))


def main() -> None:
    world = World.rectangle(-1.5, -1.5, 5.5, 3.5).with_segments(
        [(1.0, 3.5, 1.0, 2.7), (3.8, -1.5, 3.8, -0.7)]
    )
    sensor = QuadToFSensor(max_range=12.0, noise_sigma=0.01)
    steps, anchors = square_mission(side=2.0, laps=2, increments=15, scan_frames=20)
    missions = [
        simulate_mission(0, Pose2D(0, 0, 0), steps, anchors, 20, world, sensor, np.random.default_rng(1)),
        simulate_mission(1, Pose2D(2, 0, 0), steps, anchors, 20, world, sensor, np.random.default_rng(2)),
    ]

    # --- Distributed collaborative SLAM over the medium ---
    agents, medium = run_swarm(missions)
    corrected = {i: agents[i].poses for i in agents}

    # --- Communication report (the raw material for the R9 scalability study) ---
    print("--- communication over the mission ---")
    print(f"messages sent : {medium.messages_sent}")
    print(f"data sent     : {medium.bytes_sent} B  ({medium.bytes_sent / 1024:.1f} KB)")
    for name in sorted(medium.count_by_type):
        print(f"  {name:18s}: {medium.count_by_type[name]:3d} msgs, {medium.bytes_by_type[name]} B")

    # --- Map: raw odometry vs distributed-corrected ---
    odom = {i: {s.anchor_id: s.anchor_pose for s in missions[i].scans} for i in agents}
    corr = {i: {s.anchor_id: corrected[i][s.anchor_id] for s in missions[i].scans} for i in agents}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, title, poses in [(ax1, "Raw odometry", odom), (ax2, "Distributed collaborative SLAM", corr)]:
        for x1, y1, x2, y2 in world.walls:
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.5)
        for i in agents:
            cloud = scan_map(missions[i].scans, poses[i])
            ax.scatter(cloud[:, 0], cloud[:, 1], s=2, alpha=0.2, color=DRONE_COLORS[i])
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    fig.suptitle(f"R7: map built by message passing ({medium.messages_sent} msgs, "
                 f"{medium.bytes_sent / 1024:.1f} KB over the radio)")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "07_distributed_comms.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved map -> {out}")


if __name__ == "__main__":
    main()
