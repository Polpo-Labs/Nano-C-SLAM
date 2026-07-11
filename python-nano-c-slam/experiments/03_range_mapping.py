"""
R3 demo: build a map from range sensing, and see drift smear it (then R2 fix it).

Setup: a rectangular room with the robot driving a square inside it. At every
step the range sensor senses the *real* room from the *true* pose, but the robot
can only place those hits on the map using the pose it *believes* it is at:

  * map from raw odometry  -> believed poses drift, so the walls smear/rotate.
  * map after pose-graph optimization (R2) -> corrected poses, so the walls line
    up crisply on top of the true room.

This is the payoff of R2+R3 together: better poses => a better map.

Run:  ../.venv/Scripts/python.exe experiments/03_range_mapping.py
Saves a plot to experiments/output/03_range_mapping.png.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from nano_c_slam.core.geometry import relative
from nano_c_slam.core.types import EdgeType, Pose2D
from nano_c_slam.slam.mapping import build_point_cloud
from nano_c_slam.slam.pose_graph import PoseGraph
from nano_c_slam.sim.motion import integrate
from nano_c_slam.sim.sensor import RangeSensor
from nano_c_slam.sim.trajectories import square_steps, steps_per_lap
from nano_c_slam.sim.world import World


def plot_map(ax, world, cloud, path, title):
    """Draw the true walls, a point-cloud map, and a trajectory on one axis."""
    for x1, y1, x2, y2 in world.walls:
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2)
    ax.scatter(cloud[:, 0], cloud[:, 1], s=2, alpha=0.3, color="tab:orange")
    ax.plot([p.x for p in path], [p.y for p in path], color="tab:blue", linewidth=1)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")


def main() -> None:
    # A room with the robot's 2 m square path comfortably inside it.
    world = World.rectangle(-1.5, -1.5, 3.5, 3.5)

    start = Pose2D(0.0, 0.0, 0.0)
    laps, increments = 3, 20
    steps = square_steps(side=2.0, laps=laps, increments=increments)
    truth = integrate(start, steps)
    rng = np.random.default_rng(1)
    odom = integrate(start, steps, rng=rng, trans_sigma=0.01, rot_sigma=0.01)

    # The sensor observes the real world from the true pose (with a little noise).
    sensor = RangeSensor.ring(n_beams=48, max_range=6.0, noise_sigma=0.01)
    readings = [sensor.measure(pose, world, rng=rng) for pose in truth]

    # Correct the drifted odometry with loop closures at each lap return (R2).
    graph = PoseGraph(odom)
    graph.add_odometry_chain()
    for lap_end in [steps_per_lap(increments) * lap for lap in range(1, laps + 1)]:
        graph.add_edge(0, lap_end, relative(truth[0], truth[lap_end]),
                       information=np.eye(3) * 10.0, edge_type=EdgeType.INTRA_LC)
    optimized = graph.optimize(fixed_ids=(0,))

    # Same sensor readings, mapped with the believed poses before vs after PGO.
    map_odom = build_point_cloud(odom, readings)
    map_opt = build_point_cloud(optimized, readings)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plot_map(ax1, world, map_odom, odom, "Map from raw odometry (smeared)")
    plot_map(ax2, world, map_opt, optimized, "Map after pose-graph optimization")
    ax1.set_ylabel("y [m]")
    fig.suptitle("R3: range-sensor map, before vs after correcting the poses")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "03_range_mapping.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"mapped points: {len(map_odom)}   saved plot -> {out_path}")


if __name__ == "__main__":
    main()
