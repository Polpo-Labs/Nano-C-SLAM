"""
R4 demo: loop closures discovered from data with ICP (no more truth-given edges).

In R2/R3 we *told* the graph "you are back at the start". Here the robot instead
figures that out by scan-matching: when it returns to a place, it aligns the new
scan with the earlier one using ICP, and the resulting rigid transform becomes
the loop-closure edge. Fed to the same pose-graph optimizer, this corrects the
drift and sharpens the map -- now entirely from sensor data.

Two panels:
  (left)  one loop-closure scan pair: the source scan placed at the drifted
          odometry guess vs. after ICP has aligned it onto the target scan.
  (right) the map built with ICP-derived loop closures (vs. raw odometry).

We give the room a small asymmetric obstacle so the scans have a distinctive
feature -- a clean rectangle is symmetric and makes scan-matching ambiguous
(the "texture-rich location" idea from the paper).

Run:  ../.venv/Scripts/python.exe experiments/04_icp_scan_matching.py
Saves a plot to experiments/output/04_icp_scan_matching.png.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from nano_c_slam.core.geometry import relative, transform_points
from nano_c_slam.core.types import EdgeType, Pose2D
from nano_c_slam.slam.icp import icp
from nano_c_slam.slam.mapping import build_point_cloud, project_reading
from nano_c_slam.slam.pose_graph import PoseGraph
from nano_c_slam.sim.motion import integrate
from nano_c_slam.sim.sensor import RangeSensor
from nano_c_slam.sim.trajectories import square_steps, steps_per_lap
from nano_c_slam.sim.world import World


def local_scan(reading) -> np.ndarray:
    """A reading's hit points in the sensor's own body frame (pose-independent)."""
    return project_reading(Pose2D(0.0, 0.0, 0.0), reading)


def main() -> None:
    # Room + a small asymmetric obstacle so scans have a distinctive feature.
    world = World.rectangle(-1.5, -1.5, 3.5, 3.5).with_segments(
        [(2.6, 0.4, 2.6, 1.6), (2.6, 1.6, 3.1, 1.6)]  # an L-shaped obstacle
    )

    start = Pose2D(0.0, 0.0, 0.0)
    laps, increments = 3, 20
    steps = square_steps(side=2.0, laps=laps, increments=increments)
    truth = integrate(start, steps)
    rng = np.random.default_rng(1)
    odom = integrate(start, steps, rng=rng, trans_sigma=0.01, rot_sigma=0.01)

    sensor = RangeSensor.ring(n_beams=64, max_range=8.0, noise_sigma=0.01)
    readings = [sensor.measure(pose, world, rng=rng) for pose in truth]

    # Loop-closure candidates: the start pose and each lap return coincide.
    lap_ends = [steps_per_lap(increments) * lap for lap in range(1, laps + 1)]

    graph = PoseGraph(odom)
    graph.add_odometry_chain()
    for j in lap_ends:
        # Align scan j onto scan 0, seeded with the (drifted) odometry guess.
        init = relative(odom[0], odom[j])
        z_icp, _, error, iters = icp(local_scan(readings[j]), local_scan(readings[0]), init=init)
        print(f"loop closure 0<->{j}:  ICP error {error:.3f} m in {iters} iters")
        graph.add_edge(0, j, z_icp, information=np.eye(3) * 10.0, edge_type=EdgeType.INTER_LC)

    optimized = graph.optimize(fixed_ids=(0,))

    # --- Panel A: visualize one ICP loop closure (the last lap return) ---
    j = lap_ends[-1]
    tgt = local_scan(readings[0])
    src = local_scan(readings[j])
    init = relative(odom[0], odom[j])
    z_icp, aligned, _, _ = icp(src, tgt, init=init)
    src_at_guess = transform_points(init, src)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6))
    axA.scatter(tgt[:, 0], tgt[:, 1], s=14, color="black", label="target scan (pose 0)")
    axA.scatter(src_at_guess[:, 0], src_at_guess[:, 1], s=14, color="tab:red",
                alpha=0.6, label="source at odometry guess")
    axA.scatter(aligned[:, 0], aligned[:, 1], s=14, color="tab:green",
                alpha=0.6, label="source after ICP")
    axA.set_aspect("equal")
    axA.legend(loc="upper left", fontsize=8)
    axA.set_title("ICP aligns a revisited scan onto the original")

    # --- Panel B: the resulting map, raw odometry vs ICP-corrected ---
    map_odom = build_point_cloud(odom, readings)
    map_opt = build_point_cloud(optimized, readings)
    for x1, y1, x2, y2 in world.walls:
        axB.plot([x1, x2], [y1, y2], color="black", linewidth=2)
    axB.scatter(map_odom[:, 0], map_odom[:, 1], s=2, alpha=0.15, color="tab:red",
                label="map from raw odometry")
    axB.scatter(map_opt[:, 0], map_opt[:, 1], s=2, alpha=0.3, color="tab:green",
                label="map after ICP loop closures")
    axB.set_aspect("equal")
    axB.legend(loc="upper left", fontsize=8)
    axB.set_title("Map corrected by ICP-derived loop closures")

    fig.suptitle("R4: loop closures discovered from data via ICP scan-matching")
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "04_icp_scan_matching.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved plot -> {out_path}")


if __name__ == "__main__":
    main()
