"""
R2 demo: pose-graph optimization straightens the drifted trajectory.

We reuse the R1 drifting square, then build a pose graph:
  * odometry edges  = the (noisy) relative motions the robot reported, and
  * loop-closure edges = "at the end of each lap you are back at the start"
    (we simply hand it the true relative pose, standing in for what a perfect
    scan-match will later provide in R4).

Generic Gauss-Newton optimization then redistributes the loop-closure
corrections along the chain, snapping the orange odometry spiral back onto the
true square. This proves the *correction mechanism* generically -- before any of
the paper's ICP / hierarchical / distributed machinery.

Run:  ../.venv/Scripts/python.exe experiments/02_pose_graph_optimization.py
Saves a plot to experiments/output/02_pose_graph_optimization.png.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from nano_c_slam.core.geometry import relative
from nano_c_slam.core.types import EdgeType, Pose2D
from nano_c_slam.slam.pose_graph import PoseGraph
from nano_c_slam.sim.motion import integrate
from nano_c_slam.sim.trajectories import square_steps, steps_per_lap


def mean_position_error(estimate, truth) -> float:
    """Average straight-line position error between two equal-length paths (m)."""
    return float(np.mean([np.hypot(e.x - t.x, e.y - t.y) for e, t in zip(estimate, truth)]))


def main() -> None:
    start = Pose2D(0.0, 0.0, 0.0)
    laps, increments = 3, 20
    steps = square_steps(side=2.0, laps=laps, increments=increments)

    truth = integrate(start, steps)
    rng = np.random.default_rng(1)
    odom = integrate(start, steps, rng=rng, trans_sigma=0.01, rot_sigma=0.01)

    graph = PoseGraph(odom)
    graph.add_odometry_chain()  # consecutive-pose edges from the odometry estimate
    # One loop closure at each lap return (those poses coincide with the start).
    lap_ends = [steps_per_lap(increments) * lap for lap in range(1, laps + 1)]
    for lap_end in lap_ends:
        graph.add_edge(0, lap_end, relative(truth[0], truth[lap_end]),
                       information=np.eye(3) * 10.0, edge_type=EdgeType.INTRA_LC)

    before = mean_position_error(odom, truth)
    optimized = graph.optimize(fixed_ids=(0,))
    after = mean_position_error(optimized, truth)
    print(f"mean position error  before: {before:.3f} m   after: {after:.3f} m")

    plt.figure(figsize=(6, 6))
    plt.plot([p.x for p in truth], [p.y for p in truth], "-", label="ground truth")
    plt.plot([p.x for p in odom], [p.y for p in odom], "-", alpha=0.6,
             label=f"odometry (err {before:.2f} m)")
    plt.plot([p.x for p in optimized], [p.y for p in optimized], "-",
             label=f"optimized (err {after:.2f} m)")
    # Mark the loop-closure poses that were pinned back to the start.
    plt.plot([odom[k].x for k in lap_ends], [odom[k].y for k in lap_ends], "x",
             color="red", label="loop-closure poses")
    plt.plot(0, 0, "ko", label="start")
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.title("R2: pose-graph optimization corrects drift")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "02_pose_graph_optimization.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved plot -> {out_path}")


if __name__ == "__main__":
    main()
