"""
R1 demo: watch odometry drift away from ground truth.

We drive one robot around a square several times. The *ground-truth* pose comes
from integrating the true motion steps; the *odometry* pose integrates the same
steps but with per-step Gaussian noise (imperfect dead reckoning). With no map
and no correction, the odometry estimate slowly drifts away from reality -- the
core problem the next rung (pose-graph optimization) fixes.

Run:  ../.venv/Scripts/python.exe experiments/01_odometry_drift.py
Saves a plot to experiments/output/01_odometry_drift.png and prints the drift.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from nano_c_slam.core.types import Pose2D
from nano_c_slam.sim.motion import integrate
from nano_c_slam.sim.trajectories import square_steps


def main() -> None:
    start = Pose2D(0.0, 0.0, 0.0)
    steps = square_steps(side=2.0, laps=3)

    # Ground truth = exact integration; odometry = same steps with noise.
    truth = integrate(start, steps)
    rng = np.random.default_rng(1)
    odom = integrate(start, steps, rng=rng, trans_sigma=0.01, rot_sigma=0.01)

    # Final drift = how far the odometry estimate ends up from the true position.
    drift = float(np.hypot(truth[-1].x - odom[-1].x, truth[-1].y - odom[-1].y))
    print(f"steps: {len(steps)}  final drift: {drift:.3f} m")

    plt.figure(figsize=(6, 6))
    plt.plot([p.x for p in truth], [p.y for p in truth], "-", label="ground truth")
    plt.plot([p.x for p in odom], [p.y for p in odom], "-", label="odometry (drifting)")
    plt.plot(0, 0, "ko", label="start")
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.title(f"R1: odometry drift over 3 laps (final drift {drift:.2f} m)")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "01_odometry_drift.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved plot -> {out_path}")


if __name__ == "__main__":
    main()
