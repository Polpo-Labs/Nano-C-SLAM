"""
R1 demo: watch odometry drift away from ground truth.

We drive one robot around a square several times. The *ground-truth* pose comes
from integrating the true motion steps; the *odometry* pose integrates the same
steps but with per-step Gaussian noise (imperfect dead reckoning). With no map
and no correction, the odometry estimate slowly drifts away from reality -- the
core problem the next rung (pose-graph optimization) will fix.

Run:  ../.venv/Scripts/python.exe experiments/01_odometry_drift.py
Saves a plot to experiments/output/01_odometry_drift.png and prints the drift.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from nano_c_slam.core.types import Pose2D
from nano_c_slam.sim.motion import noisy_step, step


def square_steps(side: float, laps: int) -> list[tuple[float, float, float]]:
    """Motion steps (dx, dy, dpsi) that walk a `side`-metre square `laps` times.

    Each lap is 4x (drive one side forward, then turn 90 deg). Small forward
    increments keep the path smooth and give many odometry steps to accumulate.
    """
    increments = 20                       # forward sub-steps per side (smoothness)
    forward = (side / increments, 0.0, 0.0)
    turn = (0.0, 0.0, np.pi / 2)
    one_side = [forward] * increments
    return (one_side + [turn]) * 4 * laps


def integrate(steps, noisy: bool, rng=None, trans_sigma=0.0, rot_sigma=0.0):
    """Integrate a list of motion steps from the origin into a list of poses.

    `noisy=False` gives the ground-truth path; `noisy=True` adds odometry noise.
    Both branches reuse the same motion functions so there is one motion model.
    """
    pose = Pose2D(0.0, 0.0, 0.0)
    path = [pose]
    for dx, dy, dpsi in steps:
        if noisy:
            pose = noisy_step(pose, dx, dy, dpsi, rng, trans_sigma, rot_sigma)
        else:
            pose = step(pose, dx, dy, dpsi)
        path.append(pose)
    return path


def main() -> None:
    rng = np.random.default_rng(1)
    steps = square_steps(side=2.0, laps=3)

    truth = integrate(steps, noisy=False)
    odom = integrate(steps, noisy=True, rng=rng, trans_sigma=0.01, rot_sigma=0.01)

    # Final drift = how far the odometry estimate ends up from the true position.
    drift = float(np.hypot(truth[-1].x - odom[-1].x, truth[-1].y - odom[-1].y))
    print(f"steps: {len(steps)}  final drift: {drift:.3f} m")

    # Plot both trajectories so the divergence is visible.
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
