"""
R9 (part B): empirical scalability -- measure, don't just model.

We actually fly N autonomous drones in a shared room (each wall-follows the
perimeter, so their corner scans overlap and produce inter-drone loop closures),
run the distributed collaborative SLAM (R7), and measure real numbers as N grows:
  * mapping RMSE (points to nearest wall), raw odometry vs. collaborative,
  * mean trajectory error (ATE) across the swarm, and
  * the data actually sent over the medium.

This is the payoff of having a simulator: the paper's scalability is modelled,
ours is measured. (Coverage-time-vs-N -- the paper's other scalability result --
needs the swarm-spreading behaviour we deferred from R8, so it is out of scope.)

Run:  ../.venv/Scripts/python.exe experiments/09b_scalability_empirical.py
"""

from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from nano_c_slam.analysis.metrics import mapping_rmse, mean_position_error
from nano_c_slam.comms.swarm import run_swarm
from nano_c_slam.core.geometry import transform_points
from nano_c_slam.core.types import Pose2D
from nano_c_slam.exploration.controller import WallFollowController
from nano_c_slam.sim.explore import explore_mission
from nano_c_slam.sim.sensor import QuadToFSensor
from nano_c_slam.sim.world import World

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Staggered start poses around a 5x4 room so drones follow the perimeter in phase
# (and thus scan the same corners, producing inter-drone loop closures).
STARTS = [Pose2D(1, 1, 0), Pose2D(4, 1, np.pi / 2), Pose2D(4, 3, np.pi),
          Pose2D(1, 3, -np.pi / 2)]


def merged_map(missions, poses_per_drone):
    """World-frame cloud of all drones' scans placed at the given poses."""
    clouds = []
    for mission, poses in zip(missions, poses_per_drone):
        for s in mission.scans:
            clouds.append(transform_points(poses[s.anchor_id], s.cloud))
    return np.vstack(clouds) if clouds else np.empty((0, 2))


def run_for_n(n, world, sensor, n_steps):
    missions = [
        explore_mission(i, STARTS[i], WallFollowController(), world, sensor,
                        np.random.default_rng(10 + i), n_steps=n_steps)
        for i in range(n)
    ]
    t0 = time.perf_counter()
    agents, medium = run_swarm(missions)
    elapsed = time.perf_counter() - t0

    odom_poses = [m.odom for m in missions]
    corrected = [agents[i].poses for i in range(n)]
    ate = np.mean([mean_position_error(corrected[i], missions[i].truth) for i in range(n)])
    return {
        "n": n,
        "rmse_odom": mapping_rmse(merged_map(missions, odom_poses), world.walls),
        "rmse_slam": mapping_rmse(merged_map(missions, corrected), world.walls),
        "ate": float(ate),
        "kb": medium.bytes_sent / 1024,
        "scans": sum(len(m.scans) for m in missions),
        "time": elapsed,
    }


def main() -> None:
    world = World.rectangle(0, 0, 5, 4)
    sensor = QuadToFSensor(max_range=6.0, noise_sigma=0.01)
    swarm_sizes = [1, 2, 3, 4]

    rows = []
    for n in swarm_sizes:
        row = run_for_n(n, world, sensor, n_steps=1000)
        rows.append(row)
        print(f"N={n}: map RMSE odom {row['rmse_odom']:.3f} -> slam {row['rmse_slam']:.3f} m | "
              f"ATE {row['ate']:.3f} m | {row['scans']} scans | {row['kb']:.1f} KB | {row['time']:.1f} s")

    ns = [r["n"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: measured map accuracy, raw odometry vs collaborative SLAM.
    ax1.plot(ns, [r["rmse_odom"] for r in rows], "o--", color="tab:red", label="raw odometry")
    ax1.plot(ns, [r["rmse_slam"] for r in rows], "o-", color="tab:green", label="collaborative SLAM")
    ax1.set_xlabel("number of drones N")
    ax1.set_ylabel("mapping RMSE [m]")
    ax1.set_title("Map accuracy vs. swarm size (measured)")
    ax1.set_xticks(ns)
    ax1.legend(fontsize=8)

    # Right: the two costs of scaling -- radio data and optimization time. The
    # steep time growth is the practical "usability cliff" of naive all-pairs
    # scan matching (the paper avoids it with spatial gating).
    ax2.plot(ns, [r["kb"] for r in rows], "o-", color="tab:blue", label="data over radio [KB]")
    ax2.set_xlabel("number of drones N")
    ax2.set_ylabel("data over the radio [KB]", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_xticks(ns)
    ax2b = ax2.twinx()
    ax2b.plot(ns, [r["time"] for r in rows], "s-", color="tab:purple", label="optimize time [s]")
    ax2b.set_ylabel("optimize time [s]", color="tab:purple")
    ax2b.tick_params(axis="y", labelcolor="tab:purple")
    ax2.set_title("Cost of scaling: communication and compute")

    fig.suptitle("R9: empirical scalability -- measured accuracy and communication")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "09b_scalability_empirical.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
