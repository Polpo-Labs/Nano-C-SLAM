"""
R5 demo: the paper's sensing model -- quad-ToF frames and rotating scans.

A single depth frame from the four ToF sensors covers only four 45 deg arcs, with
45 deg blind gaps between them. By rotating in place while acquiring 20 frames,
the drone sweeps those gaps and builds a ~360 deg "scan" of the place. This is
why the paper rotates >= 45 deg during a scan.

Left  : one depth frame projected onto the room -- four arcs, big gaps.
Right : a 20-frame scan (rotating 90 deg) -- the walls are now fully covered.

Run:  ../.venv/Scripts/python.exe experiments/05_tof_scan_model.py
Saves a plot to experiments/output/05_tof_scan_model.png.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from nano_c_slam.core.geometry import transform_points
from nano_c_slam.core.types import AugmentedPose, Pose2D
from nano_c_slam.slam.mapping import depth_local_points
from nano_c_slam.slam.scan import build_scan, scan_point_cloud
from nano_c_slam.sim.sensor import QuadToFSensor
from nano_c_slam.sim.world import World


def draw_walls(ax, world):
    for x1, y1, x2, y2 in world.walls:
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2)


def main() -> None:
    world = World.rectangle(-2.0, -2.0, 2.0, 2.0).with_segments(
        [(1.2, 1.2, 1.2, 2.0), (0.6, -0.4, 1.6, -0.4)]  # a couple of features
    )
    sensor = QuadToFSensor(max_range=8.0, noise_sigma=0.01)
    rng = np.random.default_rng(0)

    # The scan is acquired at this spot; the anchor pose faces 0 deg.
    spot = (0.3, 0.2)
    anchor = Pose2D(spot[0], spot[1], 0.0)

    # --- One frame: four arcs with gaps (points shown in the world) ---
    frame = sensor.measure(anchor, world, rng=rng)
    single_world = transform_points(anchor, depth_local_points(frame, sensor.bearings))

    # --- A scan: 20 frames while rotating 90 deg in place ---
    frames = []
    for k, psi in enumerate(np.linspace(0.0, np.pi / 2, 20)):
        pose = Pose2D(spot[0], spot[1], psi)
        frames.append(AugmentedPose(k, float(k), pose, sensor.measure(pose, world, rng=rng)))
    scan = build_scan(0, frames)
    scan_world = transform_points(anchor, scan_point_cloud(scan, sensor.bearings))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for ax in (ax1, ax2):
        draw_walls(ax, world)
        ax.plot(*spot, "b^", markersize=10, label="drone")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
    ax1.scatter(single_world[:, 0], single_world[:, 1], s=18, color="tab:red", label="hits")
    ax2.scatter(scan_world[:, 0], scan_world[:, 1], s=8, color="tab:green", label="hits")
    ax1.set_title(f"Single depth frame ({single_world.shape[0]} hits, 180 deg with gaps)")
    ax2.set_title(f"20-frame rotating scan ({scan_world.shape[0]} hits, ~360 deg)")
    ax1.set_ylabel("y [m]")
    ax1.legend(loc="upper right", fontsize=8)
    fig.suptitle("R5: quad-ToF frame vs. rotating scan -- rotation fills the sensor gaps")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "05_tof_scan_model.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"single-frame hits: {single_world.shape[0]}   scan hits: {scan_world.shape[0]}")
    print(f"saved plot -> {out_path}")


if __name__ == "__main__":
    main()
