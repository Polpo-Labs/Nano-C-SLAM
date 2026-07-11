"""
Motion model: how a robot's pose changes when it moves (R1).

This is the most fundamental piece of any SLAM problem. Before mapping or
optimization, we just move a robot around and track where it *thinks* it is
(odometry) versus where it *actually* is (ground truth). The gap between the two
is drift -- the very thing SLAM later corrects.

A single motion "step" is a small relative movement (a twist) expressed in the
robot's own body frame; applying it to the current world pose is just SE(2)
composition. All that math lives in core.geometry, so this module stays tiny and
reuses it (DRY).

We keep this deliberately generic (a plain 2D kinematic step). The paper's
specific velocity-command / 7.5 Hz details are layered on later, not here.
"""

from __future__ import annotations

import numpy as np

from ..core.geometry import compose
from ..core.types import Pose2D


def step(pose: Pose2D, dx: float, dy: float, dpsi: float) -> Pose2D:
    """Advance a world pose by one motion step given in the robot's body frame.

    dx   : forward translation this step (body-frame x), metres.
    dy   : lateral  translation this step (body-frame y), metres.
    dpsi : change in heading this step, radians.

    Returns the new world pose. The body-frame step is turned into a world-frame
    update by composing it onto the current pose (core.geometry.compose).
    """
    return compose(pose, Pose2D(dx, dy, dpsi))


def noisy_step(
    pose: Pose2D,
    dx: float,
    dy: float,
    dpsi: float,
    rng: np.random.Generator,
    trans_sigma: float = 0.0,
    rot_sigma: float = 0.0,
) -> Pose2D:
    """Like `step`, but corrupts the motion with Gaussian noise (imperfect odometry).

    This models dead-reckoning error: the robot commands/senses a motion but the
    real applied motion differs slightly, and those errors accumulate into drift.

    rng         : a numpy Generator, passed in so every run is reproducible.
    trans_sigma : std-dev of the per-step translation error (metres).
    rot_sigma   : std-dev of the per-step heading error (radians).

    Reuses `step` for the actual pose update so the SE(2) math is never
    duplicated -- only the noisy deltas are computed here.
    """
    return step(
        pose,
        dx + rng.normal(0.0, trans_sigma),
        dy + rng.normal(0.0, trans_sigma),
        dpsi + rng.normal(0.0, rot_sigma),
    )


def integrate(
    start: Pose2D,
    steps: list[tuple[float, float, float]],
    rng: np.random.Generator | None = None,
    trans_sigma: float = 0.0,
    rot_sigma: float = 0.0,
) -> list[Pose2D]:
    """Integrate a sequence of motion steps from `start` into a list of poses.

    Returns the full path including the start pose, so the result has
    len(steps) + 1 entries. With `rng=None` the motion is exact -> ground truth;
    with an `rng` and sigmas it drifts -> odometry. Both branches reuse the
    motion functions above, so there is exactly one motion model in the project.
    """
    pose = start
    path = [pose]
    for dx, dy, dpsi in steps:
        if rng is None:
            pose = step(pose, dx, dy, dpsi)
        else:
            pose = noisy_step(pose, dx, dy, dpsi, rng, trans_sigma, rot_sigma)
        path.append(pose)
    return path
