"""
Reactive exploration controller (R8).

The drone decides where to go from its ToF readings alone -- no scripted path.
This is a generic first policy shaped like the paper's state machine:

  CRUISE   : drive forward (slowing as the front gets closer) while holding a
             wall on the right at a set distance -- this traces a room's
             perimeter and naturally loops back for loop closures.
  SPINNING : when something blocks the front, rotate in place until the way is
             clear again. A long spin happens at corners -- exactly where a
             20-frame rotating scan should be acquired (the runtime watches for
             this, delivering the corner-triggered scans deferred from R5).

The output is body velocities (vx, vy) and yaw rate omega. The paper's third
state (Caution, for avoiding other drones) and its wall-parallel line alignment
are convergence steps we can add later.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..core.types import DepthFrame

# DepthFrame row order (see sim.sensor.SENSOR_CENTERS): front, back, left, right.
FRONT, BACK, LEFT, RIGHT = 0, 1, 2, 3


class ExploreState(Enum):
    CRUISE = "cruise"
    SPINNING = "spinning"


def min_distance(depth: DepthFrame, sensor_index: int) -> float:
    """Nearest valid hit of one ToF sensor (a DepthFrame row), or inf if it sees
    nothing (all pixels out of range)."""
    valid = depth.valid[sensor_index]
    if not valid.any():
        return float("inf")
    return float(depth.ranges[sensor_index][valid].min())


@dataclass
class WallFollowController:
    """Follow the wall on the right; spin in place when the front is blocked.

    All distances are in metres, speeds in m/s, rates in rad/s. Defaults are tuned
    for the slow flight of a nano-drone in a room-sized space.
    """

    stop_distance: float = 0.6     # dS: start spinning when the front is closer
    clear_margin: float = 0.3      # resume cruising once front > stop_distance + this
    wall_distance: float = 0.6     # dobj: preferred distance to the right wall
    min_speed: float = 0.15
    max_speed: float = 0.5
    speed_slope: float = 0.6       # how quickly vx ramps up with frontal clearance
    wall_gain: float = 1.0         # P-gain of the right-wall follower
    max_lateral: float = 0.3
    spin_rate: float = 0.5         # omega while spinning (CCW / turning left)
    state: ExploreState = ExploreState.CRUISE

    def command(self, depth: DepthFrame) -> tuple[float, float, float]:
        """Return the body command (vx, vy, omega) for the current reading."""
        d_front = min_distance(depth, FRONT)
        d_right = min_distance(depth, RIGHT)

        # While spinning, keep turning left until the front opens up again.
        if self.state == ExploreState.SPINNING:
            if d_front > self.stop_distance + self.clear_margin:
                self.state = ExploreState.CRUISE
            else:
                return (0.0, 0.0, self.spin_rate)

        # In CRUISE, a blocked front triggers a spin.
        if d_front < self.stop_distance:
            self.state = ExploreState.SPINNING
            return (0.0, 0.0, self.spin_rate)

        # Forward speed ramps with how clear the front is (smooth deceleration).
        vx = float(np.clip(self.speed_slope * (d_front - self.stop_distance),
                           self.min_speed, self.max_speed))
        # Hold the right wall at wall_distance: if we are too far (d_right larger),
        # move right (negative body-y) to approach it; too close, ease left.
        vy = 0.0
        if d_right < 2 * self.wall_distance:
            vy = float(np.clip(-self.wall_gain * (d_right - self.wall_distance),
                               -self.max_lateral, self.max_lateral))
        return (vx, vy, 0.0)
