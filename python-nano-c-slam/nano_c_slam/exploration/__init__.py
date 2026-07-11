"""
exploration: the per-drone flight behavior.

Responsibility: the Cruise / Spinning / Caution state machine (Section IV) that,
from the ToF distances and neighbours' positions, outputs the body velocities
(vx, vy) and yaw rate omega. It biases the drone toward walls and corners
(good loop-closure spots) and spreads the swarm out at intersections.

(Populated in Phase 3.)
"""
