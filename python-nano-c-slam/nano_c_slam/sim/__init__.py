"""
sim: the simulated 2D world -- the ground-truth generator.

Responsibility: model drones and their environment, and produce the *noisy*
sensor inputs the rest of the system consumes:
  - drone kinematics driven by (vx, vy, omega) commands,
  - ToF depth frames via ray-casting against wall segments,
  - noisy odometry (the drift the SLAM layer must correct),
  - UWB range measurements between drones (with line-of-sight dropouts).

The SLAM/exploration/comms layers never see ground truth -- only what a real
drone would measure. (Populated in Phase 1.)
"""
