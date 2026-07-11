"""
Nano-C-SLAM: a clean, readable Python reproduction of the collaborative SLAM
system for robot swarms (Niculescu et al., IEEE Access 2025), built to run in
2D simulation.

The package is split into narrow, independently testable components:
    core         shared data types + SE(2) geometry (reused by everything)
    sim          the simulated 2D world and sensors (ground-truth generator)
    slam         scan building, ICP, and pose-graph optimization
    exploration  the per-drone flight state machine
    comms        the token-based swarm communication protocol
    viz          2D monitoring and Foxglove streaming

See ../README.md for the engineering principles we hold ourselves to.
"""

__version__ = "0.0.1"
