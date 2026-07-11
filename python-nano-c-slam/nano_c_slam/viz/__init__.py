"""
viz: monitoring and debugging views.

Responsibility: a live 2D monitor (walls, drones, ground-truth vs. estimated
trajectories, the growing map, loop-closure edges) and Foxglove streaming of
point clouds / pose graphs / transforms for 3D inspection and offline replay.

Kept strictly read-only w.r.t. the rest of the system -- visualization must
never influence the algorithm. (Populated in Phase 4.)
"""
