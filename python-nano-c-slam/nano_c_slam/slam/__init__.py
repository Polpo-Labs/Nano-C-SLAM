"""
slam: the mapping brain of each drone.

Responsibility: turn noisy augmented poses into a consistent map, by
  - building scans from depth frames and projecting them to point clouds,
  - ICP scan-matching to find loop-closure transforms,
  - hierarchical pose-graph optimization (PGO) to correct drift,
  - the distributed / cascaded C-SLAM logic that aligns each drone's map to
    lower-ID drones using external poses as fixed anchors (Section II).

(Populated in Phases 1-2.)
"""
