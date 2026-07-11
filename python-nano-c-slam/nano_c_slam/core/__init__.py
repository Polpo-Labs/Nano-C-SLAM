"""
core: the shared foundations every other component builds on.

  types     the plain data structures that flow through the pipeline
  geometry  the SE(2) rigid-body math used for poses, ICP, and pose-graph edges

Nothing here knows about simulation, SLAM logic, or communication -- it is pure
data + math so it can be reused (and unit-tested) in complete isolation.
"""
