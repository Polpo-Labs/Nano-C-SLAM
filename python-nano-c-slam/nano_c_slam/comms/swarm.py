"""
Running a swarm of distributed SLAM agents (R7).

This orchestrates the message exchange that produces the same collaborative map
as the R6 god-view, but through the medium only. The order mirrors the paper's
stages and its cascaded scheme (each drone aligns to lower-ID drones):

  1. Every drone broadcasts its scans to higher-ID drones (Stage 3).
  2. In increasing ID order, each drone: takes in what it has received, optimizes
     its own graph against the (fixed) external poses (Stages 2 & 4), and
     broadcasts its updated scan poses to higher-ID drones (Stage 5).

Because drone i optimizes before any higher drone uses its poses, the higher
drones align to already-corrected references -- the cascade.
"""

from __future__ import annotations

from ..sim.mission import MissionResult
from .agent import DroneAgent
from .medium import Medium


def run_swarm(missions: list[MissionResult]) -> tuple[dict[int, DroneAgent], Medium]:
    """Run distributed collaborative SLAM over the given missions (index = drone id).

    Returns the agents (with corrected `poses`) and the medium (with comms tallies).
    """
    ids = list(range(len(missions)))
    higher_ids = {i: [j for j in ids if j > i] for i in ids}

    agents = {i: DroneAgent(i, missions[i].odom) for i in ids}
    for i in ids:
        for scan in missions[i].scans:
            agents[i].acquire(scan)

    medium = Medium(ids)

    # Stage 3: broadcast every scan to higher-ID drones.
    for i in ids:
        for message in agents[i].scan_messages():
            medium.broadcast(i, higher_ids[i], message)

    # Cascade: optimize in ID order, forwarding updated poses to higher drones.
    for i in ids:
        for message in medium.deliver(i):
            agents[i].receive(message)
        agents[i].optimize()
        for message in agents[i].pose_update_messages():
            medium.broadcast(i, higher_ids[i], message)

    return agents, medium
