"""
The shared communication medium (R7).

A minimal message bus: a drone posts a message addressed to another drone, and
the medium drops it into that drone's inbox until it is delivered. It also tallies
how many messages and bytes crossed the radio -- the raw material for the
scalability study (R9).

This is the plain message-passing substrate. The paper's token-based scheduling
(one transmitter at a time, in ID order) and its fail-safes (ACK/retransmit,
token reclaim) are a convergence step layered on top later; they change *when*
drones talk, not the messages themselves.
"""

from __future__ import annotations

from collections import defaultdict


class Medium:
    """In-memory inboxes plus communication accounting."""

    def __init__(self, agent_ids: list[int]) -> None:
        self.inboxes: dict[int, list] = {i: [] for i in agent_ids}
        self.messages_sent = 0
        self.bytes_sent = 0
        # Breakdown by message class name (e.g. how much was scans vs pose updates).
        self.count_by_type: dict[str, int] = defaultdict(int)
        self.bytes_by_type: dict[str, int] = defaultdict(int)

    def send(self, to_id: int, message) -> None:
        """Queue `message` for drone `to_id` and count it."""
        self.inboxes[to_id].append(message)
        self.messages_sent += 1
        self.bytes_sent += message.size_bytes
        name = type(message).__name__
        self.count_by_type[name] += 1
        self.bytes_by_type[name] += message.size_bytes

    def broadcast(self, from_id: int, to_ids: list[int], message) -> None:
        """Send the same message to several recipients (one transmission each)."""
        for to_id in to_ids:
            if to_id != from_id:
                self.send(to_id, message)

    def deliver(self, agent_id: int) -> list:
        """Hand over and clear an agent's inbox."""
        messages = self.inboxes[agent_id]
        self.inboxes[agent_id] = []
        return messages
