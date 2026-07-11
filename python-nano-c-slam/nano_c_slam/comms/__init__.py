"""
comms: the swarm communication protocol.

Responsibility: the token-based, one-transmitter-at-a-time protocol (Section III,
Algorithm 1) that carries UWB ranging, position broadcasts, scan exchange, the
PGO trigger, and updated-pose broadcasts -- plus the fail-safes (ACK/retransmit,
token reclaim). Modelled first as a deterministic discrete-event scheduler so
experiments are reproducible.

(Populated in Phase 2.)
"""
