"""
Analytical scalability model of the swarm communication protocol (R9).

Reproduces the paper's Section VII-C study: how the token-based protocol scales
with the number of drones N. Two views:

  * loop time    -- time for one full token round, dominated by pairwise ranging
                    (N(N-1)/2 rangings x ~4 ms). Grows quadratically; this is the
                    localization / collision-avoidance latency.
  * radio demand -- the aggregate data rate the swarm must sustain to keep that
                    loop fresh (finish within one pose period). Compared against
                    the nominal rates of BLE / UWB / WiFi to get a max swarm size.

Assumptions (stated so the numbers are reproducible):
  - every pair of drones does one double-sided two-way ranging per loop, and the
    ranging message also carries the drone's position (so it covers localization);
  - at most SCAN_FRACTION of the drones transmit one scan per loop;
  - the loop must complete within one pose period (1/7.5 s) to stay fresh.

With these, the model lands close to the paper's capacities (UWB ~100, BLE ~40).
Scan size is reused from comms.messages (DRY).
"""

from __future__ import annotations

import math

from ..comms.messages import SCAN_BYTES

T_RANGING = 4e-3            # s, one pairwise double-sided ranging (paper: ~4 ms)
RANGING_BYTES = 24         # position (12 B) + ranging overhead per pair, modeled
SCAN_FRACTION = 0.2        # <= 20% of drones send a scan in a given loop (paper)
POSE_PERIOD = 1.0 / 7.5    # s, the loop must finish within this to stay fresh

# Nominal data rates of the cheap radio modules the paper considers.
RADIO_RATES_BPS = {"BLE": 2e6, "UWB": 6.8e6, "WiFi": 25e6}


def pairwise_rangings(n: int) -> int:
    """Number of ranging exchanges in one loop: every pair, once."""
    return n * (n - 1) // 2


def loop_time(n: int) -> float:
    """Token-round time from pairwise ranging (the localization latency), seconds."""
    return pairwise_rangings(n) * T_RANGING


def data_per_loop_bytes(n: int) -> int:
    """Bytes the swarm must move in one loop: all rangings plus the scans sent."""
    ranging = pairwise_rangings(n) * RANGING_BYTES
    scans = math.ceil(SCAN_FRACTION * n) * SCAN_BYTES
    return ranging + scans


def required_bandwidth(n: int) -> float:
    """Aggregate data rate (bps) needed to keep the loop within one pose period."""
    return data_per_loop_bytes(n) * 8 / POSE_PERIOD


def max_swarm_size(rate_bps: float, hard_cap: int = 2000) -> int:
    """Largest N whose required bandwidth still fits a radio of `rate_bps`.

    `required_bandwidth` increases with N, so we just grow N until it no longer
    fits.
    """
    n = 1
    while n < hard_cap and required_bandwidth(n + 1) <= rate_bps:
        n += 1
    return n
