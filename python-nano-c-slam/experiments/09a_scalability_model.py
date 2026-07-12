"""
R9 (part A): analytical scalability of the communication protocol.

Reproduces the paper's Fig. 12 study. Left: the token-loop time grows
quadratically with the number of drones (pairwise ranging), so localization
latency becomes GNSS-comparable around 30 drones. Right: the radio bandwidth the
swarm must sustain to keep the loop fresh, against the nominal rates of BLE / UWB
/ WiFi -- where each line is crossed is that radio's maximum swarm size.

Run:  ../.venv/Scripts/python.exe experiments/09a_scalability_model.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from nano_c_slam.analysis.scalability import (
    RADIO_RATES_BPS,
    loop_time,
    max_swarm_size,
    required_bandwidth,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def main() -> None:
    ns = np.arange(2, 220)

    print("--- max swarm size per radio (analytical) ---")
    for name, rate in RADIO_RATES_BPS.items():
        print(f"{name:5s} ({rate / 1e6:4.1f} Mbps): {max_swarm_size(rate)} drones")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Loop time vs N ---
    ax1.plot(ns, [loop_time(n) for n in ns], color="tab:blue")
    ax1.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax1.annotate("~GNSS latency", (ns[-1], 1.0), textcoords="offset points",
                 xytext=(-90, 5), fontsize=8, color="grey")
    ax1.set_xlabel("number of drones N")
    ax1.set_ylabel("loop time [s]")
    ax1.set_title("Token-loop time (pairwise ranging) grows quadratically")

    # --- Required bandwidth vs N, with radio limits ---
    bandwidth = np.array([required_bandwidth(n) for n in ns]) / 1e6  # Mbps
    ax2.plot(ns, bandwidth, color="tab:red", label="required bandwidth")
    for name, rate in RADIO_RATES_BPS.items():
        cap = max_swarm_size(rate)
        ax2.axhline(rate / 1e6, color="grey", linestyle="--", linewidth=1)
        ax2.annotate(f"{name} ({rate/1e6:.0f} Mbps)", (ns[0], rate / 1e6),
                     textcoords="offset points", xytext=(2, 3), fontsize=8, color="grey")
        ax2.plot(cap, rate / 1e6, "ko", markersize=5)
        ax2.annotate(f"{cap}", (cap, rate / 1e6), textcoords="offset points",
                     xytext=(4, -12), fontsize=8)
    ax2.set_xlabel("number of drones N")
    ax2.set_ylabel("required bandwidth [Mbps]")
    ax2.set_title("Radio bandwidth demand vs. swarm size")
    ax2.legend(loc="upper left", fontsize=8)

    fig.suptitle("R9: analytical scalability of the token communication protocol")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "09a_scalability_model.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
