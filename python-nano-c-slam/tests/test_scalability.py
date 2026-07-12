"""
Unit tests for the R9 analytical scalability model.

We pin the structural properties (quadratic ranging, monotonic growth, radio
ordering) and check the BLE capacity lands where the paper reports it (~40).
"""

from nano_c_slam.analysis.scalability import (
    RADIO_RATES_BPS,
    data_per_loop_bytes,
    loop_time,
    max_swarm_size,
    pairwise_rangings,
    required_bandwidth,
)


def test_pairwise_rangings_is_all_pairs():
    assert pairwise_rangings(1) == 0
    assert pairwise_rangings(4) == 6      # 4 choose 2
    assert pairwise_rangings(10) == 45


def test_loop_time_grows_quadratically():
    # Doubling N should roughly quadruple the (ranging-dominated) loop time.
    ratio = loop_time(100) / loop_time(50)
    assert 3.5 < ratio < 4.2


def test_bandwidth_and_data_increase_with_n():
    assert data_per_loop_bytes(50) > data_per_loop_bytes(20)
    assert required_bandwidth(50) > required_bandwidth(20)


def test_radio_capacities_are_ordered_and_match_paper_ballpark():
    ble = max_swarm_size(RADIO_RATES_BPS["BLE"])
    uwb = max_swarm_size(RADIO_RATES_BPS["UWB"])
    wifi = max_swarm_size(RADIO_RATES_BPS["WiFi"])
    assert ble < uwb < wifi                 # faster radio -> bigger swarm
    assert 35 <= ble <= 45                  # paper reports ~40 for BLE
    assert wifi > 150                        # paper reports ~190 for WiFi


def test_capacity_respects_the_bandwidth_limit():
    # At the reported capacity the demand fits; one more drone must exceed it.
    rate = RADIO_RATES_BPS["UWB"]
    n = max_swarm_size(rate)
    assert required_bandwidth(n) <= rate < required_bandwidth(n + 1)
