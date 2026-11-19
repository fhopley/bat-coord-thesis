"""Unit tests for model.components.vpp_event_classifier.

Ported from test_perfect_foresight_vpp.py's classifier test section as part
of removing perfect_foresight_vpp.py (superseded by vpp_sc_event.py).
vpp_event_classifier.py itself is unaffected by that removal -- it is a
standalone component consumed by both the old and new VPP dispatch paths --
so these tests are ported verbatim (no adaptation needed) rather than
rewritten. classify_vpp_events never references a datetime index, so there
is nothing here tied to perfect_foresight_vpp's own calling convention.
"""
import numpy as np
import pandas as pd
import pytest

from model.components.vpp_event_classifier import classify_vpp_events


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _idx(n: int) -> pd.DatetimeIndex:
    """n-interval index at 1-hour cadence (kW = kWh/interval)."""
    return pd.date_range('2022-01-03 01:00:00', periods=n, freq='60min',
                         tz='Etc/GMT-10')


def _s(vals, idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(vals, index=idx, dtype=float)


# ---------------------------------------------------------------------------
# vpp_event_classifier tests
# ---------------------------------------------------------------------------

def test_classifier_no_spikes():
    """No price exceeds threshold: all intervals unmarked."""
    n = 12
    idx = _idx(n)
    prices = _s(np.full(n, 50.0), idx)
    result = classify_vpp_events(prices, threshold_dollars_per_mwh=100.0,
                                 event_window_hours=2.0, interval_hours=1.0)
    assert not result.any()


def test_classifier_single_spike_correct_window():
    """Spike at t=5 with 3-hour window marks [3, 5] inclusive."""
    n = 9
    idx = _idx(n)
    prices_arr = np.full(n, 50.0); prices_arr[5] = 200.0
    result = classify_vpp_events(_s(prices_arr, idx), threshold_dollars_per_mwh=100.0,
                                 event_window_hours=3.0, interval_hours=1.0)
    expected = np.zeros(n, dtype=bool); expected[3:6] = True
    np.testing.assert_array_equal(result, expected)


def test_classifier_spike_at_start_clamped():
    """Spike at t=0 with 3-hour window marks [0, 0] only (no underflow)."""
    n = 6
    idx = _idx(n)
    prices_arr = np.full(n, 50.0); prices_arr[0] = 200.0
    result = classify_vpp_events(_s(prices_arr, idx), threshold_dollars_per_mwh=100.0,
                                 event_window_hours=3.0, interval_hours=1.0)
    assert result[0]
    assert not result[1:].any()


def test_classifier_overlapping_windows_contiguous_no_extension():
    """Two spikes 2 intervals apart with 4-hour window.

    Spike B at t=7 opens window [4, 7]. Spike A at t=5 falls inside that
    window, so it is skipped -- the block does not extend back to [2, 7].
    """
    n = 10
    idx = _idx(n)
    prices_arr = np.full(n, 50.0)
    prices_arr[5] = 200.0   # spike A (earlier in time, later in backwards scan)
    prices_arr[7] = 200.0   # spike B (triggers window [4, 7])
    result = classify_vpp_events(_s(prices_arr, idx), threshold_dollars_per_mwh=100.0,
                                 event_window_hours=4.0, interval_hours=1.0)
    expected = np.zeros(n, dtype=bool); expected[4:8] = True
    np.testing.assert_array_equal(result, expected)


def test_classifier_two_separate_windows():
    """Two spikes far enough apart produce two separate event windows."""
    n = 20
    idx = _idx(n)
    prices_arr = np.full(n, 50.0)
    prices_arr[3]  = 200.0   # spike A -> window [2, 3] (2-hour window)
    prices_arr[15] = 200.0   # spike B -> window [14, 15]
    result = classify_vpp_events(_s(prices_arr, idx), threshold_dollars_per_mwh=100.0,
                                 event_window_hours=2.0, interval_hours=1.0)
    assert result[2] and result[3]
    assert not result[4:14].any()
    assert result[14] and result[15]
    assert not result[16:].any()


def test_classifier_rejects_zero_threshold():
    idx = _idx(4)
    with pytest.raises(ValueError, match='threshold'):
        classify_vpp_events(_s([100.0]*4, idx), threshold_dollars_per_mwh=0.0,
                            event_window_hours=1.0, interval_hours=1.0)


def test_classifier_rejects_nan_prices():
    idx = _idx(4)
    prices = _s([100.0, float('nan'), 100.0, 100.0], idx)
    with pytest.raises(ValueError, match='NaN'):
        classify_vpp_events(prices, threshold_dollars_per_mwh=500.0,
                            event_window_hours=1.0, interval_hours=1.0)
