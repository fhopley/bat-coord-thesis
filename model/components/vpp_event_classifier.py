"""VPP event classifier: mark intervals belonging to a VPP event window.

Produces a boolean array of length T indicating which intervals are under
operator control. The VPP dispatchers (vpp_sc_event, and the
min-export-price derivation in the orchestrator) take this array
as an input parameter and has no knowledge of how events were classified,
so the classification logic can be replaced independently as the modelling of
operator strategy becomes more sophisticated.

Current strategy: backwards scan over wholesale price spikes
------------------------------------------------------------
The classifier scans the price series from the end of the period backwards to
the start. Whenever it finds an interval whose price exceeds the configured
threshold and that interval is not already marked as an event, it marks the
preceding window (including the spike itself) as operator-controlled. This
window represents the time the operator needs to position the battery (charge,
hold, or prepare) before discharging during the price spike.

Scanning backwards is natural for this rule: each spike anchors a window that
precedes it, and the backwards direction attaches windows correctly without
look-ahead. When two spikes are separated by less than one window length their
windows overlap and the union is marked as a single contiguous event block,
which is correct -- the operator retains control across the whole period
spanning both spikes.

The 'already marked' skip (step 2 below) means a spike that falls inside a
window opened by a later spike does not open a second window extending further
back. This prevents double-counting of that earlier interval against the annual
VPP energy cap in the LP.

This strategy is intentionally permissive: it marks every interval where forced
export could in principle be economically beneficial to the operator at the
configured threshold. Real VPP operators may trigger fewer events because they
have cheaper generation available, or stricter internal criteria. More complex
classification logic -- incorporating the operator's generation portfolio,
FCAS availability, or a learned dispatch policy -- can be substituted here
without changing the LP.
"""
import math
import numpy as np
import pandas as pd


def classify_vpp_events(
    prices_mwh: pd.Series,
    threshold_dollars_per_mwh: float,
    event_window_hours: float,
    interval_hours: float = 5 / 60,
) -> np.ndarray:
    """Mark intervals belonging to a VPP event window.

    Performs a backwards scan over the price series. Each interval whose
    wholesale price exceeds threshold_dollars_per_mwh (and which is not already
    inside a window opened by a later spike) triggers a window of
    event_window_hours ending at that interval (inclusive). All intervals in
    the window are marked True.

    Args:
        prices_mwh: NEM RRP in $/MWh for the full simulation period. Must
            contain no NaN values. Length T.
        threshold_dollars_per_mwh: Wholesale price above which an interval is
            treated as a price spike. Must be > 0. Set equal to
            exp_reward_c_per_kwh * 10 to mark all spikes where gross
            forced-export revenue covers the household reward; set higher to
            model a more conservative operator.
        event_window_hours: Length of the pre-spike window (hours) that the
            operator takes control of the battery before each spike. Must be
            > 0. (AGL example: 10.0 hours.)
        interval_hours: Length of each interval in hours. Default 5/60
            (5-minute intervals).

    Returns:
        Boolean numpy array of length T. True = operator controls the
        battery in that interval (VPP event); False = default regime.

    Raises:
        ValueError: if prices_mwh contains NaN, threshold <= 0, or
            event_window_hours <= 0.
    """
    if threshold_dollars_per_mwh <= 0:
        raise ValueError(
            f"threshold_dollars_per_mwh must be > 0, "
            f"got {threshold_dollars_per_mwh}"
        )
    if event_window_hours <= 0:
        raise ValueError(
            f"event_window_hours must be > 0, got {event_window_hours}"
        )

    prices = prices_mwh.to_numpy(dtype=float, copy=False)
    T = len(prices)

    if np.isnan(prices).any():
        n_nan = int(np.isnan(prices).sum())
        raise ValueError(
            f"classify_vpp_events: prices_mwh contains {n_nan} NaN "
            f"value(s). Missing prices must be resolved before classification."
        )

    # Number of intervals in one event window (rounded up so a fractional
    # window always covers at least the requested duration).
    window_intervals = math.ceil(event_window_hours / interval_hours)

    vpp_event = np.zeros(T, dtype=bool)

    for t in range(T - 1, -1, -1):
        # Skip if already inside a window opened by a later spike.
        if vpp_event[t]:
            continue
        if prices[t] > threshold_dollars_per_mwh:
            # Mark the window [max(0, t - window_intervals + 1), t] inclusive.
            start = max(0, t - window_intervals + 1)
            vpp_event[start : t + 1] = True

    return vpp_event
