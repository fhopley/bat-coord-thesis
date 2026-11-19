"""Local calendar facts per interval, shared by the forecast providers.

Held apart from any one forecaster because both the usage and solar providers
key off the LOCAL day boundary, and neither should have to import the other to
get it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SLOTS_PER_DAY = 288           # 5-minute slots in a nominal 24 hour local day


@dataclass(frozen=True)
class ClockArrays:
    """Per-interval local calendar facts, one entry per interval of the period.

    In production these come straight from
    model.core.localtime.local_wall_clock(index, state, interval_hours), which
    already evaluates at the interval-START instant -- which is exactly the
    decision instant. Held as a separate type so the forecasting maths is
    testable over plain arrays with no timezone machinery in the way.

    Attributes:
        slot: Local time-of-day slot, 0 .. 287, from local hour and minute.
        dayofweek: Monday=0 .. Sunday=6, local.
        is_weekend: True on Saturday and Sunday, local.
        day_ordinal: Consecutive integer per local calendar date, 0-based.
            Not a date; only equality and ordering are used.
    """
    slot: np.ndarray
    dayofweek: np.ndarray
    is_weekend: np.ndarray
    day_ordinal: np.ndarray


def clock_from_wall_clock(wc, interval_hours: float = 5 / 60) -> ClockArrays:
    """Adapt a model.core.localtime.LocalWallClock into ClockArrays.

    The only place this module touches the production timezone type. Kept as a
    free function so the maths above it never imports pandas.
    """
    slot = (np.asarray(wc.hour, dtype=np.int64) * 12
            + np.asarray(wc.minute, dtype=np.int64) // 5)
    dates = wc.date
    # date is a normalised DatetimeIndex; factorize preserves order because the
    # index is monotonic, so the codes are consecutive day ordinals.
    day_ordinal = np.asarray(dates.factorize()[0], dtype=np.int64)
    return ClockArrays(
        slot=slot,
        dayofweek=np.asarray(wc.dayofweek, dtype=np.int64),
        is_weekend=np.asarray(wc.is_weekend, dtype=bool),
        day_ordinal=day_ordinal,
    )
