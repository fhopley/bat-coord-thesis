"""Local (state-based) wall-clock derivation for tariff time-of-day logic.

AEST (Etc/GMT-10, fixed UTC+10, no DST) is the model's canonical index --
wholesale prices and physical energy flows are genuinely NEM/market-time
quantities and stay in AEST throughout the model. Tariff clock-driven
overlays (network and retail TOU periods, block windows, zero-import reward
windows, no-grid-charge windows, the TOU export-order nudge) are specified
by retailers and DNSPs in real local time, which is state-dependent and
DST-aware. QLD has no DST, so QLD local time equals AEST year-round. NSW and
VIC observe the same DST offset as each other. SA is on a permanent
half-hour offset from AEST (ACST/ACDT), on top of its own DST.

This module is the single source of truth for converting an AEST interval
index into the local wall-clock values those overlays need. It does not
touch wholesale prices or physical energy flows -- those stay in AEST,
correctly, everywhere else in the model.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


# Single source of truth for state -> local IANA timezone. Local timezone is
# always derived from the site's actual location (scenario.state), not from
# a tariff's declared state, so that a deliberate cross-state tariff
# sensitivity run still applies the site's real local clock.
STATE_TIMEZONE: dict[str, str] = {
    'NSW': 'Australia/Sydney',
    'VIC': 'Australia/Melbourne',
    'QLD': 'Australia/Brisbane',
    'SA':  'Australia/Adelaide',
}


@dataclass(frozen=True)
class LocalWallClock:
    """Per-interval local wall-clock values for a state, aligned to an index.

    All four arrays are derived from the interval-START instant (index
    timestamp minus interval_hours), respecting the interval-end timestamp
    convention used throughout the model. This matches the convention
    already used by period_rate_series, block_window_mask, and every other
    existing hour-of-day lookup -- the rate/window/day for an interval is
    the one applying at the instant the interval began.

    Attributes:
        hour: int ndarray, 0-23, local hour-of-day at the interval-start
            instant.
        minute: int ndarray, 0-59, local minute-of-hour. Needed only by
            sub-hour lookups (the TOU export-order nudge); everything else
            uses `hour` alone.
        dayofweek: int ndarray, Monday=0 .. Sunday=6, at the interval-start
            instant in local time.
        is_weekend: bool ndarray, True where dayofweek is Saturday or
            Sunday in local time.
        date: pd.DatetimeIndex of local calendar dates (normalized, i.e.
            midnight, tz-aware in the state's local timezone). Used for
            daily-reset grouping: block tier cumulative thresholds,
            zero-import reward day qualification, and _build_day_map's
            per-day LP variable construction. Grouping by this value gives
            each *local* calendar day its own group, including the 23-hour
            and 25-hour days either side of a DST transition -- no special
            casing is needed because grouping is by actual date component,
            not by a fixed interval count per day.
    """
    hour: np.ndarray
    minute: np.ndarray
    dayofweek: np.ndarray
    is_weekend: np.ndarray
    date: pd.DatetimeIndex


def local_wall_clock(
    index: pd.DatetimeIndex,
    state: str,
    interval_hours: float,
) -> LocalWallClock:
    """Convert an AEST interval index to local wall-clock values for a state.

    Args:
        index: AEST-indexed (Etc/GMT-10 or equivalent fixed-offset
            tz-aware) interval-end timestamps, as used throughout
            DataContext, SiteTraces, and the price series.
        state: One of 'NSW', 'QLD', 'SA', 'VIC'.
        interval_hours: Interval length in hours (e.g. 5/60 for 5-minute
            data), used to compute the interval-start instant.

    Returns:
        LocalWallClock with hour, minute, dayofweek, is_weekend, and date
        arrays aligned to `index`.

    Raises:
        ValueError: if `state` is not a recognised state.

    Note on DST transitions: converting from an unambiguous fixed-offset
    instant (AEST) into a local zone is always well-defined in this
    direction -- there is no "ambiguous" or "nonexistent" local time issue
    here (that only arises when *localizing* a naive local timestamp, not
    when converting a real instant into its local representation). Two
    distinct AEST instants can map to the same local wall-clock hour label
    during the DST "fall back" transition (an hour that occurs twice in
    local time); this is expected and correct -- each is still a distinct,
    unambiguous real instant, and the tariff rate that applies at each is
    the one for that instant's genuine local hour.
    """
    if state not in STATE_TIMEZONE:
        raise ValueError(
            f"local_wall_clock: unknown state {state!r}, expected one of "
            f"{sorted(STATE_TIMEZONE)}"
        )

    interval_delta = pd.Timedelta(hours=interval_hours)
    start_instants_aest = index - interval_delta
    local = start_instants_aest.tz_convert(STATE_TIMEZONE[state])

    dayofweek = local.dayofweek.to_numpy()
    return LocalWallClock(
        hour=local.hour.to_numpy(),
        minute=local.minute.to_numpy(),
        dayofweek=dayofweek,
        is_weekend=(dayofweek >= 5),
        date=local.normalize(),
    )
