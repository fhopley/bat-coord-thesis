"""Unit tests for model.core.localtime.

Expected local-time values below were independently verified against
pandas' own tz database via tz_convert before being hard-coded here (not
derived from local_wall_clock itself), so these are genuine ground-truth
checks, not tautologies. See the design conversation for the verification
commands.

Key facts these tests are pinned to (2022 Australian DST calendar):
  - DST 2021-22 season ended 2022-04-03 (AEDT -> AEST).
  - DST 2022-23 season began 2022-10-02 (AEST -> AEDT).
  - QLD has no DST: QLD local == AEST year-round.
  - NSW and VIC share the same DST offset (+1h vs AEST when DST is active).
  - SA is on a permanent half-hour offset from AEST (ACST=UTC+9:30 in
    winter, ACDT=UTC+10:30 in summer/DST) -- never aligns to an AEST hour
    boundary at all.
"""
import numpy as np
import pandas as pd
import pytest

from model.core.localtime import local_wall_clock, LocalWallClock, STATE_TIMEZONE


INTERVAL_HOURS = 5 / 60


def _idx(timestamps: list[str]) -> pd.DatetimeIndex:
    """AEST-indexed timestamps (interval-END convention, as used everywhere
    else in the model)."""
    return pd.DatetimeIndex(timestamps, tz='Etc/GMT-10')


# ---------------------------------------------------------------------------
# State timezone table
# ---------------------------------------------------------------------------

def test_state_timezone_covers_all_four_nem_states():
    assert set(STATE_TIMEZONE.keys()) == {'NSW', 'QLD', 'SA', 'VIC'}


def test_unknown_state_raises():
    idx = _idx(['2022-01-15 12:05:00'])
    with pytest.raises(ValueError, match='unknown state'):
        local_wall_clock(idx, 'TAS', INTERVAL_HOURS)


# ---------------------------------------------------------------------------
# QLD: no DST, should exactly track AEST year-round (the control case)
# ---------------------------------------------------------------------------

def test_qld_matches_aest_in_summer():
    """QLD has no DST: local hour equals the AEST interval-start hour, even
    in January when NSW/VIC/SA are all offset."""
    idx = _idx(['2022-01-15 12:05:00'])
    result = local_wall_clock(idx, 'QLD', INTERVAL_HOURS)
    assert result.hour[0] == 12
    assert result.minute[0] == 0


def test_qld_matches_aest_in_winter():
    idx = _idx(['2022-07-15 12:05:00'])
    result = local_wall_clock(idx, 'QLD', INTERVAL_HOURS)
    assert result.hour[0] == 12
    assert result.minute[0] == 0


# ---------------------------------------------------------------------------
# NSW / VIC: DST-affected, +1h vs AEST when DST is active
# ---------------------------------------------------------------------------

def test_nsw_dst_offset_in_summer():
    """January: DST active. AEST 12:00 interval-start -> NSW local 13:00."""
    idx = _idx(['2022-01-15 12:05:00'])
    result = local_wall_clock(idx, 'NSW', INTERVAL_HOURS)
    assert result.hour[0] == 13
    assert result.minute[0] == 0


def test_nsw_no_offset_in_winter():
    """July: DST inactive. AEST 12:00 interval-start -> NSW local 12:00
    (NSW local == AEST in winter)."""
    idx = _idx(['2022-07-15 12:05:00'])
    result = local_wall_clock(idx, 'NSW', INTERVAL_HOURS)
    assert result.hour[0] == 12
    assert result.minute[0] == 0


def test_vic_dst_offset_matches_nsw():
    """VIC shares NSW's DST offset."""
    idx = _idx(['2022-01-15 12:05:00'])
    result = local_wall_clock(idx, 'VIC', INTERVAL_HOURS)
    assert result.hour[0] == 13


# ---------------------------------------------------------------------------
# SA: permanent half-hour offset, on top of its own DST
# ---------------------------------------------------------------------------

def test_sa_half_hour_offset_in_summer():
    """January: SA is ACDT = AEST + 30min. AEST 12:00 -> SA local 12:30."""
    idx = _idx(['2022-01-15 12:05:00'])
    result = local_wall_clock(idx, 'SA', INTERVAL_HOURS)
    assert result.hour[0] == 12
    assert result.minute[0] == 30


def test_sa_half_hour_offset_in_winter():
    """July: SA is ACST = AEST - 30min. AEST 12:00 -> SA local 11:30."""
    idx = _idx(['2022-07-15 12:05:00'])
    result = local_wall_clock(idx, 'SA', INTERVAL_HOURS)
    assert result.hour[0] == 11
    assert result.minute[0] == 30


# ---------------------------------------------------------------------------
# Weekday/weekend and date boundary: SA's half-hour offset can put an
# interval on a different calendar day and a different day-of-week to AEST.
# This is the concrete case that motivates the whole timezone migration.
# ---------------------------------------------------------------------------

def test_sa_local_weekday_differs_from_aest_at_midnight_boundary():
    """AEST Monday 00:00 interval-start is still Sunday 23:30 in SA local
    time (winter, -30min offset) -- a different weekday, different weekend
    classification, and different calendar date to AEST.
    """
    idx = _idx(['2022-07-04 00:05:00'])  # interval-end; start = Mon 00:00 AEST
    aest_start = idx - pd.Timedelta(hours=INTERVAL_HOURS)
    assert aest_start.dayofweek[0] == 0  # Monday in AEST, for context

    result = local_wall_clock(idx, 'SA', INTERVAL_HOURS)
    assert result.hour[0] == 23
    assert result.dayofweek[0] == 6          # Sunday in SA local time
    assert result.is_weekend[0] == True      # weekend in SA local, not in AEST
    assert result.date[0] == pd.Timestamp('2022-07-03', tz='Australia/Adelaide')


def test_nsw_local_weekday_matches_aest_at_same_boundary():
    """The same instant in NSW (no half-hour offset, and no DST active in
    July) stays Monday -- confirms the SA case above is genuinely about
    SA's offset, not a bug shared by all states."""
    idx = _idx(['2022-07-04 00:05:00'])
    result = local_wall_clock(idx, 'NSW', INTERVAL_HOURS)
    assert result.hour[0] == 0
    assert result.dayofweek[0] == 0
    assert result.is_weekend[0] == False
    assert result.date[0] == pd.Timestamp('2022-07-04', tz='Australia/Sydney')


# ---------------------------------------------------------------------------
# Shape and dtype invariants
# ---------------------------------------------------------------------------

def test_output_arrays_match_index_length():
    idx = _idx(['2022-01-15 12:05:00', '2022-01-15 12:10:00', '2022-01-15 12:15:00'])
    result = local_wall_clock(idx, 'NSW', INTERVAL_HOURS)
    assert len(result.hour) == 3
    assert len(result.minute) == 3
    assert len(result.dayofweek) == 3
    assert len(result.is_weekend) == 3
    assert len(result.date) == 3


def test_is_weekend_dtype_is_bool():
    idx = _idx(['2022-01-15 12:05:00'])
    result = local_wall_clock(idx, 'NSW', INTERVAL_HOURS)
    assert result.is_weekend.dtype == np.bool_


def test_hour_range_valid():
    """Sanity check across a full day of 5-min intervals: hour always in
    [0, 23], minute always in [0, 59], for every state."""
    idx = pd.date_range('2022-01-01 00:05', periods=288, freq='5min', tz='Etc/GMT-10')
    for state in STATE_TIMEZONE:
        result = local_wall_clock(idx, state, INTERVAL_HOURS)
        assert (result.hour >= 0).all() and (result.hour <= 23).all()
        assert (result.minute >= 0).all() and (result.minute <= 59).all()
        assert (result.dayofweek >= 0).all() and (result.dayofweek <= 6).all()


# ---------------------------------------------------------------------------
# DST transition days: local calendar days can be 23h or 25h long. Grouping
# by `.date` (an actual calendar-date component from a properly localized
# timestamp) handles this correctly with no special-casing -- these tests
# confirm that rather than assume it.
# ---------------------------------------------------------------------------

def test_dst_spring_forward_day_has_23_local_hours_nsw():
    """2022-10-02 is the DST start date for NSW (clocks skip 2am -> 3am
    local, so that local hour never occurs). A full local calendar day of
    5-min intervals should contain 23*12=276 intervals, not 288 -- grouping
    by `.date` (a true local calendar-date component) gets this right with
    no special-casing, purely because the AEST instants that fall in the
    skipped local hour don't exist to be grouped there.

    Uses a 3-AEST-day window centred on the transition so the local
    calendar day's true boundaries (which don't align with AEST midnight
    once DST is offsetting the clock) are fully captured.
    """
    start_aest = pd.Timestamp('2022-10-01', tz='Etc/GMT-10')
    idx = pd.date_range(start_aest + pd.Timedelta(minutes=5), periods=3 * 288,
                        freq='5min', tz='Etc/GMT-10')
    result = local_wall_clock(idx, 'NSW', INTERVAL_HOURS)
    same_local_day = result.date == pd.Timestamp('2022-10-02', tz='Australia/Sydney')
    n_intervals = int(same_local_day.sum())
    assert n_intervals == 276, (
        f"Expected 276 intervals (23h) on the spring-forward local day, got {n_intervals}"
    )


def test_dst_autumn_fallback_day_has_25_local_hours_nsw():
    """2022-04-03 is the DST end date for NSW (clocks repeat 2am-3am local,
    so that local hour occurs twice as two distinct real instants). A full
    local calendar day of 5-min intervals should contain 25*12=300
    intervals, not 288.
    """
    start_aest = pd.Timestamp('2022-04-02', tz='Etc/GMT-10')
    idx = pd.date_range(start_aest + pd.Timedelta(minutes=5), periods=3 * 288,
                        freq='5min', tz='Etc/GMT-10')
    result = local_wall_clock(idx, 'NSW', INTERVAL_HOURS)
    same_local_day = result.date == pd.Timestamp('2022-04-03', tz='Australia/Sydney')
    n_intervals = int(same_local_day.sum())
    assert n_intervals == 300, (
        f"Expected 300 intervals (25h) on the fall-back local day, got {n_intervals}"
    )
