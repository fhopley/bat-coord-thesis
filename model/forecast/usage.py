"""usage: household load forecasting from the site's own history.

Answers, for a decision instant and a window length, "what did the controller
think the household would consume?" -- the load-side counterpart to
model/forecast/*.py on the price side, and deliberately the same shape.

See usage_forecast_spec.md for the design. Three things are worth knowing
before reading the code.

Level and shape are separate
----------------------------
A day's forecast is a scalar daily energy multiplied by a normalised
time-of-day shape that sums to one. They are estimated separately because the
day-of-week signal in household load lives almost entirely in the level, not
the shape: over a 28-day history there are only four samples per day-of-week,
which is far too few for a 5-minute slot but is fine for a daily total, since
daily totals are an order of magnitude more stable than 5-minute values.

The profile is rebuilt once per local day, not per interval
-----------------------------------------------------------
Tables built at the start of local day d use history strictly before day d,
and are then applied to EVERY target in the window, including targets falling
on day d+1. Rebuilding per target day would be a leak: at a decision instant
inside day d, a table for day d+1 would have been built from history that
includes day d, most of which has not happened yet.

Only actuals strictly at or before the decision instant are read
---------------------------------------------------------------
Load differs from price here. The settled price for the interval ending at D
is not published at D, which is why the price provider needs a lead-0
substitution. Metered load for the interval ending at D IS observed at D by a
local controller. So the rule is one interval more permissive than the price
side: labels <= decision_instant are readable, labels > decision_instant are
not. That off-by-one is asserted, not assumed, because a leak here would be
invisible in every downstream artefact.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from model.core.forecast import LoadSlice
from model.core.scenario import UsageForecastConfig
from model.forecast.clock import (
    SLOTS_PER_DAY, ClockArrays, clock_from_wall_clock,
)

_WEEKDAY, _WEEKEND = 0, 1


# ---------------------------------------------------------------------------
# Clock arrays
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Profile tables
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DayTables:
    """The profile in force for all decisions made during one local day.

    Attributes:
        level: (7,) mean daily energy in kWh, indexed by day-of-week.
        shape: (2, 288) normalised time-of-day shape summing to 1 along the
            slot axis, row 0 weekday and row 1 weekend.
        n_days: Valid history days behind these tables.
        warm_up: True when n_days is below the configured history_days, i.e.
            the tables are built on a short history near the start of a period.
    """
    level: np.ndarray
    shape: np.ndarray
    n_days: int
    warm_up: bool


class _RollingHistory:
    """Trailing window of valid history days, maintained incrementally.

    Rebuilding the tables from scratch for every local day of a five-year run
    would be O(days * history_days * 288). Adding one day and dropping one day
    is O(288), which is what makes the EDP-scale run comfortable.
    """

    def __init__(self, config: UsageForecastConfig):
        self._cfg = config
        self._days: deque = deque()
        self._shape_sum = np.zeros((2, SLOTS_PER_DAY))
        self._shape_cnt = np.zeros((2, SLOTS_PER_DAY))
        self._energy: list[deque] = [deque() for _ in range(7)]

    def push(self, daytype: int, dow: int, energy: float,
             slots: np.ndarray, norm_load: np.ndarray) -> None:
        """Add one valid history day, evicting the oldest if the window is full."""
        np.add.at(self._shape_sum[daytype], slots, norm_load)
        np.add.at(self._shape_cnt[daytype], slots, 1.0)
        self._energy[dow].append(energy)
        self._days.append((daytype, dow, energy, slots, norm_load))
        while len(self._days) > self._cfg.history_days:
            self._evict()

    def _evict(self) -> None:
        daytype, dow, _energy, slots, norm_load = self._days.popleft()
        np.subtract.at(self._shape_sum[daytype], slots, norm_load)
        np.subtract.at(self._shape_cnt[daytype], slots, 1.0)
        self._energy[dow].popleft()

    def __len__(self) -> int:
        return len(self._days)

    def tables(self) -> DayTables | None:
        """Materialise the tables for the current window, or None if too short.

        The fallback ladder is deliberate and deterministic. A day-type with no
        history falls back to the all-days shape; a day-of-week with no history
        falls back to the day-type mean level; a day-type with no level history
        falls back to the all-days mean. Near the start of a period, or for a
        site whose first weekend has not yet occurred, this degrades smoothly
        instead of producing a hole.
        """
        cfg = self._cfg
        n = len(self._days)
        if n < cfg.min_history_days:
            return None

        # ---- Shape ----
        all_sum = self._shape_sum.sum(axis=0)
        all_cnt = self._shape_cnt.sum(axis=0)
        all_shape = _finish_shape(all_sum, all_cnt, cfg.smoothing_intervals)
        if all_shape is None:
            return None

        shape = np.empty((2, SLOTS_PER_DAY))
        for dt in (_WEEKDAY, _WEEKEND):
            s = _finish_shape(self._shape_sum[dt], self._shape_cnt[dt],
                              cfg.smoothing_intervals)
            shape[dt] = all_shape if s is None else s

        # ---- Level ----
        by_dow = [np.mean(v) if v else np.nan for v in self._energy]
        dow_daytype = np.array([_WEEKDAY] * 5 + [_WEEKEND] * 2)
        by_daytype = np.full(2, np.nan)
        for dt in (_WEEKDAY, _WEEKEND):
            vals = [e for d in range(7) if dow_daytype[d] == dt
                    for e in self._energy[d]]
            if vals:
                by_daytype[dt] = float(np.mean(vals))
        all_vals = [e for d in range(7) for e in self._energy[d]]
        overall = float(np.mean(all_vals)) if all_vals else np.nan
        by_daytype = np.where(np.isnan(by_daytype), overall, by_daytype)

        level = np.empty(7)
        for d in range(7):
            fallback = by_daytype[dow_daytype[d]]
            if cfg.dow_level and not np.isnan(by_dow[d]):
                level[d] = by_dow[d]
            else:
                level[d] = fallback

        if not np.all(np.isfinite(level)):
            return None

        return DayTables(level=level, shape=shape, n_days=n,
                         warm_up=n < cfg.history_days)


def _finish_shape(total: np.ndarray, count: np.ndarray,
                  smoothing: int) -> np.ndarray | None:
    """Mean-across-days, smooth circularly, renormalise to sum 1.

    Slots with no history in the window (a 23-hour local day on the spring DST
    transition contributes nothing to twelve of them) are filled by the
    smoothing pass rather than special-cased, because a zero there would be a
    claim that the household uses no power at 02:15, which is a different
    statement from having no observation.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(count > 0, total / np.maximum(count, 1e-12), np.nan)
    if np.all(np.isnan(mean)):
        return None
    if np.any(np.isnan(mean)):
        # Circular linear fill of the missing slots before smoothing.
        idx = np.arange(SLOTS_PER_DAY)
        good = ~np.isnan(mean)
        mean = np.interp(idx, idx[good], mean[good], period=SLOTS_PER_DAY)
    if smoothing > 1:
        mean = uniform_filter1d(mean, size=smoothing, mode='wrap')
    total_sum = mean.sum()
    if not np.isfinite(total_sum) or total_sum <= 0:
        return None
    return mean / total_sum


def build_day_tables(load: np.ndarray, clock: ClockArrays,
                     config: UsageForecastConfig,
                     valid: np.ndarray | None = None) -> dict[int, DayTables]:
    """Build the tables in force for each local day of the period.

    Returns a mapping from local day ordinal to the tables a controller would
    have held throughout that day, i.e. built from days strictly before it.
    Days with no usable history are absent from the mapping.

    Only whole local days that are fully inside the period contribute to the
    history. The first and last local days of a period are usually partial, and
    a partial day has a depressed daily total that would bias the level
    downward, so the coverage gate excludes them without needing a special
    case.

    `valid` marks intervals that carry a real measurement. Intervals that were
    INFILLED must be excluded, and one class of them makes this essential
    rather than tidy: the EDP pipeline's Method 2 load infill is same-day-of-week
    copying, which is close to what this module does. A day rebuilt that way is
    not independent evidence about the household's habits, it is a restatement
    of them, and letting it into the history would inflate the apparent
    regularity of the profile. The gate is applied at day level, against
    `min_day_coverage`, so a lightly patched day still contributes and a mostly
    synthetic one does not.
    """
    if valid is None:
        valid = np.ones(len(load), dtype=bool)
    elif len(valid) != len(load):
        raise ValueError(
            f"valid covers {len(valid)} intervals but load covers {len(load)}")

    day_ord = clock.day_ordinal
    n_days_total = int(day_ord[-1]) + 1 if len(day_ord) else 0
    starts = np.searchsorted(day_ord, np.arange(n_days_total), side='left')
    ends = np.searchsorted(day_ord, np.arange(n_days_total), side='right')

    hist = _RollingHistory(config)
    out: dict[int, DayTables] = {}

    for d in range(n_days_total):
        # Tables for day d are built from days strictly before d, so they are
        # materialised BEFORE day d is pushed into the history.
        t = hist.tables()
        if t is not None:
            out[d] = t

        s, e = int(starts[d]), int(ends[d])
        if e <= s:
            continue
        n_intervals = int(valid[s:e].sum())
        expected = SLOTS_PER_DAY
        if (e - s) < config.min_day_coverage * expected:
            continue
        if n_intervals < config.min_day_coverage * expected:
            continue
        day_load = load[s:e]
        energy = float(day_load.sum())
        if not np.isfinite(energy) or energy <= 0:
            continue
        hist.push(
            daytype=_WEEKEND if bool(clock.is_weekend[s]) else _WEEKDAY,
            dow=int(clock.dayofweek[s]),
            energy=energy,
            slots=clock.slot[s:e],
            norm_load=day_load / energy,
        )
    return out


# ---------------------------------------------------------------------------
# Base forecast
# ---------------------------------------------------------------------------

def base_forecast(tables: DayTables, clock: ClockArrays,
                  lo: int, hi: int) -> np.ndarray:
    """Base forecast over interval positions [lo, hi), under one day's tables.

    Vectorised over the span. Applying one build day's tables to targets on a
    later calendar day is the intended behaviour, not an approximation: it is
    what a controller that retrains overnight actually holds.
    """
    dt = clock.is_weekend[lo:hi].astype(np.int64)
    return (tables.level[clock.dayofweek[lo:hi]]
            * tables.shape[dt, clock.slot[lo:hi]])


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class UsageForecastProvider:
    """Load forecast from the site's own trailing history.

    Constructed once per site. get() is called once per 5-minute interval,
    roughly 105,000 times per site-year, so the tables and the per-build-day
    base arrays are precomputed and the hot path is a slice plus one multiply.

    Args:
        load: Actual gross load, kWh per interval, over the whole period.
        clock: Per-interval local calendar facts for the same index.
        config: UsageForecastConfig.
        max_horizon_intervals: Longest window get() will be asked for. Sets how
            far each build day's base array reaches forward.
        interval_hours: Interval length in hours.
        valid: Optional per-interval mask of real measurements. Infilled
            intervals must be excluded; see build_day_tables. Only the PROFILE
            HISTORY is gated by it. The adjustment still reads whatever the
            trace holds at the decision instant, because a controller reading
            its own meter sees a real number there -- the infill is an artefact
            of this dataset, not of the controller's information.
    """

    def __init__(self, load: np.ndarray, clock: ClockArrays,
                 config: UsageForecastConfig, max_horizon_intervals: int,
                 interval_hours: float = 5 / 60,
                 valid: np.ndarray | None = None):
        if len(load) != len(clock.slot):
            raise ValueError(
                f"load covers {len(load)} intervals but clock covers "
                f"{len(clock.slot)}"
            )
        self._load = np.asarray(load, dtype=float)
        self._clock = clock
        self._cfg = config
        self._T = len(self._load)
        self._interval_hours = interval_hours
        self._H = int(max_horizon_intervals)
        self._m = max(1, int(round(config.lookback_hours / interval_hours)))

        self._tables = build_day_tables(self._load, clock, config, valid)

        day_ord = clock.day_ordinal
        n_days_total = int(day_ord[-1]) + 1 if self._T else 0
        self._day_start = np.searchsorted(day_ord, np.arange(n_days_total),
                                          side='left')
        self._day_end = np.searchsorted(day_ord, np.arange(n_days_total),
                                        side='right')

        # Per build day, the base forecast over everything that day's decisions
        # can reach: back by the adjustment lookback, forward by the horizon.
        self._base: dict[int, tuple[int, np.ndarray]] = {}
        for d, t in self._tables.items():
            lo = max(0, int(self._day_start[d]) - self._m)
            hi = min(self._T, int(self._day_end[d]) + self._H)
            self._base[d] = (lo, base_forecast(t, clock, lo, hi))

        # Decay applied by window position. Position j covers the interval
        # ending (j+1) intervals after the decision instant, so lead is
        # measured to interval end. decay_hours == 0 disables the adjustment.
        if config.decay_hours > 0:
            lead_h = (np.arange(self._H) + 1) * interval_hours
            self._decay = np.exp(-lead_h / config.decay_hours)
        else:
            self._decay = np.zeros(self._H)

    @classmethod
    def from_series(cls, load_kwh, state: str,
                    config: UsageForecastConfig,
                    max_horizon_intervals: int,
                    interval_hours: float = 5 / 60,
                    valid: np.ndarray | None = None
                    ) -> 'UsageForecastProvider':
        """Build from a load Series and a state, resolving the local clock.

        The intended construction point for callers, so nothing outside this
        module needs to know about ClockArrays.

        local_wall_clock is imported lazily so the forecasting maths above
        stays importable, and testable, without the timezone machinery.

        Args:
            load_kwh: Actual gross load, kWh per interval, indexed on the
                component's market-time index.
            state: Site state. Sets the LOCAL day boundary, which is what the
                day-type split and the day-of-week level key off. A wrong
                state silently mislabels every weekend.
            config: UsageForecastConfig.
            max_horizon_intervals: Longest window get() will be asked for.
            interval_hours: Interval length in hours.
            valid: Optional mask of intervals carrying a real measurement.
                Infilled intervals must be excluded from the profile history.
        """
        from model.core.localtime import local_wall_clock
        wc = local_wall_clock(load_kwh.index, state, interval_hours)
        return cls(np.asarray(load_kwh, dtype=float),
                   clock_from_wall_clock(wc, interval_hours),
                   config, max_horizon_intervals, interval_hours, valid)

    # -- diagnostics -------------------------------------------------------

    @property
    def n_days_with_tables(self) -> int:
        return len(self._tables)

    def base_at(self, d: int, lo: int, hi: int) -> np.ndarray | None:
        """Base forecast over [lo, hi) under build day d's tables, or None.

        Exposed for the fit harness and the check script, which need the base
        residual without going through get().
        """
        entry = self._base.get(d)
        if entry is None:
            return None
        start, arr = entry
        if lo < start or hi > start + len(arr):
            return None
        return arr[lo - start:hi - start]

    # -- the contract ------------------------------------------------------

    def get(self, k: int, n_intervals: int) -> LoadSlice | None:
        """Forecast for the window beginning at interval k.

        Indexed by interval position rather than by timestamp, unlike the price
        provider. The load forecast is built from the site's own trace on the
        component's own index, so there is no external run store to align to
        and a timestamp lookup would be a searchsorted per call for nothing.

        Returns None when no tables exist for the decision instant's local day,
        which happens only at the very start of a period.
        """
        if not 0 <= k < self._T:
            raise ValueError(f"k={k} outside [0, {self._T})")
        n = int(min(n_intervals, self._T - k))
        if n > self._H:
            raise ValueError(
                f"requested {n} intervals but the provider was built for a "
                f"maximum horizon of {self._H}"
            )

        d = int(self._clock.day_ordinal[k])
        entry = self._base.get(d)
        if entry is None:
            return None
        start, arr = entry

        base = arr[k - start:k - start + n]

        # Adjustment. The lookback covers intervals labelled at or before the
        # decision instant, which is interval k-1 and earlier. The upper bound
        # is exclusive at k, and that is the whole causality rule.
        lo = max(0, k - self._m)
        n_lb = k - lo
        adj = 0.0
        if n_lb > 0 and self._cfg.decay_hours > 0:
            resid = self._load[lo:k] - arr[lo - start:k - start]
            adj = float(resid.mean())

        raw = base + adj * self._decay[:n]
        clipped = raw < 0.0
        out = np.where(clipped, 0.0, raw)

        return LoadSlice(
            load_kwh=out,
            adjustment_kwh=adj,
            n_lookback=n_lb,
            n_clipped=int(clipped.sum()),
            warm_up=self._tables[d].warm_up,
        )
