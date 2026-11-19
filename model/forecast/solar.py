"""solar: generation forecasting from the site's own history.

See solar_forecast_spec.md. Everything here is naive relative to an operator
forecasting from satellite imagery, so a run using it bounds the cost of solar
forecast error from above rather than modelling any retailer's behaviour.

Solar splits into a deterministic part and a stochastic one. Sun position
fixes the shape; cloud fixes the level. So the envelope is estimated from
history and only the level -- one scalar per day -- is forecast.

Every index in this module is a RATIO OF SUMS, not a mean of per-slot ratios.
Being energy-weighted, it needs no daylight window: dawn and dusk carry almost
no envelope energy and so contribute almost nothing without being excluded.
It also makes the index and the elapsed-fraction `f` the same kind of object,
both fractions of envelope energy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.core.forecast import SolarSlice
from model.core.scenario import SolarForecastConfig
from model.forecast.clock import SLOTS_PER_DAY, ClockArrays, clock_from_wall_clock


@dataclass(frozen=True)
class DayTables:
    """What a controller holds throughout one local day.

    Attributes:
        envelope: (288,) clear-sky proxy, kWh per slot.
        day_energy: Total envelope energy over a nominal local day.
        r_prev: Index realised on the previous local day, or NaN.
        r_mean: Index over the whole trailing history.
        n_days: Valid history days behind the envelope.
        warm_up: True when n_days is below the configured history_days.
    """
    envelope: np.ndarray
    day_energy: float
    r_prev: float
    r_mean: float
    n_days: int
    warm_up: bool


def envelope_from(profiles: np.ndarray, quantile: float) -> np.ndarray:
    """Clear-sky proxy from a stack of daily profiles.

    A slot with no history in the window is an all-NaN column, which maps to
    zero envelope -- the right answer, not an error.

    Deliberately not np.nanquantile: with axis=0 that dispatches to
    apply_along_axis, a Python loop over all 288 columns, which dominated the
    fit harness at roughly 1.5 million calls per site. Sorting once (NaN sorts
    to the end) and indexing the per-column valid count is fully vectorised
    and matches numpy's linear interpolation exactly.
    """
    M = np.asarray(profiles, dtype=float)
    if M.size == 0:
        return np.zeros(SLOTS_PER_DAY)
    out = np.zeros(M.shape[1])
    cnt = np.count_nonzero(~np.isnan(M), axis=0)
    good = np.flatnonzero(cnt > 0)
    if good.size == 0:
        return out
    Ms = np.sort(M, axis=0)
    pos = (cnt[good] - 1) * quantile
    lo = np.floor(pos).astype(np.intp)
    hi = np.ceil(pos).astype(np.intp)
    v_lo = Ms[lo, good]
    v_hi = Ms[hi, good]
    out[good] = v_lo + (v_hi - v_lo) * (pos - lo)
    return np.nan_to_num(out)


def day_profiles(solar: np.ndarray, clock: ClockArrays,
                 config: SolarForecastConfig,
                 valid: np.ndarray | None = None
                 ) -> tuple[list[int], np.ndarray]:
    """Reduce each usable local day to a (288,) profile of energy per slot.

    Returns the day ordinals and the stacked profiles, NaN where a day has no
    interval in a slot. Split out from build_day_tables so the fit harness can
    derive envelopes for several history lengths and quantiles from a single
    pass, rather than re-deriving the profiles -- and, more importantly, so
    both use the same construction rather than two that can drift.
    """
    if valid is None:
        valid = np.ones(len(solar), dtype=bool)
    day_ord = clock.day_ordinal
    n_days_total = int(day_ord[-1]) + 1 if len(day_ord) else 0
    starts = np.searchsorted(day_ord, np.arange(n_days_total), side='left')
    ends = np.searchsorted(day_ord, np.arange(n_days_total), side='right')

    days, profs = [], []
    for d in range(n_days_total):
        s_, e_ = int(starts[d]), int(ends[d])
        if e_ <= s_:
            continue
        if (e_ - s_) < config.min_day_coverage * SLOTS_PER_DAY:
            continue
        if int(valid[s_:e_].sum()) < config.min_day_coverage * SLOTS_PER_DAY:
            continue
        day_solar = solar[s_:e_]
        if not np.isfinite(day_solar).all() or day_solar.sum() <= 0:
            continue
        # Mean, not sum, where a DST day maps two intervals to one slot: the
        # profile is energy in a typical 5-minute slot, not the day's total.
        tot = np.zeros(SLOTS_PER_DAY)
        cnt = np.zeros(SLOTS_PER_DAY)
        np.add.at(tot, clock.slot[s_:e_], day_solar)
        np.add.at(cnt, clock.slot[s_:e_], 1.0)
        days.append(d)
        profs.append(np.where(cnt > 0, tot / np.maximum(cnt, 1e-12), np.nan))
    return days, (np.vstack(profs) if profs else np.zeros((0, SLOTS_PER_DAY)))


def index_of(profile: np.ndarray, envelope: np.ndarray) -> float:
    """Ratio of sums index for one profile against an envelope."""
    present = ~np.isnan(profile)
    den = float((present * envelope).sum())
    return float(np.nansum(profile) / den) if den > 0 else np.nan


def build_day_tables(solar: np.ndarray, clock: ClockArrays,
                     config: SolarForecastConfig,
                     valid: np.ndarray | None = None) -> dict[int, DayTables]:
    """Tables in force for each local day, built from days strictly before it.

    Rebuilding per target day would leak: at a decision instant inside day d, a
    table for day d+1 would draw on day d, most of which has not happened.

    The envelope is a high quantile per slot rather than a maximum, so a single
    anomalous reading cannot define the clear-sky proxy for a slot.

    r_prev is measured against the envelope in force at the time rather than
    stored raw, because the same generation means a different index once the
    envelope moves.
    """
    days, profs = day_profiles(solar, clock, config, valid)
    out: dict[int, DayTables] = {}
    if not days:
        return out

    # Tables for day d draw on history days strictly before d, so the loop
    # walks the day list and looks only backwards.
    next_day = {d: (days[i + 1] if i + 1 < len(days) else None)
                for i, d in enumerate(days)}
    n_days_total = int(clock.day_ordinal[-1]) + 1
    pos = 0
    for d in range(n_days_total):
        while pos < len(days) and days[pos] < d:
            pos += 1
        if pos == 0:
            continue
        M = profs[max(0, pos - config.history_days):pos]
        envelope = envelope_from(M, config.envelope_quantile)
        day_energy = float(envelope.sum())
        if day_energy <= 0:
            continue
        present = ~np.isnan(M)
        den = float((present * envelope).sum())
        if den <= 0:
            continue
        out[d] = DayTables(
            envelope=envelope, day_energy=day_energy,
            r_prev=index_of(profs[pos - 1], envelope),
            r_mean=float(np.nansum(M) / den),
            n_days=M.shape[0],
            warm_up=M.shape[0] < config.history_days)
    del next_day
    return out


class SolarForecastProvider:
    """Solar forecast from the site's own trailing history.

    Args:
        solar: Actual generation, kWh per interval, over the whole period.
        clock: Per-interval local calendar facts for the same index.
        config: SolarForecastConfig.
        max_horizon_intervals: Longest window get() will be asked for.
        interval_hours: Interval length in hours.
        valid: Optional mask of intervals carrying a real measurement. Gates
            the ENVELOPE HISTORY only. Unlike load infill, pv infill is not the
            site's own history restated, so excluding it is a fitting nicety
            rather than a correctness requirement; dispatch passes None.
    """

    def __init__(self, solar: np.ndarray, clock: ClockArrays,
                 config: SolarForecastConfig, max_horizon_intervals: int,
                 interval_hours: float = 5 / 60,
                 valid: np.ndarray | None = None):
        if len(solar) != len(clock.slot):
            raise ValueError(
                f"solar covers {len(solar)} intervals but clock covers "
                f"{len(clock.slot)}")
        self._solar = np.asarray(solar, dtype=float)
        self._clock = clock
        self._cfg = config
        self._T = len(self._solar)
        self._H = int(max_horizon_intervals)
        self._interval_hours = interval_hours
        self._m = max(1, int(round(config.lookback_hours / interval_hours)))
        self._lag = SLOTS_PER_DAY          # persistence: 24 h of market time

        self._cum_solar = np.concatenate([[0.0], np.cumsum(self._solar)])
        self._tables = build_day_tables(self._solar, clock, config, valid)

        day_ord = clock.day_ordinal
        n_days_total = int(day_ord[-1]) + 1 if self._T else 0
        self._day_start = np.searchsorted(day_ord, np.arange(n_days_total),
                                          side='left')

        # Per build day, the envelope evaluated along the actual index, and its
        # running sum. Lets every "envelope energy between here and there" be a
        # subtraction rather than a gather.
        self._env: dict[int, tuple[int, np.ndarray]] = {}
        for d, t in self._tables.items():
            lo = max(0, int(self._day_start[d]) - self._m)
            hi = min(self._T, int(self._day_start[d]) + SLOTS_PER_DAY + self._H)
            arr = t.envelope[clock.slot[lo:hi]]
            self._env[d] = (lo, np.concatenate([[0.0], np.cumsum(arr)]))

        if config.decay_hours > 0:
            lead_h = (np.arange(self._H) + 1) * interval_hours
            self._decay = np.exp(-lead_h / config.decay_hours)
        else:
            self._decay = np.zeros(self._H)

    @classmethod
    def from_series(cls, solar_kwh, state: str, config: SolarForecastConfig,
                    max_horizon_intervals: int,
                    interval_hours: float = 5 / 60,
                    valid: np.ndarray | None = None) -> 'SolarForecastProvider':
        """Build from a generation Series and a state, resolving the clock."""
        from model.core.localtime import local_wall_clock
        wc = local_wall_clock(solar_kwh.index, state, interval_hours)
        return cls(np.asarray(solar_kwh, dtype=float),
                   clock_from_wall_clock(wc, interval_hours),
                   config, max_horizon_intervals, interval_hours, valid)

    @property
    def n_days_with_tables(self) -> int:
        return len(self._tables)

    def _env_sum(self, d: int, lo: int, hi: int) -> float:
        """Envelope energy over [lo, hi) under build day d's envelope."""
        start, cum = self._env[d]
        i, j = lo - start, hi - start
        if i < 0 or j > len(cum) - 1 or j <= i:
            return 0.0
        return float(cum[j] - cum[i])

    def get(self, k: int, n_intervals: int) -> SolarSlice | None:
        """Forecast for the window beginning at interval k.

        Reads generation only at labels at or before the decision instant,
        which is interval k-1 and earlier.
        """
        if not 0 <= k < self._T:
            raise ValueError(f"k={k} outside [0, {self._T})")
        n = int(min(n_intervals, self._T - k))
        if n > self._H:
            raise ValueError(
                f"requested {n} intervals but the provider was built for a "
                f"maximum horizon of {self._H}")

        if self._cfg.mode == 'persistence':
            if k - self._lag < 0:
                return None
            out = self._solar[k - self._lag:k - self._lag + n].copy()
            return SolarSlice(solar_kwh=np.maximum(out, 0.0))

        d = int(self._clock.day_ordinal[k])
        t = self._tables.get(d)
        if t is None or not np.isfinite(t.r_mean):
            return None
        start, _ = self._env[d]

        # History-only estimate. Formed once per day; the whole forecast
        # overnight, and the whole forecast for any later-day target.
        r_prev = t.r_prev if np.isfinite(t.r_prev) else t.r_mean
        w = self._cfg.w_prev_share
        r_hist = w * r_prev + (1.0 - w) * t.r_mean

        # Today so far. f is envelope energy elapsed as a fraction of the local
        # day's envelope energy, so it is zero overnight and one at dusk with
        # no separate daylight calculation.
        day0 = int(self._day_start[d])
        env_elapsed = self._env_sum(d, day0, k)
        f = env_elapsed / t.day_energy if t.day_energy > 0 else 0.0
        a_w = min(1.0, f / self._cfg.f_on) * self._cfg.w_today_max
        if env_elapsed > 0:
            r_today = (self._cum_solar[k] - self._cum_solar[day0]) / env_elapsed
        else:
            r_today, a_w = r_hist, 0.0
        r_blend = a_w * r_today + (1.0 - a_w) * r_hist

        # Additive index adjustment. With the index energy-weighted, the
        # residual reduces to the difference between the recently realised
        # index and the one being predicted.
        adj = 0.0
        if self._cfg.decay_hours > 0 and k > 0:
            lo = max(0, k - self._m, start)
            env_lb = self._env_sum(d, lo, k)
            if env_lb > 0:
                r_lb = (self._cum_solar[k] - self._cum_solar[lo]) / env_lb
                adj = r_lb - r_blend

        # Same-day targets get the blend; later days fall back to history,
        # since today's partial index says nothing about tomorrow beyond what
        # today's completed index will say, and that is not yet known.
        same_day = self._clock.day_ordinal[k:k + n] == d
        r_hat = np.where(same_day, r_blend, r_hist)

        idx = r_hat + adj * self._decay[:n]
        clipped = (idx < 0.0) | (idx > self._cfg.index_max)
        idx = np.clip(idx, 0.0, self._cfg.index_max)
        out = idx * t.envelope[self._clock.slot[k:k + n]]

        return SolarSlice(
            solar_kwh=np.maximum(out, 0.0),
            index_hist=float(r_hist), index_today=float(r_blend),
            intraday_weight=float(a_w), adjustment=float(adj),
            n_clipped=int(clipped.sum()), warm_up=t.warm_up)
