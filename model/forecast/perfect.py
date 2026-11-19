"""Providers built from actual prices. No external data required.

PerfectForecastProvider is the correctness gate for rolling_optimisation.
TruncatedPerfectForecastProvider is rung 2 of the measurement ladder: zero
forecast error, realistic availability structure.

Both take the actual price series the model already has, so the whole rolling
component can be built and validated before any AEMO forecast data exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.core.forecast import ForecastSlice
from model.forecast.assembly import (
    assemble_window, mode_allowance_intervals, uses_actual_for_committed,
)
from model.forecast.runset import RunSet

_P5MIN_TARGETS = 12          # leads 0..55 min, constant, no exceptions observed
_COARSE_INTERVALS = 6        # 30 min at 5-minute resolution


def _positions(index: pd.DatetimeIndex, interval_minutes: int) -> tuple[pd.Timestamp, int]:
    """(grid start, interval minutes), validating the index is a regular grid."""
    if len(index) < 2:
        raise ValueError("price index must contain at least two intervals")
    step = index[1] - index[0]
    if step != pd.Timedelta(minutes=interval_minutes):
        raise ValueError(
            f"price index step is {step}, expected {interval_minutes} minutes")
    return index[0], interval_minutes


class _GridProvider:
    """Shared position arithmetic for providers built on the actuals index."""

    def __init__(self, prices_mwh: pd.Series, interval_minutes: int = 5):
        if prices_mwh.isna().any():
            raise ValueError("prices_mwh contains NaN")
        self._index = prices_mwh.index
        self._start, self._step = _positions(self._index, interval_minutes)
        self._actuals = prices_mwh.to_numpy(dtype=np.float64)

    def _pos(self, ts: pd.Timestamp) -> int:
        delta = ts - self._start
        q, r = divmod(int(delta.total_seconds()), self._step * 60)
        if r != 0:
            raise ValueError(
                f"{ts} does not fall on the {self._step}-minute grid starting "
                f"{self._start}"
            )
        return q

    def _ts(self, pos: int) -> pd.Timestamp:
        return self._start + pd.Timedelta(minutes=self._step * pos)


class PerfectForecastProvider(_GridProvider):
    """Actual prices, unlimited horizon, no infill.

    Not a benchmark. It exists for the correctness gate: with the horizon set
    to the full period, every rolling window reaches the period end and pins
    the same global terminal SOC, so each is the tail subproblem of the first
    and the committed sequence must reproduce perfect_foresight_lp.

    This is the one provider that legitimately reads actuals at and beyond the
    decision instant. It therefore bypasses the causality assertion BY
    CONSTRUCTION rather than by disabling it: it does not go through
    assemble_window at all. That exemption is confined to this class, and is
    the reason it must never produce a reported result.
    """

    def horizon_available(self) -> int:
        return len(self._actuals)

    def get(self, decision_instant: pd.Timestamp,
            n_intervals: int) -> ForecastSlice | None:
        k = self._pos(decision_instant) + 1
        if k >= len(self._actuals):
            return None
        stop = min(k + n_intervals, len(self._actuals))
        vals = self._actuals[k:stop]
        # No splice, so the provenance fields are trivial rather than absent:
        # there is one notional run and every value comes from it.
        return ForecastSlice(
            run_time=decision_instant,
            coarse_run_time=decision_instant,
            rrp_mwh=vals.copy(),
            is_infill=np.zeros(vals.size, dtype=bool),
            is_fine=np.ones(vals.size, dtype=bool),
        )


class TruncatedPerfectForecastProvider(_GridProvider):
    """Actual prices restricted to the availability structure of pre-dispatch.

    Forecast error is exactly zero; everything else is realistic. The gap
    against perfect foresight therefore isolates horizon truncation, terminal
    pin drift and (optionally) coarse resolution loss, with no forecast error
    mixed in.

    The sawtooth is the closed form validated against all 1,345 February 2026
    runs at 100% match:

        R = latest 30-minute boundary at or before the decision instant
        horizon_end(R) = 04:00 on (R.date + 2 days) if R.hour >= 13
                         04:00 on (R.date + 1 day)  otherwise

    Both 13:00 and the 04:00 trading day boundary are MARKET time, and the
    model index is already market time (Etc/GMT-10), so no local-time
    conversion is involved. core/localtime.py is for tariff periods, which are
    specified in local time, and does not apply here.

    coarse_resolution controls what the coarse segment carries:
        '5min'  actual 5-minute prices, isolating horizon and pin effects alone
        '30min' 30-minute means, adding the cost of only seeing half-hour
                averages beyond the first hour

    Running both separates those two costs. Neither needs new data.
    """

    def __init__(self, prices_mwh: pd.Series, interval_minutes: int = 5,
                 coarse_resolution: str = '30min',
                 mode: str = 'forecast',
                 infill_lag_intervals: int = 288):
        super().__init__(prices_mwh, interval_minutes)
        if coarse_resolution not in ('5min', '30min'):
            raise ValueError(
                f"coarse_resolution must be '5min' or '30min', got "
                f"{coarse_resolution!r}"
            )
        self._coarse_resolution = coarse_resolution
        self._mode = mode
        self._lag = int(infill_lag_intervals)
        mode_allowance_intervals(mode)          # validates the mode name
        self._fine = self._build_fine()
        self._coarse = self._build_coarse()

    # -- construction ------------------------------------------------------

    def _build_fine(self) -> RunSet:
        """P5MIN-equivalent: a run every interval, 12 targets at leads 0..55."""
        n = len(self._actuals)
        runs = np.arange(n, dtype=np.int64)
        leads = np.arange(_P5MIN_TARGETS, dtype=np.int64)
        target = (runs[:, None] + leads[None, :]).ravel()
        run_rep = np.repeat(runs, _P5MIN_TARGETS)
        keep = target < n
        return RunSet.from_long(
            run_rep[keep], target[keep], self._actuals[target[keep]],
            resolution_intervals=1,
        )

    def _horizon_end_pos(self, run_pos: np.ndarray) -> np.ndarray:
        """Closed-form sawtooth horizon end, as interval positions."""
        ts = self._start + pd.to_timedelta(run_pos * self._step, unit='m')
        days = np.where(ts.hour >= 13, 2, 1)
        end = (ts.normalize() + pd.to_timedelta(days, unit='D')
               + pd.Timedelta(hours=4))
        return ((end - self._start).total_seconds().to_numpy()
                / (self._step * 60)).astype(np.int64)

    def _build_coarse(self) -> RunSet:
        """PREDISPATCH-equivalent: a run every 30 min, targets on the sawtooth."""
        n = len(self._actuals)
        # Runs sit on 30-minute market boundaries. The series need not start on
        # one (interval-end labelling means it typically starts at :05), so
        # find the first boundary position and step from there.
        offs = [i for i in range(_COARSE_INTERVALS)
                if self._ts(i).minute % 30 == 0]
        if not offs:
            raise ValueError(
                "no 30-minute boundary found in the first six intervals; the "
                "price index is not aligned to the market clock"
            )
        runs = np.arange(offs[0], n, _COARSE_INTERVALS, dtype=np.int64)
        ends = self._horizon_end_pos(runs)

        # Target spacing follows the requested resolution. At '5min' the coarse
        # segment carries a target for every interval, so horizon truncation is
        # isolated with no resolution loss mixed in. At '30min' it carries one
        # per half hour, matching what PREDISPATCH actually publishes.
        step = _COARSE_INTERVALS if self._coarse_resolution == '30min' else 1
        run_out, tgt_out = [], []
        for r, e in zip(runs, ends):
            t = np.arange(r, min(e, n - 1) + 1, step, dtype=np.int64)
            if t.size == 0:
                continue
            run_out.append(np.full(t.size, r, dtype=np.int64))
            tgt_out.append(t)
        run_arr = np.concatenate(run_out)
        tgt_arr = np.concatenate(tgt_out)

        if self._coarse_resolution == '30min':
            vals = self._block_mean(tgt_arr)
        else:
            vals = self._actuals[tgt_arr]
        return RunSet.from_long(run_arr, tgt_arr, vals,
                                resolution_intervals=step)

    def _block_mean(self, target_pos: np.ndarray) -> np.ndarray:
        """Mean of the six 5-minute actuals a 30-minute target covers.

        Target p covers positions p-5 .. p, matching the interval-end
        convention. Positions before the series start are excluded from the
        mean rather than treated as zero.
        """
        back = np.arange(_COARSE_INTERVALS - 1, -1, -1, dtype=np.int64)
        src = target_pos[:, None] - back[None, :]
        valid = src >= 0
        src_clipped = np.where(valid, src, 0)
        vals = np.where(valid, self._actuals[src_clipped], np.nan)
        return np.nanmean(vals, axis=1)

    # -- interface ---------------------------------------------------------

    def horizon_available(self) -> int:
        return self._coarse.max_target_span()

    def get(self, decision_instant: pd.Timestamp,
            n_intervals: int) -> ForecastSlice | None:
        k = self._pos(decision_instant) + 1
        if k >= len(self._actuals):
            return None
        if k < 1:
            # The very first interval of the period has no prior interval to
            # act as its decision instant, so there is no history and no run
            # to select. One interval out of ~105,000; the caller falls back
            # to self-consumption, which is the honest answer.
            return None
        n = min(n_intervals, len(self._actuals) - k)
        if n < 1:
            return None

        w = assemble_window(
            k=k, n_intervals=n,
            coarse=self._coarse, fine=self._fine,
            actuals=self._actuals,
            infill_lag_intervals=self._lag,
            mode=self._mode,
        )
        if w is None:
            return None
        rrp = w.rrp

        if uses_actual_for_committed(self._mode):
            # The settled price for interval k is published ~9 s after k
            # begins, well before the P5MIN run that some modes also use, so
            # pricing k at the actual is the realistic case, not a bound.
            rrp = rrp.copy()
            rrp[0] = self._actuals[k]

        return ForecastSlice(run_time=self._ts(w.run_pos),
                             coarse_run_time=self._ts(w.coarse_run_pos),
                             rrp_mwh=rrp, is_infill=w.is_infill,
                             is_fine=w.is_fine)
