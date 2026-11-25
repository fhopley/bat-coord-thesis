"""Forecasts from the canonical AEMO pre-dispatch store.

The first provider carrying real forecast error. Everything before it,
PerfectForecastProvider and TruncatedPerfectForecastProvider, is built from
actual prices and exists to isolate the cost of the rolling machinery itself.
This one measures what a controller acting on AEMO's published signal could
actually have achieved.

Does no cleaning. The store has already applied the INTERVENTION == 0 filter,
the explicit dtype cast, the uniqueness assertion and the month-boundary
correction, so this module loads, converts to interval positions, and runs the
shared assembly. See predispatch_data_spec.md.

Causality
---------
Run labels are used as published, with no adjustment. Both source tables
publish BEFORE their nominal label (P5MIN by 4.35 minutes on average and 3.12
in the worst case observed, PREDISPATCH by 28.39 and 27.37), so treating the
label as the moment the forecast became available is conservative. The hazard
runs the other way: subtracting the observed offset to "correct" the label to a
publication time would hand the battery information minutes before it existed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from model.core.forecast import ForecastSlice
from model.forecast.assembly import (
    assemble_window, mode_allowance_intervals, uses_actual_for_committed,
)
from model.forecast.perfect import _GridProvider
from model.forecast.runset import RunSet
from pipeline.pre_dispatch.B_Clean_Predispatch import read_store

DegradedPolicy = Literal['fail', 'self_consumption', 'actuals', 'infill']

# Runs before the modelled period still matter: the earliest decision instants
# need a run at or before them, and a coarse run can be up to 30 minutes old.
_LOOKBACK = pd.Timedelta(days=1)


class PredispatchForecastProvider(_GridProvider):
    """AEMO pre-dispatch, spliced and infilled, at any decision instant.

    Args:
        store_root: Canonical store root, e.g. data/clean/predispatch.
        region: NEM region id, e.g. 'SA1'.
        prices_mwh: Actual RRP over the modelled period. Defines the interval
            grid, and is read by the infill rule for intervals strictly before
            the decision instant.
        mode: Current-interval price mode. See forecast_providers_spec.md.
        infill_lag_intervals: Lag used to extrapolate beyond forecast coverage.
            288 is 24 hours. At the current 18 hour horizon default this fires
            for about 1% of window entries.
        on_degraded_period: What to do about periods the store marks as having
            no usable lead dimension, October 2022 being the known case.
                'fail'             raise at construction, naming the period
                'self_consumption' return None throughout it
                'actuals'          serve actual prices, marked in diagnostics
                'infill'           no special handling: falls through to the
                                    ordinary assemble_window(...) call below,
                                    same as any interval whose horizon exceeds
                                    real coverage. For October 2022 this means
                                    the fine (P5MIN) segment, which is
                                    unaffected and complete, still supplies a
                                    genuine forecast for its own ~55 minute
                                    reach; only the tail beyond that -- which
                                    the coarse segment's surviving 00:00 to
                                    04:00 runs on the 1st cannot reach into
                                    mid-month -- falls
                                    to the standard 24-hour-lag infill. This is
                                    the recommended choice for a gap of this
                                    shape: 'actuals' would substitute genuine
                                    foresight and overstate achievable value,
                                    'self_consumption' discards the real
                                    near-term forecast the fine segment still
                                    has. RollingDiagnostics.infill_fraction
                                    reports the effect automatically, as an
                                    elevated fraction for that period's
                                    windows, with no separate accounting.
            Default 'fail' so the choice is conscious: a result that quietly
            used actuals for a month would be indistinguishable from one that
            did not.
        max_staleness_hours: Largest age any decision instant may see in the
            freshest available run. This is a COVERAGE check, not the staleness
            gate that was deliberately rejected: the provider still returns the
            best run it has, however old, but refuses to start when the store
            does not span the modelled period.

            It exists because the failure it catches is silent. A store ending
            before the period does not read as empty: the last partition is
            found, its final runs are carried forward indefinitely, and infill
            supplies the rest from 24-hour-lagged actuals. The result looks
            like a forecast, scores plausibly, and is really yesterday's
            prices. Observed on a real cohort run where two sites sat entirely
            after the store ended and returned 98.7% infill while reporting no
            fallbacks and no error.

            Six hours is generous against real data: a full sweep of 2021-01
            to 2026-06 found four isolated missing PREDISPATCH runs in roughly
            96,000, none longer than one cadence.
    """

    def __init__(
        self,
        store_root: Path,
        region: str,
        prices_mwh: pd.Series,
        interval_minutes: int = 5,
        mode: str = 'forecast',
        infill_lag_intervals: int = 288,
        on_degraded_period: DegradedPolicy = 'fail',
        manifest_path: Optional[Path] = None,
        max_staleness_hours: float = 6.0,
    ):
        super().__init__(prices_mwh, interval_minutes)
        mode_allowance_intervals(mode)              # validates the mode name
        if on_degraded_period not in ('fail', 'self_consumption', 'actuals',
                                     'infill'):
            raise ValueError(
                f"on_degraded_period must be 'fail', 'self_consumption', "
                f"'actuals' or 'infill', got {on_degraded_period!r}")
        self._store_root = Path(store_root)
        self._region = region
        self._mode = mode
        self._lag = int(infill_lag_intervals)
        self._on_degraded = on_degraded_period

        start = self._index[0] - _LOOKBACK
        end = self._index[-1]
        self._fine = self._load('p5min', 1, start, end)
        self._coarse = self._load('predispatch', 6, start, end)
        self._check_coverage('predispatch', self._coarse, max_staleness_hours)
        self._check_coverage('p5min', self._fine, max_staleness_hours)
        self._degraded = self._degraded_ranges(manifest_path)

    # -- construction ------------------------------------------------------

    def _load(self, table: str, resolution_intervals: int,
              start: pd.Timestamp, end: pd.Timestamp) -> RunSet:
        df = read_store(self._store_root, table, self._region, start, end)
        if df.empty:
            raise ValueError(
                f"no {table} data for {self._region} between {start} and "
                f"{end} under {self._store_root}. Run "
                "pipeline.pre_dispatch.A_Download_Predispatch and "
                "B_Clean_Predispatch for this period."
            )
        # Interval positions on the model grid. Runs before the period start
        # get negative positions, which RunSet.latest_at handles and which
        # _paint filters out of the window; that is how an early decision
        # instant still finds a run.
        delta = (df['run_time'] - self._start).dt.total_seconds().to_numpy()
        step_s = self._step * 60
        if (delta % step_s != 0).any():
            raise ValueError(
                f"{table}: {int((delta % step_s != 0).sum())} run time(s) do "
                f"not fall on the {self._step}-minute model grid starting "
                f"{self._start}"
            )
        run_pos = (delta // step_s).astype(np.int64)
        lead = df['lead_minutes'].to_numpy().astype(np.int64)
        if (lead % self._step != 0).any():
            raise ValueError(
                f"{table}: lead_minutes not a whole multiple of "
                f"{self._step} minutes, so target positions are not integral")
        return RunSet.from_long(run_pos, run_pos + lead // self._step,
                                df['rrp_mwh'].to_numpy(), resolution_intervals)

    def _check_coverage(self, table: str, runs: RunSet,
                        max_staleness_hours: float) -> None:
        """Refuse to start when the store does not span the modelled period.

        Measures the worst staleness any decision instant would experience:
        the largest gap between consecutive runs, plus the gap from the period
        start to the first run and from the last run to the period end. Those
        endpoint gaps are the ones that matter here, because a store that
        stops early produces the largest and least visible staleness of all.
        """
        limit = max_staleness_hours * 60 / self._step        # in intervals
        first_pos, last_pos = 0, len(self._actuals) - 1
        rp = runs.run_pos

        lead_gap = int(rp[0] - first_pos) if rp[0] > first_pos else 0
        trail_gap = int(last_pos - rp[-1]) if rp[-1] < last_pos else 0
        internal = int(np.diff(rp).max()) if rp.size > 1 else 0

        worst, where = max(
            (lead_gap, f"between the period start {self._ts(first_pos)} and the "
                       f"first run {self._ts(int(rp[0]))}"),
            (trail_gap, f"between the last run {self._ts(int(rp[-1]))} and the "
                        f"period end {self._ts(last_pos)}"),
            (internal, "between two consecutive runs"),
            key=lambda t: t[0])
        if worst <= limit:
            return
        hours = worst * self._step / 60
        raise ValueError(
            f"{table}: the store does not cover the modelled period. Worst "
            f"staleness {hours:,.1f} h {where}, against a limit of "
            f"{max_staleness_hours:g} h.\n"
            "This does NOT read as an empty store: the nearest runs are "
            "carried forward and infill supplies the rest from 24-hour-lagged "
            "actuals, so the run would complete and produce a plausible "
            "number that is really yesterday's prices.\n"
            "Acquire the missing months (pipeline.pre_dispatch.A_Download_"
            "Predispatch then B_Clean_Predispatch), or raise "
            "max_staleness_hours deliberately if a gap this size is intended."
        )

    def _degraded_ranges(self, manifest_path: Optional[Path]):
        """Interval-position ranges the store flags as unfit to forecast from.

        Read from the manifest rather than inferred from the data, for two
        DIFFERENT reasons covering two DIFFERENT shapes, both checked here:

        - `degraded`: a lead-0-only month LOOKS complete from the data alone
          (full run coverage, no gaps, every price present), which is the
          whole trap. Only the manifest distinguishes it.
        - `known_archive_gap`: October 2022's surviving runs (00:00 to 04:00
          on the 1st, held in September's source file) have perfectly
          ordinary leads (24 to 28 hours) -- nothing about THEIR OWN
          data marks it as unusable, and stage C's `degraded` flag is
          deliberately NOT widened to catch it by run-count alone, because
          that also caught every merely-partial or in-progress partition
          (see C_Validate_Predispatch.py's TableReport.degraded docstring).
          What actually marks this period is the curated, manually-confirmed
          KNOWN_ARCHIVE_GAPS table, which is exactly what `known_archive_gap`
          reports.

        Either flag being set is sufficient; a period can be degraded without
        being a known archive gap (some future lead-0 `_D`-only month) or a
        known archive gap without being degraded in the lead-0 sense (this
        one).
        """
        path = Path(manifest_path) if manifest_path else (
            self._store_root / 'manifest.json')
        if not path.exists():
            if self._on_degraded == 'fail':
                return []                       # nothing to check against
            return []
        entries = json.loads(path.read_text()).get('entries', [])
        ranges, named = [], []
        for e in entries:
            flagged = e.get('degraded') or e.get('known_archive_gap')
            if not flagged or e.get('region') != self._region:
                continue
            period = pd.Period(e['period'], freq='M')
            lo = period.to_timestamp().tz_localize(self._index.tz)
            hi = (period + 1).to_timestamp().tz_localize(self._index.tz)
            if hi <= self._index[0] or lo > self._index[-1]:
                continue                        # outside the modelled period
            named.append(f"{e['table']} {e['region']} {e['period']} "
                         f"(max lead {e['max_lead_minutes']} min)")
            ranges.append((self._pos(max(lo, self._index[0])),
                           self._pos(min(hi, self._index[-1]))))
        if named and self._on_degraded == 'fail':
            raise ValueError(
                "the store marks the following period(s) as having no usable "
                f"forecast lead: {'; '.join(named)}. Rolling optimisation "
                "against a one-step forecast would be indistinguishable from a "
                "real result in every downstream artefact. Set "
                "on_degraded_period to 'infill' (recommended -- falls through "
                "to the ordinary infill path, using any real fine-segment "
                "coverage that still exists), 'self_consumption', or "
                "'actuals' to proceed deliberately."
            )
        return ranges

    def _is_degraded(self, pos: int) -> bool:
        return any(lo <= pos <= hi for lo, hi in self._degraded)

    # -- interface ---------------------------------------------------------

    def horizon_available(self) -> int:
        """Structural horizon of the coarse segment, in intervals.

        Measured from the RUN LABEL, so it peaks near 39 hours. The horizon
        usable from a DECISION INSTANT is shorter, because the coarse run in
        hand can be up to 30 minutes old; that floor is about 15.08 hours.
        """
        return self._coarse.max_target_span()

    def get(self, decision_instant: pd.Timestamp,
            n_intervals: int) -> ForecastSlice | None:
        k = self._pos(decision_instant) + 1
        if k < 1 or k >= len(self._actuals):
            return None
        n = min(n_intervals, len(self._actuals) - k)
        if n < 1:
            return None

        if self._degraded and self._is_degraded(k):
            if self._on_degraded == 'self_consumption':
                return None
            if self._on_degraded == 'actuals':
                vals = self._actuals[k:k + n]
                # Every value is a substituted actual, so nothing came from
                # either forecast segment. is_fine is all False rather than
                # all True: this window has no fine segment, it has no
                # forecast at all.
                return ForecastSlice(run_time=decision_instant,
                                     coarse_run_time=decision_instant,
                                     rrp_mwh=vals.copy(),
                                     is_infill=np.ones(n, dtype=bool),
                                     is_fine=np.zeros(n, dtype=bool))

        w = assemble_window(
            k=k, n_intervals=n, coarse=self._coarse, fine=self._fine,
            actuals=self._actuals, infill_lag_intervals=self._lag,
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
