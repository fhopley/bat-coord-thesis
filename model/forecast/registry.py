"""Build a forecast provider from a configuration string.

Exists so that `--forecast predispatch` works the same way in the check script,
the horizon sweep and eventually the orchestrator, without each of them
carrying its own if-chain over provider constructors. It was deliberately not
written while there were only two providers, since indirection with nothing to
indirect is just a layer.

The names map onto the measurement ladder in rolling_optimisation_spec.md:

    perfect                 rung 1 equivalent. Actual prices, unlimited
                            horizon. The correctness gate ONLY; never a
                            reported result, because it reads actuals at and
                            beyond the decision instant.
    truncated_perfect       rung 2b. Actual prices restricted to pre-dispatch's
                            availability structure, 30-minute coarse segment.
                            Zero forecast error, realistic everything else.
    truncated_perfect_5min  rung 2a. As above with a 5-minute coarse segment,
                            isolating horizon and terminal pin effects from
                            resolution loss.
    predispatch             rung 3. Real AEMO forecasts from the canonical
                            store. The first provider carrying forecast error.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from model.core.forecast import ForecastProvider
from model.forecast.perfect import (
    PerfectForecastProvider, TruncatedPerfectForecastProvider,
)
from model.forecast.predispatch import PredispatchForecastProvider

FORECAST_SOURCES = ('perfect', 'truncated_perfect', 'truncated_perfect_5min',
                    'predispatch')

# Providers that read actuals at or beyond the decision instant, and so cannot
# produce a defensible result. Named here rather than left to reviewer memory.
_NON_CAUSAL = ('perfect',)


def is_reportable(name: str, horizon_hours: float | None = None,
                  period_hours: float | None = None) -> bool:
    """False only where a provider genuinely sees beyond its own window.

    'perfect' returns actual prices, but that is only a problem when the
    HORIZON covers the whole period, which is the correctness-gate
    configuration. At a finite horizon it means perfect price foresight over
    the window and nothing more, which is a legitimate and useful reference:
    it isolates the cost of the rolling window itself from every forecast
    effect.

    An earlier version flagged 'perfect' unconditionally, which would have
    warned spuriously on exactly that reference run.

    With horizons unknown, errs toward flagging.
    """
    if name not in _NON_CAUSAL:
        return True
    if horizon_hours is None or period_hours is None:
        return False
    return horizon_hours < period_hours


def build_provider(
    name: str,
    prices_mwh: pd.Series,
    *,
    interval_minutes: int = 5,
    mode: str = 'forecast',
    infill_lag_intervals: int = 288,
    store_root: Optional[Path] = None,
    region: Optional[str] = None,
    on_degraded_period: str = 'fail',
    max_staleness_hours: float = 6.0,
) -> ForecastProvider:
    """Construct a provider by name.

    Args:
        name: One of FORECAST_SOURCES.
        prices_mwh: Actual RRP over the modelled period. Defines the interval
            grid for every provider, and supplies the forecast itself for the
            perfect and truncated variants.
        store_root, region: Required for 'predispatch' only.
        max_staleness_hours: 'predispatch' only. See
            PredispatchForecastProvider's own docstring -- this is a coverage
            check (does the store span the modelled period at all), not the
            staleness gate that provider deliberately does not have. Ignored
            for every other source.

    Raises:
        ValueError: on an unknown name, or a missing store_root/region.
    """
    if name not in FORECAST_SOURCES:
        raise ValueError(
            f"unknown forecast source {name!r}, expected one of "
            f"{list(FORECAST_SOURCES)}")

    if name == 'perfect':
        return PerfectForecastProvider(prices_mwh, interval_minutes)

    if name.startswith('truncated_perfect'):
        coarse = '5min' if name.endswith('_5min') else '30min'
        return TruncatedPerfectForecastProvider(
            prices_mwh, interval_minutes=interval_minutes,
            coarse_resolution=coarse, mode=mode,
            infill_lag_intervals=infill_lag_intervals)

    if store_root is None or region is None:
        raise ValueError(
            "forecast source 'predispatch' requires store_root and region, "
            "e.g. store_root='data/clean/predispatch', region='SA1'")
    return PredispatchForecastProvider(
        Path(store_root), region, prices_mwh,
        interval_minutes=interval_minutes, mode=mode,
        infill_lag_intervals=infill_lag_intervals,
        on_degraded_period=on_degraded_period,
        max_staleness_hours=max_staleness_hours)


def region_for_state(state: str) -> str:
    """NEM region id for a model state code. 'SA' -> 'SA1'."""
    state = state.strip().upper()
    if state not in ('NSW', 'QLD', 'SA', 'TAS', 'VIC'):
        raise ValueError(f"unknown state {state!r}")
    return f"{state}1"
