"""The price forecast contract consumed by rolling_optimisation.

A provider answers exactly one question: what did the price forecast look like
at this instant, over this many intervals? Sources differ in horizon,
resolution, update cadence, units and publication lag, and all of that is
absorbed inside the provider. The component receives a finished vector of
wholesale prices, or nothing.

See forecast_providers_spec.md for the design and forecast/assembly.py for the
shared machinery every provider runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastSlice:
    """One window of forecast prices, aligned to the rolling window.

    Position j corresponds to the interval whose interval-END label is
    decision_instant + (j+1) * interval. Position 0 is the interval the caller
    is about to commit.

    Arrays rather than Series: get() is called once per 5-minute interval,
    roughly 105,000 times per site-year, so nothing in the assembly path
    constructs a pandas object.

    Attributes:
        run_time: Nominal label of the underlying real forecast run, for the
            causality assertion in the caller. For providers that synthesise
            runs from actuals this is still the instant the information would
            have been available.

            Where a provider splices two segments this is the FINE run, which
            is the freshest thing the window contains. It is therefore not a
            measure of how old the window is: see coarse_run_time.
        coarse_run_time: Nominal label of the underlying COARSE run, which
            supplies everything beyond the fine segment.

            Reported separately because the two differ systematically and the
            difference matters. A fine run exists at every 5-minute decision
            instant, so run_time is zero-stale by construction whenever fine
            coverage is complete, while the coarse run in hand can be up to
            25 minutes old and is what shapes the forward window the plan is
            actually built on. Defaults to run_time for providers that do not
            splice.
        rrp_mwh: Wholesale regional reference price in $/MWh, one per window
            interval, in target order. Every provider returns wholesale
            prices; the customer price stack is applied downstream by
            PriceStackOffsets.
        is_infill: True where the value was extrapolated rather than forecast.
            Without this, a plot of a 48-hour forecast surface gives no
            indication that a third of it is yesterday repeated, and no
            diagnostic can separate a forecast failure from an extrapolation
            failure.
        is_fine: True where the value came from the fine (5-minute) segment
            rather than the coarse (30-minute) one.

            Retained rather than recomputed downstream because the boundary is
            not simply "the first twelve positions": it moves with the mode
            allowance and shortens when the fine run is stale. Anything
            plotting the splice would otherwise have to duplicate an assembly
            assumption, which is the shape of most of the silent errors this
            subsystem has produced. Defaults to all True for providers that do
            not splice.

    len(rrp_mwh) may be SHORTER than the requested n_intervals when the
    provider cannot fill the whole window, which happens near the start of
    available history. The caller shortens its window to match.
    """
    run_time: pd.Timestamp
    rrp_mwh: np.ndarray
    is_infill: np.ndarray
    coarse_run_time: pd.Timestamp | None = None
    is_fine: np.ndarray | None = None

    def __post_init__(self):
        if self.rrp_mwh.shape != self.is_infill.shape:
            raise ValueError(
                f"rrp_mwh {self.rrp_mwh.shape} and is_infill "
                f"{self.is_infill.shape} must have the same shape"
            )
        # Defaults are filled here rather than required at every construction
        # site, so a provider that does not splice says nothing rather than
        # saying something wrong. Frozen dataclass, hence object.__setattr__.
        if self.coarse_run_time is None:
            object.__setattr__(self, 'coarse_run_time', self.run_time)
        if self.is_fine is None:
            object.__setattr__(
                self, 'is_fine', np.ones(self.rrp_mwh.shape, dtype=bool))
        elif self.is_fine.shape != self.rrp_mwh.shape:
            raise ValueError(
                f"rrp_mwh {self.rrp_mwh.shape} and is_fine "
                f"{self.is_fine.shape} must have the same shape"
            )
        if self.rrp_mwh.size == 0:
            raise ValueError("ForecastSlice must cover at least one interval")
        if not np.isfinite(self.rrp_mwh).all():
            n_bad = int((~np.isfinite(self.rrp_mwh)).sum())
            raise ValueError(
                f"rrp_mwh contains {n_bad} non-finite value(s); a provider "
                "must return a fully populated window or None"
            )

    def __len__(self) -> int:
        return int(self.rrp_mwh.size)

    @property
    def infill_fraction(self) -> float:
        """Share of the window that was extrapolated rather than forecast.

        Reported per window by rolling_optimisation. Worth watching in its own
        right: because the pre-dispatch horizon sawtooths between roughly 15
        and 39 hours, this has a strong daily rhythm that would otherwise
        contaminate any time-of-day analysis.
        """
        return float(self.is_infill.mean())

    @property
    def fine_fraction(self) -> float:
        """Share of the window supplied by the 5-minute segment.

        Small by construction: the fine segment reaches roughly one hour, so
        at an 18-hour horizon this is about 5 per cent. Worth having as a
        number because it is the reader's check that the splice landed where
        it should.
        """
        return float(self.is_fine.mean())


@dataclass(frozen=True)
class LoadSlice:
    """One window of forecast household load, aligned to the rolling window.

    The load-side counterpart to ForecastSlice. Position j corresponds to the
    interval whose interval-END label is decision_instant + (j+1) * interval.

    Position 0 is supplied by the CALLER from actual load, not by the provider,
    so its value here is the provider's honest forecast and is overwritten
    before the LP sees it. See rolling_optimisation and usage_forecast_spec.md,
    "What the component commits".

    Attributes:
        load_kwh: Forecast gross household load per window interval, kWh.
        adjustment_kwh: The additive recent-observation term before decay,
            kWh per interval. Positive means observed load has been running
            above the profile.
        n_lookback: Intervals of observed history behind the adjustment.
        n_clipped: Window positions clipped at zero. Clipping biases the
            forecast upward wherever it fires, so it is counted rather than
            absorbed.
        warm_up: True when the profile behind this slice rests on less than
            the configured history window.

    len(load_kwh) may be SHORTER than the requested n_intervals near the end
    of the period; the caller shortens its window to match.
    """
    load_kwh: np.ndarray
    adjustment_kwh: float = 0.0
    n_lookback: int = 0
    n_clipped: int = 0
    warm_up: bool = False

    def __post_init__(self):
        if self.load_kwh.size == 0:
            raise ValueError("LoadSlice must cover at least one interval")
        if not np.isfinite(self.load_kwh).all():
            n_bad = int((~np.isfinite(self.load_kwh)).sum())
            raise ValueError(
                f"load_kwh contains {n_bad} non-finite value(s); a provider "
                "must return a fully populated window or None"
            )
        if (self.load_kwh < 0).any():
            raise ValueError("load_kwh must be non-negative")

    def __len__(self) -> int:
        return int(self.load_kwh.size)


class ForecastProvider(Protocol):
    """Source of price forecasts at a point in time.

    Implementations own causality. rolling_optimisation checks the returned
    run_time but never computes what was available; any publication-lag
    knowledge specific to a source lives in the provider.
    """

    def horizon_available(self) -> int:
        """Maximum number of intervals get() can normally fill."""
        ...

    def get(self, decision_instant: pd.Timestamp,
            n_intervals: int) -> ForecastSlice | None:
        """Forecast available at decision_instant, over n_intervals.

        Returns None when no run exists at or before decision_instant at all,
        or when a degraded period is configured to fall back. The caller then
        commits the self-consumption action for that interval.
        """
        ...


@dataclass(frozen=True)
class SolarSlice:
    """One window of forecast solar generation, aligned to the rolling window.

    Position 0 is supplied by the CALLER from actual generation, as with load;
    the provider's value there is its honest forecast and is overwritten before
    the LP sees it.

    Attributes:
        solar_kwh: Forecast generation per window interval, kWh.
        index_hist: The history-only index estimate, R_hist.
        index_today: The blended index used for same-day targets.
        intraday_weight: The weight given to today's observation, a.
        adjustment: The additive index correction before decay.
        n_clipped: Window positions clipped at an index bound.
        warm_up: Envelope built on less than the configured history.
    """
    solar_kwh: np.ndarray
    index_hist: float = 0.0
    index_today: float = 0.0
    intraday_weight: float = 0.0
    adjustment: float = 0.0
    n_clipped: int = 0
    warm_up: bool = False

    def __post_init__(self):
        if self.solar_kwh.size == 0:
            raise ValueError("SolarSlice must cover at least one interval")
        if not np.isfinite(self.solar_kwh).all():
            raise ValueError("solar_kwh contains non-finite values")
        if (self.solar_kwh < 0).any():
            raise ValueError("solar_kwh must be non-negative")

    def __len__(self) -> int:
        return int(self.solar_kwh.size)


class SolarForecastProvider(Protocol):
    """Source of solar generation forecasts at a point in time.

    Indexed by interval position, as LoadForecastProvider is, and owning the
    same causality rule: generation metered for the interval ending at the
    decision instant is observed; anything later is not.
    """

    def get(self, k: int, n_intervals: int) -> SolarSlice | None:
        """Forecast for the window beginning at interval k."""
        ...


class LoadForecastProvider(Protocol):
    """Source of household load forecasts at a point in time.

    Indexed by interval POSITION rather than by timestamp, unlike
    ForecastProvider. A load forecast is built from the site's own trace on the
    component's own index, so there is no external run store to align to and a
    timestamp lookup would be a searchsorted per call for nothing.

    Implementations own causality. The rule is one interval more permissive
    than the price side: the settled price for the interval ending at D is not
    published at D, but metered load for that interval IS observed at D by a
    local controller. Labels at or before the decision instant are readable;
    labels after it are not.
    """

    def get(self, k: int, n_intervals: int) -> LoadSlice | None:
        """Forecast for the window beginning at interval k, over n_intervals.

        Returns None when no forecast can be produced, which for a
        history-based provider means the start of the period. The caller then
        commits the self-consumption action for that interval.
        """
        ...
