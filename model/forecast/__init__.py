"""Price forecast providers for rolling_optimisation.

The protocol and ForecastSlice live in model.core.forecast; this package holds
the shared assembly machinery and the concrete providers.
"""
from model.forecast.runset import RunSet
from model.forecast.assembly import assemble_window, expand_coarse
from model.forecast.perfect import (
    PerfectForecastProvider, TruncatedPerfectForecastProvider,
)
from model.forecast.predispatch import PredispatchForecastProvider
from model.forecast.registry import (
    FORECAST_SOURCES, build_provider, is_reportable, region_for_state,
)

__all__ = [
    'RunSet',
    'assemble_window',
    'expand_coarse',
    'PerfectForecastProvider',
    'TruncatedPerfectForecastProvider',
    'PredispatchForecastProvider',
    'FORECAST_SOURCES',
    'build_provider',
    'is_reportable',
    'region_for_state',
]
