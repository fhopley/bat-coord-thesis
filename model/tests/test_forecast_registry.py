"""Tests for the forecast provider registry."""
import numpy as np
import pandas as pd
import pytest

from model.forecast.perfect import (
    PerfectForecastProvider, TruncatedPerfectForecastProvider,
)
from model.forecast.registry import (
    FORECAST_SOURCES, build_provider, is_reportable, region_for_state,
)

TZ = 'Etc/GMT-10'


def _prices(days=3):
    idx = pd.date_range('2026-02-09 00:05', periods=288 * days, freq='5min', tz=TZ)
    rng = np.random.default_rng(0)
    return pd.Series(60 + rng.normal(0, 20, len(idx)), index=idx)


@pytest.mark.parametrize("name,cls", [
    ('perfect', PerfectForecastProvider),
    ('truncated_perfect', TruncatedPerfectForecastProvider),
    ('truncated_perfect_5min', TruncatedPerfectForecastProvider),
])
def test_builds_the_data_free_providers(name, cls):
    assert isinstance(build_provider(name, _prices()), cls)


def test_coarse_resolution_follows_the_name():
    p = _prices()
    a = build_provider('truncated_perfect', p)
    b = build_provider('truncated_perfect_5min', p)
    assert a._coarse_resolution == '30min'
    assert b._coarse_resolution == '5min'


def test_unknown_source_rejected():
    with pytest.raises(ValueError, match="unknown forecast source"):
        build_provider('amber_spot', _prices())


def test_predispatch_requires_store_and_region():
    with pytest.raises(ValueError, match="requires store_root and region"):
        build_provider('predispatch', _prices())


def test_perfect_is_flagged_not_reportable():
    """It reads actuals at and beyond the decision instant. Naming it here
    rather than relying on reviewer memory is the point."""
    assert not is_reportable('perfect')
    assert all(is_reportable(n) for n in FORECAST_SOURCES if n != 'perfect')


@pytest.mark.parametrize("state,region", [
    ('SA', 'SA1'), ('nsw', 'NSW1'), ('VIC', 'VIC1'), ('QLD', 'QLD1'),
])
def test_region_for_state(state, region):
    assert region_for_state(state) == region


def test_unknown_state_rejected():
    with pytest.raises(ValueError, match="unknown state"):
        region_for_state('WA')
