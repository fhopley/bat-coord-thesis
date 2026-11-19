"""Tests for NEM market price limits by financial year."""
import pandas as pd
import pytest

from model.core.market_prices import (
    MARKET_PRICE_CAP_BY_FY, MARKET_PRICE_FLOOR_MWH,
    financial_year_start, market_price_cap, market_price_floor,
    widest_market_price_cap,
)

TZ = 'Etc/GMT-10'


def _ts(s):
    return pd.Timestamp(s, tz=TZ)


@pytest.mark.parametrize("stamp,fy", [
    ('2025-06-30 23:55', 2024),
    ('2025-07-01 00:00', 2025),
    ('2025-09-01', 2025),
    ('2026-03-01', 2025),
    ('2026-06-30', 2025),
    ('2026-07-01', 2026),
])
def test_financial_year_boundaries(stamp, fy):
    assert financial_year_start(_ts(stamp)) == fy


@pytest.mark.parametrize("stamp,cap", [
    ('2021-03-01', 15_000.0),   # FY2020: before the 1 Jul 2021 step
    ('2022-06-01', 15_100.0),   # FY2021
    ('2022-08-01', 15_500.0),
    ('2024-01-01', 16_600.0),
    ('2025-01-01', 17_500.0),
    ('2026-02-10', 20_300.0),   # the Energy Masters window
    ('2026-08-01', 23_200.0),
])
def test_cap_by_date(stamp, cap):
    assert market_price_cap(_ts(stamp)) == cap


def test_cap_for_2021_calendar_start_of_edp_window():
    """The EDP window opens 2021-01, which is financial year 2020."""
    assert market_price_cap(_ts('2021-01-01')) == 15_000.0


def test_unknown_year_raises_rather_than_falling_back():
    """A silent fallback would validate forecast data against the wrong cap."""
    with pytest.raises(KeyError, match="No Market Price Cap recorded"):
        market_price_cap(_ts('2035-01-01'))


def test_floor_is_constant_across_the_modelled_period():
    for stamp in ('2021-01-01', '2023-07-01', '2026-06-30'):
        assert market_price_floor(_ts(stamp)) == -1_000.0
    assert MARKET_PRICE_FLOOR_MWH == -1_000.0


def test_widest_cap_is_the_maximum_on_record():
    assert widest_market_price_cap() == max(MARKET_PRICE_CAP_BY_FY.values())


def test_cap_series_is_monotonic_except_where_determinations_differ():
    """The cap is CPI-adjusted annually and has risen every year on record."""
    years = sorted(MARKET_PRICE_CAP_BY_FY)
    caps = [MARKET_PRICE_CAP_BY_FY[y] for y in years]
    assert years == list(range(years[0], years[-1] + 1)), "no gaps in the table"
    assert all(b > a for a, b in zip(caps, caps[1:]))


def test_prices_module_uses_the_widest_cap():
    from model.core import prices
    assert prices.MARKET_PRICE_CAP_MWH == widest_market_price_cap()
    assert prices.MARKET_PRICE_FLOOR_MWH == MARKET_PRICE_FLOOR_MWH
