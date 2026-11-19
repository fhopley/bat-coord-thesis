"""Tests for analysis/scripts/asx_caps.py ($300 cap settlement).

See docs/cap_hedge_spec.md (ASX $300 cap futures; Verification).
asx_caps.py lives with the analysis scripts, not in a package, so it is
loaded by file path.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    'asx_caps', REPO_ROOT / 'analysis' / 'scripts' / 'asx_caps.py')
asx_caps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(asx_caps)

TZ = 'Etc/GMT-10'


def _quarter_index(year: int, quarter: int) -> pd.DatetimeIndex:
    """Complete interval-end 5-minute index for one calendar quarter."""
    start = pd.Timestamp(year=year, month=3 * (quarter - 1) + 1, day=1, tz=TZ)
    end = start + pd.DateOffset(months=3)
    return pd.date_range(start + pd.Timedelta(minutes=5), end, freq='5min')


# ---------------------------------------------------------------------------
# Quarter assignment and contract size
# ---------------------------------------------------------------------------

def test_contract_quarter_uses_interval_start():
    idx = pd.DatetimeIndex(['2022-01-01 00:00', '2022-01-01 00:05',
                            '2022-04-01 00:00', '2022-04-01 00:05'], tz=TZ)
    got = list(asx_caps.contract_quarter(idx))
    assert got == [(2021, 4), (2022, 1), (2022, 1), (2022, 2)], got


@pytest.mark.parametrize('year,quarter,hours', [
    (2023, 1, 90 * 24), (2024, 1, 91 * 24), (2023, 2, 91 * 24),
    (2023, 3, 92 * 24), (2023, 4, 92 * 24)])
def test_quarter_hours_match_asx_contract_sizes(year, quarter, hours):
    assert asx_caps.quarter_hours(year, quarter) == hours


# ---------------------------------------------------------------------------
# Settlement arithmetic
# ---------------------------------------------------------------------------

def test_interval_at_strike_pays_nothing_and_just_above_pays():
    idx = _quarter_index(2022, 1)
    n = len(idx)
    assert n == 90 * 288
    at = pd.Series(50.0, index=idx)
    at.iloc[1000] = asx_caps.CAP_STRIKE
    res = asx_caps.cap_settlement(at)
    assert res.loc[(2022, 1), 'csp'] == 0.0

    above = at.copy()
    above.iloc[1000] = asx_caps.CAP_STRIKE + 0.01
    res = asx_caps.cap_settlement(above)
    assert res.loc[(2022, 1), 'csp'] == pytest.approx(0.01 / n, rel=1e-9)


def test_payout_is_per_interval_not_on_quarterly_average():
    """One $10,300 interval in an otherwise $50 quarter: the quarterly
    average is far below $300, but the cap pays 10,000 $/MWh for 5 minutes."""
    idx = _quarter_index(2023, 2)
    rrp = pd.Series(50.0, index=idx)
    rrp.iloc[500] = 10_300.0
    assert rrp.mean() < asx_caps.CAP_STRIKE
    res = asx_caps.cap_settlement(rrp).loc[(2023, 2)]
    payout_per_mw = res['csp'] * res['hours']
    assert payout_per_mw == pytest.approx(10_000.0 / 12, rel=1e-9), res.to_dict()


def test_pre_5ms_backfilled_prices_settle_as_30_minute_series():
    """B_Clean_NEM_Prices.py back-fills each 30-minute price over its six
    5-minute slots; CSP·H_q on that series equals the 30-minute settlement."""
    rng = np.random.default_rng(7)
    idx5 = _quarter_index(2021, 1)
    idx30 = idx5[5::6]                     # interval-end on the half hour
    p30 = pd.Series(rng.gamma(2.0, 40.0, len(idx30)), index=idx30)
    spikes = rng.choice(len(idx30), 40, replace=False)
    p30.iloc[spikes] = rng.uniform(301, 14_000, 40)
    assert (p30 > asx_caps.CAP_STRIKE).sum() >= 40

    p5 = p30.reindex(idx5).bfill(limit=5)
    assert p5.notna().all()
    res = asx_caps.cap_settlement(p5).loc[(2021, 1)]
    expected = ((p30 - asx_caps.CAP_STRIKE).clip(lower=0) * 0.5).sum()
    assert res['csp'] * res['hours'] == pytest.approx(expected, rel=1e-9), (
        f"5-min {res['csp'] * res['hours']:.4f} vs 30-min {expected:.4f}")


def test_nan_prices_are_counted_and_excluded_from_e():
    idx = _quarter_index(2022, 3)
    rrp = pd.Series(50.0, index=idx)
    rrp.iloc[10] = 1_300.0
    rrp.iloc[20:23] = np.nan
    res = asx_caps.cap_settlement(rrp).loc[(2022, 3)]
    assert res['n_nan'] == 3
    assert res['csp'] == pytest.approx(1_000.0 / (len(idx) - 3), rel=1e-9)


# ---------------------------------------------------------------------------
# Real data: AEMO Quarterly Energy Dynamics Q1 2024, Figure 19 cap returns
# ---------------------------------------------------------------------------

PRICE_2024 = REPO_ROOT / 'data' / 'clean' / 'nem_prices' / '2024_nem_prices.parquet'
QED_Q1_2024 = {'NSW1': 17.0, 'QLD1': 32.0, 'VIC1': 15.0, 'SA1': 11.0}


@pytest.mark.skipif(not PRICE_2024.exists(), reason='2024 clean price file not present')
def test_q1_2024_matches_aemo_qed_cap_returns():
    prices = pd.read_parquet(PRICE_2024)
    if 'datetime_aest' in prices.columns:
        prices = prices.set_index('datetime_aest')
    got = {}
    for region, published in QED_Q1_2024.items():
        res = asx_caps.cap_settlement(prices[f'{region}_rrp'])
        assert list(res.index) == [(2024, q) for q in (1, 2, 3, 4)], list(res.index)
        got[region] = res.loc[(2024, 1), 'csp']
    for region, published in QED_Q1_2024.items():
        assert abs(got[region] - published) <= 0.5, (
            f"{region}: computed {got[region]:.2f} vs QED {published:.0f}; all {got}")
