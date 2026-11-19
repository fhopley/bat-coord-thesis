"""Tests for the providers built from actual prices.

The sawtooth assertions are cross-checked against the pre-dispatch
investigation: horizon 15.5 h minimum and 39.0 h maximum measured from the run
label, stepping at 13:00 market time, validated there against all 1,345
February 2026 runs.
"""
import numpy as np
import pandas as pd
import pytest

from model.core.forecast import ForecastSlice
from model.forecast import (
    PerfectForecastProvider, TruncatedPerfectForecastProvider,
)

TZ = 'Etc/GMT-10'


def _prices(days=6, start='2026-02-09 00:05', seed=0):
    idx = pd.date_range(start, periods=288 * days, freq='5min', tz=TZ)
    rng = np.random.default_rng(seed)
    tod = (np.arange(len(idx)) % 288) / 288
    base = 60 + 200 * np.exp(-((tod - 0.79) ** 2) / 0.0012) \
        - 70 * np.exp(-((tod - 0.5) ** 2) / 0.006)
    return pd.Series(base + rng.normal(0, 30, len(idx)), index=idx)


# ------------------------------------------------------------ ForecastSlice

def test_slice_rejects_non_finite_values():
    with pytest.raises(ValueError, match="non-finite"):
        ForecastSlice(pd.Timestamp('2026-02-09', tz=TZ),
                      np.array([1.0, np.nan]), np.zeros(2, bool))


def test_slice_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        ForecastSlice(pd.Timestamp('2026-02-09', tz=TZ),
                      np.array([1.0, 2.0]), np.zeros(3, bool))


def test_slice_infill_fraction():
    s = ForecastSlice(pd.Timestamp('2026-02-09', tz=TZ),
                      np.ones(4), np.array([False, False, True, True]))
    assert s.infill_fraction == 0.5
    assert len(s) == 4


def test_slice_defaults_provenance_for_providers_that_do_not_splice():
    """A provider with no splice says nothing rather than saying something
    wrong: one notional run, every value from it."""
    t = pd.Timestamp('2026-02-09', tz=TZ)
    s = ForecastSlice(t, np.ones(4), np.zeros(4, bool))
    assert s.coarse_run_time == t
    assert s.is_fine.all()
    assert s.fine_fraction == 1.0


def test_slice_rejects_mismatched_is_fine():
    with pytest.raises(ValueError, match="is_fine"):
        ForecastSlice(pd.Timestamp('2026-02-09', tz=TZ),
                      np.array([1.0, 2.0]), np.zeros(2, bool),
                      is_fine=np.zeros(3, bool))


def test_slice_fine_fraction():
    s = ForecastSlice(pd.Timestamp('2026-02-09', tz=TZ),
                      np.ones(4), np.zeros(4, bool),
                      is_fine=np.array([True, True, False, False]))
    assert s.fine_fraction == 0.5


# --------------------------------------------------------- PerfectForecast

def test_perfect_returns_actuals_exactly_with_no_infill():
    p = _prices()
    pf = PerfectForecastProvider(p)
    s = pf.get(p.index[100], 48)
    np.testing.assert_array_equal(s.rrp_mwh, p.to_numpy()[101:149])
    assert not s.is_infill.any()
    assert s.infill_fraction == 0.0


def test_perfect_truncates_at_the_end_of_the_period():
    p = _prices(days=1)
    pf = PerfectForecastProvider(p)
    s = pf.get(p.index[-10], 100)
    assert len(s) == 9


def test_perfect_returns_none_past_the_end():
    p = _prices(days=1)
    assert PerfectForecastProvider(p).get(p.index[-1], 10) is None


def test_perfect_horizon_is_the_whole_period():
    p = _prices(days=1)
    assert PerfectForecastProvider(p).horizon_available() == len(p)


def test_perfect_reports_trivial_provenance():
    p = _prices()
    s = PerfectForecastProvider(p).get(p.index[100], 48)
    assert s.coarse_run_time == s.run_time == p.index[100]
    assert s.is_fine.all()


def test_provider_rejects_nan_prices():
    p = _prices(days=1)
    p.iloc[5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        PerfectForecastProvider(p)


def test_provider_rejects_off_grid_decision_instant():
    p = _prices(days=1)
    pf = PerfectForecastProvider(p)
    with pytest.raises(ValueError, match="grid"):
        pf.get(p.index[10] + pd.Timedelta(minutes=2), 10)


# ------------------------------------------------ TruncatedPerfectForecast

def test_truncated_reproduces_the_measured_sawtooth_maximum():
    """39.0 h from the run label, matching the investigation."""
    tp = TruncatedPerfectForecastProvider(_prices(days=6))
    assert tp.horizon_available() == 39 * 12


def test_truncated_sawtooth_never_falls_below_the_measured_minimum():
    """15.5 h from the run label. Measured from the DECISION INSTANT the
    guaranteed minimum is 15.08 h, because the coarse run is up to 25 minutes
    stale; that is the figure the 48 h horizon default is justified against."""
    p = _prices(days=6)
    tp = TruncatedPerfectForecastProvider(p)
    coarse = tp._coarse
    spans = []
    for i in range(coarse.n_runs):
        run_pos, targets, _ = coarse.run(i)
        spans.append(int(targets[-1] - run_pos))
    spans = np.array(spans[:-80])          # drop runs clipped by series end
    assert spans.max() == 39 * 12
    assert spans.min() >= int(15.5 * 12)


def test_truncated_steps_at_thirteen_hundred_market_time():
    p = _prices(days=6)
    tp = TruncatedPerfectForecastProvider(p)
    coarse = tp._coarse
    by_hour = {}
    for i in range(coarse.n_runs - 80):
        run_pos, targets, _ = coarse.run(i)
        ts = tp._ts(int(run_pos))
        by_hour.setdefault(ts.hour, []).append(int(targets[-1] - run_pos))
    # the longest horizons occur at 13:00, the shortest just before it
    assert max(by_hour[13]) == 39 * 12
    assert max(by_hour[12]) < max(by_hour[13])


def test_truncated_fills_the_whole_window_using_infill():
    p = _prices(days=6)
    tp = TruncatedPerfectForecastProvider(p)
    s = tp.get(p.index[900], 576)
    assert len(s) == 576
    assert np.isfinite(s.rrp_mwh).all()
    assert 0.0 < s.infill_fraction < 1.0


def test_truncated_needs_no_infill_inside_the_forecast_horizon():
    p = _prices(days=6)
    tp = TruncatedPerfectForecastProvider(p)
    s = tp.get(p.index[900], 144)          # 12 h, always inside the sawtooth
    assert not s.is_infill.any()


def test_infill_fraction_varies_with_the_sawtooth():
    """Because the horizon sawtooths, the share of a 48 h window that is
    extrapolated has a strong daily rhythm. Worth asserting, since it would
    otherwise contaminate any time-of-day analysis unnoticed."""
    p = _prices(days=6)
    tp = TruncatedPerfectForecastProvider(p)
    fracs = [tp.get(p.index[i], 576).infill_fraction
             for i in range(600, 600 + 288, 12)]
    assert max(fracs) - min(fracs) > 0.1


def test_five_minute_coarse_matches_actuals_within_the_horizon():
    p = _prices(days=6)
    tp = TruncatedPerfectForecastProvider(p, coarse_resolution='5min')
    k = 900
    s = tp.get(p.index[k - 1], 144)
    np.testing.assert_allclose(s.rrp_mwh, p.to_numpy()[k:k + 144])


def test_thirty_minute_coarse_loses_resolution_beyond_the_fine_segment():
    p = _prices(days=6)
    fine_res = TruncatedPerfectForecastProvider(p, coarse_resolution='5min')
    coarse_res = TruncatedPerfectForecastProvider(p, coarse_resolution='30min')
    k = 900
    a = fine_res.get(p.index[k - 1], 144).rrp_mwh
    b = coarse_res.get(p.index[k - 1], 144).rrp_mwh
    # identical over the fine segment, different beyond it
    np.testing.assert_allclose(a[:11], b[:11])
    assert not np.allclose(a[12:], b[12:])


def test_thirty_minute_coarse_is_a_block_mean():
    p = _prices(days=6)
    tp = TruncatedPerfectForecastProvider(p, coarse_resolution='30min')
    k = 900
    s = tp.get(p.index[k - 1], 144)
    vals = p.to_numpy()
    # Coarse targets sit on 30-minute MARKET boundaries. The series starts at
    # 00:05, so boundary positions are those with pos % 6 == 5, not multiples
    # of 6. Position 17 into the window sits in the coarse segment.
    j = 17
    pos = k + j
    block_end = pos + ((5 - pos) % 6)
    assert tp._ts(block_end).minute % 30 == 0
    expected = vals[block_end - 5:block_end + 1].mean()
    assert s.rrp_mwh[j] == pytest.approx(expected, rel=1e-9)


def test_truncated_marks_the_splice_boundary():
    """is_fine must agree with where the values actually stop matching the
    5-minute actuals, which is the whole point of recording it."""
    p = _prices(days=6)
    tp = TruncatedPerfectForecastProvider(p, coarse_resolution='30min')
    k = 900
    s = tp.get(p.index[k - 1], 144)
    vals = p.to_numpy()
    assert s.is_fine[:11].all()
    assert not s.is_fine[11:].any()
    np.testing.assert_allclose(s.rrp_mwh[s.is_fine], vals[k:k + 11])


def test_truncated_coarse_run_is_older_than_the_fine_run():
    """The fine segment publishes every interval, the coarse one every 30
    minutes, so the two stalenesses are structurally different."""
    p = _prices(days=6)
    tp = TruncatedPerfectForecastProvider(p)
    leads = []
    for i in range(900, 906):
        s = tp.get(p.index[i], 144)
        assert s.run_time == p.index[i]              # fine is never stale
        leads.append((p.index[i] - s.coarse_run_time).total_seconds() / 60)
    assert max(leads) > 0, "the coarse run must go stale between publications"
    assert max(leads) <= 30


def test_rejects_unknown_coarse_resolution():
    with pytest.raises(ValueError, match="coarse_resolution"):
        TruncatedPerfectForecastProvider(_prices(days=1), coarse_resolution='1h')


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        TruncatedPerfectForecastProvider(_prices(days=1), mode='actual')


# -------------------------------------------------------------------- modes

def test_price_and_next_prices_the_committed_interval_at_outturn():
    """AEMO publishes the settled price ~9 s after the interval begins, so
    this is the realistic case rather than an unachievable bound."""
    p = _prices(days=6)
    tp = TruncatedPerfectForecastProvider(p, mode='price_and_next')
    k = 900
    s = tp.get(p.index[k - 1], 48)
    assert s.rrp_mwh[0] == p.to_numpy()[k]


def test_price_known_uses_the_actual_but_the_earlier_forward_window():
    """The information set available ~9 s into the interval: AEMO has published
    the settled price, but the next P5MIN run is still 30 s away."""
    p = _prices(days=6)
    k = 900
    base = TruncatedPerfectForecastProvider(p, mode='forecast').get(p.index[k-1], 48)
    pk = TruncatedPerfectForecastProvider(p, mode='price_known').get(p.index[k-1], 48)
    assert pk.rrp_mwh[0] == p.to_numpy()[k], "committed interval at the actual"
    assert pk.run_time == base.run_time, "forward window from the earlier run"
    np.testing.assert_allclose(pk.rrp_mwh[1:], base.rrp_mwh[1:])


def test_the_three_coherent_modes_form_an_information_ladder():
    """0 s, ~9 s and ~39 s into the interval, distinguished by which RUN
    supplies the forward window.

    With this provider the forecast IS the outturn, so all three modes agree on
    the committed interval; that is correct, not a bug, and it is why the
    committed-interval distinction is asserted in the pre-dispatch tests where
    forecast error actually exists.
    """
    p = _prices(days=6)
    k = 900
    got = {m: TruncatedPerfectForecastProvider(p, mode=m).get(p.index[k-1], 48)
           for m in ('forecast', 'price_known', 'price_and_next')}
    for m in got:
        assert got[m].rrp_mwh[0] == p.to_numpy()[k]
    # forward window: earlier run, earlier run, later run
    assert got['price_known'].run_time == got['forecast'].run_time
    assert got['price_and_next'].run_time > got['forecast'].run_time


def test_price_and_next_uses_the_later_run_for_the_forward_window():
    """The incoherent predecessor took the committed interval from the actual
    but the forward window from the run available BEFORE the interval. Nobody
    would build that: if you have the price at 9 s you have the run at 39 s."""
    p = _prices(days=6)
    k = 900
    a = TruncatedPerfectForecastProvider(p, mode='next_forecast').get(p.index[k-1], 48)
    b = TruncatedPerfectForecastProvider(p, mode='price_and_next').get(p.index[k-1], 48)
    assert b.run_time == a.run_time
    np.testing.assert_allclose(b.rrp_mwh[1:], a.rrp_mwh[1:])


def test_next_forecast_uses_a_run_labelled_one_interval_later():
    p = _prices(days=6)
    k = 900
    a = TruncatedPerfectForecastProvider(p, mode='forecast').get(p.index[k - 1], 48)
    b = TruncatedPerfectForecastProvider(p, mode='next_forecast').get(p.index[k - 1], 48)
    assert b.run_time == a.run_time + pd.Timedelta(minutes=5)
    assert b.run_time == p.index[k]


def test_all_modes_produce_a_full_window():
    p = _prices(days=6)
    for mode in ('forecast', 'next_forecast', 'price_and_next'):
        s = TruncatedPerfectForecastProvider(p, mode=mode).get(p.index[899], 288)
        assert len(s) == 288 and np.isfinite(s.rrp_mwh).all()


def test_run_time_never_exceeds_the_permitted_instant():
    """The causality property, checked across many decision instants."""
    p = _prices(days=6)
    for mode, allowance in (('forecast', 0), ('price_known', 0),
                            ('next_forecast', 1), ('price_and_next', 1)):
        tp = TruncatedPerfectForecastProvider(p, mode=mode)
        for i in range(400, 1200, 37):
            di = p.index[i]
            s = tp.get(di, 288)
            limit = di + pd.Timedelta(minutes=5 * allowance)
            assert s.run_time <= limit, (mode, di, s.run_time)
