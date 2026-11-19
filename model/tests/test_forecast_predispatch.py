"""Tests for PredispatchForecastProvider.

Every store used here is built by running synthetic raw frames through the
REAL stage B cleaner and partition writer, so the test exercises the whole
chain from raw MMSDM shape to assembled window. A provider that passes against
a hand-built parquet but fails against stage B's output would be worthless.
"""
import json

import numpy as np
import pandas as pd
import pytest

from model.core.forecast import ForecastSlice
from model.forecast.predispatch import PredispatchForecastProvider
from pipeline.pre_dispatch.B_Clean_Predispatch import (
    clean_forecast_frame, write_partitions,
)
from pipeline.pre_dispatch.C_Validate_Predispatch import (
    build_manifest, predispatch_horizon_end, validate_store,
    KNOWN_ARCHIVE_GAPS,
)

TZ = 'Etc/GMT-10'
REGION = 'SA1'


# ------------------------------------------------------------------ fixtures

def _actuals(start='2026-02-10 00:05', days=4, seed=0):
    idx = pd.date_range(start, periods=288 * days, freq='5min', tz=TZ)
    rng = np.random.default_rng(seed)
    tod = (np.arange(len(idx)) % 288) / 288
    base = (60 + 240 * np.exp(-((tod - 0.79) ** 2) / 0.0012)
            - 50 * np.exp(-((tod - 0.5) ** 2) / 0.006))
    return pd.Series(base + rng.normal(0, 25, len(idx)), index=idx)


def _build_store(tmp_path, actuals, lead_bias=0.0, lead0_only=False,
                 skip_fine=False):
    """Write a store whose forecasts are the actuals plus a known bias.

    A bias rather than noise so a test can assert exactly which source a value
    came from: a fine value is actuals+bias, a coarse value is actuals+2*bias,
    and an infilled value is a 24-hour-lagged actual.
    """
    start = actuals.index[0] - pd.Timedelta(days=1)
    a = actuals.reindex(pd.date_range(start, actuals.index[-1], freq='5min',
                                      tz=TZ)).bfill()

    if not skip_fine:
        write_partitions(clean_forecast_frame(_fine_rows(a, lead_bias), 'p5min'),
                         'p5min', tmp_path, merge=True)
    write_partitions(
        clean_forecast_frame(_coarse_rows(a, lead_bias, lead0_only),
                             'predispatch'),
        'predispatch', tmp_path, merge=True)
    return tmp_path


# The two raw frames are built with positional arrays rather than a row loop
# with per-element .loc lookups, which took about a minute across this file.
# a.index is a gap-free 5-minute range, so "t in a.index" is "position < n".

def _fine_rows(a, lead_bias):
    """P5MIN raw frame: a run at every interval, leads 0-11, a lead kept
    only where its target lies inside a."""
    n = len(a)
    naive = a.index.tz_localize(None)
    vals = a.to_numpy(dtype=float)
    run_pos = np.repeat(np.arange(n), 12)
    tgt_pos = run_pos + np.tile(np.arange(12), n)
    keep = tgt_pos < n
    run_pos, tgt_pos = run_pos[keep], tgt_pos[keep]
    return pd.DataFrame({'RUN_DATETIME': naive[run_pos],
                         'INTERVAL_DATETIME': naive[tgt_pos],
                         'REGIONID': REGION,
                         'RRP': vals[tgt_pos] + lead_bias,
                         'INTERVENTION': 0})


def _coarse_rows(a, lead_bias, lead0_only):
    """PREDISPATCH raw frame: a run at every half hour, 30-minute targets to
    the run's horizon end; a target beyond a takes a's last value."""
    n = len(a)
    naive = a.index.tz_localize(None)
    vals = a.to_numpy(dtype=float)
    frames = []
    for pos in np.flatnonzero((a.index.minute % 30 == 0) & (a.index.second == 0)):
        run = naive[pos]
        end = predispatch_horizon_end(run)
        k = 1 if lead0_only else int((end - run).total_seconds() // 1800) + 1
        offs = 6 * np.arange(k)
        src = np.minimum(pos + offs, n - 1)
        frames.append(pd.DataFrame({
            'PREDISPATCH_RUN_DATETIME': np.repeat(run, k),
            'DATETIME': run + pd.to_timedelta(30 * np.arange(k), unit='min'),
            'REGIONID': REGION,
            'RRP': vals[src] + 2 * lead_bias,
            'INTERVENTION': 0}))
    return pd.concat(frames, ignore_index=True)


def _provider(tmp_path, actuals, **kw):
    return PredispatchForecastProvider(tmp_path, REGION, actuals, **kw)


# ------------------------------------------------------------------- loading

def test_loads_and_returns_a_full_window(tmp_path):
    a = _actuals()
    _build_store(tmp_path, a)
    p = _provider(tmp_path, a)
    s = p.get(a.index[600], 216)                 # 18 h, the current default
    assert isinstance(s, ForecastSlice)
    assert len(s) == 216
    assert np.isfinite(s.rrp_mwh).all()


def test_missing_store_raises_with_a_pointer(tmp_path):
    a = _actuals()
    with pytest.raises(ValueError, match="A_Download_Predispatch"):
        _provider(tmp_path, a)


def test_horizon_available_reflects_the_sawtooth(tmp_path):
    a = _actuals()
    _build_store(tmp_path, a)
    p = _provider(tmp_path, a)
    assert p.horizon_available() == 39 * 12      # 39 h from the run label


# ---------------------------------------------------------------- causality

def test_run_label_is_never_after_the_decision_instant(tmp_path):
    a = _actuals()
    _build_store(tmp_path, a)
    for mode, allowance in (('forecast', 0), ('next_forecast', 1),
                            ('price_and_next', 1)):
        p = _provider(tmp_path, a, mode=mode)
        for i in range(400, 900, 53):
            s = p.get(a.index[i], 216)
            limit = a.index[i] + pd.Timedelta(minutes=5 * allowance)
            assert s.run_time <= limit, (mode, a.index[i], s.run_time)


def test_fine_segment_supplies_the_committed_interval(tmp_path):
    """Interval k sits at P5MIN lead 5 minutes, never lead 0."""
    a = _actuals()
    _build_store(tmp_path, a, lead_bias=1000.0)
    p = _provider(tmp_path, a)
    k = 600
    s = p.get(a.index[k - 1], 216)
    assert s.rrp_mwh[0] == pytest.approx(a.iloc[k] + 1000.0)
    assert not s.is_infill[0]


def test_is_fine_agrees_with_the_value_provenance(tmp_path):
    """The bias encodes the source, so is_fine is checkable against it rather
    than against the same arithmetic that produced it."""
    a = _actuals()
    _build_store(tmp_path, a, lead_bias=1000.0)
    p = _provider(tmp_path, a)
    s = p.get(a.index[599], 216)
    k = 600
    vals = a.to_numpy()
    fine_pos = np.flatnonzero(s.is_fine)
    assert fine_pos.size == 11
    assert (fine_pos == np.arange(11)).all()
    for j in fine_pos:
        assert s.rrp_mwh[j] == pytest.approx(vals[k + j] + 1000.0)


def test_coarse_run_time_is_reported_and_can_be_stale(tmp_path):
    """PREDISPATCH publishes every 30 minutes, so the run in hand is 0 to 25
    minutes old, while P5MIN is fresh at every decision instant."""
    a = _actuals()
    _build_store(tmp_path, a)
    p = _provider(tmp_path, a)
    leads = []
    for i in range(600, 606):
        s = p.get(a.index[i], 216)
        assert s.run_time == a.index[i]
        leads.append((a.index[i] - s.coarse_run_time).total_seconds() / 60)
    assert max(leads) > 0
    assert max(leads) <= 30


def test_degraded_actuals_window_claims_no_forecast_provenance(tmp_path):
    """Serving actuals is not a forecast, so nothing is marked fine."""
    a = _actuals()
    _build_store(tmp_path, a, lead0_only=True)
    _degraded_manifest(tmp_path)
    p = _provider(tmp_path, a, on_degraded_period='actuals')
    s = p.get(a.index[600], 48)
    assert s.is_infill.all()
    assert not s.is_fine.any()
    assert s.coarse_run_time == a.index[600]


def test_coarse_segment_takes_over_beyond_the_fine_hour(tmp_path):
    a = _actuals()
    _build_store(tmp_path, a, lead_bias=1000.0)
    p = _provider(tmp_path, a)
    s = p.get(a.index[599], 216)
    # fine covers ~11 intervals from the decision instant; coarse beyond
    assert s.rrp_mwh[0] - a.iloc[600] == pytest.approx(1000.0)
    assert s.rrp_mwh[60] - a.iloc[660] == pytest.approx(2000.0, abs=200.0)


# ------------------------------------------------------------------- modes

def test_price_and_next_uses_the_outturn(tmp_path):
    a = _actuals()
    _build_store(tmp_path, a, lead_bias=1000.0)
    p = _provider(tmp_path, a, mode='price_and_next')
    k = 600
    s = p.get(a.index[k - 1], 48)
    assert s.rrp_mwh[0] == pytest.approx(a.iloc[k])


def test_next_forecast_uses_a_later_run(tmp_path):
    a = _actuals()
    _build_store(tmp_path, a)
    k = 600
    base = _provider(tmp_path, a, mode='forecast').get(a.index[k - 1], 48)
    nxt = _provider(tmp_path, a, mode='next_forecast').get(a.index[k - 1], 48)
    assert nxt.run_time == base.run_time + pd.Timedelta(minutes=5)


def test_unknown_mode_rejected(tmp_path):
    a = _actuals()
    _build_store(tmp_path, a)
    with pytest.raises(ValueError, match="unknown mode"):
        _provider(tmp_path, a, mode='actual')


# ------------------------------------------------------------------ infill

def test_infill_fires_only_beyond_the_forecast_horizon(tmp_path):
    a = _actuals(days=4)
    _build_store(tmp_path, a)
    p = _provider(tmp_path, a)
    short = p.get(a.index[600], 144)             # 12 h, inside coverage
    assert not short.is_infill.any()
    long = p.get(a.index[600], 576)              # 48 h, beyond it
    assert long.is_infill.any()
    assert long.is_infill[-1]


def test_infill_share_at_the_eighteen_hour_default_is_small(tmp_path):
    """About 1% MEAN on real data, worth roughly 0.000025 VCR.

    The mean is the meaningful figure, not the max. The pre-dispatch horizon
    sawtooths, so at its 15.08 h floor an 18 h window has roughly 16% infill,
    while for the 87.8% of decision instants with 18 h or more available it has
    none. Asserting on the max would be asserting the sawtooth away.
    """
    a = _actuals(days=4)
    _build_store(tmp_path, a)
    p = _provider(tmp_path, a)
    fracs = [p.get(a.index[i], 216).infill_fraction
             for i in range(400, 900, 25)]
    assert np.mean(fracs) < 0.05
    assert max(fracs) < 0.25


# -------------------------------------------------------- degraded periods

def _degraded_manifest(tmp_path):
    reports = validate_store(tmp_path, tables=('predispatch',),
                             regions=(REGION,), verbose=False)
    return build_manifest(reports, tmp_path)


def test_degraded_period_fails_by_default(tmp_path):
    """The October 2022 case. A lead-0-only month LOOKS complete: full run
    coverage, no gaps, every price present. Only the manifest distinguishes
    it."""
    a = _actuals()
    _build_store(tmp_path, a, lead0_only=True)   # P5MIN unaffected
    manifest = _degraded_manifest(tmp_path)
    assert any(e['degraded'] for e in manifest['entries'])
    with pytest.raises(ValueError, match="no usable forecast lead"):
        _provider(tmp_path, a)


def test_degraded_period_can_fall_back_to_self_consumption(tmp_path):
    a = _actuals()
    _build_store(tmp_path, a, lead0_only=True)   # P5MIN unaffected
    _degraded_manifest(tmp_path)
    p = _provider(tmp_path, a, on_degraded_period='self_consumption')
    assert p.get(a.index[600], 216) is None


def test_degraded_period_can_serve_actuals_marked_as_infill(tmp_path):
    """Marked so a result that used actuals is distinguishable from one that
    did not."""
    a = _actuals()
    _build_store(tmp_path, a, lead0_only=True)   # P5MIN unaffected
    _degraded_manifest(tmp_path)
    p = _provider(tmp_path, a, on_degraded_period='actuals')
    s = p.get(a.index[600], 48)
    assert s.is_infill.all()
    np.testing.assert_allclose(s.rrp_mwh, a.to_numpy()[601:649])


def test_invalid_degraded_policy_rejected(tmp_path):
    a = _actuals()
    _build_store(tmp_path, a)
    with pytest.raises(ValueError, match="on_degraded_period"):
        _provider(tmp_path, a, on_degraded_period='ignore')


def test_degraded_period_infill_uses_real_fine_data_where_it_exists(tmp_path):
    """'infill' adds no special handling of its own: it falls through to the
    ordinary assemble_window(...) path used everywhere else. With P5MIN
    unaffected (as in October 2022), the near-term part of the window should
    be genuine forecast, not infill, and only the tail -- beyond fine's ~55
    minute reach, which the lead-0 coarse segment cannot extend -- should be
    marked infill. This is the mechanism rolling_optimisation_spec.md's
    Failure handling section describes; this test is what confirms it rather
    than just asserting it in prose."""
    a = _actuals()
    _build_store(tmp_path, a, lead0_only=True)   # P5MIN unaffected
    _degraded_manifest(tmp_path)
    p = _provider(tmp_path, a, on_degraded_period='infill')
    s = p.get(a.index[600], 48)
    assert s is not None
    # Fine covers roughly the first ~11 intervals from the decision instant
    # (see test_coarse_segment_takes_over_beyond_the_fine_hour above, same
    # store shape); those must be real forecast, not infill.
    assert not s.is_infill[:10].any()
    assert s.is_fine[:10].any()
    # Coarse is lead-0 only here, so it cannot reach past the fine segment;
    # the tail must fall back to infill rather than reading a phantom coarse
    # value.
    assert s.is_infill[20:].all()


def test_degraded_period_infill_does_not_construct_specially(tmp_path):
    """A period marked degraded and 'infill' selected must be
    INDISTINGUISHABLE, at the assembled-window level, from an ordinary window
    whose horizon simply runs past real coverage -- there is no separate code
    path to test other than assemble_window itself."""
    a = _actuals()
    _build_store(tmp_path, a, lead0_only=True)
    _degraded_manifest(tmp_path)
    p_infill = _provider(tmp_path, a, on_degraded_period='infill')
    # The comparison provider must never have a chance to see the degraded
    # flag at all, not merely be configured with a permissive policy --
    # pointing it at a manifest path that does not exist makes
    # _degraded_ranges return [] unconditionally (see predispatch.py), so
    # _is_degraded(k) can never be True and 'fail' (the default) never has
    # anything to raise about. Reusing tmp_path's real manifest.json here
    # (the original bug) meant BOTH providers saw the same degraded entry,
    # so construction of p_fail_ok raised before any comparison happened.
    p_fail_ok = _provider(tmp_path, a,
                          manifest_path=tmp_path / 'no_manifest_here.json')
    # Same underlying store, same request, only the degraded-period POLICY
    # and manifest visibility differ -- 'fail' with no manifest in view and
    # 'infill' with the real degraded one must produce the identical window,
    # since 'infill' takes no special branch and neither does an unflagged
    # period.
    s_infill = p_infill.get(a.index[600], 48)
    s_plain = p_fail_ok.get(a.index[600], 48)
    np.testing.assert_array_equal(s_infill.rrp_mwh, s_plain.rrp_mwh)
    np.testing.assert_array_equal(s_infill.is_infill, s_plain.is_infill)


def test_known_archive_gap_raises_even_without_lead0_degradation(tmp_path,
                                                                   monkeypatch):
    """October 2022's actual shape: a single surviving run with a perfectly
    ordinary lead (so max_lead_minutes != 0, and the lead-0-only 'degraded'
    check alone does NOT catch it) but confirmed absent from the archive.
    known_archive_gap is the signal that catches this one; it must be
    sufficient on its own, independent of 'degraded'."""
    a = _actuals()
    _build_store(tmp_path, a)   # healthy store: normal lead, no degradation
    reports = validate_store(tmp_path, tables=('predispatch',),
                             regions=(REGION,), verbose=False)
    assert not any(r.degraded for r in reports), \
        "fixture must NOT be lead-0 degraded, to isolate known_archive_gap"
    period = str(pd.Period(a.index[0], freq='M'))
    monkeypatch.setitem(KNOWN_ARCHIVE_GAPS, ('predispatch', period),
                        'synthetic gap for this test')
    build_manifest(reports, tmp_path)
    with pytest.raises(ValueError, match="no usable forecast lead"):
        _provider(tmp_path, a)
    # 'infill' still gets through, exactly as for a lead-0 degradation.
    p = _provider(tmp_path, a, on_degraded_period='infill')
    assert p.get(a.index[600], 48) is not None


def test_absent_manifest_does_not_block_a_healthy_store(tmp_path):
    a = _actuals()
    _build_store(tmp_path, a)
    assert not (tmp_path / 'manifest.json').exists()
    p = _provider(tmp_path, a)
    assert p.get(a.index[600], 216) is not None


def test_healthy_manifest_permits_construction(tmp_path):
    a = _actuals()
    _build_store(tmp_path, a)
    reports = validate_store(tmp_path, tables=('p5min', 'predispatch'),
                             regions=(REGION,), verbose=False)
    build_manifest(reports, tmp_path)
    entries = json.loads((tmp_path / 'manifest.json').read_text())['entries']
    assert not any(e['degraded'] for e in entries)
    assert _provider(tmp_path, a).get(a.index[600], 216) is not None


# ------------------------------------------------------------------- edges

def test_first_interval_falls_back(tmp_path):
    a = _actuals()
    _build_store(tmp_path, a)
    p = _provider(tmp_path, a)
    assert p.get(a.index[0] - pd.Timedelta(minutes=5), 216) is None


def test_window_shortens_at_the_period_end(tmp_path):
    a = _actuals()
    _build_store(tmp_path, a)
    p = _provider(tmp_path, a)
    s = p.get(a.index[-20], 216)
    assert len(s) == 19


def test_off_grid_decision_instant_rejected(tmp_path):
    a = _actuals()
    _build_store(tmp_path, a)
    p = _provider(tmp_path, a)
    with pytest.raises(ValueError, match="grid"):
        p.get(a.index[100] + pd.Timedelta(minutes=2), 216)


# --------------------------------------------------------- store coverage

def test_store_ending_before_the_period_is_refused(tmp_path):
    """The failure a real cohort run hit: two sites sat entirely after the
    store ended. Nothing raised, no fallbacks were reported, and the provider
    carried months-old runs forward while infill supplied 98.7% of every
    window from 24-hour-lagged actuals. The numbers looked plausible and were
    really yesterday's prices."""
    a = _actuals(days=8)
    _build_store(tmp_path, a.iloc[:288 * 2])          # store covers 2 of 8 days
    with pytest.raises(ValueError, match="does not cover the modelled period"):
        _provider(tmp_path, a)


def test_store_starting_after_the_period_is_refused(tmp_path):
    a = _actuals(days=8)
    _build_store(tmp_path, a.iloc[288 * 5:])
    with pytest.raises(ValueError, match="does not cover the modelled period"):
        _provider(tmp_path, a)


def test_the_error_names_the_gap_and_the_remedy(tmp_path):
    a = _actuals(days=8)
    _build_store(tmp_path, a.iloc[:288 * 2])
    with pytest.raises(ValueError) as exc:
        _provider(tmp_path, a)
    msg = str(exc.value)
    assert 'period end' in msg or 'first run' in msg
    assert 'A_Download_' in msg, "must say how to fix it"
    assert 'max_staleness_hours' in msg, "must say how to override deliberately"


def test_a_deliberate_override_is_possible(tmp_path):
    """Refusing must not be unconditional: a caller who understands the gap
    can proceed."""
    a = _actuals(days=8)
    _build_store(tmp_path, a.iloc[:288 * 2])
    p = _provider(tmp_path, a, max_staleness_hours=24 * 10)
    assert p.get(a.index[100], 216) is not None


def test_full_coverage_passes(tmp_path):
    a = _actuals(days=4)
    _build_store(tmp_path, a)
    assert _provider(tmp_path, a).get(a.index[600], 216) is not None


def test_an_isolated_missing_run_does_not_trip_the_guard(tmp_path):
    """Four isolated missing runs in ~96,000 across 5.5 years. Carry-forward
    handles them and the guard must not confuse them with a truncated store."""
    a = _actuals(days=4)
    _build_store(tmp_path, a)
    path = tmp_path / 'predispatch' / REGION / '202602.parquet'
    df = pd.read_parquet(path)
    drop = sorted(df.run_time.unique())[40]
    df[df.run_time != drop].to_parquet(path, index=False)
    assert _provider(tmp_path, a).get(a.index[600], 216) is not None


def test_price_known_beats_the_forecast_on_the_committed_interval(tmp_path):
    """Where forecast error exists, `price_known` prices interval k at the
    outturn while `forecast` prices it at a five-minute-ahead estimate. This is
    the distinction the truncated provider cannot show."""
    a = _actuals()
    _build_store(tmp_path, a, lead_bias=1000.0)
    k = 600
    fc = _provider(tmp_path, a, mode='forecast').get(a.index[k - 1], 48)
    pk = _provider(tmp_path, a, mode='price_known').get(a.index[k - 1], 48)
    assert fc.rrp_mwh[0] == pytest.approx(a.iloc[k] + 1000.0)
    assert pk.rrp_mwh[0] == pytest.approx(a.iloc[k])
    assert pk.run_time == fc.run_time, "forward window from the same run"
    np.testing.assert_allclose(pk.rrp_mwh[1:], fc.rrp_mwh[1:])
