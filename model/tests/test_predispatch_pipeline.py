"""Tests for the pre-dispatch store pipeline, stages B and C.

Synthetic raw frames shaped like real MMSDM output. Every trap from
predispatch_data_spec.md is reproduced deliberately, because each one produces
a store that LOADS CLEANLY AND IS WRONG, and a test is the only thing standing
between that and a plausible-looking result.

Stage A's download is not tested here: it talks to nemweb.com.au, which no
test environment should depend on. nemweb_direct.py was validated against the
real archive during the investigation.
"""
import json

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.pre_dispatch.B_Clean_Predispatch import (
    clean_forecast_frame, partition_path, read_store, write_partitions,
)
from pipeline.pre_dispatch.C_Validate_Predispatch import (
    build_manifest, expected_target_count, predispatch_horizon_end,
    validate_partition, validate_store,
)

TZ = 'Etc/GMT-10'


# ------------------------------------------------------------------ fixtures

def raw_p5min(start='2026-02-10 00:05', n_runs=12, regions=('SA1', 'NSW1'),
              intervention=False, price=50.0):
    """P5MIN-shaped raw frame: 12 targets per run at leads 0..55 min."""
    rows = []
    base = pd.Timestamp(start)
    for i in range(n_runs):
        run = base + pd.Timedelta(minutes=5 * i)
        for region in regions:
            for lead in range(12):
                rows.append({
                    'RUN_DATETIME': run,
                    'INTERVAL_DATETIME': run + pd.Timedelta(minutes=5 * lead),
                    'REGIONID': region, 'RRP': price + lead,
                    'INTERVENTION': 0,
                    'NOISE_COL': 'ignored',      # schema drift is additive
                })
                if intervention:
                    r = dict(rows[-1])
                    r['INTERVENTION'] = 1
                    r['RRP'] = 9999.0            # the case we must NOT keep
                    rows.append(r)
    return pd.DataFrame(rows)


def raw_predispatch(start='2026-02-10 13:00', n_runs=4, regions=('SA1',),
                    price=80.0):
    """PREDISPATCH-shaped raw frame with the real sawtooth horizon."""
    rows = []
    base = pd.Timestamp(start)
    for i in range(n_runs):
        run = base + pd.Timedelta(minutes=30 * i)
        end = predispatch_horizon_end(run)
        n = int((end - run).total_seconds() // 1800) + 1
        for region in regions:
            for k in range(n):
                rows.append({
                    'PREDISPATCH_RUN_DATETIME': run,
                    'DATETIME': run + pd.Timedelta(minutes=30 * k),
                    'REGIONID': region, 'RRP': price + k,
                    'INTERVENTION': 0, 'PERIODID': k + 1,
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------- stage B: cleaning

def test_canonical_schema_and_types():
    out = clean_forecast_frame(raw_p5min(), 'p5min')
    assert list(out.columns) == ['run_time', 'region', 'lead_minutes', 'rrp_mwh']
    assert out['lead_minutes'].dtype == np.int16
    assert out['rrp_mwh'].dtype == np.float64
    assert str(out['run_time'].dt.tz) == TZ


def test_target_time_is_reconstructible_from_run_time_and_lead():
    """The whole reason target_time is not stored."""
    raw = raw_p5min()
    out = clean_forecast_frame(raw, 'p5min')
    rebuilt = out['run_time'] + pd.to_timedelta(out['lead_minutes'], unit='m')
    expected = (raw.sort_values(['REGIONID', 'RUN_DATETIME', 'INTERVAL_DATETIME'])
                ['INTERVAL_DATETIME'].dt.tz_localize(TZ).to_numpy())
    np.testing.assert_array_equal(np.sort(rebuilt.to_numpy()), np.sort(expected))


def test_intervention_case_one_is_dropped():
    """TRAP: case 1 is the physical dispatch run's shadow prices and does NOT
    match settled prices. Keeping it would corrupt exactly the intervals where
    arbitrage value concentrates."""
    out = clean_forecast_frame(raw_p5min(intervention=True), 'p5min')
    assert (out['rrp_mwh'] != 9999.0).all()
    assert out.attrs['n_intervention_rows_dropped'] == len(out)


def test_intervention_filter_leaves_the_key_unique():
    out = clean_forecast_frame(raw_p5min(intervention=True), 'p5min')
    assert not out.duplicated(subset=['run_time', 'region', 'lead_minutes']).any()


def test_surviving_duplicate_raises_rather_than_being_dropped():
    """A duplicate that survives the filter differs in VALUE, so silently
    keeping one would pick an arbitrary price."""
    raw = raw_p5min(n_runs=1)
    dup = raw.iloc[[0]].copy()
    dup['RRP'] = 123.0
    with pytest.raises(ValueError, match="duplicated"):
        clean_forecast_frame(pd.concat([raw, dup], ignore_index=True), 'p5min')


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_price_dtype_is_normalised_whatever_the_source_gave(dtype):
    """TRAP: nemseer's downcasting makes the SAME column change dtype between
    adjacent months of the same table."""
    raw = raw_p5min(price=50.0)
    raw['RRP'] = raw['RRP'].astype(dtype)
    out = clean_forecast_frame(raw, 'p5min')
    assert out['rrp_mwh'].dtype == np.float64


def test_missing_column_names_what_is_missing():
    raw = raw_p5min().drop(columns='INTERVENTION')
    with pytest.raises(ValueError, match="INTERVENTION"):
        clean_forecast_frame(raw, 'p5min')


def test_negative_lead_raises():
    """Points at the PREDISPATCHSEQNO arithmetic, which is derived rather than
    read and would otherwise fail silently."""
    raw = raw_predispatch(n_runs=1)
    raw.loc[0, 'DATETIME'] = raw.loc[0, 'PREDISPATCH_RUN_DATETIME'] - pd.Timedelta(minutes=30)
    with pytest.raises(ValueError, match="target before"):
        clean_forecast_frame(raw, 'predispatch')


def test_misaligned_lead_raises():
    raw = raw_predispatch(n_runs=1)
    raw.loc[1, 'DATETIME'] += pd.Timedelta(minutes=7)
    with pytest.raises(ValueError, match="whole multiple"):
        clean_forecast_frame(raw, 'predispatch')


def test_null_price_raises():
    raw = raw_p5min()
    raw.loc[3, 'RRP'] = np.nan
    with pytest.raises(ValueError, match="null price"):
        clean_forecast_frame(raw, 'p5min')


def test_unknown_table_raises():
    with pytest.raises(ValueError, match="unknown table"):
        clean_forecast_frame(raw_p5min(), 'pdpasa')


# -------------------------------------------------- stage B: partitioning

def test_partitions_split_by_region_and_calendar_month(tmp_path):
    raw = raw_p5min(start='2026-02-28 23:30', n_runs=24)   # crosses into March
    out = clean_forecast_frame(raw, 'p5min')
    paths = write_partitions(out, 'p5min', tmp_path)
    names = {f"{p.parent.name}/{p.name}" for p in paths}
    assert names == {'SA1/202602.parquet', 'SA1/202603.parquet',
                     'NSW1/202602.parquet', 'NSW1/202603.parquet'}


def test_partition_holds_only_its_own_month(tmp_path):
    """The store is partitioned on the CALENDAR month of run_time, not on
    AEMO's shifted trading-day clock. A reader must be able to assume that."""
    out = clean_forecast_frame(raw_p5min(start='2026-02-28 23:30', n_runs=24),
                               'p5min')
    write_partitions(out, 'p5min', tmp_path)
    feb = pd.read_parquet(partition_path(tmp_path, 'p5min', 'SA1',
                                         pd.Period('2026-02', freq='M')))
    assert (feb['run_time'].dt.month == 2).all()


def test_merge_combines_two_source_months_into_one_partition(tmp_path):
    """One source file spills across a month boundary, so consecutive source
    months can both contribute to a single partition."""
    # 12 runs from 23:30 straddle midnight: 6 land in February, 6 in March.
    a = clean_forecast_frame(raw_p5min(start='2026-02-28 23:30', n_runs=12), 'p5min')
    b = clean_forecast_frame(raw_p5min(start='2026-03-01 00:35', n_runs=6), 'p5min')
    write_partitions(a, 'p5min', tmp_path, merge=True)
    write_partitions(b, 'p5min', tmp_path, merge=True)
    march = pd.read_parquet(partition_path(tmp_path, 'p5min', 'SA1',
                                           pd.Period('2026-03', freq='M')))
    assert march['run_time'].nunique() == 12
    assert not march.duplicated(subset=['run_time', 'lead_minutes']).any()


def test_read_store_round_trips(tmp_path):
    out = clean_forecast_frame(raw_p5min(n_runs=24), 'p5min')
    write_partitions(out, 'p5min', tmp_path)
    got = read_store(tmp_path, 'p5min', 'SA1',
                     pd.Timestamp('2026-02-10 00:05', tz=TZ),
                     pd.Timestamp('2026-02-10 01:00', tz=TZ))
    assert len(got) == 12 * 12
    assert got['run_time'].is_monotonic_increasing


def test_read_store_returns_typed_empty_when_absent(tmp_path):
    got = read_store(tmp_path, 'p5min', 'SA1',
                     pd.Timestamp('2026-02-10', tz=TZ),
                     pd.Timestamp('2026-02-11', tz=TZ))
    assert got.empty and list(got.columns) == ['run_time', 'lead_minutes', 'rrp_mwh']


# ----------------------------------------------- stage C: the closed form

@pytest.mark.parametrize("run,expected_hours", [
    ('2026-02-10 13:00', 39.0),    # just after bid closure: the maximum
    ('2026-02-10 12:30', 15.5),    # just before: the minimum
    ('2026-02-10 04:00', 24.0),
    ('2026-02-10 00:00', 28.0),
])
def test_sawtooth_closed_form(run, expected_hours):
    ts = pd.Timestamp(run)
    got = (predispatch_horizon_end(ts) - ts).total_seconds() / 3600
    assert got == pytest.approx(expected_hours)


def test_sawtooth_bounds_match_the_investigation():
    """15.5 to 39.0 hours from the run label, validated there against all
    1,345 February 2026 runs."""
    runs = pd.date_range('2026-02-10 00:00', '2026-02-12 00:00', freq='30min')
    spans = [(predispatch_horizon_end(r) - r).total_seconds() / 3600 for r in runs]
    assert min(spans) == pytest.approx(15.5)
    assert max(spans) == pytest.approx(39.0)


def test_target_counts_span_the_documented_range():
    runs = pd.date_range('2026-02-10 00:00', '2026-02-11 23:30', freq='30min')
    counts = [expected_target_count(r, 'predispatch') for r in runs]
    assert min(counts) == 32 and max(counts) == 79


def test_p5min_target_count_is_constant():
    assert expected_target_count(pd.Timestamp('2026-02-10 07:35'), 'p5min') == 12


# --------------------------------------------------- stage C: validation

def _store(tmp_path, table='predispatch', **kw):
    raw = raw_predispatch(**kw) if table == 'predispatch' else raw_p5min(**kw)
    write_partitions(clean_forecast_frame(raw, table), table, tmp_path)
    return tmp_path


def test_healthy_partition_passes(tmp_path):
    _store(tmp_path, 'predispatch', n_runs=4)
    rep = validate_partition(tmp_path, 'predispatch', 'SA1',
                             pd.Period('2026-02', freq='M'))
    assert rep.ok and not rep.wrong_target_count and not rep.degraded
    assert rep.max_lead_minutes == 39 * 60


def test_missing_run_is_detected(tmp_path):
    raw = raw_predispatch(n_runs=6)
    drop = raw['PREDISPATCH_RUN_DATETIME'].unique()[2]
    raw = raw[raw['PREDISPATCH_RUN_DATETIME'] != drop]
    write_partitions(clean_forecast_frame(raw, 'predispatch'), 'predispatch', tmp_path)
    rep = validate_partition(tmp_path, 'predispatch', 'SA1',
                             pd.Period('2026-02', freq='M'))
    assert len(rep.missing_runs) == 1


def test_wrong_target_count_detected_against_the_closed_form(tmp_path):
    """Not against a median: the sawtooth would flag half of all runs."""
    raw = raw_predispatch(n_runs=4)
    first = raw['PREDISPATCH_RUN_DATETIME'].unique()[0]
    idx = raw.index[raw['PREDISPATCH_RUN_DATETIME'] == first][:3]
    write_partitions(clean_forecast_frame(raw.drop(index=idx), 'predispatch'),
                     'predispatch', tmp_path)
    rep = validate_partition(tmp_path, 'predispatch', 'SA1',
                             pd.Period('2026-02', freq='M'))
    assert len(rep.wrong_target_count) == 1
    assert rep.wrong_target_count[0]['expected'] - rep.wrong_target_count[0]['got'] == 3


def test_price_above_the_financial_year_cap_is_flagged(tmp_path):
    """Range validation belongs at ingest because the cap resets every 1 July,
    so the correct value is a per-year fact."""
    raw = raw_predispatch(n_runs=2)
    raw.loc[0, 'RRP'] = 25_000.0                      # above every cap on record
    write_partitions(clean_forecast_frame(raw, 'predispatch'), 'predispatch', tmp_path)
    rep = validate_partition(tmp_path, 'predispatch', 'SA1',
                             pd.Period('2026-02', freq='M'))
    assert rep.out_of_range == 1 and not rep.ok
    assert '20,300' in rep.errors[0] or '23,200' in rep.errors[0]


def test_price_below_the_floor_is_flagged(tmp_path):
    raw = raw_predispatch(n_runs=2)
    raw.loc[0, 'RRP'] = -1500.0
    write_partitions(clean_forecast_frame(raw, 'predispatch'), 'predispatch', tmp_path)
    rep = validate_partition(tmp_path, 'predispatch', 'SA1',
                             pd.Period('2026-02', freq='M'))
    assert rep.out_of_range == 1


def test_lead_zero_only_month_is_marked_degraded(tmp_path):
    """The October 2022 case. The DATA/ `_D` variant loads at 100% coverage
    with no lead dimension, so without this flag it is indistinguishable from
    a healthy month and a rolling run would silently use a one-step forecast."""
    raw = raw_predispatch(n_runs=4)
    raw = raw[raw['DATETIME'] == raw['PREDISPATCH_RUN_DATETIME']]
    write_partitions(clean_forecast_frame(raw, 'predispatch'), 'predispatch', tmp_path)
    rep = validate_partition(tmp_path, 'predispatch', 'SA1',
                             pd.Period('2026-02', freq='M'))
    assert rep.degraded
    assert rep.max_lead_minutes == 0


def test_absent_partition_reports_rather_than_raising(tmp_path):
    rep = validate_partition(tmp_path, 'p5min', 'SA1',
                             pd.Period('2021-01', freq='M'))
    assert not rep.ok and 'missing' in rep.errors[0]


def test_manifest_records_degraded_periods(tmp_path):
    raw = raw_predispatch(n_runs=4)
    raw = raw[raw['DATETIME'] == raw['PREDISPATCH_RUN_DATETIME']]
    write_partitions(clean_forecast_frame(raw, 'predispatch'), 'predispatch', tmp_path)
    reports = validate_store(tmp_path, tables=('predispatch',),
                             regions=('SA1',), verbose=False)
    manifest = build_manifest(reports, tmp_path)
    entry = manifest['entries'][0]
    assert entry['degraded'] is True and entry['max_lead_minutes'] == 0
    assert 'no multi-lead forecast' in entry['reason']
    on_disk = json.loads((tmp_path / 'manifest.json').read_text())
    assert on_disk['entries'] == manifest['entries']


def test_validate_store_walks_every_partition(tmp_path):
    write_partitions(clean_forecast_frame(raw_p5min(n_runs=24), 'p5min'),
                     'p5min', tmp_path)
    reports = validate_store(tmp_path, tables=('p5min',),
                             regions=('SA1', 'NSW1'), verbose=False)
    assert {r.region for r in reports} == {'SA1', 'NSW1'}
    assert all(r.ok for r in reports)


# ------------------------------------------------- stage A: orchestration
# The download itself talks to nemweb.com.au and is not tested here.
# compile_data is injected so the month selection, region filter, idempotency
# and fail-soft behaviour can be exercised without a network.

from pipeline.pre_dispatch.A_Download_Predispatch import (  # noqa: E402
    MonthResult, SOURCE, _run_window, acquire, download_month,
)


def _fake_compile(frames):
    """Stand in for nemweb_direct.compile_data, recording its arguments."""
    calls = []

    def compile_data(run_start, run_end, forecasted_start, forecasted_end,
                     forecast_type, tables, raw_cache, data_format='df', **kw):
        calls.append(dict(run_start=run_start, run_end=run_end,
                          forecasted_start=forecasted_start,
                          forecasted_end=forecasted_end,
                          forecast_type=forecast_type, tables=list(tables)))
        key = (forecast_type, tables[0])
        if key not in frames:
            raise ValueError(f"No files for {key}")
        return {tables[0]: frames[key]}

    compile_data.calls = calls
    return compile_data


def test_run_window_is_the_calendar_month():
    """Calendar months in, and nemweb_direct shifts the SOURCE selection back
    to compensate for AEMO's trading-day partitioning. Passing an
    already-shifted window here would double-shift and lose runs."""
    rs, re_, fs, fe = _run_window(pd.Period('2026-02', freq='M'))
    assert rs == '2026/02/01 00:00'
    assert re_ == '2026/03/01 00:00'
    assert fs == '2026/02/01 00:00'
    assert fe == '2026/03/04 00:00'      # +3 days covers the 39 h horizon


def test_source_routing_uses_the_multi_lead_table():
    """PREDISPATCH price must come from PREDISP_ALL_DATA, which nemweb_direct
    selects from the table name. The DATA/ `_D` variant is lead-0 only and
    would load cleanly with no lead dimension."""
    assert SOURCE['predispatch'] == ('PREDISPATCH', 'PRICE')
    assert SOURCE['p5min'] == ('P5MIN', 'REGIONSOLUTION')


def test_download_month_passes_the_right_forecast_type():
    fake = _fake_compile({('P5MIN', 'REGIONSOLUTION'): raw_p5min()})
    df = download_month(pd.Period('2026-02', freq='M'), 'p5min',
                        compile_data=fake)
    assert len(df) > 0
    assert fake.calls[0]['forecast_type'] == 'P5MIN'
    assert fake.calls[0]['tables'] == ['REGIONSOLUTION']


def test_download_month_rejects_unknown_table():
    with pytest.raises(ValueError, match="unknown table"):
        download_month(pd.Period('2026-02', freq='M'), 'pdpasa',
                       compile_data=_fake_compile({}))


def test_acquire_builds_partitions(tmp_path):
    fake = _fake_compile({('P5MIN', 'REGIONSOLUTION'): raw_p5min(n_runs=24)})
    res = acquire('2026-02', '2026-02', tables=('p5min',),
                  regions=('SA1', 'NSW1'), raw_cache=tmp_path / 'raw',
                  store_root=tmp_path / 'store', clean=True, compile_data=fake)
    assert all(r.ok for r in res)
    assert res[0].n_clean_rows == 24 * 12 * 2
    for region in ('SA1', 'NSW1'):
        assert partition_path(tmp_path / 'store', 'p5min', region,
                              pd.Period('2026-02', freq='M')).exists()


def test_acquire_filters_to_requested_regions(tmp_path):
    fake = _fake_compile({('P5MIN', 'REGIONSOLUTION'): raw_p5min(n_runs=6)})
    res = acquire('2026-02', '2026-02', tables=('p5min',), regions=('SA1',),
                  raw_cache=tmp_path / 'raw', store_root=tmp_path / 'store',
                  clean=True, compile_data=fake)
    assert res[0].n_clean_rows == 6 * 12
    assert not (tmp_path / 'store' / 'p5min' / 'NSW1').exists()


def test_acquire_is_idempotent(tmp_path):
    """Extending the store by one month must cost one month."""
    fake = _fake_compile({('P5MIN', 'REGIONSOLUTION'): raw_p5min(n_runs=6)})
    kw = dict(tables=('p5min',), regions=('SA1',), raw_cache=tmp_path / 'raw',
              store_root=tmp_path / 'store', clean=True, compile_data=fake)
    acquire('2026-02', '2026-02', **kw)
    n_after_first = len(fake.calls)
    second = acquire('2026-02', '2026-02', **kw)
    assert second[0].skipped
    assert len(fake.calls) == n_after_first, "must not re-download"


def test_acquire_force_rebuilds(tmp_path):
    fake = _fake_compile({('P5MIN', 'REGIONSOLUTION'): raw_p5min(n_runs=6)})
    kw = dict(tables=('p5min',), regions=('SA1',), raw_cache=tmp_path / 'raw',
              store_root=tmp_path / 'store', clean=True, compile_data=fake)
    acquire('2026-02', '2026-02', **kw)
    res = acquire('2026-02', '2026-02', force=True, **kw)
    assert not res[0].skipped and res[0].ok


def test_acquire_fails_soft_on_a_missing_month(tmp_path):
    """The October 2022 case: one unavailable month must not lose the rest."""
    fake = _fake_compile({('P5MIN', 'REGIONSOLUTION'): raw_p5min(n_runs=6)})
    res = acquire('2026-02', '2026-02', tables=('p5min', 'predispatch'),
                  regions=('SA1',), raw_cache=tmp_path / 'raw',
                  store_root=tmp_path / 'store', compile_data=fake)
    ok = [r for r in res if r.ok]
    bad = [r for r in res if not r.ok]
    assert len(ok) == 1 and ok[0].table == 'p5min'
    assert len(bad) == 1 and bad[0].table == 'predispatch'
    assert 'No files' in bad[0].error


def test_acquire_no_clean_leaves_the_store_empty(tmp_path):
    fake = _fake_compile({('P5MIN', 'REGIONSOLUTION'): raw_p5min(n_runs=6)})
    res = acquire('2026-02', '2026-02', tables=('p5min',), regions=('SA1',),
                  raw_cache=tmp_path / 'raw', store_root=tmp_path / 'store',
                  clean=False, compile_data=fake)
    assert res[0].ok and res[0].n_raw_rows > 0 and res[0].n_clean_rows == 0
    assert not (tmp_path / 'store').exists()


def test_acquire_reports_intervention_rows_dropped(tmp_path):
    fake = _fake_compile({('P5MIN', 'REGIONSOLUTION'):
                          raw_p5min(n_runs=6, intervention=True)})
    res = acquire('2026-02', '2026-02', tables=('p5min',), regions=('SA1',),
                  raw_cache=tmp_path / 'raw', store_root=tmp_path / 'store',
                  clean=True, compile_data=fake)
    assert res[0].n_intervention_dropped == 6 * 12


# ------------------------------- stage C: calendar-month completeness

def _full_month_p5min(period='2026-02', regions=('SA1',)):
    """A complete calendar month of P5MIN runs at 5-minute cadence."""
    p = pd.Period(period, freq='M')
    runs = pd.date_range(p.to_timestamp(), (p + 1).to_timestamp(),
                         freq='5min', inclusive='left')
    rows = []
    for run in runs:
        for region in regions:
            for lead in range(12):
                rows.append({'RUN_DATETIME': run,
                             'INTERVAL_DATETIME': run + pd.Timedelta(minutes=5 * lead),
                             'REGIONID': region, 'RRP': 50.0, 'INTERVENTION': 0})
    return pd.DataFrame(rows)


def test_complete_month_reports_full_expected_count(tmp_path):
    write_partitions(clean_forecast_frame(_full_month_p5min(), 'p5min'),
                     'p5min', tmp_path)
    rep = validate_partition(tmp_path, 'p5min', 'SA1', pd.Period('2026-02', freq='M'))
    assert rep.expected_runs == 28 * 288 == rep.n_runs
    assert rep.ok and not rep.partial


def test_truncated_month_is_flagged_partial_not_complete(tmp_path):
    """The failure the first run exposed: September was one P5MIN run and nine
    PREDISPATCH runs short, and the old check reported 8639/8639 OK because it
    derived the grid from the runs it happened to see."""
    raw = _full_month_p5min()
    first = raw['RUN_DATETIME'].min()
    raw = raw[raw['RUN_DATETIME'] != first]          # drop the 00:00 run
    write_partitions(clean_forecast_frame(raw, 'p5min'), 'p5min', tmp_path)
    rep = validate_partition(tmp_path, 'p5min', 'SA1', pd.Period('2026-02', freq='M'))
    assert rep.expected_runs == 28 * 288
    assert rep.n_runs == 28 * 288 - 1
    assert rep.partial and rep.n_missing_leading == 1
    assert rep.ok, "edge truncation is visible, not a failure"


def test_internal_gap_fails_while_edge_truncation_does_not(tmp_path):
    raw = _full_month_p5min()
    runs = sorted(raw['RUN_DATETIME'].unique())
    raw = raw[~raw['RUN_DATETIME'].isin([runs[0], runs[500], runs[-1]])]
    write_partitions(clean_forecast_frame(raw, 'p5min'), 'p5min', tmp_path)
    rep = validate_partition(tmp_path, 'p5min', 'SA1', pd.Period('2026-02', freq='M'))
    assert rep.n_missing_leading == 1 and rep.n_missing_trailing == 1
    assert rep.n_missing_internal == 1
    assert not rep.ok, "an internal gap is a defect"
    assert len(rep.missing_runs) == 1


def test_one_day_partition_does_not_look_complete(tmp_path):
    """A partition holding one day of a month must not pass."""
    raw = _full_month_p5min()
    day = raw['RUN_DATETIME'].dt.date == pd.Timestamp('2026-02-15').date()
    write_partitions(clean_forecast_frame(raw[day], 'p5min'), 'p5min', tmp_path)
    rep = validate_partition(tmp_path, 'p5min', 'SA1', pd.Period('2026-02', freq='M'))
    assert rep.n_runs == 288 and rep.expected_runs == 28 * 288
    assert rep.partial


def test_manifest_records_partial_months(tmp_path):
    raw = _full_month_p5min()
    raw = raw[raw['RUN_DATETIME'] != raw['RUN_DATETIME'].min()]
    write_partitions(clean_forecast_frame(raw, 'p5min'), 'p5min', tmp_path)
    reports = validate_store(tmp_path, tables=('p5min',), regions=('SA1',),
                             verbose=False)
    entry = build_manifest(reports, tmp_path)['entries'][0]
    assert entry['partial'] is True
    assert entry['missing_runs_leading'] == 1
    assert entry['missing_runs_internal'] == 0


# ------------------- stage B: the month-boundary read (regression)

from pipeline.pre_dispatch.B_Clean_Predispatch import (  # noqa: E402
    RAW_TOKEN, clean_range, raw_cache_path, raw_cache_path as _rcp,
)


def test_raw_cache_filenames_match_nemweb_direct():
    """nemweb_direct prefixes PREDISPATCH tables rather than joining with an
    underscore. Getting this wrong means stage B silently finds no cache."""
    assert RAW_TOKEN['p5min'] == 'P5MIN_REGIONSOLUTION'
    assert RAW_TOKEN['predispatch'] == 'PREDISPATCHPRICE'
    p = raw_cache_path(Path('/raw'), 'predispatch', pd.Period('2026-02', freq='M'))
    assert p.name == 'PREDISPATCHPRICE_202602.parquet'


def test_clean_range_skips_when_the_months_own_cache_is_absent(tmp_path):
    out = clean_range('2026-02', '2026-02', tables=('p5min',),
                      regions=('SA1',), raw_cache=tmp_path / 'raw',
                      store_root=tmp_path / 'store', verbose=False)
    assert 'raw cache missing' in out[0]['error']


def _source_file_p5min(period, out_dir, regions=('SA1',)):
    """A P5MIN SOURCE file as AEMO partitions it: M-01 00:05 to M+1-01 00:00.

    Note the offset. The 00:00 run of month M lives in month M-1's file, which
    is the whole point of this fixture.
    """
    p = pd.Period(period, freq='M')
    runs = pd.date_range(p.to_timestamp() + pd.Timedelta(minutes=5),
                         (p + 1).to_timestamp(), freq='5min')
    rows = []
    for run in runs:
        for region in regions:
            for lead in range(12):
                rows.append({'RUN_DATETIME': run,
                             'INTERVAL_DATETIME': run + pd.Timedelta(minutes=5 * lead),
                             'REGIONID': region, 'RRP': 50.0, 'INTERVENTION': 0})
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(_rcp(out_dir, 'p5min', p), index=False)


def test_clean_range_reads_the_previous_months_source_file(tmp_path):
    """Regression. Reading only month M's source file loses the 00:00 run for
    P5MIN and the 00:00-to-04:00 block for PREDISPATCH, because MMSDM
    partitions on a shifted trading-day clock. Observed on the first real
    acquisition: September was 1 and 9 runs short and looked complete."""
    raw = tmp_path / 'raw'
    _source_file_p5min('2026-01', raw)
    _source_file_p5min('2026-02', raw)
    clean_range('2026-02', '2026-02', tables=('p5min',), regions=('SA1',),
                raw_cache=raw, store_root=tmp_path / 'store', verbose=False)
    rep = validate_partition(tmp_path / 'store', 'p5min', 'SA1',
                             pd.Period('2026-02', freq='M'))
    assert rep.n_runs == rep.expected_runs == 28 * 288
    assert not rep.partial, "the 00:00 run must be recovered from January's file"


def test_clean_range_without_the_previous_file_is_partial_not_wrong(tmp_path):
    """When the previous month genuinely was not acquired, the result is
    short. That must be VISIBLE rather than silently complete."""
    raw = tmp_path / 'raw'
    _source_file_p5min('2026-02', raw)
    clean_range('2026-02', '2026-02', tables=('p5min',), regions=('SA1',),
                raw_cache=raw, store_root=tmp_path / 'store', verbose=False)
    rep = validate_partition(tmp_path / 'store', 'p5min', 'SA1',
                             pd.Period('2026-02', freq='M'))
    assert rep.n_runs == 28 * 288 - 1
    assert rep.partial and rep.n_missing_leading == 1


def test_clean_range_does_not_rewrite_the_previous_month(tmp_path):
    """Reading January's file must not leave a partial January partition."""
    raw = tmp_path / 'raw'
    _source_file_p5min('2026-01', raw)
    _source_file_p5min('2026-02', raw)
    clean_range('2026-02', '2026-02', tables=('p5min',), regions=('SA1',),
                raw_cache=raw, store_root=tmp_path / 'store', verbose=False)
    assert not partition_path(tmp_path / 'store', 'p5min', 'SA1',
                              pd.Period('2026-01', freq='M')).exists()


# ------------------- known archive gaps (October 2022 shape)

from pipeline.pre_dispatch.C_Validate_Predispatch import (  # noqa: E402
    KNOWN_ARCHIVE_GAPS,
)


def _source_file_predispatch(period, out_dir, regions=('SA1',)):
    """A PREDISPATCH SOURCE file as AEMO partitions it: M-01 04:30 to
    M+1-01 04:00, so the next month's 00:00 to 04:00 runs live here."""
    p = pd.Period(period, freq='M')
    runs = pd.date_range(p.to_timestamp() + pd.Timedelta(hours=4, minutes=30),
                         (p + 1).to_timestamp() + pd.Timedelta(hours=4),
                         freq='30min')
    rows = []
    for run in runs:
        n = int((predispatch_horizon_end(run) - run).total_seconds() // 1800) + 1
        for region in regions:
            for k in range(n):
                rows.append({'PREDISPATCH_RUN_DATETIME': run,
                             'DATETIME': run + pd.Timedelta(minutes=30 * k),
                             'REGIONID': region, 'RRP': 80.0 + k,
                             'INTERVENTION': 0})
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(_rcp(out_dir, 'predispatch', p), index=False)


def test_clean_range_writes_the_leading_runs_of_a_known_gap(tmp_path,
                                                            monkeypatch):
    """Regression. A known-gap month used to be skipped outright, discarding
    the nine runs the previous month's source file holds for it. With no
    partition the month dropped out of the manifest, and with it the
    known_archive_gap flag the forecast provider's on_degraded_period policy
    keys on."""
    monkeypatch.setitem(KNOWN_ARCHIVE_GAPS, ('predispatch', '2026-02'),
                        'synthetic gap for this test')
    raw = tmp_path / 'raw'
    _source_file_predispatch('2026-01', raw)
    out = clean_range('2026-02', '2026-02', tables=('predispatch',),
                      regions=('SA1',), raw_cache=raw,
                      store_root=tmp_path / 'store', verbose=False)
    assert out[0]['known_gap'] and not out[0]['error']
    rep = validate_partition(tmp_path / 'store', 'predispatch', 'SA1',
                             pd.Period('2026-02', freq='M'))
    assert rep.n_runs == 9                       # 00:00 to 04:00 on the 1st
    assert rep.known_gap and rep.ok and not rep.wrong_target_count
    entry = build_manifest([rep], tmp_path / 'store')['entries'][0]
    assert entry['known_archive_gap'] is True


def test_clean_range_known_gap_without_previous_file_is_not_an_error(
        tmp_path, monkeypatch):
    monkeypatch.setitem(KNOWN_ARCHIVE_GAPS, ('predispatch', '2026-02'),
                        'synthetic gap for this test')
    out = clean_range('2026-02', '2026-02', tables=('predispatch',),
                      regions=('SA1',), raw_cache=tmp_path / 'raw',
                      store_root=tmp_path / 'store', verbose=False)
    assert out[0]['known_gap'] and not out[0]['error']
    assert not partition_path(tmp_path / 'store', 'predispatch', 'SA1',
                              pd.Period('2026-02', freq='M')).exists()


def test_manifest_lists_a_known_gap_with_no_partition(tmp_path, monkeypatch):
    """A known gap inside the store's span is reported even when nothing is on
    disk for it; one outside the span is not."""
    monkeypatch.setitem(KNOWN_ARCHIVE_GAPS, ('predispatch', '2026-02'),
                        'synthetic gap inside the span')
    monkeypatch.setitem(KNOWN_ARCHIVE_GAPS, ('predispatch', '2025-06'),
                        'synthetic gap outside the span')
    for start in ('2026-01-10 13:00', '2026-03-10 13:00'):
        write_partitions(clean_forecast_frame(raw_predispatch(start=start),
                                              'predispatch'),
                         'predispatch', tmp_path)
    reports = validate_store(tmp_path, tables=('predispatch',),
                             regions=('SA1',), verbose=False)
    assert [r.period for r in reports] == ['2026-01', '2026-02', '2026-03']
    gap = reports[1]
    assert gap.known_gap and gap.ok and gap.n_runs == 0
    entry = build_manifest(reports, tmp_path)['entries'][1]
    assert entry['known_archive_gap'] is True and entry['ok'] is True
