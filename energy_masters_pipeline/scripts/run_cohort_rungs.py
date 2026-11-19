"""Run the full measurement ladder across the Energy Masters cohort.

Produces two CSVs for charting rather than terminal output, because the
interesting structure is per-site and per-day and does not survive being read
one site at a time.

Six dispatches per site, forming a ladder in which each step changes exactly
one thing. See RUNGS below for the sequence and what each step adds.

Levels, not deltas
------------------
Results are reported as VCR levels side by side. Differences are named only
for the four adjacent pairs that differ in a single respect (CLEAN_CONTRASTS);
no attempt is made to decompose the gap to `actual` into shares that sum to a
total, because Amber's better price forecast, worse site forecast and own risk
appetite push in opposite directions and cannot be separated from these runs.

Every column is computed on the SAME sites, days, batteries and c_deg, so
capacity, headroom per kWh and spike-day luck set the LEVEL of the whole chart
identically and do not reorder it. That is what makes the spacing between
columns meaningful even though the level is specific to these sites and this
window.

Usage (from project root):
    python3 -m energy_masters_pipeline.scripts.run_cohort_rungs
    python3 -m energy_masters_pipeline.scripts.run_cohort_rungs --sites EM-427,EM-414
    python3 -m energy_masters_pipeline.scripts.run_cohort_rungs --horizon 18 \\
        --rungs truncated_perfect,predispatch_next
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from model.components.billing import compute_bill
from model.components.rolling_optimisation import rolling_optimisation
from model.core.prices import build_price_stack_offsets
from model.core.scenario import (
    RollingConfig, SolarForecastConfig, UsageForecastConfig,
)
from model.dev.progress import ConsoleProgress
from model.forecast.usage import UsageForecastProvider
from model.forecast.solar import SolarForecastProvider
from model.forecast.registry import (
    build_provider, is_reportable, region_for_state,
)
from energy_masters_pipeline.scripts.rolling_compare import (
    build_daily, dispatch_metrics,
)
from energy_masters_pipeline.scripts.sweep_horizon import simulation_inputs

DEFAULT_HORIZON_HOURS = 18.0
DEFAULT_STORE = Path('data/clean/predispatch')
DEFAULT_OUT_DIR = Path('data/processed/energy_masters')
DEFAULT_SITE_CSV = DEFAULT_OUT_DIR / 'cohort_rungs_site.csv'
DEFAULT_DAILY_CSV = DEFAULT_OUT_DIR / 'cohort_rungs_daily.csv'

# Below this a site's VCR is a ratio of two small numbers and says nothing
# about coordination quality. Same threshold as run_cohort_attribution.
SITE_HEADROOM_FLOOR_DOLLARS = 5.0

# The ladder, in chart order. Each column differs from the one before it in
# EXACTLY ONE respect, which is what lets the chart carry the reasoning without
# a caption claiming what each gap means:
#
#   perfect_foresight   whole period, unlimited horizon
#   perfect_18h         + finite window, re-optimised every interval
#   shaped_18h          + 30-min coarse beyond the hour, and infill beyond ~15 h
#   predispatch_full    + the FORWARD window is forecast (~39 s into interval k)
#   predispatch_price   + the forward window is not refreshed (~9 s)
#   predispatch_before  + the current price is not yet known (0 s)
#   actual              SEVERAL things at once, and not attributable
#
# The last three are the three moments at which a real controller's
# information changes during a 5-minute interval, measured rather than assumed:
#
#     0 s     the run labelled index[k-1], forecasting interval k 5 min ahead
#     ~9 s    + the settled dispatch price for k
#     ~39 s   + the run labelled index[k], refreshing the forward window
#
# The settled price arriving BEFORE the forecast of the same interval is the
# counter-intuitive part. P5MIN does publish its own estimate for an interval
# whose price is already set, because pre-dispatch makes approximations to run
# faster and is not the same algorithm as dispatch, but no controller would use
# it in preference to the published price.
#
# `predispatch_full` is the most realistic column, not a bound. AEMO writes the
# dispatch price for an interval about 4.85 minutes BEFORE that interval ends,
# i.e. roughly 9 seconds after it begins. Measured over 8,928 January 2026
# intervals: median -4.85 min, p1 -4.93, p99 -4.65.
#
# THE LOAD DIMENSION
#
# Every price rung assumes household load is known perfectly over the horizon.
# The two `_lf` rungs relax that, forecasting load from the site's own trailing
# history instead (see usage_forecast_spec.md). Only two of the five price
# rungs get a twin, which makes a 2x2 rather than a longer ladder:
#
#                          perfect load        forecast load
#     actual prices        shaped_18h          shaped_18h_lf
#     pre-dispatch         predispatch_full    predispatch_full_lf
#
#     load error alone  = shaped_18h        - shaped_18h_lf
#     price error alone = shaped_18h        - predispatch_full
#     both              = shaped_18h        - predispatch_full_lf
#     interaction       = both - load - price
#
# Anchored on shaped_18h, not perfect_18h, so the price leg is exactly the
# CLEAN_CONTRASTS pair 'forward prices are forecasts' and nothing else.
# Anchoring on perfect_18h would fold coarse resolution and infill into the
# price leg as well, weighing three causes against the load leg's one.
#
# Expect the interaction to be positive rather than zero: both errors bite
# hardest on the same hot, high-price days, and a controller that mis-forecasts
# the evening peak AND mis-sizes household load walks into the spike twice.
#
# The load forecast is applied ONLY inside rolling_optimisation. The
# self_consumption, perfect_foresight and actual columns are produced upstream
# by load_and_analyze and never see it.
#
# THE SOLAR DIMENSION
#
# The `_sf` rungs add a solar forecast on top of the load one, forecasting
# generation from the site's own history (see solar_forecast_spec.md). Only
# the `index` mode is run. Its cheaper `persistence` sibling exists in the
# provider but earns no rung here: every history-only method is naive relative
# to an operator forecasting cloud movement from satellite imagery, so `index`
# already gives a bound, and a second looser one would add a column without
# adding an argument.
#
# predispatch_full_lf_sf is the deliverable: a controller built entirely from
# information available before each decision instant, and therefore a LOWER
# BOUND on what a wholesale-exposed plan can achieve, whatever any retailer
# actually achieves.
#
# (key, forecast_source, mode, load, solar, label)
RUNGS = (
    ('perfect_18h', 'perfect', 'forecast', 'perfect', 'perfect',
     'rolling, actual prices'),
    ('shaped_18h', 'truncated_perfect', 'forecast', 'perfect', 'perfect',
     'rolling, actual prices in pre-dispatch shape'),
    ('shaped_18h_lf', 'truncated_perfect', 'forecast', 'forecast', 'perfect',
     'rolling, actual prices in pre-dispatch shape, forecast load'),
    ('shaped_18h_lf_sf', 'truncated_perfect', 'forecast', 'forecast', 'index',
     'rolling, actual prices in pre-dispatch shape, forecast load and solar'),
    ('predispatch_full', 'predispatch', 'price_and_next', 'perfect', 'perfect',
     'pre-dispatch, price known + refreshed forecast'),
    ('predispatch_full_lf', 'predispatch', 'price_and_next', 'forecast', 'perfect',
     'pre-dispatch, price known + refreshed, forecast load'),
    ('predispatch_full_lf_sf', 'predispatch', 'price_and_next', 'forecast', 'index',
     'pre-dispatch, price known + refreshed, forecast load and solar'),
    ('predispatch_price', 'predispatch', 'price_known', 'perfect', 'perfect',
     'pre-dispatch, price known'),
    ('predispatch_before', 'predispatch', 'forecast', 'perfect', 'perfect',
     'pre-dispatch, forecast only'),
)
RUNG_KEYS = tuple(r[0] for r in RUNGS)

# Adjacent pairs that differ in exactly one respect, so their gap has a single
# cause. Everything else in the ladder is reported as a level, not a delta.
CLEAN_CONTRASTS = (
    ('perfect_foresight', 'perfect_18h', 'finite window'),
    ('perfect_18h', 'shaped_18h', 'coarse resolution and infill'),
    ('shaped_18h', 'predispatch_full', 'forward prices are forecasts'),
    ('predispatch_full', 'predispatch_price', 'forward window not refreshed'),
    ('predispatch_price', 'predispatch_before', 'current price not yet known'),
    ('shaped_18h', 'shaped_18h_lf', 'load is forecast, prices actual'),
    ('shaped_18h_lf', 'shaped_18h_lf_sf', 'solar is forecast, prices actual'),
    ('predispatch_full', 'predispatch_full_lf', 'load is forecast'),
    ('predispatch_full_lf', 'predispatch_full_lf_sf', 'solar is forecast'),
)


def run_site(
    site_id: str,
    horizon_hours: float = DEFAULT_HORIZON_HOURS,
    rungs=RUNG_KEYS,
    usage_config: UsageForecastConfig = UsageForecastConfig(),
    solar_config: SolarForecastConfig = SolarForecastConfig(),
    store_root: Path = DEFAULT_STORE,
    c_deg_per_kwh_override: Optional[float] = None,
    curtail_cost_per_kwh_dollars=None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the ladder for one site. Returns (per-dispatch rows, per-day rows)."""
    from energy_masters_pipeline.scripts.viz_energy_masters import (
        load_and_analyze, DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS,
    )
    if curtail_cost_per_kwh_dollars is None:
        curtail_cost_per_kwh_dollars = DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS

    a = load_and_analyze(
        site_id, simulate_lp=True,
        curtail_cost_per_kwh_dollars=curtail_cost_per_kwh_dollars,
        c_deg_per_kwh_override=c_deg_per_kwh_override,
    )
    if a.prices_mwh is None:
        raise RuntimeError("EMAnalysis.prices_mwh is None; update "
                           "viz_energy_masters.py")

    index = a.actual_result.import_kwh.index
    # The EXACT inputs the SC and PF baselines were given. load_and_analyze
    # clips negative gross_load before simulating but leaves a.trace unclipped,
    # so taking the trace would hand rolling a marginally different problem
    # from the one its own benchmarks solved.
    solar_s, load_s = simulation_inputs(a)
    offsets = build_price_stack_offsets(index, a.network_tariff,
                                        a.retail_tariff, a.state,
                                        a.interval_hours)
    export_limit = (np.inf if a.export_limit_kw is None
                    else a.export_limit_kw * a.interval_hours)
    cap = a.battery_spec.capacity_kwh
    bill = lambda t: compute_bill(t, a.prices_mwh, a.network_tariff,
                                  a.retail_tariff, a.state, a.interval_hours)
    mk = lambda lab, t, b: dispatch_metrics(lab, t, a.p_imp, a.p_exp, cap, b)

    sc_m = mk('self_consumption', a.sc_result, a.sc_bill)
    pf_m = mk('perfect_foresight', a.lp_result, a.lp_bill)
    ac_m = mk('actual', a.actual_result, a.actual_bill)
    headroom = sc_m.energy_cost - pf_m.energy_cost

    traces_by_key = {'self_consumption': a.sc_result,
                     'actual': a.actual_result,
                     'perfect_foresight': a.lp_result}
    rows = []

    # Built once per site, not once per rung: it depends only on the load
    # trace, the state and the horizon. Deferred until a `_lf` rung is
    # actually requested so a perfect-load-only run costs nothing.
    #
    # valid=None: Energy Masters traces carry no infill flag, unlike EDP clean
    # files, so every interval enters the profile history. Where EM data has
    # been gap-filled the profile will partly restate that filling.
    n_h = int(round(horizon_hours / a.interval_hours))
    usage_provider = None
    if any(r[3] == 'forecast' for r in RUNGS if r[0] in rungs):
        usage_provider = UsageForecastProvider.from_series(
            load_s, a.state, usage_config, max_horizon_intervals=n_h,
            interval_hours=a.interval_hours)
    # Same deferral and the same valid=None: Energy Masters traces carry no
    # pv infill flag. Unlike load infill that is only a fitting nicety, since
    # EDP's pv infill comes from a state-average profile rather than the
    # site's own history, so there is no circularity to avoid.
    solar_provider = None
    if any(r[4] != 'perfect' for r in RUNGS if r[0] in rungs):
        solar_provider = SolarForecastProvider.from_series(
            solar_s, a.state, solar_config, max_horizon_intervals=n_h,
            interval_hours=a.interval_hours)

    def add(key, label, m, diag=None):
        rows.append(dict(
            site_id=site_id, dispatch=key, label=label,
            horizon_hours=(horizon_hours if diag is not None else np.nan),
            battery_kwh=cap, power_kw=a.battery_spec.power_kw,
            c_deg_per_kwh=a.battery_spec.c_deg_per_kwh,
            n_intervals=len(index),
            period_start=str(index[0].date()), period_end=str(index[-1].date()),
            energy_cost=m.energy_cost, bill_total=m.bill_total,
            vcr=((sc_m.energy_cost - m.energy_cost) / headroom
                 if abs(headroom) > 1e-9 else np.nan),
            headroom=headroom,
            import_kwh=m.import_kwh, export_kwh=m.export_kwh,
            grid_to_bat_kwh=m.grid_to_bat_kwh, bat_to_grid_kwh=m.bat_to_grid_kwh,
            sol_to_bat_kwh=m.sol_to_bat_kwh, curtail_kwh=m.curtail_kwh,
            throughput_kwh=m.throughput_kwh, equivalent_cycles=m.equivalent_cycles,
            terminal_soc_kwh=m.terminal_soc_kwh,
            infill_fraction=(diag.infill_fraction_mean if diag else np.nan),
            # Staleness is what makes a truncated store visible: an infill
            # share near 1.0 with a staleness of weeks means the "forecast" is
            # really 24-hour-lagged actuals carried forward.
            #
            # These two measure the FINE run and read 0.00 on a healthy store,
            # because a 5-minute source publishes at every decision instant.
            # That is correct and is not the whole picture: the coarse columns
            # below carry the age of the run that shapes the forward window.
            staleness_min_mean=(diag.staleness_minutes_mean if diag else np.nan),
            staleness_min_max=(diag.staleness_minutes_max if diag else np.nan),
            coarse_staleness_min_mean=(diag.coarse_staleness_minutes_mean
                                       if diag else np.nan),
            coarse_staleness_min_max=(diag.coarse_staleness_minutes_max
                                      if diag else np.nan),
            n_fallbacks=(diag.n_fallbacks if diag else np.nan),
            # Load forecast. Warm-up matters on this cohort: periods run 61 to
            # 273 days and the profile wants 28 days of history, so on the
            # shortest sites a large share of windows plan on a partial
            # profile. It is reported per site rather than excluded, because
            # excluding it would leave the shortest sites with no result.
            load_forecast_used=(diag.load_forecast_used if diag else False),
            load_warm_up_windows=(diag.load_windows_warm_up if diag else np.nan),
            load_warm_up_frac=((diag.load_windows_warm_up / len(index))
                               if diag and diag.load_forecast_used else np.nan),
            load_sc_fallbacks=(diag.sc_fallback_no_load_forecast
                               if diag else np.nan),
            load_clipped=(diag.load_positions_clipped if diag else np.nan),
            infeasible_from_site=(diag.infeasible_from_site_forecast
                                  if diag else np.nan),
            solar_forecast_used=(diag.solar_forecast_used if diag else False),
            solar_warm_up_windows=(diag.solar_windows_warm_up
                                   if diag else np.nan),
            solar_sc_fallbacks=(diag.sc_fallback_no_solar_forecast
                                if diag else np.nan),
            solar_clipped=(diag.solar_positions_clipped if diag else np.nan),
            solve_ms_median=(diag.solve_ms_median if diag else np.nan),
            causality_ok=(diag.causality_ok if diag else True),
            error='',
        ))

    add('self_consumption', 'self-consumption', sc_m)
    add('actual', 'actual (Amber)', ac_m)

    for key, source, mode, load_mode, solar_mode, label in RUNGS:
        if key not in rungs:
            continue
        use_load_fc = (load_mode == 'forecast')
        use_solar_fc = (solar_mode != 'perfect')
        cfg = RollingConfig(forecast_source=source, horizon_hours=horizon_hours,
                            current_interval_price=mode,
                            usage_forecast=(usage_config if use_load_fc
                                            else None),
                            solar_forecast=(solar_config if use_solar_fc
                                            else None))
        try:
            period_hours = len(index) * a.interval_hours
            if not is_reportable(source, horizon_hours, period_hours):
                raise ValueError(
                    f"'{source}' at horizon {horizon_hours} h over a "
                    f"{period_hours:.0f} h period sees the whole period and "
                    "is the correctness gate, not a reportable result")
            provider = build_provider(
                source, a.prices_mwh,
                interval_minutes=int(round(a.interval_hours * 60)),
                mode=mode, store_root=store_root,
                region=region_for_state(a.state))
            with ConsoleProgress(f"{site_id} {label}", enabled=show_progress) as pr:
                traces, diag = rolling_optimisation(
                    site_id, solar_s, load_s, provider, offsets,
                    a.battery_spec, a.soc_init_kwh, a.soc_final_kwh, cfg,
                    export_limit_kwh_per_interval=export_limit,
                    interval_hours=a.interval_hours,
                    curtail_cost_per_kwh_dollars=a.curtail_cost_per_kwh_dollars,
                    progress=pr,
                    load_forecast_provider=(usage_provider if use_load_fc
                                            else None),
                    solar_forecast_provider=(solar_provider if use_solar_fc
                                             else None),
                    commit_actual_load=cfg.commit_actual_load)
        except Exception as e:                              # noqa: BLE001
            rows.append(dict(site_id=site_id, dispatch=key, label=label,
                             horizon_hours=horizon_hours, battery_kwh=cap,
                             headroom=headroom, vcr=np.nan,
                             error=f"{type(e).__name__}: {e}"))
            print(f"    {site_id} {label} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        traces_by_key[key] = traces
        add(key, label, mk(key, traces, bill(traces)), diag)

    add('perfect_foresight', 'perfect foresight', pf_m)

    # Per-day energy cost for every dispatch present. Value is concentrated:
    # on EM-427 a single day carries 35% of the period headroom, so a
    # cohort chart built only on period totals hides what needs explaining.
    # build_daily needs one 'primary' rolling series; every rung is added
    # explicitly below, so this is just a placeholder column that is dropped.
    daily = build_daily(a.sc_result, a.lp_result,
                        traces_by_key.get('predispatch_full', a.lp_result),
                        a.actual_result, a.p_imp, a.p_exp)
    daily = daily.rename(columns={'rolling': '_primary_rolling',
                                  'vcr_rolling': '_primary_vcr'})
    for key in RUNG_KEYS:
        if key in traces_by_key:
            t = traces_by_key[key]
            cost = (t.import_kwh * a.p_imp - t.export_kwh * a.p_exp).groupby(
                t.import_kwh.index.date).sum()
            daily[key] = cost
            with np.errstate(divide='ignore', invalid='ignore'):
                daily[f'vcr_{key}'] = (daily['sc'] - cost) / daily['headroom']
    daily = daily.drop(columns=['_primary_rolling', '_primary_vcr'])
    daily.insert(0, 'site_id', site_id)
    return pd.DataFrame(rows), daily.reset_index()


def run_cohort(
    sites: Optional[list] = None,
    horizon_hours: float = DEFAULT_HORIZON_HOURS,
    rungs=RUNG_KEYS,
    usage_config: UsageForecastConfig = UsageForecastConfig(),
    solar_config: SolarForecastConfig = SolarForecastConfig(),
    store_root: Path = DEFAULT_STORE,
    site_csv: Optional[Path] = DEFAULT_SITE_CSV,
    daily_csv: Optional[Path] = DEFAULT_DAILY_CSV,
    c_deg_per_kwh_override: Optional[float] = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the ladder across the cohort. Fail-soft per site."""
    from energy_masters_pipeline.scripts.run_cohort_attribution import accepted_sites
    if sites is None:
        sites = accepted_sites()

    print(f"Cohort rung comparison: {len(sites)} site(s), horizon "
          f"{horizon_hours:g} h, rungs {list(rungs)}")
    if any(r[3] == 'forecast' for r in RUNGS if r[0] in rungs):
        u = usage_config
        print(f"  load forecast: history {u.history_days}d, smoothing "
              f"{u.smoothing_intervals}, dow_level {u.dow_level}, "
              f"x {u.lookback_hours:g} h, y {u.decay_hours:g} h")
    if any(r[4] != 'perfect' for r in RUNGS if r[0] in rungs):
        c = solar_config
        print(f"  solar forecast: {c.mode}, history {c.history_days}d, "
              f"quantile {c.envelope_quantile:g}, prev_share "
              f"{c.w_prev_share:g}, today_max {c.w_today_max:g}, f_on "
              f"{c.f_on:g}, x {c.lookback_hours:g} h, y {c.decay_hours:g} h")
    print(f"Sites: {', '.join(sites)}\n")

    site_frames, daily_frames, failures = [], [], []
    t0 = time.perf_counter()
    for i, site_id in enumerate(sites, 1):
        print(f"[{i}/{len(sites)}] {site_id} ...", flush=True)
        try:
            s_rows, d_rows = run_site(
                site_id, horizon_hours=horizon_hours, rungs=rungs,
                usage_config=usage_config, solar_config=solar_config,
                store_root=store_root,
                c_deg_per_kwh_override=c_deg_per_kwh_override,
                show_progress=show_progress)
        except Exception as e:                              # noqa: BLE001
            failures.append((site_id, e))
            print(f"    SITE FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        site_frames.append(s_rows)
        daily_frames.append(d_rows)
        got = s_rows.set_index('dispatch')['vcr']
        print("    " + "  ".join(
            f"{k}={got.get(k, float('nan')):.3f}"
            for k in ('actual',) + tuple(k for k in RUNG_KEYS if k in rungs)),
            flush=True)

    if not site_frames:
        print("\nNo sites completed.")
        return pd.DataFrame(), pd.DataFrame()

    site_df = pd.concat(site_frames, ignore_index=True)
    daily_df = pd.concat(daily_frames, ignore_index=True)
    for path, df in ((site_csv, site_df), (daily_csv, daily_df)):
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            print(f"\nWritten {Path(path).resolve()}  ({len(df):,} rows)")

    print(f"\nTotal wall time {(time.perf_counter() - t0) / 60:.1f} min")
    if failures:
        print(f"\n{len(failures)} site(s) failed:")
        for site_id, e in failures:
            print(f"  {site_id}: {type(e).__name__}: {e}")
    print_cohort_table(site_df)
    return site_df, daily_df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def cohort_vcr(site_df: pd.DataFrame) -> pd.DataFrame:
    """Cohort VCR per dispatch, summed numerator over summed denominator.

    Never a mean of per-site ratios: that silently overweights sites with
    little headroom. Sites below the headroom floor are excluded because their
    VCR is a ratio of two small numbers.
    """
    thick = site_df[(site_df['headroom'] >= SITE_HEADROOM_FLOOR_DOLLARS)
                    & site_df['error'].fillna('').eq('')].copy()
    sc = (thick[thick['dispatch'] == 'self_consumption']
          .set_index('site_id')['energy_cost'])
    thick['num'] = thick['site_id'].map(sc) - thick['energy_cost']
    g = thick.groupby('dispatch', sort=False)
    out = g.agg(cohort_vcr=('num', 'sum'), denom=('headroom', 'sum'),
                n_sites=('site_id', 'nunique'),
                cycles=('equivalent_cycles', 'mean'),
                grid_to_bat=('grid_to_bat_kwh', 'sum'),
                bat_to_grid=('bat_to_grid_kwh', 'sum'),
                infill=('infill_fraction', 'mean')).reset_index()
    out['cohort_vcr'] = out['cohort_vcr'] / out['denom']
    order = ['self_consumption', 'actual'] + list(RUNG_KEYS) + ['perfect_foresight']
    out['_o'] = out['dispatch'].map({d: i for i, d in enumerate(order)})
    return out.sort_values('_o').drop(columns=['_o', 'denom'])


def print_cohort_table(site_df: pd.DataFrame) -> None:
    coh = cohort_vcr(site_df)
    print("\n=== Cohort VCR by dispatch ===")
    print("Summed numerator / summed denominator; sites below "
          f"${SITE_HEADROOM_FLOOR_DOLLARS:.0f} headroom excluded.")
    print(coh.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    v = coh.set_index('dispatch')['cohort_vcr']
    print("\n=== Adjacent contrasts (single-cause differences only) ===")
    for a, b, cause in CLEAN_CONTRASTS:
        if a in v.index and b in v.index:
            print(f"  {v[a]:.4f} -> {v[b]:.4f}   {v[b] - v[a]:+.4f}   {cause}")
    if 'actual' in v.index:
        print(f"\n  actual (Amber) sits at {v['actual']:.4f}. The gap to any "
              "rolling column is NOT attributed:")
        print("  Amber has a better price forecast than pre-dispatch, a worse "
              "site forecast than")
        print("  our perfect solar and load, and its own risk appetite. Those "
              "push in opposite")
        print("  directions and cannot be separated from these runs.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sites', type=str, default=None,
                        help='Comma-separated. Default: all accepted sites.')
    parser.add_argument('--horizon', type=float, default=DEFAULT_HORIZON_HOURS)
    parser.add_argument('--rungs', type=str, default=','.join(RUNG_KEYS),
                        help=f'Subset of {list(RUNG_KEYS)}')
    parser.add_argument('--store', type=Path, default=DEFAULT_STORE)
    parser.add_argument('--c-deg-per-kwh', type=float, default=None)
    parser.add_argument('--site-csv', type=Path, default=DEFAULT_SITE_CSV)
    parser.add_argument('--daily-csv', type=Path, default=DEFAULT_DAILY_CSV)
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args()

    run_cohort(
        sites=[s.strip() for s in args.sites.split(',')] if args.sites else None,
        horizon_hours=args.horizon,
        rungs=tuple(r.strip() for r in args.rungs.split(',')),
        store_root=args.store,
        site_csv=args.site_csv, daily_csv=args.daily_csv,
        c_deg_per_kwh_override=args.c_deg_per_kwh,
        show_progress=not args.no_progress,
    )
