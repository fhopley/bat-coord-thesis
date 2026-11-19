"""Run rolling_optimisation on ONE real Energy Masters site and check it.

The gate before any sweep. A sweep across horizons, battery sizes and sites is
only worth running once a single run on a single site is known to produce
something sensible next to actual, self-consumption and perfect foresight.
If it does not, a sweep multiplies the same problem forty times and takes
hours doing it.

Defaults to TruncatedPerfectForecastProvider, which needs NO pre-dispatch
store: real prices restricted to the pre-dispatch availability structure
(sawtooth horizon, fine and coarse split, infill beyond). Forecast error is
exactly zero, so anything that looks wrong is the component or the plumbing.

Pass --forecast predispatch for real AEMO forecasts. Running both on the same
site and differencing the VCRs is the measurement this whole chain exists for:
everything above rung 2 is what forecast error costs.

Usage (from project root):
    python3 -m energy_masters_pipeline.scripts.check_rolling_optimisation EM-427
    python3 -m energy_masters_pipeline.scripts.check_rolling_optimisation EM-427 \\
        --forecast predispatch
    python3 -m energy_masters_pipeline.scripts.check_rolling_optimisation EM-427 \\
        --forecast predispatch --mode next_forecast --horizon 12

Exits 0 if every check passes, 1 otherwise.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model.components.billing import compute_bill
from model.components.rolling_optimisation import rolling_optimisation
from model.core.prices import build_price_stack_offsets
from model.core.scenario import RollingConfig
from model.forecast.registry import (
    FORECAST_SOURCES, build_provider, is_reportable, region_for_state,
)
from energy_masters_pipeline.scripts.rolling_compare import (
    RollingComparison, build_daily, dispatch_metrics, format_report, run_checks,
)

DEFAULT_HORIZON_HOURS = 18.0        # set by the five-site horizon sweep
DEFAULT_FORECAST = 'truncated_perfect'
DEFAULT_STORE = Path('data/clean/predispatch')


def check_site(
    site_id: str,
    horizon_hours: float = DEFAULT_HORIZON_HOURS,
    forecast_source: str = DEFAULT_FORECAST,
    store_root: Path = DEFAULT_STORE,
    days: int | None = None,
    c_deg_per_kwh_override: float | None = None,
    curtail_cost_per_kwh_dollars=None,
    mode: str = 'forecast',
    on_degraded_period: str = 'fail',
    quiet: bool = False,
) -> RollingComparison:
    """Load one EM site, run rolling optimisation, and compare four dispatches.

    Args:
        site_id: Energy Masters site, e.g. 'EM-427'.
        horizon_hours: Rolling planning horizon.
        forecast_source: One of FORECAST_SOURCES.
            'truncated_perfect' is rung 2, zero forecast error with realistic
            availability, and answers "does the rolling machinery work".
            'predispatch' is rung 3, real AEMO forecasts, and answers "what
            does forecast error actually cost". The difference between the two
            on the same site is the measurement this whole chain exists for.
        store_root: Canonical pre-dispatch store. Used by 'predispatch' only.
        days: Restrict to the first N days of the site's valid period. Useful
            for a fast first look; None uses the whole period.
        c_deg_per_kwh_override: Passed through to load_and_analyze so the
            rolling run and the perfect foresight benchmark share it.
        curtail_cost_per_kwh_dollars: As the EM default when omitted.
    """
    from energy_masters_pipeline.scripts.viz_energy_masters import (
        load_and_analyze, DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS,
    )
    if curtail_cost_per_kwh_dollars is None:
        curtail_cost_per_kwh_dollars = DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS

    if not quiet:
        print(f"Loading {site_id} (self-consumption + perfect foresight) ...",
              flush=True)
    a = load_and_analyze(
        site_id, simulate_lp=True,
        curtail_cost_per_kwh_dollars=curtail_cost_per_kwh_dollars,
        c_deg_per_kwh_override=c_deg_per_kwh_override,
    )
    if a.prices_mwh is None:
        raise RuntimeError(
            "EMAnalysis.prices_mwh is None. This script needs the version of "
            "viz_energy_masters.py that populates it from prices_mwh_for_bill."
        )
    if a.lp_result is None or a.sc_result is None:
        raise RuntimeError(f"{site_id}: load_and_analyze returned no LP or SC "
                           "result; rerun with simulate_lp=True")

    index = a.actual_result.import_kwh.index
    if days is not None:
        cutoff = index[0] + pd.Timedelta(days=days)
        keep = index < cutoff
        if keep.sum() < 288:
            raise ValueError(f"--days {days} leaves under one day of data")
        raise NotImplementedError(
            "Restricting to the first N days would require re-running the SC "
            "and PF baselines over the same slice, since a shorter period has "
            "a different terminal SOC and therefore a different benchmark. "
            "Run the whole period, or add slicing to load_and_analyze."
        )

    # Use the EXACT inputs the SC and PF baselines were given, not the raw
    # trace: load_and_analyze clips negative gross_load before simulating but
    # leaves a.trace unclipped, so taking the trace would compare rolling
    # against benchmarks that solved a marginally different problem.
    from energy_masters_pipeline.scripts.sweep_horizon import simulation_inputs
    solar_s, load_s = simulation_inputs(a)
    solar = solar_s.to_numpy(dtype=float)
    load = load_s.to_numpy(dtype=float)

    offsets = build_price_stack_offsets(
        index, a.network_tariff, a.retail_tariff, a.state, a.interval_hours)

    if not quiet:
        print(f"Building forecast provider ({forecast_source}) ...", flush=True)
    provider = build_provider(
        forecast_source, a.prices_mwh,
        interval_minutes=int(round(a.interval_hours * 60)),
        mode=mode, store_root=store_root,
        region=region_for_state(a.state),
        on_degraded_period=on_degraded_period,
    )
    period_hours = len(index) * a.interval_hours
    if not is_reportable(forecast_source, horizon_hours, period_hours) and not quiet:
        print(f"  NOTE: '{forecast_source}' reads actuals at and beyond the "
              "decision instant.\n  It is the correctness gate, not a "
              "reportable result.", flush=True)

    export_limit = (np.inf if a.export_limit_kw is None
                    else a.export_limit_kw * a.interval_hours)
    # The component does not read current_interval_price (mode selection lives
    # in the provider), but the config is what a caller reads back to find out
    # what was run, so it must not disagree with the provider.
    cfg = RollingConfig(forecast_source=forecast_source,
                        horizon_hours=horizon_hours,
                        current_interval_price=mode)

    if not quiet:
        print(f"Rolling optimisation, horizon {horizon_hours:g} h, "
              f"{len(index):,} intervals ...", flush=True)
    t0 = time.perf_counter()
    rolling_traces, diagnostics = rolling_optimisation(
        site_id, solar_s, load_s, provider, offsets, a.battery_spec,
        a.soc_init_kwh, a.soc_final_kwh, cfg,
        export_limit_kwh_per_interval=export_limit,
        interval_hours=a.interval_hours,
        curtail_cost_per_kwh_dollars=a.curtail_cost_per_kwh_dollars,
    )
    wall = time.perf_counter() - t0
    if not quiet:
        print(f"  done in {wall:.1f} s\n", flush=True)

    bill = lambda t: compute_bill(t, a.prices_mwh, a.network_tariff,
                                  a.retail_tariff, a.state, a.interval_hours)
    rolling_bill = bill(rolling_traces)
    cap = a.battery_spec.capacity_kwh
    mk = lambda lab, t, b: dispatch_metrics(lab, t, a.p_imp, a.p_exp, cap, b)

    sc_m = mk('self-consumption', a.sc_result, a.sc_bill)
    pf_m = mk('perfect foresight', a.lp_result, a.lp_bill)
    rl_m = mk(f'rolling {horizon_hours:g}h', rolling_traces, rolling_bill)
    ac_m = mk('actual (Amber)', a.actual_result, a.actual_bill)

    soc_min = (a.battery_spec.backup_reserve_pct / 100.0) * cap
    return RollingComparison(
        site_id=site_id,
        horizon_hours=horizon_hours,
        coarse_resolution=forecast_source,
        period_start=index[0], period_end=index[-1],
        sc=sc_m, pf=pf_m, rolling=rl_m, actual=ac_m,
        daily=build_daily(a.sc_result, a.lp_result, rolling_traces,
                          a.actual_result, a.p_imp, a.p_exp),
        checks=run_checks(sc_m, pf_m, rl_m, rolling_traces, solar, load,
                          diagnostics, cap, soc_min, a.soc_final_kwh,
                          len(index)),
    ), diagnostics


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('site_id')
    parser.add_argument('--horizon', type=float, default=DEFAULT_HORIZON_HOURS,
                        help=f'Planning horizon in hours (default {DEFAULT_HORIZON_HOURS:g})')
    parser.add_argument('--forecast', default=DEFAULT_FORECAST,
                        choices=FORECAST_SOURCES,
                        help="'truncated_perfect' is rung 2 (zero forecast "
                             "error); 'predispatch' is rung 3 (real AEMO "
                             "forecasts). Their difference is what forecast "
                             "error costs.")
    parser.add_argument('--store', type=Path, default=DEFAULT_STORE,
                        help='Canonical pre-dispatch store, for --forecast '
                             'predispatch')
    parser.add_argument('--mode', default='forecast',
                        choices=('forecast', 'price_known', 'price_and_next',
                                 'next_forecast'),
                        help="Current-interval price mode. 'forecast' is the "
                             "run available at the interval start; "
                             "'price_known' adds the settled price published "
                             "~9 s in; 'price_and_next' also refreshes the "
                             "forward window from the run arriving ~39 s in. "
                             "'next_forecast' is decomposition only.")
    parser.add_argument('--on-degraded-period', default='fail',
                        choices=('fail', 'self_consumption', 'actuals'))
    parser.add_argument('--c-deg-per-kwh', type=float, default=None,
                        help='Override the LP degradation cost for BOTH the '
                             'rolling run and the perfect foresight benchmark')
    parser.add_argument('--top-days', type=int, default=8)
    parser.add_argument('--csv', type=Path, default=None,
                        help='Write the per-day table to this path')
    args = parser.parse_args(argv)

    cmp_, diagnostics = check_site(
        args.site_id, horizon_hours=args.horizon,
        forecast_source=args.forecast, store_root=args.store,
        mode=args.mode, on_degraded_period=args.on_degraded_period,
        c_deg_per_kwh_override=args.c_deg_per_kwh,
    )
    print(format_report(cmp_, diagnostics, n_days=args.top_days))

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        cmp_.daily.to_csv(args.csv)
        print(f"\nPer-day table written to {args.csv.resolve()}")

    return 0 if cmp_.passed else 1


if __name__ == '__main__':
    sys.exit(main())
