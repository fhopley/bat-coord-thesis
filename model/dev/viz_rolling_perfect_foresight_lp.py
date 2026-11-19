"""Interactive visualisation of rolling_optimisation output for a single real site.

Usage (from project root):
    python3 -m model.dev.viz_rolling_perfect_foresight_lp SITE_ID YEAR
    python3 -m model.dev.viz_rolling_perfect_foresight_lp S0008 2022 --on-degraded-period infill --max-staleness-hours 800
    python3 -m model.dev.viz_rolling_perfect_foresight_lp SITE_ID YEAR --state NSW \\
        --network-tariff NSW_ausgrid_ea025_ea029_2026 \\
        --retail-tariff NSW_amber_wholesale_2026 --export-limit-kw 10 \\
        --horizon 18 --forecast predispatch

Defaults to the "predispatch_full_lf_sf" rung -- forecast_source=predispatch,
current_interval_price=price_and_next (the settled price plus the refreshed
forecast -- "full" means FULL information, the most a real controller has,
NOT the absence of privileged current-interval information; see
RollingConfig's own docstring on current_interval_price), usage_forecast and
solar_forecast both enabled with their fitted defaults -- which is the same
rolling: block full_run_rolling_amber.yaml uses. That matters here
specifically: this script exists to let a person LOOK at what the bulk
orchestrator run is actually doing on one real site before trusting it
across hundreds, so its defaults track the production config rather than a
diagnostic rung of the measurement ladder. Override any of it with
--forecast / --horizon / --mode / --no-load-forecast / --no-solar-forecast.

Sibling to viz_perfect_foresight_lp.py, and deliberately built ON it rather
than beside it: site/price/tariff loading, the flows/prices figures, the
day-by-day interactive loop, and the SiteTraces plotting are all imported
unchanged from there. The two scripts must show identical days for identical
sites given identical inputs, which is only guaranteed if they share the
literal plotting code rather than two copies of it.

The dispatch itself is NOT re-derived here either. build_rolling_price_provider
and run_rolling_perfect_foresight_lp_site are imported directly from
model.orchestrate -- the same functions the bulk orchestrator's per-site
worker calls (see orchestrate._rolling_perfect_foresight_site_worker). This
is deliberate: a single-site diagnostic script that hand-mirrored the SOC-pin
and forecast-provider wiring, rather than calling the shared code, would be
exactly the kind of two-independently-maintained-copies risk that has
already caused drift elsewhere in this project (the B_Clean_Predispatch /
C_Validate_Predispatch known-gaps table being the reason that pair now shares
one source of truth instead of two).

RollingAnalysis subclasses LPAnalysis (viz_perfect_foresight_lp.py) rather
than defining a parallel dataclass, purely so every existing plotting
function -- which reads analysis.lp_result / .sc_result / .lp_bill / .p_imp
etc by attribute -- works unchanged on a rolling result. Field names carry
their original LP-era meaning (lp_result IS the rolling dispatch here, not a
perfect_foresight_lp one); backend is set to 'rolling' rather than a solver
name so that distinction stays visible in the printed verification block and
the two scripts are never confused for one another from their output alone.

Controls: same as viz_perfect_foresight_lp.py (Enter/n, b, YYYY-MM-DD, q).
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from model.core.paths import default_paths, Paths
from model.core.scenario import (
    BatterySpec, RollingConfig, UsageForecastConfig, SolarForecastConfig,
)
from model.core.tariff import NetworkTariff, RetailTariff
from model.core.prices import build_price_vectors, build_price_stack_offsets
from model.components.billing import compute_bill
from model.components.rolling_optimisation import RollingDiagnostics
from model.forecast.registry import region_for_state
from model.dev._dev_common import DEV_BATTERY
from model.orchestrate import (
    build_rolling_price_provider, run_rolling_perfect_foresight_lp_site,
    DEFAULT_PREDISPATCH_STORE,
)

# Reused unchanged from viz_perfect_foresight_lp.py -- see this module's
# docstring for why identical plotting code matters here specifically.
from model.dev.viz_perfect_foresight_lp import (
    LPAnalysis, load_inputs, run_lp_analysis, _recover_curtail,
    _setup_flows_figure, _setup_prices_figure,
    _plot_flows_day, _plot_prices_day, _interactive_loop,
    BATTERY_MODE,
)


@dataclass(frozen=True)
class RollingAnalysis(LPAnalysis):
    """LPAnalysis plus the rolling-specific fields the plotting code never
    touches but the verification printer and CLI need.

    diagnostics: RollingDiagnostics for this run (see rolling_optimisation.py).
    rolling_config: the RollingConfig actually used, so the printed
        verification block states exactly what was run rather than the
        caller having to cross-reference argv.
    """
    diagnostics: Optional[RollingDiagnostics] = None
    rolling_config: Optional[RollingConfig] = None


def run_rolling_analysis(
    site_id: str,
    year: int,
    trace: pd.DataFrame,
    prices_mwh: pd.Series,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    battery_spec: BatterySpec,
    rolling_config: RollingConfig,
    export_limit_kw: float,
    state: str,
    interval_hours: float = 5 / 60,
    store_root: Path = DEFAULT_PREDISPATCH_STORE,
) -> RollingAnalysis:
    """Run self_consumption then rolling_optimisation and bill both.

    Structured to mirror run_lp_analysis's shape exactly (same site/price/
    tariff inputs, same SOC-pin story, same return-a-frozen-dataclass style)
    so the two are easy to read side by side -- but the actual dispatch call
    goes through model.orchestrate.run_rolling_perfect_foresight_lp_site,
    not a local reimplementation. See this module's docstring.

    on_degraded_period and max_staleness_hours are NOT separate parameters
    here -- they are real RollingConfig fields (model/core/scenario.py) and
    are read from rolling_config by build_rolling_price_provider, same as
    forecast_source/horizon_hours/current_interval_price already are. Set
    them on the RollingConfig passed in, via --on-degraded-period /
    --max-staleness-hours in main().
    """
    solar = trace['pv_site_net_kwh']
    load = trace['gross_load_kwh']

    # Customer-facing price vectors, for the price panels and for billing --
    # the same construction viz_perfect_foresight_lp.py uses, and a DIFFERENT
    # representation of the same tariff stack from the affine offsets built
    # below (which rolling_optimisation itself needs). Both are legitimate;
    # they serve different consumers with different mathematical shapes, not
    # two copies of the same thing.
    p_imp, p_exp = build_price_vectors(
        solar.index, prices_mwh, network_tariff, retail_tariff, state,
        interval_hours,
    )
    offsets = build_price_stack_offsets(
        solar.index, network_tariff, retail_tariff, state, interval_hours,
    )
    region = region_for_state(state)
    provider = build_rolling_price_provider(
        rolling_config, prices_mwh, interval_hours,
        region=region, store_root=store_root,
    )

    export_limit_kwh = export_limit_kw * interval_hours
    sc_result, roll_result, diagnostics = run_rolling_perfect_foresight_lp_site(
        site_id, solar, load, battery_spec, offsets, provider, rolling_config,
        export_limit_kwh_per_interval=export_limit_kwh,
        interval_hours=interval_hours, state=state,
    )

    soc_min = (battery_spec.backup_reserve_pct / 100.0) * battery_spec.capacity_kwh
    soc_final = float(sc_result.battery_soc_kwh.iloc[-1])
    soc_final = min(max(soc_final, soc_min), battery_spec.capacity_kwh)

    sc_bill = compute_bill(sc_result, prices_mwh, network_tariff, retail_tariff,
                           state, interval_hours)
    roll_bill = compute_bill(roll_result, prices_mwh, network_tariff, retail_tariff,
                             state, interval_hours)

    curtail = _recover_curtail(solar, roll_result)

    return RollingAnalysis(
        site_id=site_id, year=year, trace=trace,
        sc_result=sc_result, lp_result=roll_result,
        sc_bill=sc_bill, lp_bill=roll_bill,
        p_imp=p_imp, p_exp=p_exp, curtail=curtail,
        battery_spec=battery_spec,
        network_tariff=network_tariff, retail_tariff=retail_tariff,
        export_limit_kw=export_limit_kw,
        state=state,
        soc_init_kwh=soc_min, soc_final_kwh=soc_final,
        interval_hours=interval_hours,
        backend='rolling',
        lp_solve_seconds=diagnostics.solve_seconds_total,
        diagnostics=diagnostics,
        rolling_config=rolling_config,
    )


def _print_verification(analysis: RollingAnalysis,
                        pf_analysis: Optional[LPAnalysis] = None) -> None:
    """Print startup verification. Distinct from viz_perfect_foresight_lp's
    _print_verification rather than reused: that one checks LP-specific
    invariants (a global objective, an LP<=self_consumption bound) that do
    not hold for rolling_optimisation, which has no single objective and can
    fall back to self_consumption interval-by-interval. RollingDiagnostics
    already carries the checks that ARE meaningful for a rolling run.

    pf_analysis: optional perfect_foresight_lp result on the SAME site/year/
    trace/prices/tariffs/battery/export limit (from run_lp_analysis, run by
    the caller when --compare-perfect-foresight is set). Not run by default:
    it's a second LP solve, and most invocations of this script are looking
    at a genuinely imperfect rung (predispatch) where the comparison isn't
    the point. It IS the point for a near-zero-forecast-error rung like
    shaped_18h or perfect_18h (see RUNGS in run_cohort_rungs.py) -- the
    closeness of rolling to perfect_foresight_lp there is a machinery
    correctness check, not a result, mirroring what
    check_rolling_optimisation.py calls "the correctness gate before any
    sweep is worth running".
    """
    a = analysis
    cap = a.battery_spec.capacity_kwh
    min_soc = (a.battery_spec.backup_reserve_pct / 100.0) * cap
    tol = 1e-6

    def status(ok: bool) -> str:
        return 'PASS' if ok else 'FAIL'

    print()
    print(f"=== Verification: site {a.site_id} year {a.year} "
          f"(rolling_perfect_foresight_lp) ===")
    rc = a.rolling_config
    print(f"forecast_source={rc.forecast_source}  horizon={rc.horizon_hours:g}h  "
          f"current_interval_price={rc.current_interval_price}  "
          f"load_forecast={'on' if rc.usage_forecast is not None else 'off'}  "
          f"solar_forecast={'on' if rc.solar_forecast is not None else 'off'}  "
          f"commit_actual_load={rc.commit_actual_load}")
    if rc.forecast_source == 'predispatch':
        print(f"on_degraded_period={rc.on_degraded_period}  "
              f"max_staleness_hours={rc.max_staleness_hours:g}")
    print()
    print(a.diagnostics.summary())
    print()

    # Causality is a hard fail here too, matching
    # orchestrate._run_rolling_perfect_foresight_lp: a causality violation
    # means the result is not a legitimate rolling-optimisation output, so
    # this script should not go on to plot it as though it were one.
    if not a.diagnostics.causality_ok:
        print(f"CAUSALITY: {status(False)} -- see max_run_time_lead_minutes "
              "above. Aborting rather than plotting an invalid dispatch.")
        raise RuntimeError(
            f"{a.site_id}: RollingDiagnostics.causality_ok is False."
        )
    print(f"causality                 {status(True)}")

    soc = a.lp_result.battery_soc_kwh
    bounds_ok = bool(((soc >= min_soc - tol) & (soc <= cap + tol)).all())
    print(f"SOC within [reserve, capacity]  {status(bounds_ok)} "
          f"[{soc.min():.4f}, {soc.max():.4f}] kWh vs [{min_soc:.4f}, {cap:.4f}]")

    terminal_ok = abs(float(soc.iloc[-1]) - a.soc_final_kwh) <= 1e-3
    print(f"terminal SOC matches self_consumption target  {status(terminal_ok)} "
          f"{float(soc.iloc[-1]):.4f} vs target {a.soc_final_kwh:.4f} kWh")

    solar = a.trace['pv_site_net_kwh']
    r = a.lp_result
    balance = solar - (r.sol_to_home_kwh + r.sol_to_bat_kwh + r.sol_to_grid_kwh
                       + a.curtail)
    solar_ok = bool((balance.abs() < 1e-6).all())
    print(f"solar balance (sol_to_home+bat+grid+curtail == solar)  "
          f"{status(solar_ok)}  max |residual| {balance.abs().max():.2e} kWh")

    print()
    print(f"--- rolling solve timing ---")
    print(f"solve time total:   {a.diagnostics.solve_seconds_total:.2f}s  "
          f"({a.diagnostics.solve_ms_median:.2f}ms median, "
          f"{a.diagnostics.solve_ms_p95:.2f}ms p95 per window)")
    print()
    print(f"self_consumption bill:   ${a.sc_bill.household_total:.2f}")
    print(f"rolling bill:            ${a.lp_bill.household_total:.2f}")
    if not (bounds_ok and terminal_ok and solar_ok):
        print("\n*** ONE OR MORE CHECKS FAILED -- see above ***")

    if pf_analysis is not None:
        pf = pf_analysis
        sc_total = a.sc_bill.household_total
        pf_total = pf.lp_bill.household_total
        roll_total = a.lp_bill.household_total
        headroom = sc_total - pf_total
        print()
        print("--- rolling vs perfect_foresight_lp (same site/prices/tariffs/"
              "battery) ---")
        print(f"perfect_foresight_lp bill:  ${pf_total:.2f}")
        print(f"rolling bill:               ${roll_total:.2f}")
        if abs(headroom) < 1e-6:
            print("headroom (self_consumption - perfect_foresight) is ~$0 -- "
                  "VCR is undefined for this site/period, comparing bills "
                  "directly instead.")
            print(f"rolling - perfect_foresight:  ${roll_total - pf_total:+.2f}")
        else:
            vcr = (sc_total - roll_total) / headroom
            print(f"headroom (self_consumption - perfect_foresight): "
                  f"${headroom:.2f}")
            print(f"rolling value-capture ratio vs perfect_foresight: "
                  f"{vcr:.1%}")
            print("For a near-zero-forecast-error rung (shaped_18h, "
                  "perfect_18h) this should be close to 100% -- the gap is "
                  "the rolling machinery's own cost (window/terminal-SOC "
                  "effects), not forecast error. See "
                  "rolling_optimisation_spec.md's measurement ladder for "
                  "what to expect at each rung.")


def _parse_forecast_sub_config(enabled: bool, cls):
    """Enabled -> cls() with fitted defaults; disabled -> None.

    No override surface for individual sub-fields here (unlike the YAML
    parser's usage_forecast: {key: value} form) -- this script is for
    inspecting the production config on one site, not for sweeping
    forecaster hyperparameters. Edit usage_config/solar_config below in
    code if a specific diagnostic run needs non-default values.
    """
    return cls() if enabled else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('site_id', help='Site identifier, e.g. S0008')
    parser.add_argument('year', type=int, help='Year, e.g. 2022')
    parser.add_argument('--state', default='NSW', help='NEM region for prices (default NSW)')
    parser.add_argument('--network-tariff', default='NSW_ausgrid_ea025_ea029_2026')
    parser.add_argument('--retail-tariff', default='NSW_amber_wholesale_2026')
    parser.add_argument('--export-limit-kw', type=float, default=10.0,
                        help='Per-phase export limit kW, 1 phase assumed (default 10)')
    parser.add_argument('--forecast', default='predispatch',
                        help="forecast_source (default 'predispatch', the "
                             "production setting). See "
                             "model.forecast.registry.FORECAST_SOURCES for "
                             "the full list, including diagnostic rungs like "
                             "'truncated_perfect'.")
    parser.add_argument('--horizon', type=float, default=18.0,
                        help='horizon_hours (default 18, per the EDP horizon '
                             'sweep result).')
    parser.add_argument('--mode', default='price_and_next',
                        choices=('forecast', 'price_known', 'price_and_next',
                                'next_forecast'),
                        help="current_interval_price (default "
                             "'price_and_next' -- FULL information, the "
                             "settled price plus the refreshed forecast, "
                             "matching predispatch_full_lf_sf; 'forecast' "
                             "is the LEAST informed mode, not a 'plainer' "
                             "version of full -- see RollingConfig's "
                             "docstring before changing this).")
    parser.add_argument('--infill-lag-hours', type=float, default=24.0)
    parser.add_argument('--no-load-forecast', action='store_true',
                        help='Disable usage_forecast (default: enabled, '
                             'matching predispatch_full_lf_sf).')
    parser.add_argument('--no-solar-forecast', action='store_true',
                        help='Disable solar_forecast (default: enabled, '
                             'matching predispatch_full_lf_sf).')
    parser.add_argument('--no-commit-actual-load', action='store_true',
                        help='Set commit_actual_load=False (default: True, '
                             'the production setting -- see RollingConfig '
                             'docstring for what this changes).')
    parser.add_argument('--store', type=Path, default=DEFAULT_PREDISPATCH_STORE,
                        help='Pre-dispatch store root, for --forecast predispatch.')
    parser.add_argument('--on-degraded-period', default='fail',
                        choices=('fail', 'self_consumption', 'actuals', 'infill'),
                        help="What a --forecast predispatch provider does "
                             "about a period the store marks as having no "
                             "usable forecast lead (October 2022 being the "
                             "known case). 'infill' is recommended for a gap "
                             "shaped like that one: it adds no special "
                             "handling, falling through to the same "
                             "24-hour-lag infill used for any ordinary "
                             "horizon shortfall, so real fine-segment "
                             "coverage (P5MIN, unaffected in October) still "
                             "counts. Default 'fail' -- deliberately "
                             "conscious rather than silently substituting. "
                             "See PredispatchForecastProvider's docstring.")
    parser.add_argument('--max-staleness-hours', type=float, default=6.0,
                        help="Largest gap a --forecast predispatch provider "
                             "may see between consecutive store runs before "
                             "refusing to start (coverage check, not a "
                             "staleness gate on dispatch itself). Default "
                             "6.0. A year spanning October 2022's archive "
                             "gap needs this raised past ~750 as well as "
                             "--on-degraded-period set to 'infill' (or "
                             "another non-'fail' value) -- the coverage "
                             "check runs first and will refuse before "
                             "on_degraded_period is ever consulted.")
    parser.add_argument('--compare-perfect-foresight', action='store_true',
                        help="Also run perfect_foresight_lp on the same "
                             "site/prices/tariffs/battery and print the "
                             "rolling value-capture ratio against it. The "
                             "check this exists for: at a near-zero-"
                             "forecast-error rung (--forecast truncated_perfect "
                             "--mode forecast --no-load-forecast "
                             "--no-solar-forecast, i.e. shaped_18h) rolling "
                             "should land close to 100% -- confirming the "
                             "rolling machinery itself before trusting it "
                             "at forecast_source=predispatch. A second LP "
                             "solve, so off by default.")
    parser.add_argument('--project-root', type=Path, default=None)
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    trace, prices_mwh, network_tariff, retail_tariff = load_inputs(
        args.site_id, args.year, args.state,
        args.network_tariff, args.retail_tariff, paths,
    )
    for col in ('pv_site_net_kwh', 'gross_load_kwh'):
        if col not in trace.columns:
            print(f"ERROR: trace missing {col}", file=sys.stderr)
            return 1

    rolling_config = RollingConfig(
        forecast_source=args.forecast,
        horizon_hours=args.horizon,
        infill_lag_hours=args.infill_lag_hours,
        current_interval_price=args.mode,
        usage_forecast=_parse_forecast_sub_config(
            not args.no_load_forecast, UsageForecastConfig),
        solar_forecast=_parse_forecast_sub_config(
            not args.no_solar_forecast, SolarForecastConfig),
        commit_actual_load=not args.no_commit_actual_load,
        on_degraded_period=args.on_degraded_period,
        max_staleness_hours=args.max_staleness_hours,
    )

    analysis = run_rolling_analysis(
        args.site_id, args.year, trace, prices_mwh,
        network_tariff, retail_tariff, DEV_BATTERY, rolling_config,
        export_limit_kw=args.export_limit_kw,
        state=args.state,
        store_root=args.store,
    )

    pf_analysis = None
    if args.compare_perfect_foresight:
        pf_analysis = run_lp_analysis(
            args.site_id, args.year, trace, prices_mwh,
            network_tariff, retail_tariff, DEV_BATTERY,
            export_limit_kw=args.export_limit_kw,
            state=args.state,
        )

    _print_verification(analysis, pf_analysis)
    _interactive_loop(analysis)
    return 0


if __name__ == '__main__':
    sys.exit(main())
