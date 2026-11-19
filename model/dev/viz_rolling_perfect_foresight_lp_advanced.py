"""Interactive visualisation of rolling advanced-TOU dispatch for one real site.

Usage (from project root):
    python3 -m model.dev.viz_rolling_perfect_foresight_lp_advanced SITE_ID YEAR
    python3 -m model.dev.viz_rolling_perfect_foresight_lp_advanced S0008 2022 --on-degraded-period infill --max-staleness-hours 800 --compare-perfect-foresight
    python3 -m model.dev.viz_rolling_perfect_foresight_lp_advanced S0008 2022 \\
        --state NSW --retail-tariff NSW_globird_zerohero_2026 \\
        --horizon 18 --compare-perfect-foresight

The rolling counterpart of viz_perfect_foresight_lp_advanced.py, and the
advanced counterpart of viz_rolling_perfect_foresight_lp.py. It descends from
both rather than sitting beside them:

  - The dispatch goes through
    model.orchestrate.run_rolling_perfect_foresight_lp_advanced_site, which is
    the same function the bulk orchestrator's per-site worker calls. A
    single-site script that hand-mirrored the SOC pins and forecast wiring
    would be a second copy of exactly the conventions that must not drift.
  - The plotting is imported unchanged from viz_perfect_foresight_lp.py and
    viz_perfect_foresight_lp_advanced.py, so this script and the
    perfect-foresight advanced viz show identical days for identical inputs.
    Block tier bands and marginal-tier display prices come from there too.

RollingAdvancedAnalysis subclasses LPAdvancedAnalysis so every existing
advanced plotting function works on it unchanged. As in the wholesale rolling
viz, field names keep their LP-era meaning: lp_result IS the rolling dispatch,
and backend is 'rolling' rather than a solver name so the two scripts are
never confused from their output alone.

What the price panels show, and the one case where it is not what the plan saw
---------------------------------------------------------------------------
The panels plot prices built from the SETTLED wholesale price. On this tariff
the settled price only enters through the critical rates, and under the
default current_interval_price mode ('price_and_next', the production setting)
the committed interval is priced at the settled price -- so for every interval
actually committed, the critical state shown is the one the plan acted on.

Under --mode forecast the committed interval is priced from the forecast
instead. An interval can then be critical in the plot and not in the plan, or the
reverse, and the dispatch will look inexplicable at exactly those intervals.
That is a real property of that mode, not a plotting bug. The verification
block says which mode was used.

On a mixed plan the settled price enters every interval's dispatch prices,
and the panels show those dispatch prices built from the settled price. The
plan acted on the forecast, so under --mode forecast (the setting for the
Tesla runs) every committed interval was priced differently from what is
plotted.

Controls: same as the other viz scripts (Enter/n, b, YYYY-MM-DD, q).
"""
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from model.core.paths import default_paths
from model.core.scenario import (
    BatterySpec, RollingConfig, UsageForecastConfig, SolarForecastConfig,
)
from model.core.tariff import NetworkTariff, RetailTariff
from model.core.prices import build_lp_price_vectors_advanced
from model.components.billing import compute_bill
from model.components.rolling_optimisation import RollingDiagnostics
from model.forecast.registry import region_for_state
from model.dev._dev_common import (
    DEV_BATTERY, build_billing_prices, build_display_prices,
)
from model.orchestrate import (
    advanced_plan_needs_price_forecast, build_rolling_price_provider,
    run_rolling_perfect_foresight_lp_advanced_site, DEFAULT_PREDISPATCH_STORE,
)

from model.dev.viz_perfect_foresight_lp import (
    load_inputs, _recover_curtail,
)
from model.dev.viz_perfect_foresight_lp_advanced import (
    LPAdvancedAnalysis, run_lp_advanced_analysis, _interactive_loop_advanced,
)


@dataclass(frozen=True)
class RollingAdvancedAnalysis(LPAdvancedAnalysis):
    """LPAdvancedAnalysis plus the rolling-specific fields the plotting code
    never touches but the verification printer needs.

    diagnostics: RollingDiagnostics for this run, including the advanced
        counters (reward days lost, block tier days used up).
    rolling_config: the RollingConfig actually used, so the printed
        verification block states what was run rather than the reader having
        to cross-reference argv.
    """
    diagnostics: Optional[RollingDiagnostics] = None
    rolling_config: Optional[RollingConfig] = None


def run_rolling_advanced_analysis(
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
) -> RollingAdvancedAnalysis:
    """Run self_consumption then the rolling advanced dispatch, and bill both.

    Pure computation given inputs (no disk access apart from the pre-dispatch
    store, and only when the plan has a critical rate).

    Mirrors run_lp_advanced_analysis's shape so the two read side by side, but
    the dispatch call goes through the shared orchestrator function rather
    than a local reimplementation.
    """
    solar = trace['pv_site_net_kwh']
    load = trace['gross_load_kwh']

    # A plan with no critical rate reads no wholesale price at all, so
    # building a provider for it would query a store for a forecast nothing
    # can use -- and, worse, a gap in that store would push the run into
    # self-consumption for a reason unconnected to the plan.
    needs_prices = advanced_plan_needs_price_forecast(retail_tariff)
    provider = None
    if needs_prices:
        provider = build_rolling_price_provider(
            rolling_config, prices_mwh, interval_hours,
            region=region_for_state(state), store_root=store_root,
        )

    export_limit_kwh = export_limit_kw * interval_hours
    _t0 = time.perf_counter()
    sc_result, roll_result, diagnostics = (
        run_rolling_perfect_foresight_lp_advanced_site(
            site_id, solar, load, battery_spec,
            network_tariff, retail_tariff, provider, rolling_config,
            export_limit_kwh_per_interval=export_limit_kwh,
            interval_hours=interval_hours, state=state,
            settled_rrp_mwh=prices_mwh,
        )
    )
    wall_seconds = time.perf_counter() - _t0

    soc_min = (battery_spec.backup_reserve_pct / 100.0) * battery_spec.capacity_kwh
    soc_final = float(sc_result.battery_soc_kwh.iloc[-1])
    soc_final = min(max(soc_final, soc_min), battery_spec.capacity_kwh)

    # LP objective prices, built from the SETTLED price. See this module's
    # docstring for the one mode where these differ from what the plan saw.
    p_imp, p_exp = build_lp_price_vectors_advanced(
        solar.index, prices_mwh, network_tariff, retail_tariff, state,
        interval_hours,
    )

    sc_bill = compute_bill(sc_result, prices_mwh, network_tariff,
                           retail_tariff, state, interval_hours)
    roll_bill = compute_bill(roll_result, prices_mwh, network_tariff,
                             retail_tariff, state, interval_hours)
    curtail = _recover_curtail(solar, roll_result)

    # Billing prices: what the customer actually pays and earns. Display only.
    p_imp_billing, p_exp_billing = build_billing_prices(
        roll_result.import_kwh.index, prices_mwh, network_tariff,
        retail_tariff, state, interval_hours,
    )
    # A mixed plan displays its dispatch prices (see module docstring).
    p_imp_display, p_exp_display = build_display_prices(
        roll_result.import_kwh.index, roll_result, retail_tariff,
        *((p_imp, p_exp) if retail_tariff.type == 'mixed'
          else (p_imp_billing, p_exp_billing)),
        prices_mwh, state, interval_hours,
    )

    return RollingAdvancedAnalysis(
        site_id=site_id, year=year, trace=trace,
        sc_result=sc_result, lp_result=roll_result,
        sc_bill=sc_bill, lp_bill=roll_bill,
        p_imp=p_imp, p_exp=p_exp,
        p_imp_billing=p_imp_billing, p_exp_billing=p_exp_billing,
        p_imp_display=p_imp_display, p_exp_display=p_exp_display,
        prices_mwh=prices_mwh,
        curtail=curtail,
        battery_spec=battery_spec,
        network_tariff=network_tariff, retail_tariff=retail_tariff,
        export_limit_kw=export_limit_kw,
        state=state,
        soc_init_kwh=soc_min, soc_final_kwh=soc_final,
        interval_hours=interval_hours,
        backend='rolling',
        lp_solve_seconds=wall_seconds,
        diagnostics=diagnostics,
        rolling_config=rolling_config,
    )


def _print_verification(analysis: RollingAdvancedAnalysis,
                        pf_analysis: Optional[LPAdvancedAnalysis] = None) -> None:
    """Print startup verification.

    Distinct from viz_perfect_foresight_lp_advanced's _print_verification_advanced
    for the same reason the wholesale rolling viz has its own: that one checks
    LP-specific invariants (a single global objective, an LP <= self_consumption
    bound) which do not hold for a rolling run, since there is no single
    objective and the dispatch can fall back to self-consumption interval by
    interval.
    """
    a = analysis
    cap = a.battery_spec.capacity_kwh
    min_soc = (a.battery_spec.backup_reserve_pct / 100.0) * cap
    tol = 1e-6

    def status(ok: bool) -> str:
        return 'PASS' if ok else 'FAIL'

    rt = a.retail_tariff
    rc = a.rolling_config
    print()
    print(f"=== Verification: site {a.site_id} year {a.year} "
          f"(rolling_perfect_foresight_lp_advanced) ===")
    print(f"retail plan: {rt.name}  "
          f"block import {len(rt.block_import_tiers or ())}  "
          f"block export {len(rt.block_export_tiers or ())}  "
          f"zero-import reward "
          f"{'yes' if rt.zero_import_reward is not None else 'no'}  "
          f"critical rates {'yes' if rt.has_critical_rates else 'no'}  "
          f"type {rt.type}")
    print(f"horizon={rc.horizon_hours:g}h  "
          f"load_forecast={'on' if rc.usage_forecast is not None else 'off'}  "
          f"solar_forecast={'on' if rc.solar_forecast is not None else 'off'}  "
          f"commit_actual_load={rc.commit_actual_load}")
    if advanced_plan_needs_price_forecast(rt):
        print(f"forecast_source={rc.forecast_source}  "
              f"current_interval_price={rc.current_interval_price}")
        if rt.type == 'mixed':
            print("mixed plan: every window price is built from the price "
                  "forecast; the panels show dispatch prices built from the "
                  "settled price.")
        if rc.current_interval_price == 'forecast':
            print("NOTE: --mode forecast prices the committed interval from "
                  "the forecast, so the prices in the panels are not always "
                  "the ones the plan acted on. See this module's docstring.")
        if rc.forecast_source == 'predispatch':
            print(f"on_degraded_period={rc.on_degraded_period}  "
                  f"max_staleness_hours={rc.max_staleness_hours:g}")
    else:
        print("price forecast: NOT USED -- this plan has no critical "
              "rate, so nothing in its prices depends on the wholesale "
              "price. The rolling problem here is site forecast error alone.")
    print()
    print(a.diagnostics.summary())
    print()

    # A causality violation means the result is not a legitimate rolling
    # output, so refuse to plot it as though it were one. Matches the
    # wholesale rolling viz and the orchestrator.
    if not a.diagnostics.causality_ok:
        print(f"causality  {status(False)} -- see max_run_time_lead_minutes "
              "above. Aborting rather than plotting an invalid dispatch.")
        raise RuntimeError(
            f"{a.site_id}: RollingDiagnostics.causality_ok is False.")
    print(f"causality                       {status(True)}")

    soc = a.lp_result.battery_soc_kwh
    bounds_ok = bool(((soc >= min_soc - tol) & (soc <= cap + tol)).all())
    print(f"SOC within [reserve, capacity]  {status(bounds_ok)} "
          f"[{soc.min():.4f}, {soc.max():.4f}] kWh vs "
          f"[{min_soc:.4f}, {cap:.4f}]")

    terminal_ok = abs(float(soc.iloc[-1]) - a.soc_final_kwh) <= 1e-3
    print(f"terminal SOC matches SC target  {status(terminal_ok)} "
          f"{float(soc.iloc[-1]):.4f} vs target {a.soc_final_kwh:.4f} kWh")

    solar = a.trace['pv_site_net_kwh']
    r = a.lp_result
    balance = solar - (r.sol_to_home_kwh + r.sol_to_bat_kwh
                       + r.sol_to_grid_kwh + a.curtail)
    solar_ok = bool((balance.abs() < 1e-6).all())
    print(f"solar balance                   {status(solar_ok)} "
          f"max |residual| {balance.abs().max():.2e} kWh")

    # The bill's zero-import credits against the planner's own count of days
    # it gave up on. These are computed by different code from different
    # inputs -- billing walks the committed traces, the planner accumulates as
    # it goes -- so agreement is worth seeing rather than assuming.
    if rt.zero_import_reward is not None:
        reward_per_day = rt.zero_import_reward.reward_c_per_day / 100.0
        credits = -a.lp_bill.retail_other_credits
        days_paid = credits / reward_per_day if reward_per_day else float('nan')
        print(f"zero-import reward              "
              f"${credits:.2f} credited (~{days_paid:.0f} day(s) paid); "
              f"planner recorded {a.diagnostics.reward_days_lost} day(s) lost")

    print()
    print(f"--- timing ---")
    print(f"wall clock:  {a.lp_solve_seconds:.1f}s   "
          f"LP solve only: {a.diagnostics.solve_seconds_total:.1f}s "
          f"({a.diagnostics.solve_ms_median:.2f}ms median, "
          f"{a.diagnostics.solve_ms_p95:.2f}ms p95 per window)")
    print()
    print(f"self_consumption bill:  ${a.sc_bill.household_total:.2f}")
    print(f"rolling bill:           ${a.lp_bill.household_total:.2f}")
    if not (bounds_ok and terminal_ok and solar_ok):
        print("\n*** ONE OR MORE CHECKS FAILED -- see above ***")

    if pf_analysis is not None:
        sc_total = a.sc_bill.household_total
        pf_total = pf_analysis.lp_bill.household_total
        roll_total = a.lp_bill.household_total
        headroom = sc_total - pf_total
        print()
        print("--- rolling vs perfect_foresight_lp_advanced "
              "(same site/prices/tariffs/battery) ---")
        print(f"perfect_foresight_lp_advanced bill:  ${pf_total:.2f}")
        print(f"rolling bill:                        ${roll_total:.2f}")
        if abs(headroom) < 1e-6:
            print("headroom (self_consumption - perfect_foresight) is ~$0 -- "
                  "VCR is undefined for this site/period, comparing bills "
                  "directly instead.")
            print(f"rolling - perfect_foresight:  ${roll_total - pf_total:+.2f}")
        else:
            vcr = (sc_total - roll_total) / headroom
            print(f"headroom (self_consumption - perfect_foresight): "
                  f"${headroom:.2f}")
            print(f"rolling value-capture ratio: {vcr:.1%}")
            if rt.has_critical_rates:
                print("Expect a noisier, lower VCR here than on the wholesale "
                      "path at equal forecast quality: a critical rate turns a "
                      "small price error into a whole rate won or lost, and on "
                      "a plan where the reward window and the block export "
                      "window coincide, chasing a spike that does not arrive can "
                      "cost the reward as well. That is a predicted result, not "
                      "a bug.")


def _sub_config(enabled: bool, cls):
    """Enabled -> cls() with fitted defaults; disabled -> None."""
    return cls() if enabled else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('site_id', help='Site identifier, e.g. S0008')
    parser.add_argument('year', type=int, help='Year, e.g. 2022')
    parser.add_argument('--state', default='NSW',
                        help='NEM region for prices (default NSW)')
    parser.add_argument('--network-tariff',
                        default='NSW_ausgrid_ea025_ea029_2026')
    parser.add_argument('--retail-tariff',
                        default='NSW_globird_zerohero_2026',
                        help='Advanced TOU plan (default the GloBird '
                             'ZeroHero reference plan).')
    parser.add_argument('--export-limit-kw', type=float, default=10.0,
                        help='Per-phase export limit kW, 1 phase assumed '
                             '(default 10)')
    parser.add_argument('--horizon', type=float, default=18.0,
                        help='horizon_hours (default 18, carried over from '
                             'the wholesale path; not yet swept for this '
                             'one, which has strong 24-hour structure from '
                             'the daily tier reset).')
    parser.add_argument('--forecast', default='predispatch',
                        help="forecast_source. Only consulted when the plan "
                             "has a critical rate; ignored entirely "
                             "otherwise.")
    parser.add_argument('--mode', default='price_and_next',
                        choices=('forecast', 'price_known', 'price_and_next',
                                 'next_forecast'),
                        help="current_interval_price (default "
                             "'price_and_next'). Under 'forecast' the price "
                             "panels can disagree with what the plan saw at "
                             "a critical threshold -- see this module's "
                             "docstring.")
    parser.add_argument('--infill-lag-hours', type=float, default=24.0)
    parser.add_argument('--no-load-forecast', action='store_true',
                        help='Disable usage_forecast (default: enabled).')
    parser.add_argument('--no-solar-forecast', action='store_true',
                        help='Disable solar_forecast (default: enabled).')
    parser.add_argument('--no-commit-actual-load', action='store_true',
                        help='Set commit_actual_load=False (default: True).')
    parser.add_argument('--store', type=Path,
                        default=DEFAULT_PREDISPATCH_STORE,
                        help='Pre-dispatch store root, for --forecast '
                             'predispatch.')
    parser.add_argument('--on-degraded-period', default='fail',
                        choices=('fail', 'self_consumption', 'actuals',
                                 'infill'))
    parser.add_argument('--max-staleness-hours', type=float, default=6.0)
    parser.add_argument('--compare-perfect-foresight', action='store_true',
                        help='Also run perfect_foresight_lp_advanced on the '
                             'same inputs and print the value-capture ratio. '
                             'A second full-year LP solve, so off by default.')
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

    if retail_tariff.type not in ('tou', 'mixed'):
        print(f"ERROR: --retail-tariff {args.retail_tariff} is type "
              f"'{retail_tariff.type}'. This script is for advanced TOU "
              f"and mixed plans; use viz_rolling_perfect_foresight_lp.py for "
              f"wholesale-exposed plans.", file=sys.stderr)
        return 1

    rolling_config = RollingConfig(
        forecast_source=args.forecast,
        horizon_hours=args.horizon,
        infill_lag_hours=args.infill_lag_hours,
        current_interval_price=args.mode,
        usage_forecast=_sub_config(not args.no_load_forecast,
                                   UsageForecastConfig),
        solar_forecast=_sub_config(not args.no_solar_forecast,
                                   SolarForecastConfig),
        commit_actual_load=not args.no_commit_actual_load,
        on_degraded_period=args.on_degraded_period,
        max_staleness_hours=args.max_staleness_hours,
    )

    analysis = run_rolling_advanced_analysis(
        args.site_id, args.year, trace, prices_mwh,
        network_tariff, retail_tariff, DEV_BATTERY, rolling_config,
        export_limit_kw=args.export_limit_kw,
        state=args.state,
        store_root=args.store,
    )

    pf_analysis = None
    if args.compare_perfect_foresight:
        pf_analysis = run_lp_advanced_analysis(
            args.site_id, args.year, trace, prices_mwh,
            network_tariff, retail_tariff, DEV_BATTERY,
            export_limit_kw=args.export_limit_kw,
            state=args.state,
        )

    _print_verification(analysis, pf_analysis)
    _interactive_loop_advanced(analysis)
    return 0


if __name__ == '__main__':
    sys.exit(main())
