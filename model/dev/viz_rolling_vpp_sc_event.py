"""Interactive visualisation of rolling_vpp_sc_event output for a single site.

Usage (from project root):
    python3 -m model.dev.viz_rolling_vpp_sc_event SITE_ID YEAR
    python3 -m model.dev.viz_rolling_vpp_sc_event S0008 2022 --state NSW \\
        --forecast predispatch --horizon 18 \\
        --retail-tariff NSW_agl_battery_rewards_vpp_2026 \\
        --op-tariff NSW_operator_wholesale_2026
    python3 -m model.dev.viz_rolling_vpp_sc_event S0008 2022 \\
        --compare-perfect-foresight

For a year spanning the October 2022 archive gap, both of these are needed --
the coverage check runs before on_degraded_period is consulted:

    python3 -m model.dev.viz_rolling_vpp_sc_event S0008 2022 \\
        --on-degraded-period infill --max-staleness-hours 800

Stands to `viz_vpp_sc_event` as `viz_rolling_perfect_foresight_lp` stands to
`viz_perfect_foresight_lp`: same site, same tariffs, same three-panel windows,
same day navigation, with the dispatch produced under imperfect foresight
instead of perfect. All plotting is imported from `viz_vpp_sc_event` rather
than reimplemented, so the two are directly comparable on screen -- if a panel
looks different, the dispatch differs, not the drawing code.

What the event shading means here
---------------------------------
In `viz_vpp_sc_event` the red shading is `classify_vpp_events`, computed from
settled prices over the whole year. Here it is the mask the operator actually
declared, interval by interval, from the forecast it held at the time. The two
differ for two reasons at once, and the verification block reports both:

  - The declaration rule is the skip-free forward form of the classifier's
    backwards scan, so it starts blocks earlier where spikes cluster closer
    than one event window. This difference exists even with a perfect
    forecast.
  - The forecast is wrong sometimes. Blocks declared on a spike that never
    settled are phantom events, and are the headline cost of imperfect
    operator foresight.

`--compare-perfect-foresight` runs `vpp_sc_event` on the same inputs and
reports the mask difference and the bill difference side by side. It is a
second full solve, so it is off by default.

Comparing against a bulk orchestrator run of the same scenario
----------------------------------------------------------------
`--op-tariff` defaults to a placeholder, not to whatever a given scenario's
YAML sets as `op_retail_tariff_id`. Pass the SAME value the scenario uses, or
the operator LP here runs on a different price stack and every downstream
number -- forced import/export split, budget clamps, resettlement volume --
will disagree with the bulk run for a reason that has nothing to do with
either being wrong. The event mask and the minimum export price resolution do
NOT depend on the operator tariff (they come from the household `vpp:` block
and the settled/forecast prices only), so those two agreeing across a
mismatch is expected and does not mean the rest will.

Controls (from viz_vpp_sc_event):
    Enter / n       advance one day
    b               go back one day
    <YYYY-MM-DD>    jump to a specific date
    e               jump to the first event day
    ne              jump to the next event day
    q               quit
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model.core.paths import default_paths
from model.core.prices import (
    build_price_vectors, build_price_stack_offsets, charge_window_mask,
)
from model.core.scenario import (
    RollingConfig, UsageForecastConfig, SolarForecastConfig,
)
from model.components.billing import compute_bill
from model.components.self_consumption import self_consumption
from model.components.rolling_vpp_event import (
    count_blocks_without_a_settled_spike, run_with_planner,
)
from model.components.vpp_declaration import declared_mask, window_intervals
from model.components.vpp_event_classifier import classify_vpp_events
from model.components.vpp_window_planner import OPERATOR
from model.forecast.registry import region_for_state
from model.orchestrate import (
    build_rolling_price_provider, DEFAULT_PREDISPATCH_STORE,
    resolve_min_export_price,
)
from model.dev._dev_common import DEV_BATTERY, build_display_prices

# Plotting and interaction reused unchanged. The mask, traces and price series
# this module builds are the same shapes viz_vpp_sc_event already draws, so
# nothing about the display needs to know a rolling run produced them.
from model.dev.viz_vpp_sc_event import (
    EovAnalysis, load_inputs, _build_event_summary, _interactive_loop,
    write_event_csv,
)

BATTERY_MODE = 'Rolling VPP Events, Self-Consumption Between'


def run_rolling_vpp_analysis(
    site_id: str,
    year: int,
    trace,
    prices_mwh,
    network_tariff,
    retail_tariff,
    op_tariff,
    battery_spec,
    rolling_config: RollingConfig,
    export_limit_kw: float,
    state: str,
    interval_hours: float = 5 / 60,
    store_root: Path = DEFAULT_PREDISPATCH_STORE,
    gap_strategy: str = 'self_consumption',
) -> tuple[EovAnalysis, object, object]:
    """Run rolling_vpp_event and package it for the shared display code.

    Returns (EovAnalysis, RollingDiagnostics, MinExportPriceResolution). The
    analysis object carries the DECLARED event mask in its `vpp_event` field,
    so the shared plotting shades what the operator actually did rather than
    what a perfect-foresight classifier would have marked. The resolution is
    returned separately, not folded into EovAnalysis, because that dataclass
    is shared with the perfect-foresight viz tools and has no VPP-mode
    concept; pass it to print_verification to report the breakdown.
    """
    solar = trace['pv_site_net_kwh']
    load = trace['gross_load_kwh']
    idx = solar.index

    vpp_cfg = retail_tariff.vpp
    if vpp_cfg is None:
        raise ValueError(
            f"Retail tariff '{retail_tariff.name}' has no VppConfig block. "
            "Use a VPP-enabled retail tariff YAML."
        )

    # Resolved exactly as the orchestrator resolves it for a bulk run of the
    # same scenario, so the two dispatch against the same number. Before this
    # the raw configured value was passed straight through regardless of
    # min_export_price_mode, so a based_on_this_year tariff dispatched here
    # with a threshold the bulk run never used -- and the difference showed up
    # nowhere in this tool's own output, only as numbers that didn't reconcile
    # against a bulk run of the same scenario. Raises the same way
    # validate_scenario's rules do: absent under 'fixed', or on
    # 'based_on_last_year', which is not implemented.
    min_export_resolution = resolve_min_export_price(
        vpp_cfg=vpp_cfg,
        prices_mwh=prices_mwh,
        site_traces={site_id: {'gross_load_kwh': load}},
        site_ids=[site_id],
        power_kw=battery_spec.power_kw,
        interval_hours=interval_hours,
    )
    min_export_price_dollars_per_mwh = min_export_resolution.value_dollars_per_mwh

    # Household display prices, and the operator price stack the LP runs on.
    real_p_imp, real_p_exp = build_price_vectors(
        idx, prices_mwh, network_tariff, retail_tariff, state, interval_hours)
    op_p_imp, op_p_exp = build_price_vectors(
        idx, prices_mwh, network_tariff, op_tariff, state, interval_hours)
    op_offsets = build_price_stack_offsets(
        idx, network_tariff, op_tariff, state, interval_hours)

    provider = build_rolling_price_provider(
        rolling_config, prices_mwh, interval_hours,
        region=region_for_state(state), store_root=store_root,
    )

    _C = 100.0
    p_curtail = (vpp_cfg.curtail_c_per_kwh / _C
                 if vpp_cfg.curtail_c_per_kwh is not None else None)

    no_grid_charge_mask = None
    if vpp_cfg.no_grid_charge_windows:
        no_grid_charge_mask = np.zeros(len(idx), dtype=bool)
        for window in vpp_cfg.no_grid_charge_windows:
            no_grid_charge_mask |= charge_window_mask(
                idx, window, state, interval_hours)

    sc_result = self_consumption(
        site_id, solar, load, battery_spec, interval_hours)
    sc_bill = compute_bill(sc_result, prices_mwh, network_tariff,
                           retail_tariff, state, interval_hours)

    soc_min = (battery_spec.backup_reserve_pct / 100.0) * battery_spec.capacity_kwh

    t0 = time.perf_counter()
    result, vt, diagnostics, planner = run_with_planner(
        site_id=site_id,
        solar_kwh=solar, load_kwh=load,
        prices_mwh=prices_mwh,
        op_price_offsets=op_offsets,
        forecast_provider=provider,
        battery_spec=battery_spec,
        soc_init_kwh=soc_min, soc_final_kwh=soc_min,
        config=rolling_config,
        vpp_threshold_dollars_per_mwh=vpp_cfg.vpp_threshold_dollars_per_mwh,
        event_window_hours=vpp_cfg.event_window_hours,
        vpp_exp_reward=vpp_cfg.exp_reward_c_per_kwh / _C,
        vpp_imp_reward=vpp_cfg.imp_reward_c_per_kwh / _C,
        vpp_kwh_limit=vpp_cfg.kwh_limit,
        vpp_limit_applies_to=vpp_cfg.limit_applies_to,
        min_export_price_dollars_per_mwh=min_export_price_dollars_per_mwh,
        gap_strategy=gap_strategy,
        # Only read on 'tou_advanced', where the wrapper builds the household
        # AdvancedWindowPlanner from them.
        retail_tariff=retail_tariff,
        network_tariff=network_tariff,
        state=state,
        export_limit_kwh_per_interval=export_limit_kw * interval_hours,
        interval_hours=interval_hours,
        p_curtail=p_curtail,
        no_grid_charge_mask=no_grid_charge_mask,
    )
    solve_seconds = time.perf_counter() - t0

    bill = compute_bill(result, prices_mwh, network_tariff, retail_tariff,
                        state, interval_hours, vpp_traces=vt)

    # The declared mask comes from the planner, not from the traces: an
    # event interval where the operator held control without dispatching
    # carries no forced flow and would be invisible in a flow-derived mask.
    declared = planner.mode == OPERATOR

    disp_p_imp, disp_p_exp = build_display_prices(
        idx, result, retail_tariff, real_p_imp, real_p_exp,
        prices_mwh, state, interval_hours,
    )

    analysis = EovAnalysis(
        site_id=site_id, year=year, trace=trace,
        sc_result=sc_result, eov_result=result, eov_vt=vt,
        sc_bill=sc_bill, eov_bill=bill,
        real_p_imp=real_p_imp, real_p_exp=real_p_exp,
        disp_p_imp=disp_p_imp, disp_p_exp=disp_p_exp,
        op_p_imp=op_p_imp, op_p_exp=op_p_exp,
        prices_mwh=prices_mwh,
        vpp_event=declared,
        battery_spec=battery_spec,
        network_tariff=network_tariff,
        retail_tariff=retail_tariff, op_tariff=op_tariff,
        export_limit_kw=export_limit_kw,
        state=state,
        soc_init_kwh=soc_min,
        interval_hours=interval_hours,
        eov_solve_seconds=solve_seconds,
        event_summary=_build_event_summary(
            declared, prices_mwh, interval_hours, vpp_cfg),
        skip_wins=[],
        min_export_price_mwh=min_export_price_dollars_per_mwh,
    )
    return analysis, diagnostics, min_export_resolution


def print_verification(analysis: EovAnalysis, diagnostics,
                       rolling_config: RollingConfig,
                       pf_analysis: EovAnalysis | None = None,
                       label: str = 'rolling_vpp_sc_event',
                       pf_label: str = 'vpp_sc_event',
                       min_export_price_resolution=None) -> None:
    """Startup verification for a rolling VPP run.

    Reports the things that can only go wrong on this path: how the declared
    mask relates to the perfect-foresight classifier, how much of the declared
    time was spent on spikes that never arrived, whether the cap bound, and
    whether the import budget or the LP itself ran into trouble.
    """
    d = diagnostics
    cfg = analysis.retail_tariff.vpp
    ih = analysis.interval_hours
    w = window_intervals(cfg.event_window_hours, ih)

    print()
    print('=' * 72)
    print(f"{label}  --  {analysis.site_id}  {analysis.year}  "
          f"{analysis.state}")
    print('=' * 72)
    print(f"  forecast source      : {rolling_config.forecast_source}")
    print(f"  horizon              : {rolling_config.horizon_hours} h "
          f"({max(1, int(round(rolling_config.horizon_hours / ih)))} intervals)")
    print(f"  event window         : {cfg.event_window_hours} h ({w} intervals)")
    print(f"  threshold            : "
          f"${cfg.vpp_threshold_dollars_per_mwh:,.0f}/MWh")
    print(f"  min export price     : "
          f"${analysis.min_export_price_mwh:,.0f}/MWh")
    r = min_export_price_resolution
    if r is not None and r.mode != 'fixed':
        cfg_str = ('none' if r.configured_dollars_per_mwh is None
                  else f"${r.configured_dollars_per_mwh:,.0f}/MWh")
        print(f"    mode                     : {r.mode}")
        print(f"    derived                  : "
              f"${r.derived_dollars_per_mwh:,.0f}/MWh over "
              f"{r.n_mask_intervals:,} event-mask interval(s), "
              f"{r.available_kwh:,.1f} kWh available (cap "
              f"{'reached' if r.budget_exhausted else 'NOT reached'})")
        print(f"    configured floor         : {cfg_str}")
        print(f"    operative = max(derived, floor), from {r.source}")
    print(f"  cap                  : {cfg.kwh_limit:,.0f} kWh "
          f"({cfg.limit_applies_to})")
    print(f"  solve wall-clock     : {analysis.eov_solve_seconds:,.1f} s")

    # --- declaration vs the perfect-foresight classifier ------------------
    prices = analysis.prices_mwh.to_numpy()
    classifier = classify_vpp_events(
        analysis.prices_mwh,
        threshold_dollars_per_mwh=cfg.vpp_threshold_dollars_per_mwh,
        event_window_hours=cfg.event_window_hours,
        interval_hours=ih,
    )
    skip_free = declared_mask(prices, w, cfg.vpp_threshold_dollars_per_mwh)
    declared = analysis.vpp_event

    print()
    print('  event mask (intervals)')
    print(f"    classify_vpp_events (settled, backwards scan) : "
          f"{int(classifier.sum()):>7,}")
    print(f"    forward rule on settled prices (skip-free)    : "
          f"{int(skip_free.sum()):>7,}")
    print(f"    declared by the operator from its forecast    : "
          f"{int(declared.sum()):>7,}")
    print(f"    rule difference  (skip-free - classifier)     : "
          f"{int(skip_free.sum() - classifier.sum()):>+7,}  "
          f"exists even with a perfect forecast")
    print(f"    forecast difference (declared - skip-free)    : "
          f"{int(declared.sum() - skip_free.sum()):>+7,}  "
          f"the cost of imperfect operator foresight")

    n_phantom = count_blocks_without_a_settled_spike(
        np.where(declared, OPERATOR, 'household'), prices,
        cfg.vpp_threshold_dollars_per_mwh)
    print()
    print('  events')
    print(f"    blocks declared            : {_fmt(d.events_anchored)}")
    print(f"    intervals under operator   : {_fmt(d.event_intervals)}")
    print(f"    blocks with no real spike  : {n_phantom}  "
          f"(declared on a forecast spike that never settled)")
    print(f"    block length min/mean/max  : "
          f"{_fmt(d.event_block_len_min)} / "
          f"{_fmt(d.event_block_len_mean, '.1f')} / "
          f"{_fmt(d.event_block_len_max)} intervals")
    print(f"    single-interval blocks     : {_fmt(d.event_blocks_len_1)}  "
          f"(flapping indicator)")

    print()
    print('  dispatch')
    print(f"    forced export              : "
          f"{_fmt(d.f_exp_total_kwh, ',.1f')} kWh")
    print(f"    forced import              : "
          f"{_fmt(d.f_imp_total_kwh, ',.1f')} kWh")
    print(f"    cap used / limit           : "
          f"{_fmt(d.cap_used_kwh, ',.1f')} / {cfg.kwh_limit:,.0f} kWh")
    if d.cap_exhausted_at is not None:
        ts = analysis.prices_mwh.index[d.cap_exhausted_at]
        print(f"    cap EXHAUSTED at           : interval "
              f"{d.cap_exhausted_at:,} ({ts})")
        print(f"                                 no events declared after "
              f"this point")
    else:
        print(f"    cap exhausted              : no")
    print(f"    forced export blocked      : "
          f"{_fmt(d.f_exp_blocked_by_price)} interval(s) below the min "
          f"export price")

    print()
    print('  accounting')
    print(f"    import budget clamps       : {_fmt(d.n_imp_budget_clamps)}")
    print(f"    blocks resettled           : {_fmt(d.blocks_resettled)}")
    print(f"    volume moved n_imp<->f_imp : "
          f"{_fmt(d.n_imp_resettled_kwh, '+,.2f')} kWh")

    print()
    print('  degraded intervals')
    print(f"    no price forecast          : {d.sc_fallback_no_forecast:,}")
    print(f"    infeasible window          : {d.sc_fallback_infeasible:,}")
    print(f"    operator LP infeasible     : {_fmt(d.operator_infeasible)}")
    print(f"    fallback inside an event   : "
          f"{_fmt(d.event_intervals_fallback)}")
    if d.n_lt_W:
        print(f"    lookahead truncated        : {d.n_lt_W:,} interval(s), "
              f"first {d.n_lt_W_first}, last {d.n_lt_W_last}")
        print(f"                                 expected at the period "
              f"edges only")

    # --- invariants -------------------------------------------------------
    vt = analysis.eov_vt
    res = analysis.eov_result
    imp_err = float(np.abs((vt.n_imp_kwh + vt.f_imp_kwh - res.import_kwh)
                           .to_numpy()).max())
    exp_err = float(np.abs((vt.n_exp_kwh + vt.f_exp_kwh - res.export_kwh)
                           .to_numpy()).max())
    forced_flow = (vt.f_imp_kwh.to_numpy() + vt.f_exp_kwh.to_numpy())
    forced_outside = int((~declared & (forced_flow > 1e-9)).sum())
    print()
    print('  invariants')
    print(f"    max |n_imp+f_imp - import| : {imp_err:.3e} kWh")
    print(f"    max |n_exp+f_exp - export| : {exp_err:.3e} kWh")
    print(f"    forced flow outside events : {forced_outside}  (must be 0)")
    ok = imp_err < 1e-6 and exp_err < 1e-6 and forced_outside == 0
    print(f"    ALL INVARIANTS            : {'PASS' if ok else 'FAIL'}")

    # --- bills ------------------------------------------------------------
    print()
    print('  household bill ($/yr)')
    print(f"    self-consumption           : "
          f"{analysis.sc_bill.household_total:>10,.2f}")
    print(f"    rolling VPP                : "
          f"{analysis.eov_bill.household_total:>10,.2f}")
    print(f"    delta                      : "
          f"{analysis.eov_bill.household_total - analysis.sc_bill.household_total:>+10,.2f}")

    if pf_analysis is not None:
        pf_total = pf_analysis.eov_bill.household_total
        roll_total = analysis.eov_bill.household_total
        sc_total = analysis.sc_bill.household_total
        pf_gain = sc_total - pf_total
        roll_gain = sc_total - roll_total
        print()
        print(f'  against perfect foresight ({pf_label}, same inputs)')
        print(f"    perfect foresight bill     : {pf_total:>10,.2f}")
        print(f"    perfect foresight events   : "
              f"{int(pf_analysis.vpp_event.sum()):>10,} intervals")
        if abs(pf_gain) > 1e-9:
            print(f"    value capture ratio        : "
                  f"{roll_gain / pf_gain:>10.1%}")
            print(f"                                 mixes forecast error "
                  f"with the declaration rule difference above; the two are "
                  f"not separable in this number")
        else:
            print(f"    value capture ratio        : n/a "
                  f"(perfect foresight gained nothing to capture)")
    print('=' * 72)
    print()


def _fmt(v, spec='') -> str:
    """None-safe formatting: a VPP counter is None on a non-VPP path, which a
    reader must be able to tell from a counter that stayed at zero."""
    if v is None:
        return 'n/a'
    return format(v, spec) if spec else f"{v:,}"


def _parse_forecast_sub_config(enabled: bool, cls):
    return cls() if enabled else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('site_id', help='Site identifier, e.g. S0008')
    parser.add_argument('year', type=int, help='Year, e.g. 2022')
    parser.add_argument('--state', default='NSW')
    parser.add_argument('--network-tariff',
                        default='NSW_ausgrid_ea025_ea029_2026')
    parser.add_argument('--retail-tariff',
                        default='agl_battery_rewards_vpp_2026')
    parser.add_argument('--op-tariff', default='NSW_operator_wholesale_2026')
    parser.add_argument('--export-limit-kw', type=float, default=10.0)
    parser.add_argument('--forecast', default='predispatch',
                        help="Price forecast source, or 'perfect' to isolate "
                             "the declaration rule difference from forecast "
                             "error.")
    parser.add_argument('--horizon', type=float, default=18.0)
    parser.add_argument('--infill-lag-hours', type=float, default=24.0)
    parser.add_argument('--mode', default='price_and_next',
                        choices=('forecast', 'price_known', 'price_and_next',
                                 'next_forecast'),
                        help="current_interval_price (default "
                             "'price_and_next' -- the settled price plus the "
                             "refreshed forecast. 'forecast' is the LEAST "
                             "informed mode, not a plainer version of full; "
                             "see RollingConfig's docstring before changing "
                             "it).")
    parser.add_argument('--on-degraded-period', default='fail',
                        choices=('fail', 'self_consumption', 'actuals',
                                 'infill'),
                        help="What a predispatch provider does about a period "
                             "the store marks as having no usable forecast "
                             "lead. Default 'fail', deliberately conscious "
                             "rather than silently substituting. A year "
                             "spanning October 2022 needs 'infill' AND "
                             "--max-staleness-hours raised past ~750.")
    parser.add_argument('--max-staleness-hours', type=float, default=6.0,
                        help="Largest gap between consecutive store runs "
                             "before refusing to start. Coverage check, not a "
                             "staleness gate on dispatch. The check runs "
                             "BEFORE on_degraded_period is consulted, so a "
                             "year with an archive gap needs both raised.")
    parser.add_argument('--no-load-forecast', action='store_true')
    parser.add_argument('--no-solar-forecast', action='store_true')
    parser.add_argument('--no-commit-actual-load', action='store_true')
    parser.add_argument('--compare-perfect-foresight', action='store_true',
                        help='Also run vpp_sc_event on the same inputs. A '
                             'second full solve, so off by default.')
    parser.add_argument('--event-csv', type=Path, default=None)
    parser.add_argument('--store', type=Path,
                        default=DEFAULT_PREDISPATCH_STORE)
    parser.add_argument('--project-root', type=Path, default=None)
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    trace, prices_mwh, network, retail, op_tariff = load_inputs(
        args.site_id, args.year, args.state,
        args.network_tariff, args.retail_tariff, args.op_tariff, paths,
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

    analysis, diagnostics, min_export_resolution = run_rolling_vpp_analysis(
        args.site_id, args.year, trace, prices_mwh,
        network, retail, op_tariff, DEV_BATTERY, rolling_config,
        export_limit_kw=args.export_limit_kw,
        state=args.state,
        store_root=args.store,
    )

    pf_analysis = None
    if args.compare_perfect_foresight:
        from model.dev.viz_vpp_sc_event import run_eov_analysis
        pf_analysis = run_eov_analysis(
            args.site_id, args.year, trace, prices_mwh,
            network, retail, op_tariff, DEV_BATTERY,
            export_limit_kw=args.export_limit_kw,
            state=args.state,
        )

    print_verification(analysis, diagnostics, rolling_config, pf_analysis,
                       min_export_price_resolution=min_export_resolution)
    if args.event_csv is not None:
        write_event_csv(analysis, args.event_csv)
    _interactive_loop(analysis, battery_mode=BATTERY_MODE)
    return 0


if __name__ == '__main__':
    sys.exit(main())
