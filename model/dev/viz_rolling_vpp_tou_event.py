"""Interactive visualisation of rolling_vpp_tou_event output for a single site.

Usage (from project root):
    python3 -m model.dev.viz_rolling_vpp_tou_event SITE_ID YEAR
    python3 -m model.dev.viz_rolling_vpp_tou_event S0008 2022 \\
        --retail-tariff ea_battery_ease_vpp_2026
    python3 -m model.dev.viz_rolling_vpp_tou_event S0008 2022 \\
        --retail-tariff origin_battery_maximiser_vpp_2026 \\
        --compare-perfect-foresight

For a year spanning the October 2022 archive gap, both of these are needed --
the coverage check runs before on_degraded_period is consulted:

    python3 -m model.dev.viz_rolling_vpp_tou_event S0008 2022 \\
        --on-degraded-period infill --max-staleness-hours 800

Stands to `viz_rolling_vpp_sc_event` as `viz_vpp_tou_event` stands to
`viz_vpp_sc_event`: same site, same tariffs, same windows, same navigation,
with between-event gaps filled by the TOU planner instead of self-consumption.
The analysis, verification and display code is imported rather than
reimplemented, so a difference on screen is a difference in dispatch.

What to watch here that does not matter on the SC path
------------------------------------------------------
Under `rolling_vpp_sc_event` a declared event that never dispatches is close to
harmless: the operator runs the battery at roughly self-consumption anyway.
Here it is not. Every declared interval is an interval the household is NOT
optimising its TOU bill over, and the operator has no reason to care about the
household's tariff. So the cost of a phantom event is the TOU value forgone,
not just the forced flows.

Run-on blocks make this worse than the event share suggests. A block stays
declared while any forecast price within one event window exceeds the
threshold, and through a volatile period that is almost always true, so
consecutive events merge. On the AGL plan at a 10-hour window a single block
reached 3,124 intervals -- 10.8 days of continuous operator control. The
verification block prints the block length distribution first for that reason.
Shorter event windows shrink it: EA Battery Ease and Origin both specify 2
hours.

The household is billed on TOTAL import and export, including the flows the
operator forces during events, so forced export consumes the same block-tier
headroom the household's own exports use. That coupling is the reason the
model runs one planner with two modes rather than two planners.

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
"""
import argparse
import sys
from pathlib import Path

from model.core.paths import default_paths
from model.core.scenario import (
    RollingConfig, SolarForecastConfig, UsageForecastConfig,
)
from model.dev._dev_common import DEV_BATTERY
from model.dev.viz_rolling_vpp_sc_event import (
    print_verification, run_rolling_vpp_analysis,
)
from model.dev.viz_vpp_sc_event import _interactive_loop, load_inputs, write_event_csv
from model.orchestrate import DEFAULT_PREDISPATCH_STORE

BATTERY_MODE = 'Rolling VPP Events, TOU Optimised Between'


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
                        default='ea_battery_ease_vpp_2026',
                        help='Must be a TOU plan carrying a vpp block.')
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
                             "'price_and_next'). 'forecast' is the LEAST "
                             "informed mode, not a plainer version of full.")
    parser.add_argument('--on-degraded-period', default='fail',
                        choices=('fail', 'self_consumption', 'actuals',
                                 'infill'),
                        help="Default 'fail', deliberately conscious rather "
                             "than silently substituting. A year spanning "
                             "October 2022 needs 'infill' AND "
                             "--max-staleness-hours raised past ~750.")
    parser.add_argument('--max-staleness-hours', type=float, default=6.0,
                        help="Coverage check, not a staleness gate on "
                             "dispatch. It runs BEFORE on_degraded_period is "
                             "consulted, so an archive gap needs both raised.")
    parser.add_argument('--no-load-forecast', action='store_true')
    parser.add_argument('--no-solar-forecast', action='store_true')
    parser.add_argument('--no-commit-actual-load', action='store_true')
    parser.add_argument('--compare-perfect-foresight', action='store_true',
                        help='Also run vpp_tou_event on the same inputs. A '
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
    if retail.type != 'tou':
        print(f"ERROR: retail tariff '{args.retail_tariff}' is type "
              f"'{retail.type}'. rolling_vpp_tou_event optimises the household "
              f"against a TOU plan between events; use "
              f"viz_rolling_vpp_sc_event for a household that self-consumes.",
              file=sys.stderr)
        return 1

    rolling_config = RollingConfig(
        forecast_source=args.forecast,
        horizon_hours=args.horizon,
        infill_lag_hours=args.infill_lag_hours,
        current_interval_price=args.mode,
        usage_forecast=None if args.no_load_forecast else UsageForecastConfig(),
        solar_forecast=None if args.no_solar_forecast else SolarForecastConfig(),
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
        gap_strategy='tou_advanced',
    )

    pf_analysis = None
    if args.compare_perfect_foresight:
        from model.dev.viz_vpp_tou_event import run_tou_analysis
        pf_analysis = run_tou_analysis(
            args.site_id, args.year, trace, prices_mwh,
            network, retail, op_tariff, DEV_BATTERY,
            export_limit_kw=args.export_limit_kw,
            state=args.state,
        )

    print_verification(analysis, diagnostics, rolling_config, pf_analysis,
                       label='rolling_vpp_tou_event',
                       pf_label='vpp_tou_event',
                       min_export_price_resolution=min_export_resolution)
    if args.event_csv is not None:
        write_event_csv(analysis, args.event_csv)
    _interactive_loop(analysis, battery_mode=BATTERY_MODE)
    return 0


if __name__ == '__main__':
    sys.exit(main())
