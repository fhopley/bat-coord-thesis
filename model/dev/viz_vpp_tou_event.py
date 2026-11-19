"""Interactive visualisation of vpp_tou_event output for a single real site.

Usage (from project root):
    python3 -m model.dev.viz_vpp_tou_event SITE_ID YEAR
    python3 -m model.dev.viz_vpp_tou_event SITE_ID YEAR --state NSW \\
        --retail-tariff ea_battery_ease_2025 \\
        --op-tariff agl_operator_wholesale_2025 \\
        --export-limit-kw 10

Identical to viz_vpp_sc_event except that between-event gaps are filled by
perfect_foresight_lp_advanced (TOU-optimised) rather than self-consumption.
The event LP, verification checks, and interactive display are unchanged.

Three interactive windows open together, navigated by the same day prompt:

  FLOWS window (3 panels):
    1. Solar generation stacked (sol_to_home, sol_to_bat, sol_to_grid).
       Solar line. Event intervals shaded red.
    2. Serviced load stacked (sol_to_home, bat_to_home, grid_to_home).
       Load line. Event intervals shaded red.
    3. SOC (left axis, kWh) + charge/discharge bars (right axis, kW).
       Event intervals shaded red.

  PRICES window (3 panels):
    1. Export (kW): n_exp (solar) + f_exp (forced, red). Real TOU export
       price (c/kWh right axis, dotted). Op export price (dashed).
    2. Import (kW): n_imp (load blue) + f_imp (forced, red). Real TOU import
       price and op import price (right axis).
    3. SOC + charge/discharge panel (same as FLOWS window).

Controls:
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

from model.core.paths import default_paths
from model.core.prices import build_lp_price_vectors_advanced, charge_window_mask
from model.components.vpp_sc_event import vpp_sc_event
from model.dev._dev_common import DEV_BATTERY, build_display_prices
from model.dev.viz_vpp_sc_event import (
    EovAnalysis,
    load_inputs,
    _build_event_summary,
    _print_verification,
    _interactive_loop,
    write_event_csv,
)
from model.components.self_consumption import self_consumption
from model.components.vpp_event_classifier import classify_vpp_events
from model.components.billing import compute_bill
from model.orchestrate import _compute_min_export_price


# Battery coordination mode label shown in chart titles after the retail tariff
# name.  Edit this string to describe the strategy being visualised.
BATTERY_MODE = 'VPP Events & Household Bill Optimised'

# Set True to bias the TOU gap LP toward charging with solar first and exporting
# the surplus -- produces more realistic per-interval dispatch profiles but adds
# ~3x solve time overhead. Set False to exactly match scenario bulk runs (which
# default to apply_export_order_nudge=False). Note: the TOU LP horizon extends
# past event windows and its gap-only portion is used; even a small nudge change
# can alter the gap dispatch and therefore the SOC handed to the event LP, so
# profiles produced with True may not be bill-identical to scenario output.
APPLY_EXPORT_ORDER_NUDGE = True


def run_tou_analysis(
    site_id: str,
    year: int,
    trace,
    prices_mwh,
    network_tariff,
    retail_tariff,
    op_tariff,
    battery_spec,
    export_limit_kw: float,
    state: str,
    interval_hours: float = 5 / 60,
    apply_export_order_nudge: bool = True,
) -> EovAnalysis:
    """Run the full vpp_tou_event analysis pipeline (no disk access).

    Identical to viz_vpp_sc_event.run_eov_analysis except gap_strategy is
    set to 'tou_advanced' and the household TOU price vectors are built and
    passed through to the component.

    Args:
        state: the site's actual location (scenario.state), used to resolve
            local time for every clock-driven tariff feature (TOU periods,
            no-grid-charge windows, VPP event billing). Required -- there is
            no sane default, and passing the wrong state silently bills the
            wrong tariff periods rather than raising, so this parameter is
            deliberately mandatory rather than defaulted to 'NSW' or similar.
    """
    from model.core.prices import build_price_vectors

    solar = trace['pv_site_net_kwh']
    load  = trace['gross_load_kwh']
    idx   = solar.index

    vpp_cfg = retail_tariff.vpp
    if vpp_cfg is None:
        raise ValueError(
            f"Retail tariff '{retail_tariff.name}' has no VppConfig block. "
            "Use a VPP-enabled retail tariff YAML."
        )

    # Operator price vectors
    op_p_imp, op_p_exp = build_price_vectors(
        idx, prices_mwh, network_tariff, op_tariff, state, interval_hours)

    # Household TOU price vectors (for the TOU gap LP)
    hh_p_imp, hh_p_exp = build_lp_price_vectors_advanced(
        idx, prices_mwh, network_tariff, retail_tariff, state, interval_hours)

    # Real household prices for display (same vectors, used for axis labels)
    real_p_imp, real_p_exp = build_price_vectors(
        idx, prices_mwh, network_tariff, retail_tariff, state, interval_hours)

    # Self-consumption baseline for billing delta
    sc_result = self_consumption(site_id, solar, load, battery_spec, interval_hours)
    sc_bill   = compute_bill(sc_result, prices_mwh, network_tariff, retail_tariff,
                             state, interval_hours)

    soc_min  = (battery_spec.backup_reserve_pct / 100.0) * battery_spec.capacity_kwh
    soc_init = soc_min

    # Event classification
    vpp_event = classify_vpp_events(
        prices_mwh,
        threshold_dollars_per_mwh=vpp_cfg.vpp_threshold_dollars_per_mwh,
        event_window_hours=vpp_cfg.event_window_hours,
        interval_hours=interval_hours,
    )

    event_summary = _build_event_summary(vpp_event, prices_mwh, interval_hours, vpp_cfg)

    _C = 100.0
    export_limit_kwh = export_limit_kw * interval_hours

    p_curtail = (vpp_cfg.curtail_c_per_kwh / _C
                 if vpp_cfg.curtail_c_per_kwh is not None else None)

    # Resolve minimum forced export price ($/MWh -> $/kWh)
    if vpp_cfg.min_export_price_dollars_per_mwh is not None:
        min_export_price_mwh = vpp_cfg.min_export_price_dollars_per_mwh
    else:
        min_export_price_mwh = _compute_min_export_price(
            prices_mwh=prices_mwh,
            site_traces={site_id: trace},
            site_ids=[site_id],
            power_kw=battery_spec.power_kw,
            interval_hours=interval_hours,
            kwh_limit=vpp_cfg.kwh_limit,
            vpp_event=vpp_event,
        )
    min_export_price_per_kwh = min_export_price_mwh / 1000.0

    # No-grid-charge mask: identical construction to _run_vpp_sc_event in orchestrate.py
    no_grid_charge_mask = None
    if vpp_cfg.no_grid_charge_windows:
        no_grid_charge_mask = np.zeros(len(idx), dtype=bool)
        for window in vpp_cfg.no_grid_charge_windows:
            no_grid_charge_mask |= charge_window_mask(idx, window, state, interval_hours)

    t0 = time.perf_counter()
    import warnings as _warnings
    skip_wins: list[str] = []
    with _warnings.catch_warnings(record=True) as _caught:
        _warnings.simplefilter("always")
        eov_result, eov_vt = vpp_sc_event(
            site_id, solar, load,
            op_p_imp, op_p_exp,
            vpp_event=vpp_event,
            battery_spec=battery_spec,
            soc_init_kwh=soc_init,
            vpp_exp_reward=vpp_cfg.exp_reward_c_per_kwh / _C,
            vpp_imp_reward=vpp_cfg.imp_reward_c_per_kwh / _C,
            vpp_kwh_limit=vpp_cfg.kwh_limit,
            vpp_limit_applies_to=vpp_cfg.limit_applies_to,
            export_limit_kwh_per_interval=export_limit_kwh,
            interval_hours=interval_hours,
            p_curtail=p_curtail,
            gap_strategy='tou_advanced',
            retail_tariff=retail_tariff,
            prices_mwh=prices_mwh,
            price_imp=hh_p_imp,
            price_exp=hh_p_exp,
            min_export_price_per_kwh=min_export_price_per_kwh,
            apply_export_order_nudge=apply_export_order_nudge,
            no_grid_charge_mask=no_grid_charge_mask,
            state=state,
        )
    eov_solve_seconds = time.perf_counter() - t0

    for w in _caught:
        msg = str(w.message)
        if msg.startswith('EOV_SKIP:'):
            parts = msg.split(':')
            skip_wins.append(
                f"    t={parts[2]}  remaining_cap={float(parts[3]):.2f} kWh  "
                f"load-solar={float(parts[4]):.2f} kWh  "
                f"n_imp_hi_sum={float(parts[5]):.2f} kWh  "
                f"{parts[6]}"
            )
        else:
            _warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    eov_bill = compute_bill(eov_result, prices_mwh, network_tariff, retail_tariff,
                            state, interval_hours, vpp_traces=eov_vt)

    # See run_eov_analysis: must be built after the component returns, since
    # the marginal tier rate depends on the solved cumulative daily flow.
    disp_p_imp, disp_p_exp = build_display_prices(
        idx, eov_result, retail_tariff,
        real_p_imp, real_p_exp, prices_mwh, state, interval_hours,
    )

    return EovAnalysis(
        site_id=site_id, year=year, trace=trace,
        sc_result=sc_result, eov_result=eov_result, eov_vt=eov_vt,
        sc_bill=sc_bill, eov_bill=eov_bill,
        real_p_imp=real_p_imp, real_p_exp=real_p_exp,
        disp_p_imp=disp_p_imp, disp_p_exp=disp_p_exp,
        op_p_imp=op_p_imp, op_p_exp=op_p_exp,
        prices_mwh=prices_mwh,
        vpp_event=vpp_event,
        battery_spec=battery_spec,
        network_tariff=network_tariff,
        retail_tariff=retail_tariff, op_tariff=op_tariff,
        export_limit_kw=export_limit_kw,
        state=state,
        soc_init_kwh=soc_init,
        interval_hours=interval_hours,
        eov_solve_seconds=eov_solve_seconds,
        event_summary=event_summary,
        skip_wins=skip_wins,
        min_export_price_mwh=min_export_price_mwh,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('site_id', help='Site identifier, e.g. S0008')
    parser.add_argument('year', type=int, help='Year, e.g. 2022')
    parser.add_argument('--state', default='NSW')
    parser.add_argument('--network-tariff', default='NSW_ausgrid_ea025_ea029_2026')
    parser.add_argument('--retail-tariff', default='NSW_agl_battery_rewards_vpp_2026')
    parser.add_argument('--op-tariff', default='NSW_operator_wholesale_2026')
    parser.add_argument('--export-limit-kw', type=float, default=10.0)
    parser.add_argument('--event-csv', type=Path, default=None,
                        help='If provided, write per-interval event diagnostic CSV to this path.')
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

    analysis = run_tou_analysis(
        args.site_id, args.year, trace, prices_mwh,
        network, retail, op_tariff, DEV_BATTERY,
        export_limit_kw=args.export_limit_kw,
        state=args.state,
        apply_export_order_nudge=APPLY_EXPORT_ORDER_NUDGE,
    )
    _print_verification(analysis)
    if args.event_csv is not None:
        write_event_csv(analysis, args.event_csv)
    _interactive_loop(analysis, battery_mode=BATTERY_MODE)
    return 0


if __name__ == '__main__':
    sys.exit(main())