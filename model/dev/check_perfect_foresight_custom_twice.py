"""Non-interactive two-pass verification of perfect_foresight_custom.

Default mode (two-pass):
    Runs perfect_foresight_custom twice, using the first pass output as the
    baseline for the second pass. Tests whether the greedy can improve on its
    own solution. If the full-battery-barrier hypothesis is correct, pass 2
    finds no further improvement.

--seed-from-lp mode:
    Runs one custom pass seeded from the LP (HiGHS) solution. Tests whether
    custom finds any improving shifts over the LP optimum, and whether LP is
    also a local optimum under the greedy's criteria.

Usage (from project root):
    python3 -m model.dev.check_perfect_foresight_custom_twice SITE_ID YEAR
    python3 -m model.dev.check_perfect_foresight_custom_twice SITE_ID YEAR \\
        --state NSW --retail-tariff amber_wholesale_2024 --export-limit-kw 10
    python3 -m model.dev.check_perfect_foresight_custom_twice SITE_ID YEAR \\
        --seed-from-lp
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

from model.core.paths import default_paths
from model.core.prices import build_price_vectors
from model.components.self_consumption import self_consumption
from model.components.billing import compute_bill
from model.components.perfect_foresight_custom import perfect_foresight_custom
from model.dev._dev_common import DEV_BATTERY
from model.dev.viz_perfect_foresight_lp import load_inputs, run_lp_analysis


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('site_id')
    parser.add_argument('year', type=int)
    parser.add_argument('--state',           default='NSW')
    parser.add_argument('--network-tariff',  default='ausgrid_ea025_network_2024')
    parser.add_argument('--retail-tariff',   default='amber_wholesale_2024')
    parser.add_argument('--export-limit-kw', type=float, default=10.0)
    parser.add_argument('--opt-factor',      type=float, default=1.0)
    parser.add_argument('--seed-from-lp',    action='store_true',
                        help='Run one custom pass seeded from the LP solution '
                             'instead of two passes from self_consumption.')
    parser.add_argument('--project-root',    type=Path, default=None)
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    trace, prices_mwh, nt, rt = load_inputs(
        args.site_id, args.year, args.state,
        args.network_tariff, args.retail_tariff, paths,
    )
    spec    = DEV_BATTERY
    IH      = 5 / 60
    soc_min = spec.backup_reserve_pct / 100.0 * spec.capacity_kwh
    export  = args.export_limit_kw
    opt     = args.opt_factor
    solar   = trace['pv_site_net_kwh']
    load_s  = trace['gross_load_kwh']

    p_imp, p_exp = build_price_vectors(solar.index, prices_mwh, nt, rt, IH)
    pi = p_imp.to_numpy()
    pe = p_exp.to_numpy()

    def obj(tr):
        return float(np.sum(tr.import_kwh * pi - tr.export_kwh * pe))

    def bill(tr):
        return compute_bill(tr, prices_mwh, nt, rt, IH).household_total

    # Always needed
    sc_r   = self_consumption(args.site_id, solar, load_s, spec, IH)
    sf     = float(min(max(sc_r.battery_soc_kwh.iloc[-1], soc_min), spec.capacity_kwh))
    sc_obj = obj(sc_r)
    sc_b   = bill(sc_r)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lp = run_lp_analysis(
            args.site_id, args.year, trace, prices_mwh, nt, rt, spec,
            export_limit_kw=export,
        )
    lp_obj = obj(lp.lp_result)
    lp_b   = bill(lp.lp_result)

    # -------------------------------------------------------------------------
    # --seed-from-lp mode
    # -------------------------------------------------------------------------
    if args.seed_from_lp:
        soc_init_lp  = float(lp.lp_result.battery_soc_kwh.iloc[0])
        soc_final_lp = float(lp.lp_result.battery_soc_kwh.iloc[-1])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r_lp = perfect_foresight_custom(
                args.site_id, solar, load_s, p_imp, p_exp, spec,
                soc_init_kwh=soc_init_lp, soc_final_kwh=soc_final_lp,
                export_limit_kwh_per_interval=export * IH,
                interval_hours=IH, opt_factor=opt,
                sc_traces=lp.lp_result,
            )
        obj_seed = obj(r_lp)
        b_seed   = bill(r_lp)

        print()
        print(f"=== Custom seeded from LP: {args.site_id} {args.year}  "
              f"opt_factor={opt:g}  export_limit={export:.0f} kW ===")
        print()
        print(f"{'Stage':<36} {'obj ($)':>12} {'bill ($)':>11} "
              f"{'gap to LP ($)':>14} {'gap (%)':>8}")
        print("-" * 84)

        def row_lp(label, o, b):
            gap = o - lp_obj
            pct = 100 * gap / (abs(lp_obj) + 1e-9)
            print(f"{label:<36} {o:>12,.4f} {b:>11,.2f} {gap:>+14,.4f} {pct:>+8.2f}%")

        row_lp(f"LP ({lp.backend})  [seed]",  lp_obj,   lp_b)
        row_lp("custom on LP output",          obj_seed, b_seed)

        print()
        delta = obj_seed - lp_obj
        if abs(delta) < 1e-4:
            print("Custom finds no improvements over LP.")
            print("  -> LP is also a local optimum under the greedy's criteria.")
            print("     The gap is structural — the greedy cannot reach those shifts")
            print("     from its own starting point (full-battery-barrier SUPPORTED).")
        elif delta < 0:
            print(f"Custom improved on LP by ${-delta:.4f} — LP was not optimal!")
            print("  -> Unexpected: check LP formulation or solver precision.")
        else:
            print(f"Custom made LP solution worse by ${delta:.4f}.")
            print("  -> Invariant violation or baseline inconsistency — investigate.")
        return 0

    # -------------------------------------------------------------------------
    # Default two-pass mode
    # -------------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r1 = perfect_foresight_custom(
            args.site_id, solar, load_s, p_imp, p_exp, spec,
            soc_init_kwh=soc_min, soc_final_kwh=sf,
            export_limit_kwh_per_interval=export * IH,
            interval_hours=IH, opt_factor=opt,
        )
    obj1 = obj(r1)
    b1   = bill(r1)

    soc_init_p2  = float(r1.battery_soc_kwh.iloc[0])
    soc_final_p2 = float(r1.battery_soc_kwh.iloc[-1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r2 = perfect_foresight_custom(
            args.site_id, solar, load_s, p_imp, p_exp, spec,
            soc_init_kwh=soc_init_p2, soc_final_kwh=soc_final_p2,
            export_limit_kwh_per_interval=export * IH,
            interval_hours=IH, opt_factor=opt,
            sc_traces=r1,
        )
    obj2 = obj(r2)
    b2   = bill(r2)

    print()
    print(f"=== Two-pass greedy: {args.site_id} {args.year}  "
          f"opt_factor={opt:g}  export_limit={export:.0f} kW ===")
    print()
    print(f"{'Stage':<28} {'obj ($)':>12} {'bill ($)':>11} "
          f"{'gap to LP ($)':>14} {'gap (%)':>8}")
    print("-" * 76)

    def row(label, o, b):
        gap = o - lp_obj
        pct = 100 * gap / (abs(lp_obj) + 1e-9)
        print(f"{label:<28} {o:>12,.4f} {b:>11,.2f} {gap:>+14,.4f} {pct:>+8.2f}%")

    row("self_consumption",            sc_obj, sc_b)
    row("pass 1  (custom)",            obj1,   b1)
    row("pass 2  (custom on pass 1)",  obj2,   b2)
    row(f"LP ({lp.backend})",          lp_obj, lp_b)

    print()
    delta = obj2 - obj1
    print(f"Pass 2 improvement over pass 1: {delta:+.4f}  "
          f"({'better' if delta < -1e-4 else 'no meaningful change'})")
    print()
    if abs(delta) < 1e-4:
        print("Hypothesis check: pass 2 finds no improvement.")
        print("  -> Full-battery-barrier explanation SUPPORTED.")
        print("     The greedy is at a local optimum it cannot escape by re-running.")
    else:
        print("Hypothesis check: pass 2 finds further improvement.")
        print("  -> Full-battery-barrier explanation NOT fully supported.")
        print(f"     Additional ${-delta:.4f} recovered in pass 2.")
    return 0


if __name__ == '__main__':
    sys.exit(main())