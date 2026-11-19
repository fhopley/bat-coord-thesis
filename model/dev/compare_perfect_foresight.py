"""Per-interval dispatch comparison: perfect_foresight_custom vs perfect_foresight_lp.

Usage (from project root):
    python3 -m model.dev.compare_perfect_foresight SITE_ID YEAR
    python3 -m model.dev.compare_perfect_foresight SITE_ID YEAR \\
        --state NSW --retail-tariff amber_wholesale_2024 \\
        --export-limit-kw 10 --opt-factor 1.0 \\
        --top-n 20 --csv results/S0088_compare.csv

Runs both the LP (HiGHS recommended; use the solver selected by lp_backend())
and the custom heuristic on identical inputs, then reports where and why the
custom solution leaves value on the table.

Output sections:
  1. Gap summary             — total $, %, gross custom-worse vs custom-better
  2. Mechanism breakdown     — how much of the gap comes from missed export vs
                               excess import vs mixed effects
  3. Hour-of-day pattern     — mean gap per clock hour (identifies peak timing)
  4. Month-of-year pattern   — seasonal concentration
  5. Top-N worst intervals   — timestamp, prices (c/kWh), Δflows, SOC, mechanism
  6. Top-5 best intervals    — where custom beats LP (offsetting wins)

An optional CSV writes the full per-interval comparison for deeper analysis
in e.g. a Jupyter notebook.
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from model.core.paths import default_paths
from model.dev._dev_common import DEV_BATTERY
from model.dev.viz_perfect_foresight_lp import load_inputs, run_lp_analysis
from model.dev.viz_perfect_foresight_custom import run_custom_analysis


# ---------------------------------------------------------------------------
# Comparison dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DispatchComparison:
    site_id: str
    year: int
    index: pd.DatetimeIndex
    # Per-interval prices ($/kWh)
    p_imp: np.ndarray
    p_exp: np.ndarray
    # Per-interval dispatch costs ($/interval): p_imp*import - p_exp*export
    cost_lp: np.ndarray
    cost_custom: np.ndarray
    # Gap: positive = custom worse at this interval
    gap: np.ndarray
    # Raw flows (kWh/interval) — kept for CSV and sub-flow inspection
    lp_import: np.ndarray
    lp_export: np.ndarray
    custom_import: np.ndarray
    custom_export: np.ndarray
    # Flow deltas: custom - lp (kWh/interval)
    delta_import: np.ndarray
    delta_export: np.ndarray
    # Gap decomposition (both positive = custom worse)
    import_contribution: np.ndarray   # p_imp * delta_import
    export_contribution: np.ndarray   # -p_exp * delta_export
    # SOC traces (kWh) — useful in top-N display
    soc_custom: np.ndarray
    soc_lp: np.ndarray
    # Scalar summary
    lp_obj: float
    custom_obj: float
    lp_solve_s: float
    custom_solve_s: float
    opt_factor: float
    export_limit_kw: float
    lp_backend: str


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def run_comparison(
    site_id: str,
    year: int,
    trace: pd.DataFrame,
    prices_mwh: pd.Series,
    network_tariff,
    retail_tariff,
    battery_spec,
    export_limit_kw: float,
    opt_factor: float = 1.0,
    interval_hours: float = 5 / 60,
) -> DispatchComparison:
    """Run both solvers and compute the per-interval comparison."""
    lp_an = run_lp_analysis(
        site_id, year, trace, prices_mwh,
        network_tariff, retail_tariff, battery_spec,
        export_limit_kw=export_limit_kw,
        interval_hours=interval_hours,
    )
    custom_an = run_custom_analysis(
        site_id, year, trace, prices_mwh,
        network_tariff, retail_tariff, battery_spec,
        export_limit_kw=export_limit_kw,
        opt_factor=opt_factor,
        interval_hours=interval_hours,
    )

    p_imp = lp_an.p_imp.to_numpy()
    p_exp = lp_an.p_exp.to_numpy()

    lp_imp = lp_an.lp_result.import_kwh.to_numpy()
    lp_exp = lp_an.lp_result.export_kwh.to_numpy()
    cu_imp = custom_an.custom_result.import_kwh.to_numpy()
    cu_exp = custom_an.custom_result.export_kwh.to_numpy()

    cost_lp     = lp_imp * p_imp - lp_exp * p_exp
    cost_custom = cu_imp * p_imp - cu_exp * p_exp
    gap         = cost_custom - cost_lp

    delta_import = cu_imp - lp_imp
    delta_export = cu_exp - lp_exp

    # Decompose the gap at each interval:
    #   gap[t] = p_imp[t]*Δimport[t]  +  (-p_exp[t])*Δexport[t]
    #          = import_contribution   +  export_contribution
    # Both positive → custom was worse on both dimensions.
    import_contribution =  p_imp * delta_import
    export_contribution = -p_exp * delta_export

    return DispatchComparison(
        site_id=site_id, year=year,
        index=lp_an.lp_result.import_kwh.index,
        p_imp=p_imp, p_exp=p_exp,
        cost_lp=cost_lp, cost_custom=cost_custom, gap=gap,
        lp_import=lp_imp, lp_export=lp_exp,
        custom_import=cu_imp, custom_export=cu_exp,
        delta_import=delta_import, delta_export=delta_export,
        import_contribution=import_contribution,
        export_contribution=export_contribution,
        soc_custom=custom_an.custom_result.battery_soc_kwh.to_numpy(),
        soc_lp=lp_an.lp_result.battery_soc_kwh.to_numpy(),
        lp_obj=float(np.sum(cost_lp)),
        custom_obj=float(np.sum(cost_custom)),
        lp_solve_s=lp_an.lp_solve_seconds,
        custom_solve_s=custom_an.solve_seconds,
        opt_factor=opt_factor,
        export_limit_kw=export_limit_kw,
        lp_backend=lp_an.backend,
    )


# ---------------------------------------------------------------------------
# Mechanism classification
# ---------------------------------------------------------------------------

# A contribution is "significant" if it accounts for at least this share of
# the interval gap (avoids labelling tiny offsetting effects as the mechanism).
_MECH_THRESHOLD = 0.25

def _mechanism(imp_c: float, exp_c: float, gap: float) -> str:
    """Classify the dominant cost mechanism at one interval."""
    if gap <= 0:
        return 'CUSTOM_BETTER'
    if abs(gap) < 1e-8:
        return 'NEUTRAL'
    imp_share = imp_c / gap if gap != 0 else 0
    exp_share = exp_c / gap if gap != 0 else 0
    both = imp_share > _MECH_THRESHOLD and exp_share > _MECH_THRESHOLD
    if both:
        return 'BOTH'
    if imp_share > _MECH_THRESHOLD:
        return 'EXCESS_IMPORT'
    if exp_share > _MECH_THRESHOLD:
        return 'MISSED_EXPORT'
    return 'OTHER'


_MECH_DESC = {
    'MISSED_EXPORT': 'custom exports less (missed battery/solar export)',
    'EXCESS_IMPORT': 'custom imports more (excess grid import)',
    'BOTH':          'both: custom imports more AND exports less',
    'OTHER':         'other / offsetting sub-interval effects',
}


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

_C = 100.0  # $/kWh → c/kWh


def print_report(cmp: DispatchComparison, top_n: int = 20) -> None:
    T = len(cmp.gap)
    total_gap = float(np.sum(cmp.gap))
    pct_gap   = 100 * total_gap / (abs(cmp.lp_obj) + 1e-9)

    worse_mask  = cmp.gap >  1e-6
    better_mask = cmp.gap < -1e-6
    n_worse  = int(worse_mask.sum())
    n_better = int(better_mask.sum())
    gross_worse  = float(cmp.gap[worse_mask].sum())
    gross_better = float(-cmp.gap[better_mask].sum())

    print()
    print(f"=== Dispatch comparison: custom vs LP ({cmp.lp_backend}) ===")
    print(f"Site {cmp.site_id}  Year {cmp.year}  "
          f"opt_factor={cmp.opt_factor:g}  export_limit={cmp.export_limit_kw:.0f} kW")
    print(f"LP solve: {cmp.lp_solve_s:.1f}s  "
          f"Custom solve: {cmp.custom_solve_s:.1f}s  T={T:,}")

    # --- 1. Summary ---
    print()
    print("--- Gap summary ---")
    print(f"LP objective:     ${cmp.lp_obj:>11,.4f}")
    print(f"Custom objective: ${cmp.custom_obj:>11,.4f}")
    print(f"Net gap (custom - LP): ${total_gap:>+,.4f}  ({pct_gap:+.2f}% of LP)")
    print(f"Intervals custom worse:  {n_worse:>7,}  ({100*n_worse/T:.1f}%)  "
          f"gross ${gross_worse:,.2f}")
    print(f"Intervals custom better: {n_better:>7,}  ({100*n_better/T:.1f}%)  "
          f"gross ${gross_better:,.2f}  (partially offsets)")
    print(f"Intervals equal:         {T-n_worse-n_better:>7,}  ({100*(T-n_worse-n_better)/T:.1f}%)")

    # --- 2. Mechanism breakdown ---
    print()
    print("--- Mechanism breakdown (intervals where custom is worse) ---")
    mechs = {}
    for i in np.where(worse_mask)[0]:
        m = _mechanism(float(cmp.import_contribution[i]),
                       float(cmp.export_contribution[i]),
                       float(cmp.gap[i]))
        mechs[m] = mechs.get(m, 0.0) + float(cmp.gap[i])

    imp_total = float(cmp.import_contribution[worse_mask].sum())
    exp_total = float(cmp.export_contribution[worse_mask].sum())
    for mech in ['MISSED_EXPORT', 'EXCESS_IMPORT', 'BOTH', 'OTHER']:
        val = mechs.get(mech, 0.0)
        pct = 100 * val / (gross_worse + 1e-9)
        if abs(val) > 0.001:
            print(f"  {mech:<16}  ${val:>7,.2f}  ({pct:5.1f}%)  "
                  f"— {_MECH_DESC[mech]}")
    print(f"  (import contribution total: ${imp_total:>+,.2f}  "
          f"export contribution total: ${exp_total:>+,.2f})")

    # --- 3. Hour-of-day pattern ---
    print()
    print("--- Gap by hour of day (mean ¢/interval across year) ---")
    hours = cmp.index.hour
    hour_gap_c = cmp.gap * _C  # convert to cents
    print("  Hr  mean¢  |  Hr  mean¢  |  Hr  mean¢  |  Hr  mean¢  |  Hr  mean¢  |  Hr  mean¢")
    row = ""
    for h in range(24):
        mask_h = hours == h
        mean_c = float(hour_gap_c[mask_h].mean()) if mask_h.any() else 0.0
        col = f"  {h:02d}  {mean_c:+.3f}"
        row += col
        if (h + 1) % 4 == 0:
            print(row)
            row = ""

    # --- 4. Month pattern ---
    print()
    print("--- Gap by month (total $ custom worse / better) ---")
    months = cmp.index.month
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    header = "  ".join(f"{m:>5}" for m in month_names)
    values = []
    for m in range(1, 13):
        mask_m = months == m
        values.append(float(cmp.gap[mask_m].sum()))
    print("  " + header)
    print("  " + "  ".join(f"{v:>+5.2f}" for v in values))

    # --- 5. Top-N worst intervals ---
    print()
    print(f"--- Top {top_n} worst intervals (custom most expensive vs LP) ---")
    _print_top_intervals(cmp, top_n, worst=True)

    # --- 6. Top-5 best intervals (offsetting wins) ---
    print()
    print("--- Top 5 best intervals (custom cheapest vs LP — offsetting wins) ---")
    _print_top_intervals(cmp, 5, worst=False)

    print()


def _print_top_intervals(cmp: DispatchComparison, n: int, worst: bool) -> None:
    gap = cmp.gap
    if worst:
        order = np.argsort(-gap)
        subset = order[gap[order] > 1e-8][:n]
    else:
        order = np.argsort(gap)
        subset = order[gap[order] < -1e-8][:n]

    if len(subset) == 0:
        print("  (none)")
        return

    hdr = (f"  {'Timestamp':<20} {'p_imp':>6} {'p_exp':>6} "
           f"{'Δimp':>6} {'Δexp':>6} "
           f"{'imp¢':>6} {'exp¢':>6} {'gap$':>6} "
           f"{'cSOC':>5} {'lSOC':>5}  mechanism")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for i in subset:
        ts  = cmp.index[i]
        pi  = cmp.p_imp[i] * _C
        pe  = cmp.p_exp[i] * _C
        di  = cmp.delta_import[i]
        de  = cmp.delta_export[i]
        ic  = cmp.import_contribution[i] * _C
        ec  = cmp.export_contribution[i] * _C
        g   = cmp.gap[i]
        cs  = cmp.soc_custom[i]
        ls  = cmp.soc_lp[i]
        mech = _mechanism(float(cmp.import_contribution[i]),
                          float(cmp.export_contribution[i]), float(g))
        ts_str = str(ts)[:16]
        print(f"  {ts_str:<20} {pi:>6.1f} {pe:>6.1f} "
              f"{di:>+6.3f} {de:>+6.3f} "
              f"{ic:>+6.2f} {ec:>+6.2f} {g:>+6.3f} "
              f"{cs:>5.2f} {ls:>5.2f}  {mech}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(cmp: DispatchComparison, path: Path) -> None:
    """Write the full per-interval comparison to CSV."""
    mechs = [
        _mechanism(float(ic), float(ec), float(g))
        for ic, ec, g in zip(cmp.import_contribution,
                              cmp.export_contribution, cmp.gap)
    ]
    df = pd.DataFrame({
        'datetime':                   cmp.index,
        'hour':                       cmp.index.hour,
        'month':                      cmp.index.month,
        'p_imp_dollar_kwh':           cmp.p_imp,
        'p_exp_dollar_kwh':           cmp.p_exp,
        'lp_import_kwh':              cmp.lp_import,
        'lp_export_kwh':              cmp.lp_export,
        'custom_import_kwh':          cmp.custom_import,
        'custom_export_kwh':          cmp.custom_export,
        'delta_import_kwh':           cmp.delta_import,
        'delta_export_kwh':           cmp.delta_export,
        'import_contribution_dollar': cmp.import_contribution,
        'export_contribution_dollar': cmp.export_contribution,
        'gap_dollar':                 cmp.gap,
        'soc_custom_kwh':             cmp.soc_custom,
        'soc_lp_kwh':                 cmp.soc_lp,
        'mechanism':                  mechs,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Full per-interval comparison saved to {path}  ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('site_id')
    parser.add_argument('year', type=int)
    parser.add_argument('--state', default='NSW')
    parser.add_argument('--network-tariff', default='ausgrid_ea025_network_2024')
    parser.add_argument('--retail-tariff', default='amber_wholesale_2024')
    parser.add_argument('--export-limit-kw', type=float, default=10.0)
    parser.add_argument('--opt-factor', type=float, default=1.0)
    parser.add_argument('--top-n', type=int, default=20,
                        help='Number of worst intervals to display (default 20)')
    parser.add_argument('--csv', type=Path, default=None,
                        help='Save full per-interval comparison to this CSV path')
    parser.add_argument('--project-root', type=Path, default=None)
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    trace, prices_mwh, network_tariff, retail_tariff = load_inputs(
        args.site_id, args.year, args.state,
        args.network_tariff, args.retail_tariff, paths,
    )

    cmp = run_comparison(
        args.site_id, args.year, trace, prices_mwh,
        network_tariff, retail_tariff, DEV_BATTERY,
        export_limit_kw=args.export_limit_kw,
        opt_factor=args.opt_factor,
    )

    print_report(cmp, top_n=args.top_n)

    if args.csv:
        save_csv(cmp, args.csv)

    return 0


if __name__ == '__main__':
    sys.exit(main())
