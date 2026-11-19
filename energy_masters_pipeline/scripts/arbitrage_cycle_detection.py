"""Grid arbitrage cycle extraction: the lower bound of Amber's spread.

Tracks a running
MAX import price during charging and running MIN export price during
discharging, within each session bounded by the battery's SOC touching its
floor or ceiling, on the actual reconstruction's 5-minute series directly
-- no contiguous-run grouping needed.

The output is read as a LOWER ENVELOPE (plot spread vs time, look at where
it bottoms out), not as a distribution to summarise with one statistic --
see plot_spread_over_time.

Usage (from project root):
    python3 -m energy_masters_pipeline.scripts.arbitrage_cycle_detection EM-427
    python3 -m energy_masters_pipeline.scripts.arbitrage_cycle_detection --synthetic

The opportunity-availability scan (coordination-off detection) from the
first draft of this module is deferred until this extraction is confirmed
correct on real data -- see the spec.

TESTING STATUS
--------------
The state machine (max/min tracking, floor/ceiling reset, a higher second
charge price replacing vs a lower one not, a lower second discharge price
replacing vs a higher one not, an unpaired discharge being ignored) IS
verified by a synthetic self-test exercising each rule individually
against a hand-computed expected outcome, deliberately using SCATTERED
(non-contiguous) intervals to confirm no run-grouping is needed. The
real-data path is not yet run.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_SOC_TOLERANCE_KWH = 0.05
MIN_FLOW_KWH = 0.2
"""Minimum flow to count as a genuine charge/discharge interval, not
floating-point noise. Matches the tolerance viz_energy_masters.py's own
simultaneous-charge-discharge check uses (> 1e-6) -- that check is what
established '0 simultaneous intervals' for the actual reconstruction.
This extraction originally used a bare > 0 check with no tolerance at
all, which is inconsistent with that established convention and, when
run against real EM-427 data, let floating-point-noise-level residuals
(e.g. 1e-13 kWh, numerically zero but technically positive) in BOTH
grid_to_bat and bat_to_grid at the same interval count as a genuine
same-instant charge-and-discharge -- producing charge_time == discharge_time
exactly, and a 'spread' that is really just p_exp(t) - p_imp(t) at one
instant (the tariff's built-in markup), not a market outcome. That is
almost certainly the source of the two suspiciously fixed-value clusters
seen in a real run, not a genuine tariff-driven finding.
"""


def extract_arbitrage_cycles(
    grid_to_bat_kwh: pd.Series,
    discharge_flow_kwh: pd.Series,
    battery_soc_kwh: pd.Series,
    charge_price: pd.Series,
    discharge_reference_price: pd.Series,
    capacity_kwh: float,
    soc_tolerance_kwh: float = DEFAULT_SOC_TOLERANCE_KWH,
    min_flow_kwh: float = MIN_FLOW_KWH,
) -> pd.DataFrame:
    """Extract one row per session-bounded arbitrage cycle, for ONE
    discharge pathway at a time.

    Generalised (was hardcoded to bat_to_grid_kwh / p_exp): the two real
    pathways -- grid charge -> export, and grid charge -> serve home load
    -- are economically different (export earns p_exp, serving home
    avoids paying p_imp) and must NOT be pooled into one spread number,
    since p_imp > p_exp structurally would bias a combined threshold. Call
    this once per pathway (see extract_grid_to_export_cycles /
    extract_grid_to_home_cycles below) and keep the two results separate,
    not merged.

    discharge_flow_kwh: bat_to_grid_kwh for the export pathway,
        bat_to_home_kwh for the home pathway.
    discharge_reference_price: p_exp (revenue) for export, p_imp (avoided
        cost) for home.

    A single forward
    pass over the 5-minute series: no contiguous-run grouping, since only
    a running max (charge) and running min (discharge reference price) are
    tracked. The OTHER discharge pathway is not looked at at all in a
    given call -- e.g. when extracting grid->home cycles, bat_to_grid
    activity is invisible to this pass, neither counted as a discharge nor
    triggering a session reset, so the two pathways' extractions cannot
    contaminate each other.

    Returns columns: charge_price, charge_time, discharge_price,
    discharge_time, spread. Empty (with those columns) if no cycle was
    ever completed.
    """
    idx = grid_to_bat_kwh.index
    g2b = grid_to_bat_kwh.to_numpy()
    dis = discharge_flow_kwh.to_numpy()
    soc = battery_soc_kwh.to_numpy()
    pi = charge_price.to_numpy()
    pd_ref = discharge_reference_price.to_numpy()

    floor = soc_tolerance_kwh
    ceiling = capacity_kwh - soc_tolerance_kwh

    pending_charge_price = None
    pending_charge_time = None
    candidate_discharge_price = None
    candidate_discharge_time = None

    cycles = []
    for i in range(len(idx)):
        if g2b[i] > min_flow_kwh:
            price = pi[i]
            if candidate_discharge_price is not None:
                # A charge arrives while a discharge candidate is already
                # active: treat the reversal as closing the current pair
                # (confirmed necessary against real data -- SOC
                # floor/ceiling alone let sessions run for days, mixing
                # many unrelated charge/discharge moments together).
                # Finalise if complete,
                # then start fresh from this interval.
                if pending_charge_price is not None:
                    cycles.append({
                        'charge_price': pending_charge_price,
                        'charge_time': pending_charge_time,
                        'discharge_price': candidate_discharge_price,
                        'discharge_time': candidate_discharge_time,
                        'spread': candidate_discharge_price - pending_charge_price,
                    })
                pending_charge_price = price
                pending_charge_time = idx[i]
                candidate_discharge_price = None
                candidate_discharge_time = None
            elif pending_charge_price is None or price > pending_charge_price:
                pending_charge_price = price
                pending_charge_time = idx[i]
            # else: a lower-priced charge (with no discharge candidate yet)
            # does not replace the existing (higher, worse-case) pending charge.

        if dis[i] > min_flow_kwh and pending_charge_price is not None:
            price = pd_ref[i]
            if candidate_discharge_price is None or price < candidate_discharge_price:
                candidate_discharge_price = price
                candidate_discharge_time = idx[i]
            # else: a higher-priced discharge does not replace the
            # existing (lower, worse-case) candidate.

        if soc[i] <= floor or soc[i] >= ceiling:
            if pending_charge_price is not None and candidate_discharge_price is not None:
                cycles.append({
                    'charge_price': pending_charge_price,
                    'charge_time': pending_charge_time,
                    'discharge_price': candidate_discharge_price,
                    'discharge_time': candidate_discharge_time,
                    'spread': candidate_discharge_price - pending_charge_price,
                })
            pending_charge_price = None
            pending_charge_time = None
            candidate_discharge_price = None
            candidate_discharge_time = None

    columns = ['charge_price', 'charge_time', 'discharge_price', 'discharge_time', 'spread']
    if not cycles:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(cycles, columns=columns)


def extract_grid_to_export_cycles(
    grid_to_bat_kwh: pd.Series, bat_to_grid_kwh: pd.Series, battery_soc_kwh: pd.Series,
    p_imp: pd.Series, p_exp: pd.Series, capacity_kwh: float, **kwargs,
) -> pd.DataFrame:
    """Grid charge -> export pathway: spread = p_exp(discharge) - p_imp(charge)."""
    return extract_arbitrage_cycles(
        grid_to_bat_kwh, bat_to_grid_kwh, battery_soc_kwh, p_imp, p_exp, capacity_kwh, **kwargs,
    )


def extract_grid_to_home_cycles(
    grid_to_bat_kwh: pd.Series, bat_to_home_kwh: pd.Series, battery_soc_kwh: pd.Series,
    p_imp: pd.Series, capacity_kwh: float, **kwargs,
) -> pd.DataFrame:
    """Grid charge -> serve home load pathway: spread = p_imp(avoided, at
    discharge time) - p_imp(charge time). Both sides reference p_imp --
    this measures an AVOIDED COST, not earned revenue, which is why it
    must not be pooled with the export pathway's spread.
    """
    return extract_arbitrage_cycles(
        grid_to_bat_kwh, bat_to_home_kwh, battery_soc_kwh, p_imp, p_imp, capacity_kwh, **kwargs,
    )



def summarise_spreads(cycles: pd.DataFrame) -> dict:
    """Rough anchors only -- the real read is the plot (see
    plot_spread_over_time). Reported so there is SOME number to look at
    before opening a plot, not as a replacement for looking at one.

    Includes fraction of cycles with a genuinely positive spread and the
    session-duration distribution as standing diagnostics: a real EM-427
    run found 0% positive and sessions up to 126 hours before the
    charge-after-discharge reset rule was added, which is exactly the
    kind of thing worth catching automatically rather than by manual
    inspection every time.
    """
    if len(cycles) == 0:
        return {'n_cycles': 0}
    spreads = cycles['spread'].to_numpy()
    duration_hours = (
        (pd.to_datetime(cycles['discharge_time']) - pd.to_datetime(cycles['charge_time']))
        .dt.total_seconds().to_numpy() / 3600.0
    )
    return {
        'n_cycles': len(cycles),
        'spread_min_c': float(spreads.min()) * 100,
        'spread_p5_c': float(np.percentile(spreads, 5)) * 100,
        'spread_p10_c': float(np.percentile(spreads, 10)) * 100,
        'spread_median_c': float(np.median(spreads)) * 100,
        'frac_positive': float((spreads > 0).mean()),
        'duration_median_hours': float(np.median(duration_hours)),
        'duration_p90_hours': float(np.percentile(duration_hours, 90)),
        'duration_max_hours': float(duration_hours.max()),
    }


def plot_spread_over_time(cycles: pd.DataFrame, site_id: str = '',
                          pathway: str = '', out_path: Optional[Path] = None,
                          ylim: tuple = (-10, 30)):
    """Scatter of each cycle's spread against time, for reading the lower
    envelope by eye -- this is the primary artifact, not the summary
    statistics in summarise_spreads.

    Plotted in c/kWh (spread * 100). y-axis limited to `ylim` by default:
    a few genuine losses ("mistakes") are worth keeping visible near zero,
    but extreme spike-driven outliers are deliberately clipped from view,
    since only the lower-bound clustering near zero is of interest.

    pathway: a short label ('grid->export' or 'grid->home') for the title,
    since these must never be plotted together -- see
    extract_grid_to_export_cycles / extract_grid_to_home_cycles.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))
    if len(cycles) == 0:
        ax.text(0.5, 0.5, 'No cycles extracted', transform=ax.transAxes,
                ha='center', va='center')
    else:
        t = pd.to_datetime(cycles['charge_time'])
        spread_c = cycles['spread'] * 100.0
        ax.scatter(t, spread_c, s=14, alpha=0.6, color='tab:blue')
        ax.axhline(0, color='black', linewidth=0.6, alpha=0.4)
        ordered = cycles.assign(t=t, spread_c=spread_c).sort_values('t')
        if len(ordered) >= 5:
            roll_min = ordered['spread_c'].rolling(10, min_periods=1, center=True).min()
            ax.plot(ordered['t'], roll_min, color='tab:red', linewidth=1.2,
                    label='rolling min (visual aid only)')
            ax.legend(loc='upper right')
        ax.set_ylim(*ylim)
    ax.set_ylabel('spread (c/kWh) = discharge price - charge price')
    ax.set_xlabel('charge time')
    pathway_str = f" [{pathway}]" if pathway else ""
    ax.set_title(f"{site_id}{pathway_str}: {len(cycles)} arbitrage cycle(s) -- "
                f"read the lower envelope by eye, do not just take a percentile")
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    return fig


def _report_pathway(site_id: str, pathway: str, cycles: pd.DataFrame) -> None:
    summary = summarise_spreads(cycles)
    print(f"=== {site_id} [{pathway}] arbitrage cycle extraction ===")
    print(f"{summary.get('n_cycles', 0)} cycle(s) extracted")
    if summary.get('n_cycles', 0) > 0:
        print(f"  spread (c/kWh): min={summary['spread_min_c']:.2f}  p5={summary['spread_p5_c']:.2f}  "
              f"p10={summary['spread_p10_c']:.2f}  median={summary['spread_median_c']:.2f}")
        print(f"  fraction with genuinely positive spread: {summary['frac_positive']:.2%}  "
              f"(0% would mean something is still wrong with session boundaries -- "
              "real profitable round trips should show up positive)")
        print(f"  session duration (charge to discharge): median={summary['duration_median_hours']:.1f}h  "
              f"p90={summary['duration_p90_hours']:.1f}h  max={summary['duration_max_hours']:.1f}h")
    print("These are rough anchors. Open the plot and read the lower envelope by eye --")
    print("that is the actual answer, not any single number above.")
    print()


def flow_magnitude_summary(flow_kwh: pd.Series, name: str) -> dict:
    """Percentiles of nonzero flow magnitude, for choosing min_flow_kwh
    empirically rather than guessing. If there's a distinct low cluster
    (allocation-residual/measurement noise) separate from a higher cluster
    (genuine dispatch), the gap between them is where min_flow_kwh should
    sit -- this alone won't show that shape (see the histogram in
    threshold_sweep's printed output for that), but gives the percentile
    anchors to interpret it against.
    """
    nonzero = flow_kwh[flow_kwh > 0]
    if len(nonzero) == 0:
        return {'name': name, 'n_nonzero': 0}
    return {
        'name': name,
        'n_nonzero': int(len(nonzero)),
        'min': float(nonzero.min()),
        'p1': float(nonzero.quantile(0.01)),
        'p5': float(nonzero.quantile(0.05)),
        'p10': float(nonzero.quantile(0.10)),
        'p25': float(nonzero.quantile(0.25)),
        'median': float(nonzero.median()),
    }


def print_flow_magnitude_summary(grid_to_bat_kwh: pd.Series, bat_to_grid_kwh: pd.Series,
                                 bat_to_home_kwh: pd.Series) -> None:
    print("=== Nonzero flow magnitude (kWh per interval) ===")
    print(f"{'':14s} {'n':>8s} {'min':>10s} {'p1':>10s} {'p5':>10s} {'p10':>10s} "
          f"{'p25':>10s} {'median':>10s}")
    for name, s in [('grid_to_bat', grid_to_bat_kwh), ('bat_to_grid', bat_to_grid_kwh),
                    ('bat_to_home', bat_to_home_kwh)]:
        m = flow_magnitude_summary(s, name)
        if m.get('n_nonzero', 0) == 0:
            print(f"{name:14s} {'0':>8s}  (no nonzero intervals)")
            continue
        print(f"{name:14s} {m['n_nonzero']:>8d} {m['min']:>10.6f} {m['p1']:>10.6f} "
              f"{m['p5']:>10.6f} {m['p10']:>10.6f} {m['p25']:>10.6f} {m['median']:>10.6f}")
    print("Look for a distinct LOW cluster (allocation-residual or measurement noise, not")
    print("a deliberate dispatch decision) separated by a gap from a HIGHER cluster (genuine")
    print("flows) -- min_flow_kwh should sit in that gap, not at the floating-point-noise")
    print("floor (1e-6) currently used by default.")
    print()


def threshold_sweep(
    grid_to_bat_kwh: pd.Series, discharge_flow_kwh: pd.Series, battery_soc_kwh: pd.Series,
    charge_price: pd.Series, discharge_reference_price: pd.Series, capacity_kwh: float,
    thresholds: tuple = (1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2),
) -> pd.DataFrame:
    """Re-run the extraction at several min_flow_kwh values and report
    frac_positive and n_cycles at each -- the direct test of whether
    noise-level residual flows (not floating-point noise, genuinely small
    but nonzero allocation residuals) are contaminating the real-data
    result, separately from anything about Amber's actual decisions.

    Rising frac_positive as the threshold increases, with n_cycles
    shrinking but not collapsing to near-zero, is evidence of real signal
    hidden under noise. A frac_positive that stays flat regardless of
    threshold is evidence the earlier conclusion (imperfect-foresight
    dispatch does not decompose into clean round trips) stands, not that
    noise was the explanation.
    """
    rows = []
    for th in thresholds:
        cycles = extract_arbitrage_cycles(
            grid_to_bat_kwh, discharge_flow_kwh, battery_soc_kwh,
            charge_price, discharge_reference_price, capacity_kwh, min_flow_kwh=th,
        )
        n = len(cycles)
        frac_pos = float((cycles['spread'] > 0).mean()) if n else float('nan')
        median_spread_c = float(cycles['spread'].median() * 100) if n else float('nan')
        rows.append({
            'min_flow_kwh': th, 'n_cycles': n,
            'frac_positive': frac_pos, 'median_spread_c': median_spread_c,
        })
    return pd.DataFrame(rows)


def print_threshold_sweep(site_id: str, pathway: str, sweep: pd.DataFrame) -> None:
    print(f"=== {site_id} [{pathway}] threshold sweep ===")
    print(f"{'min_flow_kwh':>14s} {'n_cycles':>10s} {'frac_positive':>14s} {'median_spread_c':>16s}")
    for _, row in sweep.iterrows():
        print(f"{row['min_flow_kwh']:>14.6f} {int(row['n_cycles']):>10d} "
              f"{row['frac_positive']:>14.1%} {row['median_spread_c']:>16.2f}")
    print()


def run_checks(site_id: str, source: str = 'actual', diagnose_noise: bool = False) -> dict:
    """Runs BOTH pathways (grid->export, grid->home) separately -- never
    combine them, see the module docstring.

    source='actual' (default): the real reconstructed behaviour.
    source='lp': the perfect-foresight simulation instead, as a CONTROL --
        the LP is optimal by construction (given its price signals,
        constraints, and c_deg_per_kwh), so any grid-charge cycle it
        performs should net out positive. A clean, mostly-positive result
        here confirms the extraction method is sound; a messy result here
        would mean the extraction itself still has a flaw, independent of
        anything about Amber's real coordination quality.

    diagnose_noise: if True, ALSO print the nonzero-flow-magnitude summary
        and a min_flow_kwh threshold sweep for both pathways, before the
        normal report. Only meaningful for source='actual' -- the LP's
        flows are a clean simulated output with nothing analogous to
        allocation-residual measurement noise to sweep for.

    Returns {'export': cycles_df, 'home': cycles_df}.
    """
    from energy_masters_pipeline.scripts.viz_energy_masters import load_and_analyze

    if source not in ('actual', 'lp'):
        raise ValueError(f"source must be 'actual' or 'lp', got {source!r}")

    analysis = load_and_analyze(site_id, simulate_lp=(source == 'lp'))
    r = analysis.lp_result if source == 'lp' else analysis.actual_result
    if r is None:
        raise ValueError(f"{site_id}: no lp_result -- load_and_analyze did not simulate it")

    if diagnose_noise:
        print_flow_magnitude_summary(r.grid_to_bat_kwh, r.bat_to_grid_kwh, r.bat_to_home_kwh)
        export_sweep = threshold_sweep(
            r.grid_to_bat_kwh, r.bat_to_grid_kwh, r.battery_soc_kwh,
            analysis.p_imp, analysis.p_exp, analysis.battery_spec.capacity_kwh,
        )
        print_threshold_sweep(site_id, 'grid->export', export_sweep)
        home_sweep = threshold_sweep(
            r.grid_to_bat_kwh, r.bat_to_home_kwh, r.battery_soc_kwh,
            analysis.p_imp, analysis.p_imp, analysis.battery_spec.capacity_kwh,
        )
        print_threshold_sweep(site_id, 'grid->home', home_sweep)
        sweep_dir = Path('data/processed/energy_masters')
        sweep_dir.mkdir(parents=True, exist_ok=True)
        export_sweep.to_csv(
            sweep_dir / f'{site_id}_threshold_sweep_export_{source}.csv', index=False)
        home_sweep.to_csv(
            sweep_dir / f'{site_id}_threshold_sweep_home_{source}.csv', index=False)

    export_cycles = extract_grid_to_export_cycles(
        r.grid_to_bat_kwh, r.bat_to_grid_kwh, r.battery_soc_kwh,
        analysis.p_imp, analysis.p_exp, analysis.battery_spec.capacity_kwh,
    )
    home_cycles = extract_grid_to_home_cycles(
        r.grid_to_bat_kwh, r.bat_to_home_kwh, r.battery_soc_kwh,
        analysis.p_imp, analysis.battery_spec.capacity_kwh,
    )

    site_label = f"{site_id} [{source}]"
    _report_pathway(site_label, 'grid->export', export_cycles)
    _report_pathway(site_label, 'grid->home', home_cycles)

    for label, cycles in [('export', export_cycles), ('home', home_cycles)]:
        csv_dir = Path('data/processed/energy_masters')
        csv_dir.mkdir(parents=True, exist_ok=True)
        out_csv = csv_dir / f'{site_id}_arbitrage_cycles_{label}_{source}.csv'
        cycles.to_csv(out_csv, index=False)
        print(f"[{label}/{source}] cycle detail written to {out_csv.resolve()}")
        png_dir = Path('energy_masters_pipeline/charts/diagnostics')
        png_dir.mkdir(parents=True, exist_ok=True)
        out_png = png_dir / f'{site_id}_arbitrage_spread_over_time_{label}_{source}.png'
        pathway_label = ('grid->export' if label == 'export' else 'grid->home') + f' [{source}]'
        plot_spread_over_time(cycles, site_id, pathway_label, out_png)

    return {'export': export_cycles, 'home': home_cycles}


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------

def _synthetic_self_test() -> None:
    print("=== Synthetic self-test ===")
    n = 300
    idx = pd.date_range('2025-06-01', periods=n, freq='5min')
    g2b = pd.Series(0.0, index=idx)
    b2g = pd.Series(0.0, index=idx)
    soc = pd.Series(5.0, index=idx)
    p_imp = pd.Series(0.20, index=idx)
    p_exp = pd.Series(0.10, index=idx)
    capacity = 10.0

    g2b.iloc[[5, 9, 14]] = 1.0
    p_imp.iloc[[5, 9, 14]] = [0.05, 0.08, 0.03]
    b2g.iloc[[30, 40]] = 1.0
    p_exp.iloc[[30, 40]] = [0.35, 0.60]
    soc.iloc[45] = capacity
    soc.iloc[46:60] = 6.0

    cycles = extract_arbitrage_cycles(g2b, b2g, soc, p_imp, p_exp, capacity)
    print(f"Case 1: {len(cycles)} cycle(s)")
    print(cycles.to_string())

    ok = True
    check1 = (len(cycles) >= 1 and abs(cycles.iloc[0]['charge_price'] - 0.08) < 1e-9
              and abs(cycles.iloc[0]['discharge_price'] - 0.35) < 1e-9)
    print(f"Case 1 (max charge=0.08, min discharge=0.35, scattered intervals): "
          f"{'PASS' if check1 else 'FAIL'}")
    ok = ok and check1

    idx2 = pd.date_range(idx[-1] + pd.Timedelta(minutes=5), periods=100, freq='5min')
    g2b2 = pd.Series(0.0, index=idx2)
    b2g2 = pd.Series(0.0, index=idx2)
    soc2 = pd.Series(5.0, index=idx2)
    p_imp2 = pd.Series(0.20, index=idx2)
    p_exp2 = pd.Series(0.10, index=idx2)

    g2b2.iloc[2] = 1.0; p_imp2.iloc[2] = 0.10
    g2b2.iloc[6] = 1.0; p_imp2.iloc[6] = 0.15
    g2b2.iloc[10] = 1.0; p_imp2.iloc[10] = 0.05
    b2g2.iloc[20] = 1.0; p_exp2.iloc[20] = 0.50
    soc2.iloc[25] = capacity

    cycles2 = extract_arbitrage_cycles(g2b2, b2g2, soc2, p_imp2, p_exp2, capacity)
    print(f"\nCase 2: {len(cycles2)} cycle(s)")
    print(cycles2.to_string())
    check2 = len(cycles2) == 1 and abs(cycles2.iloc[0]['charge_price'] - 0.15) < 1e-9
    print(f"Case 2 (higher replaces, lower does not -> charge_price=0.15): "
          f"{'PASS' if check2 else 'FAIL'}")
    ok = ok and check2

    idx3 = pd.date_range(idx2[-1] + pd.Timedelta(minutes=5), periods=100, freq='5min')
    g2b3 = pd.Series(0.0, index=idx3)
    b2g3 = pd.Series(0.0, index=idx3)
    soc3 = pd.Series(5.0, index=idx3)
    p_imp3 = pd.Series(0.20, index=idx3)
    p_exp3 = pd.Series(0.10, index=idx3)

    g2b3.iloc[2] = 1.0; p_imp3.iloc[2] = 0.06
    b2g3.iloc[10] = 1.0; p_exp3.iloc[10] = 0.40
    b2g3.iloc[15] = 1.0; p_exp3.iloc[15] = 0.55
    b2g3.iloc[20] = 1.0; p_exp3.iloc[20] = 0.30
    soc3.iloc[25] = capacity

    cycles3 = extract_arbitrage_cycles(g2b3, b2g3, soc3, p_imp3, p_exp3, capacity)
    print(f"\nCase 3: {len(cycles3)} cycle(s)")
    print(cycles3.to_string())
    check3 = len(cycles3) == 1 and abs(cycles3.iloc[0]['discharge_price'] - 0.30) < 1e-9
    print(f"Case 3 (lower discharge replaces, higher does not -> discharge_price=0.30): "
          f"{'PASS' if check3 else 'FAIL'}")
    ok = ok and check3

    idx4 = pd.date_range(idx3[-1] + pd.Timedelta(minutes=5), periods=100, freq='5min')
    g2b4 = pd.Series(0.0, index=idx4)
    b2g4 = pd.Series(0.0, index=idx4)
    soc4 = pd.Series(5.0, index=idx4)
    p_imp4 = pd.Series(0.20, index=idx4)
    p_exp4 = pd.Series(0.10, index=idx4)

    g2b4.iloc[2] = 1.0; p_imp4.iloc[2] = 0.07
    soc4.iloc[5] = capacity
    soc4.iloc[6:20] = 6.0
    g2b4.iloc[30] = 1.0; p_imp4.iloc[30] = 0.09
    b2g4.iloc[40] = 1.0; p_exp4.iloc[40] = 0.45
    soc4.iloc[45] = 0.0

    cycles4 = extract_arbitrage_cycles(g2b4, b2g4, soc4, p_imp4, p_exp4, capacity)
    print(f"\nCase 4: {len(cycles4)} cycle(s)")
    print(cycles4.to_string())
    check4 = (len(cycles4) == 1 and abs(cycles4.iloc[0]['charge_price'] - 0.09) < 1e-9
              and abs(cycles4.iloc[0]['discharge_price'] - 0.45) < 1e-9)
    print(f"Case 4 (ceiling wipes unpaired charge; floor still works afterwards): "
          f"{'PASS' if check4 else 'FAIL'}")
    ok = ok and check4

    idx5 = pd.date_range(idx4[-1] + pd.Timedelta(minutes=5), periods=50, freq='5min')
    g2b5 = pd.Series(0.0, index=idx5)
    b2g5 = pd.Series(0.0, index=idx5)
    soc5 = pd.Series(5.0, index=idx5)
    p_imp5 = pd.Series(0.20, index=idx5)
    p_exp5 = pd.Series(0.10, index=idx5)
    b2g5.iloc[5] = 1.0; p_exp5.iloc[5] = 0.90
    soc5.iloc[10] = capacity

    cycles5 = extract_arbitrage_cycles(g2b5, b2g5, soc5, p_imp5, p_exp5, capacity)
    check5 = len(cycles5) == 0
    print(f"\nCase 5 (unpaired discharge ignored, no crash): {'PASS' if check5 else 'FAIL'}")
    ok = ok and check5

    # --- Case 6: the new rule. Charge (0.06) -> discharge starts a
    # candidate (0.40) -> ANOTHER charge arrives (0.07) with NO floor/
    # ceiling touch in between. This should CLOSE the first pair
    # (0.06 -> 0.40) immediately and start a fresh pending charge at 0.07,
    # rather than merging everything into one long session. A second
    # discharge (0.50) then closes the second pair at floor/ceiling.
    idx6 = pd.date_range(idx5[-1] + pd.Timedelta(minutes=5), periods=100, freq='5min')
    g2b6 = pd.Series(0.0, index=idx6)
    b2g6 = pd.Series(0.0, index=idx6)
    soc6 = pd.Series(5.0, index=idx6)
    p_imp6 = pd.Series(0.20, index=idx6)
    p_exp6 = pd.Series(0.10, index=idx6)

    g2b6.iloc[2] = 1.0; p_imp6.iloc[2] = 0.06
    b2g6.iloc[10] = 1.0; p_exp6.iloc[10] = 0.40
    g2b6.iloc[20] = 1.0; p_imp6.iloc[20] = 0.07  # reversal: should close the first pair here
    b2g6.iloc[30] = 1.0; p_exp6.iloc[30] = 0.50
    soc6.iloc[35] = capacity  # finalises the second pair

    cycles6 = extract_arbitrage_cycles(g2b6, b2g6, soc6, p_imp6, p_exp6, capacity)
    print(f"\nCase 6: {len(cycles6)} cycle(s)")
    print(cycles6.to_string())
    check6 = (
        len(cycles6) == 2
        and abs(cycles6.iloc[0]['charge_price'] - 0.06) < 1e-9
        and abs(cycles6.iloc[0]['discharge_price'] - 0.40) < 1e-9
        and abs(cycles6.iloc[1]['charge_price'] - 0.07) < 1e-9
        and abs(cycles6.iloc[1]['discharge_price'] - 0.50) < 1e-9
    )
    print(f"Case 6 (charge-after-discharge closes the first pair, not one long session): "
          f"{'PASS' if check6 else 'FAIL'}")
    ok = ok and check6

    # --- Case 7: reproduces the real bug found against EM-427 data.
    # Genuine charge at t=2 (price 0.06), genuine discharge at t=10
    # (price 0.40). At t=5, BOTH channels carry floating-point-noise-level
    # residuals (1e-13 kWh) that are numerically zero but technically > 0
    # -- with the old bare ">0" check these register as a same-instant
    # charge AND discharge, corrupting the pairing. With the fix
    # (min_flow_kwh tolerance) they must be ignored entirely.
    idx7 = pd.date_range(idx6[-1] + pd.Timedelta(minutes=5), periods=50, freq='5min')
    g2b7 = pd.Series(0.0, index=idx7)
    b2g7 = pd.Series(0.0, index=idx7)
    soc7 = pd.Series(5.0, index=idx7)
    p_imp7 = pd.Series(0.20, index=idx7)
    p_exp7 = pd.Series(0.10, index=idx7)

    g2b7.iloc[2] = 1.0; p_imp7.iloc[2] = 0.06
    g2b7.iloc[5] = 1e-13; p_imp7.iloc[5] = 99.0    # noise-level charge at an absurd price
    b2g7.iloc[5] = 1e-13; p_exp7.iloc[5] = -99.0   # noise-level discharge at an absurd price
    b2g7.iloc[10] = 1.0; p_exp7.iloc[10] = 0.40
    soc7.iloc[15] = capacity

    cycles7 = extract_arbitrage_cycles(g2b7, b2g7, soc7, p_imp7, p_exp7, capacity)
    print(f"\nCase 7: {len(cycles7)} cycle(s)")
    print(cycles7.to_string())
    check7 = (
        len(cycles7) == 1
        and abs(cycles7.iloc[0]['charge_price'] - 0.06) < 1e-9
        and abs(cycles7.iloc[0]['discharge_price'] - 0.40) < 1e-9
        and cycles7.iloc[0]['charge_time'] != cycles7.iloc[0]['discharge_time']
    )
    print(f"Case 7 (floating-point-noise blips ignored, no same-instant corruption): "
          f"{'PASS' if check7 else 'FAIL'}")
    ok = ok and check7

    # --- Case 8: the home pathway, and non-contamination between
    # pathways. Same underlying battery activity, but this time bat_to_home
    # discharges are interleaved with bat_to_grid discharges. Extracting
    # the export pathway must be blind to the home discharges (and vice
    # versa) -- neither counted as a discharge nor triggering a session
    # reset in the OTHER pathway's extraction.
    idx8 = pd.date_range(idx6[-1] + pd.Timedelta(minutes=5), periods=100, freq='5min')
    g2b8 = pd.Series(0.0, index=idx8)
    b2h8 = pd.Series(0.0, index=idx8)   # bat_to_home
    b2g8 = pd.Series(0.0, index=idx8)   # bat_to_grid
    soc8 = pd.Series(5.0, index=idx8)
    p_imp8 = pd.Series(0.20, index=idx8)
    p_exp8 = pd.Series(0.10, index=idx8)

    g2b8.iloc[2] = 1.0; p_imp8.iloc[2] = 0.05          # charge
    b2h8.iloc[6] = 1.0; p_imp8.iloc[6] = 0.35           # home discharge (avoided cost 0.35)
    b2g8.iloc[10] = 1.0; p_exp8.iloc[10] = 0.45         # export discharge (revenue 0.45)
    soc8.iloc[15] = capacity                             # finalises both candidates at once

    export_cycles8 = extract_grid_to_export_cycles(g2b8, b2g8, soc8, p_imp8, p_exp8, capacity)
    home_cycles8 = extract_grid_to_home_cycles(g2b8, b2h8, soc8, p_imp8, capacity)
    print(f"\nCase 8 export: {len(export_cycles8)} cycle(s)")
    print(export_cycles8.to_string())
    print(f"Case 8 home: {len(home_cycles8)} cycle(s)")
    print(home_cycles8.to_string())

    check8 = (
        len(export_cycles8) == 1
        and abs(export_cycles8.iloc[0]['charge_price'] - 0.05) < 1e-9
        and abs(export_cycles8.iloc[0]['discharge_price'] - 0.45) < 1e-9
        and len(home_cycles8) == 1
        and abs(home_cycles8.iloc[0]['charge_price'] - 0.05) < 1e-9
        and abs(home_cycles8.iloc[0]['discharge_price'] - 0.35) < 1e-9
    )
    print(f"Case 8 (both pathways extracted independently from the same charge, "
          f"neither contaminated by the other's discharge events): "
          f"{'PASS' if check8 else 'FAIL'}")
    ok = ok and check8

    print(f"\n=== Overall: {'ALL PASS' if ok else 'SOME FAILED'} ===")

    import matplotlib
    matplotlib.use('Agg')
    plot_spread_over_time(cycles, site_id='SYNTH-TEST')
    print(f"\nplot_spread_over_time rendered without error: PASS")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--synthetic':
        _synthetic_self_test()
        sys.exit(0)
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('site_id')
    parser.add_argument('--source', choices=['actual', 'lp'], default='actual',
                        help="'lp' runs against the perfect-foresight simulation instead "
                             "of the real reconstruction, as a control on the extraction "
                             "method itself (the LP is optimal by construction).")
    parser.add_argument('--diagnose-noise', action='store_true',
                        help='Print nonzero-flow-magnitude percentiles and a min_flow_kwh '
                             'threshold sweep before the normal report -- checks whether '
                             'allocation-residual/measurement noise (not floating-point '
                             'noise) in the actual reconstruction is contaminating the '
                             'result, separately from anything about real dispatch decisions.')
    args = parser.parse_args()
    run_checks(args.site_id, source=args.source, diagnose_noise=args.diagnose_noise)