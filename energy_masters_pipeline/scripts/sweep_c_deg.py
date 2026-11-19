"""Sweep c_deg_per_kwh and fit it against actual battery use.

Reverse-engineering Amber's degradation cost from extracted arbitrage
cycles (arbitrage_cycle_detection.py) gave mixed results, so this script
calibrates it instead: solve perfect_foresight_lp at several candidate c_deg_per_kwh values and
compare its battery-use metrics against the actual (Amber-coordinated)
reconstruction for the SAME real sites and periods. Where the LP matches
actual's level of battery use is the c_deg that puts the benchmark on
comparable terms, which is the right footing for a fair LP-vs-actual
comparison regardless of what Amber's own internal cost assumption is.

ONE VALUE ACROSS THE COHORT, not one per site. c_deg here stands in for
Amber's arbitrage threshold, a property of the coordinator rather than of
any individual site, and a per-site fit would overfit badly given how much
a single site's numbers move on one extreme price day.

Four metrics, one panel each:
  - total throughput (C + Q): HEADLINE. The LP charges c_deg per kWh of
    charge plus discharge, so throughput is the quantity c_deg acts on
    directly.
  - AC charge (C = sol_to_bat + grid_to_bat): the charge side of throughput.
  - grid_to_bat: arbitrage charging from the grid, a cross-check.
  - bat_to_grid: arbitrage discharging, a cross-check.

All four are reported so their agreement can be judged; agreement between
them is evidence, disagreement says something is wrong upstream.

Actual's value does not depend on c_deg at all, so per site it is a single
fixed number and only the LP curve moves. The cohort chart plots the RATIO
LP/actual so sites of different sizes share an axis and the answer is read
off one reference line at 1.0.

Usage (from project root):
    python3 -m energy_masters_pipeline.scripts.sweep_c_deg                 # whole cohort, pooled fit
    python3 -m energy_masters_pipeline.scripts.sweep_c_deg EM-427          # one site
    python3 -m energy_masters_pipeline.scripts.sweep_c_deg --sites EM-427,EM-414
    python3 -m energy_masters_pipeline.scripts.sweep_c_deg --c-deg-values 0.015,0.03,0.06,0.09,0.12
    python3 -m energy_masters_pipeline.scripts.sweep_c_deg --from-csv      # redraw from the last cohort CSV

Cohort runs also write a single-pane thesis figure of the total throughput
ratio (--thesis-out, default energy_masters_pipeline/charts/
thesis_c_deg_throughput.png, format per docs/thesis_figure_spec.md). --from-csv rebuilds the
tables and both charts from the per-site CSV without re-solving, so the
figure can be restyled without the LP run.

Runtime: each site/c_deg pair is a full load_and_analyze including a fresh
LP solve, so a 10-site 5-value run is tens of minutes. The ingest and
self-consumption work is repeated for every c_deg even though only the LP
depends on it; splitting that out would cut the time several-fold and is a
deliberate non-goal for now.

CAVEAT: the fit is only as good as the per-site overrides feeding it.
Power and export-limit estimates that are a few percent high inflate LP
throughput, which is exactly the quantity being matched. Until every site
has been reviewed by hand, treat the result as provisional and re-run
after the review pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.scripts import thesis_style as ts

from energy_masters_pipeline.scripts.viz_energy_masters import (
    load_and_analyze, resolve_curtailment_arg, describe_curtailment,
    CURTAILMENT_ARG_HELP, DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS,
    COLOUR_ACTUAL, COLOUR_LP,
)

# DEFAULT_C_DEG_VALUES = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07)
DEFAULT_C_DEG_VALUES = (0.015, 0.03, 0.06, 0.09, 0.12)
DEFAULT_COHORT_CSV = Path('data/processed/energy_masters/c_deg_sweep_cohort.csv')
DEFAULT_THESIS_CHART = Path('energy_masters_pipeline/charts/thesis_c_deg_throughput.png')

# The metric the final c_deg is read from. The LP's degradation term is
# c_deg * (charge + discharge), so total throughput is the quantity c_deg
# acts on directly. The other three are computed and charted as cross-checks.
HEADLINE_METRIC = 'total_throughput_C+Q_kwh'

# The metric the thesis figure is drawn from.
THESIS_METRIC = HEADLINE_METRIC

# Thesis figure height (in); width and format come from thesis_style.
THESIS_HEIGHT_IN = 3.0

# Metric name -> function computing it from a SiteTraces.
_METRICS = {
    'total_throughput_C+Q_kwh': lambda tr: float(
        (tr.sol_to_bat_kwh + tr.grid_to_bat_kwh
         + tr.bat_to_home_kwh + tr.bat_to_grid_kwh).sum()
    ),
    'ac_charge_C_kwh': lambda tr: float(
        (tr.sol_to_bat_kwh + tr.grid_to_bat_kwh).sum()
    ),
    'grid_to_bat_kwh': lambda tr: float(tr.grid_to_bat_kwh.sum()),
    'bat_to_grid_kwh': lambda tr: float(tr.bat_to_grid_kwh.sum()),
}


@dataclass
class SiteSweep:
    """One site's sweep: LP metrics per c_deg, plus actual's fixed values."""
    site_id: str
    lp: pd.DataFrame                      # index c_deg_per_kwh, one column per metric
    actual: Optional[dict]                # metric -> actual's single value
    failures: list                        # (c_deg, exception) for solves that failed


def sweep_c_deg(
    site_id: str,
    c_deg_values: tuple[float, ...] = DEFAULT_C_DEG_VALUES,
    export_limit_kw: Optional[float] = None,
    curtail_cost_per_kwh_dollars: Optional[float] = DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS,
    quiet: bool = False,
) -> SiteSweep:
    """Solve the LP at each c_deg value and collect the four metrics.

    Fail-soft PER c_deg VALUE. With curtailment pinned by default a solve can
    legitimately come back infeasible, and losing an entire site's sweep to
    one bad value would be a poor trade. Failed values become NaN rows and
    are reported; they are not silently dropped, because a gap changes which
    sites are eligible for pooling.
    """
    actual_values: Optional[dict] = None
    failures: list = []
    rows = []
    for c_deg in c_deg_values:
        if not quiet:
            print(f"  solving LP at c_deg_per_kwh={c_deg} ...", flush=True)
        try:
            analysis = load_and_analyze(
                site_id, simulate_lp=True, export_limit_kw=export_limit_kw,
                curtail_cost_per_kwh_dollars=curtail_cost_per_kwh_dollars,
                c_deg_per_kwh_override=c_deg,
            )
        except Exception as e:
            failures.append((c_deg, e))
            print(f"    FAILED at c_deg={c_deg}: {type(e).__name__}: {e}")
            row = {name: float('nan') for name in _METRICS}
            row['c_deg_per_kwh'] = c_deg
            rows.append(row)
            continue
        if actual_values is None:
            # Actual doesn't depend on c_deg -- computed once, from whichever
            # call happens to succeed first.
            actual_values = {
                name: fn(analysis.actual_result) for name, fn in _METRICS.items()
            }
        row = {name: fn(analysis.lp_result) for name, fn in _METRICS.items()}
        row['c_deg_per_kwh'] = c_deg
        rows.append(row)

    lp = pd.DataFrame(rows).set_index('c_deg_per_kwh')
    return SiteSweep(site_id=site_id, lp=lp, actual=actual_values, failures=failures)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def implied_c_deg(c_deg_values, ratios) -> tuple:
    """Where the LP/actual ratio crosses 1.0, linearly interpolated.

    Returns (value_or_None, status). The ratio is expected to fall with
    c_deg -- a higher degradation cost means less cycling -- so the crossing
    is the c_deg at which the LP uses the battery as much as actual did.

    Reports rather than hides the awkward cases: a curve that never crosses
    means the sweep grid does not bracket the answer, and more than one
    crossing means the curve is not monotone, which is a signal about the
    data rather than something to average away.
    """
    xs = np.asarray(list(c_deg_values), dtype=float)
    ys = np.asarray(list(ratios), dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if len(xs) < 2:
        return None, 'too few points'

    d = ys - 1.0
    if np.all(d > 0):
        return None, f'no crossing: LP still above actual at c_deg={xs.max():g}'
    if np.all(d < 0):
        return None, f'no crossing: LP already below actual at c_deg={xs.min():g}'

    crossings = []
    for i in range(len(xs) - 1):
        a, b = d[i], d[i + 1]
        if a == 0.0:
            crossings.append(xs[i])
        elif a * b < 0:
            crossings.append(xs[i] + (xs[i + 1] - xs[i]) * (-a) / (b - a))
    if d[-1] == 0.0:
        crossings.append(xs[-1])
    # Dedupe values that coincide at a shared grid point.
    deduped = []
    for c in crossings:
        if not deduped or abs(c - deduped[-1]) > 1e-12:
            deduped.append(c)
    if not deduped:
        return None, 'no crossing found'
    status = 'ok' if len(deduped) == 1 else f'{len(deduped)} crossings, using the lowest'
    return float(deduped[0]), status


def build_long_table(sweeps: list) -> pd.DataFrame:
    """Tidy site x c_deg x metric table with LP, actual, and their ratio."""
    rows = []
    for sw in sweeps:
        if sw.actual is None:
            continue   # every solve for this site failed; nothing to compare
        for metric in _METRICS:
            actual = sw.actual[metric]
            for c_deg, lp_value in sw.lp[metric].items():
                rows.append({
                    'site_id': sw.site_id,
                    'c_deg_per_kwh': float(c_deg),
                    'metric': metric,
                    'lp_kwh': float(lp_value),
                    'actual_kwh': float(actual),
                    # A site that never did the thing being measured has no
                    # meaningful ratio -- NaN rather than a division blow-up.
                    'ratio': (float(lp_value) / float(actual)
                              if abs(actual) > 1e-9 else float('nan')),
                })
    return pd.DataFrame(rows)


def pooled_ratio(long: pd.DataFrame, metric: str) -> tuple:
    """Pooled LP/actual ratio per c_deg, and the sites it was pooled over.

    Sums kWh across sites rather than averaging per-site ratios. A mean of
    ratios silently overweights sites with small denominators, which is the
    same trap run_cohort_attribution.py already documents for cohort VCR.
    Pooling is therefore energy-weighted: a bigger battery counts for more.

    Only sites with a usable value at EVERY c_deg are pooled. Otherwise the
    pooled curve would step whenever the site set changed, and that step
    would look exactly like a real response to c_deg.
    """
    sub = long[long['metric'] == metric]
    if sub.empty:
        return pd.Series(dtype=float), [], []
    lp_piv = sub.pivot(index='c_deg_per_kwh', columns='site_id', values='lp_kwh')
    actual_by_site = sub.groupby('site_id')['actual_kwh'].first()

    usable = lp_piv.notna().all(axis=0) & (actual_by_site.abs() > 1e-9).reindex(lp_piv.columns)
    complete = sorted(lp_piv.columns[usable])
    dropped = sorted(set(lp_piv.columns) - set(complete))
    if not complete:
        return pd.Series(dtype=float), [], dropped

    pooled = lp_piv[complete].sum(axis=1) / actual_by_site[complete].sum()
    return pooled.sort_index(), complete, dropped


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_sweep_table(sweep: SiteSweep) -> None:
    print()
    print(f"=== {sweep.site_id}: c_deg sweep, LP vs actual (kWh) ===")
    if sweep.actual is None:
        print("  every solve failed; nothing to report")
        return
    for name in _METRICS:
        print(f"\n--- {name} ---  (actual: {sweep.actual[name]:,.1f})")
        for c_deg, lp_value in sweep.lp[name].items():
            if not np.isfinite(lp_value):
                print(f"  c_deg={c_deg:<7.3f} LP=    FAILED")
                continue
            delta = lp_value - sweep.actual[name]
            ratio = (lp_value / sweep.actual[name]
                     if abs(sweep.actual[name]) > 1e-9 else float('nan'))
            print(f"  c_deg={c_deg:<7.3f} LP={lp_value:>10,.1f}   "
                  f"delta={delta:>+10,.1f}   ratio={ratio:>6.3f}")


def print_cohort_table(long: pd.DataFrame, c_deg_values) -> dict:
    """Per-metric implied c_deg, pooled and per site. Returns the pooled series."""
    pooled_by_metric = {}
    print()
    print("=== Implied c_deg_per_kwh: where LP battery use matches actual ===")
    print("Pooled = sum(LP kWh) / sum(actual kWh) across sites complete at every")
    print("c_deg value, NOT a mean of per-site ratios (which would overweight")
    print("sites with small denominators).")

    for metric in _METRICS:
        pooled, complete, dropped = pooled_ratio(long, metric)
        pooled_by_metric[metric] = pooled
        tag = '  [HEADLINE]' if metric == HEADLINE_METRIC else ''
        print(f"\n--- {metric}{tag} ---")
        if pooled.empty:
            print("  no sites usable for this metric")
            continue
        value, status = implied_c_deg(pooled.index, pooled.values)
        shown = 'none' if value is None else f"{value:.4f}"
        print(f"  pooled implied c_deg: {shown}   ({status})")
        print(f"  pooled over {len(complete)} site(s): {', '.join(complete)}")
        if dropped:
            print(f"  EXCLUDED from pooling ({len(dropped)}): {', '.join(dropped)}")
            print("    (a failed solve or a zero actual denominator at some c_deg)")

        sub = long[long['metric'] == metric]
        per_site = []
        for site_id, grp in sub.groupby('site_id'):
            g = grp.sort_values('c_deg_per_kwh')
            v, st = implied_c_deg(g['c_deg_per_kwh'], g['ratio'])
            per_site.append((site_id, v, st))
        finite = [v for _, v, _ in per_site if v is not None]
        if finite:
            print(f"  per-site spread: {min(finite):.4f} to {max(finite):.4f} "
                  f"({len(finite)} of {len(per_site)} sites cross)")
        for site_id, v, st in per_site:
            shown = f"{v:.4f}" if v is not None else "   --  "
            note = '' if st == 'ok' else f"   {st}"
            print(f"    {site_id:<8} {shown}{note}")

    return pooled_by_metric


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_sweep(sweep: SiteSweep, out_path: Optional[Path] = None) -> None:
    """Single-site absolute view: LP curve against actual's flat reference."""
    if sweep.actual is None:
        print("(no chart: every solve failed)")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ax, name in zip(axes, _METRICS):
        ax.plot(sweep.lp.index, sweep.lp[name], marker='o', color=COLOUR_LP,
                label='perfect foresight (LP)')
        ax.axhline(sweep.actual[name], color=COLOUR_ACTUAL, linestyle='--',
                   label='actual (Amber-coordinated)')
        ax.set_title(name + ('  [headline]' if name == HEADLINE_METRIC else ''))
        ax.set_xlabel('c_deg_per_kwh ($/kWh)')
        ax.set_ylabel('kWh (period total)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{sweep.site_id}: battery-use metrics vs c_deg_per_kwh")
    fig.tight_layout()
    _finish(fig, out_path)


def plot_cohort_ratio(long: pd.DataFrame, pooled_by_metric: dict,
                      out_path: Optional[Path] = None) -> None:
    """Cohort view: LP/actual ratio per site, plus the pooled curve.

    Ratio rather than kWh so sites of very different sizes share an axis, and
    so the answer is read off a single reference line at 1.0. Individual site
    lines are kept rather than collapsed to a box plot: the question is how
    CONSISTENT the sites are, and whether one site is dragging the pooled
    curve, neither of which survives aggregation.
    """
    sites = sorted(long['site_id'].unique())
    cmap = plt.get_cmap('tab20' if len(sites) > 10 else 'tab10')
    colours = {s: cmap(i % cmap.N) for i, s in enumerate(sites)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    for ax, metric in zip(axes, _METRICS):
        sub = long[long['metric'] == metric]
        for site_id in sites:
            g = sub[sub['site_id'] == site_id].sort_values('c_deg_per_kwh')
            ax.plot(g['c_deg_per_kwh'], g['ratio'], color=colours[site_id],
                    linewidth=1.1, alpha=0.8, label=site_id)
        pooled = pooled_by_metric.get(metric)
        if pooled is not None and not pooled.empty:
            ax.plot(pooled.index, pooled.values, color='black', linewidth=2.6,
                    zorder=5, label='pooled')
            value, _ = implied_c_deg(pooled.index, pooled.values)
            if value is not None:
                ax.axvline(value, color='black', linestyle=':', linewidth=1.2)
                ax.annotate(f"{value:.4f}", xy=(value, 1.0), xytext=(5, 8),
                            textcoords='offset points', fontsize=9)
        ax.axhline(1.0, color=COLOUR_ACTUAL, linestyle='--', linewidth=1.3)
        ax.set_title(metric + ('  [headline]' if metric == HEADLINE_METRIC else ''))
        ax.set_xlabel('c_deg_per_kwh ($/kWh)')
        ax.set_ylabel('LP / actual')
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(7, len(labels)),
               fontsize=8, frameon=False)
    fig.suptitle('Cohort: LP battery use relative to actual, vs c_deg_per_kwh '
                 '(crossing 1.0 = same battery use)')
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    _finish(fig, out_path)


def plot_thesis_throughput(long: pd.DataFrame,
                           out_path: Optional[Path] = DEFAULT_THESIS_CHART
                           ) -> None:
    """Total throughput ratio on one pane, in the thesis figure format
    (docs/thesis_figure_spec.md).

    One colour per site, the pooled cohort curve in black over the sites,
    a two-column legend in the upper right with the pooled curve at the foot
    of the first column, and the pooled crossing of 1.0 marked and labelled.
    """
    sub = long[long['metric'] == THESIS_METRIC]
    if sub.empty:
        print("(no thesis chart: no total throughput results)")
        return
    pooled, complete, _ = pooled_ratio(long, THESIS_METRIC)

    sites = sorted(sub['site_id'].unique())
    if len(sites) > len(ts.CATEGORICAL_COLOURS):
        raise ValueError(f"{len(sites)} sites but only "
                         f"{len(ts.CATEGORICAL_COLOURS)} categorical colours defined")

    ts.apply_style()
    fig = ts.new_figure(THESIS_HEIGHT_IN)
    ax = fig.subplots()
    for site_id, colour in zip(sites, ts.CATEGORICAL_COLOURS):
        g = sub[sub['site_id'] == site_id].sort_values('c_deg_per_kwh')
        ax.plot(g['c_deg_per_kwh'], g['ratio'], color=colour,
                lw=ts.MEMBER_LW, zorder=2, label=site_id)
    ax.axhline(1.0, color=ts.REFERENCE_COLOUR, ls='--', lw=ts.REFERENCE_LW,
               zorder=1)
    if not pooled.empty:
        ax.plot(pooled.index, pooled.values, color=ts.EMPHASIS_COLOUR,
                lw=ts.EMPHASIS_LW, marker='o', zorder=4, label='Pooled')
        value, _ = implied_c_deg(pooled.index, pooled.values)
        if value is not None:
            ax.axvline(value, color=ts.REFERENCE_COLOUR, ls='--',
                       lw=ts.REFERENCE_LW, zorder=1)
            # Labelled at the crossing it identifies. The crossing sits in
            # the bundle of site lines, so the label takes a white box.
            ax.annotate(f"{value:.3f} \\$/kWh", xy=(value, 1.0),
                        xytext=(4, 5), textcoords='offset points',
                        ha='left', va='bottom', fontsize=ts.ANNOTATION_SIZE,
                        color=ts.REFERENCE_COLOUR, zorder=6,
                        bbox=ts.LABEL_BOX)

    xs = np.sort(sub['c_deg_per_kwh'].unique())
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    pad = 0.03 * (xs.max() - xs.min())
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_xlabel(r'Degradation cost $c_{\mathrm{deg}}$ (\$/kWh)')
    ax.set_ylabel('LP / actual battery throughput')
    ax.set_title(r'LP vs actual battery throughput by $c_{\mathrm{deg}}$')
    # matplotlib fills legend columns top to bottom, so with two columns and
    # nine entries the fifth handle is the foot of the first column. Putting
    # 'Pooled' there leaves the eight sites as two aligned columns of four.
    handles, labels = ax.get_legend_handles_labels()
    if 'Pooled' in labels:
        k = labels.index('Pooled')
        h, l = handles.pop(k), labels.pop(k)
        half = (len(labels) + 1) // 2
        handles.insert(half, h)
        labels.insert(half, l)
    leg = ax.legend(handles, labels, loc='upper right', ncol=2)
    leg.set_zorder(20)

    if out_path is None:
        plt.show()
        return
    ts.save_figure(fig, out_path)


def _finish(fig, out_path: Optional[Path]) -> None:
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"\nChart written to {out_path.resolve()}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_sweep(
    site_id: str,
    c_deg_values: tuple[float, ...] = DEFAULT_C_DEG_VALUES,
    export_limit_kw: Optional[float] = None,
    curtail_cost_per_kwh_dollars: Optional[float] = DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS,
    out_path: Optional[Path] = None,
) -> SiteSweep:
    print(f"Sweeping c_deg_per_kwh for {site_id}: {list(c_deg_values)}")
    print(describe_curtailment(curtail_cost_per_kwh_dollars))
    sweep = sweep_c_deg(
        site_id, c_deg_values=c_deg_values, export_limit_kw=export_limit_kw,
        curtail_cost_per_kwh_dollars=curtail_cost_per_kwh_dollars,
    )
    print_sweep_table(sweep)
    plot_sweep(sweep, out_path=out_path)
    return sweep


def run_cohort_sweep(
    sites: Optional[list] = None,
    c_deg_values: tuple[float, ...] = DEFAULT_C_DEG_VALUES,
    curtail_cost_per_kwh_dollars: Optional[float] = DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS,
    out_path: Optional[Path] = None,
    csv_path: Optional[Path] = DEFAULT_COHORT_CSV,
    thesis_out: Optional[Path] = DEFAULT_THESIS_CHART,
) -> pd.DataFrame:
    """Sweep every accepted site and fit one c_deg across the cohort.

    One fitted value, not one per site: c_deg here is standing in for
    Amber's arbitrage threshold, which is a property of the COORDINATOR
    rather than of any site. Fitting per site would also overfit badly given
    how much a single site's numbers move on one extreme price day.

    No per-site export_limit_kw parameter: the whole point is to run every
    site under its own reviewed settings, which now come from the review CSV
    (export_kw_override) rather than from a single command line value that
    could only ever be right for one site.
    """
    # Deferred: keeps this module's import cheap, and accepted_sites is the
    # existing single source of truth for which sites count. It arguably
    # belongs in capacity_review, and could move there during the rename.
    from energy_masters_pipeline.scripts.run_cohort_attribution import accepted_sites

    if sites is None:
        sites = accepted_sites()
    print(f"Cohort c_deg sweep: {len(sites)} site(s), "
          f"{len(c_deg_values)} c_deg value(s) = {len(sites) * len(c_deg_values)} LP solves")
    print(f"Sites: {', '.join(sites)}")
    print(describe_curtailment(curtail_cost_per_kwh_dollars))
    print()

    sweeps = []
    for i, site_id in enumerate(sites, 1):
        print(f"[{i}/{len(sites)}] {site_id}", flush=True)
        try:
            sweeps.append(sweep_c_deg(
                site_id, c_deg_values=c_deg_values,
                curtail_cost_per_kwh_dollars=curtail_cost_per_kwh_dollars,
                quiet=True,
            ))
        except Exception as e:
            # Fail-soft at site level too: a site can fail before any solve
            # (unreviewed, excluded, missing data) and must not lose the rest.
            print(f"    SITE FAILED: {type(e).__name__}: {e}")

    long = build_long_table(sweeps)
    if long.empty:
        print("\nNo usable results.")
        return long

    solve_failures = [(sw.site_id, c, e) for sw in sweeps for c, e in sw.failures]
    if solve_failures:
        print(f"\n{len(solve_failures)} individual solve(s) failed:")
        for site_id, c_deg, e in solve_failures:
            print(f"  {site_id} at c_deg={c_deg}: {type(e).__name__}: {e}")

    pooled_by_metric = print_cohort_table(long, c_deg_values)

    if csv_path is not None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        long.to_csv(csv_path, index=False)
        print(f"\nPer-site detail written to {Path(csv_path).resolve()}")

    # Thesis chart first: plot_cohort_ratio blocks on plt.show() when no
    # --out is given.
    if thesis_out is not None:
        plot_thesis_throughput(long, out_path=thesis_out)
    plot_cohort_ratio(long, pooled_by_metric, out_path=out_path)
    return long


def redraw_from_csv(
    csv_path: Path = DEFAULT_COHORT_CSV,
    out_path: Optional[Path] = None,
    thesis_out: Optional[Path] = DEFAULT_THESIS_CHART,
) -> pd.DataFrame:
    """Rebuild the cohort table and both charts from a saved per-site CSV."""
    long = pd.read_csv(csv_path)
    c_deg_values = sorted(long['c_deg_per_kwh'].unique())
    print(f"Read {csv_path}: {long['site_id'].nunique()} site(s), "
          f"c_deg values {[float(v) for v in c_deg_values]}")
    pooled_by_metric = print_cohort_table(long, c_deg_values)
    if thesis_out is not None:
        plot_thesis_throughput(long, out_path=thesis_out)
    plot_cohort_ratio(long, pooled_by_metric, out_path=out_path)
    return long


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('site_id', nargs='?', default=None,
                        help='Single site, e.g. EM-427. Omit to sweep the whole '
                             'cohort and fit one pooled c_deg across it.')
    parser.add_argument('--sites', type=str, default=None,
                        help='Comma-separated site list for a cohort run. '
                             'Default: all accept/override sites in the review CSV.')
    parser.add_argument('--c-deg-values', type=str, default=None,
                        help='Comma-separated c_deg_per_kwh values to sweep. '
                             f'Default: {",".join(str(v) for v in DEFAULT_C_DEG_VALUES)}')
    parser.add_argument('--export-limit-kw', type=float, default=None,
                        help='Single-site runs only. Cohort runs take each site\'s '
                             'export limit from the review CSV.')
    parser.add_argument('--curtailment', type=str, default=None, metavar='MODE',
                        help=CURTAILMENT_ARG_HELP)
    parser.add_argument('--out', type=Path, default=None,
                        help='Save chart to this path instead of opening it interactively.')
    parser.add_argument('--csv', type=Path, default=DEFAULT_COHORT_CSV,
                        help='Cohort runs only: per-site detail CSV.')
    parser.add_argument('--thesis-out', type=Path, default=DEFAULT_THESIS_CHART,
                        help='Cohort runs only: path for the single-pane total '
                             'throughput figure. '
                             f'Default: {DEFAULT_THESIS_CHART}')
    parser.add_argument('--from-csv', action='store_true',
                        help='Redraw the cohort table and charts from --csv '
                             'without solving.')
    args = parser.parse_args()

    if args.from_csv:
        if args.site_id is not None or args.sites is not None or args.c_deg_values:
            parser.error('--from-csv takes no site or c_deg arguments; it '
                         'redraws whatever the CSV holds')
        redraw_from_csv(args.csv, out_path=args.out, thesis_out=args.thesis_out)
        raise SystemExit(0)

    try:
        curtail_cost = resolve_curtailment_arg(args.curtailment)
    except ValueError as e:
        parser.error(str(e))

    c_deg_values = (
        tuple(float(v) for v in args.c_deg_values.split(','))
        if args.c_deg_values else DEFAULT_C_DEG_VALUES
    )

    if args.site_id is not None and args.sites is not None:
        parser.error('give either a single site_id or --sites, not both')

    if args.site_id is not None:
        run_sweep(
            args.site_id, c_deg_values=c_deg_values,
            export_limit_kw=args.export_limit_kw,
            curtail_cost_per_kwh_dollars=curtail_cost, out_path=args.out,
        )
    else:
        if args.export_limit_kw is not None:
            parser.error('--export-limit-kw applies to single-site runs only; '
                         'set export_kw_override in the review CSV instead')
        sites = [s.strip() for s in args.sites.split(',')] if args.sites else None
        run_cohort_sweep(
            sites=sites, c_deg_values=c_deg_values,
            curtail_cost_per_kwh_dollars=curtail_cost,
            out_path=args.out, csv_path=args.csv,
            thesis_out=args.thesis_out,
        )