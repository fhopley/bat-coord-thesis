"""Cap account aggregation: combine the two headline household charts
(annual bill impact, and bill impact relative to coordination savings)
across every year and state already run for one battery size.

See cap_account_spec.md, Section 12, for the full design. This script
reads already-written cap_account_analysis.py output
(`{state}_{year}_{battery_name}/per_site_annual.csv`) -- it does not
touch flows, prices, or settlement, and does not extend the fuller
per-scenario output (retailer shortfall, concentration, monthly boxplot)
to this aggregation.

--results-root points at one run's output directory, i.e.
{cap_account_analysis --output-dir}/{run_name}; --variant selects which
parameter combination within each scenario to aggregate (spec section 12).
Charts are written under {output-dir}/{battery}/{variant}/, so one battery
size or variant never overwrites another.

Usage (from project root):
    python3 analysis/scripts/cap_account_aggregate.py --battery-name 20kWh \\
        --variant cohort_x300_monthly_s0.01
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The two metrics this script aggregates, and how to read/label them.
# Titles use Title Case, matching financial_analysis.py's own convention
# (e.g. 'Household Total Bill Savings by Battery Size and Retail Plan').
# Axis labels stay sentence case, also matching that script.
_METRICS = {
    'diff': {
        'stem': 'annual_diff_histogram',
        'xlabel': 'Household bill impact, capped - baseline ($)',
        'xlabel_sub': 'Household bill impact incl. subscription change, '
                      'capped - baseline ($)',
        'title': 'Annual Household Bill Impact Under Cap',
    },
    'pct_of_coordination_savings': {
        'stem': 'pct_of_coordination_savings_histogram',
        'xlabel': 'Bill impact as % of savings from coordination '
                  '(vs self_consumption, same battery)',
        'xlabel_sub': 'Bill impact incl. subscription change, as % of '
                      'savings from coordination (vs self_consumption, '
                      'same battery)',
        'title': 'Annual Bill Impact Relative to Coordination Savings',
    },
}

# Offer C (cap_account_spec.md Section 20): the same two metrics, but diff
# is a full-bill difference against Amber rather than a capped-minus-baseline
# wholesale-layer one, so the labels say so.
_DYNAMIC_PREFIX = 'dynamic_'
_DYNAMIC_LABELS = {
    'diff': {
        'xlabel': 'Household bill impact, Offer C - Amber, full bill ($)',
        'title': 'Annual Household Bill Impact, Offer C vs Amber',
    },
    'pct_of_coordination_savings': {
        'xlabel': 'Bill impact, Offer C - Amber, as % of savings from '
                  'coordination (vs self_consumption, same battery)',
        'title': 'Annual Bill Impact Relative to Coordination Savings, '
                 'Offer C vs Amber',
    },
}

_N_BINS = 40


def _apply_subscription(df: pd.DataFrame, change_per_month: float,
                        path) -> None:
    """Fold a change in the fixed subscription fee into the household
    metrics, in place.

    per_site_annual.csv's `diff` is the wholesale layer only (spec section
    2), so on its own it describes an offer stripped of the thing the
    offer's alpha was chosen against. At alpha=0.6 with the fee removed
    entirely, the median site-year moves from +$246 to -$54: the sign of
    the household outcome flips. Comparing variants on the unadjusted
    figure compares them on an incomplete bill.

    `pct_of_coordination_savings` is adjusted with it, or the two charts
    disagree about the same households. Its denominator is
    `annual_coordination_savings` where present; for files written before
    that column existed it is recovered as 100 * diff / pct, which is exact
    -- both scale together -- except where diff is exactly zero, and those
    sites drop out of that metric only.
    """
    if not change_per_month:
        return
    annual = change_per_month * 12.0

    if 'annual_coordination_savings' in df.columns:
        coord = df['annual_coordination_savings']
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            coord = 100.0 * df['diff'] / df['pct_of_coordination_savings']
        coord = coord.replace([np.inf, -np.inf], np.nan)
        n_lost = int(df['pct_of_coordination_savings'].notna().sum()
                     - coord.notna().sum())
        if n_lost:
            print(f"  NOTE: {path.parent.name}: {n_lost} site(s) dropped "
                  f"from pct_of_coordination_savings -- diff is exactly "
                  f"zero so the denominator cannot be recovered. Re-run "
                  f"cap_account_analysis.py to write it directly.",
                  file=sys.stderr)

    df['diff'] = df['diff'] + annual
    df['pct_of_coordination_savings'] = np.where(
        coord.notna() & (coord >= 1.0), df['diff'] / coord * 100.0, np.nan)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate cap_account_analysis.py's two headline "
                    "household charts across every year/state already run "
                    "for one battery size.",
    )
    parser.add_argument('--results-root', type=Path,
                        default=Path('analysis/output/cap_account/'
                                     'full_run_rolling_amber'),
                        help="One run's output directory, i.e. "
                             "{cap_account_analysis --output-dir}/"
                             "{run_name}. Default: "
                             "analysis/output/cap_account/"
                             "full_run_rolling_amber.")
    parser.add_argument('--variant', required=True,
                        help="Variant subfolder within each scenario, e.g. "
                             "'cohort_x300_monthly_s0.01' or "
                             "'retailer_x300_daily_a0.5'. One aggregate "
                             "covers one variant; mixing offers or "
                             "settlement periods in one histogram would be "
                             "meaningless.")
    parser.add_argument('--data-root', type=Path, default=Path('.'),
                        help="Project root for site metadata (PV banding "
                             "needs EDP metadata, same convention as "
                             "financial_analysis.py). Default: cwd.")
    parser.add_argument('--battery-name', required=True,
                        help="e.g. '20kWh' -- aggregates every "
                             "{state}_{year}_{battery-name} folder found "
                             "under --results-root, across every state and "
                             "year present.")
    parser.add_argument('--subscription-change-per-month', type=float,
                        default=0.0,
                        help="Change in the fixed subscription fee this "
                             "offer allows, $/month, NEGATIVE where the fee "
                             "falls. Added to every site's annual bill "
                             "impact as 12x this value. This is a design "
                             "input, not a model output: the fee is a "
                             "choice, and cap_account_retailer.py's "
                             "subscription_change_vs_amber is what the "
                             "retailer could afford. Default 0, i.e. the "
                             "wholesale layer alone.")
    parser.add_argument('--unweighted', action='store_true',
                        help="Plain concatenation: every site-year counts "
                             "once, no PV weighting, no year-equalization, "
                             "no state weighting (default: off, weighted "
                             "mode is on).")
    parser.add_argument('--weights-file', type=Path,
                        default=Path('analysis/config/pv_size_bands.yaml'),
                        help="Path to the weighting YAML (default: "
                             "analysis/config/pv_size_bands.yaml, the same "
                             "file financial_analysis.py already uses).")
    parser.add_argument('--output-dir', type=Path,
                        default=Path('analysis/output/cap_account_aggregate'),
                        help="Charts are written under "
                             "{output-dir}/{battery}/{variant}/ so "
                             "aggregates of different battery sizes or "
                             "variants don't overwrite each other.")
    parser.add_argument('--pct-axis-limit', type=float, default=None,
                        metavar='PCT',
                        help="Fix the x-axis of the "
                             "pct_of_coordination_savings histograms to "
                             "[-PCT, +PCT] regardless of the data, so runs "
                             "at different thresholds or settlement periods "
                             "can be compared on one scale. Site-years "
                             "outside the range are not drawn. Applies to "
                             "the single, per-state and grid figures; the "
                             "dollar histograms are unaffected.")
    parser.add_argument('--interval', type=float, default=90.0,
                        help="Central interval to shade and annotate on the "
                             "per-state and national histograms (default: "
                             "90, i.e. the 5th-95th weighted percentile "
                             "range). Pass 0 to disable the annotation "
                             "entirely. Not drawn on the state x year grid "
                             "(20 small panels have no room for it).")
    return parser.parse_args(argv)


def _discover_scenario_dirs(results_root: Path, battery_name: str
                            ) -> list[tuple[str, int, Path]]:
    """Every {state}_{year}_{battery_name} directory under results_root,
    as (state, year, path) tuples. battery_name must match exactly."""
    pattern = re.compile(r'^([A-Z]{2,3})_(\d{4})_' + re.escape(battery_name) + r'$')
    found = []
    if not results_root.exists():
        return found
    for entry in sorted(results_root.iterdir()):
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if m:
            found.append((m.group(1), int(m.group(2)), entry))
    return found


def _site_weight_for_state_year(
    data_root: Path, state: str, year: int, weights_cfg: dict,
    dc_ac_ratio_cache: dict[int, float], all_meta_cache: dict[int, pd.DataFrame],
    fa,
) -> pd.Series:
    """PV size-band site_weight for one (state, year), reusing
    financial_analysis.py's own banding pipeline exactly (fa is that
    imported module) -- one source of truth for PV banding logic, not
    duplicated here. Mirrors that script's own main()-loop caching: the
    pooled (national) DC:AC ratio and pooled metadata are computed once
    per year, across every state, then reused for every state that year.
    """
    if year not in all_meta_cache:
        pooled_frames = []
        for other_state in weights_cfg['by_state'].keys():
            try:
                pooled_frames.append(fa.load_site_meta(data_root, str(year), other_state))
            except FileNotFoundError:
                continue
        pooled_meta = (pd.concat(pooled_frames, ignore_index=True) if pooled_frames
                      else pd.DataFrame(columns=['site_id', 'subarray_dc_rating_kw',
                                                 'inverter_ac_rating_kw',
                                                 'annual_generation_kwh']))
        ratio = fa.compute_dc_ac_ratio(pooled_meta)
        dc_ac_ratio_cache[year] = ratio
        all_meta_cache[year] = fa.assign_direct_size(pooled_meta, ratio)

    meta = fa.load_site_meta(data_root, str(year), state)
    meta = fa.assign_direct_size(meta, dc_ac_ratio_cache[year])
    meta = fa.estimate_missing_sizes(meta, state, fallback_meta=all_meta_cache[year])
    site_weight, excluded = fa.assign_bands_and_weights(
        meta, state, weights_cfg['bands'], weights_cfg['by_state'])
    if excluded:
        print(f"  {len(excluded)} site(s) excluded from weighted "
             f"calculation for {state} {year}.", file=sys.stderr)
    return site_weight


def _shared_hist_bins(values_by_panel: dict[str, np.ndarray],
                      fixed_abs_limit: float | None = None) -> np.ndarray:
    """Common bin edges across every panel, computed once before any panel
    is drawn, so a small-multiples grid shares one x-axis honestly --
    same principle as financial_analysis.py's _shared_ylim, applied to
    histogram bins instead of curve y-ranges.

    With fixed_abs_limit set, the edges span [-limit, +limit] regardless of
    the data, so separate runs can be laid side by side on one scale.
    Values outside the range fall outside every bin and are not drawn."""
    if fixed_abs_limit is not None:
        return np.linspace(-fixed_abs_limit, fixed_abs_limit, _N_BINS + 1)
    all_vals = np.concatenate([v for v in values_by_panel.values() if len(v)])
    lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.02
    return np.linspace(lo - pad, hi + pad, _N_BINS + 1)


def _shared_hist_ylim(values_by_panel: dict[str, np.ndarray],
                      weights_by_panel: dict[str, np.ndarray | None],
                      bins: np.ndarray) -> float:
    """Common y-axis (count) upper limit across every panel, from the
    tallest bar any panel would draw with the shared bins above."""
    max_count = 0.0
    for label, vals in values_by_panel.items():
        if len(vals) == 0:
            continue
        w = weights_by_panel.get(label)
        counts, _ = np.histogram(vals, bins=bins, weights=w)
        max_count = max(max_count, float(counts.max()))
    return max_count * 1.08 if max_count > 0 else 1.0


def _weighted_percentile(values: np.ndarray, weights: np.ndarray | None,
                         q: float) -> float:
    """q in [0, 100]. Weighted percentile via cumulative-weight
    interpolation (midpoint convention): sorts values, builds the
    cumulative weight fraction at each point's *midpoint* (so the smallest
    value sits at a small positive fraction, not exactly 0), and linearly
    interpolates the target quantile against it. Falls back to plain
    np.percentile when weights is None, and the two agree exactly in that
    case (checked directly, not just assumed) since equal weights reduce
    the same interpolation to the unweighted case.
    """
    if weights is None:
        return float(np.percentile(values, q))
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w) - 0.5 * w
    cum = cum / w.sum()
    return float(np.interp(q / 100.0, cum, v))


def _draw_histogram(ax, values: np.ndarray, weights: np.ndarray | None,
                    bins: np.ndarray, ylim: float, color: str) -> None:
    ax.hist(values, bins=bins, weights=weights, color=color, edgecolor='white')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_ylim(0, ylim)


def make_histogram_grid(
    cells: list[list[str | None]], values_by_panel: dict[str, np.ndarray],
    weights_by_panel: dict[str, np.ndarray | None], n_by_panel: dict[str, int],
    xlabel: str, suptitle: str, battery_name: str, weight_label: str,
    output_dir: Path, stem: str,
    color: str, panel_size: tuple[float, float] = (3.2, 2.6),
    fixed_abs_limit: float | None = None,
) -> None:
    """State x year small-multiples grid of histograms, mirroring
    financial_analysis.py's make_grid_figure/_grid_cells_from_flat:
    row-major cells (rows = states, columns = years), missing combinations
    left blank, one shared x-axis and y-axis across every included panel.
    """
    n_rows = len(cells)
    n_cols = len(cells[0]) if n_rows else 0
    flat_labels = [c for row in cells for c in row if c is not None]
    if not flat_labels:
        return

    bins = _shared_hist_bins({l: values_by_panel[l] for l in flat_labels},
                             fixed_abs_limit)
    ylim = _shared_hist_ylim({l: values_by_panel[l] for l in flat_labels},
                             weights_by_panel, bins)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_w = panel_size[0] * n_cols + 0.6
    fig_h = panel_size[1] * n_rows + 0.7
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             sharex=True, sharey=True, squeeze=False)

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r][c]
            label = cells[r][c]
            if label is None:
                ax.axis('off')
                continue
            _draw_histogram(ax, values_by_panel[label], weights_by_panel.get(label),
                           bins, ylim, color)
            if fixed_abs_limit is not None:
                ax.set_xlim(-fixed_abs_limit, fixed_abs_limit)
            ax.set_title(label, fontsize=9)
            ax.text(0.02, 0.03, f'n={n_by_panel[label]}', transform=ax.transAxes,
                   ha='left', va='bottom', fontsize=7.5, color='grey')

            is_bottom = all(cells[rr][c] is None for rr in range(r + 1, n_rows))
            if c == 0:
                ax.set_ylabel('Count')
            if is_bottom:
                ax.tick_params(labelbottom=True)

    fig.suptitle(suptitle, fontsize=12, y=0.995)
    fig.text(0.01, 0.99, f'{battery_name} battery  |  {weight_label}',
             ha='left', va='top', fontsize=8, color='grey')
    fig.text(0.5, 0.01, xlabel, ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save(fig, output_dir, f'{stem}_grid')
    plt.close(fig)


def _write_site_year_csv(output_dir: Path, stem: str,
                         key_parts: list[pd.DataFrame], values: np.ndarray,
                         weights: np.ndarray | None) -> None:
    """The national pool behind a histogram, one row per site-year: the
    plotted value and the weight the histogram uses (1 when unweighted), so
    figures that compare variants can be drawn without re-running the
    aggregation. Written as {stem}_site_years.csv beside the PNG."""
    out = pd.concat(key_parts, ignore_index=True)
    out['value'] = values
    out['weight'] = 1.0 if weights is None else weights
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'{stem}_site_years.csv'
    out.to_csv(path, index=False)
    print(f"  Saved {path.name}")


def _write_household_outcomes(output_dir: Path, args, sub: float,
                              weight_tag: str, pools: dict,
                              n_scenarios: int) -> None:
    """The national pool's household outcome statistics, one row, written as
    household_outcomes_{weight_tag}.csv beside the histograms. Bill change is
    capped - baseline (or Offer C - Amber), so a site is better off where it
    is negative. The percentage columns are of the savings from coordination,
    as in the pct_of_coordination_savings histogram."""
    diff, w_diff = pools['diff']
    pct, w_pct = pools['pct_of_coordination_savings']

    def pctile(v, w, q):
        return _weighted_percentile(v, w, q)

    row = {
        'variant': args.variant,
        'battery_name': args.battery_name,
        'subscription_change_per_month': sub,
        'n_scenarios': n_scenarios,
        'n_site_years': len(diff),
        'mean_bill_change': float(np.average(diff, weights=w_diff)),
        'p5_bill_change': pctile(diff, w_diff, 5.0),
        'p50_bill_change': pctile(diff, w_diff, 50.0),
        'p95_bill_change': pctile(diff, w_diff, 95.0),
        'share_better_off': float(np.average(diff < 0, weights=w_diff)),
        'n_site_years_pct': len(pct),
        'p5_pct_of_coordination_savings': pctile(pct, w_pct, 5.0),
        'p50_pct_of_coordination_savings': pctile(pct, w_pct, 50.0),
        'p95_pct_of_coordination_savings': pctile(pct, w_pct, 95.0),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'household_outcomes_{weight_tag}.csv'
    pd.DataFrame([row]).to_csv(path, index=False)
    print(f"  Saved {path.name}")


def _save(fig, output_dir: Path, stem: str) -> None:
    """Save matching financial_analysis.py's own _save: same dpi,
    bbox_inches='tight', and a per-file confirmation line."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f'{stem}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved {stem}.png")


def _annotate_text(ax, text: str, loc: str = 'upper left') -> None:
    """Small grey in-axes note, matching financial_analysis.py's own
    _annotate_text -- detail (scope, weighting, n, interval bounds) goes
    here, not stuffed into the title, so the title stays a short statement
    of what the chart is, not a running log of everything about it."""
    if loc == 'upper left':
        ax.text(0.02, 0.97, text, transform=ax.transAxes,
               ha='left', va='top', fontsize=8, color='grey')
    else:
        ax.text(0.98, 0.04, text, transform=ax.transAxes,
               ha='right', va='bottom', fontsize=8, color='grey')


def _year_range(years) -> str:
    """'2021 - 2025' for a span, '2022' for a single year. Compact enough
    to sit inline in a note alongside the scope, rather than listing every
    year individually."""
    ys = sorted(years)
    return str(ys[0]) if len(ys) == 1 else f'{ys[0]} - {ys[-1]}'


def _draw_single_histogram(values: np.ndarray, weights: np.ndarray | None,
                           n: int, xlabel: str, title: str, scope: str,
                           battery_name: str, weight_label: str,
                           output_dir: Path, stem: str,
                           color: str, interval_pct: float | None = 90.0,
                           fixed_abs_limit: float | None = None) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = _shared_hist_bins({'_': values}, fixed_abs_limit)
    _draw_histogram(ax, values, weights, bins,
                    _shared_hist_ylim({'_': values}, {'_': weights}, bins), color)
    if fixed_abs_limit is not None:
        ax.set_xlim(-fixed_abs_limit, fixed_abs_limit)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count' if weights is None else 'Weighted count')

    # Note lines use flow_exposure_analysis.py's inline '  |  ' separator to
    # pack related facts onto one line rather than one fact per line -- keeps
    # the note to 4 lines instead of 5.
    note = (f'{scope}  |  {battery_name} battery\n'
           f'{weight_label}\n'
           f'n = {n} site-years')
    if interval_pct is not None:
        tail = (100.0 - interval_pct) / 2.0
        lo = _weighted_percentile(values, weights, tail)
        hi = _weighted_percentile(values, weights, 100.0 - tail)
        ax.axvspan(lo, hi, color=color, alpha=0.12, zorder=0)
        for edge in (lo, hi):
            ax.axvline(edge, color=color, linewidth=1.2, linestyle=':')
        note += f'\nCentral {interval_pct:.0f}% interval: {lo:+.2f} to {hi:+.2f}'
    _annotate_text(ax, note, 'upper left')

    ax.set_title(title)
    fig.tight_layout()
    _save(fig, output_dir, stem)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    weighted = not args.unweighted
    weight_label = 'Weighted by PV size, year, and state' if weighted else 'Unweighted'
    # Suffix output filenames by mode, matching financial_analysis.py's own
    # weight_tag convention -- without this, a weighted and an unweighted
    # run of the same battery size silently overwrite each other's charts.
    weight_tag = 'weighted' if weighted else 'unweighted'

    # The fee change belongs in the label and the path, not just the flag:
    # two runs of the same variant at different fees are different offers,
    # and a reader of the chart cannot otherwise tell which they are seeing.
    sub = args.subscription_change_per_month
    is_dynamic = args.variant.startswith(_DYNAMIC_PREFIX)
    if is_dynamic and sub:
        print(f"ERROR: --subscription-change-per-month does not apply to "
              f"{args.variant}: Offer C's diff is the full bill, fixed "
              f"charges included (cap_account_spec.md Section 20).",
              file=sys.stderr)
        return 1
    if sub:
        weight_label += (f'  |  subscription {sub:+.2f}/month '
                         f'({sub * 12:+.0f}/year) included')
        weight_tag += f'_sub{sub:+g}'

    # Battery before variant: --battery-name selects which scenario
    # directories are read but does not appear in the variant folder name,
    # so without it here a 10 kWh run overwrites a 20 kWh one.
    output_root = args.output_dir / args.battery_name / args.variant

    scenario_dirs = _discover_scenario_dirs(args.results_root, args.battery_name)
    if not scenario_dirs:
        print(f"No {{state}}_{{year}}_{args.battery_name} directories found "
             f"under {args.results_root}. --results-root should point at "
             f"one run's output directory, e.g. "
             f"analysis/output/cap_account/full_run_rolling_amber.",
             file=sys.stderr)
        return 1
    print(f"Found {len(scenario_dirs)} scenario(s): "
         f"{[(s, y) for s, y, _ in scenario_dirs]}")

    weights_cfg = None
    if weighted:
        if not args.weights_file.exists():
            print(f"ERROR: weighted mode requires a weights file, none "
                 f"found at {args.weights_file}", file=sys.stderr)
            return 1
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import financial_analysis as fa
        weights_cfg = fa.load_weights_config(args.weights_file)
    else:
        fa = None

    # Load every scenario's per_site_annual.csv once, keyed by (state, year).
    records: dict[tuple[str, int], pd.DataFrame] = {}
    dc_ac_ratio_cache: dict[int, float] = {}
    all_meta_cache: dict[int, pd.DataFrame] = {}
    site_weight_cache: dict[tuple[str, int], pd.Series] = {}

    for state, year, path in scenario_dirs:
        csv_path = path / args.variant / 'per_site_annual.csv'
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping {state} {year}.",
                 file=sys.stderr)
            continue
        df = pd.read_csv(csv_path, index_col=0)
        df.index = df.index.astype(str)
        _apply_subscription(df, args.subscription_change_per_month, csv_path)
        records[(state, year)] = df

        if weighted:
            site_weight_cache[(state, year)] = _site_weight_for_state_year(
                args.data_root, state, year, weights_cfg,
                dc_ac_ratio_cache, all_meta_cache, fa)

    if not records:
        print("No usable per_site_annual.csv files found.", file=sys.stderr)
        return 1

    states_in_order = sorted({s for s, _ in records.keys()})
    years_in_order = sorted({y for _, y in records.keys()})
    national_pools: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}

    for metric, meta in _METRICS.items():
        # The limit is a percentage, so it applies only to the percentage
        # metric; the dollar histogram keeps its data-driven range.
        pct_limit = (args.pct_axis_limit
                     if metric == 'pct_of_coordination_savings' else None)
        color = '#4c72b0' if metric == 'diff' else '#c44e52'
        xlabel = meta['xlabel_sub'] if sub else meta['xlabel']
        title = meta['title']
        if is_dynamic:
            xlabel = _DYNAMIC_LABELS[metric]['xlabel']
            title = _DYNAMIC_LABELS[metric]['title']

        # Per-(state, year) panel data, for the grid and reused for the
        # per-state pools below.
        panel_values: dict[str, np.ndarray] = {}
        panel_weights: dict[str, np.ndarray | None] = {}
        panel_n: dict[str, int] = {}
        panel_sites: dict[str, np.ndarray] = {}

        for (state, year), df in records.items():
            series = df[metric].dropna()
            if weighted:
                sw = site_weight_cache[(state, year)].reindex(series.index)
                keep = sw.notna()
                series = series[keep]
                w = sw[keep].to_numpy()
            else:
                w = None
            label = f'{state} {year}'
            panel_values[label] = series.to_numpy()
            panel_weights[label] = w
            panel_n[label] = len(series)
            panel_sites[label] = series.index.to_numpy()

        # --- Per-state, all-years pool ---
        for state in states_in_order:
            state_years = sorted(y for s, y in records.keys() if s == state)
            if len(state_years) < 2:
                continue
            vals_parts, w_parts = [], []
            for year in state_years:
                label = f'{state} {year}'
                vals_parts.append(panel_values[label])
                if weighted:
                    w_parts.append(panel_weights[label])
            values = np.concatenate(vals_parts)
            weights = np.concatenate(w_parts) if weighted else None
            out_dir = output_root / f'{state}_all_years'
            print(f"\n{state}, all years ({weight_tag}) -- {metric}:")
            _draw_single_histogram(
                values, weights, len(values), xlabel, title,
                f'{state}, {_year_range(state_years)}', args.battery_name,
                weight_label,
                out_dir, f"{meta['stem']}_{weight_tag}", color,
                interval_pct=(args.interval if args.interval > 0 else None),
                fixed_abs_limit=pct_limit)

        # --- National pool ---
        if len(states_in_order) > 1:
            vals_parts, w_parts, key_parts = [], [], []
            for (state, year), df in records.items():
                label = f'{state} {year}'
                n_years_state = len({y for s, y in records.keys() if s == state})
                vals_parts.append(panel_values[label])
                if weighted:
                    state_weight = weights_cfg['state_mix'].get(state)
                    if state_weight is None:
                        print(f"WARNING: no state_mix entry for {state}, "
                             f"excluding from national pool for {metric}.",
                             file=sys.stderr)
                        vals_parts.pop()
                        continue
                    national_w = (panel_weights[label] / n_years_state
                                 * state_weight)
                    w_parts.append(national_w)
                key_parts.append(pd.DataFrame({
                    'state': state, 'year': year,
                    'site_id': panel_sites[label]}))
            values = np.concatenate(vals_parts)
            weights = np.concatenate(w_parts) if weighted else None
            out_dir = output_root / 'national'
            national_pools[metric] = (values, weights)
            _write_site_year_csv(out_dir, f"{meta['stem']}_{weight_tag}",
                                 key_parts, values, weights)
            print(f"\nNational ({weight_tag}) -- {metric}:")
            _draw_single_histogram(
                values, weights, len(values), xlabel, title,
                f'National, {_year_range(years_in_order)}', args.battery_name,
                weight_label, out_dir,
                f"{meta['stem']}_{weight_tag}", color,
                interval_pct=(args.interval if args.interval > 0 else None),
                fixed_abs_limit=pct_limit)

        # --- State x year grid ---
        cells = [[f'{state} {year}' if (state, year) in records else None
                 for year in years_in_order] for state in states_in_order]
        print(f"\nState x year grid ({weight_tag}) -- {metric}:")
        make_histogram_grid(
            cells, panel_values, panel_weights, panel_n, xlabel,
            title, args.battery_name, weight_label,
            output_root / 'state_year_grid',
            f"{meta['stem']}_{weight_tag}", color,
            fixed_abs_limit=pct_limit)

    if national_pools:
        _write_household_outcomes(
            output_root / 'national', args, sub, weight_tag, national_pools,
            len({k for k in records}))

    print(f"\nDone. Output written to {output_root}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())