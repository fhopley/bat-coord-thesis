"""Charts for the rolling optimisation horizon sweep.

Consumes the CSV written by sweep_horizon.py. Three charts, each answering one
question that the numbers alone make hard to see:

  A. Where does the VCR curve flatten, per site and for the cohort?
  B. Where does the chosen horizon sit relative to what pre-dispatch actually
     guarantees, and therefore whether infill fires at all?
  C. What does each extra hour of horizon cost in compute?

Follows the conventions in chart_vcr_results.py: matplotlib only, _finish()
writes a 150 dpi PNG or shows interactively, one figure per question.

Usage (from project root):
    python3 -m energy_masters_pipeline.scripts.chart_horizon_sweep
    python3 -m energy_masters_pipeline.scripts.chart_horizon_sweep \\
        --csv data/processed/energy_masters/horizon_sweep.csv \\
        --out-dir data/processed/energy_masters/charts
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from energy_masters_pipeline.scripts.sweep_horizon import (
    COHORT_TOLERANCE, SITE_HEADROOM_FLOOR_DOLLARS, cohort_vcr,
    recommend_horizon,
)

# The horizon pre-dispatch guarantees from any decision instant, allowing for
# PREDISPATCH being up to 25 minutes stale. Measured, not assumed: see
# predispatch_findings.md. A chosen horizon at or below this needs no infill.
PREDISPATCH_GUARANTEED_HOURS = 15.08


def _finish(fig, name: str, out_dir: Optional[Path]):
    if out_dir is None:
        plt.show()
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {path}")


def _site_colours(sites):
    cmap = plt.get_cmap('tab10' if len(sites) <= 10 else 'tab20')
    return {s: cmap(i % cmap.N) for i, s in enumerate(sorted(sites))}


def chart_vcr_by_horizon(long: pd.DataFrame, out_dir: Optional[Path] = None):
    """VCR against horizon, one line per site plus the cohort.

    The cohort line is the one the decision rule reads, and it is drawn heavy
    because a per-site line can look alarming while contributing almost nothing
    to the cohort: a site's weight is its headroom, not one vote.
    """
    thick = long[long['headroom'] >= SITE_HEADROOM_FLOOR_DOLLARS].copy()
    if 'battery_group' not in thick.columns:
        thick['battery_group'] = 'site_native'
    coh = cohort_vcr(long)

    # One line per (site, battery size). A size sweep runs many capacities on
    # ONE site, so keying colour on site alone would collapse every size onto a
    # single indistinguishable line.
    swept = thick['battery_group'].nunique() > 1
    thick['_key'] = (thick['site_id'] + ' ' +
                     thick['battery_kwh'].map(lambda v: f"{v:.1f} kWh"))
    colours = _site_colours(thick['_key'].unique())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key, grp in thick.groupby('_key'):
        grp = grp.sort_values('horizon_hours')
        ax.plot(grp['horizon_hours'], grp['vcr'], marker='o', ms=4, lw=1.2,
                alpha=0.75, color=colours[key], label=key)

    coh = coh.sort_values('horizon_hours')
    if swept:
        # With sizes swept there is no single cohort curve; each size is its
        # own population and the per-size lines above already say it.
        pass
    else:
        ax.plot(coh['horizon_hours'], coh['cohort_vcr'], marker='s', ms=7,
                lw=2.8, color='black',
                label=f"cohort (n={int(coh['n_sites'].max())})", zorder=5)

    best = coh['cohort_vcr'].max()
    if swept:
        best = thick['vcr'].max()
    ax.axhline(best - COHORT_TOLERANCE, ls=':', lw=1, color='grey')
    ax.annotate(f"decision threshold: best - {COHORT_TOLERANCE}",
                xy=(long['horizon_hours'].min(), best - COHORT_TOLERANCE),
                xytext=(2, 4), textcoords='offset points',
                fontsize=8, color='grey')

    ax.axvline(PREDISPATCH_GUARANTEED_HOURS, ls='--', lw=1.2, color='tab:red')
    ax.annotate("pre-dispatch guarantees 15.08 h\n(right of this, infill fires)",
                xy=(PREDISPATCH_GUARANTEED_HOURS, ax.get_ylim()[0]),
                xytext=(4, 18), textcoords='offset points',
                fontsize=8, color='tab:red')

    chosen = recommend_horizon(long)
    for (_, _), hh in chosen.items():
        if hh is not None:
            ax.axvline(hh, ls='-', lw=1.2, color='tab:green', alpha=0.6)
            ax.annotate(f"chosen: {hh:g} h", xy=(hh, best),
                        xytext=(5, -14), textcoords='offset points',
                        fontsize=9, color='tab:green', fontweight='bold')

    ax.set_xlabel('Planning horizon (hours)')
    ax.set_ylabel('Value Capture Ratio vs perfect foresight')
    ax.set_title('Rolling optimisation: value captured against planning horizon\n'
                 'Zero forecast error, so every loss here is horizon, terminal '
                 'pin or coarse resolution', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    _finish(fig, 'horizon_sweep_vcr', out_dir)


def chart_infill_cost(long: pd.DataFrame, out_dir: Optional[Path] = None):
    """Cohort VCR against the share of each window that is extrapolated.

    The point is the turn at the right-hand end. Beyond about 24 h the extra
    horizon is bought entirely with infilled prices, and value stops rising and
    starts falling: more window, worse information, net negative. It is the
    cleanest available evidence that infill is not free, and it is what makes a
    small infill share at a shorter horizon defensible.
    """
    coh = cohort_vcr(long).sort_values('horizon_hours')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(coh['infill'] * 100, coh['cohort_vcr'], marker='o', lw=1.8,
            color='tab:blue')
    for _, r in coh.iterrows():
        ax.annotate(f"{r['horizon_hours']:g} h",
                    xy=(r['infill'] * 100, r['cohort_vcr']),
                    xytext=(5, -3), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Share of the planning window that is infilled (%)')
    ax.set_ylabel('Cohort VCR')
    ax.set_title('More horizon is only worth having if it carries real prices\n'
                 'Labels are horizon length', fontsize=10)
    ax.grid(alpha=0.3)
    _finish(fig, 'horizon_sweep_infill_cost', out_dir)


def chart_compute_cost(long: pd.DataFrame, out_dir: Optional[Path] = None):
    """Wall time against horizon, with cohort VCR on a twin axis.

    Puts the trade-off in one frame: the value curve is flat from about 18 h
    while the cost curve keeps climbing, so everything to the right of the
    chosen horizon is compute spent for nothing.
    """
    coh = cohort_vcr(long).sort_values('horizon_hours')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(coh['horizon_hours'], coh['wall_s'] / 60, width=1.6,
           color='tab:grey', alpha=0.55, label='wall time (min)')
    ax.set_xlabel('Planning horizon (hours)')
    ax.set_ylabel('Sweep wall time across all sites (minutes)')

    ax2 = ax.twinx()
    ax2.plot(coh['horizon_hours'], coh['cohort_vcr'], marker='o', lw=2,
             color='tab:blue', label='cohort VCR')
    ax2.set_ylabel('Cohort VCR', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    chosen = [h for h in recommend_horizon(long).values() if h is not None]
    if chosen:
        ax.axvline(chosen[0], ls='-', lw=1.4, color='tab:green', alpha=0.7)
    ax.set_title('Value flattens well before cost does', fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    _finish(fig, 'horizon_sweep_compute', out_dir)


def chart_all(csv_path: Path, out_dir: Optional[Path] = None) -> pd.DataFrame:
    long = pd.read_csv(csv_path)
    if 'battery_group' not in long.columns:
        long['battery_group'] = 'site_native'
    chart_vcr_by_horizon(long, out_dir)
    chart_infill_cost(long, out_dir)
    chart_compute_cost(long, out_dir)
    return long


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', type=Path,
                        default=Path('data/processed/energy_masters/horizon_sweep.csv'))
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='Omit to show interactively')
    args = parser.parse_args()
    chart_all(args.csv, args.out_dir)