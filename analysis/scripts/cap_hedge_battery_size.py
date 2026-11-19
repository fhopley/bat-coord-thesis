"""
cap_hedge_battery_size.py

Battery-size sweep of the cap hedge analysis. See
docs/cap_hedge_battery_size_spec.md.

  --check   flows present for each run and battery size, then exit
  --run     cap_hedge_analysis.py at each size (size-band weighting,
            start-of-quarter premium, No CER, Solar Only, Self Consumption
            and Amber achievable) into runs/<N>kWh, then the conservative
            Amber run (CONSERVATIVE_RUN) with No CER into
            runs/<N>kWh_conservative at the sizes where all its flows exist.
            Sizes already run are skipped unless --force.
  always    collects the per-size national hedge summaries and state-month
            tables into cap_hedge_by_battery_size.csv, writes the values for
            the text to text_values.txt (and prints them), and draws
            retailer_cap_by_battery_size.png, copied to
            thesis/figures/results/.

Output: analysis/output/cap_hedge_battery_size/

Each state-month value is the cohort's mean per site in that state and
month (sites averaged per interval), so there are states x months rows.
State-month classes (state_months.csv, no cap cover): net cost (N < 0),
net earnings (N > 0), no intervals above $300/MWh. A month with intervals
above $300/MWh and N exactly 0 is counted separately (n_zero_with_intervals).

Run from repo root:
  python3 analysis/scripts/cap_hedge_battery_size.py --data-root "/Volumes/Manual/Thesis Project" --check
  python3 analysis/scripts/cap_hedge_battery_size.py --data-root "/Volumes/Manual/Thesis Project" --run
  python3 analysis/scripts/cap_hedge_battery_size.py
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import cap_hedge_analysis as cha
import financial_analysis as fa
import thesis_style as ts
from flow_exposure_analysis import discover_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / 'analysis' / 'output' / 'cap_hedge_battery_size'
RUNS_DIR = OUT_DIR / 'runs'
BASE_DIR = REPO_ROOT / 'analysis' / 'output' / 'cap_hedge'   # 20 kWh thesis results
THESIS_FIG = REPO_ROOT / 'thesis' / 'figures' / 'results' / 'retailer_cap_by_battery_size.png'
CSV_NAME = 'cap_hedge_by_battery_size.csv'
FIG_NAME = 'retailer_cap_by_battery_size.png'

SIZES = tuple(range(5, 55, 5))
TEXT_SIZES = (10, 20, 50)
CONSERVATIVE_RUN = 'full_run_rolling_amber_predispatchonly'
CONS_SUFFIX = '_conservative'

# (cohort name in the CSV and chart, run directory suffix, scenario label in
# that run's outputs). The conservative pass labels its run 'Amber' so the
# per-size charts cap_hedge_analysis.py draws treat it as Amber.
COHORTS = [
    ('No CER', '', cha.NO_CER),
    ('Solar only', '', 'Solar Only'),
    ('Self-consumption', '', 'Self Consumption'),
    ('Amber', '', 'Amber'),
    ('Amber (conservative)', CONS_SUFFIX, 'Amber'),
]
REFERENCE_COHORTS = ('No CER', 'Solar only')

# Chart legend: order and wording (the cohort names above are the CSV's).
LEGEND_ORDER = ('Self-consumption', 'Amber (conservative)', 'Amber')
LEGEND_LABELS = {'Amber': 'Amber (achievable)'}

HEIGHT_IN = 3.0
ZERO_LW = 0.6


def size_dir(kwh: int, suffix: str = '') -> Path:
    return RUNS_DIR / f'{kwh}kWh{suffix}'


def is_done(d: Path) -> bool:
    return ((d / 'national' / 'hedge_summary.csv').exists()
            and (d / 'state_months' / 'state_months.csv').exists())


# ── Check ─────────────────────────────────────────────────────────────────────

def flows_present(runs_root: Path, run_dir: str, scenarios: list[tuple[str, str]],
                  kwh: int | None) -> int:
    sfx = f'_{kwh}kWh' if kwh else ''
    return sum((runs_root / run_dir / 'scenarios' / f'{s}_{y}{sfx}' / 'flows.parquet').exists()
               for s, y in scenarios)


def check(data_root: Path, sizes: list[int]) -> dict[int, bool]:
    """Print flows present per run and size. Returns, per size, whether the
    conservative run has every (state, year)."""
    scenarios = discover_scenarios(data_root)
    if not scenarios:
        print(f"ERROR: no No CER scenarios under {data_root / 'data/output/runs'}.",
              file=sys.stderr)
        return {}
    runs_root = data_root / 'data' / 'output' / 'runs'
    n = len(scenarios)
    print(f"{n} state-years: {', '.join(f'{s}_{y}' for s, y in scenarios)}\n")
    scen = cha.build_scenarios(core_only=True)
    rows = [(sc['run_dir'], sc['has_battery']) for sc in scen]
    rows.append((CONSERVATIVE_RUN, True))
    print(f"{'Run':<42}" + ''.join(f'{k:>7}' for k in sizes))
    cons_ok = {}
    for run_dir, has_battery in rows:
        if not has_battery:
            print(f"{run_dir:<42}{flows_present(runs_root, run_dir, scenarios, None):>4}/{n}"
                  "  (no battery)")
            continue
        counts = [flows_present(runs_root, run_dir, scenarios, k) for k in sizes]
        print(f"{run_dir:<42}" + ''.join(f'{f"{c}/{n}":>7}' for c in counts))
        if run_dir == CONSERVATIVE_RUN:
            cons_ok = {k: c == n for k, c in zip(sizes, counts)}
    return cons_ok


# ── Run ───────────────────────────────────────────────────────────────────────

def run_size(data_root: Path, kwh: int, conservative: bool) -> int:
    """cap_hedge_analysis.main at one size. The conservative pass swaps in a
    scenario list of No CER and the conservative Amber run; quarters are
    intersected across a tier's scenarios, so it is kept apart from the main
    pass."""
    out = size_dir(kwh, CONS_SUFFIX if conservative else '')
    print(f"\n{'=' * 78}\n{kwh} kWh{' conservative' if conservative else ''} -> {out}\n"
          f"{'=' * 78}")
    argv = ['--data-root', str(data_root), '--battery-kwh', str(kwh), '--core-only',
            '--no-notes', '--output-dir', str(out)]
    original = cha.build_scenarios
    if conservative:
        def build(core_only: bool) -> list[dict]:
            scen = []
            for sc in original(core_only=True):
                if sc['label'] == cha.NO_CER:
                    scen.append(sc)
                elif sc['label'] == 'Amber':
                    scen.append(dict(sc, run_dir=CONSERVATIVE_RUN))
            return scen
        cha.build_scenarios = build
    try:
        rc = cha.main(argv)
    finally:
        cha.build_scenarios = original
        plt.close('all')
        gc.collect()
    return rc


# ── Collect ───────────────────────────────────────────────────────────────────

def collect(sizes: list[int]) -> pd.DataFrame:
    rows = []
    for kwh in sizes:
        main_sm = size_dir(kwh) / 'state_months' / 'state_months.csv'
        expected = (int((pd.read_csv(main_sm)['scenario'] == cha.NO_CER).sum())
                    if main_sm.exists() else 0)
        for cohort, suffix, label in COHORTS:
            d = size_dir(kwh, suffix)
            if not is_done(d):
                continue
            hs = pd.read_csv(d / 'national' / 'hedge_summary.csv')
            sm = pd.read_csv(d / 'state_months' / 'state_months.csv')
            hs = hs[hs['scenario'] == label]
            sm = sm[sm['scenario'] == label]
            if hs.empty and sm.empty:
                continue
            has_int = sm['n_intervals_gt300'] > 0
            row = dict(
                cohort=cohort, battery_kwh=kwh,
                x0_kw=float(hs['x0_kw'].iloc[0]) if not hs.empty else np.nan,
                n_state_months=len(sm),
                n_net_cost=int((has_int & (sm['N_at_0'] < 0)).sum()),
                n_net_earnings=int((has_int & (sm['N_at_0'] > 0)).sum()),
                n_no_intervals_gt300=int((~has_int).sum()),
                n_zero_with_intervals=int((has_int & (sm['N_at_0'] == 0)).sum()),
            )
            # Worst month among those with intervals above $300/MWh: a net
            # cost if any, otherwise the smallest net earnings.
            if has_int.any():
                w = sm[has_int].loc[sm.loc[has_int, 'N_at_0'].idxmin()]
                row.update(worst_N_at_0=float(w['N_at_0']), worst_state=w['state'],
                           worst_year=int(w['year']), worst_month=int(w['month']))
            row['complete'] = bool(not hs.empty and expected > 0
                                   and len(sm) == expected)
            rows.append(row)
    return pd.DataFrame(rows)


def checks(df: pd.DataFrame) -> list[str]:
    """Reference cohorts constant across sizes; 20 kWh against the cap hedge
    results in the thesis."""
    lines = []
    for cohort in REFERENCE_COHORTS:
        v = df.loc[(df['cohort'] == cohort) & df['complete'], 'x0_kw']
        if len(v):
            lines.append(f"CHECK {cohort} x0 across sizes: {v.min():.4f} to {v.max():.4f} kW/site")
    hs_p = BASE_DIR / 'national' / 'hedge_summary.csv'
    sm_p = BASE_DIR / 'state_months' / 'state_months.csv'
    if not (hs_p.exists() and sm_p.exists()):
        return lines
    hs, sm = pd.read_csv(hs_p), pd.read_csv(sm_p)
    for cohort, suffix, label in COHORTS:
        if suffix:
            continue
        r = df[(df['cohort'] == cohort) & (df['battery_kwh'] == 20)]
        b_hs, b_sm = hs[hs['scenario'] == label], sm[sm['scenario'] == label]
        if r.empty or b_hs.empty:
            continue
        r = r.iloc[0]
        b_cost = int(((b_sm['n_intervals_gt300'] > 0) & (b_sm['N_at_0'] < 0)).sum())
        dx = r['x0_kw'] - float(b_hs['x0_kw'].iloc[0])
        lines.append(f"CHECK 20 kWh {cohort} vs analysis/output/cap_hedge: "
                     f"x0 diff {dx:+.2e} kW/site, net-cost months "
                     f"{r['n_net_cost']} vs {b_cost}")
    return lines


def _month(r) -> str:
    return f"{r['worst_state']} {pd.Timestamp(int(r['worst_year']), int(r['worst_month']), 1):%b %Y}"


def _dollars(v: float) -> str:
    return f"{'−' if v < 0 else ''}${abs(v):,.2f}"


def text_values(df: pd.DataFrame) -> list[str]:
    lines = []
    for cohort in ('Amber', 'Amber (conservative)', 'Self-consumption'):
        d = df[(df['cohort'] == cohort) & df['complete']].sort_values('battery_kwh')
        if d.empty:
            continue
        lines.append(f"\n{cohort} (sizes {', '.join(str(k) for k in d['battery_kwh'])} kWh)")
        neg = d.loc[d['x0_kw'] < 0, 'battery_kwh']
        lines.append(f"  Smallest size with x0 < 0: "
                     f"{f'{neg.min()} kWh' if len(neg) else 'none'}")
        # Smallest size from which every larger size has no net-cost month.
        from_kwh = None
        for kwh, n in zip(d['battery_kwh'][::-1], d['n_net_cost'][::-1]):
            if n > 0:
                break
            from_kwh = kwh
        lines.append(f"  No state-month with a net cost from: "
                     f"{f'{from_kwh} kWh' if from_kwh is not None else 'none'}")
        for kwh in TEXT_SIZES:
            r = d[d['battery_kwh'] == kwh]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(
                f"  {kwh:>2} kWh: x0 {r['x0_kw']:+.2f} kW/site; net cost {r['n_net_cost']}, "
                f"net earnings {r['n_net_earnings']}, no intervals {r['n_no_intervals_gt300']}"
                f" of {r['n_state_months']}; worst {_dollars(r['worst_N_at_0'])}/site ({_month(r)})")
    return lines


# ── Chart ─────────────────────────────────────────────────────────────────────

def _sweep_axis(ax, sizes: list[int]) -> None:
    ax.set_xticks(sizes)
    ax.set_xticklabels([f'{k:g}' for k in sizes])
    pad = 0.03 * (max(sizes) - min(sizes))
    ax.set_xlim(min(sizes) - pad, max(sizes) + pad)
    ax.set_xlabel('Battery capacity (kWh)')


def draw_chart(df: pd.DataFrame, out_png: Path) -> None:
    mpl.rcdefaults()                     # financial_analysis sets its own on import
    ts.apply_style()
    scen = {sc['label']: sc for sc in cha.build_scenarios(core_only=True)}
    amber, amber_ink = scen['Amber']['color'], fa.PLAN_INK['SC']
    styles = [('Self-consumption', scen['Self Consumption']['color'], None, '-'),
              ('Amber', amber, amber_ink, '-'),
              ('Amber (conservative)', amber, amber_ink, fa.DASHED)]
    ok = df[df['complete']]
    sizes = sorted(ok['battery_kwh'].unique())
    if not sizes:
        print("No complete results to chart.")
        return
    n_total = int(ok['n_state_months'].max())

    fig = ts.new_figure(HEIGHT_IN)
    ax_x, ax_n = fig.subplots(1, 2)
    handles = {}
    for cohort, colour, ink, ls in styles:
        d = ok[ok['cohort'] == cohort].sort_values('battery_kwh')
        if d.empty:
            continue
        edge = (dict(markeredgecolor=ink, markeredgewidth=ts.MARKER_EDGE_WIDTH)
                if ink else {})
        kw = dict(color=colour, lw=ts.DATA_LW, ls=ls, marker='o',
                  ms=ts.MARKER_SIZE, **edge)
        ax_x.plot(d['battery_kwh'], d['x0_kw'], zorder=3, **kw)
        ax_n.plot(d['battery_kwh'], d['n_net_cost'], zorder=3, **kw)
        handles[cohort] = Line2D([], [], label=LEGEND_LABELS.get(cohort, cohort),
                                 **kw)

    ax_x.axhline(0, color='black', lw=ZERO_LW, zorder=2.5)
    ax_x.set_title('Exposure-neutral cover $x_0$ (kW/site)', fontsize=ts.SUBTITLE_SIZE)
    ax_n.set_title('State-months with a net cost above \\$300/MWh',
                   fontsize=ts.SUBTITLE_SIZE)
    ax_n.set_ylim(0, n_total)
    if n_total % 60 == 0:
        ax_n.set_yticks(np.arange(0, n_total + 1, 60))
    for ax in (ax_x, ax_n):
        _sweep_axis(ax, sizes)
    legend = [handles[c] for c in LEGEND_ORDER if c in handles]
    fig.legend(handles=legend, loc='outside lower center', ncol=len(legend),
               frameon=False)
    ts.panel_suptitle(fig, 'Cap position by battery size, imperfect foresight')
    ts.save_figure(fig, out_png)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=None,
                        help='Project root holding data/output/runs (needed for '
                             '--check and --run)')
    parser.add_argument('--sizes', type=int, nargs='+', default=list(SIZES),
                        help='Battery sizes, kWh (default: 5 to 50 in steps of 5)')
    parser.add_argument('--check', action='store_true',
                        help='Print the flows present per run and size, then exit')
    parser.add_argument('--run', action='store_true',
                        help='Run cap_hedge_analysis.py at each size before collecting')
    parser.add_argument('--force', action='store_true',
                        help='With --run, rerun sizes that already have outputs')
    parser.add_argument('--no-conservative', action='store_true',
                        help='With --run, skip the conservative Amber pass')
    args = parser.parse_args(argv)
    sizes = sorted(set(args.sizes))

    if args.check or args.run:
        if args.data_root is None:
            parser.error('--check and --run need --data-root')
        data_root = args.data_root.resolve()
        cons_ok = check(data_root, sizes)
        if args.check:
            return 0
        os.chdir(REPO_ROOT)      # cap_hedge_analysis reads its weights file relative to root
        failed = []
        for kwh in sizes:
            passes = [False]
            if not args.no_conservative:
                if cons_ok.get(kwh):
                    passes.append(True)
                else:
                    print(f"NOTE: {kwh} kWh conservative pass skipped: "
                          f"{CONSERVATIVE_RUN} flows incomplete.")
            for cons in passes:
                d = size_dir(kwh, CONS_SUFFIX if cons else '')
                if is_done(d) and not args.force:
                    print(f"{d.name}: outputs exist, skipped (--force to rerun).")
                    continue
                if run_size(data_root, kwh, cons) != 0:
                    failed.append(d.name)
        if failed:
            print(f"\nFAILED: {', '.join(failed)}", file=sys.stderr)

    df = collect(sizes)
    if df.empty:
        print(f"No per-size outputs under {RUNS_DIR}; run with --run first.", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / CSV_NAME, index=False)
    print(f"\nSaved: {OUT_DIR / CSV_NAME}")

    incomplete = df[~df['complete']]
    lines = [f"INCOMPLETE (left off the chart): {r.cohort} {r.battery_kwh} kWh, "
             f"{r.n_state_months} state-months" for r in incomplete.itertuples()]
    lines += checks(df) + text_values(df)
    text = '\n'.join(lines)
    print(text)
    (OUT_DIR / 'text_values.txt').write_text(text.lstrip('\n') + '\n')

    draw_chart(df, OUT_DIR / FIG_NAME)
    if (OUT_DIR / FIG_NAME).exists():
        THESIS_FIG.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT_DIR / FIG_NAME, THESIS_FIG)
        print(f"  Copied: {THESIS_FIG}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
