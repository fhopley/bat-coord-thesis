"""Average day net import profile: the cohort's mean net import by local time of
day for each cohort and offer.

See docs/average_day_profile_spec.md. The loading, site weighting, local-time
and run-basis machinery is imported from flow_exposure_analysis.py
(docs/flow_exposure_spec.md), not copied.

Usage (from project root):
    python3 analysis/scripts/average_day_profile.py --data-root "/Volumes/Manual/Thesis Project"
    python3 analysis/scripts/average_day_profile.py --basis conservative --battery-kwh 10
    python3 analysis/scripts/average_day_profile.py --switch-off no_cer solar_only
    python3 analysis/scripts/average_day_profile.py --months 6 7 8 --per-state
    python3 analysis/scripts/average_day_profile.py --include-price
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import financial_analysis as fa
import flow_exposure_analysis as fe
import thesis_style as ts
from financial_analysis import load_weights_config

BINS_PER_DAY = 288
BIN_HOURS = 24 / BINS_PER_DAY
PEAK_WINDOW = (16, 21)       # local hours, as the peak-window exposure charts
MIDDAY_WINDOW = (11, 14)
HEIGHT_IN = 3.0
EVENING_SHADE = '0.93'
NATIONAL = 'national'

# Wholesale price (--include-price), drawn on a secondary y axis. The axes are
# scaled so the flow lines fill the lower FLOW_BAND of the panel and the price
# fills PRICE_BAND, leaving the top of the panel for the data note.
PRICE_KEY = 'price'
PRICE_NAME = 'Wholesale price'
PRICE_COLOUR = '#7b3294'
FLOW_BAND_TOP = 0.9
PRICE_BAND = (0.62, 0.9)


# ── Series ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Series:
    key: str
    name: str
    colour: str
    linestyle: str
    has_battery: bool
    plan_label: str | None = None     # coordination plan whose lp_dir is the run
    run_dir: str | None = None        # fixed run, where there is no plan


def _series_table() -> list[Series]:
    """The series in chart order. Colours and names for the offers are those
    of the other Results charts (flow_exposure_analysis)."""
    plan_colour = {p['label']: p['color'] for p in fe._BASE_PLANS}
    refs = {r['label']: r for r in fe.REFERENCE_SCENARIOS}
    table = [
        Series('no_cer', 'No CER', 'black', '-', False,
               run_dir=refs['No CER (load only)']['run_dir']),
        Series('solar_only', 'Solar only', refs['Solar Only']['color'], '-',
               False, run_dir=refs['Solar Only']['run_dir']),
        Series('self_consumption', fe.SELF_CONSUMPTION_NAME, '0.45', '--', True,
               run_dir=fe.SELF_CONSUMPTION_REFERENCE_RUN),
    ]
    for key, label in (('amber', 'Amber'), ('globird', 'GloBird'),
                       ('agl', 'AGL'), ('energyaustralia', 'Energy Australia'),
                       ('origin', 'Origin Battery Lite')):
        table.append(Series(key, fe.OFFER_NAMES[label], plan_colour[label], '-',
                            True, plan_label=label))
    return table


SERIES = _series_table()
SERIES_KEYS = [s.key for s in SERIES]


def _series_plan(s: Series) -> dict | None:
    """The registry entry behind an offer series (None for the cohorts)."""
    return next((p for p in fe._BASE_PLANS if p['label'] == s.plan_label), None)


def series_excluded(s: Series, state: str) -> bool:
    """True where the offer is left out in state (EnergyAustralia outside
    NSW), from financial_analysis.RETAIL_PLANS excluded_states."""
    plan = _series_plan(s)
    return plan is not None and state in fe._excluded_states(plan)


def resolve_runs(basis: str) -> dict[str, str | None]:
    """Run directory for every series on one basis (None where there is no run
    to use). The optimised run of each offer is the plan's lp_dir from
    plans_for_basis. Where that run is a placeholder (Amber, conservative) the
    placeholder's real run stands in; flows are checked at load time."""
    plans = {p['label']: p for p in fe.plans_for_basis(basis)}
    placeholders = fa.PLACEHOLDERS.get(basis, {})
    runs: dict[str, str | None] = {}
    for s in SERIES:
        if s.run_dir is not None:
            runs[s.key] = s.run_dir
        elif s.plan_label in plans:
            runs[s.key] = plans[s.plan_label]['lp_dir']
        else:
            base = next(p for p in fe._BASE_PLANS if p['label'] == s.plan_label)
            ph = placeholders.get(base['lp_mode'])
            runs[s.key] = ph['run'] if ph else None
            if ph:
                print(f"  NOTE: {s.name} on the {basis} basis uses '{ph['run']}' "
                      f"flows in place of the placeholder run.")
    return runs


# ── Profiles ──────────────────────────────────────────────────────────────────

class BinSums:
    """Per-bin sum and count of interval values, so pooled means combine
    exactly across years and states."""

    def __init__(self):
        self.total = np.zeros(BINS_PER_DAY)
        self.count = np.zeros(BINS_PER_DAY)

    def add(self, bins: np.ndarray, values: np.ndarray) -> None:
        ok = ~np.isnan(values)
        self.total += np.bincount(bins[ok], weights=values[ok],
                                  minlength=BINS_PER_DAY)
        self.count += np.bincount(bins[ok], minlength=BINS_PER_DAY)

    def mean(self) -> np.ndarray:
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(self.count > 0, self.total / self.count, np.nan)


def gather_profiles(
    data_root: Path, runs_root: Path, scenarios: list[tuple[str, str]],
    runs: dict[str, str | None], battery_kwh: int, weighted: bool,
    weights_cfg: dict | None, months: list[int] | None,
    include_price: bool = False,
    by_year: dict[tuple[str, str, str], BinSums] | None = None,
) -> dict[tuple[str, str], BinSums]:
    """One pass: each flows.parquet is read once. Returns the pooled
    per-interval cohort means as BinSums by (series key, state). With
    include_price the state's wholesale price (RRP, $/MWh) is pooled the same
    way under the key PRICE_KEY, on the intervals of the first run loaded for
    each state-year. When by_year is given it is filled with the same sums by
    (series key, state, year), for the single state-year outputs."""
    sums: dict[tuple[str, str], BinSums] = {}
    price_cache: dict[str, pd.DataFrame] = {}
    price_done: set[tuple[str, str]] = set()
    site_weight_cache: dict[tuple[str, str], pd.Series] = {}
    dc_ac_ratio_cache: dict[str, float] = {}
    all_meta_cache: dict[str, pd.DataFrame] = {}
    clock_cache: dict[tuple[str, str], tuple] = {}

    for s in SERIES:
        run_dir = runs[s.key]
        if run_dir is None:
            continue
        size = battery_kwh if s.has_battery else None
        loaded = 0
        for state, year in scenarios:
            if series_excluded(s, state):
                continue
            df = fe.load_flows(runs_root, run_dir, state, year, battery_kwh=size)
            if df is None:
                print(f"  WARNING: flows.parquet not found for '{run_dir}' "
                      f"{state}_{year}{f'_{size}kWh' if size else ''}, skipping.",
                      file=sys.stderr)
                continue
            site_weight = None
            if weighted:
                key = (state, year)
                if key not in site_weight_cache:
                    site_weight_cache[key] = fe.compute_site_weight_for_scenario(
                        data_root, weights_cfg, state, year,
                        dc_ac_ratio_cache, all_meta_cache)
                site_weight = site_weight_cache[key]
            net_kw = fe.weighted_row_mean(df, site_weight) / fe.INTERVAL_HOURS

            cached = clock_cache.get((state, year))
            if cached is None or not cached[0].equals(df.index):
                clock = fe.local_wall_clock(df.index, state, fe.INTERVAL_HOURS)
                bins = clock.hour * fe.INTERVALS_PER_HOUR + clock.minute // 5
                cached = (df.index, bins.astype(int), clock.date.month.to_numpy())
                clock_cache[(state, year)] = cached
            _, bins, month = cached
            values = net_kw.to_numpy()
            keep = np.isin(month, months) if months else slice(None)
            sums.setdefault((s.key, state), BinSums()).add(
                bins[keep], values[keep])
            if by_year is not None:
                by_year.setdefault((s.key, state, year), BinSums()).add(
                    bins[keep], values[keep])
            if include_price and (state, year) not in price_done:
                rrp = fe.get_prices(data_root, year, price_cache)[
                    fe.RRP_COL[state]].reindex(df.index).to_numpy()
                sums.setdefault((PRICE_KEY, state), BinSums()).add(
                    bins[keep], rrp[keep])
                if by_year is not None:
                    by_year.setdefault((PRICE_KEY, state, year), BinSums()).add(
                        bins[keep], rrp[keep])
                price_done.add((state, year))
            loaded += 1
        if not loaded:
            print(f"  NOTE: {s.name} left out: no flows for '{run_dir}'"
                  f"{f' at {size} kWh' if size else ''}.")
    return sums


def state_profile(sums, key: str, state: str) -> np.ndarray | None:
    b = sums.get((key, state))
    return None if b is None else b.mean()


def national_profile(sums, key: str, states: list[str], state_mix: dict | None
                     ) -> np.ndarray | None:
    """state_mix-weighted combination of the state profiles, renormalised over
    the states that have the series; without state_mix (unweighted or a single
    state) the states' intervals are pooled."""
    have = [s for s in states if (key, s) in sums]
    if not have:
        return None
    if state_mix is None or len(have) == 1:
        pooled = BinSums()
        for s in have:
            pooled.total += sums[(key, s)].total
            pooled.count += sums[(key, s)].count
        return pooled.mean()
    profiles = np.array([sums[(key, s)].mean() for s in have])
    out = np.full(BINS_PER_DAY, np.nan)
    for i in range(BINS_PER_DAY):
        v = fe._state_mix_weighted_average(list(profiles[:, i]), have, state_mix)
        if v is not None:
            out[i] = v
    return out


# ── Outputs ───────────────────────────────────────────────────────────────────

def bin_label(i: int) -> str:
    return f'{i // 12:02d}:{(i % 12) * 5:02d}'


def profile_frame(profiles: dict[str, np.ndarray],
                  price: np.ndarray | None = None) -> pd.DataFrame:
    df = pd.DataFrame({s.name: profiles[s.key] for s in SERIES
                       if s.key in profiles})
    if price is not None:
        df[f'{PRICE_NAME} ($/MWh)'] = price
    df.insert(0, 'local_time', [bin_label(i) for i in range(BINS_PER_DAY)])
    return df


def summary_frame(profiles: dict[str, np.ndarray]) -> pd.DataFrame:
    def window_mean(p, w):
        return float(np.nanmean(p[w[0] * 12:w[1] * 12]))

    rows = []
    for s in SERIES:
        p = profiles.get(s.key)
        if p is None:
            continue
        step = np.diff(np.append(p, p[0]))          # change into bin i+1 (wraps)
        j = int(np.nanargmax(np.abs(step)))
        rows.append({
            'series': s.name,
            'peak_window_mean_kw': window_mean(p, PEAK_WINDOW),
            'midday_window_mean_kw': window_mean(p, MIDDAY_WINDOW),
            'max_kw': float(np.nanmax(p)),
            'max_time': bin_label(int(np.nanargmax(p))),
            'min_kw': float(np.nanmin(p)),
            'min_time': bin_label(int(np.nanargmin(p))),
            'max_step_kw': float(step[j]),
            'max_step_time': bin_label((j + 1) % BINS_PER_DAY),
        })
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        for line in header:
            f.write(f'# {line}\n')
        df.to_csv(f, index=False, float_format='%.6g')
    print(f"  Saved: {path}")


def _band_limits(lo: float, hi: float, band: tuple[float, float]
                 ) -> tuple[float, float]:
    """Axis limits that place the data range [lo, hi] at the fractions `band`
    of the axis height."""
    if hi <= lo:
        hi = lo + 1.0
    span = (hi - lo) / (band[1] - band[0])
    bottom = lo - band[0] * span
    return bottom, bottom + span


def fig_average_day(profiles: dict[str, np.ndarray], shown: list[Series],
                    out_path: Path, title: str, note: str,
                    price: np.ndarray | None = None) -> None:
    x = np.arange(BINS_PER_DAY) * BIN_HOURS
    with fe._thesis_rc():
        fig = ts.new_figure(HEIGHT_IN)
        ax = fig.add_subplot(1, 1, 1)
        ax.axvspan(*PEAK_WINDOW, color=EVENING_SHADE, lw=0, zorder=0)
        ax.axhline(0, color='black', lw=0.8, zorder=1)
        for s in shown:
            ax.plot(x, profiles[s.key], color=s.colour, ls=s.linestyle,
                    lw=ts.DATA_LW, label=s.name, zorder=2)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f'{int(v):02d}:00'))
        ax.set_xlabel('Local time of day')
        ax.set_ylabel('Mean net import (kW per site)')
        ax.set_title(title)
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)
        handles, labels = ax.get_legend_handles_labels()
        if price is not None and np.isfinite(price).any():
            flows = np.concatenate([profiles[s.key] for s in shown])
            flo, fhi = np.nanmin(flows), np.nanmax(flows)
            flo = min(flo, 0.0)
            # Flow lines fill the lower part of the panel (zero line included).
            ax.set_ylim(*_band_limits(flo, fhi, (0.04, FLOW_BAND_TOP)))
            ax2 = ax.twinx()
            pmin, pmax = np.nanmin(price), np.nanmax(price)
            ax2.set_ylim(*_band_limits(pmin, pmax, PRICE_BAND))
            line, = ax2.plot(x, price, color=PRICE_COLOUR, lw=ts.MEMBER_LW,
                             label=f'{PRICE_NAME} (right axis)', zorder=3)
            ticks = mticker.MaxNLocator(nbins=4).tick_values(pmin, pmax)
            ax2.set_yticks([t for t in ticks if pmin <= t <= pmax])
            ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda v, _: f'{v:,.0f}'))
            ax2.set_ylabel('Mean wholesale price (\\$/MWh)', color=PRICE_COLOUR)
            ax2.tick_params(axis='y', colors=PRICE_COLOUR)
            ax2.grid(False)
            handles.append(line)
            labels.append(line.get_label())
        fa._draw_notes(ax, note, 'upper left')
        fig.legend(handles, labels, loc='outside lower center',
                   ncol=len(labels) if len(labels) <= 8 else -(-len(labels) // 2),
                   frameon=False, handlelength=1.6,
                   columnspacing=0.9, handletextpad=0.4)
        ts.save_figure(fig, out_path)


def print_summary(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print(f"{'Series':<18} {'16-21h':>8} {'11-14h':>8} {'Max':>7} {'at':>6} "
          f"{'Min':>7} {'at':>6} {'Max step':>9} {'at':>6}")
    for r in df.itertuples(index=False):
        print(f"{r.series:<18} {r.peak_window_mean_kw:>8.3f} "
              f"{r.midday_window_mean_kw:>8.3f} {r.max_kw:>7.3f} {r.max_time:>6} "
              f"{r.min_kw:>7.3f} {r.min_time:>6} {r.max_step_kw:>9.3f} "
              f"{r.max_step_time:>6}")


def chart_title(args) -> str:
    return (f'Average day net import by offer, '
            f'{fa.RUN_BASES[args.basis]["title"]}, {args.battery_kwh} kWh')


def data_note(state_list: list[str], years: list[str], args, national: bool
              ) -> str:
    """The data note drawn top left, in the form of the flow exposure charts
    (flow_exposure_analysis._summary_note)."""
    state_txt = 'National' if national else '/'.join(state_list)
    year_txt = f'{years[0]}–{years[-1]}' if len(years) > 1 else years[0]
    months_txt = (f', months {", ".join(map(str, args.months))}'
                  if args.months else '')
    lines = [f'{state_txt}, {year_txt}{months_txt}, {args.battery_kwh} kWh battery',
             fe._weight_note(not args.unweighted, national=national)]
    if national:
        shown = [s for s in SERIES if s.key not in (args.switch_off or [])]
        plans = [fa_plan for s in shown
                 if (plan := _series_plan(s)) is not None
                 for fa_plan in fa._BASE_PLANS
                 if fa_plan['sc_mode'] == plan['sc_mode']]
        lines += fa.restricted_offer_notes(state_list, plans)
    return '\n'.join(lines)


def write_outputs(profiles, out_dir: Path, args, header: list[str],
                  label: str, note: str,
                  price: np.ndarray | None = None) -> None:
    size = f'{args.battery_kwh}kWh'
    months_sfx = ('_months_' + '-'.join(str(m) for m in args.months)
                  if args.months else '')
    excl_sfx = ('_excl_' + '_'.join(args.switch_off) if args.switch_off else '')
    price_sfx = '_price' if price is not None else ''
    stem = f'average_day_profile_{size}'
    shown = [s for s in SERIES
             if s.key in profiles and s.key not in (args.switch_off or [])]

    write_csv(profile_frame(profiles, price),
              out_dir / f'{stem}{months_sfx}.csv', header)
    summary = summary_frame(profiles)
    write_csv(summary, out_dir / f'{stem}{months_sfx}_summary.csv', header)
    if shown:
        fig_average_day(profiles, shown,
                        out_dir / f'{stem}{excl_sfx}{price_sfx}{months_sfx}',
                        chart_title(args), note, price)
    else:
        print("  WARNING: every series is switched off or missing, no chart drawn.",
              file=sys.stderr)
    print_summary(summary, f'Summary ({label})')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('.'),
                        help='Project root with data/output/runs and '
                             'data/clean/nem_prices (default: cwd)')
    parser.add_argument('--basis', choices=list(fa.RUN_BASES),
                        default='achievable', help='Run basis (default: achievable)')
    parser.add_argument('--battery-kwh', type=int, default=20,
                        help='Battery size for the battery series (default: 20)')
    parser.add_argument('--switch-off', nargs='+', default=None, metavar='KEY',
                        help='Series left off the chart (the CSV keeps every '
                             f'series): {", ".join(SERIES_KEYS)}')
    parser.add_argument('--states', nargs='+', default=None,
                        help='Restrict states (default: all found)')
    parser.add_argument('--years', nargs='+', default=None,
                        help='Restrict years (default: all found)')
    parser.add_argument('--months', nargs='+', type=int, default=None,
                        help='Restrict to these months, 1-12 (default: all)')
    parser.add_argument('--per-state', action='store_true',
                        help='Also write one chart and CSV per state, in '
                             '{state}_all_years/ and in each {state}_{year}/')
    parser.add_argument('--unweighted', action='store_true',
                        help='Plain per-site means; states pooled without '
                             'state_mix')
    parser.add_argument('--include-price', action='store_true',
                        help='Also plot the mean wholesale price (RRP) by '
                             'local time of day on a secondary y axis; the '
                             'price is added to the CSV and "_price" to the '
                             'chart filename')
    parser.add_argument('--weights-file', type=Path,
                        default=Path('analysis/config/pv_size_bands.yaml'),
                        help='Weighting YAML '
                             '(default: analysis/config/pv_size_bands.yaml)')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('analysis/output/exposure'),
                        help='Output directory (default: analysis/output/exposure)')
    args = parser.parse_args(argv)

    bad = [k for k in (args.switch_off or []) if k not in SERIES_KEYS]
    if bad:
        parser.error(f"unknown --switch-off key(s): {', '.join(bad)}. "
                     f"Valid keys: {', '.join(SERIES_KEYS)}")
    if args.months and any(not 1 <= m <= 12 for m in args.months):
        parser.error('--months values must be 1 to 12')

    weighted = not args.unweighted
    weights_cfg = None
    if weighted:
        if not args.weights_file.exists():
            print(f"ERROR: weighted mode requires a weights file, none found at "
                  f"{args.weights_file}", file=sys.stderr)
            return 1
        weights_cfg = load_weights_config(args.weights_file)

    scenarios = fe.discover_scenarios(args.data_root)
    if args.states:
        scenarios = [(s, y) for s, y in scenarios if s in args.states]
    if args.years:
        scenarios = [(s, y) for s, y in scenarios if y in args.years]
    if not scenarios:
        print("ERROR: no scenarios match.", file=sys.stderr)
        return 1
    print(f"Scenarios found: {scenarios}")

    runs_root = args.data_root / 'data' / 'output' / 'runs'
    print(f"\n=== Run basis: {args.basis} ===")
    runs = resolve_runs(args.basis)
    print("\nLoading flows (single pass)...")
    year_sums: dict[tuple[str, str, str], BinSums] = {}
    sums = gather_profiles(args.data_root, runs_root, scenarios, runs,
                           args.battery_kwh, weighted, weights_cfg, args.months,
                           include_price=args.include_price,
                           by_year=year_sums if args.per_state else None)
    if not any(k != PRICE_KEY for k, _ in sums):
        print("ERROR: no flows.parquet files found for any series.",
              file=sys.stderr)
        return 1

    states = sorted({st for _, st in sums})
    years = sorted({y for _, y in scenarios})
    state_mix = weights_cfg['state_mix'] if weighted and len(states) > 1 else None

    def header(state_list, year_list=None):
        return [
            f'Average day net import profile (kW per site) by local time of day, '
            f'bin labelled by interval start',
            f'basis: {args.basis}',
            f'battery_kwh: {args.battery_kwh}',
            f'weighting: {"site size-band weighted" if weighted else "unweighted"}',
            'state_combination: ' + (
                'state_mix' if state_mix and len(state_list) > 1
                else 'pooled intervals'),
            f'states: {" ".join(state_list)}',
            f'years: {" ".join(year_list or years)}',
            f'months: {" ".join(map(str, args.months)) if args.months else "all"}',
        ]

    out_base = args.output_dir / fa.RUN_BASES[args.basis]['folder']

    print("\nNational profile..." if len(states) > 1 else "\nProfile...")
    national = {s.key: p for s in SERIES
                if (p := national_profile(sums, s.key, states, state_mix)) is not None}
    price = (national_profile(sums, PRICE_KEY, states, state_mix)
             if args.include_price else None)
    write_outputs(national, out_base / NATIONAL, args, header(states),
                  'national' if len(states) > 1 else states[0],
                  data_note(states, years, args, national=state_mix is not None),
                  price)

    if args.per_state:
        for state in states:
            profiles = {s.key: p for s in SERIES
                        if (p := state_profile(sums, s.key, state)) is not None}
            print(f"\n{state}...")
            write_outputs(profiles, out_base / f'{state}_all_years', args,
                          header([state]), state,
                          data_note([state], years, args, national=False),
                          state_profile(sums, PRICE_KEY, state)
                          if args.include_price else None)

    if args.per_state:
        for state, year in scenarios:
            profiles = {s.key: b.mean() for s in SERIES
                        if (b := year_sums.get((s.key, state, year))) is not None}
            if not profiles:
                continue
            print(f"\n{state} {year}...")
            price_b = year_sums.get((PRICE_KEY, state, year))
            write_outputs(profiles, out_base / f'{state}_{year}', args,
                          header([state], [year]), f'{state} {year}',
                          data_note([state], [year], args, national=False),
                          price_b.mean()
                          if args.include_price and price_b is not None else None)

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
