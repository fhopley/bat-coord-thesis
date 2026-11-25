"""
reliability_response_analysis.py

Physical response of each cohort and offer in the intervals of greatest
system stress. The selection is the intervals priced at or above the market
price cap (MPC) in force less an offset x (--mpc-offsets, default 0 and
$1,000/MWh), or above a static price (--thresholds). The MPC rose from
$15,000 to $20,300/MWh over 2021 to 2025, so a static threshold sits ever
further below it; selecting relative to the cap also leaves out intervals
where a price-following battery charges ahead of higher prices later in the
same event. Net import is counted per interval with equal weight (no
weighting by price), since the question is the physical response, not the
financial one.

Two stages:

  extract   one pass over the flows.parquet files (achievable imperfect
            foresight, optimised mode, as the thesis event exposure charts),
            caching per run, state and year the PV-size-weighted cohort net
            import and the weighted share of sites importing, for every
            interval within CONTEXT_HOURS of an interval above EXTRACT_FLOOR.
            A cache file that exists is not rebuilt, so the extract can be
            run in pieces (--states, --years).
  charts    from the cache only, for each selection, into MPC (x = 0),
            MPC-<x> or T<threshold>:

  event_exposure_national.png   mean net import per site in the selected
                                intervals, by cohort and offer (as
                                Figure D.18)
  event_exposure_<state>.png    the same for each state
  flow_duration_count.png       cohort net import in every selected
                                interval, sorted, equal width (as
                                Figure D.17)
  flow_duration_count_all.png   the same with every offer available in
                                every state
  event_response.png            mean net import by time from the start of
                                an event, events pooled, with the share of
                                events still in progress shaded
  events.csv                    one row per event: state, start, duration,
                                prices, net import per scenario (mean and
                                worst interval)
  interval_summary.csv          per scenario: selected intervals, mean net
                                import, share of intervals and of events
                                with cohort net import, worst interval, mean
                                share of sites importing
  state_year_counts.csv         selected intervals and events per state-year

An event is a run of selected intervals in one state, with gaps of up to
EVENT_GAP_MIN minutes joined into one event.

Sign convention: net import positive, kW/site (flows are kWh per 5-minute
interval, so kW = kWh × 12).

Run from repo root:
  python3 analysis/scripts/reliability_response_analysis.py --data-root "/Volumes/Manual/Thesis Project"
  python3 analysis/scripts/reliability_response_analysis.py --charts-only --mpc-offsets 0 500 1000 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import financial_analysis as fa
import flow_exposure_analysis as fe
import savings_bar_figure as sbf
import thesis_style as ts
from financial_analysis import load_weights_config
from flow_exposure_analysis import (
    INTERVAL_HOURS, RRP_COL, compute_site_weight_for_scenario,
    discover_scenarios, get_prices, load_flows, weighted_row_mean,
)

BASIS = 'achievable'
MODE = 'optimised'
EXTRACT_FLOOR = 5000.0          # $/MWh; the cache covers any T at or above this
CONTEXT_HOURS = 12              # cached either side of every >EXTRACT_FLOOR interval
EVENT_GAP_MIN = 30
RESPONSE_WINDOW_H = (-12.0, 6.0)
DEFAULT_MPC_OFFSETS = (0, 1000)
HEIGHT_IN = 3.0
DOLLAR = '\\$'
BASIS_SUFFIX = f", {fa.RUN_BASES[BASIS]['title']}"
# With --with-conservative: the basis is carried by the bar legend and the
# line labels, so the titles keep their length.
CONSERVATIVE = 'conservative'
RANGE_SUFFIX = ', imperfect foresight'
RANGE_STEM = '_range'
CORE_LABELS = ('No CER (load only)', 'Solar Only', 'Self Consumption (Amber)', 'Amber')
REFERENCE_LABELS = CORE_LABELS[:3]
# Market price cap by financial year (starting 1 July of the key year), $/MWh.
MPC_BY_FY = {2020: 15000, 2021: 15100, 2022: 15500, 2023: 16600, 2024: 17500,
             2025: 20300}


def scenario_list(with_conservative: bool = False) -> list[dict]:
    """The event exposure bars of flow_exposure_analysis.py in optimised mode
    on the achievable basis: label, colour, run_dir, hatch, plus has_battery,
    line style, basis, the states left out and a label tag for the duration
    and response charts. With with_conservative, each offer's conservative
    run follows, labelled '<offer> (conservative)', with 'of' naming the
    achievable scenario it pairs with."""
    fe.set_basis(BASIS)
    plans = {p['label']: p for p in fe.COORDINATION_PLANS}
    out = []
    for label, colour, run_dir, hatch in fe._bar_order(MODE):
        plan = plans.get(label) if label not in REFERENCE_LABELS else None
        out.append(dict(label=label, color=colour, run_dir=run_dir, hatch=hatch,
                        has_battery=label not in REFERENCE_LABELS[:2],
                        ls='-' if label in REFERENCE_LABELS else '--', basis=BASIS,
                        excluded=fe._excluded_states(plan) if plan else set(),
                        tag=' (achievable)' if plan and with_conservative else ''))
    if with_conservative:
        cons = {p['label']: p for p in fe.plans_for_basis(CONSERVATIVE)}
        for sc in list(out):
            if sc['label'] in REFERENCE_LABELS or sc['label'] not in cons:
                continue
            out.append(dict(sc, label=f"{sc['label']} ({CONSERVATIVE})", of=sc['label'],
                            run_dir=cons[sc['label']]['lp_dir'], ls=':',
                            basis=CONSERVATIVE, tag=f' ({CONSERVATIVE})'))
    return out


def _fmt_dollars(t: float) -> str:
    return f'{DOLLAR}{t:,.0f}'


def _name(sc: dict, battery_kwh: int) -> str:
    # Only one self-consumption scenario is shown, so it needs no offer name.
    label = sc.get('of', sc['label'])
    name = (fe.SELF_CONSUMPTION_NAME if label == 'Self Consumption (Amber)'
            else fe._display_name(label))
    name = f'{name}, {battery_kwh} kWh' if sc['has_battery'] else name
    return name + sc['tag']


def _minus(text: str) -> str:
    return text.replace('-', '−')


def mpc_at(index: pd.DatetimeIndex) -> np.ndarray:
    """Market price cap in force at each interval end."""
    start = index - pd.Timedelta(hours=INTERVAL_HOURS)
    fy = np.where(start.month >= 7, start.year, start.year - 1)
    return np.array([MPC_BY_FY.get(int(y), np.nan) for y in fy])


class Selection:
    """The intervals taken as system stress: price above a static threshold
    (kind 'static', value $/MWh), or at or above the market price cap in
    force less an offset (kind 'mpc', value = offset $/MWh, 0 for intervals
    at the cap)."""

    def __init__(self, kind: str, value: float):
        self.kind, self.value = kind, float(value)

    def mask(self, rrp: pd.Series) -> pd.Series:
        if self.kind == 'static':
            return rrp > self.value
        return rrp >= mpc_at(rrp.index) - self.value - 0.01

    @property
    def phrase(self) -> str:
        if self.kind == 'static':
            return f'>{_fmt_dollars(self.value)}/MWh'
        if self.value == 0:
            return 'at the market price cap'
        return f'within {_fmt_dollars(self.value)}/MWh of the market price cap'

    @property
    def folder(self) -> str:
        if self.kind == 'static':
            return f'T{self.value:.0f}'
        return 'MPC' if self.value == 0 else f'MPC-{self.value:.0f}'

    @property
    def floor(self) -> float:
        """Lowest price the selection can take, for the cache floor check."""
        return self.value if self.kind == 'static' else min(MPC_BY_FY.values()) - self.value


# ── Extract ──────────────────────────────────────────────────────────────────

def _cache_path(cache_dir: Path, run_dir: str, state: str, year: str) -> Path:
    return cache_dir / run_dir / f'{state}_{year}.parquet'


def context_mask(rrp: pd.Series) -> np.ndarray:
    """True within CONTEXT_HOURS of any interval with rrp > EXTRACT_FLOOR."""
    hot = (rrp > EXTRACT_FLOOR).to_numpy()
    k = int(CONTEXT_HOURS / INTERVAL_HOURS)
    if not hot.any():
        return hot
    c = np.concatenate([[0], np.cumsum(hot)])
    n = len(hot)
    i = np.arange(n)
    lo = np.clip(i - k, 0, n)
    hi = np.clip(i + k + 1, 0, n)
    return (c[hi] - c[lo]) > 0


def extract(data_root: Path, cache_dir: Path, scenarios: list[tuple[str, str]],
            scen_list: list[dict], weights_cfg: dict, battery_kwh: int) -> None:
    runs_root = data_root / 'data' / 'output' / 'runs'
    price_cache: dict[str, pd.DataFrame] = {}
    sw_cache: dict[tuple[str, str], pd.Series] = {}
    dc_ac_cache: dict[str, float] = {}
    meta_cache: dict[str, pd.DataFrame] = {}
    rows = []
    for sc in scen_list:
        for state, year in scenarios:
            if state in sc['excluded']:
                continue
            out = _cache_path(cache_dir, sc['run_dir'], state, year)
            if out.exists():
                continue
            df = load_flows(runs_root, sc['run_dir'], state, year,
                            battery_kwh=battery_kwh if sc['has_battery'] else None)
            if df is None:
                continue
            if (state, year) not in sw_cache:
                sw_cache[(state, year)] = compute_site_weight_for_scenario(
                    data_root, weights_cfg, state, year, dc_ac_cache, meta_cache)
            sw = sw_cache[(state, year)]
            rrp = get_prices(data_root, year, price_cache)[RRP_COL[state]]
            rrp = rrp.reindex(df.index)
            keep = context_mask(rrp.fillna(-np.inf))
            sub = df.loc[keep]
            net_kw = weighted_row_mean(sub, sw) / INTERVAL_HOURS
            share_imp = weighted_row_mean((sub > 0).astype(float), sw)
            cols = [c for c in df.columns if c in sw.index]
            frame = pd.DataFrame({'rrp': rrp.loc[keep], 'net_kw': net_kw,
                                  'share_importing': share_imp})
            frame.index.name = 'datetime_aest'
            out.parent.mkdir(parents=True, exist_ok=True)
            frame.reset_index().to_parquet(out, index=False)
            rows.append((sc['run_dir'], f'{state}_{year}', len(df.columns), len(cols),
                         int((frame['rrp'] > EXTRACT_FLOOR).sum()), len(frame)))
    if rows:
        print(f"\n{'Run':<42} {'Scenario':<9} {'Sites':>5} {'Used':>5} "
              f"{'>floor':>6} {'Cached':>6}")
        for r in rows:
            print(f"{r[0]:<42} {r[1]:<9} {r[2]:>5} {r[3]:>5} {r[4]:>6} {r[5]:>6}")
    else:
        print("  Nothing new to extract.")


# ── Load cache ───────────────────────────────────────────────────────────────

def load_cache(cache_dir: Path, scen_list: list[dict],
               scenarios: list[tuple[str, str]]) -> dict:
    """data[(label, state, year)] -> frame indexed by datetime_aest."""
    data, missing = {}, []
    for sc in scen_list:
        for state, year in scenarios:
            if state in sc['excluded']:
                continue
            p = _cache_path(cache_dir, sc['run_dir'], state, year)
            if not p.exists():
                missing.append(f"{sc['run_dir']} {state}_{year}")
                continue
            data[(sc['label'], state, year)] = (pd.read_parquet(p)
                                                .set_index('datetime_aest'))
    if missing:
        print(f"  WARNING: {len(missing)} run/state-years not in the cache, left out:",
              file=sys.stderr)
        for m in missing:
            print(f"    {m}", file=sys.stderr)
    return data


# ── Events ───────────────────────────────────────────────────────────────────

def find_events(rrp: pd.Series, sel: Selection) -> pd.DataFrame:
    """Runs of selected intervals in one state, joined across gaps of up to
    EVENT_GAP_MIN. Returns start/end (interval ends)."""
    hot = rrp[sel.mask(rrp)].sort_index()
    if hot.empty:
        return pd.DataFrame(columns=['start', 'end'])
    gap = hot.index.to_series().diff() > pd.Timedelta(minutes=EVENT_GAP_MIN
                                                       + 60 * INTERVAL_HOURS)
    eid = gap.cumsum().to_numpy()
    g = pd.Series(hot.index, index=eid)
    return pd.DataFrame({'start': g.groupby(level=0).min(),
                         'end': g.groupby(level=0).max()}).reset_index(drop=True)


# ── Range bar chart ──────────────────────────────────────────────────────────

def fig_range_bar(stat: dict, scen_list: list[dict], title: str, stem: str,
                  state: str, year_label: str, battery_kwh: int, out_dir: Path,
                  notes: bool) -> None:
    """The event exposure bars of fe.fig_scenario_bar, with each offer's bar
    running to its conservative value and a tinted segment continuing it to
    the achievable value, as in savings_bar_figure.py. Bars are labelled
    with the achievable value only."""
    cons_run = {sc['of']: sc['run_dir'] for sc in scen_list if sc['basis'] == CONSERVATIVE}
    bars = []
    for sc in scen_list:
        if sc['basis'] != BASIS or sc['run_dir'] not in stat['stats']:
            continue
        ach = stat['stats'][sc['run_dir']]['ev']
        cons = stat['stats'].get(cons_run.get(sc['label']), {}).get('ev')
        bars.append((sc, cons, ach))
    if not bars:
        print(f"  Skipping {stem}: no data.", file=sys.stderr)
        return
    mean_rrp = stat['mean_rrp'].get('ev') if stat['mean_rrp'] else None
    rrp_str = (f'\nMean RRP: {DOLLAR}{mean_rrp:,.0f}/MWh'
               if mean_rrp is not None and not np.isnan(mean_rrp) else '')
    note = (f'{state}, {year_label}, {battery_kwh} kWh battery\n'
            f'{fe._weight_note(True, national=(state == "National"))}{rrp_str}'
            f'{fe._national_lines(state)}')
    vals = []
    with fe._thesis_rc():
        fig = ts.new_figure(fe.WATERFALL_HEIGHT_IN)
        ax = fig.subplots()
        for i, (sc, cons, ach) in enumerate(bars):
            base = ach if cons is None else cons
            # The range segment goes under the solid bar, so a conservative
            # value on the other side of zero from the achievable one stays
            # visible.
            if cons is not None and abs(ach - cons) > 1e-4:
                ax.bar([i], [ach - cons], bottom=[cons], width=fe.BAR_WIDTH, zorder=3,
                       color=sbf._tint(sc['color'], sbf.RANGE_TINT), linewidth=0)
            if fe._display_name(sc['label']) == fe.SC_REFERENCE_NAME:
                for kw in sbf._sc_patches(sc['color']):
                    ax.bar([i], [base], width=fe.BAR_WIDTH, zorder=3.5, **kw)
            else:
                ax.bar([i], [base], width=fe.BAR_WIDTH, zorder=3.5,
                       **fe._bar_style(sc['hatch'], sc['color']))
            if abs(ach) > 1e-4:
                fe._signed_label(ax, i, ach, fe._num2(ach))
            vals += [v for v in (cons, ach) if v is not None]
        fe._frame_categories(ax, [fe._tick_label(sc['label']) for sc, _, _ in bars])
        ax.axhline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=2)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fe._num2(v)))
        fe._signed_ylim(ax, vals)
        ax.set_ylabel('Mean per-site net import (kW)')
        ax.set_title(title)
        if notes:
            fa._draw_notes(ax, note, 'upper right')
        handles, labels = fe._hatch_legend_handles(MODE)
        handles.append(mpatches.Patch(color=sbf._tint(sbf.KEY_COLOUR, sbf.RANGE_TINT)))
        labels.append(sbf.RANGE_LABEL)
        fig.legend(handles=handles, labels=labels, loc='outside lower center',
                   ncol=len(handles), frameon=False)
        ts.save_figure(fig, out_dir / stem)


# ── Analysis per threshold ───────────────────────────────────────────────────

def analyse_threshold(sel: Selection, data: dict, scen_list: list[dict],
                      scenarios: list[tuple[str, str]], state_mix: dict,
                      battery_kwh: int, out_dir: Path, notes: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    states = sorted({s for s, _ in scenarios})
    labels = [sc['label'] for sc in scen_list]
    # With conservative runs present, every output is the _range version.
    ranged = any(sc['basis'] == CONSERVATIVE for sc in scen_list)
    sfx = RANGE_STEM if ranged else ''
    suffix = RANGE_SUFFIX if ranged else BASIS_SUFFIX
    core = CORE_LABELS + ((f'Amber ({CONSERVATIVE})',) if ranged else ())
    ref = 'No CER (load only)'

    # Prices per state from the No CER cache (every run shares the grid).
    rrp_state = {}
    for state in states:
        parts = [data[(ref, state, y)]['rrp'] for s, y in scenarios
                 if s == state and (ref, state, y) in data]
        if parts:
            rrp_state[state] = pd.concat(parts).sort_index()

    # Per state: pooled-years mean net import over >T intervals (as the
    # flow_exposure_analysis.py per-state multi-year tier), then national by
    # state_mix over the states where the run has >T intervals.
    per_state: dict[str, dict] = {}
    for state in states:
        stat = {'stats': {}, 'mean_rrp': {}, 'n_by_run': {}}
        for sc in scen_list:
            frames = [data[(sc['label'], state, y)] for s, y in scenarios
                      if s == state and (sc['label'], state, y) in data]
            if not frames:
                continue
            comb = pd.concat(frames)
            m = sel.mask(comb['rrp'])
            if not m.any():
                continue
            stat['stats'][sc['run_dir']] = {'ev': float(comb.loc[m, 'net_kw'].mean())}
            stat['n_by_run'][sc['run_dir']] = 0
            if not stat['mean_rrp']:
                stat['mean_rrp'] = {'ev': float(comb.loc[m, 'rrp'].mean())}
        if stat['stats']:
            per_state[state] = stat

    national = {'stats': {}, 'mean_rrp': {}, 'n_by_run': {}}
    for sc in scen_list:
        st = [s for s in per_state if sc['run_dir'] in per_state[s]['stats']]
        if not st:
            continue
        v = fe._state_mix_weighted_average(
            [per_state[s]['stats'][sc['run_dir']]['ev'] for s in st], st, state_mix)
        national['stats'][sc['run_dir']] = {'ev': v}
        national['n_by_run'][sc['run_dir']] = 0
    national['mean_rrp'] = {'ev': fe._state_mix_weighted_average(
        [per_state[s]['mean_rrp']['ev'] for s in per_state], list(per_state), state_mix)}

    years = sorted({y for _, y in scenarios})
    year_label = f'{years[0]}–{years[-1]}' if len(years) > 1 else years[0]
    # The mode is left out of the title (always optimised here) so it fits
    # the axes width at the longest selection phrase.
    title = f'Net import in intervals {sel.phrase}{suffix}'
    fe.NATIONAL_NOTE_LINES[:] = fa.restricted_offer_notes(list(per_state))
    for state, stat in [('National', national)] + list(per_state.items()):
        stem = f"event_exposure_{'national' if state == 'National' else state}{sfx}"
        if ranged:
            fig_range_bar(stat, scen_list, title, stem, state, year_label,
                          battery_kwh, out_dir, notes)
        else:
            fe.fig_scenario_bar('ev', MODE, title, stem, state, year_label, stat,
                                True, battery_kwh, out_dir, show_n=False, notes=notes)

    # Flow duration, equal width: each (state, interval) weighted by
    # state_mix[s] / Y_s as in cap_hedge_analysis.py, so states count by
    # their mix and not by how often their price exceeded T.
    curves = {}
    for sc in scen_list:
        st = [s for s in states if any((sc['label'], s, y) in data for _, y in scenarios)]
        if set(st) != set(per_state) and sc['label'] not in core:
            continue
        ms, ws = [], []
        for s in st:
            yrs = [y for ss, y in scenarios if ss == s and (sc['label'], s, y) in data]
            vals = np.concatenate([
                data[(sc['label'], s, y)].loc[lambda d: sel.mask(d['rrp']), 'net_kw']
                .to_numpy() for y in yrs])
            ms.append(vals)
            ws.append(np.full(len(vals), state_mix[s] / len(yrs)))
        m, w = np.concatenate(ms), np.concatenate(ws)
        if len(m) == 0:
            continue
        order = np.argsort(-m, kind='stable')
        x = np.concatenate([[0.0], np.cumsum(w[order]) / w.sum() * 100.0])
        y = np.concatenate([m[order], m[order][-1:]])
        curves[sc['label']] = (x, y)
    for stem, keep in ((f'flow_duration_count{sfx}', core),
                       (f'flow_duration_count_all{sfx}',
                        tuple(sc['label'] for sc in scen_list if sc['basis'] == BASIS))):
        fig = ts.new_figure(HEIGHT_IN)
        ax = fig.subplots()
        for sc in scen_list:
            if sc['label'] not in keep or sc['label'] not in curves:
                continue
            x, y = curves[sc['label']]
            ax.step(x, y, where='post', color=sc['color'], ls=sc['ls'], lw=ts.DATA_LW,
                    label=_name(sc, battery_kwh), zorder=3)
        ax.axhline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=1)
        ax.set_xlim(0, 100)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _minus(f'{v:g}')))
        ax.set_title(f'Cohort net import in intervals {sel.phrase}' + suffix)
        ax.set_xlabel(f'Share of intervals {sel.phrase} (%)')
        ax.set_ylabel('Cohort net import (kW/site)')
        if keep == core and not ranged:
            ax.legend(loc='lower left')
        elif keep == core:
            # The conservative Amber curve fills the lower left; two columns
            # below the axes put the two Amber lines side by side.
            fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center',
                       ncol=2, frameon=False)
        else:
            # Seven entries do not fit inside the axes clear of the curves.
            fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center',
                       ncol=4, frameon=False)
        if notes:
            n_int = sum(int(sel.mask(r).sum()) for r in rrp_state.values())
            fa._draw_notes(ax, f'National, {year_label}, {n_int:,} state-intervals\n'
                               'Weighted by PV size & state mix', 'upper right')
        ts.save_figure(fig, out_dir / stem)

    # Events, per state, with each scenario's response.
    ev_rows = []
    resp = {lab: [] for lab in labels}
    offs = np.arange(int(RESPONSE_WINDOW_H[0] / INTERVAL_HOURS),
                     int(RESPONSE_WINDOW_H[1] / INTERVAL_HOURS) + 1)
    step = pd.Timedelta(hours=INTERVAL_HOURS)
    for state, rrp in rrp_state.items():
        ev = find_events(rrp, sel)
        series = {lab: pd.concat([data[(lab, state, y)]['net_kw'] for s, y in scenarios
                                  if s == state and (lab, state, y) in data]).sort_index()
                  for lab in labels
                  if any((lab, state, y) in data for s, y in scenarios if s == state)}
        shares = {lab: pd.concat([data[(lab, state, y)]['share_importing']
                                  for s, y in scenarios
                                  if s == state and (lab, state, y) in data]).sort_index()
                  for lab in series}
        for _, e in ev.iterrows():
            win = rrp.loc[e['start']:e['end']]
            hot = win[sel.mask(win)]
            row = dict(state=state,
                       start=e['start'] - step, end=e['end'],
                       n_intervals=len(hot),
                       span_h=(e['end'] - e['start']) / pd.Timedelta(hours=1)
                       + INTERVAL_HOURS,
                       mean_rrp=hot.mean(), max_rrp=hot.max(),
                       mpc=float(mpc_at(hot.index[:1])[0]),
                       share_at_mpc=float((hot >= mpc_at(hot.index) - 1e-6).mean()))
            for lab, s in series.items():
                v = s.reindex(hot.index)
                row[f'{lab} mean kW'] = v.mean()
                row[f'{lab} max kW'] = v.max()
                row[f'{lab} share sites importing'] = shares[lab].reindex(hot.index).mean()
                idx = e['start'] + offs * step
                resp[lab].append(s.reindex(idx).to_numpy())
            ev_rows.append(row)
    events = pd.DataFrame(ev_rows).sort_values('start').reset_index(drop=True)
    events.to_csv(out_dir / f'events{sfx}.csv', index=False, float_format='%.4g')
    print(f"  Saved: {out_dir / f'events{sfx}.csv'} ({len(events)} events)")

    # Mean response around event start, events pooled with equal weight.
    fig = ts.new_figure(HEIGHT_IN)
    ax = fig.subplots()
    h = offs * INTERVAL_HOURS
    for sc in scen_list:
        if sc['label'] not in core or not resp[sc['label']]:
            continue
        arr = np.vstack(resp[sc['label']])
        ax.plot(h, np.nanmean(arr, axis=0), color=sc['color'], ls=sc['ls'],
                lw=ts.DATA_LW, label=_name(sc, battery_kwh), zorder=3)
    # Share of events still above T (any later >T interval) at each offset.
    in_ev = np.zeros(len(offs))
    for _, e in events.iterrows():
        rel = (e['end'] - (e['start'] + step)) / step
        in_ev += (offs >= 0) & (offs <= rel)
    ax2 = ax.twinx()
    ax2.fill_between(h, in_ev / max(len(events), 1) * 100, step='mid',
                     color=ts.REFERENCE_COLOUR, alpha=0.12, lw=0, zorder=0)
    ax2.set_ylim(0, 100)
    ax2.grid(False)
    ax2.set_ylabel('Events still in progress (%)')
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.axhline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=1)
    ax.axvline(0, color=ts.REFERENCE_COLOUR, ls='--', lw=ts.REFERENCE_LW, zorder=2)
    ax.set_xlim(*RESPONSE_WINDOW_H)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(3))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _minus(f'{v:g}')))
    ax.set_title(f'Cohort net import around events {sel.phrase}' + suffix)
    ax.set_xlabel('Hours from event start')
    ax.set_ylabel('Mean net import (kW/site)')
    legend = ax.legend(loc='lower left')
    if notes:
        # Notes stacked directly above the legend, both in the lower left.
        fig.canvas.draw()
        top = legend.get_window_extent().transformed(ax.transAxes.inverted()).y1
        fa._draw_notes(ax, f'{len(events)} events, all states, {year_label}\n'
                           'Each event equal weight; weighted by PV size',
                       'lower left', bottom=top + 0.02)
    ts.save_figure(fig, out_dir / f'event_response{sfx}')

    # Interval and event summary per scenario (national, as the bars).
    summ = []
    for sc in scen_list:
        lab = sc['label']
        frames = [data[(lab, s, y)].loc[lambda d: sel.mask(d['rrp'])] for s, y in scenarios
                  if (lab, s, y) in data]
        if not frames:
            continue
        hot = pd.concat(frames)
        if hot.empty:
            continue
        col = f'{lab} max kW'
        evs = events.dropna(subset=[col]) if col in events else events.iloc[0:0]
        summ.append(dict(
            scenario=lab, run_dir=sc['run_dir'],
            national_mean_kw=national['stats'].get(sc['run_dir'], {}).get('ev'),
            intervals=len(hot),
            pooled_mean_kw=hot['net_kw'].mean(),
            share_intervals_importing=(hot['net_kw'] > 0).mean(),
            worst_interval_kw=hot['net_kw'].max(),
            p95_interval_kw=hot['net_kw'].quantile(0.95),
            mean_share_sites_importing=hot['share_importing'].mean(),
            events=len(evs),
            events_with_import=int((evs[col] > 0).sum()) if len(evs) else 0,
            events_mean_import=int((evs[f'{lab} mean kW'] > 0).sum()) if len(evs) else 0))
    summary = pd.DataFrame(summ)
    summary.to_csv(out_dir / f'interval_summary{sfx}.csv', index=False,
                   float_format='%.4g')
    print(f"  Saved: {out_dir / f'interval_summary{sfx}.csv'}")
    with pd.option_context('display.width', 200, 'display.max_columns', 20,
                           'display.float_format', '{:.3f}'.format):
        print(summary.drop(columns='run_dir').to_string(index=False))

    counts = []
    for state, rrp in rrp_state.items():
        start = rrp.index - step
        for y in sorted(set(start.year)):
            r = rrp[start.year == y]
            ev = find_events(r, sel)
            counts.append(dict(state=state, year=y, intervals=int(sel.mask(r).sum()),
                               events=len(ev)))
    pd.DataFrame(counts).to_csv(out_dir / f'state_year_counts{sfx}.csv', index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('.'),
                        help='Root holding data/output/runs, data/clean/nem_prices '
                             'and data/processed/edp.')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('analysis/output/reliability'))
    parser.add_argument('--mpc-offsets', type=float, nargs='*',
                        default=list(DEFAULT_MPC_OFFSETS),
                        help='x in $/MWh: intervals at or above the market price '
                             'cap in force less x (0 = at the cap).')
    parser.add_argument('--thresholds', type=float, nargs='*', default=[],
                        help='Static thresholds in $/MWh: intervals above each.')
    parser.add_argument('--states', nargs='+', default=None)
    parser.add_argument('--years', nargs='+', default=None)
    parser.add_argument('--battery-kwh', type=int, default=20)
    parser.add_argument('--weights-file', type=Path,
                        default=Path('analysis/config/pv_size_bands.yaml'))
    parser.add_argument('--charts-only', action='store_true',
                        help='Draw from the cache without reading any flows.')
    parser.add_argument('--extract-only', action='store_true')
    parser.add_argument('--no-notes', dest='notes', action='store_false')
    parser.add_argument('--with-conservative', action='store_true',
                        help='Add each offer\'s conservative run: range bars, the '
                             'conservative Amber line, outputs named *_range.')
    args = parser.parse_args(argv)

    selections = ([Selection('mpc', x) for x in args.mpc_offsets]
                  + [Selection('static', t) for t in args.thresholds])
    if any(s.floor < EXTRACT_FLOOR for s in selections):
        print(f"ERROR: every selection must stay at or above {EXTRACT_FLOOR:,.0f} $/MWh, "
              "the cache floor.", file=sys.stderr)
        return 1
    ts.apply_style()
    weights_cfg = load_weights_config(args.weights_file)
    scen_list = scenario_list(args.with_conservative)
    cache_dir = args.output_dir / 'cache'

    if args.charts_only:
        scenarios = sorted({tuple(p.stem.split('_')) for p in cache_dir.glob('*/*.parquet')})
    else:
        scenarios = discover_scenarios(args.data_root)
    if args.states:
        scenarios = [(s, y) for s, y in scenarios if s in args.states]
    if args.years:
        scenarios = [(s, y) for s, y in scenarios if y in args.years]
    if not scenarios:
        print("ERROR: no scenarios found or none match the filter.", file=sys.stderr)
        return 1

    if not args.charts_only:
        print("Extracting...")
        extract(args.data_root, cache_dir, scenarios, scen_list, weights_cfg,
                args.battery_kwh)
        if args.extract_only:
            return 0

    data = load_cache(cache_dir, scen_list, scenarios)
    for sel in selections:
        print(f"\n=== Intervals {sel.phrase.replace(DOLLAR, '$')} ===")
        analyse_threshold(sel, data, scen_list, scenarios, weights_cfg['state_mix'],
                          args.battery_kwh, args.output_dir / sel.folder, args.notes)
    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
