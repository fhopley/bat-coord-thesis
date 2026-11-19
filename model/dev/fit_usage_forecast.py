"""fit_usage_forecast: choose the adjustment lookback x and decay y.

Run from the project root:

    python3 -m model.dev.fit_usage_forecast --year 2022 \
        --data-root "/Volumes/Manual/Thesis Project"

    python3 -m model.dev.fit_usage_forecast --year 2022 --state NSW --sites 25 \
        --data-root "/Volumes/Manual/Thesis Project"

Sites and their states are discovered from the EDP metadata CSV under
`--data-root`, the same resolution orchestrate.validate_scenario performs.
Results are written to <data-root>/data/processed/usage_fit by default,
alongside the metadata they were derived from.

Pure forecast-error work. The LP is not involved, and neither is any price
data, so this is cheap and can run over many more sites than a dispatch sweep.

What it fits, and why that metric
---------------------------------
The primary metric is CUMULATIVE ENERGY ERROR over 1, 3, 6 and 12 hours from
each decision instant, not per-interval MAE. The battery plans an energy
trajectory: what the LP consumes from the load forecast is how much energy the
house will take between now and the evening peak, not what it will draw in any
particular five minutes. A forecast can have a mediocre MAE and still size the
overnight charge correctly, or a good MAE and get the total wrong. Per-interval
MAE is reported alongside, and if the two disagree the cumulative one is the
one that bears on dispatch.

Window position 0 is excluded from every metric, because the component commits
the decision interval with actual load. Only positions 1 and beyond reach the
plan.

Why the sweep is nearly free
----------------------------
Write r(t) for the residual of the BASE profile, r = base - actual. The
adjustment is a(D) = -mean of r over the last x hours, and the forecast error
at lead l is then

    e(D, l) = r(D+l) + a(D) * exp(-l / y)

so both terms are linear in quantities that do not depend on y at all, and the
only thing x changes is a scalar per decision instant. Cumulative error over
lead L is therefore

    E(D, L) = R(D, L) - mean_x(r)(D) * S(L, y)

with R the cumulative base residual and S the cumulative decay. R is computed
once per build day and reused across the whole grid.

Held-out discipline
-------------------
Fit on EDP and apply the result FIXED to the Energy Masters cohort. Fitting x
and y on EM -- even on forecast error rather than on VCR -- puts a free
parameter into the number being reported. A per-site fit is reported too, but
only to show the dispersion: a per-site (x, y) is one fitted parameter per site
and would weaken the cohort claim considerably. If the global optimum sits
inside the spread of the per-site optima, use the global one and say so.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from model.forecast.usage import (
    SLOTS_PER_DAY, UsageForecastConfig, UsageForecastProvider,
)
from model.dev.check_usage_forecast import wall_clock

# Cumulative-energy horizons, in 5-minute intervals.
LEADS = (12, 36, 72, 144)                    # 1h, 3h, 6h, 12h
LEAD_LABELS = ('1h', '3h', '6h', '12h')

# The floor of x is ONE INTERVAL: the adjustment cannot average over less than
# a single observation. The grid reaches it, because the first cohort fit used
# a grid starting at 0.5 h and 74 % of sites optimised to that lowest value --
# an edge, not an optimum.
DEFAULT_X = (5 / 60, 1 / 6, 0.5, 1.0, 2.0, 3.0)
DEFAULT_Y = (0.5, 1.0, 2.0, 3.0, 6.0, 12.0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_site(path: str, site_id: str | None = None):
    """Read one site's gross load and validity mask from a clean EDP parquet.

    Clean EDP files are one site per file, named
    data/clean/edp/{year}/edp_clean_{site}_{year}.parquet, with `datetime_aest`
    as a COLUMN and no site id column at all -- the id lives in the filename.
    A long file carrying a site id column is also accepted, so a slim extract
    works without pre-processing.

    Returns (load_kwh Series, valid mask, site_id).

    `valid` is False wherever `load_infill` is non-zero. The exclusion is not
    optional for this work: EDP's Method 2 load infill copies the same day of
    week, which is close to what the forecaster does, so infilled intervals
    would both inflate the profile's apparent regularity and, as evaluation
    targets, measure the forecaster against a near-copy of itself.
    """
    df = pd.read_parquet(path)
    if 'datetime_aest' in df.columns:
        df = df.set_index('datetime_aest')
    if df.index.tz is None:
        df.index = df.index.tz_localize('Etc/GMT-10')
    df = df.sort_index()

    id_col = next((c for c in ('site_id', 'edp_site_id') if c in df.columns), None)
    if id_col is not None:
        ids = df[id_col].unique()
        if site_id is None:
            if len(ids) != 1:
                raise ValueError(
                    f"{path} holds {len(ids)} sites; pass --site. "
                    f"First few: {list(ids[:5])}")
            site_id = ids[0]
        if site_id not in set(ids):
            raise ValueError(f"site {site_id!r} not in {path}")
        df = df[df[id_col] == site_id]
    elif site_id is None:
        stem = Path(path).stem                       # edp_clean_S0008_2022
        parts = stem.split('_')
        site_id = parts[2] if len(parts) >= 3 else stem

    load = None
    for col, scale in (('gross_load_kwh', 1.0), ('gross_load_wh', 1e-3)):
        if col in df.columns:
            load = df[col].astype(float) * scale
            break
    if load is None:
        raise ValueError(
            f"{path} has no gross_load_kwh or gross_load_wh column. "
            f"Columns present: {list(df.columns)}")

    if 'load_infill' in df.columns:
        valid = (df['load_infill'].to_numpy() == 0)
    else:
        valid = np.ones(len(load), dtype=bool)
    return load.rename('gross_load_kwh'), valid, site_id


@dataclass(frozen=True)
class SiteRef:
    """One site to fit, with the state its local clock must use."""
    path: str
    site_id: str
    state: str


def fit_output_dir(data_root: str) -> Path:
    """Default destination for fit results.

    Sits beside data/processed/edp, which is where the metadata the fit reads
    lives, so the result and its inputs stay together on whichever volume the
    data is on.
    """
    return Path(data_root) / 'data' / 'processed' / 'usage_fit'


def resolve_sites(data_root: str, year: int, states: list[str] | None,
                  limit: int | None) -> list[SiteRef]:
    """Discover site files and their states the way validate_scenario does.

    The state map is the EDP metadata CSV at
    data/processed/edp/{year}_selected_sites_clean.csv, keyed by `edp_site_id`
    with a `state` column. Traces sit at
    data/clean/edp/{year}/edp_clean_{site}_{year}.parquet.

    State is resolved PER SITE rather than passed once for the run, because a
    year folder holds every state at once and the local day boundary that the
    weekday/weekend split and the day-of-week level both depend on is
    state-specific. SA sits half an hour off market time permanently and
    NSW/VIC swing an hour seasonally.
    """
    r = Path(data_root)
    meta_path = r / 'data' / 'processed' / 'edp' / f'{year}_selected_sites_clean.csv'
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata not found: {meta_path}. Pass --edp with an explicit "
            f"--state to fit specific files instead.")
    meta = pd.read_csv(meta_path)
    for col in ('edp_site_id', 'state'):
        if col not in meta.columns:
            raise ValueError(
                f"{meta_path} has no {col!r} column; found {list(meta.columns)}")
    if states:
        meta = meta[meta['state'].isin(states)]

    trace_dir = r / 'data' / 'clean' / 'edp' / str(year)
    refs, missing = [], []
    for sid, st in zip(meta['edp_site_id'], meta['state']):
        f = trace_dir / f'edp_clean_{sid}_{year}.parquet'
        (refs if f.exists() else missing).append(
            SiteRef(str(f), str(sid), str(st)) if f.exists() else str(sid))
    if missing:
        print(f"  {len(missing)} site(s) in metadata have no trace file "
              f"(first 5: {missing[:5]})")
    refs.sort(key=lambda x: (x.state, x.site_id))
    return refs[:limit] if limit else refs


def explicit_sites(paths: list[str], state: str) -> list[SiteRef]:
    """Expand explicit files, directories and globs under one stated state."""
    out: list[str] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out += [str(q) for q in sorted(p.rglob('*.parquet'))]
        elif any(ch in raw for ch in '*?['):
            out += [str(q) for q in sorted(Path().glob(raw))]
        else:
            out.append(raw)
    refs = []
    for f in out:
        parts = Path(f).stem.split('_')
        refs.append(SiteRef(f, parts[2] if len(parts) >= 3 else Path(f).stem,
                            state))
    return refs


# ---------------------------------------------------------------------------
# Residual assembly
# ---------------------------------------------------------------------------

def residual_blocks(load: np.ndarray, clock, cfg: UsageForecastConfig,
                    H: int, m_list: tuple[int, ...],
                    valid: np.ndarray | None = None):
    """Per build day, the base residual matrix and the lookback residuals.

    Returns (R_cum, LB) where

        R_cum : (n_decisions, len(LEADS)) cumulative base residual, positions
                1..lead inclusive
        LB    : (n_decisions, n_x) mean base residual over each candidate
                lookback, one column per x

    Both are stacked over every usable decision instant in the period.
    """
    m_max = max(m_list)
    # The provider's per-build-day base array reaches back by ITS OWN
    # lookback, so it must be built with the longest x in the grid or the
    # widest lookback would fall off the front of the array.
    if int(round(cfg.lookback_hours / (5 / 60))) < m_max:
        raise ValueError(
            f"provider lookback {cfg.lookback_hours}h is shorter than the "
            f"widest x in the grid ({m_max} intervals); build the config with "
            f"lookback_hours=max(x)"
        )
    if valid is None:
        valid = np.ones(len(load), dtype=bool)
    provider = UsageForecastProvider(load, clock, cfg, H, valid=valid)
    day_ord = clock.day_ordinal
    n_days = int(day_ord[-1]) + 1
    starts = np.searchsorted(day_ord, np.arange(n_days), side='left')
    ends = np.searchsorted(day_ord, np.arange(n_days), side='right')

    r_cum_all, lb_all = [], []
    max_lead = max(LEADS)

    for d in range(n_days):
        s, e = int(starts[d]), int(ends[d])
        lo = max(0, s - m_max)
        hi = min(len(load), e + max_lead)
        base = provider.base_at(d, lo, hi)
        if base is None:
            continue
        r = base - load[lo:hi]                      # base residual on this span

        # Decision instants inside day d that have the full lead available.
        ks = np.arange(s, min(e, len(load) - max_lead))
        ks = ks[ks - lo >= 1]                       # need >= 1 lookback interval
        if len(ks) == 0:
            continue
        # Drop decision instants whose evaluated span, or whose lookback,
        # touches an infilled interval. A contaminated target would be scored
        # against synthetic data; a contaminated lookback would feed the
        # adjustment a number the meter never produced.
        vpad = np.concatenate([[0], np.cumsum(~valid)])
        bad_fwd = (vpad[ks + max_lead] - vpad[ks + 1]) > 0
        bad_back = (vpad[ks] - vpad[np.maximum(ks - max(m_list), 0)]) > 0
        ks = ks[~(bad_fwd | bad_back)]
        if len(ks) == 0:
            continue
        off = ks - lo

        # Cumulative residual over window positions 1..lead.
        r_pad = np.concatenate([[0.0], np.cumsum(r)])
        r_cum = np.empty((len(ks), len(LEADS)))
        for i, lead in enumerate(LEADS):
            # positions 1..lead-1 map to intervals k+1 .. k+lead-1
            r_cum[:, i] = r_pad[off + lead] - r_pad[off + 1]
        r_cum_all.append(r_cum)

        # Mean residual over each candidate lookback, ending at k-1 inclusive.
        lb = np.empty((len(ks), len(m_list)))
        for j, m in enumerate(m_list):
            start = np.maximum(off - m, 0)
            n = off - start
            lb[:, j] = (r_pad[off] - r_pad[start]) / np.maximum(n, 1)
        lb_all.append(lb)

    if not r_cum_all:
        raise ValueError("no usable decision instants; period too short")
    return np.vstack(r_cum_all), np.vstack(lb_all)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep(R_cum: np.ndarray, LB: np.ndarray, xs, ys,
          interval_hours: float = 5 / 60) -> pd.DataFrame:
    """Mean absolute cumulative energy error for every (x, y) plus the null.

    The null (y = 0, no adjustment) is included as its own row. If the
    adjustment cannot beat it the adjustment should not be built.
    """
    rows = []
    null = np.abs(R_cum).mean(axis=0)
    rows.append({'x_hours': np.nan, 'y_hours': 0.0,
                 **{f'cum_{lab}': v for lab, v in zip(LEAD_LABELS, null)}})

    for j, x in enumerate(xs):
        a = -LB[:, j]                              # a = -mean(base - actual)
        for y in ys:
            lead_h = (np.arange(max(LEADS)) + 1) * interval_hours
            decay = np.exp(-lead_h / y)
            dcum = np.concatenate([[0.0], np.cumsum(decay)])
            vals = []
            for i, lead in enumerate(LEADS):
                s = dcum[lead] - dcum[1]
                vals.append(np.abs(R_cum[:, i] + a * s).mean())
            rows.append({'x_hours': x, 'y_hours': y,
                         **{f'cum_{lab}': v for lab, v in
                            zip(LEAD_LABELS, vals)}})

    df = pd.DataFrame(rows)
    # Selection scalar: mean of each horizon's error normalised by the null at
    # that horizon, so no single horizon dominates through its scale alone.
    for lab, n in zip(LEAD_LABELS, null):
        df[f'rel_{lab}'] = df[f'cum_{lab}'] / n
    df['score'] = df[[f'rel_{lab}' for lab in LEAD_LABELS]].mean(axis=1)
    return df.sort_values('score').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def equal_weight_table(site_sweeps: dict, xs, ys) -> pd.DataFrame:
    """Per-grid-point median site score and count of sites harmed.

    One vote per household, against the pooled score's one vote per kWh of
    error. Cheap: every site's sweep is already in hand.
    """
    keys, rows = None, []
    for sid, sw in site_sweeps.items():
        sw = sw.sort_values(['x_hours', 'y_hours'], na_position='first')
        if keys is None:
            keys = list(zip(sw.x_hours, sw.y_hours))
        rows.append(sw.score.to_numpy())
    if not rows:
        return pd.DataFrame()
    M = np.vstack(rows)                       # sites x grid points
    return pd.DataFrame({
        'x_hours': [k[0] for k in keys],
        'y_hours': [k[1] for k in keys],
        'median_site_score': np.median(M, axis=0),
        'mean_site_score': M.mean(axis=0),
        'n_harmed': (M >= 1.0).sum(axis=0),
        'n_sites': M.shape[0],
    }).sort_values('median_site_score').reset_index(drop=True)


def _edge_warnings(pooled: pd.DataFrame, xs, ys) -> list[str]:
    """Flag a pooled optimum sitting on a grid boundary.

    A fit that lands on the edge has not found an optimum, it has run out of
    grid, and reporting it as a fitted value would be wrong. The one exception
    is x at one interval, which is the floor of the parameter space rather than
    of the grid.
    """
    best = pooled.iloc[0]
    out = []
    xs_s, ys_s = sorted(xs), sorted(ys)
    if best.x_hours == xs_s[0] and abs(best.x_hours - 5 / 60) > 1e-9:
        out.append(f"x={best.x_hours}h is the LOWEST value tried; extend the "
                   f"grid downward (floor is one interval, {5 / 60:.4f}h)")
    if best.x_hours == xs_s[-1]:
        out.append(f"x={best.x_hours}h is the HIGHEST value tried; extend the "
                   f"grid upward")
    if best.y_hours == ys_s[0]:
        out.append(f"y={best.y_hours}h is the LOWEST value tried; extend the "
                   f"grid downward")
    if best.y_hours == ys_s[-1]:
        out.append(f"y={best.y_hours}h is the HIGHEST value tried; extend the "
                   f"grid upward")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--year', type=int, default=None,
                   help='discover sites and their states from '
                        'data/processed/edp/{year}_selected_sites_clean.csv')
    p.add_argument('--data-root', default='.',
                   help='root holding data/clean/edp and data/processed/edp; '
                        'the EDP files may live on an external volume, e.g. '
                        '"/Volumes/Manual/Thesis Project"')
    p.add_argument('--state', action='append', default=None,
                   help='filter with --year, or the state to assume with '
                        '--edp; repeatable')
    p.add_argument('--edp', nargs='+', default=None,
                   help='explicit parquet file(s), directory, or glob; '
                        'requires a single --state')
    p.add_argument('--sites', type=int, default=None,
                   help='cap on the number of sites to fit over')
    p.add_argument('--horizon-hours', type=float, default=18.0)
    p.add_argument('--history-days', type=int, default=28)
    p.add_argument('--smoothing', type=int, default=7)
    p.add_argument('--dow-level', action='store_true')
    p.add_argument('--x', type=float, nargs='+', default=list(DEFAULT_X))
    p.add_argument('--y', type=float, nargs='+', default=list(DEFAULT_Y))
    p.add_argument('--out', default=None,
                   help='directory for CSV output '
                        '(default: <data-root>/data/processed/usage_fit)')
    args = p.parse_args()

    interval_hours = 5 / 60
    H = int(round(args.horizon_hours / interval_hours))
    m_list = tuple(max(1, int(round(x / interval_hours))) for x in args.x)

    if args.year is not None:
        refs = resolve_sites(args.data_root, args.year, args.state,
                             args.sites)
    elif args.edp:
        if not args.state or len(args.state) != 1:
            print("--edp requires exactly one --state, since the metadata "
                  "CSV that normally supplies it is not being read.")
            return 1
        refs = explicit_sites(args.edp, args.state[0])
        if args.sites:
            refs = refs[:args.sites]
    else:
        print("Pass either --year (with --root) or --edp with --state.")
        return 1
    if not refs:
        print("No site files found.")
        return 1
    cfg = UsageForecastConfig(history_days=args.history_days,
                              smoothing_intervals=args.smoothing,
                              dow_level=args.dow_level,
                              lookback_hours=max(args.x), decay_hours=1.0)

    print(f"fit_usage_forecast")
    print(f"  data root     {args.data_root}")
    by_state: dict[str, int] = {}
    for r in refs:
        by_state[r.state] = by_state.get(r.state, 0) + 1
    print(f"  sites         {len(refs)} "
          f"({', '.join(f'{k} {v}' for k, v in sorted(by_state.items()))})")
    print(f"  profile       history {args.history_days}d, smoothing "
          f"{args.smoothing}, dow_level {args.dow_level}")
    print(f"  grid          x {args.x}")
    print(f"                y {args.y}")
    print(f"  horizons      {LEAD_LABELS}")

    per_site, R_all, LB_all = [], [], []
    site_sweeps: dict[str, pd.DataFrame] = {}
    for ref in refs:
        try:
            series, valid, sid = load_site(ref.path, ref.site_id)
        except Exception as exc:                       # noqa: BLE001
            print(f"  skip {Path(ref.path).name}: {exc}")
            continue
        clock = wall_clock(series.index, ref.state, interval_hours)
        load = series.to_numpy(dtype=float)
        if len(load) < SLOTS_PER_DAY * (args.history_days + 7):
            print(f"  skip {sid}: only {len(load) / SLOTS_PER_DAY:.0f} days")
            continue
        try:
            R, LB = residual_blocks(load, clock, cfg, H, m_list, valid)
        except ValueError as exc:
            print(f"  skip {sid}: {exc}")
            continue
        R_all.append(R)
        LB_all.append(LB)
        sw = sweep(R, LB, args.x, args.y)
        site_sweeps[str(sid)] = sw
        best = sw.iloc[0]
        per_site.append({'site_id': sid, 'state': ref.state,
                         'n_decisions': len(R),
                         'infill_frac': float((~valid).mean()),
                         'x_hours': best.x_hours, 'y_hours': best.y_hours,
                         'score': best.score})
        print(f"  {str(sid)[:12]:<12} {ref.state:<4} {len(R):>7,} decisions  "
              f"{100 * (~valid).mean():4.1f}% infilled  "
              f"best x={best.x_hours:>4}h y={best.y_hours:>4}h  "
              f"score {best.score:.4f}")

    if not R_all:
        print("No usable sites.")
        return 1

    pooled = sweep(np.vstack(R_all), np.vstack(LB_all), args.x, args.y)

    print("\nPooled fit, mean absolute cumulative energy error (kWh)")
    print("  score is the mean of each horizon's error relative to the null.")
    head = pooled.head(10).copy()
    head['x_hours'] = head['x_hours'].fillna(-1)
    print(head[['x_hours', 'y_hours'] + [f'cum_{l}' for l in LEAD_LABELS]
               + ['score']].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    null = pooled[pooled.y_hours == 0].iloc[0]
    best = pooled.iloc[0]

    print(f"\n  null (no adjustment)  score 1.0000")
    print(f"  best  x={best.x_hours}h y={best.y_hours}h  "
          f"score {best.score:.4f}  "
          f"({100 * (best.score - 1):+.1f}% vs null)")
    for lab in LEAD_LABELS:
        print(f"    {lab:>4}  {null[f'cum_{lab}']:.4f} -> "
              f"{best[f'cum_{lab}']:.4f} kWh "
              f"({100 * (best[f'cum_{lab}'] / null[f'cum_{lab}'] - 1):+.1f}%)")

    for w in _edge_warnings(pooled, args.x, args.y):
        print(f"\n  ** GRID EDGE: {w}")

    # What a site actually gives up by taking the pooled parameters instead of
    # its own. Counting how many sites share the pooled optimum answers a
    # different and much less useful question: on a flat surface almost none
    # will, and it costs them almost nothing. The loss is the number that
    # decides whether one global (x, y) is defensible.
    losses = []
    for row in per_site:
        sw = site_sweeps[str(row['site_id'])]
        m = ((sw.x_hours == best.x_hours) & (sw.y_hours == best.y_hours))
        if not m.any():
            continue
        at_pooled = float(sw.loc[m, 'score'].iloc[0])
        row['score_at_pooled'] = at_pooled
        row['loss_vs_own_pct'] = 100 * (at_pooled / row['score'] - 1)
        losses.append(row['loss_vs_own_pct'])
    ps = pd.DataFrame(per_site)

    if losses:
        L = np.array(losses)
        print("\nCost of using the pooled (x, y) instead of each site's own")
        print(f"  median {np.median(L):.2f}%  p90 {np.percentile(L, 90):.2f}%  "
              f"p99 {np.percentile(L, 99):.2f}%  max {L.max():.2f}%")
        print(f"  sites losing more than 1%: {int((L > 1).sum())} of {len(L)}; "
              f"more than 5%: {int((L > 5).sum())}")
        print("  A small loss here is what justifies one global pair; the "
              "count of sites\n  sharing the pooled optimum does not, since a "
              "flat surface makes it low\n  regardless of whether the choice "
              "matters.")

    ew = equal_weight_table(site_sweeps, args.x, args.y)
    if not ew.empty:
        ew_best = ew.iloc[0]
        print("\nEqual-weight view: one vote per site, not per kWh of error")
        print(ew.head(5).to_string(index=False,
                                   float_format=lambda v: f"{v:.4f}"))
        agree = (ew_best.x_hours == best.x_hours
                 and ew_best.y_hours == best.y_hours)
        if agree:
            print(f"  Agrees with the pooled optimum "
                  f"(x={best.x_hours}h, y={best.y_hours}h).")
        else:
            print(f"  ** DISAGREES with the pooled optimum: equal-weight "
                  f"prefers x={ew_best.x_hours}h y={ew_best.y_hours}h, "
                  f"pooled prefers x={best.x_hours}h y={best.y_hours}h.")
            print(f"  The pooled criterion is volume weighted, so "
                  f"high-consumption households pull it.")
        at_pooled = ew[(ew.x_hours == best.x_hours)
                       & (ew.y_hours == best.y_hours)]
        if not at_pooled.empty:
            r = at_pooled.iloc[0]
            print(f"  At the pooled optimum: median site score "
                  f"{r.median_site_score:.4f}, and the adjustment is WORSE "
                  f"than no adjustment\n  for {int(r.n_harmed)} of "
                  f"{int(r.n_sites)} sites.")

    print("\nPer-site dispersion of the optimum")
    print(f"  x: {ps.x_hours.value_counts().sort_index().to_dict()}")
    print(f"  y: {ps.y_hours.value_counts().sort_index().to_dict()}")
    at_global = pooled[(pooled.x_hours == best.x_hours)
                       & (pooled.y_hours == best.y_hours)]
    if ps.state.nunique() > 1:
        print("\n  per-state pooled optimum")
        for st, grp in ps.groupby('state'):
            print(f"    {st:<4} {len(grp):>3} sites  "
                  f"x mode {grp.x_hours.mode().iloc[0]}h  "
                  f"y mode {grp.y_hours.mode().iloc[0]}h")
    print(f"  sites whose own optimum equals the pooled optimum: "
          f"{int(((ps.x_hours == best.x_hours) & (ps.y_hours == best.y_hours)).sum())}"
          f" of {len(ps)}")

    out = Path(args.out) if args.out else fit_output_dir(args.data_root)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.year}" if args.year is not None else "adhoc"
    pooled.to_csv(out / f'usage_fit_pooled_{tag}.csv', index=False)
    ps.to_csv(out / f'usage_fit_per_site_{tag}.csv', index=False)
    print(f"\nWrote {out / f'usage_fit_pooled_{tag}.csv'}")
    print(f"      {out / f'usage_fit_per_site_{tag}.csv'}")
    if not ew.empty:
        ew.to_csv(out / f'usage_fit_equal_weight_{tag}.csv', index=False)
        print(f"      {out / f'usage_fit_equal_weight_{tag}.csv'}")
    # The full per-site grid, so any alternative selection criterion can be
    # tried later without re-running the fit. 433 sites by ~43 grid points is
    # a small file and it is the expensive thing to reproduce.
    long = pd.concat(
        [sw.assign(site_id=sid) for sid, sw in site_sweeps.items()],
        ignore_index=True)
    long.to_csv(out / f'usage_fit_grid_by_site_{tag}.csv', index=False)
    print(f"      {out / f'usage_fit_grid_by_site_{tag}.csv'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
