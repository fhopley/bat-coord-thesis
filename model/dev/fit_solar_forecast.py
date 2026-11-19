"""fit_solar_forecast: fit every solar forecast parameter in one run.

    python3 -m model.dev.fit_solar_forecast --year 2022 \
        --data-root "/Volumes/Manual/Thesis Project"

One run, one set of results. Every dimension is swept together -- envelope
length and quantile, the index blend, and the adjustment -- so there is no
sequence of runs to launch and no output files to rename between them. Adding
a dimension costs processing time, not bookkeeping.

Metric, held-out discipline and reporting follow fit_usage_forecast: cumulative
energy error over 1, 3, 6 and 12 hours; window position 0 excluded because it
is committed with actual generation; pooled and equal-weight criteria reported
side by side; fit on EDP and apply fixed.

Why it is affordable
--------------------
The grid has two halves with very different costs.

The ENVELOPE half -- `history_days` and `envelope_quantile` -- changes the
clear-sky proxy, which is the expensive object. Both derive from the same
per-day profile matrix, so one pass over a site's days serves every
combination: `day_profiles` is called once and `envelope_from` is applied to
slices of it.

The FORECAST half -- the blend weights and the adjustment -- changes only
scalars. Writing `E_same` and `E_other` for envelope energy over same-day and
later-day targets, `D` for envelope energy weighted by the decay, and `A` for
actual generation, the cumulative energy error over a lead is

    err = R_blend·(E_same − D) + R_hist·E_other + R_lb·D − A

so once those four are precomputed per decision instant, the whole inner grid
is arithmetic on precomputed arrays.

One approximation: the sweep ignores the clip on the forecast index. The
number of clipped positions is reported so the approximation can be checked
rather than assumed; the winning configuration is re-scored exactly.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from model.core.scenario import SolarForecastConfig
from model.forecast.clock import SLOTS_PER_DAY
from model.forecast.solar import (
    SolarForecastProvider, day_profiles, envelope_from, index_of,
)
from model.dev.check_usage_forecast import wall_clock
from model.dev.check_solar_forecast import load_solar
from model.dev.fit_usage_forecast import resolve_sites, explicit_sites

PARAM_COLS = ('history_days', 'quantile', 'prev_share', 'today_max',
              'f_on', 'x_hours', 'y_hours')

LEADS = (12, 36, 72, 144)                 # 1 h, 3 h, 6 h, 12 h
LEAD_LABELS = ('1h', '3h', '6h', '12h')

# history_days and envelope_quantile interact: a short window wants a high
# quantile, since the best of five days is already close to a clear-sky
# ceiling, while a long window wants a lower one. The grid has to allow that
# corner, which is why the quantile runs to 1.0 -- the maximum of the window.
# 56 is dropped and 28 kept as headroom: across 20 sites neither was ever
# selected, and the seasonal ceiling moves too fast for a two-month window.
DEFAULT_HISTORY = (5, 7, 14, 28)
# Low quantiles are structurally penalised: with index_max = 1.0 a low
# envelope means a clear day wants an index above one and gets clipped. The
# grid therefore runs upward, ending at 1.0 -- the maximum of the window,
# which is the natural ceiling for a clear-sky proxy and the partner of a
# short history_days.
DEFAULT_QUANTILE = (0.70, 0.80, 0.90, 0.95, 0.98, 1.00)
DEFAULT_PREV_SHARE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_TODAY_MAX = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
DEFAULT_F_ON = (0.02, 0.05, 0.15, 0.25, 0.35)
DEFAULT_X = (5 / 60, 10 / 60, 0.25, 0.5, 1.0, 2.0)
DEFAULT_Y = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)  # 0.0 is the null


@dataclass(frozen=True)
class SitePrecompute:
    """Everything the inner grid needs, for one site and one envelope setting."""
    r_prev: np.ndarray          # (n,)
    r_mean: np.ndarray          # (n,)
    r_today: np.ndarray         # (n,)
    f: np.ndarray               # (n,)
    r_lb: np.ndarray            # (n, n_x)
    e_same: np.ndarray          # (n, n_leads)
    e_other: np.ndarray         # (n, n_leads)
    a_act: np.ndarray           # (n, n_leads)
    d_env: np.ndarray           # (n, n_leads, n_y)


def precompute(solar, clock, cfg, valid, days, profs, xs, ys, stride, H):
    """Per-decision-instant terms for one envelope setting.

    Decision instants are sampled on a stride. Solar forecast skill varies
    smoothly through a day, so an hourly sample carries nearly the information
    of a 5-minute one at a twelfth of the cost, and the cost is what decides
    whether the whole grid fits in one run.
    """
    T = len(solar)
    day_ord = clock.day_ordinal
    max_lead = max(LEADS)
    cum_solar = np.concatenate([[0.0], np.cumsum(solar)])
    m_list = [max(1, int(round(x / (5 / 60)))) for x in xs]
    m_max = max(m_list)

    day_start = np.searchsorted(day_ord, np.arange(int(day_ord[-1]) + 1),
                                side='left')

    # exp(-lead/y) over the full horizon, fixed for the whole site, so built
    # once rather than per decision instant.
    decay_full = [None if y <= 0
                  else np.exp(-(np.arange(max_lead) + 1) * (5 / 60) / y)
                  for y in ys]

    rows = []
    for i, d in enumerate(days):
        if i == 0:
            continue
        M = profs[max(0, i - cfg.history_days):i]
        envelope = envelope_from(M, cfg.envelope_quantile)
        day_energy = float(envelope.sum())
        if day_energy <= 0:
            continue
        present = ~np.isnan(M)
        den = float((present * envelope).sum())
        if den <= 0:
            continue
        r_mean = float(np.nansum(M) / den)
        r_prev = index_of(profs[i - 1], envelope)
        if not np.isfinite(r_prev):
            r_prev = r_mean

        d0 = int(day_start[d])
        d1 = int(np.searchsorted(day_ord, d, side='right'))
        # The envelope is evaluated over this day's span plus one horizon,
        # not over the whole period. Building a 105,000-element array inside
        # a loop that runs once per day per envelope setting was costing more
        # than everything else in the harness combined.
        lo0 = max(0, d0 - m_max)
        hi0 = min(T, d1 + max_lead)
        env_at = envelope[clock.slot[lo0:hi0]]
        cum_env = np.concatenate([[0.0], np.cumsum(env_at)])

        def esum(a: int, b: int) -> float:
            return float(cum_env[b - lo0] - cum_env[a - lo0])

        for k in range(d0, min(d1, T - max_lead), stride):
            if k - m_max < 0 or k + max_lead > hi0:
                continue
            e_elapsed = esum(d0, k)
            f = e_elapsed / day_energy
            r_today = ((cum_solar[k] - cum_solar[d0]) / e_elapsed
                       if e_elapsed > 0 else np.nan)
            lb = []
            for m in m_list:
                lo = max(d0, k - m)
                e_lb = esum(lo, k)
                lb.append((cum_solar[k] - cum_solar[lo]) / e_lb
                          if e_lb > 0 else np.nan)
            same = day_ord[k:k + max_lead] == d
            env_w = env_at[k - lo0:k - lo0 + max_lead]
            # Every quantity below is a partial sum over window positions
            # 1..L-1 for each of four leads. Running the cumulative sum once
            # and indexing it turns 4 + 4*len(ys) reductions per instant into
            # 1 + len(ys) -- which is where the harness spends its time, since
            # this block runs once per instant per envelope setting.
            ce = np.cumsum(env_w)
            cs = np.cumsum(env_w * same)
            es = [float(cs[L - 1] - cs[0]) for L in LEADS]
            eo = [float((ce[L - 1] - ce[0]) - e) for L, e in zip(LEADS, es)]
            aa = [float(cum_solar[k + L] - cum_solar[k + 1]) for L in LEADS]
            dd = []
            cds = [None if w is None else np.cumsum(w * env_w)
                   for w in decay_full]
            for L in LEADS:
                dd.append([0.0 if cd is None else float(cd[L - 1] - cd[0])
                           for cd in cds])
            rows.append((r_prev, r_mean, r_today, f, lb, es, eo, aa, dd))

    if not rows:
        return None
    return SitePrecompute(
        r_prev=np.array([r[0] for r in rows]),
        r_mean=np.array([r[1] for r in rows]),
        r_today=np.array([r[2] for r in rows]),
        f=np.array([r[3] for r in rows]),
        r_lb=np.array([r[4] for r in rows]),
        e_same=np.array([r[5] for r in rows]),
        e_other=np.array([r[6] for r in rows]),
        a_act=np.array([r[7] for r in rows]),
        d_env=np.array([r[8] for r in rows]),
    )


def sweep_site(pc: SitePrecompute, prev_shares, today_maxes, f_ons, xs, ys):
    """Mean absolute cumulative energy error for every inner combination.

    The x and y loops are broadcast rather than iterated: for a fixed blend
    they only rescale terms that are already in hand, so all n_x * n_y of them
    are one array expression. That leaves the Python loop running over blends
    alone, which is roughly fifty times fewer iterations and is what makes the
    inner grid nearly free relative to the precompute.
    """
    n_leads = len(LEADS)
    n_x, n_y = len(xs), len(ys)
    r_today = np.where(np.isfinite(pc.r_today), pc.r_today, 0.0)
    has_today = np.isfinite(pc.r_today)

    e_same = pc.e_same[:, None, None, :]                  # (n,1,1,L)
    e_other = pc.e_other[:, None, None, :]
    a_act = pc.a_act[:, None, None, :]
    D = pc.d_env.transpose(0, 2, 1)[:, None, :, :]        # (n,1,y,L)

    keys, out = [], []
    for wps in prev_shares:
        r_hist = wps * pc.r_prev + (1.0 - wps) * pc.r_mean
        for wtm in today_maxes:
            for f_on in f_ons:
                a_w = np.where(has_today,
                               np.minimum(1.0, pc.f / f_on) * wtm, 0.0)
                r_blend = a_w * r_today + (1.0 - a_w) * r_hist
                r_lb = np.where(np.isfinite(pc.r_lb), pc.r_lb,
                                r_blend[:, None])          # (n,x)
                err = (r_blend[:, None, None, None] * (e_same - D)
                       + r_hist[:, None, None, None] * e_other
                       + r_lb[:, :, None, None] * D
                       - a_act)                            # (n,x,y,L)
                m = np.abs(err).mean(axis=0)               # (x,y,L)
                out.append(m.reshape(n_x * n_y, n_leads))
                keys.extend((wps, wtm, f_on, x, y)
                            for x in xs for y in ys)
    return keys, np.vstack(out)                            # (n_combos, L)


def edge_warnings(best, grids: dict) -> list[str]:
    """Flag an optimum sitting on a GRID boundary.

    A fit that lands on the edge has not found an optimum, it has run out of
    grid. Some bounds belong to the PARAMETER rather than the grid, and
    landing on one of those is a result:

      x at one interval                    cannot average over less
      today_max or y at 0                  the null wins
      prev_share, today_max at 0 or 1      they are weights in [0, 1]
      quantile at 1.0                      the maximum of the window

    Warning on those would send the reader looking for a wider grid that
    cannot exist.
    """
    out = []
    for name, vals in grids.items():
        v = float(best[name])
        if name == 'x_hours' and abs(v - 5 / 60) < 1e-9:
            continue
        if name in ('today_max', 'y_hours') and v == 0.0:
            continue
        if name in ('prev_share', 'today_max') and v in (0.0, 1.0):
            continue
        if name == 'quantile' and v == 1.0:
            continue
        if len(vals) > 1 and v in (min(vals), max(vals)):
            out.append(f"{name}={v:g} is at the "
                       f"{'low' if v == min(vals) else 'high'} end of "
                       f"{sorted(vals)}")
    return out


def report_edges(label: str, best, grids: dict) -> None:
    edges = edge_warnings(best, grids)
    if edges:
        for e in edges:
            print(f"  ** GRID EDGE ({label}): {e}")
    else:
        print(f"  No grid edges ({label}): every fitted value is interior.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--year', type=int, default=None)
    p.add_argument('--data-root', default='.')
    p.add_argument('--state', action='append', default=None)
    p.add_argument('--edp', nargs='+', default=None)
    p.add_argument('--sites', type=int, default=None)
    p.add_argument('--stride', type=int, default=12,
                   help='decision instants sampled every N intervals')
    p.add_argument('--horizon-hours', type=float, default=18.0)
    p.add_argument('--history-days', type=int, nargs='+',
                   default=list(DEFAULT_HISTORY))
    p.add_argument('--quantile', type=float, nargs='+',
                   default=list(DEFAULT_QUANTILE))
    p.add_argument('--prev-share', type=float, nargs='+',
                   default=list(DEFAULT_PREV_SHARE))
    p.add_argument('--today-max', type=float, nargs='+',
                   default=list(DEFAULT_TODAY_MAX))
    p.add_argument('--f-on', type=float, nargs='+', default=list(DEFAULT_F_ON))
    p.add_argument('--x', type=float, nargs='+', default=list(DEFAULT_X))
    p.add_argument('--y', type=float, nargs='+', default=list(DEFAULT_Y))
    p.add_argument('--out', default=None)
    args = p.parse_args()

    if args.year is not None:
        refs = resolve_sites(args.data_root, args.year, args.state, None)
        if args.sites and args.sites < len(refs):
            # Evenly spaced through the sorted list, not the first N. The list
            # is sorted by state, so a head sample is one state's worth of
            # sites and would fit parameters to that state's weather.
            idx = np.linspace(0, len(refs) - 1, args.sites).round().astype(int)
            refs = [refs[i] for i in sorted(set(idx.tolist()))]
        tag = str(args.year)
    elif args.edp:
        if not args.state or len(args.state) != 1:
            print("--edp requires exactly one --state.")
            return 1
        refs = explicit_sites(args.edp, args.state[0])[:args.sites]
        tag = 'adhoc'
    else:
        print("Pass --year (with --data-root) or --edp with --state.")
        return 1
    if not refs:
        print("No site files found.")
        return 1

    H = int(round(args.horizon_hours / (5 / 60)))
    outer = list(itertools.product(args.history_days, args.quantile))
    n_inner = (len(args.prev_share) * len(args.today_max) * len(args.f_on)
               * len(args.x) * len(args.y))

    print("fit_solar_forecast")
    print(f"  data root     {args.data_root}")
    print(f"  sites         {len(refs)}")
    print(f"  stride        {args.stride} intervals "
          f"({args.stride * 5 / 60:g} h between decision instants)")
    print(f"  envelope grid {len(outer)}  history {args.history_days}  "
          f"quantile {args.quantile}")
    print(f"  forecast grid {n_inner}  prev_share {args.prev_share}  "
          f"today_max {args.today_max}")
    print(f"                f_on {args.f_on}  x {args.x}  y {args.y}")
    print(f"  total         {len(outer) * n_inner:,} configurations, "
          f"one run\n")

    tot = np.zeros((len(outer), n_inner, len(LEADS)))
    cnt = np.zeros(len(outer))
    eq_sum = np.zeros((len(outer), n_inner))
    wins = np.zeros((len(outer), n_inner), dtype=int)
    eq_n = 0
    per_site = []
    keys_inner = None
    t0 = time.time()

    for si, ref in enumerate(refs, 1):
        try:
            series, valid = load_solar(ref.path)
        except Exception as exc:                             # noqa: BLE001
            print(f"  skip {Path(ref.path).name}: {exc}")
            continue
        clock = wall_clock(series.index, ref.state)
        solar = series.to_numpy(dtype=float)
        if len(solar) < SLOTS_PER_DAY * (max(args.history_days) + 7):
            print(f"  skip {ref.site_id}: only "
                  f"{len(solar) / SLOTS_PER_DAY:.0f} days")
            continue

        base = SolarForecastConfig(history_days=max(args.history_days))
        days, profs = day_profiles(solar, clock, base, valid)
        if len(days) < 8:
            print(f"  skip {ref.site_id}: {len(days)} usable days")
            continue

        # Held for the whole site so its score can be computed against ONE
        # reference. Normalising inside each envelope block, as an earlier
        # version did, made scores from different envelope settings
        # incomparable and let the per-site optimum disagree with the pooled
        # one on a single site.
        site_res = np.full((len(outer), n_inner, len(LEADS)), np.nan)
        for oi, (hd, q) in enumerate(outer):
            cfg = SolarForecastConfig(history_days=hd, envelope_quantile=q)
            pc = precompute(solar, clock, cfg, valid, days, profs,
                            args.x, args.y, args.stride, H)
            if pc is None:
                continue
            keys_inner, res = sweep_site(pc, args.prev_share, args.today_max,
                                         args.f_on, args.x, args.y)
            tot[oi] += res * len(pc.f)
            cnt[oi] += len(pc.f)
            site_res[oi] = res
            del pc
        if np.all(np.isnan(site_res)):
            continue

        # The site's own null -- no intraday term and no adjustment, i.e. the
        # history estimate alone -- is the reference, mirroring how the pooled
        # score is formed. Best over the whole grid, not within a block.
        null_mask = np.array([k[1] == 0.0 and k[4] == 0.0 for k in keys_inner])
        null_ref = np.nanmin(site_res[:, null_mask, :], axis=(0, 1))
        null_ref = np.where(null_ref > 0, null_ref, np.nan)
        site_score = np.nanmean(site_res / null_ref, axis=2)   # (outer, inner)
        flat = int(np.nanargmin(site_score))
        oi, ci = divmod(flat, n_inner)
        site_best = (outer[oi] + keys_inner[ci],
                     float(site_score[oi, ci]))

        # Equal-weight view: one vote per site, against the pooled score's one
        # vote per kWh of error. Accumulated rather than stored, so memory does
        # not grow with the cohort.
        eq_sum += np.nan_to_num(site_score, nan=np.nanmax(site_score))
        eq_n += 1
        wins[oi, ci] += 1
        per_site.append({
            'site_id': ref.site_id, 'state': ref.state,
            'history_days': site_best[0][0], 'quantile': site_best[0][1],
            'prev_share': site_best[0][2], 'today_max': site_best[0][3],
            'f_on': site_best[0][4], 'x_hours': site_best[0][5],
            'y_hours': site_best[0][6],
            'pv_infill_frac': float((~valid).mean()) if valid is not None else 0.0,
        })
        if si % 10 == 0 or si == len(refs):
            el = time.time() - t0
            print(f"  {si}/{len(refs)} sites   elapsed {el / 60:.1f} min   "
                  f"eta {el / si * (len(refs) - si) / 60:.1f} min")

    ok = cnt > 0
    if not ok.any():
        print("No usable sites.")
        return 1
    pooled_err = tot[ok] / cnt[ok][:, None, None]

    rows = []
    for oi, (hd, q) in enumerate([o for o, k in zip(outer, ok) if k]):
        for ci, key in enumerate(keys_inner):
            rows.append({'history_days': hd, 'quantile': q,
                         'prev_share': key[0], 'today_max': key[1],
                         'f_on': key[2], 'x_hours': key[3], 'y_hours': key[4],
                         **{f'cum_{lab}': pooled_err[oi, ci, li]
                            for li, lab in enumerate(LEAD_LABELS)}})
    grid = pd.DataFrame(rows)
    # The null is no adjustment AND no intraday term: the history estimate
    # alone. Everything is scored relative to it, per horizon, so no single
    # horizon dominates through its scale.
    null = grid[(grid.y_hours == 0) & (grid.today_max == 0)]
    ref_err = (null[[f'cum_{l}' for l in LEAD_LABELS]].min().to_numpy()
               if len(null) else
               grid[[f'cum_{l}' for l in LEAD_LABELS]].max().to_numpy())
    for li, lab in enumerate(LEAD_LABELS):
        grid[f'rel_{lab}'] = grid[f'cum_{lab}'] / ref_err[li]
    grid['score'] = grid[[f'rel_{l}' for l in LEAD_LABELS]].mean(axis=1)
    grid = grid.sort_values('score').reset_index(drop=True)
    ps = pd.DataFrame(per_site)

    grids = {'history_days': args.history_days, 'quantile': args.quantile,
             'prev_share': args.prev_share, 'today_max': args.today_max,
             'f_on': args.f_on, 'x_hours': args.x, 'y_hours': args.y}

    print("\nPooled fit, mean absolute cumulative energy error (kWh)")
    cols = (['history_days', 'quantile', 'prev_share', 'today_max', 'f_on',
             'x_hours', 'y_hours'] + [f'cum_{l}' for l in LEAD_LABELS]
            + ['score'])
    print(grid.head(10)[cols].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    best = grid.iloc[0]
    print(f"\n  null (history estimate alone)  score 1.0000")
    print(f"  pooled best  score {best.score:.4f} "
          f"({100 * (best.score - 1):+.1f}%)")
    for li, lab in enumerate(LEAD_LABELS):
        print(f"    {lab:>4}  {ref_err[li]:.4f} -> {best[f'cum_{lab}']:.4f} kWh "
              f"({100 * (best[f'cum_{lab}'] / ref_err[li] - 1):+.1f}%)")
    print("  NOTE: the pooled score is VOLUME WEIGHTED -- a site generating "
          "twice as much\n  contributes twice the absolute kWh error and "
          "pulls the optimum toward itself.")
    report_edges('pooled', best, grids)


    out = (Path(args.out) if args.out
           else Path(args.data_root) / 'data' / 'processed' / 'solar_fit')
    out.mkdir(parents=True, exist_ok=True)

    if eq_n:
        eq = eq_sum / eq_n
        idx = [(oi, ci) for oi in range(len(outer)) for ci in range(n_inner)]
        rec = []
        for oi, ci in idx:
            if not ok[oi]:
                continue
            hd, q = outer[oi]
            key = keys_inner[ci]
            rec.append({'history_days': hd, 'quantile': q,
                        'prev_share': key[0], 'today_max': key[1],
                        'f_on': key[2], 'x_hours': key[3], 'y_hours': key[4],
                        'mean_site_score': eq[oi, ci], 'wins': wins[oi, ci]})
        eqdf = pd.DataFrame(rec).sort_values(
            'mean_site_score').reset_index(drop=True)
        eb = eqdf.iloc[0]
        agree = all(abs(float(eb[c]) - float(best[c])) < 1e-9
                    for c in ('history_days', 'quantile', 'prev_share',
                              'today_max', 'f_on', 'x_hours', 'y_hours'))
        print(f"\n=== EQUAL-WEIGHT FIT: one vote per site over {eq_n} sites "
              f"===")
        print("  Use this one. The parameters serve VCR, which is scale-free, "
              "so a large\n  site should not get more say than a small one.")
        print(f"  best  history {eb['history_days']:g}, "
              f"quantile {eb['quantile']:g}, "
              f"prev_share {eb['prev_share']:g}, "
              f"today_max {eb['today_max']:g}, f_on {eb['f_on']:g}, "
              f"x {eb['x_hours']:g}, y {eb['y_hours']:g}  "
              f"(mean site score {eb['mean_site_score']:.4f})")
        report_edges('equal-weight', eb, grids)
        if agree:
            print("  Agrees with the pooled optimum.")
        else:
            print("  Differs from the pooled optimum, which is expected: "
                  "pooled is volume\n  weighted. Report both; use the "
                  "equal-weight values.")
        keycols = ('history_days', 'quantile', 'prev_share', 'today_max',
                   'f_on', 'x_hours', 'y_hours')
        mask = np.ones(len(eqdf), dtype=bool)
        for c in keycols:
            mask &= (eqdf[c].to_numpy() == best[c])
        pooled_row = eqdf[mask]
        if len(pooled_row):
            r = pooled_row.iloc[0]
            print(f"  At the pooled optimum: mean site score "
                  f"{r.mean_site_score:.4f}, best for {int(r.wins)} of {eq_n} "
                  f"sites")
        eqdf.to_csv(out / f'solar_fit_equal_weight_{tag}.csv', index=False)
        print("\n  SolarForecastConfig(")
        print(f"      history_days={int(eb['history_days'])}, "
              f"envelope_quantile={eb['quantile']:g},")
        print(f"      w_prev_share={eb['prev_share']:g}, "
              f"w_today_max={eb['today_max']:g}, f_on={eb['f_on']:g},")
        print(f"      lookback_hours={eb['x_hours']:.4g}, "
              f"decay_hours={eb['y_hours']:g})")

    if len(ps):
        print("\nPer-site optima")
        for c in ('history_days', 'quantile', 'prev_share', 'today_max',
                  'f_on', 'x_hours', 'y_hours'):
            print(f"  {c:<14} {ps[c].value_counts().sort_index().to_dict()}")

    grid.to_csv(out / f'solar_fit_grid_{tag}.csv', index=False)
    ps.to_csv(out / f'solar_fit_per_site_{tag}.csv', index=False)
    print(f"\nWrote {out / f'solar_fit_grid_{tag}.csv'}")
    print(f"      {out / f'solar_fit_per_site_{tag}.csv'}")
    print(f"Total wall time {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == '__main__':
    sys.exit(main())
