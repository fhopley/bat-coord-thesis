"""check_solar_forecast: mechanics verification for the solar forecast.

Run from the project root:

    python3 -m model.dev.check_solar_forecast --year 2022 --site S0008 \
        --data-root "/Volumes/Manual/Thesis Project"

Reports diagnostic context rather than a bare pass or fail.

Whether the index mode beats persistence, or either beats nothing, is an
empirical question answered by the fit. The checks here assert mechanism and
causality only, and print the comparison so the mechanism can be seen working.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model.core.scenario import SolarForecastConfig
from model.forecast.clock import SLOTS_PER_DAY, ClockArrays
from model.forecast.solar import SolarForecastProvider, build_day_tables
from model.dev.check_usage_forecast import wall_clock

_FAILURES: list[str] = []


def report(name: str, ok: bool, detail: str = '') -> None:
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}")
    for line in (detail or '').rstrip().splitlines():
        print(f"         {line}")
    if not ok:
        _FAILURES.append(name)


def load_solar(path: str):
    """Generation and its infill mask from a clean EDP parquet."""
    df = pd.read_parquet(path)
    if 'datetime_aest' in df.columns:
        df = df.set_index('datetime_aest')
    if df.index.tz is None:
        df.index = df.index.tz_localize('Etc/GMT-10')
    df = df.sort_index()
    col = next((c for c in ('pv_site_net_kwh', 'pv_site_net_wh')
                if c in df.columns), None)
    if col is None:
        raise ValueError(f"{path} has no pv_site_net column; has "
                         f"{list(df.columns)}")
    # Clipped at zero, NOT abs(): pv_site_net is negative overnight, when the
    # inverter draws from the house rather than generating. Taking magnitudes
    # turns that draw into generation, which puts a non-zero envelope on night
    # slots and then leaks a non-zero elapsed-energy fraction into the dawn
    # blend.
    solar = df[col].clip(lower=0).astype(float) * (
        1e-3 if col.endswith('_wh') else 1.0)
    valid = (df['pv_infill'].to_numpy() == 0) if 'pv_infill' in df else None
    return solar.rename('solar_kwh'), valid


# ---------------------------------------------------------------------------

def check_envelope(solar, clock, cfg, valid):
    print("\nEnvelope and index")
    tables = build_day_tables(solar, clock, cfg, valid)
    n_days = int(clock.day_ordinal[-1]) + 1
    first = min(tables) if tables else n_days
    report("tables exist contiguously from the first usable day",
           bool(tables) and all(d in tables for d in range(first, n_days - 1)),
           f"{len(tables)} tables over {n_days} local days, from day {first}")

    bad = [d for d, t in tables.items() if (t.envelope < 0).any()]
    report("no negative envelope values", not bad, f"first: {bad[:5]}" if bad else '')

    t = tables[max(tables)]
    lit = int((t.envelope > 0).sum())
    peak = int(np.argmax(t.envelope))
    report("the envelope is a daylight-shaped curve",
           0 < lit < SLOTS_PER_DAY and 96 <= peak <= 192,
           f"{lit} of {SLOTS_PER_DAY} slots non-zero, peak at slot {peak} "
           f"({peak / 12:.2f} h local), day energy {t.day_energy:.2f} kWh")

    # The envelope is a high quantile, so a few per cent of DAYLIGHT
    # observations must exceed it. Measured across every day with tables, not
    # one: a single day can sit entirely under the envelope simply by being
    # cloudy, which would make this a test of the weather.
    num = den = 0
    for d, tt in tables.items():
        s0 = int(np.searchsorted(clock.day_ordinal, d, 'left'))
        e0 = int(np.searchsorted(clock.day_ordinal, d, 'right'))
        env_of = tt.envelope[clock.slot[s0:e0]]
        lit = env_of > 0
        num += int((solar[s0:e0][lit] > env_of[lit] + 1e-12).sum())
        den += int(lit.sum())
    over = num / den if den else 0.0
    report("some daylight observations exceed the envelope, as a quantile "
           "implies",
           0.005 < over < 0.35,
           f"{100 * over:.2f}% of {den:,} daylight intervals above the "
           f"envelope (quantile {cfg.envelope_quantile}; a perfect quantile "
           f"on stationary data would give {100 * (1 - cfg.envelope_quantile):.0f}%)")

    idx = [t.r_mean for t in tables.values() if np.isfinite(t.r_mean)]
    report("the trailing index is a sane fraction",
           bool(idx) and 0.05 < float(np.mean(idx)) < 1.0,
           f"r_mean over {len(idx)} days: mean {np.mean(idx):.3f}, "
           f"range {np.min(idx):.3f}-{np.max(idx):.3f}")


def tables_of(solar, clock, cfg, valid):
    return build_day_tables(solar, clock, cfg, valid)


def check_blend(solar, clock, cfg, valid, H):
    print("\nIndex blend")
    pr = SolarForecastProvider(solar, clock, cfg, H, valid=valid)
    T = len(solar)
    rows = []
    for k in range(SLOTS_PER_DAY * (cfg.history_days + 1), T - H, 13):
        sl = pr.get(k, H)
        if sl is None:
            continue
        day0 = int(np.searchsorted(clock.day_ordinal,
                                   clock.day_ordinal[k], 'left'))
        rows.append((k, sl.intraday_weight, sl.index_hist, sl.index_today,
                     int(clock.slot[k])))
    r = np.array(rows)
    # "Before dawn" is defined by the envelope in force on that instant's own
    # day, not by the clock and not by one day's envelope: dawn moves through
    # the year, so a summer morning genuinely has envelope energy elapsed at
    # an hour that is night in winter.
    tables = tables_of(solar, clock, cfg, valid)
    first_lit = {}
    for d, tt in tables.items():
        lit = np.flatnonzero(tt.envelope > 0)
        first_lit[d] = int(lit[0]) if lit.size else SLOTS_PER_DAY
    pre = np.array([row for row in rows
                    if row[4] < first_lit.get(
                        int(clock.day_ordinal[int(row[0])]), 0)])
    lo, hi = min(first_lit.values()), max(first_lit.values())
    report("the intraday weight is zero before the envelope lights up",
           len(pre) > 0 and float(pre[:, 1].max()) == 0.0,
           f"max weight over {len(pre)} pre-dawn instants: "
           f"{pre[:, 1].max():.2e}; first lit slot runs {lo} to {hi} "
           f"({lo / 12:.2f}-{hi / 12:.2f} h local) across the year")
    report("the intraday weight never exceeds its ceiling",
           float(r[:, 1].max()) <= cfg.w_today_max + 1e-12,
           f"max {r[:, 1].max():.4f}, ceiling {cfg.w_today_max}")
    mid = r[(r[:, 4] > 150) & (r[:, 4] < 200)]
    report("the weight reaches its ceiling by mid-afternoon",
           float(mid[:, 1].min()) >= cfg.w_today_max - 1e-9,
           f"min weight over slots 12:30-16:40: {mid[:, 1].min():.4f}")
    zero_w = r[r[:, 1] == 0.0]
    report("wherever the weight is zero the forecast is the history estimate",
           len(zero_w) > 0 and bool(np.allclose(zero_w[:, 2], zero_w[:, 3])),
           f"index_today == index_hist at all {len(zero_w)} such instants; "
           f"max |difference| "
           f"{np.abs(zero_w[:, 2] - zero_w[:, 3]).max() if len(zero_w) else 0:.2e}")


def check_causality(solar, clock, cfg, valid, H):
    print("\nCausality")
    T = len(solar)
    pr = SolarForecastProvider(solar, clock, cfg, H, valid=valid)
    # Each probe rebuilds the provider, which is seconds per build, so this
    # samples rather than sweeps. Twenty instants spread across the period is
    # enough to catch a leak: a provider that reads ahead does so at every
    # instant, not at a chosen few.
    ks = [k for k in range(SLOTS_PER_DAY * 2, T - H, max(1, (T - H) // 20))]
    bad = []
    for k in ks:
        ref = pr.get(k, H)
        if ref is None:
            continue
        poisoned = solar.copy()
        poisoned[k:] = 50.0
        got = SolarForecastProvider(poisoned, clock, cfg, H,
                                    valid=valid).get(k, H)
        if got is None or not np.array_equal(ref.solar_kwh, got.solar_kwh):
            bad.append(k)
        if len(bad) >= 3:
            break
    report("no forecast depends on generation at or after the decision instant",
           not bad,
           f"{len(ks)} decision instants re-run against a trace overwritten "
           f"from k onward; all identical" if not bad
           else f"first offenders at intervals {bad}")

    # The complementary direction: the interval ending AT the decision instant
    # is observed, so poisoning k-1 must move the forecast. Without this, an
    # implementation reading only <= k-2 would pass the check above.
    if cfg.decay_hours > 0:
        moved = 0
        for k in ks:
            if not (72 < int(clock.slot[k]) < 216):
                continue                       # needs daylight to register
            ref = pr.get(k, H)
            poisoned = solar.copy()
            poisoned[k - 1] += 2.0
            got = SolarForecastProvider(poisoned, clock, cfg, H,
                                        valid=valid).get(k, H)
            if ref is not None and got is not None and not np.array_equal(
                    ref.solar_kwh, got.solar_kwh):
                moved += 1
            if moved:
                break
        report("the interval ending at the decision instant IS used",
               moved > 0, "perturbing k-1 changes the forecast")


def check_modes(solar, clock, cfg, valid, H):
    print("\nModes")
    T = len(solar)
    idx = SolarForecastProvider(solar, clock, cfg, H, valid=valid)
    per = SolarForecastProvider(
        solar, clock, SolarForecastConfig(**{**cfg.__dict__,
                                             'mode': 'persistence'}), H)
    ei, ep, n = [], [], 0
    vp = (np.concatenate([[0], np.cumsum(~valid)]) if valid is not None
          else np.zeros(T + 1, dtype=int))
    for k in range(SLOTS_PER_DAY * (cfg.history_days + 1), T - H, 29):
        if vp[k + H] - vp[k + 1] > 0:
            continue
        a, b = idx.get(k, H), per.get(k, H)
        if a is None or b is None:
            continue
        act = solar[k + 1:k + H]
        ei.append(np.abs(a.solar_kwh[1:] - act).sum())
        ep.append(np.abs(b.solar_kwh[1:] - act).sum())
        n += 1
    report("both modes produce finite non-negative forecasts", n > 0,
           f"{n} windows compared on non-infilled targets")
    print(f"         mean |error| over the horizon: persistence "
          f"{np.mean(ep):.3f} kWh, index {np.mean(ei):.3f} kWh "
          f"({100 * (np.mean(ei) / np.mean(ep) - 1):+.1f}%)")
    print("         NOTE: both are naive bounds. Which is tighter is measured "
          "here, not asserted.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--year', type=int, default=None)
    p.add_argument('--data-root', default='.')
    p.add_argument('--edp', help='explicit clean EDP parquet')
    p.add_argument('--site')
    p.add_argument('--state', default=None)
    p.add_argument('--horizon-hours', type=float, default=18.0)
    args = p.parse_args()

    if args.year is not None:
        from model.dev.fit_usage_forecast import resolve_sites
        refs = resolve_sites(args.data_root, args.year, None, None)
        if args.site:
            refs = [r for r in refs if r.site_id == args.site]
        if not refs:
            print(f"site {args.site!r} not found for {args.year}")
            return 1
        ref = refs[0]
        path, state, sid = ref.path, ref.state, ref.site_id
    elif args.edp:
        if not args.state:
            print("--edp requires --state")
            return 1
        path, state, sid = args.edp, args.state, args.site or Path(args.edp).stem
    else:
        print("Pass --year (with --data-root) or --edp with --state.")
        return 1

    series, valid = load_solar(path)
    clock = wall_clock(series.index, state)
    solar = series.to_numpy(dtype=float)
    cfg = SolarForecastConfig()
    H = int(round(args.horizon_hours / (5 / 60)))

    print("check_solar_forecast")
    print(f"  source        {Path(path).name} site {sid} ({state})")
    print(f"  intervals     {len(solar):,}  mean daily "
          f"{solar.sum() / (len(solar) / SLOTS_PER_DAY):.2f} kWh")
    if valid is not None:
        print(f"  pv infill     {100 * (~valid).mean():.1f}% "
              f"(excluded from the envelope history)")
    print(f"  config        mode {cfg.mode}, history {cfg.history_days}d, "
          f"quantile {cfg.envelope_quantile}, w_prev_share "
          f"{cfg.w_prev_share}, w_today_max {cfg.w_today_max}, f_on {cfg.f_on}")

    check_envelope(solar, clock, cfg, valid)
    check_blend(solar, clock, cfg, valid, H)
    check_causality(solar, clock, cfg, valid, H)
    check_modes(solar, clock, cfg, valid, H)

    print()
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} check(s): {', '.join(_FAILURES)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
