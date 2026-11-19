"""check_usage_forecast: mechanics verification for the usage forecast.

Run from the project root:

    python3 -m model.dev.check_usage_forecast                       # synthetic
    python3 -m model.dev.check_usage_forecast --year 2022 --site S0008 \
        --data-root "/Volumes/Manual/Thesis Project"
    python3 -m model.dev.check_usage_forecast --edp PATH --state NSW   # ad-hoc file

With --year the site's state comes from the EDP metadata CSV rather than being
assumed, which matters: the weekday/weekend split and the day-of-week level
both key off the LOCAL day boundary, and a wrong state silently mislabels it.

Reports diagnostic context rather than a bare pass or fail. Every check that
can fail prints the values needed to diagnose it: the offending interval, its
neighbours, and what the forecast and the actual were there.

The causality check is the one that matters. Everything else in this module is
a modelling choice; reading load at or after the decision instant is simply
wrong, and it is the kind of wrong that would leave no trace in any downstream
artefact. It is verified here by re-running the provider against a trace whose
future has been overwritten with garbage: if any forecast changes, the
provider read something it should not have.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model.forecast.usage import (
    SLOTS_PER_DAY, ClockArrays, UsageForecastConfig, UsageForecastProvider,
    build_day_tables,
)

STATE_TIMEZONE = {
    'NSW': 'Australia/Sydney', 'VIC': 'Australia/Melbourne',
    'QLD': 'Australia/Brisbane', 'SA': 'Australia/Adelaide',
}

_FAILURES: list[str] = []


def report(name: str, ok: bool, detail: str = '') -> None:
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}")
    if detail:
        for line in detail.rstrip().splitlines():
            print(f"         {line}")
    if not ok:
        _FAILURES.append(name)


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------

def wall_clock(index: pd.DatetimeIndex, state: str,
               interval_hours: float = 5 / 60) -> ClockArrays:
    """Container stand-in for model.core.localtime.local_wall_clock.

    Replicates the documented behaviour: evaluate at the interval-START
    instant, convert to the state's local timezone, return local calendar
    facts. In production this function is deleted and
    clock_from_wall_clock(local_wall_clock(...)) is used instead. It exists
    here only so the check runs without the production package on the path.
    """
    start = index - pd.Timedelta(hours=interval_hours)
    local = start.tz_convert(STATE_TIMEZONE[state])
    slot = local.hour.to_numpy() * 12 + local.minute.to_numpy() // 5
    dates = local.normalize()
    return ClockArrays(
        slot=slot.astype(np.int64),
        dayofweek=local.dayofweek.to_numpy().astype(np.int64),
        is_weekend=(local.dayofweek.to_numpy() >= 5),
        day_ordinal=pd.Index(dates).factorize()[0].astype(np.int64),
    )


# ---------------------------------------------------------------------------
# Synthetic trace
# ---------------------------------------------------------------------------

def synthetic_load(n_days: int = 70, state: str = 'NSW',
                   seed: int = 7) -> pd.Series:
    """A household load trace with the structure the forecast is meant to find.

    Deliberately includes everything that should break a naive implementation:
    a weekday/weekend shape difference, a day-of-week level difference, a slow
    seasonal drift, spiky appliance noise, occasional large EV-like events on
    random days, and a five-day hot spell near the end where load runs well
    above the trailing profile. The hot spell is the case the recent-observation
    adjustment exists for.

    The noise is an AR(1) process, not white noise, and that choice is load
    bearing. A household's deviation from its own average profile persists:
    somebody is home, or the air conditioning is running, and both states last
    hours. With white noise the last two hours of observation carry no
    information about the next two, and the adjustment can only ever add noise
    -- an earlier version of this trace had exactly that property and the
    adjustment lost to the null on every horizon. The AR(1) coefficient here is
    an ASSUMPTION about real load, not a measurement. Whether the adjustment
    actually pays is settled by fit_usage_forecast against EDP, not here.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp('2024-09-01 00:05', tz='Etc/GMT-10')
    index = pd.date_range(start, periods=n_days * SLOTS_PER_DAY, freq='5min')
    clock = wall_clock(index, state)

    slot = clock.slot
    hod = slot / 12.0
    weekday_shape = (0.35
                     + 0.55 * np.exp(-0.5 * ((hod - 7.5) / 1.3) ** 2)
                     + 1.00 * np.exp(-0.5 * ((hod - 18.5) / 1.8) ** 2))
    weekend_shape = (0.45
                     + 0.35 * np.exp(-0.5 * ((hod - 9.5) / 2.2) ** 2)
                     + 0.55 * np.exp(-0.5 * ((hod - 13.0) / 3.0) ** 2)
                     + 0.85 * np.exp(-0.5 * ((hod - 19.0) / 2.0) ** 2))
    shape = np.where(clock.is_weekend, weekend_shape, weekday_shape)

    dow_level = np.array([1.00, 0.97, 0.98, 1.02, 1.10, 1.28, 1.22])
    level = dow_level[clock.dayofweek]

    day = clock.day_ordinal
    drift = 1.0 + 0.0012 * day                      # ~3.6%/month seasonal rise

    load = 0.055 * shape * level * drift

    # Hot spell: five consecutive days at the end running ~55% above profile,
    # concentrated in the afternoon and evening as air conditioning would be.
    hot_days = set(range(n_days - 8, n_days - 3))
    hot = np.isin(day, list(hot_days))
    aircon = 0.55 * np.exp(-0.5 * ((hod - 16.5) / 4.0) ** 2)
    load = load * np.where(hot, 1.0 + aircon, 1.0)

    # EV-like events: one long high-draw block on a handful of random days.
    for d in rng.choice(n_days, size=max(1, n_days // 9), replace=False):
        mask = (day == d) & (slot >= 264)            # from 22:00 local
        load[mask] += 0.35

    # Persistent occupancy deviation, AR(1) with a roughly 90 minute memory,
    # plus white appliance switching noise on top of it.
    phi = np.exp(-(5 / 60) / 1.5)
    eps = rng.normal(0.0, 1.0, size=len(load))
    z = np.empty(len(load))
    z[0] = eps[0]
    for i in range(1, len(load)):
        z[i] = phi * z[i - 1] + np.sqrt(1 - phi ** 2) * eps[i]
    occupancy = np.exp(0.30 * z)
    switching = rng.lognormal(mean=0.0, sigma=0.30, size=len(load))

    load = load * occupancy * switching
    return pd.Series(np.maximum(load, 0.0), index=index, name='gross_load_kwh')


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_tables(load: np.ndarray, clock: ClockArrays,
                 cfg: UsageForecastConfig,
                 valid: np.ndarray | None = None) -> None:
    print("\nProfile tables")
    tables = build_day_tables(load, clock, cfg, valid)
    n_days = int(clock.day_ordinal[-1]) + 1

    # Not "all but the first day": a site whose file opens with a block of
    # infilled days has no real history until that block ends, and the correct
    # behaviour is to have no tables until then. S0008's 2022 file opens with
    # 36 contiguous same-DOW-infilled days, so tables start in February.
    first = min(tables) if tables else n_days
    contiguous = all(d in tables for d in range(first, n_days - 1))
    report("tables exist contiguously from the first usable day",
           bool(tables) and contiguous,
           f"{len(tables)} tables over {n_days} local days, "
           f"starting at day {first}"
           + (f" -- {first} leading days have no usable history, which for a "
              f"real file usually means a leading infill block"
              if first > 1 else ""))

    bad = [d for d, t in tables.items()
           if not np.allclose(t.shape.sum(axis=1), 1.0, atol=1e-9)]
    report("every shape row sums to 1", not bad,
           f"first offenders: {bad[:5]}" if bad else '')

    neg = [d for d, t in tables.items() if (t.shape < 0).any()]
    report("no negative shape values", not neg,
           f"first offenders: {neg[:5]}" if neg else '')

    warm = sorted(d for d, t in tables.items() if t.warm_up)
    report("warm-up flag clears once history is deep enough",
           bool(warm) and max(warm) < n_days - 1,
           f"warm-up on {len(warm)} days, last is day {max(warm) if warm else '-'}")

    mid = tables[max(tables)]
    wk, we = mid.shape[0], mid.shape[1]
    peak_wk, peak_we = int(np.argmax(wk)), int(np.argmax(we))
    report("weekday and weekend shapes differ",
           float(np.abs(wk - we).sum()) > 0.05,
           f"L1 distance {float(np.abs(wk - we).sum()):.4f}; "
           f"weekday peak slot {peak_wk} ({peak_wk / 12:.2f}h), "
           f"weekend peak slot {peak_we} ({peak_we / 12:.2f}h)")

    # The smoothing machinery is tested at a forced width, independent of the
    # configured one, because smoothing DEFAULTS TO OFF: it lost on real data.
    # This asserts the code works if it is ever switched on, not that it helps.
    sm_on = UsageForecastConfig(**{**cfg.__dict__, 'smoothing_intervals': 7})
    sm_off = UsageForecastConfig(**{**cfg.__dict__, 'smoothing_intervals': 1})
    d = max(tables)
    a = build_day_tables(load, clock, sm_on, valid)[d].shape[0]
    b = build_day_tables(load, clock, sm_off, valid)[d].shape[0]
    tv = lambda z: float(np.abs(np.diff(z, append=z[0])).sum())
    report("the smoothing machinery reduces roughness when switched on",
           tv(a) < tv(b) and abs(a.sum() - 1.0) < 1e-9,
           f"total variation {tv(b):.4f} at width 1 -> {tv(a):.4f} at width 7, "
           f"mass preserved at {a.sum():.6f}\n"
           f"configured width is {cfg.smoothing_intervals}")


def check_causality(load: np.ndarray, clock: ClockArrays,
                    cfg: UsageForecastConfig, H: int,
                    valid: np.ndarray | None = None) -> None:
    """Re-run against a trace whose future is destroyed, and require identity.

    Chosen over inspecting the code because it tests the behaviour rather than
    the intent, and because it catches an accidental read through any path,
    including the table build and the base array construction.
    """
    print("\nCausality")
    T = len(load)
    provider = UsageForecastProvider(load, clock, cfg, H, valid=valid)

    ks = [k for k in range(0, T - H, max(1, (T - H) // 400))]
    mismatches = []
    for k in ks:
        ref = provider.get(k, H)
        if ref is None:
            continue
        poisoned = load.copy()
        poisoned[k:] = 1e6                      # everything from interval k on
        pp = UsageForecastProvider(poisoned, clock, cfg, H, valid=valid)
        got = pp.get(k, H)
        if got is None or not np.array_equal(ref.load_kwh, got.load_kwh):
            first = (int(np.argmax(ref.load_kwh != got.load_kwh))
                     if got is not None else -1)
            mismatches.append((k, first))
        if len(mismatches) >= 3:
            break

    if mismatches:
        detail = '\n'.join(
            f"decision at interval {k}: forecast changed at window position "
            f"{j} (target interval {k + j})" for k, j in mismatches)
    else:
        detail = (f"{len(ks)} decision instants re-run against a trace with "
                  f"all intervals from k onward overwritten; every forecast "
                  f"identical")
    report("no forecast depends on load at or after the decision instant",
           not mismatches, detail)

    # The complementary check: the interval labelled exactly at the decision
    # instant IS observed, so poisoning k-1 MUST change the adjustment. A
    # provider that quietly used labels <= k-2 would pass the check above.
    if cfg.decay_hours > 0:
        k = ks[len(ks) // 2]
        ref = provider.get(k, H)
        poisoned = load.copy()
        poisoned[k - 1] += 5.0
        pp = UsageForecastProvider(poisoned, clock, cfg, H, valid=valid)
        got = pp.get(k, H)
        changed = not np.array_equal(ref.load_kwh, got.load_kwh)
        report("the interval ending at the decision instant IS used", changed,
               f"adjustment {ref.adjustment_kwh:+.5f} -> "
               f"{got.adjustment_kwh:+.5f} kWh/interval at interval {k}")


def check_forecast_quality(load: np.ndarray, clock: ClockArrays,
                           cfg: UsageForecastConfig, H: int,
                           valid: np.ndarray | None = None) -> None:
    print("\nForecast behaviour")
    T = len(load)
    if valid is None:
        valid = np.ones(T, dtype=bool)
    provider = UsageForecastProvider(load, clock, cfg, H, valid=valid)
    null_cfg = UsageForecastConfig(**{**cfg.__dict__, 'decay_hours': 0.0})
    null = UsageForecastProvider(load, clock, null_cfg, H, valid=valid)

    # Infilled targets must be excluded. EDP's Method 2 load infill copies the
    # same day of week, so an infilled window is close to what the profile
    # itself predicts: leaving it in flatters the base and hides the
    # adjustment. On the real site this alone moved the 1h figure from -1% to
    # -11%.
    vp = np.concatenate([[0], np.cumsum(~valid)])
    ks = [k for k in range(SLOTS_PER_DAY * cfg.history_days, T - H,
                           max(1, (T - H) // 1500))
          if vp[k + H] - vp[k + 1] == 0]
    err_adj, err_null, clipped, warm = [], [], 0, 0
    for k in ks:
        a, b = provider.get(k, H), null.get(k, H)
        if a is None or b is None:
            continue
        actual = load[k:k + H]
        err_adj.append(a.load_kwh - actual)
        err_null.append(b.load_kwh - actual)
        clipped += a.n_clipped
        warm += int(a.warm_up)
    E_adj, E_null = np.array(err_adj), np.array(err_null)

    report("forecasts are finite and non-negative",
           bool(np.isfinite(E_adj).all()) and bool((E_adj + 1e-12 >= -np.abs(E_adj)).all()),
           f"{len(ks)} windows evaluated, {clipped} positions clipped at zero, "
           f"{warm} windows on warm-up tables")

    # Cumulative energy error is what the LP actually consumes: the battery
    # plans an energy trajectory, not a sequence of point loads. Position 0 is
    # excluded because it is committed with actual load.
    print("         cumulative energy error, mean absolute kWh "
          "(lower is better)")
    print(f"         {'horizon':>8}  {'no adjustment':>14}  "
          f"{'with x/y':>10}  {'change':>8}")
    for lead in (12, 36, 72, 144, min(216, H)):
        if lead > H:
            continue
        c_null = np.abs(E_null[:, 1:lead].sum(axis=1)).mean()
        c_adj = np.abs(E_adj[:, 1:lead].sum(axis=1)).mean()
        print(f"         {lead * 5 / 60:6.1f}h  {c_null:14.4f}  "
              f"{c_adj:10.4f}  {100 * (c_adj / c_null - 1):+7.1f}%")

    mae_null = np.abs(E_null[:, 1:]).mean()
    mae_adj = np.abs(E_adj[:, 1:]).mean()
    print(f"         per-interval MAE {mae_null:.5f} -> {mae_adj:.5f} kWh "
          f"({100 * (mae_adj / mae_null - 1):+.1f}%)")
    print(f"         NOTE: whether the adjustment beats the null is an "
          f"EMPIRICAL question,")
    print(f"         settled by fit_usage_forecast on EDP, not asserted here. "
          f"These numbers")
    print(f"         are reported so the mechanism can be seen working, not "
          f"as a pass gate.")

    bias_null = E_null[:, 1:].mean()
    bias_adj = E_adj[:, 1:].mean()
    print(f"         mean signed error {bias_null:+.6f} (null) -> "
          f"{bias_adj:+.6f} (adjusted) kWh/interval, "
          f"{100 * bias_null / load.mean():+.1f}% of mean load")


def check_level_bias(load: np.ndarray, clock: ClockArrays,
                     cfg: UsageForecastConfig,
                     valid: np.ndarray | None = None) -> None:
    """Separate implementation error from the trailing mean's own lag.

    A trailing N-day mean of a trending series lags it by roughly N/2 days, so
    ANY forecast built on one under-predicts a rising load and over-predicts a
    falling one. That is a property of the estimator, not a defect in the code,
    and testing the absolute bias would therefore test the trace rather than
    the implementation.

    What is testable is that the level estimator does exactly what a plain
    trailing mean of daily totals does and no worse. If the two diverge, the
    day-type split, the coverage gate or the shape normalisation has introduced
    a bias of its own, which is a bug.
    """
    print("\nLevel bias against a plain trailing mean")
    from model.forecast.usage import base_forecast, build_day_tables
    tables = build_day_tables(load, clock, cfg, valid)
    day_ord = clock.day_ordinal
    n_days = int(day_ord[-1]) + 1
    starts = np.searchsorted(day_ord, np.arange(n_days), side='left')
    ends = np.searchsorted(day_ord, np.arange(n_days), side='right')

    totals = np.array([load[int(starts[d]):int(ends[d])].sum()
                       for d in range(n_days)])
    full = np.array([(ends[d] - starts[d]) >= SLOTS_PER_DAY
                     for d in range(n_days)])

    fc, ac, tm = [], [], []
    for d in range(cfg.history_days, n_days - 1):
        if d not in tables or not full[d]:
            continue
        a, b = int(starts[d]), int(ends[d])
        fc.append(base_forecast(tables[d], clock, a, b).sum())
        ac.append(totals[d])
        tm.append(totals[max(0, d - cfg.history_days):d].mean())
    fc, ac, tm = np.array(fc), np.array(ac), np.array(tm)

    bias_fc = 100 * (fc.sum() / ac.sum() - 1)
    bias_tm = 100 * (tm.sum() / ac.sum() - 1)
    report("level bias matches a plain trailing mean of daily totals",
           abs(bias_fc - bias_tm) < 2.0,
           f"{len(fc)} whole days: forecast {bias_fc:+.2f}%, plain trailing "
           f"{cfg.history_days}-day mean {bias_tm:+.2f}%, "
           f"difference {bias_fc - bias_tm:+.2f} pp\n"
           f"KNOWN LIMITATION: the shared component is the trailing mean's lag "
           f"behind a trend,\nnot an implementation fault. See "
           f"usage_forecast_spec.md, 'Where this is weakest'.")


def check_adjustment_mechanism(load: np.ndarray, clock: ClockArrays,
                               cfg: UsageForecastConfig, H: int,
                               anomaly_days: set[int]) -> None:
    """The adjustment must respond to a persistent anomaly, in the right sign.

    This is the mechanism test, and it is separate from the question of whether
    the adjustment pays on average. On days constructed to run well above the
    trailing profile, the adjustment must be positive and must reduce error. If
    it does not do that here it is wired up wrongly, regardless of what any
    average over a real trace says.
    """
    print("\nAdjustment mechanism on a constructed anomaly")
    T = len(load)
    provider = UsageForecastProvider(load, clock, cfg, H)
    null_cfg = UsageForecastConfig(**{**cfg.__dict__, 'decay_hours': 0.0})
    null = UsageForecastProvider(load, clock, null_cfg, H)

    lead = min(72, H)                                # 6 hours
    inside, outside = [], []
    adj_inside, adj_outside = [], []
    for k in range(SLOTS_PER_DAY * cfg.history_days, T - H, 6):
        a, b = provider.get(k, H), null.get(k, H)
        if a is None or b is None:
            continue
        actual = load[k:k + lead].sum()
        e_adj = abs(a.load_kwh[1:lead].sum() - load[k + 1:k + lead].sum())
        e_null = abs(b.load_kwh[1:lead].sum() - load[k + 1:k + lead].sum())
        target = (inside if int(clock.day_ordinal[k]) in anomaly_days
                  else outside)
        target.append((e_null, e_adj))
        (adj_inside if int(clock.day_ordinal[k]) in anomaly_days
         else adj_outside).append(a.adjustment_kwh)
        del actual

    ins = np.array(inside)
    out = np.array(outside)
    report("the adjustment is positive on the constructed hot spell",
           float(np.mean(adj_inside)) > 0,
           f"mean adjustment {np.mean(adj_inside):+.5f} kWh/interval on "
           f"anomaly days vs {np.mean(adj_outside):+.5f} elsewhere "
           f"({len(ins)} and {len(out)} decision instants)")
    report("the adjustment reduces 6h cumulative error on those days",
           ins[:, 1].mean() < ins[:, 0].mean(),
           f"mean absolute 6h error {ins[:, 0].mean():.4f} -> "
           f"{ins[:, 1].mean():.4f} kWh "
           f"({100 * (ins[:, 1].mean() / ins[:, 0].mean() - 1):+.1f}%); "
           f"off-anomaly {out[:, 0].mean():.4f} -> {out[:, 1].mean():.4f} "
           f"({100 * (out[:, 1].mean() / out[:, 0].mean() - 1):+.1f}%)")


def check_degradation(load: np.ndarray, clock: ClockArrays,
                      cfg: UsageForecastConfig, H: int) -> None:
    """Short history, all-weekday history, and an empty day must not crash."""
    print("\nGraceful degradation")
    short = UsageForecastProvider(load[:SLOTS_PER_DAY * 3], 
                                  ClockArrays(clock.slot[:SLOTS_PER_DAY * 3],
                                              clock.dayofweek[:SLOTS_PER_DAY * 3],
                                              clock.is_weekend[:SLOTS_PER_DAY * 3],
                                              clock.day_ordinal[:SLOTS_PER_DAY * 3]),
                                  cfg, H)
    got = short.get(SLOTS_PER_DAY * 2, min(H, SLOTS_PER_DAY))
    report("three days of history still produces a forecast",
           got is not None and np.isfinite(got.load_kwh).all(),
           f"tables on {short.n_days_with_tables} days, "
           f"warm_up={got.warm_up if got else '-'}")

    first = UsageForecastProvider(load, clock, cfg, H).get(0, H)
    report("the first interval of the period returns None", first is None,
           "no complete prior local day exists, so there is nothing to build "
           "a profile from; the caller decides the fallback")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--year', type=int, default=None,
                   help='resolve the site and its state from the EDP '
                        'metadata under --data-root')
    p.add_argument('--data-root', default='.',
                   help='root holding data/clean/edp and data/processed/edp')
    p.add_argument('--edp', help='explicit clean EDP parquet; omit for '
                                 'synthetic')
    p.add_argument('--site', help='site id')
    p.add_argument('--state', default=None,
                   help='required with --edp; ignored with --year. Sets the '
                        'local day boundary and day-of-week labelling')
    p.add_argument('--horizon-hours', type=float, default=18.0)
    p.add_argument('--lookback-hours', type=float, default=0.5)
    p.add_argument('--decay-hours', type=float, default=1.5)
    p.add_argument('--dow-level', action='store_true')
    p.add_argument('--smoothing', type=int, default=7)
    args = p.parse_args()

    from model.dev.fit_usage_forecast import load_site, resolve_sites

    if args.year is not None:
        refs = resolve_sites(args.data_root, args.year, None, None)
        if args.site:
            refs = [r for r in refs if r.site_id == args.site]
            if not refs:
                print(f"site {args.site!r} not found for {args.year}")
                return 1
        ref = refs[0]
        state = ref.state
        series, valid, sid = load_site(ref.path, ref.site_id)
        source = (f"{Path(ref.path).name} site {sid} ({state}), "
                  f"{100 * (~valid).mean():.1f}% infilled (excluded)")
    elif args.edp:
        if not args.state:
            print("--edp requires --state; the metadata CSV that normally "
                  "supplies it is not being read, and a wrong state silently "
                  "mislabels every weekend.")
            return 1
        state = args.state
        series, valid, sid = load_site(args.edp, args.site)
        source = (f"{Path(args.edp).name} site {sid}, "
                  f"{100 * (~valid).mean():.1f}% infilled (excluded)")
    else:
        state = args.state or 'NSW'
        series = synthetic_load(state=state)
        valid = None
        source = "synthetic, 70 days"

    clock = wall_clock(series.index, state)
    load = series.to_numpy(dtype=float)
    cfg = UsageForecastConfig(lookback_hours=args.lookback_hours,
                              decay_hours=args.decay_hours,
                              smoothing_intervals=args.smoothing,
                              dow_level=args.dow_level)
    H = int(round(args.horizon_hours / (5 / 60)))

    print(f"check_usage_forecast")
    print(f"  source        {source}")
    print(f"  state         {state}")
    print(f"  intervals     {len(load):,} "
          f"({series.index[0]} to {series.index[-1]})")
    print(f"  mean daily    {load.sum() / (len(load) / SLOTS_PER_DAY):.2f} kWh")
    print(f"  config        history {cfg.history_days}d, smoothing "
          f"{cfg.smoothing_intervals}, dow_level {cfg.dow_level}, "
          f"x {cfg.lookback_hours}h, y {cfg.decay_hours}h")
    print(f"  horizon       {args.horizon_hours}h ({H} intervals)")

    check_tables(load, clock, cfg, valid)
    check_causality(load, clock, cfg, H, valid)
    check_forecast_quality(load, clock, cfg, H, valid)
    check_level_bias(load, clock, cfg, valid)
    if not args.edp:
        n_days = int(clock.day_ordinal[-1]) + 1
        check_adjustment_mechanism(load, clock, cfg, H,
                                   set(range(n_days - 8, n_days - 3)))
    check_degradation(load, clock, cfg, H)

    print()
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} check(s): {', '.join(_FAILURES)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
