"""When is the dispatch price actually known, and is our actuals series it?

Settles three competing explanations for a puzzle: AEMO's fact sheet says the
dispatch solution is published about 12 seconds after a 5-minute boundary, yet
the P5MIN lead-0 value for the same interval, published about 27 seconds later
still, matches our actuals only 10.5% of the time with a median error of $1.91
and a mean of $70.

    1. P5MIN RE-SOLVES rather than reporting the dispatch outcome. Its lead-0
       is its own estimate for the interval in progress. Everything we have
       built stands, and `forecast within` is correctly conservative.

    2. OUR ACTUALS ARE NOT THE 5-MINUTE DISPATCH PRICE. The price pipeline
       uses TRADINGPRICE via NEMOSIS; if anything in that path delivers
       half-hour values, a median error of $1.91 with a $70 mean is exactly
       the signature: agreement in flat periods, divergence on spikes. This
       would be serious, because every price in the model, including perfect
       foresight's objective, would be an average.

    3. THE PRICE BELONGS TO THE INTERVAL THAT JUST ENDED, not the one
       beginning. That would put the actual roughly 5 minutes later than the
       fact sheet implies and restore the original framing.

Each has a different consequence, so guessing is not an option.

The decisive measurement is LASTCHANGED minus SETTLEMENTDATE in DISPATCHPRICE,
the same technique that established the P5MIN and PREDISPATCH publication
offsets. SETTLEMENTDATE labels the interval END, so:

    lag about -4.8 min  the price is set at the START of the interval it
                        applies to, confirming the fact sheet, and a
                        controller knows it with ~4.7 min of the interval left
    lag about +0.2 min  the price is published just AFTER its interval ends,
                        and cannot be acted on within it

Usage (from project root):
    python3 -m pipeline.pre_dispatch.check_dispatch_timing --month 2026-01
    python3 -m pipeline.pre_dispatch.check_dispatch_timing --month 2026-01 \\
        --region SA1 --state SA
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.pre_dispatch.B_Clean_Predispatch import MARKET_TZ, read_store
from pipeline.pre_dispatch.nemweb_direct import (
    _dir_url, _list_links, _matches, _read_zip,
)

# DISPATCHPRICE lives in DATA/ and is small: about 45,000 rows per month
# across five regions, against roughly 500,000 for P5MIN.
DISPATCH_TOKEN = 'DISPATCHPRICE'
RAW_CACHE = Path('data/raw/predispatch')


def fetch_dispatch_price(period: pd.Period,
                         raw_cache: Path = RAW_CACHE) -> pd.DataFrame:
    """Download or read from cache one month of DISPATCHPRICE.

    Uses nemweb_direct's fetching primitives directly rather than
    compile_data, which only knows the forecast tables. Caches under the same
    naming scheme so a repeat run is free.
    """
    cached = Path(raw_cache) / f'{DISPATCH_TOKEN}_{period.strftime("%Y%m")}.parquet'
    if cached.exists():
        print(f"  cache hit {cached.name}")
        return pd.read_parquet(cached)

    url = _dir_url(period.year, period.month, 'DATA')
    parts = [u for u in _list_links(url) if _matches(u, DISPATCH_TOKEN)]
    if not parts:
        raise ValueError(
            f"No {DISPATCH_TOKEN} files for {period} at {url}. The table may "
            "be named differently in this era; list the directory to check.")
    print(f"  {len(parts)} part file(s)")
    df = pd.concat([_read_zip(p) for p in parts], ignore_index=True)
    Path(raw_cache).mkdir(parents=True, exist_ok=True)
    df.to_parquet(cached, index=False)
    print(f"  cached {cached.name} ({len(df):,} rows)")
    return df


def load_model_actuals(state: str, year: int) -> pd.Series:
    """The price series the model bills against."""
    path = Path('data/clean/nem_prices') / f'{year}_nem_prices.parquet'
    if not path.exists():
        raise FileNotFoundError(f"model price file not found: {path}")
    df = pd.read_parquet(path)
    col = f'{state}1_rrp'
    if col not in df.columns:
        raise ValueError(f"{path} has no column {col}; present: {list(df.columns)}")
    s = df.set_index('datetime_aest')[col]
    s.index = pd.to_datetime(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize(MARKET_TZ)
    return s.astype(float)


def _describe(a: pd.Series, b: pd.Series, name_a: str, name_b: str) -> None:
    j = pd.DataFrame({'a': a, 'b': b}).dropna()
    if j.empty:
        print(f"  {name_a} vs {name_b}: no overlap")
        return
    e = (j['a'] - j['b']).abs()
    # Agreement is tested TO THE CENT, not bit-for-bit. Two price series can
    # differ in the seventh significant figure through float32 downcasting or
    # rounding somewhere in an extraction path while being the same series in
    # every sense that matters. An earlier version used a bitwise-tight
    # tolerance and reported two identical series as 48.6% equal, with an MAE
    # of $0.00, which is a contradiction that should have been caught in
    # review rather than in the field.
    exact = float((j['a'] - j['b']).abs().lt(0.005).mean())
    spike = j[j['b'] > 300]
    line = (f"  {name_a:<22} vs {name_b:<18} n={len(j):>7,}  "
            f"exact {exact:6.1%}  MAE ${e.mean():8.2f}  "
            f"median ${e.median():7.2f}  p99 ${e.quantile(0.99):9.2f}")
    if len(spike):
        se = (spike['a'] - spike['b']).abs()
        line += f"   | spikes n={len(spike):,} MAE ${se.mean():,.0f}"
    print(line)


def run(month: str, region: str = 'SA1', state: str = 'SA',
        store_root: Path = Path('data/clean/predispatch')) -> None:
    period = pd.Period(month, freq='M')
    print(f"=== Dispatch price timing and identity: {region}, {period} ===\n")

    print("Fetching DISPATCHPRICE ...")
    raw = fetch_dispatch_price(period)
    need = ['SETTLEMENTDATE', 'REGIONID', 'RRP', 'INTERVENTION', 'LASTCHANGED']
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise ValueError(f"DISPATCHPRICE missing {missing}; present: "
                         f"{sorted(raw.columns)}")

    d = raw.loc[:, need].copy()
    d = d[(d['REGIONID'].astype(str) == region) & (d['INTERVENTION'] == 0)]
    d['SETTLEMENTDATE'] = pd.to_datetime(d['SETTLEMENTDATE'])
    d['LASTCHANGED'] = pd.to_datetime(d['LASTCHANGED'])
    d = d.drop_duplicates(subset=['SETTLEMENTDATE'], keep='last')
    print(f"  {len(d):,} intervals after INTERVENTION == 0\n")

    # ---- QUESTION 3: which interval does the price belong to? -----------
    lag_min = (d['LASTCHANGED'] - d['SETTLEMENTDATE']).dt.total_seconds() / 60
    print("--- Publication timing (LASTCHANGED - SETTLEMENTDATE) ---")
    print(f"  mean {lag_min.mean():+.2f} min   median {lag_min.median():+.2f}   "
          f"p1 {lag_min.quantile(0.01):+.2f}   p99 {lag_min.quantile(0.99):+.2f}")
    med = lag_min.median()
    if med < -2:
        # SETTLEMENTDATE labels the interval END, so a median of -4.85 means
        # the price is written 4.85 min BEFORE the interval ends, i.e. 0.15 min
        # after it begins. Remaining time is -med, not 5 + med.
        print(f"  => the price is set near the START of the interval it applies")
        print(f"     to: written {-med:.2f} min before the interval ends, so "
              f"{5 + med:.2f} min")
        print(f"     after it begins. A controller knows it with about "
              f"{-med:.1f} min of the interval remaining.")
        print("     This CONFIRMS the fact sheet and refutes explanation 3.")
    elif med > -1:
        print("  => the price is published at or after the END of its interval,")
        print("     so it cannot be acted on within that interval.")
        print("     This SUPPORTS explanation 3 and the original framing.")
    else:
        print("  => ambiguous; inspect the distribution before concluding.")
    print()

    # ---- QUESTIONS 1 and 2: what is what? -------------------------------
    disp = d.set_index(d['SETTLEMENTDATE'].dt.tz_localize(MARKET_TZ))['RRP'].astype(float)
    disp.name = 'dispatch'

    actuals = load_model_actuals(state, period.year)
    lo = disp.index.min()
    hi = disp.index.max()
    fine = read_store(store_root, 'p5min', region, lo - pd.Timedelta(hours=1), hi)
    lead0 = (fine[fine['lead_minutes'] == 0]
             .set_index('run_time')['rrp_mwh'].astype(float))
    lead5 = (fine[fine['lead_minutes'] == 5]
             .assign(t=lambda x: x['run_time'] + pd.Timedelta(minutes=5))
             .set_index('t')['rrp_mwh'].astype(float))

    print("--- Which series is which? ---")
    _describe(disp, actuals, 'DISPATCHPRICE.RRP', 'model actuals')
    _describe(lead0, disp, 'P5MIN lead 0', 'DISPATCHPRICE')
    _describe(lead0, actuals, 'P5MIN lead 0', 'model actuals')
    _describe(lead5, disp, 'P5MIN lead 5', 'DISPATCHPRICE')
    print()

    # ---- ALIGNMENT: does P5MIN lead 0 mean the same interval as
    # DISPATCHPRICE at the same label? Everything downstream assumes it does.
    # If P5MIN's first row were really the NEXT interval, `next_forecast` would
    # price interval k with a forecast of k+1, which would be a serious bug and
    # would also explain a large lead-0 error. Scanning offsets settles it: the
    # true alignment is whichever shift minimises the error.
    print("--- Alignment scan: P5MIN lead 0 against DISPATCHPRICE, shifted ---")
    print("    (offset 0 winning confirms both label the same interval)")
    best, best_mae = None, np.inf
    for off in (-10, -5, 0, 5, 10):
        shifted = lead0.copy()
        shifted.index = shifted.index + pd.Timedelta(minutes=off)
        k = pd.DataFrame({'a': shifted, 'b': disp}).dropna()
        if k.empty:
            continue
        mae = float((k['a'] - k['b']).abs().mean())
        corr = float(k['a'].corr(k['b']))
        same = float((k['a'] - k['b']).abs().lt(0.005).mean())
        flag = ''
        if mae < best_mae:
            best, best_mae = off, mae
        print(f"    offset {off:+3d} min: MAE ${mae:8.2f}   corr {corr:6.4f}   "
              f"same to 1c {same:5.1%}{flag}")
    print(f"    => best alignment at {best:+d} min", end='')
    if best == 0:
        print(". Confirms the assumption: P5MIN lead 0 and DISPATCHPRICE at the")
        print("       same label are the same interval, and lead 0 is a genuine")
        print("       re-forecast of an interval whose price is already set.")
    else:
        print(f". CONTRADICTS the assumption. P5MIN's first row is NOT the")
        print(f"       interval its label suggests; every lead is offset by "
              f"{best:+d} min.")
        print("       `next_forecast` is misaligned and must be corrected.")
    print()

    j = pd.DataFrame({'disp': disp, 'act': actuals, 'l0': lead0}).dropna()
    if j.empty:
        print("  no overlap; cannot conclude")
        return
    disp_eq_act = float((j['disp'] - j['act']).abs().lt(0.005).mean())
    l0_eq_disp = float((j['l0'] - j['disp']).abs().lt(0.005).mean())

    print("--- Conclusion ---")
    if disp_eq_act > 0.99 and l0_eq_disp < 0.5:
        print("  EXPLANATION 1. Our actuals ARE the 5-minute dispatch price, and")
        print("  P5MIN lead-0 is a separate re-solve rather than a copy of it.")
        print("  Everything built so far stands. 'forecast within' is")
        print("  conservative: a real controller also knows the dispatch price.")
    elif disp_eq_act < 0.99 and l0_eq_disp > 0.99:
        print("  EXPLANATION 2. P5MIN lead-0 IS the dispatch price, but our")
        print("  actuals are NOT. Every price in the model is suspect,")
        print("  including perfect foresight's objective. Check whether the")
        print("  nem_prices pipeline is delivering 30-minute averages.")
    elif disp_eq_act < 0.99 and l0_eq_disp < 0.99:
        print("  NEITHER matches the dispatch price exactly. Check the")
        print("  half-hourly signature below before drawing conclusions.")
    else:
        print("  Both match the dispatch price; the original puzzle dissolves.")

    # A 30-minute average shows up as constant runs of six 5-minute values.
    for name, s in (('model actuals', j['act']), ('DISPATCHPRICE', j['disp'])):
        v = s.to_numpy()
        n = len(v) - len(v) % 6
        blocks = v[:n].reshape(-1, 6)
        flat = float((blocks.std(axis=1) < 1e-9).mean())
        print(f"  {name:<16} constant within half-hour blocks: {flat:.1%}"
              + ("   <- looks like a 30-minute average" if flat > 0.5 else ""))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--month', required=True, help='YYYY-MM')
    parser.add_argument('--region', default='SA1')
    parser.add_argument('--state', default='SA')
    parser.add_argument('--store', type=Path,
                        default=Path('data/clean/predispatch'))
    args = parser.parse_args()
    run(args.month, region=args.region, state=args.state, store_root=args.store)
