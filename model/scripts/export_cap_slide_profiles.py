"""Export the NSW price profiles needed by the cap-mechanism slides.

Reads the per-year NEM price parquets and writes three small CSVs. Nothing
here recomputes the price stack: it calls build_layer_rates, so the profiles
carry the same loss factors, network periods and retail adders that billing
and the optimiser use.

Place at model/scripts/export_cap_slide_profiles.py and run from the project
root:

    python3 -m model.scripts.export_cap_slide_profiles --out data/output/cap_slides

Outputs (all indexed by local time-of-day, 5-minute bins, 288 rows):

  nsw_profile_pooled.csv
      Mean profile pooled over --profile-years (default 2021-2025), for the
      slide 4 left-hand panels. Columns: tod, spot_mwh, import_c_kwh,
      export_c_kwh, network_import_c_kwh, n.
      import_c_kwh is the full wholesale-passthrough stack the household
      pays: LF x (spot x (1 + GST) + network) + retail adders, with GST on
      spot where the retail tariff sets gst_on_wholesale_import (network
      rates and adders already include GST; exports carry none).

  nsw_wholesale_by_year.csv
      Mean spot price by time-of-day, one column per year, for slide 7.
      No network layer and no adders, so the loss factor can be applied (or
      not) at plot time. Columns: tod, spot_mwh_2021 .. spot_mwh_2026.

  nsw_annual_summary.csv
      One row per year: interval count, first/last timestamp, interval
      minutes, mean/median/p95/p99/max spot, and the share of intervals above
      $150 and $300 per MWh. The first/last timestamps are what tell the deck
      whether a year is partial.

Time-of-day binning uses local_wall_clock, so bins follow the site's real
local clock and DST is handled the same way tariff period lookups handle it.
Pooling across years is interval-weighted: years are concatenated before
grouping, so a partial year contributes in proportion to its intervals. 2026
is excluded from the pooled profile by default for that reason.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model.components.billing import GST_RATE
from model.core.localtime import local_wall_clock
from model.core.prices import build_layer_rates
from model.core.tariff import load_network_tariff, load_retail_tariff

RRP_COL = {'NSW': 'NSW1_rrp', 'QLD': 'QLD1_rrp',
           'SA': 'SA1_rrp', 'VIC': 'VIC1_rrp'}


def load_year(nem_prices_dir: Path, year: int,
              state: str) -> tuple[pd.Series, int]:
    """Return that year's RRP in $/MWh indexed by datetime_aest, and the
    number of trailing unsettled intervals dropped.

    The current year's file is a full-year skeleton, so everything after the
    last settled interval is NaN. Those rows are truncated. A NaN anywhere
    before the last settled interval is a gap in the middle of the data
    rather than an unwritten future, so it raises instead: filling it would
    silently bias whichever time-of-day bins it falls in.
    """
    path = nem_prices_dir / f'{year}_nem_prices.parquet'
    if not path.exists():
        raise FileNotFoundError(f"price file not found: {path}")
    col = RRP_COL[state]
    df = pd.read_parquet(path, columns=['datetime_aest', col])
    df = df.sort_values('datetime_aest').set_index('datetime_aest')
    s = df[col].astype(float)
    s.index = pd.DatetimeIndex(s.index)
    if s.index.tz is None:
        # local_wall_clock needs a real instant to convert from; the model
        # index is market time, so a naive index is AEST by convention.
        s.index = s.index.tz_localize('Etc/GMT-10')

    last_settled = s.last_valid_index()
    if last_settled is None:
        raise ValueError(f"{path.name}: {col} is entirely NaN.")
    n_trailing = int(len(s) - s.index.get_loc(last_settled) - 1)
    s = s.loc[:last_settled]

    if s.isna().any():
        gaps = s.index[s.isna()]
        raise ValueError(
            f"{path.name}: {len(gaps)} NaN values in {col} before the last "
            f"settled interval ({last_settled}), first at {gaps[0]}. These "
            "are gaps, not unsettled intervals. Clean them in the price "
            "pipeline before exporting."
        )
    return s, n_trailing


def interval_hours_of(index: pd.DatetimeIndex) -> float:
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        raise ValueError("cannot infer interval length from a single row")
    return deltas.mode().iloc[0].total_seconds() / 3600.0


def tod_bin(index: pd.DatetimeIndex, state: str, ih: float) -> np.ndarray:
    """Local time-of-day bin index, 5-minute resolution, 0..287."""
    lwc = local_wall_clock(index, state, ih)
    return lwc.hour * 12 + lwc.minute // 5


def tod_labels() -> list[str]:
    return [f"{b // 12:02d}:{(b % 12) * 5:02d}" for b in range(288)]


def stack_for_year(prices: pd.Series, network, retail, state: str,
                   ih: float) -> pd.DataFrame:
    """Per-interval import/export price in c/kWh, plus the network layer."""
    lr = build_layer_rates(prices.index, prices, network, retail, state, ih)
    gst = GST_RATE if retail.gst_on_wholesale_import else 0.0
    return pd.DataFrame({
        'spot_mwh': prices.to_numpy(),
        'import_c_kwh': (lr.wholesale_import * (1.0 + gst) + lr.network_import
                         + lr.retail_import) * 100.0,
        'export_c_kwh': (lr.wholesale_export + lr.network_export
                         + lr.retail_export) * 100.0,
        'network_import_c_kwh': lr.network_import * 100.0,
    }, index=prices.index)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--project-root', type=Path, default=Path('.'))
    ap.add_argument('--nem-prices-dir', type=Path, default=None,
                    help="default: <project-root>/data/clean/nem_prices")
    ap.add_argument('--tariffs-dir', type=Path, default=None,
                    help="default: <project-root>/model/tariffs")
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--state', default='NSW')
    ap.add_argument('--years', default='2021,2022,2023,2024,2025,2026',
                    help="all years to summarise and export by year")
    ap.add_argument('--profile-years', default='2021,2022,2023,2024,2025',
                    help="years pooled into nsw_profile_pooled.csv")
    ap.add_argument('--network-tariff-id',
                    default='NSW_ausgrid_ea025_ea029_2026')
    ap.add_argument('--retail-tariff-id', default='NSW_amber_wholesale_2026')
    args = ap.parse_args(argv)

    root = args.project_root
    nem_dir = args.nem_prices_dir or root / 'data' / 'clean' / 'nem_prices'
    tariffs_dir = args.tariffs_dir or root / 'model' / 'tariffs'
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    years = [int(y) for y in args.years.split(',')]
    profile_years = [int(y) for y in args.profile_years.split(',')]
    state = args.state

    network = load_network_tariff(args.network_tariff_id, tariffs_dir)
    retail = load_retail_tariff(args.retail_tariff_id, tariffs_dir)
    if retail.type != 'wholesale_exposed':
        raise ValueError(
            f"--retail-tariff-id must be a wholesale_exposed plan, "
            f"got type={retail.type!r}"
        )
    print(f"network : {network.name}  "
          f"(LF import {network.import_loss_factor:.6f}, "
          f"export {network.export_loss_factor:.6f})")
    print(f"retail  : {retail.name}")

    summary_rows = []
    by_year_spot = {}
    pooled_parts = []

    for year in years:
        prices, n_trailing = load_year(nem_dir, year, state)
        ih = interval_hours_of(prices.index)
        bins = tod_bin(prices.index, state, ih)
        df = stack_for_year(prices, network, retail, state, ih)
        df['tod_bin'] = bins

        summary_rows.append({
            'year': year,
            'n_intervals': len(prices),
            'unsettled_dropped': n_trailing,
            'interval_minutes': round(ih * 60, 3),
            'first': prices.index[0],
            'last': prices.index[-1],
            'mean_spot_mwh': prices.mean(),
            'median_spot_mwh': prices.median(),
            'p95_spot_mwh': prices.quantile(0.95),
            'p99_spot_mwh': prices.quantile(0.99),
            'max_spot_mwh': prices.max(),
            'min_spot_mwh': prices.min(),
            'share_above_150': float((prices > 150).mean()),
            'share_above_300': float((prices > 300).mean()),
            'share_negative': float((prices < 0).mean()),
        })

        g = df.groupby('tod_bin')['spot_mwh'].mean()
        by_year_spot[f'spot_mwh_{year}'] = g.reindex(range(288))

        if year in profile_years:
            pooled_parts.append(df)
        tail = (f"  (+{n_trailing:,} unsettled intervals dropped)"
                if n_trailing else "")
        print(f"  {year}: {len(prices):>7,} intervals @ "
              f"{ih * 60:g} min, {prices.index[0]} .. {prices.index[-1]}"
              f"{tail}")

    # ---- pooled profile (slide 4)
    #
    # Both the mean and the quartiles are exported. The mean of an evening
    # bin is pulled well above its own median by the small share of
    # intervals above $300/MWh, so the mean is the right summary for an
    # "average price" chart but the wrong base curve for a typical-day
    # illustration that adds a spike of its own.
    pooled = pd.concat(pooled_parts)
    cols = ['spot_mwh', 'import_c_kwh', 'export_c_kwh',
            'network_import_c_kwh']
    g = pooled.groupby('tod_bin')
    agg = g[cols].mean().reindex(range(288))
    for q, tag in ((0.25, 'p25'), (0.50, 'p50'), (0.75, 'p75')):
        qd = g[cols].quantile(q).reindex(range(288))
        for c in cols:
            agg[f'{c}_{tag}'] = qd[c]
    agg['n'] = g.size().reindex(range(288))
    agg.insert(0, 'tod', tod_labels())
    agg.to_csv(out / 'nsw_profile_pooled.csv', index=False)

    # ---- per-year wholesale profile (slide 7)
    by_year = pd.DataFrame(by_year_spot)
    by_year.insert(0, 'tod', tod_labels())
    by_year.to_csv(out / 'nsw_wholesale_by_year.csv', index=False)

    # ---- annual summary
    pd.DataFrame(summary_rows).to_csv(out / 'nsw_annual_summary.csv',
                                      index=False)

    print(f"\nwrote 3 files to {out}")
    print(f"  pooled profile over {profile_years}, "
          f"{int(agg['n'].min()):,}-{int(agg['n'].max()):,} intervals per bin")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
