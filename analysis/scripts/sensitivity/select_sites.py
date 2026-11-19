"""A stratified site subset for the sensitivity runs.

The sensitivity runs cover one state and year, so their cost is set by the
site count. Cutting sites at random, or with --max-sites, moves the PV size
band mix and so moves the weighted means the results are compared against.
This picks a subset that holds the band mix of the full site set.

Within each PV size band the sites are ordered by size_kw and taken at evenly
spaced ranks, so the subset spans the band rather than clustering at one end,
and the same subset comes back on every call. Band assignment is
financial_analysis's, so a site lands in the same band here as in the
analysis that consumes the runs.

Sites are restricted to those present in --reference-run, since every
sensitivity comparison is paired against that run's own scenario, and the
study-wide exclusions are applied.

Usage (from project root):
    python3 analysis/scripts/sensitivity/select_sites.py --data-root "/Volumes/Manual/Thesis Project"
    python3 analysis/scripts/sensitivity/select_sites.py --data-root ... --fraction 0.6
    python3 analysis/scripts/sensitivity/select_sites.py --data-root ... --n 60 --quiet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import financial_analysis as fa  # noqa: E402


def band_table(data_root: Path, state: str, year: str,
               weights_path: Path) -> pd.DataFrame:
    """site_id, size_kw and band for every site in the state and year."""
    weights = fa.load_weights_config(weights_path)

    # The DC:AC ratio is pooled nationally and computed once per year, and
    # the pooled frame is the regression fallback for states with too few
    # tier-1/2 sites of their own. Same order as financial_analysis.main.
    pooled_frames = []
    for other_state in weights['by_state'].keys():
        try:
            pooled_frames.append(fa.load_site_meta(data_root, year, other_state))
        except FileNotFoundError:
            continue
    pooled_meta = pd.concat(pooled_frames, ignore_index=True)
    ratio = fa.compute_dc_ac_ratio(pooled_meta)
    pooled = fa.assign_direct_size(pooled_meta, ratio)

    meta = fa.assign_direct_size(fa.load_site_meta(data_root, year, state), ratio)
    meta = fa.estimate_missing_sizes(meta, state, fallback_meta=pooled)

    lo = weights['bands'][0][1]
    hi = weights['bands'][-1][2]

    def band_of(kw: float) -> str | None:
        if pd.isna(kw) or kw < lo or kw > hi:
            return None
        for label, band_lo, band_hi in weights['bands']:
            if band_lo <= kw <= band_hi:
                return label
        return None

    meta['band'] = meta['size_kw'].apply(band_of)
    return meta[['site_id', 'size_kw', 'band']]


def sites_in_run(data_root: Path, run: str, state: str, year: str,
                 battery_kwh: int) -> set[str]:
    bills = (data_root / 'data' / 'output' / 'runs' / run / 'scenarios'
             / f'{state}_{year}_{battery_kwh}kWh' / 'bills.csv')
    if not bills.exists():
        raise SystemExit(f'Reference run scenario not found: {bills}')
    return set(pd.read_csv(bills, usecols=['site_id'])['site_id'])


def evenly_spaced(values: list, k: int) -> list:
    """k items from values at evenly spaced ranks, endpoints included."""
    n = len(values)
    if k >= n:
        return list(values)
    if k <= 1:
        return [values[n // 2]]
    step = (n - 1) / (k - 1)
    return [values[round(i * step)] for i in range(k)]


def select(meta: pd.DataFrame, target: int) -> pd.DataFrame:
    """A subset of about `target` sites holding the band mix of `meta`."""
    banded = meta[meta['band'].notna()].copy()
    total = len(banded)
    chosen = []
    for band, rows in banded.groupby('band'):
        share = len(rows) / total
        k = max(1, round(share * target))
        ordered = rows.sort_values('size_kw')['site_id'].tolist()
        chosen.extend(evenly_spaced(ordered, k))
    return meta[meta['site_id'].isin(chosen)].sort_values('site_id')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('.'))
    parser.add_argument('--state', default='NSW')
    parser.add_argument('--year', default='2024')
    parser.add_argument('--fraction', type=float, default=None,
                        help='Share of the eligible sites to keep.')
    parser.add_argument('--n', type=int, default=None,
                        help='Target site count (overrides --fraction).')
    parser.add_argument('--reference-run', default='full_run_rolling_amber',
                        help='Run whose site set the subset is drawn from.')
    parser.add_argument('--reference-size', type=int, default=20,
                        help='Battery size of the reference scenario.')
    parser.add_argument('--weights', type=Path,
                        default=Path('analysis/config/pv_size_bands.yaml'))
    parser.add_argument('--quiet', action='store_true',
                        help='Print only the comma-separated site list.')
    args = parser.parse_args(argv)

    meta = band_table(args.data_root, args.state, args.year, args.weights)
    runnable = sites_in_run(args.data_root, args.reference_run, args.state,
                            args.year, args.reference_size)
    meta = meta[meta['site_id'].isin(runnable)]

    eligible = meta[meta['band'].notna()]
    if args.n is not None:
        target = args.n
    elif args.fraction is not None:
        target = round(args.fraction * len(eligible))
    else:
        target = len(eligible)

    picked = select(meta, target)
    ids = picked['site_id'].tolist()

    if not args.quiet:
        print(f'{args.state} {args.year}: {len(runnable)} sites in '
              f'{args.reference_run}, {len(eligible)} inside the size bands, '
              f'{len(ids)} selected.\n')
        full = eligible.groupby('band')['site_id'].count()
        sub = picked[picked['band'].notna()].groupby('band')['site_id'].count()
        print(f"{'band':<10} {'all':>5} {'selected':>9} {'all share':>10} "
              f"{'sel share':>10}")
        for band in full.index:
            n_all = int(full[band])
            n_sel = int(sub.get(band, 0))
            print(f'{band:<10} {n_all:>5} {n_sel:>9} '
                  f'{n_all / full.sum():>10.3f} {n_sel / max(1, sub.sum()):>10.3f}')
        dropped = picked[picked['band'].isna()]
        if len(dropped):
            print(f'\n{len(dropped)} site(s) outside the band range are not '
                  f'selected.')
        print('\n--sites argument:\n')

    print(','.join(ids))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
