"""Shared loading and weighting for the sensitivity result scripts.

The sensitivity runs cover one state and year and are read against the full
run's own scenario for the same state, year and battery size. Two properties
hold for every comparison built here.

Paired site sets. A sensitivity run may cover a subset of the full run's
sites, so every comparison is restricted to the sites present in all of the
runs being compared, and the weights are renormalised over that set. Comparing
a subset's mean against the full run's mean would read a change in the site
mix as a change in the result.

Weighting. Site weights are financial_analysis's PV size band weights, so a
site carries the same weight here as in the results these are compared
against.

See docs/sensitivity_runs_spec.md.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import financial_analysis as fa          # noqa: E402
import flow_exposure_analysis as fea     # noqa: E402
import thesis_style as ts                # noqa: E402
from site_exclusions import EXCLUDED_SITES  # noqa: E402

STATE = 'NSW'
YEAR = '2024'
OUTPUT_ROOT = Path('analysis/output/sensitivity')
WEIGHTS_PATH = Path('analysis/config/pv_size_bands.yaml')
BATTERY_SIZES = fa.BATTERY_SIZES

# Reference runs the four sets are read against.
REF_ROLLING_AMBER = 'full_run_rolling_amber'              # 18 h, c_deg 3.9
REF_OPTIMISED_AMBER = 'full_run_optimised_amber'          # perfect foresight
REF_SC_AMBER = 'full_run_self_consumption_amber'
REF_SOLAR_AMBER = 'full_run_solar_only_amber'
REF_ORIGIN_1H = 'full_run_optimised_origin_lite'
REF_ORIGIN_10H = '10hr_full_run_optimised_origin_lite'
REF_SC_ORIGIN = 'full_run_self_consumption_origin_novpp'
REF_SOLAR_ORIGIN = 'full_run_solar_only_origin_starter'
REF_ROLLING_ORIGIN = 'full_run_rolling_origin_novpp'
REF_ROLLING_EA = 'full_run_rolling_ea_novpp'
REF_SC_EA = 'full_run_self_consumption_ea'
REF_SOLAR_EA = 'full_run_solar_only_ea'

# Annual VPP enrolment payment for an enrolled but undispatched battery, as
# financial_analysis applies it on the conservative basis. EnergyAustralia's
# Battery Ease pays a monthly credit; Origin Battery Starter is not an
# enrolled plan and receives nothing.
EA_ENROLMENT = fa.ENROLMENT_PAYMENTS['ea']


def runs_root(data_root: Path) -> Path:
    return Path(data_root) / 'data' / 'output' / 'runs'


def scenario_id(size: int | None, state: str = STATE, year: str = YEAR) -> str:
    return f'{state}_{year}' if size is None else f'{state}_{year}_{size}kWh'


# ── Bills ─────────────────────────────────────────────────────────────────────

def load_bills(runs: Path, run: str, size: int | None,
               state: str = STATE, year: str = YEAR) -> pd.DataFrame | None:
    """Per-site bill quantities for one run and scenario, or None if absent.

    Columns: site_id, household_total, net_wholesale, retailer_margin and
    vpp_reward. vpp_reward is retail_other_credits with its sign flipped, so
    a reward the household receives is positive; that field carries the VPP
    per-kWh rewards and zero-import rewards (docs/model_spec.md).
    """
    f = runs / run / 'scenarios' / scenario_id(size, state, year) / 'bills.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df['net_wholesale'] = (df['wholesale_usage_cost']
                           + df['wholesale_feedin_credit'])
    df['vpp_reward'] = -df.get('retail_other_credits', 0.0)
    keep = ['site_id', 'household_total', 'net_wholesale', 'vpp_reward']
    if 'retailer_margin' in df.columns:
        keep.append('retailer_margin')
    df = df[keep]
    return df[~df['site_id'].isin(EXCLUDED_SITES)].set_index('site_id')


def require_bills(runs: Path, run: str, size: int | None,
                  state: str = STATE, year: str = YEAR) -> pd.DataFrame:
    df = load_bills(runs, run, size, state, year)
    if df is None:
        raise SystemExit(
            f'Missing bills.csv for {run} {scenario_id(size, state, year)}. '
            f'Run it, or point --data-root at the drive holding it.')
    return df


def savings(bills: pd.DataFrame, solar: pd.DataFrame) -> pd.DataFrame:
    """Per-site saving against a solar-only run, as positive dollars saved."""
    joined = bills.join(solar[['hh_solar', 'ws_solar']], how='inner')
    out = pd.DataFrame(index=joined.index)
    out['hh_saving'] = joined['hh_solar'] - joined['household_total']
    out['ws_saving'] = joined['ws_solar'] - joined['net_wholesale']
    for col in ('vpp_reward', 'retailer_margin'):
        if col in joined.columns:
            out[col] = joined[col]
    return out


def load_solar_only(runs: Path, run: str, state: str = STATE,
                    year: str = YEAR) -> pd.DataFrame:
    """Solar-only baseline bills, indexed by site_id."""
    f = runs / run / 'scenarios' / f'{state}_{year}' / 'bills.csv'
    if not f.exists():
        raise SystemExit(f'Missing solar-only baseline: {f}')
    df = pd.read_csv(f)
    df['ws_solar'] = df['wholesale_usage_cost'] + df['wholesale_feedin_credit']
    df = df.rename(columns={'household_total': 'hh_solar'})
    df = df[~df['site_id'].isin(EXCLUDED_SITES)]
    return df[['site_id', 'hh_solar', 'ws_solar']].set_index('site_id')


# ── Weighting ─────────────────────────────────────────────────────────────────

def site_weights(data_root: Path, state: str = STATE, year: str = YEAR,
                 weights_path: Path = WEIGHTS_PATH) -> pd.Series:
    """PV size band weights for one state and year, as financial_analysis
    computes them: a nationally pooled DC:AC ratio, per-state regression
    fallback, then the band weights spread over each band's sites."""
    cfg = fa.load_weights_config(weights_path)
    pooled_frames = []
    for other in cfg['by_state'].keys():
        try:
            pooled_frames.append(fa.load_site_meta(data_root, year, other))
        except FileNotFoundError:
            continue
    pooled_meta = pd.concat(pooled_frames, ignore_index=True)
    ratio = fa.compute_dc_ac_ratio(pooled_meta)
    pooled = fa.assign_direct_size(pooled_meta, ratio)

    meta = fa.assign_direct_size(fa.load_site_meta(data_root, year, state), ratio)
    meta = fa.estimate_missing_sizes(meta, state, fallback_meta=pooled)
    weight, _ = fa.assign_bands_and_weights(
        meta, state, cfg['bands'], cfg['by_state'])
    return weight


def common_index(*frames: pd.DataFrame | pd.Series) -> pd.Index:
    """Sites present in every frame, so a comparison is paired."""
    idx = None
    for f in frames:
        if f is None:
            continue
        idx = f.index if idx is None else idx.intersection(f.index)
    return pd.Index([]) if idx is None else idx


def wmean(values: pd.Series, weight: pd.Series | None) -> float:
    """Weighted mean over the intersection, renormalised. Unweighted when
    weight is None."""
    return fea.weighted_series_mean(values, weight)


# ── Rolling diagnostics and battery flows ─────────────────────────────────────

def load_diagnostics(runs: Path, run: str, size: int,
                     state: str = STATE, year: str = YEAR) -> pd.DataFrame | None:
    """rolling_diagnostics.csv for one run and scenario, indexed by site_id."""
    f = (runs / run / 'scenarios' / scenario_id(size, state, year)
         / 'rolling_diagnostics.csv')
    if not f.exists():
        return None
    return pd.read_csv(f).set_index('site_id')


def battery_throughput(runs: Path, run: str, size: int,
                       state: str = STATE, year: str = YEAR
                       ) -> tuple[pd.DataFrame, float] | None:
    """Per-site annual battery charge and discharge (kWh) and equivalent full
    cycles per day, from battery_flows.parquet.

    battery_flows carries the net battery AC flow per interval, positive when
    charging. Discharge is the sum of the negative side. Equivalent full
    cycles per day is annual discharge over the usable capacity, over the
    days covered, where usable capacity is the nameplate less the backup
    reserve, matching the cycling chart's convention.
    """
    df = fea.load_battery_flows(runs, run, state, year, size)
    if df is None:
        return None
    charge = df.clip(lower=0).sum()
    discharge = df.clip(upper=0).abs().sum()
    days = len(df) * 5.0 / (60.0 * 24.0)
    out = pd.DataFrame({'charge_kwh': charge, 'discharge_kwh': discharge})
    out.index.name = 'site_id'
    return out[~out.index.isin(EXCLUDED_SITES)], days


def cycles_per_day(discharge_kwh: float, size: int, days: float,
                   backup_reserve_pct: float = 20.0) -> float:
    usable = size * (1.0 - backup_reserve_pct / 100.0)
    return discharge_kwh / usable / days


# ── Output ────────────────────────────────────────────────────────────────────

def save_figure(fig, path: Path, thesis_dir: Path | None = None) -> None:
    """Write the figure, and copy it into the thesis figures directory when
    one is given. thesis_style.save_figure closes the figure, so the copy is
    made from the written file rather than by drawing it twice."""
    ts.save_figure(fig, path)
    if thesis_dir is not None:
        thesis_dir = Path(thesis_dir)
        thesis_dir.mkdir(parents=True, exist_ok=True)
        dest = thesis_dir / Path(path).with_suffix('.png').name
        shutil.copyfile(Path(path).with_suffix('.png'), dest)
        print(f'  Saved: {dest}')


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f'  Saved: {path}')


def add_common_args(parser) -> None:
    parser.add_argument('--data-root', type=Path, default=Path('.'),
                        help='Project root holding data/output/runs.')
    parser.add_argument('--state', default=STATE)
    parser.add_argument('--year', default=YEAR)
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Override the output directory.')
    parser.add_argument('--thesis-dir', type=Path, default=None,
                        help='Also copy the figure here, e.g. '
                             'thesis/figures/results.')
    parser.add_argument('--unweighted', action='store_true',
                        help='Straight site mean instead of PV size band '
                             'weighting.')
    parser.add_argument('--print-values', action='store_true',
                        help='Print the values the thesis text cites.')
