"""Interactive visualisation of perfect_foresight_lp output for a single real site.

Usage (from project root):
    python3 -m model.dev.viz_perfect_foresight_lp SITE_ID YEAR
    python3 -m model.dev.viz_perfect_foresight_lp SITE_ID YEAR --state NSW \\
        --network-tariff ausgrid_ea025_network_2024 \\
        --retail-tariff amber_wholesale_2024 --export-limit-kw 10

Loads the clean EDP trace and NEM price file for one site/year, builds the
customer-facing per-interval price vectors (via model.core.prices), runs
self_consumption first to obtain a baseline bill and the end-of-period SOC,
then runs perfect_foresight_lp with:
  - the hardcoded DEV_BATTERY (no degradation cost; c_deg defaults to 0),
  - SOC initialised at the reserve floor and required to END at the
    self_consumption end-of-period SOC (fair LP-vs-self_consumption comparison),
  - a per-phase export limit (default 10 kW, 1 phase assumed for this dev run).

Two interactive windows open at once, both navigated together:

  FLOWS window (3 panels):
    1. Solar generation: stacked sol_to_home (solar), sol_to_bat (battery),
       sol_to_grid (grid), curtail (light orange) at the top. Solar line.
    2. Serviced load: stacked sol_to_home, bat_to_home, grid_to_home. Load line.
    3. SOC (left axis, green line, capacity/reserve dashed) + charge/discharge
       columns (right axis, kW): charge = sol_to_bat (solar) + grid_to_bat (grid);
       discharge = bat_to_home (load blue) + bat_to_grid (battery). Both positive,
       drawn side by side within each interval.

  PRICES window (3 panels):
    1. Export (left axis, kW): sol_to_grid (solar) + bat_to_grid (battery),
       green total outline; export price (right axis, c/kWh) dotted, solar colour.
    2. Import (left axis, kW): grid_to_home (load blue) + grid_to_bat (battery),
       grey total outline; import price (right axis, c/kWh) dotted, grey.
    3. Same SOC + charge/discharge panel as the FLOWS window.

Controls:
    Enter / n       advance one day
    b               go back one day
    <YYYY-MM-DD>    jump to a specific date
    q               quit

CURTAILMENT NOTE
----------------
The LP may curtail solar (when export-limited, or when the export rate is
negative and exporting would cost money). SiteTraces has no explicit
curtail_kwh field, so curtailment is recovered here as
    curtail = solar - (sol_to_home + sol_to_bat + sol_to_grid).
The solar invariant therefore reads
    solar = sol_to_home + sol_to_bat + sol_to_grid + curtail.
A clean long-term fix is to add curtail_kwh to SiteTraces (self_consumption
would set it to zeros); that is a deliberate downstream change left for later.
"""
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.core.paths import default_paths, Paths
from model.core.scenario import BatterySpec
from model.core.tariff import (
    NetworkTariff, RetailTariff, load_network_tariff, load_retail_tariff,
)
from model.core.prices import build_price_vectors
from model.core.site_result import SiteBill
from model.components.self_consumption import self_consumption
from model.components.perfect_foresight_lp import perfect_foresight_lp, lp_backend
from model.components.billing import compute_bill
from model.dev._dev_common import DEV_BATTERY, load_site_trace as _load_site_trace


# Colour palette (matching viz_self_consumption)
COLOUR_SOLAR = '#F5A623'    # yellow/orange
COLOUR_BATTERY = '#5DA13E'  # green
COLOUR_GRID = '#888888'     # grey
COLOUR_LOAD = '#1F77B4'     # blue
COLOUR_CURTAIL = '#FBD38D'  # light orange (curtailed solar)

_C_PER_DOLLAR = 100.0       # $/kWh -> c/kWh for price axes

# Battery coordination mode label shown in chart titles after the retail tariff
# name.  Edit this string to describe the strategy being visualised.
BATTERY_MODE = 'Household Bill Optimised'


# ---------------------------------------------------------------------------
# Data bundle and analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LPAnalysis:
    site_id: str
    year: int
    trace: pd.DataFrame
    sc_result: object         # SiteTraces from self_consumption
    lp_result: object         # SiteTraces from perfect_foresight_lp
    sc_bill: SiteBill
    lp_bill: SiteBill
    p_imp: pd.Series          # $/kWh
    p_exp: pd.Series          # $/kWh
    curtail: pd.Series        # kWh per interval (recovered)
    battery_spec: BatterySpec
    network_tariff: NetworkTariff
    retail_tariff: RetailTariff
    export_limit_kw: float
    state: str
    soc_init_kwh: float
    soc_final_kwh: float
    interval_hours: float
    backend: str          # 'gurobi' or 'highs'
    lp_solve_seconds: float


def _load_prices_mwh(state: str, year: int, paths: Paths) -> pd.Series:
    """Load the NEM RRP series ($/MWh) for the state and year.

    Mirrors ingest._load_prices minimally (no price_flag warning here; the
    dev run surfaces any NaN via the price-vector validation downstream).
    """
    price_path = paths.nem_prices_dir / f'{year}_nem_prices.parquet'
    if not price_path.exists():
        raise FileNotFoundError(f"NEM price file not found: {price_path}")
    df = pd.read_parquet(price_path)
    rrp_col = f'{state}1_rrp'
    if rrp_col not in df.columns:
        raise ValueError(
            f"NEM price file {price_path} missing column {rrp_col!r}. "
            f"Available: {list(df.columns)}"
        )
    series = pd.Series(
        df[rrp_col].to_numpy(),
        index=pd.to_datetime(df['datetime_aest']).dt.tz_convert('Etc/GMT-10'),
        name=f'{state}_rrp_dollars_per_mwh',
    )
    series.index.name = 'datetime_aest'
    return series


def load_inputs(
    site_id: str,
    year: int,
    state: str,
    network_tariff_id: str,
    retail_tariff_id: str,
    paths: Paths,
) -> tuple[pd.DataFrame, pd.Series, NetworkTariff, RetailTariff]:
    """Load the real site trace, NEM prices, and tariffs for a dev run."""
    trace = _load_site_trace(site_id, year, paths)
    prices_mwh = _load_prices_mwh(state, year, paths)
    # Align prices to the trace index (both full-year 5-min AEST); any genuine
    # mismatch surfaces as NaN, which build_price_vectors rejects.
    prices_mwh = prices_mwh.reindex(trace.index)
    network_tariff = load_network_tariff(network_tariff_id, paths.tariffs_dir)
    retail_tariff = load_retail_tariff(retail_tariff_id, paths.tariffs_dir)
    return trace, prices_mwh, network_tariff, retail_tariff


def run_lp_analysis(
    site_id: str,
    year: int,
    trace: pd.DataFrame,
    prices_mwh: pd.Series,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    battery_spec: BatterySpec,
    export_limit_kw: float,
    state: str,
    interval_hours: float = 5 / 60,
) -> LPAnalysis:
    """Run self_consumption then perfect_foresight_lp and bill both.

    Pure computation given inputs (no disk access), so it can be exercised on
    synthetic data in tests.

    Args:
        state: the site's actual location (scenario.state), used to resolve
            local time for TOU period lookups in build_price_vectors and
            compute_bill. Required -- there is no sane default, and passing
            the wrong state silently bills the wrong tariff periods rather
            than raising, so this parameter is deliberately mandatory rather
            than defaulted to 'NSW' or similar. perfect_foresight_lp itself
            does not need state: it optimises against the already-resolved
            p_imp/p_exp vectors and has no clock-driven tariff features of
            its own (that's only perfect_foresight_lp_advanced).
    """
    solar = trace['pv_site_net_kwh']
    load = trace['gross_load_kwh']

    # Customer-facing price vectors (the prices the LP optimises against and
    # billing charges — same source, by construction).
    p_imp, p_exp = build_price_vectors(
        solar.index, prices_mwh, network_tariff, retail_tariff, state, interval_hours,
    )

    # Self-consumption first: baseline bill + end-of-period SOC for the LP.
    sc_result = self_consumption(site_id, solar, load, battery_spec, interval_hours)

    soc_min = (battery_spec.backup_reserve_pct / 100.0) * battery_spec.capacity_kwh
    soc_init = soc_min
    soc_final = float(sc_result.battery_soc_kwh.iloc[-1])
    # Clamp tiny float drift into bounds so the LP's bound check is satisfied.
    soc_final = min(max(soc_final, soc_min), battery_spec.capacity_kwh)

    export_limit_kwh = export_limit_kw * interval_hours
    _lp_t0 = time.perf_counter()
    lp_result = perfect_foresight_lp(
        site_id, solar, load, p_imp, p_exp, battery_spec,
        soc_init_kwh=soc_init, soc_final_kwh=soc_final,
        export_limit_kwh_per_interval=export_limit_kwh,
        interval_hours=interval_hours,
    )
    lp_solve_seconds = time.perf_counter() - _lp_t0

    sc_bill = compute_bill(sc_result, prices_mwh, network_tariff, retail_tariff,
                           state, interval_hours)
    lp_bill = compute_bill(lp_result, prices_mwh, network_tariff, retail_tariff,
                           state, interval_hours)

    curtail = _recover_curtail(solar, lp_result)

    return LPAnalysis(
        site_id=site_id, year=year, trace=trace,
        sc_result=sc_result, lp_result=lp_result,
        sc_bill=sc_bill, lp_bill=lp_bill,
        p_imp=p_imp, p_exp=p_exp, curtail=curtail,
        battery_spec=battery_spec,
        network_tariff=network_tariff, retail_tariff=retail_tariff,
        export_limit_kw=export_limit_kw,
        state=state,
        soc_init_kwh=soc_init, soc_final_kwh=soc_final,
        interval_hours=interval_hours,
        backend=lp_backend(),
        lp_solve_seconds=lp_solve_seconds,
    )


def _recover_curtail(solar: pd.Series, result) -> pd.Series:
    """Return curtailment series from SiteTraces (now an explicit field)."""
    return result.curtail_kwh


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _print_verification(analysis: LPAnalysis) -> None:
    """Print startup verification stats. Emits 'FAIL' tokens on any check failure."""
    a = analysis
    r = a.lp_result
    solar = a.trace['pv_site_net_kwh']
    load = a.trace['gross_load_kwh']
    cap = a.battery_spec.capacity_kwh
    min_soc = (a.battery_spec.backup_reserve_pct / 100.0) * cap
    tol = 1e-6  # LP carries solver float noise; looser than self_consumption's 1e-9

    def status(ok: bool) -> str:
        return 'PASS' if ok else 'FAIL'

    print()
    print(f"=== Verification: site {a.site_id} year {a.year} (perfect_foresight_lp) ===")
    T = len(a.lp_result.import_kwh)
    print(f"--- LP solve timing ---")
    print(f"backend:         {a.backend}")
    print(f"LP solve time:   {a.lp_solve_seconds:.2f}s")
    print(f"T (intervals):   {T:,}")
    print(f"variables:       {6*T:,}  (6T)")
    print()
    print(f"Battery: {cap} kWh, {a.battery_spec.power_kw} kW, "
          f"reserve {a.battery_spec.backup_reserve_pct}%, "
          f"eff {a.battery_spec.charge_efficiency}/{a.battery_spec.discharge_efficiency}, "
          f"c_deg {a.battery_spec.c_deg_per_kwh}")
    print(f"Export limit: {a.export_limit_kw} kW "
          f"({a.export_limit_kw * a.interval_hours:.4f} kWh/interval)")
    print(f"Network: {a.network_tariff.name} | Retail: {a.retail_tariff.name} "
          f"({a.retail_tariff.type})")
    print(f"SOC: init={a.soc_init_kwh:.4f} (floor={min_soc:.4f}), "
          f"final={a.soc_final_kwh:.4f} (= self_consumption end SOC)")
    print()

    # --- Flow invariants ---
    print("--- Flow invariants (atol 1e-6) ---")
    load_recon = (r.sol_to_home_kwh + r.bat_to_home_kwh + r.grid_to_home_kwh).to_numpy()
    d = np.abs(load_recon - load.to_numpy()).max()
    print(f"Load balance   max|recon-load|:   {d:.3e}  {status(d < tol)}")

    solar_recon = (r.sol_to_home_kwh + r.sol_to_bat_kwh + r.sol_to_grid_kwh
                   + a.curtail).to_numpy()
    d = np.abs(solar_recon - solar.to_numpy()).max()
    print(f"Solar balance  max|recon+curtail-solar|: {d:.3e}  {status(d < tol)}")

    d = np.abs((r.grid_to_home_kwh + r.grid_to_bat_kwh - r.import_kwh).to_numpy()).max()
    print(f"Import recon   max|g2h+g2b-import|: {d:.3e}  {status(d < tol)}")

    d = np.abs((r.sol_to_grid_kwh + r.bat_to_grid_kwh - r.export_kwh).to_numpy()).max()
    print(f"Export recon   max|s2g+b2g-export|: {d:.3e}  {status(d < tol)}")

    neg_ok = True
    for fld in ('sol_to_bat_kwh', 'sol_to_grid_kwh', 'bat_to_home_kwh',
                'bat_to_grid_kwh', 'grid_to_home_kwh', 'grid_to_bat_kwh'):
        mn = getattr(r, fld).min()
        if mn < -tol:
            neg_ok = False
            print(f"  negative flow in {fld}: min={mn:.3e}")
    print(f"Non-negativity (flows except sol_to_home): {status(neg_ok)}")
    print()

    # --- LP-specific checks ---
    print("--- LP constraints ---")
    soc = r.battery_soc_kwh.to_numpy()
    soc_ok = soc.min() >= min_soc - tol and soc.max() <= cap + tol
    print(f"SOC bounds     min={soc.min():.4f} max={soc.max():.4f} "
          f"[{min_soc:.4f},{cap:.4f}]: {status(soc_ok)}")

    d = abs(float(r.battery_soc_kwh.iloc[-1]) - a.soc_final_kwh)
    print(f"Terminal SOC   |soc[-1]-soc_final|: {d:.3e}  {status(d < tol)}")

    exp_lim = a.export_limit_kw * a.interval_hours
    max_exp = float(r.export_kwh.max())
    print(f"Export limit   max export={max_exp:.4f} <= {exp_lim:.4f}: "
          f"{status(max_exp <= exp_lim + tol)}")

    charge = (r.sol_to_bat_kwh + r.grid_to_bat_kwh).to_numpy()
    discharge = (r.bat_to_home_kwh + r.bat_to_grid_kwh).to_numpy()
    n_simul_cd = int(((charge > tol) & (discharge > tol)).sum())
    n_simul_ie = int(((r.import_kwh.to_numpy() > tol) & (r.export_kwh.to_numpy() > tol)).sum())
    # Degenerate ties are not correctness failures; report as INFO, not FAIL.
    print(f"Simultaneous charge+discharge intervals: {n_simul_cd} (INFO)")
    print(f"Simultaneous import+export intervals:    {n_simul_ie} (INFO)")

    pwr = a.battery_spec.power_kw * a.interval_hours
    pwr_ok = charge.max() <= pwr + tol and discharge.max() <= pwr + tol
    print(f"Power limit    max charge={charge.max():.4f} "
          f"max discharge={discharge.max():.4f} <= {pwr:.4f}: {status(pwr_ok)}")
    print()

    # --- Optimisation sanity / VCR direction ---
    print("--- Optimisation sanity ---")
    sc_total = a.sc_bill.household_total
    lp_total = a.lp_bill.household_total
    le_ok = lp_total <= sc_total + 1e-4
    print(f"self_consumption household bill: ${sc_total:,.2f}")
    print(f"perfect_foresight household bill: ${lp_total:,.2f}")
    print(f"LP <= self_consumption: {status(le_ok)}  "
          f"(saving ${sc_total - lp_total:,.2f})")

    # LP objective should reconcile with the dispatch-dependent part of its bill.
    lp_obj = float(np.sum(r.import_kwh.to_numpy() * a.p_imp.to_numpy()
                          - r.export_kwh.to_numpy() * a.p_exp.to_numpy()))
    if a.retail_tariff.type == 'wholesale_exposed':
        daily = a.lp_bill.network_daily_charges + a.lp_bill.retail_daily_charges
    else:
        daily = a.lp_bill.retail_daily_charges
    bill_dispatch = lp_total - daily
    d = abs(lp_obj - bill_dispatch)
    print(f"LP objective vs bill dispatch part: obj=${lp_obj:,.4f} "
          f"bill=${bill_dispatch:,.4f} |diff|={d:.3e}  {status(d < 1e-3)}")
    print()

    # --- Annual totals ---
    print("--- Annual totals (kWh) ---")
    print(f"Solar generated:    {solar.sum():>11.1f}")
    print(f"Gross load:         {load.sum():>11.1f}")
    print(f"Solar -> home:      {r.sol_to_home_kwh.sum():>11.1f}")
    print(f"Solar -> battery:   {r.sol_to_bat_kwh.sum():>11.1f}")
    print(f"Solar -> grid:      {r.sol_to_grid_kwh.sum():>11.1f}")
    print(f"Solar curtailed:    {a.curtail.sum():>11.1f}  "
          f"({a.curtail.sum() / solar.sum() * 100 if solar.sum() else 0:.2f}% of solar)")
    print(f"Grid -> battery:    {r.grid_to_bat_kwh.sum():>11.1f}")
    print(f"Battery -> home:    {r.bat_to_home_kwh.sum():>11.1f}")
    print(f"Battery -> grid:    {r.bat_to_grid_kwh.sum():>11.1f}")
    print(f"Grid -> home:       {r.grid_to_home_kwh.sum():>11.1f}")
    print(f"Total import:       {r.import_kwh.sum():>11.1f}")
    print(f"Total export:       {r.export_kwh.sum():>11.1f}")
    self_suff = ((r.sol_to_home_kwh.sum() + r.bat_to_home_kwh.sum())
                 / load.sum() * 100 if load.sum() else 0)
    print(f"Self-sufficiency:   {self_suff:>10.1f}%")
    print()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _day_slice(analysis: LPAnalysis, day: pd.Timestamp):
    """Return (t, dict-of-kW-arrays, soc, p_imp_c, p_exp_c, day_start, day_end)."""
    a = analysis
    day_start = day.normalize()
    day_end = day_start + pd.Timedelta(days=1)
    idx = a.trace.index
    mask = (idx >= day_start) & (idx < day_end)
    t = idx[mask]
    k = 1.0 / a.interval_hours  # kWh/interval -> kW

    r = a.lp_result
    d = dict(
        solar=a.trace.loc[mask, 'pv_site_net_kwh'].to_numpy() * k,
        load=a.trace.loc[mask, 'gross_load_kwh'].to_numpy() * k,
        s2h=r.sol_to_home_kwh.loc[mask].to_numpy() * k,
        s2b=r.sol_to_bat_kwh.loc[mask].to_numpy() * k,
        s2g=r.sol_to_grid_kwh.loc[mask].to_numpy() * k,
        b2h=r.bat_to_home_kwh.loc[mask].to_numpy() * k,
        b2g=r.bat_to_grid_kwh.loc[mask].to_numpy() * k,
        g2h=r.grid_to_home_kwh.loc[mask].to_numpy() * k,
        g2b=r.grid_to_bat_kwh.loc[mask].to_numpy() * k,
        curtail=a.curtail.loc[mask].to_numpy() * k,
    )
    soc = r.battery_soc_kwh.loc[mask].to_numpy()  # kWh
    p_imp_c = a.p_imp.loc[mask].to_numpy() * _C_PER_DOLLAR  # c/kWh
    p_exp_c = a.p_exp.loc[mask].to_numpy() * _C_PER_DOLLAR
    return t, d, soc, p_imp_c, p_exp_c, day_start, day_end, mask.sum()


def _reapply_labels(axes, titles, ylabels, xlabel='Hour of day'):
    for ax, title, yl in zip(axes, titles, ylabels):
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_ylabel(yl)
    axes[-1].set_xlabel(xlabel)


def _format_time_axis(ax, t, day_start, day_end):
    tz = t.tz if len(t) else None
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3), tz=tz))
    ax.set_xlim(day_start, day_end)


def _plot_soc_panel(ax, t, soc, d, battery_spec, interval_hours):
    """SOC line on `ax` (left, kWh) + charge/discharge bars on a twin (right, kW).

    Charge flows (s2b, g2b) are plotted as positive bars stacked upward.
    Discharge flows (b2h, b2g) are plotted as negative bars stacked downward,
    so the two directions are visually separated across the zero axis.
    """
    cap = battery_spec.capacity_kwh
    min_soc = (battery_spec.backup_reserve_pct / 100.0) * cap

    ax2 = ax.twinx()
    # SOC line must sit above the bars: raise ax, make its background transparent.
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    if len(t):
        tnum = mdates.date2num(t)
        w = (interval_hours / 24.0) * 0.45

        # Charge bars: positive (stacked upward)
        ax2.bar(tnum, d['s2b'], width=w, align='center',
                color=COLOUR_SOLAR, edgecolor='none', label='charge: sol_to_bat')
        ax2.bar(tnum, d['g2b'], width=w, bottom=d['s2b'], align='center',
                color=COLOUR_GRID, edgecolor='none', label='charge: grid_to_bat')

        # Discharge bars: negative (stacked downward)
        ax2.bar(tnum, -d['b2h'], width=w, align='center',
                color=COLOUR_LOAD, edgecolor='none', label='discharge: bat_to_home')
        ax2.bar(tnum, -d['b2g'], width=w, bottom=-d['b2h'], align='center',
                color=COLOUR_BATTERY, edgecolor='none', label='discharge: bat_to_grid')

        ax2.axhline(0, color='black', linewidth=0.6, alpha=0.4)

    ax2.set_ylabel('Charge (+) / Discharge (−) (kW)')
    pwr = battery_spec.power_kw * 1.1
    ax2.set_ylim(-max(pwr, 0.1), max(pwr, 0.1))

    ax.plot(t, soc, color=COLOUR_BATTERY, linewidth=1.8, label='SOC', zorder=10)
    ax.axhline(cap, ls='--', color='black', lw=0.8, alpha=0.5)
    ax.axhline(min_soc, ls='--', color='red', lw=0.8, alpha=0.5)
    ax.set_ylim(0, cap * 1.1)
    # Combined legend on ax (higher z-order) so it renders above capacity line.
    # facecolor must be set explicitly because ax.patch.set_visible(False) above
    # would otherwise make the legend frame transparent too.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=7, ncol=1,
              framealpha=0.9, borderpad=0.4, facecolor='white').set_zorder(20)


# ---------------------------------------------------------------------------
# FLOWS figure
# ---------------------------------------------------------------------------

def _set_window_title(fig, title):
    mgr = getattr(fig.canvas, 'manager', None)
    if mgr is not None:
        try:
            mgr.set_window_title(title)
        except Exception:
            pass


def _setup_flows_figure(site_id):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 3, 3]})
    _set_window_title(fig, f'{site_id} — flows')
    fig._viz_site_id = site_id
    fig._viz_kind = 'Generation, Load & Battery SOC'
    return fig, axes


def _plot_flows_day(axes, analysis, day):
    for ax in axes:
        ax.clear()
        for tw in [c for c in ax.figure.axes if getattr(c, '_is_twin', False)]:
            pass  # twins are recreated each redraw below
    # Remove stale twin axes (created by _plot_soc_panel on the previous redraw)
    _clear_twins(axes[2])

    t, d, soc, _, _, day_start, day_end, n = _day_slice(analysis, day)
    if n == 0:
        for ax in axes:
            ax.text(0.5, 0.5, f"No data for {day.normalize().date()}",
                    transform=ax.transAxes, ha='center', va='center')
        _finalise(axes[0].figure, day_start, analysis.retail_tariff.name)
        return

    # Panel 1: Solar side (+ curtail at top)
    base = np.zeros_like(d['s2h'])
    l1 = d['s2h']
    l2 = d['s2h'] + d['s2b']
    l3 = l2 + d['s2g']
    l4 = l3 + d['curtail']
    axes[0].fill_between(t, base, l1, step='post', color=COLOUR_SOLAR, label='sol_to_home')
    axes[0].fill_between(t, l1, l2, step='post', color=COLOUR_BATTERY, label='sol_to_bat')
    axes[0].fill_between(t, l2, l3, step='post', color=COLOUR_GRID, label='sol_to_grid')
    axes[0].fill_between(t, l3, l4, step='post', color=COLOUR_CURTAIL, label='curtailed')
    axes[0].step(t, d['solar'], where='post', color=COLOUR_SOLAR, linewidth=1.8,
                 label='gross solar')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Load side
    p1 = d['s2h']
    p2 = d['s2h'] + d['b2h']
    p3 = p2 + d['g2h']
    axes[1].fill_between(t, base, p1, step='post', color=COLOUR_SOLAR, label='sol_to_home')
    axes[1].fill_between(t, p1, p2, step='post', color=COLOUR_BATTERY, label='bat_to_home')
    axes[1].fill_between(t, p2, p3, step='post', color=COLOUR_GRID, label='grid_to_home')
    axes[1].step(t, d['load'], where='post', color=COLOUR_LOAD, linewidth=1.8,
                 label='gross load')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    y_max = max(float(np.max(l4)) if len(l4) else 0,
                float(np.max(p3)) if len(p3) else 0,
                float(np.max(d['solar'])) if len(d['solar']) else 0,
                float(np.max(d['load'])) if len(d['load']) else 0, 0.1)
    y_min = min(0.0, float(np.min(d['solar'])) if len(d['solar']) else 0.0)
    axes[0].set_ylim(y_min * 1.05 if y_min < 0 else 0, y_max * 1.05)
    axes[1].set_ylim(0, y_max * 1.05)

    # Panel 3: SOC + charge/discharge
    _plot_soc_panel(axes[2], t, soc, d, analysis.battery_spec, analysis.interval_hours)
    axes[2].grid(True, alpha=0.3)
    _format_time_axis(axes[2], t, day_start, day_end)

    _reapply_labels(
        axes,
        ['Solar Generation', 'Serviced Load', 'Battery SOC + Charge/Discharge'],
        ['Power (kW)', 'Power (kW)', 'SOC (kWh)'],
    )
    _finalise(axes[0].figure, day_start, analysis.retail_tariff.name)


# ---------------------------------------------------------------------------
# Price axis scaling
# ---------------------------------------------------------------------------
#
# Wholesale-exposed prices span roughly -100 c/kWh to several thousand c/kWh,
# and the interesting behaviour is at BOTH ends: the sign of a small negative
# export price decides whether exporting is a cost or a credit, while a
# 2,000 c/kWh spike is what dominates the day's bill. A linear axis renders the
# small values as a flat line against the spike; a pure log axis cannot show
# zero or negatives at all.
#
# symlog solves this: linear within +/- _PRICE_LINTHRESH_C of zero, logarithmic
# outside it. Two things the bare scale does not give us:
#
#  1. The transition point has to be visible. Explicit horizontal rules at 0
#     and at the threshold were tried and removed: they added more clutter than
#     they resolved. The changing TICK SPACING carries it instead, which is
#     quieter and always present. The linear region is therefore labelled on a
#     uniform 10 c/kWh ladder rather than at a few round stops: evenly spaced
#     labels give way to widening ones at exactly +/- _PRICE_LINTHRESH_C, so
#     the eye can locate the boundary without being told where it is. The
#     ladder is affordable because the linear region is the LEAST crowded part
#     of the axis -- the tightest label pair on any day shape is in the log
#     section, not this one.
#  2. Limits are SNAPPED to a small standard set rather than fitted to each
#     day. Free autoscale means the eye recalibrates on every step of the day
#     browser, which defeats browsing; fixed limits waste the pane on calm
#     days. Snapping means the axis only ever takes a few recognisable forms.
#
# Note the snap stops are in RETAIL c/kWh, not wholesale: these prices carry
# network and retail layers and loss factors on top of the wholesale rate, so
# they legitimately exceed the wholesale market cap. The top stop is therefore
# fitted to the data rather than hardcoded to a market ceiling.
#
# The threshold sits just above the top of the normal household retail band
# (roughly 30-60 c/kWh for TOU import) so that ordinary tariff structure stays
# LINEAR and only genuine spikes go logarithmic. Pushing it lower would put a
# peak-vs-shoulder import step into the log region, where it compresses to a
# fraction of a decade and stops being readable.
#
# The two sides snap against DIFFERENT stop sets, because they do different
# jobs. Above zero the axis has to reach a spike. Below zero it only has to
# clear the axis floor: real export prices sit a few cents under zero, not
# tens, so a floor matched to the threshold would spend a large share of the
# pane rendering an empty band. The -10 floor keeps the price lines off the
# floor -- which is what separates them from the flow columns -- while leaving
# the height for the 3-to-8 c/kWh range where the export signal actually
# lives. It still extends on the rare days that go deeper.

_PRICE_LINTHRESH_C = 50.0
_PRICE_HI_STOPS_C = (50.0, 100.0, 200.0, 1000.0, 2000.0)
_PRICE_LO_STOPS_C = (10.0, 50.0, 200.0, 1000.0)   # magnitudes; -10 is the floor
_PRICE_TICK_CANDIDATES_C = (
    -5000.0, -2000.0, -1000.0, -500.0, -200.0, -100.0,
    -50.0, -40.0, -30.0, -20.0, -10.0,
    0.0, 10.0, 20.0, 30.0, 40.0, 50.0,
    100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0,
)


def _price_axis_limits(*price_arrays):
    """Snapped (lo, hi) price-axis limits in c/kWh.

    Uses the smallest standard stop that contains the day's data, falling
    back to a data-fitted limit beyond the largest stop (the "ceiling" case).
    The two sides snap against separate stop sets: see the note above on why
    the floor is much tighter than the threshold.
    """
    # Filter for emptiness AFTER masking, not before: an array that is
    # non-empty but entirely non-finite masks down to size zero, and testing
    # v.size on the pre-mask array would leave `vals` as a list of empty
    # arrays -- truthy, so the guard below falls through and np.max raises on
    # a zero-size reduction.
    vals = [np.asarray(p, dtype=float).ravel() for p in price_arrays]
    vals = [v[np.isfinite(v)] for v in vals]
    vals = [v for v in vals if v.size]
    if not vals:
        return -_PRICE_LO_STOPS_C[0], _PRICE_LINTHRESH_C
    allv = np.concatenate(vals)
    vmax = float(np.max(allv))
    vmin = float(np.min(allv))

    hi = next((s for s in _PRICE_HI_STOPS_C if vmax <= s), None)
    if hi is None:
        hi = vmax * 1.05
    lo = next((-s for s in _PRICE_LO_STOPS_C if vmin >= -s), None)
    if lo is None:
        lo = vmin * 1.05
    return lo, hi


def _apply_price_scale(ax, lo, hi):
    """Apply the symlog price scale with snapped limits and ticks.

    No horizontal rules at zero or at the linear/log threshold: the tick
    spacing already shows where the scale changes, and the extra lines read
    as noise against the flow columns.
    """
    ax.set_yscale('symlog', linthresh=_PRICE_LINTHRESH_C, linscale=1.4)
    ax.set_ylim(lo, hi)
    ticks = [v for v in _PRICE_TICK_CANDIDATES_C if lo <= v <= hi]
    if ticks:
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:,.0f}" for v in ticks])


# ---------------------------------------------------------------------------
# PRICES figure
# ---------------------------------------------------------------------------

def _setup_prices_figure(site_id):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 3, 3]})
    _set_window_title(fig, f'{site_id} — prices')
    fig._viz_site_id = site_id
    fig._viz_kind = 'Exports, Imports & Battery SOC'
    return fig, axes


def _plot_prices_day(axes, analysis, day):
    for ax in axes:
        ax.clear()
    _clear_twins(axes[0])
    _clear_twins(axes[1])
    _clear_twins(axes[2])

    t, d, soc, p_imp_c, p_exp_c, day_start, day_end, n = _day_slice(analysis, day)
    if n == 0:
        for ax in axes:
            ax.text(0.5, 0.5, f"No data for {day.normalize().date()}",
                    transform=ax.transAxes, ha='center', va='center')
        _finalise(axes[0].figure, day_start, analysis.retail_tariff.name)
        return

    # Both price axes share one transform and one set of limits, so that a
    # given height means the same c/kWh in both panes.
    p_lo, p_hi = _price_axis_limits(p_exp_c, p_imp_c)

    # Panel 1: Export vs export price
    e1 = d['s2g']
    e2 = d['s2g'] + d['b2g']
    axes[0].fill_between(t, 0, e1, step='post', color=COLOUR_SOLAR, label='sol_to_grid')
    axes[0].fill_between(t, e1, e2, step='post', color=COLOUR_BATTERY, label='bat_to_grid')
    axes[0].step(t, e2, where='post', color=COLOUR_BATTERY, linewidth=1.5,
                 label='total export')
    axes[0].set_ylim(0, max(float(np.max(e2)) if len(e2) else 0, 0.1) * 1.1)
    axes[0].grid(True, alpha=0.3)
    axp0 = axes[0].twinx()
    axp0.step(t, p_exp_c, where='post', color=COLOUR_GRID, linestyle=':',
              linewidth=1.6, label='export price')
    _apply_price_scale(axp0, p_lo, p_hi)
    axp0.set_ylabel('Export price (c/kWh)')
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axp0.get_legend_handles_labels()
    axp0.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

    # Panel 2: Import vs import price
    i1 = d['g2h']
    i2 = d['g2h'] + d['g2b']
    axes[1].fill_between(t, 0, i1, step='post', color=COLOUR_LOAD, label='grid_to_home')
    axes[1].fill_between(t, i1, i2, step='post', color=COLOUR_BATTERY, label='grid_to_bat')
    axes[1].step(t, i2, where='post', color=COLOUR_GRID, linewidth=1.5,
                 label='total import')
    axes[1].set_ylim(0, max(float(np.max(i2)) if len(i2) else 0, 0.1) * 1.1)
    axes[1].grid(True, alpha=0.3)
    axp1 = axes[1].twinx()
    axp1.step(t, p_imp_c, where='post', color=COLOUR_GRID, linestyle=':',
              linewidth=1.6, label='import price')
    _apply_price_scale(axp1, p_lo, p_hi)
    axp1.set_ylabel('Import price (c/kWh)')
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = axp1.get_legend_handles_labels()
    axp1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

    # Panel 3: SOC + charge/discharge (shared design)
    _plot_soc_panel(axes[2], t, soc, d, analysis.battery_spec, analysis.interval_hours)
    axes[2].grid(True, alpha=0.3)
    _format_time_axis(axes[2], t, day_start, day_end)

    _reapply_labels(
        axes,
        ['Export vs Export Price', 'Import vs Import Price',
         'Battery SOC + Charge/Discharge'],
        ['Export (kW)', 'Import (kW)', 'SOC (kWh)'],
    )
    _finalise(axes[0].figure, day_start, analysis.retail_tariff.name)


# ---------------------------------------------------------------------------
# Shared figure utilities
# ---------------------------------------------------------------------------

def _clear_twins(ax):
    """Remove any twin axes previously attached to `ax` (they accumulate on redraw)."""
    fig = ax.figure
    for other in list(fig.axes):
        if other is ax:
            continue
        # A twin shares the same subplot position (bbox) as ax.
        if other.get_position().bounds == ax.get_position().bounds:
            other.remove()


def _finalise(fig, day_start, suffix=''):
    kind = getattr(fig, '_viz_kind', '')
    site = getattr(fig, '_viz_site_id', '')
    fig.suptitle(f"Site {site}  {day_start.date()}  [{kind}]  {suffix} - {BATTERY_MODE}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def _interactive_loop(analysis: LPAnalysis) -> None:
    a = analysis

    fig_flows, axes_flows = _setup_flows_figure(a.site_id)
    fig_prices, axes_prices = _setup_prices_figure(a.site_id)
    plt.ion()

    first_day = a.trace.index[0].normalize()
    last_day = a.trace.index[-1].normalize()
    current = first_day

    _plot_flows_day(axes_flows, a, current)
    _plot_prices_day(axes_prices, a, current)
    plt.show()
    plt.pause(0.1)

    print(f"Days: {first_day.date()} to {last_day.date()}")
    print("Controls: Enter=next, b=back, YYYY-MM-DD=jump, q=quit "
          "(two windows update together)")

    while True:
        try:
            raw = input(f"[{current.date()}] > ").strip().lower()
        except EOFError:
            break
        if raw == 'q':
            break
        elif raw == 'b':
            current = max(first_day, current - pd.Timedelta(days=1))
        elif raw in ('', 'n'):
            current = min(last_day, current + pd.Timedelta(days=1))
        else:
            try:
                current = pd.Timestamp(raw).tz_localize(a.trace.index.tz).normalize()
                if current < first_day or current > last_day:
                    print(f"(out of range {first_day.date()}..{last_day.date()})")
                    current = max(first_day, min(last_day, current))
            except Exception as e:
                print(f"  (unrecognised input: {e})")
                continue

        _plot_flows_day(axes_flows, a, current)
        _plot_prices_day(axes_prices, a, current)
        fig_flows.canvas.draw_idle()
        fig_prices.canvas.draw_idle()
        plt.pause(0.05)

    plt.close(fig_flows)
    plt.close(fig_prices)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('site_id', help='Site identifier, e.g. S0008')
    parser.add_argument('year', type=int, help='Year, e.g. 2022')
    parser.add_argument('--state', default='NSW', help='NEM region for prices (default NSW)')
    parser.add_argument('--network-tariff', default='NSW_ausgrid_ea025_ea029_2026')
    parser.add_argument('--retail-tariff', default='NSW_amber_wholesale_2026')
    parser.add_argument('--export-limit-kw', type=float, default=10.0,
                        help='Per-phase export limit kW, 1 phase assumed (default 10)')
    parser.add_argument('--project-root', type=Path, default=None)
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    trace, prices_mwh, network_tariff, retail_tariff = load_inputs(
        args.site_id, args.year, args.state,
        args.network_tariff, args.retail_tariff, paths,
    )
    for col in ('pv_site_net_kwh', 'gross_load_kwh'):
        if col not in trace.columns:
            print(f"ERROR: trace missing {col}", file=sys.stderr)
            return 1

    analysis = run_lp_analysis(
        args.site_id, args.year, trace, prices_mwh,
        network_tariff, retail_tariff, DEV_BATTERY,
        export_limit_kw=args.export_limit_kw,
        state=args.state,
    )
    _print_verification(analysis)
    _interactive_loop(analysis)
    return 0


if __name__ == '__main__':
    sys.exit(main())