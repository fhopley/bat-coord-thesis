"""Interactive visualisation of self_consumption output for a single real site.

Usage (from project root):
    python3 -m model.dev.viz_self_consumption SITE_ID YEAR
    python3 -m model.dev.viz_self_consumption SITE_ID YEAR --no-battery

Loads the clean EDP trace and NEM price file for one site/year, runs
self_consumption with a hardcoded battery spec (or no battery via flag),
and opens an interactive 3-panel daily window.

Controls:
    Enter           advance one day
    b               go back one day
    <YYYY-MM-DD>    jump to a specific date
    q               quit

Panels (top to bottom):
  1. Solar side: stacked bars of sol_to_home (yellow), sol_to_bat (green),
     sol_to_grid (grey). Solar trace as a thin yellow line on top.
  2. Load side: stacked bars of sol_to_home (yellow), bat_to_home (green),
     grid_to_home (grey). Gross load trace as a thin blue line on top.
  3. Battery state of charge as a green line, with dashed lines at
     capacity (top) and backup reserve floor (bottom).

At startup prints verification stats:
  - cross-check: import_kwh - export_kwh vs ac_load_net_kwh
  - load and solar balance invariants
  - SOC bound check
  - total energy in / out / through battery
  - self-sufficiency percentage
"""
import argparse
import sys
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
from model.components.billing import compute_bill
from model.dev._dev_common import DEV_BATTERY, load_site_trace as _load_site_trace


# Colour palette per Fred's preferences
COLOUR_SOLAR = '#F5A623'    # yellow/orange
COLOUR_BATTERY = '#5DA13E'  # green
COLOUR_GRID = '#888888'     # grey
COLOUR_LOAD = '#1F77B4'     # blue

_C_PER_DOLLAR = 100.0       # $/kWh -> c/kWh for price axes

# Battery coordination mode label shown in chart titles after the retail tariff
# name.  Edit this string to describe the strategy being visualised.
BATTERY_MODE = 'Self Consumption'


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def _load_prices_mwh(state: str, year: int, paths: Paths) -> pd.Series:
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
    """Load site trace, NEM prices, and tariffs."""
    trace = _load_site_trace(site_id, year, paths)
    prices_mwh = _load_prices_mwh(state, year, paths)
    prices_mwh = prices_mwh.reindex(trace.index)
    network_tariff = load_network_tariff(network_tariff_id, paths.tariffs_dir)
    retail_tariff = load_retail_tariff(retail_tariff_id, paths.tariffs_dir)
    return trace, prices_mwh, network_tariff, retail_tariff


def _print_verification(
    site_id: str,
    year: int,
    trace: pd.DataFrame,
    result,
    battery_present: bool,
    battery_spec: BatterySpec | None,
    sc_bill: SiteBill | None = None,
    retail_tariff: RetailTariff | None = None,
) -> None:
    """Print startup verification stats."""
    solar = trace['pv_site_net_kwh']
    load = trace['gross_load_kwh']
    ac_load_net = trace['ac_load_net_kwh'] if 'ac_load_net_kwh' in trace else None

    print()
    print(f"=== Verification for site {site_id} year {year} ===")
    print(f"Mode: {'with battery' if battery_present else 'NO BATTERY'}")
    if battery_present:
        print(f"Battery: {battery_spec.capacity_kwh} kWh, "
              f"{battery_spec.power_kw} kW, reserve {battery_spec.backup_reserve_pct}%, "
              f"eff {battery_spec.charge_efficiency}/{battery_spec.discharge_efficiency}")
    if retail_tariff is not None:
        print(f"Retail:  {retail_tariff.name} ({retail_tariff.type})")
    print()

    # Cross-check 1: load balance
    load_recon = (result.sol_to_home_kwh + result.bat_to_home_kwh
                  + result.grid_to_home_kwh).values
    load_balance_diff = np.abs(load_recon - load.values).max()
    print(f"Load balance     (max |reconstructed - load|):  "
          f"{load_balance_diff:.6e}  "
          f"{'PASS' if load_balance_diff < 1e-9 else 'FAIL'}")

    # Cross-check 2: solar balance
    solar_recon = (result.sol_to_home_kwh + result.sol_to_bat_kwh
                   + result.sol_to_grid_kwh).values
    solar_balance_diff = np.abs(solar_recon - solar.values).max()
    print(f"Solar balance    (max |reconstructed - solar|): "
          f"{solar_balance_diff:.6e}  "
          f"{'PASS' if solar_balance_diff < 1e-9 else 'FAIL'}")

    # Cross-check 3: ac_load_net relationship
    # For no-battery: import - export should equal ac_load_net exactly.
    # With battery: they differ by (bat_to_home - sol_to_bat).
    if ac_load_net is not None:
        net_grid = (result.import_kwh - result.export_kwh).values
        if not battery_present:
            diff = np.abs(net_grid - ac_load_net.values).max()
            print(f"ac_load_net check (max |net_grid - ac_load_net|): "
                  f"{diff:.6e}  "
                  f"{'PASS' if diff < 1e-6 else 'FAIL'}")
        else:
            # net_grid = load - solar + (sol_to_bat - bat_to_home)
            # so net_grid - ac_load_net = sol_to_bat - bat_to_home (approximately,
            # since ac_load_net = load - solar with EDP sign convention)
            expected_offset = (result.sol_to_bat_kwh - result.bat_to_home_kwh).values
            diff = np.abs((net_grid - ac_load_net.values) - expected_offset).max()
            print(f"ac_load_net check (with battery offset): "
                  f"{diff:.6e}  "
                  f"{'PASS' if diff < 1e-6 else 'FAIL'}")
    else:
        print("ac_load_net check: SKIPPED (column not present)")

    # Cross-check 4: SOC bounds
    if battery_present:
        capacity = battery_spec.capacity_kwh
        min_soc = (battery_spec.backup_reserve_pct / 100) * capacity
        soc = result.battery_soc_kwh.values
        ceiling_ok = soc.max() <= capacity + 1e-9
        floor_ok = soc.min() >= min_soc - 1e-9
        print(f"SOC bounds       (min={soc.min():.4f}, max={soc.max():.4f}, "
              f"floor={min_soc}, ceiling={capacity}): "
              f"{'PASS' if ceiling_ok and floor_ok else 'FAIL'}")

    # Cross-check 5: no grid<->battery flow under self_consumption
    g2b_max = result.grid_to_bat_kwh.abs().max()
    b2g_max = result.bat_to_grid_kwh.abs().max()
    print(f"No grid<->battery flow: "
          f"max grid_to_bat={g2b_max:.2e}, max bat_to_grid={b2g_max:.2e}  "
          f"{'PASS' if g2b_max < 1e-9 and b2g_max < 1e-9 else 'FAIL'}")

    # Summary totals
    print()
    print("--- Annual totals (kWh) ---")
    print(f"Solar generated:    {solar.sum():>10.1f}")
    print(f"Gross load:         {load.sum():>10.1f}")
    print(f"Solar -> home:      {result.sol_to_home_kwh.sum():>10.1f}")
    print(f"Solar -> battery:   {result.sol_to_bat_kwh.sum():>10.1f}")
    print(f"Solar -> grid:      {result.sol_to_grid_kwh.sum():>10.1f}")
    print(f"Battery -> home:    {result.bat_to_home_kwh.sum():>10.1f}")
    print(f"Grid -> home:       {result.grid_to_home_kwh.sum():>10.1f}")
    print(f"Total import:       {result.import_kwh.sum():>10.1f}")
    print(f"Total export:       {result.export_kwh.sum():>10.1f}")

    self_sufficiency = (
        (result.sol_to_home_kwh.sum() + result.bat_to_home_kwh.sum())
        / load.sum() * 100
    )
    print(f"Self-sufficiency:   {self_sufficiency:>9.1f}%")

    if battery_present and result.sol_to_bat_kwh.sum() > 0:
        rt_eff = (result.bat_to_home_kwh.sum()
                  / result.sol_to_bat_kwh.sum() * 100)
        print(f"Observed RT eff:    {rt_eff:>9.1f}%  "
              f"(expected ~{battery_spec.charge_efficiency * battery_spec.discharge_efficiency * 100:.1f}%)")

    if sc_bill is not None:
        print()
        print("--- Billing ---")
        print(f"Household total:    ${sc_bill.household_total:>9.2f}")
        print(f"  Daily charges:    ${sc_bill.retail_daily_charges:>9.2f}")
        if sc_bill.network_daily_charges:
            print(f"  Network daily:    ${sc_bill.network_daily_charges:>9.2f}")

    print()


# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------

def _set_window_title(fig, title):
    mgr = getattr(fig.canvas, 'manager', None)
    if mgr is not None:
        try:
            mgr.set_window_title(title)
        except Exception:
            pass


def _clear_twins(ax):
    """Remove any twin axes previously attached to `ax`."""
    fig = ax.figure
    for other in list(fig.axes):
        if other is ax:
            continue
        if other.get_position().bounds == ax.get_position().bounds:
            other.remove()


def _format_time_axis(ax, t, day_start, day_end):
    tz = t.tz if len(t) else None
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3), tz=tz))
    ax.set_xlim(day_start, day_end)


def _finalise(fig, day_start, suffix=''):
    kind = getattr(fig, '_viz_kind', '')
    site = getattr(fig, '_viz_site_id', '')
    fig.suptitle(f"Site {site}  {day_start.date()}  [{kind}]  {suffix} - {BATTERY_MODE}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])


def _plot_soc_panel(ax, t, soc, d, battery_spec, interval_hours):
    """SOC line (left, kWh) + bipolar charge/discharge bars (right, kW).

    Charge flows (s2b, g2b) are plotted as positive bars stacked upward.
    Discharge flows (b2h, b2g) are plotted as negative bars stacked downward.
    Under self-consumption, g2b and b2g are always zero but the structure
    matches the other viz scripts for consistency.
    """
    cap = battery_spec.capacity_kwh
    min_soc = (battery_spec.backup_reserve_pct / 100.0) * cap

    ax2 = ax.twinx()
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
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=7, ncol=1,
              framealpha=0.9, borderpad=0.4, facecolor='white').set_zorder(20)


# ---------------------------------------------------------------------------
# FLOWS figure
# ---------------------------------------------------------------------------

def _setup_flows_figure(site_id: str) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(14, 9), sharex=True,
        gridspec_kw={'height_ratios': [3, 3, 3]},
    )
    _set_window_title(fig, f'{site_id} — flows')
    fig._viz_site_id = site_id
    fig._viz_kind = 'Generation, Load & Battery SOC'
    return fig, axes


def _plot_flows_day(
    axes: list[plt.Axes],
    day: pd.Timestamp,
    trace: pd.DataFrame,
    result,
    battery_spec: BatterySpec | None,
    interval_hours: float,
    retail_tariff_name: str,
) -> None:
    """Replace the contents of the 3 FLOWS panels with one day's data."""
    for ax in axes:
        ax.clear()
    _clear_twins(axes[2])

    day_start = day.normalize()
    day_end = day_start + pd.Timedelta(days=1)
    mask = (trace.index >= day_start) & (trace.index < day_end)

    if mask.sum() == 0:
        for ax in axes:
            ax.text(0.5, 0.5, f"No data for {day_start.date()}",
                    transform=ax.transAxes, ha='center', va='center')
        _finalise(axes[0].figure, day_start, retail_tariff_name)
        return

    t = trace.index[mask]
    k = 1.0 / interval_hours  # kWh/interval -> kW
    solar = trace.loc[mask, 'pv_site_net_kwh'].values * k
    load  = trace.loc[mask, 'gross_load_kwh'].values * k
    s2h = result.sol_to_home_kwh.loc[mask].values * k
    s2b = result.sol_to_bat_kwh.loc[mask].values * k
    s2g = result.sol_to_grid_kwh.loc[mask].values * k
    b2h = result.bat_to_home_kwh.loc[mask].values * k
    b2g = result.bat_to_grid_kwh.loc[mask].values * k
    g2h = result.grid_to_home_kwh.loc[mask].values * k
    g2b = result.grid_to_bat_kwh.loc[mask].values * k
    soc = result.battery_soc_kwh.loc[mask].values

    d = dict(s2h=s2h, s2b=s2b, s2g=s2g, b2h=b2h, b2g=b2g,
             g2h=g2h, g2b=g2b)

    line_lw = 1.8

    # --- Panel 1: Solar side ---
    base = np.zeros_like(s2h)
    axes[0].fill_between(t, base, s2h, step='post',
                         color=COLOUR_SOLAR, label='sol_to_home')
    axes[0].fill_between(t, s2h, s2h + s2b, step='post',
                         color=COLOUR_BATTERY, label='sol_to_bat')
    axes[0].fill_between(t, s2h + s2b, s2h + s2b + s2g, step='post',
                         color=COLOUR_GRID, label='sol_to_grid')
    axes[0].step(t, solar, where='post', color=COLOUR_SOLAR,
                 linewidth=line_lw, label='gross solar')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # --- Panel 2: Load side ---
    axes[1].fill_between(t, base, s2h, step='post',
                         color=COLOUR_SOLAR, label='sol_to_home')
    axes[1].fill_between(t, s2h, s2h + b2h, step='post',
                         color=COLOUR_BATTERY, label='bat_to_home')
    axes[1].fill_between(t, s2h + b2h, s2h + b2h + g2h, step='post',
                         color=COLOUR_GRID, label='grid_to_home')
    axes[1].step(t, load, where='post', color=COLOUR_LOAD,
                 linewidth=line_lw, label='gross load')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Shared y range for panels 1 & 2
    top1 = s2h + s2b + s2g
    top2 = s2h + b2h + g2h
    y_max = max(
        float(top1.max()) if len(top1) else 0,
        float(top2.max()) if len(top2) else 0,
        float(solar.max()) if len(solar) else 0,
        float(load.max()) if len(load) else 0,
        0.1,
    )
    y_min = min(0.0, float(solar.min()) if len(solar) else 0.0)
    axes[0].set_ylim(y_min * 1.05 if y_min < 0 else 0, y_max * 1.05)
    axes[1].set_ylim(0, y_max * 1.05)

    # --- Panel 3: SOC + charge/discharge ---
    if battery_spec is not None:
        _plot_soc_panel(axes[2], t, soc, d, battery_spec, interval_hours)
    else:
        axes[2].plot(t, soc, color=COLOUR_BATTERY, linewidth=line_lw,
                     label='battery SOC')
        axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True, alpha=0.3)
    _format_time_axis(axes[2], t, day_start, day_end)

    titles  = ['Solar Generation', 'Serviced Load', 'Battery SOC + Charge/Discharge']
    ylabels = ['Power (kW)', 'Power (kW)', 'SOC (kWh)']
    for ax, title, yl in zip(axes, titles, ylabels):
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_ylabel(yl)
    axes[-1].set_xlabel('Hour of day')
    _finalise(axes[0].figure, day_start, retail_tariff_name)


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

def _setup_prices_figure(site_id: str) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(14, 9), sharex=True,
        gridspec_kw={'height_ratios': [3, 3, 3]},
    )
    _set_window_title(fig, f'{site_id} — prices')
    fig._viz_site_id = site_id
    fig._viz_kind = 'Exports, Imports & Battery SOC'
    return fig, axes


def _plot_prices_day(
    axes: list[plt.Axes],
    day: pd.Timestamp,
    trace: pd.DataFrame,
    result,
    p_imp: pd.Series,
    p_exp: pd.Series,
    battery_spec: BatterySpec | None,
    interval_hours: float,
    retail_tariff_name: str,
) -> None:
    """Replace the contents of the 3 PRICES panels with one day's data."""
    for ax in axes:
        ax.clear()
    _clear_twins(axes[0])
    _clear_twins(axes[1])
    _clear_twins(axes[2])

    day_start = day.normalize()
    day_end = day_start + pd.Timedelta(days=1)
    mask = (trace.index >= day_start) & (trace.index < day_end)

    if mask.sum() == 0:
        for ax in axes:
            ax.text(0.5, 0.5, f"No data for {day_start.date()}",
                    transform=ax.transAxes, ha='center', va='center')
        _finalise(axes[0].figure, day_start, retail_tariff_name)
        return

    t = trace.index[mask]
    k = 1.0 / interval_hours
    s2g = result.sol_to_grid_kwh.loc[mask].values * k
    b2g = result.bat_to_grid_kwh.loc[mask].values * k
    g2h = result.grid_to_home_kwh.loc[mask].values * k
    g2b = result.grid_to_bat_kwh.loc[mask].values * k
    soc = result.battery_soc_kwh.loc[mask].values
    p_exp_c = p_exp.loc[mask].values * _C_PER_DOLLAR
    p_imp_c = p_imp.loc[mask].values * _C_PER_DOLLAR

    # Both price axes share one transform and one set of limits, so that a
    # given height means the same c/kWh in both panes.
    p_lo, p_hi = _price_axis_limits(p_exp_c, p_imp_c)

    d = dict(
        s2b=result.sol_to_bat_kwh.loc[mask].values * k,
        g2b=g2b,
        b2h=result.bat_to_home_kwh.loc[mask].values * k,
        b2g=b2g,
    )

    # Panel 1: Export vs export price
    e2 = s2g + b2g
    axes[0].fill_between(t, 0, s2g, step='post',
                         color=COLOUR_SOLAR, label='sol_to_grid')
    axes[0].fill_between(t, s2g, e2, step='post',
                         color=COLOUR_BATTERY, label='bat_to_grid')
    axes[0].step(t, e2, where='post', color=COLOUR_BATTERY,
                 linewidth=1.5, label='total export')
    axes[0].set_ylim(0, max(float(e2.max()) if len(e2) else 0, 0.1) * 1.1)
    axes[0].grid(True, alpha=0.3)
    axp0 = axes[0].twinx()
    axp0.step(t, p_exp_c, where='post', color=COLOUR_GRID,
              linestyle=':', linewidth=1.6, label='export price')
    _apply_price_scale(axp0, p_lo, p_hi)
    axp0.set_ylabel('Export price (c/kWh)')
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axp0.get_legend_handles_labels()
    axp0.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

    # Panel 2: Import vs import price
    i2 = g2h + g2b
    axes[1].fill_between(t, 0, g2h, step='post',
                         color=COLOUR_LOAD, label='grid_to_home')
    axes[1].fill_between(t, g2h, i2, step='post',
                         color=COLOUR_BATTERY, label='grid_to_bat')
    axes[1].step(t, i2, where='post', color=COLOUR_GRID,
                 linewidth=1.5, label='total import')
    axes[1].set_ylim(0, max(float(i2.max()) if len(i2) else 0, 0.1) * 1.1)
    axes[1].grid(True, alpha=0.3)
    axp1 = axes[1].twinx()
    axp1.step(t, p_imp_c, where='post', color=COLOUR_GRID,
              linestyle=':', linewidth=1.6, label='import price')
    _apply_price_scale(axp1, p_lo, p_hi)
    axp1.set_ylabel('Import price (c/kWh)')
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = axp1.get_legend_handles_labels()
    axp1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

    # Panel 3: SOC + charge/discharge
    if battery_spec is not None:
        _plot_soc_panel(axes[2], t, soc, d, battery_spec, interval_hours)
    else:
        axes[2].plot(t, soc, color=COLOUR_BATTERY, linewidth=1.8,
                     label='battery SOC')
        axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True, alpha=0.3)
    _format_time_axis(axes[2], t, day_start, day_end)

    titles  = ['Export: solar + battery vs price',
               'Import: load + battery vs price',
               'Battery SOC + Charge/Discharge']
    ylabels = ['Export (kW)', 'Import (kW)', 'SOC (kWh)']
    for ax, title, yl in zip(axes, titles, ylabels):
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_ylabel(yl)
    axes[-1].set_xlabel('Hour of day')
    _finalise(axes[0].figure, day_start, retail_tariff_name)


def _interactive_loop(
    trace: pd.DataFrame,
    result,
    battery_spec: BatterySpec | None,
    p_imp: pd.Series,
    p_exp: pd.Series,
    site_id: str,
    retail_tariff_name: str,
    interval_hours: float,
) -> None:
    """Day-by-day interactive viewer — two figures navigated together."""
    fig_flows,  axes_flows  = _setup_flows_figure(site_id)
    fig_prices, axes_prices = _setup_prices_figure(site_id)
    plt.ion()

    first_day = trace.index[0].normalize()
    last_day  = trace.index[-1].normalize()
    current   = first_day

    _plot_flows_day(axes_flows, current, trace, result,
                    battery_spec, interval_hours, retail_tariff_name)
    _plot_prices_day(axes_prices, current, trace, result,
                     p_imp, p_exp, battery_spec, interval_hours, retail_tariff_name)
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
                current = pd.Timestamp(raw).tz_localize(trace.index.tz).normalize()
                if current < first_day or current > last_day:
                    print(f"(out of range, must be {first_day.date()}..{last_day.date()})")
                    current = max(first_day, min(last_day, current))
            except Exception as e:
                print(f"  (unrecognised input: {e})")
                continue

        _plot_flows_day(axes_flows, current, trace, result,
                        battery_spec, interval_hours, retail_tariff_name)
        _plot_prices_day(axes_prices, current, trace, result,
                         p_imp, p_exp, battery_spec, interval_hours, retail_tariff_name)
        fig_flows.canvas.draw_idle()
        fig_prices.canvas.draw_idle()
        plt.pause(0.05)

    plt.close(fig_flows)
    plt.close(fig_prices)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site_id', help='Site identifier, e.g. S0008')
    parser.add_argument('year', type=int, help='Year, e.g. 2022')
    parser.add_argument('--no-battery', action='store_true',
                        help='Run without a battery (overrides DEV_BATTERY).')
    parser.add_argument('--state', default='NSW',
                        help='NEM region for prices (default NSW)')
    parser.add_argument('--network-tariff', default='ausgrid_ea025_network_2024')
    parser.add_argument('--retail-tariff', default='amber_wholesale_2024')
    parser.add_argument('--project-root', type=Path, default=None,
                        help='Override project root (default: cwd).')
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

    battery_spec = None if args.no_battery else DEV_BATTERY
    battery_present = battery_spec is not None

    result = self_consumption(
        site_id=args.site_id,
        solar_kwh=trace['pv_site_net_kwh'],
        load_kwh=trace['gross_load_kwh'],
        battery_spec=battery_spec,
    )

    p_imp, p_exp = build_price_vectors(
        trace.index, prices_mwh, network_tariff, retail_tariff,
        args.state, interval_hours=5 / 60,
    )

    sc_bill = compute_bill(result, prices_mwh, network_tariff, retail_tariff,
                           args.state, interval_hours=5 / 60)

    _print_verification(
        site_id=args.site_id,
        year=args.year,
        trace=trace,
        result=result,
        battery_present=battery_present,
        battery_spec=battery_spec,
        sc_bill=sc_bill,
        retail_tariff=retail_tariff,
    )

    _interactive_loop(
        trace=trace,
        result=result,
        battery_spec=battery_spec,
        p_imp=p_imp,
        p_exp=p_exp,
        site_id=args.site_id,
        retail_tariff_name=retail_tariff.name,
        interval_hours=5 / 60,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())