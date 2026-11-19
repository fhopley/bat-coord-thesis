"""Interactive visualisation of vpp_sc_event output for a single real site.

Usage (from project root):
    python3 -m model.dev.viz_vpp_sc_event SITE_ID YEAR
    python3 -m model.dev.viz_vpp_sc_event SITE_ID YEAR --state NSW \\
        --retail-tariff agl_battery_rewards_vpp_2025 \\
        --op-tariff agl_operator_wholesale_2025 \\
        --export-limit-kw 10

Loads the clean EDP trace, NEM prices, household retail tariff (which must carry
a VppConfig block), and operator tariff for one site/year. Runs self_consumption
for the SC baseline bill, runs the VPP event classifier, then runs
vpp_sc_event.

Three interactive windows open together, navigated by the same day prompt:

  FLOWS window (3 panels):
    1. Solar generation stacked (sol_to_home, sol_to_bat, sol_to_grid).
       Solar line. Event intervals shaded red.
    2. Serviced load stacked (sol_to_home, bat_to_home, grid_to_home).
       Load line. Event intervals shaded red.
    3. SOC (left axis, kWh) + charge/discharge bars (right axis, kW).
       Event intervals shaded red.

  PRICES window (3 panels):
    1. Export (kW): n_exp (solar) + f_exp (forced, red). Real TOU export
       price (c/kWh right axis, dotted). Op export price (dashed).
    2. Import (kW): n_imp (load blue) + f_imp (forced, red). Real TOU import
       price and op import price (right axis).
    3. SOC + charge/discharge panel (same as FLOWS window).

Controls:
    Enter / n       advance one day
    b               go back one day
    <YYYY-MM-DD>    jump to a specific date
    e               jump to the first event day
    ne              jump to the next event day
    q               quit
"""
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.core.paths import default_paths, Paths
from model.core.scenario import BatterySpec
from model.core.tariff import (
    NetworkTariff, RetailTariff, load_network_tariff, load_retail_tariff,
)
from model.core.prices import (
    build_price_vectors, block_window_mask, charge_window_mask,
)
from model.core.localtime import local_wall_clock
from model.core.site_result import SiteBill, VppTraces
from model.components.self_consumption import self_consumption
from model.components.vpp_sc_event import vpp_sc_event
from model.components.vpp_event_classifier import classify_vpp_events
from model.components.billing import compute_bill
from model.dev._dev_common import (
    DEV_BATTERY, load_site_trace as _load_site_trace, build_display_prices,
)
from model.orchestrate import _compute_min_export_price


# Colour palette (matching existing viz tools)
COLOUR_SOLAR    = '#F5A623'   # yellow/orange
COLOUR_BATTERY  = '#5DA13E'   # green
COLOUR_GRID     = '#888888'   # grey
COLOUR_LOAD     = '#1F77B4'   # blue
COLOUR_FORCED   = '#D0021B'   # red for forced flows
COLOUR_EVENT_BG = '#FFF3F4'   # very light red background for event intervals (lightened)

_C_PER_DOLLAR   = 100.0       # $/kWh -> c/kWh for price axes

# Battery coordination mode label shown in chart titles after the retail tariff
# name.  Edit this string to describe the strategy being visualised.
BATTERY_MODE = 'VPP Events & Self Consumption'


# ---------------------------------------------------------------------------
# Data bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EovAnalysis:
    site_id:        str
    year:           int
    trace:          pd.DataFrame
    sc_result:      object       # SiteTraces from self_consumption
    eov_result:     object       # SiteTraces from vpp_sc_event
    eov_vt:         VppTraces    # Natural/forced split
    sc_bill:        SiteBill
    eov_bill:       SiteBill
    real_p_imp:     pd.Series    # $/kWh real TOU household base rate (no block tiers)
    real_p_exp:     pd.Series    # $/kWh real TOU household base rate (no block tiers)
    disp_p_imp:     pd.Series    # $/kWh as real_p_imp, but marginal block tier rate
    disp_p_exp:     pd.Series    #   substituted inside block windows (for display)
    op_p_imp:       pd.Series    # $/kWh operator
    op_p_exp:       pd.Series    # $/kWh operator
    prices_mwh:     pd.Series    # raw NEM RRP $/MWh
    vpp_event:      np.ndarray   # bool, length T
    battery_spec:   BatterySpec
    network_tariff: NetworkTariff
    retail_tariff:  RetailTariff   # household VPP plan
    op_tariff:      RetailTariff   # operator wholesale plan
    export_limit_kw: float
    state:           str          # the site's actual location (scenario.state)
    soc_init_kwh:   float
    interval_hours: float
    eov_solve_seconds: float
    event_summary:  str          # pre-formatted event summary for printing
    skip_wins:      list         # per-window SC-fallback messages; empty if none
    min_export_price_mwh: float  # resolved threshold $/MWh (0.0 = no restriction)


# ---------------------------------------------------------------------------
# Input loading and analysis
# ---------------------------------------------------------------------------

def _load_prices_mwh(state: str, year: int, paths: Paths) -> pd.Series:
    price_path = paths.nem_prices_dir / f'{year}_nem_prices.parquet'
    if not price_path.exists():
        raise FileNotFoundError(f"NEM price file not found: {price_path}")
    df = pd.read_parquet(price_path)
    rrp_col = f'{state}1_rrp'
    if rrp_col not in df.columns:
        raise ValueError(f"Missing column {rrp_col!r} in {price_path}")
    return pd.Series(
        df[rrp_col].to_numpy(),
        index=pd.to_datetime(df['datetime_aest']).dt.tz_convert('Etc/GMT-10'),
        name=f'{state}_rrp_dollars_per_mwh',
    )


def load_inputs(
    site_id: str,
    year: int,
    state: str,
    network_tariff_id: str,
    retail_tariff_id: str,
    op_tariff_id: str,
    paths: Paths,
) -> tuple[pd.DataFrame, pd.Series, NetworkTariff, RetailTariff, RetailTariff]:
    """Load site trace, NEM prices, and all three tariffs."""
    trace = _load_site_trace(site_id, year, paths)
    prices_mwh = _load_prices_mwh(state, year, paths)
    prices_mwh = prices_mwh.reindex(trace.index)
    network  = load_network_tariff(network_tariff_id, paths.tariffs_dir)
    retail   = load_retail_tariff(retail_tariff_id,   paths.tariffs_dir)
    op_tariff = load_retail_tariff(op_tariff_id,       paths.tariffs_dir)
    return trace, prices_mwh, network, retail, op_tariff


def _build_event_summary(
    vpp_event: np.ndarray,
    prices_mwh: pd.Series,
    interval_hours: float,
    vpp_cfg,
) -> str:
    """Build a multi-line event summary string for the verification printout."""
    idx = prices_mwh.index
    T = len(vpp_event)
    n_event_intervals = int(vpp_event.sum())
    n_event_hours = n_event_intervals * interval_hours

    # Identify contiguous event blocks
    blocks = []
    in_block = False
    for t in range(T):
        if vpp_event[t] and not in_block:
            block_start = t
            in_block = True
        elif not vpp_event[t] and in_block:
            blocks.append((block_start, t - 1))
            in_block = False
    if in_block:
        blocks.append((block_start, T - 1))

    n_blocks = len(blocks)
    lines = [
        f"VPP events: {n_blocks} window(s), "
        f"{n_event_intervals} event intervals "
        f"({n_event_hours:.1f} h, "
        f"{n_event_intervals / T * 100:.1f}% of year)",
        f"  threshold: {vpp_cfg.vpp_threshold_dollars_per_mwh:.0f} $/MWh | "
        f"window: {vpp_cfg.event_window_hours:.0f} h | "
        f"cap: {vpp_cfg.kwh_limit:.0f} kWh ({vpp_cfg.limit_applies_to})",
    ]

    # List first 10 event blocks with date/time, price at spike, and duration
    n_show = min(10, n_blocks)
    if n_blocks:
        lines.append(f"  First {n_show} of {n_blocks} event window(s):")
        for i, (ts, te) in enumerate(blocks[:n_show]):
            start_ts = idx[ts]
            end_ts   = idx[te]
            dur_h    = (te - ts + 1) * interval_hours
            # Peak price within this block
            block_prices = prices_mwh.iloc[ts:te + 1]
            peak_price   = float(block_prices.max())
            peak_ts      = block_prices.idxmax()
            lines.append(
                f"    [{i+1:2d}] {start_ts.strftime('%Y-%m-%d %H:%M')} -- "
                f"{end_ts.strftime('%H:%M')} "
                f"({dur_h:.1f} h)  "
                f"peak {peak_price:,.0f} $/MWh @ {peak_ts.strftime('%H:%M')}"
            )
        if n_blocks > n_show:
            lines.append(f"    ... and {n_blocks - n_show} more")

    return "\n".join(lines)


def run_eov_analysis(
    site_id: str,
    year: int,
    trace: pd.DataFrame,
    prices_mwh: pd.Series,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    op_tariff: RetailTariff,
    battery_spec: BatterySpec,
    export_limit_kw: float,
    state: str,
    interval_hours: float = 5 / 60,
) -> EovAnalysis:
    """Run the full vpp_sc_event analysis pipeline (no disk access).

    Args:
        state: the site's actual location (scenario.state), used to resolve
            local time for every clock-driven tariff feature (TOU periods,
            no-grid-charge windows, VPP event billing). Required -- there is
            no sane default, and passing the wrong state silently bills the
            wrong tariff periods rather than raising, so this parameter is
            deliberately mandatory rather than defaulted to 'NSW' or similar.
    """
    solar = trace['pv_site_net_kwh']
    load  = trace['gross_load_kwh']
    idx   = solar.index

    vpp_cfg = retail_tariff.vpp
    if vpp_cfg is None:
        raise ValueError(
            f"Retail tariff '{retail_tariff.name}' has no VppConfig block. "
            "Use a VPP-enabled retail tariff YAML."
        )

    # Operator price vectors
    op_p_imp, op_p_exp = build_price_vectors(
        idx, prices_mwh, network_tariff, op_tariff, state, interval_hours)

    # Real household TOU prices for display
    real_p_imp, real_p_exp = build_price_vectors(
        idx, prices_mwh, network_tariff, retail_tariff, state, interval_hours)

    # Self-consumption baseline for billing delta
    sc_result = self_consumption(site_id, solar, load, battery_spec, interval_hours)
    sc_bill   = compute_bill(sc_result, prices_mwh, network_tariff, retail_tariff,
                             state, interval_hours)

    soc_min  = (battery_spec.backup_reserve_pct / 100.0) * battery_spec.capacity_kwh
    soc_init = soc_min

    # Event classification
    vpp_event = classify_vpp_events(
        prices_mwh,
        threshold_dollars_per_mwh=vpp_cfg.vpp_threshold_dollars_per_mwh,
        event_window_hours=vpp_cfg.event_window_hours,
        interval_hours=interval_hours,
    )

    event_summary = _build_event_summary(vpp_event, prices_mwh, interval_hours, vpp_cfg)

    _C = 100.0
    export_limit_kwh = export_limit_kw * interval_hours

    p_curtail = (vpp_cfg.curtail_c_per_kwh / _C
                 if vpp_cfg.curtail_c_per_kwh is not None else None)

    # Resolve minimum forced export price ($/MWh -> $/kWh)
    if vpp_cfg.min_export_price_dollars_per_mwh is not None:
        min_export_price_mwh = vpp_cfg.min_export_price_dollars_per_mwh
    else:
        min_export_price_mwh = _compute_min_export_price(
            prices_mwh=prices_mwh,
            site_traces={site_id: trace},
            site_ids=[site_id],
            power_kw=battery_spec.power_kw,
            interval_hours=interval_hours,
            kwh_limit=vpp_cfg.kwh_limit,
            vpp_event=vpp_event,
        )
    min_export_price_per_kwh = min_export_price_mwh / 1000.0

    # No-grid-charge mask: identical construction to _run_vpp_sc_event in orchestrate.py
    no_grid_charge_mask = None
    if vpp_cfg.no_grid_charge_windows:
        no_grid_charge_mask = np.zeros(len(idx), dtype=bool)
        for window in vpp_cfg.no_grid_charge_windows:
            no_grid_charge_mask |= charge_window_mask(idx, window, state, interval_hours)

    t0 = time.perf_counter()
    import warnings as _warnings
    skip_wins: list[str] = []
    with _warnings.catch_warnings(record=True) as _caught:
        _warnings.simplefilter("always")
        eov_result, eov_vt = vpp_sc_event(
            site_id, solar, load,
            op_p_imp, op_p_exp,
            vpp_event=vpp_event,
            battery_spec=battery_spec,
            soc_init_kwh=soc_init,
            vpp_exp_reward=vpp_cfg.exp_reward_c_per_kwh / _C,
            vpp_imp_reward=vpp_cfg.imp_reward_c_per_kwh / _C,
            vpp_kwh_limit=vpp_cfg.kwh_limit,
            vpp_limit_applies_to=vpp_cfg.limit_applies_to,
            export_limit_kwh_per_interval=export_limit_kwh,
            interval_hours=interval_hours,
            p_curtail=p_curtail,
            min_export_price_per_kwh=min_export_price_per_kwh,
            no_grid_charge_mask=no_grid_charge_mask,
            state=state,
        )
    eov_solve_seconds = time.perf_counter() - t0

    # Parse structured warning tokens; re-raise any other warnings normally.
    for w in _caught:
        msg = str(w.message)
        if msg.startswith('EOV_SKIP:'):
            # EOV_SKIP:site_id:win_start:remaining_cap:rhs_total:n_imp_hi_total:status=N
            parts = msg.split(':')
            skip_wins.append(
                f"    t={parts[2]}  remaining_cap={float(parts[3]):.2f} kWh  "
                f"load-solar={float(parts[4]):.2f} kWh  "
                f"n_imp_hi_sum={float(parts[5]):.2f} kWh  "
                f"{parts[6]}"
            )
        else:
            _warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    eov_bill = compute_bill(eov_result, prices_mwh, network_tariff, retail_tariff,
                            state, interval_hours, vpp_traces=eov_vt)

    # Display prices depend on the solved flows (the marginal block tier rate
    # at each interval is a function of that local day's cumulative flow so
    # far), so they can only be built here, not alongside real_p_imp/exp.
    disp_p_imp, disp_p_exp = build_display_prices(
        idx, eov_result, retail_tariff,
        real_p_imp, real_p_exp, prices_mwh, state, interval_hours,
    )

    return EovAnalysis(
        site_id=site_id, year=year, trace=trace,
        sc_result=sc_result, eov_result=eov_result, eov_vt=eov_vt,
        sc_bill=sc_bill, eov_bill=eov_bill,
        real_p_imp=real_p_imp, real_p_exp=real_p_exp,
        disp_p_imp=disp_p_imp, disp_p_exp=disp_p_exp,
        op_p_imp=op_p_imp, op_p_exp=op_p_exp,
        prices_mwh=prices_mwh,
        vpp_event=vpp_event,
        battery_spec=battery_spec,
        network_tariff=network_tariff,
        retail_tariff=retail_tariff, op_tariff=op_tariff,
        export_limit_kw=export_limit_kw,
        state=state,
        soc_init_kwh=soc_init,
        interval_hours=interval_hours,
        eov_solve_seconds=eov_solve_seconds,
        event_summary=event_summary,
        skip_wins=skip_wins,
        min_export_price_mwh=min_export_price_mwh,
    )


# ---------------------------------------------------------------------------
# Verification printout
# ---------------------------------------------------------------------------

def _print_verification(analysis: EovAnalysis) -> None:
    """Print verification stats. Emits 'FAIL' on any check failure."""
    a   = analysis
    r   = a.eov_result
    vt  = a.eov_vt
    cfg = a.retail_tariff.vpp
    solar = a.trace['pv_site_net_kwh']
    load  = a.trace['gross_load_kwh']
    cap   = a.battery_spec.capacity_kwh
    soc_min = (a.battery_spec.backup_reserve_pct / 100.0) * cap
    tol = 1e-6

    def status(ok: bool) -> str:
        return 'PASS' if ok else 'FAIL'

    T = len(r.import_kwh)
    print()
    print(f"=== Verification: site {a.site_id} year {a.year} (vpp_sc_event) ===")
    print(f"--- Solve timing ---")
    print(f"EOV solve time:    {a.eov_solve_seconds:.2f}s")
    print(f"T (intervals):     {T:,}")
    src = 'YAML' if a.retail_tariff.vpp.min_export_price_dollars_per_mwh is not None \
          else 'auto-computed'
    if a.min_export_price_mwh > 0:
        print(f"Min export price:  {a.min_export_price_mwh:.1f} $/MWh  "
              f"({a.min_export_price_mwh/10:.2f} c/kWh)  [{src}]")
    else:
        print(f"Min export price:  none  (f_exp unrestricted)")
    if a.skip_wins:
        print(f"SC fallbacks:      {len(a.skip_wins)} window(s) were infeasible "
              f"and fell back to self-consumption:")
        for line in a.skip_wins:
            print(line)
    print()
    print(f"Battery: {cap} kWh, {a.battery_spec.power_kw} kW, "
          f"reserve {a.battery_spec.backup_reserve_pct}%, "
          f"eff {a.battery_spec.charge_efficiency}/{a.battery_spec.discharge_efficiency}, "
          f"c_deg {a.battery_spec.c_deg_per_kwh}")
    print(f"Export limit: {a.export_limit_kw} kW "
          f"({a.export_limit_kw * a.interval_hours:.4f} kWh/interval)")
    print(f"Network: {a.network_tariff.name}")
    print(f"Retail:  {a.retail_tariff.name} ({a.retail_tariff.type})")
    print(f"SOC: init={a.soc_init_kwh:.4f} (floor={soc_min:.4f}), "
          f"terminal floats (no terminal constraint -- SC runs between events)")
    print()

    # --- VPP event summary ---
    print("--- VPP event classification ---")
    print(a.event_summary)
    print()

    # --- Flow invariants ---
    print("--- SiteTraces flow invariants (atol 1e-6) ---")
    load_recon = (r.sol_to_home_kwh + r.bat_to_home_kwh + r.grid_to_home_kwh).to_numpy()
    d = np.abs(load_recon - load.to_numpy()).max()
    print(f"Load balance       max|recon-load|:         {d:.3e}  {status(d < tol)}")

    solar_recon = (r.sol_to_home_kwh + r.sol_to_bat_kwh + r.sol_to_grid_kwh
                   + r.curtail_kwh).to_numpy()
    d = np.abs(solar_recon - solar.to_numpy()).max()
    print(f"Solar balance      max|recon-solar|:        {d:.3e}  {status(d < tol)}")

    d = np.abs((r.grid_to_home_kwh + r.grid_to_bat_kwh - r.import_kwh).to_numpy()).max()
    print(f"Import recon       max|g2h+g2b-import|:     {d:.3e}  {status(d < tol)}")

    d = np.abs((r.sol_to_grid_kwh + r.bat_to_grid_kwh - r.export_kwh).to_numpy()).max()
    print(f"Export recon       max|s2g+b2g-export|:     {d:.3e}  {status(d < tol)}")

    neg_ok = True
    for fld in ('sol_to_bat_kwh', 'sol_to_grid_kwh', 'bat_to_home_kwh',
                'bat_to_grid_kwh', 'grid_to_home_kwh', 'grid_to_bat_kwh',
                'curtail_kwh'):
        mn = getattr(r, fld).min()
        if mn < -tol:
            neg_ok = False
            print(f"  negative flow in {fld}: min={mn:.3e}")
    print(f"Non-negativity (flows except sol_to_home):  {status(neg_ok)}")
    print()

    # --- VppTraces invariants ---
    print("--- VppTraces split invariants (atol 1e-6) ---")
    d = np.abs((vt.n_imp_kwh + vt.f_imp_kwh - r.import_kwh).to_numpy()).max()
    print(f"n_imp+f_imp == import_kwh  max|diff|: {d:.3e}  {status(d < tol)}")
    d = np.abs((vt.n_exp_kwh + vt.f_exp_kwh - r.export_kwh).to_numpy()).max()
    print(f"n_exp+f_exp == export_kwh  max|diff|: {d:.3e}  {status(d < tol)}")

    n_imp = vt.n_imp_kwh.to_numpy(); f_imp = vt.f_imp_kwh.to_numpy()
    n_exp = vt.n_exp_kwh.to_numpy(); f_exp = vt.f_exp_kwh.to_numpy()
    neg_vt = (n_imp.min() >= -tol and f_imp.min() >= -tol
              and n_exp.min() >= -tol and f_exp.min() >= -tol)
    print(f"VppTraces non-negativity:              {status(neg_vt)}")

    # Forced flows only outside events should be zero
    non_event = ~a.vpp_event
    f_imp_outside = float(f_imp[non_event].max()) if non_event.any() else 0.0
    f_exp_outside = float(f_exp[non_event].max()) if non_event.any() else 0.0
    zero_outside = f_imp_outside < tol and f_exp_outside < tol
    print(f"Forced flows zero outside events:      {status(zero_outside)}"
          f"  (max f_imp={f_imp_outside:.2e}, max f_exp={f_exp_outside:.2e})")
    print()

    # --- LP constraints ---
    print("--- LP constraints ---")
    soc = r.battery_soc_kwh.to_numpy()
    soc_ok = soc.min() >= soc_min - tol and soc.max() <= cap + tol
    print(f"SOC bounds     min={soc.min():.4f} max={soc.max():.4f} "
          f"[{soc_min:.4f},{cap:.4f}]: {status(soc_ok)}")

    exit_soc = float(r.battery_soc_kwh.iloc[-1])
    print(f"Terminal SOC   soc[-1]={exit_soc:.4f} kWh (unconstrained; INFO)")

    exp_lim = a.export_limit_kw * a.interval_hours
    max_exp = float((vt.n_exp_kwh + vt.f_exp_kwh).max())
    print(f"Export limit   max export={max_exp:.4f} <= {exp_lim:.4f}: "
          f"{status(max_exp <= exp_lim + tol)}")

    if cfg.limit_applies_to == 'export_only':
        forced_total = float(f_exp.sum())
        cap_ok = forced_total <= cfg.kwh_limit + tol
        print(f"VPP cap        f_exp total={forced_total:.2f} <= {cfg.kwh_limit:.0f} kWh "
              f"({cfg.limit_applies_to}): {status(cap_ok)}")
    else:
        forced_total = float(f_imp.sum() + f_exp.sum())
        cap_ok = forced_total <= cfg.kwh_limit + tol
        print(f"VPP cap        f_imp+f_exp total={forced_total:.2f} <= {cfg.kwh_limit:.0f} kWh "
              f"({cfg.limit_applies_to}): {status(cap_ok)}")

    charge    = (r.sol_to_bat_kwh + r.grid_to_bat_kwh).to_numpy()
    discharge = (r.bat_to_home_kwh + r.bat_to_grid_kwh).to_numpy()
    simul_cd_mask = (charge > tol) & (discharge > tol)
    # Use the LP-canonical import/export from SiteTraces (= n_imp+f_imp, n_exp+f_exp)
    simul_ie_mask = (r.import_kwh.to_numpy() > tol) & (r.export_kwh.to_numpy() > tol)
    n_simul_cd = int(simul_cd_mask.sum())
    n_simul_ie = int(simul_ie_mask.sum())
    print(f"Simultaneous charge+discharge: {n_simul_cd} intervals (INFO)")
    print(f"Simultaneous import+export:    {n_simul_ie} intervals "
          f"{'(INFO)' if n_simul_ie == 0 else '(UNEXPECTED -- investigate)'}")

    # Detailed diagnostic: show timestamps and position relative to event
    # window boundaries. Hypothesis: simultaneous intervals cluster at the
    # first interval of each event window (SOC handoff from SC to LP).
    if n_simul_cd > 0:
        print()
        print("--- Simultaneous charge+discharge diagnostic ---")
        idx_full = r.battery_soc_kwh.index
        # Build event window boundary set: first interval of each event window
        from model.components.vpp_sc_event import _find_event_windows
        event_arr = a.vpp_event.astype(bool)
        windows   = _find_event_windows(event_arr)
        win_starts = {ws for ws, _ in windows}
        win_ends   = {we - 1 for _, we in windows}  # last interval of each window

        simul_indices = np.where(simul_cd_mask)[0]
        at_start  = sum(1 for i in simul_indices if i in win_starts)
        at_end    = sum(1 for i in simul_indices if i in win_ends)
        in_event  = sum(1 for i in simul_indices if event_arr[i])
        in_sc     = sum(1 for i in simul_indices if not event_arr[i])

        print(f"  Total simultaneous intervals:  {n_simul_cd}")
        print(f"  In event windows:              {in_event}")
        print(f"  In SC (non-event) intervals:   {in_sc}  "
              f"{'(unexpected -- investigate)' if in_sc > 0 else '(OK)'}")
        print(f"  At first interval of window:   {at_start}")
        print(f"  At last interval of window:    {at_end}")
        print(f"  At other event intervals:      {n_simul_cd - at_start - at_end - in_sc}")

        # Show first 10 occurrences with context
        n_show = min(10, n_simul_cd)
        print(f"  First {n_show} of {n_simul_cd} simultaneous intervals:")
        op_imp_arr   = a.op_p_imp.to_numpy() * _C_PER_DOLLAR  # c/kWh for readability
        op_exp_arr   = a.op_p_exp.to_numpy() * _C_PER_DOLLAR
        n_imp_arr    = vt.n_imp_kwh.to_numpy()
        f_imp_arr    = vt.f_imp_kwh.to_numpy()
        n_exp_arr    = vt.n_exp_kwh.to_numpy()
        f_exp_arr    = vt.f_exp_kwh.to_numpy()
        ih           = a.interval_hours
        solar_kw     = a.trace['pv_site_net_kwh'].to_numpy() / ih  # kW
        load_kw      = a.trace['gross_load_kwh'].to_numpy()  / ih  # kW
        for i in simul_indices[:n_show]:
            ts      = idx_full[i]
            pos     = ('win_start' if i in win_starts else
                       'win_end'   if i in win_ends   else
                       'event'     if event_arr[i]    else 'SC')
            soc_val = float(r.battery_soc_kwh.iloc[i])
            chg     = float(charge[i])
            dis     = float(discharge[i])
            net_bat = dis - chg   # positive = net discharge
            print(f"    t={i:6d}  {ts.strftime('%Y-%m-%d %H:%M')}  "
                  f"sol={solar_kw[i]:+.4f}kW  load={load_kw[i]:.4f}kW  "
                  f"net(dis-chg)={net_bat:+.4f}kW  "
                  f"soc={soc_val:.4f}  "
                  f"chg={chg:.4f}  dis={dis:.4f}  "
                  f"op_imp={op_imp_arr[i]:.2f}c  op_exp={op_exp_arr[i]:.2f}c  "
                  f"n_imp={n_imp_arr[i]:.4f}  f_imp={f_imp_arr[i]:.4f}  "
                  f"n_exp={n_exp_arr[i]:.4f}  f_exp={f_exp_arr[i]:.4f}")
    print()

    # --- VPP dispatch summary ---
    print("--- EOV dispatch summary ---")
    print(f"Total f_exp (forced export):   {float(f_exp.sum()):>10.2f} kWh")
    print(f"Total f_imp (forced import):   {float(f_imp.sum()):>10.2f} kWh")
    print(f"Total n_exp (natural export):  {float(n_exp.sum()):>10.2f} kWh")
    print(f"Total n_imp (natural import):  {float(n_imp.sum()):>10.2f} kWh")
    _C = 100.0
    reward_exp = float(f_exp.sum()) * cfg.exp_reward_c_per_kwh / _C
    reward_imp = float(f_imp.sum()) * cfg.imp_reward_c_per_kwh / _C
    print(f"HH VPP reward (exp):           ${reward_exp:>9.2f}")
    print(f"HH VPP reward (imp):           ${reward_imp:>9.2f}")
    print(f"retail_other_credits:          ${a.eov_bill.retail_other_credits:>9.2f}")
    print()

    # --- Pre-event SOC hold analysis ---
    # Quantifies how much the VPP formulation withholds battery discharge from
    # household self-consumption use in non-event intervals to preserve SOC
    # ahead of VPP events.
    #
    # For each non-event interval t, avoidable_import[t] is the grid import
    # that the battery could have covered given the SOC it actually held at
    # the start of that interval, subject to the power cap:
    #
    #   available_ac[t] = min((soc[t-1] - soc_min) * eta_d,
    #                         power_kw * interval_hours)
    #   avoidable_import[t] = min(n_imp[t], max(0, available_ac[t]))
    #
    # This uses the VPP's own SOC trajectory (so post-event SOC depletion is
    # correctly reflected) and asks: given where the battery actually was,
    # could it have served this import? The sum is the total grid import the
    # household paid for when the battery had capacity to cover it but chose
    # not to (because it was holding SOC for a future event).
    print("--- Pre-event SOC hold analysis ---")
    non_event = ~a.vpp_event.astype(bool)
    soc_arr   = r.battery_soc_kwh.to_numpy()
    n_imp_arr = n_imp  # already numpy from earlier in _print_verification

    eta_d       = a.battery_spec.discharge_efficiency
    power_kwh   = a.battery_spec.power_kw * a.interval_hours
    soc_min_val = (a.battery_spec.backup_reserve_pct / 100.0) * a.battery_spec.capacity_kwh

    # soc[t-1]: use soc_init for t=0, soc[t-1] otherwise
    T    = len(soc_arr)
    soc_prev        = np.empty(T)
    soc_prev[0]     = a.soc_init_kwh
    soc_prev[1:]    = soc_arr[:-1]

    available_ac    = np.minimum(
        np.maximum(0.0, (soc_prev - soc_min_val) * eta_d),
        power_kwh,
    )
    avoidable       = np.minimum(n_imp_arr, available_ac)
    avoidable_non_event = avoidable[non_event]
    avoidable_total = float(avoidable_non_event.sum())

    non_event_import_total = float(n_imp_arr[non_event].sum())
    hold_fraction = (avoidable_total / non_event_import_total * 100
                     if non_event_import_total > 0 else 0.0)
    n_held_intervals = int((avoidable_non_event > 0.001).sum())

    print(f"Non-event intervals:               {non_event.sum():>10,}")
    print(f"Non-event grid import:             {non_event_import_total:>10.2f} kWh")
    print(f"Avoidable import (SOC held):       {avoidable_total:>10.2f} kWh"
          f"  ({hold_fraction:.1f}% of non-event import)")
    print(f"  Intervals with avoidable import: {n_held_intervals:>10,}")
    print(f"  (import that could have been avoided by discharging held SOC)")
    print()

    # --- Block export tiers ---
    # Whether the daily tier threshold is actually crossed. This is the same
    # question the export price panel answers visually; printing it means a
    # flat 8c line can be attributed to the data rather than to the axis.
    # Day grouping is local (local_wall_clock), matching billing's
    # _block_tier_components and _dev_common.build_display_prices -- an AEST
    # grouping here would disagree with the chart on DST days.
    rt = a.retail_tariff
    if rt.block_export_tiers:
        print("--- Block export tiers ---")
        idx_all = r.export_kwh.index
        lwc = local_wall_clock(idx_all, a.state, a.interval_hours)
        for w in rt.block_export_tiers:
            mask = block_window_mask(idx_all, w, a.state, a.interval_hours)
            if not mask.any():
                print(f"  [{w.start_hour:02d},{w.end_hour:02d}): no window intervals")
                continue
            daily = (pd.Series(r.export_kwh.to_numpy()[mask])
                     .groupby(lwc.date[mask]).sum())
            t1_cap = w.tiers[0].threshold_kwh if w.tiers else 0.0
            in_t1  = daily.clip(upper=t1_cap)
            in_t2  = (daily - t1_cap).clip(lower=0)
            spill  = in_t2 > 1e-4
            print(f"  [{w.start_hour:02d},{w.end_hour:02d})  "
                  f"daily export avg={daily.mean():.2f} kWh  "
                  f"tier1 avg={in_t1.mean():.2f}/{t1_cap:.1f} kWh  "
                  f"tier2 on {int(spill.sum())}/{len(daily)} days "
                  f"(avg {in_t2[spill].mean() if spill.any() else 0.0:.2f} kWh)")
            if spill.any():
                days = [str(pd.Timestamp(d).date()) for d in in_t2[spill].index[:5]]
                print(f"    first tier2 days: {', '.join(days)}")
        print()

    # --- Billing breakdown ---
    # Show all SiteBill components side by side for SC and EOV.
    # Credits are stored as negative numbers in SiteBill (sign convention).
    # Rows marked (*) only appear for wholesale-exposed plans; they are zero
    # for TOU plans and omitted if both SC and EOV are zero on that row.
    sc  = a.sc_bill
    eov = a.eov_bill
    pt  = sc.plan_type  # 'tou' or 'wholesale_exposed'

    def _brow(label, sc_val, eov_val, indent=0):
        pad = '  ' * indent
        delta = eov_val - sc_val
        sign  = '+' if delta >= 0 else ''
        print(f"  {pad}{label:<38}  {sc_val:>10.2f}   {eov_val:>10.2f}   {sign}{delta:.2f}")

    print("--- Billing breakdown ---")
    print(f"  {'':38}  {'SC':>10}   {'EOV':>10}   {'Delta':>8}")
    print(f"  {'':-<38}  {'':-<10}   {'':-<10}   {'':-<8}")

    if pt == 'wholesale_exposed':
        _brow("Wholesale import cost", sc.wholesale_usage_cost,   eov.wholesale_usage_cost)
        _brow("Wholesale export credit", sc.wholesale_feedin_credit, eov.wholesale_feedin_credit)
        _brow("Network import charge",   sc.network_usage_cost,    eov.network_usage_cost)
        _brow("Network export credit",   sc.network_feedin_credit, eov.network_feedin_credit)
        _brow("Network daily charges",   sc.network_daily_charges, eov.network_daily_charges)

    _brow("Retail import (usage)",   sc.retail_usage_cost,     eov.retail_usage_cost)
    _brow("Retail export credit",    sc.retail_feedin_credit,  eov.retail_feedin_credit)
    _brow("Retail daily charges",    sc.retail_daily_charges,  eov.retail_daily_charges)

    # retail_other_credits: zero-import reward, VPP rewards, block tier
    # adjustments, etc. Only show if non-zero on either bill.
    if abs(sc.retail_other_credits) > 0.005 or abs(eov.retail_other_credits) > 0.005:
        _brow("Other credits (rewards/tiers)", sc.retail_other_credits, eov.retail_other_credits)

    print(f"  {'':-<38}  {'':-<10}   {'':-<10}   {'':-<8}")
    _brow("Household total",  sc.household_total, eov.household_total)
    saving = sc.household_total - eov.household_total
    sign   = '+' if saving >= 0 else ''
    pct    = saving / sc.household_total * 100 if sc.household_total else 0.0
    print(f"  {'Saving vs SC':<38}  {'':>10}   {'':>10}   {sign}{saving:.2f}  ({pct:.1f}%)")
    print()
    print(f"  {'Retailer margin':<38}  {sc.retailer_margin:>10.2f}   {eov.retailer_margin:>10.2f}"
          f"   {'+' if eov.retailer_margin >= sc.retailer_margin else ''}"
          f"{eov.retailer_margin - sc.retailer_margin:.2f}")
    print()

    # --- Annual totals ---
    print("--- Annual totals (kWh) ---")
    print(f"Solar generated:    {solar.sum():>11.1f}")
    print(f"Gross load:         {load.sum():>11.1f}")
    print(f"Total import:       {r.import_kwh.sum():>11.1f}"
          f"  (natural {float(n_imp.sum()):>8.1f}, forced {float(f_imp.sum()):>7.1f})")
    print(f"Total export:       {r.export_kwh.sum():>11.1f}"
          f"  (natural {float(n_exp.sum()):>8.1f}, forced {float(f_exp.sum()):>7.1f})")
    print()


# ---------------------------------------------------------------------------
# Event CSV diagnostic
# ---------------------------------------------------------------------------

def write_event_csv(a: 'EovAnalysis', path: 'str | Path') -> None:
    """Write a per-interval CSV for all event-window intervals above the
    min export price threshold, with enough detail to diagnose why the LP
    did or did not discharge.

    Columns:
        timestamp           interval end time (index)
        event_window        event window index (1-based)
        op_price_exp_mwh    operator export price $/MWh
        above_min_price     True if op_price_exp >= min_export_price threshold
        solar_kwh           gross solar generation
        load_kwh            gross household load
        net_load_kwh        load - solar (negative = surplus)
        soc_before_kwh      DC SOC at start of this interval
        max_possible_discharge_kwh  physical discharge ceiling:
                            min(power_kwh, soc_before - soc_floor,
                                export_limit - max(0, solar - load))
        f_exp_kwh           actual forced export from LP
        n_exp_kwh           natural export from LP
        total_export_kwh    f_exp + n_exp
        export_limit_kwh    per-interval export ceiling
        headroom_kwh        export_limit - total_export (unused export capacity)
        limiting_factor     categorised reason for less-than-maximum discharge
    """
    import csv
    from pathlib import Path as _Path

    r       = a.eov_result
    vt      = a.eov_vt
    cfg     = a.retail_tariff.vpp
    bspec   = a.battery_spec
    cap     = bspec.capacity_kwh
    soc_min = (bspec.backup_reserve_pct / 100.0) * cap
    ih      = a.interval_hours
    power_kwh = bspec.power_kw * ih
    export_limit_kwh = a.export_limit_kw * ih
    min_exp_per_kwh  = a.min_export_price_mwh / 1000.0

    idx       = r.import_kwh.index
    solar     = a.trace['pv_site_net_kwh'].to_numpy()
    load      = a.trace['gross_load_kwh'].to_numpy()
    op_exp    = a.op_p_exp.to_numpy()          # $/kWh
    soc       = r.battery_soc_kwh.to_numpy()
    f_exp     = vt.f_exp_kwh.to_numpy()
    n_exp     = vt.n_exp_kwh.to_numpy()
    vpp_event = a.vpp_event

    # Build event window index map (interval -> 1-based window number)
    windows    = _find_event_windows(vpp_event.astype(bool))
    win_of_t   = np.zeros(len(idx), dtype=int)
    for w_idx, (ws, we) in enumerate(windows, start=1):
        win_of_t[ws:we] = w_idx

    rows = []
    for t in range(len(idx)):
        if not vpp_event[t]:
            continue
        op_exp_mwh    = op_exp[t] * 1000.0
        above         = op_exp[t] >= min_exp_per_kwh
        sol           = float(solar[t])
        ld            = float(load[t])
        net_ld        = ld - sol
        soc_before    = float(soc[t - 1]) if t > 0 else (bspec.backup_reserve_pct / 100.0 * cap)
        solar_surplus = max(0.0, sol - ld)  # solar already serving load surplus
        discharge_cap = min(
            power_kwh,
            soc_before - soc_min,
            export_limit_kwh - solar_surplus,
        )
        discharge_cap = max(0.0, discharge_cap)
        tot_exp       = float(f_exp[t] + n_exp[t])
        headroom      = max(0.0, export_limit_kwh - tot_exp)

        # Categorise limiting factor
        f_exp_t = float(f_exp[t])
        if not above:
            factor = 'below_min_price'
        elif soc_before - soc_min < 1e-6:
            factor = 'soc_constrained'
        elif tot_exp >= export_limit_kwh - 1e-6:
            factor = 'export_limit_hit'
        elif solar_surplus >= export_limit_kwh - 1e-6:
            factor = 'solar_surplus_only'
        elif ld >= bspec.power_kw * ih - 1e-6:
            factor = 'load_absorbed_all'
        elif f_exp_t >= discharge_cap - 1e-6:
            factor = 'at_capacity'
        elif f_exp_t > 1e-6:
            factor = 'partial_discharge'
        else:
            factor = 'unexplained_gap'

        rows.append({
            'timestamp':                 idx[t].isoformat(),
            'event_window':              win_of_t[t],
            'op_price_exp_mwh':          round(op_exp_mwh, 2),
            'above_min_price':           above,
            'solar_kwh':                 round(sol, 5),
            'load_kwh':                  round(ld, 5),
            'net_load_kwh':              round(net_ld, 5),
            'soc_before_kwh':            round(soc_before, 5),
            'max_possible_discharge_kwh': round(discharge_cap, 5),
            'f_exp_kwh':                 round(f_exp_t, 5),
            'n_exp_kwh':                 round(float(n_exp[t]), 5),
            'total_export_kwh':          round(tot_exp, 5),
            'export_limit_kwh':          round(export_limit_kwh, 5),
            'headroom_kwh':              round(headroom, 5),
            'limiting_factor':           factor,
        })

    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Event CSV written: {path}  ({len(rows):,} rows, "
          f"{sum(1 for r in rows if r['above_min_price'])} above min price)")


# ---------------------------------------------------------------------------
# Plotting helpers (shared with LP viz)
# ---------------------------------------------------------------------------

def _set_window_title(fig, title):
    mgr = getattr(fig.canvas, 'manager', None)
    if mgr is not None:
        try:
            mgr.set_window_title(title)
        except Exception:
            pass


def _clear_twins(ax):
    fig = ax.figure
    for other in list(fig.axes):
        if other is ax:
            continue
        if other.get_position().bounds == ax.get_position().bounds:
            other.remove()


def _finalise(fig, day_start, suffix=''):
    kind = getattr(fig, '_viz_kind', '')
    site = getattr(fig, '_viz_site_id', '')
    mode = getattr(fig, '_viz_battery_mode', BATTERY_MODE)
    fig.suptitle(f"Site {site}  {day_start.date()}  [{kind}]  {suffix} - {mode}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])


def _format_time_axis(ax, t, day_start, day_end):
    tz = t.tz if len(t) else None
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3), tz=tz))
    ax.set_xlim(day_start, day_end)


# ---------------------------------------------------------------------------
# Price axis scaling
# ---------------------------------------------------------------------------
#
# Wholesale-exposed prices span roughly -100 c/kWh to several thousand c/kWh,
# and the interesting behaviour is at BOTH ends: the sign of a small negative
# export price decides whether exporting is a cost or a credit, while a
# 2,000 c/kWh spike is what calls the event in the first place. A linear axis
# renders the small values as a flat line against the spike; a pure log axis
# cannot show zero or negatives at all.
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
    # non-empty but entirely non-finite masks down to size zero, and the
    # earlier form tested v.size on the pre-mask array. That left `vals` as a
    # list of empty arrays, which is truthy, so the guard below fell through
    # and np.max raised on a zero-size reduction.
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
    as noise against the event shading.
    """
    ax.set_yscale('symlog', linthresh=_PRICE_LINTHRESH_C, linscale=1.4)
    ax.set_ylim(lo, hi)
    ticks = [v for v in _PRICE_TICK_CANDIDATES_C if lo <= v <= hi]
    if ticks:
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:,.0f}" for v in ticks])


def _shade_event_intervals(ax, t, vpp_event_day: np.ndarray, day_start, day_end):
    """Shade event intervals with a light red background.

    Contiguous event intervals are merged into one span. Drawing a separate
    axvspan per 5-minute interval created hundreds of patches per panel, and
    every patch was edge-stroked -- axvspan applies ``color`` to both face and
    edge, at the default 1pt patch linewidth -- so each interval boundary
    appeared as a faint vertical rule inside what should read as flat shading.
    ``linewidth=0`` suppresses the stroke; merging removes the internal
    boundaries altogether.

    Interval width is derived from len(t) rather than assumed, so a DST
    transition day (with 23 or 25 hours of intervals) still shades correctly.
    """
    ev = np.asarray(vpp_event_day, dtype=bool)
    if not ev.any():
        return
    ih = pd.Timedelta(seconds=(day_end - day_start).total_seconds() / len(t))

    # Run-length encode: a run starts where False -> True and ends where
    # True -> False. Padding with False at both ends makes the first and last
    # intervals fall out of the same diff without special-casing.
    edges  = np.diff(np.concatenate(([False], ev, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends   = np.flatnonzero(edges == -1)   # exclusive positional index into t

    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[e - 1] + ih,
                   color=COLOUR_EVENT_BG, alpha=0.6, linewidth=0, zorder=0)


def _day_slice(analysis: EovAnalysis, day: pd.Timestamp):
    """Return per-day arrays for plotting."""
    a = analysis
    day_start = day.normalize()
    day_end   = day_start + pd.Timedelta(days=1)
    idx  = a.trace.index
    mask = (idx >= day_start) & (idx < day_end)
    t    = idx[mask]
    k    = 1.0 / a.interval_hours  # kWh/interval -> kW

    r  = a.eov_result
    vt = a.eov_vt
    d  = dict(
        solar = a.trace.loc[mask, 'pv_site_net_kwh'].to_numpy() * k,
        load  = a.trace.loc[mask, 'gross_load_kwh'].to_numpy() * k,
        s2h   = r.sol_to_home_kwh.loc[mask].to_numpy() * k,
        s2b   = r.sol_to_bat_kwh.loc[mask].to_numpy() * k,
        s2g   = r.sol_to_grid_kwh.loc[mask].to_numpy() * k,
        b2h   = r.bat_to_home_kwh.loc[mask].to_numpy() * k,
        b2g   = r.bat_to_grid_kwh.loc[mask].to_numpy() * k,
        g2h   = r.grid_to_home_kwh.loc[mask].to_numpy() * k,
        g2b   = r.grid_to_bat_kwh.loc[mask].to_numpy() * k,
        curtail = r.curtail_kwh.loc[mask].to_numpy() * k,
        n_imp = vt.n_imp_kwh.loc[mask].to_numpy() * k,
        f_imp = vt.f_imp_kwh.loc[mask].to_numpy() * k,
        n_exp = vt.n_exp_kwh.loc[mask].to_numpy() * k,
        f_exp = vt.f_exp_kwh.loc[mask].to_numpy() * k,
    )
    soc           = r.battery_soc_kwh.loc[mask].to_numpy()
    # Display series, not real_p_*: inside a block window these carry the
    # marginal tier rate rather than the base feedin/usage rate, which on a
    # tiered plan is the whole of the price signal. Identical to real_p_*
    # everywhere else and on every plan with no block tiers.
    real_p_imp_c  = a.disp_p_imp.loc[mask].to_numpy() * _C_PER_DOLLAR
    real_p_exp_c  = a.disp_p_exp.loc[mask].to_numpy() * _C_PER_DOLLAR
    op_p_imp_c    = a.op_p_imp.loc[mask].to_numpy() * _C_PER_DOLLAR
    op_p_exp_c    = a.op_p_exp.loc[mask].to_numpy() * _C_PER_DOLLAR
    event_day     = a.vpp_event[mask]
    return (t, d, soc, real_p_imp_c, real_p_exp_c, op_p_imp_c, op_p_exp_c,
            event_day, day_start, day_end, int(mask.sum()))


def _plot_soc_panel(ax, t, soc, d, battery_spec, interval_hours, event_day):
    """SOC line (left) + charge/discharge bars (right), event intervals shaded.

    Charge flows (s2b, g2b) are plotted as positive bars.
    Discharge flows (b2h, b2g) are plotted as negative bars so that the two
    directions are visually separated across the zero axis.
    """
    cap     = battery_spec.capacity_kwh
    soc_min = (battery_spec.backup_reserve_pct / 100.0) * cap
    ax2 = ax.twinx()
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    if len(t):
        _shade_event_intervals(ax2, t, event_day,
                               t[0].normalize(),
                               t[0].normalize() + pd.Timedelta(days=1))
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
    ax.axhline(cap,     ls='--', color='black', lw=0.8, alpha=0.5)
    ax.axhline(soc_min, ls='--', color='red',   lw=0.8, alpha=0.5)
    ax.set_ylim(0, cap * 1.1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=7, ncol=1,
              framealpha=0.9, borderpad=0.4, facecolor='white').set_zorder(20)


# ---------------------------------------------------------------------------
# FLOWS figure
# ---------------------------------------------------------------------------

def _setup_flows_figure(site_id, battery_mode=BATTERY_MODE):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 3, 3]})
    _set_window_title(fig, f'{site_id} — EOV flows')
    fig._viz_site_id = site_id; fig._viz_kind = 'Generation, Load & Battery SOC'
    fig._viz_battery_mode = battery_mode
    return fig, axes


def _plot_flows_day(axes, analysis, day):
    for ax in axes:
        ax.clear()
    _clear_twins(axes[2])

    res = _day_slice(analysis, day)
    t, d, soc, _, _, _, _, event_day, day_start, day_end, n = res
    if n == 0:
        for ax in axes:
            ax.text(0.5, 0.5, f"No data for {day.normalize().date()}",
                    transform=ax.transAxes, ha='center', va='center')
        _finalise(axes[0].figure, day_start, analysis.retail_tariff.name)
        return

    for ax in axes[:2]:
        _shade_event_intervals(ax, t, event_day, day_start, day_end)

    # Panel 1: Solar
    base = np.zeros_like(d['s2h'])
    l1 = d['s2h']; l2 = l1 + d['s2b']; l3 = l2 + d['s2g']; l4 = l3 + d['curtail']
    axes[0].fill_between(t, base, l1, step='post', color=COLOUR_SOLAR,
                         linewidth=0, label='sol_to_home')
    axes[0].fill_between(t, l1, l2, step='post', color=COLOUR_BATTERY,
                         linewidth=0, label='sol_to_bat')
    axes[0].fill_between(t, l2, l3, step='post', color=COLOUR_GRID,
                         linewidth=0, label='sol_to_grid')
    axes[0].fill_between(t, l3, l4, step='post', color='#FBD38D',
                         linewidth=0, label='curtailed')
    axes[0].step(t, d['solar'], where='post', color=COLOUR_SOLAR, linewidth=1.8,
                 label='gross solar')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Load
    p1 = d['s2h']; p2 = p1 + d['b2h']; p3 = p2 + d['g2h']
    axes[1].fill_between(t, base, p1, step='post', color=COLOUR_SOLAR,
                         linewidth=0, label='sol_to_home')
    axes[1].fill_between(t, p1, p2, step='post', color=COLOUR_BATTERY,
                         linewidth=0, label='bat_to_home')
    axes[1].fill_between(t, p2, p3, step='post', color=COLOUR_GRID,
                         linewidth=0, label='grid_to_home')
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

    # Panel 3: SOC
    _plot_soc_panel(axes[2], t, soc, d, analysis.battery_spec,
                    analysis.interval_hours, event_day)
    axes[2].grid(True, alpha=0.3)
    _format_time_axis(axes[2], t, day_start, day_end)

    titles = ['Solar Generation', 'Serviced Load', 'Battery SOC + Charge/Discharge']
    ylabels = ['Power (kW)', 'Power (kW)', 'SOC (kWh)']
    for ax, title, yl in zip(axes, titles, ylabels):
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_ylabel(yl)
    axes[-1].set_xlabel('Hour of day')
    _finalise(axes[0].figure, day_start, analysis.retail_tariff.name)


# ---------------------------------------------------------------------------
# PRICES figure
# ---------------------------------------------------------------------------

def _setup_prices_figure(site_id, battery_mode=BATTERY_MODE):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 3, 3]})
    _set_window_title(fig, f'{site_id} — EOV prices + split')
    fig._viz_site_id = site_id; fig._viz_kind = 'Exports, Imports & Battery SOC'
    fig._viz_battery_mode = battery_mode
    return fig, axes


def _plot_prices_day(axes, analysis, day):
    for ax in axes:
        ax.clear()
    for ax in axes:
        _clear_twins(ax)

    res = _day_slice(analysis, day)
    (t, d, soc, real_p_imp_c, real_p_exp_c, op_p_imp_c, op_p_exp_c,
     event_day, day_start, day_end, n) = res
    if n == 0:
        for ax in axes:
            ax.text(0.5, 0.5, f"No data for {day.normalize().date()}",
                    transform=ax.transAxes, ha='center', va='center')
        _finalise(axes[0].figure, day_start, analysis.retail_tariff.name)
        return

    for ax in axes[:2]:
        _shade_event_intervals(ax, t, event_day, day_start, day_end)

    # Both price axes share one transform and one set of limits, so that a
    # given height means the same c/kWh in both panes, and so that the
    # household and operator series stay directly comparable.
    p_lo, p_hi = _price_axis_limits(real_p_exp_c, op_p_exp_c,
                                    real_p_imp_c, op_p_imp_c)

    # Panel 1: Export split (n_exp + f_exp) vs prices
    axes[0].fill_between(t, 0, d['n_exp'], step='post', color=COLOUR_SOLAR,
                         linewidth=0, label='n_exp (natural)')
    axes[0].fill_between(t, d['n_exp'], d['n_exp'] + d['f_exp'], step='post',
                         color=COLOUR_FORCED, alpha=0.8, linewidth=0,
                         label='f_exp (forced)')
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.3)
    axp0 = axes[0].twinx()
    axp0.step(t, real_p_exp_c, where='post', color=COLOUR_GRID,
              linestyle=':', linewidth=1.6, label='hh export price (TOU)')
    axp0.step(t, op_p_exp_c, where='post', color=COLOUR_FORCED,
              linestyle='--', linewidth=1.2, label='op export price', alpha=0.6)
    _apply_price_scale(axp0, p_lo, p_hi)
    axp0.set_ylabel('Export price (c/kWh)')
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axp0.get_legend_handles_labels()
    axp0.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

    # Panel 2: Import split (n_imp + f_imp) vs import prices
    axes[1].fill_between(t, 0, d['n_imp'], step='post', color=COLOUR_LOAD,
                         linewidth=0, label='n_imp (natural)')
    axes[1].fill_between(t, d['n_imp'], d['n_imp'] + d['f_imp'], step='post',
                         color=COLOUR_FORCED, alpha=0.8, linewidth=0,
                         label='f_imp (forced)')
    axes[1].set_ylim(bottom=0)
    axes[1].grid(True, alpha=0.3)
    axp1 = axes[1].twinx()
    axp1.step(t, real_p_imp_c, where='post', color=COLOUR_GRID,
              linestyle=':', linewidth=1.6, label='hh import price (TOU)')
    axp1.step(t, op_p_imp_c, where='post', color=COLOUR_FORCED,
              linestyle='--', linewidth=1.2, label='op import price', alpha=0.6)
    _apply_price_scale(axp1, p_lo, p_hi)
    axp1.set_ylabel('Import price (c/kWh)')
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = axp1.get_legend_handles_labels()
    axp1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

    # Panel 3: SOC
    _plot_soc_panel(axes[2], t, soc, d, analysis.battery_spec,
                    analysis.interval_hours, event_day)
    axes[2].grid(True, alpha=0.3)
    _format_time_axis(axes[2], t, day_start, day_end)

    titles  = ['Export: natural + forced vs price',
               'Import: natural + forced vs price',
               'Battery SOC + Charge/Discharge']
    ylabels = ['Export (kW)', 'Import (kW)', 'SOC (kWh)']
    for ax, title, yl in zip(axes, titles, ylabels):
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_ylabel(yl)
    axes[-1].set_xlabel('Hour of day')
    _finalise(axes[0].figure, day_start, analysis.retail_tariff.name)


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def _first_event_day(analysis: EovAnalysis) -> pd.Timestamp:
    """Return the date of the first VPP event interval, or first day if none."""
    idx = analysis.trace.index
    event_indices = np.where(analysis.vpp_event)[0]
    if len(event_indices):
        return idx[event_indices[0]].normalize()
    return idx[0].normalize()


def _next_event_day(analysis: EovAnalysis, current: pd.Timestamp) -> pd.Timestamp:
    """Return the date of the next VPP event day strictly after current."""
    idx = analysis.trace.index
    current_end = current.normalize() + pd.Timedelta(days=1)
    # Find event intervals that start after the end of the current day
    future_mask = (idx >= current_end) & analysis.vpp_event.astype(bool)
    future_indices = np.where(future_mask)[0]
    if len(future_indices):
        return idx[future_indices[0]].normalize()
    print("  (no further event days)")
    return current


def _interactive_loop(analysis: EovAnalysis, battery_mode: str = BATTERY_MODE) -> None:
    a = analysis

    fig_flows,  axes_flows  = _setup_flows_figure(a.site_id, battery_mode)
    fig_prices, axes_prices = _setup_prices_figure(a.site_id, battery_mode)
    plt.ion()

    first_day  = a.trace.index[0].normalize()
    last_day   = a.trace.index[-1].normalize()
    current    = _first_event_day(a)  # default to first event day for quick inspection

    _plot_flows_day(axes_flows, a, current)
    _plot_prices_day(axes_prices, a, current)
    plt.show()
    plt.pause(0.1)

    print(f"Days: {first_day.date()} to {last_day.date()}")
    print(f"Starting on first event day: {current.date()}")
    print("Controls: Enter=next day, b=back, YYYY-MM-DD=jump, "
          "e=first event, ne=next event, q=quit")

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
        elif raw == 'e':
            current = _first_event_day(a)
        elif raw == 'ne':
            current = _next_event_day(a, current)
        else:
            try:
                current = pd.Timestamp(raw).tz_localize(a.trace.index.tz).normalize()
                if current < first_day or current > last_day:
                    print(f"(out of range {first_day.date()}..{last_day.date()})")
                    current = max(first_day, min(last_day, current))
            except Exception as ex:
                print(f"  (unrecognised input: {ex})")
                continue

        _plot_flows_day(axes_flows, a, current)
        _plot_prices_day(axes_prices, a, current)
        fig_flows.canvas.draw_idle()
        fig_prices.canvas.draw_idle()
        plt.pause(0.05)

    plt.close(fig_flows)
    plt.close(fig_prices)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('site_id', help='Site identifier, e.g. S0008')
    parser.add_argument('year', type=int, help='Year, e.g. 2022')
    parser.add_argument('--state', default='NSW')
    parser.add_argument('--network-tariff', default='NSW_ausgrid_ea025_ea029_2026')
    parser.add_argument('--retail-tariff', default='NSW_agl_battery_rewards_vpp_2026')
    parser.add_argument('--op-tariff', default='NSW_operator_wholesale_2026')
    parser.add_argument('--export-limit-kw', type=float, default=10.0)
    parser.add_argument('--event-csv', type=Path, default=None,
                        help='If provided, write per-interval event diagnostic CSV to this path.')
    parser.add_argument('--project-root', type=Path, default=None)
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    trace, prices_mwh, network, retail, op_tariff = load_inputs(
        args.site_id, args.year, args.state,
        args.network_tariff, args.retail_tariff, args.op_tariff, paths,
    )
    for col in ('pv_site_net_kwh', 'gross_load_kwh'):
        if col not in trace.columns:
            print(f"ERROR: trace missing {col}", file=sys.stderr)
            return 1

    analysis = run_eov_analysis(
        args.site_id, args.year, trace, prices_mwh,
        network, retail, op_tariff, DEV_BATTERY,
        export_limit_kw=args.export_limit_kw,
        state=args.state,
    )
    _print_verification(analysis)
    if args.event_csv is not None:
        write_event_csv(analysis, args.event_csv)
    _interactive_loop(analysis)
    return 0


if __name__ == '__main__':
    sys.exit(main())