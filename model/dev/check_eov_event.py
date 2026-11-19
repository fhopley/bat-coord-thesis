"""Deep-dive diagnostic for a single vpp_sc_event event window.

Checks the natural/forced split accounting, ending SOC, household and retailer
economics across the event day, and operator P&L for the event window itself.

Usage (from project root):
    python3 -m model.dev.check_eov_event SITE_ID YEAR EVENT_DATE
    python3 -m model.dev.check_eov_event S0579 2022 2022-05-03

Runs the full EOV analysis pipeline (same as viz_vpp_sc_event) then
isolates the specified event window and day for detailed checks.

Arguments:
    SITE_ID         Site identifier, e.g. S0579
    YEAR            Year, e.g. 2022
    EVENT_DATE      Date of the event to examine (YYYY-MM-DD, AEST)

Optional:
    --state             NEM state (default: NSW)
    --network-tariff    Network tariff ID (default: ausgrid_ea025_network_2024)
    --retail-tariff     Retail tariff ID (default: agl_battery_rewards_vpp_2025)
    --op-tariff         Operator tariff ID (default: agl_operator_wholesale_2025)
    --export-limit-kw   Export limit kW (default: 10.0)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model.core.paths import default_paths
from model.core.scenario import BatterySpec
from model.core.tariff import load_network_tariff, load_retail_tariff
from model.components.billing import compute_bill
from model.dev._dev_common import DEV_BATTERY
from model.dev.viz_vpp_sc_event import load_inputs, run_eov_analysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title: str = '') -> None:
    if title:
        print(f"\n--- {title} ---")
    else:
        print()


def _find_event_window(
    vpp_event: np.ndarray,
    index: pd.DatetimeIndex,
    event_date: str,
) -> tuple[int, int] | None:
    """Return (win_start, win_end) for the event window that falls on event_date.

    win_end is exclusive. Returns None if no event window found on that date.
    """
    target = pd.Timestamp(event_date, tz='Etc/GMT-10')
    day_start = target.normalize()
    day_end   = day_start + pd.Timedelta(days=2)  # search up to next day for 10h window

    # Find contiguous event blocks
    T = len(vpp_event)
    blocks = []
    in_block = False
    for t in range(T):
        if vpp_event[t] and not in_block:
            block_start = t
            in_block = True
        elif not vpp_event[t] and in_block:
            blocks.append((block_start, t))
            in_block = False
    if in_block:
        blocks.append((block_start, T))

    for ws, we in blocks:
        block_date = index[ws].normalize()
        if block_date == day_start:
            return ws, we

    return None


def _day_mask(index: pd.DatetimeIndex, event_date: str) -> pd.Series:
    """Boolean Series selecting the calendar day of event_date (AEST)."""
    target = pd.Timestamp(event_date, tz='Etc/GMT-10').normalize()
    return pd.Series(
        (index >= target) & (index < target + pd.Timedelta(days=1)),
        index=index,
    )


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def run_check(
    site_id: str,
    year: int,
    event_date: str,
    state: str,
    network_tariff_id: str,
    retail_tariff_id: str,
    op_tariff_id: str,
    export_limit_kw: float,
    paths,
) -> int:

    print(f"\n=== EOV Event Diagnostic: {site_id} {year} event on {event_date} ===")

    # ------------------------------------------------------------------ #
    # Load and solve                                                       #
    # ------------------------------------------------------------------ #
    trace, prices_mwh, network, retail, op_tariff = load_inputs(
        site_id, year, state,
        network_tariff_id, retail_tariff_id, op_tariff_id, paths,
    )
    a = run_eov_analysis(
        site_id, year, trace, prices_mwh,
        network, retail, op_tariff, DEV_BATTERY,
        export_limit_kw=export_limit_kw,
    )

    idx         = a.eov_result.import_kwh.index
    solar       = a.trace['pv_site_net_kwh'].to_numpy()
    load        = a.trace['gross_load_kwh'].to_numpy()
    vpp_event   = a.vpp_event
    interval_h  = a.interval_hours
    vpp_cfg     = a.retail_tariff.vpp
    _C          = 100.0

    # ------------------------------------------------------------------ #
    # Locate the event window                                              #
    # ------------------------------------------------------------------ #
    window = _find_event_window(vpp_event, idx, event_date)
    if window is None:
        print(f"ERROR: no VPP event window found starting on {event_date}",
              file=sys.stderr)
        return 1

    ws, we = window   # we is exclusive
    Te = we - ws
    win_idx   = idx[ws:we]
    print(f"\nEvent window: {idx[ws]} -- {idx[we-1]}  ({Te} intervals, "
          f"{Te * interval_h:.1f} h)")

    # Pull arrays for the event window
    ew_solar     = solar[ws:we]
    ew_load      = load[ws:we]
    ew_n_imp     = a.eov_vt.n_imp_kwh.iloc[ws:we].to_numpy()
    ew_f_imp     = a.eov_vt.f_imp_kwh.iloc[ws:we].to_numpy()
    ew_n_exp     = a.eov_vt.n_exp_kwh.iloc[ws:we].to_numpy()
    ew_f_exp     = a.eov_vt.f_exp_kwh.iloc[ws:we].to_numpy()
    ew_soc_eov   = a.eov_result.battery_soc_kwh.iloc[ws:we].to_numpy()
    ew_soc_sc    = a.sc_result.battery_soc_kwh.iloc[ws:we].to_numpy()
    ew_op_imp    = a.op_p_imp.iloc[ws:we].to_numpy() * _C   # c/kWh
    ew_op_exp    = a.op_p_exp.iloc[ws:we].to_numpy() * _C   # c/kWh
    ew_ret_imp   = a.real_p_imp.iloc[ws:we].to_numpy() * _C
    ew_ret_exp   = a.real_p_exp.iloc[ws:we].to_numpy() * _C

    ew_sc_n_imp  = a.sc_result.import_kwh.iloc[ws:we].to_numpy()
    ew_sc_n_exp  = a.sc_result.export_kwh.iloc[ws:we].to_numpy()
    ew_charge    = (a.eov_result.sol_to_bat_kwh.iloc[ws:we].to_numpy()
                  + a.eov_result.grid_to_bat_kwh.iloc[ws:we].to_numpy())
    ew_discharge = (a.eov_result.bat_to_home_kwh.iloc[ws:we].to_numpy()
                  + a.eov_result.bat_to_grid_kwh.iloc[ws:we].to_numpy())

    # ------------------------------------------------------------------ #
    # Check 1: Natural import -- LP total vs SC total                     #
    # ------------------------------------------------------------------ #
    _sep("Check 1: Natural import budget (LP total vs SC total)")
    sc_n_imp_total  = float(ew_sc_n_imp.sum())
    lp_n_imp_total  = float(ew_n_imp.sum())
    lp_f_imp_total  = float(ew_f_imp.sum())
    lp_imp_total    = lp_n_imp_total + lp_f_imp_total

    print(f"  SC  natural import total:  {sc_n_imp_total:8.4f} kWh  (the budget)")
    print(f"  LP  natural import total:  {lp_n_imp_total:8.4f} kWh")
    print(f"  LP  forced  import total:  {lp_f_imp_total:8.4f} kWh")
    print(f"  LP  total   import:        {lp_imp_total:8.4f} kWh")
    excess = lp_n_imp_total - sc_n_imp_total
    print(f"  n_imp excess over budget:  {excess:+.6f} kWh  "
          f"{'PASS' if excess <= 1e-4 else 'FAIL -- budget exceeded'}")

    # ------------------------------------------------------------------ #
    # Check 2: Natural export -- LP total vs SC total (LP <= SC expected) #
    # ------------------------------------------------------------------ #
    _sep("Check 2: Natural export (LP total <= SC total expected)")
    sc_n_exp_total  = float(ew_sc_n_exp.sum())
    lp_n_exp_total  = float(ew_n_exp.sum())
    lp_f_exp_total  = float(ew_f_exp.sum())
    lp_exp_total    = lp_n_exp_total + lp_f_exp_total
    export_diff     = lp_n_exp_total - sc_n_exp_total  # expected <= 0

    print(f"  SC  natural export total:  {sc_n_exp_total:8.4f} kWh")
    print(f"  LP  natural export total:  {lp_n_exp_total:8.4f} kWh")
    print(f"  LP  forced  export total:  {lp_f_exp_total:8.4f} kWh")
    print(f"  LP  total   export:        {lp_exp_total:8.4f} kWh")
    print(f"  LP n_exp vs SC n_exp:      {export_diff:+.4f} kWh  "
          f"{'PASS (LP <= SC as expected)' if export_diff <= 1e-4 else 'FAIL -- LP natural export exceeds SC'}")
    if export_diff < -1e-4:
        print(f"  Solar redirected to battery instead of natural export: "
              f"{-export_diff:.4f} kWh")

    # Per-interval ceiling check (n_exp[t] <= n_exp_hi[t] = sc_n_exp[t])
    ceiling_violations = int(np.sum(ew_n_exp > ew_sc_n_exp + 1e-6))
    print(f"  Per-interval ceiling violations: {ceiling_violations}  "
          f"{'PASS' if ceiling_violations == 0 else 'FAIL'}")

    # ------------------------------------------------------------------ #
    # Check 3: Ending SOC -- LP vs SC                                     #
    # ------------------------------------------------------------------ #
    _sep("Check 3: Ending SOC (LP free to exit lower than SC)")
    soc_entry    = float(a.eov_result.battery_soc_kwh.iloc[ws - 1]) if ws > 0 \
                   else (a.battery_spec.backup_reserve_pct / 100.0) * a.battery_spec.capacity_kwh
    soc_exit_eov = float(ew_soc_eov[-1])
    soc_exit_sc  = float(ew_soc_sc[-1])
    soc_diff     = soc_exit_eov - soc_exit_sc  # typically negative

    print(f"  Entry SOC:       {soc_entry:6.4f} kWh")
    print(f"  Exit SOC (SC):   {soc_exit_sc:6.4f} kWh")
    print(f"  Exit SOC (LP):   {soc_exit_eov:6.4f} kWh")
    print(f"  SOC delta (LP-SC): {soc_diff:+.4f} kWh  "
          f"({'LP exits lower -- battery more discharged' if soc_diff < -1e-4 else 'LP exits higher or equal'})")
    soc_min = (a.battery_spec.backup_reserve_pct / 100.0) * a.battery_spec.capacity_kwh
    print(f"  Exit SOC within bounds [{soc_min:.4f}, {a.battery_spec.capacity_kwh:.4f}]: "
          f"{'PASS' if soc_min - 1e-6 <= soc_exit_eov <= a.battery_spec.capacity_kwh + 1e-6 else 'FAIL'}")

    # ------------------------------------------------------------------ #
    # Check 4: Household economics -- full event day                      #
    # ------------------------------------------------------------------ #
    _sep("Check 4: Household economics across event day")
    day_mask = _day_mask(idx, event_date)

    def _day_bill_component(traces, bill_fn_kwargs):
        """Compute bill for full year and report day-subset economics from traces."""
        pass  # We use the full-year bills; isolate day via mask

    # Use pre-computed full-year bills; isolate day contribution via flows
    # Household pays: import * retail_imp_price - export * retail_exp_price
    # (daily charges are the same for both so net out; reward only in EOV)
    day_imp_sc  = a.sc_result.import_kwh[day_mask].to_numpy()
    day_exp_sc  = a.sc_result.export_kwh[day_mask].to_numpy()
    day_imp_eov = a.eov_result.import_kwh[day_mask].to_numpy()
    day_exp_eov = a.eov_result.export_kwh[day_mask].to_numpy()
    day_ret_imp = a.real_p_imp[day_mask].to_numpy()  # $/kWh
    day_ret_exp = a.real_p_exp[day_mask].to_numpy()  # $/kWh

    sc_dispatch_cost  = float((day_imp_sc  * day_ret_imp).sum()
                             - (day_exp_sc  * day_ret_exp).sum())
    eov_dispatch_cost = float((day_imp_eov * day_ret_imp).sum()
                             - (day_exp_eov * day_ret_exp).sum())

    # VPP reward received (retail_other_credits is negative = credit)
    # Isolate day reward from event window flows
    day_f_imp = a.eov_vt.f_imp_kwh[day_mask].to_numpy()
    day_f_exp = a.eov_vt.f_exp_kwh[day_mask].to_numpy()
    vpp_imp_reward = vpp_cfg.imp_reward_c_per_kwh / _C
    vpp_exp_reward = vpp_cfg.exp_reward_c_per_kwh / _C
    day_reward    = float(day_f_imp.sum() * vpp_imp_reward
                        + day_f_exp.sum() * vpp_exp_reward)

    hh_dispatch_saving = sc_dispatch_cost - eov_dispatch_cost
    hh_net_benefit     = hh_dispatch_saving + day_reward

    print(f"  SC  dispatch cost (day):   ${sc_dispatch_cost:8.4f}")
    print(f"  EOV dispatch cost (day):   ${eov_dispatch_cost:8.4f}")
    print(f"  Dispatch saving:           ${hh_dispatch_saving:8.4f}")
    print(f"  VPP reward received:       ${day_reward:8.4f}  "
          f"(f_imp={day_f_imp.sum():.3f} kWh @ ${vpp_imp_reward:.2f}, "
          f"f_exp={day_f_exp.sum():.3f} kWh @ ${vpp_exp_reward:.2f})")
    print(f"  Net household benefit:     ${hh_net_benefit:8.4f}  "
          f"{'PASS (HH better off)' if hh_net_benefit > -1e-4 else 'WARN -- HH worse off on this day'}")

    # Full-year comparison for context
    _sep("  Full-year household bill comparison (context)")
    print(f"  SC  annual bill:           ${a.sc_bill.household_total:8.2f}")
    print(f"  EOV annual bill:           ${a.eov_bill.household_total:8.2f}")
    print(f"  Annual saving:             ${a.sc_bill.household_total - a.eov_bill.household_total:8.2f}")

    # ------------------------------------------------------------------ #
    # Check 5: Unified entity P&L -- retailer and operator are the same  #
    # entity (e.g. AGL), so their economics are combined into one view.  #
    #                                                                     #
    # The entity:                                                         #
    #   - Collects retail revenue from the household (import bill)        #
    #   - Pays wholesale cost for all energy imported from the grid       #
    #   - Receives wholesale revenue for all energy exported to the grid  #
    #   - Pays VPP rewards to the household for forced flows              #
    #   - The network costs and charges net out (entity neither pays nor  #
    #     receives network tariffs directly in this simplified view)      #
    # ------------------------------------------------------------------ #
    _sep("Check 5: Unified retailer/operator entity P&L (event day)")
    day_prices_mwh = a.prices_mwh[day_mask].to_numpy()  # $/MWh
    ilf = a.network_tariff.import_loss_factor
    elf = a.network_tariff.export_loss_factor

    # Wholesale costs and revenues (loss-factor adjusted, same for SC and EOV)
    day_ws_cost_sc  = float((day_imp_sc  * day_prices_mwh / 1000.0 * ilf).sum())
    day_ws_cost_eov = float((day_imp_eov * day_prices_mwh / 1000.0 * ilf).sum())
    day_ws_rev_sc   = float((day_exp_sc  * day_prices_mwh / 1000.0 * elf).sum())
    day_ws_rev_eov  = float((day_exp_eov * day_prices_mwh / 1000.0 * elf).sum())

    # Entity P&L = retail revenue - wholesale import cost + wholesale export revenue
    #            - VPP rewards paid to household
    entity_pnl_sc  = sc_dispatch_cost  - day_ws_cost_sc  + day_ws_rev_sc
    entity_pnl_eov = eov_dispatch_cost - day_ws_cost_eov + day_ws_rev_eov - day_reward
    entity_delta   = entity_pnl_eov - entity_pnl_sc

    # Decompose the delta into its drivers
    retail_rev_change = eov_dispatch_cost - sc_dispatch_cost   # negative = entity earns less
    ws_import_saving  = day_ws_cost_sc - day_ws_cost_eov       # positive = entity saves
    ws_export_gain    = day_ws_rev_eov - day_ws_rev_sc         # positive = entity earns more
    reward_cost       = -day_reward                            # negative = entity pays out

    print(f"                          {'SC':>10}   {'EOV':>10}")
    print(f"  Retail revenue (HH bill): ${sc_dispatch_cost:>9.4f}   ${eov_dispatch_cost:>9.4f}")
    print(f"  Wholesale import cost:   -${day_ws_cost_sc:>9.4f}  -${day_ws_cost_eov:>9.4f}")
    print(f"  Wholesale export rev:    +${day_ws_rev_sc:>9.4f}  +${day_ws_rev_eov:>9.4f}")
    print(f"  VPP rewards paid:        -${0.0:>9.4f}  -${day_reward:>9.4f}")
    print(f"  {'':->50}")
    print(f"  Entity P&L (day):         ${entity_pnl_sc:>9.4f}   ${entity_pnl_eov:>9.4f}")
    print()
    print(f"  Entity P&L delta:        {entity_delta:+.4f}  "
          f"{'PASS (entity better off)' if entity_delta > -1e-4 else 'NOTE -- entity worse off on this day'}")
    print()
    print(f"  Delta decomposition:")
    print(f"    Retail revenue change:   {retail_rev_change:+.4f}  "
          f"(household imports less at retail)")
    print(f"    Wholesale import saving: {ws_import_saving:+.4f}  "
          f"(entity buys cheaper wholesale energy)")
    print(f"    Wholesale export gain:   {ws_export_gain:+.4f}  "
          f"(entity sells more at spike price)")
    print(f"    VPP rewards paid:        {reward_cost:+.4f}  "
          f"(paid to household for forced dispatch)")
    print(f"    {'':->42}")
    print(f"    Total:                   {retail_rev_change + ws_import_saving + ws_export_gain + reward_cost:+.4f}")

    # ------------------------------------------------------------------ #
    # Check 6: Per-interval table for the event window                    #
    # ------------------------------------------------------------------ #
    _sep("Check 6: Per-interval detail for event window")
    print(f"  {'Time':>5}  {'sol':>6}  {'load':>6}  {'n_imp':>6}  {'f_imp':>6}  "
          f"{'n_exp':>6}  {'f_exp':>6}  {'chg':>6}  {'dis':>6}  "
          f"{'soc':>6}  {'op_imp':>7}  {'op_exp':>7}  {'ret_imp':>7}")
    print(f"  {'':->5}  {'':->6}  {'':->6}  {'':->6}  {'':->6}  "
          f"{'':->6}  {'':->6}  {'':->6}  {'':->6}  "
          f"{'':->6}  {'':->7}  {'':->7}  {'':->7}")

    for i, ts in enumerate(win_idx):
        # Flag intervals with forced flows
        flag = ''
        if ew_f_imp[i] > 1e-4:
            flag = ' <f_imp'
        elif ew_f_exp[i] > 1e-4:
            flag = ' <f_exp'
        print(
            f"  {ts.strftime('%H:%M'):>5}  "
            f"{ew_solar[i]*12:6.3f}  "    # kW (x12 for 5-min)
            f"{ew_load[i]*12:6.3f}  "
            f"{ew_n_imp[i]*12:6.3f}  "
            f"{ew_f_imp[i]*12:6.3f}  "
            f"{ew_n_exp[i]*12:6.3f}  "
            f"{ew_f_exp[i]*12:6.3f}  "
            f"{ew_charge[i]*12:6.3f}  "
            f"{ew_discharge[i]*12:6.3f}  "
            f"{ew_soc_eov[i]:6.3f}  "
            f"{ew_op_imp[i]:7.2f}  "
            f"{ew_op_exp[i]:7.2f}  "
            f"{ew_ret_imp[i]:7.2f}"
            f"{flag}"
        )
    print(f"\n  (kW values = kWh/interval * 12 for 5-min; SOC in kWh; prices in c/kWh)")

    print(f"\n=== Diagnostic complete ===")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('site_id',    help='Site identifier, e.g. S0579')
    parser.add_argument('year',       type=int, help='Year, e.g. 2022')
    parser.add_argument('event_date', help='Event date YYYY-MM-DD (AEST)')
    parser.add_argument('--state',          default='NSW')
    parser.add_argument('--network-tariff', default='ausgrid_ea025_network_2024')
    parser.add_argument('--retail-tariff',  default='agl_battery_rewards_vpp_2025')
    parser.add_argument('--op-tariff',      default='agl_operator_wholesale_2025')
    parser.add_argument('--export-limit-kw', type=float, default=10.0)
    parser.add_argument('--project-root',   type=Path, default=None)
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    return run_check(
        args.site_id, args.year, args.event_date,
        args.state, args.network_tariff, args.retail_tariff,
        args.op_tariff, args.export_limit_kw, paths,
    )


if __name__ == '__main__':
    sys.exit(main())