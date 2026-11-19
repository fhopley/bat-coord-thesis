"""Tests for model.components.perfect_foresight_lp.

All test cases use interval_hours=1.0 so that kWh per interval equals kW,
making the arithmetic easy to verify by hand. Battery power limits and
capacity are set generously unless the test specifically targets those
constraints. The LP solver (HiGHS) is fast enough that even calling it
many times in this suite adds negligible wall time.

Known-answer tests are structured around cases where the optimal solution
is analytically unambiguous:
  - flat prices + high c_deg → no cycling
  - clear cheap/expensive price spread → predictable charge/discharge
  - degradation cost exceeding spread → no cycling
  - solar surplus → sol_to_bat flow
  - battery exporting to grid → bat_to_grid flow
  - export limit → cap binding
  - terminal SOC constraint
  - SiteTraces invariants (algebraic, atol=1e-6)
  - validation errors
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from model.core.scenario import BatterySpec
from model.components.perfect_foresight_lp import perfect_foresight_lp


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _idx(n: int) -> pd.DatetimeIndex:
    """n-interval index at 1-hour cadence (simplifies kW = kWh/interval)."""
    return pd.date_range('2022-01-03 01:00:00', periods=n, freq='60min')


def _s(vals, idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(vals, index=idx, dtype=float)


def _bspec(
    capacity: float = 10.0,
    reserve_pct: float = 0.0,
    power: float = 10.0,
    eta_c: float = 1.0,
    eta_d: float = 1.0,
    c_deg: float = 0.0,
) -> BatterySpec:
    return BatterySpec(
        capacity_kwh=capacity,
        backup_reserve_pct=reserve_pct,
        power_kw=power,
        charge_efficiency=eta_c,
        discharge_efficiency=eta_d,
        c_deg_per_kwh=c_deg,
    )


def _run(solar, load, p_imp, p_exp, bspec, soc_init=0.0, soc_final=0.0,
         export_limit=np.inf, site_id='S0001', curtail_cost=0.0):
    idx = _idx(len(solar))
    return perfect_foresight_lp(
        site_id,
        _s(solar, idx), _s(load, idx),
        _s(p_imp, idx), _s(p_exp, idx),
        bspec,
        soc_init_kwh=soc_init,
        soc_final_kwh=soc_final,
        export_limit_kwh_per_interval=export_limit,
        interval_hours=1.0,
        curtail_cost_per_kwh_dollars=curtail_cost,
    )


# ---------------------------------------------------------------------------
# No-cycling cases
# ---------------------------------------------------------------------------

def test_no_cycling_high_degradation_cost():
    """c_deg >> any price spread: battery should sit idle."""
    # Spread = 0.3 - 0.05 = 0.25; cycle cost = 2*1.0 = 2.0 >> 0.25.
    traces = _run(
        solar=[0., 0., 0., 0.], load=[1., 1., 1., 1.],
        p_imp=[0.3, 0.3, 0.3, 0.3], p_exp=[0.05, 0.05, 0.05, 0.05],
        bspec=_bspec(c_deg=1.0),
    )
    np.testing.assert_allclose(traces.battery_soc_kwh.to_numpy(), 0., atol=1e-6)
    np.testing.assert_allclose(traces.import_kwh.to_numpy(), 1., atol=1e-6)
    np.testing.assert_allclose(traces.export_kwh.to_numpy(), 0., atol=1e-6)


def test_no_cycling_flat_price_with_degradation():
    """Flat prices and c_deg > 0: no incentive to cycle."""
    traces = _run(
        solar=[0., 0.], load=[1., 1.],
        p_imp=[0.3, 0.3], p_exp=[0.05, 0.05],
        bspec=_bspec(c_deg=0.01),
    )
    np.testing.assert_allclose(traces.battery_soc_kwh.to_numpy(), 0., atol=1e-6)


def test_single_interval():
    """T=1 edge case: the LP must respect the terminal SOC constraint."""
    traces = _run(
        solar=[0.], load=[1.],
        p_imp=[0.3], p_exp=[0.0],
        bspec=_bspec(capacity=5., power=5., c_deg=1.),
        soc_init=0., soc_final=0.,
    )
    np.testing.assert_allclose(traces.import_kwh.iloc[0], 1., atol=1e-6)
    np.testing.assert_allclose(traces.export_kwh.iloc[0], 0., atol=1e-6)
    np.testing.assert_allclose(traces.battery_soc_kwh.iloc[0], 0., atol=1e-6)


# ---------------------------------------------------------------------------
# Arbitrage cases
# ---------------------------------------------------------------------------

def test_arbitrage_unit_efficiency():
    """Cheap t=0 import -> charge; expensive t=1 load -> discharge.

    p_imp=[0.1, 0.5]: spread 0.4 > 0 -> cycle.
    η_c = η_d = 1.0. Charge 1 kWh at t=0 (cost 0.1), discharge at t=1
    (avoids 0.5 import). Net savings 0.4.
    Expected: grid_to_bat=1 at t=0, bat_to_home=1 at t=1, imp=[1,0].
    """
    traces = _run(
        solar=[0., 0.], load=[0., 1.],
        p_imp=[0.1, 0.5], p_exp=[0., 0.],
        bspec=_bspec(capacity=1., power=1.),
        soc_init=0., soc_final=0.,
    )
    np.testing.assert_allclose(traces.grid_to_bat_kwh.iloc[0], 1., atol=1e-6)
    np.testing.assert_allclose(traces.bat_to_home_kwh.iloc[1], 1., atol=1e-6)
    np.testing.assert_allclose(traces.import_kwh.to_numpy(), [1., 0.], atol=1e-6)
    np.testing.assert_allclose(traces.battery_soc_kwh.to_numpy(), [1., 0.], atol=1e-6)


def test_arbitrage_with_efficiency_losses():
    """Same as above but η_c = η_d = 0.9.

    Charge 1 AC -> soc = 0.9 DC.
    Discharge to soc_final=0: y = 0.9 * 0.9 = 0.81 AC kWh.
    Remaining load at t=1: 1 - 0.81 = 0.19 from grid.
    """
    traces = _run(
        solar=[0., 0.], load=[0., 1.],
        p_imp=[0.1, 0.5], p_exp=[0., 0.],
        bspec=_bspec(capacity=1., power=1., eta_c=0.9, eta_d=0.9),
        soc_init=0., soc_final=0.,
    )
    np.testing.assert_allclose(traces.grid_to_bat_kwh.iloc[0], 1., atol=1e-6)
    np.testing.assert_allclose(traces.battery_soc_kwh.iloc[0], 0.9, atol=1e-6)
    np.testing.assert_allclose(traces.battery_soc_kwh.iloc[1], 0., atol=1e-6)
    np.testing.assert_allclose(traces.bat_to_home_kwh.iloc[1], 0.81, atol=1e-5)
    np.testing.assert_allclose(traces.import_kwh.iloc[1], 0.19, atol=1e-5)


def test_degradation_threshold_prevents_cycling():
    """Cycle cost = 2 * c_deg = 0.5 > spread = 0.4 -> LP should not cycle."""
    traces = _run(
        solar=[0., 0.], load=[0., 1.],
        p_imp=[0.1, 0.5], p_exp=[0., 0.],
        bspec=_bspec(capacity=1., power=1., c_deg=0.25),
        soc_init=0., soc_final=0.,
    )
    np.testing.assert_allclose(traces.battery_soc_kwh.to_numpy(), 0., atol=1e-6)
    np.testing.assert_allclose(traces.import_kwh.to_numpy(), [0., 1.], atol=1e-6)


def test_three_period_arbitrage():
    """Load shifts from expensive to cheap: LP charges cheap, discharges dear.

    p_imp=[0.5, 0.1, 0.5], load=[1, 0, 1].
    Optimal: charge 1 kWh at t=1 (cost 0.1), discharge at t=2 (saves 0.5).
    At t=0 import 1 (nothing else available).
    """
    traces = _run(
        solar=[0., 0., 0.], load=[1., 0., 1.],
        p_imp=[0.5, 0.1, 0.5], p_exp=[0., 0., 0.],
        bspec=_bspec(capacity=1., power=1.),
        soc_init=0., soc_final=0.,
    )
    np.testing.assert_allclose(traces.import_kwh.iloc[0], 1., atol=1e-6)
    np.testing.assert_allclose(traces.grid_to_bat_kwh.iloc[1], 1., atol=1e-6)
    np.testing.assert_allclose(traces.bat_to_home_kwh.iloc[2], 1., atol=1e-6)
    np.testing.assert_allclose(traces.import_kwh.iloc[2], 0., atol=1e-6)


# ---------------------------------------------------------------------------
# Solar flow decomposition
# ---------------------------------------------------------------------------

def test_solar_charges_battery_not_grid():
    """Surplus solar fills the battery; grid_to_bat should be zero.

    t=0: solar=3, load=1 -> surplus=2 -> sol_to_bat=2, soc=2, no grid_to_bat.
    t=1: solar=0, load=2 -> bat_to_home=2, soc=0, no import.
    """
    traces = _run(
        solar=[3., 0.], load=[1., 2.],
        p_imp=[0.5, 0.5], p_exp=[0.1, 0.1],
        bspec=_bspec(capacity=2., power=2.),
        soc_init=0., soc_final=0.,
    )
    np.testing.assert_allclose(traces.sol_to_bat_kwh.iloc[0], 2., atol=1e-6)
    np.testing.assert_allclose(traces.grid_to_bat_kwh.iloc[0], 0., atol=1e-6)
    np.testing.assert_allclose(traces.bat_to_home_kwh.iloc[1], 2., atol=1e-6)
    np.testing.assert_allclose(traces.import_kwh.to_numpy(), [0., 0.], atol=1e-6)


def test_solar_split_between_home_bat_grid():
    """Solar surplus exceeds battery capacity: excess exports to grid.

    solar=5, load=1 -> sol_to_home=1, surplus=4.
    Battery capacity=2, power=2: sol_to_bat=2, sol_to_grid=2.
    """
    traces = _run(
        solar=[5., 0.], load=[1., 0.],
        p_imp=[0.3, 0.3], p_exp=[0.1, 0.1],
        bspec=_bspec(capacity=2., power=2.),
        soc_init=0., soc_final=2.,
    )
    np.testing.assert_allclose(traces.sol_to_home_kwh.iloc[0], 1., atol=1e-6)
    np.testing.assert_allclose(traces.sol_to_bat_kwh.iloc[0], 2., atol=1e-6)
    np.testing.assert_allclose(traces.sol_to_grid_kwh.iloc[0], 2., atol=1e-6)


# ---------------------------------------------------------------------------
# Battery-to-grid (bat_to_grid) — distinguishes LP from self_consumption
# ---------------------------------------------------------------------------

def test_battery_exports_to_grid():
    """LP charges from grid (cheap) and exports from battery (expensive).

    p_imp=[0.1, 0.5], p_exp=[0.0, 0.4], load=0.
    Cycle: import 1 at t=0 (cost 0.1), export 1 at t=1 (earn 0.4). Net -0.3.
    Better than doing nothing (cost 0) -> LP should cycle.
    """
    traces = _run(
        solar=[0., 0.], load=[0., 0.],
        p_imp=[0.1, 0.5], p_exp=[0.0, 0.4],
        bspec=_bspec(capacity=1., power=1.),
        soc_init=0., soc_final=0.,
    )
    np.testing.assert_allclose(traces.grid_to_bat_kwh.iloc[0], 1., atol=1e-6)
    np.testing.assert_allclose(traces.bat_to_grid_kwh.iloc[1], 1., atol=1e-6)
    np.testing.assert_allclose(traces.import_kwh.iloc[0], 1., atol=1e-6)
    np.testing.assert_allclose(traces.export_kwh.iloc[1], 1., atol=1e-6)


# ---------------------------------------------------------------------------
# Constraint tests
# ---------------------------------------------------------------------------

def test_soc_bounds_respected():
    """SOC must stay within [soc_min, capacity] at every interval."""
    soc_min = 0.2 * 5.0  # reserve_pct=20, capacity=5 -> soc_min=1
    traces = _run(
        solar=[0., 0., 0., 0.], load=[2., 2., 2., 2.],
        p_imp=[0.5, 0.1, 0.5, 0.1], p_exp=[0., 0., 0., 0.],
        bspec=_bspec(capacity=5., reserve_pct=20., power=5.),
        soc_init=soc_min, soc_final=soc_min,
    )
    soc = traces.battery_soc_kwh.to_numpy()
    assert (soc >= soc_min - 1e-6).all(), f"SOC below soc_min: {soc.min():.6f}"
    assert (soc <= 5.0 + 1e-6).all(), f"SOC above capacity: {soc.max():.6f}"


def test_power_limit_respected():
    """Charge/discharge must not exceed power_kw * interval_hours."""
    power = 0.5   # kW; interval_hours=1 -> cap = 0.5 kWh per interval
    traces = _run(
        solar=[0., 0., 0.], load=[0., 1., 0.],
        p_imp=[0.1, 0.5, 0.1], p_exp=[0., 0., 0.],
        bspec=_bspec(capacity=2., power=power),
        soc_init=0., soc_final=0.,
    )
    # Derived charge and discharge from flow decomposition
    charge = traces.grid_to_bat_kwh.to_numpy() + traces.sol_to_bat_kwh.to_numpy()
    discha = traces.bat_to_home_kwh.to_numpy() + traces.bat_to_grid_kwh.to_numpy()
    assert (charge  <= power + 1e-6).all(), f"Charge exceeds power cap: {charge.max():.6f}"
    assert (discha  <= power + 1e-6).all(), f"Discharge exceeds power cap: {discha.max():.6f}"


def test_export_limit_binding():
    """Export cap must be respected; LP curtails excess solar it cannot store.

    solar=2, load=0, power=1, capacity=2, export_limit=1.0.
    Without limit: exp=2 (all solar exported or stored).
    With limit: LP charges 1 kWh (power cap), exports 1 kWh (limit cap).
    """
    export_cap = 1.0
    traces = _run(
        solar=[2., 2.], load=[0., 0.],
        p_imp=[0.3, 0.3], p_exp=[0.1, 0.1],
        bspec=_bspec(capacity=2., power=1.),
        soc_init=0., soc_final=0.,
        export_limit=export_cap,
    )
    assert (traces.export_kwh.to_numpy() <= export_cap + 1e-6).all()


def test_curtailment_makes_export_limited_scenario_feasible():
    """A scenario infeasible without curtailment should solve when solar can be curtailed.

    T=1, solar=5, export_limit=1, battery holds 0 net change (soc_init=soc_final=0).
    Without curtailment the LP would be forced to export 5 kWh (violating the 1 kWh cap).
    With curtailment: curtail=4, exp=1, charge=discharge=0. Feasible.
    """
    traces = _run(
        solar=[5.], load=[0.],
        p_imp=[0.3], p_exp=[0.1],
        bspec=_bspec(capacity=5., power=5.),
        soc_init=0., soc_final=0.,
        export_limit=1.0,
    )
    assert (traces.export_kwh.to_numpy() <= 1.0 + 1e-6).all()
    np.testing.assert_allclose(traces.battery_soc_kwh.to_numpy(), 0., atol=1e-6)


def test_curtailment_avoids_export_penalty():
    """When exporting costs money (negative p_exp), LP should curtail instead.

    solar=[3,0], load=0, p_exp=-0.5 everywhere. Charging at t=0 and discharging
    at t=1 would incur a penalty on the t=1 export (cost 0.5). Curtailing
    all solar at t=0 and doing nothing at t=1 costs 0. LP should choose 0.
    """
    traces = _run(
        solar=[3., 0.], load=[0., 0.],
        p_imp=[0.3, 0.3], p_exp=[-0.5, -0.5],
        bspec=_bspec(capacity=1., power=1.),
        soc_init=0., soc_final=0.,
    )
    # Export everywhere should be zero (curtailing is free, exporting costs money)
    np.testing.assert_allclose(traces.export_kwh.to_numpy(), 0., atol=1e-6)
    np.testing.assert_allclose(traces.battery_soc_kwh.to_numpy(), 0., atol=1e-6)


# ---------------------------------------------------------------------------
# Curtailment cost modes (curtail_cost_per_kwh_dollars)
# ---------------------------------------------------------------------------

def test_curtail_default_matches_omitted_argument():
    """curtail_cost_per_kwh_dollars=0.0 (explicit) must match the parameter
    being omitted entirely -- this is the backward-compatibility contract:
    no existing caller passes this argument, so the default must reproduce
    exactly what the function did before the parameter existed.
    """
    solar = [5., 0.]
    load  = [0., 0.]
    p_imp = [0.3, 0.3]
    p_exp = [0.1, 0.1]
    idx = _idx(2)
    bspec = _bspec(capacity=1., power=1.)

    traces_explicit = _run(solar, load, p_imp, p_exp, bspec,
                            soc_init=0., soc_final=0., export_limit=1.0,
                            curtail_cost=0.0)
    traces_omitted = perfect_foresight_lp(
        'S0001',
        _s(solar, idx), _s(load, idx),
        _s(p_imp, idx), _s(p_exp, idx),
        bspec, soc_init_kwh=0., soc_final_kwh=0.,
        export_limit_kwh_per_interval=1.0, interval_hours=1.0,
    )
    np.testing.assert_allclose(
        traces_explicit.curtail_kwh.to_numpy(),
        traces_omitted.curtail_kwh.to_numpy(), atol=1e-9,
    )
    np.testing.assert_allclose(
        traces_explicit.import_kwh.to_numpy(),
        traces_omitted.import_kwh.to_numpy(), atol=1e-9,
    )
    np.testing.assert_allclose(
        traces_explicit.export_kwh.to_numpy(),
        traces_omitted.export_kwh.to_numpy(), atol=1e-9,
    )


def test_curtail_pinned_to_zero():
    """curtail_cost_per_kwh_dollars=None pins curtailment to zero.

    Same scenario as test_curtailment_makes_export_limited_scenario_feasible,
    which relies on curtailment being available to stay feasible. With
    curtailment pinned, that same scenario must become infeasible.
    """
    with pytest.raises(RuntimeError, match='did not find an optimal solution'):
        _run(
            solar=[5.], load=[0.],
            p_imp=[0.3], p_exp=[0.1],
            bspec=_bspec(capacity=5., power=5.),
            soc_init=0., soc_final=0.,
            export_limit=1.0,
            curtail_cost=None,
        )


def test_curtail_pinned_zero_when_feasible():
    """When pinning doesn't cause infeasibility, curtail_kwh must be all-zero
    and the flow balance must still hold (solar surplus goes to storage
    instead, since curtailment is unavailable).
    """
    traces = _run(
        solar=[3., 0.], load=[0., 0.],
        p_imp=[0.3, 0.3], p_exp=[-0.5, -0.5],
        bspec=_bspec(capacity=5., power=5.),
        soc_init=0., soc_final=3.,
        curtail_cost=None,
    )
    np.testing.assert_allclose(traces.curtail_kwh.to_numpy(), 0., atol=1e-9)


def test_curtail_priced_reduces_curtailment_between_free_and_zero():
    """A positive curtailment price should curtail less than the free
    (0.0) case, without needing to hit the infeasibility wall of pinning.

    Same export-penalty scenario as test_curtailment_avoids_export_penalty,
    but relaxed so curtailing isn't the LP's only escape (battery has
    headroom to store some of the surplus and shift it to a later,
    less-penalised interval).
    """
    solar = [3., 0., 0.]
    load  = [0., 0., 1.]
    p_imp = [0.3, 0.3, 0.3]
    p_exp = [-0.5, -0.5, -0.5]
    bspec = _bspec(capacity=2., power=2.)

    traces_free = _run(solar, load, p_imp, p_exp, bspec,
                        soc_init=0., soc_final=0., curtail_cost=0.0)
    traces_priced = _run(solar, load, p_imp, p_exp, bspec,
                          soc_init=0., soc_final=0., curtail_cost=0.05)

    curtail_free = traces_free.curtail_kwh.sum()
    curtail_priced = traces_priced.curtail_kwh.sum()
    assert curtail_priced <= curtail_free + 1e-9, (
        f"priced curtailment ({curtail_priced:.6f}) exceeds free curtailment "
        f"({curtail_free:.6f})"
    )


def test_curtail_priced_appears_in_objective():
    """A steep enough curtailment price should change dispatch relative to
    free curtailment: the LP should prefer storing/exporting surplus over
    curtailing once curtailment is expensive enough, even at an export
    penalty, provided the battery has room to absorb it.
    """
    solar = [4., 0.]
    load  = [0., 0.]
    p_imp = [0.3, 0.3]
    p_exp = [-0.05, -0.05]   # small export penalty
    bspec = _bspec(capacity=4., power=4.)

    traces_free = _run(solar, load, p_imp, p_exp, bspec,
                        soc_init=0., soc_final=0., curtail_cost=0.0)
    traces_priced = _run(solar, load, p_imp, p_exp, bspec,
                          soc_init=0., soc_final=0., curtail_cost=1.0)

    # Free: curtailing costs nothing, cheaper than paying the export penalty.
    assert traces_free.curtail_kwh.sum() > 0.
    # Expensive curtailment: LP should curtail strictly less (store/export
    # instead), since curtailing is now pricier than the small export penalty.
    assert traces_priced.curtail_kwh.sum() < traces_free.curtail_kwh.sum()


# ---------------------------------------------------------------------------
# Terminal SOC
# ---------------------------------------------------------------------------

def test_terminal_soc_enforced():
    """soc[T-1] must equal soc_final_kwh exactly."""
    soc_final = 2.5
    traces = _run(
        solar=[3., 3., 3.], load=[1., 1., 1.],
        p_imp=[0.3, 0.3, 0.3], p_exp=[0.1, 0.1, 0.1],
        bspec=_bspec(capacity=5., power=5.),
        soc_init=0., soc_final=soc_final,
    )
    np.testing.assert_allclose(traces.battery_soc_kwh.iloc[-1], soc_final, atol=1e-6)


# ---------------------------------------------------------------------------
# SiteTraces invariants (the key regression guard for the decomposition)
# ---------------------------------------------------------------------------

def test_sitetraces_invariants():
    """All four SiteTraces invariants must hold algebraically (atol=1e-6).

    import  = grid_to_home + grid_to_bat
    export  = sol_to_grid  + bat_to_grid
    load    = sol_to_home  + bat_to_home + grid_to_home
    solar   = sol_to_home  + sol_to_bat  + sol_to_grid
    """
    solar = [0., 2., 4., 1., 0., 3.]
    load  = [1., 1., 2., 1., 2., 1.]
    p_imp = [0.4, 0.1, 0.2, 0.4, 0.5, 0.1]
    p_exp = [0.1, 0.05, 0.1, 0.1, 0.1, 0.05]
    traces = _run(
        solar=solar, load=load, p_imp=p_imp, p_exp=p_exp,
        bspec=_bspec(capacity=3., power=2., eta_c=0.95, eta_d=0.95, c_deg=0.05),
        soc_init=0., soc_final=0.,
    )
    tol = 1e-6
    imp  = traces.import_kwh.to_numpy()
    exp  = traces.export_kwh.to_numpy()
    s    = np.array(solar)
    l    = np.array(load)

    np.testing.assert_allclose(
        imp, traces.grid_to_home_kwh.to_numpy() + traces.grid_to_bat_kwh.to_numpy(), atol=tol,
        err_msg="import != grid_to_home + grid_to_bat",
    )
    np.testing.assert_allclose(
        exp, traces.sol_to_grid_kwh.to_numpy() + traces.bat_to_grid_kwh.to_numpy(), atol=tol,
        err_msg="export != sol_to_grid + bat_to_grid",
    )
    np.testing.assert_allclose(
        l, traces.sol_to_home_kwh.to_numpy() + traces.bat_to_home_kwh.to_numpy()
        + traces.grid_to_home_kwh.to_numpy(), atol=tol,
        err_msg="load != sol_to_home + bat_to_home + grid_to_home",
    )
    np.testing.assert_allclose(
        s, traces.sol_to_home_kwh.to_numpy() + traces.sol_to_bat_kwh.to_numpy()
        + traces.sol_to_grid_kwh.to_numpy() + traces.curtail_kwh.to_numpy(), atol=tol,
        err_msg="solar != sol_to_home + sol_to_bat + sol_to_grid + curtail_kwh",
    )
    # All flows except sol_to_home must be non-negative
    for field in ('sol_to_bat_kwh', 'sol_to_grid_kwh', 'bat_to_home_kwh',
                  'bat_to_grid_kwh', 'grid_to_home_kwh', 'grid_to_bat_kwh'):
        arr = getattr(traces, field).to_numpy()
        assert (arr >= -tol).all(), f"{field} has negative values (min={arr.min():.2e})"


def test_lp_bill_le_no_battery_bill():
    """LP dispatch cost must not exceed the cost with no cycling (baseline).

    With a free battery (c_deg=0), the LP can always replicate the
    no-cycling solution, so its cost must be <= cost without cycling.
    """
    solar = [0., 0., 0.]
    load  = [1., 1., 1.]
    p_imp = [0.5, 0.1, 0.5]
    p_exp = [0.,  0.,  0. ]

    # No cycling (large c_deg)
    t_nc = _run(solar=solar, load=load, p_imp=p_imp, p_exp=p_exp,
                bspec=_bspec(c_deg=100.), soc_init=0., soc_final=0.)
    cost_nc = float(np.sum(
        t_nc.import_kwh.to_numpy() * np.array(p_imp)
        - t_nc.export_kwh.to_numpy() * np.array(p_exp)
    ))

    # LP optimal (c_deg=0)
    t_lp = _run(solar=solar, load=load, p_imp=p_imp, p_exp=p_exp,
                bspec=_bspec(c_deg=0.), soc_init=0., soc_final=0.)
    cost_lp = float(np.sum(
        t_lp.import_kwh.to_numpy() * np.array(p_imp)
        - t_lp.export_kwh.to_numpy() * np.array(p_exp)
    ))

    assert cost_lp <= cost_nc + 1e-6, (
        f"LP cost ({cost_lp:.6f}) > no-cycle cost ({cost_nc:.6f})"
    )


# ---------------------------------------------------------------------------
# Validation error cases
# ---------------------------------------------------------------------------

def test_price_ordering_violation_raises():
    """p_exp > p_imp at any interval must raise ValueError."""
    idx = _idx(2)
    with pytest.raises(ValueError, match='price_exp exceeds price_imp'):
        perfect_foresight_lp(
            'S',
            _s([0., 0.], idx), _s([0., 0.], idx),
            _s([0.3, 0.3], idx), _s([0.1, 0.4], idx),  # p_exp[1] > p_imp[1]
            _bspec(), 0., 0., interval_hours=1.,
        )


def test_nan_solar_raises():
    idx = _idx(2)
    with pytest.raises(ValueError, match='NaN'):
        perfect_foresight_lp(
            'S',
            _s([float('nan'), 1.], idx), _s([0., 0.], idx),
            _s([0.3, 0.3], idx), _s([0., 0.], idx),
            _bspec(), 0., 0., interval_hours=1.,
        )


def test_nan_price_raises():
    idx = _idx(2)
    with pytest.raises(ValueError, match='NaN'):
        perfect_foresight_lp(
            'S',
            _s([0., 0.], idx), _s([0., 0.], idx),
            _s([0.3, float('nan')], idx), _s([0., 0.], idx),
            _bspec(), 0., 0., interval_hours=1.,
        )


def test_negative_load_raises():
    idx = _idx(2)
    with pytest.raises(ValueError, match='negative'):
        perfect_foresight_lp(
            'S',
            _s([0., 0.], idx), _s([-1., 1.], idx),
            _s([0.3, 0.3], idx), _s([0., 0.], idx),
            _bspec(), 0., 0., interval_hours=1.,
        )


def test_soc_init_out_of_bounds_raises():
    idx = _idx(2)
    bspec = _bspec(capacity=5., reserve_pct=20.)  # soc_min=1
    with pytest.raises(ValueError, match='soc_init_kwh'):
        perfect_foresight_lp(
            'S',
            _s([0., 0.], idx), _s([0., 0.], idx),
            _s([0.3, 0.3], idx), _s([0., 0.], idx),
            bspec, soc_init_kwh=0.5, soc_final_kwh=1.,  # 0.5 < soc_min=1
            interval_hours=1.,
        )


def test_misaligned_index_raises():
    idx2 = _idx(2)
    idx3 = _idx(3)
    with pytest.raises(ValueError, match='index'):
        perfect_foresight_lp(
            'S',
            _s([0., 0.], idx2), _s([0., 0.], idx2),
            _s([0.3, 0.3, 0.3], idx3), _s([0., 0., 0.], idx3),
            _bspec(), 0., 0., interval_hours=1.,
        )