"""Unit tests for model.core.prices.

Uses synthetic tariffs with single all-day periods so period_rate_series
returns a constant per layer regardless of weekday/weekend/hour mapping, which
lets us assert exact $/kWh values for unit conversions, loss-factor
application, plan-type branching, and the customer-facing combined vectors.

NOTE: tariffs are constructed with the documented field names. If the real
NetworkTariff/RetailTariff constructors differ (extra required fields, or
construction via loaders/fixtures), mirror whatever model/tests/test_billing.py
does to build them and adjust the helpers below.
"""
import numpy as np
import pandas as pd
import pytest

from model.core.tariff import NetworkTariff, RetailTariff, TariffPeriod
from model.core.prices import build_layer_rates, build_price_vectors
from model.core.localtime import local_wall_clock

INTERVAL = 5 / 60
STATE = "NSW"  # matches the state="NSW" set on every tariff fixture below


def _index(n=288):
    # one day at 5-min, interval-END timestamps.
    # tz-aware AEST (Etc/GMT-10), matching the convention used throughout
    # the model (DataContext, SiteTraces, price series) -- this fixture was
    # previously tz-naive, which happened to work only because the old
    # period_rate_series read .hour/.dayofweek directly off it with no local
    # time conversion. local_wall_clock() requires a real tz-aware instant.
    return pd.date_range("2024-06-03 00:05", periods=n, freq="5min",
                         tz="Etc/GMT-10")


def _network(usage_c=10.0, feedin_c=2.0,
             usage_dlf=1.05, import_tlf=1.0, export_dlf=0.95, export_tlf=1.0):
    return NetworkTariff(
        name="net", state="NSW",
        usage_tariffs=(TariffPeriod(0, 24, usage_c),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, feedin_c),),
        daily_c_per_day=60.0,
        usage_dlf=usage_dlf, export_dlf=export_dlf,
        import_tlf=import_tlf, export_tlf=export_tlf,
    )


def _wholesale_exposed(usage_other_c=4.0, feedin_other_c=1.0):
    return RetailTariff(
        name="amber", state="NSW", type="wholesale_exposed",
        daily_c_per_day=82.0,
        usage_tariffs=(), usage_tariffs_weekend=None, feedin_tariffs=(),
        usage_other_c_per_kwh=usage_other_c,
        feedin_other_c_per_kwh=feedin_other_c,
    )


def _tou(usage_c=20.0, feedin_c=5.0, usage_other_c=3.0, feedin_other_c=0.0):
    return RetailTariff(
        name="tou", state="NSW", type="tou",
        daily_c_per_day=151.0,
        usage_tariffs=(TariffPeriod(0, 24, usage_c),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, feedin_c),),
        usage_other_c_per_kwh=usage_other_c,
        feedin_other_c_per_kwh=feedin_other_c,
    )


def _flat_price(index, mwh=100.0):
    return pd.Series(np.full(len(index), mwh), index=index)


# ---------------------------------------------------------------- layers

def test_layer_unit_conversions_and_loss_factors_wholesale_exposed():
    idx = _index()
    lr = build_layer_rates(idx, _flat_price(idx, 100.0),
                           _network(), _wholesale_exposed(), STATE, INTERVAL)
    # wholesale: 100/1000 = 0.1, x lf
    assert lr.wholesale_import == pytest.approx(0.1 * 1.05)
    assert lr.wholesale_export == pytest.approx(0.1 * 0.95)
    # network: 10c -> 0.10, x lf ; 2c -> 0.02, x lf
    assert lr.network_import == pytest.approx(0.10 * 1.05)
    assert lr.network_export == pytest.approx(0.02 * 0.95)
    # retail: flat adders only, no loss factor
    assert lr.retail_import == pytest.approx(0.04)
    assert lr.retail_export == pytest.approx(0.01)


def test_retail_layer_not_loss_adjusted_tou():
    idx = _index()
    lr = build_layer_rates(idx, _flat_price(idx), _network(), _tou(), STATE, INTERVAL)
    # 20c usage + 3c adder = 0.23 ; 5c feedin + 0 = 0.05, no loss factor
    assert lr.retail_import == pytest.approx(0.23)
    assert lr.retail_export == pytest.approx(0.05)


def test_negative_spot_price_makes_wholesale_rate_negative():
    idx = _index()
    lr = build_layer_rates(idx, _flat_price(idx, -200.0),
                           _network(), _wholesale_exposed(), STATE, INTERVAL)
    assert lr.wholesale_import == pytest.approx(-0.2 * 1.05)
    assert lr.wholesale_export == pytest.approx(-0.2 * 0.95)


def test_export_penalty_negative_network_feedin():
    idx = _index()
    lr = build_layer_rates(idx, _flat_price(idx),
                           _network(feedin_c=-1.0), _wholesale_exposed(), STATE, INTERVAL)
    assert lr.network_export == pytest.approx(-0.01 * 0.95)


# ---------------------------------------------------- combined vectors

def test_wholesale_exposed_combined_is_full_stack():
    idx = _index()
    p_imp, p_exp = build_price_vectors(idx, _flat_price(idx, 100.0),
                                       _network(), _wholesale_exposed(), STATE, INTERVAL)
    # 0.105 + 0.105 + 0.04 = 0.25 ; 0.095 + 0.019 + 0.01 = 0.124
    assert p_imp.iloc[0] == pytest.approx(0.25)
    assert p_exp.iloc[0] == pytest.approx(0.124)
    assert p_imp.index.equals(idx)


def test_tou_combined_is_retail_only():
    idx = _index()
    p_imp, p_exp = build_price_vectors(idx, _flat_price(idx, 100.0),
                                       _network(), _tou(), STATE, INTERVAL)
    # retail only: wholesale/network excluded entirely
    assert p_imp.iloc[0] == pytest.approx(0.23)
    assert p_exp.iloc[0] == pytest.approx(0.05)


def test_combined_equals_sum_of_layers_wholesale_exposed():
    idx = _index()
    net, ret = _network(), _wholesale_exposed()
    lr = build_layer_rates(idx, _flat_price(idx), net, ret, STATE, INTERVAL)
    p_imp, p_exp = build_price_vectors(idx, _flat_price(idx), net, ret, STATE, INTERVAL)
    np.testing.assert_allclose(
        p_imp.to_numpy(), lr.wholesale_import + lr.network_import + lr.retail_import)
    np.testing.assert_allclose(
        p_exp.to_numpy(), lr.wholesale_export + lr.network_export + lr.retail_export)


# ---------------------------------------------------------------- errors

def test_nan_prices_raise():
    idx = _index()
    price = _flat_price(idx)
    price.iloc[5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        build_layer_rates(idx, price, _network(), _wholesale_exposed(), STATE, INTERVAL)


def test_misaligned_index_raises():
    idx = _index()
    price = _flat_price(idx).iloc[:-1]  # one short
    with pytest.raises(ValueError, match="index"):
        build_layer_rates(idx, price, _network(), _wholesale_exposed(), STATE, INTERVAL)

# ------------------------------------------- rolling optimisation price stack

from model.core.prices import (  # noqa: E402
    PriceStackOffsets, build_price_stack_offsets, validate_price_ordering,
    MARKET_PRICE_FLOOR_MWH, MARKET_PRICE_CAP_MWH,
)


def _varying_price(index, seed=0):
    """A price series that VARIES, including negative and spike values.

    A flat series cannot distinguish the affine reconstruction from any other
    formula that happens to be right at one price, so the reconstruction tests
    below must not use _flat_price.
    """
    rng = np.random.default_rng(seed)
    vals = rng.normal(80.0, 250.0, len(index))
    vals[3] = -950.0      # near the market floor, where the spread is tightest
    vals[7] = 14000.0     # spike
    vals[11] = 0.0
    return pd.Series(vals, index=index)


def test_offsets_reconstruct_build_price_vectors_exactly():
    """The whole design rests on this: if it fails, the rolling optimiser and
    the bill are computed from different price stacks."""
    idx = _index()
    net, ret = _network(), _wholesale_exposed()
    prices = _varying_price(idx)

    offsets = build_price_stack_offsets(idx, net, ret, STATE, INTERVAL)
    p_imp_ref, p_exp_ref = build_price_vectors(idx, prices, net, ret, STATE, INTERVAL)
    p_imp, p_exp = offsets.window_price_vectors(prices.to_numpy(), start=0)

    np.testing.assert_allclose(p_imp, p_imp_ref.to_numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(p_exp, p_exp_ref.to_numpy(), rtol=0, atol=1e-12)


def test_offsets_reconstruct_a_mid_period_window():
    """Windows start at arbitrary offsets, so slicing must line up."""
    idx = _index()
    net, ret = _network(), _wholesale_exposed()
    prices = _varying_price(idx, seed=1)
    offsets = build_price_stack_offsets(idx, net, ret, STATE, INTERVAL)
    p_imp_ref, p_exp_ref = build_price_vectors(idx, prices, net, ret, STATE, INTERVAL)

    start, stop = 100, 220
    p_imp, p_exp = offsets.window_price_vectors(
        prices.to_numpy()[start:stop], start=start)

    np.testing.assert_allclose(p_imp, p_imp_ref.to_numpy()[start:stop], atol=1e-12)
    np.testing.assert_allclose(p_exp, p_exp_ref.to_numpy()[start:stop], atol=1e-12)


def test_window_beyond_period_raises():
    idx = _index()
    offsets = build_price_stack_offsets(idx, _network(), _wholesale_exposed(),
                                        STATE, INTERVAL)
    with pytest.raises(ValueError, match="outside the offsets period"):
        offsets.window_price_vectors(np.zeros(10), start=len(idx) - 5)


def test_tou_plan_rejected():
    """A TOU customer price does not depend on RRP, so rolling optimisation on
    one is perfect foresight with a truncated horizon."""
    idx = _index()
    with pytest.raises(ValueError, match="wholesale_exposed"):
        build_price_stack_offsets(idx, _network(), _tou(), STATE, INTERVAL)


# ------------------------------------------------------------ price ordering

def test_price_ordering_passes_with_equal_loss_factors():
    """The SA Energy Masters case: export_dlf copied from usage_dlf, so the RRP
    term cancels and the spread cannot depend on price."""
    idx = _index()
    net = _network(usage_dlf=1.0811, export_dlf=1.0811,
                   import_tlf=1.0012, export_tlf=1.0012)
    offsets = build_price_stack_offsets(idx, net, _wholesale_exposed(),
                                        STATE, INTERVAL)
    assert offsets.import_loss_factor == pytest.approx(offsets.export_loss_factor)
    validate_price_ordering(offsets)  # must not raise


def test_price_ordering_fails_when_export_dlf_is_lower():
    """import_lf > export_lf shrinks the spread as prices fall, so a real
    export loss factor can push the crossover inside the market range."""
    idx = _index()
    net = _network(usage_c=3.124, feedin_c=-1.1,
                   usage_dlf=1.0811, export_dlf=0.95,
                   import_tlf=1.0012, export_tlf=1.0012)
    with pytest.raises(ValueError, match="market price floor"):
        build_price_stack_offsets(idx, net, _wholesale_exposed(usage_other_c=4.081),
                                  STATE, INTERVAL)


def test_price_ordering_binding_case_is_the_floor_not_the_cap():
    """Regression guard: checking only the cap would pass a tariff that fails."""
    idx = _index()
    net = _network(usage_c=3.124, feedin_c=-1.1,
                   usage_dlf=1.0811, export_dlf=0.95,
                   import_tlf=1.0012, export_tlf=1.0012)
    lr = build_layer_rates(idx, pd.Series(0.0, index=idx), net,
                           _wholesale_exposed(usage_other_c=4.081), STATE, INTERVAL)
    offsets = PriceStackOffsets(
        import_loss_factor=net.import_loss_factor,
        export_loss_factor=net.export_loss_factor,
        offset_import=lr.network_import + lr.retail_import,
        offset_export=lr.network_export + lr.retail_export,
    )
    # cap alone: fine. floor: violation.
    validate_price_ordering(offsets, price_floor_mwh=0.0,
                            price_cap_mwh=MARKET_PRICE_CAP_MWH)
    with pytest.raises(ValueError, match="floor"):
        validate_price_ordering(offsets, price_floor_mwh=MARKET_PRICE_FLOOR_MWH,
                                price_cap_mwh=MARKET_PRICE_CAP_MWH)


def test_price_ordering_error_names_the_offending_interval():
    """Diagnostic quality: the message must say which interval and by how much."""
    idx = _index()
    net = _network(usage_c=3.124, feedin_c=-1.1, usage_dlf=1.0811,
                   export_dlf=0.95, import_tlf=1.0012, export_tlf=1.0012)
    with pytest.raises(ValueError) as exc:
        build_price_stack_offsets(idx, net, _wholesale_exposed(usage_other_c=4.081),
                                  STATE, INTERVAL)
    msg = str(exc.value)
    assert "interval" in msg and "$/kWh" in msg
    assert "import_loss_factor" in msg and "export_loss_factor" in msg

# ------------------------------------------- advanced base / critical-rate split

from model.core.prices import (  # noqa: E402
    build_base_price_vectors_advanced, build_lp_price_vectors_advanced,
    apply_critical_rates, critical_masks, block_tier_masks, block_window_mask,
)
from model.core.tariff import (  # noqa: E402
    BlockTariffWindow, BlockTier, CriticalRate,
)


def _advanced_tou(usage_c=20.0, feedin_c=5.0, usage_other_c=3.0,
                  feedin_other_c=0.0, critical=True):
    """A TOU plan with a block import window, a block export window, and
    optionally both critical rates.

    Window hours and critical thresholds and rates mirror GloBird ZeroHero so
    the test exercises the same shape as the reference plan.
    """
    return RetailTariff(
        name="advanced_tou", state="NSW", type="tou",
        daily_c_per_day=181.5,
        usage_tariffs=(TariffPeriod(0, 24, usage_c),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, feedin_c),),
        usage_other_c_per_kwh=usage_other_c,
        feedin_other_c_per_kwh=feedin_other_c,
        block_import_tiers=(
            BlockTariffWindow(12, 15, (BlockTier(50.0, 0.0001),
                                       BlockTier(np.inf, 30.8))),
        ),
        block_export_tiers=(
            BlockTariffWindow(18, 21, (BlockTier(15.0, 10.0),
                                       BlockTier(np.inf, 2.0))),
        ),
        critical_export_rate=(
            CriticalRate(2000.0, 100.0) if critical else None),
        critical_import_rate=(
            CriticalRate(-100.0, -5.0) if critical else None),
    )


def _threshold_crossing_price(index, seed=3):
    """Prices that land on both sides of both critical thresholds, outside
    and inside both block windows, including on the opposite side's window.

    A series that never crosses a threshold would let a broken critical-rate
    step pass, so the reconstruction test must not use a flat or narrow series.
    """
    prices = _varying_price(index, seed=seed)
    vals = prices.to_numpy().copy()
    vals[20:30] = 5000.0     # above the export threshold, outside windows
    vals[40:50] = -300.0     # below the import threshold, outside windows
    vals[60:70] = 200.0      # between the two, neither fires
    lwc = local_wall_clock(index, STATE, INTERVAL)
    hour, minute = np.asarray(lwc.hour), np.asarray(lwc.minute)
    vals[(hour == 12) & (minute < 30)] = -300.0   # import-critical in import window
    vals[(hour == 13) & (minute < 30)] = 5000.0   # export-critical in import window
    vals[(hour == 19) & (minute < 30)] = 5000.0   # export-critical in export window
    vals[(hour == 20) & (minute < 30)] = -300.0   # import-critical in export window
    return pd.Series(vals, index=index)


def _assert_crosses_both_thresholds(prices, retail):
    above = (prices.to_numpy()
             > retail.critical_export_rate.threshold_dollars_per_mwh).sum()
    below = (prices.to_numpy()
             < retail.critical_import_rate.threshold_dollars_per_mwh).sum()
    assert above > 0 and below > 0, (
        f"test price series does not exercise both critical rates: "
        f"{above} intervals above the export threshold, "
        f"{below} below the import threshold"
    )


def test_advanced_base_plus_critical_rates_reconstructs_the_public_vectors():
    """The rolling advanced path builds the base once per site and applies
    critical rates per window against the FORECAST price. If this fails, the
    rolling window and the perfect-foresight run are pricing different plans.
    """
    idx = _index()
    net, ret = _network(), _advanced_tou()
    prices = _threshold_crossing_price(idx)
    _assert_crosses_both_thresholds(prices, ret)

    base_imp, base_exp = build_base_price_vectors_advanced(
        idx, prices, net, ret, STATE, INTERVAL)
    got_imp, got_exp = apply_critical_rates(
        base_imp, base_exp, prices, ret, shift_opposite=True)
    ref_imp, ref_exp = build_lp_price_vectors_advanced(
        idx, prices, net, ret, STATE, INTERVAL)

    np.testing.assert_allclose(got_imp.to_numpy(), ref_imp.to_numpy(),
                               rtol=0, atol=0)
    np.testing.assert_allclose(got_exp.to_numpy(), ref_exp.to_numpy(),
                               rtol=0, atol=0)


def test_advanced_base_applies_block_masking():
    """Guards the reconstruction test above from passing vacuously: the base
    must actually differ from the unmasked vectors inside a block window."""
    idx = _index()
    net, ret = _network(), _advanced_tou()
    prices = _threshold_crossing_price(idx)
    base_imp, base_exp = build_base_price_vectors_advanced(
        idx, prices, net, ret, STATE, INTERVAL)

    lwc = local_wall_clock(idx, STATE, INTERVAL)
    in_import = (lwc.hour >= 12) & (lwc.hour < 15)
    in_export = (lwc.hour >= 18) & (lwc.hour < 21)
    assert in_import.any() and in_export.any()

    # inside: flat adder only, the tier variables carry the rest
    np.testing.assert_allclose(base_imp.to_numpy()[in_import], 0.03)
    np.testing.assert_allclose(base_exp.to_numpy()[in_export], 0.0)
    # outside: full TOU period rate plus adder
    np.testing.assert_allclose(base_imp.to_numpy()[~in_import], 0.23)
    np.testing.assert_allclose(base_exp.to_numpy()[~in_export], 0.05)


def test_advanced_base_does_not_depend_on_wholesale_price_for_tou():
    """The rolling advanced path builds the base ONCE for the whole period and
    reuses it for every window. That is only sound because a TOU plan's base
    vectors do not depend on the wholesale price."""
    idx = _index()
    net, ret = _network(), _advanced_tou()

    lo_imp, lo_exp = build_base_price_vectors_advanced(
        idx, pd.Series(np.full(len(idx), -900.0), index=idx),
        net, ret, STATE, INTERVAL)
    hi_imp, hi_exp = build_base_price_vectors_advanced(
        idx, pd.Series(np.full(len(idx), 12000.0), index=idx),
        net, ret, STATE, INTERVAL)

    np.testing.assert_array_equal(lo_imp.to_numpy(), hi_imp.to_numpy())
    np.testing.assert_array_equal(lo_exp.to_numpy(), hi_exp.to_numpy())


def test_advanced_base_equals_public_when_no_critical_rates():
    """The common TOU plan has no critical rates, so nothing in the price
    vectors reads the wholesale price and no price forecast is needed at all."""
    idx = _index()
    net, ret = _network(), _advanced_tou(critical=False)
    prices = _threshold_crossing_price(idx)

    base_imp, base_exp = build_base_price_vectors_advanced(
        idx, prices, net, ret, STATE, INTERVAL)
    ref_imp, ref_exp = build_lp_price_vectors_advanced(
        idx, prices, net, ret, STATE, INTERVAL)

    np.testing.assert_array_equal(base_imp.to_numpy(), ref_imp.to_numpy())
    np.testing.assert_array_equal(base_exp.to_numpy(), ref_exp.to_numpy())


def test_critical_rates_preserve_the_import_export_spread():
    """The opposite-side shift is what stops the LP importing and exporting in
    the same interval to harvest a critical rate, and it is why price
    ordering under critical rates needs no separate range analysis."""
    idx = _index()
    net, ret = _network(), _advanced_tou()
    prices = _threshold_crossing_price(idx)
    _assert_crosses_both_thresholds(prices, ret)

    base_imp, base_exp = build_base_price_vectors_advanced(
        idx, prices, net, ret, STATE, INTERVAL)
    amp_imp, amp_exp = build_lp_price_vectors_advanced(
        idx, prices, net, ret, STATE, INTERVAL)

    np.testing.assert_allclose(
        (amp_imp - amp_exp).to_numpy(), (base_imp - base_exp).to_numpy(),
        rtol=0, atol=1e-15)


def _critical_positions(prices, retail):
    exp_mask, imp_mask = critical_masks(prices, retail)
    assert exp_mask.any() and imp_mask.any(), "fixture must cross both thresholds"
    return exp_mask, imp_mask


@pytest.mark.parametrize('shift_opposite', [True, False])
def test_critical_rates_replace_the_critical_side(shift_opposite):
    """The critical side takes rho (critical rate plus that side's flat adder)
    whatever the base rate was, inside a block window or not. The other side
    moves by the same delta for the LP, and not at all for billing."""
    idx = _index()
    net, ret = _network(), _advanced_tou()
    prices = _threshold_crossing_price(idx)
    exp_mask, imp_mask = _critical_positions(prices, ret)
    base_imp, base_exp = build_base_price_vectors_advanced(
        idx, prices, net, ret, STATE, INTERVAL)
    got_imp, got_exp = apply_critical_rates(
        base_imp, base_exp, prices, ret, shift_opposite=shift_opposite)
    b_imp, b_exp = base_imp.to_numpy(), base_exp.to_numpy()
    g_imp, g_exp = got_imp.to_numpy(), got_exp.to_numpy()

    rho_x = (100.0 + 0.0) / 100      # critical export rate + feedin_other
    rho_m = (-5.0 + 3.0) / 100       # critical import rate + usage_other
    np.testing.assert_allclose(g_exp[exp_mask], rho_x, rtol=0, atol=1e-15)
    np.testing.assert_allclose(g_imp[imp_mask], rho_m, rtol=0, atol=1e-15)

    if shift_opposite:
        np.testing.assert_allclose(
            g_imp[exp_mask], b_imp[exp_mask] + (rho_x - b_exp[exp_mask]),
            rtol=0, atol=1e-15)
        np.testing.assert_allclose(
            g_exp[imp_mask], b_exp[imp_mask] + (rho_m - b_imp[imp_mask]),
            rtol=0, atol=1e-15)
    else:
        np.testing.assert_array_equal(g_imp[exp_mask], b_imp[exp_mask])
        np.testing.assert_array_equal(g_exp[imp_mask], b_exp[imp_mask])

    neither = ~exp_mask & ~imp_mask
    np.testing.assert_array_equal(g_imp[neither], b_imp[neither])
    np.testing.assert_array_equal(g_exp[neither], b_exp[neither])


def test_block_tier_masks_remove_only_the_same_side_critical_intervals():
    """Critical flow leaves the tier on its own side only: a critical export
    inside the import window, or a critical import inside the export window,
    stays in that window's tier."""
    idx = _index()
    ret = _advanced_tou()
    prices = _threshold_crossing_price(idx)
    exp_mask, imp_mask = _critical_positions(prices, ret)
    plain_imp = np.asarray(block_window_mask(idx, ret.block_import_tiers[0], STATE, INTERVAL))
    plain_exp = np.asarray(block_window_mask(idx, ret.block_export_tiers[0], STATE, INTERVAL))
    # Non-vacuity: each kind of overlap is present in the fixture.
    assert (plain_imp & imp_mask).any() and (plain_imp & exp_mask).any()
    assert (plain_exp & exp_mask).any() and (plain_exp & imp_mask).any()

    (eff_imp,), (eff_exp,) = block_tier_masks(idx, ret, STATE, INTERVAL, prices)
    np.testing.assert_array_equal(eff_imp, plain_imp & ~imp_mask)
    np.testing.assert_array_equal(eff_exp, plain_exp & ~exp_mask)


def test_block_tier_masks_needs_prices_only_with_critical_rates():
    idx = _index()
    with pytest.raises(ValueError, match='prices_mwh is required'):
        block_tier_masks(idx, _advanced_tou(), STATE, INTERVAL, None)
    (eff_imp,), (eff_exp,) = block_tier_masks(
        idx, _advanced_tou(critical=False), STATE, INTERVAL, None)
    np.testing.assert_array_equal(
        eff_imp, block_window_mask(idx, _advanced_tou().block_import_tiers[0],
                                   STATE, INTERVAL))


# ------------------------------------------------- mixed-plan dispatch prices
# model_spec.md, Mixed-plan dispatch prices.

from model.core.prices import (  # noqa: E402
    MIXED_CURTAIL_COST_PER_KWH, build_mixed_dispatch_stack,
    dispatch_curtail_cost, mixed_export_rate, validate_mixed_price_ordering,
    MARKET_PRICE_CAP_MWH, MARKET_PRICE_FLOOR_MWH,
)

_TESLA_LIKE_PERIODS = ((0, 15, 24.02), (15, 21, 49.71), (21, 24, 24.02))


def _mixed(periods=_TESLA_LIKE_PERIODS, usage_other_c=0.821, feedin_other_c=0.0,
           share=0.9, basis='wholesale', floor_c=0.0):
    return RetailTariff(
        name="mixed", state="NSW", type="mixed", daily_c_per_day=110.0,
        usage_tariffs=tuple(TariffPeriod(*p) for p in periods),
        usage_tariffs_weekend=None, feedin_tariffs=(),
        usage_other_c_per_kwh=usage_other_c,
        feedin_other_c_per_kwh=feedin_other_c,
        wholesale_export_share=share, export_share_basis=basis,
        export_floor_c_per_kwh=floor_c,
    )


def _spiky_prices(index, seed=11):
    """Ordinary prices with spikes, zeros and negatives down to the floor."""
    rng = np.random.default_rng(seed)
    p = rng.normal(90.0, 120.0, len(index))
    p[::37] = 0.0
    p[5::41] = rng.uniform(1000.0, 17000.0, len(p[5::41]))
    p[11::29] = rng.uniform(-1000.0, -1.0, len(p[11::29]))
    assert (p < 0).any() and (p == 0).any() and (p > 1000).any()
    return pd.Series(p, index=index)


def _tou_by_hour(index, periods):
    hours = local_wall_clock(index, STATE, INTERVAL).hour
    out = np.full(len(index), np.nan)
    for start, end, c in periods:
        out[(hours >= start) & (hours < end)] = c
    assert not np.isnan(out).any()
    return out / 100.0


@pytest.mark.parametrize("basis", ["wholesale", "wholesale_and_network"])
@pytest.mark.parametrize("floor_c", [0.0, 2.0, None])
def test_mixed_dispatch_prices_follow_the_formula(basis, floor_c):
    """p_imp = TOU + usage_other + max(0, RRP)/1000 * import_lf, and p_exp is
    the floored share, computed here from the definitions rather than
    through build_layer_rates."""
    idx = _index()
    net = _network(feedin_c=-1.5, usage_dlf=1.04, import_tlf=1.01,
                   export_dlf=0.98, export_tlf=0.99)
    rt = _mixed(basis=basis, floor_c=floor_c, feedin_other_c=0.3)
    prices = _spiky_prices(idx)
    p_imp, p_exp = build_price_vectors(idx, prices, net, rt, STATE, INTERVAL)

    lf_i, lf_e = 1.04 * 1.01, 0.98 * 0.99
    rrp = prices.to_numpy() / 1000.0
    exp_imp = (_tou_by_hour(idx, _TESLA_LIKE_PERIODS) + 0.00821
               + np.maximum(0.0, rrp) * lf_i)
    v = rrp * lf_e + (lf_e * -0.015 if basis == "wholesale_and_network" else 0.0)
    credit = 0.9 * v
    if floor_c is not None:
        credit = np.maximum(floor_c / 100.0, credit)
    exp_exp = credit + 0.003

    np.testing.assert_allclose(p_imp.to_numpy(), exp_imp, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(p_exp.to_numpy(), exp_exp, rtol=1e-12, atol=1e-15)


def test_mixed_dispatch_export_is_the_billed_export_and_import_is_not():
    """The export price an optimiser sees is exactly the credit billing pays.
    The import price is the billed rate plus the positive spot term."""
    idx = _index()
    net = _network()
    rt = _mixed()
    prices = _spiky_prices(idx)
    p_imp, p_exp = build_price_vectors(idx, prices, net, rt, STATE, INTERVAL)
    lr = build_layer_rates(idx, prices, net, rt, STATE, INTERVAL)
    np.testing.assert_array_equal(p_exp.to_numpy(), lr.retail_export)
    spot_term = np.maximum(0.0, prices.to_numpy()) / 1000.0 * net.import_loss_factor
    np.testing.assert_allclose(p_imp.to_numpy() - lr.retail_import, spot_term,
                               rtol=1e-12, atol=1e-15)
    assert (spot_term > 0).any() and (spot_term == 0).any()


def test_mixed_window_prices_are_slices_of_the_whole_period():
    idx = _index()
    net, rt = _network(), _mixed()
    prices = _spiky_prices(idx)
    full_imp, full_exp = build_price_vectors(idx, prices, net, rt, STATE, INTERVAL)
    stack = build_mixed_dispatch_stack(idx, net, rt, STATE, INTERVAL)
    for start, n in ((0, 216), (37, 1), (100, 188)):
        w_imp, w_exp = stack.window_price_vectors(
            prices.to_numpy()[start:start + n], start)
        np.testing.assert_array_equal(w_imp, full_imp.to_numpy()[start:start + n])
        np.testing.assert_array_equal(w_exp, full_exp.to_numpy()[start:start + n])
    with pytest.raises(ValueError, match="falls outside"):
        stack.window_price_vectors(np.zeros(10), len(idx) - 5)


def test_mixed_dispatch_stack_rejects_other_types():
    with pytest.raises(ValueError, match="needs a 'mixed' retail tariff"):
        build_mixed_dispatch_stack(_index(), _network(), _tou(), STATE, INTERVAL)


def test_dispatch_curtail_cost():
    assert dispatch_curtail_cost(_mixed()) == MIXED_CURTAIL_COST_PER_KWH
    assert dispatch_curtail_cost(_tou()) == 0.0
    assert dispatch_curtail_cost(_wholesale_exposed()) == 0.0


def test_mixed_ordering_fails_only_at_zero():
    """No floor, a zero TOU rate and a positive network export rate in V:
    at RRP = 0 the import price is 0 and the export credit is
    share * network export > 0. At the market floor and the cap the spread is
    positive, so only the 0 candidate catches it."""
    idx = _index(12)
    net = _network(feedin_c=2.0, export_dlf=1.0, usage_dlf=1.0)
    rt = _mixed(periods=((0, 24, 0.0),), usage_other_c=0.0,
                basis="wholesale_and_network", floor_c=None)
    with pytest.raises(ValueError, match=r"at 0 \$/MWh"):
        build_mixed_dispatch_stack(idx, net, rt, STATE, INTERVAL)
    # Non-vacuity: the other two candidates pass on their own.
    for rrp in (MARKET_PRICE_FLOOR_MWH, MARKET_PRICE_CAP_MWH):
        lr = build_layer_rates(idx, pd.Series(np.full(12, rrp), index=idx),
                               net, rt, STATE, INTERVAL)
        p_imp = lr.retail_import + max(0.0, rrp) / 1000.0
        assert (p_imp - lr.retail_export > 0).all(), rrp


def test_mixed_ordering_fails_at_the_cap():
    """An export loss factor above the import loss factor at a share of 1:
    the spread falls with price and goes negative at the cap only."""
    idx = _index(12)
    net = _network(usage_dlf=1.0, export_dlf=1.2)
    rt = _mixed(share=1.0)
    with pytest.raises(ValueError, match="at the market cap"):
        build_mixed_dispatch_stack(idx, net, rt, STATE, INTERVAL)


def test_mixed_ordering_error_carries_its_context():
    idx = _index(12)
    net = _network(usage_dlf=1.0, export_dlf=1.2)
    with pytest.raises(ValueError) as e:
        build_mixed_dispatch_stack(idx, net, _mixed(share=1.0), STATE, INTERVAL)
    msg = str(e.value)
    for part in ("interval", "p_imp=", "p_exp=", "import_loss_factor=1.0",
                 "export_loss_factor=1.2", "share=1", "basis=wholesale"):
        assert part in msg, (part, msg)


@pytest.mark.parametrize("state, network_id", [
    ("NSW", "NSW_ausgrid_ea025_ea029_2026"),
    ("QLD", "QLD_energex_6900_2026"),
    ("SA", "SA_sapn_rtou_2026"),
    ("VIC", "VIC_united_energy_URSTOU_2026"),
])
def test_real_tesla_tariffs_pass_the_ordering_proof(state, network_id):
    """The production plan with the network tariffs the full runs use.
    Asserts the files exist rather than skipping, so a rename fails here."""
    from pathlib import Path
    from model.core.tariff import load_network_tariff, load_retail_tariff
    tdir = Path(__file__).parent.parent / "tariffs"
    tid = f"{state}_tesla_electric_mixed_2026"
    for name in (tid, network_id):
        assert (tdir / f"{name}.yaml").exists(), name
    rt = load_retail_tariff(tid, tdir)
    net = load_network_tariff(network_id, tdir)
    idx = pd.date_range("2024-01-01 00:05", periods=288 * 7, freq="5min",
                        tz="Etc/GMT-10")
    stack = build_mixed_dispatch_stack(idx, net, rt, state, INTERVAL)
    validate_mixed_price_ordering(stack)      # explicit, as well as inside
