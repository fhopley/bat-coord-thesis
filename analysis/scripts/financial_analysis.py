"""Financial analysis: household outcomes and wholesale market impact.

Produces, for every scenario the script can find under --data-root, and for
each run basis:

  Tier 1 - one figure set per (state, year) scenario found.
  Tier 2 - for any state with more than one year of data, an additional
           figure set that is the straight (unweighted) mean of that
           state's per-year results across years.
  Tier 3 - if more than one state is present, an additional national
           figure set combining each state's tier-2 (or tier-1, if only
           one year exists) result using the state_mix weights from the
           weighting YAML. Only produced when weighting is enabled, since
           state_mix is itself a weighting concept.

Each figure set covers:

  Section 1 - Household outcomes
    household_savings_default_mode
    household_savings_optimised_mode
    household_savings_full_range

  Section 2 - Wholesale market impact
    wholesale_savings

  with small-multiples grids of the optimised view by state, by state and
  year, and by year within each state.

  Section 3 - Retailer outcomes: not yet implemented. The loading and
  weighting machinery below is written so that section can be added later
  without needing its own data path.

Run bases (see RUN_BASES): perfect_foresight, conservative and achievable
imperfect foresight. Each is written to its own subdirectory of
--output-dir, with its curves as curves_<weighting>.csv, and named in every
figure title. With both imperfect foresight bases drawn, the national
optimised household and wholesale views are also drawn side by side in
imperfect_foresight_comparison/.

Figures follow docs/thesis_figure_spec.md through thesis_style. The data
notes (n, period, weighting, baseline) are drawn for presentation use and
printed for captions; --no-notes leaves them off for the thesis.

See docs/financial_analysis_spec.md for the full design rationale.

Usage (from project root):
    python3 analysis/scripts/financial_analysis.py --data-root "/Volumes/Manual/Thesis Project"
    python3 analysis/scripts/financial_analysis.py --data-root "/Volumes/Manual/Thesis Project" --no-notes
    python3 analysis/scripts/financial_analysis.py --basis achievable
    python3 analysis/scripts/financial_analysis.py --unweighted
    python3 analysis/scripts/financial_analysis.py --states NSW --years 2022
    python3 analysis/scripts/financial_analysis.py --only-plan ea

--only-plan is a presentation override for showing one retailer in isolation
(e.g. a meeting with that retailer). It restricts every chart to the named
plan(s) and moves the single self-consumption reference line off Amber onto
the first plan named. It writes to the normal output paths, so it overwrites
the full-registry charts; re-run without the flag to restore them. See the
SC_REFERENCE_MODE block below for what it does and does not change.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import make_interp_spline

import thesis_style as ts

mpl.rcParams.update({
    'font.family':      'serif',
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'legend.fontsize':  9,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right': False,
    'axes.grid':        True,
    'grid.alpha':       0.35,
    'grid.linewidth':   0.5,
})

BATTERY_SIZES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# Sites to exclude from all analysis. Defined once in site_exclusions.py --
# see there for the criterion and the measured figures. Do not redefine it
# here: three separate copies is how W0164 came to be excluded from the cap
# account results and included in these.
from site_exclusions import EXCLUDED_SITES

# The study-wide savings baseline: solar, no battery, on the Amber plan.
# Also used to enumerate scenarios (it is the run most likely to be complete)
# and, under --baseline own, as Amber's own solar-only run — which is why
# Amber's zero_batt_dir below is None rather than naming a run of its own.
DEFAULT_BASELINE_RUN = 'full_run_solar_only_amber'

# ── Single-plan override (--only-plan) ────────────────────────────────────────
#
# Presentation-only override for showing one retailer's results in isolation
# (e.g. a meeting with that retailer). Set from the command line in main(),
# never at import time; with no flag the module behaves exactly as before.
#
# It changes two things, and nothing else:
#
#   1. RETAIL_PLANS is narrowed *in place* to the named plan(s). Every
#      drawing function iterates RETAIL_PLANS, so filtering the registry
#      once filters every chart by construction, rather than each chart
#      needing its own filter to be kept in sync.
#   2. SC_REFERENCE_MODE moves off Amber onto the first named plan. The
#      charts that carry a single self-consumption reference line
#      (household_savings_optimised_mode, wholesale_savings, and the
#      small-multiples grids) then show that plan's self-consumption series
#      instead of Amber's — otherwise those charts would show an Amber line
#      on a figure with no other Amber content.
#
# Not changed: which sites, weights, or runs are loaded. The savings
# baseline is a separate concern, controlled by --baseline; note that
# --only-plan defaults it to 'own' (see resolve_baseline_mode).

SC_REFERENCE_MODE = 'SC'

# Command-line names for --only-plan, mapped to the sc_mode that identifies
# each plan in the registry below. Kept separate from RETAIL_PLANS so the
# registry stays exactly as it was.
PLAN_KEYS = {
    'amber':       'SC',
    'globird':     'SC_GB',
    'agl':         'SC_AGL',
    'ea':          'SC_EA',
    'origin_lite': 'SC_ORIGIN',
}

# ── Retail plan registry ──────────────────────────────────────────────────────
#
# Same fields as the original script's RETAIL_PLANS, plus:
#   excluded_states   set of state codes where this plan should NOT be
#                     expected / loaded / included in pooling. Empty set
#                     means available everywhere (the default today).
#
RETAIL_PLANS = [
    dict(
        color         = '#2bffa4',
        sc_mode       = 'SC',
        lp_mode       = 'LP',
        sc_label      = 'Amber Self Consumption',
        lp_label      = 'Amber Optimised',
        sc_dir        = 'full_run_self_consumption_amber',
        sc_no_vpp_dir = 'full_run_self_consumption_amber',
        lp_dir        = 'full_run_optimised_amber',
        zero_batt_dir = None,
        zero_batt_key = None,
        default_mode  = 'lp_dir',
        opt_type      = 'vpp',
        excluded_states = set(),
    ),
    dict(
        color         = '#e5037f',
        sc_mode       = 'SC_GB',
        lp_mode       = 'LP_GB',
        sc_label      = 'GloBird ZeroHero Self Consumption',
        lp_label      = 'GloBird ZeroHero Optimised',
        sc_dir        = 'full_run_self_consumption_globird',
        sc_no_vpp_dir = 'full_run_self_consumption_globird_novpp',
        lp_dir        = 'full_run_optimised_globird',
        zero_batt_dir = 'full_run_solar_only_globird',
        zero_batt_key = 'globird',
        default_mode  = 'sc_dir',
        opt_type      = 'vpp',
        excluded_states = set(),
    ),
    dict(
        color         = '#1018a1',
        sc_mode       = 'SC_AGL',
        lp_mode       = 'LP_AGL',
        sc_label      = 'AGL Battery Rewards Self Consumption',
        lp_label      = 'AGL Battery Rewards Optimised',
        sc_dir        = 'full_run_self_consumption_agl',
        sc_no_vpp_dir = 'full_run_self_consumption_agl_novpp',
        lp_dir        = 'full_run_optimised_agl',
        zero_batt_dir = 'full_run_solar_only_agl',
        zero_batt_key = 'agl',
        default_mode  = 'sc_dir',
        opt_type      = 'user',
        excluded_states = set(),
    ),
    dict(
        color         = '#197e2b',
        sc_mode       = 'SC_EA',
        lp_mode       = 'LP_EA',
        sc_label      = 'EnergyAustralia Battery Ease Self Consumption',
        lp_label      = 'EnergyAustralia Battery Ease Optimised',
        sc_dir        = 'full_run_self_consumption_ea',
        sc_no_vpp_dir = 'full_run_self_consumption_ea',
        lp_dir        = 'full_run_optimised_ea',
        zero_batt_dir = 'full_run_solar_only_ea',
        zero_batt_key = 'ea',
        default_mode  = 'lp_dir',
        opt_type      = 'vpp',
        # Battery Ease was offered in NSW only. The QLD, SA and VIC runs model
        # estimated versions of it and are left out, so its national results
        # are its NSW results.
        excluded_states = {'QLD', 'SA', 'VIC'},
    ),
    dict(
        color         = '#ffba46',
        sc_mode       = 'SC_ORIGIN',
        lp_mode       = 'LP_ORIGIN',
        sc_label      = 'Origin Battery Lite Self Consumption',
        lp_label      = 'Origin Battery Lite Optimised',
        sc_dir        = 'full_run_self_consumption_origin_lite',
        sc_no_vpp_dir = 'full_run_self_consumption_origin_novpp',
        lp_dir        = 'full_run_optimised_origin_lite',
        zero_batt_dir = 'full_run_solar_only_origin_starter',
        zero_batt_key = 'origin',
        default_mode  = 'sc_dir',
        opt_type      = 'user',
        excluded_states = set(),
    ),
]


# ── Run bases ─────────────────────────────────────────────────────────────────
#
# Every figure set is drawn on one run basis, which sets which runs fill each
# plan's sc_dir and lp_dir:
#
#   perfect_foresight  the registry above: perfect foresight optimisation,
#                      with maximum VPP dispatch where the plan has a VPP.
#   conservative       imperfect foresight on pre-dispatch forecasts available
#                      before each interval starts, and no VPP dispatch. A
#                      VPP plan's no-dispatch case is its _novpp run plus the
#                      enrolment payment it would still receive.
#   achievable         imperfect foresight using each interval's dispatch
#                      price and refreshed pre-dispatch, with maximum
#                      realistic VPP dispatch.
#
# 'folder' is the output subdirectory and 'title' the text added to every
# figure title.
RUN_BASES = {
    'perfect_foresight': dict(folder='perfect_foresight',
                              title='perfect foresight'),
    'conservative': dict(folder='imperfect_foresight_conservative',
                         title='imperfect foresight (conservative)'),
    'achievable': dict(folder='imperfect_foresight_achievable',
                       title='imperfect foresight (achievable)'),
}

# (sc_dir, lp_dir) for each plan, by sc_mode, on each basis other than
# perfect_foresight, which uses the registry as it stands. Rule-based self
# consumption has no foresight, so it is the same run on every basis unless
# the plan's self-consumption mode can take part in VPP events (AGL, Origin).
PLAN_RUNS = {
    'SC': {
        'conservative': ('full_run_self_consumption_amber',
                         'full_run_rolling_amber_predispatchonly'),
        'achievable':   ('full_run_self_consumption_amber',
                         'full_run_rolling_amber'),
    },
    'SC_GB': {
        'conservative': ('full_run_self_consumption_globird',
                         'full_run_rolling_globird_novpp'),
        'achievable':   ('full_run_self_consumption_globird',
                         'full_run_rolling_globird'),
    },
    'SC_AGL': {
        'conservative': ('full_run_self_consumption_agl_novpp',
                         'full_run_rolling_agl_novpp'),
        'achievable':   ('full_run_rolling_self_consumption_agl',
                         'full_run_rolling_agl'),
    },
    'SC_EA': {
        'conservative': ('full_run_self_consumption_ea',
                         'full_run_rolling_ea_novpp'),
        'achievable':   ('full_run_self_consumption_ea',
                         'full_run_rolling_ea'),
    },
    'SC_ORIGIN': {
        'conservative': ('full_run_self_consumption_origin_novpp',
                         'full_run_rolling_origin_novpp'),
        'achievable':   ('full_run_rolling_self_consumption_origin',
                         'full_run_rolling_origin'),
    },
}

# Annual VPP enrolment payments, received whether or not the battery is
# dispatched. The with-VPP runs include them; the _novpp runs do not, so on
# the conservative basis they are added to the household saving of the modes
# whose run is a _novpp run standing in for an enrolled, undispatched
# battery. EnergyAustralia's self consumption is not enrolled: its VPP runs
# the battery in TOU mode between events.
ENROLMENT_PAYMENTS = {'agl': 80.0, 'ea': 15.0 * 12}
CONSERVATIVE_PAYMENTS = {
    'SC_AGL': ENROLMENT_PAYMENTS['agl'],
    'LP_AGL': ENROLMENT_PAYMENTS['agl'],
    'LP_EA': ENROLMENT_PAYMENTS['ea'],
}

# Runs that have not completed, built from runs that have, keyed by basis
# then mode. The mode's slot loads the 'upper' run named in PLAN_RUNS, and its
# curves become
#     (1 - capture) * base + capture * upper,
# which scales the upper run's value capture ratio over base by capture.
# An entry is dict(base=<mode key>, capture=<float>, run=<name of the run it
# stands in for>). Delete the entry, and name the real run in PLAN_RUNS, once
# it exists. Empty while every run is complete.
PLACEHOLDERS: dict[str, dict[str, dict]] = {}

# Darker step of a plan's colour for marker edges, where the colour itself
# is too light to outline a marker on white.
PLAN_INK = {'SC': '#0e8f5c'}

_BASE_PLANS = [dict(p) for p in RETAIL_PLANS]


def plans_for_basis(basis: str) -> list[dict]:
    """The registry with each plan's runs set for one run basis."""
    plans = []
    for base in _BASE_PLANS:
        plan = dict(base)
        runs = PLAN_RUNS.get(plan['sc_mode'], {}).get(basis)
        if runs is not None:
            plan['sc_dir'], plan['lp_dir'] = runs
        plans.append(plan)
    return plans


# Suffix of the mode holding a plan's no-VPP self-consumption curve, where it
# is not the plan's self-consumption mode itself (see sc_no_vpp_mode).
NO_VPP_SUFFIX = '_NO_VPP'


def sc_no_vpp_mode(plan: dict) -> str:
    """Mode key of the plan's self-consumption curve without VPP
    participation: no VPP events, critical rates or enrolment payment.

    That is the plan's self-consumption mode where its sc_dir is already the
    no-VPP run and carries no conservative enrolment payment, and otherwise
    a further mode, {sc_mode}_NO_VPP, loaded from sc_no_vpp_dir. A plan with
    no sc_no_vpp_dir (e.g. one built by an external script) uses its
    self-consumption mode."""
    run = plan.get('sc_no_vpp_dir')
    if run is None or (run == plan['sc_dir']
                       and plan['sc_mode'] not in CONSERVATIVE_PAYMENTS):
        return plan['sc_mode']
    return plan['sc_mode'] + NO_VPP_SUFFIX


def blend_capture(base, upper, capture: float) -> np.ndarray:
    """(1 - capture) * base + capture * upper: a curve whose value capture
    ratio over base is capture times that of upper."""
    return ((1 - capture) * np.asarray(base, float)
            + capture * np.asarray(upper, float))


def apply_basis_adjustments(curves: dict, basis: str) -> None:
    """Placeholders and enrolment payments for one scenario's curve set, in
    place. Both are linear in the site savings, so applying them to the
    curve set gives the same result as applying them site by site, and they
    pass unchanged through the averaging across years and states."""
    for mode, p in PLACEHOLDERS.get(basis, {}).items():
        for metric_curves in curves.values():
            if mode in metric_curves and p['base'] in metric_curves:
                metric_curves[mode] = blend_capture(
                    metric_curves[p['base']], metric_curves[mode],
                    p['capture'])
    if basis == 'conservative':
        hh = curves.get('hh_saving', {})
        for mode, payment in CONSERVATIVE_PAYMENTS.items():
            if mode in hh:
                # No battery, nothing to enrol: the zero-capacity anchor
                # keeps no payment.
                hh[mode] = np.asarray(hh[mode], float).copy()
                hh[mode][1:] += payment


def apply_site_basis_adjustments(df: pd.DataFrame, basis: str) -> pd.DataFrame:
    """apply_basis_adjustments for site rows (site_id, battery_kwh, mode,
    hh_saving, ws_saving), which carry no zero-capacity row to exempt from
    the payments."""
    df = df.copy()
    for mode, p in PLACEHOLDERS.get(basis, {}).items():
        key = ['site_id', 'battery_kwh']
        base = df[df['mode'] == p['base']].set_index(key)
        rows = df['mode'] == mode
        upper = df[rows].set_index(key)
        for col in ('hh_saving', 'ws_saving'):
            df.loc[rows, col] = blend_capture(base[col].reindex(upper.index),
                                              upper[col], p['capture'])
        # A site without the base run has no blended value.
        missing = base['hh_saving'].reindex(upper.index).isna().to_numpy()
        if missing.any():
            df = df.drop(df[rows].index[missing])
    if basis == 'conservative':
        for mode, payment in CONSERVATIVE_PAYMENTS.items():
            df.loc[df['mode'] == mode, 'hh_saving'] += payment
    return df


# ── Scenario discovery ────────────────────────────────────────────────────────

_SCENARIO_RE = re.compile(r'^([A-Z]{2,3})_(\d{4})$')


def discover_scenarios(data_root: Path) -> list[tuple[str, str]]:
    """Find (state, year) pairs with a solar-only baseline present.

    The solar-only baseline directory has no battery-size suffix
    ({STATE}_{YEAR}), so scanning it directly gives the set of scenarios
    that are at least minimally complete. Individual retail-plan runs are
    checked for existence later, per plan, at load time.
    """
    baseline_scenarios = (data_root / 'data' / 'output' / 'runs'
                          / DEFAULT_BASELINE_RUN / 'scenarios')
    if not baseline_scenarios.exists():
        return []
    found = []
    for d in sorted(baseline_scenarios.iterdir()):
        if not d.is_dir():
            continue
        m = _SCENARIO_RE.match(d.name)
        if m:
            found.append((m.group(1), m.group(2)))
    return found


# ── Weighting config (YAML) ───────────────────────────────────────────────────

def load_weights_config(path: Path) -> dict:
    """Load and validate the size-band weighting YAML.

    Returns a dict with keys:
      bands       list of (label, lo, hi) tuples in size_bands order
      by_state    dict[state] -> list[float] aligned to bands
      state_mix   dict[state] -> float
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    labels = raw['size_bands']
    bands = []
    for label in labels:
        lo_str, hi_str = label.split('-')
        bands.append((label, float(lo_str), float(hi_str)))

    by_state = raw['by_state']
    for state, weights in by_state.items():
        if len(weights) != len(bands):
            raise ValueError(
                f"Weights config error: by_state['{state}'] has "
                f"{len(weights)} entries but size_bands has {len(bands)}. "
                f"These must be the same length and in the same order."
            )

    return {'bands': bands, 'by_state': by_state, 'state_mix': raw['state_mix']}


# ── Site metadata and size-band assignment ────────────────────────────────────
#
# System size (size_kw) is resolved in three tiers, in order of trust:
#
#   1. Direct: subarray_dc_rating_kw is trusted only when there is an
#      inverter_ac_rating_kw to check it against, and it is >= that inverter
#      rating. A subarray reading below the inverter's AC rating usually
#      means metadata only captured one of several DC strings, not the
#      site's true array size, so it is never trusted on its own.
#   2. Ratio: for sites with an inverter rating but no trusted subarray
#      value, size_kw is estimated as inverter_ac_rating_kw times a single
#      pooled (national, not per-state) median DC:AC oversizing ratio taken
#      from the tier-1 trusted sites. This ratio reflects a hardware/design
#      convention, not a climate-dependent quantity, so pooling nationally
#      is appropriate and gives a larger, more stable sample than any one
#      state alone.
#   3. Regression: for sites with no inverter rating at all (so tiers 1 and
#      2 don't apply), size_kw is estimated by regressing tier-1 and tier-2
#      sites' size_kw against annual_generation_kwh, fit per state (specific
#      yield is climate-dependent, unlike the DC:AC ratio), falling back to
#      a pooled all-state fit if a state doesn't have enough tier-1/2 sites
#      of its own to fit a reliable line.

def load_site_meta(data_root: Path, year: str, state: str) -> pd.DataFrame:
    """Load site metadata for one state/year: site_id, subarray_dc_rating_kw,
    inverter_ac_rating_kw, annual_generation_kwh.
    """
    meta_f = (data_root / 'data' / 'processed' / 'edp'
              / f'{year}_selected_sites_clean.csv')
    meta = pd.read_csv(meta_f, usecols=[
        'edp_site_id', 'state', 'subarray_dc_rating_kw',
        'inverter_ac_rating_kw', 'annual_generation_kwh',
    ])
    meta = meta[meta['state'] == state].copy()
    meta = meta.rename(columns={'edp_site_id': 'site_id'})
    meta = meta[~meta['site_id'].isin(EXCLUDED_SITES)]
    return meta[['site_id', 'subarray_dc_rating_kw',
                'inverter_ac_rating_kw', 'annual_generation_kwh']]


def compute_dc_ac_ratio(all_states_meta: pd.DataFrame, min_trusted: int = 10) -> float:
    """Pooled (national) median DC:AC oversizing ratio from the tier-1
    trusted subset: sites where subarray_dc_rating_kw >= inverter_ac_rating_kw
    (both present). Raises if there aren't enough trusted sites to trust
    the resulting ratio, unless no site needs it (every site with an inverter
    rating is tier 1), in which case it returns NaN. That case arises with the
    sample household H0001 alone.
    """
    both = all_states_meta.dropna(subset=['subarray_dc_rating_kw', 'inverter_ac_rating_kw'])
    trusted = both[both['subarray_dc_rating_kw'] >= both['inverter_ac_rating_kw']]
    needs_ratio = (all_states_meta['inverter_ac_rating_kw'].notna()
                   & ~all_states_meta.index.isin(trusted.index))
    if len(trusted) < min_trusted and not needs_ratio.any():
        return float('nan')
    if len(trusted) < min_trusted:
        raise RuntimeError(
            f"Only {len(trusted)} trusted sites (subarray >= inverter) found "
            f"nationally; need at least {min_trusted} to compute a reliable "
            f"DC:AC ratio."
        )
    return float((trusted['subarray_dc_rating_kw']
                 / trusted['inverter_ac_rating_kw']).median())


def assign_direct_size(meta: pd.DataFrame, dc_ac_ratio: float) -> pd.DataFrame:
    """Adds 'size_kw' and 'size_source' columns using the tier-1/tier-2 rule
    above. size_source is 'direct', 'ratio', or 'regression' (meaning: still
    needs the regression step, no inverter rating available at all).
    """
    meta = meta.copy()
    has_both = meta['subarray_dc_rating_kw'].notna() & meta['inverter_ac_rating_kw'].notna()
    trusted_mask = has_both & (meta['subarray_dc_rating_kw']
                               >= meta['inverter_ac_rating_kw'])
    ratio_mask = meta['inverter_ac_rating_kw'].notna() & ~trusted_mask

    meta['size_kw'] = np.nan
    meta['size_source'] = 'regression'
    meta.loc[trusted_mask, 'size_kw'] = meta.loc[trusted_mask, 'subarray_dc_rating_kw']
    meta.loc[trusted_mask, 'size_source'] = 'direct'
    meta.loc[ratio_mask, 'size_kw'] = (meta.loc[ratio_mask, 'inverter_ac_rating_kw']
                                       * dc_ac_ratio)
    meta.loc[ratio_mask, 'size_source'] = 'ratio'
    return meta


def estimate_missing_sizes(meta: pd.DataFrame, state: str,
                           fallback_meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fill size_kw for sites still marked size_source == 'regression' (no
    inverter rating at all) via a per-state linear regression of size_kw
    (from tier-1/tier-2 sites only) against annual_generation_kwh. Falls
    back to a pooled (all-state) regression if this state doesn't have
    enough tier-1/2 sites of its own to fit a reliable line.

    fallback_meta, if given, must already have size_kw/size_source assigned
    (i.e. have been through assign_direct_size with the same dc_ac_ratio).
    """
    meta = meta.copy()
    meta['size_estimated'] = meta['size_source'] != 'direct'

    fit_source = meta[meta['size_source'] != 'regression'].dropna(
        subset=['annual_generation_kwh'])
    MIN_FIT_SITES = 5
    if len(fit_source) < MIN_FIT_SITES and fallback_meta is not None:
        pooled_fit = fallback_meta[fallback_meta['size_source'] != 'regression'].dropna(
            subset=['annual_generation_kwh'])
        if len(pooled_fit) >= MIN_FIT_SITES:
            fit_source = pooled_fit
            print(f"  NOTE: {state} has only {len(meta[meta['size_source'] != 'regression'])} "
                  f"tier-1/2 sites; using pooled all-state regression "
                  f"({len(pooled_fit)} sites) for the remaining size estimates.",
                  file=sys.stderr)

    missing_mask = (meta['size_source'] == 'regression') & meta['annual_generation_kwh'].notna()
    if missing_mask.any():
        if len(fit_source) < 2:
            print(f"  WARNING: not enough data to fit size regression for "
                  f"{state}; {missing_mask.sum()} site(s) with missing size "
                  f"will be excluded from the weighted calculation.",
                  file=sys.stderr)
        else:
            slope, intercept = np.polyfit(
                fit_source['annual_generation_kwh'], fit_source['size_kw'], 1)
            meta.loc[missing_mask, 'size_kw'] = (
                intercept + slope * meta.loc[missing_mask, 'annual_generation_kwh'])

    return meta


def assign_bands_and_weights(meta: pd.DataFrame, state: str,
                             bands: list[tuple[str, float, float]],
                             by_state: dict[str, list[float]]
                             ) -> tuple[pd.Series, list[dict]]:
    """Assign each site to a size band and compute its weight.

    Returns (site_weight Series indexed by site_id, list of excluded-site
    records for reporting). If a band carries non-zero weight but has zero
    sites, that weight is instead applied to the site pool of the nearest
    non-empty neighbouring band (pooling both neighbours if equally near),
    with a warning. Only raises RuntimeError if even that fails, i.e. no
    neighbouring band has any sites either.
    """
    if state not in by_state:
        raise KeyError(f"No by_state weights defined for '{state}' in the "
                        f"weights config.")
    weights_for_state = by_state[state]

    lo_bound = bands[0][1]
    hi_bound = bands[-1][2]

    excluded = []

    def band_of(kw: float) -> str | None:
        if pd.isna(kw):
            return None
        if kw < lo_bound or kw > hi_bound:
            return None
        for label, lo, hi in bands:
            if lo <= kw <= hi:
                return label
        return None

    meta = meta.copy()
    meta['band'] = meta['size_kw'].apply(band_of)

    for _, row in meta[meta['band'].isna()].iterrows():
        reason = 'no size data' if pd.isna(row['size_kw']) else 'out of band range'
        kw_display = ('n/a' if pd.isna(row['size_kw'])
                      else f"{row['size_kw']:.2f} kW"
                           f"{' (modelled)' if row['size_estimated'] else ''}")
        print(f"  WARNING: site {row['site_id']} excluded from weighted "
              f"calculation in {state} — {reason} ({kw_display}).",
              file=sys.stderr)
        excluded.append({'site_id': row['site_id'], 'state': state,
                         'kw': row['size_kw'], 'reason': reason,
                         'estimated': row['size_estimated']})

    counts = meta.groupby('band')['site_id'].count()
    labels = [b[0] for b in bands]

    # ── Resolve which sites contribute to each band's mean ────────────────────
    # Normally a band's mean uses exactly the sites assigned to it. If a band
    # carries non-zero weight but has zero sites, we borrow the site pool from
    # the nearest non-empty neighbouring band(s) instead of erroring out. This
    # keeps that band's population share represented in the result (using the
    # closest available proxy for what a system of that size would do) rather
    # than silently dropping its weight or refusing to compute a result.
    # "Nearest" is by position in the band list; if both neighbours are
    # equally near (the empty band is not at either end and both sides are
    # one step away), both are pooled together for the borrowed estimate.
    band_sites: dict[str, list[str]] = {}
    for i, label in enumerate(labels):
        weight = weights_for_state[i]
        n_here = int(counts.get(label, 0))
        if n_here > 0 or weight == 0:
            band_sites[label] = meta.loc[meta['band'] == label, 'site_id'].tolist()
            continue

        dist = 1
        donors: list[str] = []
        while not donors and dist <= len(labels):
            candidates = [j for j in (i - dist, i + dist) if 0 <= j < len(labels)]
            donors = [labels[j] for j in candidates if counts.get(labels[j], 0) > 0]
            dist += 1

        if not donors:
            raise RuntimeError(
                f"Weighting error: band '{label}' in {state} carries a weight "
                f"of {weight:.3f} but has 0 sites, and no neighbouring band "
                f"has any sites to borrow from either. Cannot compute a "
                f"weighted result."
            )

        donor_sites = []
        for d in donors:
            donor_sites.extend(meta.loc[meta['band'] == d, 'site_id'].tolist())
        band_sites[label] = donor_sites
        print(f"  WARNING: band '{label}' in {state} has 0 sites (weight "
              f"{weight:.3f}); borrowing from neighbouring band(s) "
              f"{donors} ({len(donor_sites)} site(s)) instead.", file=sys.stderr)

    band_weight = dict(zip(labels, weights_for_state))
    weight_accum: dict[str, float] = {}
    for label, sites in band_sites.items():
        if not sites:
            continue
        w_per_site = band_weight[label] / len(sites)
        for sid in sites:
            weight_accum[sid] = weight_accum.get(sid, 0.0) + w_per_site

    site_weight = pd.Series(weight_accum, dtype=float)
    site_weight.index.name = 'site_id'
    return site_weight, excluded


# ── Scenario data loading ─────────────────────────────────────────────────────

def _load_battery_run(run_dir: Path, scenario: str, label: str) -> pd.DataFrame | None:
    if not run_dir.exists():
        return None
    frames = []
    for size in BATTERY_SIZES:
        f = run_dir / 'scenarios' / f'{scenario}_{size}kWh' / 'bills.csv'
        if not f.exists():
            continue
        df = pd.read_csv(f, usecols=[
            'site_id', 'household_total',
            'wholesale_usage_cost', 'wholesale_feedin_credit',
        ])
        df['battery_kwh'] = size
        df['net_wholesale'] = df['wholesale_usage_cost'] + df['wholesale_feedin_credit']
        frames.append(df[['site_id', 'battery_kwh', 'household_total', 'net_wholesale']])
    if not frames:
        return None
    result = pd.concat(frames, ignore_index=True)
    result['mode'] = label
    return result


def _load_solar_only(runs_root: Path, run_name: str, scenario: str) -> pd.DataFrame | None:
    """Load one solar-only (no battery) run as a savings baseline.

    Returns site_id / hh_solar / ws_solar, or None if the run is absent.
    """
    f = runs_root / run_name / 'scenarios' / scenario / 'bills.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f, usecols=[
        'site_id', 'household_total', 'wholesale_usage_cost', 'wholesale_feedin_credit',
    ])
    df['net_wholesale'] = df['wholesale_usage_cost'] + df['wholesale_feedin_credit']
    df = df[['site_id', 'household_total', 'net_wholesale']].rename(
        columns={'household_total': 'hh_solar', 'net_wholesale': 'ws_solar'})
    return df[~df['site_id'].isin(EXCLUDED_SITES)]


def load_scenario(data_root: Path, state: str, year: str,
                  baseline_mode: str = 'amber') -> dict | None:
    """Load one (state, year) scenario: savings baseline(s), all available
    retail-plan runs, and their zero-battery anchors. Returns None if the
    default baseline is missing (scenario not usable).

    baseline_mode selects what each plan's savings are measured against:

      'amber'  every plan is measured against DEFAULT_BASELINE_RUN, so a
               plan's curve includes the bill difference between its own
               tariff and Amber's. This is the study-wide comparison and
               the historical behaviour of this script.
      'own'    each plan is measured against its own solar-only run, so a
               curve is the incremental value of the battery on that plan
               and starts at $0 by construction. Answers "what does a
               battery get me here", not "what does switching to here get
               me", and is the right frame when showing one retailer their
               own results.

    Under 'own' a plan whose solar-only run is missing is skipped entirely
    rather than quietly falling back to the Amber baseline, since a chart
    mixing two baselines would look perfectly normal while comparing
    incomparable quantities.

    Unlike the original load_all_data, this does not load custom opt_factor
    runs or timing data, since sections 1 and 2 (household outcomes,
    wholesale impact) don't use them.
    """
    if baseline_mode not in ('amber', 'own'):
        raise ValueError(f"Unknown baseline_mode {baseline_mode!r}; "
                         f"expected 'amber' or 'own'.")

    runs_root = data_root / 'data' / 'output' / 'runs'
    scenario = f'{state}_{year}'

    solar_df = _load_solar_only(runs_root, DEFAULT_BASELINE_RUN, scenario)
    if solar_df is None:
        print(f"  WARNING: solar-only baseline missing for {scenario}, skipping.",
              file=sys.stderr)
        return None

    all_frames = []
    plan_zero = {}
    baselines: dict[str, pd.DataFrame] = {}
    for plan in RETAIL_PLANS:
        if state in plan['excluded_states']:
            continue

        # Which solar-only run this plan's savings are measured against.
        # Amber's zero_batt_dir is None because its own solar-only run is
        # DEFAULT_BASELINE_RUN, so it resolves to the same frame either way.
        baseline_df = solar_df
        if baseline_mode == 'own':
            own_run = plan['zero_batt_dir'] or DEFAULT_BASELINE_RUN
            if own_run != DEFAULT_BASELINE_RUN:
                baseline_df = _load_solar_only(runs_root, own_run, scenario)
                if baseline_df is None:
                    print(f"  WARNING: --baseline own requires '{own_run}' for "
                          f"{scenario}, which is missing. Skipping "
                          f"{plan['sc_label'].replace(' Self Consumption', '')} "
                          f"entirely for this scenario rather than falling back "
                          f"to the {DEFAULT_BASELINE_RUN} baseline.",
                          file=sys.stderr)
                    continue

        loaded_any = False
        entries = [(plan['sc_dir'], plan['sc_mode']),
                   (plan['lp_dir'], plan['lp_mode'])]
        no_vpp = sc_no_vpp_mode(plan)
        if no_vpp != plan['sc_mode']:
            entries.append((plan['sc_no_vpp_dir'], no_vpp))
        loaded: dict[str, pd.DataFrame] = {}
        for run_name, mode in entries:
            if run_name in loaded:
                # The same run under a second mode (the no-VPP curve of a
                # conservative self-consumption run that carries a payment).
                df = loaded[run_name].assign(mode=mode)
            else:
                df = _load_battery_run(runs_root / run_name, scenario, mode)
                if df is not None:
                    loaded[run_name] = df
            if df is None:
                print(f"  WARNING: run '{run_name}' not found for {scenario}, "
                      f"skipping mode '{mode}'.", file=sys.stderr)
                continue
            df = df[~df['site_id'].isin(EXCLUDED_SITES)]
            all_frames.append(df)
            baselines[mode] = baseline_df
            loaded_any = True

        if plan['zero_batt_dir'] is not None and loaded_any:
            pz = _load_solar_only(runs_root, plan['zero_batt_dir'], scenario)
            if pz is not None:
                # Renamed on load for use as a baseline; here it is the
                # measured quantity instead, so swap the roles back.
                pz = pz.rename(columns={'hh_solar': 'household_total',
                                        'ws_solar': 'net_wholesale'})
                pz = pz.merge(baseline_df, on='site_id', how='inner')
                pz['hh_saving'] = pz['hh_solar'] - pz['household_total']
                pz['ws_saving'] = pz['ws_solar'] - pz['net_wholesale']
                plan_zero[plan['zero_batt_key']] = pz[['site_id', 'hh_saving', 'ws_saving']]
            else:
                print(f"  WARNING: plan_zero '{plan['zero_batt_key']}' not found "
                      f"for {scenario}.", file=sys.stderr)

    if not all_frames:
        print(f"  WARNING: no retail-plan runs found for {scenario}, skipping.",
              file=sys.stderr)
        return None

    runs = pd.concat(all_frames, ignore_index=True)
    return {'solar_only': solar_df, 'runs': runs, 'plan_zero': plan_zero,
            'baselines': baselines, 'baseline_mode': baseline_mode}


def build_savings(scenario_data: dict) -> pd.DataFrame:
    """Attach each mode's savings against its own baseline.

    Modes are handled one at a time because under --baseline own they no
    longer share a baseline frame. Under 'amber' every mode maps to the same
    frame, so this is equivalent to the single merge it replaces.
    """
    frames = []
    for mode, sub in scenario_data['runs'].groupby('mode', sort=False):
        base = scenario_data['baselines'][mode]
        merged = sub.merge(base, on='site_id', how='inner')
        dropped = sub['site_id'].nunique() - merged['site_id'].nunique()
        if dropped:
            print(f"  WARNING: {dropped} site(s) in mode '{mode}' have no "
                  f"matching site in its solar-only baseline and were dropped. "
                  f"This mode's curve is computed over a smaller site set than "
                  f"the others.", file=sys.stderr)
        frames.append(merged)
    df = pd.concat(frames, ignore_index=True)
    df['hh_saving'] = df['hh_solar'] - df['household_total']
    df['ws_saving'] = df['ws_solar'] - df['net_wholesale']
    return df


# ── Weighted / unweighted mean ────────────────────────────────────────────────

def mean_by_size(df: pd.DataFrame, y_col: str,
                 site_weight: pd.Series | None) -> pd.Series:
    """Mean of y_col by battery_kwh, reindexed to BATTERY_SIZES.

    If site_weight is None, this is a plain per-site mean (identical to the
    original script's _mean_by_size). If given, it's a weighted mean using
    the supplied per-site weights (a 'weight' column, e.g. site-level size-
    band weight, or a combined size-band * state-mix weight for pooling).
    Sites with no weight (excluded / out of band) are dropped.
    """
    if site_weight is None:
        return df.groupby('battery_kwh')[y_col].mean().reindex(BATTERY_SIZES)

    d = df.copy()
    d['_w'] = d['site_id'].map(site_weight)
    d = d.dropna(subset=['_w'])
    if d.empty:
        return pd.Series(index=BATTERY_SIZES, dtype=float)

    def _wmean(g):
        w = g['_w'].to_numpy()
        if w.sum() == 0:
            return np.nan
        return np.average(g[y_col].to_numpy(), weights=w)

    return d.groupby('battery_kwh').apply(_wmean).reindex(BATTERY_SIZES)


def weighted_scalar_mean(df: pd.DataFrame, y_col: str,
                         site_weight: pd.Series | None) -> float:
    """Same idea as mean_by_size but for a scalar (no battery_kwh grouping),
    used for plan_zero anchors.
    """
    if site_weight is None:
        return float(df[y_col].mean())
    d = df.copy()
    d['_w'] = d['site_id'].map(site_weight)
    d = d.dropna(subset=['_w'])
    if d.empty or d['_w'].sum() == 0:
        return 0.0
    return float(np.average(d[y_col].to_numpy(), weights=d['_w'].to_numpy()))


# ── Chart helpers ──────────────────────────────────────────────────────────

def _smooth_xy_piecewise(xs: np.ndarray, ys: np.ndarray,
                         breakpoint: float = 20.0, n_pts: int = 200):
    valid = ~np.isnan(ys)
    xv, yv = xs[valid], ys[valid]
    if len(xv) < 3:
        return xv, yv
    bp_idx = np.searchsorted(xv, breakpoint, side='right') - 1
    bp_idx = max(1, min(bp_idx, len(xv) - 2))
    results_x, results_y = [], []
    for x_seg, y_seg, n_half in [
        (xv[:bp_idx + 1], yv[:bp_idx + 1], n_pts // 3),
        (xv[bp_idx:],     yv[bp_idx:],     n_pts * 2 // 3),
    ]:
        if len(x_seg) < 2:
            results_x.append(x_seg); results_y.append(y_seg)
            continue
        k = min(3, len(x_seg) - 1)
        if k < 2:
            results_x.append(x_seg); results_y.append(y_seg)
            continue
        spl = make_interp_spline(x_seg, y_seg, k=k)
        x_fine = np.linspace(x_seg[0], x_seg[-1], n_half)
        results_x.append(x_fine); results_y.append(spl(x_fine))
    return np.concatenate(results_x), np.concatenate(results_y)


# Thesis figure format (docs/thesis_figure_spec.md).
# Single charts are taller than the spec's 3.0 in by the legend rows below
# the axes, so the axes keep their height.
SINGLE_HEIGHT_IN = 3.4
PAIR_HEIGHT_IN = 3.3
DASHED = (0, (5, 2.4))
STYLE_KEY_COLOUR = '0.45'
YLABEL = r'Mean annual saving (\$/site)'

def _offer_name(plan: dict) -> str:
    return plan['sc_label'].replace(' Self Consumption', '')


NOTE_TOP = 0.97                  # axes-fraction top edge of upper notes
PAIR_NOTE_BOTTOM = 0.02          # lower notes on the basis pair, clear of the curves


def restricted_offer_notes(states, plans: list[dict] | None = None
                           ) -> list[str]:
    """Data note lines for offers whose results cover only some of the
    states in a pooled view, e.g. 'EnergyAustralia results for NSW only'.
    plans defaults to the full registry, not the installed one, so scripts
    that import this module get the same lines."""
    states = set(states)
    lines = []
    for plan in _BASE_PLANS if plans is None else plans:
        covered = sorted(states - plan['excluded_states'])
        if covered and len(covered) < len(states):
            lines.append(f"{_offer_name(plan).split()[0]} results for "
                         f"{', '.join(covered)} only")
    return lines


def _draw_notes(ax, text: str, loc: str, bottom: float = 0.04) -> None:
    """Data notes in a corner of the axes: grey 0.3 at the data note size,
    on a white box so gridlines and data behind them do not show through.
    bottom is the axes-fraction offset of lower notes from the x-axis."""
    x, ha = (0.02, 'left') if 'left' in loc else (0.98, 'right')
    y, va = (NOTE_TOP, 'top') if 'upper' in loc else (bottom, 'bottom')
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=ts.NOTE_SIZE, color=ts.REFERENCE_COLOUR,
            bbox=ts.LABEL_BOX, zorder=6)


def _frame_capacity_axis(ax, label_step: int | None = None) -> None:
    """Battery capacity sweep axis: ticks at the modelled capacities (or
    every label_step kWh where panels are too narrow to label them all),
    3 % padding either side, and dollar tick labels on y."""
    xs = [0] + BATTERY_SIZES
    pad = 0.03 * xs[-1]
    ax.set_xlim(xs[0] - pad, xs[-1] + pad)
    if label_step:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(label_step))
    else:
        ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'${v:,.0f}' if v >= 0 else f'−${-v:,.0f}'))


def _sc_reference_name() -> str | None:
    """Offer name of the plan supplying the self-consumption reference
    line, or None if that plan isn't in the (possibly filtered) registry.
    """
    for plan in RETAIL_PLANS:
        if plan['sc_mode'] == SC_REFERENCE_MODE:
            return _offer_name(plan)
    return None


def _style_handles(opt_types: set[str], sc_label: str | None,
                   marks: ts.Marks = ts.MARKS) -> list:
    """Legend entries explaining line style and marker, built only for the
    conventions actually drawn on the chart."""
    kw = dict(color=STYLE_KEY_COLOUR, lw=marks.lw, ms=marks.ms)
    handles = []
    if 'vpp' in opt_types:
        handles.append(mlines.Line2D([0], [0], ls=DASHED, marker='s',
                                     label='VPP optimised', **kw))
    if 'user' in opt_types:
        handles.append(mlines.Line2D([0], [0], ls=DASHED, marker='D',
                                     label='User optimised', **kw))
    if sc_label is not None:
        handles.append(mlines.Line2D([0], [0], ls='-', marker='o',
                                     label=sc_label, **kw))
    return handles


def _plan_anchor(plan: dict, y_col: str, data: dict,
                 site_weight: pd.Series | None) -> float:
    """The zero-battery ($0 saving by definition, or a plan-specific
    no-battery bill) anchor value for one plan, used as the x=0 point on
    every curve.
    """
    zero_key = plan['zero_batt_key']
    if zero_key is None:
        return 0.0
    pz = data['plan_zero'].get(zero_key)
    if pz is None:
        print(f"  WARNING: plan_zero['{zero_key}'] not loaded, using 0 anchor.",
              file=sys.stderr)
        return 0.0
    return weighted_scalar_mean(pz, y_col, site_weight)


# ── Curve model ────────────────────────────────────────────────────────────────
#
# A "curve set" is a dict keyed by internal mode label (e.g. 'SC', 'LP_AGL')
# whose value is a numpy array of length len(BATTERY_SIZES) + 1: index 0 is
# the zero-battery anchor for that plan, the rest follow BATTERY_SIZES order.
# This is the common representation used by:
#   - a single (state, year) scenario (computed straight from site data)
#   - a state's multi-year average (elementwise mean of per-year curve sets)
#   - the national pooled view (elementwise state_mix-weighted sum of
#     per-state curve sets, each of which may itself be a multi-year average)
#
# Working in this shape rather than re-deriving from raw site data at every
# aggregation level is what makes "straight mean across years" and "state_mix
# weighted combination across states" behave as genuine means-of-means
# rather than accidentally re-pooling site samples that were never meant to
# be pooled together (e.g. different years' sites are not the same
# population and should not be blended site-by-site).

def compute_curves(df: pd.DataFrame, data: dict,
                   site_weight: pd.Series | None) -> dict[str, dict[str, np.ndarray]]:
    """Compute curve sets for one (state, year) scenario.

    Returns {'hh_saving': {mode: array}, 'ws_saving': {mode: array}}.
    """
    sizes = BATTERY_SIZES
    curves = {'hh_saving': {}, 'ws_saving': {}}
    for plan in RETAIL_PLANS:
        anchor_hh = _plan_anchor(plan, 'hh_saving', data, site_weight)
        anchor_ws = _plan_anchor(plan, 'ws_saving', data, site_weight)
        for mode_key in dict.fromkeys((plan['sc_mode'], plan['lp_mode'],
                                       sc_no_vpp_mode(plan))):
            sub = df[df['mode'] == mode_key]
            if sub.empty:
                continue
            hh_means = mean_by_size(sub, 'hh_saving', site_weight).to_numpy()
            ws_means = mean_by_size(sub, 'ws_saving', site_weight).to_numpy()
            curves['hh_saving'][mode_key] = np.concatenate([[anchor_hh], hh_means])
            curves['ws_saving'][mode_key] = np.concatenate([[anchor_ws], ws_means])
    return curves


def average_curves_across_years(curve_sets: list[dict]) -> dict:
    """Elementwise mean of several years' curve sets for the same state.

    This is a straight (unweighted) mean across years, per the instruction
    that year-averaging within a state is not itself weighted by anything.
    A mode missing in some years is averaged only over the years where it is
    present, so one plan's data landing late doesn't blank out its earlier
    years or drag down its average with implicit zeros.
    """
    result = {'hh_saving': {}, 'ws_saving': {}}
    for y_col in ('hh_saving', 'ws_saving'):
        all_modes = set()
        for cs in curve_sets:
            all_modes.update(cs[y_col].keys())
        for mode in all_modes:
            arrays = [cs[y_col][mode] for cs in curve_sets if mode in cs[y_col]]
            result[y_col][mode] = np.nanmean(np.stack(arrays), axis=0)
    return result


def combine_curves_state_weighted(per_state_curves: dict[str, dict],
                                  state_mix: dict[str, float]) -> dict:
    """Weighted combination of several states' curve sets using state_mix.

    Weights are renormalised per mode over just the states where that mode's
    curve is present, so a plan restricted to a subset of states (e.g. a
    future NSW-only plan) still produces a sensible pooled curve using only
    the states it actually exists in, rather than being diluted by states
    where it was never expected to appear.
    """
    result = {'hh_saving': {}, 'ws_saving': {}}
    for y_col in ('hh_saving', 'ws_saving'):
        all_modes = set()
        for cs in per_state_curves.values():
            all_modes.update(cs[y_col].keys())
        for mode in all_modes:
            states_with_mode = [s for s, cs in per_state_curves.items()
                                if mode in cs[y_col]]
            weights = np.array([state_mix.get(s, 0.0) for s in states_with_mode])
            if weights.sum() == 0:
                continue
            weights = weights / weights.sum()
            arrays = np.stack([per_state_curves[s][y_col][mode]
                               for s in states_with_mode])
            result[y_col][mode] = np.average(arrays, axis=0, weights=weights)
    return result


# ── Drawing from curve sets ───────────────────────────────────────────────────
#
# Three views of a curve set, each a list of (plan, mode, linestyle, marker)
# series per plan:
#
#   default    each plan in its default (as-joined) mode
#   optimised  each plan's optimised series, plus the reference plan's
#              self-consumption series as the sole self-consumption line
#   full       both series for every plan, self-consumption without VPP
#              participation (sc_no_vpp_mode)
#
# Colour identifies the offer; linestyle and marker the mode: solid and
# circle for self consumption, dashed and square for VPP-optimised plans,
# dashed and diamond for user-optimised plans.

def _view_series(view: str) -> list[tuple[dict, str, object, str]]:
    series = []
    for plan in RETAIL_PLANS:
        lp_marker = 's' if plan['opt_type'] == 'vpp' else 'D'
        sc = (plan, plan['sc_mode'], '-', 'o')
        lp = (plan, plan['lp_mode'], DASHED, lp_marker)
        if view == 'default':
            series.append(sc if plan['default_mode'] == 'sc_dir' else lp)
        elif view == 'optimised':
            series.append(lp)
            if plan['sc_mode'] == SC_REFERENCE_MODE:
                series.append(sc)
        else:
            # The full range shows self-consumption without VPP
            # participation, so the gap to the optimised curve is the whole
            # value of the plan's coordination.
            series.extend([(plan, sc_no_vpp_mode(plan), '-', 'o'), lp])
    return series


def _draw_view(ax, curves: dict[str, np.ndarray], view: str,
               marks: ts.Marks = ts.MARKS) -> tuple[list, list]:
    """Draw one view onto axes. Returns (offer handles, style key handles):
    one handle per offer drawn, in registry order, and keys for only the
    line styles and markers actually drawn. marks sets the line width and
    marker sizes, ts.GRID_MARKS on grid panels."""
    xs_base = np.concatenate([[0], np.array(BATTERY_SIZES)])
    offers: dict[str, object] = {}
    opt_types: set[str] = set()
    sc_drawn = False
    for plan, mode, ls, marker in _view_series(view):
        if mode not in curves:
            continue
        ys_raw = np.asarray(curves[mode], dtype=float)
        xs_sm, ys_sm = _smooth_xy_piecewise(xs_base, ys_raw)
        ax.plot(xs_sm, ys_sm, color=plan['color'], lw=marks.lw, ls=ls,
                zorder=3)
        ink = PLAN_INK.get(plan['sc_mode'])
        edge = (dict(markeredgecolor=ink, markeredgewidth=marks.mew)
                if ink else {})
        ax.plot(xs_base, ys_raw, ls='none', marker=marker, ms=marks.ms,
                color=plan['color'], zorder=4, **edge)
        name = _offer_name(plan)
        offers.setdefault(name, mlines.Line2D(
            [0], [0], color=plan['color'], lw=marks.lw, label=name))
        if mode in (plan['sc_mode'], sc_no_vpp_mode(plan)):
            sc_drawn = True
        else:
            opt_types.add(plan['opt_type'])
    if view == 'optimised':
        ref = _sc_reference_name()
        sc_label = f'Self-consumption ({ref})' if sc_drawn and ref else None
    else:
        sc_label = 'Self-consumption' if sc_drawn else None
    return list(offers.values()), _style_handles(opt_types, sc_label, marks)


def _figure_single(curves: dict, view: str, title: str, notes_loc: str,
                   notes: str | None):
    """One view of a curve set. With five offers and up to three style keys
    there is no corner the curves reliably leave clear for the legend, so it
    goes below the axes, as on the grids and pairs."""
    fig = ts.new_figure(height_in=SINGLE_HEIGHT_IN)
    ax = fig.add_subplot()
    offers, styles = _draw_view(ax, curves, view)
    _frame_capacity_axis(ax)
    ax.set_xlabel('Battery capacity (kWh)')
    ax.set_ylabel(YLABEL)
    ax.set_title(title)
    handles = offers + styles
    fig.legend(handles=handles, loc='outside lower center',
               ncol=-(-len(handles) // 2), frameon=False)
    if notes:
        _draw_notes(ax, notes, notes_loc)
    return fig


# (view, metric, file stem, title, notes corner) for each single-panel
# figure in a figure set. The notes take the corner that chart's curves
# leave clear: upper left on the household charts, where the curves rise to
# the right, and lower right on the wholesale chart, above the
# self-consumption line.
SINGLE_FIGURES = [
    ('default', 'hh_saving', 'household_savings_default_mode',
     'Household bill saving by battery size and offer, default mode',
     'upper left'),
    ('optimised', 'hh_saving', 'household_savings_optimised_mode',
     'Household bill saving by battery size and offer, optimised',
     'upper left'),
    ('full', 'hh_saving', 'household_savings_full_range',
     'Household bill saving by battery size, offer and mode',
     'upper left'),
    ('optimised', 'ws_saving', 'wholesale_savings',
     'Net wholesale saving by battery size and offer', 'lower right'),
]


def make_all_figures(curve_set: dict, output_dir: Path, weight_tag: str,
                     info_text: str, basis_title: str,
                     draw_notes: bool = True) -> None:
    """Generate the four single-panel figures from one curve set (a single
    scenario, a state's multi-year average, or the national pooled result).
    info_text is printed for the caption, and drawn unless draw_notes is
    False."""
    for view, metric, stem, title, notes_loc in SINGLE_FIGURES:
        fig = _figure_single(curve_set[metric], view, f'{title}, {basis_title}',
                             notes_loc, info_text if draw_notes else None)
        ts.save_figure(fig, output_dir / f'{stem}_{weight_tag}')
    print('    ' + info_text.replace('\n', ', '))


def make_basis_pair(sets: list[tuple[str, dict]], metric: str, title: str,
                    notes: str | None, output_dir: Path, stem: str) -> None:
    """The optimised view of curve sets from different run bases side by
    side on a shared y range, under one title and one legend. sets is
    [(panel subtitle, curve set), ...]. notes, where given, go in the
    bottom right corner of each panel."""
    fig = ts.new_figure(height_in=PAIR_HEIGHT_IN)
    axes = fig.subplots(1, len(sets), sharey=True, squeeze=False)[0]
    ylim = _shared_ylim([cs for _, cs in sets], metric)
    handles: list = []
    for ax, (subtitle, curve_set) in zip(axes, sets):
        offers, styles = _draw_view(ax, curve_set[metric], 'optimised')
        if len(offers) + len(styles) > len(handles):
            handles = offers + styles
        _frame_capacity_axis(ax)
        ax.set_ylim(*ylim)
        ax.set_title(subtitle, fontsize=ts.SUBTITLE_SIZE)
        ax.set_xlabel('Battery capacity (kWh)')
        if notes:
            _draw_notes(ax, notes, 'lower right', bottom=PAIR_NOTE_BOTTOM)
    fig.supylabel(YLABEL, fontsize=mpl.rcParams['axes.labelsize'])
    fig.legend(handles=handles, loc='outside lower center',
               ncol=-(-len(handles) // 2), frameon=False)
    ts.panel_suptitle(fig, title)
    ts.save_figure(fig, output_dir / stem)


def _shared_ylim(curve_sets: list[dict], metric: str,
                 pad_frac: float = 0.08) -> tuple[float, float]:
    """Common y-range across several curve sets' optimised view, so panels
    can share one y-axis honestly."""
    vals = []
    for cs in curve_sets:
        curves = cs[metric]
        for _, mode, _, _ in _view_series('optimised'):
            if mode in curves:
                vals.append(np.asarray(curves[mode], dtype=float))
    allv = np.concatenate(vals)
    allv = allv[~np.isnan(allv)]
    lo, hi = float(allv.min()), float(allv.max())
    span = (hi - lo) if hi > lo else max(abs(hi), 1.0)
    return lo - span * pad_frac, hi + span * pad_frac


def _grid_cells_from_flat(labels: list[str], n_rows: int, n_cols: int) -> list[list[str | None]]:
    """Row-major fill of a small-multiples grid from a flat label list,
    padding any leftover cells with None (turned off / hidden panels).
    """
    cells: list[list[str | None]] = []
    it = iter(labels)
    for _ in range(n_rows):
        row = [next(it, None) for _ in range(n_cols)]
        cells.append(row)
    return cells


# Panel heights (in) by number of rows, keeping the panels wider than they
# are tall, and room for the figure title, the x-axis label and two legend
# rows below the panels.
GRID_PANEL_HEIGHT_IN = {2: 1.75, 4: 0.82}
GRID_FURNITURE_IN = 1.05


def make_grid_figure(cells: list[list[str | None]], panels: dict[str, dict],
                     metric: str, title: str, panel_notes: dict[str, str],
                     output_dir: Path, stem: str,
                     title_note: str | None = None) -> None:
    """Small-multiples grid (arbitrary rows x cols) of the optimised view,
    one panel per curve set (one per state, one per year, or a state x year
    grid). Panels share one y range and one legend below them. cells is a
    row-major list of lists of labels, None for a blank cell; panel_notes
    maps a label to the site count drawn in the panel's top left corner,
    which the curves leave clear, and is empty for the thesis. title_note,
    where given, is appended to the title: the grid's equivalent of a
    single chart's notes.
    """
    n_rows = len(cells)
    n_cols = len(cells[0]) if n_rows else 0
    flat_labels = [c for row in cells for c in row if c is not None]
    ylim = _shared_ylim([panels[l] for l in flat_labels], metric)

    panel_h = GRID_PANEL_HEIGHT_IN.get(n_rows, 1.75)
    fig = ts.new_figure(height_in=panel_h * n_rows + GRID_FURNITURE_IN)
    axes = ts.grid_axes(fig, n_rows, n_cols, 'Battery capacity (kWh)')
    legend_handles: list = []
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r][c]
            label = cells[r][c]
            if label is None:
                ax.axis('off')
                continue
            offers, styles = _draw_view(ax, panels[label][metric], 'optimised',
                                        ts.GRID_MARKS)
            if len(offers) + len(styles) > len(legend_handles):
                legend_handles = offers + styles
            ax.set_title(label, fontsize=ts.SUBTITLE_SIZE)
            note = panel_notes.get(label)
            if note:
                _draw_notes(ax, note, 'upper left')
            _frame_capacity_axis(ax, label_step=10)
            ax.set_ylim(*ylim)
            # Tick labels on the outer edges only. "Bottom" is the lowest
            # panel in this column with content, so a column that runs
            # short still gets its x tick labels.
            is_bottom = all(cells[rr][c] is None for rr in range(r + 1, n_rows))
            ax.tick_params(labelleft=(c == 0), labelbottom=is_bottom)

    fig.supylabel(YLABEL, fontsize=mpl.rcParams['axes.labelsize'])
    fig.legend(handles=legend_handles, loc='outside lower center',
               ncol=-(-len(legend_handles) // 2), frameon=False)
    ts.panel_suptitle(fig, f'{title} — {title_note}' if title_note
                      else title)
    ts.save_figure(fig, output_dir / stem)


def make_comparison_grid(panels: dict[str, dict], metric: str, title: str,
                         panel_notes: dict[str, str], output_dir: Path,
                         stem: str, title_note: str | None = None) -> None:
    """2x2 small-multiples grid comparing up to 4 curve sets (e.g. one per
    state, or one per year for a single state)."""
    cells = _grid_cells_from_flat(list(panels.keys()), 2, 2)
    make_grid_figure(cells, panels, metric, title, panel_notes, output_dir,
                     stem, title_note)


def install_plans(plans: list[dict], sc_reference_mode: str | None = None) -> None:
    """Replace the plan registry for the lifetime of the process.

    Every drawing and loading function in this module reads the module-level
    RETAIL_PLANS, so this is the single supported way for a caller to change
    what gets analysed — both --only-plan below and external scripts (e.g. the
    retailer briefing script) go through here rather than reaching in and
    rebinding globals themselves.

    Mutates in place so that callers holding a reference to RETAIL_PLANS see
    the change. sc_reference_mode selects which plan supplies the single
    self-consumption reference line on the charts that carry one; it defaults
    to the first installed plan, since the Amber default is meaningless once
    the registry no longer describes the standard plans.
    """
    if not plans:
        raise ValueError('install_plans requires at least one plan.')
    global SC_REFERENCE_MODE
    modes = {p['sc_mode'] for p in plans}
    if sc_reference_mode is not None and sc_reference_mode not in modes:
        raise ValueError(f'sc_reference_mode {sc_reference_mode!r} is not among '
                         f'the installed plans {sorted(modes)}.')
    RETAIL_PLANS[:] = plans
    SC_REFERENCE_MODE = sc_reference_mode or plans[0]['sc_mode']


def make_scenario_builder(data_root: Path, weights_cfg: dict | None,
                          baseline_mode: str = 'amber'):
    """Return a callable (state, year) -> dict | None describing one scenario.

    Wraps the per-scenario pipeline — load runs, attach savings, derive PV
    size weights, compute curves — so that callers other than main() can get
    a weighted result without restating any of it. The per-year metadata
    caches (pooled DC:AC ratio, pooled site metadata) live in the closure and
    are shared across calls, exactly as the locals in main() used to be.

    The returned dict carries:
      curves       curve set, as described under 'Curve model' above
      n_sites      effective site count behind the curves
      savings      per-site, per-size, per-mode savings frame (build_savings)
      site_weight  per-site weight Series, or None when unweighted

    savings and site_weight are the inputs compute_curves collapses. They are
    returned as well because some charts describe the spread across sites
    rather than the mean, and a distribution cannot be recovered from a curve.

    Pass weights_cfg=None for unweighted curves.
    """
    all_meta_by_year: dict[str, pd.DataFrame] = {}
    dc_ac_ratio_by_year: dict[str, float] = {}

    def build(state: str, year: str):
        data = load_scenario(data_root, state, year, baseline_mode)
        if data is None:
            return None
        df = build_savings(data)
        n_sites_total = df['site_id'].nunique()

        site_weight = None
        if weights_cfg is not None:
            if year not in all_meta_by_year:
                pooled_frames = []
                for other_state in weights_cfg['by_state'].keys():
                    try:
                        pooled_frames.append(load_site_meta(data_root, year, other_state))
                    except FileNotFoundError:
                        continue
                pooled_meta = (pd.concat(pooled_frames, ignore_index=True)
                              if pooled_frames else pd.DataFrame(
                                  columns=['site_id', 'subarray_dc_rating_kw',
                                          'inverter_ac_rating_kw',
                                          'annual_generation_kwh']))
                # Pooled (national) DC:AC ratio, computed once per year from
                # every state's tier-1 trusted sites, then applied uniformly.
                ratio = compute_dc_ac_ratio(pooled_meta)
                print(f"  NOTE: {year} pooled DC:AC ratio = {ratio:.3f} "
                      f"(from tier-1 trusted sites across all states).")
                dc_ac_ratio_by_year[year] = ratio
                all_meta_by_year[year] = assign_direct_size(pooled_meta, ratio)

            meta = load_site_meta(data_root, year, state)
            meta = assign_direct_size(meta, dc_ac_ratio_by_year[year])
            meta = estimate_missing_sizes(meta, state,
                                          fallback_meta=all_meta_by_year[year])
            site_weight, excluded = assign_bands_and_weights(
                meta, state, weights_cfg['bands'], weights_cfg['by_state'])
            if excluded:
                print(f"  {len(excluded)} site(s) excluded from weighted "
                      f"calculation for {state} {year}.")

        curves = compute_curves(df, data, site_weight)

        # The site count shown on-chart should reflect what actually went
        # into the number being plotted. In weighted mode that's the count of
        # sites in df that also have a usable weight (i.e. excludes
        # out-of-band / unbandable sites, per the weighting warnings above) —
        # not the full site count, which would silently overstate the
        # effective sample behind a weighted result.
        if site_weight is not None:
            n_sites = len(set(df['site_id']) & set(site_weight.index))
        else:
            n_sites = n_sites_total
        return {'curves': curves, 'n_sites': n_sites, 'savings': df,
                'site_weight': site_weight}

    return build


# ── Main ──────────────────────────────────────────────────────────────────────

def _span(years: list[str]) -> str:
    """Consecutive years as a span, otherwise listed."""
    ints = sorted(int(y) for y in years)
    if len(ints) > 1 and ints == list(range(ints[0], ints[-1] + 1)):
        return f'{ints[0]}–{ints[-1]}'
    return ', '.join(str(y) for y in ints)


def _curve_rows(scope: str, curve_set: dict) -> list[dict]:
    """Long-format rows of one curve set, for the curves CSV."""
    rows = []
    kwh = [0] + BATTERY_SIZES
    for plan in RETAIL_PLANS:
        modes = [(plan['sc_mode'], 'self-consumption'),
                 (plan['lp_mode'], 'optimised')]
        if sc_no_vpp_mode(plan) != plan['sc_mode']:
            modes.append((sc_no_vpp_mode(plan), 'self-consumption, no VPP'))
        for mode, series in modes:
            for metric, curves in curve_set.items():
                if mode not in curves:
                    continue
                for k, v in zip(kwh, curves[mode]):
                    rows.append(dict(scope=scope, metric=metric,
                                     offer=_offer_name(plan), series=series,
                                     mode=mode, kwh=k, value=float(v)))
    return rows


def _site_rows(state: str, year: str, built: dict, basis: str) -> pd.DataFrame:
    """One scenario's site savings by mode and capacity, with the basis
    adjustments applied, for the site savings CSV. weight is the site's PV
    size weight (1 when unweighted); sites the weighting excludes are left
    out, as they are from the curves."""
    labels = {}
    for plan in RETAIL_PLANS:
        labels[plan['sc_mode']] = (_offer_name(plan), 'self-consumption')
        labels[plan['lp_mode']] = (_offer_name(plan), 'optimised')
    df = built['savings'][['site_id', 'battery_kwh', 'mode',
                           'hh_saving', 'ws_saving']]
    df = apply_site_basis_adjustments(df[df['mode'].isin(labels)], basis)
    weight = built['site_weight']
    df['weight'] = 1.0 if weight is None else df['site_id'].map(weight)
    df = df.dropna(subset=['weight'])
    df['offer'] = df['mode'].map(lambda m: labels[m][0])
    df['series'] = df['mode'].map(lambda m: labels[m][1])
    return df.assign(state=state, year=year).rename(
        columns={'battery_kwh': 'kwh'})[
        ['state', 'year', 'site_id', 'weight', 'offer', 'series', 'mode',
         'kwh', 'hh_saving', 'ws_saving']]


def _run_basis(basis: str, ctx: dict) -> dict | None:
    """Every figure set for one run basis, written under that basis's
    folder. Returns the national curve set (None without one) for the
    comparison figures."""
    args = ctx['args']
    spec = RUN_BASES[basis]
    basis_title = spec['title']
    out_root = args.output_dir / spec['folder']
    weighted, weight_tag = ctx['weighted'], ctx['weight_tag']
    weight_label = 'Weighted by PV size' if weighted else 'Unweighted'
    baseline_note = ctx['baseline_note']
    draw_notes = args.notes
    title_note = weight_label if draw_notes else None

    plans = plans_for_basis(basis)
    if ctx['only_modes']:
        plans = [p for p in plans if p['sc_mode'] in ctx['only_modes']]
    install_plans(plans, ctx['sc_reference_mode'])

    print('\n' + '=' * 72)
    print(f'RUN BASIS: {basis_title}  ->  {out_root}')
    for mode, p in PLACEHOLDERS.get(basis, {}).items():
        print(f"  PLACEHOLDER: {mode} = {1 - p['capture']:g} x {p['base']} + "
              f"{p['capture']:g} x the run in its slot, standing in for "
              f"{p['run']}.")
    print('=' * 72)

    build_scenario = make_scenario_builder(args.data_root, ctx['weights_cfg'],
                                           ctx['baseline_mode'])
    scenarios_by_state = ctx['scenarios_by_state']
    per_state_curves: dict[str, dict] = {}
    per_state_site_counts: dict[str, list[int]] = {}
    per_state_year_curves: dict[str, dict[str, dict]] = {}
    per_state_year_ncount: dict[str, dict[str, int]] = {}
    csv_rows: list[dict] = []
    site_frames: list[pd.DataFrame] = []

    for state in ctx['states_in_order']:
        years = scenarios_by_state[state]
        year_curve_sets = []
        for year in years:
            print(f"\nLoading {state} {year}...")
            built = build_scenario(state, year)
            if built is None:
                continue
            curves, n_sites = built['curves'], built['n_sites']
            apply_basis_adjustments(curves, basis)
            site_frames.append(_site_rows(state, year, built, basis))
            year_curve_sets.append(curves)
            csv_rows += _curve_rows(f'{state}_{year}', curves)

            # Tier 1: per (state, year) figures — always produced.
            info_text = (f'{state} {year} (n = {n_sites})\n'
                         f'{weight_label}{baseline_note}')
            make_all_figures(curves, out_root / f'{state}_{year}', weight_tag,
                             info_text, basis_title, draw_notes)
            per_state_site_counts.setdefault(state, []).append(n_sites)
            per_state_year_curves.setdefault(state, {})[year] = curves
            per_state_year_ncount.setdefault(state, {})[year] = n_sites

        if not year_curve_sets:
            continue
        # Tier 2: per-state, straight mean across years — only when the
        # state actually has more than one year loaded.
        if len(year_curve_sets) > 1:
            state_avg = average_curves_across_years(year_curve_sets)
            per_state_curves[state] = state_avg
            csv_rows += _curve_rows(f'{state}_all_years', state_avg)
            counts = per_state_site_counts[state]
            info_text = (f'{state}, {_span(years)} '
                         f'(n = {min(counts)}–{max(counts)} per year)\n'
                         f'{weight_label}{baseline_note}')
            make_all_figures(state_avg, out_root / f'{state}_all_years',
                             weight_tag, info_text, basis_title, draw_notes)
        else:
            # Single year: the state's "representative" curve set for
            # national pooling is just that year's curve set.
            per_state_curves[state] = year_curve_sets[0]

    grid_dir = out_root / 'national_state_comparison'

    # Comparison grid 1: all states side by side (2x2), same y-scale, one
    # legend, for household_savings_optimised_mode and wholesale_savings.
    if len(per_state_curves) > 1:
        state_notes = {}
        for state in per_state_curves:
            counts = per_state_site_counts.get(state, [])
            if counts:
                state_notes[state] = (f'n = {min(counts)}–{max(counts)}'
                                      if len(set(counts)) > 1
                                      else f'n = {counts[0]}')
        print(f"\nState comparison grid ({weight_tag}): "
              f"{', '.join(per_state_curves)}")
        make_comparison_grid(
            per_state_curves, 'hh_saving',
            f'Household bill saving by state, optimised, {basis_title}',
            state_notes if draw_notes else {}, grid_dir,
            f'household_savings_optimised_mode_by_state_{weight_tag}',
            title_note)
        make_comparison_grid(
            per_state_curves, 'ws_saving',
            f'Net wholesale saving by state, {basis_title}',
            state_notes if draw_notes else {}, grid_dir, f'wholesale_savings_by_state_{weight_tag}',
            title_note)

    # Comparison grid 1b: full state x year grid (rows = states, columns =
    # every year found for any state). A state with no data for a given
    # year leaves that cell blank rather than distorting the row.
    if len(per_state_year_curves) > 1:
        all_years_sorted = sorted({y for ys in per_state_year_curves.values()
                                   for y in ys})
        sy_panels: dict[str, dict] = {}
        sy_notes: dict[str, str] = {}
        cells: list[list[str | None]] = []
        for state in ctx['states_in_order']:
            year_curves = per_state_year_curves.get(state, {})
            row: list[str | None] = []
            for year in all_years_sorted:
                if year in year_curves:
                    label = f'{state} {year}'
                    sy_panels[label] = year_curves[year]
                    sy_notes[label] = f'n = {per_state_year_ncount[state][year]}'
                    row.append(label)
                else:
                    row.append(None)
            cells.append(row)
        print(f"\nState x year grid ({weight_tag}): "
              f"{len(cells)} states x {len(all_years_sorted)} years")
        make_grid_figure(
            cells, sy_panels, 'hh_saving',
            f'Household bill saving by state and year, optimised, '
            f'{basis_title}', sy_notes if draw_notes else {}, grid_dir,
            f'household_savings_optimised_mode_by_state_and_year_{weight_tag}',
            title_note)
        make_grid_figure(
            cells, sy_panels, 'ws_saving',
            f'Net wholesale saving by state and year, {basis_title}',
            sy_notes if draw_notes else {}, grid_dir,
            f'wholesale_savings_by_state_and_year_{weight_tag}', title_note)

    # Comparison grid 2: for each state, the four most recent years side by
    # side (2x2), same y-scale, one legend. Only built for states with at
    # least two of the target years present.
    TARGET_YEARS = ['2022', '2023', '2024', '2025']
    for state in ctx['states_in_order']:
        year_curves = per_state_year_curves.get(state, {})
        sel_years = [y for y in TARGET_YEARS if y in year_curves]
        if len(sel_years) < 2:
            continue
        year_panels = {y: year_curves[y] for y in sel_years}
        year_notes = {y: f'n = {per_state_year_ncount[state][y]}'
                      for y in sel_years}
        out_dir = out_root / f'{state}_year_comparison'
        print(f"\n{state} year comparison grid ({weight_tag}): "
              f"{', '.join(sel_years)}")
        make_comparison_grid(
            year_panels, 'hh_saving',
            f'Household bill saving by year, {state}, optimised, '
            f'{basis_title}', year_notes if draw_notes else {}, out_dir,
            f'household_savings_optimised_mode_by_year_{weight_tag}',
            title_note)
        make_comparison_grid(
            year_panels, 'ws_saving',
            f'Net wholesale saving by year, {state}, {basis_title}',
            year_notes if draw_notes else {}, out_dir, f'wholesale_savings_by_year_{weight_tag}',
            title_note)

    # Tier 3: national pooled view, only when more than one state is present.
    national = None
    if weighted and len(per_state_curves) > 1:
        national = combine_curves_state_weighted(per_state_curves,
                                                 ctx['weights_cfg']['state_mix'])
        csv_rows += _curve_rows('national', national)
        print(f"\nNational (state-weighted) ({weight_tag}):")
        make_all_figures(national, out_root / 'national', weight_tag,
                         ctx['national_info'], basis_title, draw_notes)
    elif not weighted and len(per_state_curves) > 1:
        print("\nNOTE: multiple states found but --unweighted was given, so no "
              "state-weighted national view is produced (state_mix is a "
              "weighting concept). Per-state results above are unaffected.")

    if csv_rows:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(csv_rows).to_csv(out_root / f'curves_{weight_tag}.csv',
                                      index=False)
    if site_frames:
        pd.concat(site_frames, ignore_index=True).to_csv(
            out_root / f'site_savings_{weight_tag}.csv', index=False)
    return national


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('.'),
                        help='Project root; run outputs and site metadata are '
                             'both found at their usual relative paths '
                             'underneath this (default: cwd)')
    parser.add_argument('--output-dir', type=Path, default=Path('analysis/output/financial'),
                        help='Output directory for figures (default: analysis/output/financial)')
    parser.add_argument('--basis', nargs='+', default=['all'],
                        choices=list(RUN_BASES) + ['all'],
                        help='Run bases to draw, each into its own '
                             'subdirectory of --output-dir (default: all). '
                             'The conservative vs achievable comparison is '
                             'drawn when both are.')
    parser.add_argument('--states', nargs='+', default=None,
                        help='Restrict to specific states (default: all found)')
    parser.add_argument('--years', nargs='+', default=None,
                        help='Restrict to specific years (default: all found)')
    parser.add_argument('--unweighted', action='store_true',
                        help='Use straight per-site means instead of size-band '
                             'and state-mix weighting (default: weighting is on)')
    parser.add_argument('--no-notes', dest='notes', action='store_false',
                        help='Leave the data notes (n, period, weighting, '
                             'baseline) off the figures, for the thesis, '
                             'where they go in the caption. They are printed '
                             'either way.')
    parser.add_argument('--only-plan', nargs='+', default=None,
                        metavar='PLAN',
                        choices=sorted(PLAN_KEYS.keys()),
                        help='Presentation override: restrict every chart to '
                             'the named retail plan(s) and move the '
                             'self-consumption reference line onto the first '
                             'one named (instead of Amber). Choices: '
                             + ', '.join(sorted(PLAN_KEYS.keys()))
                             + '. Figures are written to the same paths as '
                               'normal, so this OVERWRITES the full-registry '
                               'charts; re-run without the flag to restore '
                               'them.')
    parser.add_argument('--baseline', choices=['amber', 'own'], default=None,
                        help="What each plan's savings are measured against. "
                             "'amber': every plan against "
                             + DEFAULT_BASELINE_RUN
                             + ", so a curve includes the bill difference "
                               "between that plan's tariff and Amber's. "
                               "'own': each plan against its own solar-only "
                               "run, so a curve is the incremental value of "
                               "the battery on that plan and starts at $0. "
                               "Default: 'amber', or 'own' when --only-plan "
                               "is given.")
    parser.add_argument('--weights-file', type=Path,
                        default=Path('analysis/config/pv_size_bands.yaml'),
                        help='Path to the weighting YAML '
                             '(default: analysis/config/pv_size_bands.yaml)')
    args = parser.parse_args(argv)

    # The module-level rcParams above remain for the scripts that import
    # this module; this script's own figures follow the thesis format.
    mpl.rcdefaults()
    ts.apply_style()

    bases = list(RUN_BASES) if 'all' in args.basis else list(
        dict.fromkeys(args.basis))

    # A single-plan chart measured against a different retailer's solar-only
    # bill would fold a tariff difference into a curve that looks like pure
    # battery value, so --only-plan moves the default to 'own'. An explicit
    # --baseline always wins, so either combination remains reachable.
    baseline_mode = args.baseline or ('own' if args.only_plan else 'amber')
    baseline_desc = ("each plan's own solar-only run (curves start at $0)"
                     if baseline_mode == 'own'
                     else f'{DEFAULT_BASELINE_RUN} (solar, no battery)')

    only_modes = None
    sc_reference_mode = None
    if args.only_plan:
        only_modes = [PLAN_KEYS[k] for k in args.only_plan]
        missing = [k for k in args.only_plan
                   if PLAN_KEYS[k] not in {p['sc_mode'] for p in _BASE_PLANS}]
        if missing:
            print(f"ERROR: --only-plan named {missing}, which is in PLAN_KEYS "
                  f"but not in the RETAIL_PLANS registry. These two must "
                  f"agree; not guessing which one is right.", file=sys.stderr)
            return 1
        sc_reference_mode = only_modes[0]
        print("=" * 72)
        print("SINGLE-PLAN OVERRIDE ACTIVE — these are NOT the full charts.")
        print(f"  Plans shown:              "
              f"{', '.join(_offer_name(p) for p in _BASE_PLANS if p['sc_mode'] in only_modes)}")
        print(f"  Savings baseline:         {baseline_desc}"
              f"{'' if args.baseline else ' [implied by --only-plan]'}")
        print("  Output paths are unchanged, so this OVERWRITES the full")
        print("  charts. Re-run without --only-plan to restore them.")
        print("=" * 72)

    weighted = not args.unweighted
    weight_tag = 'weighted' if weighted else 'unweighted'
    baseline_note = (' vs solar only' if baseline_mode == 'own'
                     else ' vs solar only on Amber')

    all_scenarios = discover_scenarios(args.data_root)
    if not all_scenarios:
        print("ERROR: no scenarios discovered under "
              f"{args.data_root / 'data' / 'output' / 'runs'}", file=sys.stderr)
        return 1
    if args.states:
        all_scenarios = [(s, y) for s, y in all_scenarios if s in args.states]
    if args.years:
        all_scenarios = [(s, y) for s, y in all_scenarios if y in args.years]
    if not all_scenarios:
        print("ERROR: no scenarios match the requested --states/--years filter.",
              file=sys.stderr)
        return 1
    print(f"Scenarios found: {all_scenarios}")
    if not args.only_plan:
        print(f"Savings baseline: {baseline_desc}")

    weights_cfg = None
    if weighted:
        if not args.weights_file.exists():
            print(f"ERROR: weighted mode requires a weights file, "
                  f"none found at {args.weights_file}", file=sys.stderr)
            return 1
        weights_cfg = load_weights_config(args.weights_file)

    # Group requested scenarios by state, preserving discovery order.
    states_in_order: list[str] = []
    scenarios_by_state: dict[str, list[str]] = {}
    for state, year in all_scenarios:
        if state not in scenarios_by_state:
            scenarios_by_state[state] = []
            states_in_order.append(state)
        scenarios_by_state[state].append(year)
    all_years = sorted({y for ys in scenarios_by_state.values() for y in ys})
    national_info = '\n'.join(
        [f"National ({', '.join(sorted(states_in_order))}), "
         f"{_span(all_years)}",
         f"Weighted by PV size & state mix{baseline_note}"]
        + restricted_offer_notes(
            states_in_order,
            [p for p in _BASE_PLANS
             if only_modes is None or p['sc_mode'] in only_modes]))

    ctx = dict(args=args, weighted=weighted, weight_tag=weight_tag,
               weights_cfg=weights_cfg, baseline_mode=baseline_mode,
               baseline_note=baseline_note, only_modes=only_modes,
               sc_reference_mode=sc_reference_mode,
               states_in_order=states_in_order,
               scenarios_by_state=scenarios_by_state,
               national_info=national_info)

    nationals = {basis: _run_basis(basis, ctx) for basis in bases}

    # Achievable and conservative side by side, on one y range: achievable
    # on the left, where the briefings put perfect foresight.
    if nationals.get('conservative') and nationals.get('achievable'):
        out_dir = args.output_dir / 'imperfect_foresight_comparison'
        sets = [('Achievable', nationals['achievable']),
                ('Conservative', nationals['conservative'])]
        notes = national_info if args.notes else None
        print('\nConservative vs achievable comparison:')
        make_basis_pair(sets, 'hh_saving',
                        'Household bill saving by battery size and offer, '
                        'imperfect foresight, optimised',
                        notes, out_dir,
                        f'household_savings_optimised_mode_{weight_tag}')
        make_basis_pair(sets, 'ws_saving',
                        'Net wholesale saving by battery size and offer, '
                        'imperfect foresight',
                        notes, out_dir, f'wholesale_savings_{weight_tag}')

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
