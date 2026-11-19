#!/usr/bin/env python3
"""Retailer briefing charts — battery outcomes for a single retailer.

Purpose
-------
Produces the charts used when the audience is one retailer, or when the
question is about one retailer's plan in isolation. Six briefings are
registered, all driven from the same machinery:

  agl     Their own battery outcomes with and without VPP participation, on
          their own tariff. The quantity of interest is the vertical gap
          between the participating and non-participating curves.

          Each chart is drawn under perfect foresight and imperfect
          foresight side by side, as for Origin. The imperfect foresight
          runs are full_run_rolling_self_consumption_agl,
          full_run_rolling_agl and full_run_rolling_agl_novpp. Self
          consumption without VPP is rule-based, so its one run appears in
          both panels.

  globird Their own battery outcomes with and without critical peak rewards,
          on the ZeroHero tariff. The critical rates pay 100 c/kWh for
          exports while the 5-minute price is above $2,000/MWh and -5 c/kWh
          for imports while it is below -$100/MWh. The optimised runs
          dispatch the battery to them and are paid them; the range is what
          they add over the _novpp runs.

          The _novpp runs switch off only the critical rates. The
          other VPP-like rewards, including the zero import reward and the
          higher export reward during the peak period, remain active in
          both the _novpp and with-VPP runs, so the gap between them is the
          value of the critical rates alone.

          Self consumption is drawn once, from the _novpp run. The with-VPP
          self consumption run (full_run_self_consumption_globird) still
          operates in self consumption mode and ignores the available
          rewards, so it lies on the _novpp line and is not drawn.

          Each chart is drawn under perfect foresight and imperfect
          foresight side by side, as for Origin. The imperfect foresight
          runs are full_run_rolling_globird_novpp and
          full_run_rolling_globird. Self consumption is rule-based and
          cannot respond to critical peak rewards, so it has no rolling run
          and its one run appears in both panels.

  origin  Their own battery outcomes with and without VPP participation,
          the AGL briefing on Origin's plans. The with-VPP runs use the
          Battery Lite VPP tariff (full_run_*_origin_lite); the _novpp runs
          use the Battery Starter TOU tariff (full_run_*_origin_novpp),
          which is the same tariff without the vpp block, so the gap between
          them is the VPP event exports alone. Origin pays no ongoing reward
          for VPP enrolment, only $1/kWh on forced exports during events, so
          the zero-dispatch edge of each range is the _novpp curve on both
          charts. The baseline is the solar-only Battery Starter run.

          Each chart is drawn under perfect foresight and imperfect foresight
          side by side. The imperfect foresight runs
          (full_run_rolling_self_consumption_origin, full_run_rolling_origin,
          full_run_rolling_origin_novpp) dispatch on pre-dispatch price
          forecasts and on load and solar forecasts. Self consumption without
          VPP is rule-based, so its one run appears in both panels. Under
          imperfect foresight, TOU optimisation without VPP falls below self
          consumption at small capacities, and the perfect foresight panel
          beside it shows that this comes from forecast error.

  amber   The same battery on the same tariff under three coordination
          modes — self consumption, perfect foresight, and rolling
          optimisation with imperfect foresight. The quantity of interest is
          how much of the perfect-foresight result survives when the
          optimiser can only see forecasts.

          Imperfect foresight is drawn as a range between two rolling runs
          that differ only in what is known about the interval being
          dispatched. The upper edge, full_run_rolling_amber, acts on its
          settled price and refreshed forecast (current_interval_price:
          price_and_next). The lower edge,
          full_run_rolling_amber_predispatchonly, acts on the pre-dispatch
          forecast available before the interval starts
          (current_interval_price: forecast).

  ea      The GloBird briefing on EnergyAustralia Battery Ease, for NSW
          only, the one state the offer was available in; the runs for the
          other states model estimated versions of it. The VPP pays a
          standing $15 a month for enrolment and nothing per kWh of
          dispatch, so on the household chart the zero-dispatch edge of the
          range is the _novpp curve plus $180 a year, as for AGL's
          enrolment payment. Self consumption ignores VPP events, so it has
          no with-VPP run and is drawn once, from
          full_run_self_consumption_ea, which is on the Battery Ease TOU
          tariff without the VPP. Each chart is drawn under perfect and
          imperfect foresight side by side, as for Origin.

  tesla   The amber briefing on the Tesla Electric Dynamic plan. The
          full_run_*_tesla runs come from the model/configs files of the same
          name: perfect_foresight_advanced and
          rolling_perfect_foresight_lp_advanced dispatching against the
          mixed-plan dispatch prices (docs/model_spec.md), the rolling run
          on price forecasts only. Imperfect foresight is a range as for
          Amber, its lower edge full_run_rolling_tesla_predispatchonly.

This is deliberately a separate script from financial_analysis.py. That
script answers the thesis question (how do retail plans compare to each
other), and its series names, legend layout and figure set are shaped by it.
This one answers a different question for a different audience. Rather than
bending the thesis script further with meeting-specific special cases, this
imports it and reuses its numerics unchanged:

    make_scenario_builder           run loading, per-plan savings baselines,
                                    PV size weighting, curve computation
    average_curves_across_years     multi-year means
    combine_curves_state_weighted   national pooling by state mix

Nothing about the loading, weighting or aggregation is reimplemented here, so
a fix in financial_analysis.py lands in both. What this script owns is which
plans get installed, how the series are drawn, which figures get produced,
and where they go.

The charts are drawn here rather than by financial_analysis.py's views,
because the visual encoding differs: there, linestyle separates optimised
from self consumption across five retailers; here it separates modes within
one retailer. See the visual encoding block below.

How three runs fit a two-slot registry
--------------------------------------
Each registry entry in financial_analysis.py carries exactly two runs, an
sc_dir and an lp_dir, and both the loader and the curve builder iterate
RETAIL_PLANS reading those two fields. A briefing that needs three runs
therefore needs two entries.

Rather than hand-writing those entries and their mode codes, a briefing can
declare a flat list of modes — label, mode key, run directory, colour — and
_plans_from_modes packs them into registry entries two at a time. An odd mode
count leaves one slot spare, which is filled by repeating the first mode's
run under a mode key suffixed __UNUSED. That repeat costs one extra read of a
run already being read and is never drawn; the alternative, a slot naming a
run that does not exist, would make the missing-run check fire on a run
nobody asked for.

Which curves get drawn is governed by the 'series' list, not by the packing:
the packing exists to satisfy the loader, the series list to draw the chart.
Adding a fourth coordination mode (say, actual metered Amber dispatch) is a
mode entry plus a series entry, with no change to the packing.

Baseline
--------
Always --baseline own: every series is measured against the retailer's own
solar-only run, so all curves start at $0 and the tariff difference against
other retailers is divided out. What remains is attributable to the battery
and its coordination mode.

Usage
-----
    python3 analysis/scripts/retailer_briefing.py --retailer amber
    python3 analysis/scripts/retailer_briefing.py --retailer agl
    python3 analysis/scripts/retailer_briefing.py --retailer globird
    python3 analysis/scripts/retailer_briefing.py --retailer origin
    python3 analysis/scripts/retailer_briefing.py --retailer tesla
    python3 analysis/scripts/retailer_briefing.py --retailer ea
    python3 analysis/scripts/retailer_briefing.py --retailer amber \
        --figures state_all_years --states NSW

Writes to analysis/output/retailer_briefings/<retailer>/ by default, well away
from the financial_analysis.py output tree.

Figures follow docs/thesis_figure_spec.md, through thesis_style. Notes on the
data behind each figure (n, period, weighting, box construction) are drawn in
the corner opposite the legend for presentation use, and printed to the
console in either case. --no-notes leaves them off, for the thesis, where
they belong in the caption.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

import financial_analysis as fa
import thesis_style as ts


# ── Visual encoding ───────────────────────────────────────────────────────────
#
# AGL briefing — two binary dimensions on two channels, kept orthogonal:
#
#   linestyle   dotted = VPP participation, solid = no VPP
#   colour      dark   = TOU optimised,     mid   = self consumption
#
# Colour is the stronger grouping channel, so it binds each comparison pair
# (dark-with-dark, mid-with-mid) and the VPP delta is read as the vertical
# gap inside a colour. Linestyle carries the dimension being compared.
#
# Dotted rather than dashed on purpose: financial_analysis.py already uses
# dashed to mean "optimised" and solid to mean "self consumption". Reusing
# dashed here would make one visual channel mean two different things across
# two slides in the same deck. Dotted does not appear in those charts.
#
# The mid tone is deliberately not pale. The weakest combination on the chart
# is mid + dotted, which carries self consumption with VPP — the series most
# customers actually experience, since self consumption is the default mode —
# so it has to survive a projector.
AGL_DARK = '#1018a1'   # AGL's colour in the thesis registry; keeps continuity
AGL_MID = '#6b78d6'

# GloBird briefing — the AGL encoding with the self consumption range
# omitted: colour is TOU optimised (dark) against self consumption (mid), and
# the critical peak rewards range is a dotted upper edge over shading.
#
# GB_DARK is GloBird's colour in the thesis registry, for the same continuity
# reason as AGL_DARK.
GB_DARK = '#e5037f'
GB_MID = '#f06cb0'

# Origin briefing — the AGL encoding. Origin's registry colour (#ffba46) is too
# light to carry a line or marker on white, so both shades are darker steps of
# the same hue: dark for TOU optimised, mid for self consumption.
ORIGIN_DARK = '#a85400'
ORIGIN_MID = '#f0a030'

# Amber briefing — every series is the same retailer on the same tariff, so
# colour is not carrying a distinction here and is spent entirely on identity:
# all series take Amber's registry colour, the one they already have in the
# thesis charts, and the coordination mode is carried by:
#
#   linestyle   solid = self consumption, dashed = perfect foresight,
#               dotted = rolling
#   marker      circle = self consumption, square = perfect foresight,
#               triangle = rolling
#
# Solid + circle for self consumption and dashed + square for the
# full-information optimisation are the conventions financial_analysis.py
# already uses for those two things, so a reader carries them between decks.
# Rolling is new to this chart and takes the remaining combination.
#
# Both channels are used rather than one because the curves converge at small
# capacities, where linestyle alone is hard to resolve, and because linestyle
# is the channel a projector degrades first.
#
# AMBER_INK is not a second series colour. It is a darkened step of the same
# hue used only for marker edges, box outlines and median rules — the small,
# thin marks that the registry colour is too light to carry on white. Lines
# and fills stay the registry colour exactly.
AMBER = '#2bffa4'
AMBER_INK = '#0e8f5c'

# Tesla briefing — the Amber encoding, with Tesla's colour in place of
# Amber's. Tesla Electric is not in financial_analysis.RETAIL_PLANS, so the
# colour is set here; it is dark enough to carry marker edges itself, so
# TESLA_INK is only a slightly darker step for consistency with Amber.
TESLA = '#c8202f'
TESLA_INK = '#8c1521'

# EnergyAustralia briefing — the AGL encoding. EA_DARK is EnergyAustralia's
# colour in the thesis registry, for the same continuity reason as AGL_DARK.
EA_DARK = '#197e2b'
EA_MID = '#5aae61'

# EnergyAustralia Battery Ease VPP pays $15 a month for enrolment,
# irrespective of dispatch, so it sets the floor of the with-VPP range on the
# household bill chart, as AGL_ENROLMENT_PAYMENT does for AGL.
EA_ENROLMENT_PAYMENT = fa.ENROLMENT_PAYMENTS['ea']

# Dash patterns are in multiples of the line width, which with the marker
# sizes comes from thesis_style: ts.MARKS on full-size charts, ts.GRID_MARKS
# on grid panels.
DOTTED = (0, (1.6, 1.6))
DASHED = (0, (5, 2.4))
SOLID = '-'

# Default marker shape. On the AGL briefing markers encode nothing and exist
# only so the ten modelled capacities are visible as discrete points, because
# the curves are piecewise-smoothed between them and would otherwise read as
# continuous model output. On the Amber briefing they additionally carry the
# coordination mode, via the per-series 'marker' key.
MARKER = 'o'

# With-VPP outcomes are drawn as a range rather than a line. Only the upper
# edge is stroked: it is the modelled maximum-dispatch run, so it keeps the
# dotted style and the markers. The lower edge — the no-VPP run plus the
# enrolment payment — is left unstroked and stated by the fill alone, since it
# is a derived bound rather than a modelled result and an extra line there
# would add a stroke without adding information.
BAND_ALPHA = ts.BAND_ALPHA
LEGEND_PATCH_ALPHA = 0.38
# How far the shading hangs below the dotted line in a legend key, as a
# fraction of the key's height. The line itself is not positioned by a
# fraction — see _RangeKeyHandler for why.
LEGEND_KEY_SHADING_DEPTH = 0.55

# AGL Battery Rewards pays this annually for being enrolled and controllable,
# irrespective of whether any dispatch occurs. It therefore sets the floor of
# the with-VPP range on the household bill chart.
AGL_ENROLMENT_PAYMENT = fa.ENROLMENT_PAYMENTS['agl']

# ── Ratio chart geometry ──────────────────────────────────────────────────────
#
# The ratio chart carries one box per series per capacity, so with more than
# one series the boxes have to be dodged off the capacity they describe.
# BOX_DODGE is the gap between box centres and must exceed BOX_WIDTH or the
# boxes in a group overlap; at two series, 1.6-wide boxes 1.8 apart span
# 3.4 kWh of the 5 kWh between capacities. A single series has the capacity
# to itself and takes the wider box, which gives the median rule and the IQR
# more room without crowding its neighbours.
BOX_WIDTH = 1.6
SINGLE_BOX_WIDTH = 2.6
BOX_DODGE = 1.8
# Raised from the value that suited a two-series chart: the fill is now
# the only mark carrying the series, with no line over it.
BOX_ALPHA = 0.30
BOX_EDGE_WIDTH = 0.6
# Whiskers span this percentile range, rather than the 1.5-IQR convention.
# The question the chart answers is what range of outcomes sites actually
# saw, and a stated percentile range answers it directly; a 1.5-IQR whisker
# is a statement about distribution shape, and needs outlier points drawn
# beyond it to stay honest, which puts single sites on a cohort chart.
WHISKER_PCT = (5.0, 95.0)

# Sites whose self-consumption saving at a given capacity is at or below this
# are dropped from the ratio distribution at that capacity. A ratio needs a
# denominator both positive and large enough for the quotient to mean
# something: a site saving $8 under self consumption and $240 under rolling
# is a 30× outcome that describes its denominator, not coordination.
#
# A default, not a fixed rule. Household bill savings and net wholesale
# savings are on different scales, so the same dollar floor is a different
# filter on each; a ratio chart can set min_denominator to override it. The
# count dropped is reported per chart per scenario, so a floor that is wrong
# for a metric shows up as a large drop count rather than as a quietly
# thinned distribution.
RATIO_MIN_DENOMINATOR = 25.0


def _plans_from_modes(modes: list[dict], baseline_run: str,
                      zero_batt_key: str) -> list[dict]:
    """Pack a flat mode list into financial_analysis.RETAIL_PLANS entries.

    Two modes per entry, in the order given. See 'How three runs fit a
    two-slot registry' above for why an odd count repeats the first mode's
    run into the spare slot.

    zero_batt_dir is left None when the briefing's baseline is the study-wide
    solar-only run, because fa._plan_anchor then resolves the zero-capacity
    anchor to $0 without loading anything. Under --baseline own the anchor is
    $0 either way, so naming that run again would only cost a read.

    opt_type and default_mode are inert here — they steer drawing functions
    in financial_analysis.py that this script does not call — but the
    registry shape is validated field for field, so they have to be present.

    A placeholder mode has no run to load, so it is not packed; its curves
    are added after loading by _add_placeholder_curves.
    """
    slots = [m for m in modes if 'placeholder' not in m]
    if len(slots) % 2:
        spare = dict(slots[0])
        spare['key'] = f"{slots[0]['key']}__UNUSED"
        slots.append(spare)

    zero_dir = None if baseline_run == fa.DEFAULT_BASELINE_RUN else baseline_run
    zero_key = None if zero_dir is None else zero_batt_key

    plans = []
    for first, second in zip(slots[0::2], slots[1::2]):
        plans.append(dict(
            sc_label=first['label'],
            lp_label=second['label'],
            sc_mode=first['key'],
            lp_mode=second['key'],
            sc_dir=first['run'],
            sc_no_vpp_dir=None,   # self-consumption mode stands in (fa.sc_no_vpp_mode)
            lp_dir=second['run'],
            zero_batt_dir=zero_dir,
            zero_batt_key=zero_key,
            color=first.get('color', AMBER),
            opt_type='vpp',
            default_mode='sc_dir',
            excluded_states=set(),
        ))
    return plans


# Adding a retailer to the coordination briefing is a mode list in this order
# (self consumption, perfect foresight, rolling, and optionally a second
# rolling run with less information as the lower edge of the imperfect
# foresight range) plus a BRIEFINGS entry built from it by
# _coordination_briefing.
#
# A mode carrying 'placeholder' stands in for a run that has not completed:
# nothing is loaded for it, and its savings are built from two loaded modes as
#     S = (1 − capture) · S_base + capture · S_upper,
# which scales the upper mode's value capture ratio,
# (S_upper − S_base) / (S_pf − S_base), by capture. Delete the 'placeholder'
# entry once the run exists and the run is loaded in its place.
AMBER_MODES = [
    dict(key='SC', label='Self-consumption',
         run='full_run_self_consumption_amber', color=AMBER),
    dict(key='LP', label='Perfect foresight',
         run='full_run_optimised_amber', color=AMBER),
    dict(key='LP_ROLL', label='Imperfect foresight',
         run='full_run_rolling_amber', color=AMBER),
    dict(key='LP_ROLL_PD', label='Imperfect foresight, pre-dispatch only',
         run='full_run_rolling_amber_predispatchonly', color=AMBER),
]

# Modelled Tesla Electric Dynamic runs (model/configs/full_run_*_tesla.yaml).
TESLA_MODES = [
    dict(key='SC_TE', label='Self-consumption',
         run='full_run_self_consumption_tesla', color=TESLA),
    dict(key='LP_TE', label='Perfect foresight',
         run='full_run_optimised_tesla', color=TESLA),
    dict(key='LP_ROLL_TE', label='Imperfect foresight',
         run='full_run_rolling_tesla', color=TESLA),
    dict(key='LP_ROLL_PD_TE', label='Imperfect foresight, pre-dispatch only',
         run='full_run_rolling_tesla_predispatchonly', color=TESLA),
]


def _coordination_briefing(display_name: str, modes: list[dict],
                           baseline_run: str, zero_batt_key: str,
                           color: str, ink: str) -> dict:
    """BRIEFINGS entry comparing one retailer's tariff under its coordination
    modes. modes is (self consumption, perfect foresight, rolling), in that
    order, optionally followed by a second rolling run with less information,
    which makes imperfect foresight a range with the first rolling run as its
    upper edge."""
    keys = [m['key'] for m in modes]
    sc, pf, roll = keys[:3]
    roll_low = keys[3] if len(keys) > 3 else None
    foresight_style = dict(label='Imperfect foresight', color=color,
                           linestyle=DOTTED, marker='^', marker_edge=ink)
    # The range's edges named as in the Energy Masters value capture
    # figures (chart_cohort_rungs.py).
    range_note = (None if roll_low is None else
                  'Range: predispatch only to dispatch price & new predispatch')
    if roll_low is None:
        foresight = dict(kind='line', mode=roll, **foresight_style)
    else:
        # Both edges are model runs, so both are stroked and marked, unlike
        # the VPP ranges, whose lower edge is a derived bound.
        foresight = dict(kind='band', upper_mode=roll, lower_mode=roll_low,
                         stroke_lower=True, **foresight_style)
    return dict(
        display_name=display_name,
        subject='coordination mode',
        baseline_run=baseline_run,
        plans=_plans_from_modes(modes, baseline_run,
                                zero_batt_key=zero_batt_key),
        placeholders=[m for m in modes if 'placeholder' in m],
        # Highest-saving series first, matching the vertical stacking on the
        # chart. Perfect foresight is an upper bound rather than an
        # attainable outcome; it is drawn so the rolling result can be read
        # against it.
        series=[
            dict(kind='line', label='Perfect foresight',
                 mode=pf, color=color, linestyle=DASHED, marker='s',
                 marker_edge=ink),
            foresight,
            dict(kind='line', label='Self-consumption',
                 mode=sc, color=color, linestyle=SOLID, marker='o',
                 marker_edge=ink),
        ],
        charts=[
            dict(metric='hh_saving',
                 title='Household bill saving by battery size and '
                       f'coordination type, {display_name}',
                 ylabel=r'Mean annual saving (\$/site)',
                 filename='coordination_household_savings',
                 grid_title='Household bill saving',
                 legend_loc='lower right', range_note=range_note),
            dict(metric='ws_saving',
                 title='Net wholesale saving by battery size and '
                       f'coordination type, {display_name}',
                 ylabel=r'Mean annual saving (\$/site)',
                 filename='coordination_wholesale_savings',
                 grid_title='Net wholesale saving',
                 legend_loc='upper left', range_note=range_note),
            # One ratio chart per metric, mirroring the two savings charts
            # above, because a household bill ratio and a net wholesale ratio
            # are different quantities: the first is what the customer keeps,
            # the second what the market saves, and the two need not move
            # together.
            #
            # Perfect foresight is deliberately absent from both. It appears
            # on the savings charts, where it is the upper bound the rolling
            # result is read against; carrying it here as well put two sets
            # of boxes on one axes and made the chart harder to read than the
            # number it was reporting. What these charts answer is what an
            # attainable optimiser is worth against the default mode, which
            # is one series: the upper rolling run, as site ratios need a
            # loaded run and a placeholder has none.
            #
            # legend_loc is unused while these charts have a single series
            # each, and is kept for the case where a second is added back.
            dict(kind='ratio', metric='hh_saving', denominator=sc,
                 title='Household bill saving relative to self-consumption, '
                       f'imperfect foresight (achievable), {display_name}',
                 ylabel='Ratio to self-consumption saving',
                 filename='coordination_household_value_ratio',
                 legend_loc='lower right',
                 series=[
                     dict(mode=roll, label='Imperfect foresight',
                          color=color, marker_edge=ink),
                 ]),
            dict(kind='ratio', metric='ws_saving', denominator=sc,
                 title='Net wholesale saving relative to self-consumption, '
                       f'imperfect foresight (achievable), {display_name}',
                 ylabel='Ratio to self-consumption saving',
                 filename='coordination_wholesale_value_ratio',
                 legend_loc='lower right',
                 series=[
                     dict(mode=roll, label='Imperfect foresight',
                          color=color, marker_edge=ink),
                 ]),
        ],
    )

# AGL draws four series, as Origin does: self consumption and TOU optimised,
# each with and without VPP participation, under perfect foresight and,
# from the rolling runs, imperfect foresight, except self consumption
# without VPP, which has no foresight to vary.
AGL_MODES = [
    dict(key='SC_AGL', label='AGL (VPP) self-consumption',
         run='full_run_self_consumption_agl', color=AGL_MID),
    dict(key='LP_AGL', label='AGL (VPP) TOU optimised',
         run='full_run_optimised_agl', color=AGL_DARK),
    dict(key='SC_AGL_NOVPP', label='AGL (no VPP) self-consumption',
         run='full_run_self_consumption_agl_novpp', color=AGL_MID),
    dict(key='LP_AGL_NOVPP', label='AGL (no VPP) TOU optimised',
         run='full_run_optimised_agl_novpp', color=AGL_DARK),
    dict(key='SC_AGL_IF', label='AGL (VPP) self-consumption, rolling',
         run='full_run_rolling_self_consumption_agl', color=AGL_MID),
    dict(key='LP_AGL_IF', label='AGL (VPP) TOU optimised, rolling',
         run='full_run_rolling_agl', color=AGL_DARK),
    dict(key='LP_AGL_NOVPP_IF', label='AGL (no VPP) TOU optimised, rolling',
         run='full_run_rolling_agl_novpp', color=AGL_DARK),
]

# GloBird draws three series: self consumption and TOU optimised without
# critical event rewards, and TOU optimised with them. The two TOU optimised
# series are drawn under perfect foresight and, from the rolling runs,
# imperfect foresight. The five runs are packed two to a registry entry by
# _plans_from_modes, which fills the spare slot with a repeat of the first
# mode's run. full_run_self_consumption_globird is not listed: it is
# identical to the _novpp self consumption run (see the module docstring).
GLOBIRD_MODES = [
    dict(key='SC_GB_NOVPP', label='Self-consumption',
         run='full_run_self_consumption_globird_novpp', color=GB_MID),
    dict(key='LP_GB_NOVPP', label='TOU optimised',
         run='full_run_optimised_globird_novpp', color=GB_DARK),
    dict(key='LP_GB', label='TOU optimised, with critical peak rewards',
         run='full_run_optimised_globird', color=GB_DARK),
    dict(key='LP_GB_NOVPP_IF', label='TOU optimised, rolling',
         run='full_run_rolling_globird_novpp', color=GB_DARK),
    dict(key='LP_GB_IF',
         label='TOU optimised, with critical peak rewards, rolling',
         run='full_run_rolling_globird', color=GB_DARK),
]

# EnergyAustralia draws three series, as GloBird does: self consumption, and
# TOU optimised without and with the VPP, the TOU optimised series under
# perfect foresight and, from the rolling runs, imperfect foresight.
EA_MODES = [
    dict(key='SC_EA', label='Self-consumption',
         run='full_run_self_consumption_ea', color=EA_MID),
    dict(key='LP_EA_NOVPP', label='TOU optimised',
         run='full_run_optimised_ea_novpp', color=EA_DARK),
    dict(key='LP_EA', label='TOU optimised, with VPP',
         run='full_run_optimised_ea', color=EA_DARK),
    dict(key='LP_EA_NOVPP_IF', label='TOU optimised, rolling',
         run='full_run_rolling_ea_novpp', color=EA_DARK),
    dict(key='LP_EA_IF', label='TOU optimised, with VPP, rolling',
         run='full_run_rolling_ea', color=EA_DARK),
]

# Origin draws four series, as AGL does: self consumption and TOU optimised,
# each with and without VPP participation. Each is drawn under perfect
# foresight and, from the rolling runs, imperfect foresight, except self
# consumption without VPP, which has no foresight to vary. The seven runs are
# packed two to a registry entry.
ORIGIN_MODES = [
    dict(key='SC_OR', label='Origin (VPP) self-consumption',
         run='full_run_self_consumption_origin_lite', color=ORIGIN_MID),
    dict(key='LP_OR', label='Origin (VPP) TOU optimised',
         run='full_run_optimised_origin_lite', color=ORIGIN_DARK),
    dict(key='SC_OR_NOVPP', label='Origin (no VPP) self-consumption',
         run='full_run_self_consumption_origin_novpp', color=ORIGIN_MID),
    dict(key='LP_OR_NOVPP', label='Origin (no VPP) TOU optimised',
         run='full_run_optimised_origin_novpp', color=ORIGIN_DARK),
    dict(key='SC_OR_IF', label='Origin (VPP) self-consumption, rolling',
         run='full_run_rolling_self_consumption_origin', color=ORIGIN_MID),
    dict(key='LP_OR_IF', label='Origin (VPP) TOU optimised, rolling',
         run='full_run_rolling_origin', color=ORIGIN_DARK),
    dict(key='LP_OR_NOVPP_IF', label='Origin (no VPP) TOU optimised, rolling',
         run='full_run_rolling_origin_novpp', color=ORIGIN_DARK),
]

BRIEFINGS = {
    'agl': dict(
        display_name='AGL',
        subject='VPP participation',
        baseline_run='full_run_solar_only_agl',
        # All runs share the one solar-only baseline: with no battery there
        # is no VPP to be in or out of, so the curves are directly
        # comparable.
        plans=_plans_from_modes(AGL_MODES, 'full_run_solar_only_agl',
                                zero_batt_key='agl'),
        # Drawing and legend order: highest-saving series first, so the
        # legend reads top-to-bottom in the same order the lines stack on
        # the chart. The retailer and product are named in the title, so
        # the labels carry only what distinguishes the four series. The
        # series name the perfect foresight modes; 'foresight' maps each to
        # its imperfect foresight mode for the second panel.
        series=[
            dict(kind='band', label='TOU optimised, with VPP',
                 upper_mode='LP_AGL', lower_mode='LP_AGL_NOVPP',
                 color=AGL_DARK),
            dict(kind='line', label='TOU optimised',
                 mode='LP_AGL_NOVPP', color=AGL_DARK, linestyle=SOLID),
            dict(kind='band', label='Self-consumption, with VPP',
                 upper_mode='SC_AGL', lower_mode='SC_AGL_NOVPP',
                 color=AGL_MID),
            dict(kind='line', label='Self-consumption',
                 mode='SC_AGL_NOVPP', color=AGL_MID, linestyle=SOLID),
        ],
        foresight=dict(SC_AGL='SC_AGL_IF', LP_AGL='LP_AGL_IF',
                       LP_AGL_NOVPP='LP_AGL_NOVPP_IF'),
        # One entry per output figure. Placement follows the convention in
        # financial_analysis.py: the household chart puts the legend lower
        # right, the wholesale chart upper left, since the curves occupy
        # different parts of the axes.
        charts=[
            dict(metric='hh_saving',
                 title='Household bill saving by battery size, '
                       'AGL Battery Rewards',
                 ylabel=r'Mean annual saving (\$/site)',
                 filename='vpp_participation_value',
                 grid_title='Household bill saving',
                 legend_loc='lower right',
                 # Enrolment payment, received whenever a battery is enrolled
                 # regardless of dispatch, so it applies to the zero-dispatch
                 # floor under either foresight. The maximum-dispatch runs
                 # already include it.
                 vpp_annual_payment=AGL_ENROLMENT_PAYMENT,
                 range_note='VPP range: zero to maximum dispatch'),
            dict(metric='ws_saving',
                 title='Net wholesale saving by battery size, '
                       'AGL Battery Rewards',
                 ylabel=r'Mean annual saving (\$/site)',
                 filename='vpp_wholesale_value',
                 grid_title='Net wholesale saving',
                 legend_loc='upper left',
                 # The enrolment payment is a retail transfer and does not
                 # change wholesale cost, so the zero-dispatch floor is the
                 # no-VPP curve exactly and the band's lower edge coincides
                 # with the solid line.
                 vpp_annual_payment=0.0,
                 range_note='VPP range: zero to maximum dispatch'),
        ],
    ),
    'globird': dict(
        display_name='GloBird',
        subject='VPP participation',
        baseline_run='full_run_solar_only_globird',
        plans=_plans_from_modes(GLOBIRD_MODES, 'full_run_solar_only_globird',
                                zero_batt_key='globird'),
        # Highest-saving series first, as for AGL. The series name the
        # perfect foresight modes; 'foresight' maps each to its imperfect
        # foresight mode for the second panel.
        series=[
            dict(kind='band', label='TOU optimised, with critical peak rewards',
                 upper_mode='LP_GB', lower_mode='LP_GB_NOVPP',
                 color=GB_DARK),
            dict(kind='line', label='TOU optimised',
                 mode='LP_GB_NOVPP', color=GB_DARK, linestyle=SOLID),
            dict(kind='line', label='Self-consumption',
                 mode='SC_GB_NOVPP', color=GB_MID, linestyle=SOLID),
        ],
        foresight=dict(LP_GB='LP_GB_IF', LP_GB_NOVPP='LP_GB_NOVPP_IF'),
        charts=[
            dict(metric='hh_saving',
                 title='Household bill saving by battery size, '
                       'GloBird ZeroHero',
                 ylabel=r'Mean annual saving (\$/site)',
                 filename='vpp_participation_value',
                 grid_title='Household bill saving',
                 legend_loc='lower right',
                 # No enrolment payment is modelled for ZeroHero VPP: rewards
                 # are paid per critical event, so the band's lower edge is
                 # the _novpp curve exactly and coincides with the solid line.
                 vpp_annual_payment=0.0,
                 range_note='Range: critical peak rewards off to on'),
            dict(metric='ws_saving',
                 title='Net wholesale saving by battery size, '
                       'GloBird ZeroHero',
                 ylabel=r'Mean annual saving (\$/site)',
                 filename='vpp_wholesale_value',
                 grid_title='Net wholesale saving',
                 legend_loc='upper left',
                 vpp_annual_payment=0.0,
                 range_note='Range: critical peak rewards off to on'),
        ],
    ),
    'origin': dict(
        display_name='Origin',
        subject='VPP participation',
        baseline_run='full_run_solar_only_origin_starter',
        plans=_plans_from_modes(ORIGIN_MODES,
                                'full_run_solar_only_origin_starter',
                                zero_batt_key='origin'),
        # Highest-saving series first, as for AGL. The series name the
        # perfect foresight modes; 'foresight' maps each to its imperfect
        # foresight mode for the second panel.
        series=[
            dict(kind='band', label='TOU optimised, with VPP',
                 upper_mode='LP_OR', lower_mode='LP_OR_NOVPP',
                 color=ORIGIN_DARK),
            dict(kind='line', label='TOU optimised',
                 mode='LP_OR_NOVPP', color=ORIGIN_DARK, linestyle=SOLID),
            dict(kind='band', label='Self-consumption, with VPP',
                 upper_mode='SC_OR', lower_mode='SC_OR_NOVPP',
                 color=ORIGIN_MID),
            dict(kind='line', label='Self-consumption',
                 mode='SC_OR_NOVPP', color=ORIGIN_MID, linestyle=SOLID),
        ],
        foresight=dict(SC_OR='SC_OR_IF', LP_OR='LP_OR_IF',
                       LP_OR_NOVPP='LP_OR_NOVPP_IF'),
        charts=[
            dict(metric='hh_saving',
                 title='Household bill saving by battery size, '
                       'Origin Battery Lite',
                 ylabel=r'Mean annual saving (\$/site)',
                 filename='vpp_participation_value',
                 grid_title='Household bill saving',
                 legend_loc='lower right',
                 # No enrolment payment: Origin pays only per kWh of forced
                 # event export, so the zero-dispatch floor is the _novpp
                 # curve and the band's lower edge coincides with the solid
                 # line.
                 vpp_annual_payment=0.0,
                 range_note='VPP range: zero to maximum dispatch'),
            dict(metric='ws_saving',
                 title='Net wholesale saving by battery size, '
                       'Origin Battery Lite',
                 ylabel=r'Mean annual saving (\$/site)',
                 filename='vpp_wholesale_value',
                 grid_title='Net wholesale saving',
                 legend_loc='upper left',
                 vpp_annual_payment=0.0,
                 range_note='VPP range: zero to maximum dispatch'),
        ],
    ),
    'ea': dict(
        display_name='EnergyAustralia',
        subject='VPP participation',
        baseline_run='full_run_solar_only_ea',
        plans=_plans_from_modes(EA_MODES, 'full_run_solar_only_ea',
                                zero_batt_key='ea'),
        # The offer was available in NSW only, so the briefing is drawn for
        # NSW alone and its default figure is NSW across all years.
        states=('NSW',),
        # Highest-saving series first, as for AGL. The series name the
        # perfect foresight modes; 'foresight' maps each to its imperfect
        # foresight mode for the second panel.
        series=[
            dict(kind='band', label='TOU optimised, with VPP',
                 upper_mode='LP_EA', lower_mode='LP_EA_NOVPP',
                 color=EA_DARK),
            dict(kind='line', label='TOU optimised',
                 mode='LP_EA_NOVPP', color=EA_DARK, linestyle=SOLID),
            dict(kind='line', label='Self-consumption',
                 mode='SC_EA', color=EA_MID, linestyle=SOLID),
        ],
        foresight=dict(LP_EA='LP_EA_IF', LP_EA_NOVPP='LP_EA_NOVPP_IF'),
        charts=[
            dict(metric='hh_saving',
                 title='Household bill saving by battery size, '
                       'EnergyAustralia Battery Ease',
                 ylabel=r'Mean annual saving (\$/site)',
                 filename='vpp_participation_value',
                 grid_title='Household bill saving',
                 legend_loc='lower right',
                 vpp_annual_payment=EA_ENROLMENT_PAYMENT,
                 range_note='VPP range: zero to maximum dispatch'),
            dict(metric='ws_saving',
                 title='Net wholesale saving by battery size, '
                       'EnergyAustralia Battery Ease',
                 ylabel=r'Mean annual saving (\$/site)',
                 filename='vpp_wholesale_value',
                 grid_title='Net wholesale saving',
                 legend_loc='upper left',
                 # A retail transfer, so no change to wholesale cost.
                 vpp_annual_payment=0.0,
                 range_note='VPP range: zero to maximum dispatch'),
        ],
    ),
    'amber': _coordination_briefing(
        'Amber', AMBER_MODES, 'full_run_solar_only_amber',
        zero_batt_key='amber', color=AMBER, ink=AMBER_INK),
    'tesla': _coordination_briefing(
        'Tesla Electric', TESLA_MODES, 'full_run_solar_only_tesla',
        zero_batt_key='tesla', color=TESLA, ink=TESLA_INK),
}

FIGURE_CHOICES = ['state_year', 'state_all_years', 'national',
                  'state_grid', 'state_year_grid']


def _series_for(briefing: dict, chart: dict) -> list[dict]:
    """The series a chart draws: its own list if it has one, else the
    briefing's. A chart showing a different quantity from the rest of the
    briefing carries its own — the ratio chart has no self-consumption
    series, because self consumption is its denominator.
    """
    return chart.get('series', briefing['series'])


FORESIGHT_PANELS = (('pf', 'Perfect foresight'),
                    ('if', 'Imperfect foresight'))


def _panel_sets(briefing: dict, chart: dict
                ) -> list[tuple[str | None, str | None, list[dict]]]:
    """The (key, subtitle, series) sets a chart draws, one per panel.

    One set, with no key or subtitle, unless the briefing has a 'foresight'
    map, in which case two: the series as listed, under perfect foresight,
    and the same series with each mode swapped for its imperfect foresight
    mode. A mode absent from the map is in both.
    """
    series = _series_for(briefing, chart)
    mapping = briefing.get('foresight')
    if not mapping:
        return [(None, None, series)]
    imperfect = []
    for spec in series:
        s = dict(spec)
        for k in ('mode', 'upper_mode', 'lower_mode'):
            if k in s:
                s[k] = mapping.get(s[k], s[k])
        imperfect.append(s)
    (pf, pf_name), (imp, imp_name) = FORESIGHT_PANELS
    return [(pf, pf_name, series), (imp, imp_name, imperfect)]


def _all_series(briefing: dict):
    for chart in briefing['charts']:
        for _, _, series in _panel_sets(briefing, chart):
            for spec in series:
                yield chart, spec


def _validate_plan_keys(plans: list[dict]) -> None:
    """Fail loudly if the briefing dicts drift from the live registry shape.

    These dicts are duck-typed into financial_analysis.py's loading code, so
    a key added or renamed there would otherwise surface as a KeyError deep
    inside loading, or worse, as a chart that draws but is wrong.
    """
    reference = set(fa.RETAIL_PLANS[0])
    for plan in plans:
        missing = reference - set(plan)
        extra = set(plan) - reference
        if missing or extra:
            raise KeyError(
                f"Briefing plan {plan.get('sc_label', '?')!r} does not match "
                f"the shape of financial_analysis.RETAIL_PLANS entries. "
                f"Missing: {sorted(missing)}. Unexpected: {sorted(extra)}. "
                f"Update BRIEFINGS in this script.")


def _validate_series_modes(briefing: dict) -> None:
    """Fail loudly if a series names a mode no installed plan produces.

    An unknown mode key would otherwise reach the drawing code as a missing
    curve, which is warned about and skipped. That is the right response to a
    run that failed to complete, and the wrong one for a typo in this file,
    which should produce no chart at all.
    """
    available = {m['key'] for m in briefing.get('placeholders', [])}
    for plan in briefing['plans']:
        available.update((plan['sc_mode'], plan['lp_mode']))
    for chart, spec in _all_series(briefing):
        wanted = {spec[k] for k in ('mode', 'upper_mode', 'lower_mode')
                  if k in spec}
        if 'denominator' in chart:
            wanted.add(chart['denominator'])
        unknown = wanted - available
        if unknown:
            raise KeyError(
                f"Chart {chart['filename']!r} names mode(s) {sorted(unknown)}, "
                f"which no installed plan supplies. Available: "
                f"{sorted(available)}. Check the mode keys in BRIEFINGS.")


def _check_runs_exist(data_root: Path, plans: list[dict], baseline_run: str,
                      scenarios: list[tuple[str, str]]) -> list[str]:
    """Return a list of human-readable problems, empty if all runs are present.

    Checked up front against the first scenario so a missing or misnamed run
    is reported before any loading work, rather than as a curve that quietly
    fails to appear on the chart.
    """
    runs_root = data_root / 'data' / 'output' / 'runs'
    state, year = scenarios[0]
    scenario = f'{state}_{year}'
    size = fa.BATTERY_SIZES[0]
    problems = []
    # Solar-only runs sit at scenarios/<scenario>/; battery runs are per
    # capacity, at scenarios/<scenario>_<size>kWh/. Probe the smallest size.
    checks = [(baseline_run, 'solar-only baseline', scenario)]
    for plan in plans:
        checks.append((plan['sc_dir'], plan['sc_label'], f'{scenario}_{size}kWh'))
        checks.append((plan['lp_dir'], plan['lp_label'], f'{scenario}_{size}kWh'))
    # A run repeated into a spare slot is one run, so report it once.
    seen = set()
    for run_name, label, scen_dir in checks:
        if (run_name, scen_dir) in seen:
            continue
        seen.add((run_name, scen_dir))
        f = runs_root / run_name / 'scenarios' / scen_dir / 'bills.csv'
        if not f.exists():
            problems.append(f'  {label}\n      expected: {f}')
    return problems


def _check_placeholders(data_root: Path, placeholders: list[dict],
                        scenarios: list[tuple[str, str]]) -> None:
    """Report each placeholder mode, and whether its run now exists."""
    runs_root = data_root / 'data' / 'output' / 'runs'
    state, year = scenarios[0]
    probe = f'{state}_{year}_{fa.BATTERY_SIZES[0]}kWh'
    for mode in placeholders:
        p = mode['placeholder']
        print(f"  PLACEHOLDER: '{mode['label']}' ({mode['key']}) = "
              f"{1 - p['capture']:g} \u00d7 {p['base']} + {p['capture']:g} "
              f"\u00d7 {p['upper']}, not loaded from {mode['run']}.")
        if (runs_root / mode['run'] / 'scenarios' / probe / 'bills.csv').exists():
            print(f"  NOTE: {mode['run']} now exists. Delete its 'placeholder' "
                  f"entry in this script to draw the run instead.")


def _add_placeholder_curves(curves: dict, placeholders: list[dict]) -> None:
    """Add each placeholder mode's curves to one scenario's curve set.

    Built from the scenario's curves rather than per site: each curve is a
    weighted mean over the sites in the scenario, so a fixed linear
    combination of two curves equals the curve of the same combination of the
    site savings wherever both runs cover the same sites, and it passes
    unchanged through the year and state averaging downstream.
    """
    for mode in placeholders:
        p = mode['placeholder']
        c = p['capture']
        for metric_curves in curves.values():
            base = metric_curves.get(p['base'])
            upper = metric_curves.get(p['upper'])
            if base is None or upper is None:
                continue
            metric_curves[mode['key']] = fa.blend_capture(base, upper, c)


class _RangeKey:
    """Proxy handle for a range series in the legend.

    Not an Artist: it carries only the colour, which edges are stroked and
    the line width, and _RangeKeyHandler builds the key from them. Because it is not an
    Artist it has no label of its own, which is why the legend is given an
    explicit labels list alongside the handles.
    """

    def __init__(self, color: str, both_edges: bool = False,
                 lw: float = ts.DATA_LW):
        self.color = color
        self.both_edges = both_edges
        self.lw = lw


class _RangeKeyHandler:
    """Draw a range key as a dotted line with shading hanging beneath it, or,
    where both edges are stroked on the chart, as shading between two dotted
    lines.

    A bordered box would imply a region bounded on all four sides; what the
    chart shows is a modelled upper bound with a range below it, so the key is
    built the same way round.

    The dotted line is placed at (height - ydescent) / 2, which is not an
    arbitrary choice: it is the expression matplotlib's own HandlerLine2D uses
    to centre a plain line handle. Anchoring to it means the dotted line in a
    range key and the solid line in the key beneath it sit at exactly the same
    height relative to their labels, so the rows read as one list. Computing a
    centre from the handlebox as ydescent + height/2 looks equivalent and is
    not — ydescent is a descent below the box, so that expression sits roughly
    2 * ydescent too high.
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0 = handlebox.xdescent
        width, height = handlebox.width, handlebox.height
        line_y = (height - handlebox.ydescent) / 2
        depth = height * LEGEND_KEY_SHADING_DEPTH
        if orig_handle.both_edges:
            # Centred on the same height as a plain line key.
            top, bottom = line_y + depth / 2, line_y - depth / 2
            edges = (top, bottom)
        else:
            top, bottom = line_y, line_y - depth
            edges = (top,)
        shading = mpatches.Rectangle(
            (x0, bottom), width, top - bottom, facecolor=orig_handle.color,
            alpha=LEGEND_PATCH_ALPHA, edgecolor='none',
            transform=handlebox.get_transform())
        handlebox.add_artist(shading)
        for y in edges:
            line = mlines.Line2D(
                [x0, x0 + width], [y, y], color=orig_handle.color,
                lw=orig_handle.lw, ls=DOTTED,
                transform=handlebox.get_transform())
            handlebox.add_artist(line)
        return shading


def _band_edges(curves: dict, spec: dict, payment: float,
                xs_base: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Upper and lower bound arrays for a with-VPP range series.

    Upper is the modelled maximum-dispatch run, which already includes the
    enrolment payment. Lower is the no-VPP run plus that payment, which is the
    zero-dispatch case: an enrolled battery that is never called has the same
    flows as an unenrolled one, but is still paid for being enrolled.

    The payment is not applied at zero capacity, where there is no battery to
    enrol and nothing to be paid for. Both bounds are zero there by
    construction of the own-plan baseline, so the band closes to the origin
    rather than opening to a spurious payment-sized gap.
    """
    upper = curves.get(spec['upper_mode'])
    lower = curves.get(spec['lower_mode'])
    if upper is None or lower is None:
        return None
    lower = np.asarray(lower, dtype=float).copy()
    lower[xs_base > 0] += payment
    return np.asarray(upper, dtype=float), lower


def _line_handle(ax, spec: dict, xs_sm, ys_sm, xs_raw, ys_raw, zorder: int,
                 marks: ts.Marks = ts.MARKS):
    """Draw one smoothed series with markers at the modelled capacities.

    Returns a proxy handle rather than the drawn line, because the two carry
    different marker settings: the drawn line is the smoothed curve, several
    hundred points long, and giving it a marker would fill it in solid.

    A series may set marker_edge to outline its markers in a darker step of
    its own colour, which is what makes the modelled capacities readable when
    the series colour is too light to carry a marker this small on white.

    The key carries a marker only where the series assigns one, so it appears
    on the Amber charts, where marker shape identifies the coordination mode,
    and not on the AGL charts, where every series has the same marker and a
    key showing it would state nothing.
    """
    marker = spec.get('marker')
    edge = spec.get('marker_edge')
    edge_kw = (dict(markeredgecolor=edge, markeredgewidth=marks.mew)
               if edge else {})
    ax.plot(xs_sm, ys_sm, color=spec['color'], lw=marks.lw,
            ls=spec['linestyle'], zorder=zorder)
    ax.plot(xs_raw, ys_raw, ls='none', marker=marker or MARKER, ms=marks.ms,
            color=spec['color'], zorder=zorder + 1, **edge_kw)
    return mlines.Line2D([], [], color=spec['color'], lw=marks.lw,
                         ls=spec['linestyle'], marker=marker or 'none',
                         ms=marks.ms, label=spec['label'], **edge_kw)


def _draw_series(ax, curves: dict, series: list[dict],
                 vpp_annual_payment: float, quiet: bool = False,
                 marks: ts.Marks = ts.MARKS) -> tuple[list, list[str]]:
    """Draw the briefing series onto axes and return (handles, labels).

    Everything that puts marks on the axes lives here, and nothing that
    frames them does: no titles, labels, limits or legend. That split is what
    lets a full-size chart and one panel of a small-multiples grid share the
    encoding — a series added to a briefing appears in both, drawn the same
    way, without either call site knowing about the other.

    Deliberately not fa._draw_view. That function
    walks RETAIL_PLANS and treats every lp_mode as an optimised series and
    the reference sc_mode as self consumption, which is the thesis encoding.
    Handed a briefing's plans it draws the spare packing slot as a result and
    omits a real mode, because the registry shape it reads carries no record
    of which slots a briefing meant. This walks the series list instead,
    which is that record.

    quiet suppresses the missing-curve warnings, for grid panels where one
    warning per panel per series would bury the run output. marks sets the
    line width and marker sizes, ts.GRID_MARKS on grid panels.
    """
    xs_base = np.concatenate([[0], np.array(fa.BATTERY_SIZES)])
    handles, labels = [], []

    for spec in series:
        if spec['kind'] == 'band':
            edges = _band_edges(curves, spec, vpp_annual_payment, xs_base)
            if edges is None:
                if not quiet:
                    print(f"  WARNING: band '{spec['label']}' needs modes "
                          f"{spec['upper_mode']} and {spec['lower_mode']}; one "
                          f"is missing, so it was omitted.", file=sys.stderr)
                continue
            upper, lower = edges
            xs_sm, up_sm = fa._smooth_xy_piecewise(xs_base, upper)
            _, lo_sm = fa._smooth_xy_piecewise(xs_base, lower)
            ax.fill_between(xs_sm, lo_sm, up_sm, color=spec['color'],
                            alpha=BAND_ALPHA, lw=0, zorder=2)
            # Upper edge: modelled, so it keeps the weight and the markers.
            # The lower edge is stroked only where it is modelled too
            # (stroke_lower). Where it is a derived bound (the VPP ranges'
            # no-VPP run plus enrolment payment) the fill already states
            # where it is, and an edge line there adds a stroke to the chart
            # without adding information.
            stroked = [(upper, up_sm)]
            if spec.get('stroke_lower'):
                stroked.append((lower, lo_sm))
            edge = spec.get('marker_edge')
            edge_kw = (dict(markeredgecolor=edge,
                            markeredgewidth=marks.mew) if edge else {})
            for ys_raw, ys_sm in stroked:
                ax.plot(xs_sm, ys_sm, color=spec['color'], lw=marks.lw,
                        ls=spec.get('linestyle', DOTTED), zorder=3)
                ax.plot(xs_base, ys_raw, ls='none',
                        marker=spec.get('marker', MARKER), ms=marks.ms,
                        color=spec['color'], zorder=4, **edge_kw)
            handles.append(_RangeKey(spec['color'],
                                     both_edges=bool(spec.get('stroke_lower')),
                                     lw=marks.lw))
            labels.append(spec['label'])
            continue

        ys_raw = curves.get(spec['mode'])
        if ys_raw is None:
            if not quiet:
                print(f"  WARNING: series '{spec['label']}' (mode "
                      f"{spec['mode']}) has no curve and was omitted from the "
                      f"chart.", file=sys.stderr)
            continue
        xs_sm, ys_sm = fa._smooth_xy_piecewise(xs_base, ys_raw)
        handles.append(_line_handle(ax, spec, xs_sm, ys_sm, xs_base, ys_raw, 3,
                                    marks))
        labels.append(spec['label'])

    # Sweep axis: ticks at the modelled capacities, 3 % padding either side.
    pad = 0.03 * xs_base[-1]
    ax.set_xlim(xs_base[0] - pad, xs_base[-1] + pad)
    ax.set_xticks(xs_base)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'${v:,.0f}' if v >= 0 else f'\u2212${-v:,.0f}'))
    return handles, labels


# The notes go in the corner diagonally opposite the legend, which on every
# chart here is the corner the curves leave clear.
_NOTE_LOC = {'lower right': 'upper left', 'upper left': 'lower right',
             'upper right': 'lower left', 'lower left': 'upper right'}


def _draw_notes(ax, text: str, legend_loc: str) -> None:
    """Data notes in the corner opposite the legend, grey 0.3 at the data
    note size, on a white box so gridlines and data behind them do not show
    through."""
    loc = _NOTE_LOC[legend_loc]
    x, ha = (0.02, 'left') if 'left' in loc else (0.98, 'right')
    y, va = (0.97, 'top') if 'upper' in loc else (0.04, 'bottom')
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=ts.NOTE_SIZE, color=ts.REFERENCE_COLOUR,
            bbox=ts.LABEL_BOX, zorder=6)


def _axes_briefing(ax, curves: dict, ylabel: str, title: str,
                   series: list[dict], legend_loc: str = 'lower right',
                   vpp_annual_payment: float = 0.0) -> None:
    """One full-size briefing chart: the series, framed and titled.

    Series are drawn in the order given and the legend is built from the same
    list, so legend order matches the vertical stacking on the chart.
    """
    handles, labels = _draw_series(ax, curves, series, vpp_annual_payment)

    ax.set_xlabel('Battery capacity (kWh)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Single column, so the legend reads as an ordered ranking rather than a
    # grid. No style key: each series is named outright.
    # Taller, longer keys than the default: the range key needs vertical room
    # for the shading under its dotted line to be legible at legend scale.
    ax.legend(handles=handles, labels=labels, loc=legend_loc, ncol=1,
              handleheight=1.5, handlelength=2.8,
              handler_map={_RangeKey: _RangeKeyHandler()})


# ── Ratio chart: coordination value relative to self consumption ──────────────
#
# Boxes only. Each site contributes one ratio at each capacity — its saving
# under coordination over its saving under self consumption — weighted by the
# same PV size-band weight it carries on the savings charts. The box reports
# the weighted median, the interquartile range and the stated percentiles of
# that distribution.
#
# An earlier version also drew the cohort ratio (weighted mean saving under
# coordination over weighted mean saving under self consumption) as a line
# through the boxes. On the EDP set it lands on the box median, so it was a
# second mark saying what the median already said, and it is not drawn.
#
# The two are different quantities that happen to coincide here, not the same
# quantity computed twice: a ratio of means is dollar-weighted, so a site
# saving $600 under self consumption moves it further than one saving $200,
# where every site moves the median equally. Should a later run have the
# sites with the largest self-consumption savings gaining disproportionately
# from coordination, the cohort ratio would pull away from the median and be
# worth restoring — it is c['LP_ROLL'][1:] / c['SC'][1:] on the curve set,
# which _axes_ratio would then have to be handed.
#
# Site ratios are aggregated across scenarios as observations rather than as
# curves, so a site appearing in two years contributes two observations and
# the boxes describe the spread of site-years. Their weights carry the same
# structure the curve aggregation uses — each year within a state contributes
# equally, states are combined by state_mix — so the boxes describe the same
# population the savings charts do.


def _weighted_quantile(values: np.ndarray, weights: np.ndarray,
                       quantiles: np.ndarray) -> np.ndarray:
    """Weighted quantiles by linear interpolation of the weighted CDF.

    Each sorted observation is placed at the cumulative weight below it,
    normalised so the smallest sits at 0 and the largest at 1. That is the
    weighted generalisation of numpy's default (linear) quantile, and reduces
    to it exactly when the weights are equal — so an --unweighted run's boxes
    can be checked against a plain np.quantile or pandas .quantile call.

    Weights are not repetition counts, so a weight of 3 does not give the same
    answer as three copies of the observation. Nothing here relies on it
    doing so; the weights are population shares.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if v.size == 0:
        return np.full(len(quantiles), np.nan)
    if v.size == 1:
        return np.full(len(quantiles), v[0])
    order = np.argsort(v)
    v, w = v[order], w[order]
    pos = np.cumsum(w) - w
    return np.interp(quantiles, pos / pos[-1], v)


def _size_slice(mode_frame, size: int, metric: str):
    """One mode's values at one capacity, indexed by site.

    Duplicate site rows would multiply silently on the index intersection
    below, so the first is kept and the rest dropped.
    """
    sub = mode_frame[mode_frame['battery_kwh'] == size]
    sub = sub[~sub['site_id'].duplicated()]
    return sub.set_index('site_id')[metric]


def _collect_ratio_samples(savings, site_weight, charts: list[dict]
                           ) -> tuple[dict, list[tuple[str, float, int, int]]]:
    """Per-site ratio observations for every ratio chart in one scenario.

    Returns ({(metric, mode, size): (values, weights)}, reports), where each
    report is (chart filename, floor applied, n_dropped, n_total). Counts are
    per chart rather than pooled, because the charts can be on different
    metrics with different floors, and a pooled count would hide a floor that
    is wrong for one of them.

    Sites are intersected across the numerator run, the denominator run and
    the weight index, so a site missing from any one of them contributes to
    none of the boxes rather than to some of them.
    """
    samples: dict = {}
    reports: list[tuple[str, float, int, int]] = []
    by_mode = {mode: g for mode, g in savings.groupby('mode', sort=False)}

    for chart in charts:
        if chart.get('kind') != 'ratio':
            continue
        metric = chart['metric']
        floor = chart.get('min_denominator', RATIO_MIN_DENOMINATOR)
        den_all = by_mode.get(chart['denominator'])
        if den_all is None:
            continue
        dropped = total = 0
        for spec in chart['series']:
            num_all = by_mode.get(spec['mode'])
            if num_all is None:
                continue
            for size in fa.BATTERY_SIZES:
                den = _size_slice(den_all, size, metric)
                num = _size_slice(num_all, size, metric)
                common = den.index.intersection(num.index)
                if site_weight is not None:
                    common = common.intersection(site_weight.index)
                if common.empty:
                    continue
                d, n = den.loc[common], num.loc[common]
                keep = d > floor
                total += len(common)
                dropped += int((~keep).sum())
                if not keep.any():
                    continue
                vals = (n[keep] / d[keep]).to_numpy(dtype=float)
                if site_weight is None:
                    w = np.ones(vals.size, dtype=float)
                else:
                    w = site_weight.loc[common][keep].to_numpy(dtype=float)
                samples[(metric, spec['mode'], size)] = (vals, w)
        reports.append((chart['filename'], floor, dropped, total))
    return samples, reports


def _scale_samples(samples: dict, factor: float) -> dict:
    """Rescale every observation weight, for combining scenarios."""
    return {k: (v, w * factor) for k, (v, w) in samples.items()}


def _merge_samples(sample_sets: list[dict]) -> dict:
    """Concatenate observations for the same (metric, mode, capacity)."""
    merged: dict = {}
    for samples in sample_sets:
        for key, (v, w) in samples.items():
            if key in merged:
                prev_v, prev_w = merged[key]
                merged[key] = (np.concatenate([prev_v, v]),
                               np.concatenate([prev_w, w]))
            else:
                merged[key] = (v.copy(), w.copy())
    return merged


def _box_geometry(n: int) -> tuple[float, list[float]]:
    """Box width and the horizontal offsets centring n boxes on a capacity."""
    if n == 1:
        return SINGLE_BOX_WIDTH, [0.0]
    span = BOX_DODGE * (n - 1)
    return BOX_WIDTH, [-span / 2 + i * BOX_DODGE for i in range(n)]


def _draw_box(ax, x: float, stats: np.ndarray, width: float, face: str,
              edge: str) -> None:
    """One box: IQR rectangle, median rule, whiskers to the stated percentiles.

    stats is (p_low, q1, median, q3, p_high). face is the series colour and
    edge the darker step of it that carries the thin marks; passing the same
    value for both draws a single-colour box.
    """
    p_lo, q1, med, q3, p_hi = stats
    half = width / 2
    cap = half * 0.45
    for y0, y1 in ((p_lo, q1), (q3, p_hi)):
        ax.plot([x, x], [y0, y1], color=edge, lw=BOX_EDGE_WIDTH, zorder=2)
    for y in (p_lo, p_hi):
        ax.plot([x - cap, x + cap], [y, y], color=edge, lw=BOX_EDGE_WIDTH,
                zorder=2)
    ax.add_patch(mpatches.Rectangle(
        (x - half, q1), width, q3 - q1, facecolor=face, alpha=BOX_ALPHA,
        edgecolor=edge, lw=BOX_EDGE_WIDTH, zorder=2))
    ax.plot([x - half, x + half], [med, med], color=edge,
            lw=BOX_EDGE_WIDTH * 1.6, zorder=3)


def _box_note(weighted: bool) -> str:
    """The box construction, printed with the ratio chart's caption notes.

    Names the plotted values as individual site ratios, because the notes
    otherwise carry the same "Weighted by PV size & state mix" line as the
    savings charts, where weighting produces a weighted mean. Here it does
    not touch the values at all: every value drawn is one site's own ratio,
    unweighted, and the weights set only how much of the distribution each
    site occupies, and so where the quantile marks fall.

    Built from WHISKER_PCT so the note cannot drift from the whiskers drawn,
    and from the run's weighting so an --unweighted figure does not claim a
    weighted median.
    """
    mark = 'weighted median' if weighted else 'median'
    return (f'Boxes: individual site ratios \u2014 {mark} and IQR, '
            f'whiskers {WHISKER_PCT[0]:.0f}\u2013{WHISKER_PCT[1]:.0f}th pct')


def _axes_ratio(ax, samples: dict, chart: dict) -> None:
    """Draw the coordination-value ratio chart.

    Takes no curve set. Site ratios are computed from the savings frame, not
    from the curves, so this chart is defined in views where the curves are
    not — see the unweighted national view in main().

    Zero capacity is not plotted: with no battery both numerator and
    denominator are $0 by construction of the own-plan baseline, so the ratio
    there is 0/0 rather than 1.
    """
    metric = chart['metric']
    sizes = np.array(fa.BATTERY_SIZES, dtype=float)
    quantiles = np.array([WHISKER_PCT[0] / 100, 0.25, 0.5, 0.75,
                          WHISKER_PCT[1] / 100])
    series = chart['series']
    box_width, offsets = _box_geometry(len(series))
    handles, labels = [], []
    drawn_y: list[float] = []

    for spec, dx in zip(series, offsets):
        edge = spec.get('marker_edge', spec['color'])
        drawn_any = False
        for size, x in zip(fa.BATTERY_SIZES, sizes):
            drawn = samples.get((metric, spec['mode'], size))
            if drawn is None:
                continue
            stats = _weighted_quantile(drawn[0], drawn[1], quantiles)
            if np.isnan(stats).any():
                continue
            _draw_box(ax, x + dx, stats, box_width, spec['color'], edge)
            drawn_y.extend((stats[0], stats[-1]))
            drawn_any = True

        if not drawn_any:
            print(f"  WARNING: ratio series '{spec['label']}' (mode "
                  f"{spec['mode']}) yielded no site observations and was "
                  f"omitted.", file=sys.stderr)
            continue
        # Series identity is carried by the box, so the key is a box too.
        handles.append(mpatches.Patch(facecolor=spec['color'], alpha=BOX_ALPHA,
                                      edgecolor=edge, lw=BOX_EDGE_WIDTH))
        labels.append(spec['label'])

    # Parity with self consumption. Drawn beneath everything, being a
    # reference value rather than a result.
    ax.axhline(1.0, color=ts.REFERENCE_COLOUR, lw=ts.REFERENCE_LW, ls='--',
               zorder=1)

    ax.set_xlabel('Battery capacity (kWh)')
    ax.set_ylabel(chart['ylabel'])
    ax.set_title(chart['title'])
    # Room for the outermost boxes, which hang half a box width plus their
    # dodge beyond the smallest and largest capacities.
    pad = box_width / 2 + max(abs(o) for o in offsets) + 0.6
    ax.set_xlim(sizes[0] - pad, sizes[-1] + pad)
    # Parity is always in frame, with clearance beneath it, so that the
    # distance of every series from 1.0 is readable and the reference line
    # does not sit on the axis.
    if drawn_y:
        lo, hi = min(drawn_y + [1.0]), max(drawn_y + [1.0])
        span = max(hi - lo, 0.1)
        ax.set_ylim(lo - 0.12 * span, hi + 0.10 * span)
    ax.set_xticks(sizes)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'{v:.1f}\u00d7'))

    # One series needs no legend: the title names it, and what the box means
    # is stated in the caption rather than as a key.
    if len(handles) > 1:
        ax.legend(handles=handles, labels=labels, loc=chart['legend_loc'],
                  ncol=1)


# ── Small-multiples grids ─────────────────────────────────────────────────────
#
# The equivalents of financial_analysis.py's national_state_comparison grids,
# for a briefing's own series. Two are produced: states side by side (each
# panel that state's mean across its years), and the full state x year grid
# (rows states, columns years, blank where a state has no data for a year).
#
# Panels share one y-scale and one legend so they are comparable at a glance,
# which is the only reason to put them on one figure rather than in a folder.
# Only 'series' charts get grids: a panel of box plots at grid scale would be
# a few pixels of box per capacity.


GRID_COLS = 4                     # state grid: the states in one row
# Panel heights that make the panels about 1.4 times as wide as they are tall.
STATE_PANEL_HEIGHT_IN = 1.0
STATE_YEAR_PANEL_HEIGHT_IN = 0.82
# Figure title, x-axis label and the legend row, in inches.
GRID_FURNITURE_IN = 0.85
# Grid panels are too narrow to label every modelled capacity, so they label
# every other one.
GRID_TICK_STEP = 10


def _shared_ylim(curve_sets: list[dict], metric: str, series: list[dict],
                 pad_frac: float = 0.08) -> tuple[float, float] | None:
    """Common y-range across several curve sets, over the modes the series
    actually draw.

    Not fa._shared_ylim, which spans the thesis series. Every mode a series
    references is included, band edges as well as lines, so no panel can be
    clipped by a range computed from series it does not contain.
    """
    modes = set()
    for spec in series:
        modes.update(spec[k] for k in ('mode', 'upper_mode', 'lower_mode')
                     if k in spec)
    vals = [np.asarray(cs[metric][m], dtype=float)
            for cs in curve_sets if metric in cs
            for m in modes if m in cs[metric]]
    if not vals:
        return None
    allv = np.concatenate(vals)
    allv = allv[~np.isnan(allv)]
    if allv.size == 0:
        return None
    lo, hi = float(allv.min()), float(allv.max())
    span = (hi - lo) if hi > lo else max(abs(hi), 1.0)
    return lo - span * pad_frac, hi + span * pad_frac


def _grid_cells(labels: list[str], n_cols: int) -> list[list[str | None]]:
    """Row-major fill, padding the last row with None for hidden panels."""
    rows = -(-len(labels) // n_cols)  # ceil
    it = iter(labels)
    return [[next(it, None) for _ in range(n_cols)] for _ in range(rows)]


def make_briefing_grid(cells: list[list[str | None]], panels: dict[str, dict],
                       chart: dict, series: list[dict], suptitle: str,
                       panel_notes: dict[str, str], output_dir: Path,
                       stem: str, panel_height_in: float) -> None:
    """One small-multiples grid of a briefing chart's series.

    cells is row-major, None for a blank cell. panels maps a cell label to
    that cell's curve set, and panel_notes a cell label to the site count
    drawn in the panel's top left corner, which the curves leave clear; an
    empty panel_notes draws none, for the thesis. The legend is drawn once for the figure, from
    whichever panel produced the most handles — a panel missing a series
    (a state with no data for one run) would otherwise decide the legend for
    every panel that has it.
    """
    n_rows = len(cells)
    if not n_rows:
        return
    n_cols = len(cells[0])
    flat = [c for row in cells for c in row if c is not None]
    metric = chart['metric']
    payment = chart.get('vpp_annual_payment', 0.0)
    ylim = _shared_ylim([panels[l] for l in flat], metric, series)

    fig = ts.new_figure(panel_height_in * n_rows + GRID_FURNITURE_IN)
    axes = ts.grid_axes(fig, n_rows, n_cols, 'Battery capacity (kWh)')
    legend_handles: list = []
    legend_labels: list[str] = []

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r][c]
            label = cells[r][c]
            curves = None if label is None else panels[label].get(metric)
            if curves is None:
                ax.axis('off')
                continue
            handles, labels = _draw_series(ax, curves, series, payment,
                                           quiet=True, marks=ts.GRID_MARKS)
            if len(handles) > len(legend_handles):
                legend_handles, legend_labels = handles, labels
            ax.set_title(label, fontsize=ts.SUBTITLE_SIZE)
            note = panel_notes.get(label)
            if note:
                _draw_notes(ax, note, 'lower right')
            ax.xaxis.set_major_locator(mticker.MultipleLocator(GRID_TICK_STEP))
            if ylim is not None:
                ax.set_ylim(*ylim)

            # Tick labels on the outer edges only. "Bottom" is the lowest
            # panel in this column that has content, not the last row, so a
            # column that runs short still gets its x tick labels.
            is_bottom = all(cells[rr][c] is None for rr in range(r + 1, n_rows))
            ax.tick_params(labelleft=(c == 0), labelbottom=is_bottom)

    fig.supylabel(chart['ylabel'],
                  fontsize=matplotlib.rcParams['axes.labelsize'])
    fig.legend(handles=legend_handles, labels=legend_labels,
               loc='outside lower center', ncol=len(legend_handles),
               frameon=False, handleheight=1.5, handlelength=2.8,
               handler_map={_RangeKey: _RangeKeyHandler()})
    ts.panel_suptitle(fig, suptitle)
    ts.save_figure(fig, output_dir / stem)


def make_briefing_grids(panels: dict[str, dict], cells: list[list[str | None]],
                        briefing: dict, scope: str, weight_tag: str,
                        panel_notes: dict[str, str], output_dir: Path,
                        panel_height_in: float,
                        title_note: str | None = None) -> None:
    """Emit one grid per 'series' chart in the briefing, for one scope.

    A chart drawn under both foresights gets a grid for each, rather than
    a grid of side-by-side pairs, which at state x year would be forty
    panels; the foresight is named in the title and the file name.

    title_note, where given, is appended to the figure title: the grid's
    equivalent of the notes on a single chart, the site counts being in the
    panels already.
    """
    for chart in briefing['charts']:
        if chart.get('kind', 'series') != 'series':
            continue
        for key, name, series in _panel_sets(briefing, chart):
            suptitle = (f"{chart['grid_title']} by {scope}, "
                        f"{briefing['display_name']}")
            stem = f"{chart['filename']}_by_{scope.replace(' ', '_')}"
            if key:
                suptitle += f', {name.lower()}'
                stem += f'_{key}'
            if title_note:
                suptitle += f' \u2014 {title_note}'
            make_briefing_grid(cells, panels, chart, series, suptitle,
                               panel_notes, output_dir,
                               f'{stem}_{weight_tag}', panel_height_in)


def _figure_pair(curves: dict, chart: dict,
                 sets: list[tuple[str, str, list[dict]]],
                 notes: str | None):
    """A chart's panel sets side by side on a shared y range, under one
    title and one legend. notes, where given, go in the bottom right corner
    of each panel."""
    fig = ts.new_figure(height_in=3.3)
    axes = fig.subplots(1, len(sets), sharey=True)
    payment = chart.get('vpp_annual_payment', 0.0)
    for ax, (_, name, series) in zip(axes, sets):
        handles, labels = _draw_series(ax, curves, series, payment)
        ax.set_title(name, fontsize=ts.SUBTITLE_SIZE)
        ax.set_xlabel('Battery capacity (kWh)')
        if notes:
            _draw_notes(ax, notes, 'upper left')
    fig.supylabel(chart['ylabel'],
                  fontsize=matplotlib.rcParams['axes.labelsize'])
    fig.legend(handles=handles, labels=labels, loc='outside lower center',
               ncol=len(handles), frameon=False, handleheight=1.5,
               handlelength=2.8, handler_map={_RangeKey: _RangeKeyHandler()})
    ts.panel_suptitle(fig, chart['title'])
    return fig


def make_briefing_figures(curve_set: dict | None, samples: dict,
                          output_dir: Path, weight_tag: str, info_text: str,
                          briefing: dict, kinds: set[str] | None = None,
                          draw_notes: bool = True) -> None:
    """Emit the briefing's charts from one curve set and one sample set.

    curve_set may be None for a view where the curves are not defined but the
    site ratios still are — the unweighted national view, where pooling state
    means would need state_mix but pooling site ratios does not. Charts that
    need curves are skipped there. kinds restricts the chart kinds drawn, so
    that a caller which knows in advance a view supports only some of them
    says so rather than relying on the per-chart skips.

    info_text, the chart's range_note and, on ratio charts, the box
    construction are printed for the caption, and drawn on the chart unless
    draw_notes is False, in the corner opposite the legend on a single
    chart and in the bottom right corner of each panel of a pair.

    A series chart whose briefing has a 'foresight' map is drawn as a pair
    of panels (see _panel_sets) under the chart's file name.
    """
    weighted = weight_tag == 'weighted'
    for chart in briefing['charts']:
        kind = chart.get('kind', 'series')
        if kinds is not None and kind not in kinds:
            continue
        notes = [n for n in (info_text, chart.get('range_note')) if n]
        ax = None

        if kind == 'ratio':
            if not any(key[0] == chart['metric'] for key in samples):
                print(f"  WARNING: no site ratios collected for metric "
                      f"'{chart['metric']}'; skipping {chart['filename']}.",
                      file=sys.stderr)
                continue
            fig = ts.new_figure(height_in=3.0)
            ax = fig.add_subplot()
            _axes_ratio(ax, samples, chart)
            notes.append(_box_note(weighted))
        else:
            curves = curve_set.get(chart['metric']) if curve_set else None
            if curves is None:
                print(f"  WARNING: metric '{chart['metric']}' absent from the "
                      f"curve set; skipping {chart['filename']}.",
                      file=sys.stderr)
                continue
            sets = _panel_sets(briefing, chart)
            if len(sets) > 1:
                fig = _figure_pair(curves, chart, sets,
                                   '\n'.join(notes) if draw_notes else None)
            else:
                fig = ts.new_figure(height_in=3.0)
                ax = fig.add_subplot()
                _axes_briefing(ax, curves, chart['ylabel'], chart['title'],
                               sets[0][2], legend_loc=chart['legend_loc'],
                               vpp_annual_payment=chart.get(
                                   'vpp_annual_payment', 0.0))
        if draw_notes and ax is not None:
            _draw_notes(ax, '\n'.join(notes), chart['legend_loc'])
        ts.save_figure(fig, output_dir / f"{chart['filename']}_{weight_tag}")
        print('    ' + '; '.join(n.replace('\n', ', ') for n in notes))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--retailer', choices=sorted(BRIEFINGS), default='amber',
                        help='Which briefing registry to install (default: amber)')
    parser.add_argument('--data-root', type=Path, default=Path('.'),
                        help='Project root (default: cwd)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory (default: '
                             'analysis/output/retailer_briefings/<retailer>)')
    parser.add_argument('--figures', nargs='+', default=None,
                        choices=FIGURE_CHOICES + ['all'],
                        help="Which figures to produce. 'national': pooled "
                             "state-weighted view. 'state_all_years': one per "
                             "state, mean across years. 'state_year': one per "
                             "state-year. 'state_grid': one small-multiples "
                             "grid comparing states. 'state_year_grid': one "
                             "grid of states by years. Both grids go to "
                             "national_state_comparison/. Default: national, "
                             "or state_all_years for a briefing restricted "
                             "to one state.")
    parser.add_argument('--states', nargs='+', default=None,
                        help='Restrict to these states (default: all found)')
    parser.add_argument('--years', nargs='+', default=None,
                        help='Restrict to these years (default: all found)')
    parser.add_argument('--unweighted', action='store_true',
                        help='Disable PV size weighting')
    parser.add_argument('--weights-file', type=Path,
                        default=Path('analysis/config/pv_size_bands.yaml'),
                        help='PV size band weights config')
    parser.add_argument('--no-notes', dest='notes', action='store_false',
                        help='Leave the data notes (n, period, weighting) '
                             'off the charts, for the thesis, where they go '
                             'in the caption. They are printed either way.')
    args = parser.parse_args(argv)

    # financial_analysis sets its own rcParams on import; start from the
    # matplotlib defaults so none of them survive under the thesis style.
    matplotlib.rcdefaults()
    ts.apply_style()

    briefing = BRIEFINGS[args.retailer]
    placeholders = briefing.get('placeholders', [])
    # A briefing for an offer available in only some states is drawn for
    # those states alone, and with one state has no national view.
    briefing_states = briefing.get('states')
    requested = args.figures or (
        ['state_all_years'] if briefing_states and len(briefing_states) == 1
        else ['national'])
    figures = set(FIGURE_CHOICES if 'all' in requested else requested)
    output_dir = args.output_dir or (Path('analysis/output/retailer_briefings')
                                     / args.retailer)
    weighted = not args.unweighted
    weight_tag = 'weighted' if weighted else 'unweighted'

    scenarios = fa.discover_scenarios(args.data_root)
    if briefing_states:
        scenarios = [(s, y) for s, y in scenarios if s in briefing_states]
    if args.states:
        scenarios = [(s, y) for s, y in scenarios if s in args.states]
    if args.years:
        scenarios = [(s, y) for s, y in scenarios if y in args.years]
    if not scenarios:
        print('ERROR: no scenarios match the requested --states/--years filter.',
              file=sys.stderr)
        return 1

    plans = briefing['plans']
    _validate_plan_keys(plans)
    _validate_series_modes(briefing)

    problems = _check_runs_exist(args.data_root, plans,
                                 briefing['baseline_run'], scenarios)
    if problems:
        print('ERROR: the following runs this briefing needs were not found:\n'
              + '\n'.join(problems)
              + '\n\n  If those runs completed under different directory '
                'names, update the run names in the mode lists at the top '
                'of this script.',
              file=sys.stderr)
        return 1

    _check_placeholders(args.data_root, placeholders, scenarios)

    # Installing the briefing registry replaces the six thesis plans for this
    # process only; financial_analysis.py run on its own is unaffected.
    fa.install_plans(plans, plans[0]['sc_mode'])

    weights_cfg = None
    if weighted:
        if not args.weights_file.exists():
            print(f'ERROR: weighted mode requires a weights file, none found '
                  f'at {args.weights_file}', file=sys.stderr)
            return 1
        weights_cfg = fa.load_weights_config(args.weights_file)

    drawn = list(dict.fromkeys(spec['label']
                               for _, spec in _all_series(briefing)))
    print('=' * 72)
    print(f"RETAILER BRIEFING — {briefing['display_name']} "
          f"({briefing['subject']})")
    print(f"  Series:      {', '.join(drawn)}")
    print(f"  Baseline:    own plan, solar only ({briefing['baseline_run']}) "
          f"— all curves start at $0")
    print(f"  Figures:     {', '.join(sorted(figures))}")
    print(f"  Output:      {output_dir}")
    print('=' * 72)

    build_scenario = fa.make_scenario_builder(args.data_root, weights_cfg, 'own')

    scenarios_by_state: dict[str, list[str]] = {}
    for state, year in scenarios:
        scenarios_by_state.setdefault(state, []).append(year)

    # Notes: scope (with the site count where it is not national), then the
    # weighting and baseline on one line, so that a range note makes three.
    weight_label = 'Weighted by PV size' if weighted else 'Unweighted'
    baseline_note = ' vs solar only'
    per_state_curves: dict[str, dict] = {}
    per_state_samples: dict[str, dict] = {}
    per_state_counts: dict[str, list[int]] = {}
    # Retained for the grids: the per-state-year curve sets are otherwise
    # averaged away into per_state_curves as each state finishes.
    per_state_year_curves: dict[str, dict[str, dict]] = {}
    per_state_year_counts: dict[str, dict[str, int]] = {}

    for state, years in scenarios_by_state.items():
        year_curve_sets = []
        year_sample_sets = []
        for year in years:
            print(f'\nLoading {state} {year}...')
            built = build_scenario(state, year)
            if built is None:
                continue
            curves, n_sites = built['curves'], built['n_sites']
            _add_placeholder_curves(curves, placeholders)
            samples, reports = _collect_ratio_samples(
                built['savings'], built['site_weight'], briefing['charts'])
            for name, floor, dropped, total in reports:
                if dropped:
                    print(f'  {name}: {dropped} of {total} site-capacity '
                          f'ratio(s) dropped for {state} {year} — '
                          f'self-consumption saving at or below ${floor:.0f}.')
            year_curve_sets.append(curves)
            year_sample_sets.append(samples)
            per_state_counts.setdefault(state, []).append(n_sites)
            per_state_year_curves.setdefault(state, {})[year] = curves
            per_state_year_counts.setdefault(state, {})[year] = n_sites

            if 'state_year' in figures:
                info = (f'{state} {year} (n = {n_sites})\n'
                        f'{weight_label}{baseline_note}')
                print(f'  writing {state}_{year}')
                make_briefing_figures(curves, samples,
                                      output_dir / f'{state}_{year}',
                                      weight_tag, info, briefing,
                                      draw_notes=args.notes)

        if not year_curve_sets:
            continue
        state_avg = (fa.average_curves_across_years(year_curve_sets)
                     if len(year_curve_sets) > 1 else year_curve_sets[0])
        per_state_curves[state] = state_avg
        # Each year contributes equally to the state's distribution, matching
        # the straight mean across years the curves take.
        per_state_samples[state] = _merge_samples(
            [_scale_samples(s, 1.0 / len(year_sample_sets))
             for s in year_sample_sets])

        if 'state_all_years' in figures:
            counts = per_state_counts[state]
            n_label = (f'n = {min(counts)}\u2013{max(counts)} per year'
                       if len(set(counts)) > 1 else f'n = {counts[0]}')
            # A run of consecutive years as a span, to keep the line short
            # enough for a panel of a pair.
            ints = sorted(int(y) for y in years)
            span = (f'{ints[0]}\u2013{ints[-1]}'
                    if len(ints) > 1 and ints == list(range(ints[0], ints[-1] + 1))
                    else ', '.join(years))
            info = (f'{state}, {span} ({n_label})\n'
                    f'{weight_label}{baseline_note}')
            print(f'  writing {state}_all_years')
            make_briefing_figures(state_avg, per_state_samples[state],
                                  output_dir / f'{state}_all_years',
                                  weight_tag, info, briefing,
                                  draw_notes=args.notes)

    grid_dir = output_dir / 'national_state_comparison'
    grid_note = weight_label if args.notes else None
    has_series_charts = any(c.get('kind', 'series') == 'series'
                            for c in briefing['charts'])

    if 'state_grid' in figures and len(per_state_curves) > 1 and has_series_charts:
        panels, notes = {}, {}
        for state, curve in per_state_curves.items():
            # The years are the same for every state and go in the caption.
            label = state
            panels[label] = curve
            counts = per_state_counts.get(state, [])
            if counts:
                notes[label] = (f'n = {min(counts)}\u2013{max(counts)}'
                                if len(set(counts)) > 1 else f'n = {counts[0]}')
        print(f"\nState comparison grid: {', '.join(panels)} "
              f"(n is the site count per year)")
        make_briefing_grids(panels, _grid_cells(list(panels), GRID_COLS),
                            briefing, 'state', weight_tag,
                            notes if args.notes else {}, grid_dir,
                            STATE_PANEL_HEIGHT_IN, title_note=grid_note)

    if 'state_year_grid' in figures and per_state_year_curves and has_series_charts:
        all_years = sorted({y for ys in per_state_year_curves.values()
                            for y in ys})
        panels, notes = {}, {}
        cells: list[list[str | None]] = []
        for state in scenarios_by_state:
            if state not in per_state_year_curves:
                continue
            row: list[str | None] = []
            for year in all_years:
                # A state with no data for a year leaves the cell blank
                # rather than shifting its row out of alignment with the
                # column headings the other rows establish.
                if year not in per_state_year_curves[state]:
                    row.append(None)
                    continue
                label = f'{state} {year}'
                panels[label] = per_state_year_curves[state][year]
                notes[label] = f'n = {per_state_year_counts[state][year]}'
                row.append(label)
            cells.append(row)
        print(f'\nState x year grid: {len(cells)} states x '
              f'{len(all_years)} years')
        make_briefing_grids(panels, cells, briefing, 'state and year',
                            weight_tag, notes if args.notes else {}, grid_dir,
                            STATE_YEAR_PANEL_HEIGHT_IN, title_note=grid_note)

    if 'national' in figures:
        if len(per_state_curves) < 2:
            print('\nNOTE: national view needs more than one state; only '
                  f'{sorted(per_state_curves)} loaded, so it was skipped.')
        else:
            state_list = ', '.join(sorted(per_state_curves))
            all_years = sorted({y for ys in scenarios_by_state.values() for y in ys})
            year_label = (f'{all_years[0]}\u2013{all_years[-1]}'
                          if len(all_years) > 1 else all_years[0])

        if len(per_state_curves) >= 2 and weighted:
            print('  writing national')
            national = fa.combine_curves_state_weighted(
                per_state_curves, weights_cfg['state_mix'])
            # The same state mix the curves are combined with, applied to the
            # site observations, so every national figure describes one
            # population rather than two.
            mix = weights_cfg['state_mix']
            national_samples = _merge_samples(
                [_scale_samples(per_state_samples[s], mix.get(s, 0.0))
                 for s in per_state_curves])
            info = (f'National ({state_list}), {year_label}\n'
                    f'Weighted by PV size & state mix{baseline_note}')
            make_briefing_figures(national, national_samples,
                                  output_dir / 'national',
                                  weight_tag, info, briefing,
                                  draw_notes=args.notes)

        elif len(per_state_curves) >= 2:
            # Unweighted: the savings charts have no national view, because
            # combining per-state means needs state_mix and there is no
            # defensible substitute — a straight mean over states would
            # weight SA and NSW alike, which is a weighting decision made by
            # not making one.
            #
            # The ratio charts do not have that problem. They pool site
            # ratios, not state means, so the pool is well defined without
            # state_mix; what it is not is population-representative, since
            # each state then enters in proportion to how many sites it
            # happens to contribute. That is what an unweighted view is
            # everywhere else in this script, so it is produced and labelled
            # as such rather than withheld.
            note = ('\nNOTE: --unweighted given, so the national savings '
                    'charts were not produced — pooling per-state means needs '
                    'state_mix, which is a weighting concept.')
            if any(c.get('kind') == 'ratio' for c in briefing['charts']):
                print('  writing national')
                national_samples = _merge_samples(
                    [per_state_samples[s] for s in per_state_curves])
                info = (f'National ({state_list}), {year_label}\n'
                        f'Unweighted, states by site count,'
                        f'{baseline_note}')
                make_briefing_figures(None, national_samples,
                                      output_dir / 'national', weight_tag,
                                      info, briefing, kinds={'ratio'},
                                      draw_notes=args.notes)
                note += (' The national ratio charts were, by pooling site '
                         'ratios directly.')
            else:
                note += ' This briefing has no ratio charts, so it has no '\
                        'national view at all under --unweighted.'
            print(note)

    print('\nDone.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
