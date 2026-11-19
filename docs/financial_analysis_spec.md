# Financial analysis — Specification

`analysis/scripts/financial_analysis.py` reports household outcomes and
wholesale market impact for battery-equipped solar households under
different retail plans, across states and years, with population weighting
by PV system size.

## 1. Scope

The script produces three sections of output, and does so at every level of
aggregation the available data supports, automatically:

- **Per (state, year) scenario** — always produced, one figure set for
  every scenario found.
- **Per state, averaged across years** — produced automatically whenever a
  state has more than one year of data. This is a straight (unweighted)
  mean across years, not itself subject to size-band or state-mix
  weighting, since year-averaging is a separate aggregation step from the
  population weighting.
- **National (state-weighted)** — produced automatically whenever more than
  one state is present, combining each state's result (its multi-year
  average, or its single year if that's all it has) using the `state_mix`
  weights. Only produced when weighting is enabled overall, since
  `state_mix` is itself a weighting concept and has no meaning under
  `--unweighted`.

No flag is needed to opt into the state or national aggregation: the
script produces whatever tiers the discovered data supports.

Every tier is produced once for each run basis (section 1a).

Each figure set covers two sections:

1. **Household outcomes** — bill savings by battery size, retail plan, and
   dispatch mode (default as-joined behaviour, optimised behaviour, and a
   full-range view showing both, with self-consumption without VPP
   participation, section 1a).
2. **Wholesale market impact** — net wholesale cost savings by battery
   size, mode and retail plan, against a shared wholesale baseline.

Retailer outcomes are not produced here. `retailer_briefing.py`,
`cap_account_retailer.py` and `cap_account_aggregate.py` import this
script's run registry and weighting rather than having their own.

## 1a. Run bases

Each plan has a self-consumption run and an optimised run. The run basis
sets which runs those are:

| Basis | Optimisation | VPP dispatch |
|---|---|---|
| `perfect_foresight` | perfect foresight | maximum |
| `conservative` | imperfect foresight on the pre-dispatch forecast available before each interval starts | none |
| `achievable` | imperfect foresight using each interval's dispatch price and refreshed pre-dispatch | maximum realistic |

Rule-based self consumption has no foresight, so a plan's self-consumption
run is the same on every basis, except for plans whose self-consumption
mode takes part in VPP events (AGL, Origin), which use their with-VPP run
on perfect foresight, their rolling self-consumption run on achievable and
their no-VPP run on conservative.

On the conservative basis a VPP plan's no-dispatch case is its no-VPP run
plus the enrolment payment an enrolled but undispatched battery still
receives (AGL $80 a year, EnergyAustralia $180 a year), added to the
household saving at every battery size above zero.

Amber's conservative optimised run is `full_run_rolling_amber_predispatchonly`.

Each plan also names its no-VPP self-consumption run (`sc_no_vpp_dir`): plain
self-consumption with no VPP events, critical rates or enrolment payment.

| Offer | `sc_no_vpp_dir` |
|---|---|
| Amber | `full_run_self_consumption_amber` |
| GloBird ZeroHero | `full_run_self_consumption_globird_novpp` |
| AGL Battery Rewards | `full_run_self_consumption_agl_novpp` |
| EnergyAustralia Battery Ease | `full_run_self_consumption_ea` |
| Origin Battery Lite | `full_run_self_consumption_origin_novpp` |

Where it differs from the basis's self-consumption run, or that run carries a
conservative enrolment payment, it is loaded as a further mode
(`{sc_mode}_NO_VPP`) with no basis adjustment.  It is the self-consumption
curve of the full-range view on every basis, and is written to the curves CSV
with series `self-consumption, no VPP`.  The default view keeps the basis's
self-consumption run, the plan's as-joined behaviour.

A run that has not completed can stand in as a placeholder built from two
that have: its curve is (1 - c) x base + c x upper, which scales the upper
run's value capture ratio over base by c. Placeholders are declared in
`PLACEHOLDERS` by basis and mode, and the table is empty while every run is
complete.

The basis is named in every figure title. With both imperfect foresight
bases drawn, the national optimised household and wholesale views are also
drawn as a pair of panels, achievable on the left and conservative on the
right, on one y range.

## 2. Data discovery: states and years

The script discovers which (state, year) combinations actually have data,
rather than requiring the caller to enumerate them. For a given run
directory, any subfolder of its `scenarios/` folder whose name matches
`{STATE}_{YEAR}_{BATTERY}kWh` (or `{STATE}_{YEAR}` for the no-battery case)
is treated as valid, and the set of (state, year) pairs found this way
across run directories is the available set. Anything incomplete for a
given pair (e.g. missing the solar-only baseline) is skipped with a
warning rather than failing.

An optional `--states` / `--years` filter can narrow this down to a subset;
the default is everything found.

## 3. Directory layout

The pattern is `{STATE}_{YEAR}_{BATTERY}kWh` under each run directory's
`scenarios/` folder, with no `_{BATTERY}` suffix for the no-battery
scenario. This is uniform across all states and years.

## 4. Data root

A single `--data-root` covers everything: run outputs live at
`{data-root}/data/output/runs` and site metadata at its usual relative
path underneath it. When the run data is on an external disk, `--data-root`
points there and the metadata files sit in the same relative position. There
is no separate runs-root override.

## 5. Retail plan availability by state

Every plan is modelled in every state. EnergyAustralia Battery Ease was
offered in NSW only. Its QLD, SA and VIC runs model estimated versions of
the offer and are left out of the analysis through the plan's
`excluded_states`, so its national results are its NSW results (the
state_mix weights are renormalised over the states a plan has). National figures carry the data note line
"EnergyAustralia results for NSW only" (`restricted_offer_notes`).

## 6. System size: source fields and estimation pipeline

System size (`size_kw`) is resolved per site in three tiers, in order of
trust, based on analysis of the 2022 site metadata for NSW, QLD, VIC and
SA:

**Tier 1 — direct.** `subarray_dc_rating_kw` is trusted only when there is
an `inverter_ac_rating_kw` to check it against, and it is greater than or
equal to that inverter rating. A subarray reading below the inverter's AC
rating usually means metadata only captured one of several DC strings, not
the site's true array size (confirmed empirically: 65.6% of sites with both
fields present show subarray < inverter, which is backwards for a single
fully-captured array under normal DC:AC oversizing design practice), so
`subarray_dc_rating_kw` is never trusted on its own, regardless of whether
it's present.

**Tier 2 — ratio-scaled.** For sites with an inverter rating but no
trusted subarray value (tier 1 doesn't apply), `size_kw` is estimated as
`inverter_ac_rating_kw` times a single **pooled, national** median DC:AC
oversizing ratio taken from the tier-1 trusted sites (measured at 1.20 in
the 2022 sample). This ratio reflects a hardware/design convention that
doesn't vary meaningfully by climate, so pooling nationally is appropriate
and gives a larger, more stable sample than any one state alone.
`inverter_ac_rating_kw` is used as-is, without scaling by `inverter_count`
for multi-inverter sites: that scaling gives an implied specific yield of
about half the plausible value.

**Tier 3 — regression.** For sites with no inverter rating at all (tiers 1
and 2 don't apply), `size_kw` is estimated by regressing tier-1 and tier-2
sites' `size_kw` against `annual_generation_kwh`, fit **per state** (unlike
the DC:AC ratio, specific yield is climate-dependent), falling back to a
pooled all-state fit if a state doesn't have enough tier-1/2 sites of its
own to fit a reliable line.

Sites with no `size_kw` resolved by any tier (missing both inverter and
generation data) are excluded from the weighted calculation but stay in
the unweighted calculation, which never depends on system size.

**Out-of-band sites.** The band structure only covers a defined range
(4.5 to 25 kW — deliberately excluding smaller systems, since the
analysis is meant to reflect adding a battery to an already reasonably
sized array, not an undersized one that a household would likely upgrade
before adding storage). Any site whose resolved `size_kw` falls below the
bottom of the bottom band or above the top of the top band is excluded
from the weighted calculation entirely, with a warning naming the site and
its rating (real or modelled, labelled as such) so it's easy to check
whether that's expected or a sign the estimate went wrong. These sites
remain in the unweighted calculation.

## 7. Weighting YAML

```yaml
size_bands: ["4.5-6.5", "6.5-9.5", "9.5-14", "14-25"]
by_state:
  NSW: [0.259, 0.406, 0.289, 0.046]
  VIC: [0.364, 0.435, 0.163, 0.037]
  QLD: [0.367, 0.356, 0.236, 0.041]
  SA:  [0.339, 0.405, 0.206, 0.049]
state_mix:
  NSW: 0.338
  VIC: 0.230
  QLD: 0.317
  SA:  0.115
```

`by_state` values are positional against `size_bands`, not repeated as
keyed pairs — this means adding, removing, or resizing bands only requires
editing `size_bands` and updating each state's array to match its length
and order, rather than touching every band label in every state block.
Code that reads this file must validate that every `by_state` array has the
same length as `size_bands` and fail clearly (not silently misalign) if it
doesn't.

Bands only cover 4.5kW and up, per the deliberate exclusion described in
section 6. Sites below 4.5kW are dropped from the weighted view, with the
dropped count reported, and kept in the unweighted view.

File location: `analysis/config/pv_size_bands.yaml`.

## 8. Weighting math

For a given state, rather than computing a per-band mean and then combining
band means, each site gets a per-site weight directly:

```
site_weight = by_state[band_of_site] / count_of_sites_in_band_in_state
```

The state's representative outcome (for a given battery size, mode, plan)
is then the weighted mean over all its sites using these weights. This
achieves the same effect as a two-stage band-mean approach (each band's
total contribution to the result equals its `by_state` share regardless of
how many sites happen to fall in it) but is a single weighted mean rather
than a mean-of-means, which is simpler to implement and to extend to a
further weighting dimension.

**Empty bands: borrow from the nearest neighbour.** If a band carries
non-zero weight but has zero sites in a state's sample (observed in
practice: QLD's 14-25kW band in the 2022 data), that band's weight is
applied instead to the site pool of the nearest non-empty neighbouring
band, with a warning naming the affected band, its weight, and which
neighbour(s) it borrowed from. If the empty band sits between two other
bands and both are equally near (one step away on each side), both are
pooled together as the donor set. This keeps the band's population share
represented in the result — using the closest available proxy for what a
system of that size would actually do — rather than silently dropping its
weight (which would understate that band's contribution to the state
total) or refusing to compute a result outright. It is still a hard error
if even the neighbours have no sites either, since at that point there is
no reasonable proxy left to borrow from.

For the national (state-weighted) tier, each state's representative curve
(its multi-year average, or its single year if only one exists) is combined
using `state_mix`, restricted to states where the relevant plan is marked
available (section 5), and produced automatically whenever more than one
state is present (section 1) — not behind a separate flag.

## 9. Weighted vs unweighted: single global flag

One flag controls the entire script's behaviour, applied consistently
across all sections and all charts. Weighting is now the default — the flag
is used to turn it off, not on:

```
--unweighted    use straight per-site means instead of size-band and
                state-mix weighting (default: off, i.e. weighted is on)
```

With `--unweighted`, everything in sections 6 to 8 is bypassed entirely and
every calculation is a plain mean over whatever sites are present, matching
the simplest possible baseline. This remains useful both as a comparison
view and as a sanity check: the weighted and unweighted results should
differ only in the ways weighting is supposed to change them.

## 10. Output naming

Each run basis writes to its own subdirectory of `--output-dir`
(`perfect_foresight/`, `imperfect_foresight_conservative/`,
`imperfect_foresight_achievable/`), and within it each tier to its own
subdirectory:

- `{state}_{year}/` — per-scenario results
- `{state}_all_years/` — per-state, mean across years (only when >1 year)
- `national/` — state-weighted pooled result (only when >1 state, weighted
  mode only)
- `national_state_comparison/`, `{state}_year_comparison/` — the
  small-multiples grids

with every curve set drawn written to `curves_{weighted|unweighted}.csv`
(scope, metric, offer, series, mode, kWh, value), and the site savings
behind the per-scenario curves, with the basis adjustments applied, to
`site_savings_{weighted|unweighted}.csv` (state, year, site_id, weight,
offer, series, mode, kWh, hh_saving, ws_saving; weight is the PV size weight,
1 when unweighted, and sites the weighting excludes are left out). The conservative and
achievable pair goes to `imperfect_foresight_comparison/`.

`analysis/scripts/savings_bar_figure.py` reads the conservative and
achievable CSVs and draws, for one capacity (default 20 kWh) and scope
(default national), a self-consumption and an optimised bar per offer in the
offer's colour. Self-consumption is the no-VPP run without the enrolment
payment, outlined in the offer colour with a light hatch in a tint of it.
The optimised bar is solid to the lower of the conservative and achievable
values, with a segment in a tint of the offer colour continuing it to the
higher (the imperfect foresight & VPP dispatch range): no VPP events or
critical rates to maximum dispatch, and for Amber conservative to
achievable. Each bar is labelled with its top value. Offers
are ordered by the household achievable optimised saving, the conservative
value standing in where the achievable run is missing (its slot labelled
"range pending"), and both figures use that order. Two figures, in
`capacity_comparison/`:

- `household_saving_by_offer_own_plan_{kWh}kwh_{scope}_{weighting}.png`: the
  household saving against the offer's own Solar only bill, which is the
  curve value less the value at 0 kWh (the offer's Solar only bill against
  Amber's)
- `wholesale_saving_by_offer_{kWh}kwh_{scope}_{weighting}.png`: the net
  wholesale saving

It also prints, from the site savings, the share of site-years in which each
offer gives the lowest bill with an optimised battery of that capacity, for
each imperfect foresight basis. Every saving is against the same Solar only
bill on Amber, so the highest saving is the lowest bill. A site-year is
compared over the offers modelled for it, and weighted as in the national
curves: its PV size weight normalised within its scenario, divided by its
state's number of years and multiplied by the state's renormalised
state-mix share.

Within each directory, descriptive stems rather than legacy figure numbers:

- `household_savings_default_mode`
- `household_savings_optimised_mode`
- `household_savings_full_range`
- `wholesale_savings`

Each suffixed with `_weighted` or `_unweighted` depending on the mode the
script ran in, e.g. `household_savings_default_mode_weighted.png`.

## 11. Interface (script CLI)

```
python3 analysis/scripts/financial_analysis.py
    --data-root PATH          project root; run outputs and site metadata
                              are both found at their usual relative paths
                              underneath this (default: cwd)
    --output-dir PATH         (default: analysis/output/financial)
    --states STATE [STATE...] restrict to specific states
                              (default: all states found under --data-root)
    --years YEAR [YEAR...]    restrict to specific years
                              (default: all years found under --data-root)
    --unweighted               use straight per-site means instead of
                              size-band and state-mix weighting
                              (default: off, weighting is on by default)
    --weights-file PATH       path to the weighting YAML
                              (default: analysis/config/pv_size_bands.yaml)
    --basis BASIS [BASIS...]  run bases to draw: perfect_foresight,
                              conservative, achievable or all (default: all)
    --no-notes                leave the data notes (n, period, weighting,
                              baseline) off the figures, for the thesis
    --baseline {amber,own}    savings against Amber's solar-only run (amber)
                              or each plan's own solar-only run (own)
                              (default: amber, or own with --only-plan)
    --only-plan PLAN [PLAN...] presentation override: restrict every chart
                              to the named plans; writes to the normal
                              output paths, so overwrites the full charts
```

Figures follow `docs/thesis_figure_spec.md`.

State and national aggregation happen automatically based on what data is
found; there is no separate flag to request them (see section 1).
