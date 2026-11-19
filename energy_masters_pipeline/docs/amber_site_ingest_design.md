# Amber site ingestion design

## Purpose and scope

This document designs the ingestion step that turns an Amber telemetry
site's raw data into the two shapes the existing model needs:

1. A `(pv, gross_load)` pair to feed `self_consumption` and
   `perfect_foresight_lp`, exactly the same shape `_dev_common.load_site_trace`
   already produces for EDP sites.
2. A `SiteTraces` object built from the site's *actual* measured behaviour
   (Amber's own coordination, not a model), so `compute_bill` can be run on
   it and produce a real bill to compare against the self-consumption and
   perfect-foresight bills.

The design applies generally to any site in the clean shortlist (the
family-summing and gross-load formula were validated across all 23 battery
sites, not just one), with the multi-instance summing logic (needed for
sites like EM-414 with two parallel PV/battery systems) written generally
enough to be exercised later even though the first test site doesn't need it.

**EM-427 is the first site this will be run and validated against**, chosen
for being architecturally simple (one PV instance, one battery instance)
with the cleanest reconstruction diagnostics of the shortlist. Once it works
end to end, running the same code against the rest of the clean-14 sites
should need no design change, only the per-site inputs (site_id, state,
tariff) changing.

## Two outputs, two different problems

The two things this ingest produces are not the same difficulty of problem,
and treating them as one was the mistake in an earlier version of this
design conversation.

**The LP/self-consumption feed** just needs `pv_site_net_kwh` and
`gross_load_kwh`. `self_consumption()` and `perfect_foresight_lp()` both
construct the full flow decomposition internally from just those two series
plus a `BatterySpec`; nothing else is needed. This is a data-shaping
problem, not an allocation problem, because the gross-load reconstruction
formula was already validated (see below).

**The actual-measured `SiteTraces`** is different. `compute_bill` was
confirmed (by reading `billing.py`'s `_build_settlement_base`) to only ever
read `traces.import_kwh` and `traces.export_kwh` from the `SiteTraces` it's
given. Nothing else in the settlement path touches the flow decomposition
(`sol_to_home_kwh`, `sol_to_bat_kwh`, ..., `curtail_kwh`) or
`battery_soc_kwh`. So the actual-measured `SiteTraces` for any site only
needs two real series; the other eight fields become all-zero placeholders,
exactly matching the pattern `site_result.py` already documents for
solar-only cohorts ("populated as all-zero series rather than left as
None... keeps downstream code uniform").

## Mathematical formulation

### Gross load and PV, for the LP/self-consumption feed

Already validated against EM-427 and the other 22 battery sites (see the
correlation-based reconstruction check from earlier in this thread):

    gross_load_kwh = ac_load_net_kwh + pv_dc_kwh - battery_storage_kwh

Sign convention, confirmed both algebraically and against a real sample
day: `ac_load_net` positive = net import, negative = net export;
`battery_storage` positive = charging, negative = discharging.

    pv_site_net_kwh = pv_dc_kwh

For a site with multiple instances of the same appliance family (EM-414's
two battery/PV systems, or any site with a genuine second PV string), sum
across instances within the family before applying the formula:

    pv_dc_kwh = sum(pv_dc__i for i in instances at this site)
    battery_storage_kwh = sum(battery_storage__i for i in instances at this site)

EM-427 has exactly one instance of each (`pv_dc__1`, `battery_storage__1`),
so this collapses to the plain single-channel case, but the ingestion code
should still go through the family-sum step rather than hardcoding a single
column name, so the same function works unmodified on EM-414 later.

### Import and export, for the actual-measured SiteTraces

The raw telemetry already carries the exact split billing needs, without
needing to re-derive it from the signed `ac_load_net.energy` column:

    import_kwh = ac_load_net.energy_pos_wh / 1000
    export_kwh = ac_load_net.energy_neg_wh / 1000

This was verified earlier: `energy == energy_pos - energy_neg` holds for
every row in the file with zero violations, and both components are always
non-negative. Using the source data's own decomposition directly, rather
than recomputing `max(ac_load_net, 0)` / `max(-ac_load_net, 0)`, means one
less place for the sign convention to be silently assumed twice.

### Unit conversion

Raw `energy` values are Wh per 5-minute interval (confirmed: night-time
`ac_load_net` and `pv_dc` values of -1/0/1 match parasitic inverter draw at
Wh scale, and the modal interval check confirmed 5-minute raw cadence).
Convert to kWh by dividing by 1000, matching `_dev_common.load_site_trace`'s
existing `_wh -> _kwh` convention exactly.

## Timezone handling: a real gotcha, resolved

The raw data carries two timestamp columns: `utc_reading_datetime` (true
UTC) and `site_reading_datetime` (real site-local time, DST-aware -- this
is the one that showed a genuine seasonal offset shift for EM-751 in the
original HTML site report).

**Do not index on `site_reading_datetime`.** The rest of the model
(`self_consumption`, `perfect_foresight_lp`, `compute_bill`, `prices.py`)
is built around a fixed-offset "AEST" index (`Etc/GMT-10`, no DST
observed) with real local time resolved separately, on demand, via
`core/localtime.py` and an explicit `state` parameter. This is exactly how
`L_Convert_To_Parquet.py` builds `datetime_aest` for the EDP pipeline: UTC
converted to the fixed `Etc/GMT-10` offset, not the site's actual local
clock. Indexing on `site_reading_datetime` instead would mean the model's
own DST-aware local-time resolution and the index's timestamps disagree
about what a given wall-clock moment across a DST transition actually is,
silently double-handling (or fighting) daylight saving.

So: build the index as `utc_reading_datetime + 10 hours`, localised to
`Etc/GMT-10`, discarding `site_reading_datetime` entirely once the index is
built. `state` (needed for `build_price_vectors` and `compute_bill`'s local
time resolution) is a separate, explicit parameter, not derived from either
timestamp column.

**State: SA for every Energy Masters site**, confirmed. This resolves the
open question from the previous version of this document -- `state='SA'`
is a fixed input for the whole cohort, not something the ingestion code
needs to look up or infer per site.

## The bridged-noise rows

Per your direction: assume REAL_TIME for the entire bridged valid period,
including the small number of rows inside it that weren't actually tagged
`REAL_TIME` (0.69% of EM-427's window, from the site-summary run). No
special-casing needed -- these rows get treated identically to genuinely
tagged rows throughout the ingest.

## Missing data

The EDP pipeline's approach (from `E_Flag_All_Sites.py` and
`G`/`H`/`I`) is a six-code flag system (clean, missing, flatline, DST gap,
partial phase, invalid, implausibly large) followed by two-stage infill:
Method 1 (linear interpolation, for short gaps with clean boundaries) and
Method 2 (synthetic day, built from a cross-site normalised PV profile
library for PV, and same-day-of-week matching within the site's own data
for load). That's the right eventual target if this data turns out to be
worth the investment, but implementing the full flagging taxonomy now,
before we know whether EM-427-style data is even usable at scale, is
premature.

For this first sense-check: **plain linear interpolation** on the three raw
channels (`ac_load_net`, `pv_dc`, `battery_storage`) wherever a value is
missing, applied before the gross-load formula, with no gap-length cap and
no flagging. Given EM-427's `pct_bridged_noise_in_period` was only 0.69%
and its general missingness elsewhere in the file was in the low single
digits, the volume needing interpolation should be small regardless.

**One addition confirmed necessary, not just anticipated:** EM-414 showed
a real case of a present-but-implausible value -- `pv_dc__1` jumped to
15,297 Wh in a single 5-minute interval (implying ~184kW, physically
impossible for this cohort) between two ordinary near-zero readings.
Interpolation alone doesn't catch this, since the value isn't missing.
Any raw Wh energy-flow value (never price/rate columns, where a large
value can be a genuine, important price spike) exceeding
`MAX_PLAUSIBLE_WH_PER_INTERVAL` (5000 Wh, i.e. 60kW -- generous, well
above anything this cohort should produce) is now masked to NaN before
interpolation, so it gets smoothed over the same way a genuine gap would.
Confirmed as a one-off for EM-414 (1 interval out of 68,287), not a
systemic issue.

**Documented limitation and upgrade path:** this simplified version does
not distinguish a genuinely clean value from an interpolated one (no
`pv_infill`/`load_infill`-style tracking column), does not flag
implausible or flatlined values before filling them (the plausibility
clip above catches only the single most extreme case, a physically
impossible magnitude -- it wouldn't catch a smaller but still-wrong value,
or a flatlined stretch), and has no long-gap synthetic-day fallback. If
EM-427-style data proves out and this becomes a real part of the analysis
(not just a sense-check), the upgrade path is: add the same flag-code
column convention as the EDP pipeline, split interpolation into a
short-gap linear pass and a long-gap cross-site/same-day-of-week pass
mirroring Scripts G and I, and track infill provenance the same way
(`pv_infill`, `load_infill` columns) so it's auditable which values are
original versus filled.

## Price reconciliation

Two independently-built price series should be compared for the same
interval:

1. **Our derived price**: `build_price_vectors(index, prices_mwh,
   network_tariff, retail_tariff, state='SA')` with `retail_tariff.type =
   'wholesale_exposed'` (matching Amber's actual structure: spot price
   passthrough plus flat retailer adders, no TOU periods), giving
   `(p_imp, p_exp)` in $/kWh.
2. **Amber's own published rate**: the raw `import_rate` / `export_rate`
   columns, already in the file per interval.

The comparison is a straightforward residual: `p_imp - import_rate` and
`p_exp - export_rate` over the site's valid period, reported as summary
statistics (mean, distribution, fraction of intervals beyond some
tolerance) rather than expecting an exact match. Some divergence is
expected and fine -- our derived price depends on assumed network tariff
and loss factor values that may not exactly match the site's actual
network tariff; the point of the comparison is to confirm the two are in
the right ballpark and flag it clearly if they are not, not to force
agreement.

This needs a `network_tariff` and a `retail_tariff` (wholesale_exposed,
built to approximate Amber's actual fee structure) as inputs, plus the SA
NEM RRP series for the site's valid period -- same `paths.nem_prices_dir`
source `viz_perfect_foresight_lp.py` already uses.

### Settlement resolution: some sites publish a 30-minute price, not 5-minute

Confirmed for EM-427 by direct investigation, not assumed: the raw
per-interval residual showed extreme outliers (up to +/-$23/kWh) tightly
concentrated around genuine NEM price-spike events, while the bulk of the
distribution looked fine. The cause: if a site's smart meter isn't
certified for 5-minute settlement, Amber bills against AEMO's 30-minute
time-weighted average price (TWAP) instead of the 5-minute price, even
though the underlying telemetry (`ac_load_net`, `pv_dc`, `battery_storage`)
is still logged at 5-minute resolution. A transition-timing check across
EM-427's whole valid period confirmed this cleanly: `import_rate`/
`export_rate` change value at almost exactly the same tick every time
(>95% of transitions land on a `:00`/`:30` mark, and >94% of blocks are
exactly 6 ticks wide) -- a real, present-in-the-data property, not
occasional noise.

Getting the reconstruction right needed two separate corrections, not one:

- **Content**: which six 5-minute ticks get averaged together. Our own
  `TariffPeriod` convention already assigns a 5-minute interval to a
  period by its START instant (label minus interval length). Applying
  that same logic one level up to the 30-minute AEMO trading interval
  means grouping timestamps `t+5` through `t+30` into the block labelled
  `t` -- a content phase-offset of 5 minutes relative to the naive
  `floor(30min)` grouping, not zero.
- **Broadcast**: the resulting block average needs to be shifted 5
  minutes earlier than its own constituent members to land where Amber
  actually displays it. This is a single-tick correction, not a
  whole-30-minute-block one -- an easy mistake to make, since a whole-block
  shift is the more obvious thing to try first and very nearly explains
  the pattern without actually fixing it.

Confirmed against the full 201-day valid period (not just the spike
examples used to find it): std residual dropped from 0.273 to 0.032
$/kWh for import (max from 22.8 to 0.24) and from 0.279 to 0.069 $/kWh for
export (max from 23.0 to 4.0, one lingering outlier not yet chased down
but small next to what this fixed).

**This correction applies only to the reconciliation/billing comparison,
never to the price the perfect-foresight LP optimises against.** The LP
should keep using the genuine 5-minute wholesale price regardless of what
resolution the site is billed at -- Amber's own battery coordination very
likely still operates on 5-minute granularity even at a 30-minute-settled
site (this is itself checkable later, once we're comparing actual battery
behaviour: exports during a narrow 5-minute price spike that don't spread
across the whole 30-minute settlement window would confirm 5-minute
dispatch decisions under 30-minute billing).

**5-minute settlement certification was checked across the whole clean-14
shortlist, not just EM-427, and came back unanimous**: every site shows
the same pattern (98.0-99.7% of blocks exactly 6 ticks wide, transitioning
at the same tick position) -- EM-427 is actually one of the weaker
confirmations in the cohort (95.4%/94.7%), not an outlier. This makes
practical sense for a cohort likely onboarded through the same program
with broadly similar meter hardware: 5-minute settlement certification is
a meter/config property, plausibly uniform across participants recruited
the same way. Practically, this means the `(phase_offset=5, shift=-5min)`
correction validated for EM-427 can be treated as the shared default for
price reconciliation across the whole clean-14, not something needing
fresh per-site tuning each time.

That said, "uniform settlement resolution" confirms the *regime*, not
necessarily the exact phase/shift numbers down to the same precision for
every site -- worth a lightweight spot-check (residual stats with the
correction applied, on one or two more sites) before fully relying on it
everywhere, rather than assuming zero site-to-site variation without
checking. A quick block-width check like the one already run is still
worth doing for any *new* site added to the cohort later, in case it
turns out differently (a different meter vintage, a different retailer
migration history, etc.).

## Interface contract

Two functions, mirroring the existing `_dev_common.load_site_trace` shape:

```
load_em_site_trace(site_id: str, paths) -> pd.DataFrame
    Index: datetime_aest (Etc/GMT-10, 5-min).
    Columns: pv_site_net_kwh, gross_load_kwh.
    Restricted to the site's valid REAL_TIME period(s) from the site
    summary table. Feeds self_consumption / perfect_foresight_lp exactly
    like the EDP trace does.

build_em_actual_traces(site_id: str, paths) -> SiteTraces
    site_id: the em_id.
    import_kwh, export_kwh: real, from ac_load_net's energy_pos/energy_neg.
    battery_soc_kwh and the eight flow-decomposition fields: all-zero
    pd.Series, same index. Not derived, not estimated -- genuinely zero,
    since nothing downstream reads them for billing.

reconcile_prices(site_id: str, valid_periods, ...) -> pd.DataFrame
    Our derived (p_imp, p_exp) vs Amber's published (import_rate,
    export_rate), plus the residual, over the site's valid period(s).
    Diagnostic/reporting only -- not consumed by the LP or billing.

check_settlement_resolution(site_id: str, ...) -> dict
    Dominant block width and transition tick-position for a site's
    import_rate/export_rate (see "Settlement resolution" above). Should
    run before reconcile_prices is trusted for a new site, to determine
    whether the 30-minute block-average correction applies or whether the
    raw 5-minute price should be used directly.
```

## Edge cases

- **Multiple instances of one appliance family** (EM-414 and similar):
  summed before the gross-load formula is applied, as above. Turned out
  not to be meaningfully exercised by EM-414 after all -- its second
  system (`pv_dc__4`/`battery_storage__4`) has rows somewhere in the
  site's full raw history but zero rows within its actual valid REAL_TIME
  period, so only the first instance ever contributes there. The summing
  logic is confirmed correct in the sense that it didn't silently break
  anything (verified via the EM-414 outlier investigation), but a site
  where both instances genuinely overlap within the valid period hasn't
  been tested yet.
- **No battery channel**: not applicable to any site in the selected
  cohort, since no-battery sites were already excluded at the
  site-selection stage.
- **Bridged-noise rows**: treated as REAL_TIME throughout, no special
  casing, per your direction above.
- **`TARIFF_MISCONFIGURED` rows**: not relevant to EM-427 specifically
  (that only affected EM-302 and EM-457), but the ingest should not
  silently accept a `TARIFF_MISCONFIGURED` row if one ever appears in a
  site's valid period; worth a defensive check that raises rather than
  proceeds, since the whole point of the valid-period definition was
  REAL_TIME-only data.

## Consistency contract

- `import_kwh - export_kwh` must equal `ac_load_net_kwh` exactly, by
  construction (both derived from the same `energy_pos`/`energy_neg` split,
  and `ac_load_net`'s own sign convention is already import-minus-export).
  A cheap assertion, not a design risk, but worth stating as an invariant
  the ingestion function should check before returning. (An earlier version
  of this document had the sign backwards -- caught when the corresponding
  check in em_ingest.py's run_checks flagged a large residual that turned
  out to be the check itself, not the data, being wrong.)
- The gross-load reconstruction's correlation-based sign check (near-zero
  correlation with `battery_storage`, used earlier to validate the sign
  convention across all 23 sites) should be re-run as a regression check
  whenever this ingestion code changes, not just once during design.
