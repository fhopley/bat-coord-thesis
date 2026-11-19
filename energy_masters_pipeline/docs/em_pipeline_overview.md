# Energy Masters analysis pipeline

A high-level map of the Energy Masters (EM) analysis: what has to be run
by hand, what runs automatically, and where the pipeline is known to be
weak. This is a navigation document, not a specification. Detail on the
ingestion stage lives in `amber_site_ingest_design.md`; detail on the
coordination models lives in the model spec documents.

Scripts run from the project root as modules:

    python3 -m energy_masters_pipeline.scripts.<script_name> [args]

## Layout

```
energy_masters_pipeline/
├── battery_capacity_review.csv     hand-curated, not reproducible
├── scripts/                        all thirteen scripts
├── charts/                         chart_vcr_results.py output
│   └── diagnostics/                per-site review PNGs
└── docs/                           the EM spec documents

data/processed/energy_masters/      everything the scripts generate
data/energy_masters_amber_data/     raw telemetry, unchanged
```

Paths are relative to the project root, as before. The raw telemetry keeps
its existing location: renaming it to match the `data/raw/` convention was
considered and deliberately deferred, since it is a rename with no
functional benefit.

## The five scripts that have to be run

Everything else in the pipeline is imported and called automatically.
These five are the only ones a person has to invoke, in this order.

### 1. `build_site_summary_table.py`

    python3 -m energy_masters_pipeline.scripts.build_site_summary_table

Scans all 33 sites in the telemetry parquet and writes
`data/processed/energy_masters/site_summary_results.csv`, one row per
site, carrying each site's
`valid_periods`: the contiguous REAL_TIME-tariff stretches of usable
length, with short gaps bridged over as pipeline noise rather than
genuine tariff changes.

**Run it once. Re-run only if the telemetry file is refreshed or one of
its four tunable gates changes** (`MIN_SEGMENT_DAYS`,
`MIN_GAP_TO_SPLIT_DAYS`, `MAX_PCT_NEGATIVE`,
`MAX_ABS_CORR_WITH_BATTERY`). Nothing downstream re-derives valid
periods, so this table is effectively the definition of what "the data"
is. Note that its own `include` column is a data-quality screen, not the
cohort: the cohort is decided at step 2.

### 2. `viz_energy_masters.py`, per site

    python3 -m energy_masters_pipeline.scripts.viz_energy_masters EM-427

The review instrument. It runs the whole per-site chain (see "What runs
automatically" below), prints the verification block, and opens the
interactive day browser. What you are looking for at this stage is
whether the estimated capacity, power, export limit and date range are
right, and whether the site shows evidence of automated coordination at
all.

Decisions from that review go into `battery_capacity_review.csv`, via
`capacity_review.upsert_review` or by editing the file. That CSV carries
`decision` (`accept`, `override`, `exclude`), a free-text `reason`, and
five override fields: `capacity_kwh_override`, `power_kw_override`,
`export_kw_override`, `start_date_override`, `end_date_override`.

**This is the only genuinely manual stage and the only one that cannot be
reproduced by re-running code.** A site absent from the CSV is treated as
unreviewed, not accepted, and `load_and_analyze` raises rather than
silently including it.

Other modes: `--compare-day YYYY-MM-DD` writes a static actual-versus-LP
comparison PNG (this is what the manual highest-headroom-day review
used), `--no-simulate-lp` skips the LP solve, and `--curtailment` and
`--c-deg-per-kwh` override the defaults for one run.

### 3. `sweep_c_deg.py`, cohort

    python3 -m energy_masters_pipeline.scripts.sweep_c_deg

Solves the LP at each candidate `c_deg_per_kwh` and finds where LP
throughput meets actual throughput. Called with no site argument it
sweeps the whole cohort and fits one pooled value; called with a site
argument it does one site. The default sweep is 0.01 to 0.07 $/kWh in
steps of 0.01. Cohort runs also write `charts/thesis_c_deg_throughput.png`
(and `.pdf`), the total throughput ratio on a single pane for the report;
`--from-csv` redraws the tables and charts from the saved CSV without
solving.

`c_deg` stands in for Amber's arbitrage threshold, which is a property of
the coordinator rather than of a site, so one value is fitted across the
cohort. **This has been run against the final reviewed sites and 0.039
$/kWh has been chosen.** That value is now the default in
`viz_energy_masters.DEFAULT_C_DEG_PER_KWH`, so it flows to every
downstream script without being passed explicitly.

Two design points that are easy to undo:

- The fit is pooled (sum kWh across sites, then find the crossing), never
  a mean of per-site fitted values, which would overweight sites with
  small denominators.
- Pool only over sites that solve at every `c_deg` value. A ragged site
  set makes the pooled curve jump for reasons that are purely about which
  solves succeeded. The script is fail-soft per value and reports
  failures rather than dropping them quietly.

Fit on total throughput (`total_throughput_C+Q_kwh`). The LP's
degradation term is `c_deg * (charge + discharge)`, so throughput is the
quantity `c_deg` acts on directly. AC charge and the two arbitrage metrics
(`grid_to_bat`, `bat_to_grid`) are reported as cross-checks. In the July
2026 run the arbitrage crossings were 0.034 and 0.046 $/kWh, either side
of the 0.039 throughput crossing.

### 4. `run_cohort_attribution.py`

    python3 -m energy_masters_pipeline.scripts.run_cohort_attribution

Loops every `accept` or `override` site in the review CSV, runs
`load_and_analyze(simulate_lp=True)` and the daily value attribution on
each, and writes:

- `cohort_value_attribution_summary.csv`, one row per site
- `{site_id}_daily_value_attribution.csv`, one row per site-day

Both are consumed by step 5. Fail-soft per site: one site erroring out
does not lose the rest, and failures are listed at the end.

It threads both `--curtailment` and `--c-deg-per-kwh` through to
`load_and_analyze`, and records the settings actually in force as
`c_deg_per_kwh` and `curtail_mode` columns on every summary row, so a
summary CSV can always be traced back to the assumptions that produced
it.

Runtime is minutes, not seconds: each site is a full ingest,
reconstruction, self-consumption run and LP solve.

### 5. `chart_vcr_results.py`

    python3 -m energy_masters_pipeline.scripts.chart_vcr_results

Reads the two artefacts from step 4 and writes the cohort charts to
`charts/` (or `--show` to open them): tier contribution, per-site VCR
with leave-one-out fragility, day-level scatter, concentration, and VCR
against capacity band.

Every fitted line is pooled within a bin (`sum(captured)/sum(headroom)`),
never regressed, for the same reason the c_deg fit is pooled. Days below
`MIN_HEADROOM_DOLLARS` are dropped from the scatter and the chart says
so: at the $0.50 default that removes about a quarter of site-days but
only 2.4 percent of total headroom.

Its `EXCLUDE_SITES` tuple is currently empty and should generally stay
that way. Manual exclusions belong in the review CSV, where they carry a
reason; the tuple exists only as a last-resort override at chart time.

## What runs automatically

These modules are imported and called inside
`viz_energy_masters.load_and_analyze`, which is the single entry point
that steps 2, 3 and 4 all go through. None of them needs to be run as
part of producing results.

| Module | What it contributes |
|---|---|
| `capacity_review.py` | Looks up the site's review decision, raises if unreviewed or excluded, applies date overrides to the valid periods |
| `em_ingest.py` | Loads and cleans the raw channels, builds the component flows, builds the price vectors |
| `estimate_battery_capacity.py` | Usable capacity plus charge and discharge efficiencies as a self-consistent triple, and the initial SOC |
| `estimate_battery_max_power.py` | One symmetric battery AC power limit |
| `estimate_export_limit.py` | Site export limit, when no `export_kw_override` is set |
| `reconstruct_flows.py` | Allocates measured solar, grid and battery flows into the full `SiteTraces` decomposition and reconstructs SOC |
| `analyse_daily_value_gaps.py` | Daily and mechanism attribution, called by step 4 |

The order inside `load_and_analyze` matters and is already correct in the
code: review lookup, then date overrides, then component flows, then
capacity, then power, then reconstruction, then billing, then the
self-consumption and LP simulations. Capacity estimation in particular
sits after the plausibility clip and after the date overrides, so a
period override changes the capacity estimate rather than being applied
on top of one computed over the stretch the override exists to drop.

Each of these modules also has its own `run_checks` entry point that
prints diagnostics and saves a sense-check plot for one site. Those are
for investigating a specific site, not for producing results.

`analyse_daily_value_gaps.py` additionally has a full CLI for a per-site
deep dive: mechanism buckets, concentration, VCR by opportunity size,
leave-one-out robustness, monthly view. Useful when a site looks odd in
the cohort table.

## Current settings

Read from the code, not recalled:

| Setting | Value | Where |
|---|---|---|
| `c_deg_per_kwh` | 0.039 $/kWh | `viz_energy_masters.DEFAULT_C_DEG_PER_KWH` |
| Curtailment | pinned to zero | `viz_energy_masters.DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS = None` |
| Network tariff | `SA_SAPN_RESELE_2025` | `em_ingest.DEFAULT_NETWORK_TARIFF_ID` |
| Retail tariff | `SA_amber_wholesale_2026` | `em_ingest.DEFAULT_RETAIL_TARIFF_ID` |
| State | SA, whole cohort | `em_ingest` |
| Export limit | per-site override, else auto-estimated | review CSV, `estimate_export_limit.py` |

Curtailment is pinned because Amber has no inverter access, so an
unpinned LP benchmark is not a like-for-like comparison. Pinning can make
the LP infeasible for a given site and period, which surfaces as a
per-site failure rather than a silent fallback.

## Sequencing traps

Things the code does not enforce and that will be easy to forget:

1. **Re-check the review CSV after any change to the estimators.**
   Overrides were set against particular estimates. If an estimator
   moves, an override that was a correction may become a distortion, and
   nothing will tell you.

2. **Calibrate c_deg only after per-site overrides are final.** The fit
   matches LP throughput to actual throughput, and capacity, power and
   export limit all move LP throughput directly. EM-427's power estimate
   was out by about 3 percent and its export limit by about 4, both in
   the direction that inflates the quantity being fitted.

3. **Keep curtailment mode consistent across steps 2, 3 and 4.** All
   three take `--curtailment` and share one default. A run that sets it
   explicitly at one step and not another is comparing different models.
   The summary CSV records the mode per row, so check that column before
   comparing two runs.

4. **The 30-minute price correction is for reconciliation and billing
   only, never for the LP.** The LP optimises against the genuine
   5-minute wholesale price regardless of what resolution a site is
   billed at. Easy to lose when touching price handling.

5. **Step 5 globs a whole directory.** `chart_vcr_results.py` picks up
   every `*_daily_value_attribution.csv` under
   `data/processed/energy_masters`. A stale
   per-site file from an earlier run under different settings will be
   picked up silently and pooled with the current ones.

## Known limitations

### Data handling

1. **Missing data handling is deliberately simplified.** Plain linear
   interpolation on the three raw channels, no gap-length cap, no
   flagging. It does not distinguish a genuine value from an interpolated
   one, does not flag flatlined stretches, and has no long-gap
   synthetic-day fallback. The upgrade path, if this data becomes central
   rather than a sense-check, is to mirror the EDP pipeline's flag-code
   convention and two-stage infill.

2. **The plausibility clip catches only physically impossible
   magnitudes.** Values above 5000 Wh per 5-minute interval are masked
   before interpolation. A smaller but still wrong value passes through.

3. **Multi-instance summing has never been genuinely exercised.**
   EM-414's second system has no rows inside its valid period, so only
   the first instance ever contributes. The logic is confirmed not to
   break anything, but a site where two instances genuinely overlap has
   not been tested.

4. **EM-427 has 96 unexplained negative-load intervals** (4.936 kWh,
   about 0.1 percent of load). Not the benign parasitic-draw case. Too
   small to move any headline number, root cause not investigated.
   `load_and_analyze` warns when this count is nonzero and the cohort
   summary carries it per site.

### Model assumptions

5. **`c_deg_per_kwh = 0.039` is a fitted value, not a measured one.** It
   is a single cohort-wide stand-in for Amber's arbitrage threshold, and
   the results have not yet been tested for sensitivity to it. See
   outstanding actions.

6. **Battery max power and export limit may be the same physical
   constraint.** For EM-427 they were independently estimated at 5.172 kW
   and 4.992 kW from different signals, close enough to suspect a single
   inverter AC rating. Currently modelled as two independent constraints.
   Probably immaterial, wrong in principle.

7. **Billing across year boundaries.** `_span_days` in `billing.py` is
   safe only because all EM valid periods are well under 365 days. This
   is reasoned, not enforced. An assertion would fix it.

8. **The flow allocation rule is a modelling choice, not a
   measurement.** `reconstruct_flows.py` allocates by a physical priority
   rule (solar, then grid, then battery, for serving load). Its residual
   noise is small but nonzero, and it was the confirmed contaminant that
   closed out the arbitrage cycle work.

### Inference limits

9. **Coordination-off periods cannot be detected automatically.** Amber
   can temporarily disable battery coordination, and a coordination-off
   stretch surviving into the bill comparison makes actual look
   artificially close to self-consumption, biasing cohort VCR down for
   reasons unrelated to coordination quality.

   Current mitigation: the highest-headroom days have been reviewed by
   hand for every site, and sites showing no evidence of automated
   coordination are excluded in the review CSV. This catches whole-site
   cases, not intermittent ones. An automated detector remains an
   outstanding action.

   The arbitrage-cycle approach to this is CLOSED with a nuanced negative
   result: real allocation noise (confirmed), imprecise session
   boundaries on noisy real SOC (plausible), and genuinely
   imperfect-foresight dispatch (likely) are all partly true and not
   separable with that method. Do not resurrect it expecting threshold
   tuning to resolve it.

10. **The effective sample is much smaller than it looks.** Three
    calendar dates carry roughly 45 percent of all cohort headroom, so
    spike-day estimates rest on about three independent events rather
    than hundreds of site-days. Several sites have windows entirely after
    the January price events and therefore near-zero event exposure. Any
    bootstrap over the top bucket must resample calendar dates, not
    site-days.

11. **A single headline VCR is not a stable statistic.** For EM-427 the
    leave-one-day-out range was 0.275 to 0.415 and the monthly range was
    -0.10 to 0.58. Report a range and its sensitivity, not a point value.
    The cohort summary carries `vcr_loo_min`, `vcr_loo_max` and the tier
    columns for exactly this reason.

12. **Per-day VCR is too noisy to plot directly.** Headroom is near zero
    on most days, hence the pooling rule and the minimum-headroom filter
    in the charts. Sites whose total headroom falls below
    `SITE_HEADROOM_FLOOR_DOLLARS` are reported separately rather than
    folded into the cohort figure.

## Outstanding actions

1. **c_deg sensitivity analysis over 0.034 to 0.046.** Re-run
   `run_cohort_attribution.py --c-deg-per-kwh` at the endpoints and
   compare the headline cohort VCR and the tier breakdown against the
   0.039 result. This is what converts the fitted value from an
   assumption into a bounded one.
2. **Build the coordination-off detector.** The agreed lightweight
   approach is a daily anomaly signal: compare each day's battery
   throughput against other days with similar available solar, and flag
   days with normal solar but near-zero cycling. It does not depend on
   resolving what counts as a deliberate round trip, so it avoids the
   structural problem that closed the arbitrage-cycle work.