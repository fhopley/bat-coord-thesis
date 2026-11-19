# predispatch_data — Specification

## Purpose and scope

The canonical store of AEMO pre-dispatch price forecasts and the pipeline in
`pipeline/pre_dispatch/` that builds it. Pre-dispatch records what AEMO
expected prices to be at each past instant. It is the real forecast source for
the rolling coordination types, and the only one covering the full EDP
period.

The only consumer is `PredispatchForecastProvider` (see
`forecast_providers_spec.md`), which reads with `B_Clean_Predispatch.read_store`.
The settled prices that forecasts are scored against, and that infill reads,
are specified in `nem_price_spec.md`.

The store is built once and read many times, so every expensive or error-prone
decision is made here rather than in the read path, which runs about 105,000
times per site-year.

The facts below come from a one-off investigation of the AEMO pre-dispatch
archive, which is not part of the repository.

---

## Source tables

| | P5MIN / REGIONSOLUTION | PREDISPATCH / PRICE |
|---|---|---|
| Run cadence | 5 min | 30 min |
| Targets per run | 12, constant | 32 to 79, sawtooth |
| Target spacing | 5 min | 30 min |
| Horizon | 55 min | 15.5 to 39 h from the run label |
| Run time | `RUN_DATETIME` | derived from `PREDISPATCHSEQNO` |
| Target column | `INTERVAL_DATETIME` | `DATETIME` |
| Archive folder | `DATA/` | **`PREDISP_ALL_DATA/`** |

Both tables label targets **interval-end**, on the same grid as the settled
price files. The PREDISPATCH half hour labelled 14:30 covers the 5-minute
actuals labelled 14:05 to 14:30.

Five columns are used: run time, target, `REGIONID`, `RRP`, `INTERVENTION`.

P5MIN targets are at 5-minute resolution throughout, including before the
October 2021 five-minute settlement change. That change affected settlement
and the actuals series, not the forecast grid.

### Why not nemseer

nemseer 1.0.7 cannot retrieve this data, for two reasons:

- **Filenames.** It constructs `PUBLIC_DVD_{TABLE}_{yyyymm}010000.zip`, but
  recent months are published as `PUBLIC_ARCHIVE#{TABLE}#FILE01#...` and split
  across several part files. Its table-discovery regex excludes `#`, so it
  fails with `Tables include: []`, which reads like an absent table rather than
  a broken parser.
- **pandas 3.** Its `clean_forecast_csv` assigns datetimes into a string
  column, which pandas 3 rejects.

`nemweb_direct.py` parses the archive directory listing instead of
constructing filenames, handles both naming eras and multi-part files, and
reimplements nemseer's cleaning faithfully, so results remain comparable with a
nemseer-based extraction.

---

## Four traps

Each of these produces plausible-looking output rather than an error.

### 1. Source files are partitioned on a shifted clock

PREDISPATCH files are partitioned by `PREDISPATCHSEQNO` trading date, and
trading date D covers runs from D 04:30 to D+1 04:00. The nine runs from 00:00
to 04:00 on the 1st of a month are therefore in the **previous** month's file.
P5MIN has the same offset one interval wide (the 00:00 run). A loader
selecting files by the calendar month of the run loses about 600 PREDISPATCH
and 67 P5MIN runs over 2021–2026 without any error.

**Handled in** `nemweb_direct.compile_data`, which selects source files on the
month of `run_time − 4h30m` (PREDISPATCH) or `run_time − 5 min` (P5MIN), and in
`B_Clean_Predispatch.clean_range`, which also reads the previous month's
cached file. The store itself is partitioned by the **calendar** month of
`run_time`, so a reader can assume `202602.parquet` holds February runs and
nothing else.

### 2. The wrong directory loads cleanly

The `DATA/` copy of PREDISPATCHPRICE has a `_D` suffix and holds only the
lead-0 row: one forecast per target, no lead dimension. For September 2022 it
has 7,200 rows against 399,600 in `PREDISP_ALL_DATA/`. A store built from it
would load cleanly and be useless for rolling optimisation.

**Handled in** `nemweb_direct`, which routes PREDISPATCH PRICE to
`PREDISP_ALL_DATA/`. The trailing delimiter in `_matches` keeps the table
token from also matching `_D` (and `P5MIN_CONSTRAINT` from matching
`P5MIN_CONSTRAINTSOLUTION`).

### 3. Dtypes vary by month

nemseer-style cleaning downcasts floats to integer or float32 where values
allow it, so the same column changes dtype between adjacent months (`RRP` is
float32 in P5MIN 2021-01, float64 in 2026-02 and float32 in 2026-03).

**Handled in** stage B, which casts every column explicitly.

### 4. October 2022 has no multi-lead PREDISPATCH

`PREDISP_ALL_DATA/` for 2022-10 does not contain the price table in any of its
seven subfolders. The multi-lead forecast for that month is not in the archive,
while `DATA/` holds the `_D` variant at 100% coverage, so traps 2 and 4
compound. P5MIN is complete for the month.

**Handled** by treating it as a known archive gap:

- stage A raises `SourceMonthUnavailable` for the month and records the
  failure without stopping;
- the `_D` variant is not ingested;
- stage C lists it in `KNOWN_ARCHIVE_GAPS`;
- the manifest flags it `known_archive_gap`.

The store's 2022-10 PREDISPATCH partition holds only the single run that
spills over from the September file. The provider acts on the flag (see
`forecast_providers_spec.md`, degraded periods).

---

## Canonical store

```
data/raw/predispatch/{TOKEN}_{YYYYMM}.parquet            stage A cache, all source columns
data/clean/predispatch/p5min/{REGION}/{YYYYMM}.parquet
data/clean/predispatch/predispatch/{REGION}/{YYYYMM}.parquet
data/clean/predispatch/manifest.json
```

`{TOKEN}` is `P5MIN_REGIONSOLUTION` or `PREDISPATCHPRICE`. All five NEM
regions are kept (`NSW1`, `QLD1`, `SA1`, `TAS1`, `VIC1`): source files are not
split by region, so extra regions cost nothing to acquire. The two tables are
stored separately because their cadence, spacing and role in the provider
differ (fine and coarse segments).

| Column | Type | Notes |
|---|---|---|
| `run_time` | `datetime64[us, Etc/GMT-10]` | nominal run label |
| `lead_minutes` | `int16` | 0–55 (P5MIN), 0–2340 (PREDISPATCH) |
| `rrp_mwh` | `float64` | explicitly cast |

- The region is the partition and is not stored.
- `target_time = run_time + lead_minutes` exactly. The lead is stored rather
  than the target because it is more compact and is the natural key for
  error-by-lead analysis.
- `float64` because these prices feed money calculations and the exactness
  test in `prices.py`.

**Run time is the nominal label, not publication.** For P5MIN it is
`RUN_DATETIME`. For PREDISPATCH there is no run time column. It is derived in
`nemweb_direct._parse_predispatch_seq_no` as

```
run_time = PREDISPATCHSEQNO date + 04:30 + (PP − 1) × 30 min,   PP = 1..48
```

This was verified exact: `2026021018` gives 2026-02-10 13:00. Because this is
arithmetic on a sequence number, a change in AEMO's convention would shift
every run time silently. The run-grid check in stage C is the guard.

Both tables publish **before** their label (P5MIN by 4.35 minutes on average,
3.12 minutes at the tightest observed; PREDISPATCH by 28.39 and 27.37). Using
the label as the availability instant is therefore conservative. The store does
not adjust for this and must not.

**Time zone.** MMS timestamps are market time (UTC+10, no DST). Stage B
localises to `Etc/GMT-10` with no shift, and nothing downstream converts.

---

## Cleaning (`B_Clean_Predispatch.clean_forecast_frame`)

1. **Select the five columns by name.** Schema drift over 2021–2026 is
   additive only (18 columns added to P5MIN, 2 to PREDISPATCH), so naming
   columns keeps one loader valid across the period. A missing column raises.
2. **Keep `INTERVENTION == 0`.** On intervention intervals AEMO publishes two
   cases:
   - Case 0 is the counterfactual pricing run and matches settled prices
     exactly (June 2022, all 2,672 SA1 intervention intervals: MAE $0.00).
   - Case 1 is the physical dispatch run's shadow price (MAE $89.71).

   The two agree only 59.5% of the time, and where they differ the mean gap
   is $1,143/MWh ($1,767/MWh in SA1). That is exactly where arbitrage value
   concentrates, so the choice matters. Filtering is lossless: every
   intervention key has a case-0 row. P5MIN has never been observed with a
   non-zero value; the filter is applied to it anyway.
3. **Assert uniqueness** of `(run_time, region, target_time)`. Exact duplicate
   rows are dropped upstream, so a surviving duplicate differs in value and
   raises rather than being resolved silently.
4. **Check leads.** A target before its run means the run-time derivation is
   wrong. A lead that is not a whole multiple of the target spacing means the
   target cannot be reconstructed. Both raise.
5. **Cast** (`int16` lead, `float64` price) and **localise** `run_time`.
6. **No null prices**; a null raises.

`write_partitions` writes one file per region and calendar month. With
`merge=True`, which stage A and `clean_range` use, it merges into an existing
partition, keeping the newer row per `(run_time, lead_minutes)`, because
consecutive source months write into the same calendar month.

---

## Validation (`C_Validate_Predispatch`)

Re-runnable independently of acquisition. For each table, region and month it
produces a `TableReport` and writes the manifest.

- **Run grid.** Runs are compared with the full calendar-month grid
  (5 or 30 minute). Missing runs are split into leading, trailing and internal.
  Only internal gaps fail (`ok = False`). Leading and trailing gaps mark the
  month `partial`, which is normal at the edge of an acquired range (a month's
  first runs live in the previous source file). A grid built from the observed
  first and last run would make a one-day partition look complete. Runs off
  the grid are errors.
- **Target counts** per run against the closed form, never against a median.
  The daily sawtooth would make a median check flag about half of all runs.

  ```
  P5MIN:        12
  PREDISPATCH:  horizon_end(R) = 04:00 on R.date + 2 days   if R.hour ≥ 13
                                 04:00 on R.date + 1 day    otherwise
                count(R) = (horizon_end(R) − R) / 30 min + 1
  ```

  Bid submission closes at 12:30, so runs from 13:00 see one trading day
  further.
- **Price range** per financial year. Every price must lie between
  `market_price_floor` and `market_price_cap` from `model/core/market_prices.py`,
  for the financial year of the partition's middle run. This lives here, not
  in the provider, because the cap resets each 1 July.
- **Duplicates** on `(run_time, lead_minutes)`.
- **Flags:**
  - `degraded`: runs present but `max_lead_minutes == 0`, the signature of a
    lead-0-only source;
  - `known_archive_gap`: listed in `KNOWN_ARCHIVE_GAPS`.

  Either flag makes the provider treat the period as degraded. `degraded` is
  deliberately narrow, so that a short partition is not mistaken for one with
  no forecast.

`manifest.json` has one entry per table, region and month:

- run and row counts;
- `max_lead_minutes`;
- missing runs split into leading, trailing and internal;
- `partial`, wrong target counts, out-of-range count;
- `degraded`, `known_archive_gap`, `ok`, `reason`.

It also carries `schema_version`, `generated_utc` and `market_timezone`. The
exit status is non-zero if any partition fails.

---

## Pipeline

Run from the project root:

```bash
python3 -u -m pipeline.pre_dispatch.A_Download_Predispatch --start 2021-01 --end 2026-06 [--clean] [--force] [--tables ...] [--regions ...]
python3 -u -m pipeline.pre_dispatch.B_Clean_Predispatch --start 2021-01 --end 2026-06
python3 -u -m pipeline.pre_dispatch.C_Validate_Predispatch [--start YYYY-MM --end YYYY-MM]
```

**Stage A (`A_Download_Predispatch`).** Fetches each calendar month of each
table through `nemweb_direct.compile_data` into the raw cache. With `--clean`
it also runs stage B inline, merging into the store.

- **Fail-soft per table-month:** failures are collected and summarised at
  the end. An unavailable month (`SourceMonthUnavailable`) is reported without
  a traceback because it is a data condition. Anything else prints its
  traceback.
- **Idempotent:** a month is skipped unless `--force` is given when every
  region's partition holds more runs than the spillover bound (1 for P5MIN, 9
  for PREDISPATCH). Testing only that the file exists was a real defect: the
  previous month's download creates a one-run stub, so every month after the
  first was skipped.
- The first month of a range is short by those spillover runs. Start one
  month earlier to fill them.

**Stage B (`B_Clean_Predispatch`).** Rebuilds the store from the raw cache
without the network, reading each month's file and the previous month's. It
is separate from A so a change to the cleaning rules can be applied to years of
data from cache. It fails soft per table-month when a cache file is absent;
cleaning errors fail hard.

**Stage C (`C_Validate_Predispatch`).** As above. Stages B and C fail hard on
defects; only acquisition is fail-soft.

**Diagnostic.** `check_dispatch_timing.py --month YYYY-MM` measures when the
5-minute dispatch price is published, using `LASTCHANGED − SETTLEMENTDATE` in
DISPATCHPRICE. This is the source of the "settled price known about 9 s into
the interval" fact behind the `price_known` and `price_and_next` modes.

---

## Verification

`model/tests/test_predispatch_pipeline.py`:

- **Stage B:** canonical schema and types; target reconstruction;
  INTERVENTION filtering and resulting uniqueness; a surviving duplicate,
  negative or misaligned lead, null price, missing column or unknown table
  raises; dtype normalisation; partitioning by region and calendar month;
  merging two source months; `read_store` round-trip and typed empty result.
- **Stage C:** the sawtooth closed form and its bounds; P5MIN's constant
  count; a healthy partition passes; a missing run, wrong target count or
  price above cap or below floor is detected; lead-0-only months are marked
  degraded; an absent partition is reported; complete, truncated, one-day and
  internally gappy months are classified correctly; the manifest records
  degraded and partial months.
- **Stage A,** with an injected fetcher: calendar run window; multi-lead
  source routing; region filtering; idempotence and `--force`; fail-soft on a
  missing month; no store writes without `--clean`; intervention count
  reporting.
- **`clean_range`:** cache filenames match `nemweb_direct`; skips when a
  month's cache is absent; reads the previous month's file; is partial, not
  wrong, without it; does not rewrite the previous month.

---

## Known limitations

- **No lead-0 cross-check against `_D`.** The `_D` file is an independent copy
  of the lead-0 slice, about 1.5% of the full size, and matched the full table
  exactly for September 2022. It is the only independent check available on the
  store's contents. It is not implemented. Fix path: in stage C, fetch `_D` per
  month and assert the store's lead-0 rows reproduce it.
- **INTERVENTION rule not independently confirmed.** The settled prices used
  to choose case 0 came from a NEMOSIS extraction that may itself filter on the
  same column. That is sufficient for value capture, where forecast and
  actuals must agree, but it is not proof of what AEMO settles. Fix path: check
  the choice of case 0 against TRADINGPRICE directly.
- **Range check depends on the cap table.** A partition whose financial year
  is missing from `market_prices.py` fails validation until the table is
  extended. The table holds determined values only.

---

## Validation results

From the investigation and the full sweep of 2021-01 to 2026-06:

- The PREDISPATCH closed form matched all 1,345 February 2026 runs.
- The sweep found no months with wrong target counts, no duplicates and no
  missing regions.
- There were four isolated missing PREDISPATCH runs out of about 96,000:
  2022-01-31 14:30, 2023-12-28 16:30, 2024-09-05 14:00 (two consecutive) and
  2024-11-19 14:00.
- `RRP` had no nulls in 483,840 P5MIN and 370,860 PREDISPATCH rows checked.
