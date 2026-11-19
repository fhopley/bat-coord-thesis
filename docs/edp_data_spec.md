# EDP data — Specification

## Purpose and scope

Covers the raw EDP household energy monitoring data, the preparation pipeline
in `pipeline/edp/` that turns it into clean per-site files, and the format of
those clean files. The clean files are the site traces read by
`model/components/ingest.py`.

---

## Raw data

### Files

- One file per month: `data/raw/edp/edp_data_{YYYY}_{MM}.csv`, 2020-01 to 2025-12.
- About 3.7 GB per month, so every script reads in chunks
  (`chunksize=500_000`) and loads only the columns it needs via `usecols`.
- One row per circuit measurement (phase, device or circuit type). A site
  at one timestamp typically has 2–10+ rows.
- Rows within a file are not guaranteed to be in timestamp order.
- Site metadata: `data/raw/edp/edp_sites_metadata_2020_2025.csv`. Some sites
  have more than one metadata row; the pipeline keeps the row with the
  largest `inverter_ac_rating_kw`. Sites without a metadata entry are dropped
  at site selection.

### Columns

| Column | Type | Description |
|---|---|---|
| `edp_site_id` | string | Site identifier. Prefix `S` = Solar Analytics, `W` = Wattwatchers. Sign conventions and units are consistent between the two, so both are pooled. |
| `unix_time` | integer | Epoch seconds, but in **site local time**, not UTC. DST-aware (see Timestamps). |
| `datetime` | string | `YYYY-MM-DD HH:MM:SS` equivalent of `unix_time`, in site local time. Interval-start: the first timestamp of the year is `00:00:00`. |
| `edp_device_and_circuit` | string | Device and phase/circuit identifier, e.g. `A1`, `A2`, `A3`. Has trailing whitespace; strip before use. |
| `circuit_label` | string | Circuit type, e.g. `ac_load_net`, `pv_site_net`, without phase reference. The primary filter field. May have leading/trailing whitespace; strip before use. |
| `edp_circuit_label` | string | `circuit_label` and `edp_device_and_circuit` combined, e.g. `pv_site_net_A1`. |
| `real_energy` | float | Energy over the 5-minute interval, **Wh**. The only energy field used. |
| `real_energy_negative`, `real_energy_positive` | float | Components of real energy. Not used; interpretation unknown. |
| `reactive_energy`, `current_*`, `voltage_*` | float | Not used. Not always populated. |

The pipeline reads only `edp_site_id`, `unix_time`, `datetime`,
`edp_device_and_circuit`, `circuit_label` and `real_energy`.

### Circuit labels

| `circuit_label` | Meaning | Sign |
|---|---|---|
| `ac_load_net` | Net AC load: gross consumption minus PV generation. `gross_load = ac_load_net + pv_site_net`. | Positive = net consumption |
| `pv_site_net` | PV generation. | Positive = generation |
| `battery_storage`, `load_battery`, `pv_site_net_battery` | Separately metered battery channel. Used only to reconstruct battery-free load for battery sites (see Battery sites). | Determined per site |

Other labels (e.g. `load_air_conditioner`) are ignored.

Phases are collapsed by summing all rows sharing
`(edp_site_id, unix_time, circuit_label)`. This is correct for single- and
three-phase sites alike, but understates the value when one phase is missing
at a timestamp (see Known limitations, item 9).

### Timestamps and daylight saving

Raw timestamps are site local time, interval-start. The site time zone comes
from `site_timezone` in the metadata, falling back to a state default when it
is missing.

- **Autumn-back (April).** One hour of local timestamps occurs twice, and the
  data holds only one reading for each. The clean files have 12 missing AEST
  intervals on the transition date: 02:05–03:00 AEST for NSW, VIC and TAS
  time zones, 02:35–03:30 AEST for SA. These intervals get Flag 3. See Known
  limitations for which occurrence the retained readings are assigned to.
- **Spring-forward (October).** The skipped local hour never existed.
  Pre-gap and post-gap local timestamps are consecutive in AEST, so no data is
  missing.
- QLD has no DST and is unaffected.
- 288 intervals per day: 105,120 per non-leap year, 105,408 for 2020 and
  2024.

---

## Pipeline

`pipeline/run_pipeline.py` runs scripts A to L in `pipeline/edp/` for one
year, stopping at the first failure:

```bash
python3 -u pipeline/run_pipeline.py 2022 | tee logs/pipeline_2022.log
```

All intermediate outputs are in `data/processed/edp/`, prefixed `YYYY_`.

| Script | Does | Main output |
|---|---|---|
| `A_Site_Summary.py` | Scans the raw files: per-site, per-label coverage (distinct timestamps / expected intervals), maximum gap, phase count; sign and magnitude totals for the three battery labels | `YYYY_site_coverage.csv` |
| `B0_Battery_Site_Screen.py` | Screens every site with a battery-labelled channel (see Battery sites) | `YYYY_battery_site_screen.csv` |
| `B_Site_Selection.py` | Coverage, inverter and battery filters (see Site selection) | `YYYY_selected_sites.csv` |
| `C_Check_Consistency.py` | Seven consistency checks and manual exclusions | `YYYY_consistency_checks.csv`, `YYYY_selected_sites_clean.csv`, `YYYY_selected_sites_excluded.csv` |
| `D_Summarise_Site_Funnel.py` | Site counts by state at each selection stage | `YYYY_site_funnel_summary.csv` |
| `E_Flag_All_Sites.py` | One pass over the raw files for all selected sites: phase aggregation, AEST conversion, regular grid, flags | `data/clean/edp/YYYY/edp_clean_{site_id}_{YYYY}.csv`, `YYYY_site_pv_stats.csv` |
| `F_Site_Stats.py` | Flag summary and PV statistics before infill | `YYYY_flag_summary.csv`, `YYYY_site_pv_stats.csv` |
| `G_Infill_Short_All_Sites.py` | Infill Method 1 (updates clean files in place) | clean files |
| `H_Build_PV_Profile.py` | State PV profile library for Method 2 | `YYYY_pv_profile_library.csv` |
| `I_Infill_Long_All_Sites.py` | Infill Method 2 (updates clean files in place) | clean files |
| `J_Site_Stats.py` | Flag summary and PV statistics after infill | `YYYY_flag_summary.csv`, `YYYY_site_pv_stats.csv` |
| `K_Highlight_Marginal_Sites.py` | Prints the sites closest to each consistency-check threshold, on both sides | terminal only |
| `L_Convert_To_Parquet.py` | Converts clean CSVs to Parquet and deletes the CSVs | `data/clean/edp/YYYY/edp_clean_{site_id}_{YYYY}.parquet` |

`pipeline/pipeline_funnel_summary.py` builds cross-year funnel tables from the
`D_Summarise_Site_Funnel.py` section of the pipeline logs.

Scripts prefixed `viz_` are visual inspection tools and are not part of the
pipeline run.

---

## Site selection

### Coverage, inverter and battery filters (`B_Site_Selection.py`)

Each filter is reported on its own and then applied together.

1. `ac_load_net` coverage between 85% and 100.5%.
2. `pv_site_net` coverage between 85% and 100.5%.
3. `inverter_ac_rating_kw` ≥ 1.0 kW. A null rating is treated as unknown and
   passes.
4. Battery screen. A site has battery evidence if metadata says so
   (`islandable = Yes`, `has_battery = t`, or `battery_size_make_model`
   populated) or if a battery-labelled channel is present in the raw data.
   Sites with battery evidence are excluded unless `B0_Battery_Site_Screen.py`
   marked them usable.

Coverage counts distinct timestamps present, not whether the values are
meaningful. The upper bound catches sites where more distinct timestamps
exist than a single meter can produce, as with W0017 (two households under one
site ID).

### Battery sites (`B0_Battery_Site_Screen.py`)

A site's `ac_load_net` includes battery charge and discharge. Where the
battery is separately metered, subtracting (or adding, depending on the sign
convention) the battery channel recovers the battery-free net load. A site
with any battery-labelled channel is usable if all of the following hold:

| Rule | Threshold | Reason |
|---|---|---|
| Both `ac_load_net` and `pv_site_net` present | — | Without both there is nothing to reconstruct from. |
| Exactly one distinct battery-labelled channel | two or more → excluded | Two channels may be a complementary charge/discharge pair or redundant metering, which cannot be told apart. Such sites are rare. |
| Bidirectional | ≥ 5% of intervals on the minority sign | Excludes submeter-style channels that record one direction only. |
| Annual throughput | ≥ 200,000 Wh | Below this the sign split reflects noise rather than battery behaviour. |
| Charge/discharge ratio (larger side / smaller side) | ≤ 3.0 | Round-trip losses make charge exceed discharge over a year, so the larger side is charging, which fixes the sign convention. The bound is loose (about 33% round-trip efficiency) and catches structurally one-sided channels; sites that pass still face the consistency checks. |

An implausibly large interval is not a site-level rule. It is flagged on the
reconstructed load (Flag 6) rather than excluding the site-year.

Usable sites carry `battery_channel_label` and `battery_sign_convention` into
`YYYY_selected_sites.csv`. For these sites `C_Check_Consistency.py` and
`E_Flag_All_Sites.py` use the corrected, battery-free `ac_load_net`
everywhere except Check 3 and Flag 2, which test the raw sensor reading: a
stuck sensor could be masked by a varying battery term. Where the battery
channel has no value at an interval, the raw `ac_load_net` is used for that
interval.

### Consistency checks (`C_Check_Consistency.py`)

Solar hours are 09:00–16:00 **site local time** for all checks. A site
failing any check is excluded.

| Check | Fails when | Catches |
|---|---|---|
| 1 | Correlation of `ac_load_net` with `pv_site_net`, over solar-hour intervals with PV ≥ 100 Wh, is above −0.2 | `ac_load_net` measuring gross load |
| 2a | 10 or more intervals where export exceeds PV by more than 50 Wh | undisclosed battery or unmonitored inverter |
| 2b | More than 10% of all intervals have `gross_load < 0` | partial-phase metering, other systematic faults |
| 3 | More than 5% of solar-hour raw `ac_load_net` values have magnitude ≤ 2 Wh | stuck load sensor |
| 4 | More than 5% of solar-hour `pv_site_net` values have magnitude ≤ 2 Wh | PV sensor fault or outage |
| 5a | Annual PV generation below 500 kWh | non-functional or implausibly small systems |
| 5b | Average daily gross load below 5 kWh | unoccupied sites |
| 6 | More than 1% of solar-hour `pv_site_net` values are below −20 Wh | reversed polarity and similar faults |
| 7 | Correlation of `gross_load` with `pv_site_net` during solar hours exceeds 0.475 | partial-phase metering |

Check 3 is limited to solar hours because PV drives continuous variation in
`ac_load_net` during the day even in an empty house, whereas near-zero net
load overnight is normal. The 2 Wh floor separates stuck sensors (≈ 0 Wh)
from solar-following flexible loads such as EV charging, which leave residuals
of 5–15 Wh.

Check 7 detects a CT clamp on one phase of a three-phase installation (or a
mis-set CT ratio). `ac_load_net` then captures a fixed fraction of the load, so
`gross_load` tracks PV rather than household behaviour. Sites with
solar-following loads (EV chargers, hot water, export limiting) have bounded
correlation and pass. The 0.475 threshold was set by visual review of all 2022
sites above 0.3. It should be rechecked against the `K_Highlight_Marginal_Sites.py`
output for each new year.

Sites listed in `data/raw/edp/manual_exclusions.csv` are always excluded:

- **W0017:** two homes share one site ID, each with independent three-phase
  metering (device prefixes A and B) and PV.
- **W0164:** overnight load up to about 55 kW, not a residential profile.

---

## Clean file format

One file per site per year: `data/clean/edp/{YYYY}/edp_clean_{site_id}_{YYYY}.parquet`,
snappy-compressed, read with `pd.read_parquet()`. Per-site metadata used by the
model comes from `data/processed/edp/YYYY_selected_sites_clean.csv`.

### Timestamps

Raw timestamps are converted to **AEST (UTC+10, no DST), interval-end**, the
convention AEMO uses, so price data joins directly on timestamp:

1. Localise `datetime` to the site time zone.
2. Convert to AEST.
3. Add 5 minutes (interval-start → interval-end).

For a NSW site in January (UTC+11), local `2022-01-01 00:00:00` becomes
`2021-12-31 23:05:00` AEST. Clean files cover the **AEST** calendar year.

Each file has a complete regular 5-minute grid for the year (105,120 or
105,408 rows). Intervals missing from the raw data are explicit rows with NaN
values and a flag, not absent rows.

### Columns

| Column | Parquet type | Description |
|---|---|---|
| `datetime_aest` | `timestamp[us, tz=Etc/GMT-10]` | Interval-end, AEST |
| `pv_site_net_wh` | `float64` | PV generation summed across phases, Wh |
| `ac_load_net_wh` | `float64` | Net AC load summed across phases, Wh (battery-free for usable battery sites). After infill, back-calculated as `gross_load_wh - pv_site_net_wh`. |
| `gross_load_wh` | `float64` | `ac_load_net_wh + pv_site_net_wh`. The infill target for load. |
| `pv_flag` | `int64` | Flag code for PV |
| `load_flag` | `int64` | Flag code for load |
| `pv_infill` | `int64` | 0 = original, 1 = Method 1, 2 = Method 2 |
| `load_infill` | `int64` | 0 = original, 1 = Method 1, 2 = Method 2 |
| `pv_original_wh` | `float64` | PV as extracted by `E_Flag_All_Sites.py`, before infill. NaN where the raw interval was missing. |
| `ac_load_net_original_wh` | `float64` | Net load as extracted by `E_Flag_All_Sites.py`, before infill. NaN where the raw interval was missing. |

Flags keep recording original data quality after infill; the infill columns
record what replaced the value.

### Flag codes

Codes are assigned independently per channel. When several conditions apply
to one interval, the higher code wins. Solar hours for flagging are
09:00–16:00 **AEST**.

| Code | Name | Condition |
|---|---|---|
| 0 | Clean | No issue |
| 1 | Missing | Interval absent after AEST reindex |
| 2 | Flatline | 12+ consecutive intervals where adjacent values differ by ≤ 0.01 Wh. PV: solar hours only, and only when the flatline value is below the site's daytime Q3 (flatlines at or above Q3 are inverter clipping, which is real). Load: all hours, any value, tested on the raw `ac_load_net`. |
| 3 | DST gap | The 12 autumn-back intervals, identified from a fixed table of transition times by time zone and year |
| 4 | Partial phase | Fewer phases contributed than the site's maximum for that channel. Value kept, may be understated. |
| 5 | Invalid | PV: `pv_site_net_wh < -20` Wh. Load: `gross_load_wh < 0`. |
| 6 | Implausibly large | PV: above the lower of the site's `Q3 + 3×IQR` (daytime positive intervals) and 1,667 Wh (20 kW). Load: `gross_load_wh > 2,500` Wh (30 kW). |

The 0.01 Wh flatline tolerance is tight on purpose: it catches sensors
reporting identical values without flagging genuinely low but varying
consumption. The −20 Wh PV floor (≈ −240 W) allows for inverter standby draw
overnight, which is usually around −0.04 Wh but reaches −240 W on some
inverters.

---

## Infill

Both methods replace NaN intervals and flagged non-NaN intervals (flag > 0):
a flagged value is unreliable whether or not it is NaN. After each method,
`ac_load_net_wh` is back-calculated from `gross_load_wh - pv_site_net_wh`.

### Method 1: linear interpolation (`G_Infill_Short_All_Sites.py`)

- PV: runs entirely within 21:00–03:59 AEST (dark) are interpolated at any
  length; other runs only if ≤ 6 intervals (30 min).
- Gross load: runs ≤ 12 intervals (1 hour).
- Both boundaries must be Flag 0 and non-NaN. At the start or end of the year
  the single available boundary value is carried flat. Runs without clean
  boundaries are left for Method 2.

### Method 2: synthetic day (`H_Build_PV_Profile.py`, `I_Infill_Long_All_Sites.py`)

- PV: `H_Build_PV_Profile.py` builds, for each AEST interval of the year, the
  mean normalised generation across same-state sites, using only original
  Flag 0 data. Each site is normalised by its own daytime Q3 (`pv_q3_wh` in
  `YYYY_site_pv_stats.csv`). Intervals with no contributors (overnight, year
  boundary) are 0. The script warns where fewer than 20 sites contribute.
  `I_Infill_Long_All_Sites.py` scales the profile by the target site's Q3.
- Gross load: from the site's own data. For each gap day, the two closest
  same-day-of-week days with all 288 intervals Flag 0 or Method 1 are
  averaged; if only one exists it is used alone.

The methods aim to get daily energy right rather than the timing of
individual intervals, because daily energy totals matter more to battery
dispatch than interval-level accuracy.

---

## Known data quality issues

### Undisclosed batteries (item 11)

Metadata battery flags are incomplete. Battery-labelled channels in the raw
data are also treated as battery evidence (see Battery sites). Batteries
with no metadata flag and no metered channel are caught by Check 2a (export
exceeding PV) or Check 2b (persistent negative gross load).

### Flatline / zero-value data (item 12)

A site can pass coverage while reporting stuck or zero values for long periods
(e.g. S0302, a whole month of flatline `ac_load_net`). Checks 3 and 4 screen
sites; Flag 2 marks the intervals.

### Systemic W-site platform outages (item 13)

Several W sites miss the same 7-day period in October simultaneously, which
is consistent with a platform outage. These sites can still pass coverage. The
gap is filled by Method 2. The exact dates and number of affected sites have
not been quantified.

### Partial-phase metering (item 14)

A CT clamp on one of three phases gives `gross_load ≈ (2/3) × pv_site_net`
during the day. Check 7 excludes these sites. Sites excluded for this reason
in 2022 include W0017, S0405, S0118, S0373, W0143 and S0379.

### Metadata time zone for VIC sites (item 15)

About 11 VIC sites have `site_timezone = Australia/Hobart` instead of
`Australia/Melbourne`. The DST schedules are the same, so the analysis is
unaffected. Metadata is used as-is, so summaries by state are preferred over
summaries by time zone.

### Null inverter rating (item 16)

153 of the 781 sites with 2022 raw data have no `inverter_ac_rating_kw`. They
pass the inverter filter without a verified system size. Check 5a
(minimum annual generation) is a partial safeguard.

### Daytime negative PV (item 17)

S0521, S0531 and S0554 had 19–48% of PV intervals flagged invalid, from
systematic negative readings during daylight rather than overnight standby
noise. Check 6 excludes these sites.

### Unoccupied sites (item 19)

A dwelling empty all year gives self-consistent near-zero load that passes
most checks but is not representative. Check 5b excludes these sites. Sites
with occasional short absences are retained.

---

## Known limitations

- **Partial-phase intervals (item 9).** When one phase is missing at a
  timestamp, the phase sum is understated. Flag 4 marks these intervals but
  the value is not replaced. The size of the effect has not been measured.
- **Autumn-back assignment.** The repeated local hour is localised with
  `tz_localize(..., ambiguous=True)`, which in pandas places it at the **first
  (DST) occurrence**, and the Flag 3 gap is the second (standard-time)
  occurrence. If the data system keeps the second reading, as has been
  assumed, the retained hour sits one hour early in AEST. This affects 12
  intervals per DST site-year. Fix path: localise with `ambiguous=False` and
  move the `DST_AUTUMN_BACK_UTC` start times in `E_Flag_All_Sites.py` one hour
  earlier.
- **Battery throughput floor.** The 200,000 Wh floor excludes a battery
  installed late in the year or out of use for part of it. Fix path: pro-rate
  the floor by the days the site reports data.
- **PV infill and clipping.** Sites with significant inverter clipping have a
  suppressed Q3, so Method 2 underestimates their generation.
- **PV infill and orientation.** The state profile does not capture
  site-specific orientation.
- **Load infill timing.** Method 2 reproduces daily energy but not the timing
  of individual peaks.

---

## Validation

- **Infill accuracy (W0341, 25–31 October 2022, masked week).** PV daily
  energy error −10% to +5%. Load daily energy error −15% to +35%, with six of
  seven days within ±15%.
- **Flag distribution (2022).** From `2022_flag_summary.csv`: Flag 6 load
  intervals were almost entirely from W0164, which is now manually excluded.

---

## Open questions

- Meaning of `real_energy_negative` and `real_energy_positive`. Neither is
  used.
