# NEM wholesale prices — Specification

## Purpose and scope

Covers the 5-minute regional wholesale spot price (RRP) series used by the
coordination model: its source, the two scripts in `pipeline/nem_prices/`
that produce it, and the clean file format. `model/components/ingest.py`
reads the clean files and joins them to the EDP site traces on
`datetime_aest`. Price forecasts are covered separately in
`predispatch_data_spec.md`.

---

## Source

AEMO **TRADINGPRICE**: the settlement price. DISPATCHPRICE is not used.
Before 1 October 2021 settlement was on the 30-minute average of six dispatch
prices, not the individual dispatch prices. The price that retailers settle
at, and that wholesale-exposed households (e.g. Amber customers) face, is the
trading price.

The data comes from two paths, split at `NEMOSIS_CUTOFF` (1 August 2024):

- **Before August 2024:** [NEMOSIS](https://github.com/UNSW-CEEM/NEMOSIS)
  `dynamic_data_compiler`, table `TRADINGPRICE`, columns `SETTLEMENTDATE`,
  `REGIONID`, `RRP`, filtered to the four regions. NEMOSIS caches source files
  in `data/raw/nem_prices/`.
- **From August 2024:** AEMO's aggregated price and demand files,
  `https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{YYYYMM}_{REGION}.csv`,
  one per month per region. From August 2024 AEMO's MMSDM archive files have
  `#` in their names, and NEMWeb rejects both the encoded and literal forms,
  so NEMOSIS 3.8.1 cannot download them (NEMOSIS issue #40). The aggregated
  files carry the same 5-minute trading price.

API details for NEMOSIS 3.8.1: the cache directory parameter is
`raw_data_location`; `filter_values` takes a tuple of lists, e.g.
`(['NSW1', 'QLD1', 'SA1', 'VIC1'],)`.

### Regions

| Region | State |
|---|---|
| `NSW1` | NSW |
| `QLD1` | QLD |
| `SA1` | SA |
| `VIC1` | VIC |

TAS1 is excluded because the EDP dataset has no Tasmanian sites.

### Five-minute settlement

From **1 October 2021** the NEM settles on 5-minute intervals. Before that,
TRADINGPRICE has one record per 30-minute trading interval per region; after
it, one per 5-minute interval. The statistical properties of the series change
at this boundary, which matters when comparing price volatility or intraday
shape across years.

---

## Pipeline

Both scripts process every year in the `YEARS` list at the top of the script
(currently 2020–2026). Run from the project root:

```bash
python3 pipeline/nem_prices/A_Download_NEM_Prices.py
python3 pipeline/nem_prices/B_Clean_NEM_Prices.py
```

### `A_Download_NEM_Prices.py`

Downloads each year through the path(s) covering it, drops duplicate
`(SETTLEMENTDATE, REGIONID)` rows, and writes
`data/processed/nem_prices/YYYY_nem_prices_raw.csv` with columns
`SETTLEMENTDATE` (AEST, interval-end, `YYYY-MM-DD HH:MM:SS`), `REGIONID` and
`RRP` ($/MWh). For the current year it fetches up to the current month.

### `B_Clean_NEM_Prices.py`

1. **Localise.** Parse `SETTLEMENTDATE`, attach the fixed `Etc/GMT-10` zone,
   and rename it `datetime_aest`. The source is already AEST; this makes the
   zone explicit in the dtype.
2. **De-duplicate.** Duplicate `(datetime_aest, REGIONID)` pairs are reported
   and the first is kept.
3. **Pivot** to one row per interval and one `{REGION}_rrp` column per region.
4. **Reindex** to the complete 5-minute AEST grid for the calendar year,
   split at `2021-10-01 00:05:00` AEST:
   - Before the split, `bfill(limit=5)`. `SETTLEMENTDATE` is interval-end, so
     the 00:30 price covers 00:05–00:25 as well. Between two consecutive
     30-minute prices there are exactly five empty slots; a missing
     30-minute price leaves 11 empty slots, and the 6 beyond the limit stay
     NaN.
   - After the split, no fill. Any NaN is a missing 5-minute price (e.g.
     during the June 2022 market suspension) and must not be filled silently.
5. **Completeness check.** Reports if the row count is not 105,120
   (105,408 for 2020 and 2024).
6. **Flag.** `price_flag = 1` where any region is NaN, otherwise 0. NaN
   prices are left in place.
7. **Quarterly TWAP** per region is printed for validation (see Verification).
8. Write `data/clean/nem_prices/YYYY_nem_prices.parquet`.

`pipeline/nem_prices/check_nem_parquet.py` is a diagnostic, not part of the
pipeline.

---

## Clean file format

`data/clean/nem_prices/YYYY_nem_prices.parquet`, snappy-compressed, one file
per calendar year, 105,120 rows (105,408 in leap years).

| Column | Type | Description |
|---|---|---|
| `datetime_aest` | `timestamp[us, tz=Etc/GMT-10]` | AEST (UTC+10, no DST), interval-end, complete 5-minute grid |
| `NSW1_rrp`, `QLD1_rrp`, `SA1_rrp`, `VIC1_rrp` | float | Regional Reference Price, $/MWh. NaN if missing. |
| `price_flag` | integer | 1 if any region's price is missing, otherwise 0 |

The timestamp convention and dtype are identical to the EDP clean files, so the
two join directly on `datetime_aest`. The join is many-to-one: every site in a
state takes its state's regional price (`ingest.py` maps `NSW` → `NSW1_rrp`
and so on).

`Etc/GMT-10` is UTC+10: the IANA `Etc/` zones invert the sign. A named
zone such as `Australia/Sydney` must not be used here, because it observes DST.

### Handling in the model

`ingest.py` raises if the file or the state's `_rrp` column or `price_flag`
is missing. If any interval has `price_flag = 1` it issues a warning and
passes the NaN prices through; downstream components must handle them.

---

## Verification

`B_Clean_NEM_Prices.py` prints quarterly TWAP (the simple mean over the
quarter's 5-minute intervals) per region, to compare with the TWAP in AEMO's
Quarterly Energy Dynamics reports. AEMO computes TWAP over 5-minute periods
even for pre-5MS quarters. Each 30-minute price is repeated across its six
slots, so the back-filled series gives the same TWAP as the 30-minute series
and should match QED directly. Differences under $1/MWh can come from
interval-boundary handling; larger ones indicate a data or processing error.
AER VWAP figures are demand-weighted and are not comparable.

AEMO's aggregated price and demand files (`PERIODTYPE = TRADE`) are an
independent source for spot checks of the NEMOSIS-derived years.

---

## Known limitations

- **Pre-5MS resolution.** For 2020 and January–September 2021 the price
  signal has 30-minute resolution on a 5-minute index. This is the correct
  economic representation, since no intra-period settlement variation existed,
  but battery value is not directly comparable across the boundary.
- **Cache refresh.** NEMOSIS does not re-download files already in
  `data/raw/nem_prices/`. To pick up AEMO republications, delete the relevant
  cache files (or use NEMOSIS `rebuild=True`) before rerunning
  `A_Download_NEM_Prices.py`. The direct-download path always fetches fresh.
- **Missing prices** are not imputed (see Handling in the model).
- **Current year.** Intervals after the latest available data are NaN with
  `price_flag = 1`.
