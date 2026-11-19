"""
Drop-in replacement for the three nemseer functions used by
predispatch_investigation.py. Depends on pandas only, not on nemseer.

Two independent reasons nemseer 1.0.7 cannot do this job as shipped:

1. Filenames. It constructs

       PUBLIC_DVD_{TABLE}_{yyyymm}010000.zip

   but AEMO now publishes recent months as

       PUBLIC_ARCHIVE#{TABLE}#FILE01#{yyyymm}010000.zip
       PUBLIC_ARCHIVE#{TABLE}#ALL#FILE01#{yyyymm}010000.zip   (PREDISP_ALL_DATA)

   and splits large tables across FILE01, FILE02, ... nemseer's table-discovery
   regex excludes '#', so it matches nothing and raises "Tables include: []"
   rather than a download error. This module parses the directory listing
   instead of constructing filenames, so it handles both naming eras and
   multi-part tables.

2. pandas 3. nemseer's clean_forecast_csv does `df.loc[:, col] = <datetimes>`
   against a string column, which pandas 3 rejects with
   "Invalid value for dtype 'str'". _clean below replicates that function's
   behaviour exactly, including the float downcasting and duplicate dropping,
   in a form pandas 3 accepts.

Used by A_Download_Predispatch (compile_data) as the stage A fetcher. It keeps
nemseer's function names and semantics so that results remain comparable with
a nemseer-based extraction.
"""

import io
import logging
import re
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

BASE = "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

DTFMT = "%Y/%m/%d %H:%M"
AEMO_DTFMT = "%Y/%m/%d %H:%M:%S"

# Copied verbatim from nemseer.data so behaviour matches what nemseer returns.
RUNTIME_COL = {"P5MIN": "RUN_DATETIME", "PREDISPATCH": "PREDISPATCH_RUN_DATETIME",
               "PDPASA": "RUN_DATETIME", "STPASA": "RUN_DATETIME",
               "MTPASA": "RUN_DATETIME"}
FORECASTED_COL = {"P5MIN": "INTERVAL_DATETIME", "PREDISPATCH": "DATETIME",
                  "PDPASA": "INTERVAL_DATETIME", "STPASA": "INTERVAL_DATETIME",
                  "MTPASA": "DAY"}
DATETIME_COLS = {
    "DATETIME", "EFFECTIVEDATE", "INTERVAL_DATETIME", "RUN_DATETIME",
    "AUTHORISEDDATE", "LASTCHANGED", "VERSION_DATETIME", "DAY",
    "PUBLISH_DATETIME", "LATEST_OFFER_DATETIME", "STARTDATE", "ENDDATE",
    "PERIOD_ENDING", "GENCONID_EFFECTIVEDATE", "BIDSETTLEMENTDATE",
    "SETTLEMENTDATE", "OFFERDATE",
}
CATEGORY_COLS = {"CONSTRAINTID", "INTERCONNECTORID", "DUID", "CONNECTIONPOINTID",
                 "PARTICIPANTID", "EXPORTGENCONID", "IMPORTGENCONID", "REGIONID"}

# PREDISPATCH tables served from PREDISP_ALL_DATA rather than DATA. The DATA
# copies carry a _D suffix and hold only the latest forecast per interval,
# which would silently give a store with no lead-time dimension.
PREDISP_ALL = ("CONSTRAINT", "INTERCONNECTORRES", "PRICE", "LOAD", "REGIONSUM")


class SourceMonthUnavailable(ValueError):
    """No files for a table in an archive month.

    A DATA CONDITION, not a defect: the month is genuinely not published, or
    the table is genuinely absent for it as with PREDISPATCHPRICE 2022-10.
    Distinguished from the other ValueErrors raised in this module -- a zip
    with no CSV inside, a malformed listing -- which ARE defects and must not
    be absorbed by the boundary-month tolerance in compile_data.

    Subclasses ValueError so existing callers catching ValueError still work.
    """


# ------------------------------------------------------------------ cleaning
def _parse_predispatch_seq_no(df: pd.DataFrame) -> pd.DataFrame:
    """Derive PREDISPATCH_RUN_DATETIME from PREDISPATCHSEQNO (YYYYMMDDPP).

    Nominal run time = date + 04:30 + (PP - 1) x 30 min, giving 04:30 through
    04:00 next day across PP = 1..48, matching the 04:00-to-04:00 trading day.
    Identical to nemseer's _parse_predispatch_seq_no.
    """
    seq = df["PREDISPATCHSEQNO"].astype("int64").astype(str)
    parsed = seq.str.extract(r"^([0-9]{8})([0-9]{2})$")
    day = pd.to_datetime(parsed[0], format="%Y%m%d")
    offset = ((parsed[1].astype(int) - 1) * pd.Timedelta(minutes=30)
              + pd.Timedelta(hours=4, minutes=30))
    df["PREDISPATCHSEQNO"] = seq
    df["PREDISPATCH_RUN_DATETIME"] = day + offset
    return df


def _clean(buf) -> pd.DataFrame:
    """Replicates nemseer.data_handlers.clean_forecast_csv, pandas 3 safe."""
    df = pd.read_csv(buf, skiprows=1)
    df = df.iloc[0:-1, :]                       # drop end-of-report line
    df = df.drop(df.columns[0:4], axis="columns")  # drop AEMO record metadata
    for col in DATETIME_COLS.intersection(df.columns):
        df[col] = pd.to_datetime(df[col], format=AEMO_DTFMT)
    for col in CATEGORY_COLS.intersection(df.columns):
        df[col] = df[col].astype("category")
    if "PREDISPATCHSEQNO" in df.columns:
        df = _parse_predispatch_seq_no(df)
    # nemseer downcasts float64 twice: to integer where integral, then float32.
    # This is why RRP normally arrives as float32 and why the dtype is
    # data-dependent per monthly file.
    # Integer downcast is skipped for columns containing nulls: casting NaN to
    # int raises "invalid value encountered in cast" and cannot be safe. RRP has
    # no nulls in any month checked, so prices are unaffected either way.
    for col in [c for c in df.columns if df.dtypes[c] == "float64"]:
        if df[col].notna().all():
            df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in [c for c in df.columns if df.dtypes[c] == "float64"]:
        df[col] = pd.to_numeric(df[col], downcast="float")
    n = len(df)
    df = df.drop_duplicates()
    if len(df) < n:
        logger.warning(f"Dropped {n - len(df)} exactly duplicated rows")
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ fetching
def _token(forecast_type: str, table: str) -> str:
    if forecast_type == "PREDISPATCH" and table != "MNSPBIDTRK":
        return f"PREDISPATCH{table}"
    return f"{forecast_type}_{table}"


def _folder(forecast_type: str, table: str) -> str:
    if forecast_type == "PREDISPATCH" and table in PREDISP_ALL:
        return "PREDISP_ALL_DATA"
    return "DATA"


def _dir_url(year: int, month: int, folder: str) -> str:
    return (f"{BASE}/{year}/MMSDM_{year}_{month:02d}/"
            f"MMSDM_Historical_Data_SQLLoader/{folder}/")


def _get(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _list_links(url: str) -> List[str]:
    try:
        html = _get(url, timeout=60).decode("utf8", "ignore")
    except Exception as exc:
        logger.warning(f"Listing failed for {url}: {exc!r}")
        return []
    out = []
    for h in re.findall(r'HREF="([^"]+\.zip)"', html, flags=re.I):
        out.append("https://nemweb.com.au" + h if h.startswith("/") else h)
    return sorted(set(out))


def _matches(url: str, token: str) -> bool:
    """True if a zip is a part-file of the given table, either naming era.

    Old:  PUBLIC_DVD_P5MIN_REGIONSOLUTION_202302010000.zip
    New:  PUBLIC_ARCHIVE#P5MIN_REGIONSOLUTION#FILE01#202602010000.zip
          PUBLIC_ARCHIVE#PREDISPATCHPRICE#ALL#FILE01#202602010000.zip

    The trailing delimiter matters: without it 'P5MIN_CONSTRAINT' would also
    match 'P5MIN_CONSTRAINTSOLUTION', and 'PREDISPATCHPRICE' would match the
    latest-only 'PREDISPATCHPRICE_D'.
    """
    name = url.rsplit("/", 1)[-1].replace("%23", "#")
    # Old-era files are sometimes split with a numeric part suffix, e.g.
    # PUBLIC_DVD_PREDISPATCHPRICE_1_202210010000.zip, mirroring the new era's
    # #FILE01#. The optional (?:_\d+)? covers that. It must NOT swallow the
    # _D latest-only variant, which is why the group is digits and not \w.
    pattern = (r"PUBLIC_(?:ARCHIVE#|DVD_)" + re.escape(token)
               + r"(?:#(?:ALL#)?FILE\d+#|_(?:\d+_)?)\d{12}\.zip$")
    return re.match(pattern, name, flags=re.I) is not None


def _read_zip(url: str) -> pd.DataFrame:
    blob = _get(url)
    frames = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        members = [n for n in z.namelist() if n.upper().endswith(".CSV")]
        if not members:
            raise ValueError(f"No CSV inside {url}")
        for m in members:
            with z.open(m) as fh:
                frames.append(_clean(fh))
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _month_frame(year: int, month: int, forecast_type: str, table: str,
                 raw_cache: Path) -> pd.DataFrame:
    token = _token(forecast_type, table)
    cached = raw_cache / f"{token}_{year}{month:02d}.parquet"
    if cached.exists():
        logger.info(f"Cache hit {cached.name}")
        return pd.read_parquet(cached)

    url = _dir_url(year, month, _folder(forecast_type, table))
    parts = [u for u in _list_links(url) if _matches(u, token)]
    if not parts:
        raise SourceMonthUnavailable(
            f"No files for {token} in {year}-{month:02d} at {url}. "
            f"Check the table name and that the month is published.")
    logger.info(f"{token} {year}-{month:02d}: {len(parts)} part file(s)")
    frames = []
    for i, part in enumerate(parts, 1):
        logger.info(f"  part {i}/{len(parts)}: {part.rsplit('/', 1)[-1][:70]}")
        frames.append(_read_zip(part))
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    raw_cache.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cached, index=False)
    logger.info(f"Cached {cached.name} "
                f"({cached.stat().st_size / 1e6:.1f} MB, {len(df):,} rows)")
    return df


def _months(start: datetime, end: datetime):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def compile_data(run_start: str, run_end: str, forecasted_start: str,
                 forecasted_end: str, forecast_type: str, tables,
                 raw_cache: str, processed_cache=None,
                 data_format: str = "df") -> Dict[str, pd.DataFrame]:
    """Signature-compatible stand-in for nemseer.compile_data.

    Applies the same two independent filters: rows kept where
    run_start <= RUNTIME_COL <= run_end and
    forecasted_start <= FORECASTED_COL <= forecasted_end. Unlike nemseer this
    performs no input validation, so an out-of-range window returns fewer rows
    rather than raising.

    Months are selected on the run window SHIFTED BACK by the table's
    partition offset (4h30m for PREDISPATCH, 5 min for P5MIN), because MMSDM
    partitions monthly files on a trading-day clock and a run at the very start
    of a calendar month lives in the previous month's file. See the comment in
    the loop below. An earlier month that is absent from the archive is skipped
    with a warning and recorded in the returned frame's
    `attrs['missing_source_months']`; a missing final month still raises.
    """
    if data_format != "df":
        raise NotImplementedError("Only data_format='df' is supported.")
    if isinstance(tables, str):
        tables = [tables]
    rs = datetime.strptime(run_start, DTFMT)
    re_ = datetime.strptime(run_end, DTFMT)
    fs = datetime.strptime(forecasted_start, DTFMT)
    fe = datetime.strptime(forecasted_end, DTFMT)
    cache = Path(raw_cache)

    out = {}
    for table in tables:
        # MMSDM partitions monthly files on a shifted clock, so a run at the
        # very start of a calendar month can live in the PREVIOUS month's file.
        # Verified against the 2026-02 files:
        #   PREDISPATCH  file M holds run times M-01 04:30 to M+1-01 04:00
        #                (partitioned on PREDISPATCHSEQNO trading date)
        #   P5MIN        file M holds run times M-01 00:05 to M+1-01 00:00
        #                (run time treated as interval-ending)
        # Shifting the month-selection window back by that offset maps each run
        # time to the file that actually contains it. Without this, nine
        # PREDISPATCH runs and one P5MIN run are silently lost per month
        # boundary, with no error raised.
        if forecast_type in ("PREDISPATCH", "PDPASA"):
            shift = pd.Timedelta(hours=4, minutes=30)
        elif forecast_type == "P5MIN":
            shift = pd.Timedelta(minutes=5)
        else:
            shift = pd.Timedelta(0)
        # A missing month is fatal EXCEPT when it is a strictly earlier month
        # in a multi-month span. Those earlier months are pulled in solely by
        # the shift above and contribute only the handful of runs at the start
        # of the requested window; the month carrying the bulk of the request
        # is always the last one. So this tolerates exactly the boundary-spill
        # file and nothing else.
        #
        # The case that forced it: PREDISPATCHPRICE 2022-10 is absent from the
        # archive in every form. Without this, requesting NOVEMBER 2022 also
        # fails, because November's first nine runs (00:00 to 04:00 on the 1st)
        # live in October's file. One unavailable month took a second, fully
        # available month with it. Nine runs are genuinely unreachable; 1,431
        # were sitting there.
        #
        # Requesting October itself still raises, since the missing month is
        # then the LAST in the span. So does a month AEMO has not published
        # yet, which is always last in its own span. Both keep the loud signal
        # they should have.
        months = list(_months(rs - shift, re_ - shift))
        frames, missing_months = [], []
        for i, (y, m) in enumerate(months):
            try:
                frames.append(_month_frame(y, m, forecast_type, table, cache))
            except SourceMonthUnavailable:
                if not (len(months) > 1 and i < len(months) - 1):
                    raise
                missing_months.append(f"{y}-{m:02d}")
                logger.warning(
                    f"{_token(forecast_type, table)}: source month "
                    f"{y}-{m:02d} is absent. It supplies only the runs at the "
                    "start of the requested window, so the window is served "
                    "without them and is short at its leading edge.")
        if not frames:
            raise SourceMonthUnavailable(
                f"No source month available for "
                f"{_token(forecast_type, table)} across "
                f"{months[0][0]}-{months[0][1]:02d} to "
                f"{months[-1][0]}-{months[-1][1]:02d}.")
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        rcol, fcol = RUNTIME_COL[forecast_type], FORECASTED_COL[forecast_type]
        if rcol in df.columns:
            df = df[(df[rcol] >= rs) & (df[rcol] <= re_)]
        if fcol in df.columns:
            df = df[(df[fcol] >= fs) & (df[fcol] <= fe)]
        out[table] = df.reset_index(drop=True)
        # Recorded rather than only logged, so stage A can report a short
        # leading edge at acquisition time instead of leaving it to be
        # rediscovered as an unexplained PARTIAL in stage C.
        out[table].attrs['missing_source_months'] = tuple(missing_months)
    return out


def get_tables(year: int, month: int, forecast_type: str,
               actual: bool = False) -> List[str]:
    """Tables present for a forecast type in a month, both naming eras."""
    found = set()
    folders = ["DATA"] + (["PREDISP_ALL_DATA"]
                          if forecast_type == "PREDISPATCH" else [])
    for folder in folders:
        for url in _list_links(_dir_url(year, month, folder)):
            name = url.rsplit("/", 1)[-1].replace("%23", "#")
            m = re.match(r"PUBLIC_(?:ARCHIVE#|DVD_)(.+?)"
                         r"(?:#(?:ALL#)?FILE\d+#|_)\d{12}\.zip$",
                         name, flags=re.I)
            if not m:
                continue
            tok = m.group(1).upper()
            if forecast_type == "PREDISPATCH":
                if tok.startswith("PREDISPATCH"):
                    found.add(tok[len("PREDISPATCH"):].strip("_"))
            elif tok.startswith(forecast_type + "_"):
                found.add(tok[len(forecast_type) + 1:])
    if not actual:
        found = {re.sub(r"\d+$", "", t) if t.startswith("CONSTRAINTSOLUTION")
                 else t for t in found}
    return sorted(t for t in found if t)


def get_data_daterange() -> Dict[int, List[int]]:
    """Years and months in the MMSDM archive, probing 2021 onward only.

    One request per year rather than nemseer's scrape of every year directory
    back to 2009, which is what made Q4 appear to hang.
    """
    out = {}
    for year in range(2021, datetime.now().year + 2):
        try:
            html = _get(f"{BASE}/{year}/", timeout=60).decode("utf8", "ignore")
        except Exception:
            continue
        months = sorted({int(m) for m in
                         re.findall(rf"MMSDM_{year}_(\d{{2}})", html)})
        if months:
            out[year] = months
    return out
