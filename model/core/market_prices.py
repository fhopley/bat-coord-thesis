"""NEM market price limits by financial year.

The Market Price Cap (MPC) is the maximum spot price in any dispatch or
trading interval. It is reset every 1 July, CPI-adjusted by the AEMC by 28
February each year, so it is a per-financial-year fact and not a constant.

The Market Floor Price (MFP) has been -$1,000/MWh throughout the period this
model covers. The AEMC's December 2023 rule determination explicitly kept it
at that level, noting the Reliability Panel would reconsider it in the 2026
Reliability Standard and Settings review. If that review moves it, this module
is where the change lands.

Used for two things, both of them checks rather than transformations:

  1. Range validation when ingesting forecast data (predispatch_data_spec.md).
     This is why the table is per-year: a price outside the limits for its own
     financial year is a data defect, and using a single global bound would
     miss most of them.
  2. Bounding the price-ordering proof in prices.validate_price_ordering,
     which needs the widest bounds across all years rather than a specific
     year's.

Nothing here clips or modifies a price.

Source for the cap series: WattClarity's Market Price Cap glossary page, which
tracks the AEMC determinations,
https://wattclarity.com.au/other-resources/glossary/market-price-cap/

Only DETERMINED values are listed. Scheduled future values from the AEMC's
2023 rule change are deliberately excluded: the actual determination has come
in above the schedule in successive years (2024-25 and 2025-26 both did), so
treating a scheduled value as fact would silently validate against the wrong
number. Extending the table is a one-line change once a determination is made.
"""
from __future__ import annotations

import pandas as pd


# Financial year START year -> Market Price Cap, $/MWh.
# Key 2025 means the year running 1 July 2025 to 30 June 2026.
MARKET_PRICE_CAP_BY_FY: dict[int, float] = {
    2012: 12_900.0,
    2013: 13_100.0,
    2014: 13_500.0,
    2015: 13_800.0,
    2016: 14_000.0,
    2017: 14_200.0,
    2018: 14_500.0,
    2019: 14_700.0,
    2020: 15_000.0,
    2021: 15_100.0,
    2022: 15_500.0,
    2023: 16_600.0,
    2024: 17_500.0,
    2025: 20_300.0,
    2026: 23_200.0,
}

# Market Floor Price, $/MWh. Unchanged across every year in the table above.
MARKET_PRICE_FLOOR_MWH: float = -1_000.0


def financial_year_start(ts: pd.Timestamp) -> int:
    """Return the start year of the Australian financial year containing ts.

    The financial year runs 1 July to 30 June, so 2025-09-01 and 2026-03-01
    both belong to financial year 2025.

    ts is interpreted in whatever timezone it carries. Market price limits
    change at 1 July in market time, and the model index is market time
    (Etc/GMT-10), so no conversion is applied here.
    """
    return ts.year if ts.month >= 7 else ts.year - 1


def market_price_cap(ts: pd.Timestamp) -> float:
    """Market Price Cap in $/MWh applying at ts.

    Raises:
        KeyError: if the financial year is not in the table. Deliberate: a
            missing year means either data outside the modelled period or a
            determination that has not been recorded yet, and silently
            falling back to the nearest known value would validate against
            the wrong number.
    """
    fy = financial_year_start(ts)
    try:
        return MARKET_PRICE_CAP_BY_FY[fy]
    except KeyError:
        known = sorted(MARKET_PRICE_CAP_BY_FY)
        raise KeyError(
            f"No Market Price Cap recorded for financial year {fy}-{fy + 1} "
            f"(timestamp {ts}). Known years: {known[0]} to {known[-1]}. "
            "Add the AEMC determination to MARKET_PRICE_CAP_BY_FY; scheduled "
            "values are not sufficient, the CPI-adjusted determination has "
            "exceeded the schedule in successive years."
        ) from None


def market_price_floor(ts: pd.Timestamp | None = None) -> float:
    """Market Floor Price in $/MWh applying at ts.

    Constant across the modelled period. Takes ts anyway so that callers do
    not need changing if the Reliability Panel moves it.
    """
    return MARKET_PRICE_FLOOR_MWH


def widest_market_price_cap() -> float:
    """Highest cap across all recorded years.

    For checks that must hold regardless of when a price occurred, such as the
    price-ordering proof, which is a property of a tariff pair rather than of
    a particular interval.
    """
    return max(MARKET_PRICE_CAP_BY_FY.values())
