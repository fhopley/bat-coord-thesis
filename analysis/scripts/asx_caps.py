"""
asx_caps.py

ASX Australian Electricity Base Load Calendar Quarter $300 Cap Futures:
traded prices and cash settlement. See docs/cap_hedge_spec.md.

Cash settlement price for a region and quarter:
    CSP = (C - 300·D) / E = (1/E) · Σ_i max(P_i - 300, 0)
where P_i are the region's spot prices in the quarter (5-minute from
Q4 2021, 30-minute before) and E is their count. A buyer of 1 MW receives
CSP · H_q, H_q being the hours in the quarter, i.e. Σ_i max(P_i - 300, 0) · Δt:
every interval above $300 pays, whatever the quarterly average.
"""

import pandas as pd

CAP_STRIKE = 300.0  # $/MWh

# ── ASX NSW $300 cap prices ($/MWh, at start of each quarter) ────────────────
# Historical: price observed at the start of the quarter (i.e. what hedgers
#   would have paid entering that quarter).
# Forward (Q3 2026 onwards): live ASX24 prices as at June 2026.
# Source: ASX24. NSW only — other states trade separately.
ASX_NSW_CAP = {
    # (year, quarter): $/MWh
    # ── Historical ────────────────────────────────────────────────────────────
    (2020, 1): 38.25,
    (2020, 2):  1.91,
    (2020, 3):  1.00,
    (2020, 4):  5.50,
    (2021, 1): 10.50,
    (2021, 2):  3.40,
    (2021, 3):  8.75,
    (2021, 4):  6.30,
    (2022, 1): 17.50,
    (2022, 2): 20.00,
    (2022, 3): 55.00,
    (2022, 4): 39.50,
    (2023, 1): 28.00,
    (2023, 2):  7.50,
    (2023, 3): 12.00,
    (2023, 4): 15.50,
    (2024, 1): 28.75,
    (2024, 2): 13.90,
    (2024, 3): 18.00,
    (2024, 4): 28.25,
    (2025, 1): 50.50,
    (2025, 2): 33.50,
    (2025, 3): 28.58,
    (2025, 4): 28.00,
    (2026, 1): 21.85,
    (2026, 2): 18.25,   # GNM2026 — start-of-quarter price (now near settlement)
    # ── Forward contracts (live ASX24 prices, June 2026) ──────────────────────
    (2026, 3):  8.25,   # GNU2026
    (2026, 4):  9.90,   # GNZ2026
    (2027, 1): 17.00,   # GNH2027
    (2027, 2): 11.00,   # GNM2027
}
ASX_QLD_CAP = {
    # (year, quarter): $/MWh
    # ── Historical ────────────────────────────────────────────────────────────
    (2020, 1): 9.25,
    (2020, 2):  0.9,
    (2020, 3):  0.6,
    (2020, 4):  2.8,
    (2021, 1): 6.5,
    (2021, 2):  5.5,
    (2021, 3):  16.75,
    (2021, 4):  23,
    (2022, 1): 51.25,
    (2022, 2): 47.5,
    (2022, 3): 76,
    (2022, 4): 28.5,
    (2023, 1): 33.95,
    (2023, 2):  28,
    (2023, 3): 5.5,
    (2023, 4): 10.5,
    (2024, 1): 45,
    (2024, 2): 11.5,
    (2024, 3): 15.17,
    (2024, 4): 25,
    (2025, 1): 35.5,
    (2025, 2): 20,
    (2025, 3): 7.5,
    (2025, 4): 12.8,
    (2026, 1): 7.75,
    (2026, 2): 7.45,   
}
ASX_SA_CAP = {
    # (year, quarter): $/MWh
    # ── Historical ────────────────────────────────────────────────────────────
    (2020, 1): 37.99,
    (2020, 2):  2.25,
    (2020, 3):  2.3,
    (2020, 4):  5.25,
    (2021, 1): 14,
    (2021, 2):  3.5,
    (2021, 3):  5.75,
    (2021, 4):  12,
    (2022, 1): 21.5,
    (2022, 2): 18,
    (2022, 3): 82,
    (2022, 4): 40.5,
    (2023, 1): 28,
    (2023, 2):  17.14,
    (2023, 3): 15.25,
    (2023, 4): 15.5,
    (2024, 1): 26.55,
    (2024, 2): 20,
    (2024, 3): 42,
    (2024, 4): 23,
    (2025, 1): 28,
    (2025, 2): 10,
    (2025, 3): 36.5,
    (2025, 4): 15.5,
    (2026, 1): 52,
    (2026, 2): 17.5,   
}
ASX_VIC_CAP = {
    # (year, quarter): $/MWh
    # ── Historical ────────────────────────────────────────────────────────────
    (2020, 1): 50,
    (2020, 2):  1,
    (2020, 3):  1.75,
    (2020, 4):  3.5,
    (2021, 1): 9.5,
    (2021, 2):  1,
    (2021, 3):  2.75,
    (2021, 4):  3.5,
    (2022, 1): 5.05,
    (2022, 2): 10.25,
    (2022, 3): 60,
    (2022, 4): 5,
    (2023, 1): 8.25,
    (2023, 2):  9,
    (2023, 3): 3,
    (2023, 4): 3.5,
    (2024, 1): 7.35,
    (2024, 2): 8,
    (2024, 3): 15.88,
    (2024, 4): 5.5,
    (2025, 1): 21.9,
    (2025, 2): 7.25,
    (2025, 3): 8.25,
    (2025, 4): 6.35,
    (2026, 1): 6.5,
    (2026, 2): 10.25,   
}
# Quarters from Q3 2026 onwards are forward contracts, not yet settled
FIRST_FORWARD_QUARTER = (2026, 3)

ASX_CAP_BY_REGION = {
    'NSW1': ASX_NSW_CAP,
    'QLD1': ASX_QLD_CAP,
    'SA1':  ASX_SA_CAP,
    'VIC1': ASX_VIC_CAP,
}

INTERVAL_MINUTES = 5


def contract_quarter(index: pd.DatetimeIndex) -> pd.MultiIndex:
    """(year, quarter) of each interval-end timestamp, taken from the
    interval start, so the interval ending 00:00 on 1 January falls in Q4
    of the previous year.
    """
    start = index - pd.Timedelta(minutes=INTERVAL_MINUTES)
    return pd.MultiIndex.from_arrays([start.year, start.quarter],
                                     names=['year', 'quarter'])


def quarter_hours(year: int, quarter: int) -> float:
    """Hours in a calendar quarter: the contract size in MWh per MW."""
    start = pd.Timestamp(year=year, month=3 * (quarter - 1) + 1, day=1)
    end = start + pd.DateOffset(months=3)
    return float((end - start).days * 24)


def cap_settlement(rrp: pd.Series) -> pd.DataFrame:
    """Realised $300 cap settlement per (year, quarter) present in rrp.

    rrp: 5-minute spot prices ($/MWh) on an interval-end DatetimeIndex.
    Pre-5MS 30-minute prices repeated across their six 5-minute slots give
    the same CSP as the 30-minute series.

    Returns a DataFrame indexed by (year, quarter) with columns
      csp    $/MWh, Σ max(P - CAP_STRIKE, 0) / (count of non-NaN prices)
      hours  calendar hours in the quarter (H_q)
      n_nan  count of NaN prices in the quarter
    """
    if not isinstance(rrp.index, pd.DatetimeIndex):
        raise TypeError("cap_settlement needs a DatetimeIndex (interval-end)")
    quarters = contract_quarter(rrp.index)
    frame = pd.DataFrame({
        'excess': (rrp - CAP_STRIKE).clip(lower=0).to_numpy(),
        'valid':  rrp.notna().to_numpy(),
    }, index=quarters)
    grouped = frame.groupby(level=['year', 'quarter'])
    n_valid = grouped['valid'].sum()
    n_total = grouped['valid'].size()
    out = pd.DataFrame({
        'csp':   grouped['excess'].sum() / n_valid.where(n_valid > 0),
        'n_nan': (n_total - n_valid).astype(int),
    })
    out['hours'] = [quarter_hours(y, q) for y, q in out.index]
    return out[['csp', 'hours', 'n_nan']]
