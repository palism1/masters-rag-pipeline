"""
financebench_tickers.py — Ticker/CIK registry for all 32 FinanceBench companies.

Each entry:
    company_name  : exactly as it appears in the FinanceBench dataset
    ticker        : pass to edgar.Company(ticker) — or None if CIK-only
    cik           : pass to edgar.Company(cik) when ticker lookup fails
    questions     : number of questions in the 150-question public split
    has_10q       : True if any FinanceBench questions come from 10-Q filings
    notes         : any caveats

Usage
-----
    from financebench_tickers import FINANCEBENCH_COMPANIES, ticker_or_cik
    import edgar

    for entry in FINANCEBENCH_COMPANIES:
        company = edgar.Company(ticker_or_cik(entry))
        ...
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FBCompany:
    company_name: str
    ticker: Optional[str]   # None means use cik
    cik: int
    questions: int          # count in 150-question public split
    has_10q: bool
    notes: str = ""


def ticker_or_cik(entry: FBCompany) -> str | int:
    """Return the best lookup key for edgar.Company()."""
    return entry.ticker if entry.ticker else entry.cik


# ---------------------------------------------------------------------------
# Registry — all 32 FinanceBench companies, sorted by question count desc
# ---------------------------------------------------------------------------
FINANCEBENCH_COMPANIES: list[FBCompany] = [
    FBCompany("PepsiCo",               "PEP",   77476,    11, has_10q=False),
    FBCompany("Amcor",                 "AMCR",  1748790,   9, has_10q=True),
    FBCompany("Johnson & Johnson",     "JNJ",   200406,    9, has_10q=False),
    FBCompany("3M",                    "MMM",   66740,     8, has_10q=True),
    FBCompany("AMD",                   "AMD",   2488,      8, has_10q=False),
    FBCompany("Best Buy",              "BBY",   764478,    8, has_10q=True),
    FBCompany("Boeing",                "BA",    12927,     8, has_10q=False),
    FBCompany("American Express",      "AXP",   4962,      7, has_10q=False),
    FBCompany("MGM Resorts",           "MGM",   789570,    7, has_10q=True),
    FBCompany("Pfizer",                "PFE",   78003,     6, has_10q=True),
    FBCompany("Ulta Beauty",           "ULTA",  1403568,   6, has_10q=False),
    FBCompany("Adobe",                 "ADBE",  796343,    5, has_10q=False),
    FBCompany("JPMorgan",              "JPM",   19617,     5, has_10q=True),
    FBCompany("Verizon",               "VZ",    732712,    5, has_10q=False),
    FBCompany("CVS Health",            "CVS",   64803,     4, has_10q=False),
    FBCompany("Corning",               "GLW",   24741,     4, has_10q=False),
    FBCompany("General Mills",         "GIS",   40704,     4, has_10q=False),
    FBCompany("Nike",                  "NKE",   320187,    4, has_10q=False),
    FBCompany("AES Corporation",       "AES",   874761,    3, has_10q=False),
    FBCompany("Amazon",                "AMZN",  1018724,   3, has_10q=False),
    FBCompany("American Water Works",  "AWK",   1410636,   3, has_10q=False),
    FBCompany("Block",                 "XYZ",   1512673,   3, has_10q=False,
              notes="Formerly Square; ticker changed from SQ to XYZ"),
    FBCompany("Coca-Cola",             "KO",    21344,     3, has_10q=False),
    FBCompany("Lockheed Martin",       "LMT",   936468,    3, has_10q=False),
    FBCompany("Activision Blizzard",   None,    718877,    2, has_10q=False,
              notes="Acquired by Microsoft Oct 2023; latest 10-K filed Feb 2023. Historical data only."),
    FBCompany("Foot Locker",           None,    850209,    2, has_10q=False,
              notes="Ticker 'FL' not resolvable by edgartools; use CIK 850209 directly."),
    FBCompany("Microsoft",             "MSFT",  789019,    2, has_10q=False),
    FBCompany("Netflix",               "NFLX",  1065280,   2, has_10q=False),
    FBCompany("Costco",                "COST",  909832,    1, has_10q=False),
    FBCompany("Kraft Heinz",           "KHC",   1637459,   1, has_10q=False),
    FBCompany("Paypal",                "PYPL",  1633917,   1, has_10q=False),
    FBCompany("Walmart",               "WMT",   104169,    3, has_10q=False),
]

# ---------------------------------------------------------------------------
# Convenience subsets
# ---------------------------------------------------------------------------

# Companies with 10-Q questions in FinanceBench — highest value for period-sensitivity testing
HAS_10Q = [c for c in FINANCEBENCH_COMPANIES if c.has_10q]

# Original four tickers from Stage 1
STAGE1_TICKERS = ["AAPL", "MSFT", "GOOG", "NVDA"]

# All resolvable tickers (excludes the 2 CIK-only entries if you want ticker-string interface)
ALL_TICKERS = [c.ticker for c in FINANCEBENCH_COMPANIES if c.ticker]
