"""
retrieval/query_parser.py — Extract ticker and fiscal period from a natural-language question.

Returns a filter dict suitable for Chroma's where= clause in the period-filtered retriever.
Regex-only — no LLM calls in the retrieval path. Keeps the retrieval path fast,
interpretable, and consistent with the Stage 1 tagger philosophy.

Graceful degradation:
    ticker + period  →  {"ticker": "AAPL", "fiscal_period": "FY2024-Q1"}
    period only      →  {"fiscal_period": "FY2024-Q1"}
    ticker only      →  {"ticker": "AAPL"}
    neither          →  {}    caller falls back to pure ANN (the baseline)

FILE MAP
  L001–L028  Module docstring + file map
  L030–L037  Imports
  L039–L092  CONFIG — ticker lookup tables (add aliases here)
  L094–L126  Extraction helpers — _extract_ticker(), _extract_period()
  L128–L155  Public API — parse_query()

Usage
-----
    from retrieval.query_parser import parse_query
    parse_query("What was Apple's net income in Q1 2024?")
    # → {"ticker": "AAPL", "fiscal_period": "FY2024-Q1"}
"""

from __future__ import annotations

import re

from evaluation.tagger import regex_tag
from retrieval.financebench_tickers import FINANCEBENCH_COMPANIES


# ---------------------------------------------------------------------------
# Ticker lookup — built once at import time
# ===========================================================================
# CONFIG — ticker lookup tables
# ===========================================================================

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


# All known tickers for direct mention detection (e.g. "AAPL" written in the question).
# CHANGE ME: add any new tickers here when expanding the index.
_KNOWN_TICKERS: frozenset[str] = frozenset(
    {c.ticker for c in FINANCEBENCH_COMPANIES if c.ticker}
    | {"AAPL", "GOOG", "NVDA"}  # Stage 1 tickers not in FinanceBench
)

# Auto-generated from registry (full company name → ticker).
# Extended with manual aliases below for nicknames and abbreviations.
_NAME_TO_TICKER: dict[str, str] = {
    _normalize(c.company_name): c.ticker
    for c in FINANCEBENCH_COMPANIES
    if c.ticker
}

# Manual aliases — nicknames and abbreviations not derivable from company_name.
# CHANGE ME: add new aliases here when questions use unfamiliar shorthand.
_NAME_TO_TICKER.update({
    # Stage 1 tickers not in FinanceBench registry
    "apple":      "AAPL",
    "google":     "GOOG",
    "alphabet":   "GOOG",
    "nvidia":     "NVDA",
    # Common short names and brand names
    "pepsi":      "PEP",
    "coke":       "KO",
    "coca cola":  "KO",    # "Coca-Cola" normalises to "coca cola" — coke doesn't
    "j j":        "JNJ",   # "J&J" normalises to "j j" after stripping &
    "jnj":        "JNJ",   # "JnJ" capitalisation found in FinanceBench questions
    "amex":       "AXP",
    "mgm":        "MGM",
    "cvs":        "CVS",
    "ulta":       "ULTA",
    "lockheed":   "LMT",
    "square":     "XYZ",   # Block Inc. formerly known as Square
    "kraft":      "KHC",
    "heinz":      "KHC",
    "walmart":    "WMT",
    "wal mart":   "WMT",
    "jp morgan":  "JPM",
    "chase":      "JPM",
    "aes":        "AES",
})

# Sort by length descending so longer matches are tried first.
# Prevents "american" matching before "american express" or "american water works".
_NAME_LOOKUP: list[tuple[str, str]] = sorted(
    _NAME_TO_TICKER.items(), key=lambda x: len(x[0]), reverse=True
)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_ticker(question: str) -> str | None:
    """
    Two-pass ticker extraction:
      1. Direct uppercase ticker mention in the question text
      2. Company name substring match against the normalized question
    """
    # Pass 1 — direct ticker mention ("AAPL", "MSFT", etc.)
    for match in re.finditer(r"\b([A-Z]{2,5})\b", question):
        if match.group(1) in _KNOWN_TICKERS:
            return match.group(1)

    # Pass 2 — company name substring
    q_norm = _normalize(question)
    for name, ticker in _NAME_LOOKUP:
        if name in q_norm:
            return ticker

    return None


def _extract_period(question: str) -> str | None:
    """Delegate to the Stage 1 regex tagger. Returns fy_label or None."""
    return regex_tag(question)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_query(question: str) -> dict:
    """
    Parse a natural-language financial question into a Chroma where= filter dict.

    Parameters
    ----------
    question:
        Natural-language question, e.g. "What was Apple's net income in Q1 2024?"

    Returns
    -------
    Dict with zero, one, or two keys:
        "ticker"        — exchange ticker string, e.g. "AAPL"
        "fiscal_period" — fy_label string, e.g. "FY2024-Q1"
    Empty dict means no filter could be extracted — caller should use pure ANN.
    """
    ticker = _extract_ticker(question)
    period = _extract_period(question)

    result: dict = {}
    if ticker:
        result["ticker"] = ticker
    if period:
        result["fiscal_period"] = period
    return result
