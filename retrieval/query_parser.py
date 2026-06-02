"""
retrieval/query_parser.py — Extract ticker and fiscal period from a natural-language question.

Returns a filter dict suitable for Chroma's where= clause in the period-filtered retriever.
Regex-only — no LLM calls in the retrieval path.

Graceful degradation:
    ticker + period  →  {"ticker": "AAPL", "fiscal_period": "FY2024-Q1"}
    period only      →  {"fiscal_period": "FY2024-Q1"}
    ticker only      →  {"ticker": "AAPL"}
    neither          →  {}    caller falls back to pure ANN (the baseline)

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
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


# All known tickers for direct mention detection
_KNOWN_TICKERS: frozenset[str] = frozenset(
    {c.ticker for c in FINANCEBENCH_COMPANIES if c.ticker}
    | {"AAPL", "GOOG", "NVDA"}  # Stage 1 tickers not in FinanceBench
)

# name fragment → ticker: auto-generated from registry, extended with manual aliases
_NAME_TO_TICKER: dict[str, str] = {
    _normalize(c.company_name): c.ticker
    for c in FINANCEBENCH_COMPANIES
    if c.ticker
}

_NAME_TO_TICKER.update({
    # Stage 1 tickers
    "apple":      "AAPL",
    "google":     "GOOG",
    "alphabet":   "GOOG",
    "nvidia":     "NVDA",
    # Short names and common variants
    "pepsi":      "PEP",
    "coke":       "KO",
    "coca cola":  "KO",    # "Coca-Cola" normalises to "coca cola" already, but "coke" doesn't
    "j j":        "JNJ",   # "J&J" normalises to "j j"
    "jnj":        "JNJ",   # "JnJ" capitalisation used in FinanceBench questions
    "amex":       "AXP",
    "mgm":        "MGM",
    "cvs":        "CVS",
    "ulta":       "ULTA",
    "lockheed":   "LMT",
    "square":     "XYZ",   # Block formerly known as Square
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
