"""
tests/test_query_parser.py — Unit tests for the query parser.

No live EDGAR or Chroma calls — pure string logic.
Covers all four degradation cases: ticker+period, period only, ticker only, neither.
"""

from __future__ import annotations

import os
os.environ["DRY_RUN"] = "true"

import pytest
from retrieval.query_parser import parse_query


# ---------------------------------------------------------------------------
# Full filter — ticker + period
# ---------------------------------------------------------------------------

def test_ticker_and_period_canonical():
    r = parse_query("What was Apple's net income in Q1 2024?")
    assert r == {"ticker": "AAPL", "fiscal_period": "FY2024-Q1"}


def test_ticker_and_period_direct_ticker_mention():
    r = parse_query("What was MSFT revenue in Q2 FY2024?")
    assert r == {"ticker": "MSFT", "fiscal_period": "FY2024-Q2"}


def test_ticker_and_period_annual():
    r = parse_query("PepsiCo revenue for fiscal year 2022?")
    assert r == {"ticker": "PEP", "fiscal_period": "FY2022"}


def test_ticker_and_period_variant_form():
    r = parse_query("What were Coca-Cola's earnings in FY2022?")
    assert r == {"ticker": "KO", "fiscal_period": "FY2022"}


def test_ticker_and_period_short_alias():
    r = parse_query("What was MGM's Q2 2023 operating income?")
    assert r == {"ticker": "MGM", "fiscal_period": "FY2023-Q2"}


def test_ticker_and_period_jnj_ampersand():
    r = parse_query("J&J revenue in Q1 2023")
    assert r == {"ticker": "JNJ", "fiscal_period": "FY2023-Q1"}


def test_ticker_and_period_name_with_spaces():
    r = parse_query("What was Best Buy's net income in Q3 2022?")
    assert r == {"ticker": "BBY", "fiscal_period": "FY2022-Q3"}


def test_ticker_and_period_google_alias():
    r = parse_query("Google revenue for Q4 2023")
    assert r == {"ticker": "GOOG", "fiscal_period": "FY2023-Q4"}


# ---------------------------------------------------------------------------
# Period only — no company mentioned
# ---------------------------------------------------------------------------

def test_period_only_quarterly():
    r = parse_query("What happened in Q3 2023?")
    assert r == {"fiscal_period": "FY2023-Q3"}


def test_period_only_annual():
    r = parse_query("Show me full year 2022 results.")
    assert r == {"fiscal_period": "FY2022"}


# ---------------------------------------------------------------------------
# Ticker only — no period mentioned
# ---------------------------------------------------------------------------

def test_ticker_only_company_name():
    r = parse_query("Tell me about Amazon's business.")
    assert r == {"ticker": "AMZN"}


def test_ticker_only_direct_mention():
    r = parse_query("What does NVDA do?")
    assert r == {"ticker": "NVDA"}


def test_ticker_only_alias():
    r = parse_query("What is Walmart's strategy?")
    assert r == {"ticker": "WMT"}


# ---------------------------------------------------------------------------
# No filter — neither extractable
# ---------------------------------------------------------------------------

def test_no_filter_generic():
    assert parse_query("What is the weather today?") == {}


def test_no_filter_vague_financial():
    assert parse_query("How did the company perform last year?") == {}


# ---------------------------------------------------------------------------
# Return type contract
# ---------------------------------------------------------------------------

def test_returns_dict():
    assert isinstance(parse_query("Apple Q1 2024"), dict)


def test_only_valid_keys():
    r = parse_query("Apple Q1 2024")
    assert set(r.keys()) <= {"ticker", "fiscal_period"}
