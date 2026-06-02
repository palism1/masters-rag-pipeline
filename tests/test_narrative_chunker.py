"""
tests/test_narrative_chunker.py — Unit tests for narrative_chunker.py.

Uses synthetic XbrlFact objects — no live SEC API calls.
Key properties verified:
  - No fiscal label (FY2024-Q1) appears in the chunk text
  - Annual facts use "fiscal year ended" phrasing
  - Quarterly facts use "three months ended" phrasing
  - Instant facts use "as of" phrasing
  - fy_label metadata is preserved
  - Month names are correct (December not December30)
"""

from __future__ import annotations

import os
os.environ["DRY_RUN"] = "true"

from dataclasses import dataclass
from typing import Optional

import pytest

from narrative_chunker import facts_to_narrative_chunks


# ---------------------------------------------------------------------------
# Minimal XbrlFact stand-in (only fields narrative_chunker touches)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Fact:
    concept: str
    taxonomy: str = "us-gaap"
    entity: str = "APPLE INC"
    cik: str = "0000320193"
    period_start: Optional[str] = None
    period_end: str = "2023-12-30"
    period_type: str = "duration"
    unit: str = "USD"
    value: float = 119_575_000_000.0
    scale: int = 0
    form_type: str = "10-Q"
    accession: str = "0000320193-24-000006"
    fiscal_year: Optional[int] = 2024
    fiscal_period: Optional[str] = "Q1"

    @property
    def fy_label(self):
        if not (self.fiscal_year and self.fiscal_period):
            return None
        if self.fiscal_period == "FY":
            return f"FY{self.fiscal_year}"
        return f"FY{self.fiscal_year}-{self.fiscal_period}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quarterly() -> _Fact:
    return _Fact(concept="NetIncomeLoss", period_start="2023-10-01",
                 period_end="2023-12-30", period_type="duration",
                 fiscal_year=2024, fiscal_period="Q1")

def _annual() -> _Fact:
    return _Fact(concept="NetIncomeLoss", period_start="2022-10-01",
                 period_end="2023-09-30", period_type="duration",
                 fiscal_year=2023, fiscal_period="FY")

def _instant() -> _Fact:
    return _Fact(concept="Assets", period_start=None,
                 period_end="2023-12-30", period_type="instant",
                 fiscal_year=2024, fiscal_period="Q1")


# ---------------------------------------------------------------------------
# Core contract: no FY label in text
# ---------------------------------------------------------------------------

def test_quarterly_no_fy_label_in_text():
    chunks = facts_to_narrative_chunks([_quarterly()])
    text = chunks[0]["text"]
    assert "FY2024" not in text, f"Fiscal label leaked into narrative text: {text!r}"
    assert "Q1" not in text, f"Quarter label leaked into narrative text: {text!r}"


def test_annual_no_fy_label_in_text():
    chunks = facts_to_narrative_chunks([_annual()])
    text = chunks[0]["text"]
    assert "FY2023" not in text
    assert "-FY" not in text


# ---------------------------------------------------------------------------
# Phrasing by period type
# ---------------------------------------------------------------------------

def test_quarterly_uses_three_months_phrasing():
    text = facts_to_narrative_chunks([_quarterly()])[0]["text"]
    assert "three months ended" in text.lower()


def test_annual_uses_fiscal_year_phrasing():
    text = facts_to_narrative_chunks([_annual()])[0]["text"]
    assert "fiscal year ended" in text.lower()


def test_instant_uses_as_of_phrasing():
    text = facts_to_narrative_chunks([_instant()])[0]["text"]
    assert "as of" in text.lower()


# ---------------------------------------------------------------------------
# Date formatting
# ---------------------------------------------------------------------------

def test_quarterly_month_name_correct():
    # period_end 2023-12-30 → "December 30, 2023"
    text = facts_to_narrative_chunks([_quarterly()])[0]["text"]
    assert "December 30, 2023" in text


def test_annual_month_name_correct():
    # period_end 2023-09-30 → "September 30, 2023"
    text = facts_to_narrative_chunks([_annual()])[0]["text"]
    assert "September 30, 2023" in text


# ---------------------------------------------------------------------------
# Metadata preserved
# ---------------------------------------------------------------------------

def test_fy_label_in_metadata():
    chunk = facts_to_narrative_chunks([_quarterly()])[0]
    assert chunk["fy_label"] == "FY2024-Q1"


def test_annual_fy_label_in_metadata():
    chunk = facts_to_narrative_chunks([_annual()])[0]
    assert chunk["fy_label"] == "FY2023"


def test_period_end_preserved():
    chunk = facts_to_narrative_chunks([_quarterly()])[0]
    assert chunk["period_end"] == "2023-12-30"


def test_value_preserved():
    chunk = facts_to_narrative_chunks([_quarterly()])[0]
    assert chunk["value"] == 119_575_000_000.0


def test_empty_input():
    assert facts_to_narrative_chunks([]) == []


# ---------------------------------------------------------------------------
# Entity name title-cased (SEC stores in ALL CAPS)
# ---------------------------------------------------------------------------

def test_entity_title_cased():
    text = facts_to_narrative_chunks([_quarterly()])[0]["text"]
    assert "Apple Inc" in text
    assert "APPLE INC" not in text
