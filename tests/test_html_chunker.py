"""
tests/test_html_chunker.py — Unit tests for ingestion/html_chunker.py.

Uses SYNTHETIC markdown documents and a stubbed edgartools seam — no live SEC
API calls. We monkeypatch html_chunker._fetch_filing (the isolated fetch seam)
to yield a fake (filing, company) pair whose .markdown() returns a hand-written
document, so every test is deterministic and offline.

Key properties verified:
  - Each heading surface form (Three/Six/Nine Months Ended, Year Ended,
    Fiscal Year Ended) maps to the right fy_label
  - The label propagates down onto paragraphs beneath its heading
  - Propagation switches when a later heading resolves to a DIFFERENT period
  - A narrative subheading with no period does NOT clear the inherited label
  - Paragraphs above any dated heading get fy_label=None (implicit tier)
  - Empty / unparseable markdown yields an empty list, never raises
"""

from __future__ import annotations

import os
os.environ["DRY_RUN"] = "true"

import pytest

from ingestion import html_chunker
from ingestion.html_chunker import _extract_period_from_heading, fetch_html_chunks


# ---------------------------------------------------------------------------
# Stub edgartools seam: a fake filing whose .markdown() returns canned text.
# ---------------------------------------------------------------------------

class _FakeFiling:
    def __init__(self, markdown: str):
        self._markdown = markdown
        self.accession_number = "0000000000-26-000001"
        self.form = "10-Q"
        self.period_of_report = "2022-03-31"

    def markdown(self) -> str:
        return self._markdown


class _FakeCompany:
    name = "SYNTH CORP"
    cik = 1234567


@pytest.fixture
def patch_fetch(monkeypatch):
    """Return a helper that wires _fetch_filing to yield canned markdown."""
    def _install(markdown: str):
        def _fake_fetch(ticker, form_type, limit):
            yield _FakeFiling(markdown), _FakeCompany()
        monkeypatch.setattr(html_chunker, "_fetch_filing", _fake_fetch)
    return _install


# Long enough (> MIN_CHUNK_LENGTH) to survive the short-line filter.
_P1 = "Net revenue increased 12% compared to the prior-year period on strong demand."
_P2 = "Operating income declined versus the same quarter a year earlier on higher costs."


# ---------------------------------------------------------------------------
# _extract_period_from_heading: one assertion per surface form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("heading,expected", [
    ("## Three Months Ended March 31, 2022", "FY2022-Q1"),
    ("### Six Months Ended June 30, 2023", "FY2023-Q2"),
    ("#### Nine Months Ended September 30, 2023", "FY2023-Q3"),
    ("## Year Ended December 31, 2022", "FY2022"),
    ("## Fiscal Year Ended January 28, 2024", "FY2024"),
    ("#### For the Fiscal Quarter Ended March 28, 2026", "FY2026-Q1"),
])
def test_extract_period_each_pattern(heading, expected):
    assert _extract_period_from_heading(heading) == expected


@pytest.mark.parametrize("heading", [
    "#### Gross Margin",
    "# PART I — FINANCIAL INFORMATION",
    "## Management's Discussion and Analysis",
    "",
])
def test_extract_period_none_when_no_period(heading):
    assert _extract_period_from_heading(heading) is None


# ---------------------------------------------------------------------------
# Propagation: label flows down onto paragraphs beneath the heading
# ---------------------------------------------------------------------------

def test_label_propagates_to_paragraphs_below(patch_fetch):
    patch_fetch(f"## Three Months Ended March 31, 2022\n\n{_P1}\n\n{_P2}\n")
    chunks = fetch_html_chunks("SYN", form_type="10-Q")
    assert len(chunks) == 2
    assert all(c["fy_label"] == "FY2022-Q1" for c in chunks)
    assert chunks[0]["text"] == _P1
    assert chunks[0]["section_heading"] == "Three Months Ended March 31, 2022"


def test_label_switches_on_new_period_heading(patch_fetch):
    md = (
        f"## Three Months Ended March 31, 2022\n\n{_P1}\n\n"
        f"## Three Months Ended June 30, 2022\n\n{_P2}\n"
    )
    patch_fetch(md)
    chunks = fetch_html_chunks("SYN", form_type="10-Q")
    by_text = {c["text"]: c for c in chunks}
    assert by_text[_P1]["fy_label"] == "FY2022-Q1"
    assert by_text[_P2]["fy_label"] == "FY2022-Q2"  # propagation switched


def test_undated_subheading_does_not_clear_label(patch_fetch):
    # "Gross Margin" carries no period — paragraph under it must KEEP FY2022-Q1.
    md = (
        f"## Three Months Ended March 31, 2022\n\n"
        f"#### Gross Margin\n\n{_P1}\n"
    )
    patch_fetch(md)
    chunks = fetch_html_chunks("SYN", form_type="10-Q")
    assert len(chunks) == 1
    assert chunks[0]["fy_label"] == "FY2022-Q1"


def test_paragraphs_above_any_heading_are_implicit(patch_fetch):
    # No dated heading precedes _P1 → fy_label None (the hard implicit case).
    md = f"{_P1}\n\n## Three Months Ended March 31, 2022\n\n{_P2}\n"
    patch_fetch(md)
    chunks = fetch_html_chunks("SYN", form_type="10-Q")
    by_text = {c["text"]: c for c in chunks}
    assert by_text[_P1]["fy_label"] is None
    assert by_text[_P1]["section_heading"] is None
    assert by_text[_P2]["fy_label"] == "FY2022-Q1"


# ---------------------------------------------------------------------------
# Filtering: short lines / table borders / headings never become chunks
# ---------------------------------------------------------------------------

def test_short_lines_and_headings_skipped(patch_fetch):
    md = (
        "## Three Months Ended March 31, 2022\n\n"
        "Too short.\n\n"               # below MIN_CHUNK_LENGTH
        "| --- | --- |\n\n"            # table border row
        f"{_P1}\n"
    )
    patch_fetch(md)
    chunks = fetch_html_chunks("SYN", form_type="10-Q")
    texts = [c["text"] for c in chunks]
    assert texts == [_P1]


# ---------------------------------------------------------------------------
# Metadata + source tagging
# ---------------------------------------------------------------------------

def test_metadata_and_source_present(patch_fetch):
    patch_fetch(f"## Year Ended December 31, 2022\n\n{_P1}\n")
    chunk = fetch_html_chunks("SYN", form_type="10-K")[0]
    assert chunk["source"] == "html"
    assert chunk["ticker"] == "SYN"
    assert chunk["entity"] == "SYNTH CORP"
    assert chunk["cik"] == "0001234567"            # zero-padded to 10
    assert chunk["accession"] == "0000000000-26-000001"
    assert chunk["period_end"] == "2022-03-31"
    assert chunk["fy_label"] == "FY2022"


# ---------------------------------------------------------------------------
# Graceful handling of no content
# ---------------------------------------------------------------------------

def test_empty_markdown_returns_empty_list(patch_fetch):
    patch_fetch("")
    assert fetch_html_chunks("SYN", form_type="10-K") == []


def test_whitespace_only_markdown_returns_empty_list(patch_fetch):
    patch_fetch("   \n\n   \n")
    assert fetch_html_chunks("SYN", form_type="10-K") == []


def test_fetch_exception_returns_empty_list(monkeypatch):
    def _boom(ticker, form_type, limit):
        raise RuntimeError("edgartools exploded")
        yield  # pragma: no cover — makes this a generator
    monkeypatch.setattr(html_chunker, "_fetch_filing", _boom)
    assert fetch_html_chunks("SYN", form_type="10-K") == []
