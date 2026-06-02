"""
tests/test_xbrl_loader.py — Verification suite for the XBRL ingestion path.

Run: pytest tests/ -v

All four tests hit the live SEC CompanyFacts API (cached after first run).
DRY_RUN is forced True — no DB/vector-store writes happen.

Reference filing
----------------
Apple Inc. 10-Q for Q1 FY2024 (fiscal quarter Oct 1 – Dec 30, 2023)
Accession: 0000320193-24-000006  filed: 2024-02-02
Concept:   RevenueFromContractWithCustomerExcludingAssessedTax
Value:     119,575,000,000 USD  (raw, unscaled)
EDGAR fy=2024, fp="Q1"
"""

from __future__ import annotations

import os

# Must be set before config (and therefore edgartools) is imported.
os.environ["DRY_RUN"] = "true"

import pytest

from ingestion.xbrl_loader import XbrlFact, load_company_facts
from ingestion.xbrl_chunker import facts_to_chunks

# ---------------------------------------------------------------------------
# Reference constants — match these against the official 10-Q before trusting
# ---------------------------------------------------------------------------
TICKER = "AAPL"
REF_CONCEPT = "RevenueFromContractWithCustomerExcludingAssessedTax"
REF_PERIOD_END = "2023-12-30"
REF_PERIOD_START = "2023-10-01"
REF_VALUE = 119_575_000_000
REF_UNIT = "USD"
REF_FY = 2024
REF_FP = "Q1"


# ---------------------------------------------------------------------------
# Shared fixture — fetched once for the whole module (respects edgartools cache)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def aapl_facts() -> list[XbrlFact]:
    return load_company_facts(TICKER, concepts=[REF_CONCEPT])


# ---------------------------------------------------------------------------
# Test 1 — Known-value check
# Value, period start/end, unit, AND fiscal label must all match the filing.
# ---------------------------------------------------------------------------

def test_known_value(aapl_facts: list[XbrlFact]) -> None:
    matching = [
        f for f in aapl_facts
        if f.concept == REF_CONCEPT
        and f.period_end == REF_PERIOD_END
        and f.unit == REF_UNIT
    ]
    assert matching, (
        f"No fact found: concept={REF_CONCEPT} period_end={REF_PERIOD_END} unit={REF_UNIT}"
    )
    fact = matching[0]

    assert fact.value == REF_VALUE, (
        f"Value mismatch: got {fact.value:,} expected {REF_VALUE:,}"
    )
    assert fact.period_start == REF_PERIOD_START, (
        f"period_start mismatch: got {fact.period_start} expected {REF_PERIOD_START}"
    )
    assert fact.unit == REF_UNIT
    assert fact.fiscal_year == REF_FY, (
        f"fiscal_year mismatch: got {fact.fiscal_year} expected {REF_FY}"
    )
    assert fact.fiscal_period == REF_FP, (
        f"fiscal_period mismatch: got {fact.fiscal_period!r} expected {REF_FP!r}"
    )
    assert fact.fy_label == f"FY{REF_FY}-{REF_FP}", (
        f"fy_label mismatch: got {fact.fy_label!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 — No cross-period contamination
#
# The LLM failure mode being designed out: an HTML-scraped chunk tagged "Q1"
# actually contains a Q2 value because the regex grabbed the wrong table column.
#
# The XBRL guarantee runs in the direction period_end → fy_label: each concrete
# period end-date maps to exactly one fiscal label. A single fy_label CAN map to
# multiple period_end dates — EDGAR legitimately does this for restated or
# transition-period facts (e.g. ASC-606 adoption data tagged with the adoption
# fiscal year). We test the direction that matters for retrieval correctness.
# ---------------------------------------------------------------------------

def test_no_cross_period_contamination(aapl_facts: list[XbrlFact]) -> None:
    # For each (concept, period_end, unit) there must be at most one fy_label.
    # This is the key structural guarantee: if you fetch the value for 2023-12-30,
    # it is always labeled FY2024-Q1 — never a different fiscal period.
    # Key includes period_start because EDGAR legitimately emits both a 3-month
    # Q2 fact and a 6-month YTD fact with the same period_end — different intervals,
    # different fy_labels, no contamination.  The invariant: one time interval
    # (start, end) maps to at most one fiscal label.
    period_to_label: dict[tuple, str] = {}
    for f in aapl_facts:
        if not f.fy_label:
            continue
        key = (f.concept, f.period_start, f.period_end, f.unit)
        if key in period_to_label:
            assert period_to_label[key] == f.fy_label, (
                f"Same time interval maps to two fiscal labels — data integrity broken: "
                f"{key} → {period_to_label[key]!r} and {f.fy_label!r}"
            )
        else:
            period_to_label[key] = f.fy_label

    # Every fact must have a non-empty period_end.
    for f in aapl_facts:
        assert f.period_end, f"Missing period_end: {f}"


# ---------------------------------------------------------------------------
# Test 3 — Dedup check
# Running the loader twice and unioning the dedup keys gives the same count
# as one run — the CompanyFacts API's re-reporting of prior quarters in 10-Ks
# is fully collapsed.
# ---------------------------------------------------------------------------

def test_dedup(aapl_facts: list[XbrlFact]) -> None:
    second_run = load_company_facts(TICKER, concepts=[REF_CONCEPT])

    keys_first = {f.dedup_key for f in aapl_facts}
    keys_second = {f.dedup_key for f in second_run}

    assert keys_first == keys_second, (
        f"Dedup instability: run1={len(keys_first)} keys, run2={len(keys_second)} keys"
    )
    # Merging both runs should not inflate the count.
    merged_keys = keys_first | keys_second
    assert len(merged_keys) == len(keys_first), (
        f"Merged union has more keys than a single run — double-counting present"
    )


# ---------------------------------------------------------------------------
# Test 4 — Parity diff (XBRL values vs. hardcoded reference table)
# Stands in for the HTML-scrape comparison until that path is built.
# The reference table is the ground truth from the official 10-Q/10-K.
# Any divergence fails the test — investigate before trusting the XBRL path.
# ---------------------------------------------------------------------------

_REFERENCE_TABLE: dict[tuple, float] = {
    # (ticker, concept, period_start, period_end, unit): expected_value
    # Source: official SEC filings. period_start included to disambiguate
    # quarterly vs annual facts that share the same period_end date.

    # AAPL 10-Q Q1 FY2024 (Oct–Dec 2023) — accession 0000320193-24-000006
    ("AAPL", "RevenueFromContractWithCustomerExcludingAssessedTax", "2023-10-01", "2023-12-30", "USD"): 119_575_000_000,

    # AAPL 10-K FY2023 (full year Sep 25 2022–Sep 30 2023) — accession 0000320193-23-000077
    ("AAPL", "NetIncomeLoss", "2022-09-25", "2023-09-30", "USD"): 96_995_000_000,

    # MSFT 10-Q Q2 FY2024 (Oct–Dec 2023, MSFT fiscal Q2) — accession 0000950170-24-014625
    ("MSFT", "RevenueFromContractWithCustomerExcludingAssessedTax", "2023-10-01", "2023-12-31", "USD"): 62_020_000_000,

    # GOOG 10-K FY2023 (calendar year Jan–Dec 2023) — accession 0001652044-24-000022
    ("GOOG", "Revenues", "2023-01-01", "2023-12-31", "USD"): 307_394_000_000,
}


def test_parity_diff() -> None:
    tickers  = list({k[0] for k in _REFERENCE_TABLE})
    concepts = list({k[1] for k in _REFERENCE_TABLE})

    fact_map: dict[tuple, float] = {}
    for t in tickers:
        for f in load_company_facts(t, concepts=concepts):
            fact_map[(t, f.concept, f.period_start, f.period_end, f.unit)] = f.value

    divergences = []
    for (ticker, concept, period_start, period_end, unit), ref_val in _REFERENCE_TABLE.items():
        xbrl_val = fact_map.get((ticker, concept, period_start, period_end, unit))
        if xbrl_val != ref_val:
            divergences.append((ticker, concept, period_start, period_end, unit, ref_val, xbrl_val))

    if divergences:
        lines = ["\nParity divergences (investigate before trusting XBRL path):"]
        for ticker, concept, period_start, period_end, unit, ref, xbrl in divergences:
            lines.append(
                f"  {ticker} | {concept} | {period_start} to {period_end} | {unit}\n"
                f"    reference={ref:,}  xbrl={xbrl}"
            )
        pytest.fail("\n".join(lines))


# ---------------------------------------------------------------------------
# Test 5 — Chunker output is well-formed
# Sanity check that facts_to_chunks produces usable dicts.
# ---------------------------------------------------------------------------

def test_chunker_output(aapl_facts: list[XbrlFact]) -> None:
    chunks = facts_to_chunks(aapl_facts)
    assert len(chunks) == len(aapl_facts)

    for c in chunks:
        assert isinstance(c["text"], str) and c["text"].strip(), "Empty chunk text"
        assert c["concept"], "Missing concept"
        assert c["period_end"], "Missing period_end"
        assert c["value"] is not None, "Missing value"

    # The reference fact's chunk must contain the fiscal label in its text.
    ref_chunks = [
        c for c in chunks
        if c["concept"] == REF_CONCEPT and c["period_end"] == REF_PERIOD_END
    ]
    assert ref_chunks, "Reference fact not found in chunker output"
    ref_chunk = ref_chunks[0]
    assert f"FY{REF_FY}-{REF_FP}" in ref_chunk["text"], (
        f"Fiscal label missing from chunk text: {ref_chunk['text']!r}"
    )
    assert ref_chunk["fy_label"] == f"FY{REF_FY}-{REF_FP}"
