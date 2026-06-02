"""
tests/test_stage2_xbrl_eval.py — Unit tests for the Stage-2 evaluation helper.

Tests focus on chunks_to_rows() and evaluate_xbrl(), using synthetic chunk
dicts that match the shape of facts_to_chunks() output.  No live SEC API calls.
"""

from __future__ import annotations

import os

os.environ["DRY_RUN"] = "true"

import pytest

from evaluation.xbrl_eval import chunks_to_rows, evaluate_xbrl
from evaluation.tagger import regex_tag


# ---------------------------------------------------------------------------
# Synthetic chunk factory
# ---------------------------------------------------------------------------

def _chunk(fy_label, period_type="duration", concept="NetIncomeLoss"):
    label_part = f" ({fy_label})" if fy_label else ""
    text = (
        f"ACME Corp reported {concept} of $1.000B "
        f"for the period 2024-01-01 to 2024-03-31{label_part}."
    )
    return {
        "text": text,
        "fy_label": fy_label,
        "concept": concept,
        "period_type": period_type,
        "period_start": "2024-01-01",
        "period_end": "2024-03-31",
        "unit": "USD",
        "value": 1_000_000_000.0,
    }


# ---------------------------------------------------------------------------
# chunks_to_rows
# ---------------------------------------------------------------------------

def test_none_fy_label_filtered():
    chunks = [_chunk("FY2024-Q1"), _chunk(None)]
    rows = chunks_to_rows(chunks)
    assert len(rows) == 1
    assert rows[0][1] == "FY2024-Q1"


def test_stratum_duration():
    rows = chunks_to_rows([_chunk("FY2024-Q1", period_type="duration")])
    assert rows[0][2] == "xbrl_duration"


def test_stratum_instant():
    rows = chunks_to_rows([_chunk("FY2024-Q1", period_type="instant")])
    assert rows[0][2] == "xbrl_instant"


def test_text_and_label_preserved():
    rows = chunks_to_rows([_chunk("FY2024-Q2")])
    text, label, _ = rows[0]
    assert label == "FY2024-Q2"
    assert "FY2024-Q2" in text
    assert isinstance(text, str) and text.strip()


def test_empty_input():
    assert chunks_to_rows([]) == []


def test_all_none_returns_empty():
    chunks = [_chunk(None), _chunk(None)]
    assert chunks_to_rows(chunks) == []


# ---------------------------------------------------------------------------
# evaluate_xbrl — structural checks using a small controlled dataset
# ---------------------------------------------------------------------------

sklearn = pytest.importorskip("sklearn", reason="scikit-learn required for similarity tagger fallback")


def test_evaluate_returns_all_key():
    rows = [
        ("ACME Corp reported NetIncomeLoss of $1.000B for the period 2024-01-01 to 2024-03-31 (FY2024-Q1).", "FY2024-Q1", "xbrl_duration"),
        ("ACME Corp reported NetIncomeLoss of $4.000B for the period 2024-01-01 to 2024-12-31 (FY2024).", "FY2024", "xbrl_duration"),
    ]
    result = evaluate_xbrl(rows)
    assert "ALL" in result
    assert result["ALL"]["n"] == 2


def test_evaluate_annual_regex_correct():
    # Annual facts have 'FY2023' in text — regex pattern 7 (FY\d+) should match.
    text = "ACME Corp reported Revenues of $10.000B for the period 2023-01-01 to 2023-12-31 (FY2023)."
    assert regex_tag(text) == "FY2023", (
        "Regex should correctly tag an annual fact containing 'FY2023'"
    )


def test_evaluate_quarterly_regex_wrong():
    # Quarterly facts embed 'FY2024-Q1'. Regex pattern 7 (FY\d+) extracts 'FY2024'
    # (year only) — wrong for quarterly facts. This documents the known gap.
    text = "ACME Corp reported NetIncomeLoss of $1.000B for the period 2024-01-01 to 2024-03-31 (FY2024-Q1)."
    assert regex_tag(text) != "FY2024-Q1", (
        "Regex should NOT correctly tag a quarterly XBRL chunk — "
        "pattern 7 extracts FY2024, not FY2024-Q1"
    )
