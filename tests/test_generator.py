"""
tests/test_generator.py — Unit tests for the generation layer.

No live API calls — anthropic client and retriever are mocked.
Tests cover prompt construction, response parsing, empty-chunk handling,
and the generate_both() output structure.
"""

from __future__ import annotations

import os
os.environ["DRY_RUN"] = "true"

from unittest.mock import MagicMock, patch

import pytest

import retrieval.generator as gen
from retrieval.generator import _format_chunks, _parse_response


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

_CHUNKS = [
    {
        "text":          "PepsiCo reported NetIncomeLoss of $1.322B for the period 2022-01-01 to 2022-03-19 (FY2022-Q1).",
        "fiscal_period": "FY2022-Q1",
        "ticker":        "PEP",
        "concept":       "NetIncomeLoss",
        "entity":        "PEPSICO INC",
        "accession":     "0000077476-22-000010",
        "distance":      0.04,
    },
    {
        "text":          "PepsiCo reported Revenues of $16.201B for the period 2022-01-01 to 2022-03-19 (FY2022-Q1).",
        "fiscal_period": "FY2022-Q1",
        "ticker":        "PEP",
        "concept":       "Revenues",
        "entity":        "PEPSICO INC",
        "accession":     "0000077476-22-000010",
        "distance":      0.06,
    },
]

_MOCK_RESPONSE_TEXT = (
    "ANSWER: $1.322 billion\n"
    "FISCAL_PERIOD: FY2022-Q1\n"
    "SOURCE: 0000077476-22-000010\n"
    "CONFIDENCE: HIGH"
)


# ---------------------------------------------------------------------------
# _format_chunks
# ---------------------------------------------------------------------------

def test_format_chunks_contains_text():
    formatted = _format_chunks(_CHUNKS)
    assert "1.322B" in formatted
    assert "FY2022-Q1" in formatted
    assert "0000077476-22-000010" in formatted


def test_format_chunks_numbered():
    formatted = _format_chunks(_CHUNKS)
    assert "[1]" in formatted
    assert "[2]" in formatted


def test_format_chunks_empty():
    assert _format_chunks([]) == ""


def test_format_chunks_includes_plain_english_label():
    """Concepts in CONCEPT_GLOSSARY render as 'XBRLName (Plain English)'."""
    formatted = _format_chunks(_CHUNKS)
    # NetIncomeLoss → "Net income (net earnings)" per the glossary
    assert "NetIncomeLoss (Net income (net earnings))" in formatted


def test_format_chunks_fallback_when_concept_unknown():
    """Concepts absent from the glossary keep the bare XBRL name, no parens."""
    chunk = [{
        "text":          "Some unmapped fact.",
        "fiscal_period": "FY2022-Q1",
        "concept":       "SomeUnmappedConcept",
        "entity":        "ACME INC",
        "accession":     "0000000000-22-000001",
    }]
    formatted = _format_chunks(chunk)
    assert "SomeUnmappedConcept" in formatted
    assert "SomeUnmappedConcept (" not in formatted   # no empty/parenthesized label


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

def test_parse_response_all_fields():
    result = _parse_response(_MOCK_RESPONSE_TEXT)
    assert result["answer"]        == "$1.322 billion"
    assert result["fiscal_period"] == "FY2022-Q1"
    assert result["source"]        == "0000077476-22-000010"
    assert result["confidence"]    == "HIGH"
    assert result["raw"]           == _MOCK_RESPONSE_TEXT


def test_parse_response_missing_fields():
    result = _parse_response("ANSWER: unknown\nSOURCE: abc")
    assert result["answer"]        == "unknown"
    assert result["fiscal_period"] is None
    assert result["source"]        == "abc"
    assert result["confidence"]    is None


def test_parse_response_preserves_raw():
    text = "some unexpected format"
    result = _parse_response(text)
    assert result["raw"] == text


# ---------------------------------------------------------------------------
# generate — mocked client
# ---------------------------------------------------------------------------

def _make_mock_client(response_text: str = _MOCK_RESPONSE_TEXT):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=response_text)]
    client = MagicMock()
    client.messages.create.return_value = mock_msg
    return client


@patch.object(gen, "_get_client")
def test_generate_returns_answer(mock_get_client):
    mock_get_client.return_value = _make_mock_client()
    result = gen.generate("What was PepsiCo's net income in Q1 2022?", _CHUNKS)
    assert result["answer"] == "$1.322 billion"
    assert result["fiscal_period"] == "FY2022-Q1"
    assert result["chunks_used"] == 2


@patch.object(gen, "_get_client")
def test_generate_calls_api_once(mock_get_client):
    client = _make_mock_client()
    mock_get_client.return_value = client
    gen.generate("What was PepsiCo's net income in Q1 2022?", _CHUNKS)
    assert client.messages.create.call_count == 1


def test_generate_empty_chunks_no_api_call():
    result = gen.generate("any question", [])
    assert result["answer"] is None
    assert result["chunks_used"] == 0


@patch.object(gen, "_get_client")
def test_generate_passes_model(mock_get_client):
    client = _make_mock_client()
    mock_get_client.return_value = client
    gen.generate("q", _CHUNKS)
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == gen.MODEL


# ---------------------------------------------------------------------------
# generate_both — structure check
# ---------------------------------------------------------------------------

_MOCK_RETRIEVE_BOTH = {
    "question":      "What was PepsiCo's net income in Q1 2022?",
    "parsed_filter": {"ticker": "PEP", "fiscal_period": "FY2022-Q1"},
    "filtered": {
        "question":    "What was PepsiCo's net income in Q1 2022?",
        "filter_used": {"ticker": "PEP", "fiscal_period": "FY2022-Q1"},
        "fallback":    None,
        "chunks":      _CHUNKS,
    },
    "baseline": {
        "question":    "What was PepsiCo's net income in Q1 2022?",
        "filter_used": {},
        "fallback":    None,
        "chunks":      _CHUNKS,
    },
}


@patch.object(gen, "_get_client")
@patch("retrieval.generator.retrieve_both", return_value=_MOCK_RETRIEVE_BOTH)
def test_generate_both_structure(mock_retrieve, mock_get_client):
    mock_get_client.return_value = _make_mock_client()
    result = gen.generate_both("What was PepsiCo's net income in Q1 2022?")

    assert "question"      in result
    assert "parsed_filter" in result
    assert "filtered"      in result
    assert "baseline"      in result

    for mode in ("filtered", "baseline"):
        assert "answer"        in result[mode]
        assert "fiscal_period" in result[mode]
        assert "source"        in result[mode]
        assert "confidence"    in result[mode]
        assert "retrieval"     in result[mode]
        assert "chunks_used"   in result[mode]


@patch.object(gen, "_get_client")
@patch("retrieval.generator.retrieve_both", return_value=_MOCK_RETRIEVE_BOTH)
def test_generate_both_calls_api_twice(mock_retrieve, mock_get_client):
    client = _make_mock_client()
    mock_get_client.return_value = client
    gen.generate_both("What was PepsiCo's net income in Q1 2022?")
    assert client.messages.create.call_count == 2
