"""
tests/test_retriever.py — Unit tests for the retriever helpers.

No live Chroma or embedding model calls — _get_collection and _get_model are
mocked. Integration against the real index happens in Step 5 evaluation.

Tests cover:
  - _build_where: empty, single-key, two-key Chroma where= clause construction
  - _pack_results: Chroma output → list of chunk dicts
  - retrieve / retrieve_both: correct structure, filter routing, fallback logic
"""

from __future__ import annotations

import os
os.environ["DRY_RUN"] = "true"

from unittest.mock import MagicMock, patch

import pytest

import retrieval.retriever as ret
from retrieval.retriever import _build_where, _detect_ratio_concepts, _pack_results


# ---------------------------------------------------------------------------
# _build_where
# ---------------------------------------------------------------------------

def test_build_where_empty():
    assert _build_where({}) is None


def test_build_where_single_key():
    assert _build_where({"ticker": "AAPL"}) == {"ticker": "AAPL"}


def test_build_where_two_keys():
    result = _build_where({"ticker": "AAPL", "fiscal_period": "FY2024-Q1"})
    assert result == {
        "$and": [
            {"ticker": {"$eq": "AAPL"}},
            {"fiscal_period": {"$eq": "FY2024-Q1"}},
        ]
    }


# ---------------------------------------------------------------------------
# _pack_results
# ---------------------------------------------------------------------------

_SAMPLE_CHROMA_OUTPUT = {
    "ids":       [["id1", "id2"]],
    "documents": [["chunk text A", "chunk text B"]],
    "metadatas": [
        [
            {"ticker": "AAPL", "fiscal_period": "FY2024-Q1", "concept": "NetIncomeLoss",
             "form_type": "10-Q", "accession": "0000320193-24-000006",
             "entity": "APPLE INC", "cik": "0000320193",
             "period_end": "2023-12-30", "period_type": "duration"},
            {"ticker": "AAPL", "fiscal_period": "FY2023-Q1", "concept": "NetIncomeLoss",
             "form_type": "10-Q", "accession": "0000320193-23-000006",
             "entity": "APPLE INC", "cik": "0000320193",
             "period_end": "2022-12-31", "period_type": "duration"},
        ]
    ],
    "distances": [[0.05, 0.18]],
}


def test_pack_results_length():
    assert len(_pack_results(_SAMPLE_CHROMA_OUTPUT)) == 2


def test_pack_results_fields():
    chunks = _pack_results(_SAMPLE_CHROMA_OUTPUT)
    c = chunks[0]
    assert c["id"] == "id1"
    assert c["text"] == "chunk text A"
    assert c["distance"] == 0.05
    assert c["ticker"] == "AAPL"
    assert c["fiscal_period"] == "FY2024-Q1"


def test_pack_results_distance_order():
    chunks = _pack_results(_SAMPLE_CHROMA_OUTPUT)
    assert chunks[0]["distance"] < chunks[1]["distance"]


# ---------------------------------------------------------------------------
# retrieve — mocked collection + model
# ---------------------------------------------------------------------------

def _make_mock_collection(return_chunks=_SAMPLE_CHROMA_OUTPUT, empty=False):
    col = MagicMock()
    col.query.return_value = (
        {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        if empty
        else return_chunks
    )
    return col


def _make_mock_model():
    import numpy as np
    model = MagicMock()
    model.encode.return_value = np.zeros(384)
    return model


@patch.object(ret, "_get_collection")
@patch.object(ret, "_get_model")
def test_retrieve_baseline_no_where(mock_model, mock_col):
    mock_model.return_value = _make_mock_model()
    mock_col.return_value = _make_mock_collection()

    result = ret.retrieve("Apple net income Q1 2024", filtered=False)

    call_kwargs = mock_col.return_value.query.call_args.kwargs
    assert "where" not in call_kwargs
    assert result["filter_used"] == {}
    assert result["fallback"] is None
    assert len(result["chunks"]) == 2


@patch.object(ret, "_get_collection")
@patch.object(ret, "_get_model")
def test_retrieve_filtered_applies_where(mock_model, mock_col):
    mock_model.return_value = _make_mock_model()
    mock_col.return_value = _make_mock_collection()

    result = ret.retrieve("Apple net income Q1 2024", filtered=True)

    call_kwargs = mock_col.return_value.query.call_args.kwargs
    assert "where" in call_kwargs
    assert result["filter_used"] != {}


@patch.object(ret, "_get_collection")
@patch.object(ret, "_get_model")
def test_retrieve_fallback_ticker_only(mock_model, mock_col):
    """When the period+ticker filter returns nothing, retry with ticker-only."""
    mock_model.return_value = _make_mock_model()
    empty = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    full  = _SAMPLE_CHROMA_OUTPUT
    mock_col.return_value.query.side_effect = [empty, full]

    result = ret.retrieve("Apple net income Q1 2024", filtered=True)

    assert result["fallback"] == "ticker_only"
    assert result["filter_used"] == {"ticker": "AAPL"}


@patch.object(ret, "_get_collection")
@patch.object(ret, "_get_model")
def test_retrieve_fallback_pure_ann(mock_model, mock_col):
    """When both filtered and ticker-only return nothing, fall back to pure ANN."""
    mock_model.return_value = _make_mock_model()
    empty = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    full  = _SAMPLE_CHROMA_OUTPUT
    mock_col.return_value.query.side_effect = [empty, empty, full]

    result = ret.retrieve("Apple net income Q1 2024", filtered=True)

    assert result["fallback"] == "none"
    assert result["filter_used"] == {}


# ---------------------------------------------------------------------------
# retrieve_both — structure check
# ---------------------------------------------------------------------------

@patch.object(ret, "_get_collection")
@patch.object(ret, "_get_model")
def test_retrieve_both_structure(mock_model, mock_col):
    mock_model.return_value = _make_mock_model()
    mock_col.return_value = _make_mock_collection()

    result = ret.retrieve_both("Apple net income Q1 2024")

    assert "question" in result
    assert "parsed_filter" in result
    assert "filtered" in result
    assert "baseline" in result
    assert isinstance(result["parsed_filter"], dict)
    assert isinstance(result["filtered"]["chunks"], list)
    assert isinstance(result["baseline"]["chunks"], list)


# ---------------------------------------------------------------------------
# _detect_ratio_concepts — ratio question routing
# ---------------------------------------------------------------------------

def test_detect_ratio_quick_ratio():
    concepts = _detect_ratio_concepts("What was Apple's quick ratio in FY2022?")
    assert concepts == ["AssetsCurrent", "InventoryNet", "LiabilitiesCurrent"]


def test_detect_ratio_case_insensitive():
    """Detection lowercases the question, so capitalised phrasing still fires."""
    assert _detect_ratio_concepts("Compute the GROSS MARGIN for 3M") is not None


def test_detect_ratio_capex():
    concepts = _detect_ratio_concepts("What was 3M's capex for FY2022?")
    assert concepts == ["PaymentsToAcquirePropertyPlantAndEquipment"]


def test_detect_ratio_none_for_plain_question():
    """A non-ratio question returns None so the standard top-k path is used."""
    assert _detect_ratio_concepts("What was PepsiCo's net income in Q1 2022?") is None


@patch.object(ret, "_get_collection")
@patch.object(ret, "_get_model")
def test_retrieve_both_routes_ratio_to_multi_concept(mock_model, mock_col):
    """A ratio question makes the filtered side use multi-concept retrieval."""
    mock_model.return_value = _make_mock_model()
    mock_col.return_value = _make_mock_collection()

    result = ret.retrieve_both("What was Apple's quick ratio in FY2022?")

    # multi_concept field is only present on the retrieve_multi_concept() path
    assert result["filtered"]["multi_concept"] == [
        "AssetsCurrent", "InventoryNet", "LiabilitiesCurrent"
    ]
