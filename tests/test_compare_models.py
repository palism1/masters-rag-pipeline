"""
tests/test_compare_models.py — Unit tests for compare_models metric logic.

No file I/O against real results, no Chroma, no API — all inputs are synthetic
result dicts built to exercise the aggregation edge cases. load_results is
tested only against a guaranteed-missing path (returns None).

FILE MAP
  L001–L020  Module docstring + file map
  L022–L038  CONFIG + imports + synthetic-row factory
  L040–L075  Known-set metric calculation tests
  L077–L100  non_numeric exclusion tests
  L102–L130  Missing-file / empty / partial-run handling tests
"""

from __future__ import annotations

import os
os.environ["DRY_RUN"] = "true"   # belt-and-braces: never let imports hit Chroma

import pytest

from evaluation.compare_models import (
    compute_metrics,
    load_results,
    available_models,
)

# ===========================================================================
# CONFIG — synthetic-row factory
# ===========================================================================


def _mode(answer_score="correct", retrieval=True, citation=True) -> dict:
    """One mode sub-dict (filtered or baseline) with only the scored fields."""
    return {
        "answer_score":      answer_score,
        "retrieval_correct": retrieval,
        "citation_correct":  citation,
    }


def _row(filtered: dict | None = None, baseline: dict | None = None) -> dict:
    """One per-question result row. Defaults to all-correct in both modes."""
    return {
        "filtered": filtered if filtered is not None else _mode(),
        "baseline": baseline if baseline is not None else _mode(),
    }


def _results(rows: list[dict]) -> dict:
    """Wrap rows into the {fid: row} mapping that compute_metrics consumes."""
    return {f"q{i}": r for i, r in enumerate(rows)}


# ===========================================================================


# ---------------------------------------------------------------------------
# Known-set metric calculation
# ---------------------------------------------------------------------------

def test_all_correct_gives_full_marks():
    metrics = compute_metrics(_results([_row(), _row(), _row()]))
    assert metrics["n"] == 3
    filt = metrics["modes"]["filtered"]
    assert filt["retrieval"]      == (1.0, 3, 3)
    assert filt["answer_strict"]  == (1.0, 3, 3)
    assert filt["answer_lenient"] == (1.0, 3, 3)
    assert filt["citation"]       == (1.0, 3, 3)


def test_mixed_set_rates():
    # 4 questions: 2 correct, 1 scale_mismatch, 1 wrong.
    rows = [
        _row(filtered=_mode("correct",        retrieval=True,  citation=True)),
        _row(filtered=_mode("correct",        retrieval=True,  citation=False)),
        _row(filtered=_mode("scale_mismatch", retrieval=False, citation=True)),
        _row(filtered=_mode("wrong",          retrieval=False, citation=False)),
    ]
    filt = compute_metrics(_results(rows))["modes"]["filtered"]

    assert filt["retrieval"]      == (0.5, 2, 4)        # 2/4 retrieval hits
    assert filt["citation"]       == (0.5, 2, 4)        # 2/4 citation hits
    assert filt["answer_strict"]  == (0.5, 2, 4)        # 2 correct / 4 scorable
    assert filt["answer_lenient"] == (0.75, 3, 4)       # +scale_mismatch


def test_filtered_and_baseline_scored_independently():
    rows = [_row(filtered=_mode("correct"), baseline=_mode("wrong"))]
    m = compute_metrics(_results(rows))["modes"]
    assert m["filtered"]["answer_strict"][0] == 1.0
    assert m["baseline"]["answer_strict"][0] == 0.0


def test_score_breakdown_counts():
    rows = [
        _row(filtered=_mode("correct")),
        _row(filtered=_mode("correct")),
        _row(filtered=_mode("no_answer")),
    ]
    bd = compute_metrics(_results(rows))["modes"]["filtered"]["score_breakdown"]
    assert bd == {"correct": 2, "no_answer": 1}


# ---------------------------------------------------------------------------
# non_numeric exclusion — the load-bearing scoring rule
# ---------------------------------------------------------------------------

def test_non_numeric_excluded_from_answer_denominator():
    # 3 questions, one is non_numeric → denominator must be 2, not 3.
    rows = [
        _row(filtered=_mode("correct")),
        _row(filtered=_mode("wrong")),
        _row(filtered=_mode("non_numeric")),
    ]
    filt = compute_metrics(_results(rows))["modes"]["filtered"]
    assert filt["answer_strict"]  == (0.5, 1, 2)        # 1 correct / 2 scorable
    assert filt["answer_lenient"] == (0.5, 1, 2)
    # non_numeric still appears in the raw breakdown (for transparency).
    assert filt["score_breakdown"]["non_numeric"] == 1


def test_all_non_numeric_yields_none_rate():
    rows = [_row(filtered=_mode("non_numeric")), _row(filtered=_mode("non_numeric"))]
    filt = compute_metrics(_results(rows))["modes"]["filtered"]
    assert filt["answer_strict"] == (None, 0, 0)        # empty denominator → n/a


def test_none_correctness_flags_excluded():
    # Out-of-scope questions carry None retrieval/citation flags — dropped.
    rows = [
        _row(filtered=_mode("correct", retrieval=True,  citation=True)),
        _row(filtered=_mode("correct", retrieval=None,  citation=None)),
    ]
    filt = compute_metrics(_results(rows))["modes"]["filtered"]
    assert filt["retrieval"] == (1.0, 1, 1)             # only the in-scope row
    assert filt["citation"]  == (1.0, 1, 1)


# ---------------------------------------------------------------------------
# Missing-file / empty / partial-run handling
# ---------------------------------------------------------------------------

def test_load_results_missing_file_returns_none():
    assert load_results("results/eval_results_does_not_exist_xyz.json") is None


def test_available_models_skips_unknown_and_missing(capsys):
    # 'bogus' is an unknown slug; 'sentinel_test_slug' is a known but
    # guaranteed-absent slug injected into MODELS for this test.
    import evaluation.compare_models as cm
    original = dict(cm.MODELS)
    cm.MODELS["sentinel_test_slug"] = (
        "Sentinel", "results/eval_results_sentinel_test_slug_DOES_NOT_EXIST.json"
    )
    try:
        resolved = available_models(["bogus", "sentinel_test_slug"])
    finally:
        cm.MODELS.clear()
        cm.MODELS.update(original)

    assert resolved == []
    out = capsys.readouterr().out
    assert "unknown model slug 'bogus'" in out
    assert "no results for 'sentinel_test_slug'" in out


def test_partial_run_smaller_n_still_computes():
    # A model with only 2 questions done must still produce valid metrics.
    rows = [_row(filtered=_mode("correct")), _row(filtered=_mode("wrong"))]
    metrics = compute_metrics(_results(rows))
    assert metrics["n"] == 2
    assert metrics["modes"]["filtered"]["answer_strict"] == (0.5, 1, 2)


def test_empty_results_dict():
    metrics = compute_metrics({})
    assert metrics["n"] == 0
    filt = metrics["modes"]["filtered"]
    assert filt["retrieval"]     == (None, 0, 0)
    assert filt["answer_strict"] == (None, 0, 0)
