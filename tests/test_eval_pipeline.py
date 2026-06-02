"""
tests/test_eval_pipeline.py — Unit tests for the answer scorer and ground-truth helpers.

No API calls, no Chroma, no EDGAR — pure string logic.
"""

from __future__ import annotations

import os
os.environ["DRY_RUN"] = "true"

import pytest
from evaluation.eval_pipeline import _extract_number, score_answer, _ground_truth_period


# ---------------------------------------------------------------------------
# _extract_number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("$1577.00",          1577.0),
    ("$8.70",             8.70),
    ("24.26",             24.26),
    ("0.66",              0.66),
    ("-0.02",             -0.02),
    ("0",                 0.0),
    ("1.9%",              1.9),
    ("65.4%",             65.4),
    ("$1.577B",           1.577e9),
    ("$1.577 billion",    1.577e9),
    ("$1,577 million",    1.577e9),
    ("$1,577M",           1.577e9),
    ("$11588.00",         11588.0),
    ("approximately $4.261B", 4.261e9),
    ("9.5 times",         9.5),
])
def test_extract_number(text, expected):
    result = _extract_number(text)
    assert result is not None
    assert abs(result - expected) / max(abs(expected), 1e-9) < 0.001


def test_extract_number_no_number():
    assert _extract_number("yes") is None
    assert _extract_number("Amcor is a global leader") is None


# ---------------------------------------------------------------------------
# score_answer
# ---------------------------------------------------------------------------

# Correct — same scale
@pytest.mark.parametrize("gt,gen", [
    ("$1577.00",   "$1,577.00"),
    ("$8.70",      "$8.70 per share"),
    ("24.26",      "24.26"),
    ("0.66",       "approximately 0.66"),
    ("1.9%",       "1.9%"),
    ("65.4%",      "65.4%"),
    ("0",          "$0"),
])
def test_score_correct(gt, gen):
    assert score_answer(gt, gen) == "correct"


# Scale mismatch — right value, different prefix
@pytest.mark.parametrize("gt,gen", [
    ("$1577.00",   "$1.577B"),           # millions vs billions
    ("$1577.00",   "$1.577 billion"),
    ("$11588.00",  "$11.588B"),
])
def test_score_scale_mismatch(gt, gen):
    assert score_answer(gt, gen) == "scale_mismatch"


# Wrong
@pytest.mark.parametrize("gt,gen", [
    ("$1577.00",   "$999.00"),
    ("24.26",      "12.13"),
    ("1.9%",       "5.2%"),
])
def test_score_wrong(gt, gen):
    assert score_answer(gt, gen) == "wrong"


# Non-numeric ground truth
@pytest.mark.parametrize("gt", [
    "No, the company is managing its CAPEX efficiently",
    "Amcor is a global leader in packaging",
    "Yes, dividend distribution is stable",
])
def test_score_non_numeric(gt):
    assert score_answer(gt, "some answer") == "non_numeric"


# No answer
def test_score_no_answer_none():
    assert score_answer("$1577.00", None) == "no_answer"


def test_score_no_answer_text_only():
    assert score_answer("$1577.00", "The information is not available") == "no_answer"


# Zero ground truth
def test_score_zero_correct():
    assert score_answer("0", "$0") == "correct"
    assert score_answer("0", "0") == "correct"


def test_score_zero_wrong():
    assert score_answer("0", "$100") == "wrong"


# ---------------------------------------------------------------------------
# _ground_truth_period
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("doc_name,expected", [
    ("3M_2018_10K",          "FY2018"),
    ("3M_2023Q2_10Q",        "FY2023-Q2"),
    ("BESTBUY_2024Q2_10Q",   "FY2024-Q2"),
    ("AMCOR_2023_10K",       "FY2023"),
    ("ADOBE_2015_10K",       "FY2015"),
    ("JPM_2022Q3_10Q",       "FY2022-Q3"),
])
def test_ground_truth_period(doc_name, expected):
    assert _ground_truth_period(doc_name) == expected


def test_ground_truth_period_out_of_scope():
    assert _ground_truth_period("COMPANY_2022_8K") is None
