"""
scripts/audit_fb_parser.py — Audit query parser coverage against FinanceBench.

Loads the 150-question public FinanceBench split, runs every question through
parse_query(), and reports:

  1. Scope breakdown — which doc_types are in/out of XBRL scope
  2. Ticker extraction rate — how many questions correctly identified the company
  3. Period extraction rate — how many extracted any fiscal period
  4. Period accuracy — extracted label vs ground truth from doc_name
  5. Filter coverage — how many questions get a full filter vs fallback to pure ANN
  6. Failure samples — questions where the parser missed or got the period wrong

Ground truth is derived from doc_name (e.g. "3M_2023Q2_10Q" → "FY2023-Q2"),
which is more reliable than the doc_period integer for quarterly questions.

FILE MAP
  L001–L028  Module docstring + file map
  L030–L048  Imports + CONFIG
  L050–L076  Scope classification + doc_name → period parser
  L078–L095  Ground-truth ticker lookup
  L097–L256  run_audit() — main evaluation loop and report printing

Usage
-----
    python scripts/audit_fb_parser.py
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["DRY_RUN"] = "true"

from datasets import load_dataset

from retrieval.query_parser import _KNOWN_TICKERS, _NAME_TO_TICKER, _normalize, parse_query
from retrieval.financebench_tickers import FINANCEBENCH_COMPANIES

# ---------------------------------------------------------------------------
# Doc type scoping
# ---------------------------------------------------------------------------

IN_SCOPE  = {"10k", "10q"}    # XBRL-indexed in our pipeline
OUT_SCOPE = {"8k", "earnings"} # not structured XBRL — out of scope


# ---------------------------------------------------------------------------
# Ground-truth period from doc_name
# ---------------------------------------------------------------------------

_DOC_NAME_RE = re.compile(
    r"^[A-Z0-9]+_(\d{4})(Q[1-4])?_10[KQ]$", re.IGNORECASE
)

def _ground_truth_period(doc_name: str) -> str | None:
    """
    Parse fiscal period from doc_name convention.
      3M_2018_10K        → FY2018
      3M_2023Q2_10Q      → FY2023-Q2
      BESTBUY_2024Q2_10Q → FY2024-Q2
    Returns None if the pattern doesn't match (8-K, Earnings, etc.)
    """
    m = _DOC_NAME_RE.match(doc_name)
    if not m:
        return None
    year    = m.group(1)
    quarter = m.group(2)       # e.g. "Q2" or None
    if quarter:
        return f"FY{year}-{quarter}"
    return f"FY{year}"


# ---------------------------------------------------------------------------
# Ticker ground truth from company field
# ---------------------------------------------------------------------------

# Build a company → ticker lookup from the registry
_COMPANY_TO_TICKER: dict[str, str] = {
    c.company_name.lower(): c.ticker
    for c in FINANCEBENCH_COMPANIES
    if c.ticker
}


def _ground_truth_ticker(company: str) -> str | None:
    """Best-effort ticker for a FinanceBench company name."""
    return _COMPANY_TO_TICKER.get(company.lower())


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def run_audit() -> None:
    print("Loading FinanceBench...")
    fb = load_dataset("PatronusAI/financebench", split="train")
    rows = list(fb)
    print(f"Loaded {len(rows)} rows.\n")

    # -----------------------------------------------------------------------
    # 1. Scope breakdown
    # -----------------------------------------------------------------------
    type_counts = Counter(r["doc_type"].lower() for r in rows)
    in_scope  = [r for r in rows if r["doc_type"].lower() in IN_SCOPE]
    out_scope = [r for r in rows if r["doc_type"].lower() in OUT_SCOPE]

    print("=" * 60)
    print("1. SCOPE BREAKDOWN")
    print("=" * 60)
    for doc_type, n in sorted(type_counts.items()):
        tag = "IN SCOPE" if doc_type in IN_SCOPE else "OUT OF SCOPE"
        print(f"  {doc_type:<12} {n:>3}   {tag}")
    print(f"\n  In scope (10-K + 10-Q): {len(in_scope)}")
    print(f"  Out of scope (8-K, Earnings): {len(out_scope)}")
    print(f"  → Evaluating on {len(in_scope)} questions.\n")

    # -----------------------------------------------------------------------
    # 2. Run parser on in-scope questions
    # -----------------------------------------------------------------------
    results = []
    for r in in_scope:
        question   = r["question"]
        company    = r["company"]
        doc_name   = r["doc_name"]
        doc_type   = r["doc_type"].lower()

        parsed       = parse_query(question)
        true_ticker  = _ground_truth_ticker(company)
        true_period  = _ground_truth_period(doc_name)

        ticker_extracted = parsed.get("ticker")
        period_extracted = parsed.get("fiscal_period")

        ticker_correct = (ticker_extracted == true_ticker) if true_ticker else None
        period_correct = (period_extracted == true_period) if true_period else None

        results.append({
            "question":        question,
            "company":         company,
            "doc_name":        doc_name,
            "doc_type":        doc_type,
            "true_ticker":     true_ticker,
            "true_period":     true_period,
            "ticker_extracted": ticker_extracted,
            "period_extracted": period_extracted,
            "ticker_correct":  ticker_correct,
            "period_correct":  period_correct,
            "filter_type":     (
                "full"        if ticker_extracted and period_extracted else
                "ticker_only" if ticker_extracted else
                "period_only" if period_extracted else
                "none"
            ),
        })

    # -----------------------------------------------------------------------
    # 3. Ticker extraction
    # -----------------------------------------------------------------------
    scorable_ticker = [r for r in results if r["true_ticker"] is not None]
    ticker_correct  = sum(1 for r in scorable_ticker if r["ticker_correct"])

    print("=" * 60)
    print("2. TICKER EXTRACTION")
    print("=" * 60)
    print(f"  Questions with known ticker: {len(scorable_ticker)}")
    print(f"  Correctly extracted:         {ticker_correct} / {len(scorable_ticker)}"
          f"  ({ticker_correct/len(scorable_ticker):.0%})")

    missed = [r for r in scorable_ticker if not r["ticker_correct"]]
    if missed:
        print(f"\n  Failures ({len(missed)}):")
        for r in missed[:8]:
            print(f"    [{r['company']} → want {r['true_ticker']}, got {r['ticker_extracted']!r}]")
            print(f"    {r['question'][:90]}")
    print()

    # -----------------------------------------------------------------------
    # 4. Period extraction
    # -----------------------------------------------------------------------
    scorable_period = [r for r in results if r["true_period"] is not None]
    period_any      = sum(1 for r in scorable_period if r["period_extracted"])
    period_correct  = sum(1 for r in scorable_period if r["period_correct"])

    print("=" * 60)
    print("3. PERIOD EXTRACTION")
    print("=" * 60)
    print(f"  Questions with known period:   {len(scorable_period)}")
    print(f"  Extracted any period:          {period_any} / {len(scorable_period)}"
          f"  ({period_any/len(scorable_period):.0%})")
    print(f"  Extracted correct period:      {period_correct} / {len(scorable_period)}"
          f"  ({period_correct/len(scorable_period):.0%})")

    # Break down by 10-K vs 10-Q
    for dtype in ["10k", "10q"]:
        sub = [r for r in scorable_period if r["doc_type"] == dtype]
        if not sub:
            continue
        correct = sum(1 for r in sub if r["period_correct"])
        print(f"\n  {dtype.upper()} ({len(sub)} questions):")
        print(f"    Correct: {correct} / {len(sub)}  ({correct/len(sub):.0%})")

    # Period failures
    period_wrong = [r for r in scorable_period if not r["period_correct"]]
    if period_wrong:
        print(f"\n  Failures ({len(period_wrong)}) — want → got:")
        for r in period_wrong[:10]:
            print(f"    [{r['doc_name']}]  want={r['true_period']}  got={r['period_extracted']!r}")
            print(f"    {r['question'][:90]}")
    print()

    # -----------------------------------------------------------------------
    # 5. Filter coverage
    # -----------------------------------------------------------------------
    filter_counts = Counter(r["filter_type"] for r in results)

    print("=" * 60)
    print("4. FILTER COVERAGE  (what the retriever will use)")
    print("=" * 60)
    for ftype in ["full", "ticker_only", "period_only", "none"]:
        n = filter_counts[ftype]
        pct = n / len(results)
        bar = "█" * int(pct * 30)
        print(f"  {ftype:<14} {n:>3}  {pct:>5.0%}  {bar}")
    print()
    useful = filter_counts["full"] + filter_counts["ticker_only"] + filter_counts["period_only"]
    print(f"  Questions with any filter:   {useful} / {len(results)}  ({useful/len(results):.0%})")
    print(f"  Pure ANN fallback:           {filter_counts['none']} / {len(results)}")

    # -----------------------------------------------------------------------
    # 6. Coverage by fiscal year (index check)
    # -----------------------------------------------------------------------
    year_counts = Counter(
        r["true_period"][:6] for r in results if r["true_period"]
    )
    print()
    print("=" * 60)
    print("5. FISCAL YEARS REFERENCED  (check index coverage)")
    print("=" * 60)
    for yr, n in sorted(year_counts.items()):
        print(f"  {yr}  {n:>3} questions")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  In-scope questions:       {len(results)}")
    print(f"  Ticker accuracy:          {ticker_correct}/{len(scorable_ticker)}  ({ticker_correct/len(scorable_ticker):.0%})")
    print(f"  Period accuracy:          {period_correct}/{len(scorable_period)}  ({period_correct/len(scorable_period):.0%})")
    print(f"  Full filter (both):       {filter_counts['full']}  ({filter_counts['full']/len(results):.0%})")
    print(f"  No filter (pure ANN):     {filter_counts['none']}  ({filter_counts['none']/len(results):.0%})")


if __name__ == "__main__":
    run_audit()
