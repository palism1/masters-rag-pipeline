"""
stage2_xbrl_eval.py — Stage-2 period-tagger evaluation on real XBRL chunks.

Replaces the synthetic CURATED_DATASET from Stage 1 with EDGAR-sourced chunks
that carry ground-truth fy_label values from SEC's own fy/fp tags — no labels
are inferred or regex-extracted.  Imports regex_tag and similarity_tag_all from
the Stage-1 file unchanged so accuracy numbers are directly comparable.

Usage:
    python -m evaluation.xbrl_eval                   # AAPL, default concepts
    python -m evaluation.xbrl_eval MSFT GOOG NVDA    # multiple tickers
    python -m evaluation.xbrl_eval AAPL --concepts NetIncomeLoss EarningsPerShareBasic
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict

os.environ["DRY_RUN"] = "true"

import config  # noqa: E402
from ingestion.xbrl_loader import DEFAULT_CONCEPTS, load_company_facts
from ingestion.xbrl_chunker import facts_to_chunks
from .tagger import regex_tag, similarity_tag_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Row conversion
# ---------------------------------------------------------------------------

def chunks_to_rows(chunks: list[dict]) -> list[tuple[str, str, str]]:
    """
    Convert XBRL chunk dicts to (text, true_label, stratum) rows.

    Chunks without fy_label are dropped — no ground truth, can't score.
    Stratum separates duration facts (income-statement items spanning a period)
    from instant facts (balance-sheet items at a point in time) because the
    period phrase in the text differs, which affects tagger difficulty.
    """
    rows = []
    for c in chunks:
        if not c["fy_label"]:
            continue
        stratum = "xbrl_duration" if c["period_type"] == "duration" else "xbrl_instant"
        rows.append((c["text"], c["fy_label"], stratum))
    return rows


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_xbrl(rows: list[tuple]) -> dict:
    """
    Run both taggers and print a per-stratum accuracy table.

    Returns the raw per-stratum counts dict for programmatic use / tests.
    """
    regex_preds = [regex_tag(r[0]) for r in rows]
    sim_preds, backend = similarity_tag_all(rows)

    by: dict = defaultdict(lambda: {"n": 0, "regex": 0, "sim": 0})
    misses: dict[str, list] = {"regex": [], "sim": []}

    for r, rp, sp in zip(rows, regex_preds, sim_preds):
        text, true, stratum = r
        by[stratum]["n"] += 1
        by["ALL"]["n"] += 1
        if rp == true:
            by[stratum]["regex"] += 1
            by["ALL"]["regex"] += 1
        else:
            misses["regex"].append((stratum, text, true, rp))
        if sp == true:
            by[stratum]["sim"] += 1
            by["ALL"]["sim"] += 1
        else:
            misses["sim"].append((stratum, text, true, sp))

    print(f"\nBackend: {backend}\n")
    ordered = sorted(s for s in by if s != "ALL") + ["ALL"]
    print(f"{'stratum':<18}{'n':>5}{'regex acc':>12}{'sim acc':>10}")
    print("-" * 45)
    for s in ordered:
        d = by[s]
        ra = d["regex"] / d["n"] if d["n"] else 0.0
        sa = d["sim"] / d["n"] if d["n"] else 0.0
        print(f"{s:<18}{d['n']:>5}{ra:>11.0%}{sa:>10.0%}")

    print("\nSample regex misses (stratum | true -> pred | text[:72]):")
    for stratum, text, true, pred in misses["regex"][:6]:
        print(f"  [{stratum}] {true} -> {pred!r}  | {text[:72]}")

    return dict(by)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage-2 XBRL period-tagger evaluation")
    parser.add_argument("tickers", nargs="*", default=["AAPL"])
    parser.add_argument(
        "--concepts", nargs="+", default=None,
        help="XBRL concept names (default: DEFAULT_CONCEPTS)",
    )
    args = parser.parse_args(argv)

    concepts = args.concepts or DEFAULT_CONCEPTS
    all_rows: list[tuple] = []

    for ticker in args.tickers:
        try:
            facts = load_company_facts(ticker, concepts=concepts)
            chunks = facts_to_chunks(facts)
            rows = chunks_to_rows(chunks)
            logging.info(
                "%s: %d facts → %d chunks → %d labeled rows",
                ticker, len(facts), len(chunks), len(rows),
            )
            all_rows.extend(rows)
        except Exception as exc:
            logging.error("Failed for %s: %s", ticker, exc)

    if not all_rows:
        print("No labeled rows produced — check tickers and concepts.", file=sys.stderr)
        return 1

    print("=" * 60)
    print("PERIOD-TAGGING EVALUATION  (Stage 2 — real XBRL data)")
    print("=" * 60)
    print(f"Tickers : {', '.join(args.tickers)}")
    print(f"Concepts: {', '.join(sorted(set(concepts)))}")
    print(f"Rows    : {len(all_rows)}")

    evaluate_xbrl(all_rows)

    print(
        "\nWhy regex underperforms on quarterly facts:\n"
        "  Chunk text uses 'FY2024-Q1' (XBRL label format).\n"
        "  Regex pattern 7 matches FY\\d+ and extracts year-only → wrong for Q facts.\n"
        "  The similarity tagger sees the full string and can learn the quarter token.\n"
        "  Annual facts ('FY2024') are unambiguous — both taggers should match them."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
