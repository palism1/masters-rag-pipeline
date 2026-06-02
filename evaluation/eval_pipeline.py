"""
evaluation/eval_pipeline.py — Full evaluation of filtered vs baseline retrieval on FinanceBench.

Runs generate_both() on all 127 in-scope FinanceBench questions (10-K + 10-Q),
saves results to results/eval_results.json, then prints the comparison table.

Resumable: already-completed questions are loaded from the results file and
skipped on re-run — safe to interrupt and restart without losing progress or
paying for duplicate API calls.

Metrics reported:
  - Retrieval period accuracy : top-1 chunk has correct fiscal period
  - Answer accuracy (strict)  : generated answer matches ground truth (numeric ±5%)
  - Answer accuracy (lenient) : correct OR scale_mismatch (right number, wrong unit prefix)
  - Period citation accuracy  : Claude cited the correct fiscal period in its response

All metrics reported for both filtered and baseline modes.

Usage
-----
    python evaluation/eval_pipeline.py              # full 127 questions
    python evaluation/eval_pipeline.py --limit 10   # first N questions (dev)
    python evaluation/eval_pipeline.py --reset      # clear saved results and start fresh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DRY_RUN", "false")

from datasets import load_dataset

import config
from retrieval.generator import generate_both

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RESULTS_PATH = Path("results/eval_results.json")
IN_SCOPE     = {"10k", "10q"}

# ---------------------------------------------------------------------------
# Ground-truth helpers (shared with audit_fb_parser)
# ---------------------------------------------------------------------------

_DOC_NAME_RE = re.compile(r"^[A-Z0-9]+_(\d{4})(Q[1-4])?_10[KQ]$", re.IGNORECASE)


def _ground_truth_period(doc_name: str) -> str | None:
    m = _DOC_NAME_RE.match(doc_name)
    if not m:
        return None
    year    = m.group(1)
    quarter = m.group(2)
    return f"FY{year}-{quarter}" if quarter else f"FY{year}"


# ---------------------------------------------------------------------------
# Answer scorer — Option A: numeric extraction with tolerance
# ---------------------------------------------------------------------------

_SUFFIX_MULT = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def _extract_number(text: str) -> float | None:
    """
    Extract and normalise the first significant number from text.
    Handles $, commas, B/M/K/T suffixes, and word forms (billion, million).
    Returns the value in base units (no suffix), or None if no number found.
    """
    clean = re.sub(r"[$,]", "", text.lower())
    # Match: optional minus, digits with optional decimal, optional suffix
    m = re.search(r"(-?\d+\.?\d*)\s*(billion|million|thousand|[bmkt])\b", clean)
    if m:
        val    = float(m.group(1))
        suffix = m.group(2)[0]   # first char: b/m/k/t
        return val * _SUFFIX_MULT[suffix]

    # Plain number (no suffix)
    m = re.search(r"-?\d+\.?\d*", clean)
    return float(m.group(0)) if m else None


def score_answer(ground_truth: str, generated: str | None) -> str:
    """
    Compare a generated answer against the FinanceBench ground truth.

    Returns one of:
      correct        — numeric match within 5% at the same scale
      scale_mismatch — same numeric value but at a different scale (e.g. $1577M vs $1.577B)
                       almost certainly correct, just a unit prefix difference
      wrong          — numeric values genuinely differ
      non_numeric    — ground truth is not numeric; cannot auto-score
      no_answer      — generated answer is None or contains no number
    """
    if not generated:
        return "no_answer"

    gt_val  = _extract_number(ground_truth)
    gen_val = _extract_number(generated)

    if gt_val is None:
        return "non_numeric"
    if gen_val is None:
        return "no_answer"

    # Zero case
    if gt_val == 0:
        return "correct" if abs(gen_val) < 0.01 else "wrong"

    ratio = gen_val / gt_val

    # Direct match (within 5%)
    if 0.95 <= ratio <= 1.05:
        return "correct"

    # Scale variants — handles M vs B unit confusion
    for scale in (1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9):
        if 0.95 <= ratio * scale <= 1.05:
            return "scale_mismatch"

    return "wrong"


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def _load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


def _save_results(results: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))


def run(limit: int | None = None, reset: bool = False) -> dict:
    if reset and RESULTS_PATH.exists():
        RESULTS_PATH.unlink()
        logger.info("Results cleared.")

    logger.info("Loading FinanceBench...")
    fb      = load_dataset("PatronusAI/financebench", split="train")
    rows    = [r for r in fb if r["doc_type"].lower() in IN_SCOPE]
    if limit:
        rows = rows[:limit]
    logger.info("%d in-scope questions loaded.", len(rows))

    results = _load_results()
    skipped = sum(1 for r in rows if r["financebench_id"] in results)
    if skipped:
        logger.info("Resuming — %d already done, %d remaining.", skipped, len(rows) - skipped)

    for i, row in enumerate(rows):
        fid = row["financebench_id"]
        if fid in results:
            continue

        question     = row["question"]
        ground_truth = row["answer"]
        true_period  = _ground_truth_period(row["doc_name"])

        logger.info("[%d/%d] %s — %s", i + 1, len(rows), row["company"], question[:60])

        try:
            gen = generate_both(question)
        except Exception as exc:
            logger.warning("SKIP %s — %s", fid, exc)
            continue

        def _score_mode(mode: str) -> dict:
            g = gen[mode]
            chunks = g["retrieval"]["chunks"]
            top1_period = chunks[0]["fiscal_period"] if chunks else None
            return {
                "answer":            g.get("answer"),
                "fiscal_period_cited": g.get("fiscal_period"),
                "source":            g.get("source"),
                "confidence":        g.get("confidence"),
                "raw":               g.get("raw", ""),
                "top1_period":       top1_period,
                "fallback":          g["retrieval"].get("fallback"),
                "filter_used":       g["retrieval"].get("filter_used", {}),
                "answer_score":      score_answer(ground_truth, g.get("answer")),
                "retrieval_correct": top1_period == true_period if true_period else None,
                "citation_correct":  g.get("fiscal_period") == true_period if true_period else None,
            }

        results[fid] = {
            "financebench_id": fid,
            "company":         row["company"],
            "doc_name":        row["doc_name"],
            "doc_type":        row["doc_type"],
            "question":        question,
            "ground_truth":    ground_truth,
            "true_period":     true_period,
            "parsed_filter":   gen["parsed_filter"],
            "filtered":        _score_mode("filtered"),
            "baseline":        _score_mode("baseline"),
        }

        _save_results(results)
        time.sleep(0.3)   # gentle rate-limit buffer

    return results


# ---------------------------------------------------------------------------
# Metrics + display
# ---------------------------------------------------------------------------

def print_metrics(results: dict) -> None:
    rows = list(results.values())
    n    = len(rows)

    def _pct(vals):
        valid = [v for v in vals if v is not None]
        if not valid:
            return "n/a"
        return f"{sum(valid) / len(valid):.0%}  ({sum(valid)}/{len(valid)})"

    def _acc(mode, strict=True):
        scores = [r[mode]["answer_score"] for r in rows]
        correct = ["correct"] if strict else ["correct", "scale_mismatch"]
        valid   = [s for s in scores if s not in ("non_numeric",)]
        hits    = sum(1 for s in valid if s in correct)
        if not valid:
            return "n/a"
        return f"{hits / len(valid):.0%}  ({hits}/{len(valid)})"

    print()
    print("=" * 64)
    print(f"EVALUATION RESULTS  ({n} questions)")
    print("=" * 64)
    print(f"{'Metric':<35} {'Filtered':>13} {'Baseline':>13}")
    print("-" * 64)

    print(f"{'Retrieval period accuracy':<35} "
          f"{_pct([r['filtered']['retrieval_correct'] for r in rows]):>13} "
          f"{_pct([r['baseline']['retrieval_correct'] for r in rows]):>13}")

    print(f"{'Answer accuracy (strict)':<35} "
          f"{_acc('filtered', strict=True):>13} "
          f"{_acc('baseline', strict=True):>13}")

    print(f"{'Answer accuracy (lenient)':<35} "
          f"{_acc('filtered', strict=False):>13} "
          f"{_acc('baseline', strict=False):>13}")

    print(f"{'Period citation accuracy':<35} "
          f"{_pct([r['filtered']['citation_correct'] for r in rows]):>13} "
          f"{_pct([r['baseline']['citation_correct'] for r in rows]):>13}")

    print()

    # Answer score breakdown
    from collections import Counter
    for mode in ("filtered", "baseline"):
        counts = Counter(r[mode]["answer_score"] for r in rows)
        print(f"  {mode.upper()} answer scores: " +
              "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print()
    print(f"Results saved → {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N questions (dev/test)")
    p.add_argument("--reset", action="store_true",
                   help="Clear saved results and start from scratch")
    args = p.parse_args()

    results = run(limit=args.limit, reset=args.reset)
    print_metrics(results)


if __name__ == "__main__":
    main()
