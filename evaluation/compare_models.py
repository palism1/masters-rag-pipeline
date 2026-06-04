"""
evaluation/compare_models.py — Side-by-side comparison of embedding models.

Loads the per-model evaluation result files produced by eval_pipeline.py
(one file per embedding model), computes the same headline metrics for each,
and renders both a terminal table and an HTML report.

Models compared (slug → embedding backend):
  minilm   sentence-transformers/all-MiniLM-L6-v2   (current baseline)
  finbert  ProsusAI/finbert                          (financial domain BERT)
  mpnet    sentence-transformers/all-mpnet-base-v2   (stronger general model)

Each model writes results/eval_results_<slug>.json with the SAME schema as
results/eval_results.json (per-question dict with "filtered" / "baseline"
sub-dicts carrying answer_score, retrieval_correct, citation_correct).

Metrics reported (identical definitions to eval_pipeline.print_metrics):
  - Retrieval period accuracy : top-1 chunk has correct fiscal period
  - Answer accuracy (strict)  : answer_score == correct
  - Answer accuracy (lenient) : answer_score in {correct, scale_mismatch}
  - Period citation accuracy  : Claude cited the correct fiscal period
All reported for both filtered and baseline modes.

WHY a separate script: eval_pipeline.py owns running + scoring one model.
This script is read-only aggregation across already-scored files, so it never
touches Chroma, the API, or the dataset — it only reads JSON.

FILE MAP
  L001–L045  Module docstring + file map
  L047–L078  CONFIG knobs (model slug → display name) + imports
  L080–L123  Result loading — load_results(), available_models()
  L125–L195  Metric computation — compute_metrics() (pure, testable)
  L197–L245  Terminal table — print_comparison()
  L247–L335  HTML report — render_html(), write_report()
  L337–L365  Entry point — main()

Usage
-----
    python evaluation/compare_models.py                          # all available
    python evaluation/compare_models.py --models minilm finbert  # subset
    python evaluation/compare_models.py --out my_report.html     # custom path
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DRY_RUN", "true")   # never hit Chroma/API from this script

# ===========================================================================
# CONFIG — tweak these to change which models are compared / how they display
# ===========================================================================

# CHANGE ME: model slug → (display name, results file). Order here is the
# left-to-right column order in the terminal table and HTML report.
# To add a model: drop its result file in results/ and add a row here.
MODELS: dict[str, tuple[str, str]] = {
    "minilm":  ("MiniLM (general)",   "results/eval_results_minilm.json"),
    "finbert": ("FinBERT (finance)",  "results/eval_results_finbert.json"),
    "mpnet":   ("MPNet (general+)",   "results/eval_results_mpnet.json"),
}

DEFAULT_HTML_OUT = Path("model_comparison_report.html")  # CHANGE ME: report path

# Scores that count toward answer accuracy. non_numeric is excluded from the
# denominator entirely (qualitative ground truths cannot be auto-scored), so it
# must NOT be added here. TWEAK lenient set if scale_mismatch should be dropped.
STRICT_SCORES  = {"correct"}
LENIENT_SCORES = {"correct", "scale_mismatch"}

# Colour palette for the HTML report — kept identical to make_report.py.
GREEN = "#c6efce"   # correct
RED   = "#ffc7ce"   # wrong
GREY  = "#f2f2f2"   # n/a

# ===========================================================================


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def load_results(path: str | Path) -> dict | None:
    """
    Load one model's result file. Returns the parsed dict, or None if the file
    does not exist yet.

    WHY return None instead of raising: models are evaluated incrementally, so a
    missing file simply means "not run yet" — the caller skips it with a note
    rather than crashing the whole comparison.
    """
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def available_models(slugs: list[str]) -> list[tuple[str, str, dict]]:
    """
    Resolve requested slugs to (slug, display_name, results) triples, loading
    each file. Skips (with a printed note) any model whose file is absent or
    whose results dict is empty.
    """
    resolved: list[tuple[str, str, dict]] = []
    for slug in slugs:
        if slug not in MODELS:
            print(f"  note: unknown model slug '{slug}' — skipping "
                  f"(known: {', '.join(MODELS)})")
            continue
        display, path = MODELS[slug]
        results = load_results(path)
        if not results:
            print(f"  note: no results for '{slug}' yet — skipping ({path})")
            continue
        resolved.append((slug, display, results))
    return resolved


# ---------------------------------------------------------------------------
# Metric computation — pure functions, unit-tested without file I/O
# ---------------------------------------------------------------------------

def _rate(hits: int, total: int) -> float | None:
    """Hit rate, or None when the denominator is empty (renders as 'n/a')."""
    return hits / total if total else None


def _flag_rate(rows: list[dict], mode: str, field: str) -> tuple[int, int]:
    """
    (hits, total) for a boolean correctness flag (retrieval_correct /
    citation_correct). None flags are out-of-scope questions and are dropped
    from the denominator, matching eval_pipeline._pct.
    """
    flags = [r[mode][field] for r in rows]
    valid = [f for f in flags if f is not None]
    return sum(1 for f in valid if f), len(valid)


def _answer_rate(rows: list[dict], mode: str, targets: set[str]) -> tuple[int, int]:
    """
    (hits, total) for answer accuracy. non_numeric scores are dropped from the
    denominator — they are qualitative answers that cannot be auto-scored, so
    counting them would unfairly depress every model's accuracy. Mirrors
    eval_pipeline._acc.
    """
    scores = [r[mode]["answer_score"] for r in rows]
    scorable = [s for s in scores if s != "non_numeric"]
    hits = sum(1 for s in scorable if s in targets)
    return hits, len(scorable)


def compute_metrics(results: dict) -> dict:
    """
    Compute headline metrics for one model's result set.

    Returns a dict with, per mode (filtered / baseline), the four headline rates
    plus their (hits, total) tuples, and an overall answer_score breakdown.
    This is the single source of truth consumed by both the terminal table and
    the HTML report — keeping them in lock-step.
    """
    rows = list(results.values())
    out: dict = {"n": len(rows), "modes": {}}

    for mode in ("filtered", "baseline"):
        ret_h, ret_t = _flag_rate(rows, mode, "retrieval_correct")
        cit_h, cit_t = _flag_rate(rows, mode, "citation_correct")
        str_h, str_t = _answer_rate(rows, mode, STRICT_SCORES)
        len_h, len_t = _answer_rate(rows, mode, LENIENT_SCORES)
        out["modes"][mode] = {
            "retrieval":       (_rate(ret_h, ret_t), ret_h, ret_t),
            "answer_strict":   (_rate(str_h, str_t), str_h, str_t),
            "answer_lenient":  (_rate(len_h, len_t), len_h, len_t),
            "citation":        (_rate(cit_h, cit_t), cit_h, cit_t),
            "score_breakdown": dict(Counter(r[mode]["answer_score"] for r in rows)),
        }
    return out


# ---------------------------------------------------------------------------
# Terminal table
# ---------------------------------------------------------------------------

# Display label → metrics-dict key, for the four headline rows.
_METRIC_ROWS = [
    ("Retrieval period accuracy", "retrieval"),
    ("Answer accuracy (strict)",  "answer_strict"),
    ("Answer accuracy (lenient)", "answer_lenient"),
    ("Period citation accuracy",  "citation"),
]


def _fmt_pair(metrics: dict, key: str) -> str:
    """Render 'filtered / baseline' as percentages for one metric row."""
    def one(mode: str) -> str:
        rate = metrics["modes"][mode][key][0]
        return "n/a" if rate is None else f"{rate:.0%}"
    return f"{one('filtered')} / {one('baseline')}"


def print_comparison(models: list[tuple[str, str, dict]]) -> None:
    """Print the side-by-side comparison table (filtered / baseline per cell)."""
    computed = [(slug, disp, compute_metrics(res)) for slug, disp, res in models]

    label_w = 28
    col_w   = max(20, max((len(d) for _, d, _ in computed), default=20) + 2)

    header = " " * label_w + "".join(f"{d:>{col_w}}" for _, d, _ in computed)
    print()
    print("=" * len(header))
    print("MODEL COMPARISON  (cells show filtered / baseline)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for label, key in _METRIC_ROWS:
        cells = "".join(f"{_fmt_pair(m, key):>{col_w}}" for _, _, m in computed)
        print(f"{label:<{label_w}}{cells}")

    n_cells = "".join(f"{m['n']:>{col_w}}" for _, _, m in computed)
    print(f"{'Questions evaluated':<{label_w}}{n_cells}")
    print()


# ---------------------------------------------------------------------------
# HTML report — same visual style as make_report.py
# ---------------------------------------------------------------------------

CSS = """
<style>
  body { font-family: Arial, sans-serif; font-size: 13px; margin: 32px; }
  h1 { font-size: 20px; }
  h2 { font-size: 15px; margin-top: 32px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
  table { border-collapse: collapse; width: 100%; margin-top: 12px; }
  th { background: #404040; color: white; padding: 6px 10px; text-align: left; font-size: 12px; }
  td { padding: 5px 10px; border-bottom: 1px solid #e0e0e0; vertical-align: top; }
  .summary-box { background: #f8f8f8; border: 1px solid #ddd; padding: 16px 20px;
                 display: inline-block; margin: 8px 0; border-radius: 4px; }
  .summary-box span { font-size: 28px; font-weight: bold; margin-right: 6px; }
  .metric { display: inline-block; margin-right: 40px; }
  .note { color: #555; font-style: italic; margin-top: 8px; font-size: 12px; }
</style>
"""


def _cell(metrics: dict, key: str) -> str:
    """Summary-table cell: 'filtered / baseline' percentages for one metric."""
    parts = []
    for mode in ("filtered", "baseline"):
        rate = metrics["modes"][mode][key][0]
        parts.append("n/a" if rate is None else f"{rate:.0%}")
    return f'<td title="filtered / baseline">{parts[0]} / {parts[1]}</td>'


def _summary_table(computed: list[tuple[str, str, dict]]) -> str:
    """Build the top summary table — metrics as rows, models as columns."""
    head = "<tr><th>Metric (filtered / baseline)</th>" + \
           "".join(f"<th>{disp}</th>" for _, disp, _ in computed) + "</tr>"
    body_rows = []
    for label, key in _METRIC_ROWS:
        cells = "".join(_cell(m, key) for _, _, m in computed)
        body_rows.append(f"<tr><td><b>{label}</b></td>{cells}</tr>")
    n_cells = "".join(f"<td>{m['n']}</td>" for _, _, m in computed)
    body_rows.append(f"<tr><td><b>Questions evaluated</b></td>{n_cells}</tr>")
    return f"<table>{head}{''.join(body_rows)}</table>"


def _breakdown_section(slug: str, disp: str, metrics: dict) -> str:
    """Per-model section: answer_score breakdown for filtered + baseline."""
    n = metrics["n"]
    blocks = []
    for mode in ("filtered", "baseline"):
        bd = metrics["modes"][mode]["score_breakdown"]
        rows = "".join(
            f"<tr><td>{score}</td><td>{count}</td>"
            f"<td>{count / n:.0%}</td></tr>"
            for score, count in sorted(bd.items())
        )
        blocks.append(
            f"<h3 style='font-size:13px;margin-top:16px'>{mode.capitalize()}</h3>"
            f"<table><tr><th>Answer score</th><th>Count</th><th>Share</th></tr>"
            f"{rows}</table>"
        )

    strict  = metrics["modes"]["filtered"]["answer_strict"][0]
    lenient = metrics["modes"]["filtered"]["answer_lenient"][0]
    summary = f"""
<div class="summary-box">
  <div class="metric"><span>{'n/a' if strict is None else f'{strict:.0%}'}</span>Strict (filtered)</div>
  <div class="metric"><span>{'n/a' if lenient is None else f'{lenient:.0%}'}</span>Lenient (filtered)</div>
  <div class="metric"><span>{n}</span>Questions</div>
</div>
"""
    return (f"<h2>{disp} <span style='color:#888;font-weight:normal'>({slug})</span></h2>"
            f"{summary}{''.join(blocks)}")


def render_html(computed: list[tuple[str, str, dict]]) -> str:
    """Assemble the full HTML document from the summary table + per-model sections."""
    sections = "".join(_breakdown_section(s, d, m) for s, d, m in computed)
    note = (
        '<p class="note">Cells show filtered / baseline. Answer accuracy excludes '
        "non_numeric ground truths (qualitative answers that cannot be auto-scored). "
        "Partial runs are shown as-is — compare per-model question counts before "
        "reading across rows.</p>"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Embedding model comparison</title>{CSS}</head>
<body>
<h1>Embedding model comparison</h1>
<p>FinanceBench 10-K / 10-Q · filtered vs baseline retrieval · {len(computed)} model(s)</p>
<h2>Summary</h2>
{_summary_table(computed)}
{note}
{sections}
</body>
</html>"""


def write_report(models: list[tuple[str, str, dict]], out_path: Path) -> None:
    """Compute metrics, render the HTML report, and write it to disk."""
    computed = [(slug, disp, compute_metrics(res)) for slug, disp, res in models]
    out_path.write_text(render_html(computed), encoding="utf-8")
    print(f"HTML report written → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--models", nargs="+", default=list(MODELS),
                   help="Model slugs to compare (default: all known)")
    p.add_argument("--out", type=Path, default=DEFAULT_HTML_OUT,
                   help="HTML report output path")
    args = p.parse_args(argv)

    models = available_models(args.models)
    if not models:
        print("No result files found — run eval_pipeline.py per model first.")
        return 1

    print_comparison(models)
    write_report(models, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
