"""
evaluation/make_retrieval_report.py — HTML failure report from eval results.

Reads results/eval_results.json (produced by eval_pipeline.py) and generates
retrieval_failure_report.html — a colour-coded table for every question where
filtered and baseline answers disagree.

Each row shows:
  - Question + ground truth
  - Filtered: top chunks retrieved (with fiscal_period visible), generated answer, score
  - Baseline: top chunks retrieved (with fiscal_period visible), generated answer, score

Key thesis argument made visible: the error is wrong retrieval, not hallucination.
Claude answered faithfully with whatever the retriever gave it. The wrong-period
chunk is shown explicitly so the reader can see exactly what went wrong.

FILE MAP
  L001–L030  Module docstring + file map
  L032–L048  Imports + CONFIG (paths, colours, display limits)
  L050–L120  CSS styles
  L122–L148  Score → style/label helpers
  L150–L170  Chunk summary renderer
  L172–L215  Row builder — one HTML table row per question
  L217–L272  make_report() — loads results, computes metrics, writes HTML
  L274–L285  Entry point — main()

Usage
-----
    python evaluation/make_retrieval_report.py
    python evaluation/make_retrieval_report.py --all   # include agreed-correct rows too
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ===========================================================================
# CONFIG
# ===========================================================================

RESULTS_PATH = Path("results/eval_results.json")        # CHANGE ME: input path
OUT_PATH     = Path("retrieval_failure_report.html")    # CHANGE ME: output path

# Colour palette — TWEAK to match your thesis document colours
GREEN  = "#c6efce"   # correct answer
RED    = "#ffc7ce"   # wrong answer
YELLOW = "#ffeb9c"   # scale mismatch (right value, wrong unit prefix)
GREY   = "#f5f5f5"   # no answer
BLUE   = "#dce6f1"   # non-numeric (qualitative answer, cannot auto-score)

# Max chunks shown per cell in the report — more than 3 gets cluttered
MAX_CHUNKS_SHOWN = 3                                    # TWEAK

# ===========================================================================

CSS = """
<style>
  body  { font-family: Arial, sans-serif; font-size: 12px; margin: 32px; }
  h1    { font-size: 20px; }
  h2    { font-size: 14px; margin-top: 28px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
  table { border-collapse: collapse; width: 100%; margin-top: 10px; }
  th    { background: #404040; color: white; padding: 6px 10px; text-align: left; font-size: 11px; }
  td    { padding: 5px 10px; border-bottom: 1px solid #e0e0e0; vertical-align: top; }
  .q    { font-weight: bold; color: #222; }
  .gt   { color: #1a6b1a; font-weight: bold; }
  .chunk-period { font-weight: bold; font-size: 11px; }
  .correct      { background: """ + GREEN  + """; }
  .wrong        { background: """ + RED    + """; }
  .scale        { background: """ + YELLOW + """; }
  .no-answer    { background: """ + GREY   + """; color: #888; font-style: italic; }
  .non-numeric  { background: """ + BLUE   + """; }
  .summary-box  { background: #f8f8f8; border: 1px solid #ddd; padding: 14px 20px;
                  display: inline-block; border-radius: 4px; margin: 8px 0; }
  .summary-box span { font-size: 26px; font-weight: bold; margin-right: 4px; }
  .metric       { display: inline-block; margin-right: 36px; }
  .score-tag    { display: inline-block; padding: 1px 6px; border-radius: 3px;
                  font-size: 10px; font-weight: bold; }
</style>
"""


def _score_style(score: str) -> str:
    return {
        "correct":       "correct",
        "scale_mismatch":"scale",
        "wrong":         "wrong",
        "no_answer":     "no-answer",
        "non_numeric":   "non-numeric",
    }.get(score, "")


def _score_label(score: str) -> str:
    return {
        "correct":       "CORRECT",
        "scale_mismatch":"SCALE?",
        "wrong":         "WRONG",
        "no_answer":     "NO ANSWER",
        "non_numeric":   "NON-NUMERIC",
    }.get(score, score.upper())


def _chunk_summary(mode_result: dict, max_chunks: int = MAX_CHUNKS_SHOWN) -> str:
    chunks = mode_result.get("retrieval", {}).get("chunks", [])[:max_chunks]
    if not chunks:
        return "<i>no chunks</i>"
    parts = []
    for c in chunks:
        period = c.get("fiscal_period", "?")
        concept = c.get("concept", "?")
        text = c.get("text", "")[:80]
        parts.append(
            f'<span class="chunk-period">{period}</span> | {concept}<br>'
            f'<span style="color:#555">{text}…</span>'
        )
    return "<br><br>".join(parts)


def _build_row(r: dict) -> str:
    filt = r["filtered"]
    base = r["baseline"]
    true_period = r.get("true_period", "?")

    filt_score = filt["answer_score"]
    base_score = base["answer_score"]

    # Skip if both correct and we're in failure-only mode (handled by caller)

    filt_style = _score_style(filt_score)
    base_style = _score_style(base_score)

    def _answer_cell(mode_result, style):
        answer  = mode_result.get("answer") or "<i>none</i>"
        period  = mode_result.get("fiscal_period_cited") or "?"
        label   = _score_label(mode_result["answer_score"])
        return (
            f'<td class="{style}">'
            f'<b>{answer}</b><br>'
            f'<span style="font-size:10px;color:#555">Period cited: {period}</span><br>'
            f'<span class="score-tag" style="background:#666;color:white">{label}</span>'
            f'</td>'
        )

    return (
        f"<tr>"
        f'<td><span class="q">{r["question"][:120]}</span><br>'
        f'<span class="gt">Ground truth: {r["ground_truth"]}</span><br>'
        f'<span style="font-size:10px;color:#888">{r["company"]} | {r["doc_name"]} | true period: {true_period}</span>'
        f"</td>"
        f"<td>{_chunk_summary(filt)}</td>"
        f"{_answer_cell(filt, filt_style)}"
        f"<td>{_chunk_summary(base)}</td>"
        f"{_answer_cell(base, base_style)}"
        f"</tr>"
    )


def make_report(include_all: bool = False) -> None:
    if not RESULTS_PATH.exists():
        print(f"Results file not found: {RESULTS_PATH}")
        print("Run eval_pipeline.py first.")
        return

    results = json.loads(RESULTS_PATH.read_text())
    rows    = list(results.values())
    n       = len(rows)

    # Metrics
    def _pct(vals):
        valid = [v for v in vals if v is not None]
        return (sum(valid) / len(valid)) if valid else 0.0

    def _acc(mode, strict=True):
        scores  = [r[mode]["answer_score"] for r in rows]
        correct = ["correct"] if strict else ["correct", "scale_mismatch"]
        valid   = [s for s in scores if s != "non_numeric"]
        return (sum(1 for s in valid if s in correct) / len(valid)) if valid else 0.0

    filt_ret   = _pct([r["filtered"]["retrieval_correct"] for r in rows])
    base_ret   = _pct([r["baseline"]["retrieval_correct"] for r in rows])
    filt_ans   = _acc("filtered")
    base_ans   = _acc("baseline")
    filt_lenient = _acc("filtered", strict=False)
    base_lenient = _acc("baseline", strict=False)

    summary_html = f"""
<div class="summary-box">
  <div class="metric"><span>{filt_ret:.0%}</span>Filtered retrieval<br>period accuracy</div>
  <div class="metric"><span>{base_ret:.0%}</span>Baseline retrieval<br>period accuracy</div>
  <div class="metric"><span>{filt_ans:.0%}</span>Filtered answer<br>accuracy (strict)</div>
  <div class="metric"><span>{base_ans:.0%}</span>Baseline answer<br>accuracy (strict)</div>
  <div class="metric"><span>{n}</span>questions<br>evaluated</div>
</div>
<p style="font-size:11px;color:#555;margin-top:4px">
  Lenient accuracy (correct + scale_mismatch):
  filtered {filt_lenient:.0%} | baseline {base_lenient:.0%}
</p>
"""

    # Filter rows for the table
    if include_all:
        display_rows = rows
        section_title = f"All {n} questions"
    else:
        display_rows = [
            r for r in rows
            if r["filtered"]["answer_score"] != r["baseline"]["answer_score"]
            or r["baseline"]["answer_score"] in ("wrong", "no_answer")
        ]
        section_title = (
            f"Questions where approaches disagree or baseline fails "
            f"({len(display_rows)} of {n})"
        )

    header = (
        "<tr>"
        "<th style='width:28%'>Question + Ground Truth</th>"
        "<th style='width:18%'>Filtered chunks (top 3)</th>"
        "<th style='width:12%'>Filtered answer</th>"
        "<th style='width:18%'>Baseline chunks (top 3)</th>"
        "<th style='width:12%'>Baseline answer</th>"
        "</tr>"
    )

    row_html = "".join(_build_row(r) for r in display_rows)

    legend = """
<p style="font-size:11px;margin-top:8px">
  <b>Colour key:</b>
  <span style="background:""" + GREEN + """;padding:2px 6px">CORRECT (±5%)</span>
  <span style="background:""" + YELLOW + """;padding:2px 6px">SCALE? (right value, unit prefix mismatch)</span>
  <span style="background:""" + RED + """;padding:2px 6px">WRONG</span>
  <span style="background:""" + GREY + """;padding:2px 6px">NO ANSWER</span>
  <span style="background:""" + BLUE + """;padding:2px 6px">NON-NUMERIC (manual review)</span>
</p>
<p style="font-size:11px;color:#555">
  <b>Key argument:</b> the error is wrong <i>retrieval</i>, not hallucination. Claude
  answered faithfully using whatever chunks the retriever returned. The chunk's
  fiscal_period shows exactly which period was retrieved and why the answer is wrong.
</p>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8">
<title>Retrieval failure report — fiscal-period-aware RAG</title>
{CSS}
</head>
<body>
<h1>Retrieval failure report — fiscal-period-aware RAG</h1>
<p>FinanceBench evaluation | filtered vs baseline retrieval | {n} questions</p>
{summary_html}
{legend}
<h2>{section_title}</h2>
<table>{header}{row_html}</table>
</body>
</html>"""

    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Report written → {OUT_PATH}  ({len(display_rows)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true",
                   help="Include all questions, not just disagreements/failures")
    args = p.parse_args()
    make_report(include_all=args.all)


if __name__ == "__main__":
    main()
