"""
make_report.py — HTML comparison report: regex vs similarity tagger.

Two modes:
  xbrl      (default)  Structured chunk text — "Apple Inc. reported ... (FY2024-Q1)"
  narrative            MD&A-style prose — "For the three months ended December 30, 2023..."
                       No fiscal label in the text. Tests the non-calendar fiscal year problem.

Usage:
    python make_report.py                        # AAPL xbrl mode
    python make_report.py AAPL --narrative       # AAPL narrative mode
    python make_report.py MSFT GOOG --narrative  # multiple tickers, narrative mode
    python make_report.py AAPL --concepts NetIncomeLoss EarningsPerShareBasic
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ["DRY_RUN"] = "true"

import pandas as pd

import config  # noqa: E402
from xbrl_loader import DEFAULT_CONCEPTS, load_company_facts
from xbrl_chunker import facts_to_chunks
from narrative_chunker import facts_to_narrative_chunks
from stage2_xbrl_eval import chunks_to_rows
from period_tagging_smoke_test import regex_tag, similarity_tag_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")

GREEN = "#c6efce"
RED   = "#ffc7ce"
GREY  = "#f2f2f2"

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


def _colour_cell(val, correct):
    bg = GREEN if correct else RED
    return f'<td style="background:{bg}">{val or "<i>None</i>"}</td>'


def _build_section(df_section, title, note=""):
    if df_section.empty:
        return f"<h2>{title}</h2><p><i>None.</i></p>"
    rows_html = []
    for _, r in df_section.iterrows():
        text_cell = f'<td style="color:#333;max-width:420px">{r["text"]}</td>'
        true_cell = f'<td><b>{r["true"]}</b></td>'
        regex_cell = _colour_cell(r["regex_pred"], r["regex_ok"])
        sim_cell   = _colour_cell(r["sim_pred"],   r["sim_ok"])
        rows_html.append(f"<tr>{text_cell}{true_cell}{regex_cell}{sim_cell}</tr>")
    header = (
        "<tr><th>Chunk text (first 120 chars)</th><th>True label</th>"
        "<th>Regex prediction</th><th>Similarity prediction</th></tr>"
    )
    note_html = f'<p class="note">{note}</p>' if note else ""
    return (
        f"<h2>{title}</h2>{note_html}"
        f'<table>{header}{"".join(rows_html)}</table>'
    )


def make_report(ticker: str, concepts: list[str], out_path: Path, narrative: bool = False) -> None:
    facts  = load_company_facts(ticker, concepts=concepts)
    chunks = facts_to_narrative_chunks(facts) if narrative else facts_to_chunks(facts)
    rows   = chunks_to_rows(chunks)

    if not rows:
        logging.warning("%s: no labeled rows — skipping report", ticker)
        return

    regex_preds          = [regex_tag(r[0]) for r in rows]
    sim_preds, backend   = similarity_tag_all(rows)

    records = []
    for (text, true, stratum), rp, sp in zip(rows, regex_preds, sim_preds):
        records.append({
            "text":      text[:120],
            "true":      true,
            "stratum":   stratum,
            "regex_pred": rp,
            "sim_pred":   sp,
            "regex_ok":  rp == true,
            "sim_ok":    sp == true,
        })
    df = pd.DataFrame(records)

    n          = len(df)
    regex_acc  = df["regex_ok"].mean()
    sim_acc    = df["sim_ok"].mean()

    df_rx_wrong_sim_right = df[ ~df["regex_ok"] &  df["sim_ok"]]
    df_both_wrong         = df[ ~df["regex_ok"] & ~df["sim_ok"]]
    df_rx_right_sim_wrong = df[  df["regex_ok"] & ~df["sim_ok"]]
    df_both_right         = df[  df["regex_ok"] &  df["sim_ok"]]

    summary_html = f"""
<div class="summary-box">
  <div class="metric"><span>{regex_acc:.0%}</span>Regex accuracy</div>
  <div class="metric"><span>{sim_acc:.0%}</span>Similarity accuracy ({backend.split("(")[0].strip()})</div>
  <div class="metric"><span>{n}</span>Labeled chunks</div>
</div>
"""

    sections = "".join([
        _build_section(
            df_rx_wrong_sim_right,
            f"Regex wrong, similarity right ({len(df_rx_wrong_sim_right)} rows)",
            "These are the cases that motivate the learned approach. "
            "Regex extracted the year but dropped the quarter; similarity matched the full label.",
        ),
        _build_section(
            df_both_wrong,
            f"Both wrong ({len(df_both_wrong)} rows)",
            "Edge cases where neither tagger recovers the correct label.",
        ),
        _build_section(
            df_rx_right_sim_wrong,
            f"Regex right, similarity wrong ({len(df_rx_right_sim_wrong)} rows)",
            "Cases where the regex pattern fired correctly but the nearest neighbour was misleading.",
        ),
        _build_section(
            df_both_right,
            f"Both correct ({len(df_both_right)} rows) — sample (first 20)",
            "Annual facts where 'FY2024' in the text is unambiguous for both approaches.",
        ) if not df_both_right.empty else "",
    ])

    mode_label = "narrative prose (no fiscal label in text)" if narrative else "XBRL-format chunks (fiscal label embedded)"
    mode_note  = (
        "<p><b>Narrative mode:</b> chunk text uses calendar-date phrasing only — "
        "<i>\"For the three months ended December 30, 2023...\"</i> — with no FY label. "
        "Regex maps December&nbsp;→&nbsp;Q4 (calendar year); Apple's fiscal Q1 ends in December, "
        "so regex predicts the wrong quarter for every quarterly Apple fact. "
        "This is the non-calendar fiscal year failure.</p>"
    ) if narrative else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Period-tagger report — {ticker} ({("narrative" if narrative else "xbrl")})</title>{CSS}</head>
<body>
<h1>Period-tagger comparison — {ticker}</h1>
<p>Mode: {mode_label} · Ground-truth from EDGAR fy/fp fields · {len(concepts)} concepts</p>
{mode_note}
{summary_html}
{sections}
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    logging.info("Report written → %s", out_path)
    return {"ticker": ticker, "mode": "narrative" if narrative else "xbrl",
            "n": n, "regex_acc": regex_acc, "sim_acc": sim_acc, "backend": backend}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate HTML comparison report")
    parser.add_argument("tickers", nargs="*", default=["AAPL"])
    parser.add_argument("--concepts", nargs="+", default=None)
    parser.add_argument("--narrative", action="store_true",
                        help="Use MD&A-style prose (no fiscal label in text)")
    args = parser.parse_args(argv)

    concepts = args.concepts or DEFAULT_CONCEPTS
    suffix   = "_narrative" if args.narrative else ""
    for ticker in args.tickers:
        out = Path(f"report_{ticker}{suffix}.html")
        try:
            make_report(ticker, concepts, out, narrative=args.narrative)
        except Exception as exc:
            logging.error("Failed for %s: %s", ticker, exc)


if __name__ == "__main__":
    sys.exit(main())
