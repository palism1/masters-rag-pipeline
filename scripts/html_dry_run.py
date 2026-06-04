"""
html_dry_run.py — Manual dry-run entry point for the HTML ingestion path.

Fetches one (or a few) 10-K/10-Q filings for a ticker, runs html_chunker's
markdown walk + period propagation, and prints a coverage summary WITHOUT
writing anything. Use this to eyeball how many narrative chunks inherit a
fiscal period from a heading versus how many are stranded with no nearby dated
heading — the implicit-tier problem the thesis targets.

Usage:
    python -m scripts.html_dry_run                 # defaults: AAPL, 10-K
    python -m scripts.html_dry_run AAPL            # one ticker
    python -m scripts.html_dry_run MSFT --form 10-Q --limit 2

DRY_RUN is forced True; this path never writes regardless of .env.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force DRY_RUN before config loads so the import-time check sees it.
os.environ["DRY_RUN"] = "true"

import config  # noqa: E402,F401 — import for its env validation side effect
from ingestion.html_chunker import fetch_html_chunks  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)

SAMPLE_N = 5  # how many example chunks to print from each bucket


def _trim(text: str, width: int = 96) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _print_summary(chunks: list[dict], ticker: str, form_type: str) -> None:
    labeled = [c for c in chunks if c["fy_label"]]
    implicit = [c for c in chunks if not c["fy_label"]]

    print(f"\n{'='*78}")
    print(f"  {ticker}  {form_type}  —  {len(chunks)} chunks")
    print(f"{'='*78}")
    print(f"  with fy_label (heading-propagated): {len(labeled)}")
    print(f"  without fy_label (implicit tier)  : {len(implicit)}")
    if chunks:
        pct = 100 * len(labeled) / len(chunks)
        print(f"  label coverage                    : {pct:.0f}%")

    print(f"\n  --- {SAMPLE_N} LABELED chunks (heading supplied the period) ---")
    for c in labeled[:SAMPLE_N]:
        print(f"  [{c['fy_label']:<11}] heading={_trim(c['section_heading'] or '', 40)!r}")
        print(f"      {_trim(c['text'])}")

    print(f"\n  --- {SAMPLE_N} IMPLICIT chunks (no dated heading above — the problem) ---")
    for c in implicit[:SAMPLE_N]:
        print(f"  [fy_label=None]  {_trim(c['text'])}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HTML filing dry-run chunker")
    parser.add_argument("ticker", nargs="?", default="AAPL")
    parser.add_argument("--form", default="10-K", help="filing form type (10-K / 10-Q)")
    parser.add_argument("--limit", type=int, default=1, help="most-recent N filings")
    args = parser.parse_args(argv)

    chunks = fetch_html_chunks(args.ticker, form_type=args.form, limit=args.limit)
    if not chunks:
        print(
            f"\nNo chunks returned for {args.ticker} {args.form}. "
            "edgartools could not fetch/parse the filing (see logs above).",
            file=sys.stderr,
        )
        return 1

    _print_summary(chunks, args.ticker, args.form)
    print(
        "DRY_RUN complete — nothing written. Inspect label coverage above before "
        "wiring this tier into the index builder.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
