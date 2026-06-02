"""
xbrl_dry_run.py — Manual dry-run entry point for the XBRL ingestion path.

Fetches facts + generates chunks for one or more tickers, prints a formatted
table, and exits without writing anything to any store. Use this to verify
the pipeline is producing correct values before flipping DRY_RUN=false.

Usage:
    python xbrl_dry_run.py                  # defaults: AAPL, default concepts
    python xbrl_dry_run.py MSFT GOOG        # multiple tickers
    python xbrl_dry_run.py AAPL --concepts RevenueFromContractWithCustomerExcludingAssessedTax NetIncomeLoss

Config is read from .env — DRY_RUN is forced True regardless of .env setting.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Force DRY_RUN before config loads so the import-time check sees it.
os.environ["DRY_RUN"] = "true"

import config  # noqa: E402 — must come after env override
from xbrl_loader import DEFAULT_CONCEPTS, load_company_facts
from xbrl_chunker import facts_to_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)


def _print_table(chunks: list[dict], ticker: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {ticker}  —  {len(chunks)} deduplicated facts")
    print(f"{'='*72}")
    print(f"{'CONCEPT':<52} {'PERIOD':<12} {'LABEL':<14} {'VALUE':>22}")
    print(f"{'-'*52} {'-'*12} {'-'*14} {'-'*22}")
    for c in sorted(chunks, key=lambda x: (x["concept"], x["period_end"] or "")):
        concept_short = c["concept"][-52:]
        period = (c["period_end"] or "")[-12:]
        label = (c["fy_label"] or "?")[:14]
        value = f"{c['value']:,.0f} {c['unit']}"
        print(f"{concept_short:<52} {period:<12} {label:<14} {value:>22}")
    print()
    print("Sample chunk text:")
    for c in list(chunks)[:3]:
        print(f"  {c['text']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="XBRL dry-run loader")
    parser.add_argument("tickers", nargs="*", default=["AAPL"])
    parser.add_argument(
        "--concepts", nargs="+", default=None,
        help="XBRL concept names to fetch (defaults to DEFAULT_CONCEPTS)",
    )
    args = parser.parse_args(argv)

    concepts = args.concepts or DEFAULT_CONCEPTS

    any_error = False
    for ticker in args.tickers:
        try:
            facts = load_company_facts(ticker, concepts=concepts)
            chunks = facts_to_chunks(facts)
            _print_table(chunks, ticker)
        except Exception as exc:
            logging.error("Failed for %s: %s", ticker, exc)
            any_error = True

    if any_error:
        print("\nOne or more tickers failed — see errors above.", file=sys.stderr)
        return 1

    print(
        "\nDRY_RUN complete. Review the output above, then set DRY_RUN=false "
        "in .env to enable writes.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
