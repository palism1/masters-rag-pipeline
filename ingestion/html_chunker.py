"""
html_chunker.py — Fetch SEC 10-K/10-Q filing HTML and emit period-labeled chunks.

This is the "implicit tier" ingestion path. Unlike xbrl_chunker.py — where the
fiscal period lives inside the fact and is therefore inside the chunk text — the
narrative paragraphs of a filing's MD&A say things like
    "Revenue increased 12% compared to the prior-year period."
The period for that sentence is NOT in the sentence. It lives in the section
heading above it ("Three Months Ended March 31, 2022") or a table column header.

This module fetches the filing HTML via edgartools, converts it to markdown
(edgartools' to_markdown preserves the heading hierarchy as '#'..'######'),
walks the document top-to-bottom, and PROPAGATES the period label from the most
recent period-bearing heading down onto every text chunk beneath it — until a
new heading with a different period is reached. Paragraphs that appear before any
period-bearing heading get fy_label=None: those are the hard implicit-tier cases
the thesis is about.

Output chunk dicts mirror xbrl_chunker.py's shape (text + flat metadata keys) so
both tiers can be embedded / tagged / indexed through one downstream path. The
'source' key ("html") and 'section_heading' key distinguish this tier.

Usage:
    from ingestion.html_chunker import fetch_html_chunks
    chunks = fetch_html_chunks("AAPL", form_type="10-Q")
    # chunks[0]["text"]            -> paragraph for embedding / tagging
    # chunks[0]["fy_label"]        -> propagated label or None (implicit tier)
    # chunks[0]["section_heading"] -> the heading that supplied the label

FILE MAP
  L001-L049  Module docstring + this map
  L053-L087  CONFIG knobs (CHANGE ME / TWEAK)
  L090-L127  Period-from-heading extraction (_extract_period_from_heading)
  L130-L189  Markdown walk + label propagation (_walk_markdown)
  L192-L229  Filing fetch via edgartools (_fetch_filing, _filing_metadata)
  L232-L275  Public API (fetch_html_chunks)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import edgar

import config
from evaluation.tagger import MONTH_END_TO_Q, MONTH_TO_NUM, make_label, norm_year, regex_tag

logger = logging.getLogger(__name__)


# ============================== CONFIG ======================================
# CHANGE ME: minimum character length for a markdown line to count as a chunk.
# Filters out nav crumbs, page numbers, lone table-border rows, and short
# headings that carry no narrative ("Gross Margin", "PART I"). Raise it to be
# stricter (fewer, longer chunks); lower it to keep terse sentences.
MIN_CHUNK_LENGTH = 60

# TWEAK: ordered list of heading-period patterns to try, AFTER regex_tag (from
# evaluation/tagger.py) has had first refusal. regex_tag already covers the
# canonical "(quarter|three months|period) ended <Month> <day>, <year>",
# "year ended ...", and "Q<n> <year>" surface forms. The patterns here add the
# YTD / multi-month forms regex_tag deliberately omits ("Six Months Ended",
# "Nine Months Ended") — these map to the quarter their END MONTH falls in.
# Each entry: (compiled_regex, builder(match) -> fy_label_or_None).
# Add a pattern here if a registrant uses a heading surface form both layers miss.
def _ytd_months_ended(m: re.Match) -> Optional[str]:
    """'<N> Months Ended <Month> <day>, <year>' → label by END-MONTH's quarter."""
    month_num = MONTH_TO_NUM.get(m.group("month").lower())
    quarter = MONTH_END_TO_Q.get(month_num) if month_num else None
    if not quarter:
        return None
    return make_label(norm_year(m.group("year")), quarter)


HEADING_PERIOD_PATTERNS: list[tuple[re.Pattern, "callable"]] = [
    (
        re.compile(
            r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b"
            r"[\s-]*months?\s+ended\s+(?P<month>[A-Za-z]+)\s+\d{1,2},?\s*(?P<year>\d{4})",
            re.I,
        ),
        _ytd_months_ended,
    ),
]
# ============================================================================


# ====================== PERIOD-FROM-HEADING EXTRACTION ======================
# WHY two layers: regex_tag is the project's shared, tested surface-form tagger;
# reusing it keeps heading parsing consistent with how chunk TEXT is tagged
# elsewhere. The supplementary patterns only catch YTD month forms regex_tag
# intentionally leaves out, so we never diverge on the cases it does handle.

def _extract_period_from_heading(heading: str) -> Optional[str]:
    """
    Extract a fiscal label (e.g. "FY2022-Q1", "FY2022") from a section heading.

    Returns None when the heading carries no recoverable period — that is the
    correct signal that chunks beneath it cannot inherit a period from here.

    Examples:
        "Three Months Ended March 31, 2022"      -> "FY2022-Q1"
        "Nine Months Ended September 30, 2023"   -> "FY2023-Q3"  (YTD → end-month Q)
        "Year Ended December 31, 2022"           -> "FY2022"
        "Fiscal Year Ended January 28, 2024"     -> "FY2024"
        "For the Fiscal Quarter Ended March 28, 2026" -> "FY2026-Q1"
        "Gross Margin"                           -> None
    """
    # Strip markdown heading markers / table pipes so the taggers see clean text.
    clean = heading.lstrip("#").replace("|", " ").strip()

    # Layer 1: the shared, tested tagger gets first refusal.
    label = regex_tag(clean)
    if label:
        return label

    # Layer 2: YTD / multi-month forms regex_tag deliberately omits.
    for pattern, build in HEADING_PERIOD_PATTERNS:
        m = pattern.search(clean)
        if m:
            built = build(m)
            if built:
                return built
    return None
# ============================================================================


# ===================== MARKDOWN WALK + LABEL PROPAGATION ====================
# WHY markdown: edgartools' to_markdown preserves the filing's heading hierarchy
# as leading '#' runs and keeps table column-header rows inline (as '|'-pipe
# lines). That gives us exactly the two period sources the spec calls out —
# section headings and table column headers — in a single linear stream we can
# walk once, carrying the current period forward.

_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S")
_TABLE_ROW_RE = re.compile(r"^\s*\|")


def _is_heading(line: str) -> bool:
    return bool(_HEADING_RE.match(line))


def _walk_markdown(markdown: str, base_meta: dict) -> list[dict]:
    """
    Walk markdown lines top-to-bottom, propagating the period from the most
    recent period-bearing heading (or table header row) onto each text chunk.

    Propagation rule (spec Step 3): a heading's label applies to every chunk
    beneath it UNTIL a later heading resolves to a DIFFERENT label. A heading
    that resolves to no label does NOT clear the current one — narrative
    subheadings ("Gross Margin") sit under a dated parent and must keep it.
    """
    current_label: Optional[str] = None
    current_heading: Optional[str] = None
    chunks: list[dict] = []

    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Headings AND table column-header rows can both supply a period.
        is_structural = _is_heading(line) or _TABLE_ROW_RE.match(line)
        if is_structural:
            label = _extract_period_from_heading(line)
            if label and label != current_label:
                current_label = label
                current_heading = line.lstrip("#").replace("|", " ").strip()
            # A heading is structure, not narrative — never emit it as a chunk.
            if _is_heading(line):
                continue

        # Skip table separator / border rows and anything too short to be prose.
        if _TABLE_ROW_RE.match(line) and set(line) <= set("|-: "):
            continue
        if len(line) < MIN_CHUNK_LENGTH:
            continue

        chunks.append({
            "text": line,
            "fy_label": current_label,
            "source": "html",
            "section_heading": current_heading if current_label else None,
            **base_meta,
        })
    return chunks
# ============================================================================


# ========================= FILING FETCH (edgartools) ========================
_EDGAR_CONFIGURED = False


def _configure_edgar() -> None:
    """Set SEC-required User-Agent once per process (mirrors xbrl_loader)."""
    global _EDGAR_CONFIGURED
    if _EDGAR_CONFIGURED:
        return
    edgar.set_identity(config.SEC_USER_AGENT)
    _EDGAR_CONFIGURED = True
    logger.debug("edgartools identity set: %s", config.SEC_USER_AGENT)


def _filing_metadata(filing, company) -> dict:
    """Flat metadata shared by every chunk from one filing (mirrors xbrl keys)."""
    period_end = getattr(filing, "period_of_report", None)
    return {
        "ticker": None,  # filled by caller (filing object has no ticker)
        "entity": getattr(company, "name", None),
        "cik": str(getattr(company, "cik", "")).zfill(10) if getattr(company, "cik", None) else None,
        "accession": getattr(filing, "accession_number", None),
        "form_type": getattr(filing, "form", None),
        "period_end": str(period_end) if period_end else None,
    }


def _fetch_filing(ticker: str | int, form_type: str, limit: int):
    """
    Yield (filing, company) pairs for the most-recent *limit* filings of
    *form_type*. Isolated so tests can monkeypatch it without touching edgartools.
    """
    _configure_edgar()
    company = edgar.Company(ticker)
    filings = company.get_filings(form=form_type).head(limit)
    for filing in filings:
        yield filing, company
# ============================================================================


# ============================== PUBLIC API ==================================

def fetch_html_chunks(
    ticker: str | int,
    form_type: str = "10-K",      # TWEAK: "10-Q" for quarterly MD&A narrative
    limit: int = 1,               # TWEAK: how many recent filings to fetch
) -> list[dict]:
    """
    Fetch and parse SEC filing HTML for *ticker*, returning period-labeled chunks.

    Each chunk dict carries:
        text            — paragraph / table-row text for embedding or tagging
        fy_label        — period propagated from the heading above, or None
        source          — always "html"
        section_heading — heading that supplied fy_label (None if fy_label is None)
        ticker / entity / cik / accession / form_type / period_end — filing metadata

    Returns an empty list (never raises) if edgartools cannot fetch or parse the
    filing, or if the markdown body is empty — callers treat empty as "no data".
    """
    chunks: list[dict] = []
    try:
        for filing, company in _fetch_filing(ticker, form_type, limit):
            markdown = filing.markdown()
            if not markdown or not markdown.strip():
                logger.warning(
                    "Empty markdown for %s %s (%s) — skipping",
                    ticker, form_type, getattr(filing, "accession_number", "?"),
                )
                continue
            meta = _filing_metadata(filing, company)
            meta["ticker"] = str(ticker)
            chunks.extend(_walk_markdown(markdown, meta))
    except Exception as exc:  # network / parse / lookup failure → empty, not crash
        logger.error("fetch_html_chunks failed for %s %s: %s", ticker, form_type, exc)
        return []

    labeled = sum(1 for c in chunks if c["fy_label"])
    logger.info(
        "%s %s | %d chunks (%d labeled, %d implicit)",
        ticker, form_type, len(chunks), labeled, len(chunks) - labeled,
    )
    return chunks
# ============================================================================
