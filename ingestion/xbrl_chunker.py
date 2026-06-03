"""
xbrl_chunker.py — Convert XbrlFact objects into RAG-ready text chunks.

Each chunk is a dict with a natural-language 'text' field (suitable as input
to any embedding model or period tagger) plus all raw metadata fields as
separate keys. The 'fy_label' key (e.g. "FY2024-Q1") is the ground-truth
period label — direct drop-in for the 'true_label' column in CURATED_DATASET.

Usage:
    from xbrl_loader import load_company_facts
    from xbrl_chunker import facts_to_chunks

    facts = load_company_facts("AAPL")
    chunks = facts_to_chunks(facts)
    # chunks[0]["text"]     -> sentence for embedding / tagging
    # chunks[0]["fy_label"] -> ground-truth label for evaluation
"""

from __future__ import annotations

from .xbrl_loader import XbrlFact


def _humanize_value(value: float, unit: str) -> str:
    """Format raw value into a readable string with unit suffix."""
    if unit == "USD":
        sign  = "-" if value < 0 else ""
        abs_v = abs(value)
        if abs_v >= 1e9:
            return f"{sign}${abs_v / 1e9:.3f}B"
        if abs_v >= 1e6:
            return f"{sign}${abs_v / 1e6:.3f}M"
        if abs_v >= 1e3:
            return f"{sign}${abs_v / 1e3:.3f}K"
        return f"{sign}${abs_v:,.2f}"
    if unit == "shares":
        abs_v = abs(value)
        if abs_v >= 1e9:
            return f"{value / 1e9:.3f}B shares"
        if abs_v >= 1e6:
            return f"{value / 1e6:.3f}M shares"
        return f"{value:,.0f} shares"
    # USD/share, pure ratios, etc.
    return f"{value:g} {unit}"


def _period_phrase(fact: XbrlFact) -> str:
    """
    Build the period description embedded in the chunk text.

    For duration facts with EDGAR fiscal tags: include both the raw date range
    and the standardized fiscal label so the period tagger has both signals.
    For instant (balance-sheet) facts: use 'as of <date>'.
    """
    if fact.period_type == "instant":
        label = f" ({fact.fy_label})" if fact.fy_label else ""
        return f"as of {fact.period_end}{label}"

    if fact.fy_label:
        return (
            f"for the period {fact.period_start} to {fact.period_end} "
            f"({fact.fy_label})"
        )
    return f"for the period {fact.period_start} to {fact.period_end}"


def facts_to_chunks(facts: list[XbrlFact]) -> list[dict]:
    """
    Convert a list of XbrlFact into RAG-ready chunk dicts.

    Each dict keys:
        text        — natural-language sentence; feed to embedder / period tagger
        fy_label    — ground-truth fiscal label ("FY2024-Q1", "FY2023", None)
        concept     — XBRL concept name
        taxonomy    — e.g. "us-gaap"
        entity      — company name
        cik         — 10-digit CIK string
        period_start / period_end — ISO date strings
        period_type — "duration" | "instant"
        unit        — "USD", "shares", etc.
        value       — raw numeric value (float)
        form_type   — "10-K" | "10-Q" | …
        accession   — SEC accession number
        fiscal_year / fiscal_period — as reported in EDGAR (int / str)
    """
    chunks = []
    for f in facts:
        text = (
            f"{f.entity} reported {f.concept} of "
            f"{_humanize_value(f.value, f.unit)} "
            f"{_period_phrase(f)}."
        )
        chunks.append({
            "text": text,
            "fy_label": f.fy_label,
            "concept": f.concept,
            "taxonomy": f.taxonomy,
            "entity": f.entity,
            "cik": f.cik,
            "period_start": f.period_start,
            "period_end": f.period_end,
            "period_type": f.period_type,
            "unit": f.unit,
            "value": f.value,
            "form_type": f.form_type,
            "accession": f.accession,
            "fiscal_year": f.fiscal_year,
            "fiscal_period": f.fiscal_period,
        })
    return chunks
