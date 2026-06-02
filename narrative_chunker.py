"""
narrative_chunker.py — MD&A-style prose variants of XBRL facts.

Unlike xbrl_chunker.py, the fiscal label is NOT embedded in the text.
Periods are described using calendar-date phrasing only:
    "For the three months ended December 30, 2023, Apple reported net revenue of $119.6B."

This is the harder test: the tagger must map a calendar-date phrase to the
correct fiscal label for a company that may not follow a December year-end.
Regex will map December → Q4 by default; Apple's fiscal Q1 ends in December,
so for Apple every Q prediction is off by one label.

Ground-truth fy_label is preserved as chunk metadata.
"""

from __future__ import annotations

from datetime import date as _date

from xbrl_chunker import _humanize_value
from xbrl_loader import XbrlFact

_CONCEPT_TO_PROSE: dict[str, str] = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": "net revenue",
    "Revenues":                "revenues",
    "NetIncomeLoss":           "net income",
    "EarningsPerShareBasic":   "basic earnings per share",
    "EarningsPerShareDiluted": "diluted earnings per share",
}

_MONTH_NAMES: dict[int, str] = {
    1: "January",   2: "February",  3: "March",    4: "April",
    5: "May",       6: "June",      7: "July",     8: "August",
    9: "September", 10: "October",  11: "November", 12: "December",
}


def _prose_date(iso_date: str) -> str:
    d = _date.fromisoformat(iso_date)
    return f"{_MONTH_NAMES[d.month]} {d.day}, {d.year}"


def _prose_concept(concept: str) -> str:
    return _CONCEPT_TO_PROSE.get(concept, concept.lower())


def _title_entity(entity: str) -> str:
    return entity.title()


def facts_to_narrative_chunks(facts: list[XbrlFact]) -> list[dict]:
    """
    Convert XbrlFacts into MD&A-style prose chunks.

    The fiscal label (FY2024-Q1) is intentionally absent from the text.
    Period is expressed as calendar date only so the tagger cannot cheat.
    """
    chunks = []
    for f in facts:
        concept = _prose_concept(f.concept)
        value   = _humanize_value(f.value, f.unit)
        entity  = _title_entity(f.entity)
        end     = _prose_date(f.period_end)

        if f.period_type == "instant":
            text = f"As of {end}, {entity} had {concept} of {value}."
        elif f.fiscal_period == "FY":
            text = (
                f"For the fiscal year ended {end}, "
                f"{entity} reported {concept} of {value}."
            )
        else:
            text = (
                f"For the three months ended {end}, "
                f"{entity} reported {concept} of {value}."
            )

        chunks.append({
            "text":          text,
            "fy_label":      f.fy_label,
            "concept":       f.concept,
            "taxonomy":      f.taxonomy,
            "entity":        f.entity,
            "cik":           f.cik,
            "period_start":  f.period_start,
            "period_end":    f.period_end,
            "period_type":   f.period_type,
            "unit":          f.unit,
            "value":         f.value,
            "form_type":     f.form_type,
            "accession":     f.accession,
            "fiscal_year":   f.fiscal_year,
            "fiscal_period": f.fiscal_period,
        })
    return chunks
