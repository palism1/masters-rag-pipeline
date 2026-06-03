"""
xbrl_loader.py — XBRL structured-facts loader (add-alongside path).

Sources every fact from SEC EDGAR's CompanyFacts API via edgartools, so each
fact carries concept, entity, period, unit, and value straight from the
machine-readable XBRL layer — no HTML/PDF scraping, no period/scale guessing.

Key guarantees:
  - SEC User-Agent set from config before any network call (no silent 403s).
  - Dedup: collapses (concept, period_start, period_end, unit, value) — the
    CompanyFacts API re-reports prior quarters inside 10-Ks; this kills doubles.
  - DRY_RUN=True (default): loads and logs facts but returns them WITHOUT
    performing any DB / vector-store write. Callers must check config.DRY_RUN.
  - scale: edgartools sets this from the XBRL document's precision attribute;
    numeric_value (used for all arithmetic) is always the raw base-unit amount.
  - Fiscal period (FY, Q1-Q3) and fiscal year come from edgartools' FinancialFact
    fields, which edgartools derives from the EDGAR fy/fp tags — so Apple's Q1
    (Oct-Dec) labels correctly as FY<year>-Q1 without any separate calendar lookup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import edgar

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default concept list (US-GAAP income-statement + per-share).
# Extend this list to pull additional line items.
# ---------------------------------------------------------------------------
DEFAULT_CONCEPTS: list[str] = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",  # ASC 606 revenue
    "Revenues",                                             # older revenue tag
    "NetIncomeLoss",
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
]

DEFAULT_FORM_TYPES: frozenset[str] = frozenset({"10-K", "10-Q"})

# When the same (concept, period, unit, value) appears in multiple filing types,
# prefer the more authoritative one. 10-K audited > 10-Q reviewed > amendments.
_FORM_PRIORITY: dict[str, int] = {
    "10-K": 4,
    "10-K/A": 3,
    "10-Q": 2,
    "10-Q/A": 1,
}

_EDGAR_CONFIGURED = False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class XbrlFact:
    concept: str
    taxonomy: str          # e.g. "us-gaap"
    entity: str            # company name as registered
    cik: str               # zero-padded 10-digit CIK
    period_start: Optional[str]  # ISO date; None for instant (balance-sheet) facts
    period_end: str              # ISO date
    period_type: str       # "duration" | "instant"
    unit: str              # e.g. "USD", "shares", "pure"
    value: float           # raw, unscaled value in base unit
    scale: int             # always 0 — CompanyFacts API is already unscaled
    form_type: str         # "10-K" | "10-Q" | …
    accession: str         # SEC accession number (filing identifier)
    fiscal_year: Optional[int]   # as reported by the company (e.g. 2024)
    fiscal_period: Optional[str] # "Q1" | "Q2" | "Q3" | "FY" (annual)

    @property
    def dedup_key(self) -> tuple:
        """Canonical identity of one time-series point across all filings."""
        return (self.concept, self.period_start, self.period_end, self.unit, self.value)

    @property
    def fy_label(self) -> Optional[str]:
        """Human label: 'FY2024-Q1' or 'FY2023'. None if fy/fp absent."""
        if not (self.fiscal_year and self.fiscal_period):
            return None
        if self.fiscal_period == "FY":
            return f"FY{self.fiscal_year}"
        return f"FY{self.fiscal_year}-{self.fiscal_period}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _configure_edgar() -> None:
    """Set SEC-required User-Agent. Called once per process."""
    global _EDGAR_CONFIGURED
    if _EDGAR_CONFIGURED:
        return
    edgar.set_identity(config.SEC_USER_AGENT)
    _EDGAR_CONFIGURED = True
    logger.debug("edgartools identity set: %s", config.SEC_USER_AGENT)




_FY_DRIFT = timedelta(days=14)  # 52/53-week calendars can shift the year-start by ~6 days


def _build_fy_bounds(facts: list[XbrlFact]) -> dict[tuple[str, int], tuple[str, str]]:
    """
    Return {(cik, fiscal_year): (fy_start, fy_end)} from trusted annual facts.

    Two guards:
    1. period_end.year must equal fiscal_year — rejects mislabeled annual
       comparatives (FY2007 period tagged fy=2009 because it appeared in the
       FY2009 10-K alongside the current-year data).
    2. Period must span ≥300 days — rejects early EDGAR filings where quarterly
       periods were incorrectly tagged fp="FY", which would corrupt the calendar
       and cause valid quarterly facts to be filtered out downstream.
    """
    bounds: dict[tuple[str, int], tuple[str, str]] = {}
    for f in facts:
        if f.fiscal_period != "FY" or not f.fiscal_year or not f.period_start:
            continue
        if int(f.period_end[:4]) != f.fiscal_year:
            continue
        duration = (date.fromisoformat(f.period_end) - date.fromisoformat(f.period_start)).days
        if duration < 300:
            continue  # not a real annual period (fp="FY" on a quarterly fact)
        key = (f.cik, f.fiscal_year)
        if key not in bounds:
            bounds[key] = (f.period_start, f.period_end)
    return bounds


def _filter_comparative(facts: list[XbrlFact]) -> list[XbrlFact]:
    """
    Drop comparative-period facts mislabeled with the filing's fiscal year.

    EDGAR tags every fact in a filing with the filing's fy/fp — including the
    prior-year comparison column required by SEC rules. We validate each duration
    fact against the fiscal year boundaries derived from trusted annual facts.

    The 14-day start tolerance handles 52/53-week fiscal calendars where the
    year-start shifts slightly (e.g. NVDA's late-January year-end drifts a week).
    Falls back to a year-heuristic when no annual bounds exist for that year.
    Instant facts (no period_start) are always kept.
    """
    fy_bounds = _build_fy_bounds(facts)
    kept: list[XbrlFact] = []
    dropped = 0
    for f in facts:
        if f.period_type != "duration" or not f.period_start or not f.fiscal_year:
            kept.append(f)
            continue
        key = (f.cik, f.fiscal_year)
        if key in fy_bounds:
            fy_start, fy_end = fy_bounds[key]
            start_ok = date.fromisoformat(f.period_start) >= date.fromisoformat(fy_start) - _FY_DRIFT
            end_ok   = f.period_end <= fy_end
            if start_ok and end_ok:
                kept.append(f)
            else:
                dropped += 1
        else:
            # No trusted annual fact for this year — fall back to year heuristic
            if int(f.period_start[:4]) >= f.fiscal_year - 1:
                kept.append(f)
            else:
                dropped += 1
    if dropped:
        logger.debug("Dropped %d comparative-period facts (period outside fiscal year bounds)", dropped)
    return kept


def _dedup(candidates: list[XbrlFact]) -> list[XbrlFact]:
    """
    Collapse facts that share (concept, period_start, period_end, unit, value).

    When the same fact appears under multiple form types (10-Q data re-stated
    inside a 10-K), keep the entry from the most authoritative filing.
    """
    best: dict[tuple, XbrlFact] = {}
    for fact in candidates:
        key = fact.dedup_key
        if key not in best:
            best[key] = fact
        else:
            if _FORM_PRIORITY.get(fact.form_type, 0) > _FORM_PRIORITY.get(best[key].form_type, 0):
                best[key] = fact
    return list(best.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_company_facts(
    ticker: str | int,
    concepts: list[str] | None = None,
    form_types: frozenset[str] | None = None,
) -> list[XbrlFact]:
    """
    Fetch and return deduplicated XBRL facts for *ticker*.

    Parameters
    ----------
    ticker:
        Exchange ticker symbol (e.g. "AAPL", "MSFT") or integer CIK.
        Use CIK for companies whose ticker is not resolvable by edgartools
        (e.g. Foot Locker CIK=850209, Activision CIK=718877).
    concepts:
        US-GAAP concept names to include. Defaults to DEFAULT_CONCEPTS.
    form_types:
        Filing types to accept. Defaults to {"10-K", "10-Q"}.

    Returns
    -------
    List of XbrlFact, one entry per unique time-series point.
    In DRY_RUN mode the list is still returned — the caller is responsible
    for checking config.DRY_RUN before passing facts to any write path.
    """
    _configure_edgar()

    target_concepts = set(concepts or DEFAULT_CONCEPTS)
    target_forms = form_types or DEFAULT_FORM_TYPES

    company = edgar.Company(ticker)
    entity_name: str = company.name
    cik: str = str(company.cik).zfill(10)

    logger.info("Loading XBRL facts for %s (CIK %s) — concepts: %s", ticker, cik, sorted(target_concepts))

    entity_facts = company.get_facts()

    # EntityFacts implements __iter__ over its internal List[FinancialFact].
    # FinancialFact fields used here:
    #   concept      — plain US-GAAP name, no taxonomy prefix
    #   taxonomy     — "us-gaap", "dei", etc.
    #   period_start / period_end — date objects (convert to ISO str)
    #   period_type  — "instant" | "duration"
    #   numeric_value — float in the base unit (USD, shares); use over .value
    #   scale        — precision hint from the XBRL document (may be None)
    #   fiscal_year  — int; 0 means not set → None
    #   fiscal_period — str; '' means not set → None
    candidates: list[XbrlFact] = []
    for ff in entity_facts:
        # FinancialFact.concept is prefixed: "us-gaap:ConceptName". Strip for matching.
        plain_concept = ff.concept.split(":")[-1] if ":" in ff.concept else ff.concept
        if plain_concept not in target_concepts:
            continue
        if ff.form_type not in target_forms:
            continue
        if not ff.period_end:
            continue

        try:
            numeric_val = ff.numeric_value if ff.numeric_value is not None else float(ff.value)
        except (ValueError, TypeError):
            logger.debug("Skipping non-numeric fact: concept=%s value=%r", plain_concept, ff.value)
            continue

        candidates.append(
            XbrlFact(
                concept=plain_concept,  # stored without taxonomy prefix
                taxonomy=ff.taxonomy,
                entity=entity_name,
                cik=cik,
                period_start=ff.period_start.isoformat() if ff.period_start else None,
                period_end=ff.period_end.isoformat(),
                period_type=ff.period_type,
                unit=ff.unit,
                value=numeric_val,
                scale=ff.scale or 0,
                form_type=ff.form_type,
                accession=ff.accession,
                fiscal_year=ff.fiscal_year if ff.fiscal_year else None,
                fiscal_period=ff.fiscal_period if ff.fiscal_period else None,
            )
        )

    facts = _dedup(candidates)
    facts = _filter_comparative(facts)

    if config.DRY_RUN:
        logger.info(
            "DRY_RUN | %s | %d raw entries → %d deduplicated facts (no writes)",
            ticker, len(candidates), len(facts),
        )
        for f in sorted(facts, key=lambda x: (x.concept, x.period_end))[:10]:
            logger.info(
                "  %s | %s → %s | %s | %s %g",
                f.concept, f.period_start or "instant", f.period_end,
                f.fy_label or "?", f.unit, f.value,
            )
        if len(facts) > 10:
            logger.info("  … and %d more", len(facts) - 10)
    else:
        logger.info("%s | %d facts loaded", ticker, len(facts))

    return facts
