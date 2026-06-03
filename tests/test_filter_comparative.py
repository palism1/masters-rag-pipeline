"""
tests/test_filter_comparative.py — Unit tests for the EDGAR comparative-period filter.

Background: EDGAR tags every fact in a filing with the filing's fy/fp — including
the mandatory prior-year comparison column. A 10-Q for Q3 FY2009 contains Q3 FY2008
data for comparison, but both rows carry fy=2009, fp="Q3". _filter_comparative()
detects and drops these mislabeled facts using fiscal year boundaries derived from
trusted annual (10-K) facts.

This is a thesis finding: even EDGAR's authoritative structured data requires a
validation layer. The tests here document exactly which cases are caught and why.

All tests use synthetic XbrlFact objects — no live EDGAR calls, no network.

FILE MAP
  L001–L032  Module docstring + file map
  L034–L065  Imports + CONFIG + synthetic fact factory
  L067–L110  Tests for _build_fy_bounds() — which annual facts are trusted
  L112–L210  Tests for _filter_comparative() — which facts survive the filter
"""

from __future__ import annotations

import os
os.environ["DRY_RUN"] = "true"

import pytest
from ingestion.xbrl_loader import XbrlFact, _build_fy_bounds, _filter_comparative

# ===========================================================================
# CONFIG — shared test constants
# All dates use AAPL's fiscal calendar: FY ends late September.
# FY2009: 2008-09-28 → 2009-09-26
# FY2008: 2007-09-30 → 2008-09-27
# ===========================================================================

CIK     = "0000320193"    # Apple CIK, zero-padded
FY2009  = 2009
FY2008  = 2008

# Trusted fiscal year boundaries (derived from the 10-K)
FY2009_START = "2008-09-28"
FY2009_END   = "2009-09-26"
FY2008_START = "2007-09-30"
FY2008_END   = "2008-09-27"

# ===========================================================================


def _fact(**kwargs) -> XbrlFact:
    """
    Build a synthetic XbrlFact with sensible defaults.
    Override any field via keyword argument.
    """
    defaults = dict(
        concept      = "NetIncomeLoss",
        taxonomy     = "us-gaap",
        entity       = "APPLE INC",
        cik          = CIK,
        period_start = FY2009_START,
        period_end   = FY2009_END,
        period_type  = "duration",
        unit         = "USD",
        value        = 1_000_000_000.0,
        scale        = 0,
        form_type    = "10-K",
        accession    = "0000320193-09-000001",
        fiscal_year  = FY2009,
        fiscal_period= "FY",
    )
    defaults.update(kwargs)
    return XbrlFact(**defaults)


# ===========================================================================
# _build_fy_bounds — which annual facts are trusted as calendar anchors
# ===========================================================================

class TestBuildFyBounds:

    def test_valid_annual_included(self):
        # Canonical 10-K: period_end year matches fiscal_year, duration ~365 days
        annual = _fact(period_start=FY2009_START, period_end=FY2009_END,
                       fiscal_year=FY2009, fiscal_period="FY", form_type="10-K")
        bounds = _build_fy_bounds([annual])
        assert (CIK, FY2009) in bounds
        assert bounds[(CIK, FY2009)] == (FY2009_START, FY2009_END)

    def test_annual_comparative_mislabel_excluded(self):
        # FY2007 annual data that appeared in FY2009 10-K comparison column.
        # period_end year (2008) != fiscal_year (2009) → rejected by guard 1.
        mislabeled = _fact(period_start=FY2008_START, period_end=FY2008_END,
                           fiscal_year=FY2009, fiscal_period="FY", form_type="10-K")
        bounds = _build_fy_bounds([mislabeled])
        assert (CIK, FY2009) not in bounds

    def test_short_duration_annual_excluded(self):
        # Early EDGAR filings sometimes tagged quarterly facts as fp="FY".
        # Duration < 300 days → rejected by guard 2, preventing it from
        # corrupting the fiscal year calendar used downstream.
        fake_annual = _fact(period_start="2009-07-01", period_end="2009-09-26",
                            fiscal_year=FY2009, fiscal_period="FY")
        bounds = _build_fy_bounds([fake_annual])
        assert (CIK, FY2009) not in bounds

    def test_quarterly_fact_not_used_for_bounds(self):
        # Non-FY fiscal_period is ignored entirely — only annual facts anchor the calendar
        quarterly = _fact(period_start="2008-09-28", period_end="2008-12-27",
                          fiscal_year=FY2009, fiscal_period="Q1", form_type="10-Q")
        bounds = _build_fy_bounds([quarterly])
        assert (CIK, FY2009) not in bounds

    def test_instant_fact_not_used_for_bounds(self):
        # Instant facts have no period_start — cannot establish a range
        instant = _fact(period_start=None, period_end=FY2009_END,
                        period_type="instant", fiscal_year=FY2009, fiscal_period="FY")
        bounds = _build_fy_bounds([instant])
        assert (CIK, FY2009) not in bounds

    def test_first_entry_wins_for_same_year(self):
        # Two valid annual facts for the same (cik, year) — first one is kept.
        # In practice this happens when both 10-K and 10-K/A are present.
        first  = _fact(period_start=FY2009_START, period_end=FY2009_END,
                       fiscal_year=FY2009, fiscal_period="FY", accession="AAA")
        second = _fact(period_start="2008-10-01", period_end=FY2009_END,
                       fiscal_year=FY2009, fiscal_period="FY", accession="BBB")
        bounds = _build_fy_bounds([first, second])
        assert bounds[(CIK, FY2009)][0] == FY2009_START


# ===========================================================================
# _filter_comparative — which facts survive the period validation
# ===========================================================================

class TestFilterComparative:

    def _facts_with_annual(self, *extra_facts) -> list[XbrlFact]:
        """Return a trusted annual fact plus any extra facts for the same company."""
        annual = _fact(period_start=FY2009_START, period_end=FY2009_END,
                       fiscal_year=FY2009, fiscal_period="FY", form_type="10-K")
        return [annual, *extra_facts]

    # -----------------------------------------------------------------------
    # Facts that should be KEPT
    # -----------------------------------------------------------------------

    def test_correct_quarterly_kept(self):
        # Q1 FY2009 data: period starts inside the FY2009 window
        q1 = _fact(period_start="2008-09-28", period_end="2008-12-27",
                   fiscal_year=FY2009, fiscal_period="Q1", form_type="10-Q")
        result = _filter_comparative(self._facts_with_annual(q1))
        assert any(f.fiscal_period == "Q1" for f in result)

    def test_instant_fact_always_kept(self):
        # Balance-sheet snapshot has no period_start — always passes the filter.
        # The mislabeling issue only affects duration facts with a comparison range.
        instant = _fact(period_start=None, period_end=FY2009_END,
                        period_type="instant", fiscal_year=FY2009, fiscal_period="Q4")
        result = _filter_comparative(self._facts_with_annual(instant))
        assert any(f.period_type == "instant" for f in result)

    def test_fact_with_no_fiscal_year_kept(self):
        # If fy is absent we have no basis to validate — keep rather than drop
        no_fy = _fact(fiscal_year=None, fiscal_period=None,
                      period_start=FY2009_START, period_end=FY2009_END)
        result = _filter_comparative([no_fy])
        assert len(result) == 1

    def test_52week_drift_within_tolerance_kept(self):
        # 52/53-week calendars shift the year-start by up to ~6 days.
        # _FY_DRIFT allows up to 14 days before fy_start — this should pass.
        drifted_start = "2008-09-22"   # 6 days before FY2009_START "2008-09-28"
        drifted = _fact(period_start=drifted_start, period_end="2008-12-27",
                        fiscal_year=FY2009, fiscal_period="Q1", form_type="10-Q")
        result = _filter_comparative(self._facts_with_annual(drifted))
        assert any(f.period_start == drifted_start for f in result)

    def test_annual_fact_itself_kept(self):
        # The trusted annual that anchors the bounds must also survive filtering
        annual = _fact(period_start=FY2009_START, period_end=FY2009_END,
                       fiscal_year=FY2009, fiscal_period="FY", form_type="10-K")
        result = _filter_comparative([annual])
        assert any(f.fiscal_period == "FY" for f in result)

    # -----------------------------------------------------------------------
    # Facts that should be DROPPED
    # -----------------------------------------------------------------------

    def test_comparative_quarterly_dropped(self):
        # The core mislabeling case: FY2008-Q1 data that appeared in the FY2009-Q1
        # 10-Q comparison column. EDGAR tagged it fy=2009, fp="Q1".
        # Its period_start (2007-09-30) predates FY2009_START by over a year.
        mislabeled = _fact(
            period_start = FY2008_START,   # "2007-09-30" — clearly FY2008
            period_end   = "2007-12-29",   # Q1 of FY2008
            fiscal_year  = FY2009,         # wrongly tagged as FY2009
            fiscal_period= "Q1",
            form_type    = "10-Q",
        )
        result = _filter_comparative(self._facts_with_annual(mislabeled))
        # The mislabeled Q1 should be gone; only the annual and nothing else
        periods = [f.fiscal_period for f in result]
        assert "Q1" not in periods

    def test_comparative_annual_dropped(self):
        # FY2008 annual data appearing in the FY2009 10-K comparison column,
        # mislabeled as FY2009. period_start clearly outside FY2009 window.
        mislabeled_annual = _fact(
            period_start = FY2008_START,   # "2007-09-30"
            period_end   = FY2008_END,     # "2008-09-27"
            fiscal_year  = FY2009,         # wrong label
            fiscal_period= "FY",
            form_type    = "10-K",
        )
        # Note: _build_fy_bounds rejects this as anchor (period_end year ≠ fiscal_year),
        # so it gets evaluated as a regular duration fact against bounds and dropped.
        annual_correct = _fact(period_start=FY2009_START, period_end=FY2009_END,
                               fiscal_year=FY2009, fiscal_period="FY", form_type="10-K",
                               accession="correct-accession")
        result = _filter_comparative([annual_correct, mislabeled_annual])
        # Only the correctly-dated annual should survive
        assert len(result) == 1
        assert result[0].accession == "correct-accession"

    def test_drift_exceeding_tolerance_dropped(self):
        # 20 days before fy_start — outside the 14-day _FY_DRIFT tolerance.
        # Not a 52-week calendar issue; this is a genuine comparative mislabel.
        too_early = "2008-09-08"   # 20 days before FY2009_START "2008-09-28"
        mislabeled = _fact(period_start=too_early, period_end="2008-12-27",
                           fiscal_year=FY2009, fiscal_period="Q1", form_type="10-Q")
        result = _filter_comparative(self._facts_with_annual(mislabeled))
        assert not any(f.period_start == too_early for f in result)

    # -----------------------------------------------------------------------
    # Fallback heuristic (no annual bounds available for the year)
    # -----------------------------------------------------------------------

    def test_fallback_recent_year_kept(self):
        # No annual facts → no bounds → fallback: period_start year >= fiscal_year - 1
        # 2004 >= 2005-1=2004 → kept
        fact = _fact(period_start="2004-10-01", period_end="2004-12-31",
                     fiscal_year=2005, fiscal_period="Q1")
        result = _filter_comparative([fact])
        assert len(result) == 1

    def test_fallback_old_comparative_dropped(self):
        # No bounds, period_start year (2003) < fiscal_year - 1 (2004) → dropped
        old = _fact(period_start="2003-10-01", period_end="2003-12-31",
                    fiscal_year=2005, fiscal_period="Q1")
        result = _filter_comparative([old])
        assert len(result) == 0

    # -----------------------------------------------------------------------
    # End-to-end: mixed bag (confirms counts are right)
    # -----------------------------------------------------------------------

    def test_mixed_keeps_and_drops(self):
        annual   = _fact(period_start=FY2009_START, period_end=FY2009_END,
                         fiscal_year=FY2009, fiscal_period="FY", form_type="10-K")
        good_q1  = _fact(period_start="2008-09-28", period_end="2008-12-27",
                         fiscal_year=FY2009, fiscal_period="Q1", form_type="10-Q")
        bad_q1   = _fact(period_start=FY2008_START, period_end="2007-12-29",
                         fiscal_year=FY2009, fiscal_period="Q1", form_type="10-Q",
                         value=999.0)
        instant  = _fact(period_start=None, period_end=FY2009_END,
                         period_type="instant", fiscal_year=FY2009, fiscal_period="Q4")

        result = _filter_comparative([annual, good_q1, bad_q1, instant])

        assert annual  in result   # trusted annual — kept
        assert good_q1 in result   # correct quarterly — kept
        assert bad_q1  not in result  # comparative mislabel — dropped
        assert instant in result   # instant fact — always kept
        assert len(result) == 3
