# Next Steps Plan — Fiscal-Period-Aware Financial RAG

_Last updated: 2026-06-02_

---

## Where we are

### Stage 1 — Complete

- XBRL ingestion from SEC EDGAR via `edgartools` — clean, deduplicated facts with ground-truth `fy_label` from EDGAR's own `fy`/`fp` fields
- Two chunking modes: structured XBRL format and MD&A-style narrative prose
- Two taggers benchmarked on real EDGAR data (AAPL, MSFT, GOOG, NVDA):

| Ticker | Mode | Regex | Similarity (all-MiniLM-L6-v2) |
|---|---|---|---|
| AAPL | XBRL | 54% | 44% |
| GOOG | XBRL | 19% | 41% |
| MSFT | XBRL | 55% | 42% |
| NVDA | XBRL | 52% | 55% |
| AAPL | Narrative | 35% | 41% |
| GOOG | Narrative | **77%** | 39% |
| MSFT | Narrative | 27% | 34% |
| NVDA | Narrative | 23% | 38% |

**Key results:**
- Regex ~54% on XBRL — extracts year but drops quarter from `FY2024-Q1`
- Similarity tagger 41–55% on real XBRL — WORSE than on synthetic data. All chunks
  for the same company are semantically near-identical; the embedding model cannot
  distinguish between periods even when the label is explicitly in the text
- GOOG narrative 77% vs AAPL/MSFT/NVDA 23–35% — non-calendar FY failure confirmed

### Phase 1 Steps 1–4 — Complete

**Step 1 — Index:** 13,416 documents, 32 companies, 5 income-statement concepts in Chroma.

**Step 2 — Query parser:** 99% ticker accuracy, 82% full filter coverage on FinanceBench questions.

**Step 3 — Retriever:** `retrieve_both()` validated. Smoke test:

| Question | Filtered top-3 | Baseline top-3 |
|---|---|---|
| PepsiCo net income Q1 2022 | FY2022-Q1 ✓ | FY2025-Q2, FY2023-Q2 ✗ |
| 3M revenue Q2 2023 | FY2023-Q2 ✓ | FY2025-Q2, FY2020-Q3 ✗ |
| JPMorgan EPS Q3 2022 | FY2022-Q3 ✓ | FY2020-Q3, FY2023-Q3 ✗ |

**Step 4 — Generation:** `generate_both()` validated. PepsiCo Q1 2022 — filtered: $4.261B HIGH confidence; baseline: cannot answer (retrieved FY2025-Q2 chunks).

**Data validation:** Parity test verifies 4 known values against official SEC filings (AAPL, MSFT, GOOG). Passes on every `pytest` run. Reference table needs expansion to cover FinanceBench companies before thesis submission.

---

## Remaining work

Priority order reflects what unblocks what. Items marked **[DECISION]** require the professor
conversation before proceeding.

---

### Priority 1 — Data validation (independent of any decision, do first)

**Expand the parity reference table** in `tests/test_xbrl_loader.py`.

Currently covers 4 facts across 3 companies. Before the thesis evaluation can be cited with
confidence, add manually verified values for FinanceBench companies — look up the exact figure
in the official SEC filing PDF, add the accession number, add the row.

Suggested additions (pick 2–3 per company from FinanceBench questions):
- PepsiCo net income Q1 2022 — 10-Q accession to be confirmed from EDGAR
- 3M revenue Q2 2023 — from the 10-Q filed in the Chroma index
- JPMorgan EPS Q3 2022 — from the 10-Q
- Adobe net income FY2022 — from the 10-K

This is manual work (look up the filing, read the number, add a row) but it is the only way to
make the claim "our data matches the official filings" cite-worthy in a thesis.

---

### Priority 2 — Scope decision **[DECISION — professor meeting]**

**Path A — Keep 5 concepts, focus on retrieval period accuracy**
- No index rebuild needed
- Primary metric: % of questions where filtered retriever returns the correct period's chunk vs baseline
- Answer accuracy will be low for both (coverage gap) — noted as a limitation
- Thesis scope: period-aware retrieval on income-statement XBRL facts

**Path B — Expand to ~15 concepts, enable answer accuracy comparison**
- Add: `OperatingIncomeLoss`, `GrossProfit`, `PaymentsToAcquirePropertyPlantAndEquipment` (CapEx),
  `NetCashProvidedByUsedInOperatingActivities`, `Assets`, `AssetsCurrent`, `LiabilitiesCurrent`,
  `InventoryNet`, `CashAndCashEquivalentsAtCarryingValue`
- Rebuild index (~20 min, same pipeline)
- Primary metric: retrieval period accuracy + answer accuracy gap (filtered vs baseline)
- Thesis scope: period-aware retrieval on common financial statement facts

Neither path requires changing the methodology. Both are defensible. Path B gives more evidence
for the same claim.

---

### Priority 3 — Full evaluation run (after scope decision)

```bash
python evaluation/eval_pipeline.py
```

Runs all 127 in-scope FinanceBench questions through `generate_both()`, scores each answer,
saves to `results/eval_results.json`. Takes ~5 minutes, costs ~$0.05–0.10 in API credits.
Resumable if interrupted.

Expected output:

| Metric | Filtered | Baseline |
|---|---|---|
| Retrieval period accuracy | ~80%+ | ~15–25% |
| Answer accuracy (strict) | TBD | TBD |
| Answer accuracy (lenient) | TBD | TBD |

---

### Priority 4 — Failure report HTML

```bash
python evaluation/make_retrieval_report.py
```

Generates `retrieval_failure_report.html` from the saved results. Shows every question where
filtered and baseline disagree — question, ground truth, chunks retrieved, both answers,
colour-coded correct/incorrect. This is the visual thesis argument.

---

### Priority 5 — Phase 2 (structure-aware HTML parsing)

_Goal: solve the implicit tier — fiscal period lives in a section heading or table column
header, not the chunk text._

**Step 6 — Document fetcher and structure parser**
- Use `edgartools.filing.markdown()` or `sec-parser` to retrieve 10-K/10-Q HTML
- Evaluate which tool preserves section headings and table column headers faithfully

**Step 7 — Structure-aware chunker with period propagation**
- Walk the semantic tree, inject `fiscal_period` metadata from headings/column headers
  down into child chunks
- Creates the implicit tier: chunks with no period in text but correct period in metadata

**Step 8 — Mixed index**
- Combine XBRL chunks (Phase 1) and HTML-derived chunks (Phase 2) in the same collection
- Retriever and generator unchanged

**Step 9 — Ablation study**

| Tier | Example | XBRL-only | + HTML |
|---|---|---|---|
| Explicit | "Q1 2024 net income was $X" | ✓ | ✓ |
| Variant | "fiscal '22 saw record revenue" | ✓ | ✓ |
| Implicit | "Revenue increased vs prior period" | ✗ | ✓ |

---

## Technical decisions made

| Decision | Choice | Reason |
|---|---|---|
| Vector store | Chroma | Zero infra, pure Python, fine at thesis scale |
| Embedding model | `all-MiniLM-L6-v2` | Stage 1 confirmed it cannot distinguish periods — motivates metadata filter. FinBERT ablation remains possible |
| Period extractor | Regex only | Consistent with Stage 1; keeps retrieval path fast and interpretable |
| Evaluation benchmark | FinanceBench (127 in-scope) | Positions against published baselines (naive RAG ~19%, full-context GPT-4 ~78%) |
| Generation model | Claude Haiku | Via Anthropic API. Structured prompt with source citation |
| Data source | SEC EDGAR XBRL CompanyFacts API | Authoritative, machine-readable, ground-truth fy/fp labels direct from filing |

---

## Open questions

1. **Path A or Path B?** — scope of the index (5 vs ~15 concepts). Unblocks the full evaluation.
2. **How many parity checks are enough?** — 4 currently verified. Suggest 3–4 per FinanceBench company (12–16 total) before thesis submission.
3. **Does `edgartools.filing.markdown()` preserve table column headers?** — evaluate in Phase 2 Step 6.
4. **Multi-period questions** — 18/127 FinanceBench questions reference two fiscal years ("year-over-year change from FY2016 to FY2017"). Parser takes the first match. Worth a targeted fix or noting as a limitation.
