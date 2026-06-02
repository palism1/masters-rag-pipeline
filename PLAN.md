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
- Regex ~54% on XBRL chunks — extracts year but drops quarter from `FY2024-Q1`
- Similarity tagger 41–55% on real XBRL — WORSE than on synthetic data. All chunks
  for the same company are semantically near-identical; the embedding model cannot
  distinguish between periods even when the label is explicitly in the text
- GOOG narrative 77% vs AAPL/MSFT/NVDA 23–35% — confirms the non-calendar fiscal
  year failure exactly as hypothesised
- **The embedding model's inability to distinguish periods motivates metadata filtering
  as a correctness mechanism, not just an optimisation**

### Phase 1 Steps 1–3 — Complete

**Step 1 — Index built:** 13,416 documents across 32 FinanceBench companies in Chroma.
`all-MiniLM-L6-v2` embeddings. Metadata per chunk: `fiscal_period`, `ticker`, `concept`,
`form_type`, `accession`, `entity`, `cik`, `period_end`, `period_type`.

**Step 2 — Query parser built:** regex + company name lookup → `{"ticker": "PEP", "fiscal_period": "FY2022-Q1"}`.
Audited against all 150 FinanceBench questions: 99% ticker accuracy, 82% full filter coverage,
0% pure-ANN fallback. Remaining period misses are multi-period comparison questions
(first-match limitation) and questions with no period in the text — both are expected behaviour.

**Step 3 — Retriever built:** `retrieve_both()` runs filtered and baseline modes against the
same index and returns side-by-side results. Smoke test confirmed:

| Question | Filtered top-3 periods | Baseline top-3 periods |
|---|---|---|
| PepsiCo net income Q1 2022 | FY2022-Q1, FY2022-Q1, FY2022-Q1 | FY2025-Q2, FY2025-Q2, FY2023-Q2 |
| 3M revenue Q2 2023 | FY2023-Q2, FY2023-Q2, FY2023-Q2 | FY2025-Q2, FY2020-Q3, FY2025-Q3 |
| JPMorgan EPS Q3 2022 | FY2022-Q3, FY2022-Q3, FY2022-Q3 | FY2020-Q3, FY2023-Q3, FY2023-Q2 |

**Filtered: 3/3 correct every time. Baseline: wrong every time.**
The core claim is empirically confirmed at retrieval level before any generation.

---

## Phase 1 — Remaining steps

### Step 4 — Generation with Claude

- `retrieval/generator.py`: takes `retrieve_both()` output, formats retrieved chunks as
  context, calls Claude, returns structured answer
- Two answers per question: one from filtered chunks, one from baseline chunks
- Structured prompt: answer the question, state the fiscal period explicitly, cite the
  accession number
- Add `anthropic` to `requirements.txt`

### Step 5 — Evaluation against FinanceBench

- FinanceBench public split (150 questions, 127 in XBRL scope after removing 8-K/Earnings)
- Evaluate with **RAGAS** metrics: context recall, answer accuracy
- Comparison table:
  | Approach | Context recall | Answer accuracy |
  |---|---|---|
  | Naive similarity (no filter) | ? | ? |
  | Period-filtered retrieval | ? | ? |
  | Period-filtered + Claude generation | ? | ? |

- **Retrieval failure report** (`evaluation/make_retrieval_report.py`): HTML table for
  every question where filtered and baseline disagree — shows question, ground truth,
  chunks retrieved by each approach (with `fiscal_period` metadata visible), both
  generated answers colour-coded correct/incorrect. Same structure as Stage 1
  `make_report.py`. Key argument: the error is wrong retrieval, not hallucination.
- **Coverage check completed:** FinanceBench spans FY2015–FY2024, all covered by
  EDGAR XBRL history

---

## Phase 2 — Structure-aware HTML parsing (the implicit tier)

_Goal: solve the cases where the fiscal period is NOT in the chunk text — it lives in a
section heading or table column header above the chunk_

### Step 6 — Document fetcher and structure parser

- Use `edgartools.filing.markdown()` or `sec-parser` to retrieve actual 10-K/10-Q document HTML
- Parse into a semantic tree: `TitleElement` (section headings), `TableElement`, `TextElement`
- Evaluate both tools on how faithfully they preserve:
  - Section headings (carry period for paragraphs beneath)
  - Table column headers (carry period for each data row)

### Step 7 — Structure-aware chunker with period propagation

- Walk the semantic tree and propagate the period label **down**:
  - Section heading `"Three Months Ended March 31, 2024"` → inject `fiscal_period=FY2024-Q1`
    into every paragraph chunk beneath it
  - Table column header `"Q1 2024 | Q2 2024"` → each data row becomes two chunks, one per
    period, with the correct `fiscal_period` tag
- This creates chunks for the **implicit tier** — chunks whose text alone contains no fiscal
  label, but whose metadata carries the correct one from document structure

### Step 8 — Mixed index

- Combine Phase 1 XBRL chunks and Phase 2 HTML-derived chunks in the same vector store
- Same retriever and generation pipeline from Phase 1 applies unchanged

### Step 9 — Ablation study (thesis evaluation table)

Demonstrate improvement across all three tiers:

| Tier | Example | XBRL-only | + Structure-aware HTML |
|---|---|---|---|
| Explicit | "Q1 2024 net income was $X" | ✓ retrieved | ✓ |
| Variant | "fiscal '22 saw record revenue" | ✓ retrieved | ✓ |
| Implicit | "Revenue increased vs prior period" | ✗ (no label in text) | ✓ (label from heading) |

This becomes the core evaluation table in the thesis.

---

## Technical decisions made

| Decision | Choice | Reason |
|---|---|---|
| Vector store | Chroma | Zero infra, pure Python, fine at thesis scale |
| Embedding model | `all-MiniLM-L6-v2` | Stage 1 confirmed it cannot distinguish periods — motivates metadata filter. FinBERT ablation remains a future option |
| Period extractor | Regex only | Consistent with Stage 1 philosophy; keeps the retrieval path fast and interpretable |
| Evaluation benchmark | FinanceBench (127 in-scope) | Positions work against published baselines (naive RAG ~19%, full-context GPT-4 ~78%) |
| Generation model | Claude | Via Anthropic API. Structured prompt with source citation |
| HTML parser | TBD: `sec-parser` vs `edgartools.filing.markdown()` | Evaluate in Step 6 |

---

## Open questions

- Does `edgartools.filing.markdown()` preserve table column headers, or does `sec-parser`
  need to be added as a dependency?
- Should the period extractor handle multi-period questions ("compare Q1 2024 vs Q1 2023")?
  Currently takes the first match — affects 18/127 FinanceBench questions.
