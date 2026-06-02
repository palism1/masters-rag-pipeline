# Next Steps Plan — Fiscal-Period-Aware Financial RAG

_Last updated: 2026-06-02_

---

## Where we are

Stage 1 (period tagger diagnostic) is complete:

- XBRL ingestion from SEC EDGAR via `edgartools` — clean, deduplicated facts with ground-truth `fy_label` from EDGAR's own `fy`/`fp` fields
- Two chunking modes: structured XBRL format and MD&A-style narrative prose
- Two taggers benchmarked: regex baseline vs 1-NN similarity (sentence-transformers / TF-IDF fallback)
- Cross-ticker HTML comparison reports for AAPL, MSFT, GOOG, NVDA
- Key result confirmed: regex ~54% on quarterly XBRL chunks (extracts year only from `FY2024-Q1`); similarity ~90%+. Non-calendar fiscal year failure demonstrated on narrative prose.

**What Stage 1 proved:** the tagging problem is diagnostic. The real contribution is retrieval — ensuring that when a question asks about Q1 2024, only Q1 2024 chunks are retrieved.

---

## Phase 1 — Period-filtered retrieval on clean XBRL data

_Goal: validate the core retrieval contribution end-to-end_

### Step 1 — Embed and index XBRL chunks

- Run `facts_to_chunks()` for all tickers (AAPL, MSFT, GOOG, NVDA) across all default concepts
- Embed chunk text with `sentence-transformers/all-MiniLM-L6-v2`
- Store in **Chroma** (local, zero-infra) with the following metadata fields per chunk:
  - `fiscal_period` — e.g. `"FY2024-Q1"` (from `fy_label`)
  - `ticker` — e.g. `"AAPL"`
  - `concept` — e.g. `"NetIncomeLoss"`
  - `form_type` — `"10-K"` or `"10-Q"`
  - `accession` — SEC accession number (for citation)

### Step 2 — Regex period extractor (query → filter)

- Takes a natural-language question: _"What was Apple's net income in Q1 2024?"_
- Returns a structured filter dict: `{"ticker": "AAPL", "fiscal_period": "FY2024-Q1"}`
- Regex-only, consistent with Stage 1 philosophy — no LLM call in the retrieval path
- Handles the same surface forms the tagger already covers: `Q1 2024`, `FY2024-Q1`, `first quarter 2024`, `fiscal year 2023`, etc.
- Falls back to `ticker`-only filter if no period is extractable

### Step 3 — Period-aware retriever

- Pre-filter the Chroma collection by `fiscal_period` + `ticker` before ANN search
- Run ANN search within the filtered subset
- Return top-k chunks with metadata
- **Baseline for comparison:** same ANN search with no metadata filter

### Step 4 — Generation with Claude

- Pass retrieved chunks to Claude as context
- Structured prompt: answer the question, cite the source accession number, state the fiscal period explicitly
- Return structured response: answer, evidence chunks, period label used

### Step 5 — Evaluation against FinanceBench

- FinanceBench: 10,231 expert QA pairs from real SEC filings (10-K, 10-Q)
- Map the subset of FinanceBench questions covering AAPL, MSFT, GOOG, NVDA to the pipeline
- Evaluate with **RAGAS** metrics: faithfulness, answer relevancy, context recall
- Comparison table:
  | Approach | Context recall | Answer accuracy |
  |---|---|---|
  | Naive similarity (no filter) | ? | ? |
  | Period-filtered retrieval | ? | ? |
  | Period-filtered + Claude generation | ? | ? |

---

## Phase 2 — Structure-aware HTML parsing (the implicit tier)

_Goal: solve the cases where the fiscal period is NOT in the chunk text — it lives in a section heading or table column header above the chunk_

### Step 6 — Document fetcher and structure parser

- Use `edgartools.filing.markdown()` or `sec-parser` to retrieve actual 10-K/10-Q document HTML
- Parse into a semantic tree: `TitleElement` (section headings), `TableElement`, `TextElement`
- Evaluate both tools on how faithfully they preserve:
  - Section headings (carry period for paragraphs beneath)
  - Table column headers (carry period for each data row)

### Step 7 — Structure-aware chunker with period propagation

- Walk the semantic tree and propagate the period label **down**:
  - Section heading `"Three Months Ended March 31, 2024"` → inject `fiscal_period=FY2024-Q1` into every paragraph chunk beneath it
  - Table column header `"Q1 2024 | Q2 2024"` → each data row becomes two chunks, one per period, with the correct `fiscal_period` tag
- This creates chunks for the **implicit tier** — chunks whose text alone contains no fiscal label, but whose metadata carries the correct one from document structure

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
| Vector store | Chroma | Zero infra, pure Python, fine at thesis scale. Swap to Qdrant if filter efficiency needs to be a thesis argument. |
| Embedding model | `all-MiniLM-L6-v2` | Already in stack. Finance-specific model (e.g. FinBERT) is a potential ablation. |
| Period extractor | Regex only | Consistent with Stage 1 philosophy; keeps the retrieval path fast and interpretable. |
| Evaluation benchmark | FinanceBench | Positions work against published baselines (naive RAG ~19%, full-context GPT-4 ~78%). |
| Generation model | Claude | Via Anthropic API. Structured prompt with source citation. |
| HTML parser | TBD: `sec-parser` vs `edgartools.filing.markdown()` | Evaluate in Step 6. |

---

## Open questions

- Which FinanceBench questions map cleanly to the four tickers already in the pipeline?
- Does `edgartools.filing.markdown()` preserve table column headers, or does `sec-parser` need to be added as a dependency?
- Should the period extractor handle multi-period questions ("compare Q1 2024 vs Q1 2023")?
