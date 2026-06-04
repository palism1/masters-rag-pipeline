# Fiscal-Period-Aware Financial RAG Pipeline

Masters thesis project. Evaluates whether tagging retrieval chunks with fiscal period metadata and filtering by period at query time improves answer quality on financial QA tasks compared to naive similarity search.

Benchmark: [FinanceBench](https://github.com/patronus-ai/financebench) — 150 expert QA pairs from real SEC 10-K/10-Q filings (127 in-scope for XBRL retrieval). Published baseline: naive RAG ~19%, full-context GPT-4 ~78%.

---

## Key Findings

### Finding 1 — EDGAR data quality (novel)

EDGAR progressively overwrites `fy`/`fp` tags on historical facts when those facts appear as prior-year comparison columns in newer filings. Google's FY2023 Revenues ($307.4B) carried `fiscal_year=2023` in the 2023 10-K but was re-tagged `fiscal_year=2025` after appearing in the 2025 10-K comparison column. This is undocumented in the financial NLP literature. Built and tested `_filter_comparative()` to address it — validates duration facts against trusted annual bounds derived from 10-K period dates.

### Finding 2 — Embeddings cannot distinguish fiscal periods

Cosine similarity between same-company quarterly facts is ≥95% regardless of period. `all-MiniLM-L6-v2` scores 41–55% at period classification even when the correct label is in the chunk text. The metadata filter is a correctness mechanism, not an optimisation.

### Finding 3 — Stage 1 tagger (classification task)

| Ticker | FY End | Mode | Regex | Similarity |
|---|---|---|---|---|
| AAPL | September | XBRL | 54% | 44% |
| MSFT | June | XBRL | 55% | 42% |
| GOOG | December ✓ | XBRL | 19% | 41% |
| NVDA | January | XBRL | 52% | 55% |
| AAPL | September | Narrative | 35% | 41% |
| MSFT | June | Narrative | 27% | 34% |
| GOOG | December ✓ | Narrative | **77%** | 39% |
| NVDA | January | Narrative | 23% | 38% |

Non-calendar FY companies score 23–35% on narrative prose with regex — confirmed hypothesis.

### Finding 4 — Period-filtered retrieval vs baseline (3 models, 127 questions each)

| Model | Filtered retrieval acc | Baseline retrieval acc | Filtered ans acc (lenient) |
|---|---|---|---|
| MiniLM (general, 384-dim) | **66%** | 11% | 10% |
| FinBERT (financial, 768-dim) | **67%** | 5% | 11% |
| MPNet (general+, 768-dim) | **66%** | 1% | 12% |

**The filter is the mechanism, not the model** — all three models land within 1% of each other on filtered retrieval. Counterintuitive: MPNet (strongest model) has the worst baseline (1%). A more capable embedding model makes wrong-period retrieval more confident, not less. Domain-specific FinBERT provides no advantage under the filter.

### Finding 5 — Wrong retrieval → wrong answer (end-to-end)

PepsiCo Q1 2022 question: filtered retrieved FY2022-Q1 chunks → answered $4.261B with HIGH confidence. Baseline retrieved FY2025-Q2 chunks → Claude correctly refused. The error is in retrieval, not generation.

### Finding 6 — Phase 2 HTML implicit tier assessment

Structure-aware HTML chunking of 10-K filings shows 96% heading-propagation coverage, but labeled chunks are table-of-contents rows inheriting the document-level annual heading — not the meaningful implicit tier. The real implicit tier (comparative quarterly prose with no explicit period in text) requires 10-Q column-header table parsing, identified as the specific next step for Phase 2.

---

## Structure

```
ingestion/       XBRL loader + comparative-period filter, structured + narrative chunkers, HTML chunker
evaluation/      Stage 1 tagger diagnostic, eval pipeline, model comparison, HTML reports
retrieval/       Chroma indexer (3 models), query parser, retriever, generator
scripts/         Dev utilities, dry-run tools, findings report generator
tests/           159 tests — run with pytest
config.py        Central config — reads from .env
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SEC_USER_AGENT and ANTHROPIC_API_KEY
```

| Key | Required | Default | Notes |
|---|---|---|---|
| `SEC_USER_AGENT` | Yes | — | `Firstname Lastname email@example.com` |
| `ANTHROPIC_API_KEY` | For generation | — | Required for eval pipeline |
| `DRY_RUN` | No | `true` | Set `false` to write to Chroma |
| `CHROMA_DIR` | No | `./chroma_db` | Vector store location |

---

## Reproducing results

### Stage 1 — Tagger diagnostic
```bash
python evaluation/xbrl_eval.py AAPL MSFT GOOG NVDA
python evaluation/make_summary.py
```

### Phase 1 — Build indexes (requires DRY_RUN=false)
```bash
python -m retrieval.build_index --model minilm
python -m retrieval.build_index --model finbert
python -m retrieval.build_index --model mpnet
```

### Phase 1 — Evaluate all three models
```bash
python evaluation/eval_pipeline.py --model minilm
python evaluation/eval_pipeline.py --model finbert
python evaluation/eval_pipeline.py --model mpnet
python evaluation/compare_models.py          # side-by-side comparison table + HTML report
```

### Findings report (all 6 findings, live data)
```bash
python scripts/make_findings_report.py       # → findings_report.html
```

### Retrieval failure report
```bash
python evaluation/make_retrieval_report.py   # → retrieval_failure_report.html
```

### Tests
```bash
pytest                                        # 159 tests
pytest --ignore=tests/test_xbrl_loader.py    # skip live EDGAR calls
```

---

## Literature context

| Paper | Approach | Difference from this work |
|---|---|---|
| Dadopoulos et al. 2025 (arXiv 2510.24402) | LLM-generated metadata, FinanceBench | PDF parsing, no XBRL, no fiscal period filter |
| Samuelsen et al. 2026 (arXiv 2605.25030) | Multi-agent RAG, 89.3% on FinanceBench with GPT-4.1 | PDF parsing via Docling, no XBRL, no period filter |
| Zhu et al. 2025 (arXiv 2503.05185) | Temporal-aware multi-modal RAG (FinTMMBench) | NASDAQ 100 news/prices, not SEC XBRL |
| Boritz & No 2011 (JAPP) | XBRL data quality in EDGAR filings | Computation errors in raw XBRL; does not document CompanyFacts API re-tagging |

This work is the first to use EDGAR's XBRL CompanyFacts API as a structured source for fiscal-period-aware retrieval, and the first to document the progressive re-tagging behaviour of the API.

---

See [PLAN.md](PLAN.md) for the full roadmap including Phase 2 (10-Q implicit tier).
