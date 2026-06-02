# Fiscal-Period-Aware Financial RAG Pipeline

Masters thesis project. Evaluates whether tagging and filtering retrieval chunks by
fiscal period (FY2024-Q1, FY2023, etc.) improves answer quality on financial QA tasks
compared to naive similarity search.

Benchmark: [FinanceBench](https://github.com/patronus-ai/financebench) — 150 expert QA
pairs from real SEC 10-K/10-Q filings. Published baseline: naive RAG ~19%, full-context
GPT-4 ~78%.

---

## Results so far

### Stage 1 — Tagger diagnostic (complete)

Real EDGAR data, 4 companies, 5 concepts each.

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

Key findings:
- Regex ~54% on XBRL — extracts year but drops the quarter from `FY2024-Q1`
- Similarity tagger 41–55% on real data — **worse than on synthetic data**. Quarterly
  facts for the same company are semantically near-identical; the embedding model cannot
  distinguish periods even when the label is explicitly in the chunk text
- GOOG narrative 77% vs AAPL/MSFT/NVDA 23–35% — non-calendar fiscal year failure confirmed.
  Regex maps calendar months to quarters; only GOOG (December FY) is correct

**The embedding model's inability to distinguish fiscal periods — even with the label
in the text — is the core motivation for metadata filtering.**

### Phase 1 Steps 1–3 — Complete

**Index:** 13,416 documents across all 32 FinanceBench companies in Chroma.

**Query parser:** 99% ticker accuracy, 82% full filter coverage on FinanceBench questions.

**Smoke test — retriever confirmed working:**

| Question | Filtered top-3 | Baseline top-3 |
|---|---|---|
| PepsiCo net income Q1 2022 | FY2022-Q1 ✓ | FY2025-Q2, FY2023-Q2 ✗ |
| 3M revenue Q2 2023 | FY2023-Q2 ✓ | FY2025-Q2, FY2020-Q3 ✗ |
| JPMorgan EPS Q3 2022 | FY2022-Q3 ✓ | FY2020-Q3, FY2023-Q3 ✗ |

Filtered retrieval returns the correct period every time. Baseline returns wrong periods
every time. The core thesis claim is confirmed at the retrieval level.

---

## Structure

```
ingestion/       XBRL fact loader (SEC EDGAR), structured + narrative chunkers
evaluation/      Stage 1: tagger diagnostic + HTML comparison reports
retrieval/       Phase 1: Chroma indexer, query parser, retriever, FinanceBench registry
scripts/         Dev utilities and auditing tools
tests/           55 tests — run with pytest
config.py        Central config — reads from .env
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SEC_USER_AGENT
```

`.env` keys:

| Key | Required | Default | Notes |
|-----|----------|---------|-------|
| `SEC_USER_AGENT` | Yes | — | `Firstname Lastname email@example.com` — SEC requires a real contact |
| `DRY_RUN` | No | `true` | Set to `false` to write to Chroma |
| `EDGAR_CACHE_DIR` | No | `./edgar_cache` | EDGAR API responses cached here |
| `CHROMA_DIR` | No | `./chroma_db` | Chroma vector store persisted here |

---

## Stage 1 — Tagger diagnostic (complete)

```bash
# Per-ticker accuracy table (stdout)
python evaluation/xbrl_eval.py AAPL MSFT GOOG NVDA

# Per-ticker HTML report
python evaluation/make_report.py AAPL
python evaluation/make_report.py AAPL --narrative

# Cross-ticker summary HTML (all four tickers, both modes)
python evaluation/make_summary.py
```

---

## Phase 1 — Period-filtered retrieval

### Build the index (requires DRY_RUN=false in .env)

```bash
python -m retrieval.build_index --subset 10q   # 6 companies with 10-Q questions first
python -m retrieval.build_index                # full 32 companies (~13k documents)
```

### Query the retriever

```python
from retrieval.retriever import retrieve_both

r = retrieve_both("What was PepsiCo's net income in Q1 2022?")
r["parsed_filter"]              # {"ticker": "PEP", "fiscal_period": "FY2022-Q1"}
r["filtered"]["chunks"]         # top-k from period-filtered ANN
r["baseline"]["chunks"]         # top-k from pure ANN (no filter)
```

### Parse query filters

```python
from retrieval.query_parser import parse_query

parse_query("What was Apple's net income in Q1 2024?")
# → {"ticker": "AAPL", "fiscal_period": "FY2024-Q1"}
```

### Audit parser coverage against FinanceBench

```bash
python scripts/audit_fb_parser.py
# → 99% ticker accuracy, 82% full filter, 0% pure-ANN fallback across 127 in-scope questions
```

---

## Dev utilities

```bash
# Inspect raw facts + chunk text for any ticker
python scripts/xbrl_dry_run.py AAPL
python scripts/xbrl_dry_run.py MSFT --concepts NetIncomeLoss EarningsPerShareBasic

# Run all tests
pytest

# Run tests without live EDGAR calls
pytest --ignore=tests/test_xbrl_loader.py
```

---

## Concepts

**Fiscal period tiers** (from the thesis evaluation table):

| Tier | Example | Recoverable from chunk text? |
|------|---------|------------------------------|
| Explicit | `"Q1 2024 net income was $X"` | Yes — regex or similarity |
| Variant | `"fiscal '22 saw record revenue"` | Partial — similarity beats regex |
| Implicit | `"Revenue increased vs prior period"` | No — label must come from document structure |

Phase 2 (Steps 6–9) targets the implicit tier via structure-aware HTML parsing of
SEC filings — propagating period labels from section headings and table column headers
down into individual chunks.

See [PLAN.md](PLAN.md) for the full roadmap.
