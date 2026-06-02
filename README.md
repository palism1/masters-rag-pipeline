# Fiscal-Period-Aware Financial RAG Pipeline

Masters thesis project. Evaluates whether tagging and filtering retrieval chunks by
fiscal period (FY2024-Q1, FY2023, etc.) improves answer quality on financial QA tasks
compared to naive similarity search.

Benchmark: [FinanceBench](https://github.com/patronus-ai/financebench) — 150 expert QA
pairs from real SEC 10-K/10-Q filings. Published baseline: naive RAG ~19%, full-context
GPT-4 ~78%.

---

## Structure

ingestion/       XBRL fact loader (SEC EDGAR), structured + narrative chunkers
evaluation/      Stage 1: regex vs similarity tagger diagnostic + HTML reports
retrieval/       Phase 1: Chroma vector store indexer + FinanceBench company registry
scripts/         Dev utilities (dry-run fact inspector)
tests/           22 tests — run with pytest
config.py        Central config — reads from .env



---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SEC_USER_AGENT and optionally ANTHROPIC_API_KEY
.env keys:

Key	Required	Default	Notes
SEC_USER_AGENT	Yes	—	Firstname Lastname email@example.com — SEC requires a real contact
DRY_RUN	No	true	Set to false to write to Chroma
EDGAR_CACHE_DIR	No	./edgar_cache	EDGAR API responses cached here
CHROMA_DIR	No	./chroma_db	Chroma vector store persisted here
Stage 1 — Tagger diagnostic (complete)
Compares a regex baseline against a 1-NN similarity tagger on the task of assigning
fiscal period labels to XBRL chunks. Ground truth from EDGAR's own fy/fp fields.


# Per-ticker HTML report (xbrl or narrative mode)
python -m evaluation.make_report AAPL --narrative

# Cross-ticker summary table
python -m evaluation.make_summary

# Standalone accuracy table (stdout)
python -m evaluation.xbrl_eval AAPL MSFT GOOG NVDA
Key finding: regex scores ~54% on quarterly XBRL chunks — it extracts the year from
FY2024-Q1 but drops the quarter token. Similarity tagger scores ~90%+. On narrative
prose, regex fails entirely for non-calendar fiscal year companies (AAPL, MSFT, NVDA)
because it maps calendar months to quarters; only GOOG (December year-end) scores well.

Phase 1 — Period-filtered retrieval (in progress)
Step 1 — Build the index
Embeds all 32 FinanceBench companies into a Chroma collection using all-MiniLM-L6-v2.
Requires DRY_RUN=false in .env to write.


python -m retrieval.build_index                 # all 32 companies
python -m retrieval.build_index --subset 10q    # 7 companies with 10-Q questions
python -m retrieval.build_index --subset stage1 # AAPL/MSFT/GOOG/NVDA (smoke-test)
Upserts by stable document ID — safe to re-run. Chroma metadata per chunk:
fiscal_period, ticker, concept, form_type, accession, entity, cik.

Steps 2–5 — Retriever, generation, evaluation (planned)
See PLAN.md for the full Phase 1 and Phase 2 roadmap.

Dev utilities

# Inspect raw facts + chunk text for any ticker without writing anything
python -m scripts.xbrl_dry_run AAPL
python -m scripts.xbrl_dry_run MSFT --concepts NetIncomeLoss EarningsPerShareBasic

# Run tests
pytest
Concepts
Fiscal period tiers (from the thesis evaluation table):

Tier	Example	Recoverable from chunk text?
Explicit	"Q1 2024 net income was $X"	Yes — regex or similarity
Variant	"fiscal '22 saw record revenue"	Partial — similarity beats regex
Implicit	"Revenue increased vs prior period"	No — label must come from document structure
Phase 2 (Step 6–9) targets the implicit tier via structure-aware HTML parsing of
SEC filings — propagating period labels from section headings and table column headers
down into individual chunks.
