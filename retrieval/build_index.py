"""
build_index.py — Embed all FinanceBench XBRL chunks into a Chroma vector store.

Creates (or updates) a persistent Chroma collection at config.CHROMA_DIR.
Safe to re-run: upserts by stable document ID, so re-running adds new
filings without duplicating existing ones.

Writes are gated by config.DRY_RUN (default: True).
Set DRY_RUN=false in .env to actually persist to the index.

Usage
-----
    python -m retrieval.build_index                    # all 32 FinanceBench companies (minilm)
    python -m retrieval.build_index --subset 10q       # 7 companies with 10-Q questions
    python -m retrieval.build_index --subset stage1    # AAPL, MSFT, GOOG, NVDA (dev/smoke-test)
    python -m retrieval.build_index --model finbert    # build the finbert collection (embedding ablation)

FILE MAP
  L001–L057  Module docstring + usage + file map
  L059–L080  CONFIG — collection prefix, MODEL_REGISTRY (CHANGE ME), stage-1 companies
  L082–L102  Doc ID + Chroma-safe metadata helpers
  L104–L172  build() — embed chunks into the per-model collection
  L174–L215  CLI parsing + main()
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer

import config
from retrieval.financebench_tickers import (
    FINANCEBENCH_COMPANIES,
    FBCompany,
    HAS_10Q,
    STAGE1_TICKERS,
    ticker_or_cik,
)
from ingestion.xbrl_chunker import facts_to_chunks
from ingestion.xbrl_loader import load_company_facts

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ===========================================================================
# CONFIG
# ===========================================================================

# Collection-name prefix. The per-model collection is f"{COLLECTION_PREFIX}_{slug}",
# e.g. "financebench_xbrl_minilm". Each model gets its own collection because
# embeddings from different models live in incompatible vector spaces (and
# different dimensions) — they cannot share an ANN index.
COLLECTION_PREFIX = "financebench_xbrl"

# CHANGE ME: maps a CLI slug → HuggingFace model ID. Add a row here to wire up a
# new embedding model for the ablation; the slug also names its Chroma collection
# and its results file (results/eval_results_{slug}.json).
MODEL_REGISTRY: dict[str, str] = {
    "minilm":  "sentence-transformers/all-MiniLM-L6-v2",   # general, fast, 384-dim (default)
    "finbert": "ProsusAI/finbert",                          # domain-specific financial BERT, 768-dim
    "mpnet":   "sentence-transformers/all-mpnet-base-v2",   # stronger general model, 768-dim
}

# Default model slug — keeps existing behaviour unchanged when --model is omitted.
DEFAULT_MODEL = "minilm"                                    # TWEAK

# ===========================================================================

# Stage 1 tickers are not in FinanceBench but are useful for smoke-testing the
# pipeline. CIKs are their SEC-registered identifiers.
_STAGE1_COMPANIES: list[FBCompany] = [
    FBCompany("Apple",     "AAPL", 320193,  0, has_10q=False),
    FBCompany("Microsoft", "MSFT", 789019,  2, has_10q=False),
    FBCompany("Alphabet",  "GOOG", 1652044, 0, has_10q=False),
    FBCompany("NVIDIA",    "NVDA", 1045810, 0, has_10q=False),
]


def _doc_id(chunk: dict) -> str:
    """Stable, globally unique document ID — safe for Chroma upsert."""
    period = chunk["period_start"] or "instant"
    return f"{chunk['cik']}_{chunk['concept']}_{period}_{chunk['period_end']}_{chunk['unit']}"


def _safe_metadata(chunk: dict, ticker_label: str) -> dict:
    """Chroma-safe metadata: no None values, all primitives."""
    return {
        "fiscal_period": chunk["fy_label"] or "",
        "ticker":        ticker_label,
        "concept":       chunk["concept"],
        "form_type":     chunk["form_type"],
        "accession":     chunk["accession"] or "",
        "entity":        chunk["entity"],
        "cik":           chunk["cik"],
        "period_end":    chunk["period_end"],
        "period_type":   chunk["period_type"],
    }


def build(companies: list[FBCompany], dry_run: bool, slug: str = DEFAULT_MODEL) -> None:
    embed_model      = MODEL_REGISTRY[slug]
    collection_name  = f"{COLLECTION_PREFIX}_{slug}"

    logger.info("Embedding model: %s → collection '%s'", embed_model, collection_name)
    model = SentenceTransformer(embed_model)

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    total = 0
    skipped = 0

    for company in companies:
        key = ticker_or_cik(company)
        ticker_label = company.ticker or f"CIK{company.cik}"

        try:
            facts = load_company_facts(key)
        except Exception as exc:
            logger.warning("%-12s | SKIP — %s", ticker_label, exc)
            skipped += 1
            continue

        chunks = facts_to_chunks(facts)
        if not chunks:
            logger.info("%-12s | 0 chunks — nothing to index", ticker_label)
            continue

        # Deduplicate by doc_id before embedding — EDGAR sometimes reports the same
        # (concept, period, unit) with two different values (original + restatement).
        # Both survive xbrl_loader._dedup (different values = different dedup keys)
        # but map to the same doc_id. Keep the first occurrence, which comes from
        # the most authoritative form_type already selected by _dedup.
        seen: dict[str, dict] = {}
        for c in chunks:
            did = _doc_id(c)
            if did not in seen:
                seen[did] = c

        deduped   = list(seen.values())
        ids       = list(seen.keys())
        texts     = [c["text"] for c in deduped]
        metadatas = [_safe_metadata(c, ticker_label) for c in deduped]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        dropped = len(chunks) - len(deduped)
        if dropped:
            logger.debug("%-12s | %d duplicate doc_ids dropped", ticker_label, dropped)

        if not dry_run:
            collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        logger.info("%-12s | %d chunks %s", ticker_label, len(deduped),
                    "embedded (dry_run — not written)" if dry_run else "upserted")
        total += len(chunks)

    if dry_run:
        logger.info(
            "DRY_RUN complete. %d chunks would be indexed across %d companies (%d skipped).",
            total, len(companies) - skipped, skipped,
        )
    else:
        logger.info(
            "Index built. Collection '%s' now holds %d documents. (%d companies skipped)",
            collection_name, collection.count(), skipped,
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--subset",
        choices=["all", "10q", "stage1"],
        default="all",
        help=(
            "all     — all 32 FinanceBench companies (default)\n"
            "10q     — 7 companies that have 10-Q questions in FinanceBench\n"
            "stage1  — AAPL/MSFT/GOOG/NVDA (smoke-test; only MSFT is in FinanceBench)"
        ),
    )
    p.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY),
        default=DEFAULT_MODEL,
        help=(
            "Embedding model slug — selects both the model and its dedicated\n"
            "Chroma collection (financebench_xbrl_{slug}). Default: %(default)s."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.subset == "10q":
        companies = HAS_10Q
    elif args.subset == "stage1":
        companies = _STAGE1_COMPANIES
    else:
        companies = list(FINANCEBENCH_COMPANIES)

    logger.info(
        "build_index | subset=%s | model=%s | companies=%d | dry_run=%s | chroma_dir=%s",
        args.subset, args.model, len(companies), config.DRY_RUN, config.CHROMA_DIR,
    )
    build(companies, dry_run=config.DRY_RUN, slug=args.model)


if __name__ == "__main__":
    main()
