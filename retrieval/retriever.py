"""
retrieval/retriever.py — Period-filtered and baseline retrievers over the Chroma index.

Two retrieval modes against the same collection:
  filtered  — pre-filter by fiscal_period + ticker from the query parser, then ANN
  baseline  — pure ANN, no metadata filter (the comparison point for evaluation)

retrieve_both() runs both modes and returns a side-by-side result dict — the primary
input to the evaluation harness (Step 5) and the retrieval failure report.

Filtered retrieval degrades gracefully:
  ticker + period filter  →  if empty results, retry with ticker-only
  ticker-only filter      →  if empty results, fall back to pure ANN
  fallback field records which path was taken so the eval harness can flag it

Usage
-----
    from retrieval.retriever import retrieve_both
    result = retrieve_both("What was Apple's net income in Q1 2024?")
    result["filtered"]["chunks"]   # period-filtered top-k
    result["baseline"]["chunks"]   # pure ANN top-k
    result["parsed_filter"]        # {"ticker": "AAPL", "fiscal_period": "FY2024-Q1"}
"""

from __future__ import annotations

import logging

import chromadb
from sentence_transformers import SentenceTransformer

import config
from retrieval.query_parser import parse_query

logger = logging.getLogger(__name__)

COLLECTION_NAME = "financebench_xbrl"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Module-level singletons — loaded once on first call, reused across queries.
_model: SentenceTransformer | None = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def _build_where(filter_dict: dict) -> dict | None:
    """
    Convert a {key: value} filter dict to a Chroma where= clause.

    Single condition  → {"key": "value"}
    Two conditions    → {"$and": [{"k1": {"$eq": "v1"}}, {"k2": {"$eq": "v2"}}]}
    Empty dict        → None  (no where= argument passed to Chroma)
    """
    if not filter_dict:
        return None
    if len(filter_dict) == 1:
        key, val = next(iter(filter_dict.items()))
        return {key: val}
    return {"$and": [{k: {"$eq": v}} for k, v in filter_dict.items()]}


def _pack_results(results: dict) -> list[dict]:
    """Flatten Chroma query output into a list of chunk dicts."""
    ids       = results["ids"][0]
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]
    return [
        {"id": doc_id, "text": text, "distance": dist, **meta}
        for doc_id, text, meta, dist in zip(ids, docs, metas, distances)
    ]


def _query(embedding: list[float], k: int, where: dict | None) -> list[dict]:
    """Run a single Chroma query. Returns empty list on any error."""
    collection = _get_collection()
    kwargs = {"query_embeddings": [embedding], "n_results": k}
    if where:
        kwargs["where"] = where
    try:
        return _pack_results(collection.query(**kwargs))
    except Exception as exc:
        logger.warning("Chroma query failed: %s", exc)
        return []


def retrieve(
    question: str,
    k: int = 5,
    *,
    filtered: bool = True,
) -> dict:
    """
    Retrieve top-k chunks for a question in one mode.

    Parameters
    ----------
    question:
        Natural-language financial question.
    k:
        Number of chunks to return.
    filtered:
        True  → apply period+ticker filter from the query parser (with fallback).
        False → pure ANN baseline, no filter.

    Returns
    -------
    {
        "question":     str,
        "filter_used":  dict,           # filter actually applied (may differ from parsed if fallback)
        "fallback":     None | str,     # None | "ticker_only" | "none"
        "chunks":       list[dict],     # top-k chunks, each with text + all metadata + distance
    }
    """
    embedding = _get_model().encode(question, normalize_embeddings=True).tolist()

    if not filtered:
        return {
            "question":    question,
            "filter_used": {},
            "fallback":    None,
            "chunks":      _query(embedding, k, where=None),
        }

    # Filtered path — try progressively looser filters on empty results
    parsed = parse_query(question)
    where  = _build_where(parsed)
    chunks = _query(embedding, k, where)

    if chunks:
        return {"question": question, "filter_used": parsed, "fallback": None, "chunks": chunks}

    # Fallback 1 — ticker-only (drop fiscal_period)
    if "ticker" in parsed and "fiscal_period" in parsed:
        ticker_filter = {"ticker": parsed["ticker"]}
        chunks = _query(embedding, k, _build_where(ticker_filter))
        if chunks:
            logger.warning("Period filter empty for %r — fell back to ticker-only", question)
            return {"question": question, "filter_used": ticker_filter, "fallback": "ticker_only", "chunks": chunks}

    # Fallback 2 — pure ANN
    chunks = _query(embedding, k, where=None)
    logger.warning("All filters empty for %r — fell back to pure ANN", question)
    return {"question": question, "filter_used": {}, "fallback": "none", "chunks": chunks}


def retrieve_both(question: str, k: int = 5) -> dict:
    """
    Run filtered and baseline retrieval side by side.

    This is the primary entry point for the evaluation harness and the
    retrieval failure report — both modes share the same query embedding.

    Returns
    -------
    {
        "question":      str,
        "parsed_filter": dict,   # what the query parser extracted
        "filtered":      dict,   # retrieve() result with filtered=True
        "baseline":      dict,   # retrieve() result with filtered=False
    }
    """
    return {
        "question":      question,
        "parsed_filter": parse_query(question),
        "filtered":      retrieve(question, k, filtered=True),
        "baseline":      retrieve(question, k, filtered=False),
    }
