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

FILE MAP
  L001–L038  Module docstring + file map
  L040–L062  CONFIG — default collection name, default embedding model, DEFAULT_K
  L064–L090  Lazy per-model / per-collection singleton caches
  L092–L122  Chroma where= clause builder + result packer + _query
  L124–L195  retrieve() — single-mode retrieval with fallback logic
  L197–L235  retrieve_both() — primary entry point, runs both modes

Usage
-----
    from retrieval.retriever import retrieve_both
    result = retrieve_both("What was Apple's net income in Q1 2024?")
    result["filtered"]["chunks"]   # period-filtered top-k
    result["baseline"]["chunks"]   # pure ANN top-k
    result["parsed_filter"]        # {"ticker": "AAPL", "fiscal_period": "FY2024-Q1"}

    # Embedding ablation — point both the model and collection at another slug:
    result = retrieve_both(
        "What was Apple's net income in Q1 2024?",
        model_name="ProsusAI/finbert",
        collection_name="financebench_xbrl_finbert",
    )
"""

from __future__ import annotations

import logging

import chromadb
from sentence_transformers import SentenceTransformer

import config
from retrieval.query_parser import parse_query

logger = logging.getLogger(__name__)

# ===========================================================================
# CONFIG
# ===========================================================================

# Default collection — must match the default model's collection in build_index.py
# (financebench_xbrl_{slug}). Callers override via the collection_name kwarg for
# the embedding ablation.
COLLECTION_NAME = "financebench_xbrl_minilm"   # CHANGE ME if you rename the default collection

# Default embedding model — must match the model used to build COLLECTION_NAME.
# Querying a collection with a different model than it was built with produces
# nonsense similarity scores, so model and collection are chosen together.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # CHANGE ME: ablation models in build_index.MODEL_REGISTRY

# Default number of chunks returned per query.
DEFAULT_K = 5                               # TWEAK

# ===========================================================================

# Lazy per-key singleton caches — each model/collection is loaded once on first
# use and reused across queries. Keyed (rather than a single global) so several
# embedding models can be queried in the same process during the ablation without
# reloading or clobbering each other.
_models: dict[str, SentenceTransformer] = {}
_collections: dict[str, object] = {}


def _get_model(model_name: str = EMBED_MODEL) -> SentenceTransformer:
    if model_name not in _models:
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


def _get_collection(collection_name: str = COLLECTION_NAME):
    if collection_name not in _collections:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collections[collection_name] = client.get_collection(name=collection_name)
    return _collections[collection_name]


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


def _query(
    embedding: list[float],
    k: int,
    where: dict | None,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """Run a single Chroma query. Returns empty list on any error."""
    collection = _get_collection(collection_name)
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
    model_name: str = EMBED_MODEL,
    collection_name: str = COLLECTION_NAME,
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
    model_name:
        HuggingFace embedding model ID. Defaults to the model COLLECTION_NAME was
        built with — override for the embedding ablation (pair with collection_name).
    collection_name:
        Chroma collection to query. Must have been built with model_name.

    Returns
    -------
    {
        "question":     str,
        "filter_used":  dict,           # filter actually applied (may differ from parsed if fallback)
        "fallback":     None | str,     # None | "ticker_only" | "none"
        "chunks":       list[dict],     # top-k chunks, each with text + all metadata + distance
    }
    """
    embedding = _get_model(model_name).encode(question, normalize_embeddings=True).tolist()

    if not filtered:
        return {
            "question":    question,
            "filter_used": {},
            "fallback":    None,
            "chunks":      _query(embedding, k, where=None, collection_name=collection_name),
        }

    # Filtered path — try progressively looser filters on empty results
    parsed = parse_query(question)
    where  = _build_where(parsed)
    chunks = _query(embedding, k, where, collection_name=collection_name)

    if chunks:
        return {"question": question, "filter_used": parsed, "fallback": None, "chunks": chunks}

    # Fallback 1 — ticker-only (drop fiscal_period)
    if "ticker" in parsed and "fiscal_period" in parsed:
        ticker_filter = {"ticker": parsed["ticker"]}
        chunks = _query(embedding, k, _build_where(ticker_filter), collection_name=collection_name)
        if chunks:
            logger.warning("Period filter empty for %r — fell back to ticker-only", question)
            return {"question": question, "filter_used": ticker_filter, "fallback": "ticker_only", "chunks": chunks}

    # Fallback 2 — pure ANN
    chunks = _query(embedding, k, where=None, collection_name=collection_name)
    logger.warning("All filters empty for %r — fell back to pure ANN", question)
    return {"question": question, "filter_used": {}, "fallback": "none", "chunks": chunks}


def retrieve_both(
    question: str,
    k: int = 5,
    *,
    model_name: str = EMBED_MODEL,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    """
    Run filtered and baseline retrieval side by side.

    This is the primary entry point for the evaluation harness and the
    retrieval failure report — both modes share the same query embedding.

    model_name / collection_name select the embedding model and its dedicated
    collection for the ablation; they default to the minilm build and must be
    paired (a collection is only valid for the model it was built with).

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
        "filtered":      retrieve(question, k, filtered=True,
                                  model_name=model_name, collection_name=collection_name),
        "baseline":      retrieve(question, k, filtered=False,
                                  model_name=model_name, collection_name=collection_name),
    }
