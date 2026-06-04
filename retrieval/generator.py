"""
retrieval/generator.py — Claude generation over retrieved chunks.

Takes the output of retrieve_both() and calls Claude once per retrieval mode
(filtered and baseline), returning structured answers for direct comparison.

The prompt enforces three things:
  1. Answer the question using only the provided chunks
  2. State the fiscal period the answer comes from explicitly
  3. Cite the accession number of the source chunk

This makes the failure mode visible: when baseline retrieval returns wrong-period
chunks, Claude will answer with a wrong number AND cite the wrong period — the
error is traceable to retrieval, not to the model.

FILE MAP
  L001–L032  Module docstring + file map
  L034–L063  Imports + CONFIG (model, token limit, k)
  L065–L080  Lazy client singleton
  L082–L110  Prompt construction — system prompt + chunk formatter
  L112–L130  Response parser — extracts structured fields from Claude output
  L132–L165  generate() — single-mode API call
  L167–L205  generate_both() — primary entry point, calls both modes

Usage
-----
    from retrieval.generator import generate_both
    result = generate_both("What was PepsiCo's net income in Q1 2022?")
    result["filtered"]["answer"]        # answer from period-filtered chunks
    result["baseline"]["answer"]        # answer from baseline (likely wrong period)
    result["filtered"]["fiscal_period"] # period Claude stated it used
    result["baseline"]["fiscal_period"]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic

import config
from retrieval.retriever import (
    COLLECTION_NAME as DEFAULT_COLLECTION,
    EMBED_MODEL as DEFAULT_MODEL_NAME,
    retrieve_both,
)

logger = logging.getLogger(__name__)

# ===========================================================================
# CONFIG
# ===========================================================================

# Generation model. Haiku is cheapest and sufficient for structured extraction.
# CHANGE ME: swap to "claude-sonnet-4-6" to test whether a stronger model
# compensates for wrong-period retrieval (thesis ablation).
MODEL = "claude-haiku-4-5-20251001"

# Max tokens for Claude response — 512 is enough for the structured 4-line format.
# TWEAK: increase if answers are being truncated.
MAX_TOKENS = 512

# Number of chunks passed to Claude as context per call.
# TWEAK: increase to give Claude more evidence; watch for context window limits.
TOP_K = 5

# ===========================================================================

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not config.ANTHROPIC_API_KEY:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set in .env — required for generation."
            )
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a precise financial analyst. Answer questions using only the provided \
context chunks from SEC filings. Be concise and exact.

Always structure your response as:
ANSWER: <your direct answer, including the specific dollar amount or value>
FISCAL_PERIOD: <the fiscal period this answer covers, e.g. FY2022-Q1 or FY2022>
SOURCE: <the SEC accession number of the chunk you relied on most>
CONFIDENCE: <HIGH if the answer is directly stated in a chunk / LOW if inferred>"""


def _format_chunks(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] {c.get('entity', '')} | {c.get('fiscal_period', '?')} | "
            f"{c.get('concept', '')} | accession: {c.get('accession', '?')}\n"
            f"    {c['text']}"
        )
    return "\n\n".join(lines)


def _parse_response(text: str) -> dict:
    """Extract structured fields from Claude's response."""
    result = {
        "answer":        None,
        "fiscal_period": None,
        "source":        None,
        "confidence":    None,
        "raw":           text,
    }
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("ANSWER:"):
            result["answer"] = line[len("ANSWER:"):].strip()
        elif line.startswith("FISCAL_PERIOD:"):
            result["fiscal_period"] = line[len("FISCAL_PERIOD:"):].strip()
        elif line.startswith("SOURCE:"):
            result["source"] = line[len("SOURCE:"):].strip()
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line[len("CONFIDENCE:"):].strip()
    return result


# ---------------------------------------------------------------------------
# Core call
# ---------------------------------------------------------------------------

def generate(question: str, chunks: list[dict]) -> dict:
    """
    Call Claude with a question and a list of retrieved chunks.

    Parameters
    ----------
    question:
        The natural-language financial question.
    chunks:
        Retrieved chunks from the Chroma index (output of retrieve()).

    Returns
    -------
    {
        "answer":        str | None,   # direct answer with value
        "fiscal_period": str | None,   # period Claude cited
        "source":        str | None,   # accession number
        "confidence":    str | None,   # HIGH | LOW
        "raw":           str,          # full Claude response
        "chunks_used":   int,          # number of chunks passed as context
    }
    """
    if not chunks:
        return {
            "answer": None, "fiscal_period": None,
            "source": None, "confidence": None,
            "raw": "", "chunks_used": 0,
        }

    context = _format_chunks(chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text
    result = _parse_response(raw)
    result["chunks_used"] = len(chunks)
    return result


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------

def generate_both(
    question: str,
    k: int = TOP_K,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    collection_name: str = DEFAULT_COLLECTION,
) -> dict:
    """
    Run full pipeline — retrieve then generate — for both filtered and baseline.

    This is the primary entry point for the evaluation harness and the
    retrieval failure report.

    model_name / collection_name select the embedding model and its dedicated
    Chroma collection for the embedding ablation; they pass straight through to
    retrieve_both() and default to the minilm build.

    Returns
    -------
    {
        "question":      str,
        "parsed_filter": dict,   # what the query parser extracted
        "filtered": {
            "retrieval":  dict,  # retrieve() result (chunks, filter_used, fallback)
            "answer":     str | None,
            "fiscal_period": str | None,
            "source":     str | None,
            "confidence": str | None,
            "raw":        str,
            "chunks_used": int,
        },
        "baseline": { ... }      # same shape
    }
    """
    retrieval = retrieve_both(question, k=k, model_name=model_name, collection_name=collection_name)

    filtered_gen = generate(question, retrieval["filtered"]["chunks"])
    baseline_gen = generate(question, retrieval["baseline"]["chunks"])

    return {
        "question":      question,
        "parsed_filter": retrieval["parsed_filter"],
        "filtered": {
            "retrieval": retrieval["filtered"],
            **filtered_gen,
        },
        "baseline": {
            "retrieval": retrieval["baseline"],
            **baseline_gen,
        },
    }
