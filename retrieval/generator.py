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
  Module docstring + file map
  Imports + CONFIG (model, token limit, k, CONCEPT_GLOSSARY)
  Lazy client singleton
  Prompt construction — system prompt + concept-label + chunk formatter
  Response parser — extracts structured fields from Claude output
  generate() — single-mode API call
  generate_both() — primary entry point, calls both modes

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

# Plain-English label for every XBRL concept in DEFAULT_CONCEPTS.
# WHY: chunks are tagged with raw XBRL names (PaymentsToAcquirePropertyPlantAndEquipment)
# but questions use plain terms ("capital expenditure"). Without the bridge, Claude
# fails to connect the two and answers "Unable to determine" even with the right chunk
# in context — the dominant cause of the 69%-retrieval / 5%-answer accuracy gap.
# CHANGE ME: add an entry whenever you add a concept to DEFAULT_CONCEPTS.
CONCEPT_GLOSSARY: dict[str, str] = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue (net sales, ASC 606)",
    "Revenues":                                            "Revenue (net sales)",
    "NetIncomeLoss":                                       "Net income (net earnings)",
    "EarningsPerShareBasic":                               "Basic EPS (earnings per share)",
    "EarningsPerShareDiluted":                             "Diluted EPS (earnings per share)",
    "OperatingIncomeLoss":                                 "Operating income (EBIT)",
    "GrossProfit":                                         "Gross profit",
    "CostOfGoodsAndServicesSold":                          "Cost of goods sold (COGS)",
    "NetCashProvidedByUsedInOperatingActivities":          "Operating cash flow (cash from operations)",
    "PaymentsToAcquirePropertyPlantAndEquipment":          "Capital expenditures (CapEx, PP&E purchases)",
    "Assets":                                              "Total assets",
    "AssetsCurrent":                                       "Current assets",
    "CashAndCashEquivalentsAtCarryingValue":               "Cash and cash equivalents",
    "InventoryNet":                                        "Inventory (net)",
    "AccountsReceivableNetCurrent":                        "Accounts receivable (net AR)",
    "LiabilitiesCurrent":                                  "Current liabilities",
}

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

Each chunk header shows the XBRL concept name and its plain-English equivalent \
in parentheses, e.g. "NetIncomeLoss (Net income)". Use the plain-English name to \
match the question — a question about "capital expenditure" is answered by the \
chunk labelled "CapEx", and so on.

Always structure your response as:
ANSWER: <your direct answer, including the specific dollar amount or value>
FISCAL_PERIOD: <the fiscal period this answer covers, e.g. FY2022-Q1 or FY2022>
SOURCE: <the SEC accession number of the chunk you relied on most>
CONFIDENCE: <HIGH if the answer is directly stated in a chunk / LOW if inferred>"""


def _concept_label(concept: str) -> str:
    """
    Render a concept as "XBRLName (Plain English)" for the chunk header.

    WHY: the plain-English label is what lets Claude match the XBRL tag to the
    question's wording. Concepts absent from the glossary fall back to the bare
    XBRL name (no empty parentheses) so the header degrades gracefully.
    """
    plain = CONCEPT_GLOSSARY.get(concept)
    return f"{concept} ({plain})" if plain else concept


def _format_chunks(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] {c.get('entity', '')} | {c.get('fiscal_period', '?')} | "
            f"{_concept_label(c.get('concept', ''))} | accession: {c.get('accession', '?')}\n"
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
