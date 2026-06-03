"""
config.py — Central configuration for the XBRL ingestion path.

Reads from .env (via python-dotenv). Fails fast at import time if the SEC
User-Agent is missing or still holds the example placeholder — no silent 403s.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SEC_USER_AGENT: str = os.getenv("SEC_USER_AGENT", "").strip()
if not SEC_USER_AGENT or "YOUR_EMAIL_HERE" in SEC_USER_AGENT:
    raise EnvironmentError(
        "SEC_USER_AGENT is not configured.\n"
        "  1. Copy .env.example → .env\n"
        '  2. Set: SEC_USER_AGENT=Firstname Lastname your@email.com\n'
        "SEC requires a real contact email; a missing one returns 403."
    )

EDGAR_CACHE_DIR: Path = Path(os.getenv("EDGAR_CACHE_DIR", "./edgar_cache"))
EDGAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_DIR: Path = Path(os.getenv("CHROMA_DIR", "./chroma_db"))
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Safety flag: True blocks any code path that would write to a vector store / DB.
# Flip to False only after dry-run output has been manually verified correct.
DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes")

# Anthropic API key — only required when running generation (Step 4+).
# Intentionally not validated at import time so ingestion/retrieval scripts
# work without it.
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# Vector store and embedding model — single source of truth used by both
# build_index.py and retriever.py. Change here when running ablations so
# the index and retriever always stay in sync.
COLLECTION_NAME: str = "financebench_xbrl"
EMBED_MODEL: str = "all-MiniLM-L6-v2"
