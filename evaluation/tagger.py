#!/usr/bin/env python3
"""
period_tagging_smoke_test.py

Stage-1 smoke test for the fiscal-period-aware retrieval project.
Compares a REGEX period tagger (baseline) against a SIMILARITY period tagger
("v1" stand-in) on the task: given a chunk, assign its fiscal period.
Scores overall AND per difficulty stratum (explicit / variant / implicit).

>>> READ THIS BEFORE RUNNING <<<
- Data here is SYNTHETIC (hand-written to control difficulty). Not real EDGAR text.
- The similarity tagger uses sentence-transformers IF available, else a TF-IDF
  char-ngram fallback so it runs offline. The fallback is a PROXY, not the real
  embedding model. Numbers from a fallback run validate the harness, not the thesis.
- To make this real: replace CURATED_DATASET with real chunks (step-2 XBRL puller)
  and run on a machine that can load a sentence-transformer.

FILE MAP
  L001-L033  Module docstring + this map
  L035-L052  CONFIG knobs (CHANGE ME / TWEAK)
  L054-L121  CURATED_DATASET (synthetic, 3 strata)
  L123-L146  Label / date normalization helpers
  L148-L205  REGEX TAGGER (the baseline)
  L207-L257  SIMILARITY TAGGER ("v1" stand-in; ST if available, else TF-IDF)
  L259-L304  SCORING + REPORT
  L306-L324  main()
"""

import re
import sys
from collections import defaultdict

# ============================== CONFIG ======================================
# CHANGE ME: if a company's fiscal year does NOT end in December, the regex
# month->quarter mapping below is wrong for them. Calendar-year is assumed here.
ASSUME_CALENDAR_FISCAL_YEAR = True  # DO NOT TOUCH unless you also fix MONTH_END_TO_Q

# TWEAK: similarity tagger abstains (predicts None) below this cosine score.
SIM_ABSTAIN_THRESHOLD = 0.18

# TWEAK: char-ngram range for the offline TF-IDF fallback embedder.
TFIDF_NGRAM_RANGE = (2, 5)
# ============================================================================


# ============================ CURATED DATASET ===============================
# Each row: (text, true_label, stratum). Labels: "FY2023-Q1", "FY2023" (annual),
# or None. Implicit rows carry the period a human-WITH-context would assign, but
# it is NOT recoverable from the chunk text alone -- both taggers should miss them.
CURATED_DATASET = [
    # ---- explicit: canonical phrasing, regex should nail these ----
    ("Net sales for Q1 2023 were $4.2 billion, up 6% year over year.", "FY2023-Q1", "explicit"),
    ("During the second quarter of 2022, operating income increased to $812 million.", "FY2022-Q2", "explicit"),
    ("For the quarter ended September 30, 2023, the Company reported diluted EPS of $1.34.", "FY2023-Q3", "explicit"),
    ("Revenue for the three months ended December 31, 2022 totaled $9.1 billion.", "FY2022-Q4", "explicit"),
    ("In Q4 2023, free cash flow reached a record $2.7 billion.", "FY2023-Q4", "explicit"),
    ("Fiscal year 2022 net income was $5.4 billion.", "FY2022", "explicit"),
    ("For the year ended December 31, 2023, total revenue grew 11%.", "FY2023", "explicit"),
    ("Q3 2022 gross margin expanded 120 basis points to 38.4%.", "FY2022-Q3", "explicit"),
    ("First quarter 2024 bookings rose to $1.8 billion.", "FY2024-Q1", "explicit"),
    ("The quarter ended June 30, 2024 saw R&D expense of $640 million.", "FY2024-Q2", "explicit"),
    ("Full year 2023 adjusted EBITDA was $3.2 billion.", "FY2023", "explicit"),
    ("Second quarter 2023 operating cash flow was $1.1 billion.", "FY2023-Q2", "explicit"),
    ("For the three months ended March 31, 2024, revenue was $5.6 billion.", "FY2024-Q1", "explicit"),
    ("Q2 2022 results included a $200 million restructuring charge.", "FY2022-Q2", "explicit"),
    ("Net income for fiscal year 2024 totaled $6.9 billion.", "FY2024", "explicit"),

    # ---- variant: real period present but in a form regex may miss ----
    ("FY23 revenue climbed to $18.5 billion.", "FY2023", "variant"),
    ("1Q24 net sales were $4.9 billion.", "FY2024-Q1", "variant"),
    ("Results for fiscal 2022 reflected strong demand.", "FY2022", "variant"),
    ("3Q22 operating margin was 21%.", "FY2022-Q3", "variant"),
    ("For the three months ended 3/31/23, EPS was $0.88.", "FY2023-Q1", "variant"),
    ("Sales in the first quarter of fiscal year 2024 rose 7%.", "FY2024-Q1", "variant"),
    ("FY2022 saw record shipments.", "FY2022", "variant"),
    ("2Q23 cash from operations totaled $980 million.", "FY2023-Q2", "variant"),
    ("As of December 31, 2022, total assets were $54 billion.", "FY2022", "variant"),
    ("Revenue for 4Q FY2023 was $6.2 billion.", "FY2023-Q4", "variant"),
    ("The fourth fiscal quarter of 2022 closed with $1.2 billion in orders.", "FY2022-Q4", "variant"),
    ("For the period ended 9/30/24, the company recorded a gain.", "FY2024-Q3", "variant"),
    ("FY 2023 fourth quarter revenue was $6.0 billion.", "FY2023-Q4", "variant"),
    ("In fiscal '22, margins compressed.", "FY2022", "variant"),
    ("Three-month results ending June 2024 showed growth.", "FY2024-Q2", "variant"),

    # ---- implicit: period NOT recoverable from chunk text alone ----
    ("Revenue increased 12% compared with the prior-year period.", "FY2023-Q2", "implicit"),
    ("Operating income declined versus the same quarter a year earlier.", "FY2023-Q3", "implicit"),
    ("Compared to the preceding fiscal year, expenses fell 4%.", "FY2023", "implicit"),
    ("The current quarter benefited from favorable foreign exchange.", "FY2024-Q1", "implicit"),
    ("Sequential improvement was driven by higher volumes.", "FY2024-Q2", "implicit"),
    ("Year-over-year, gross margin was roughly flat.", "FY2022-Q4", "implicit"),
    ("Results for the comparable prior period have been restated.", "FY2021", "implicit"),
    ("This quarter's bookings exceeded the trailing four-quarter average.", "FY2023-Q4", "implicit"),
    ("Relative to last year, headcount grew 8%.", "FY2024", "implicit"),
    ("The prior quarter included a one-time tax benefit.", "FY2022-Q1", "implicit"),
    ("Management reaffirmed guidance issued earlier in the year.", "FY2023", "implicit"),
    ("Amounts for the earlier period reflect the new revenue standard.", "FY2022", "implicit"),
]
# ============================================================================


# ====================== NORMALIZATION HELPERS ===============================
WORD_QUARTER = {"first": 1, "second": 2, "third": 3, "fourth": 4}
MONTH_TO_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
# DO NOT TOUCH: quarter-END month -> quarter number, calendar-year assumption.
# Reason: this encoding is exactly the regex brittleness we are measuring against.
MONTH_END_TO_Q = {3: 1, 6: 2, 9: 3, 12: 4}


def norm_year(y: str) -> int:
    """Two-digit -> 20xx; four-digit passthrough."""
    y = y.strip().lstrip("'")
    return 2000 + int(y) if len(y) == 2 else int(y)


def make_label(year: int, quarter=None) -> str:
    return f"FY{year}-Q{quarter}" if quarter else f"FY{year}"
# ============================================================================


# ============================== REGEX TAGGER ================================
# Baseline. Returns the FIRST confident period found, else None.
# Note: taking the first match is itself a known limitation for multi-period
# chunks -- that is part of what the learned method is meant to beat.
def regex_tag(text: str):
    t = text

    # 1) "Q1 2023", "Q1 FY2023"
    m = re.search(r"\bQ([1-4])\s*(?:of\s*)?(?:FY\s*)?(\d{4})\b", t, re.I)
    if m:
        return make_label(norm_year(m.group(2)), int(m.group(1)))

    # 2) "1Q23", "1Q2023", "4Q FY2023"
    m = re.search(r"\b([1-4])Q\s*(?:FY\s*)?(\d{2,4})\b", t, re.I)
    if m:
        return make_label(norm_year(m.group(2)), int(m.group(1)))

    # 3) "first/second/third/fourth quarter [of] [fiscal [year]] 2024"
    m = re.search(r"\b(first|second|third|fourth)\s+(?:fiscal\s+)?quarter\s+(?:of\s+)?"
                  r"(?:fiscal\s+(?:year\s+)?)?(\d{4})\b", t, re.I)
    if m:
        return make_label(norm_year(m.group(2)), WORD_QUARTER[m.group(1).lower()])

    # 4) "(quarter|three months|period) ended <Month> <day>, <year>"
    m = re.search(r"(?:quarter|three\s+months|period)\s+ended\s+([A-Za-z]+)\s+\d{1,2},?\s*(\d{4})", t, re.I)
    if m:
        mn = MONTH_TO_NUM.get(m.group(1).lower())
        q = MONTH_END_TO_Q.get(mn) if mn else None
        if q:
            return make_label(norm_year(m.group(2)), q)

    # 5) "(... ended) M/D/YY" numeric date
    m = re.search(r"(?:ended|ending)\s+(\d{1,2})/(\d{1,2})/(\d{2,4})", t, re.I)
    if m:
        mn = int(m.group(1))
        q = MONTH_END_TO_Q.get(mn)
        if q:
            return make_label(norm_year(m.group(3)), q)

    # 6) "year ended December 31, 2023" / "full year 2023" / "fiscal year 2024"
    m = re.search(r"(?:year\s+ended\s+[A-Za-z]+\s+\d{1,2},?\s*(\d{4}))", t, re.I)
    if m:
        return make_label(norm_year(m.group(1)))
    m = re.search(r"\b(?:full\s+year|fiscal\s+year|fiscal)\s+(\d{4})\b", t, re.I)
    if m:
        return make_label(norm_year(m.group(1)))

    # 7) "FY23" / "FY2023" / "FY 2023" (annual, no quarter nearby)
    m = re.search(r"\bFY\s*(\d{2,4})\b", t, re.I)
    if m:
        return make_label(norm_year(m.group(1)))

    return None
# ============================================================================


# =========================== SIMILARITY TAGGER ==============================
# "v1" stand-in: 1-NN over the labeled set (leave-one-out). Uses a real
# sentence-transformer if importable+loadable, else TF-IDF char-ngrams offline.
def build_vectorizer():
    try:
        from sentence_transformers import SentenceTransformer  # noqa
        model = SentenceTransformer("all-MiniLM-L6-v2")  # CHANGE ME: try a finance model

        def encode(texts):
            return model.encode(texts, normalize_embeddings=True)
        return encode, "sentence-transformers (all-MiniLM-L6-v2)"
    except Exception as e:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=TFIDF_NGRAM_RANGE)

        def encode(texts):
            X = vec.fit_transform(texts).toarray()
            n = np.linalg.norm(X, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return X / n
        return encode, f"TF-IDF char-ngram FALLBACK (no embedding model: {type(e).__name__})"


def similarity_tag_all(rows):
    import numpy as np
    encode, backend = build_vectorizer()
    texts = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    V = np.asarray(encode(texts))
    sims = V @ V.T            # cosine (vectors are normalized)
    np.fill_diagonal(sims, -1.0)  # leave-one-out: ignore self
    preds = []
    for i in range(len(rows)):
        j = int(np.argmax(sims[i]))
        preds.append(labels[j] if sims[i][j] >= SIM_ABSTAIN_THRESHOLD else None)
    return preds, backend
# ============================================================================


# ============================ SCORING + REPORT ==============================
def evaluate(rows):
    regex_preds = [regex_tag(r[0]) for r in rows]
    sim_preds, backend = similarity_tag_all(rows)

    by = defaultdict(lambda: {"n": 0, "regex": 0, "sim": 0})
    misses = {"regex": [], "sim": []}
    for r, rp, sp in zip(rows, regex_preds, sim_preds):
        text, true, stratum = r
        by[stratum]["n"] += 1
        by["ALL"]["n"] += 1
        if rp == true:
            by[stratum]["regex"] += 1; by["ALL"]["regex"] += 1
        else:
            misses["regex"].append((stratum, text, true, rp))
        if sp == true:
            by[stratum]["sim"] += 1; by["ALL"]["sim"] += 1
        else:
            misses["sim"].append((stratum, text, true, sp))

    print(f"\nBackend for similarity tagger: {backend}\n")
    print(f"{'stratum':<10}{'n':>4}{'regex acc':>12}{'sim acc':>10}")
    print("-" * 36)
    for s in ["explicit", "variant", "implicit", "ALL"]:
        d = by[s]
        ra = d["regex"] / d["n"] if d["n"] else 0
        sa = d["sim"] / d["n"] if d["n"] else 0
        print(f"{s:<10}{d['n']:>4}{ra:>11.0%}{sa:>10.0%}")

    print("\nSample regex misses (stratum | true -> pred):")
    for stratum, text, true, pred in misses["regex"][:6]:
        print(f"  [{stratum}] {true} -> {pred}  | {text[:64]}")
    return by
# ============================================================================


def main():
    print("=" * 60)
    print("PERIOD-TAGGING SMOKE TEST  (synthetic data; harness check)")
    print("=" * 60)
    evaluate(CURATED_DATASET)
    print("\nReminder: synthetic data + (likely) fallback embedder.")
    print("These numbers prove the harness runs, NOT the thesis.")


if __name__ == "__main__":
    sys.exit(main())
