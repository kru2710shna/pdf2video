from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"
PAGE_MARK_RE = re.compile(r"\[PAGE\s+(\d+)\]", re.IGNORECASE)

# -----------------------------------------------------------------------------
# Deterministic section tagging (NO guessing)
# Tag only if explicit heading tokens exist.
# Otherwise: "unknown"
# -----------------------------------------------------------------------------
_HEADING_PATTERNS: List[Tuple[str, str, re.Pattern]] = [
    ("abstract", "ABSTRACT_HEADING", re.compile(r"(?im)^\s*abstract\s*$")),
    ("intro", "INTRO_HEADING", re.compile(r"(?im)^\s*(?:\d+\s+)?introduction\s*$")),
    ("related_work", "RELATED_WORK_HEADING", re.compile(r"(?im)^\s*(?:\d+\s+)?related\s+work\s*$")),
    ("conclusion", "CONCLUSION_HEADING", re.compile(r"(?im)^\s*(?:\d+\s+)?conclusion(s)?\s*$")),
    ("references", "REFERENCES_HEADING", re.compile(r"(?im)^\s*(references|bibliography)\s*$")),
]


def detect_section_with_reason(text: str) -> Tuple[str, str, str]:
    """
    Returns: (section, reason_name, matched_text)

    Deterministic rules:
    - Standalone heading lines = strongest
    - Inline detection ONLY for Abstract + Introduction (arXiv reality)
    - DO NOT inline-detect conclusion/related/references (false positives like "in conclusion")
    """
    if not text:
        return ("unknown", "EMPTY", "")

    # 1) Standalone headings
    for section, reason, pat in _HEADING_PATTERNS:
        m = pat.search(text)
        if m:
            snippet = text[m.start():m.end()].strip()
            return (section, reason, snippet[:80])

    # 2) Inline only for Abstract/Intro near beginning
    head = text[:500]

    m = re.search(r"\bAbstract\b", head)
    if m:
        return ("abstract", "ABSTRACT_INLINE", head[m.start():m.start() + 80].strip())

    m = re.search(r"\b\d+\s+Introduction\b", head) or re.search(r"\bIntroduction\b", head)
    if m:
        return ("intro", "INTRO_INLINE", head[m.start():m.start() + 80].strip())

    return ("unknown", "NO_MATCH", "")



def tag_section_deterministic(text: str) -> str:
    section, _, _ = detect_section_with_reason(text)
    return section





def tag_section_deterministic(text: str) -> str:
    """
    Deterministic, literal matching only.
    Handles two common PDF extraction realities:

    1) Headings appear as standalone lines (best case)
    2) arXiv sometimes inlines: "... emails ... Abstract We present ..."
       We still treat this as deterministic because it is literal token matching.
    """
    if not text:
        return "unknown"

    # 1) strict standalone heading lines
    for name, pat in _HEADING_PATTERNS:
        if pat.search(text):
            return name

    # 2) inline headings near the beginning (still literal matching)
    head = text[:500]

    if re.search(r"\bAbstract\b", head):
        return "abstract"
    if re.search(r"\bIntroduction\b", head):
        return "intro"
    if re.search(r"\bRelated\s+Work\b", head):
        return "related_work"
    if re.search(r"\bConclusions?\b", head):
        return "conclusion"

    return "unknown"


# -----------------------------------------------------------------------------
# Citation-density junk filter (Step 2.4)
# Prevents "In International Conference on..." reference-heavy chunks polluting FAISS.
# -----------------------------------------------------------------------------
def looks_like_reference_signature(text: str) -> bool:
    if not text:
        return False

    bracket_refs = len(re.findall(r"\[\d+\]", text))
    year_hits = len(re.findall(r"\b(19|20)\d{2}\b", text))
    venue_hits = sum(
        text.count(x) for x in ["Conference", "Proceedings", "arXiv", "NeurIPS", "ICLR", "ICML"]
    )

    words = re.findall(r"\w+", text)
    commas = text.count(",")
    comma_density = commas / max(1, len(words))

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    short_lines_frac = sum(1 for ln in lines if len(ln) < 60) / max(1, len(lines))

    # Drop rules (simple + effective)
    return (
        bracket_refs >= 8
        or (year_hits >= 6 and venue_hits >= 2)
        or (bracket_refs >= 5 and comma_density > 0.08)
        or (short_lines_frac > 0.45 and bracket_refs >= 4)
    )


# -----------------------------------------------------------------------------
# Abstract embed cleaning
# Strip title/authors/emails BEFORE "Abstract" (supports inline Abstract too)
# -----------------------------------------------------------------------------
def extract_after_abstract_token(text: str) -> str:
    """
    Return substring starting at the FIRST literal "Abstract" token.
    Supports:
      - inline: "... Abbeel ... Abstract We present ..."
      - standalone heading line: "\nAbstract\nWe present..."
    """
    if not text:
        return text

    m = re.search(r"\bAbstract\b", text)
    if not m:
        return text.strip()

    return text[m.start():].strip()


def make_text_for_embedding(text_raw: str, section: str) -> str:
    """
    Create embedding text that is better aligned to semantic retrieval.

    - abstract: strip everything before first "Abstract"
    - everything else: unchanged
    """
    sec = (section or "unknown").lower()
    if sec == "abstract":
        return extract_after_abstract_token(text_raw)
    return (text_raw or "").strip()


# -----------------------------------------------------------------------------
# Page range inference (optional)
# -----------------------------------------------------------------------------
def infer_page_range_from_text(text: str) -> Tuple[int, int]:
    nums = [int(m.group(1)) for m in PAGE_MARK_RE.finditer(text or "")]
    if not nums:
        return (1, 1)
    return (min(nums), max(nums))


# -----------------------------------------------------------------------------
# JSONL loading
# Adds:
# - text_raw (original text)
# - section (deterministic)
# - text_embed (abstract-cleaned for embeddings)
# -----------------------------------------------------------------------------
def load_chunks_jsonl(
    chunks_path: str | Path,
    *,
    drop_junk: bool = True,
    debug_tags: bool = False,
) -> List[Dict]:
    chunks_path = Path(chunks_path)
    chunks: List[Dict] = []

    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)

            text_raw = (c.get("text") or "").strip()
            c["text_raw"] = text_raw

            section, reason, match = detect_section_with_reason(text_raw)
            c["section"] = section

            if debug_tags:
                cid = c.get("chunk_id", "UNKNOWN_CHUNK")
                shown = match.replace("\n", " ").strip()[:80]
                print(f"{cid} tagged={section} via={reason} match=\"{shown}\"")

            # Junk drop (citation-density)
            if drop_junk and looks_like_reference_signature(text_raw):
                continue

            c["text_embed"] = make_text_for_embedding(text_raw, section)
            chunks.append(c)

    return chunks



def save_chunks_jsonl(chunks: List[Dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# FAISS build
# IMPORTANT:
# - embed text_embed
# - index ids match order of 'chunks' list you pass in
# -----------------------------------------------------------------------------
def build_faiss_index(
    chunks: List[Dict],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
) -> Tuple[faiss.Index, np.ndarray]:
    if not chunks:
        raise ValueError("No chunks provided to build_faiss_index().")

    texts = [
        (c.get("text_embed") or c.get("text_raw") or c.get("text") or "").strip()
        for c in chunks
    ]

    model = SentenceTransformer(model_name)

    emb = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=int(batch_size),
        show_progress_bar=True,
    )
    emb = np.asarray(emb, dtype="float32")

    dim = int(emb.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index, emb
