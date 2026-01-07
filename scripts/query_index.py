# query_index.py
from __future__ import annotations

import sys
from pathlib import Path

# --- MUST be first: add repo root to sys.path -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- now safe to import everything else -------------------------------------
import argparse
import json
from typing import List, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.retrieval.index import load_chunks_jsonl


# -----------------------------------------------------------------------------
# Intent routing (rule-based)
# -----------------------------------------------------------------------------
CONTRIB_KEYWORDS = [
    "main contribution",
    "contribution",
    "key idea",
    "summary",
    "what is the paper about",
    "what does the paper propose",
]

CONTRIB_SECTIONS = {"abstract", "intro", "conclusion"}


def is_contrib_intent(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in CONTRIB_KEYWORDS)


def sec(c: dict) -> str:
    return (c.get("section") or "unknown").lower().strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2.3: Query FAISS index (intent-routed)")
    parser.add_argument("--pdf_name", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--extracted_dir", default="data/extracted")
    parser.add_argument("--model", default=None, help="Override model (otherwise auto from index_meta.json)")
    args = parser.parse_args()

    base = Path(args.extracted_dir) / args.pdf_name

    # Prefer indexed_chunks.jsonl if it exists (keeps FAISS ids aligned when drop_junk=True)
    indexed_chunks_path = base / "indexed_chunks.jsonl"
    chunks_path = indexed_chunks_path if indexed_chunks_path.exists() else (base / "chunks.jsonl")

    index_path = base / "faiss.index"
    meta_path = base / "index_meta.json"

    if not chunks_path.exists() or not index_path.exists():
        raise FileNotFoundError("Missing chunks jsonl or faiss.index. Run run_chunk.py then build_index.py.")

    # Load EXACT corpus that FAISS was built on
    chunks = load_chunks_jsonl(chunks_path, drop_junk=False, debug_tags=args.debug_tags)
    index = faiss.read_index(str(index_path))

    # Resolve model from meta unless overridden
    model_name: Optional[str] = args.model
    if model_name is None:
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}. Rebuild index or pass --model explicitly.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_name = meta.get("model_name")

    if not model_name:
        raise ValueError("Could not resolve model name. Pass --model or rebuild index.")

    model = SentenceTransformer(model_name)
    qv = model.encode([args.query], normalize_embeddings=True)
    qv = np.asarray(qv, dtype="float32")

    if qv.shape[1] != index.d:
        raise ValueError(
            f"Embedding dim mismatch: query_dim={qv.shape[1]} vs index_dim={index.d}. "
            f"Query model='{model_name}'. Fix: rebuild index with same model."
        )

    # ---- FAISS search (always broad) ----
    raw_k = max(50, args.k * 10)
    scores, ids = index.search(qv, raw_k)

    want_contrib = is_contrib_intent(args.query)

    # ---- pass 1: strict section filter for contribution intent ----
    selected: List[Tuple[float, dict]] = []
    selected_count = 0

    if want_contrib:
        for idx, score in zip(ids[0], scores[0]):
            i = int(idx)
            if i < 0 or i >= len(chunks):
                continue
            c = chunks[i]
            if sec(c) in CONTRIB_SECTIONS:
                selected.append((float(score), c))
                selected_count += 1
                if selected_count >= args.k:
                    break

        # ---- fallback if too few (use early pages) ----
        if selected_count < 2:
            selected = []
            selected_count = 0
            for idx, score in zip(ids[0], scores[0]):
                i = int(idx)
                if i < 0 or i >= len(chunks):
                    continue
                c = chunks[i]
                if int(c.get("page_start", 999999)) <= 3:
                    selected.append((float(score), c))
                    selected_count += 1
                    if selected_count >= args.k:
                        break
    else:
        # normal: take top-k directly
        for idx, score in zip(ids[0], scores[0]):
            i = int(idx)
            if i < 0 or i >= len(chunks):
                continue
            selected.append((float(score), chunks[i]))
            if len(selected) >= args.k:
                break

    print("\n=== STEP 2.3 RETRIEVAL RESULTS (INTENT-ROUTED) ===")
    print(f"Query: {args.query}")
    print(f"Model: {model_name}")
    print(f"Index dim: {index.d}")
    print(f"Chunks source: {chunks_path.name}")
    print(f"Intent: {'contribution/summary' if want_contrib else 'general'}\n")

    for rank, (score, c) in enumerate(selected, start=1):
        snippet = (c.get("text") or "").replace("\n", " ")
        snippet = (snippet[:220] + "...") if len(snippet) > 220 else snippet
        print(f"[{rank}] {c['chunk_id']} | {sec(c)} | pages {c['page_start']}-{c['page_end']} | score={score:.4f}")
        print(f"     {snippet}\n")

    print("===============================================\n")


if __name__ == "__main__":
    main()
