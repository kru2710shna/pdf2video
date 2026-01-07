# build_index.py

from __future__ import annotations

import sys
from pathlib import Path

# --- MUST be first: add repo root to sys.path -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import numpy as np
import faiss

from src.retrieval.index import (
    load_chunks_jsonl,
    build_faiss_index,
    save_chunks_jsonl,
    DEFAULT_EMBEDDING_MODEL,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2.2: Build FAISS index from chunks.jsonl")
    parser.add_argument("--pdf_name", required=True)
    parser.add_argument("--extracted_dir", default="data/extracted")
    parser.add_argument("--drop_junk", action="store_true")
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument( "--debug_tags", action="store_true", help="Print why each chunk got its section tag (from load_chunks_jsonl).",
)

    args = parser.parse_args()

    base = Path(args.extracted_dir) / args.pdf_name
    chunks_path = base / "chunks.jsonl"

    index_path = base / "faiss.index"
    indexed_chunks_path = base / "indexed_chunks.jsonl"
    emb_path = base / "embeddings.npy"
    meta_path = base / "index_meta.json"

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing {chunks_path}. Run scripts/run_chunk.py first.")

    # Load + tag + (optionally) drop junk
    chunks = load_chunks_jsonl(chunks_path, drop_junk=args.drop_junk, debug_tags=args.debug_tags)
    if not chunks:
        raise RuntimeError("No chunks left after filtering (drop_junk=True removed everything).")

    # Build index from text_embed (abstract cleaned)
    index, emb = build_faiss_index(
        chunks,
        model_name=args.model,
        batch_size=args.batch_size,
    )

    # Persist
    faiss.write_index(index, str(index_path))
    save_chunks_jsonl(chunks, indexed_chunks_path)

    if args.save_embeddings:
        np.save(str(emb_path), emb)

    meta = {
        "model_name": args.model,
        "dim": int(emb.shape[1]),
        "chunks_indexed": len(chunks),
        "drop_junk": bool(args.drop_junk),
        "has_section_tags": ("section" in chunks[0]),
        "has_text_embed": ("text_embed" in chunks[0]),
        "indexed_chunks_file": indexed_chunks_path.name,
        "batch_size": int(args.batch_size),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== STEP 2.2 INDEX BUILD REPORT ===")
    print(f"PDF: {args.pdf_name}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Model: {meta['model_name']}")
    print(f"Dim: {meta['dim']}")
    print(f"Drop junk: {meta['drop_junk']}")
    print(f"Section tags: {meta['has_section_tags']}")
    print(f"Text embed: {meta['has_text_embed']}")
    print(f"Saved: {index_path}")
    print(f"Saved: {indexed_chunks_path}")
    if args.save_embeddings:
        print(f"Saved: {emb_path}")
    print(f"Saved: {meta_path}")
    print("==================================\n")


if __name__ == "__main__":
    main()
