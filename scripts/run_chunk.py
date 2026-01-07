# run_chunk.py
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

from src.ingest.chunker import pages_to_chunks
from src.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2.1: pages.json -> chunks.jsonl")
    parser.add_argument(
        "--pdf_name", required=True, help="Folder name under data/extracted/<pdf_name>"
    )
    parser.add_argument("--extracted_dir", default="data/extracted")
    parser.add_argument("--min_chars", type=int, default=1500)
    parser.add_argument("--max_chars", type=int, default=2500)
    parser.add_argument("--overlap", type=int, default=250)
    parser.add_argument("--target", type=int, default=2000)
    args = parser.parse_args()

    base = Path(args.extracted_dir) / args.pdf_name
    pages_path = base / "pages.json"
    if not pages_path.exists():
        raise FileNotFoundError(f"Missing: {pages_path}")

    pages = json.loads(pages_path.read_text(encoding="utf-8"))

    chunks = pages_to_chunks(
        pages,
        pdf_name=args.pdf_name,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        overlap_chars=args.overlap,
    )

    out_path = base / "chunks.jsonl"
    ensure_dir(out_path.parent)

    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "char_count": c.char_count,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # logs
    total_chars = sum(c.char_count for c in chunks)
    avg_chars = total_chars / max(1, len(chunks))
    print("\n=== STEP 2.1 CHUNKING REPORT ===")
    print(f"PDF: {args.pdf_name}")
    print(f"Chunks: {len(chunks)}")
    print(f"Total chunk chars: {total_chars}")
    print(f"Avg chars/chunk: {avg_chars:.2f}")
    print(f"Saved: {out_path}")
    print("================================\n")


if __name__ == "__main__":
    main()
