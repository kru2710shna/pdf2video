# run_extract.py
from __future__ import annotations

import sys
from pathlib import Path

# --- make project root importable ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from scripts.ingest.pdf_loader import extract_pdf_text
from src.utils.io import ensure_dir, write_json, write_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: Extract PDF text page-by-page.")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument(
        "--out",
        default="data/extracted",
        help="Output base directory (default: data/extracted)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=50,
        help="Min chars threshold to trigger pdfplumber fallback (default: 50)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    result = extract_pdf_text(
        pdf_path,
        min_chars_threshold=args.min_chars,
        use_pdfplumber_fallback=True,
    )

    pdf_name = result["pdf_name"]
    out_dir = ensure_dir(Path(args.out) / pdf_name)

    # Write deliverables
    raw_text_path = out_dir / "raw_text.txt"
    pages_json_path = out_dir / "pages.json"

    write_text(raw_text_path, result["raw_text"])
    write_json(pages_json_path, result["pages"])

    # Print logs / stats
    stats = result["stats"]
    print("\n=== PDF2VIDEO :: STEP 1 EXTRACTION REPORT ===")
    print(f"PDF: {result['pdf_path']}")
    print(f"Pages: {result['page_count']}")
    print(f"Total chars: {stats['total_chars']}")
    print(f"Avg chars/page: {stats['avg_chars_per_page']:.2f}")
    print(f"Empty pages: {stats['empty_pages']}")
    print(f"Fallback used pages: {stats['fallback_used_pages']}")
    print(f"Saved: {raw_text_path}")
    print(f"Saved: {pages_json_path}")
    print("===========================================\n")


if __name__ == "__main__":
    main()
