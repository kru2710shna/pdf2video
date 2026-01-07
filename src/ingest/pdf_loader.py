"""
pdf_loader.py
Step 1: Load a PDF and extract readable text page-by-page.

Strategy:
- Primary extractor: pypdf (fast)
- Fallback: pdfplumber (better layout extraction for tricky PDFs)
- Per-page selection: choose the extractor that yields more text
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pypdf import PdfReader



@dataclass
class PageText:
    page_num: int  # 1-indexed
    text: str
    extractor: str  # "pypdf" or "pdfplumber"
    char_count: int


def _extract_with_pypdf(pdf_path: Path) -> Tuple[List[str], int]:
    reader = PdfReader(str(pdf_path))
    pages_text: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        # normalize
        t = t.replace("\r\n", "\n").replace("\r", "\n").strip()
        pages_text.append(t)
    return pages_text, len(reader.pages)


def _extract_with_pdfplumber(pdf_path: Path, page_count_hint: Optional[int] = None) -> List[str]:
    # imported lazily so pypdf-only environments still work (but we recommend pdfplumber installed)
    import pdfplumber  # type: ignore

    pages_text: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        # sometimes page_count differs from pypdf due to parsing edge cases; trust pdfplumber here
        for i, page in enumerate(pdf.pages):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            t = t.replace("\r\n", "\n").replace("\r", "\n").strip()
            pages_text.append(t)

    # If pypdf gave a page_count_hint and plumber returned different count,
    # we still return what plumber sees. Caller will align using min length.
    return pages_text


def extract_pdf_text(
    pdf_path: str | Path,
    *,
    min_chars_threshold: int = 50,
    use_pdfplumber_fallback: bool = True,
) -> Dict:
    """
    Extract per-page text. For each page:
    - attempt pypdf
    - if too short, optionally try pdfplumber and choose longer result

    Returns dict:
    {
      "pdf_name": "...",
      "pdf_path": "...",
      "page_count": N,
      "pages": [ {page_num, text, extractor, char_count}, ... ],
      "raw_text": "...",
      "stats": {...}
    }
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF: {pdf_path}")

    pypdf_pages, page_count = _extract_with_pypdf(pdf_path)

    plumber_pages: Optional[List[str]] = None
    if use_pdfplumber_fallback:
        try:
            plumber_pages = _extract_with_pdfplumber(pdf_path, page_count_hint=page_count)
        except Exception:
            plumber_pages = None  # if pdfplumber fails, we still proceed with pypdf

    pages: List[PageText] = []
    chosen_chars_total = 0
    pypdf_chars_total = 0
    plumber_chars_total = 0
    fallback_used_pages = 0

    # align lengths safely
    max_len = len(pypdf_pages)
    if plumber_pages is not None:
        max_len = min(max_len, len(plumber_pages))

    for idx in range(max_len):
        t1 = pypdf_pages[idx] or ""
        pypdf_chars = len(t1)
        pypdf_chars_total += pypdf_chars

        chosen_text = t1
        chosen_extractor = "pypdf"

        if plumber_pages is not None:
            t2 = plumber_pages[idx] or ""
            plumber_chars = len(t2)
            plumber_chars_total += plumber_chars

            # If pypdf is weak, or plumber is clearly better, choose plumber
            if pypdf_chars < min_chars_threshold and plumber_chars > pypdf_chars:
                chosen_text = t2
                chosen_extractor = "pdfplumber"
                fallback_used_pages += 1
            else:
                # also allow "plumber wins by a lot" even if pypdf is above threshold
                if plumber_chars > pypdf_chars * 1.25 and plumber_chars > min_chars_threshold:
                    chosen_text = t2
                    chosen_extractor = "pdfplumber"
                    fallback_used_pages += 1

        chosen_chars = len(chosen_text)
        chosen_chars_total += chosen_chars

        pages.append(
            PageText(
                page_num=idx + 1,
                text=chosen_text,
                extractor=chosen_extractor,
                char_count=chosen_chars,
            )
        )

    raw_text = "\n\n".join(
        [f"--- Page {p.page_num} ({p.extractor}) ---\n{p.text}" for p in pages]
    ).strip()

    avg_chars = (chosen_chars_total / len(pages)) if pages else 0.0
    empty_pages = sum(1 for p in pages if p.char_count == 0)

    return {
        "pdf_name": pdf_path.stem,
        "pdf_path": str(pdf_path),
        "page_count": len(pages),
        "pages": [
            {
                "page_num": p.page_num,
                "text": p.text,
                "extractor": p.extractor,
                "char_count": p.char_count,
            }
            for p in pages
        ],
        "raw_text": raw_text,
        "stats": {
            "total_chars": chosen_chars_total,
            "avg_chars_per_page": avg_chars,
            "empty_pages": empty_pages,
            "fallback_used_pages": fallback_used_pages,
            "pypdf_total_chars": pypdf_chars_total,
            "pdfplumber_total_chars": plumber_chars_total if plumber_pages is not None else None,
        },
    }
