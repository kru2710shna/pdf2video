# chunker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import re

PAGE_MARK_RE = re.compile(r"\[PAGE\s+(\d+)\]")
BRACKET_REF_RE = re.compile(r"\[\s*\d+\s*\]")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    char_count: int
    section: str  # strict tags


def infer_page_range_from_text(text: str) -> Tuple[int, int]:
    nums = [int(m.group(1)) for m in PAGE_MARK_RE.finditer(text)]
    if not nums:
        return (1, 1)
    return (min(nums), max(nums))


def strict_section_tag(text: str) -> str:
    """
    HARD string matching only. No heuristics.
    If the literal header isn't present, return 'unknown'.
    """
    t = text

    # common patterns in your pipeline: "[PAGE 1]\nAbstract", "\n1 Introduction", etc.
    if re.search(r"(\n|\r)\s*Abstract(\s|\r|\n)", t):
        return "abstract"
    if re.search(r"(\n|\r)\s*1\s*Introduction(\s|\r|\n)", t) or re.search(r"(\n|\r)\s*Introduction(\s|\r|\n)", t):
        return "intro"
    if re.search(r"(\n|\r)\s*(Conclusion|Conclusions)(\s|\r|\n)", t):
        return "conclusion"
    if re.search(r"(\n|\r)\s*(Related Work)(\s|\r|\n)", t):
        return "related_work"
    if re.search(r"(\n|\r)\s*(Experiments?|Results?)(\s|\r|\n)", t):
        return "results"

    return "unknown"


def pages_to_chunks(
    pages: List[Dict],
    pdf_name: str,
    *,
    min_chars: int = 1500,
    max_chars: int = 2500,
    overlap_chars: int = 250,
) -> List[Chunk]:
    if overlap_chars >= min_chars:
        raise ValueError("overlap_chars must be < min_chars")

    chunks: List[Chunk] = []
    buf = ""
    idx = 0

    def _new_id(i: int) -> str:
        return f"{pdf_name}_{i:04d}"

    for p in pages:
        page_num = int(p["page_num"])
        page_text = (p.get("text") or "")
        addition = f"\n\n[PAGE {page_num}]\n{page_text}"

        if len(buf) + len(addition) <= max_chars:
            buf += addition
            continue

        remainder = addition
        if len(buf) < min_chars:
            remaining = max_chars - len(buf)
            if remaining > 0:
                buf += addition[:remaining]
                remainder = addition[remaining:]

        chunk_text = buf.strip()
        ps, pe = infer_page_range_from_text(chunk_text)
        sec = strict_section_tag(chunk_text)

        chunks.append(
            Chunk(
                chunk_id=_new_id(idx),
                text=chunk_text,
                page_start=ps,
                page_end=pe,
                char_count=len(chunk_text),
                section=sec,
            )
        )
        idx += 1

        overlap = buf[-overlap_chars:] if overlap_chars > 0 else ""
        buf = (overlap + remainder).strip()

        while len(buf) > max_chars:
            part = buf[:max_chars].strip()
            ps, pe = infer_page_range_from_text(part)
            sec = strict_section_tag(part)

            chunks.append(
                Chunk(
                    chunk_id=_new_id(idx),
                    text=part,
                    page_start=ps,
                    page_end=pe,
                    char_count=len(part),
                    section=sec,
                )
            )
            idx += 1

            cut = max_chars - overlap_chars if overlap_chars > 0 else max_chars
            buf = buf[cut:].strip()

    if buf.strip():
        chunk_text = buf.strip()
        ps, pe = infer_page_range_from_text(chunk_text)
        sec = strict_section_tag(chunk_text)
        chunks.append(
            Chunk(
                chunk_id=_new_id(idx),
                text=chunk_text,
                page_start=ps,
                page_end=pe,
                char_count=len(chunk_text),
                section=sec,
            )
        )

    return chunks
