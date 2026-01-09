# src/render/pdf_render.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import re
import os

from PIL import Image, ImageDraw

import fitz  # PyMuPDF
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))




@dataclass
class RenderConfig:
    dpi: int = 180
    highlight_fill_rgba: Tuple[int, int, int, int] = (255, 235, 59, 110)  # yellow-ish
    highlight_outline_rgba: Tuple[int, int, int, int] = (255, 193, 7, 220)
    pad_px: int = 18  # padding around highlight rects


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _phrase_variants(phrase: str) -> List[str]:
    """Generate multiple search strings to survive PDF weirdness."""
    p = _norm_ws(phrase)
    variants = [p]

    # Lowercase variant (search_for is case-sensitive-ish depending on flags)
    variants.append(p.lower())

    # Remove punctuation
    variants.append(re.sub(r"[^\w\s]", "", p))
    variants.append(re.sub(r"[^\w\s]", "", p).lower())

    # Collapse spaces around commas/periods etc. (already removed above, but keep simple)
    return list(dict.fromkeys([v for v in variants if v]))


def _ngrams(words: List[str], n: int) -> List[str]:
    return [" ".join(words[i : i + n]) for i in range(0, max(0, len(words) - n + 1))]


def _fallback_chunks(phrase: str) -> List[str]:
    """
    If a long phrase fails, try shorter sub-phrases (5-grams then 4 then 3).
    This is the single biggest “it works now” trick.
    """
    words = _norm_ws(phrase).split()
    if len(words) <= 6:
        return [_norm_ws(phrase)]
    out = []
    for n in (5, 4, 3):
        out.extend(_ngrams(words, n))
    # keep unique, preserve order
    seen = set()
    final = []
    for x in out:
        if x not in seen:
            final.append(x)
            seen.add(x)
    return final[:25]  # cap


def _search_rects(page: fitz.Page, phrase: str) -> List[fitz.Rect]:
    """
    Try strict phrase search; if it fails, try variants; if still fails, try n-gram fallback.
    """
    rects: List[fitz.Rect] = []

    # Flags help with hyphenation sometimes
    flags = fitz.TEXT_DEHYPHENATE

    # 1) direct + variants
    for v in _phrase_variants(phrase):
        rects = page.search_for(v, flags=flags)
        if rects:
            return rects

    # 2) fallback n-grams for long phrases
    for chunk in _fallback_chunks(phrase):
        for v in _phrase_variants(chunk):
            rects2 = page.search_for(v, flags=flags)
            rects.extend(rects2)

    return rects


def render_page_png(pdf_path: str, page_1based: int, out_path: str, cfg: RenderConfig) -> None:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_1based - 1)
    zoom = cfg.dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pix.save(out_path)


def render_page_with_highlights(
    pdf_path: str,
    page_1based: int,
    highlight_phrases: List[str],
    out_path: str,
    cfg: RenderConfig,
) -> Dict:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_1based - 1)

    # Render page to image
    zoom = cfg.dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("RGBA")

    # Find rects in PDF coordinate space -> map to pixel space
    hits: Dict[str, int] = {}
    all_rects_px: List[Tuple[float, float, float, float]] = []

    for phrase in highlight_phrases:
        rects = _search_rects(page, phrase)
        hits[phrase] = len(rects)

        for r in rects:
            # PDF coordinates are in points; after rendering with zoom, multiply by zoom
            x0, y0, x1, y1 = r.x0 * zoom, r.y0 * zoom, r.x1 * zoom, r.y1 * zoom
            all_rects_px.append((x0, y0, x1, y1))

    # Draw highlights
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for (x0, y0, x1, y1) in all_rects_px:
        draw.rectangle(
            [x0, y0, x1, y1],
            fill=cfg.highlight_fill_rgba,
            outline=cfg.highlight_outline_rgba,
            width=2,
        )

    out = Image.alpha_composite(img, overlay)

    # Optional: crop around highlight region so each scene feels different
    if all_rects_px:
        minx = min(r[0] for r in all_rects_px) - cfg.pad_px
        miny = min(r[1] for r in all_rects_px) - cfg.pad_px
        maxx = max(r[2] for r in all_rects_px) + cfg.pad_px
        maxy = max(r[3] for r in all_rects_px) + cfg.pad_px

        minx = max(0, int(minx))
        miny = max(0, int(miny))
        maxx = min(out.size[0], int(maxx))
        maxy = min(out.size[1], int(maxy))

        # if crop is too tiny, keep full page
        if (maxx - minx) > 250 and (maxy - miny) > 250:
            out = out.crop((minx, miny, maxx, maxy))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.convert("RGB").save(out_path)

    return {
        "page_1based": page_1based,
        "dpi": cfg.dpi,
        "highlights": hits,
        "total_rects": len(all_rects_px),
    }
