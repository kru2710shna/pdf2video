# scripts/render_frames.py
from __future__ import annotations

import sys
from pathlib import Path

# ---- FIX: add repo root to PYTHONPATH (so `import src...` works) ----
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# -------------------------------------------------------------------

import argparse
import json
from typing import Any, Dict, List

from src.render.pdf_render import (
    RenderConfig,
    render_page_png,
    render_page_with_highlights,
)


def load_storyboard(storyboard_path: Path) -> Dict[str, Any]:
    with storyboard_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_name", required=True, help="PDF name key (e.g., YOUR)")
    ap.add_argument("--dpi", type=int, default=180, help="Render DPI (150–200 recommended)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing renders")
    args = ap.parse_args()

    extracted_dir = REPO_ROOT / "data" / "extracted" / args.pdf_name
    pdf_path = REPO_ROOT / "data" / "input_pdfs" / f"{args.pdf_name}.pdf"
    storyboard_path = extracted_dir / "storyboard.json"

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not storyboard_path.exists():
        raise FileNotFoundError(f"Storyboard not found: {storyboard_path}")

    sb = load_storyboard(storyboard_path)
    scenes: List[Dict[str, Any]] = sb.get("scenes", [])
    if not scenes:
        raise ValueError(f"No scenes found in storyboard: {storyboard_path}")

    pages_img_dir = extracted_dir / "pages_img"
    renders_dir = extracted_dir / "renders"
    pages_img_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)

    cfg = RenderConfig(dpi=args.dpi)

    # 1) Render unique pages referenced by storyboard
    unique_pages = sorted({int(s["visual_plan"]["page"]) for s in scenes})
    for p in unique_pages:
        out_page = pages_img_dir / f"page_{p:03d}.png"
        if out_page.exists() and not args.overwrite:
            continue

        render_page_png(
            pdf_path=pdf_path,
            page_1based=p,
            out_path=out_page,
            cfg=cfg,
        )

    # 2) Render per-scene highlighted frame
    reports: List[Dict[str, Any]] = []
    for s in scenes:
        scene_id = s["scene_id"]
        vp = s["visual_plan"]
        page = int(vp["page"])
        phrases = list(vp.get("highlight_phrases", []))

        out_scene = renders_dir / f"{scene_id}.png"
        if out_scene.exists() and not args.overwrite:
            continue

        report_item = render_page_with_highlights(
            pdf_path=pdf_path,
            page_1based=page,
            highlight_phrases=phrases,
            out_path=out_scene,
            cfg=cfg,
        )

        # Ensure report stores 1-based page + scene id
        report_item["scene_id"] = scene_id
        report_item["page"] = page
        reports.append(report_item)

    # Save report for debugging
    report_path = renders_dir / "render_report.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")

    print("\n=== STEP 4 RENDER REPORT ===")
    print(f"PDF: {args.pdf_name}")
    print(f"Pages rendered: {len(unique_pages)} -> {pages_img_dir}")
    print(f"Scene frames: {len(scenes)} -> {renders_dir}")
    print(f"Report: {report_path}")
    print("============================\n")


if __name__ == "__main__":
    main()
