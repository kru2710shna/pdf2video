# scripts/retarget_storyboard_by_pages.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def norm(s: str) -> str:
    # normalize for robust matching
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)  # drop punctuation
    return s.strip()


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, obj: Dict[str, Any]) -> None:
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def find_best_page(pages: List[Dict[str, Any]], phrases: List[str]) -> Tuple[int, Dict[str, int]]:
    """
    Return best page_1based and counts per phrase.
    We score pages by how many phrases match (substring match on normalized text).
    """
    phrase_norm = [norm(x) for x in phrases if x and x.strip()]
    counts_best: Dict[str, int] = {}
    best_page = 1
    best_score = -1

    for page_obj in pages:
        pnum = int(page_obj.get("page", page_obj.get("page_num", 0)) or 0)
        if pnum <= 0:
            continue
        page_txt = norm(page_obj.get("text", ""))

        counts: Dict[str, int] = {}
        score = 0
        for raw, ph in zip(phrases, phrase_norm):
            if not ph:
                counts[raw] = 0
                continue
            c = page_txt.count(ph)
            counts[raw] = c
            if c > 0:
                score += 1  # 1 point per phrase found (simple + robust)

        if score > best_score:
            best_score = score
            best_page = pnum
            counts_best = counts

    return best_page, counts_best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_name", required=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    extracted_dir = repo_root / "data" / "extracted" / args.pdf_name

    pages_path = extracted_dir / "pages.json"
    storyboard_path = extracted_dir / "storyboard.json"

    if not pages_path.exists():
        raise FileNotFoundError(f"Missing: {pages_path}")
    if not storyboard_path.exists():
        raise FileNotFoundError(f"Missing: {storyboard_path}")

    pages_json = load_json(pages_path)
    pages = pages_json["pages"] if isinstance(pages_json, dict) else pages_json


    sb = load_json(storyboard_path)
    scenes = sb.get("scenes", [])

    changed = 0
    debug = []

    for s in scenes:
        vp = s.get("visual_plan", {})
        phrases = list(vp.get("highlight_phrases", []))
        if not phrases:
            continue

        current_page = int(vp.get("page", 1))
        best_page, counts = find_best_page(pages, phrases)

        debug.append({
            "scene_id": s.get("scene_id"),
            "from_page": current_page,
            "to_page": best_page,
            "counts": counts,
        })

        if best_page != current_page:
            vp["page"] = best_page
            s["visual_plan"] = vp
            changed += 1

    sb["scenes"] = scenes
    save_json(storyboard_path, sb)

    (extracted_dir / "retarget_debug.json").write_text(
        json.dumps(debug, indent=2),
        encoding="utf-8"
    )

    print(f"Retargeted pages for {changed} scenes.")
    print(f"Wrote: {storyboard_path}")
    print(f"Debug: {extracted_dir / 'retarget_debug.json'}")


if __name__ == "__main__":
    main()
