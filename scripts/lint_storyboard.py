from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


# -----------------------------------------------------------------------------
# Patterns: forbid garbage in narration + quotes
# -----------------------------------------------------------------------------
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PAGE_MARK_RE = re.compile(r"\[PAGE\b", re.IGNORECASE)
AUTHOR_BLOCK_RE = re.compile(r"\bUC\s+Berkeley\b", re.IGNORECASE)

# Conservative: block known author/title junk that often leaks from abstract headers
AUTHOR_JUNK_RE = re.compile(
    r"(jonathan\s+ho|ajay\s+jain|pieter\s+abbeel|denoising\s+diffusion\s+probabilistic\s+models)",
    re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def load_storyboard(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing storyboard: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _scene_id(scene: Dict[str, Any]) -> str:
    return str(scene.get("scene_id") or "UNKNOWN")


def get_scene_text(scene: Dict[str, Any]) -> str:
    return str(scene.get("narration") or "").strip()


def get_scene_citations(scene: Dict[str, Any]) -> List[str]:
    """
    Support both:
      - new: source_chunk_ids
      - legacy: citations
    Returns de-duped list preserving order.
    """
    out: List[str] = []

    c1 = scene.get("source_chunk_ids")
    if isinstance(c1, list):
        out.extend([str(x).strip() for x in c1 if str(x).strip()])

    c2 = scene.get("citations")
    if isinstance(c2, list):
        out.extend([str(x).strip() for x in c2 if str(x).strip()])

    # de-dupe preserve order
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def contains_forbidden(s: str) -> bool:
    if not s:
        return False
    return bool(
        EMAIL_RE.search(s)
        or PAGE_MARK_RE.search(s)
        or AUTHOR_BLOCK_RE.search(s)
        or AUTHOR_JUNK_RE.search(s)
    )




def lint_visual_plan(scene: Dict[str, Any], sid: str) -> List[str]:
    """
    Step 3.4: enforce machine-usable visual_plan.
    Expected:
      visual_plan = {
        "type": "pdf_highlight",
        "page": int,
        "highlight_phrases": [str, ...]  # non-empty
      }
    """
    errors: List[str] = []
    vp = scene.get("visual_plan")

    if not isinstance(vp, dict):
        return [f"{sid}: visual_plan missing or not a dict."]

    vtype = vp.get("type")
    if vtype != "pdf_highlight":
        errors.append(f"{sid}: visual_plan.type must be 'pdf_highlight' (got {vtype!r}).")

    page = vp.get("page")
    if not isinstance(page, int) or page < 1:
        errors.append(f"{sid}: visual_plan.page must be an int >= 1 (got {page!r}).")

    hp = vp.get("highlight_phrases")
    if not isinstance(hp, list) or len(hp) == 0:
        errors.append(f"{sid}: visual_plan.highlight_phrases must be a non-empty list.")
    else:
        for i, phrase in enumerate(hp):
            if not isinstance(phrase, str) or not phrase.strip():
                errors.append(f"{sid}: highlight_phrases[{i}] must be a non-empty string.")
                continue
            if contains_forbidden(phrase):
                errors.append(f"{sid}: highlight_phrases[{i}] contains forbidden junk.")

    return errors


def lint_source_quotes(scene: Dict[str, Any], sid: str) -> List[str]:
    """
    Step 3.3: optional source_quotes (max 1 short quote <= 20 words).
    """
    errors: List[str] = []
    sq = scene.get("source_quotes")

    if sq is None:
        return errors

    if not isinstance(sq, list):
        return [f"{sid}: source_quotes must be a list if present."]

    if len(sq) > 1:
        errors.append(f"{sid}: source_quotes must contain at most 1 quote (got {len(sq)}).")

    for q in sq:
        if not isinstance(q, str):
            errors.append(f"{sid}: source_quotes contains non-string.")
            continue
        qq = q.strip()
        words = [w for w in qq.split() if w]
        if len(words) > 20:
            errors.append(f"{sid}: source_quote exceeds 20 words.")
        if contains_forbidden(qq):
            errors.append(f"{sid}: source_quote contains forbidden junk (email/page/author/title).")

    return errors


def lint_scene(scene: Dict[str, Any], *, min_chars: int, max_chars: int) -> List[str]:
    errors: List[str] = []
    sid = _scene_id(scene)

    # required keys (basic schema)
    if not isinstance(scene.get("title"), str) or not str(scene.get("title")).strip():
        errors.append(f"{sid}: missing/empty title.")
    if not isinstance(scene.get("duration_sec"), int) or int(scene.get("duration_sec")) <= 0:
        errors.append(f"{sid}: duration_sec must be a positive int.")

    # narration checks
    narration = get_scene_text(scene)
    if contains_forbidden(narration):
        if EMAIL_RE.search(narration):
            errors.append(f"{sid}: narration contains email (@).")
        if PAGE_MARK_RE.search(narration):
            errors.append(f"{sid}: narration contains [PAGE marker.")
        if AUTHOR_BLOCK_RE.search(narration):
            errors.append(f"{sid}: narration contains 'UC Berkeley' author block stuff.")
        if AUTHOR_JUNK_RE.search(narration):
            errors.append(f"{sid}: narration contains author/title junk keywords.")

    n = len(narration)
    if n < min_chars:
        errors.append(f"{sid}: narration too short ({n} < {min_chars}).")
    if n > max_chars:
        errors.append(f"{sid}: narration too long ({n} > {max_chars}).")

    # citations must exist (new or legacy)
    citations = get_scene_citations(scene)
    if len(citations) == 0:
        errors.append(f"{sid}: zero citations/source_chunk_ids.")

    # optional checks
    errors.extend(lint_source_quotes(scene, sid))
    errors.extend(lint_visual_plan(scene, sid))

    return errors


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3.2: Lint storyboard.json (quality gate).")
    parser.add_argument("--pdf_name", required=True)
    parser.add_argument("--extracted_dir", default="data/extracted")
    parser.add_argument("--min_chars", type=int, default=250)
    parser.add_argument("--max_chars", type=int, default=450)
    args = parser.parse_args()

    base = Path(args.extracted_dir) / args.pdf_name
    sb_path = base / "storyboard.json"

    sb = load_storyboard(sb_path)

    scenes = sb.get("scenes")
    if not isinstance(scenes, list) or len(scenes) == 0:
        print("\n=== STORYBOARD LINT: FAIL ===")
        print(f"Storyboard: {sb_path}")
        print("Errors: 1\n")
        print("- storyboard has no scenes.")
        print("\n============================\n")
        sys.exit(1)

    all_errors: List[str] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            all_errors.append("Scene entry is not a dict.")
            continue
        all_errors.extend(lint_scene(scene, min_chars=int(args.min_chars), max_chars=int(args.max_chars)))

    if all_errors:
        print("\n=== STORYBOARD LINT: FAIL ===")
        print(f"Storyboard: {sb_path}")
        print(f"Errors: {len(all_errors)}\n")
        for e in all_errors:
            print(f"- {e}")
        print("\n============================\n")
        sys.exit(1)

    print("\n=== STORYBOARD LINT: PASS ===")
    print(f"Storyboard: {sb_path}")
    print(f"Scenes: {len(scenes)}")
    print(f"Char range enforced: {args.min_chars}–{args.max_chars}")
    print("Visual plan enforced: type=pdf_highlight, page=int>=1, highlight_phrases=non-empty list")
    print("Source quote enforced: <= 1 quote, <= 20 words")
    print("=============================\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
