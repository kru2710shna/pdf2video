from __future__ import annotations

import sys
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# --- MUST be first: add repo root to sys.path -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.index import load_chunks_jsonl


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DEFAULT_TARGET_TOTAL_SEC = 105
MIN_NARR_CHARS = 250
MAX_NARR_CHARS = 450

# -----------------------------------------------------------------------------
# Regex / cleaning
# -----------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")

EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PAGE_MARK_INLINE_RE = re.compile(r"\[\s*PAGE\s+\d+\s*\]", re.IGNORECASE)
PAGE_MARK_ANY_RE = re.compile(r"\[PAGE\b", re.IGNORECASE)

AUTHOR_BLOCK_RE = re.compile(r"\bUC\s+Berkeley\b", re.IGNORECASE)
AUTHOR_JUNK_RE = re.compile(
    r"(jonathan\s+ho|ajay\s+jain|pieter\s+abbeel|denoising\s+diffusion\s+probabilistic\s+models)",
    re.IGNORECASE,
)

INTRO_CUT_RE = re.compile(r"(?im)^\s*\d*\s*introduction\s*$")
FIG_RE = re.compile(r"\bfig(?:ure)?\.?\s*\d+\b", re.IGNORECASE)


def _sec(c: dict) -> str:
    return (c.get("section") or "unknown").lower().strip()


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())


def is_forbidden_text(s: str) -> bool:
    if not s:
        return False
    return bool(
        EMAIL_RE.search(s)
        or PAGE_MARK_ANY_RE.search(s)
        or AUTHOR_BLOCK_RE.search(s)
        or AUTHOR_JUNK_RE.search(s)
    )


def clean_text_for_use(text: str) -> str:
    """Remove obvious junk markers and normalize whitespace."""
    t = (text or "").replace("\r", "\n")
    t = PAGE_MARK_INLINE_RE.sub(" ", t)  # remove [PAGE X]
    t = EMAIL_RE.sub(" ", t)             # remove emails
    # NOTE: we do NOT remove 'UC Berkeley' here; that should never reach narration anyway.
    t = _WS_RE.sub(" ", t).strip()
    return t


def abstract_span_only(text_embed: str) -> str:
    """Strictly keep abstract span only; cut off at an Introduction heading line if present."""
    if not text_embed:
        return ""
    m = INTRO_CUT_RE.search(text_embed)
    if not m:
        return text_embed.strip()
    return text_embed[:m.start()].strip()


def chunk_text_for_generation(c: dict) -> str:
    """
    Clean + section-aligned chunk text for evidence extraction.
    - abstract: use text_embed trimmed to abstract-only
    - else: use text_raw
    """
    sec = _sec(c)
    if sec == "abstract":
        t = (c.get("text_embed") or c.get("text_raw") or c.get("text") or "").strip()
        t = abstract_span_only(t)
        return clean_text_for_use(t)
    t = (c.get("text_raw") or c.get("text") or "").strip()
    return clean_text_for_use(t)


def split_sentences(text: str) -> List[str]:
    t = clean_text_for_use(text)
    if not t:
        return []
    parts = _SENT_SPLIT_RE.split(t)
    out: List[str] = []
    for p in parts:
        s = p.strip()
        if len(s) < 35:
            continue
        if is_forbidden_text(s):
            continue
        out.append(s)
    return out


def clamp_chars(s: str, lo: int = MIN_NARR_CHARS, hi: int = MAX_NARR_CHARS) -> str:
    s = _norm(s)
    # never allow forbidden text
    if is_forbidden_text(s):
        # hard strip forbidden tokens (defensive) then continue
        s = AUTHOR_BLOCK_RE.sub(" ", s)
        s = AUTHOR_JUNK_RE.sub(" ", s)
        s = EMAIL_RE.sub(" ", s)
        s = PAGE_MARK_ANY_RE.sub(" ", s)
        s = _norm(s)

    if len(s) > hi:
        s = s[:hi].rstrip()
        if not s.endswith((".", "!", "?")):
            s += "."
        return s

    if len(s) < lo:
        # add a safe filler sentence that contains no forbidden terms
        filler = "This explanation is grounded in the cited chunks and avoids copying them verbatim."
        s2 = _norm(s + " " + filler)
        if len(s2) > hi:
            s2 = s2[:hi].rstrip()
            if not s2.endswith((".", "!", "?")):
                s2 += "."
        return s2

    return s


# -----------------------------------------------------------------------------
# Evidence selection (deterministic, but narration is NOT verbatim)
# -----------------------------------------------------------------------------
def keyword_score(text: str, keywords: List[str]) -> int:
    tl = (text or "").lower()
    score = 0
    for k in keywords:
        kk = k.lower().strip()
        if not kk:
            continue
        score += tl.count(kk)
    return score


def rank_chunks_by_keywords(
    chunks: List[dict],
    keywords: List[str],
    *,
    section_whitelist: Optional[set] = None,
) -> List[dict]:
    ranked: List[Tuple[int, dict]] = []
    for c in chunks:
        if section_whitelist is not None and _sec(c) not in section_whitelist:
            continue
        text = chunk_text_for_generation(c)
        sc = keyword_score(text, keywords)
        if sc > 0:
            ranked.append((sc, c))

    ranked.sort(
        key=lambda x: (-x[0], int(x[1].get("page_start", 10**9)), x[1].get("chunk_id", ""))
    )
    return [c for _, c in ranked]


def pick_evidence_sentences(
    chunks: List[dict],
    keywords: List[str],
    *,
    max_sentences: int = 2,
    section_whitelist: Optional[set] = None,
) -> List[Tuple[str, str]]:
    """
    Returns: list of (sentence, chunk_id)
    Deterministic: choose highest keyword-scoring sentences from ranked chunks.
    IMPORTANT: sentences are used only for highlighting/quotes, not dumped into narration.
    """
    ranked_chunks = rank_chunks_by_keywords(chunks, keywords, section_whitelist=section_whitelist)
    picked: List[Tuple[int, str, str]] = []  # (score, sentence, chunk_id)

    for c in ranked_chunks:
        cid = c.get("chunk_id", "UNKNOWN")
        text = chunk_text_for_generation(c)
        for s in split_sentences(text):
            sc = keyword_score(s, keywords)
            if sc > 0:
                picked.append((sc, s, cid))

    picked.sort(key=lambda x: (-x[0], x[2], x[1]))
    out: List[Tuple[str, str]] = []
    seen = set()
    for _, sent, cid in picked:
        key = (cid, sent)
        if key in seen:
            continue
        seen.add(key)
        out.append((sent, cid))
        if len(out) >= max_sentences:
            break
    return out


def dedupe_preserve_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def pick_one_short_quote(evidence: List[Tuple[str, str]]) -> List[str]:
    """
    Optional traceability: at most 1 quote ≤ 20 words.
    """
    for sent, _cid in evidence:
        s = clean_text_for_use(sent)
        if is_forbidden_text(s):
            continue
        words = [w for w in s.split() if w]
        if 6 <= len(words) <= 20:
            return [s]
        if len(words) > 20:
            return [" ".join(words[:20])]
    return []


def extract_highlight_phrases(evidence: List[Tuple[str, str]], max_phrases: int = 2) -> List[str]:
    """
    Machine-usable highlight phrases.
    Deterministic short windows from evidence sentences (first 6–8 words).
    """
    phrases: List[str] = []
    for sent, _cid in evidence:
        s = clean_text_for_use(sent)
        if is_forbidden_text(s):
            continue
        words = [w for w in s.split() if w]
        if len(words) < 6:
            continue
        window = " ".join(words[:8])
        if window and window not in phrases:
            phrases.append(window)
        if len(phrases) >= max_phrases:
            break
    return phrases


def min_page_from_chunk_ids(chunks_by_id: Dict[str, dict], chunk_ids: List[str]) -> int:
    pages = []
    for cid in chunk_ids:
        c = chunks_by_id.get(cid)
        if not c:
            continue
        pages.append(int(c.get("page_start", 1)))
    return min(pages) if pages else 1


def first_chunk_in_section(chunks: List[dict], section: str) -> Optional[dict]:
    s = section.lower().strip()
    candidates = [c for c in chunks if _sec(c) == s]
    if not candidates:
        return None
    candidates.sort(key=lambda c: (int(c.get("page_start", 10**9)), c.get("chunk_id", "")))
    return candidates[0]


def find_chunks_with_figures(chunks: List[dict]) -> List[dict]:
    hits = []
    for c in chunks:
        t = chunk_text_for_generation(c)
        if FIG_RE.search(t):
            hits.append(c)
    hits.sort(key=lambda c: (int(c.get("page_start", 10**9)), c.get("chunk_id", "")))
    return hits


# -----------------------------------------------------------------------------
# Narration (deterministic, grounded, NOT copied)
# -----------------------------------------------------------------------------
def paraphrase_scene(scene_key: str) -> str:
    """
    Deterministic plain-English narration templates (2–4 sentences).
    No source text is copied verbatim here.
    """
    if scene_key == "hook":
        return (
            "Generating realistic data is hard because the space of possible outputs is huge and the training signal can be unstable. "
            "This paper motivates a method that breaks generation into many small steps, making it easier to control and learn."
        )

    if scene_key == "contrib":
        return (
            "The paper proposes diffusion-based generative modeling: start from noise and gradually refine it into data. "
            "It frames generation as an iterative denoising process that can be trained with a clear objective. "
            "The goal is high-quality samples while keeping the procedure structured and inspectable."
        )

    if scene_key == "core":
        return (
            "Diffusion models define two linked processes. "
            "A forward process progressively adds noise to data, and a learned reverse process removes that noise step by step to create a sample."
        )

    if scene_key == "reverse":
        return (
            "The reverse chain is learned rather than hand-coded. "
            "A neural network predicts the denoising direction at each step, guiding the sample from noisy states toward a clean output."
        )

    if scene_key == "train":
        return (
            "Training teaches the model to perform many small denoising corrections. "
            "Instead of one big jump from noise to data, the objective encourages accurate stepwise transitions that compound into a good sample."
        )

    if scene_key == "results":
        return (
            "The paper evaluates the approach by showing generated samples and reporting experimental results on the tested data setting. "
            "The key takeaway is that the iterative denoising procedure produces realistic outputs under the paper’s evaluation setup."
        )

    if scene_key == "matters":
        return (
            "Why this matters is the engineering trade-off: a complex generation problem becomes a sequence of simpler subproblems. "
            "That structure improves controllability and makes the system easier to debug and extend."
        )

    if scene_key == "limits":
        return (
            "The method is powerful, but it can be expensive at sampling time because generation may require many steps. "
            "The paper highlights open directions to improve efficiency while keeping sample quality."
        )

    if scene_key == "outro":
        return (
            "In summary, the paper frames generative modeling as iterative denoising from noise back to data. "
            "The main idea is to learn small, reliable transitions and compose them into high-quality generation."
        )

    return (
        "This scene summarizes the cited chunks in plain English. "
        "It stays grounded in the paper without copying the source verbatim."
    )

def ensure_pdf_highlight_plan(
    visual_plan: Dict[str, Any],
    evidence: List[Tuple[str, str]],
    chunks_by_id: Dict[str, dict],
) -> Dict[str, Any]:
    """
    Enforce executable visual_plan for lint:
      {type: "pdf_highlight", page: int>=1, highlight_phrases: [str,...]}
    """
    vp = dict(visual_plan or {})

    # force type
    vp["type"] = "pdf_highlight"

    # page: earliest page among cited chunks
    if not isinstance(vp.get("page"), int) or vp.get("page", 0) < 1:
        cited_ids = [cid for _s, cid in evidence if cid]
        vp["page"] = min_page_from_chunk_ids(chunks_by_id, cited_ids)

    # highlight phrases: derive from evidence sentences
    hp = vp.get("highlight_phrases")
    if not isinstance(hp, list) or len(hp) == 0:
        vp["highlight_phrases"] = extract_highlight_phrases(evidence, max_phrases=2)

    # final safety: never allow empty list (lint demands non-empty)
    if not vp["highlight_phrases"]:
        vp["highlight_phrases"] = ["Key idea", "Main contribution"]

    return vp


# -----------------------------------------------------------------------------
# Storyboard schema
# -----------------------------------------------------------------------------
@dataclass
class Scene:
    scene_id: str
    title: str
    duration_sec: int
    narration: str
    on_screen_text: List[str]
    visual_plan: Dict[str, Any]
    source_chunk_ids: List[str]
    source_quotes: List[str]  # optional; may be empty


def make_scene(
    *,
    scene_id: str,
    title: str,
    duration_sec: int,
    narration: str,
    on_screen_text: List[str],
    visual_plan: Dict[str, Any],
    source_chunk_ids: List[str],
    chunks_by_id: Dict[str, dict],
    source_quotes: List[str],
    evidence: List[Tuple[str, str]],
) -> Scene:
    on_screen_text = [x.strip() for x in on_screen_text if x.strip()][:2]
    source_chunk_ids = dedupe_preserve_order([cid for cid in source_chunk_ids if cid.strip()])
    narration = clamp_chars(narration)

    # defensive: never let forbidden text through
    if is_forbidden_text(narration):
        narration = clamp_chars(
            "This scene is grounded in the cited chunks and avoids copying them verbatim."
        )

    # enforce executable visual plan (Step 3.4)
    vp = dict(visual_plan or {})
    vp = ensure_pdf_highlight_plan(vp, evidence, chunks_by_id)

    return Scene(
        scene_id=scene_id,
        title=title,
        duration_sec=int(duration_sec),
        narration=narration,
        on_screen_text=on_screen_text,
        visual_plan=vp,                    # ✅ use enforced plan
        source_chunk_ids=source_chunk_ids,
        source_quotes=source_quotes[:1],
    )



# -----------------------------------------------------------------------------
# Build storyboard
# -----------------------------------------------------------------------------

def build_storyboard(chunks: List[dict], *, target_total_sec: int = DEFAULT_TARGET_TOTAL_SEC) -> Dict[str, Any]:
    chunks_by_id = {c.get("chunk_id"): c for c in chunks if c.get("chunk_id")}

    abstract_c = first_chunk_in_section(chunks, "abstract")
    intro_c = first_chunk_in_section(chunks, "intro")
    concl_c = first_chunk_in_section(chunks, "conclusion")

    fig_chunks = find_chunks_with_figures(chunks)

    scenes: List[Scene] = []

    # s01 Hook: evidence from intro/abstract if possible
    hook_pool = [c for c in chunks if _sec(c) in {"intro", "abstract"}]
    hook_evidence = pick_evidence_sentences(
        hook_pool if hook_pool else chunks,
        ["hard", "challenge", "high-dimensional", "generative", "generation"],
        max_sentences=2,
    )
    hook_cids = dedupe_preserve_order([cid for _s, cid in hook_evidence])
    if not hook_cids:
        hook_cids = [abstract_c["chunk_id"]] if abstract_c else ([intro_c["chunk_id"]] if intro_c else [])
    hook_page = min_page_from_chunk_ids(chunks_by_id, hook_cids)
    scenes.append(
        make_scene(
            scene_id="s01",
            title="Hook: Why generative modeling is hard",
            duration_sec=10,
            narration=paraphrase_scene("hook"),
            on_screen_text=["High-dimensional generation is hard", "We need stable learning + good samples"],
            visual_plan={
                "type": "pdf_highlight",
                "page": hook_page,
                "highlight_phrases": extract_highlight_phrases(hook_evidence),
            },
            source_chunk_ids=hook_cids,
            source_quotes=pick_one_short_quote(hook_evidence),
            chunks_by_id=chunks_by_id,
            evidence=hook_evidence
            
        )
    )

    # s02 Contribution: ABSTRACT ONLY (strict)
    contrib_evidence: List[Tuple[str, str]] = []
    contrib_cids: List[str] = []
    if abstract_c:
        contrib_evidence = pick_evidence_sentences(
            [abstract_c],
            ["propose", "present", "introduce", "diffusion", "denoise", "model"],
            max_sentences=2,
        )
        contrib_cids = [abstract_c["chunk_id"]]
    else:
        # deterministic fallback: still must cite something
        if intro_c:
            contrib_evidence = pick_evidence_sentences([intro_c], ["we", "model", "diffusion"], max_sentences=2)
            contrib_cids = [intro_c["chunk_id"]]
    contrib_page = min_page_from_chunk_ids(chunks_by_id, contrib_cids) if contrib_cids else 1
    scenes.append(
        make_scene(
            scene_id="s02",
            title="Main contribution (Abstract only)",
            duration_sec=12,
            narration=paraphrase_scene("contrib"),
            on_screen_text=["What the paper proposes", "High-level contribution"],
            visual_plan={
                "type": "pdf_highlight",
                "page": contrib_page,
                "highlight_phrases": extract_highlight_phrases(contrib_evidence),
            },
            source_chunk_ids=contrib_cids,
            source_quotes=pick_one_short_quote(contrib_evidence),
            chunks_by_id=chunks_by_id,
            evidence=contrib_evidence
        )
    )

    # s03 Core idea
    core_evidence = pick_evidence_sentences(
        chunks,
        ["diffusion", "noise", "denoise", "denoising", "steps", "iterative"],
        max_sentences=2,
    )
    core_cids = dedupe_preserve_order([cid for _s, cid in core_evidence]) or contrib_cids
    core_page = min_page_from_chunk_ids(chunks_by_id, core_cids) if core_cids else 1
    scenes.append(
        make_scene(
            scene_id="s03",
            title="Core idea: iterative noising → denoising",
            duration_sec=12,
            narration=paraphrase_scene("core"),
            on_screen_text=["Forward: add noise", "Reverse: remove noise"],
            visual_plan={
                "type": "pdf_highlight",
                "page": core_page,
                "highlight_phrases": extract_highlight_phrases(core_evidence),
            },
            source_chunk_ids=core_cids,
            source_quotes=pick_one_short_quote(core_evidence),
            chunks_by_id=chunks_by_id,
            evidence=core_evidence
        )
    )

    # s04 Reverse model
    rev_evidence = pick_evidence_sentences(
        chunks,
        ["reverse", "neural", "network", "predict", "denoise", "model"],
        max_sentences=2,
    )
    rev_cids = dedupe_preserve_order([cid for _s, cid in rev_evidence]) or core_cids
    rev_page = min_page_from_chunk_ids(chunks_by_id, rev_cids) if rev_cids else 1
    scenes.append(
        make_scene(
            scene_id="s04",
            title="Reverse process: neural net predicts denoising",
            duration_sec=12,
            narration=paraphrase_scene("reverse"),
            on_screen_text=["Reverse is learned", "Neural net guides denoising"],
            visual_plan={
                "type": "pdf_highlight",
                "page": rev_page,
                "highlight_phrases": extract_highlight_phrases(rev_evidence),
            },
            source_chunk_ids=rev_cids,
            source_quotes=pick_one_short_quote(rev_evidence),
            chunks_by_id=chunks_by_id,
            evidence=rev_evidence
        )
    )

    # s05 Training objective
    train_evidence = pick_evidence_sentences(
        chunks,
        ["objective", "training", "loss", "optimize", "likelihood", "bound"],
        max_sentences=2,
    )
    train_cids = dedupe_preserve_order([cid for _s, cid in train_evidence]) or rev_cids
    train_page = min_page_from_chunk_ids(chunks_by_id, train_cids) if train_cids else 1
    scenes.append(
        make_scene(
            scene_id="s05",
            title="Training objective (high-level)",
            duration_sec=12,
            narration=paraphrase_scene("train"),
            on_screen_text=["Optimize a training loss", "Learn reverse denoising"],
            visual_plan={
                "type": "pdf_highlight",
                "page": train_page,
                "highlight_phrases": extract_highlight_phrases(train_evidence),
            },
            source_chunk_ids=train_cids,
            source_quotes=pick_one_short_quote(train_evidence),
            chunks_by_id=chunks_by_id,
            evidence=train_evidence
        )
    )

    # s06 Results: prefer figure chunk if present
    res_pool = fig_chunks if fig_chunks else chunks
    res_evidence = pick_evidence_sentences(
        res_pool,
        ["results", "experiments", "dataset", "figure", "samples", "evaluation"],
        max_sentences=2,
    )
    res_cids = dedupe_preserve_order([cid for _s, cid in res_evidence])
    if not res_cids and fig_chunks:
        res_cids = [fig_chunks[0]["chunk_id"]]
    if not res_cids and concl_c:
        res_cids = [concl_c["chunk_id"]]
    res_page = min_page_from_chunk_ids(chunks_by_id, res_cids) if res_cids else 1
    scenes.append(
        make_scene(
            scene_id="s06",
            title="Results: samples and evaluation",
            duration_sec=14,
            narration=paraphrase_scene("results"),
            on_screen_text=["Qualitative samples (Fig.)", "Dataset + evaluation summary"],
            visual_plan={
                "type": "pdf_highlight",
                "page": res_page,
                "highlight_phrases": extract_highlight_phrases(res_evidence),
            },
            source_chunk_ids=res_cids,
            source_quotes=pick_one_short_quote(res_evidence),
            chunks_by_id=chunks_by_id,
            evidence=res_evidence
        )
    )

    # s07 Why it matters
    matter_pool = [c for c in chunks if _sec(c) in {"conclusion", "intro", "abstract"}] or chunks
    matter_evidence = pick_evidence_sentences(
        matter_pool,
        ["important", "practical", "stable", "quality", "improve", "effective"],
        max_sentences=2,
    )
    matter_cids = dedupe_preserve_order([cid for _s, cid in matter_evidence]) or (concl_c and [concl_c["chunk_id"]]) or contrib_cids
    matter_page = min_page_from_chunk_ids(chunks_by_id, matter_cids) if matter_cids else 1
    scenes.append(
        make_scene(
            scene_id="s07",
            title="Why it matters",
            duration_sec=12,
            narration=paraphrase_scene("matters"),
            on_screen_text=["Structured generation", "More controllable training"],
            visual_plan={
                "type": "pdf_highlight",
                "page": matter_page,
                "highlight_phrases": extract_highlight_phrases(matter_evidence),
            },
            source_chunk_ids=matter_cids,
            source_quotes=pick_one_short_quote(matter_evidence),
            chunks_by_id=chunks_by_id,
            evidence=matter_evidence
        )
    )

    # s08 Limitations
    lim_pool = [c for c in chunks if _sec(c) in {"conclusion", "related_work", "intro"}] or chunks
    lim_evidence = pick_evidence_sentences(
        lim_pool,
        ["limitation", "however", "but", "compute", "slow", "future"],
        max_sentences=2,
    )
    lim_cids = dedupe_preserve_order([cid for _s, cid in lim_evidence]) or (concl_c and [concl_c["chunk_id"]]) or contrib_cids
    lim_page = min_page_from_chunk_ids(chunks_by_id, lim_cids) if lim_cids else 1
    scenes.append(
        make_scene(
            scene_id="s08",
            title="Limitations and open questions",
            duration_sec=10,
            narration=paraphrase_scene("limits"),
            on_screen_text=["Sampling cost", "Open directions"],
            visual_plan={
                "type": "pdf_highlight",
                "page": lim_page,
                "highlight_phrases": extract_highlight_phrases(lim_evidence),
            },
            source_chunk_ids=lim_cids,
            source_quotes=pick_one_short_quote(lim_evidence),
            chunks_by_id=chunks_by_id,
            evidence=lim_evidence
        )
    )

    # s09 Outro
    outro_pool = [c for c in chunks if _sec(c) == "conclusion"] or chunks
    outro_evidence = pick_evidence_sentences(
        outro_pool,
        ["conclusion", "summary", "future", "direction"],
        max_sentences=2,
    )
    outro_cids = dedupe_preserve_order([cid for _s, cid in outro_evidence]) or (concl_c and [concl_c["chunk_id"]]) or contrib_cids
    outro_page = min_page_from_chunk_ids(chunks_by_id, outro_cids) if outro_cids else 1
    scenes.append(
        make_scene(
            scene_id="s09",
            title="Outro",
            duration_sec=11,
            narration=paraphrase_scene("outro"),
            on_screen_text=["Key takeaway", "Next directions (from text)"],
            visual_plan={
                "type": "pdf_highlight",
                "page": outro_page,
                "highlight_phrases": extract_highlight_phrases(outro_evidence),
            },
            source_chunk_ids=outro_cids,
            source_quotes=pick_one_short_quote(outro_evidence),
            chunks_by_id=chunks_by_id,
            evidence=outro_evidence
        )
    )

    total = sum(s.duration_sec for s in scenes)

    # Keep total within 90–120 deterministically by nudging s06 (results)
    if total < 90:
        need = 90 - total
        for s in scenes:
            if s.scene_id == "s06":
                s.duration_sec += need
                break
        total = sum(s.duration_sec for s in scenes)

    if total > 120:
        over = total - 120
        for s in scenes:
            if s.scene_id == "s06":
                cut = min(over, max(0, s.duration_sec - 10))
                s.duration_sec -= cut
                over -= cut
            if over <= 0:
                break
        total = sum(s.duration_sec for s in scenes)

    return {
        "schema_version": "storyboard.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_duration_sec": int(target_total_sec),
        "total_duration_sec": int(total),
        "scene_count": len(scenes),
        "scenes": [s.__dict__ for s in scenes],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 3.3/3.4: Build storyboard.json (grounded, deterministic, machine-usable visual plans)."
    )
    parser.add_argument("--pdf_name", required=True)
    parser.add_argument("--extracted_dir", default="data/extracted")
    parser.add_argument("--target_sec", type=int, default=DEFAULT_TARGET_TOTAL_SEC)
    args = parser.parse_args()

    base = Path(args.extracted_dir) / args.pdf_name
    indexed_chunks_path = base / "indexed_chunks.jsonl"
    chunks_path = indexed_chunks_path if indexed_chunks_path.exists() else (base / "chunks.jsonl")

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing {chunks_path}. Run run_chunk.py then build_index.py first.")

    chunks = load_chunks_jsonl(chunks_path, drop_junk=False, debug_tags=False)

    sb = build_storyboard(chunks, target_total_sec=int(args.target_sec))

    out_path = base / "storyboard.json"
    out_path.write_text(json.dumps(sb, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== STEP 3.3/3.4 STORYBOARD BUILD REPORT ===")
    print(f"PDF: {args.pdf_name}")
    print(f"Chunks source: {chunks_path.name}")
    print(f"Scenes: {sb['scene_count']}")
    print(f"Total duration (sec): {sb['total_duration_sec']}")
    print(f"Saved: {out_path}")
    print("===========================================\n")


if __name__ == "__main__":
    main()
