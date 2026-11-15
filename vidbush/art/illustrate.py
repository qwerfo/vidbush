"""Illustrations as a declared style (run 3, Part C).

Principle: an illustration is openly a painting, never a photograph. It is made only for what no
footage or photograph can show (a process, a comparison, a sense of scale, a scene from before
cameras), in one style line per brand (brand.illustration_style) applied to every illustration of a
video so they read as one set. It carries no text and no named real person's likeness. It is judged
against the narration of its shot window like any still, and its candidates carry source
`illustration`, so reports count them as designed animation, apart from photo-style AI stills.

choose(shotlist, ctx, budget) -> picks; prompt_for(brief, style, ctx) -> prompt;
illustrate(shot, pick, ctx, out_dir, window) -> report (chosen Candidate under "candidate");
apply_to_shot(shot, cand) puts the chosen illustration on the shot.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from ..models import (STILL_TYPES, Candidate, MotionKind, Shot, ShotList, ShotType,
                      SourceKind)
from ..util import stable_hash
from . import common

CATEGORIES = ("process", "comparison", "scale", "before_cameras", "other")
MIN_SHOT_S = 2.5
SPACING_S = 20.0  # no two illustrations closer than this, unless the budget cannot be met otherwise
REAL_KEEP_SCORE = 70.0  # real footage or photographs scoring this or more are never replaced
GENERATED = {SourceKind.ai_image, SourceKind.ai_video}  # photo-style AI fill may be replaced

DEFAULT_STYLE = ("Hand-painted illustration with visible brush texture, muted natural palette, soft light, "
                 "simplified shapes; clearly a painting, never a photograph.")
RULES = ("Clearly an illustration, not a photograph. No text, letters, numbers, signs, captions or logos. "
         "No recognisable real person. One single scene, no split panels, no collage. A full-bleed 16:9 "
         "landscape picture that fills the frame edge to edge, with no borders, frame or bars; the subject sits "
         "in the middle band, nothing important near the top or bottom edge.")
EDGE_TRIM = 0.04  # crops the provider's small corner mark; an illustration has no geometry to keep

ILLUSTRATE_SYSTEM = """You pick moments of a documentary that should be shown as an illustration.
An illustration is for what no footage or photograph can show: a process (how something works or
happens), a comparison, a sense of scale, or a scene from before cameras existed. Pick only such
moments, where the narration makes the illustration meaningful.
For each pick write a brief: what to draw, in plain words. No person names, no brand names, no
readable text, never the likeness of a named real person (an anonymous figure is fine).
Never a map, chart, graph, diagram with labels, document or screen: maps and charts are drawn by the
renderer, where every word is checked.
Answer JSON: {"picks": [{"shot_id": "...", "category": "process|comparison|scale|before_cameras|other",
"brief": "..."}]} with at most the number of picks asked for, best first."""

HEURISTIC_WORDS = re.compile(
    r"\b(how|process|compared|times (?:larger|bigger|smaller)|bigger than|larger than|smaller than|"
    r"the size of|years ago|centur(?:y|ies)|thousands of years|before|once)\b", re.I)


# a brief naming a lettered artifact is the renderer's job (maps, charts) or text the model would invent
TEXT_ARTIFACT = re.compile(r"(maps?|charts?|graphs?|diagrams?|infographics?|documents?|newspapers?|headlines?|"
                           r"signs?|screens?|labels?|captions?)", re.I)


def _eligible(shot: Shot) -> bool:
    if shot.duration < MIN_SHOT_S or shot.type in (ShotType.motion_graphic, ShotType.ai_video):
        return False
    if shot.overlay is not None and shot.overlay.kind == MotionKind.map:
        return False
    sel = shot.selected_candidate()
    if sel is not None and sel.source not in GENERATED and (sel.score or 0) >= REAL_KEEP_SCORE:
        return False  # good real material is never replaced by a drawing
    return True


def _spread(picks: list[dict], by_id: dict[str, Shot], budget: int) -> list[dict]:
    """Keep picks in order, skipping ones within SPACING_S of a kept one; fill from the skipped if short."""
    kept: list[dict] = []
    skipped: list[dict] = []
    for p in picks:
        t = by_id[p["shot_id"]].start
        (skipped if any(abs(t - by_id[k["shot_id"]].start) < SPACING_S for k in kept) else kept).append(p)
    return (kept + skipped)[:budget]


def _validate(raw: Any, by_id: dict[str, Shot]) -> list[dict]:
    items = raw.get("picks") if isinstance(raw, dict) else raw
    out, seen = [], set()
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        sid, brief = str(it.get("shot_id") or ""), str(it.get("brief") or "").strip()
        if sid in seen or sid not in by_id or not brief or not _eligible(by_id[sid]) or TEXT_ARTIFACT.search(brief):
            continue
        cat = str(it.get("category") or "other")
        seen.add(sid)
        out.append({"shot_id": sid, "category": cat if cat in CATEGORIES else "other", "brief": brief})
    return out


def _drop_capitalised(text: str) -> str:
    """Remove runs of capitalised words that do not open a sentence (names, places, brands)."""
    words, out = text.split(), []
    for i, w in enumerate(words):
        start = i == 0 or words[i - 1][-1:] in ".!?"
        if w[:1].isupper() and not start:
            continue
        out.append(w)
    return " ".join(out)


def _heuristic(shots: list[Shot], narr: Any, budget: int) -> list[dict]:
    hits = [s for s in shots if s.type == ShotType.ai_image and _eligible(s)
            and HEURISTIC_WORDS.search(common.shot_text(narr, s))]
    hits.sort(key=lambda s: -s.duration)
    return [{"shot_id": s.id, "category": "other", "brief": _drop_capitalised(s.intent)} for s in hits]


def choose(shotlist: ShotList, ctx: Any, budget: int) -> list[dict]:
    """Up to budget illustration picks [{shot_id, category, brief}], from the director, else heuristics."""
    if budget <= 0:
        return []
    narr = common.narration(ctx)
    by_id = {s.id: s for s in shotlist.shots}
    rows = []
    for s in shotlist.shots:
        sel = s.selected_candidate()
        rows.append({"id": s.id, "start": round(s.start, 1), "duration": round(s.duration, 1), "type": s.type.value,
                     "selected_source": sel.source.value if sel else None, "score": sel.score if sel else None,
                     "intent": s.intent, "narration": common.shot_text(narr, s)})
    user = f"Pick at most {budget} illustrations.\nShots:\n" + json.dumps(rows, ensure_ascii=False)
    stage = ctx.stage or "visuals"
    raw = ctx.seats.llm_json("director", "art_illustrations", ILLUSTRATE_SYSTEM, user, stage)
    picks = _validate(raw, by_id) if raw is not None else []
    if not picks:
        picks = _heuristic(shotlist.shots, narr, budget)
        for p in picks[:budget]:
            ctx.seats.note_fallback("director", stage, by_id[p["shot_id"]].scene_id,
                                    "illustration picks by heuristic (director gave none usable)")
    return _spread(picks, by_id, budget)


def prompt_for(brief: str, style: str, ctx: Any = None) -> str:
    """Style line first, then the brief with names removed, then the fixed rules."""
    from ..visuals.ai_images import guess_names
    clean = common.strip_names(brief, guess_names(brief)) or brief
    return f"{(style or DEFAULT_STYLE).strip()}\nIllustration: {clean.strip()}\n{RULES}"


def _threshold(ctx: Any) -> float:
    return float(getattr(getattr(getattr(ctx, "settings", None), "judge", None), "threshold", 70.0) or 70.0)


def illustrate(shot: Shot, pick: dict, ctx: Any, out_dir: Path, window: str) -> dict:
    """Generate, check and judge variants for one pick. Report dict; the chosen Candidate under "candidate"."""
    from ..judge.judge import apply_result, score_candidate
    from ..providers.image import fit_16x9
    from .elevate import generate
    spec = common.elevate_spec(ctx)
    if spec is None:
        return {"skipped": "no Elevate image seat"}
    art = ctx.profile.art
    style = getattr(ctx.brand, "illustration_style", "") or DEFAULT_STYLE
    prompt = prompt_for(pick["brief"], style, ctx)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants: list[dict] = []
    best: Optional[Candidate] = None
    for i in range(max(1, int(art.illustration_variants))):
        raw = out_dir / f"{shot.id}_ill_{i}.png"
        row: dict[str, Any] = {"path": "", "score": None, "reason": "", "reject": None, "text": []}
        try:
            if not raw.exists():  # resume: a finished variant is never paid for twice
                seed = int(stable_hash(shot.id, i), 16) % 2147483647
                common.call_image(ctx, spec, model=art.illustration_model, fn=lambda sp, r=raw, sd=seed: generate(
                    sp, prompt, r, model=art.illustration_model, seed=sd), note="art illustration")
            fitted = fit_16x9(raw.read_bytes(), out_dir / f"{shot.id}_ill_{i}_16x9.png", trim=EDGE_TRIM)
            row["path"] = str(fitted)
            row["text"] = common.find_text(fitted)
            if row["text"] or common.has_bars(fitted):
                row["reject"] = "text in picture" if row["text"] else "bars painted at the frame edge"
                variants.append(row)
                continue
            cand = Candidate(id=f"{shot.id}_ill_{i}", source=SourceKind.illustration, local_path=str(fitted),
                             media="image", width=1920, height=1080, licence="AI generated illustration",
                             attribution="Illustration (AI generated)", title="Illustration",
                             keyframes=[str(fitted)])
            result = score_candidate(shot, cand, ctx, narration=window)
            apply_result(cand, result)
            row.update(score=cand.score, reason=cand.judge_reason, reject=cand.reject_reason)
            if not cand.reject_reason and (cand.score or 0) >= _threshold(ctx) \
                    and (best is None or (cand.score or 0) > (best.score or 0)):
                best = cand
        except Exception as e:  # noqa: BLE001  one failed variant never sinks the shot
            ctx.log.warn("illustration variant failed", shot=shot.id, variant=i, error=str(e)[:300])
            row["reject"] = f"failed: {str(e)[:200]}"
        variants.append(row)
    return {"shot_id": shot.id, "category": pick.get("category", "other"), "brief": pick["brief"], "prompt": prompt,
            "variants": variants, "chosen": best.id if best else None, "candidate": best}


def apply_to_shot(shot: Shot, cand: Candidate) -> None:
    """Select the illustration on the shot. A footage slot now holds a still, so it becomes the matching still
    type (footage -> ai_image, footage_overlay -> image_overlay keeps its overlay) and Ken Burns runs."""
    shot.candidates.append(cand)
    shot.selected = cand.id
    shot.judge_score = cand.score
    shot.judge_notes = cand.judge_reason
    if shot.type == ShotType.footage_overlay:
        shot.type = ShotType.image_overlay
    elif shot.type not in STILL_TYPES:
        shot.type = ShotType.ai_image
