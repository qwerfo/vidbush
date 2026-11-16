"""Part 3 of the clause-first planner: one vision call per scene assigns found pictures to the scene's shots.

The model compares candidates side by side on numbered contact sheets. Code checks afterwards only the
physical and legal limits and Adam's rulings on AI stills: no AI still for a named real person, a real event
or a document; no more than two AI stills in a row; an AI still holds at most 4 s; no accidental repeat of an
asset; a label only on a real (not AI) picture. Problems go back to the model in the retry; whatever is still
wrong after that is resolved by the fallback ladder in code.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from ..models import Candidate, Narration, ShotList, SourceKind
from . import prompts as P
from . import sheets as SH
from . import vocab as V
from .clauses import Clause

RETRIES = 1
SEAT, TASK, STAGE = "vision", "plan_edit", "judge"
FRAME_POS = {"a": 0.2, "b": 0.5, "c": 0.8}  # where source/keyframes.py samples a clip
FALLBACKS = ("layout", "map", "hold_previous", "text_over_footage", "ai_still")
NO_AI_KINDS = ("person", "event", "document")
_YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")


def is_ai(c: Candidate) -> bool:
    return c.source in (SourceKind.ai_image, SourceKind.ai_video)


def _usable(c: Candidate) -> bool:
    """Drop what the local prefilter rejected for physical or legal reasons, and mocks."""
    rr = (c.reject_reason or "").lower()
    if "mock" in c.flags or not c.keyframes or not Path(c.keyframes[0]).exists():
        return False
    return not (rr.startswith("prefilter") or rr.startswith("resolution") or "burned_in" in rr)


def asset_key(c: Candidate) -> str:
    return (c.local_path or c.url or c.id).lower()


def scene_pool(old: ShotList, lo: float, hi: float) -> list[Candidate]:
    """Candidates found for any stored shot overlapping [lo, hi), one per asset."""
    seen: set[str] = set()
    pool = []
    for s in old.shots:
        if s.end <= lo or s.start >= hi:
            continue
        for c in s.candidates:
            k = asset_key(c)
            if k in seen or not _usable(c):
                continue
            seen.add(k)
            pool.append(c)
    return pool


def shots_of(plans: list[dict], clauses: list[Clause]) -> list[dict]:
    """One entry per shot: head clause plan plus the clauses its holds cover."""
    out: list[dict] = []
    for p, c in zip(plans, clauses):
        if p["treatment"] == "hold" and out:
            out[-1]["clauses"].append(c)
        else:
            out.append({"plan": p, "clauses": [c]})
    return out


def needs_pick(plan: dict) -> bool:
    t = plan["treatment"]
    if t == "picture" or t in V.OVER_PICTURE_KINDS:
        return True
    return t in V.PRESENTED_KINDS and plan.get("presentation") in ("panel", "full_then_dock", "panel_then_expand")


def _clause_row(shot: dict, narration: Narration) -> dict:
    p, cl = shot["plan"], shot["clauses"]
    lo, hi = cl[0].words[0], cl[-1].words[1]
    row = {"clause": cl[0].id, "seconds": round(cl[-1].end - cl[0].start, 2),
           "words": " ".join(w.text for w in narration.words[lo:hi + 1]),
           "subject": p.get("subject", ""), "treatment": p["treatment"], "needs_pick": needs_pick(p)}
    if p.get("presentation"):
        row["presentation"] = p["presentation"]
    if p.get("real_thing"):
        row["real_thing"] = p["real_thing"]
    els = [{"element": k, "picture_subject": e.get("picture_subject", ""), "text": e.get("text", "")}
           for k, e in enumerate((p.get("layout") or {}).get("elements", [])) if e.get("picture_subject")]
    if els:
        row["element_pictures"] = els
    if p.get("grammar"):
        row["grammar"] = p["grammar"]
    return row


def catalogue(pool: list[Candidate], used_before: dict[str, str]) -> list[dict]:
    out = []
    for n, c in enumerate(pool, 1):
        y = _YEAR.findall(c.title or "")
        e = {"n": n, "source": c.source.value, "media": c.media, "ai": is_ai(c),
             "title": (c.title or "")[:110]}
        if y:
            e["year_in_title"] = y[0]
        if c.media == "video":
            e["clip_seconds"] = round(max(0.0, c.out_point - c.in_point), 1)
        if c.width and c.height:
            e["size"] = f"{c.width}x{c.height}"
        if asset_key(c) in used_before:
            e["used_earlier_at"] = used_before[asset_key(c)]
        out.append(e)
    return out


def check(raw: Any, shots: list[dict], pool: list[Candidate], used_before: dict[str, str],
          ai_tail: int) -> tuple[list[dict], list[str]]:
    """Normalised picks, one per shot in order, and plain-language problems."""
    errors: list[str] = []
    items = raw.get("picks") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return [{"clause": s["clauses"][0].id, "pick": None} for s in shots], ["The reply needs a picks list."]
    by = {str(i.get("clause")): i for i in items if isinstance(i, dict)}
    picks: list[dict] = []
    used: dict[str, str] = {}
    run = ai_tail
    for s in shots:
        cid = s["clauses"][0].id
        p = dict(by.get(cid) or {"clause": cid, "pick": None, "reason": "no answer for this clause"})
        dur = s["clauses"][-1].end - s["clauses"][0].start
        n = p.get("pick")
        cand: Optional[Candidate] = None
        if n is not None:
            try:
                cand = pool[int(n) - 1]
                if int(n) < 1:
                    raise IndexError
            except (TypeError, ValueError, IndexError):
                errors.append(f"{cid}: pick {n} is not a candidate number.")
                p["pick"] = cand = None
        if cand is not None:
            fr = str(p.get("frame") or "a").lower()
            if cand.media != "video" and fr != "a":
                fr = "a"
            if fr not in FRAME_POS:
                errors.append(f"{cid}: frame must be a, b or c.")
                fr = "a"
            p["frame"] = fr
            key = asset_key(cand)
            prev_cid = picks[-1]["clause"] if picks else None
            adjacent = prev_cid is not None and picks[-1].get("_key") == key
            earlier = used.get(key) or used_before.get(key)
            if earlier and not adjacent and not p.get("callback"):
                errors.append(f"{cid}: candidate {n} is already used at {earlier}; pick another, or mark it a "
                              f"callback and say what changed.")
            p["_key"] = key
            rt = s["plan"].get("real_thing") or {}
            if is_ai(cand):
                if rt.get("kind") in NO_AI_KINDS:
                    errors.append(f"{cid}: candidate {n} is an AI still, never allowed for a real {rt['kind']} "
                                  f"({rt.get('name', '')}).")
                if dur > V.AI_HOLD_MAX_S + 1e-6:
                    errors.append(f"{cid}: this shot lasts {dur:.1f} s; an AI still may hold at most "
                                  f"{V.AI_HOLD_MAX_S:g} s. Use a real picture or a fallback.")
                if p.get("label"):
                    errors.append(f"{cid}: a label is only for a real picture of the thing named, not an AI still.")
                run += 1
                if run > V.AI_RUN_MAX:
                    errors.append(f"{cid}: this would be AI still number {run} in a row; at most "
                                  f"{V.AI_RUN_MAX}. Go up the ladder: a layout, a map, hold the previous real "
                                  f"picture, or text over related real footage.")
            else:
                run = 0
            used.setdefault(key, cid)
        else:
            if needs_pick(s["plan"]) and p.get("fallback") not in FALLBACKS:
                errors.append(f"{cid}: no pick, so give fallback, one of {', '.join(FALLBACKS)}.")
            if p.get("fallback") == "ai_still":
                rt = s["plan"].get("real_thing") or {}
                if rt.get("kind") in NO_AI_KINDS:
                    errors.append(f"{cid}: ai_still is not allowed for a real {rt['kind']}; choose another fallback.")
                run += 1
            else:
                run = 0 if p.get("fallback") in ("hold_previous", "text_over_footage") else run
        eps = []
        for ep in p.get("element_picks") or []:
            try:
                ec = pool[int(ep.get("pick")) - 1]
                eps.append({"element": int(ep.get("element", 0)), "pick": int(ep["pick"]),
                            "frame": str(ep.get("frame") or "a").lower(), "_cand": ec})
            except (TypeError, ValueError, IndexError, KeyError):
                continue
        p["element_picks"] = eps
        p["_cand"] = cand
        picks.append(p)
    return picks, errors


def resolve(picks: list[dict], shots: list[dict], ai_tail: int) -> list[dict]:
    """Make the picks legal after the retries, going up the fallback ladder; records what was changed."""
    run = ai_tail
    last_real: Optional[dict] = None
    for p, s in zip(picks, shots):
        dur = s["clauses"][-1].end - s["clauses"][0].start
        rt = s["plan"].get("real_thing") or {}
        cand = p.get("_cand")
        ai = cand is not None and is_ai(cand)
        bad_ai = ai and (rt.get("kind") in NO_AI_KINDS or dur > V.AI_HOLD_MAX_S + 1e-6 or run + 1 > V.AI_RUN_MAX)
        if bad_ai:
            p["resolved"] = f"AI pick {p.get('pick')} removed by the AI rulings"
            p["pick"] = p["_cand"] = None
            p["fallback"] = "hold_previous" if last_real is not None else "text_over_footage"
            cand, ai = None, False
        if p.get("fallback") == "ai_still" and (rt.get("kind") in NO_AI_KINDS or dur > V.AI_HOLD_MAX_S
                                                 or run + 1 > V.AI_RUN_MAX):
            p["resolved"] = "ai_still fallback refused by the AI rulings"
            p["fallback"] = "hold_previous" if last_real is not None else "text_over_footage"
        if p.get("label") and (cand is None or ai):
            p["label"] = ""
        p.pop("_key", None)
        if cand is not None and not ai:
            last_real = p
            run = 0
        elif ai or p.get("fallback") == "ai_still":
            run += 1
        else:
            run = 0
    return picks


def edit_scene(seats, scene_plan: dict, clauses: list[Clause], narration: Narration, old: ShotList,
               sheet_dir: Path, used_before: dict[str, str], ai_tail: int, log=None) -> dict:
    shots = shots_of(scene_plan["clauses"], clauses)
    lo, hi = clauses[0].start, clauses[-1].end
    pool = scene_pool(old, lo, hi)
    entries = [{"n": n, "keyframes": c.keyframes, "ai": is_ai(c), "media": c.media} for n, c in enumerate(pool, 1)]
    paths = SH.build(entries, sheet_dir, scene_plan["scene"])
    note = (f"{len(paths)} sheets follow in order, {SH.ROWS} candidates per sheet, one per row, labelled #n. "
            "A video row shows keyframes a, b, c (20, 50 and 80 percent through the clip); a photo row shows a.")
    rows = [_clause_row(s, narration) for s in shots]
    prompt = P.edit_prompt(scene_plan["scene_purpose"], rows, catalogue(pool, used_before), note)
    base = prompt
    attempts = []
    picks: list[dict] = []
    raw: Any = None
    for attempt in range(RETRIES + 1):
        raw = seats.vision_json(SEAT, TASK, P.EDIT_SYSTEM, prompt, paths, STAGE, details=["high"] * len(paths))
        picks, errors = check(raw, shots, pool, used_before, ai_tail)
        attempts.append({"attempt": attempt, "errors": errors})
        if log:
            log(f"{scene_plan['scene']} edit attempt {attempt}: {len(errors)} problems, pool {len(pool)}")
        if not errors:
            break
        prompt = P.edit_retry(base, {k: v for k, v in (raw or {}).items() if k != "_mock"}, errors)
    picks = resolve(picks, shots, ai_tail)
    return {"scene": scene_plan["scene"], "pool": pool, "picks": picks, "attempts": attempts,
            "sheets": paths, "mock": bool(isinstance(raw, dict) and raw.get("_mock"))}


def edit_all(seats, scene_plans: list[dict], clauses: list[Clause], narration: Narration, old: ShotList,
             sheet_dir: Path, log=None) -> list[dict]:
    out = []
    used: dict[str, str] = {}
    ai_tail = 0
    for sp in scene_plans:
        cl = [c for c in clauses if c.scene == sp["scene"]]
        e = edit_scene(seats, sp, cl, narration, old, sheet_dir, used, ai_tail, log)
        for p in e["picks"]:
            c = p.get("_cand")
            if c is not None:
                used.setdefault(asset_key(c), p["clause"])
        ai_tail = 0
        for p in reversed(e["picks"]):
            c = p.get("_cand")
            if (c is not None and is_ai(c)) or p.get("fallback") == "ai_still":
                ai_tail += 1
            else:
                break
        out.append(e)
    return out


def in_point(c: Candidate, frame: str, seconds: float) -> tuple[float, float]:
    """In and out points that start at the named keyframe and fit the shot inside the clip."""
    if c.media != "video":
        return c.in_point, c.out_point
    span = max(0.0, c.out_point - c.in_point)
    t = c.in_point + span * FRAME_POS.get(frame, 0.2)
    t = max(c.in_point, min(t, c.out_point - seconds)) if span >= seconds else c.in_point
    return round(t, 3), round(min(c.out_point, t + seconds) if span else c.out_point, 3)
