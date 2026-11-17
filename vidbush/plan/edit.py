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
FALLBACKS = ("map", "hold_previous", "text_over_footage", "ai_still")
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
    """A picture on screen beneath (or as) this shot is needed."""
    t = plan["treatment"]
    if t == "picture" or t in V.OVER_PICTURE_KINDS:
        return True
    return t in V.PRESENTED_KINDS and plan.get("presentation") in ("panel", "full_then_dock", "panel_then_expand")


def needs_document(plan: dict) -> bool:
    return plan["treatment"] == "document_focus"


def _clause_row(shot: dict, narration: Narration) -> dict:
    p, cl = shot["plan"], shot["clauses"]
    lo, hi = cl[0].words[0], cl[-1].words[1]
    row = {"clause": cl[0].id, "seconds": round(cl[-1].end - cl[0].start, 2),
           "words": " ".join(w.text for w in narration.words[lo:hi + 1]),
           "subject": p.get("subject", ""), "treatment": p["treatment"], "needs_pick": needs_pick(p)}
    if needs_document(p):
        row["needs_document_pick"] = True
        row["document_purpose"] = (p.get("layout") or {}).get("purpose", "")
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


def _cand_at(pool: list[Candidate], n: Any) -> Optional[Candidate]:
    try:
        i = int(n)
    except (TypeError, ValueError):
        return None
    return pool[i - 1] if 1 <= i <= len(pool) else None


def _limit(c: Candidate) -> float:
    return V.AI_HOLD_MAX_S if is_ai(c) else V.HOLD_MAX_S


def check(raw: Any, shots: list[dict], pool: list[Candidate], used_before: dict[str, str],
          ai_tail: int) -> tuple[list[dict], list[str]]:
    """Normalised picks, one per shot in order, and plain-language problems."""
    errors: list[str] = []
    items = raw.get("picks") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return ([{"clause": s["clauses"][0].id, "pick": None, "_cand": None, "element_picks": []} for s in shots],
                ["The reply needs a picks list."])
    by = {str(i.get("clause")): i for i in items if isinstance(i, dict)}
    picks: list[dict] = []
    used: dict[str, str] = {}
    run = ai_tail
    prev_key, prev_secs = None, 0.0

    def seen(key: str) -> Optional[str]:
        return used.get(key) or used_before.get(key)

    for s in shots:
        cid = s["clauses"][0].id
        p = dict(by.get(cid) or {"clause": cid, "pick": None, "reason": "no answer for this clause"})
        dur = s["clauses"][-1].end - s["clauses"][0].start
        rt = s["plan"].get("real_thing") or {}
        n = p.get("pick")
        cand = _cand_at(pool, n) if n is not None else None
        if n is not None and cand is None:
            errors.append(f"{cid}: pick {n} is not a candidate number.")
            p["pick"] = None
        ai_here = False
        if cand is not None:
            fr = str(p.get("frame") or "a").lower()
            if cand.media != "video":
                fr = "a"
            if fr not in FRAME_POS:
                errors.append(f"{cid}: frame must be a, b or c.")
                fr = "a"
            p["frame"] = fr
            key = asset_key(cand)
            adjacent = key == prev_key
            if seen(key) and not adjacent and not p.get("callback"):
                errors.append(f"{cid}: candidate {n} is already used at {seen(key)}; pick another, or mark it a "
                              f"callback and say what changed.")
            on_screen = dur + (prev_secs if adjacent else 0.0)
            if on_screen > _limit(cand) + 1e-6:
                errors.append(f"{cid}: candidate {n} would stay on screen {on_screen:.1f} s; the limit is "
                              f"{_limit(cand):g} s{' for an AI still' if is_ai(cand) else ''}. Use another picture.")
            if is_ai(cand):
                ai_here = True
                if rt.get("kind") in NO_AI_KINDS:
                    errors.append(f"{cid}: candidate {n} is an AI still, never allowed for a real {rt['kind']} "
                                  f"({rt.get('name', '')}).")
                if p.get("label"):
                    errors.append(f"{cid}: a label is only for a real picture of the thing named, not an AI still.")
            used.setdefault(key, cid)
            prev_key, prev_secs = key, on_screen
        else:
            prev_key, prev_secs = None, 0.0
            fb = p.get("fallback")
            if needs_pick(s["plan"]) and fb not in FALLBACKS:
                errors.append(f"{cid}: no pick, so give fallback, one of {', '.join(FALLBACKS)}.")
            if fb == "ai_still":
                ai_here = True
                if rt.get("kind") in NO_AI_KINDS:
                    errors.append(f"{cid}: ai_still is not allowed for a real {rt['kind']}; choose another fallback.")
                if dur > V.AI_HOLD_MAX_S + 1e-6:
                    errors.append(f"{cid}: ai_still on a {dur:.1f} s shot is over the {V.AI_HOLD_MAX_S:g} s limit.")
            if fb == "text_over_footage":
                fc = _cand_at(pool, p.get("fallback_pick"))
                if fc is None or is_ai(fc):
                    errors.append(f"{cid}: text_over_footage needs fallback_pick, a real candidate number.")
                elif seen(asset_key(fc)):
                    errors.append(f"{cid}: fallback_pick {p.get('fallback_pick')} is already used at "
                                  f"{seen(asset_key(fc))}.")
                else:
                    used.setdefault(asset_key(fc), cid)
                p["_fallback_cand"] = fc if fc is not None and not is_ai(fc) else None
            if fb == "map" and not str(p.get("map_place") or (rt.get("place") or rt.get("name") or "")).strip():
                errors.append(f"{cid}: a map fallback needs map_place.")
        if ai_here:
            run += 1
            if run > V.AI_RUN_MAX:
                errors.append(f"{cid}: this would be AI still number {run} in a row; at most {V.AI_RUN_MAX}. Go up "
                              f"the ladder: a map, hold the previous real picture, or text over related real footage.")
        else:
            run = 0
        n_els = len((s["plan"].get("layout") or {}).get("elements", []))
        eps = []
        for ep in p.get("element_picks") or []:
            ec = _cand_at(pool, ep.get("pick")) if isinstance(ep, dict) else None
            try:
                k = int(ep.get("element", -1))
            except (TypeError, ValueError, AttributeError):
                k = -1
            if ec is None or not 0 <= k < n_els:
                errors.append(f"{cid}: element pick {ep} names no element of this layout or no candidate.")
                continue
            if is_ai(ec) and (rt.get("kind") in NO_AI_KINDS or s["plan"]["treatment"] == "name_card"):
                errors.append(f"{cid}: element {k} cannot be an AI still: this layout shows a real "
                              f"{rt.get('kind') or 'person'}.")
            ek = asset_key(ec)
            if seen(ek) and seen(ek) != cid and not p.get("callback"):
                errors.append(f"{cid}: element {k} candidate {ep.get('pick')} is already used at {seen(ek)}.")
            used.setdefault(ek, cid)
            eps.append({"element": k, "pick": int(ep["pick"]), "frame": str(ep.get("frame") or "a").lower(),
                        "_cand": ec})
        p["element_picks"] = eps
        p["_doc"] = None
        if needs_document(s["plan"]) and p.get("document_pick") is not None:
            dc = _cand_at(pool, p.get("document_pick"))
            if dc is None or is_ai(dc):
                errors.append(f"{cid}: document_pick {p.get('document_pick')} must be a real candidate, never AI.")
            elif seen(asset_key(dc)) and seen(asset_key(dc)) != cid and not p.get("callback"):
                errors.append(f"{cid}: document_pick {p.get('document_pick')} is already used at {seen(asset_key(dc))}.")
            else:
                lay = s["plan"].get("layout") or {}
                need = len(lay.get("elements", [])) if lay.get("purpose") in V.MARKED_PURPOSES else 0
                vis = p.get("passages_visible") if isinstance(p.get("passages_visible"), list) else []
                if need and (len(vis) < need or not all(v is True for v in vis[:need])):
                    errors.append(f"{cid}: document_pick {p.get('document_pick')} is used as "
                                  f"{lay.get('purpose')}, so every passage to mark must be visible on it "
                                  f"(passages_visible all true); otherwise leave document_pick null.")
                    p["_doc_unverified"] = True
                used.setdefault(asset_key(dc), cid)
                p["_doc"] = dc
        p["_cand"] = cand
        picks.append(p)
    return picks, errors


def resolve(picks: list[dict], shots: list[dict], ai_tail: int,
            used_before: Optional[dict[str, str]] = None) -> list[dict]:
    """Make the picks legal after the retries, going up the fallback ladder; records what was changed.

    Applies the same rules as check() and never adds content: an illegal pick becomes a hold of the previous real
    picture, or a clause with nothing on it (text over footage with no footage) for the edit to report.
    """
    run = ai_tail
    last_real: Optional[dict] = None
    used = dict(used_before or {})
    prev_key, prev_secs = None, 0.0

    def drop(p: dict, why: str) -> None:
        p["resolved"] = (p.get("resolved", "") + "; " + why).strip("; ")
        p["pick"] = p["_cand"] = None
        p["fallback"] = "hold_previous" if last_real is not None else "text_over_footage"
        p["_fallback_cand"] = None

    for p, s in zip(picks, shots):
        dur = s["clauses"][-1].end - s["clauses"][0].start
        rt = s["plan"].get("real_thing") or {}
        cand = p.get("_cand")
        if cand is not None:
            key = asset_key(cand)
            adjacent = key == prev_key
            on_screen = dur + (prev_secs if adjacent else 0.0)
            if key in used and not adjacent and not p.get("callback"):
                drop(p, f"pick {p.get('pick')} removed: asset already used at {used[key]}")
            elif on_screen > _limit(cand) + 1e-6:
                drop(p, f"pick {p.get('pick')} removed: {on_screen:.1f} s on screen is over the limit")
            elif is_ai(cand) and (rt.get("kind") in NO_AI_KINDS or run + 1 > V.AI_RUN_MAX):
                drop(p, f"AI pick {p.get('pick')} removed by the AI rulings")
            cand = p.get("_cand")
            if cand is not None:
                used.setdefault(key, p.get("clause", ""))
                prev_key, prev_secs = key, on_screen
            else:
                prev_key, prev_secs = None, 0.0
        else:
            prev_key, prev_secs = None, 0.0
        fc = p.get("_fallback_cand")
        if fc is not None and asset_key(fc) in used and used[asset_key(fc)] != p.get("clause"):
            p["_fallback_cand"] = None
        elif fc is not None:
            used.setdefault(asset_key(fc), p.get("clause", ""))
        if p.get("fallback") == "ai_still" and (rt.get("kind") in NO_AI_KINDS or dur > V.AI_HOLD_MAX_S + 1e-6
                                                 or run + 1 > V.AI_RUN_MAX):
            p["resolved"] = (p.get("resolved", "") + "; ai_still fallback refused by the AI rulings").strip("; ")
            p["fallback"] = "hold_previous" if last_real is not None else "text_over_footage"
        ai = cand is not None and is_ai(cand)
        if p.get("label") and (cand is None or ai):
            p["label"] = ""
        keep = []
        for ep in p.get("element_picks", []):
            ec = ep["_cand"]
            ek = asset_key(ec)
            if is_ai(ec) and (rt.get("kind") in NO_AI_KINDS or s["plan"]["treatment"] == "name_card"):
                continue
            if ek in used and used[ek] != p.get("clause") and not p.get("callback"):
                continue
            used.setdefault(ek, p.get("clause", ""))
            keep.append(ep)
        p["element_picks"] = keep
        dc = p.get("_doc")
        if dc is not None and p.get("_doc_unverified"):
            p["_doc"] = dc = None
            p["resolved"] = (p.get("resolved", "") + "; document pick removed: passages not visible on it").strip("; ")
        if dc is not None:
            if is_ai(dc) or (asset_key(dc) in used and used[asset_key(dc)] != p.get("clause") and not p.get("callback")):
                p["_doc"] = None
            else:
                used.setdefault(asset_key(dc), p.get("clause", ""))
        p.pop("_key", None)
        if cand is not None and not ai:
            last_real = p
        if ai or (cand is None and p.get("fallback") == "ai_still"):
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
    picks = resolve(picks, shots, ai_tail, used_before)
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
