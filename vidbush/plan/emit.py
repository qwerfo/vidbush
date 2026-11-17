"""Part 4 of the clause-first planner: turn clause plans and picks into the existing ShotList contract.

One shot per clause or merged clause (a clause plus its holds). Start and end come from the clause cuts.
Type comes from the pick (video footage, photograph, AI still) or, for a full-frame layout, motion_graphic.
A layout travels in Shot.layout (kind, presentation, elements); a motion_graphic shot also carries a plain
title_card placeholder in Shot.motion so the current renderer can show it until the layouts lane lands.
"""
from __future__ import annotations

from typing import Any, Optional

from ..models import (Candidate, LayoutElement, LayoutSpec, MotionKind, MotionSpec, Narration, Script, Shot,
                      ShotList, ShotType)
from . import anchors as A
from . import vocab as V
from .clauses import Clause
from .edit import in_point, is_ai, needs_pick, shots_of

QUERY_CAP = 8
# order in which per-class queries are flattened into Shot.queries for the current sourcing stage
CLASS_ORDER = ("encyclopedic", "archive", "web_image", "newspaper", "web_video", "stock")


def flat_queries(q: dict[str, list[str]]) -> list[str]:
    cols = [q.get(k, []) for k in CLASS_ORDER]
    out: list[str] = []
    for i in range(max((len(c) for c in cols), default=0)):
        for c in cols:
            if i < len(c) and c[i] not in out:
                out.append(c[i])
    return out[:QUERY_CAP]


def _shot_type(plan: dict, cand: Optional[Candidate], fallback: str) -> ShotType:
    t = plan["treatment"]
    if t in V.LAYOUT_KINDS and not needs_pick(plan) and cand is None:
        return ShotType.motion_graphic
    if cand is not None:
        if is_ai(cand):
            return ShotType.ai_image
        return ShotType.footage if cand.media == "video" else ShotType.image
    if fallback == "ai_still":
        return ShotType.ai_image
    return ShotType.image if plan.get("medium") == "still" else ShotType.footage


def _layout(plan: dict, shot_start: float, picks: dict) -> Optional[LayoutSpec]:
    t = plan["treatment"]
    if t not in V.LAYOUT_KINDS:
        return None
    lay = plan.get("layout") or {}
    els = []
    by_el = {ep["element"]: ep for ep in picks.get("element_picks", [])}
    for k, e in enumerate(lay.get("elements", [])):
        ep = by_el.get(k)
        els.append(LayoutElement(role=str(e.get("role", "")), text=str(e.get("text", "")),
                                 anchor=str(e.get("anchor", "")), picture_subject=str(e.get("picture_subject", "")),
                                 stage=int(e.get("stage", 1) or 1), at=round(max(0.0, e.get("at", shot_start) - shot_start), 3),
                                 pick=ep["_cand"].id if ep else ""))
    return LayoutSpec(kind=t, presentation=plan.get("presentation"),
                      reason=str(plan.get("presentation_reason") or plan.get("reason", "")),
                      title=str(lay.get("title", "")), timeline_id=str(lay.get("timeline_id", "")),
                      purpose=str(lay.get("purpose", "")), elements=els)


def _placeholder(layout: LayoutSpec, duration: float) -> MotionSpec:
    title = layout.title or " / ".join(e.text for e in layout.elements[:3]) or layout.kind.replace("_", " ")
    return MotionSpec(kind=MotionKind.title_card, duration=round(duration, 3),
                      payload={"title": title[:80], "subtitle": layout.kind.replace("_", " ")})


def _cand_copy(c: Candidate, frame: str, seconds: float) -> Candidate:
    cc = c.model_copy(deep=True)
    cc.in_point, cc.out_point = in_point(c, frame, seconds)
    return cc


def _retime(shot: Shot, orig: Optional[tuple[Candidate, str]]) -> None:
    """After a shot grows, its clip and its specs follow the new duration."""
    if orig is not None:
        c = shot.selected_candidate()
        if c is not None:
            c.in_point, c.out_point = in_point(orig[0], orig[1], shot.duration)
    for spec in (shot.motion, shot.overlay):
        if spec is not None:
            spec.duration = shot.duration


def emit(project_id: str, script: Script, narration: Narration, clauses: list[Clause], scene_plans: list[dict],
         edits: Optional[list[dict]] = None) -> tuple[dict, ShotList]:
    """plan_v2 dict (everything decided) and the ShotList. edits may be None (live run before sourcing)."""
    edit_by_scene = {e["scene"]: e for e in (edits or [])}
    shots: list[Shot] = []
    plan_scenes: list[dict[str, Any]] = []
    origin: dict[str, tuple[Candidate, str]] = {}  # emitted candidate id: (stored candidate, frame)
    for sp in scene_plans:
        cl = [c for c in clauses if c.scene == sp["scene"]]
        groups = shots_of(sp["clauses"], cl)
        ed = edit_by_scene.get(sp["scene"])
        picks = ed["picks"] if ed else [{} for _ in groups]
        rows = []
        for g, pk in zip(groups, picks):
            plan, gc = g["plan"], g["clauses"]
            start, end = gc[0].start, gc[-1].end
            dur = round(end - start, 3)
            cand = pk.get("_cand")
            doc = pk.get("_doc")
            if doc is not None and plan.get("presentation") == "full":
                cand = doc  # the document fills the frame, its passages marked: it is the picture
            fallback = pk.get("fallback", "") if cand is None else ""
            # hold_previous: extend the previous shot when it stays inside the hold limit
            if fallback == "hold_previous" and shots and shots[-1].scene_id == sp["scene"]:
                prev = shots[-1]
                limit = V.HOLD_MAX_S if prev.type != ShotType.ai_image else V.AI_HOLD_MAX_S
                if end - prev.start <= limit:
                    prev.end = end
                    prev.duration = round(end - prev.start, 3)
                    _retime(prev, origin.get(prev.selected or ""))
                    prev.flags.append(f"held_over:{gc[0].id}")
                    rows.append(_row(plan, gc, pk, "held into previous shot"))
                    continue
                fallback = "text_over_footage"
                pk["resolved"] = (pk.get("resolved", "") + "; hold over the limit, text over footage").strip("; ")
            overlay_text = ""
            if fallback == "text_over_footage" and cand is None:
                cand = pk.get("_fallback_cand")  # the model's related real footage; never the previous shot's
                overlay_text = str(pk.get("overlay_text") or (plan.get("real_thing") or {}).get("name", "")
                                   or plan.get("anchor", ""))[:60]
            if fallback == "map" and cand is None:
                place = str(pk.get("map_place") or (plan.get("real_thing") or {}).get("place", "")
                            or (plan.get("real_thing") or {}).get("name", ""))
                plan = dict(plan, treatment="route_map", presentation="full",
                            layout={"title": place, "elements": [{"role": "place", "text": place,
                                                                  "anchor": plan.get("anchor", ""), "at": gc[0].speech_start}]})
            typ = _shot_type(plan, cand, fallback)
            lay = _layout(plan, start, pk)
            shot = Shot(id="", beat_id=gc[0].beat, scene_id=sp["scene"], start=start, end=end, duration=dur,
                        type=typ, intent=str(plan.get("subject", "")),
                        must_show=[plan["real_thing"]["name"]] if plan.get("real_thing") else [],
                        queries=flat_queries(plan.get("queries", {})), layout=lay)
            shot.flags = list(plan.get("grammar", []))
            if fallback:
                shot.flags.append(f"fallback:{fallback}")
            if needs_pick(plan) and cand is None and ed:
                shot.flags.append("nothing_fits")
            if typ == ShotType.ai_image:
                shot.ai_prompt = str(plan.get("subject", ""))
            if typ == ShotType.motion_graphic and lay is not None:
                shot.motion = _placeholder(lay, dur)
            if cand is not None:
                cc = _cand_copy(cand, pk.get("frame", "a"), dur)
                shot.candidates.append(cc)
                shot.selected = cc.id
                origin[cc.id] = (cand, pk.get("frame", "a"))
            if overlay_text and cand is not None:
                shot.type = ShotType.footage_overlay if cand.media == "video" else ShotType.image_overlay
                shot.overlay = MotionSpec(kind=MotionKind.text_overlay, duration=dur, background="transparent",
                                          payload={"title": overlay_text, "subtitle": ""},
                                          anchor=str(plan.get("anchor", "")))
            if doc is not None and lay is not None:
                if doc.id not in {c.id for c in shot.candidates}:
                    shot.candidates.append(_cand_copy(doc, "a", dur))
                lay.document = doc.id
            for ep in pk.get("element_picks", []):
                if ep["_cand"].id not in {c.id for c in shot.candidates}:
                    shot.candidates.append(_cand_copy(ep["_cand"], ep.get("frame", "a"), dur))
            if pk.get("label") and cand is not None and not is_ai(cand):
                shot.flags.append(f"label:{pk['label']}")
            shots.append(shot)
            rows.append(_row(plan, gc, pk, ""))
        plan_scenes.append({"scene": sp["scene"], "scene_purpose": sp.get("scene_purpose", ""),
                            "director_attempts": sp.get("attempts", []), "director_unresolved": sp.get("unresolved", []),
                            "edit_attempts": (ed or {}).get("attempts", []), "pool_size": len((ed or {}).get("pool", [])),
                            "shots": rows})
    for i, s in enumerate(shots):
        s.id = f"shot_{i + 1:03d}"
        s.index = i
    total = round(narration.duration + 1.0, 3)
    sl = ShotList(project_id=project_id, duration=total, shots=shots)
    plan = {"version": 1, "planner": "clause", "clauses": len(clauses), "scenes": plan_scenes}
    return plan, sl


def _row(plan: dict, gc: list[Clause], pk: dict, note: str) -> dict:
    cand = pk.get("_cand")
    row = {k: v for k, v in plan.items() if not k.startswith("_")}
    row["clauses"] = [c.id for c in gc]
    row["start"], row["end"] = gc[0].start, gc[-1].end
    if pk:
        row["edit"] = {"pick": cand.id if cand is not None else None, "frame": pk.get("frame"),
                       "label": pk.get("label", ""), "callback": bool(pk.get("callback")),
                       "reason": pk.get("reason", ""), "fallback": pk.get("fallback", ""),
                       "resolved": pk.get("resolved", ""), "ai": bool(cand is not None and is_ai(cand)),
                       "element_picks": [{"element": e["element"], "pick": e["_cand"].id} for e in pk.get("element_picks", [])],
                       "document_pick": pk["_doc"].id if pk.get("_doc") is not None else None}
    if note:
        row["note"] = note
    return row


def anchor_time(plan_row: dict, narration: Narration, clauses_by_id: dict[str, Clause]) -> Optional[float]:
    gc = [clauses_by_id[c] for c in plan_row["clauses"]]
    words = narration.words[gc[0].words[0]:gc[-1].words[1] + 1]
    return A.time_of(words, str(plan_row.get("anchor", "")))
