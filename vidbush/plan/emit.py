"""Part 4 of the clause-first planner: turn clause plans and picks into the existing ShotList contract.

One shot per clause or merged clause (a clause plus its holds). Start and end come from the clause cuts.
Type comes from the pick (video footage, photograph, AI still) or, for a full-frame layout, motion_graphic.
A layout travels in Shot.layout (kind, presentation, elements); a motion_graphic shot also carries a plain
title_card placeholder in Shot.motion so the current renderer can show it until the layouts lane lands.
"""
from __future__ import annotations

from typing import Any, Optional

from ..models import (LAYOUT_ROLES, PANEL_LAYOUT_KINDS, Candidate, LayoutElement, LayoutKind, LayoutSpec,
                      MotionKind, MotionSpec, Narration, RealThing, Script, Shot, ShotList, ShotType)
from . import anchors as A
from . import vocab as V
from .clauses import Clause
from .edit import in_point, is_ai, needs_pick, shots_of

QUERY_CAP = 8
LAYOUT_LEAD_S = 0.25  # a full layout's cut comes this long before its first anchored word
LAYOUT_MIN_S = 3.0  # layouts lane FULL_MIN_S: a full layout shorter than this is not read
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
    if t in V.LAYOUT_KINDS and not needs_pick(plan) and not (t == "document_focus" and cand is not None):
        return ShotType.motion_graphic
    if cand is not None:
        if is_ai(cand):
            return ShotType.ai_image
        return ShotType.footage if cand.media == "video" else ShotType.image
    if fallback == "ai_still":
        return ShotType.ai_image
    return ShotType.image if plan.get("medium") == "still" else ShotType.footage


# Roles a viewer reads as one repeated list, per kind: an element with an unknown role joins that list.
REPEAT_ROLE = {"list_slideshow": "item", "timeline": "event", "chart": "bar", "table": "row", "route_map": "place",
               "document_focus": "mark"}


def _role(kind: LayoutKind, asked: str, used: list[str], has_picture: bool) -> Optional[str]:
    """The renderer's role for a planner element: its own when the kind reads it, else the kind's repeated
    list role, else the next free role in reading order (a text never takes the portrait), else None."""
    roles = LAYOUT_ROLES[kind]
    if asked in roles:
        return asked
    if kind.value in REPEAT_ROLE:
        return REPEAT_ROLE[kind.value]
    free = [r for r in roles if r not in used and r not in ("background", "document", "source")
            and (has_picture or r != "portrait")]
    return free[0] if free else None


def _layout(plan: dict, picks: dict, cand: Optional[Candidate], doc: Optional[Candidate]) -> Optional[LayoutSpec]:
    """The layouts lane's contract (models.LayoutSpec, LAYOUT_ROLES): roles the renderer reads, pictures as
    candidate ids, a document as its own element, the planner's own notes left in plan_v2."""
    t = plan["treatment"]
    if t not in V.LAYOUT_KINDS:
        return None
    kind = LayoutKind(t)
    lay = plan.get("layout") or {}
    by_el = {ep["element"]: ep for ep in picks.get("element_picks", [])}
    els: list[LayoutElement] = []
    used: list[str] = []
    for k, e in enumerate(lay.get("elements", [])):
        ep = by_el.get(k)
        pic = ep["_cand"].id if ep else None
        role = _role(kind, str(e.get("role", "")), used, pic is not None and not str(e.get("text", "")).strip())
        if role is None:
            continue
        used.append(role)
        els.append(LayoutElement(role=role, text=str(e.get("text", "")), anchor=str(e.get("anchor", "")),
                                 picture=pic, stage=2 if int(e.get("stage", 1) or 1) == 2 else 1))
    first_anchor = next((e.anchor for e in els if e.anchor), str(plan.get("anchor", "")))
    if kind == LayoutKind.name_card and "portrait" not in used:
        pic = next((e for e in els if e.picture), None)
        if pic is not None:  # the planner's picked face is the portrait, its words stay the name
            els.insert(0, LayoutElement(role="portrait", anchor=pic.anchor or first_anchor, picture=pic.picture))
            pic.picture = None
    if t in V.OVER_PICTURE_KINDS and "background" not in used and cand is not None:
        els.insert(0, LayoutElement(role="background", anchor=first_anchor, picture=cand.id))
    if kind == LayoutKind.document_focus and doc is not None:
        els.insert(0, LayoutElement(role="document", anchor=first_anchor, picture=doc.id))
        rt = plan.get("real_thing") or {}
        src = ", ".join(x for x in (str(rt.get("name", "")), str(rt.get("date", ""))) if x)
        if src and "source" not in used:
            els.append(LayoutElement(role="source", text=src[:60], anchor=first_anchor))
    purpose = lay.get("purpose") if lay.get("purpose") in V.DOC_PURPOSES else None
    pres = plan.get("presentation") if plan.get("presentation") in V.PRESENTATIONS else None
    return LayoutSpec(kind=kind, elements=els, title=str(lay.get("title", "")), id=str(lay.get("timeline_id", "")),
                      presentation=pres if t in V.PRESENTED_KINDS else None,
                      purpose=purpose if kind == LayoutKind.document_focus else None)


def _full_layout(shot: Shot) -> bool:
    lay = shot.layout
    return lay is not None and (lay.kind not in PANEL_LAYOUT_KINDS or (lay.presentation or "full") == "full")


def start_full_layouts_on_words(shots: list[Shot], first_word: dict[str, float],
                                origin: dict[str, tuple[Candidate, str]]) -> None:
    """A full layout replaces the picture from its cut, so the cut moves to its first anchored word: the
    previous shot of the same scene holds on until then, inside its own hold limit, and the layout keeps at
    least LAYOUT_MIN_S. Panels need no move: they enter on their words over the shot's own picture."""
    for i in range(1, len(shots)):
        s, prev = shots[i], shots[i - 1]
        t = first_word.get(s.id)
        if t is None or not _full_layout(s) or prev.scene_id != s.scene_id or _full_layout(prev):
            continue
        cut = round(t - LAYOUT_LEAD_S, 3)
        if cut - s.start < 0.5 or s.end - cut < LAYOUT_MIN_S:
            continue
        limit = V.AI_HOLD_MAX_S if prev.type == ShotType.ai_image else V.HOLD_MAX_S
        if cut - prev.start > limit:
            continue
        prev.end, s.start = cut, cut
        prev.duration, s.duration = round(prev.end - prev.start, 3), round(s.end - s.start, 3)
        for sh in (prev, s):
            _retime(sh, origin.get(sh.selected or ""))
        s.layout_at = 0.0
        s.flags.append("cut_on_layout_words")


def _real_thing(plan: dict) -> Optional[RealThing]:
    rt = plan.get("real_thing")
    if not isinstance(rt, dict) or not str(rt.get("name") or "").strip():
        return None
    return RealThing(**{k: str(rt.get(k) or "") for k in ("name", "kind", "date", "place")})


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
    first_at: list[Optional[float]] = []  # per emitted shot: narration time of its layout's first anchored word
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
                passages = [str(e.get("text", "")) for e in (plan.get("layout") or {}).get("elements", [])]
                overlay_text = str(pk.get("overlay_text")
                                   or (" / ".join(passages) if plan["treatment"] == "document_focus" else "")
                                   or (plan.get("real_thing") or {}).get("name", "")
                                   or plan.get("anchor", ""))[:60]
            if fallback == "map" and cand is None:
                place = str(pk.get("map_place") or (plan.get("real_thing") or {}).get("place", "")
                            or (plan.get("real_thing") or {}).get("name", ""))
                plan = dict(plan, treatment="route_map", presentation="full", planned_treatment=plan["treatment"],
                            layout={"title": place, "elements": [{"role": "place", "text": place,
                                                                  "anchor": plan.get("anchor", ""), "at": gc[0].speech_start}]})
            typ = _shot_type(plan, cand, fallback)
            lay = _layout(plan, pk, cand, doc)
            if lay is not None and lay.kind == LayoutKind.document_focus and doc is None and overlay_text:
                lay = None  # no page carries the quote: its words go over related footage instead (overlay)
            shot = Shot(id="", beat_id=gc[0].beat, scene_id=sp["scene"], start=start, end=end, duration=dur,
                        type=typ, intent=str(plan.get("subject", "")),
                        must_show=[plan["real_thing"]["name"]] if plan.get("real_thing") else [],
                        queries=flat_queries(plan.get("queries", {})), layout=lay,
                        source_queries={k: [str(q) for q in v] for k, v in (plan.get("queries") or {}).items()
                                        if isinstance(v, list)},
                        real_thing=_real_thing(plan))
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
            if plan["treatment"] == "document_focus" and doc is None and ed:
                shot.flags.append("document_not_found")
            if doc is not None and lay is not None and doc.id not in {c.id for c in shot.candidates}:
                shot.candidates.append(_cand_copy(doc, "a", dur))
            ats = [e["at"] for e in (plan.get("layout") or {}).get("elements", []) if isinstance(e.get("at"), (int, float))]
            first_at.append(min(ats) if ats and lay is not None else None)
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
    start_full_layouts_on_words(shots, {s.id: t for s, t in zip(shots, first_at) if t is not None}, origin)
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
