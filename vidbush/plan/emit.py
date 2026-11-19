"""Part 4 of the clause-first planner: turn clause plans and picks into the existing ShotList contract.

One shot per clause or merged clause (a clause plus its holds). Start and end come from the clause cuts.
Type comes from the pick (video footage, photograph, AI still) or, for a full-frame layout, motion_graphic.
A layout travels in Shot.layout (kind, presentation, elements) and is drawn by the layouts renderer; the
picker's real alternates ride on the shot (flags alt:fitting, alt:loose) for when a layout cannot be drawn.
"""
from __future__ import annotations

from typing import Any, Optional

from ..models import (LAYOUT_ROLES, PANEL_LAYOUT_KINDS, Candidate, LayoutElement, LayoutKind, LayoutSpec,
                      Narration, RealThing, Script, Shot, ShotList, ShotType)
from . import anchors as A
from . import vocab as V
from .clauses import Clause
from .edit import in_point, is_ai, needs_pick, shots_of

QUERY_CAP = 8
LAYOUT_LEAD_S = 0.25  # a full layout's cut comes this long before its first anchored word
LAYOUT_MIN_S = V.LAYOUT_MIN_S  # Adam: every layout and panel is visible for at least 3 s
LATE_WORDS_S = 0.75  # a full layout whose first word comes later than this after its cut shows bare ground
ALT_FLAGS = {"_fit": "alt:fitting", "_loose": "alt:loose"}  # the picker's real alternates kept on the shot
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
        pc = origin.get(prev.selected or "")
        if pc is not None and pc[0].media == "video":
            limit = min(limit, max(0.0, pc[0].out_point - pc[0].in_point))  # its clip must cover the longer shot
        if cut - prev.start > limit:
            continue
        prev.end, s.start = cut, cut
        prev.duration, s.duration = round(prev.end - prev.start, 3), round(s.end - s.start, 3)
        for sh in (prev, s):
            _retime(sh, origin.get(sh.selected or ""))
        s.layout_at = 0.0
        s.flags.append("cut_on_layout_words")


def ai_brief(plan: dict) -> str:
    """What an AI still must show: the clause's meaning (who does what) in its setting (where, when, season,
    light), because a still of the nouns alone shows the wrong situation (Adam)."""
    st = plan.get("setting") or {}
    where = ", ".join(x for x in (st.get("place"), st.get("period"),
                                  st.get("season") if st.get("season") not in ("", "any") else "") if x)
    brief = str(plan.get("subject", ""))
    if st.get("circumstance"):
        brief += f"; {st['circumstance']}"
    return f"{brief} ({where})" if where else brief


def alternate_of(shot: Shot) -> Optional[Candidate]:
    """The shot's best real alternate (fitting before loose), or None."""
    for flag in ALT_FLAGS.values():
        for c in shot.candidates:
            if flag in c.flags and not c.reject_reason:
                return c
    return None


def picture_type(c: Candidate) -> ShotType:
    return ShotType.footage if c.media == "video" else ShotType.image


def drop_layout(shot: Shot, why: str) -> bool:
    """Remove a layout that cannot be shown whole; the shot keeps its own picture, else its alternate. Returns
    False when the shot is left with no picture (the live fill then holds a neighbour)."""
    shot.layout_notes.append(f"{shot.layout.kind.value if shot.layout else 'layout'} not shown: {why}")
    shot.flags.append("layout_dropped")
    shot.layout, shot.motion, shot.layout_at = None, None, 0.0
    own = shot.selected_candidate()
    if own is not None and not own.id.endswith("_layout_1"):
        shot.type = picture_type(own) if shot.type == ShotType.motion_graphic else shot.type
        return True
    alt = alternate_of(shot)
    shot.selected = alt.id if alt is not None else None
    if alt is not None:
        shot.type = picture_type(alt)
    elif shot.type == ShotType.motion_graphic:
        shot.type = ShotType.footage  # no graphic any more: the live fill gives it a neighbour's picture
    return alt is not None


def keep_layouts_whole(shots: list[Shot], first_word: dict[str, float],
                       origin: dict[str, tuple[Candidate, str]]) -> list[str]:
    """Adam: a graphic is shown whole or not at all. Every layout and panel stays at least LAYOUT_MIN_S from
    its first word to the cut, else it is dropped and the shot keeps a picture. A full layout whose first word
    comes more than LATE_WORDS_S after its cut (the cut could not move) is shown over the shot's alternate
    picture from its words, so the viewer never sees seconds of bare ground; without an alternate it is
    dropped."""
    notes = []
    for s in shots:
        t = first_word.get(s.id)
        if s.layout is None or t is None:
            continue
        on = max(s.start, t - LAYOUT_LEAD_S)
        if s.end - on < LAYOUT_MIN_S - 1e-6:
            drop_layout(s, f"only {s.end - on:.1f} s from its first word to the cut, under {LAYOUT_MIN_S:g} s")
            notes.append(f"{s.id}: layout dropped, {s.end - on:.1f} s on screen")
            continue
        if _full_layout(s) and s.selected_candidate() is None and on - s.start > LATE_WORDS_S:
            alt = alternate_of(s)
            if alt is None:
                drop_layout(s, f"its first word comes {on - s.start:.1f} s after the cut and no picture can "
                               f"cover the wait")
                notes.append(f"{s.id}: layout dropped, words late and no picture")
                continue
            s.selected = alt.id
            s.type = picture_type(alt)
            s.layout_at = round(on - s.start, 3)
            s.flags.append("layout_over_picture_from_words")
            notes.append(f"{s.id}: full layout shown from its words over {alt.id}")
    return notes


def _real_thing(plan: dict) -> Optional[RealThing]:
    rt = plan.get("real_thing")
    if not isinstance(rt, dict) or not str(rt.get("name") or "").strip():
        return None
    return RealThing(**{k: str(rt.get(k) or "") for k in ("name", "kind", "date", "place")})


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
        lead: Optional[tuple[float, list]] = None  # (start, clauses) of a clause the next shot starts early for
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
                fallback = ""
                pk["resolved"] = (pk.get("resolved", "") + "; hold over the limit, left for the fill").strip("; ")
            if fallback == "extend_next":  # the next shot starts here: its picture covers this clause too
                lead = (start, gc)
                rows.append(_row(plan, gc, pk, "covered by the next shot starting early"))
                continue
            if lead is not None:
                if lead[1][0].scene == gc[0].scene:
                    start = lead[0]
                    dur = round(end - start, 3)
                lead = None
            typ = _shot_type(plan, cand, fallback)
            lay = _layout(plan, pk, cand, doc)
            if lay is not None and lay.kind == LayoutKind.document_focus and doc is None:
                lay = None  # no page carries it: a document is never drawn without its page; the shot keeps a picture
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
                shot.ai_prompt = ai_brief(plan)
            if cand is not None:
                cc = _cand_copy(cand, pk.get("frame", "a"), dur)
                shot.candidates.append(cc)
                shot.selected = cc.id
                origin[cc.id] = (cand, pk.get("frame", "a"))
            for key, flag in ALT_FLAGS.items():  # alternates for a dropped layout or a repeat
                alt = pk.get(key)
                if alt and alt[0].id not in {c.id for c in shot.candidates}:
                    ac = _cand_copy(alt[0], alt[1], dur)
                    ac.flags = list(ac.flags) + [flag]
                    shot.candidates.append(ac)
                    origin[ac.id] = alt
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
    first = {s.id: t for s, t in zip(shots, first_at) if t is not None}
    start_full_layouts_on_words(shots, first, origin)
    drops = keep_layouts_whole(shots, first, origin)
    total = round(narration.duration + 1.0, 3)
    sl = ShotList(project_id=project_id, duration=total, shots=shots)
    plan = {"version": 1, "planner": "clause", "clauses": len(clauses), "scenes": plan_scenes,
            "layout_drops": drops}
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
