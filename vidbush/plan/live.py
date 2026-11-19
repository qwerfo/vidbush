"""The clause-first planner's sourcing and pick stages in a live run (style profile `planner: clause`).

shotlist (planner.plan_shotlist): clauses, then the director per scene; writes work/clauses.json,
    work/direct.json (the scene plans) and a shot list with per-class queries and real things.
sourcing (source_clause): one pool per scene through registry.source_pool, each shot searching with the
    director's real_thing and per-class queries; a full layout that shows real pictures (a portrait, two
    sides, list items, a document) searches for stills. Candidates land on their shots.
pick (pick_clause, the judge stage's place): the picker lane's stage (judge/pick_stage.py: describe each
    asset once, pick per scene by comparison, its ladder, AI stills under the fixed checks); then every
    layout that cannot be shown whole is dropped to a picture, an asset serves at most two shots in a row and
    never again within a minute, an empty shot holds a neighbour, and Adam's AI rulings are measured.
    No narration is ever typed over a picture (Adam).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from ..models import GENERATED_SOURCES, PANEL_LAYOUT_KINDS, LayoutKind, Narration, ShotList, ShotType, SourceKind
from . import edit as E
from . import vocab as V

# full layouts that draw real pictures: their shots source stills even though nothing plays beneath them
PICTURE_LAYOUTS = {LayoutKind.name_card, LayoutKind.split_comparison, LayoutKind.list_slideshow,
                   LayoutKind.document_focus}
DIRECT_FILE, CLAUSES_FILE, PLAN_FILE = "direct.json", "clauses.json", "plan_v2.json"
SCENE_WORKERS = 3


def is_clause(profile: Any) -> bool:
    return str(getattr(profile, "planner", "") or "") == "clause"


# ---------------------------------------------------------------- sourcing

def source_clause(shotlist: ShotList, ctx: Any) -> ShotList:
    """Every scene sourced into one pool (each asset fetched once, at most max_candidates distinct assets a
    shot); each shot keeps the candidates the pool fetched for it."""
    from ..source.registry import shot_media, source_pool
    scenes: dict[str, list[Any]] = {}
    for s in shotlist.shots:
        probe = s
        if shot_media(s) is None and s.layout is not None and s.layout.kind in PICTURE_LAYOUTS:
            probe = s.model_copy(update={"type": ShotType.image})  # a layout's pictures are stills
        scenes.setdefault(s.scene_id, []).append((s, probe))
    from concurrent.futures import ThreadPoolExecutor

    def one(item: tuple[str, list[Any]]) -> None:
        sid, pairs = item
        got = source_pool([p for _, p in pairs], ctx)  # one call per scene: its dedupe is the scene's own
        for s, p in pairs:
            have = {c.id for c in s.candidates}
            s.candidates += [c for c in got.get(p.id, []) if c.id not in have]
        ctx.log.info("scene pool", scene=sid, shots=len(pairs),
                     candidates=sum(len(got.get(p.id, [])) for _, p in pairs))

    # scenes in parallel: one short scene must not hold the downloads of the rest (adapters throttle per host)
    with ThreadPoolExecutor(max_workers=SCENE_WORKERS) as pool:
        list(pool.map(one, scenes.items()))
    return shotlist


# ---------------------------------------------------------------- pick

def pick_clause(sourced: ShotList, ctx: Any) -> ShotList:
    """The picker lane's stage (judge/pick_stage.pick_shotlist: describe once, pick per scene by comparison,
    the ladder, AI stills under the checks), then Adam's AI rulings measured on the result and flagged."""
    from ..judge.pick_stage import pick_shotlist
    work = Path(ctx.work)
    for name in (CLAUSES_FILE, PLAN_FILE):
        if not (work / name).exists():
            raise RuntimeError(f"{work / name} is missing: rerun the shotlist stage (resume --force shotlist)")
    sl = pick_shotlist(sourced, ctx)
    for n in drop_incomplete_layouts(sl):
        ctx.log.info("layout dropped", note=n)
    for n in enforce_repeats(sl):
        ctx.log.info("repeat", note=n)
    for n in fill_empty_shots(sl):
        ctx.log.info("empty shot filled", note=n)
    renumber(sl)
    rulings = check_ai_rulings(sl)
    plan = json.loads((work / PLAN_FILE).read_text(encoding="utf-8"))
    plan["ai_rulings"] = rulings
    (work / PLAN_FILE).write_text(json.dumps(plan, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    if any(rulings.values()):
        ctx.log.warn("AI rulings broken on the picked shot list", **{k: v for k, v in rulings.items() if v})
    return sl


def _real(c: Any) -> bool:
    return (c is not None and c.source not in GENERATED_SOURCES and c.source != SourceKind.motion
            and not c.reject_reason and bool(c.local_path))


def key_of(c: Any) -> str:
    return (c.local_path or c.url or c.id).lower()


def _file_s(c: Any) -> float:
    from ..util import media_duration
    try:
        return media_duration(c.local_path)
    except Exception:  # noqa: BLE001 - an unreadable length gives no room to grow
        return c.out_point or 0.0


def can_cover(c: Any, start: float, end: float, earlier: float = 0.0) -> bool:
    """The picture may stay on screen over [start, end): inside its hold limit (4 s for an AI still), and a
    video has the frames for it (earlier: seconds it must start before its current in point)."""
    if c.source in (SourceKind.ai_image, SourceKind.ai_video):
        return end - start <= V.AI_HOLD_MAX_S + 1e-6
    if end - start > V.HOLD_MAX_S + 1e-6:
        return False
    if c.media != "video":
        return True
    first = (c.in_point or 0.0) - earlier
    return first >= -0.05 and first + (end - start) <= _file_s(c) + 0.05


# full layouts made of real pictures: every one of them must be there, or the layout is not drawn
PICTURE_ROLES = {LayoutKind.document_focus: ("document",), LayoutKind.name_card: ("portrait",),
                 LayoutKind.list_slideshow: ("item",), LayoutKind.split_comparison: ("left", "right")}


def incomplete(lay: Any) -> Optional[str]:
    """Why a layout cannot be shown whole, or None (Adam: a graphic is shown whole or not at all)."""
    roles = PICTURE_ROLES.get(lay.kind)
    if roles:
        need = [e for e in lay.elements if e.role in roles]
        if not need or any(not e.picture for e in need):
            return f"{lay.kind.value} is missing real pictures"
        if lay.kind == LayoutKind.split_comparison and {e.role for e in need} != {"left", "right"}:
            return "split_comparison needs both sides"
    if lay.kind == LayoutKind.route_map:
        from ..judge.ladder import place_resolves
        names = []
        for e in lay.elements:
            if e.role == "route":
                a, _, b = e.text.partition(" to ")
                names += [a, b.split(" by ")[0]]
            elif e.role == "place":
                names.append(e.text)
        names = [n.strip() for n in names if n.strip()]
        lost = [n for n in names if not place_resolves(n)]
        if not names or lost:
            return f"route_map places not found: {', '.join(lost) or 'none given'}"
    return None


def drop_incomplete_layouts(sl: ShotList) -> list[str]:
    """A layout missing a picture or a place is not drawn (empty boxes, one-sided comparisons and empty maps
    were the worst opening faults); the shot keeps its own picture, else the picker's alternate."""
    from .emit import drop_layout
    notes = []
    for s in sl.shots:
        why = incomplete(s.layout) if s.layout is not None else None
        if why:
            drop_layout(s, why)
            notes.append(f"{s.id}: {why}; the shot keeps a picture")
    return notes


drop_pictureless_layouts = drop_incomplete_layouts  # earlier name


def _shown_near(shots: list, s: Any, c: Any) -> bool:
    """Another shot shows this asset within a minute of s."""
    k = key_of(c)
    for t in shots:
        o = t.selected_candidate() if t is not s else None
        if o is not None and key_of(o) == k and t.start < s.end + V.REPEAT_GAP_S and t.end > s.start - V.REPEAT_GAP_S:
            return True
    return False


def _merge(prev: Any, s: Any) -> None:
    """prev stays on screen over s: one shot, one move."""
    pc = prev.selected_candidate()
    if pc is not None and pc.media == "video":
        pc.out_point = round(pc.out_point + (s.end - prev.end), 3)
    prev.end = s.end
    prev.duration = round(prev.end - prev.start, 3)
    for spec in (prev.motion, prev.overlay):
        if spec is not None:
            spec.duration = prev.duration
    prev.flags.append(f"held_over:{s.id}")


def enforce_repeats(sl: ShotList) -> list[str]:
    """Adam: one asset serves at most two consecutive shots (a whole view, then one closer view) and is never
    shown again within a minute. A shot that breaks it takes its real alternate when that is free; else, when
    it continues the same asset, the previous shot holds over it inside its limit (one shot, one move); else
    its picture is removed and the fill holds a neighbour."""
    from .emit import alternate_of, picture_type
    notes: list[str] = []
    out: list[Any] = []
    for s in sl.shots:
        c = s.selected_candidate()
        if c is None or not _real(c):
            out.append(s)
            continue
        k = key_of(c)
        run = 1
        for t in reversed(out):
            if t.selected_candidate() is None or key_of(t.selected_candidate()) != k:
                break
            run += 1
        before = out[:len(out) - (run - 1)]
        soon = any(t.selected_candidate() is not None and key_of(t.selected_candidate()) == k
                   and s.start - t.end < V.REPEAT_GAP_S for t in before)
        if run <= V.REPEAT_RUN_MAX and not soon:
            out.append(s)
            continue
        why = "a third shot in a row" if run > V.REPEAT_RUN_MAX else "again within a minute"
        alt = alternate_of(s)
        if alt is not None and key_of(alt) != k and not _shown_near(sl.shots, s, alt):
            s.selected = alt.id
            if s.layout is None:
                s.type = picture_type(alt)
            s.flags.append("repeat:alternate")
            notes.append(f"{s.id}: {c.id} {why}; shows {alt.id}")
            out.append(s)
            continue
        prev = out[-1] if out else None
        pc = prev.selected_candidate() if prev is not None else None
        if (run > 1 and pc is not None and prev.layout is None and s.layout is None and s.overlay is None
                and can_cover(pc, prev.start, s.end)):
            _merge(prev, s)
            notes.append(f"{s.id}: {why}; {prev.id} holds over it as one shot")
            continue
        if s.layout is None:
            s.selected = None
            s.flags.append("repeat:removed")
            notes.append(f"{s.id}: {c.id} {why}; picture removed for the fill")
        out.append(s)
    sl.shots = out
    return notes


OVER_PICTURE_KINDS = (LayoutKind.number_over_footage, LayoutKind.date_title)


def needs_picture(s: Any) -> bool:
    """The shot shows nothing unless it gets a picture: no selected candidate, and no full-frame layout."""
    return s.selected_candidate() is None and (s.layout is None or s.type != ShotType.motion_graphic)


def layout_without_picture(s: Any, notes: list[str]) -> None:
    """A panel with no picture beneath shows full screen (the graphic is all there is to see); a number or
    date over footage with no footage is dropped, and the shot is then filled like any empty shot."""
    from .emit import drop_layout
    if s.layout.kind in PANEL_LAYOUT_KINDS:
        s.layout.presentation = "full"
        s.type = ShotType.motion_graphic
        s.flags.append("panel_shown_full:no_picture")
        notes.append(f"{s.id}: {s.layout.kind.value} panel has no picture beneath; shown full screen")
    else:
        drop_layout(s, "no real picture to show it over")
        notes.append(f"{s.id}: {s.layout.kind.value if s.layout else 'layout'} dropped, no picture beneath")


def fill_empty_shots(sl: ShotList, *_: Any, **__: Any) -> list[str]:
    """No shot leaves the pick with nothing on screen, and nothing is ever typed over a picture to fill one
    (Adam). In order: the shot's own real alternate (the picker's fitting, then loose picture); the previous
    shot's real picture holds over it (one shot, one move) inside its hold limit; the next shot's real picture
    starts early over it; else it stays empty and is reported. A shot with a layout is left to the renderer.
    Word overlap with the narration no longer ranks anything: fit is the picker's judgement of the situation."""
    from .emit import alternate_of, picture_type
    notes: list[str] = []
    out: list[Any] = []
    shots = sl.shots
    for i, s in enumerate(shots):
        if not needs_picture(s):
            out.append(s)
            continue
        alt = alternate_of(s)
        if alt is not None and not _shown_near(shots, s, alt):
            s.selected = alt.id
            if s.layout is None:
                s.type = picture_type(alt)
            s.flags.append("filled:alternate")
            notes.append(f"{s.id}: its alternate {alt.id}")
            out.append(s)
            continue
        if s.layout is not None:
            layout_without_picture(s, notes)
            if not needs_picture(s):
                out.append(s)
                continue
        prev = out[-1] if out else None
        pc = prev.selected_candidate() if prev is not None else None
        if (prev is not None and prev.scene_id == s.scene_id and _real(pc) and prev.layout is None
                and can_cover(pc, prev.start, s.end)):
            _merge(prev, s)
            notes.append(f"{s.id}: {prev.id}'s picture holds over it")
            continue
        nxt = shots[i + 1] if i + 1 < len(shots) else None
        nc = nxt.selected_candidate() if nxt is not None else None
        early = nxt.start - s.start if nxt is not None else 0.0
        back = min(early, nc.in_point or 0.0) if nc is not None and nc.media == "video" else early
        if (nxt is not None and nxt.scene_id == s.scene_id and _real(nc) and nxt.layout is None
                and can_cover(nc, s.start, nxt.end, earlier=back)):
            if nc.media == "video":  # play earlier frames where there are some, else the same window longer
                nc.in_point = round(nc.in_point - back, 3)
            nxt.start = s.start
            nxt.duration = round(nxt.end - nxt.start, 3)
            if nc.media == "video":
                nc.out_point = round(nc.in_point + nxt.duration, 3)
            for spec in (nxt.motion, nxt.overlay):
                if spec is not None:
                    spec.duration = nxt.duration
            nxt.flags.append(f"started_early_over:{s.id}")
            notes.append(f"{s.id}: {nxt.id}'s picture starts early over it")
            continue
        if split_between(s, prev, pc, nxt, nc):
            notes.append(f"{s.id}: {prev.id} holds into it and {nxt.id} starts early, meeting inside it")
            continue
        s.flags.append("filled:nothing")
        notes.append(f"{s.id}: nothing real to hold or show, left empty")
        out.append(s)
    sl.shots = out
    return notes


def split_between(s: Any, prev: Any, pc: Any, nxt: Any, nc: Any) -> bool:
    """Cover an empty shot with both neighbours: the previous picture holds into it and the next starts early,
    each inside its own limit, meeting at a point inside it. Only within one scene, real pictures, no layout."""
    def ok(n, c):
        return n is not None and n.scene_id == s.scene_id and _real(c) and n.layout is None
    if not (ok(prev, pc) and ok(nxt, nc)):
        return False
    for frac in (0.5, 0.35, 0.65, 0.2, 0.8):
        cut = round(s.start + frac * s.duration, 3)
        if can_cover(pc, prev.start, cut) and can_cover(nc, cut, nxt.end, earlier=nxt.start - cut):
            prev.end, prev.duration = cut, round(cut - prev.start, 3)
            if pc.media == "video":
                pc.out_point = round(pc.in_point + prev.duration, 3)
            if nc.media == "video":
                nc.in_point = round(nc.in_point - (nxt.start - cut), 3)
            nxt.start, nxt.duration = cut, round(nxt.end - cut, 3)
            for spec in (prev.motion, prev.overlay, nxt.motion, nxt.overlay):
                if spec is not None:
                    spec.duration = prev.duration if spec in (prev.motion, prev.overlay) else nxt.duration
            prev.flags.append(f"held_into:{s.id}")
            nxt.flags.append(f"started_early_over:{s.id}")
            return True
    return False


def renumber(sl: ShotList) -> None:
    """Shot ids follow screen order again after merges (the AI takes on disk were named before, by emit's
    deterministic ids, and are matched there, so renaming here is safe)."""
    for i, s in enumerate(sl.shots):
        s.id = f"shot_{i + 1:03d}"
        s.index = i


def check_ai_rulings(sl: ShotList) -> dict[str, Any]:
    """Adam's AI rulings measured on the emitted shot list: runs longer than two, stills over 4 s, AI on a
    named real person or a document, AI in the opening 15 s. Each offending shot is flagged; the counts go into
    plan_v2.json."""
    out = {"run_over_two": [], "over_4s": [], "real_thing": [], "opening": []}
    run = 0
    for s in sl.shots:
        ai = s.type in (ShotType.ai_image, ShotType.ai_video)
        run = run + 1 if ai else 0
        if not ai:
            continue
        if run > V.AI_RUN_MAX:
            out["run_over_two"].append(s.id)
        if s.duration > V.AI_HOLD_MAX_S + 1e-6:
            out["over_4s"].append(s.id)
        rt = getattr(s, "real_thing", None)
        if rt is not None and rt.kind in E.NO_AI_KINDS:
            out["real_thing"].append(s.id)
        if not E.ai_allowed_at(s.start):
            out["opening"].append(s.id)
    for key, ids in out.items():
        for s in sl.shots:
            if s.id in ids:
                s.flags.append(f"ai_ruling:{key}")
    return out
