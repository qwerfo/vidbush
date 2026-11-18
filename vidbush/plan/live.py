"""The clause-first planner's sourcing and pick stages in a live run (style profile `planner: clause`).

shotlist (planner.plan_shotlist): clauses, then the director per scene; writes work/clauses.json,
    work/direct.json (the scene plans) and a shot list with per-class queries and real things.
sourcing (source_clause): one pool per scene through registry.source_pool, each shot searching with the
    director's real_thing and per-class queries; a full layout that shows real pictures (a portrait, two
    sides, list items, a document) searches for stills. Candidates land on their shots.
pick (pick_clause, the judge stage's place): the picker lane's stage (judge/pick_stage.py: describe each
    asset once, pick per scene by comparison, its ladder, AI stills under the fixed checks), then Adam's AI
    rulings measured on the result and flagged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from ..models import LayoutKind, Narration, ShotList, ShotType
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
    for n in drop_pictureless_layouts(sl):
        ctx.log.info("document without its page", note=n)
    nar = ctx.load("narration", Narration)
    for n in fill_empty_shots(sl, lambda a, b: spoken(nar, a, b), sourced, describe_records(ctx.settings)):
        ctx.log.info("empty shot filled", note=n)
    rulings = check_ai_rulings(sl)
    plan = json.loads((work / PLAN_FILE).read_text(encoding="utf-8"))
    plan["ai_rulings"] = rulings
    (work / PLAN_FILE).write_text(json.dumps(plan, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    if any(rulings.values()):
        ctx.log.warn("AI rulings broken on the picked shot list", **{k: v for k, v in rulings.items() if v})
    return sl


def spoken(narration: Any, a: float, b: float) -> str:
    """The words whose middle falls in [a, b)."""
    return " ".join(w.text for w in narration.words if a <= (w.start + w.end) / 2 < b)


def _real(c: Any) -> bool:
    from ..models import GENERATED_SOURCES, SourceKind
    return (c is not None and c.source not in GENERATED_SOURCES and c.source != SourceKind.motion
            and not c.reject_reason and bool(c.local_path))


def _limit(c: Any) -> float:
    if c.media == "video":
        return max(0.0, min(V.HOLD_MAX_S, (c.out_point or 0) - (c.in_point or 0)))
    return V.HOLD_MAX_S


# full layouts made of real pictures: without any of them they draw empty boxes or a black frame
PICTURE_ROLES = {LayoutKind.document_focus: ("document",), LayoutKind.name_card: ("portrait",),
                 LayoutKind.list_slideshow: ("item",), LayoutKind.split_comparison: ("left", "right")}


def drop_pictureless_layouts(sl: ShotList) -> list[str]:
    """A picture layout with none of its real pictures is not drawn (the renderer shows empty boxes, an empty
    ground or black: a document's marks need a page, a name card a face). Adam's ruling for a missing
    document holds for all of them: the words go over related real footage, so the layout goes and
    fill_empty_shots gives the shot a related picture with its spoken words."""
    notes = []
    for s in sl.shots:
        lay = s.layout
        roles = PICTURE_ROLES.get(lay.kind) if lay is not None else None
        if not roles or any(e.role in roles and e.picture for e in lay.elements):
            continue
        s.layout, s.motion, s.selected = None, None, None
        s.type = ShotType.footage
        s.flags.append(f"{lay.kind.value}_without_pictures")
        notes.append(f"{s.id}: {lay.kind.value} has none of its real pictures; its words over related footage")
    return notes


drop_pageless_documents = drop_pictureless_layouts  # earlier name


_WORD = re.compile(r"[a-z]{4,}")


def _words(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def clean_record(rec: Optional[dict]) -> bool:
    """A picture words may go over: described, usable, nothing blocking, not a talking head or a screen,
    not a document, and no words of its own (slates, leaders, title cards, burned captions clash with ours)."""
    if not rec:
        return False
    v = rec.get("visual") or {}
    bad = {"burned_in_subtitles", "watermark", "stock_preview_mark", "channel_logo", "presenter_to_camera",
           "screen_recording"}
    return (bool(v.get("usable", True)) and not bad & set(v.get("quality") or []) and v.get("look") != "document"
            and not str(v.get("text_in_frame") or "").strip())


def describe_records(settings: Any) -> Any:
    """record(candidate) -> its cached describe record, or None; never calls a model."""
    from ..judge import describe as D
    root = D.cache_dir(settings)

    def record(c: Any) -> Optional[dict]:
        try:
            f = root / f"{D.record_key(c)}.json"
            return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        except (OSError, ValueError):
            return None
    return record


def fill_empty_shots(sl: ShotList, words_at: Any = None, pool: Optional[ShotList] = None,
                     record: Any = None) -> list[str]:
    """No shot leaves the pick with nothing on screen (the renderer would show a plain slate). In order:
    the previous shot's real picture holds on when the two together stay inside its hold limit; else the
    shot's own best unused real candidate, or the scene's, with the clause's words over it; else it stays
    empty and is reported. A shot with a layout is left to the layouts renderer. Returns notes."""
    from ..models import MotionKind, MotionSpec
    notes: list[str] = []
    shown = {c.url or c.local_path for s in sl.shots for c in [s.selected_candidate()] if c is not None}
    for i, s in enumerate(sl.shots):
        if s.selected_candidate() is not None or s.layout is not None:
            continue
        prev = sl.shots[i - 1] if i else None
        pc = prev.selected_candidate() if prev is not None else None
        if (prev is not None and prev.scene_id == s.scene_id and _real(pc) and prev.overlay is None
                and prev.duration + s.duration <= _limit(pc) + 1e-6):
            s.candidates.append(pc.model_copy(update={"in_point": (pc.in_point or 0) + prev.duration}))
            s.selected, s.type = pc.id, prev.type
            s.flags.append("filled:hold_previous")
            notes.append(f"{s.id}: holds {prev.id}'s picture")
            continue
        said = words_at(s.start, s.end) if words_at else ""
        want = _words(said) | _words(s.intent)

        def fits(c: Any) -> bool:
            return _real(c) and (c.url or c.local_path) not in shown and (record is None or clean_record(record(c)))

        def rank(c: Any) -> tuple[int, float]:
            rec = record(c) if record is not None else None
            v = (rec or {}).get("visual") or {}
            seen = _words(" ".join(str(v.get(k) or "") for k in ("shows", "place", "period")) + " " + (c.title or ""))
            return (len(want & seen), c.score or 0.0)

        own = [c for c in s.candidates if fits(c)]
        # the scene's whole sourced pool (the pick keeps only what it chose), else the picked list's
        src = pool.shots if pool is not None else sl.shots
        scene = [c for t in src if t.scene_id == s.scene_id and t.start < s.end + 30 and t.end > s.start - 30
                 for c in t.candidates if fits(c)]
        pick = max(own or scene, key=rank, default=None)
        if pick is None:
            s.flags.append("filled:nothing")
            notes.append(f"{s.id}: nothing real in the shot or scene, left empty")
            continue
        if pick.id not in {c.id for c in s.candidates}:
            s.candidates.append(pick)
        s.selected = pick.id
        shown.add(pick.url or pick.local_path)
        words = " ".join(said.split()[:6])
        video = pick.media == "video"
        if words:
            s.type = ShotType.footage_overlay if video else ShotType.image_overlay
            s.overlay = MotionSpec(kind=MotionKind.text_overlay, duration=s.duration, background="transparent",
                                   payload={"title": words, "subtitle": ""})
        else:
            s.type = ShotType.footage if video else ShotType.image
        s.flags.append("filled:related_with_words")
        notes.append(f"{s.id}: related real picture {pick.id} with its words")
    return notes


def check_ai_rulings(sl: ShotList) -> dict[str, Any]:
    """Adam's AI rulings measured on the emitted shot list: runs longer than two, stills over 4 s, AI on a
    named real person, event or document. Each offending shot is flagged; the counts go into plan_v2.json."""
    out = {"run_over_two": [], "over_4s": [], "real_thing": []}
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
    for key, ids in out.items():
        for s in sl.shots:
            if s.id in ids:
                s.flags.append(f"ai_ruling:{key}")
    return out
