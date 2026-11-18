"""The clause-first planner's sourcing and pick stages in a live run (style profile `planner: clause`).

shotlist (planner.plan_shotlist): clauses, then the director per scene; writes work/clauses.json,
    work/direct.json (the scene plans) and a shot list with per-class queries and real things.
sourcing (source_clause): one pool per scene through registry.source_pool, each shot searching with the
    director's real_thing and per-class queries; a full layout that shows real pictures (a portrait, two
    sides, list items, a document) searches for stills. Candidates land on their shots.
pick (pick_clause, the judge stage's place): the planner's comparative edit per scene over that pool, the
    fallback ladder applied in code, then the shot list emitted again with the picks; AI stills are generated
    only where the ladder ends at one, under Adam's rulings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..models import LayoutKind, Narration, Script, ShotList, ShotType
from . import clauses as C
from . import edit as E
from . import emit as M
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


# ---------------------------------------------------------------- the fallback ladder

def _place(plan: dict, pick: dict) -> str:
    rt = plan.get("real_thing") or {}
    if rt.get("kind") == "place":
        return str(pick.get("map_place") or rt.get("name") or rt.get("place") or "").strip()
    return str(pick.get("map_place") or rt.get("place") or "").strip()


def ladder(edits: list[dict], scene_plans: list[dict], clauses: list[C.Clause]) -> list[dict]:
    """A shot with nothing picked takes the highest rung it can: a layout with real pictures (kept as
    planned when its pictures were found), then a map of a named place, then text over related real footage,
    then an AI still (never for a real person, event or document; at most 4 s; at most two in a row). A hold
    of the previous real picture stays where the edit chose it (emit checks its hold limit). The AI run is
    counted across scenes in timeline order, so a ladder change never makes a third AI still in a row."""
    by_scene = {e["scene"]: e for e in edits}
    run = 0
    for sp in scene_plans:
        ed = by_scene.get(sp["scene"])
        if ed is None:
            continue
        cl = [c for c in clauses if c.scene == sp["scene"]]
        span: Optional[tuple[float, float]] = None  # (start, limit) of the picture on screen, for holds
        for p, s in zip(ed["picks"], E.shots_of(sp["clauses"], cl)):
            plan = s["plan"]
            cand = p.get("_cand")
            start, end = s["clauses"][0].start, s["clauses"][-1].end
            dur = end - start
            rt = plan.get("real_thing") or {}
            if cand is None and p.get("fallback") == "hold_previous":
                if span is not None and end - span[0] <= span[1] + 1e-6:
                    continue  # the previous picture holds on: not a new picture, the AI run is unchanged
                p["resolved"] = (p.get("resolved", "") + "; ladder: hold over the previous picture's limit").strip("; ")
                p["fallback"] = ""
            if cand is None and E.needs_pick(plan):
                was = p.get("fallback", "")
                place = _place(plan, p) if not E.needs_document(plan) else ""
                ai_ok = (rt.get("kind") not in E.NO_AI_KINDS and not E.needs_document(plan)
                         and dur <= V.AI_HOLD_MAX_S + 1e-6 and run < V.AI_RUN_MAX)
                if place:
                    p["fallback"], p["map_place"] = "map", place
                elif p.get("_fallback_cand") is not None:
                    p["fallback"] = "text_over_footage"
                elif ai_ok:
                    p["fallback"] = "ai_still"
                else:
                    p["fallback"] = "text_over_footage"  # no rung is open: the words alone, reported
                if p["fallback"] != was:
                    p["resolved"] = (p.get("resolved", "") + f"; ladder: {was or 'none'} to {p['fallback']}").strip("; ")
            ai = (cand is not None and E.is_ai(cand)) or (cand is None and p.get("fallback") == "ai_still")
            run = run + 1 if ai else 0
            limit = V.AI_HOLD_MAX_S if ai else V.HOLD_MAX_S
            shown = cand if cand is not None else p.get("_fallback_cand")
            if shown is not None and shown.media == "video":
                limit = min(limit, max(0.0, shown.out_point - shown.in_point))  # a hold never runs past its clip
            span = (start, limit)
    return edits


def edit_scenes(seats: Any, scene_plans: list[dict], clauses: list[C.Clause], narration: Narration,
                sourced: ShotList, sheet_dir: Path, log=None) -> list[dict]:
    """The planner's edit per scene in timeline order. Every asset a scene puts on screen (pick, element
    pictures, document, footage under text) is passed on as used, so no picture repeats by accident in a later
    scene; the AI run carried into the next scene skips holds, which are not new pictures."""
    out, used, ai_tail = [], {}, 0
    for sp in scene_plans:
        cl = [c for c in clauses if c.scene == sp["scene"]]
        e = E.edit_scene(seats, sp, cl, narration, sourced, sheet_dir, used, ai_tail, log)
        for p in e["picks"]:
            for c in [p.get("_cand"), p.get("_doc"), p.get("_fallback_cand")] + [
                    ep.get("_cand") for ep in p.get("element_picks", [])]:
                if c is not None:
                    used.setdefault(E.asset_key(c), p["clause"])
        ai_tail = _trailing_ai(e["picks"], ai_tail)
        out.append(e)
    return out


def _trailing_ai(picks: list[dict], carried: int) -> int:
    """AI stills at the end of a scene's picks (holds skipped); a scene of only AI and holds adds to carried."""
    n = 0
    for p in reversed(picks):
        c = p.get("_cand")
        if c is None and p.get("fallback") == "hold_previous":
            continue
        if (c is not None and E.is_ai(c)) or (c is None and p.get("fallback") == "ai_still"):
            n += 1
            continue
        return n
    return carried + n


# ---------------------------------------------------------------- pick

def pick_clause(sourced: ShotList, ctx: Any) -> ShotList:
    """Comparative edit per scene over the sourced pools, the ladder, the shot list emitted with the picks,
    and AI stills generated where the ladder ends at one."""
    from ..shots.director import _finalize
    work = Path(ctx.work)
    if not (work / DIRECT_FILE).exists():
        raise RuntimeError(f"{work / DIRECT_FILE} is missing: rerun the shotlist stage (resume --force shotlist), "
                           "which writes the scene plans the pick stage edits")
    clauses = C.load(work / CLAUSES_FILE)
    scenes = json.loads((work / DIRECT_FILE).read_text(encoding="utf-8"))
    script = ctx.load("script", Script)
    narration = ctx.load("narration", Narration)
    edits = edit_scenes(ctx.seats, scenes, clauses, narration, sourced, work / "edit_sheets",
                        log=lambda m: ctx.log.info(m))
    ladder(edits, scenes, clauses)
    plan, sl = M.emit(ctx.project.id, script, narration, clauses, scenes, edits)
    _finalize(sl, ctx.profile)
    for shot in sl.shots:
        if shot.type == ShotType.ai_image and shot.selected is None:
            _ai_still(shot, ctx)
    plan["ai_rulings"] = check_ai_rulings(sl)
    (work / PLAN_FILE).write_text(json.dumps(plan, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return sl


def _ai_still(shot: Any, ctx: Any) -> None:
    from ..judge.director_loop import add_ai
    try:
        cand = add_ai(shot, ctx, prompt=shot.ai_prompt or None)
    except Exception as e:  # noqa: BLE001 - a failed still leaves the shot to the renderer's empty-shot rule
        ctx.log.warn("AI still failed", shot=shot.id, error=str(e)[:300])
        cand = None
    if cand is not None:
        shot.selected = cand.id
    else:
        shot.flags.append("ai_still_failed")


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


def saved_scene_plans(work: Path) -> Optional[list[dict]]:
    p = Path(work) / DIRECT_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
