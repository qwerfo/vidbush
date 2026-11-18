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
from pathlib import Path
from typing import Any, Optional

from ..models import LayoutKind, ShotList, ShotType
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
    rulings = check_ai_rulings(sl)
    plan = json.loads((work / PLAN_FILE).read_text(encoding="utf-8"))
    plan["ai_rulings"] = rulings
    (work / PLAN_FILE).write_text(json.dumps(plan, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    if any(rulings.values()):
        ctx.log.warn("AI rulings broken on the picked shot list", **{k: v for k, v in rulings.items() if v})
    return sl


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
