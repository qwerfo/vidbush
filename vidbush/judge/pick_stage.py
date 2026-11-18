"""The judge stage for a clause-planned project: describe once, pick by comparison, the ladder, labels.

A project planned by the clause-first planner (style profile `planner: clause`) has work/clauses.json and
work/plan_v2.json from the plan stage, and sourcing has filled the shot list with candidates. Here every
asset is described once (cached by content hash), the picker seat assigns pictures per scene, the ladder
fills the clauses where nothing fits, and the shot list is emitted again from the plan and the picks.

The 0 to 100 judge score, its thresholds and its per-candidate calls are not used on this path; they stay
for the old slot planner (director_loop.judge_and_select). AI stills the ladder allows are generated here
and checked with the fixed questions (ai_checks.check_still), never scored.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..models import Candidate, Narration, Script, Shot, ShotList, ShotType

PLAN_KEYS_DROPPED = ("clauses", "start", "end", "edit", "note")
AI_WORKERS = 3  # the image seat's max_concurrency (Elevate: 3 at once)
RESOURCE_ASSETS = 3  # ladder step 1: assets fetched per open clause (sourcing fetches max_candidates, 8)


def is_clause_project(ctx: Any) -> bool:
    try:
        w = Path(ctx.project.paths.work)
        planner = getattr(ctx.profile, "planner", "")
    except AttributeError:  # a context without a project (tests, the editor's one-shot lists)
        return False
    return planner == "clause" and (w / "clauses.json").exists() and (w / "plan_v2.json").exists()


def scene_plans_from(plan: dict) -> list[dict]:
    """The director's per-clause plans back from plan_v2.json: each shot row is its head clause's plan and
    the shot's other clauses are holds."""
    out = []
    for sc in plan.get("scenes", []):
        cl = []
        for r in sc.get("shots", []):
            ids = r.get("clauses") or []
            if not ids:
                continue
            head = {k: v for k, v in r.items() if k not in PLAN_KEYS_DROPPED}
            head["id"] = ids[0]
            if "planned_treatment" in head:  # a fallback treatment from an earlier edit is not the plan
                head["treatment"] = head.pop("planned_treatment")
            cl.append(head)
            cl += [{"id": c, "treatment": "hold"} for c in ids[1:]]
        out.append({"scene": sc["scene"], "scene_purpose": sc.get("scene_purpose", ""), "clauses": cl})
    return out


def _earlier_take(shot: Shot, ctx: Any) -> Optional[Candidate]:
    """A still this shot already has on disk from an earlier run of the stage, checked again with the fixed
    questions (stored replies answer an unchanged check for free): a rerun never pays to make a picture it
    already made. Newest take first; None when none passes."""
    from ..models import SourceKind
    from ..util import video_size
    from .ai_checks import check_still
    files = sorted((Path(ctx.work) / "ai").glob(f"{shot.id}_*.png"),
                   key=lambda f: int(f.stem.rsplit("_", 1)[1]) if f.stem.rsplit("_", 1)[1].isdigit() else 0,
                   reverse=True)
    for f in files:
        n = f.stem.rsplit("_", 1)[1]
        if not n.isdigit():
            continue
        try:
            w, h = video_size(f)
        except Exception:  # noqa: BLE001 - an unreadable file is not a picture
            continue
        cand = Candidate(id=f"{shot.id}_ai_image_{n}", source=SourceKind.ai_image, local_path=str(f), media="image",
                         width=w, height=h, licence="AI generated", attribution="AI generated image",
                         title="AI generated still", keyframes=[str(f)])
        shot.candidates = [c for c in shot.candidates if c.id != cand.id] + [cand]
        check_still(shot, cand, ctx)
        if not cand.reject_reason:
            ctx.log.info("AI still kept from an earlier run", shot=shot.id, candidate=cand.id)
            return cand
    return None


def _generate_ai(shot: Shot, ctx: Any) -> Optional[Candidate]:
    """One AI still, regenerated with the failed fixed-question checks as direction up to ai_retries times."""
    from ..visuals.ai_images import generate_ai_candidate
    from .ai_checks import check_still
    retries = int(getattr(ctx.settings.judge, "ai_retries", 2))
    kept = _earlier_take(shot, ctx)
    if kept is not None:
        return kept
    problem = ""
    last = None
    for take in range(1 + max(0, retries)):
        try:
            cand = generate_ai_candidate(shot, ctx, problem=problem)
        except Exception as e:  # noqa: BLE001 - a failed take leaves the shot to its fallback text
            ctx.log.warn("AI still failed", shot=shot.id, take=take + 1, error=str(e)[:300])
            break
        shot.candidates.append(cand)
        check_still(shot, cand, ctx)
        last = cand
        if not cand.reject_reason:
            return cand
        problem = (cand.reject_reason or "")[:300]
    return last if last is not None and not last.reject_reason else None


def pick_shotlist(shotlist: ShotList, ctx: Any) -> ShotList:
    from ..plan import clauses as C
    from ..plan import edit as E
    from ..plan import emit as M
    from ..shots.director import _finalize
    from .ladder import Ladder
    from .resource import shots_for

    work = Path(ctx.project.paths.work)
    clauses = C.load(work / "clauses.json")
    plan = json.loads((work / "plan_v2.json").read_text(encoding="utf-8"))
    # the director's plans as the planner wrote them: this stage rewrites plan_v2.json, so reading the scene
    # plans back from it would make a rerun pick from its own earlier output (and miss every stored reply)
    direct = work / "direct.json"
    scenes = json.loads(direct.read_text(encoding="utf-8")) if direct.exists() else scene_plans_from(plan)
    script = ctx.load("script", Script)
    narration = ctx.load("narration", Narration)

    def log(m: str) -> None:
        ctx.log.info(m)

    def resource(wants: dict, scene_id: str) -> list[Candidate]:
        import copy
        from ..source.registry import source_pool
        # a repair costs less than the sourcing it repairs: at most RESOURCE_ASSETS distinct assets a clause
        small = copy.copy(ctx)
        small.settings = ctx.settings.model_copy(deep=True)
        small.settings.sources.max_candidates = min(RESOURCE_ASSETS, ctx.settings.sources.max_candidates)
        found = source_pool(shots_for(wants, scene_id), small)
        return [c for cs in found.values() for c in cs]

    on = bool(getattr(ctx.settings.judge, "ladder_resource", False))
    ladder = Ladder(ctx.seats, ctx.settings, work / "pick_sheets", log, resource if on else None)
    edits = E.edit_all(ctx.seats, scenes, clauses, narration, shotlist, work / "pick_sheets", log=log,
                       settings=ctx.settings, ladder=ladder)
    new_plan, sl = M.emit(ctx.project.id, script, narration, clauses, scenes, edits)
    for sc, e in zip(new_plan["scenes"], edits):
        by = {p["clause"]: p for p in e["picks"]}
        for r in sc["shots"]:
            p = by.get(r["clauses"][0])
            if p is not None and "edit" in r:
                r["edit"].update(filled_by=p.get("filled_by", ""), identity=p.get("identity", ""),
                                 start=p.get("start"), restated=p.get("restated"))
    _finalize(sl, ctx.profile)
    from concurrent.futures import ThreadPoolExecutor
    todo = [s for s in sl.shots if s.type == ShotType.ai_image and s.selected_candidate() is None]
    # shots in parallel (each touches only its own shot and files); takes within a shot stay serial
    with ThreadPoolExecutor(max_workers=AI_WORKERS) as pool:
        results = list(pool.map(lambda s: _generate_ai(s, ctx), todo))
    for s, cand in zip(todo, results):
        if cand is not None:
            s.selected = cand.id
        else:
            s.flags.append("ai_still_failed")
    (work / "plan_v2.json").write_text(json.dumps(new_plan, indent=1, ensure_ascii=False), encoding="utf-8")
    ctx.log.info("pick done", shots=len(sl.shots), ladder=ladder.stats)
    return sl
