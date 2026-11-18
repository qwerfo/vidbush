"""Live entry point of the clause-first planner, selected by a style profile with `planner: clause`.

Runs Part 1 (clauses), Part 2 (director per scene) and Part 4 (ShotList) before sourcing. Part 3, the
comparative edit, needs the found pool: until it is wired into the judge stage, sourcing and the judge pick
per shot as they do for the old planner, using the queries and subjects written here.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

from ..models import Brand, DirectorPlan, Narration, Project, Research, Script, ShotList, StyleProfile, VisualIdea
from . import clauses as C
from . import direct as D
from . import emit as M


def plan_shotlist(project: Project, script: Script, narration: Narration, research: Optional[Research],
                  profile: StyleProfile, brand: Brand, seats: Any,
                  rng: random.Random) -> tuple[ShotList, DirectorPlan]:
    from ..shots.director import _finalize

    log = getattr(seats, "log", None)
    clauses = C.segment(script, narration)
    scenes = D.direct_all(seats, script, clauses, narration,
                          log=(lambda m: log.info(m)) if log is not None and hasattr(log, "info") else None)
    plan, sl = M.emit(project.id, script, narration, clauses, scenes, None)
    _finalize(sl, profile)
    work = Path(project.paths.work) if project.paths and project.paths.work else None
    if work is not None:
        work.mkdir(parents=True, exist_ok=True)
        C.save(clauses, work / "clauses.json")
        # the scene plans, so the pick stage can run the edit on the sourced pools and emit again
        (work / "direct.json").write_text(json.dumps(scenes, indent=1, ensure_ascii=False), encoding="utf-8")
        (work / "plan_v2.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    ideas = [VisualIdea(beat_id=s.beat_id, family="footage" if s.type.value.startswith("footage") else "still",
                        intent=s.intent, must_show=s.must_show, queries=s.queries) for s in sl.shots
             if s.type.value != "motion_graphic"]
    return sl, DirectorPlan(ideas=ideas)
