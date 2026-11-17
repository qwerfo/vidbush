"""Per-run cost cap: stop a project at the next stage boundary once its cost total crosses the cap.

The runner (pipeline._run_locked) calls reached() after every stage when run_project got a cost_cap.
A capped project ends as needs_review with project.status_note set, a work/cost_cap.json record and a
log line. It resumes normally later (python -m vidbush resume <id>, without a cap or with a higher one).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from ..models import STAGES, StageStatus
from ..util import read_json, write_json
from .context import StageContext

CAP_FILE = "cost_cap.json"
CAP_PREFIX = "cost cap"


def cap_path(work: Path) -> Path:
    """output/<id>/work/cost_cap.json."""
    return Path(work) / CAP_FILE


def reset(ctx: StageContext) -> None:
    """Forget an earlier cap at the start of a run (the cap is per run)."""
    cap_path(ctx.work).unlink(missing_ok=True)
    if ctx.project.status_note.startswith(CAP_PREFIX):
        ctx.project.status_note = ""


def reached(ctx: StageContext, stage: str, cap_usd: float, force: set[str]) -> bool:
    """True (and the project marked needs_review) when the total crossed cap_usd after `stage`.

    Never caps after the last stage: the video is finished by then, so stopping saves nothing.
    Forced stages that have not run yet are marked pending so a later resume still reruns them.
    """
    total = ctx.tracker.total_usd
    if total <= cap_usd or stage == STAGES[-1]:
        return False
    rest = STAGES[STAGES.index(stage) + 1:]
    reason = (f"{CAP_PREFIX} reached: ${total:.2f} spent, cap ${cap_usd:.2f}; stopped after {stage}, "
              f"resume continues at {rest[0]}")
    mark_forced_pending(ctx, stage, force)
    ctx.project.status = "needs_review"
    ctx.project.status_note = reason
    write_json(cap_path(ctx.work), {"capped": True, "cap_usd": cap_usd, "total_usd": round(total, 5),
                                    "stage": stage, "next_stage": rest[0], "reason": reason, "at": time.time()})
    ctx.log.warn("cost cap reached, run stopped", stage=stage, total_usd=round(total, 5), cap_usd=cap_usd)
    return True


def mark_forced_pending(ctx: StageContext, stage: str, force: set[str]) -> None:
    """Forced stages after `stage` that have not run yet go back to pending, so a resume after a stop
    (cap or failure) reruns them instead of keeping outputs made from the old input."""
    for name in STAGES[STAGES.index(stage) + 1:]:
        rec = ctx.project.stages.get(name)
        if rec and name in force and rec.status == StageStatus.done:
            rec.status = StageStatus.pending


def stopped_inside(ctx: StageContext, stage: str, total: float, cap_usd: float) -> None:
    """Record a cap reached inside `stage` (Seats.check_cap raised before a paid call): the stage stays
    pending, so resume reruns it, and the replies already paid for come back from the reply cache."""
    reason = (f"{CAP_PREFIX} reached: ${total:.2f} spent, cap ${cap_usd:.2f}; stopped inside {stage}, "
              f"resume reruns {stage} (stored replies cost nothing)")
    ctx.project.status = "needs_review"
    ctx.project.status_note = reason
    write_json(cap_path(ctx.work), {"capped": True, "cap_usd": cap_usd, "total_usd": round(total, 5),
                                    "stage": stage, "next_stage": stage, "inside_stage": True,
                                    "reason": reason, "at": time.time()})
    ctx.log.warn("cost cap reached inside a stage, run stopped", stage=stage, total_usd=round(total, 5),
                 cap_usd=cap_usd)


def read_cap(work: Path) -> Optional[dict[str, Any]]:
    """The cap record of the last run, or None when that run was not capped."""
    path = cap_path(work)
    try:
        return read_json(path) if path.exists() else None
    except (OSError, ValueError):
        return None
