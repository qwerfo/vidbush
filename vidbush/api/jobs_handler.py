"""Worker pool handler: turns a Job row into a pipeline run or an edit render."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..config import Settings
from ..jobs.runner import Job, upsert_project_row
from ..models import EditOp, Project
from ..util import write_json

JOB_KINDS = ("run", "resume", "apply_ops", "batch")
BATCH_JOB_PREFIX = "batch:"  # Job.project_id of a batch job is "batch:<batch_id>"


def handle_job(settings: Settings, job: Job) -> None:
    """Run one job. Raises on failure so the pool marks the job failed with the traceback."""
    payload: dict[str, Any] = json.loads(job.payload or "{}")
    if job.kind in ("run", "resume"):
        _run(settings, job.project_id, payload)
    elif job.kind == "apply_ops":
        _apply_ops(settings, job, payload)
    elif job.kind == "batch":
        from ..jobs.batch import run_batch
        run_batch(settings, job.project_id.removeprefix(BATCH_JOB_PREFIX), echo=False)
    else:
        raise ValueError(f"unknown job kind: {job.kind}")


def _run(settings: Settings, project_id: str, payload: dict[str, Any]) -> None:
    run_pipeline(settings, project_id, payload)


def run_pipeline(settings: Settings, project_id: str, payload: dict[str, Any], echo: bool = False) -> Project:
    """The worker's pipeline run (DB row kept in sync); batches call it too. payload: force, cost_cap_usd."""
    from ..jobs import pipeline

    def progress(project: Project) -> None:
        upsert_project_row(settings, project)

    return pipeline.run_project(settings, project_id, echo=echo, on_progress=progress,
                                force=payload.get("force") or (), cost_cap=payload.get("cost_cap_usd"))


def _apply_ops(settings: Settings, job: Job, payload: dict[str, Any]) -> None:
    """Apply EditOps and re-render. The EditResult goes to work/last_edit.json.

    Idempotent on worker recovery: apply_and_render skips the ops when edit_plan.json already
    records this job id, and only renders.
    """
    from ..editor.ops import apply_and_render
    from ..jobs import pipeline

    ops = [EditOp.model_validate(o) for o in payload.get("ops", [])]
    out = pipeline.project_dir(settings, job.project_id) / "work" / "last_edit.json"
    record: dict[str, Any] = {"job_id": job.id, "started_at": time.time(),
                              "ops": [o.model_dump(mode="json") for o in ops]}
    try:
        result = apply_and_render(settings, job.project_id, ops, echo=False, job_id=int(job.id))
    except Exception as e:
        record.update(ok=False, error=f"{type(e).__name__}: {e}"[:2000], finished_at=time.time())
        write_json(out, record)
        _sync_row(settings, job.project_id)
        raise
    record.update(ok=not result.errors, result=result.model_dump(mode="json"), finished_at=time.time())
    write_json(out, record)
    _sync_row(settings, job.project_id)


def _sync_row(settings: Settings, project_id: str) -> None:
    from ..jobs import pipeline
    if (pipeline.project_dir(settings, project_id) / "project.json").exists():
        upsert_project_row(settings, pipeline.load_project(settings, project_id))


def last_edit_path(settings: Settings, project_id: str) -> Path:
    from ..jobs import pipeline
    return pipeline.project_dir(settings, project_id) / "work" / "last_edit.json"
