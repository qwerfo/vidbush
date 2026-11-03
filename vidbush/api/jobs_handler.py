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

JOB_KINDS = ("run", "resume", "apply_ops")


def handle_job(settings: Settings, job: Job) -> None:
    """Run one job. Raises on failure so the pool marks the job failed with the traceback."""
    payload: dict[str, Any] = json.loads(job.payload or "{}")
    if job.kind in ("run", "resume"):
        _run(settings, job.project_id, payload)
    elif job.kind == "apply_ops":
        _apply_ops(settings, job, payload)
    else:
        raise ValueError(f"unknown job kind: {job.kind}")


def _run(settings: Settings, project_id: str, payload: dict[str, Any]) -> None:
    from ..jobs import pipeline

    def progress(project: Project) -> None:
        upsert_project_row(settings, project)

    pipeline.run_project(settings, project_id, echo=False, on_progress=progress,
                         force=payload.get("force") or ())


def _apply_ops(settings: Settings, job: Job, payload: dict[str, Any]) -> None:
    """Apply EditOps and re-render. The EditResult goes to work/last_edit.json."""
    from ..editor.ops import apply_and_render
    from ..jobs import pipeline

    ops = [EditOp.model_validate(o) for o in payload.get("ops", [])]
    out = pipeline.project_dir(settings, job.project_id) / "work" / "last_edit.json"
    record: dict[str, Any] = {"job_id": job.id, "started_at": time.time(),
                              "ops": [o.model_dump(mode="json") for o in ops]}
    try:
        result = apply_and_render(settings, job.project_id, ops, echo=False)
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
