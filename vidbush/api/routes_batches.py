"""Batches: list, detail with report rows, create from a server side folder or list file, pause, resume.

A batch runs as one worker job (kind "batch", Job.project_id "batch:<id>"), which runs its projects one
after another (vidbush/jobs/batch.py), so the queue survives a server restart like any other job.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..config import Settings, list_ids
from ..jobs import batch as batches
from ..jobs.batch_report import REPORT_MD, report_json, summary
from .app import get_settings
from .jobs_handler import BATCH_JOB_PREFIX
from .media import check_id

router = APIRouter()


class BatchCreate(BaseModel):
    """POST /api/batches body. source is a folder or list file path on the server."""

    source: str
    brand: str = "default"
    profile: Optional[str] = None
    format: Optional[str] = None
    minutes: Optional[float] = None
    cost_cap_usd: Optional[float] = None  # None = config batch.cost_cap_usd, 0 = no cap
    id_prefix: str = ""
    batch_id: Optional[str] = None
    start: bool = True


class BatchResume(BaseModel):
    retry_failed: bool = False
    cost_cap_usd: Optional[float] = None


def _job_key(batch_id: str) -> str:
    return f"{BATCH_JOB_PREFIX}{batch_id}"


def _active_job(settings: Settings, batch_id: str) -> Any:
    from .routes_projects import active_job
    return active_job(settings, _job_key(batch_id), ("batch",))


def _enqueue(settings: Settings, batch_id: str) -> int:
    from .routes_projects import enqueue_job
    job = _active_job(settings, batch_id)
    return int(job.id) if job else enqueue_job(settings, _job_key(batch_id), "batch")


def _load(settings: Settings, batch_id: str) -> batches.BatchState:
    check_id(batch_id, "batch id")
    try:
        return batches.load_batch(settings, batch_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, f"batch not found: {batch_id}") from e


def _detail(settings: Settings, state: batches.BatchState) -> dict[str, Any]:
    data = report_json(state)
    data["status"] = batches.effective_status(settings, state)
    job = _active_job(settings, state.id)
    data.update(created_at=state.created_at, profile=state.profile, format=state.format, minutes=state.minutes,
                active_job=job.id if job else None,
                report_path=str(batches.batch_dir(settings, state.id) / REPORT_MD))
    if job and data["status"] == "interrupted":
        data["status"] = "queued"  # a worker job will pick it up
    return data


@router.get("/batches")
def list_all(settings: Settings = Depends(get_settings)) -> list[dict[str, Any]]:
    """Every batch, newest first, with counts and totals."""
    return [{"id": b.id, "status": batches.effective_status(settings, b), "paused": b.paused, "brand": b.brand,
             "source": b.source, "created_at": b.created_at, "items": len(b.items), **summary(b)}
            for b in batches.list_batches(settings)]


@router.get("/batches/{batch_id}")
def detail(batch_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """The batch_report rows (status, duration, cost per minute, QA percent) plus batch state."""
    return _detail(settings, _load(settings, batch_id))


@router.post("/batches")
def create(body: BatchCreate, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Plan a batch from a folder or list file on the server and (start=true) enqueue its worker job."""
    if check_id(body.brand, "brand") not in list_ids(settings, "brands"):
        raise HTTPException(422, f"unknown brand: {body.brand}")
    if body.format and body.format not in ("documentary", "top10"):
        raise HTTPException(422, "format must be documentary or top10")
    if body.batch_id:
        check_id(body.batch_id, "batch id")
    try:
        state = batches.create_batch(settings, body.source, brand=body.brand, profile=body.profile,
                                     fmt=body.format, minutes=body.minutes, cost_cap=body.cost_cap_usd,
                                     id_prefix=body.id_prefix, batch_id=body.batch_id)
    except FileNotFoundError as e:
        raise HTTPException(422, str(e)) from e
    except FileExistsError as e:
        raise HTTPException(409, str(e)) from e
    job_id = _enqueue(settings, state.id) if body.start else None
    return {**_detail(settings, state), "job_id": job_id}


@router.post("/batches/{batch_id}/pause")
def pause(batch_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Pause between projects: the running project finishes, the next one does not start."""
    _load(settings, batch_id)
    return _detail(settings, batches.pause_batch(settings, batch_id))


@router.post("/batches/{batch_id}/resume")
def resume(batch_id: str, body: Optional[BatchResume] = None,
           settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Unpause and enqueue the worker job (unless a runner already has the batch)."""
    _load(settings, batch_id)
    body = body or BatchResume()
    state = batches.prepare_resume(settings, batch_id, body.retry_failed, body.cost_cap_usd)
    job_id = None
    if not batches.is_running(settings, batch_id) and any(i.status in batches.UNFINISHED for i in state.items):
        job_id = _enqueue(settings, batch_id)
    return {**_detail(settings, state), "job_id": job_id}
