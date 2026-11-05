"""Review sheet: review.json per project, review.md export, summary, fixes plan and apply, calibration."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..config import Settings
from ..models import EditOp
from ..review import store
from ..review.store import REJECT_LABELS, REJECTS, VERDICT_LABELS, VERDICTS, ReviewError
from .app import get_settings
from .media import media_url
from .routes_projects import enqueue_job, project_root

router = APIRouter()


class ShotReviewBody(BaseModel):
    """Fields left out stay as they are. Comments have no length limit, on purpose."""
    verdict: Optional[str] = None
    comment: Optional[str] = None
    rejects: Optional[list[str]] = None
    clear_verdict: bool = False


class ProjectCommentBody(BaseModel):
    comment: str = ""


class FixesBody(BaseModel):
    ops: list[EditOp] = Field(min_length=1, max_length=500)


def _review_json(settings: Settings, project_id: str, review: store.Review) -> dict[str, Any]:
    root = project_root(settings, project_id)
    md = root / store.EXPORT_FILE
    return {**review.model_dump(mode="json"), "reviewed": review.reviewed, "total": review.total,
            "exists": store.exists(settings, project_id),
            "export_url": media_url(project_id, root, md) if md.exists() else None,
            "verdicts": [{"key": v, "label": VERDICT_LABELS[v]} for v in VERDICTS],
            "rejects": [{"key": r, "label": REJECT_LABELS[r]} for r in REJECTS],
            "threshold": float(settings.judge.threshold)}


def _shotlist(settings: Settings, project_id: str) -> Any:
    project_root(settings, project_id)
    sl = store.load_shotlist(settings, project_id)
    if sl is None:
        raise HTTPException(404, "this project has no shot list yet")
    return sl


@router.get("/projects/{project_id}/review")
def get_review(project_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """The review (an empty one when nothing was saved yet), one entry per shot in shot list order."""
    sl = _shotlist(settings, project_id)
    return _review_json(settings, project_id, store.load(settings, project_id, sl))


@router.put("/projects/{project_id}/review/shots/{shot_id}")
def put_shot_review(project_id: str, shot_id: str, body: ShotReviewBody,
                    settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    sl = _shotlist(settings, project_id)
    try:
        store._resolve(sl, shot_id)
    except ReviewError:
        raise HTTPException(404, f"shot not found: {shot_id}") from None
    kw: dict[str, Any] = {}
    if body.clear_verdict:
        kw["verdict"] = None
    elif body.verdict is not None:
        kw["verdict"] = body.verdict
    if body.comment is not None:
        kw["comment"] = body.comment
    if body.rejects is not None:
        kw["rejects"] = body.rejects
    try:
        review, entry = store.update_shot(settings, project_id, shot_id, **kw)
    except ReviewError as e:
        raise HTTPException(422, str(e)) from e
    return {"shot": entry.model_dump(mode="json"), "reviewed": review.reviewed, "total": review.total,
            "updated_at": review.updated_at}


@router.put("/projects/{project_id}/review/comment")
def put_project_comment(project_id: str, body: ProjectCommentBody,
                        settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    _shotlist(settings, project_id)
    review = store.set_project_comment(settings, project_id, body.comment)
    return {"project_comment": review.project_comment, "updated_at": review.updated_at,
            "reviewed": review.reviewed, "total": review.total}


@router.get("/projects/{project_id}/review/summary")
def review_summary(project_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    from ..review import export
    sl = _shotlist(settings, project_id)
    review = store.load(settings, project_id, sl)
    return {**export.summary(review), "text": export.summary_text(review)}


@router.post("/projects/{project_id}/review/export")
def review_export(project_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Write output/<id>/review.md and return its URL and text."""
    from ..review import export
    _shotlist(settings, project_id)
    path = export.write_export(settings, project_id)
    root = project_root(settings, project_id)
    return {"path": str(path), "url": media_url(project_id, root, path), "markdown": path.read_text(encoding="utf-8")}


@router.get("/projects/{project_id}/review/fixes")
def review_fixes(project_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """The planned EditOps (not applied), the wrong timing shots and what stays a comment."""
    from ..review import fixes
    sl = _shotlist(settings, project_id)
    try:
        return fixes.plan(settings, project_id, shotlist=sl)
    except ReviewError as e:
        raise HTTPException(422, str(e)) from e


@router.post("/projects/{project_id}/review/fixes")
def apply_review_fixes(project_id: str, body: FixesBody, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Enqueue the ops the operator saw (one apply_ops job) and mark those shots as fixed."""
    _shotlist(settings, project_id)
    dumped = [o.model_dump(mode="json") for o in body.ops]
    job_id = enqueue_job(settings, project_id, "apply_ops", {"ops": dumped})
    store.mark_fixed(settings, project_id, dumped, job_id)
    return {"job_id": job_id, "ops": dumped}


@router.get("/projects/{project_id}/review/calibration")
def review_calibration(project_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    from ..review import calibrate
    _shotlist(settings, project_id)
    c = calibrate.calibrate_project(settings, project_id)
    return {**c, "text": calibrate.calibration_text(c)}
