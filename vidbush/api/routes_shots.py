"""Shot list, thumbnails, project media, shot actions (EditOps), edit ops and chat revisions."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..config import Settings
from ..models import Candidate, EditOp, EditOpName, Shot, ShotList
from ..util import slug, stable_hash
from .app import get_settings
from .media import IMAGE_EXT, VIDEO_EXT, file_response, media_url, safe_join, thumbnail
from .routes_projects import enqueue_job, project_root

router = APIRouter()


class SwapBody(BaseModel):
    candidate_id: str


class RegenerateBody(BaseModel):
    prompt_delta: str = ""


class DurationBody(BaseModel):
    seconds: float = Field(gt=0.5, le=60.0)


class OpsBody(BaseModel):
    ops: list[EditOp] = Field(min_length=1, max_length=50)


class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    apply: bool = False


# ---------------------------------------------------------------- loading

def load_shotlist(root: Path) -> ShotList:
    """Working copy work/shotlist.json, else the published shotlist.json (paths rebased onto root)."""
    from ..jobs.paths import read_rebased, stored_root
    old = stored_root(root / "project.json")
    for p in (root / "work" / "shotlist.json", root / "shotlist.json"):
        if p.exists():
            return read_rebased(p, ShotList, root.name, root, [old] if old else [])
    raise HTTPException(404, "this project has no shot list yet")


def _shot(shotlist: ShotList, shot_id: str) -> Shot:
    try:
        return shotlist.by_id(shot_id)
    except KeyError:
        raise HTTPException(404, f"shot not found: {shot_id}") from None


def _candidate(shot: Shot, cand_id: str) -> Candidate:
    for c in shot.candidates:
        if c.id == cand_id:
            return c
    raise HTTPException(404, f"candidate not found: {cand_id}")


def _resolve(settings: Settings, root: Path, path: str) -> Optional[Path]:
    """A path from the shot list: absolute, or relative to the project or repo root."""
    if not path:
        return None
    p = Path(path)
    for cand in ([p] if p.is_absolute() else [root / p, Path(settings.repo_root) / p]):
        if cand.is_file():
            return cand
    return None


# ---------------------------------------------------------------- shot list

@router.get("/projects/{project_id}/shots")
def shots(project_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    root = project_root(settings, project_id)
    sl = load_shotlist(root)
    return {"project_id": project_id, "duration": sl.duration, "fps": sl.fps, "width": sl.width,
            "height": sl.height, "shots": [_shot_json(settings, project_id, root, s) for s in sl.shots]}


def _shot_json(settings: Settings, pid: str, root: Path, s: Shot) -> dict[str, Any]:
    base = f"/api/projects/{pid}/shots/{s.id}"
    sel = s.selected_candidate()
    bust = stable_hash(s.selected, s.render_hash, sel.keyframes if sel else [], s.render_segment_path)[:10]
    return {
        "index": s.index, "id": s.id, "type": s.type.value, "start": s.start, "end": s.end,
        "duration": s.duration, "intent": s.intent, "beat_id": s.beat_id, "scene_id": s.scene_id,
        "selected": s.selected, "judge_score": s.judge_score, "judge_notes": s.judge_notes,
        "attempts": s.attempts, "flags": s.flags, "ai_prompt": s.ai_prompt,
        "overlay": s.overlay.kind.value if s.overlay else None,
        "overlay_url": media_url(pid, root, s.overlay_path) if s.overlay_path else None,
        "motion": s.motion.kind.value if s.motion else None,
        "segment_url": media_url(pid, root, s.render_segment_path) if s.render_segment_path else None,
        "thumb_url": f"{base}/thumb?v={bust}",
        "candidates": [_cand_json(base, c, c.id == s.selected) for c in s.candidates],
    }


def _cand_json(base: str, c: Candidate, selected: bool) -> dict[str, Any]:
    cb = f"{base}/candidates/{c.id}"
    bust = stable_hash(c.keyframes, c.local_path)[:10]
    return {"id": c.id, "source": c.source.value, "media": c.media, "score": c.score,
            "judge_reason": c.judge_reason, "reject_reason": c.reject_reason, "licence": c.licence,
            "author": c.author, "attribution": c.attribution, "title": c.title, "url": c.url,
            "flags": c.flags, "selected": selected, "in_point": c.in_point, "out_point": c.out_point,
            "thumb_url": f"{cb}/thumb?v={bust}", "media_url": f"{cb}/media?v={bust}"}


# ---------------------------------------------------------------- thumbnails and media

def _candidate_still(settings: Settings, root: Path, c: Candidate) -> tuple[Optional[Path], Optional[float]]:
    """(source file, seek seconds) for a candidate preview frame: middle keyframe, else the media itself."""
    if c.keyframes:
        kf = _resolve(settings, root, c.keyframes[len(c.keyframes) // 2])
        if kf:
            return kf, None
    media = _resolve(settings, root, c.local_path)
    if media and c.media == "video":
        mid = c.in_point + max(c.out_point - c.in_point, 0.0) / 2
        return media, mid
    return media, None


@router.get("/projects/{project_id}/shots/{shot_id}/thumb")
def shot_thumb(project_id: str, shot_id: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    """JPEG of the selected candidate keyframe, else a frame of the rendered segment or final video."""
    root = project_root(settings, project_id)
    shot = _shot(load_shotlist(root), shot_id)
    cache = root / "work" / "thumbs"
    sel = shot.selected_candidate()
    src, at = _candidate_still(settings, root, sel) if sel else (None, None)
    if src is None:
        seg = _resolve(settings, root, shot.render_segment_path)
        src, at = (seg, shot.duration / 2) if seg else (None, None)
    if src is None and (root / "final.mp4").exists():
        src, at = root / "final.mp4", shot.start + shot.duration / 2
    if src is None:
        raise HTTPException(404, "no preview for this shot yet")
    return FileResponse(thumbnail(src, cache, at), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@router.get("/projects/{project_id}/shots/{shot_id}/candidates/{cand_id}/thumb")
def candidate_thumb(project_id: str, shot_id: str, cand_id: str,
                    settings: Settings = Depends(get_settings)) -> FileResponse:
    root = project_root(settings, project_id)
    cand = _candidate(_shot(load_shotlist(root), shot_id), cand_id)
    src, at = _candidate_still(settings, root, cand)
    if src is None:
        raise HTTPException(404, "candidate has no local media")
    return FileResponse(thumbnail(src, root / "work" / "thumbs", at), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@router.get("/projects/{project_id}/shots/{shot_id}/candidates/{cand_id}/media")
def candidate_media(project_id: str, shot_id: str, cand_id: str,
                    settings: Settings = Depends(get_settings)) -> FileResponse:
    """The candidate's own file (video or image), with range support. Only paths named in the shot list."""
    root = project_root(settings, project_id)
    cand = _candidate(_shot(load_shotlist(root), shot_id), cand_id)
    src = _resolve(settings, root, cand.local_path)
    if src is None or src.suffix.lower() not in IMAGE_EXT | VIDEO_EXT:
        raise HTTPException(404, "candidate has no local media")
    return file_response(src)


@router.get("/projects/{project_id}/media/{path:path}")
def project_media(project_id: str, path: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    """Any file under output/<id>/, guarded against traversal, with HTTP range support."""
    root = project_root(settings, project_id)
    return file_response(safe_join(root, path), no_cache=path.endswith(".json"))


# ---------------------------------------------------------------- shot actions

def _enqueue_ops(settings: Settings, project_id: str, ops: list[EditOp]) -> dict[str, Any]:
    dumped = [o.model_dump(mode="json") for o in ops]
    return {"job_id": enqueue_job(settings, project_id, "apply_ops", {"ops": dumped}), "ops": dumped}


def _shot_op(settings: Settings, project_id: str, shot_id: str, op: EditOpName, reason: str,
             **args: Any) -> dict[str, Any]:
    root = project_root(settings, project_id)
    shot = _shot(load_shotlist(root), shot_id)
    if "candidate_id" in args:
        _candidate(shot, args["candidate_id"])
    if op == EditOpName.remove_overlay and not (shot.overlay or shot.overlay_path):
        raise HTTPException(422, "this shot has no overlay")
    return _enqueue_ops(settings, project_id, [EditOp(op=op, args={"shot_id": shot_id, **args}, reason=reason)])


@router.post("/projects/{project_id}/shots/{shot_id}/swap")
def swap(project_id: str, shot_id: str, body: SwapBody, settings: Settings = Depends(get_settings)) -> dict:
    return _shot_op(settings, project_id, shot_id, EditOpName.swap_shot, "operator picked a candidate",
                    candidate_id=body.candidate_id)


@router.post("/projects/{project_id}/shots/{shot_id}/upload")
async def upload(project_id: str, shot_id: str, file: UploadFile = File(...),
                 settings: Settings = Depends(get_settings)) -> dict:
    """Save the operator's file to inputs/uploads/ and enqueue an upload_shot op."""
    root = project_root(settings, project_id)
    _shot(load_shotlist(root), shot_id)
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in IMAGE_EXT | VIDEO_EXT:
        raise HTTPException(422, f"unsupported file type {ext or '(none)'}: upload an image or a video")
    dest = root / "inputs" / "uploads" / f"{slug(shot_id)}_{uuid.uuid4().hex[:6]}_{slug(Path(name).stem)}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    return _shot_op(settings, project_id, shot_id, EditOpName.upload_shot, "operator uploaded a file",
                    path=str(dest))


@router.post("/projects/{project_id}/shots/{shot_id}/regenerate")
def regenerate(project_id: str, shot_id: str, body: RegenerateBody,
               settings: Settings = Depends(get_settings)) -> dict:
    return _shot_op(settings, project_id, shot_id, EditOpName.regenerate_ai, "operator asked for an AI image",
                    prompt_delta=body.prompt_delta.strip())


@router.post("/projects/{project_id}/shots/{shot_id}/duration")
def duration(project_id: str, shot_id: str, body: DurationBody,
             settings: Settings = Depends(get_settings)) -> dict:
    return _shot_op(settings, project_id, shot_id, EditOpName.set_duration, "operator set the duration",
                    seconds=round(body.seconds, 3))


@router.post("/projects/{project_id}/shots/{shot_id}/remove_overlay")
def remove_overlay(project_id: str, shot_id: str, settings: Settings = Depends(get_settings)) -> dict:
    return _shot_op(settings, project_id, shot_id, EditOpName.remove_overlay, "operator removed the overlay")


@router.post("/projects/{project_id}/ops")
def ops(project_id: str, body: OpsBody, settings: Settings = Depends(get_settings)) -> dict:
    load_shotlist(project_root(settings, project_id))
    return _enqueue_ops(settings, project_id, body.ops)


@router.post("/projects/{project_id}/chat")
def chat(project_id: str, body: ChatBody, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Map a plain-language revision to EditOps; enqueue them when apply is true."""
    load_shotlist(project_root(settings, project_id))
    try:
        from ..editor.revise import map_message
    except ImportError as e:
        raise HTTPException(503, f"chat revisions are not available: {e}") from e
    mapped = [EditOp.model_validate(o) for o in map_message(settings, project_id, body.message)]
    out: dict[str, Any] = {"ops": [o.model_dump(mode="json") for o in mapped], "job_id": None}
    if body.apply and mapped:
        out["job_id"] = _enqueue_ops(settings, project_id, mapped)["job_id"]
    return out
