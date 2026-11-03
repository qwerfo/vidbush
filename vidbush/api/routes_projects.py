"""Projects: list, create (brief, script or voiceover), detail, log tail, jobs, resume."""
from __future__ import annotations

import json
import shutil
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlmodel import Session, select

from ..config import Settings, list_ids
from ..jobs import runner
from ..models import STAGES, Brief, InputMode, Project, VideoFormat
from ..util import read_json, read_model, slug
from .app import get_settings
from .media import check_id, media_url

router = APIRouter()

SCRIPT_EXT = {".md", ".txt"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
ACTIVE = ("queued", "running")


# ---------------------------------------------------------------- helpers shared with routes_shots

def project_root(settings: Settings, project_id: str) -> Path:
    """output/<id>/ for an existing project, else 404."""
    from ..jobs.pipeline import project_dir
    root = project_dir(settings, check_id(project_id, "project id"))
    if not (root / "project.json").exists():
        raise HTTPException(404, f"project not found: {project_id}")
    return root


def load(settings: Settings, project_id: str) -> Project:
    return read_model(project_root(settings, project_id) / "project.json", Project)


def job_dict(job: runner.Job) -> dict[str, Any]:
    d = job.model_dump()
    d["payload"] = json.loads(job.payload or "{}")
    return d


def active_job(settings: Settings, project_id: str, kinds: tuple[str, ...] = ()) -> Optional[runner.Job]:
    for job in runner.list_jobs(settings, project_id, limit=20):
        if job.status in ACTIVE and (not kinds or job.kind in kinds):
            return job
    return None


def enqueue_job(settings: Settings, project_id: str, kind: str, payload: Optional[dict[str, Any]] = None) -> int:
    from .jobs_handler import JOB_KINDS
    if kind not in JOB_KINDS:
        raise ValueError(f"unknown job kind: {kind}")
    return runner.enqueue(settings, project_id, kind, payload)


# ---------------------------------------------------------------- list and detail

@router.get("/projects")
def list_projects(settings: Settings = Depends(get_settings)) -> list[dict[str, Any]]:
    """Every project: DB rows merged with projects found on disk (CLI runs). project.json wins."""
    from ..jobs.pipeline import project_dir
    with Session(runner.engine_for(settings)) as s:
        rows = {r.id: r.model_dump() for r in s.exec(select(runner.ProjectRow))}
    out_dir = settings.path(settings.output_folder)
    ids = set(rows) | ({p.parent.name for p in out_dir.glob("*/project.json")} if out_dir.exists() else set())
    out = []
    for pid in ids:
        row = rows.get(pid, {"id": pid})
        pj = project_dir(settings, pid) / "project.json"
        if pj.exists():
            try:
                p = read_model(pj, Project)
            except ValueError:
                continue
            row.update(title=p.title or pid, status=p.status, current_stage=p.current_stage or "",
                       input_mode=p.input_mode.value, format=p.format.value, created_at=p.created_at,
                       brand_id=p.brand_id, target_minutes=p.target_minutes,
                       stages_done=sum(1 for r in p.stages.values() if r.status.value in ("done", "skipped")),
                       has_final=(pj.parent / "final.mp4").exists())
        job = active_job(settings, pid)
        row["active_job"] = job_dict(job) if job else None
        out.append(row)
    return sorted(out, key=lambda r: r.get("created_at") or 0, reverse=True)


@router.get("/projects/{project_id}")
def get_project(project_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """project.json with stage records, plus output URLs (cache busted) and job state."""
    root = project_root(settings, project_id)
    data = read_json(root / "project.json")
    data["stage_order"] = STAGES
    data["final_url"] = media_url(project_id, root, root / "final.mp4") if (root / "final.mp4").exists() else None
    data["outputs"] = _outputs(project_id, root)
    job = active_job(settings, project_id)
    data["active_job"] = job_dict(job) if job else None
    last = root / "work" / "last_edit.json"
    data["last_edit"] = read_json(last) if last.exists() else None
    return data


def _outputs(project_id: str, root: Path) -> dict[str, Any]:
    names = ["captions.srt", "captions.ass", "chapters.txt", "description.md", "qa_report.md",
             "qa_report.json", "cost_report.json", "shotlist.json"]
    files = {n: media_url(project_id, root, root / n) for n in names if (root / n).exists()}
    thumbs = sorted((root / "thumbnails").glob("thumb_*.png"))
    files["thumbnails"] = [media_url(project_id, root, t) for t in thumbs]
    stats = root / "work" / "render_stats.json"
    files["render_stats"] = read_json(stats) if stats.exists() else None
    return files


@router.get("/projects/{project_id}/log")
def project_log(project_id: str, n: int = Query(200, ge=1, le=5000),
                settings: Settings = Depends(get_settings)) -> list[dict[str, Any]]:
    """Last n records of data/logs/<id>.jsonl."""
    check_id(project_id, "project id")
    path = settings.path(settings.data_folder) / "logs" / f"{project_id}.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", errors="replace") as f:
        lines = deque(f, maxlen=n)
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


@router.get("/projects/{project_id}/jobs")
def project_jobs(project_id: str, settings: Settings = Depends(get_settings)) -> list[dict[str, Any]]:
    check_id(project_id, "project id")
    return [job_dict(j) for j in runner.list_jobs(settings, project_id)]


@router.get("/projects/{project_id}/jobs/{job_id}")
def project_job(project_id: str, job_id: int, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    check_id(project_id, "project id")
    with Session(runner.engine_for(settings)) as s:
        job = s.get(runner.Job, job_id)
    if not job or job.project_id != project_id:
        raise HTTPException(404, "job not found")
    return job_dict(job)


@router.post("/projects/{project_id}/resume")
def resume(project_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    load(settings, project_id)
    job = active_job(settings, project_id, ("run", "resume"))
    if job:
        raise HTTPException(409, f"project already has a {job.kind} job {job.status} (job {job.id})")
    return {"job_id": enqueue_job(settings, project_id, "resume")}


# ---------------------------------------------------------------- create

@router.post("/projects")
async def create(mode: str = Form(...), topic: str = Form(""), target_minutes: Optional[float] = Form(None),
                 format: Optional[str] = Form(None), audience: str = Form(""), angle: str = Form(""),
                 brand: str = Form("default"), style_profile: str = Form(""), script_text: str = Form(""),
                 project_id: str = Form(""), script: Optional[UploadFile] = File(None),
                 voiceover: Optional[UploadFile] = File(None),
                 settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Create a project from a multipart form and enqueue its run job."""
    from ..jobs.pipeline import create_project
    try:
        input_mode = InputMode(mode)
        fmt = VideoFormat(format) if format else None
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    minutes = _minutes(target_minutes)
    _check_brand(settings, brand, style_profile)
    pid = _new_id(settings, project_id, topic, script, voiceover)
    staging = settings.path(settings.data_folder) / "uploads" / uuid.uuid4().hex[:12]
    try:
        kw = await _inputs(input_mode, staging, pid, topic, minutes, fmt, audience, angle, script_text,
                           script, voiceover)
        proj = create_project(settings, pid, input_mode, brand_id=brand, fmt=fmt, target_minutes=minutes,
                              style_profile_id=style_profile or None, **kw)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    runner.upsert_project_row(settings, proj)
    job_id = enqueue_job(settings, pid, "run")
    return {"project": proj.model_dump(mode="json"), "job_id": job_id}


def _minutes(value: Optional[float]) -> Optional[float]:
    if value is not None and not 1.0 <= value <= 90.0:
        raise HTTPException(422, "target_minutes must be between 1 and 90")
    return value


def _check_brand(settings: Settings, brand: str, style_profile: str) -> None:
    if check_id(brand, "brand") not in list_ids(settings, "brands"):
        raise HTTPException(422, f"unknown brand: {brand}")
    if style_profile and style_profile not in list_ids(settings, "style_profiles"):
        raise HTTPException(422, f"unknown style profile: {style_profile}")


def _new_id(settings: Settings, wanted: str, topic: str, script: Optional[UploadFile],
            voiceover: Optional[UploadFile]) -> str:
    """Use the requested id (refusing to overwrite) or derive a fresh one from the input."""
    from ..jobs.pipeline import project_dir
    if wanted:
        check_id(wanted, "project id")
        if (project_dir(settings, wanted) / "project.json").exists():
            raise HTTPException(409, f"project {wanted} already exists")
        return wanted
    files = [u.filename for u in (voiceover, script) if u and u.filename]
    base = topic.strip() or (Path(files[0]).stem if files else "project")
    return f"{slug(base, 30)}_{uuid.uuid4().hex[:6]}"


async def _inputs(mode: InputMode, staging: Path, pid: str, topic: str, minutes: Optional[float],
                  fmt: Optional[VideoFormat], audience: str, angle: str, script_text: str,
                  script: Optional[UploadFile], voiceover: Optional[UploadFile]) -> dict[str, Any]:
    """Validate the inputs for a mode and stage uploaded files. Returns create_project kwargs."""
    if mode == InputMode.brief:
        if not topic.strip():
            raise HTTPException(422, "a brief needs a topic")
        return {"brief": Brief(topic=topic.strip(), target_minutes=minutes or 8.0,
                               format=fmt or VideoFormat.documentary, audience=audience, angle=angle)}
    script_path = await _stage_script(staging, pid, script_text, script)
    if mode == InputMode.script:
        if not script_path:
            raise HTTPException(422, "script mode needs script_text or a script file")
        return {"script_path": script_path}
    if not voiceover or not voiceover.filename:
        raise HTTPException(422, "voiceover mode needs an audio file")
    return {"voiceover_path": await _save_upload(voiceover, staging, AUDIO_EXT), "script_path": script_path}


async def _stage_script(staging: Path, pid: str, text: str, upload: Optional[UploadFile]) -> Optional[str]:
    if upload and upload.filename:
        return await _save_upload(upload, staging, SCRIPT_EXT)
    if text.strip():
        staging.mkdir(parents=True, exist_ok=True)
        path = staging / f"{pid}.md"
        path.write_text(text, encoding="utf-8")
        return str(path)
    return None


async def _save_upload(upload: UploadFile, folder: Path, allowed: set[str]) -> str:
    """Save an upload under a sanitised name that keeps its stem (the stem is the fixture key)."""
    name = Path(upload.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in allowed:
        raise HTTPException(422, f"unsupported file type {ext or '(none)'}; use {', '.join(sorted(allowed))}")
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{slug(Path(name).stem, 60)}{ext}"
    with dest.open("wb") as f:
        while chunk := await upload.read(1 << 20):
            f.write(chunk)
    return str(dest)
