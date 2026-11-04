"""Stage graph runner with resume.

A stage is skipped when its record says done and its declared outputs exist.
Every stage writes to disk, so a crash at any point resumes from the first
stage that is not done.
"""
from __future__ import annotations

import random
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from ..config import Settings
from ..models import (STAGES, Brief, InputMode, Project, ProjectPaths, StageRecord, StageStatus,
                      VideoFormat)
from ..util import read_json, write_json
from . import costcap, lock
from .context import StageContext, build_context
from .paths import read_rebased, stored_root

StageFn = Callable[[StageContext], list[str]]

# stage -> stages whose outputs it reads; a stage that reruns forces every stage downstream of it
STAGE_INPUTS: dict[str, tuple[str, ...]] = {
    "intake": (),
    "research": ("intake",),
    "outline": ("research",),
    "script": ("intake", "outline"),
    "narration": ("script",),
    "shotlist": ("script", "narration", "research"),
    "sourcing": ("shotlist",),
    "judge": ("sourcing",),
    "visuals": ("judge",),
    "motion": ("visuals",),
    "mix": ("script", "narration", "shotlist"),
    "assembly": ("motion", "mix"),
    "packaging": ("assembly",),
}


class InjectedFailure(RuntimeError):
    """Raised when VIDBUSH_FAIL_AT names the current stage (resume test hook)."""


def project_dir(settings: Settings, project_id: str) -> Path:
    return settings.path(settings.output_folder) / project_id


def make_paths(root: Path) -> ProjectPaths:
    paths = ProjectPaths(root=str(root), work=str(root / "work"), sources=str(root / "sources"),
                         segments=str(root / "work" / "segments"), thumbnails=str(root / "thumbnails"))
    for p in (paths.root, paths.work, paths.sources, paths.segments, paths.thumbnails):
        Path(p).mkdir(parents=True, exist_ok=True)
    return paths


def create_project(settings: Settings, project_id: str, mode: InputMode, brand_id: str = "default",
                   brief: Optional[Brief] = None, script_path: Optional[str] = None,
                   voiceover_path: Optional[str] = None, fmt: Optional[VideoFormat] = None,
                   target_minutes: Optional[float] = None, style_profile_id: Optional[str] = None,
                   fresh: bool = True, seed: Optional[int] = None) -> Project:
    """Create output/<id>/ and project.json. fresh=True wipes an existing project of that id."""
    root = project_dir(settings, project_id)
    lock.check_free(root, project_id)
    if fresh and root.exists():
        shutil.rmtree(root)
    paths = make_paths(root)
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    sp = _copy_input(script_path, inputs)
    vp = _copy_input(voiceover_path, inputs)
    if brief is not None and (fmt or target_minutes):  # CLI or form overrides beat the brief front matter
        brief = brief.model_copy(update={"format": fmt or brief.format,
                                         "target_minutes": target_minutes or brief.target_minutes})
    fmt = fmt or (brief.format if brief else VideoFormat.documentary)
    proj = Project(id=project_id, brand_id=brand_id, input_mode=mode, format=fmt,
                   target_minutes=target_minutes or (brief.target_minutes if brief else 8.0),
                   brief=brief, script_path=sp, voiceover_path=vp, paths=paths,
                   style_profile_id=style_profile_id,
                   seed=seed if seed is not None else random.randint(1, 10_000_000),
                   stages={s: StageRecord(name=s) for s in STAGES})
    proj.title = brief.topic if brief else Path(sp or vp or project_id).stem
    write_json(root / "project.json", proj)
    return proj


def _copy_input(path: Optional[str], inputs: Path) -> Optional[str]:
    if not path:
        return None
    src = Path(path)
    dst = inputs / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return str(dst)


def load_project(settings: Settings, project_id: str) -> Project:
    """project.json with every stored path rebased onto the current project root (moved projects)."""
    root = project_dir(settings, project_id)
    pj = root / "project.json"
    old = stored_root(pj)
    project = read_rebased(pj, Project, project_id, root, [old] if old else [])
    fresh = make_paths(root) if root.exists() else project.paths
    project.paths = fresh
    return project


def stage_applies(project: Project, stage: str) -> bool:
    if stage in ("research", "outline"):
        return project.input_mode == InputMode.brief
    return True


def _outputs_exist(rec: StageRecord) -> bool:
    return all(Path(p).exists() for p in rec.outputs)


def run_project(settings: Settings, project_id: str, stages: Optional[dict[str, StageFn]] = None,
                force: Iterable[str] = (), until: Optional[str] = None, echo: bool = True,
                on_progress: Optional[Callable[[Project], None]] = None,
                cost_cap: Optional[float] = None) -> Project:
    """Run every pending stage in order under the project lock. Done stages with outputs on disk are
    skipped; a stage that runs forces every stage downstream of it (STAGE_INPUTS).

    cost_cap (USD, None = no cap): checked after each stage against the project's cost total; when
    crossed the run stops at that stage boundary and the project is marked needs_review (costcap.py)."""
    root = project_dir(settings, project_id)
    with lock.project_lock(root, project_id, owner=f"run_project {project_id}"):
        return _run_locked(settings, project_id, stages, set(force), until, echo, on_progress, cost_cap)


def _run_locked(settings: Settings, project_id: str, stages: Optional[dict[str, StageFn]], force: set[str],
                until: Optional[str], echo: bool, on_progress: Optional[Callable[[Project], None]],
                cost_cap: Optional[float] = None) -> Project:
    from .stages import STAGE_FUNCS
    funcs = stages or STAGE_FUNCS
    project = load_project(settings, project_id)
    ctx = build_context(settings, project, echo=echo)
    project.status = "running"
    costcap.reset(ctx)
    ctx.save_project()
    capped = False
    for name in STAGES:
        rec = project.stages.setdefault(name, StageRecord(name=name))
        if not stage_applies(project, name):
            rec.status = StageStatus.skipped
            continue
        if rec.status == StageStatus.done and name not in force and _outputs_exist(rec):
            ctx.log.info("stage skipped (done)", stage=name)
            continue
        lock.touch(ctx.root)
        _run_stage(ctx, name, funcs[name], rec, on_progress)
        force |= downstream(name)
        if cost_cap is not None and costcap.reached(ctx, name, cost_cap, force):
            capped = True
            break
        if until and name == until:
            break
    if capped:
        project.current_stage = None
    elif not until or until == STAGES[-1]:
        project.status = final_status(ctx)
        project.current_stage = None
    ctx.save_project()
    if on_progress:
        on_progress(project)
    return project


def downstream(stage: str) -> set[str]:
    """Every stage that reads, directly or not, the outputs of stage."""
    out: set[str] = set()
    for name in STAGES[STAGES.index(stage) + 1:]:
        if any(dep == stage or dep in out for dep in STAGE_INPUTS.get(name, ())):
            out.add(name)
    return out


def final_status(ctx: StageContext) -> str:
    """done, or needs_review when real seats exist but QA reports that some role used mock, or that some
    writer section or director chunk shipped from heuristics although a real provider was configured."""
    qa = ctx.out_path("qa_report.json")
    if ctx.settings.offline or not qa.exists():
        return "done"
    try:
        data = read_json(qa)
        mock_used = data.get("mock_used") or []
        heuristic = data.get("heuristic_fallbacks") or []
    except (OSError, ValueError, AttributeError):
        return "done"
    if heuristic:
        return "needs_review"
    real = any(v.get("real") for v in ctx.seats.status().values())
    return "needs_review" if mock_used and real else "done"


def _run_stage(ctx: StageContext, name: str, fn: StageFn, rec: StageRecord,
               on_progress: Optional[Callable[[Project], None]]) -> None:
    project = ctx.project
    project.current_stage = name
    rec.status, rec.started_at, rec.error = StageStatus.running, time.time(), None
    rec.attempts += 1
    ctx.stage = name
    ctx.save_project()
    if on_progress:
        on_progress(project)
    ctx.log.info("stage start", stage=name)
    try:
        if ctx.settings.fail_at_stage == name:
            raise InjectedFailure(f"injected failure at stage {name} (VIDBUSH_FAIL_AT)")
        rec.outputs = [str(p) for p in fn(ctx)]
    except Exception as e:
        rec.status, rec.error, rec.finished_at = StageStatus.failed, f"{type(e).__name__}: {e}", time.time()
        project.status = "failed"
        ctx.log.error("stage failed", stage=name, error=str(e)[:2000], trace=traceback.format_exc()[-4000:])
        ctx.save_project()
        if on_progress:
            on_progress(project)
        raise
    rec.status, rec.finished_at = StageStatus.done, time.time()
    ctx.log.info("stage done", stage=name, seconds=round(rec.finished_at - (rec.started_at or 0), 2))
    ctx.save_project()


def invalidate_from(settings: Settings, project_id: str, stage: str) -> Project:
    """Mark a stage and everything after it pending (used by edits)."""
    return mark_pending(settings, project_id, STAGES[STAGES.index(stage):])


def mark_pending(settings: Settings, project_id: str, stages: Iterable[str]) -> Project:
    """Mark the given stages and every stage downstream of them pending in project.json.

    Edits call this so a render that fails part way, followed by resume, still reruns them.
    """
    project = load_project(settings, project_id)
    names: set[str] = set()
    for st in stages:
        names |= {st} | downstream(st)
    for name in names:
        rec = project.stages.get(name)
        if rec and rec.status != StageStatus.skipped:
            rec.status = StageStatus.pending
    write_json(Path(project.paths.root) / "project.json", project)
    return project


def read_work_model(settings: Settings, project_id: str, name: str, model: type) -> Any:
    """Read output/<id>/work/<name> with stored project paths rebased onto the current root."""
    root = project_dir(settings, project_id)
    old = stored_root(root / "project.json")
    return read_rebased(root / "work" / name, model, project_id, root, [old] if old else [])
