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
from typing import Callable, Iterable, Optional

from ..config import Settings
from ..models import (STAGES, Brief, InputMode, Project, ProjectPaths, StageRecord, StageStatus,
                      VideoFormat)
from ..util import read_model, write_json
from .context import StageContext, build_context

StageFn = Callable[[StageContext], list[str]]


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
    if fresh and root.exists():
        shutil.rmtree(root)
    paths = make_paths(root)
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    sp = _copy_input(script_path, inputs)
    vp = _copy_input(voiceover_path, inputs)
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
    return read_model(project_dir(settings, project_id) / "project.json", Project)


def stage_applies(project: Project, stage: str) -> bool:
    if stage in ("research", "outline"):
        return project.input_mode == InputMode.brief
    return True


def _outputs_exist(rec: StageRecord) -> bool:
    return all(Path(p).exists() for p in rec.outputs)


def run_project(settings: Settings, project_id: str, stages: Optional[dict[str, StageFn]] = None,
                force: Iterable[str] = (), until: Optional[str] = None, echo: bool = True,
                on_progress: Optional[Callable[[Project], None]] = None) -> Project:
    """Run every pending stage in order. Done stages with outputs on disk are skipped."""
    from .stages import STAGE_FUNCS
    funcs = stages or STAGE_FUNCS
    project = load_project(settings, project_id)
    ctx = build_context(settings, project, echo=echo)
    force = set(force)
    project.status = "running"
    ctx.save_project()
    for name in STAGES:
        rec = project.stages.setdefault(name, StageRecord(name=name))
        if not stage_applies(project, name):
            rec.status = StageStatus.skipped
            continue
        if rec.status == StageStatus.done and name not in force and _outputs_exist(rec):
            ctx.log.info("stage skipped (done)", stage=name)
            continue
        _run_stage(ctx, name, funcs[name], rec, on_progress)
        if until and name == until:
            break
    if not until or until == STAGES[-1]:
        project.status = "done"
        project.current_stage = None
    ctx.save_project()
    if on_progress:
        on_progress(project)
    return project


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
    project = load_project(settings, project_id)
    hit = False
    for name in STAGES:
        if name == stage:
            hit = True
        if hit and project.stages.get(name) and project.stages[name].status != StageStatus.skipped:
            project.stages[name].status = StageStatus.pending
    write_json(Path(project.paths.root) / "project.json", project)
    return project
