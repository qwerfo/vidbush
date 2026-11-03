"""Motion graphics stage entry points.

render_spec(spec, out_path, alpha, settings) -> Path dispatches on
settings.render.renderer: "python" (built in, default) or "remotion" (the
node project in renderer/, falling back to python on any failure).

render_all_motion(shotlist, ctx) -> (ShotList, list[Path]) renders every
motion_graphic shot's spec to work/mg/<stable_hash(spec)>.mp4 and every
overlay to work/mg/<hash>.mov (alpha), reusing files that already exist,
in a process pool of settings.render.workers.
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from ..models import Candidate, MotionSpec, ShotList, ShotType, SourceKind
from ..util import ToolError, run, stable_hash, write_json

OVERLAY_SHOT_TYPES = {ShotType.footage_overlay, ShotType.image_overlay, ShotType.ai_image, ShotType.ai_video}
REMOTION_DIR = "renderer"


# ---------------------------------------------------------------- single spec

def render_spec(spec: MotionSpec, out_path: Path, alpha: bool, settings: Any,
                notes: Optional[list[str]] = None) -> Path:
    """Render spec to out_path with the configured renderer; remotion falls back to python on failure
    (the reason is appended to `notes` when given, else logged)."""
    out_path = Path(out_path)
    if getattr(settings.render, "renderer", "python") == "remotion":
        try:
            return render_remotion(spec, out_path, alpha, settings)
        except Exception as e:  # noqa: BLE001  any remotion failure falls back
            msg = f"remotion render failed, using python renderer: {str(e)[-300:]}"
            if notes is not None:
                notes.append(msg)
            else:
                logging.getLogger("vidbush.mg").warning(msg)
    from ..render.mg_python import render_spec_python
    return render_spec_python(spec, out_path, alpha, settings)


def render_remotion(spec: MotionSpec, out_path: Path, alpha: bool, settings: Any) -> Path:
    """Render through renderer/render.mjs (needs node, node_modules and a Chromium)."""
    root = settings.path(REMOTION_DIR)
    if not (root / "node_modules").exists():
        raise ToolError("renderer/node_modules missing (run npm install in renderer/)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spec_file = out_path.with_suffix(".spec.json")
    write_json(spec_file, spec)
    try:
        cmd = ["node", str(root / "render.mjs"), str(spec_file), str(out_path)] + (["--alpha"] if alpha else [])
        run(cmd, timeout=max(120.0, spec.duration * 30))
    finally:
        spec_file.unlink(missing_ok=True)
    if not out_path.exists():
        raise ToolError("remotion produced no output")
    return out_path


# ---------------------------------------------------------------- stage

def _job(spec_json: str, out: str, alpha: bool, settings_json: str) -> tuple[str, Optional[str]]:
    """Process pool worker: render one spec. Returns (path, warning or None)."""
    from ..config import Settings
    from ..render.mg_python import fallback_spec
    settings = Settings.model_validate_json(settings_json)
    spec = MotionSpec.model_validate_json(spec_json)
    warn = None
    notes: list[str] = []
    try:
        render_spec(spec, Path(out), alpha, settings, notes)
    except Exception as e:  # noqa: BLE001  a broken spec still renders as text
        warn = f"motion render failed ({type(e).__name__}: {str(e)[-200:]}), rendered as text overlay"
        title = spec.payload.get("title", "") if isinstance(spec.payload, dict) else ""
        from ..render.mg_python import render_spec_python
        render_spec_python(fallback_spec(spec, str(title)), Path(out), alpha, settings)
    return out, warn or (notes[0] if notes else None)


def prepare_spec(spec: MotionSpec, duration: float, shotlist: ShotList, intent: str,
                 overlay: bool) -> tuple[MotionSpec, Optional[str]]:
    """Spec retimed to the shot, sized to the shot list; invalid payloads become a text_overlay of the intent."""
    from ..render.mg_python import fallback_spec
    from .builders import retime_spec
    warn = None
    try:
        spec.typed_payload()
        spec = retime_spec(spec, duration)
    except (ValidationError, TypeError, ValueError) as e:
        warn = f"invalid {spec.kind.value} payload, using text overlay: {str(e)[:160]}"
        spec = fallback_spec(spec, intent)
    upd: dict[str, Any] = {"duration": round(duration, 3), "width": shotlist.width, "height": shotlist.height,
                           "fps": shotlist.fps}
    if overlay and spec.background == "solid":
        upd["background"] = "transparent"
    return spec.model_copy(update=upd), warn


def _plan(shotlist: ShotList, ctx: Any, mg_dir: Path) -> list[tuple[Any, MotionSpec, Path, bool]]:
    """(shot, spec, out path, alpha) for every render this shot list needs; clears stale overlay paths."""
    jobs = []
    for shot in shotlist.shots:
        if shot.type == ShotType.motion_graphic and shot.motion is not None:
            spec, warn = prepare_spec(shot.motion, shot.duration, shotlist, shot.intent, False)
            if warn:
                ctx.log.warn(warn, shot=shot.id)
            jobs.append((shot, spec, mg_dir / f"{stable_hash(spec)}.mp4", False))
        if shot.overlay is not None and shot.type in OVERLAY_SHOT_TYPES:
            spec, warn = prepare_spec(shot.overlay, shot.duration, shotlist, shot.intent, True)
            if warn:
                ctx.log.warn(warn, shot=shot.id)
            jobs.append((shot, spec, mg_dir / f"{stable_hash(spec)}.mov", True))
        elif shot.overlay_path:
            shot.overlay_path = ""
    return jobs


def _run_jobs(todo: list[tuple[MotionSpec, Path, bool]], ctx: Any) -> None:
    settings_json = ctx.settings.model_dump_json()
    args = [(s.model_dump_json(), str(p), a, settings_json) for s, p, a in todo]
    workers = max(1, int(getattr(ctx.settings.render, "workers", 1) or 1))
    if workers == 1 or len(args) == 1:
        results = [_job(*a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(args)), mp_context=get_context("spawn")) as pool:
            results = list(pool.map(_job, *zip(*args)))
    for out, warn in results:
        if warn:
            ctx.log.warn(warn, path=Path(out).name)


def render_all_motion(shotlist: ShotList, ctx: Any) -> tuple[ShotList, list[Path]]:
    """Render motion graphic shots and overlays (cached by spec hash) and wire them into the shot list."""
    mg_dir = ctx.work / "mg"
    mg_dir.mkdir(parents=True, exist_ok=True)
    jobs = _plan(shotlist, ctx, mg_dir)
    seen: set[Path] = set()
    todo = []
    for _, spec, out, alpha in jobs:
        if out not in seen and not out.exists():
            todo.append((spec, out, alpha))
        seen.add(out)
    if todo:
        _run_jobs(todo, ctx)
    ctx.log.info("motion graphics", rendered=len(todo), cached=len(seen) - len(todo))
    outs: list[Path] = []
    for shot, spec, out, alpha in jobs:
        outs.append(out)
        if alpha:
            shot.overlay_path = str(out)
            continue
        key = mid_frame(out, spec)
        cand = Candidate(id=f"{shot.id}_motion_1", source=SourceKind.motion, media="video", local_path=str(out),
                         licence="generated", keyframes=[str(key)], score=None, width=spec.width,
                         height=spec.height, in_point=0.0, out_point=spec.duration, title=spec.kind.value)
        shot.candidates = [c for c in shot.candidates if c.source != SourceKind.motion] + [cand]
        shot.selected = cand.id
    return shotlist, outs


def mid_frame(video: Path, spec: MotionSpec) -> Path:
    """PNG of the middle frame next to the clip (extracted once)."""
    png = video.with_name(video.stem + "_mid.png")
    if not png.exists():
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{spec.duration / 2:.3f}", "-i", str(video), "-frames:v", "1",
             str(png)], timeout=60)
    return png
