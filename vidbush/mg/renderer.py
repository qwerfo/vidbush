"""Motion graphics stage entry points.

render_spec(spec, out_path, alpha, settings) -> Path dispatches on
settings.render.renderer: "python" (built in, default) or "remotion" (the
node project in renderer/, falling back to python on any failure).

render_all_motion(shotlist, ctx) -> (ShotList, list[Path]) renders every
motion_graphic shot's spec to work/mg/<key>.mp4 and every overlay to
work/mg/<key>.mov (alpha), reusing files that already exist, in a process pool
of settings.render.workers. key = mg_key(spec, settings) hashes the spec, the
configured renderer and MG_RENDER_VERSION (bump it when drawing code changes
output). A degraded render (remotion fell back to python, or a text card after a
renderer error) is written to <key>.fallback.<ext> instead, so the next motion
run retries the real render; an existing fallback is reused when it fails again.
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
MG_RENDER_VERSION = "mg-4"  # mg-4: labelled major rivers only, state borders, road and rail context lines


def renderer_kind(settings: Any) -> str:
    return str(getattr(settings.render, "renderer", "python") or "python")


def mg_key(spec: MotionSpec, settings: Any) -> str:
    """Cache key of a rendered clip: spec (size and fps included), renderer kind and renderer version."""
    return stable_hash(spec, renderer_kind(settings), MG_RENDER_VERSION)


def fallback_path(out: Path) -> Path:
    """Where a degraded render of out is kept: <key>.fallback.<ext>."""
    return out.with_name(f"{out.stem}.fallback{out.suffix}")


# ---------------------------------------------------------------- single spec

def render_spec(spec: MotionSpec, out_path: Path, alpha: bool, settings: Any,
                notes: Optional[list[str]] = None, degraded_path: Optional[Path] = None) -> Path:
    """Render spec with the configured renderer and return the written path.

    Remotion falls back to python on failure (the reason is appended to `notes`
    when given, else logged); that python render goes to degraded_path when given.
    """
    out_path = Path(out_path)
    from .inset import is_inset
    if renderer_kind(settings) == "remotion" and not is_inset(spec):  # insets are a python composition
        try:
            return render_remotion(spec, out_path, alpha, settings)
        except Exception as e:  # noqa: BLE001  any remotion failure falls back
            msg = f"remotion render failed, using python renderer: {str(e)[-300:]}"
            if notes is not None:
                notes.append(msg)
            else:
                logging.getLogger("vidbush.mg").warning(msg)
            out_path = Path(degraded_path) if degraded_path else out_path
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

def _render_real(spec: MotionSpec, out: Path, alpha: bool, settings: Any) -> Path:
    """The configured renderer only, with no fallback."""
    if renderer_kind(settings) == "remotion":
        return render_remotion(spec, out, alpha, settings)
    from ..render.mg_python import render_spec_python
    return render_spec_python(spec, out, alpha, settings)


def _job(spec_json: str, out: str, alpha: bool, settings_json: str) -> tuple[str, Optional[str]]:
    """Process pool worker: render one spec. Returns (path written, warning or None).

    Degraded output goes to fallback_path(out). When a fallback already exists only
    the real render is retried; on another failure the old fallback is kept as is,
    so its bytes (and the segment cache keyed on them) stay stable.
    """
    from ..config import Settings
    from ..render.mg_python import fallback_spec, render_spec_python
    settings = Settings.model_validate_json(settings_json)
    spec = MotionSpec.model_validate_json(spec_json)
    target, degraded = Path(out), fallback_path(Path(out))
    if degraded.exists():
        try:
            _render_real(spec, target, alpha, settings)
        except Exception as e:  # noqa: BLE001  keep the previous fallback
            return str(degraded), f"motion render still failing ({type(e).__name__}: {str(e)[-200:]}), reusing fallback"
        degraded.unlink(missing_ok=True)
        return str(target), None
    notes: list[str] = []
    try:
        written = render_spec(spec, target, alpha, settings, notes, degraded_path=degraded)
    except Exception as e:  # noqa: BLE001  a broken spec still renders as text
        title = spec.payload.get("title", "") if isinstance(spec.payload, dict) else ""
        render_spec_python(fallback_spec(spec, str(title)), degraded, alpha, settings)
        return str(degraded), f"motion render failed ({type(e).__name__}: {str(e)[-200:]}), rendered as text overlay"
    return str(written), notes[0] if notes else None


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
            jobs.append((shot, spec, mg_dir / f"{mg_key(spec, ctx.settings)}.mp4", False))
        if shot.overlay is not None and shot.type in OVERLAY_SHOT_TYPES:
            spec, warn = prepare_spec(shot.overlay, shot.duration, shotlist, shot.intent, True)
            if warn:
                ctx.log.warn(warn, shot=shot.id)
            jobs.append((shot, spec, mg_dir / f"{mg_key(spec, ctx.settings)}.mov", True))
        elif shot.overlay_path:
            shot.overlay_path = ""
    return jobs


def _run_jobs(todo: list[tuple[MotionSpec, Path, bool]], ctx: Any) -> dict[Path, Path]:
    """Render todo; returns {planned path: path actually written} (a fallback path when degraded)."""
    settings_json = ctx.settings.model_dump_json()
    args = [(s.model_dump_json(), str(p), a, settings_json) for s, p, a in todo]
    workers = max(1, int(getattr(ctx.settings.render, "workers", 1) or 1))
    if workers == 1 or len(args) == 1:
        results = [_job(*a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(args)), mp_context=get_context("spawn")) as pool:
            results = list(pool.map(_job, *zip(*args)))
    written: dict[Path, Path] = {}
    for (_, planned, _), (out, warn) in zip(todo, results):
        written[planned] = Path(out)
        if warn:
            ctx.log.warn(warn, path=Path(out).name)
    return written


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
    written = _run_jobs(todo, ctx) if todo else {}
    ctx.log.info("motion graphics", rendered=len(todo), cached=len(seen) - len(todo),
                 degraded=sum(1 for k, v in written.items() if k != v))
    outs: list[Path] = []
    for shot, spec, planned, alpha in jobs:
        out = written.get(planned, planned)
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
