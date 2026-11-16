"""Rendering the shot list's layouts in the motion stage.

For every shot carrying a LayoutSpec (Shot.layout, written by the planner):
- the elements are timed from the narration words (timing.cue_times);
- a full layout (every non-panel kind, or presentation "full"/None) renders to an opaque clip for the whole
  shot and becomes the shot's selected candidate (source motion, so it is never credited to a third party);
- a panel presentation is placed from the shot's own picture (placement.py) and renders to an alpha
  overlay that starts when its first element is spoken (Shot.layout_at) and ends when its content stops
  being relevant (last element plus PANEL_HOLD_S), not at the cut. With no clear position the content is
  shown full instead, and the shot's notes say why.
Every decision goes into Shot.layout_notes. Clips are cached by a hash of everything that changes pixels.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ...models import (PANEL_LAYOUT_KINDS, Candidate, LayoutKind, Narration, Shot, ShotList, SourceKind,
                       StyleProfile)
from ...util import stable_hash
from ..house import house_style
from . import comp_class, render_full, render_panel
from .base import FrameProfile, LayoutJob
from .jobs import make_job
from .panel import DOCK_S, NEVER, PanelPlan, change_time, large_rect
from .placement import place_panel

LAYOUT_RENDER_VERSION = "layouts-1"
PANEL_HOLD_S = 2.5  # a panel stays this long after its last element is spoken, then leaves
PANEL_MIN_S = 3.0
FULL_MIN_S, FULL_MAX_S = 3.0, 7.0
SAMPLES = 4  # frames of the underlying picture read for placement


def frame_profile(profile: StyleProfile) -> FrameProfile:
    return FrameProfile(letterbox=profile.letterbox.enabled, captions=profile.captions.enabled,
                        ratio=profile.letterbox.ratio)


def presentation_of(shot: Shot) -> str:
    spec = shot.layout
    if spec is None or spec.kind not in PANEL_LAYOUT_KINDS:
        return "full"
    return spec.presentation or "full"


def _history(shots: list[Shot], upto: int, spec: Any) -> list[str]:
    """Earlier events of timelines with the same id, for a timeline that returns."""
    if spec.kind != LayoutKind.timeline or not spec.id:
        return []
    out: list[str] = []
    for s in shots[:upto]:
        lay = s.layout
        if lay is not None and lay.kind == LayoutKind.timeline and lay.id == spec.id:
            out += [e.text for e in lay.elements if e.role == "event" and e.text not in out]
    return out


def panel_share(spec: Any) -> tuple[float, float]:
    cls = comp_class(spec.kind)
    share = getattr(cls, "panel_share", None)
    return tuple(share(spec)) if callable(share) else tuple(cls.panel_size)  # type: ignore[return-value]


def _panel_window(job: LayoutJob, presentation: str, shot_dur: float) -> tuple[float, float]:
    """(start, end) of a panel inside the shot: first element (lead included) to last element + hold."""
    found = [c for c in job.cues if c.found]
    start = min((c.appear for c in found), default=0.0)
    end = max((c.word_end for c in found), default=start) + PANEL_HOLD_S
    if presentation in ("full_then_dock", "panel_then_expand"):
        end = max(end, start + 2 * DOCK_S + 4.0)
    end = min(shot_dur, max(end, start + PANEL_MIN_S))
    return max(0.0, start), end


def _shift(job: LayoutJob, by: float) -> None:
    for c in job.cues:
        c.appear, c.word, c.word_end = c.appear - by, c.word - by, c.word_end - by


# ---------------------------------------------------------------- underlying picture

def underlying_frames(shot: Shot, profile: StyleProfile, settings: Any, times: list[float],
                      width: int = 480) -> list[np.ndarray]:
    """RGB frames of the shot's own picture at times (seconds into the shot), as it moves on screen."""
    from ...render.cache import is_still, kenburns_for, resolve_media
    cand = shot.selected_candidate()
    if cand is None or cand.source == SourceKind.motion:
        return []
    path = resolve_media(cand.local_path, settings)
    if path is None or not path.is_file():
        return []
    r = settings.render
    h = round(width * r.height / r.width)
    if is_still(shot):
        import cv2

        from ...render.segment import _prepare_still, still_frames, visible_height
        kb = kenburns_for(shot, profile)
        img = _prepare_still(path, cand.crop, kb, settings, Path(settings.path("data")) / "_stills_probe",
                             visible_height(profile, r.width, r.height))
        n = max(2, round(shot.duration * r.fps))
        frames = []
        for t in times:
            i = min(n - 1, int(t * r.fps))
            gen = still_frames(img, kb, n, r.width, r.height)
            for _ in range(i):
                next(gen)
            raw = np.frombuffer(next(gen), np.uint8).reshape(r.height, r.width, 3)
            frames.append(cv2.resize(raw, (width, h), interpolation=cv2.INTER_AREA))
        return frames
    import subprocess

    from ...render.segment import footage_plan
    plan = footage_plan(cand, path, shot.duration, r.fps)
    out = []
    vf = f"scale={width}:{h}:force_original_aspect_ratio=increase,crop={width}:{h}"
    for t in times:
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{plan.start + t / plan.slow:.3f}", "-i", str(path),
                            "-frames:v", "1", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           capture_output=True, timeout=60)
        if len(p.stdout) == width * h * 3:
            out.append(np.frombuffer(p.stdout, np.uint8).reshape(h, width, 3))
    return out


# ---------------------------------------------------------------- stage

def _pool(shotlist: ShotList) -> dict[str, Candidate]:
    return {c.id: c for s in shotlist.shots for c in s.candidates}


def _key(job: LayoutJob, plan: Optional[PanelPlan]) -> str:
    return stable_hash(LAYOUT_RENDER_VERSION, job.spec, [(c.appear, c.word, c.word_end, c.stage) for c in job.cues],
                       {k: (v.path, v.in_point) for k, v in job.pictures.items()}, job.duration, job.width,
                       job.height, job.fps, vars(job.frame), vars(job.house) if hasattr(job.house, "__dict__")
                       else repr(job.house), job.extra, vars(plan) if plan else None)


def render_shot_layouts(shotlist: ShotList, ctx: Any) -> dict[str, float]:
    """Render every shot's layout; returns render seconds per kind (summed), for the stage report."""
    shots = [s for s in shotlist.shots if s.layout is not None]
    if not shots:
        return {}
    nar = ctx.maybe_load("narration", Narration)
    words = nar.words if nar else []
    frame = frame_profile(ctx.profile)
    house = house_style(getattr(ctx.brand, "house_style", None))
    r = ctx.settings.render
    pool = _pool(shotlist)
    out_dir = ctx.work / "mg"
    out_dir.mkdir(parents=True, exist_ok=True)
    seconds: dict[str, float] = {}
    previous = ""
    for idx, shot in enumerate(shotlist.shots):
        if shot.layout is None:
            continue
        spec = shot.layout
        shot.layout_notes = []
        extra = {"history": _history(shotlist.shots, idx, spec)} if spec.kind == LayoutKind.timeline else {}
        pres = presentation_of(shot)
        job = make_job(spec, words, shot.start, shot.duration, pool, frame, house, r.width, r.height, r.fps, extra)
        plan = None
        if pres != "full":
            plan, why = _plan_panel(shot, job, pres, previous, ctx)
            if plan is None:
                shot.layout_notes.append(f"panel not placed ({why}); shown full frame instead")
                pres = "full"
            else:
                previous = plan.position
                shot.layout_notes.append(f"panel {pres} at {why}")
        if pres == "full":
            job = make_job(spec, words, shot.start, shot.duration, pool, frame, house, r.width, r.height, r.fps,
                           extra)
            if not FULL_MIN_S <= shot.duration <= FULL_MAX_S:
                shot.layout_notes.append(f"full layout held {shot.duration:.1f} s, outside {FULL_MIN_S:g} to "
                                         f"{FULL_MAX_S:g} s")
        shot.layout_notes += job.notes
        key = _key(job, plan)
        out = out_dir / f"layout_{key}.{'mov' if plan else 'mp4'}"
        if not out.exists():
            secs = render_panel(job, plan, out) if plan else render_full(job, out)
            seconds[spec.kind.value] = seconds.get(spec.kind.value, 0.0) + secs
            ctx.log.info("layout rendered", shot=shot.id, kind=spec.kind.value, presentation=pres,
                         seconds=round(secs, 2), clip_s=round(job.duration, 2))
        if plan:
            shot.overlay_path = str(out)
            shot.layout_at = round(plan_start(shot, job), 3)
        else:
            shot.layout_at = 0.0
            cand = Candidate(id=f"{shot.id}_layout_1", source=SourceKind.motion, media="video", local_path=str(out),
                             licence="generated", width=r.width, height=r.height, in_point=0.0,
                             out_point=shot.duration, title=f"layout {spec.kind.value}")
            shot.candidates = [c for c in shot.candidates if c.id != cand.id] + [cand]
            shot.selected = cand.id
    return seconds


def plan_start(shot: Shot, job: LayoutJob) -> float:
    return float(job.extra.get("_start", 0.0))


def _plan_panel(shot: Shot, job: LayoutJob, pres: str, previous: str, ctx: Any) -> tuple[Optional[PanelPlan], str]:
    """Place the panel from the picture beneath; trims the job to the panel's visible window."""
    start, end = _panel_window(job, pres, shot.duration)
    _shift(job, start)
    job.duration = max(0.5, end - start)
    job.extra["_start"] = start
    W, H = job.width, job.height
    share = panel_share(job.spec)
    size = (share[0] * W, share[1] * H)
    change = change_time(job, pres)
    if pres == "full_then_dock":  # the panel rests only after the dock; read the picture over that time
        lo = min(job.duration, change + DOCK_S)
    else:
        lo = 0.0
    hi = job.duration if pres != "panel_then_expand" else min(job.duration, change if change < NEVER else job.duration)
    times = [start + lo + (hi - lo) * (k + 0.5) / SAMPLES for k in range(SAMPLES)]
    frames = underlying_frames(shot, ctx.profile, ctx.settings, times)
    visible = job.frame.visible(W, H)
    if not frames:
        from .placement import rect_for
        pos = "upper_right" if previous != "upper_right" else "upper_left"
        return PanelPlan(pres, rect_for(pos, size, visible, W), large_rect(job, pres == "panel_then_expand"),
                         change, pos), f"{pos}: no picture to read beneath, default corner"
    pl = place_panel(frames, size, visible, W, H, previous)
    if pl.position is None:
        return None, pl.reason
    return PanelPlan(pres, pl.rect, large_rect(job, pres == "panel_then_expand"), change, pl.position), pl.reason
