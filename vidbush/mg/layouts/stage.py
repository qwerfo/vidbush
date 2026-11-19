"""Rendering the shot list's layouts in the motion stage.

For every shot carrying a LayoutSpec (Shot.layout, written by the planner):
- the elements are timed from the narration words (timing.cue_times);
- a layout is shown whole or not at all: every anchored element must be spoken inside the shot, every
  picture, place and route it names must be real and drawable (job.missing), and it must stay at least
  3 s from its first word. Otherwise it is dropped (a render decision: the spec stays) and the shot shows a
  real picture of its own, the planner's fitting alternate before its loose one;
- a full layout (every non-panel kind, or presentation "full"/None) whose first words come within
  FULL_LATE_S of the cut renders to an opaque clip for the whole shot and becomes the shot's selected
  candidate (source motion, so it is never credited to a third party); one whose words come later is an
  alpha overlay entering on its first words over the shot's own picture, so no bare ground is ever held;
- a panel presentation is placed from the shot's own picture (placement.py) and renders to an alpha
  overlay that starts when its first element is spoken (Shot.layout_at) and ends when its content stops
  being relevant (last element plus PANEL_HOLD_S), not at the cut. With no clear position it is dropped.
Every decision goes into Shot.layout_notes. Clips are cached by a hash of everything that changes pixels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from ...models import (GENERATED_SOURCES, PANEL_LAYOUT_KINDS, Candidate, LayoutKind, Narration, Shot, ShotList,
                       SourceKind, StyleProfile)
from ...util import stable_hash
from ..house import house_style
from . import comp_class, render_full, render_panel
from .base import FrameProfile, LayoutJob
from .jobs import make_job
from .panel import DOCK_S, NEVER, PanelPlan, change_time, large_rect

LAYOUT_RENDER_VERSION = "layouts-9"  # bump when drawing code changes pixels
PANEL_HOLD_S = 2.5  # a panel stays this long after its last element is spoken, then leaves
PANEL_MIN_S = 3.0  # a panel is on screen at least this long from its first word, else it is not shown
FULL_MIN_S, FULL_MAX_S = 3.0, 7.0  # a full layout lasts at least FULL_MIN_S from its first word to the cut
FULL_LATE_S = 0.75  # first words later than this after the cut: the layout enters on them, over the picture
DOC_MAX_S = 14.0  # a document read in two or three marked passages needs longer than a card
SAMPLES = 4  # frames of the underlying picture read for placement
PANEL_SHRINK = (0.85, 0.72)  # smaller sizes tried when no position clears the picture at the preferred size


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


def panel_share(spec: Any, extra: Optional[dict] = None) -> tuple[float, float]:
    cls = comp_class(spec.kind)
    share = getattr(cls, "panel_share", None)
    if not callable(share):
        return tuple(cls.panel_size)  # type: ignore[return-value]
    try:
        return tuple(share(spec, extra))  # type: ignore[return-value]
    except TypeError:  # kinds whose size does not depend on the job's extra
        return tuple(share(spec))  # type: ignore[return-value]


def first_word(job: LayoutJob) -> float:
    """Seconds into the clip of the layout's first spoken word (0 when nothing is anchored)."""
    return max(0.0, min((c.word for c in job.cues if c.found), default=0.0))


def _panel_window(job: LayoutJob, presentation: str, shot_dur: float) -> Optional[tuple[float, float]]:
    """(start, end) of a panel inside the shot: first element (lead included) to last element + hold, and at
    least PANEL_MIN_S from the first word; None when the cut comes sooner (never clipped short)."""
    found = [c for c in job.cues if c.found]
    start = max(0.0, min((c.appear for c in found), default=0.0))
    first = first_word(job)
    if shot_dur - first < PANEL_MIN_S - 1e-6:
        return None
    end = max((c.word_end for c in found), default=start) + PANEL_HOLD_S
    if presentation in ("full_then_dock", "panel_then_expand"):
        end = max(end, start + 2 * DOCK_S + 4.0)
    return start, min(shot_dur, max(end, first + PANEL_MIN_S))


def incomplete(job: LayoutJob, shot_dur: float) -> str:
    """Why the layout cannot be shown whole in this shot ('' when it can): a picture or element it cannot draw
    (job.missing), or an anchored element spoken outside the shot."""
    if job.missing:
        return job.missing[0]
    for c in job.cues:
        el = job.spec.elements[c.index]
        if el.anchor and c.found and not -1.0 <= c.word < shot_dur:
            return f"element {c.index} ({el.role}): its words are spoken outside the shot"
    return ""


def real_picture(shot: Shot) -> Optional[Candidate]:
    """The shot's selected picture when it is a real one (not generated, not motion), else None."""
    c = shot.selected_candidate()
    if c is None or c.source == SourceKind.motion or c.source in GENERATED_SOURCES or not c.local_path:
        return None
    return c


def keep_real_picture(shot: Shot, ctx: Any) -> bool:
    """Make sure the shot shows a real picture of its own: its selection, else its best real candidate (the
    planner's fitting alternate before its loose one). False, with a warning, when it has none."""
    if real_picture(shot) is not None:
        return True
    from ..renderer import picture_instead
    pick = picture_instead(shot, real=True)
    if pick is None:
        ctx.log.warn("shot has no real picture to show", shot=shot.id)
        return False
    shot.selected = pick.id
    shot.layout_notes.append(f"shows {pick.id}")
    return True


def _drop(shot: Shot, spec: Any, why: str, ctx: Any) -> None:
    """The layout is not shown: a render decision only (the spec stays, so a rerun with other timings or
    pictures can show it). The shot shows a real picture of its own."""
    shot.layout_notes.append(f"{spec.kind.value} not shown: {why}; the shot's picture stays")
    keep_real_picture(shot, ctx)


def _shift(job: LayoutJob, by: float) -> None:
    if "clip_start" in job.extra:
        job.extra["clip_start"] += by
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

def legacy_panel(shot: Shot) -> Optional[Any]:
    """A planned map or chart inset (old MotionSpec overlay) as a panel LayoutSpec: every marker or bar is an
    element anchored on its own name, so it appears when named instead of arriving complete in a fixed box."""
    from ...models import LayoutElement, LayoutSpec, MotionKind
    from ..inset import is_inset
    ov = shot.overlay
    if shot.layout is not None or ov is None or not is_inset(ov):
        return None
    try:
        pay = ov.typed_payload()
    except Exception:  # noqa: BLE001  an invalid payload stays on the old path
        return None
    els: list[Any] = []
    if ov.kind == MotionKind.map:
        names: list[str] = []
        for m in pay.markers:
            if m.label and m.label not in names:
                names.append(m.label)
        for r in pay.routes:
            for n in (r.from_place, r.to_place):
                if n and n not in names:
                    names.append(n)
        els = [LayoutElement(role="place", text=n, anchor=n) for n in names]
        for r in pay.routes:
            if r.from_place and r.to_place:
                via = f" by {r.via}" if r.via else ""
                els.append(LayoutElement(role="route", text=f"{r.from_place} to {r.to_place}{via}",
                                         anchor=r.to_place))
        kind = LayoutKind.route_map
    else:
        vals = pay.series[0].values if pay.series else []
        role = "point" if ov.kind == MotionKind.line_chart else "bar"
        els = [LayoutElement(role=role, text=f"{lab}: {v:g}{pay.units}", anchor=lab)
               for lab, v in zip(pay.labels, vals)]
        kind = LayoutKind.chart
    if not els:
        return None
    return LayoutSpec(kind=kind, elements=els, title=getattr(pay, "title", ""), presentation="panel")


def _pool(shotlist: ShotList) -> dict[str, Candidate]:
    return {c.id: c for s in shotlist.shots for c in s.candidates}


def _key(job: LayoutJob, plan: Optional[PanelPlan]) -> str:
    return stable_hash(LAYOUT_RENDER_VERSION, job.spec, [(c.appear, c.word, c.word_end, c.stage) for c in job.cues],
                       {k: (v.path, v.in_point) for k, v in job.pictures.items()}, job.duration, job.width,
                       job.height, job.fps, vars(job.frame), vars(job.house) if hasattr(job.house, "__dict__")
                       else repr(job.house), job.extra, vars(plan) if plan else None)


def render_shot_layouts(shotlist: ShotList, ctx: Any) -> dict[str, float]:
    """Render every shot's layout; returns render seconds per kind (summed), for the stage report."""
    for s in shotlist.shots:
        conv = legacy_panel(s)
        if conv is not None:
            s.layout, s.overlay, s.overlay_path = conv, None, ""
            s.layout_notes = ["legacy map or chart inset converted to a word-timed, placed panel"]
    shots = [s for s in shotlist.shots if s.layout is not None]
    if not shots:
        return {}
    nar = ctx.maybe_load("narration", Narration)
    if nar is None or not nar.words:  # no word timings: render nothing (specs are kept), shots show pictures
        ctx.log.warn("layouts skipped: no narration word timings")
        for s in shots:
            keep_real_picture(s, ctx)
        return {}
    words = nar.words
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
        _reset(shot)
        extra: dict[str, Any] = {}
        if spec.kind == LayoutKind.timeline:
            extra["history"] = _history(shotlist.shots, idx, spec)
        if spec.kind == LayoutKind.route_map:  # the art lane's registered painted bases for this project
            art = ctx.work / "art" / "maps"
            if art.is_dir():
                extra["art_bases"] = [str(art)]
                # which paintings exist (and when they changed) is part of the cache key
                extra["art_fp"] = sorted((f.name, int(f.stat().st_mtime)) for f in art.glob("*_base.*"))
        pres = presentation_of(shot)
        job = make_job(spec, words, shot.start, shot.duration, pool, frame, house, r.width, r.height, r.fps, extra)
        why = incomplete(job, shot.duration)
        if why:  # a graphic missing a part, or at the wrong moment, is worse than the shot's own picture
            _drop(shot, spec, why, ctx)
            continue
        first = first_word(job)
        plan, overlay = None, False
        if pres != "full":
            if shot.selected_candidate() is None and not keep_real_picture(shot, ctx):
                _drop(shot, spec, "a panel needs the shot's own picture beneath it", ctx)
                continue
            plan, why = _plan_panel(shot, job, pres, previous, ctx)
            if plan is None:
                _drop(shot, spec, f"panel not placed ({why})", ctx)
                continue
            previous = plan.position
            shot.layout_notes.append(f"panel {pres} at {why}")
        else:
            if shot.duration - first < FULL_MIN_S:
                _drop(shot, spec, f"{shot.duration - first:.1f} s from its first words to the cut, under "
                                  f"{FULL_MIN_S:g} s", ctx)
                continue
            if first > FULL_LATE_S:  # never seconds of bare ground: the picture shows until the words
                if not keep_real_picture(shot, ctx):
                    _drop(shot, spec, f"its first words come {first:.1f} s after the cut and the shot has no "
                                      "real picture to show until then", ctx)
                    continue
                start = min(c.appear for c in job.cues if c.found)
                _shift(job, start)
                job.duration = shot.duration - start
                job.extra["_start"] = start
                overlay = True
                shot.layout_notes.append(f"full layout enters on its first words, {start:.1f} s after the cut, "
                                         "over the shot's picture")
            top = DOC_MAX_S if spec.kind == LayoutKind.document_focus else FULL_MAX_S
            if shot.duration > top:
                shot.layout_notes.append(f"full layout held {shot.duration:.1f} s, over {top:g} s")
        key = _key(job, plan)
        out = out_dir / f"layout_{key}.{'mov' if plan or overlay else 'mp4'}"
        if not out.exists():
            secs = render_panel(job, plan, out) if plan else render_full(job, out, overlay)
            seconds[spec.kind.value] = seconds.get(spec.kind.value, 0.0) + secs
            if out.exists():
                ctx.log.info("layout rendered", shot=shot.id, kind=spec.kind.value, presentation=pres,
                             seconds=round(secs, 2), clip_s=round(job.duration, 2))
        else:  # cached clip: build the composition once so its notes (base used, fallbacks) are still recorded
            comp = comp_class(spec.kind)(job, job.width, job.height, plan is not None)
            comp.close()
        shot.layout_notes += list(dict.fromkeys(job.notes))
        if job.missing or not out.exists():  # the composition found a part it cannot draw: nothing was written
            _drop(shot, spec, job.missing[0] if job.missing else "no clip was written", ctx)
            continue
        if plan or overlay:
            shot.overlay_path = str(out)
            shot.layout_at = round(plan_start(shot, job), 3)
        else:
            shot.layout_at = 0.0
            cand = Candidate(id=f"{shot.id}_layout_1", source=SourceKind.motion, media="video", local_path=str(out),
                             licence="generated", width=r.width, height=r.height, in_point=0.0,
                             out_point=shot.duration, title=f"layout {spec.kind.value}")
            cand.judge_reason = f"replaces {shot.selected}" if shot.selected else ""
            shot.candidates = [c for c in shot.candidates if c.id != cand.id] + [cand]
            shot.selected = cand.id
    return seconds


def _reset(shot: Shot) -> None:
    """Undo an earlier render of this shot's layout, so a rerun starts from the shot's own picture."""
    shot.layout_notes = []
    shot.overlay_path, shot.layout_at = "", 0.0
    lay = next((c for c in shot.candidates if c.id == f"{shot.id}_layout_1"), None)
    if lay is not None:
        prev = lay.judge_reason[len("replaces "):] if lay.judge_reason.startswith("replaces ") else None
        shot.candidates = [c for c in shot.candidates if c.id != lay.id]
        if shot.selected == lay.id:
            shot.selected = prev if any(c.id == prev for c in shot.candidates) else None


def plan_start(shot: Shot, job: LayoutJob) -> float:
    return float(job.extra.get("_start", 0.0))


def _framed_rect(shot: Shot, ctx: Any, W: int, H: int) -> Optional[tuple[float, float, float, float]]:
    """0..1 box of a narrow still framed on the house ground (segment.framed_image geometry), else None."""
    from ...render.segment import MAX_STILL_UPSCALE, visible_height
    from ...render.stillmotion import is_narrow
    c = shot.selected_candidate()
    if c is None or c.media != "image" or not c.width or not c.height:
        return None
    vis = visible_height(ctx.profile, W, H)
    if not is_narrow(c.width, c.height, W, vis):
        return None
    s = min(0.8 * vis / c.height, 0.9 * W / c.width, MAX_STILL_UPSCALE)
    fw, fh = c.width * s / W, c.height * s / H
    m = 0.03  # the frame, its shadow and the slow push
    return 0.5 - fw / 2 - m, 0.5 - fh / 2 - m, 0.5 + fw / 2 + m, 0.5 + fh / 2 + m


def _fits(job: LayoutJob, size: tuple[float, float]) -> bool:
    """The content laid out at this panel size keeps every text whole at the floors (kinds that report it)."""
    missing = list(job.missing)
    comp = comp_class(job.spec.kind)(job, int(size[0]) // 2 * 2, int(size[1]) // 2 * 2, True)
    try:
        check = getattr(comp, "fits", None)
        return bool(check()) if callable(check) else bool(getattr(comp, "ok", True))
    finally:
        comp.close()
        job.missing[:] = missing  # a trial size never decides whether the layout is whole


def _plan_panel(shot: Shot, job: LayoutJob, pres: str, previous: str, ctx: Any) -> tuple[Optional[PanelPlan], str]:
    """Place the panel from the picture beneath; trims the job to the panel's visible window. None with the
    reason when it has under PANEL_MIN_S before the cut or no clear position."""
    win = _panel_window(job, pres, shot.duration)
    if win is None:
        return None, f"under {PANEL_MIN_S:g} s from its first words to the cut"
    start, end = win
    _shift(job, start)
    job.duration = max(0.5, end - start)
    job.extra["_start"] = start
    W, H = job.width, job.height
    # a document's panel size comes from its page (OCR line heights), so the sizing sees the pictures too
    share = panel_share(job.spec, dict(job.extra, pictures={i: r.path for i, r in job.pictures.items()}))
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
    from .placement import analyse, choose
    samples = [analyse(f) for f in frames]
    framed = _framed_rect(shot, ctx, W, H)
    if framed is not None:  # a framed narrow still: the whole framed picture is the subject
        for a in samples:
            a.protect.append(framed)
    pl = choose(samples, size, visible, W, H, previous)
    for k in PANEL_SHRINK:  # a smaller panel (content relaid out at the type floors) before giving up
        if pl.position is not None:
            break
        small = (size[0] * k, size[1] * k)
        if not _fits(job, small):
            break  # smaller would cut or shrink text under the floors
        pl = choose(samples, small, visible, W, H, previous)
        if pl.position is not None:
            pl.reason += f"; panel at {k:.0%} of its preferred size to clear the subject"
    if pl.position is None:
        return None, pl.reason
    return PanelPlan(pres, pl.rect, large_rect(job, pres == "panel_then_expand"), change, pl.position), pl.reason
