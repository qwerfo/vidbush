"""Segment cache key (see the cache rule in docs/ARCHITECTURE.md).

segment_hash(shot, profile, caption_lines_in_window, settings) hashes every
input that changes a segment's pixels, so editing one shot changes only that
shot's hash. The segment lives at work/segments/<hash>.mp4 and is reused
when that file exists. Bump RENDER_VERSION when rendering code changes output.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable, Optional

from ..config import Settings
from ..models import Shot, SourceKind, StyleProfile
from ..util import file_fingerprint, stable_hash

RENDER_VERSION = "seg-4"
KB_KEYS = ("zoom0", "zoom1", "x0", "y0", "x1", "y1")


def resolve_media(path: str, settings: Settings) -> Path:
    """Absolute path for a media reference (relative paths are against the repo root)."""
    return settings.path(path) if path else Path("")


def frame_range(shot: Shot, fps: int) -> tuple[int, int]:
    """Frames [start, end) from cumulative rounding of the shot's start and end times."""
    return int(round(shot.start * fps)), int(round(shot.end * fps))


def caption_dict(line: Any) -> dict[str, Any]:
    """{text, start, end} from a CaptionLine object or a plain dict."""
    if isinstance(line, dict):
        return {"text": str(line.get("text", "")), "start": float(line.get("start", 0.0)),
                "end": float(line.get("end", 0.0))}
    return {"text": str(line.text), "start": float(line.start), "end": float(line.end)}


def lines_in_window(lines: Iterable[Any], t0: float, t1: float) -> list[Any]:
    """Lines overlapping [t0, t1] (same rule as captions.write_ass_window)."""
    out = []
    for ln in lines:
        d = caption_dict(ln)
        if d["end"] <= t0 + 1e-3 or d["start"] >= t1 - 1e-3:
            continue
        out.append(ln)
    return out


def _fallback_kenburns(shot: Shot, intensity: float) -> dict[str, float]:
    """Deterministic bounded pan and zoom used when visuals.kenburns is unavailable."""
    rng = random.Random(stable_hash("kenburns", shot.id))
    hi = 1.0 + max(0.0, min(0.5, intensity)) * rng.uniform(0.6, 1.0)
    z0, z1 = (1.0, hi) if rng.random() < 0.6 else (hi, 1.0)
    p: dict[str, float] = {"zoom0": z0, "zoom1": z1}
    for c, z in (("x0", z0), ("y0", z0), ("x1", z1), ("y1", z1)):
        bound = 0.5 - 0.5 / z
        p[c] = 0.5 + rng.uniform(-0.85, 0.85) * bound
    for k, v in (shot.kenburns or {}).items():
        if k in KB_KEYS and v is not None:
            p[k] = float(v)
    p["zoom0"], p["zoom1"] = max(1.0, p["zoom0"]), max(1.0, p["zoom1"])
    for c, zk in (("x0", "zoom0"), ("y0", "zoom0"), ("x1", "zoom1"), ("y1", "zoom1")):
        bound = 0.5 - 0.5 / p[zk]
        p[c] = min(0.5 + bound, max(0.5 - bound, p[c]))
    return {k: round(p[k], 5) for k in KB_KEYS}


def kenburns_for(shot: Shot, profile: StyleProfile) -> dict[str, float]:
    """Ken Burns parameters for a still shot (visuals.kenburns, or the local fallback)."""
    intensity = profile.motion.ken_burns_intensity
    try:
        from ..visuals.kenburns import kenburns_params
    except ImportError:
        return _fallback_kenburns(shot, intensity)
    return {k: float(v) for k, v in kenburns_params(shot, intensity).items() if k in KB_KEYS}


def is_still(shot: Shot) -> bool:
    c = shot.selected_candidate()
    return c is not None and c.media == "image"


def _candidate_part(shot: Shot, settings: Settings) -> Optional[dict[str, Any]]:
    c = shot.selected_candidate()
    if c is None:
        return None
    src = c.source.value if isinstance(c.source, SourceKind) else str(c.source)
    return {"source": src, "media": c.media, "fp": file_fingerprint(resolve_media(c.local_path, settings)),
            "in": round(c.in_point, 3), "out": round(c.out_point, 3), "crop": c.crop}


def segment_hash(shot: Shot, profile: StyleProfile, caption_lines_in_window: list[Any], settings: Settings,
                 frames: Optional[tuple[int, int]] = None, fade_out: bool = False) -> str:
    """Stable render hash of everything that affects this shot's segment pixels."""
    r = settings.render
    frames = frames or frame_range(shot, r.fps)
    overlay = None
    if shot.overlay_path:
        overlay = {"spec": shot.overlay.model_dump(mode="json") if shot.overlay else None,
                   "fp": file_fingerprint(resolve_media(shot.overlay_path, settings))}
    lut = profile.colour.lut_path
    parts = {
        "v": RENDER_VERSION,
        "candidate": _candidate_part(shot, settings),
        "type": shot.type.value,
        "duration": round(shot.duration, 3),
        "frames": list(frames),
        "overlay": overlay,
        "kenburns": kenburns_for(shot, profile) if is_still(shot) else None,
        "captions": [caption_dict(ln) for ln in caption_lines_in_window] if profile.captions.enabled else [],
        "caption_style": profile.captions.model_dump(mode="json"),
        "letterbox": profile.letterbox.model_dump(mode="json"),
        "colour": {"lut": lut, "fp": file_fingerprint(resolve_media(lut, settings)) if lut else None},
        "transition": {"in": shot.transition_in, "out": fade_out, "index0": shot.index == 0,
                       "d": profile.transitions.crossfade_duration},
        "render": {"w": r.width, "h": r.height, "fps": r.fps, "crf": r.crf, "preset": r.preset},
    }
    return stable_hash(parts)


def segment_file(out_dir: Path, render_hash: str) -> Path:
    return Path(out_dir) / f"{render_hash}.mp4"
