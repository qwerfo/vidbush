"""Keyframe extraction for candidates (inputs to the judge and the shot editor).

extract_keyframes(candidate, out_dir) writes 640 px wide jpgs: three frames
at 20, 50 and 80 percent of [in_point, out_point] for video, the image
itself for stills. It also fills candidate.width/height (ffprobe or
Pillow), fixes a missing out_point and sets candidate.keyframes.
"""
from __future__ import annotations

from pathlib import Path

from ..models import Candidate
from ..util import ToolError, ffprobe, run

KEYFRAME_WIDTH = 640
POSITIONS = (0.2, 0.5, 0.8)


def _probe_video(path: Path) -> tuple[int, int, float]:
    info = ffprobe(path)
    w = h = 0
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            w, h = int(s.get("width", 0)), int(s.get("height", 0))
            rot = str((s.get("tags") or {}).get("rotate", "0"))
            if rot in {"90", "270", "-90"}:
                w, h = h, w
            break
    try:
        dur = float(info.get("format", {}).get("duration", 0.0))
    except ValueError:
        dur = 0.0
    return w, h, dur


def _frame(path: Path, t: float, out: Path) -> Path:
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", str(path),
         "-frames:v", "1", "-vf", f"scale={KEYFRAME_WIDTH}:-2", "-q:v", "4", str(out)], timeout=60)
    if not out.exists():
        raise ToolError(f"no frame at {t:.2f}s")
    return out


def video_keyframes(cand: Candidate, out_dir: Path) -> list[str]:
    src = Path(cand.local_path)
    w, h, dur = _probe_video(src)
    if w and h:
        cand.width, cand.height = w, h
    if cand.out_point <= cand.in_point or (dur and cand.out_point > dur + 0.05):
        cand.in_point = min(cand.in_point, max(0.0, dur - 0.5))
        cand.out_point = round(dur, 3)
    span = max(0.0, cand.out_point - cand.in_point)
    outs = []
    for i, p in enumerate(POSITIONS):
        t = min(cand.in_point + span * p, max(0.0, dur - 0.1)) if dur else cand.in_point + span * p
        outs.append(str(_frame(src, t, out_dir / f"{cand.id}_k{i + 1}.jpg")))
    return outs


def image_keyframe(cand: Candidate, out_dir: Path) -> list[str]:
    from PIL import Image, ImageOps
    with Image.open(cand.local_path) as im:
        im = ImageOps.exif_transpose(im)
        cand.width, cand.height = im.size
        rgb = im.convert("RGB")
        ratio = KEYFRAME_WIDTH / max(1, rgb.width)
        thumb = rgb.resize((KEYFRAME_WIDTH, max(1, round(rgb.height * ratio))), Image.LANCZOS)
    out = out_dir / f"{cand.id}_k1.jpg"
    thumb.save(out, quality=85)
    return [str(out)]


def extract_keyframes(candidate: Candidate, out_dir: Path | str) -> list[str]:
    """Write keyframes for one candidate; returns their paths and stores them on the candidate."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if candidate.media == "video":
        frames = video_keyframes(candidate, out_dir)
    else:
        frames = image_keyframe(candidate, out_dir)
    candidate.keyframes = frames
    return frames
