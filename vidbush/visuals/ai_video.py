"""Optional image to video for ai_video shots.

Used only when the style profile asks for ai_video (visual_mix ai_video > 0)
and the video seat resolves to a real provider. animate_still(shot, still,
ctx) -> Candidate(source=ai_video, media=video) or None, in which case the
caller keeps the still as an ai_image shot with Ken Burns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..models import Candidate, Shot, SourceKind
from ..util import ToolError, media_duration, run, video_size


def ai_video_enabled(ctx: Any) -> bool:
    """True when the profile wants ai_video and the video seat is real."""
    return float(ctx.profile.visual_mix.get("ai_video", 0) or 0) > 0 and bool(ctx.seats.is_real("video"))


def animate_still(shot: Shot, still: Candidate, ctx: Any) -> Optional[Candidate]:
    """Animate an approved still through the video seat. None when the seat gives nothing."""
    if not ai_video_enabled(ctx) or not still.local_path:
        return None
    out = ctx.work / "ai" / f"{shot.id}_video_{_next(shot)}.mp4"
    prompt = f"{still.title or shot.intent} Slow, natural camera move. Keep the scene realistic."
    try:
        path = ctx.seats.video(prompt, Path(still.local_path), out, shot.duration, ctx.stage or "visuals")
    except Exception as e:  # noqa: BLE001  any provider failure keeps the still
        ctx.log.warn("ai video failed, keeping still", shot=shot.id, error=str(e)[:300])
        return None
    if not path or not Path(path).exists():
        return None
    dur = _duration(Path(path)) or shot.duration
    width, height = _size(Path(path))
    return Candidate(id=f"{shot.id}_ai_video_{_next(shot)}", source=SourceKind.ai_video, local_path=str(path),
                     media="video", in_point=0.0, out_point=round(dur, 3), width=width, height=height,
                     licence="AI generated", attribution="AI generated video", title=still.title,
                     keyframes=keyframes(Path(path), dur, ctx.work / "ai") or list(still.keyframes),
                     score=still.score, judge_reason=still.judge_reason, flags=list(still.flags))


def _next(shot: Shot) -> int:
    return 1 + sum(1 for c in shot.candidates if c.source == SourceKind.ai_video)


def _duration(path: Path) -> float:
    try:
        return media_duration(path)
    except (ToolError, OSError, ValueError):
        return 0.0


def _size(path: Path) -> tuple[int, int]:
    try:
        return video_size(path)
    except (ToolError, OSError, ValueError):
        return 0, 0


def keyframes(video: Path, duration: float, out_dir: Path) -> list[str]:
    """Three jpg keyframes at 20, 50 and 80 percent of the clip."""
    out: list[str] = []
    for i, frac in enumerate((0.2, 0.5, 0.8)):
        dst = out_dir / f"{video.stem}_k{i}.jpg"
        try:
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{duration * frac:.3f}",
                 "-i", str(video), "-frames:v", "1", "-vf", "scale=640:-2", str(dst)], timeout=60)
        except ToolError:
            continue
        if dst.exists():
            out.append(str(dst))
    return out
