"""Optional DepthFlow parallax for still shots.

When profile.motion.depthflow is true and the `depthflow` CLI is installed,
a still is turned into a 2.5D parallax clip of the shot's length, cached in
work/depth/<hash>.mp4. Any failure keeps the still (Ken Burns).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from ..models import Candidate, Shot
from ..util import ToolError, file_fingerprint, run, stable_hash
from .ai_video import keyframes

CLI = "depthflow"


def available() -> bool:
    """True when the depthflow CLI is on PATH."""
    return shutil.which(CLI) is not None


def render_parallax(image: Path, out: Path, duration: float, width: int, height: int, fps: int) -> Path:
    """Render a parallax clip with the depthflow CLI (raises ToolError on failure)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [CLI, "input", "-i", str(image), "main", "-o", str(out), "-w", str(width), "-h", str(height),
           "-f", str(fps), "-t", f"{duration:.3f}"]
    run(cmd, timeout=max(180.0, duration * 60))
    if not out.exists():
        raise ToolError("depthflow wrote no output")
    return out


def parallax_candidate(shot: Shot, cand: Candidate, ctx: Any) -> Optional[Candidate]:
    """Copy of a still candidate swapped to a parallax video, or None to keep Ken Burns."""
    if cand.media != "image" or not cand.local_path or not Path(cand.local_path).exists():
        return None
    r = ctx.settings.render
    key = stable_hash("depthflow", file_fingerprint(cand.local_path), cand.crop, round(shot.duration, 3),
                      r.width, r.height, r.fps)
    out = ctx.work / "depth" / f"{key}.mp4"
    if not out.exists():
        try:
            src = _cropped_source(Path(cand.local_path), cand.crop, out.with_suffix(".src.png"))
            render_parallax(src, out, shot.duration, r.width, r.height, r.fps)
        except (ToolError, OSError) as e:
            ctx.log.warn("depthflow failed, keeping Ken Burns", shot=shot.id, error=str(e)[-300:])
            return None
    frames = keyframes(out, shot.duration, ctx.work / "depth") or list(cand.keyframes)
    return cand.model_copy(update={"local_path": str(out), "media": "video", "in_point": 0.0,
                                   "out_point": round(shot.duration, 3), "keyframes": frames, "crop": None,
                                   "flags": list(cand.flags) + ["depthflow"]})


def _cropped_source(image: Path, crop: Optional[list[float]], out: Path) -> Path:
    """The derived crop baked into a temporary still, so the parallax shows the same framing."""
    if not crop:
        return image
    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image) as im:
        w, h = im.size
        x, y, cw, ch = crop
        im.convert("RGB").crop((round(x * w), round(y * h), round((x + cw) * w), round((y + ch) * h))).save(out)
    return out
