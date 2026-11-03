"""File serving helpers: path traversal guard, range-capable responses, thumbnails."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from fastapi.responses import FileResponse

from ..util import ToolError, file_fingerprint, run, stable_hash

PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,79}$")
THUMB_W, THUMB_H = 384, 216
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}


def check_id(value: str, what: str = "id") -> str:
    """Reject ids that could escape a folder or break a file name."""
    if not PROJECT_ID_RE.match(value or ""):
        raise HTTPException(400, f"invalid {what}: use letters, digits, _ and -")
    return value


def safe_join(root: Path, rel: str) -> Path:
    """Resolve rel under root. Raises 403 on traversal, absolute paths or symlink escape, 404 if missing."""
    if not rel or "\x00" in rel or rel.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", rel):
        raise HTTPException(403, "path not allowed")
    parts = Path(rel.replace("\\", "/")).parts
    if any(p in ("..", "") for p in parts):
        raise HTTPException(403, "path not allowed")
    base = root.resolve()
    target = (base / rel).resolve()
    if target != base and not target.is_relative_to(base):
        raise HTTPException(403, "path not allowed")
    if not target.is_file():
        raise HTTPException(404, "file not found")
    return target


def file_response(path: Path, no_cache: bool = False) -> FileResponse:
    """FileResponse with HTTP range support (Starlette handles Range and 206)."""
    headers = {"Accept-Ranges": "bytes"}
    if no_cache:
        headers["Cache-Control"] = "no-cache"
    return FileResponse(path, headers=headers)


def media_url(project_id: str, root: Path, path: str | Path, bust: bool = True) -> Optional[str]:
    """URL of a file under the project root for the media endpoint, or None when outside it."""
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    try:
        rel = p.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    url = f"/api/projects/{project_id}/media/{rel.as_posix()}"
    if bust and p.exists():
        url += f"?v={p.stat().st_mtime_ns}"
    return url


def thumbnail(src: Path, cache_dir: Path, at: Optional[float] = None) -> Path:
    """JPEG thumbnail (384x216, letterboxed to 16:9) of an image, or of a video frame at `at` seconds.

    Cached under cache_dir by source fingerprint, so a changed source gives a new file.
    """
    key = stable_hash(str(src), file_fingerprint(src), at, THUMB_W)
    out = cache_dir / f"{key}.jpg"
    if out.exists():
        return out
    cache_dir.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in VIDEO_EXT:
        _grab_frame(src, out, at or 0.0)
    else:
        _image_thumb(src, out)
    return out


def _image_thumb(src: Path, out: Path) -> None:
    from PIL import Image
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
        canvas = Image.new("RGB", (THUMB_W, THUMB_H), (0, 0, 0))
        canvas.paste(im, ((THUMB_W - im.width) // 2, (THUMB_H - im.height) // 2))
        tmp = out.with_suffix(".tmp.jpg")
        canvas.save(tmp, "JPEG", quality=82)
        tmp.replace(out)


def _grab_frame(src: Path, out: Path, at: float) -> None:
    tmp = out.with_suffix(".tmp.jpg")
    vf = (f"scale={THUMB_W}:{THUMB_H}:force_original_aspect_ratio=decrease,"
          f"pad={THUMB_W}:{THUMB_H}:(ow-iw)/2:(oh-ih)/2")
    tail = ["-frames:v", "1", "-vf", vf, "-q:v", "4", str(tmp)]
    for seek in ([] if at <= 0 else ["-ss", f"{at:.3f}"], []):
        # a seek past the end of a short clip writes nothing, so retry from the first frame
        try:
            run(["ffmpeg", "-y", "-v", "error", *seek, "-i", str(src), *tail], timeout=30)
        except ToolError:
            continue
        if tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(out)
            return
    raise HTTPException(404, "could not grab a frame")
