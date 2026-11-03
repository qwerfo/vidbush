"""Environment presence checks: ffmpeg, ffprobe, yt-dlp, fonts, whisper, seats."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

from .config import Settings
from .providers.base import CostTracker
from .providers.seats import Seats


def doctor(settings: Settings) -> dict[str, Any]:
    bins = {b: bool(shutil.which(b)) for b in ("ffmpeg", "ffprobe", "yt-dlp", "espeak-ng", "node", "depthflow")}
    mods = {m: importlib.util.find_spec(m) is not None
            for m in ("faster_whisper", "scenedetect", "cv2", "yt_dlp", "fastapi", "sqlmodel")}
    fonts = sorted(p.name for p in (Path(settings.repo_root) / "assets" / "fonts").glob("*.otf"))
    seats = Seats(settings, CostTracker("doctor")).status()
    ok = bins["ffmpeg"] and bins["ffprobe"] and bool(fonts)
    return {"ok": ok, "binaries": bins, "python_modules": mods, "fonts": fonts,
            "offline": settings.offline, "seats": {k: v["real"] for k, v in seats.items()}}
