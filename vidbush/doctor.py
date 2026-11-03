"""Environment checks: ffmpeg, ffprobe, yt-dlp, fonts, whisper, seats, optional live probe."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

from .config import Settings, load_brand
from .providers.base import CostTracker
from .providers.seats import Seats


def doctor(settings: Settings, probe: bool = False) -> dict[str, Any]:
    """Presence checks. probe=True also sends one tiny live request per configured role."""
    bins = {b: bool(shutil.which(b)) for b in ("ffmpeg", "ffprobe", "yt-dlp", "espeak-ng", "node", "depthflow")}
    mods = {m: importlib.util.find_spec(m) is not None
            for m in ("faster_whisper", "scenedetect", "cv2", "yt_dlp", "fastapi", "sqlmodel")}
    fonts = sorted(p.name for p in (Path(settings.repo_root) / "assets" / "fonts").glob("*.otf"))
    seats_obj = Seats(settings, CostTracker("doctor"))
    status = seats_obj.status()
    ok = bins["ffmpeg"] and bins["ffprobe"] and bool(fonts)
    out: dict[str, Any] = {"ok": ok, "binaries": bins, "python_modules": mods, "fonts": fonts,
                           "offline": settings.offline, "seats": {k: v["real"] for k, v in status.items()},
                           "mock_policy": seats_obj.mock_policy(),
                           "tts_voice_id": seats_obj.voice_id(_default_voice(settings))}
    try:
        from .source.youtube import runtime_warning
        out["youtube_runtime"] = runtime_warning() or "ok"
    except Exception as e:  # noqa: BLE001
        out["youtube_runtime"] = f"unknown: {e}"
    if probe and not settings.offline:
        from .providers.probe import probe_seats
        results = probe_seats(seats_obj, out["tts_voice_id"])
        out["probe"] = results
        out["ok"] = ok and not any(r["ok"] is False for r in results.values())
    elif probe:
        out["probe"] = "skipped: offline"
    return out


def _default_voice(settings: Settings) -> Any:
    try:
        return load_brand(settings, "default").voice
    except Exception:  # noqa: BLE001
        from .models import VoiceConfig
        return VoiceConfig()
