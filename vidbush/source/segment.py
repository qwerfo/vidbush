"""Choose the part of a long video to use from what is on screen, not from speech or a fixed offset (v2).

Run 3 evidence: inside a long video the segment came from transcript windows (ties went to the first
window, usually the intro) or a fixed offset (10 percent in, 20 s when the length was unknown), so the
judge compared clips that often did not show the subject at all.

choose_time(key, frames_fn, shot, settings, title) -> Optional[float]
    Frames sampled across the whole video go on one numbered contact sheet; one vision call (the
    cheap_classifier seat, stage sourcing) asks which frames show the shot's subject, and the time of
    the best one is returned. One call per video and subject, cached on disk under
    <source_cache>/segments/ (a "nothing shows it" answer is cached too). None when no real vision seat
    is configured, no frames could be read or no frame shows the subject: the caller keeps its old rule.
youtube_frames(info, n)
    Frames from YouTube's storyboard sprites (the scrub bar previews): no video download at all.
sampled_frames(src, duration, n)
    Frames read with one ffmpeg seek each, from a remote URL or a local file.
focus_window(...)
    The (start, end) seconds of a local file to cut clips from, centred on the chosen time.

configure(seats, log) is called by the registry once per sourcing run (active_adapters).
"""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import Settings
from ..models import Shot
from ..util import clamp, run

Frame = tuple[float, Any]  # (seconds, PIL image)
SHEET_COLS = 5
TILE_W, TILE_H = 256, 144
YOUTUBE_FRAMES = 20
REMOTE_FRAMES = 12
LOCAL_FRAMES = 16
EDGE = 0.04  # skip the first and last 4 percent (titles, end cards)
SEAT, TASK, STAGE = "cheap_classifier", "segment", "sourcing"
SYSTEM = ("You help a documentary editor find the right part of a long video. You see numbered frames "
          "sampled in time order. Reply with JSON only.")

_state: dict[str, Any] = {"seats": None, "log": None}
_lock = threading.Lock()


def configure(seats: Any, log: Any = None) -> None:
    with _lock:
        _state["seats"], _state["log"] = seats, log


def available() -> bool:
    seats = _state["seats"]
    try:
        return bool(seats is not None and seats.is_real(SEAT))
    except Exception:  # noqa: BLE001 - a seats object without the role means no chooser
        return False


def subject_of(shot: Shot) -> str:
    """What the frames must show: the real thing the shot names, else must_show and the intent."""
    rt = shot.real_thing
    if rt is not None:
        extra = ", ".join(x for x in (rt.kind, rt.date, rt.place) if x)
        return f"{rt.name} ({extra})" if extra else rt.name
    parts = [", ".join(shot.must_show), shot.intent]
    return "; ".join(p for p in parts if p).strip()


def _cache_path(settings: Settings, key: str, subject: str) -> Path:
    from .base import cache_dir
    h = hashlib.sha1(f"{key}|{' '.join(subject.lower().split())}".encode("utf-8")).hexdigest()[:24]
    return cache_dir(settings) / "segments" / f"{h}.json"


def contact_sheet(frames: list[Frame], out: Path) -> Path:
    """Numbered grid of frames (1-based), SHEET_COLS wide."""
    from PIL import Image, ImageDraw
    rows = (len(frames) + SHEET_COLS - 1) // SHEET_COLS
    sheet = Image.new("RGB", (SHEET_COLS * TILE_W, rows * TILE_H), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for i, (_t, im) in enumerate(frames):
        x, y = (i % SHEET_COLS) * TILE_W, (i // SHEET_COLS) * TILE_H
        sheet.paste(im.convert("RGB").resize((TILE_W, TILE_H)), (x, y))
        draw.rectangle([x, y, x + 34, y + 22], fill=(0, 0, 0))
        draw.text((x + 5, y + 4), str(i + 1), fill=(255, 255, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return out


def ask(frames: list[Frame], subject: str, title: str = "") -> Optional[dict[str, Any]]:
    """One vision call; {"frames": [...], "best": n} or None (mock seat, error, unparseable)."""
    seats = _state["seats"]
    if seats is None or not frames:
        return None
    with tempfile.TemporaryDirectory(prefix="vidbush_seg_") as tmp:
        sheet = contact_sheet(frames, Path(tmp) / "sheet.jpg")
        prompt = (f"{len(frames)} numbered frames sampled in time order from one video"
                  + (f' titled "{title[:120]}"' if title else "") + f". The shot needs: {subject[:300]}.\n"
                  "Which frames show it: the thing itself on screen, not a title card, a presenter talking, a "
                  "map or an unrelated scene? Reply {\"frames\": [the frame numbers that show it, best first], "
                  "\"best\": the best frame number, or 0 when none shows it}.")
        try:
            raw = seats.vision_json(SEAT, TASK, SYSTEM, prompt, [str(sheet)], STAGE, intent=subject,
                                    details=["high"])
        except Exception as e:  # noqa: BLE001 - the chooser is an improvement, never a failure
            log = _state["log"]
            if log is not None:
                log.warn("segment chooser call failed", error=str(e)[:200])
            return None
    if not isinstance(raw, dict) or raw.get("_mock"):
        return None
    return raw


def pick(raw: dict[str, Any], n: int) -> Optional[int]:
    """0-based index of the best frame from a reply, None when it names none."""
    nums: list[int] = []
    for v in [raw.get("best"), *(raw.get("frames") or [])]:
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= n:
            nums.append(i)
    return nums[0] - 1 if nums else None


def choose_time(key: str, frames_fn: Callable[[], list[Frame]], shot: Shot, settings: Settings,
                title: str = "") -> Optional[float]:
    """Seconds into the video where the shot's subject is best seen, or None (see module doc)."""
    subject = subject_of(shot)
    if not subject or not available():
        return None
    path = _cache_path(settings, key, subject)
    try:
        hit = json.loads(path.read_text(encoding="utf-8"))
        return None if hit.get("t") is None else float(hit["t"])
    except (OSError, ValueError, TypeError, KeyError):
        pass
    try:
        frames = frames_fn()
    except Exception as e:  # noqa: BLE001 - frames are best effort
        log = _state["log"]
        if log is not None:
            log.info("segment frames unavailable", key=key[:120], error=str(e)[:160])
        return None
    raw = ask(frames, subject, title)
    if raw is None:
        return None  # not cached: a later run with a working seat may answer
    idx = pick(raw, len(frames))
    t = frames[idx][0] if idx is not None else None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": key, "subject": subject, "t": t, "times": [round(f[0], 2) for f in frames],
                                    "reply": {k: v for k, v in raw.items() if not k.startswith("_")}}), encoding="utf-8")
    except OSError:
        pass
    return t


def even_times(duration: float, n: int) -> list[float]:
    lo, hi = duration * EDGE, duration * (1 - EDGE)
    return [lo + (hi - lo) * (i + 0.5) / n for i in range(n)]


# ---------------------------------------------------------------- frame readers

def storyboard_format(info: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The storyboard format with the largest tiles (yt-dlp lists them as mhtml formats sb0, sb1...)."""
    boards = [f for f in info.get("formats") or [] if f.get("format_note") == "storyboard" and f.get("fragments")
              and f.get("rows") and f.get("columns")]
    return max(boards, key=lambda f: int(f.get("width") or 0) * int(f.get("height") or 0), default=None)


def storyboard_tiles(fmt: dict[str, Any], duration: float) -> list[tuple[float, int, int]]:
    """(time, fragment index, tile index) for every tile of a storyboard format."""
    per = int(fmt["rows"]) * int(fmt["columns"])
    out: list[tuple[float, int, int]] = []
    t0 = 0.0
    for fi, frag in enumerate(fmt["fragments"]):
        dur = float(frag.get("duration") or 0.0) or (per / float(fmt.get("fps") or 1.0))
        step = dur / per
        for j in range(per):
            t = t0 + j * step
            if duration and t >= duration:
                break
            out.append((t, fi, j))
        t0 += dur
    return out


def youtube_frames(info: dict[str, Any], n: int = YOUTUBE_FRAMES) -> list[Frame]:
    """Frames at n even times from the storyboard sprites (only the sprite sheets holding them are read)."""
    from PIL import Image
    from . import base
    fmt = storyboard_format(info)
    duration = float(info.get("duration") or 0.0)
    if fmt is None or duration <= 0:
        return []
    tiles = storyboard_tiles(fmt, duration)
    if not tiles:
        return []
    cols, w, h = int(fmt["columns"]), int(fmt.get("width") or 0), int(fmt.get("height") or 0)
    want = [min(tiles, key=lambda x: abs(x[0] - t)) for t in even_times(duration, n)]
    sheets: dict[int, Any] = {}
    frames: list[Frame] = []
    for t, fi, j in dict.fromkeys(want):
        if fi not in sheets:
            data = base.http_get_bytes(fmt["fragments"][fi]["url"], timeout=30)
            sheets[fi] = Image.open(io.BytesIO(data)).convert("RGB")
        sheet = sheets[fi]
        tw, th = (w or sheet.width // cols), (h or sheet.height // int(fmt["rows"]))
        x, y = (j % cols) * tw, (j // cols) * th
        if x + tw <= sheet.width and y + th <= sheet.height:
            frames.append((t, sheet.crop((x, y, x + tw, y + th))))
    return frames


def sampled_frames(src: str, duration: float, n: int = REMOTE_FRAMES) -> list[Frame]:
    """Frames at n even times read with ffmpeg seeks (fast input seek; a remote mp4 is read by range)."""
    from PIL import Image
    from . import base
    if duration <= 0:
        return []
    frames: list[Frame] = []
    proxy = ["-http_proxy", base.HTTP_PROXY] if base.HTTP_PROXY and "://" in src else []
    ua = ["-user_agent", base.USER_AGENT] if "://" in src else []
    with tempfile.TemporaryDirectory(prefix="vidbush_frames_") as tmp:
        for i, t in enumerate(even_times(duration, n)):
            out = Path(tmp) / f"f{i}.jpg"
            try:
                run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", *ua, *proxy, "-i", src,
                     "-frames:v", "1", "-vf", f"scale={TILE_W}:-2", "-q:v", "5", str(out)], timeout=60)
                with Image.open(out) as im:
                    frames.append((t, im.convert("RGB").copy()))
            except Exception:  # noqa: BLE001 - one unreadable position loses one frame
                continue
    return frames


def remote_duration(url: str) -> float:
    """Length of a remote video from its container header (ffprobe), 0 when it cannot be read."""
    from ..util import ffprobe
    try:
        return float((ffprobe(url).get("format") or {}).get("duration") or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def centred(t: float, length: float, duration: float) -> tuple[float, float]:
    a = clamp(t - length / 2, 0.0, max(0.0, duration - length))
    return a, min(duration, a + length)


def focus_window(path: Path, total: float, shot: Shot, settings: Settings, key: str, title: str,
                 length: float) -> Optional[tuple[float, float]]:
    """(start, end) in a local file around where the subject is seen, or None."""
    if total <= 2 * length:
        return None
    t = choose_time(key, lambda: sampled_frames(str(path), total, LOCAL_FRAMES), shot, settings, title)
    return None if t is None else centred(t, length, total)
