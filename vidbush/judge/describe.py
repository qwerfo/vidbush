"""Describe each asset once: one vision call on frames spread across the usable part of the asset.

The record says what is visible (what it shows, people, place, period and look, text in the frame,
quality problems, and for video the time ranges worth using) and, kept apart, what sourcing already
knows (title, source, date, description, page, licence, the query that found it). Stills also get the
words the local text reader (rapidocr) finds. The describer is not shown the title or the query, so
the visible fields say only what the frames show; identity is judged later by comparing the two.

Visible fields are cached by the asset's content hash (and, for a video, the clip range), so an asset
is described once across all shots and all runs. The local prefilter runs first: its rejects never
cost a call.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

from ..models import Candidate, SourceKind

VERSION = 1
ROLE, FALLBACK_ROLE, TASK, STAGE = "describer", "vision", "describe_asset", "judge"
VIDEO_FRAMES = 3  # spread over the clip; 4 cost a third more for little more seen
FRAME_W = 640  # the vision seat fits every image into 640 px; a frame is sent at that width
OCR_MAX_SIDE = 2000
OCR_MIN_CONF = 0.5
WORKERS = 8
QUALITY = ("burned_in_subtitles", "watermark", "stock_preview_mark", "channel_logo", "low_resolution",
           "presenter_to_camera", "screen_recording")
# a record listing one of these is never picked (Adam: no subtitles, marks or logos on screen)
BLOCKING = ("burned_in_subtitles", "watermark", "stock_preview_mark", "channel_logo")
LOOKS = ("black_and_white", "colour_film", "modern_digital", "illustration", "document", "unknown")

SYSTEM = (
    "You describe one picture asset for a documentary picture editor who will compare many assets and "
    "choose between them. Describe only what is visible in the frames: never guess a name, a title or a "
    "place that nothing in the frames shows. Reply with one JSON object only."
)

RULES = [
    "What your output is used for: the editor reads your record instead of watching the asset, and code "
    "refuses an asset whose record lists a watermark, a stock preview mark, burned-in subtitles or a channel "
    "logo. A missed mark ships a faulty picture; an invented one loses a good picture. Report only what you see.",
    "shows: one plain sentence of what the asset shows, the main subject first.",
    "people: how many people are visible (0 when none), what they are doing, and whether anyone faces or "
    "speaks to the camera.",
    "place: the place as far as the frames show it (landscape, buildings, signs). Name a specific place only "
    "when visible text or an unmistakable landmark shows it; otherwise describe it (example, invented: "
    "'a harbour town with wooden houses under snowy hills').",
    "period: the period as far as the frames show it (clothing, vehicles, film stock), and look: one of "
    + ", ".join(LOOKS) + ".",
    "text_in_frame: every readable text in the frames, transcribed exactly (headlines, signs, captions, "
    "page text). Empty when there is none. Do not transcribe a watermark here; list it under quality.",
    "quality: the problems you see, from this list only: " + ", ".join(QUALITY) + ". burned_in_subtitles "
    "is dialogue or narration text burned into the picture; a sign or a newspaper page is not. "
    "stock_preview_mark is a stock agency's preview mark or tiled watermark. screen_recording is a "
    "computer or phone screen, a browser or an app window. presenter_to_camera is someone talking to the "
    "camera.",
    "usable, for a video only: the time ranges worth using, from the frame times given, each with one line "
    "of what plays there. Leave out ranges with a quality problem, a black frame or a title card. Empty for "
    "a still.",
    "No em dash characters anywhere.",
]

OUTPUT = {"shows": "", "people": {"count": 0, "doing": "", "facing_camera": False}, "place": "",
          "period": "", "look": "unknown", "text_in_frame": "", "quality": [],
          "usable": [{"from": 0.0, "to": 0.0, "note": ""}]}

_OCR_LOCK = threading.Lock()


def is_ai(c: Candidate) -> bool:
    return c.source in (SourceKind.ai_image, SourceKind.ai_video)


def cache_dir(settings: Any = None) -> Path:
    """DESCRIBE_CACHE_FOLDER, else <source cache>/describe (on E: through the data/source_cache link)."""
    env = os.environ.get("DESCRIBE_CACHE_FOLDER", "")
    if env:
        return Path(env)
    base = getattr(settings, "source_cache_folder", "") or "data/source_cache"
    root = Path(getattr(settings, "repo_root", ".") or ".")
    p = Path(base)
    return (p if p.is_absolute() else root / p) / "describe"


def content_hash(path: str) -> str:
    """sha1 of size, the first 4 MB and the last 1 MB: identical files share it; reading whole videos is slow."""
    p = Path(path)
    h = hashlib.sha1()
    size = p.stat().st_size
    h.update(str(size).encode())
    with p.open("rb") as f:
        h.update(f.read(4 << 20))
        if size > 5 << 20:
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read())
    return h.hexdigest()[:24]


def record_key(c: Candidate) -> str:
    """Content hash, plus the clip range for a video (two clips of one file show different things)."""
    k = content_hash(c.local_path)
    if c.media == "video":
        k += f"_{c.in_point:.1f}-{c.out_point:.1f}"
    return k


def frame_times(c: Candidate, n: int = VIDEO_FRAMES) -> list[float]:
    """n times spread across [in_point, out_point], away from the edges."""
    a, b = c.in_point, max(c.in_point, c.out_point)
    if b - a < 0.2:
        return [a]
    return [round(a + (b - a) * (i + 0.5) / n, 2) for i in range(n)]


def extract_frame(video: str, t: float, out: Path, width: int = FRAME_W) -> Optional[str]:
    if out.exists() and out.stat().st_size > 0:
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1",
           "-vf", f"scale={width}:-2", "-q:v", "3", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    except (subprocess.SubprocessError, OSError):
        return None
    return str(out) if out.exists() and out.stat().st_size > 0 else None


def still_frame(path: str, out: Path, width: int = FRAME_W) -> Optional[str]:
    from PIL import Image
    if out.exists() and out.stat().st_size > 0:
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((width, width))
            im.save(out, quality=90)
    except OSError:
        return None
    return str(out)


def frames_for(c: Candidate, key: str, root: Path) -> list[tuple[float, str]]:
    """(time, path) frames at 640 px; a still gives one frame at time 0."""
    fdir = root / "frames"
    if c.media != "video":
        p = still_frame(c.local_path, fdir / f"{key}_0.jpg")
        return [(0.0, p)] if p else []
    out = []
    for i, t in enumerate(frame_times(c)):
        p = extract_frame(c.local_path, t, fdir / f"{key}_{i}.jpg")
        if p:
            out.append((t, p))
    return out


def ocr_words(path: str) -> str:
    """Words the local text reader finds on a still, in reading order; empty when the reader is missing."""
    try:
        import numpy as np
        from PIL import Image
        from .prefilter import ocr_engine
    except ImportError:
        return ""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((OCR_MAX_SIDE, OCR_MAX_SIDE))
            arr = np.ascontiguousarray(np.asarray(im))
        with _OCR_LOCK:  # one ONNX session is shared; it is not safe across threads
            result, _ = ocr_engine()(arr)
    except Exception:  # noqa: BLE001 - an unreadable image has no words, the record still stands
        return ""
    lines = []
    for box, text, conf in result or []:
        try:
            if float(conf) >= OCR_MIN_CONF and str(text).strip():
                lines.append((min(p[1] for p in box), min(p[0] for p in box), str(text).strip()))
        except (TypeError, ValueError):
            continue
    lines.sort(key=lambda x: (round(x[0] / 12), x[1]))
    return " ".join(t for _, _, t in lines)[:4000]


def source_facts(c: Candidate) -> dict[str, Any]:
    """What sourcing already knows about the asset (never shown to the describer)."""
    f = {"title": c.title, "source": c.source.value, "date": getattr(c, "date", "") or "",
         "description": (getattr(c, "description", "") or "")[:600], "page_url": getattr(c, "page_url", "") or "",
         "licence": c.licence, "author": c.author, "query": getattr(c, "query", "") or "",
         "found_by": getattr(c, "found_by", "") or "", "flags": list(c.flags)}
    if c.width and c.height:
        f["size"] = f"{c.width}x{c.height}"
    if c.media == "video":
        f["clip"] = [round(c.in_point, 2), round(c.out_point, 2)]
    return {k: v for k, v in f.items() if v not in ("", [], None)}


def _prompt(c: Candidate, frames: list[tuple[float, str]]) -> str:
    if c.media == "video":
        what = (f"A video clip from {c.in_point:.1f} s to {c.out_point:.1f} s. The {len(frames)} images are frames "
                "at these times, in order: " + ", ".join(f"{t:.1f} s" for t, _ in frames) + ".")
    else:
        what = "A still image (photograph, document page, painting or graphic). The one image is the asset."
    return json.dumps({"asset": what, "rules": RULES, "output": OUTPUT}, ensure_ascii=False)


def _clean(raw: Any, c: Candidate) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    ppl = raw.get("people") if isinstance(raw.get("people"), dict) else {}
    try:
        count = int(ppl.get("count") or 0)
    except (TypeError, ValueError):
        count = 0
    q = [str(x).strip().lower() for x in (raw.get("quality") or []) if str(x).strip().lower() in QUALITY]
    usable = []
    if c.media == "video":
        for u in raw.get("usable") or []:
            try:
                a, b = float(u.get("from")), float(u.get("to"))
            except (TypeError, ValueError, AttributeError):
                continue
            a, b = max(a, c.in_point), min(b, c.out_point)
            if b - a >= 0.5:
                usable.append({"from": round(a, 2), "to": round(b, 2), "note": str(u.get("note", ""))[:120]})
    look = str(raw.get("look") or "unknown").strip().lower()
    return {"shows": str(raw.get("shows") or "")[:300],
            "people": {"count": count, "doing": str(ppl.get("doing") or "")[:160],
                       "facing_camera": bool(ppl.get("facing_camera"))},
            "place": str(raw.get("place") or "")[:200], "period": str(raw.get("period") or "")[:160],
            "look": look if look in LOOKS else "unknown",
            "text_in_frame": str(raw.get("text_in_frame") or "")[:1200], "quality": sorted(set(q)),
            "usable": usable}


def _role(seats) -> str:
    try:
        return ROLE if any(p.configured for p in seats.providers_for(ROLE)) else FALLBACK_ROLE
    except AttributeError:  # a test double without providers_for
        return ROLE


def describe_one(seats, c: Candidate, root: Path) -> Optional[dict[str, Any]]:
    """The asset's record (visible fields cached), or None when it has no readable frame."""
    if not c.local_path or not Path(c.local_path).exists():
        return None
    key = record_key(c)
    path = root / f"{key}.json"
    vis: Optional[dict] = None
    if path.exists():
        try:
            vis = json.loads(path.read_text(encoding="utf-8"))
            if vis.get("version") != VERSION:
                vis = None
        except (OSError, ValueError):
            vis = None
    cached = vis is not None
    if vis is None:
        frames = frames_for(c, key, root)
        if not frames:
            return None
        raw = seats.vision_json(_role(seats), TASK, SYSTEM, _prompt(c, frames), [p for _, p in frames], STAGE,
                                details=["high"] * len(frames))
        vis = {"version": VERSION, "key": key, "media": c.media,
               "frames": [{"t": t, "path": p} for t, p in frames], "visual": _clean(raw, c),
               "ocr_text": ocr_words(c.local_path) if c.media != "video" else "",
               "mock": bool(isinstance(raw, dict) and raw.get("_mock"))}
        if not vis["mock"]:  # a mock answer is never cached as a description
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(vis, ensure_ascii=False, indent=1), encoding="utf-8")
    rec = dict(vis)
    rec["cached"] = cached
    rec["ai"] = is_ai(c)
    rec["source_facts"] = source_facts(c)
    return rec


def blocked(rec: dict[str, Any]) -> list[str]:
    return [q for q in rec.get("visual", {}).get("quality", []) if q in BLOCKING]


def describe_pool(seats, pool: list[Candidate], settings: Any = None, log: Optional[Callable] = None,
                  root: Optional[Path] = None) -> dict[str, dict[str, Any]]:
    """Records by candidate id for every candidate that passes the prefilter; one call per uncached asset."""
    from .prefilter import apply_prefilter
    root = root or cache_dir(settings)
    root.mkdir(parents=True, exist_ok=True)
    fresh = [c for c in pool if c.score is None and c.reject_reason is None and not is_ai(c)]
    if settings is not None and fresh:
        apply_prefilter(fresh, settings)
    todo = [c for c in pool if not (c.reject_reason or "").lower().startswith(("prefilter", "resolution"))]
    out: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()

    def one(c: Candidate) -> None:
        try:
            r = describe_one(seats, c, root)
        except Exception as e:  # noqa: BLE001 - one broken asset never sinks the pool; it is simply not offered
            if log:
                log(f"describe failed for {c.id}: {str(e)[:160]}")
            return
        if r is not None:
            with lock:
                out[c.id] = r

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, todo))
    if log:
        log(f"described {len(out)} of {len(pool)} assets ({len(pool) - len(todo)} prefilter rejects, "
            f"{sum(1 for r in out.values() if not r['cached'])} new calls)")
    return out
