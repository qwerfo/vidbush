"""Source adapter contract and shared helpers.

An adapter has a name, the media kinds it supports, an availability check,
search(query, media, n, settings, shot=None) -> list[RawResult] and
fetch(raw, out_dir, shot, settings) -> list[Candidate]. fetch may return
several candidates for one asset (a long video cut into clips). Candidate
ids returned by fetch are provisional: the registry renumbers them to
f"{shot.id}_{source}_{n}" and extracts keyframes.

All network access goes through http_get_json / http_get_bytes /
ffmpeg_section in this module so tests can monkeypatch one place.
Downloads land in a shared cache (settings.source_cache_folder, keyed by
URL hash) and are hardlinked or copied into the project, each with a
<file>.json sidecar.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, runtime_checkable
from urllib.parse import urlparse

from pydantic import Field

from ..config import Settings
from ..models import Candidate, Model, Shot, SourceKind
from ..providers import http as provider_http
from ..util import clamp, run, write_json
from .licence import licence_flags, normalise_licence

Media = Literal["video", "image"]
USER_AGENT = "Vidbush/1.0 (self-hosted documentary tool; contact: operator)"
SOURCE_LABELS: dict[str, str] = {
    "youtube": "YouTube", "wikimedia": "Wikimedia Commons", "archive": "Internet Archive",
    "loc": "Library of Congress", "nasa": "NASA", "pexels": "Pexels", "pixabay": "Pixabay",
    "local": "footage library",
}


class RawResult(Model):
    """One search hit before download."""

    source: str
    url: str  # canonical asset URL (dedupe and diversity key)
    title: str = ""
    author: str = ""
    licence: str = "unknown"
    media: Media = "image"
    width: int = 0
    height: int = 0
    duration: float = 0.0
    download_url: str = ""
    page_url: str = ""
    score: float = 1.0  # relevance in 0..1; 0 means filler (no keyword match)
    note: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class SourceAdapter(Protocol):
    name: str
    media: set[str]

    def available(self, settings: Settings) -> bool: ...

    def search(self, query: str, media: Media, n: int, settings: Settings,
               shot: Optional[Shot] = None) -> list[RawResult]: ...

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Settings) -> list[Candidate]: ...


# ---------------------------------------------------------------- network seams

def http_get_json(url: str, params: Optional[dict[str, Any]] = None, headers: Optional[dict[str, str]] = None,
                  timeout: float = 20.0) -> Any:
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    return provider_http.get_json(url, params=params, headers=hdrs, timeout=timeout)


def http_get_bytes(url: str, headers: Optional[dict[str, str]] = None, timeout: float = 60.0) -> bytes:
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    return provider_http.get_bytes(url, headers=hdrs, timeout=max(timeout, 60.0))


def ffmpeg_section(url: str, start: float, length: float, out: Path) -> Path:
    """Copy one section of a remote (or local) video without downloading the rest."""
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{length:.2f}",
         "-user_agent", USER_AGENT, "-i", url, "-map", "0:v:0", "-c", "copy", "-avoid_negative_ts", "make_zero",
         str(out)], timeout=600)
    return out


# ---------------------------------------------------------------- cache and files

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]


def ext_from_url(url: str, default: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix and len(suffix) <= 5 and suffix[1:].isalnum() else default


def cache_dir(settings: Settings) -> Path:
    d = settings.path(settings.source_cache_folder)
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached_download(url: str, ext: str, settings: Settings, headers: Optional[dict[str, str]] = None) -> Path:
    """Download url once into the shared cache; later calls reuse the file."""
    dest = cache_dir(settings) / f"{url_hash(url)}{ext}"
    with _lock_for(str(dest)):
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        data = http_get_bytes(url, headers=headers, timeout=settings.sources.timeout_s)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
    return dest


def cached_section(url: str, start: float, length: float, settings: Settings) -> Path:
    """ffmpeg_section through the shared cache, keyed by url and range."""
    dest = cache_dir(settings) / f"{url_hash(f'{url}#{start:.1f}+{length:.1f}')}.mp4"
    with _lock_for(str(dest)):
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        tmp = dest.with_name(dest.stem + ".part.mp4")
        ffmpeg_section(url, start, length, tmp)
        tmp.replace(dest)
    return dest


def link_into(src: Path, out_dir: Path, name: str) -> Path:
    """Hardlink (or copy when linking fails) src into out_dir/name."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / name
    if dst.exists():
        return dst
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def attribution_for(source: str, title: str, author: str) -> str:
    label = SOURCE_LABELS.get(source, source)
    title = title or "Untitled"
    return f"{title} by {author} ({label})" if author else f"{title} ({label})"


def write_sidecar(path: Path, raw: RawResult, attribution: str, flags: list[str]) -> Path:
    """<file>.json next to a project asset with provenance and licence."""
    data = {"source": raw.source, "source_url": raw.download_url or raw.url, "page_url": raw.page_url or raw.url,
            "title": raw.title, "author": raw.author, "licence": normalise_licence(raw.licence),
            "attribution": attribution, "downloaded_at": round(time.time(), 3), "flags": flags}
    if raw.note:
        data["note"] = raw.note
    return write_json(path.with_name(path.name + ".json"), data)


def make_candidates(raw: RawResult, path: Path, shot: Shot, settings: Settings,
                    spans: Optional[list[tuple[float, float]]] = None, media: Optional[Media] = None) -> list[Candidate]:
    """Candidates for one stored asset plus its sidecar. spans are (in, out) clip points."""
    licence = normalise_licence(raw.licence)
    attribution = raw.extra.get("attribution") or attribution_for(raw.source, raw.title, raw.author)
    flags = licence_flags(licence, settings)
    if raw.note == "fixture rotation":
        flags.append("fixture_rotation")
    write_sidecar(path, raw, attribution, flags)
    media = media or raw.media
    spans = spans or [(0.0, raw.duration if media == "video" else 0.0)]
    return [Candidate(id=f"{shot.id}_{raw.source}_{i}", source=SourceKind(raw.source), url=raw.page_url or raw.url,
                      local_path=str(path), media=media, in_point=round(a, 3), out_point=round(b, 3),
                      width=raw.width, height=raw.height, licence=licence, author=raw.author,
                      attribution=attribution, title=raw.title, flags=list(flags))
            for i, (a, b) in enumerate(spans)]


# ---------------------------------------------------------------- clip cutting

def detect_scenes(path: Path) -> list[tuple[float, float]]:
    """Scene spans in seconds via PySceneDetect ContentDetector; whole file on failure."""
    from ..util import media_duration
    total = media_duration(path)
    try:
        from scenedetect import ContentDetector, detect
        scenes = detect(str(path), ContentDetector(threshold=27.0))
        spans = [(s.get_seconds(), e.get_seconds()) for s, e in scenes]
    except Exception:  # noqa: BLE001 - scene detection is best effort
        spans = []
    return spans or [(0.0, total)]


def cut_clips(scenes: list[tuple[float, float]], shot_duration: float, max_clips: int = 4,
              min_len: float = 3.0, max_len: float = 12.0) -> list[tuple[float, float]]:
    """Turn scene spans into 3 to 12 s clips, spread over the video, best fits first."""
    target = clamp(max(shot_duration + 1.0, 6.0), min_len, max_len)
    clips: list[tuple[float, float]] = []
    for a, b in scenes:
        a, b = a + 0.2, b - 0.2  # stay clear of the cut itself
        if b - a < min_len:
            continue
        if b - a <= max_len:
            clips.append((a, b))
            continue
        t = a
        while b - t >= min_len:
            clips.append((t, min(b, t + target)))
            t += target
    fits = [c for c in clips if c[1] - c[0] >= shot_duration] or clips
    if len(fits) <= max_clips:
        return fits
    step = len(fits) / max_clips
    return [fits[int(i * step)] for i in range(max_clips)]


class BaseAdapter:
    """Common availability rule: no network in offline mode, keys when required."""

    name: str = ""
    media: set[str] = {"video", "image"}
    key_name: str = ""

    def available(self, settings: Settings) -> bool:
        if settings.offline:
            return False
        return not self.key_name or bool(settings.key(self.key_name))

    def timeout(self, settings: Settings) -> float:
        return settings.sources.timeout_s
