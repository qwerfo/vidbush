"""Source adapter contract and shared helpers.

An adapter has a name, the media kinds it supports, an availability check,
search(query, media, n, settings, shot=None) -> list[RawResult] and
fetch(raw, out_dir, shot, settings) -> list[Candidate]. fetch may return
several candidates for one asset (a long video cut into clips). Candidate
ids returned by fetch are provisional: the registry renumbers them to
f"{shot.id}_{source}_{n}" and extracts keyframes.

All network access goes through http_get_json / http_download /
ffmpeg_section in this module so tests can monkeypatch one place. Every
request is rate limited per host and concurrency capped (source/throttle.py);
a 429 backs off using Retry-After, is retried, and cools the host down.
Downloads stream to disk with a size cap and land in a shared cache
(settings.source_cache_folder, keyed by URL hash); videos too big for a full
download are read as a section with ffmpeg (cached_video). Files are
hardlinked or copied into the project, each with a <file>.json sidecar.
configure(settings) sets the User-Agent contact (VIDBUSH_CONTACT) and the
throttle overrides; the registry calls it once per sourcing run.
"""
from __future__ import annotations

import re

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
from ..providers.base import RateLimitError, TransportError, http_error_to_exc
from ..util import ToolError, clamp, run, write_json
from . import throttle
from .licence import licence_flags, normalise_licence

Media = Literal["video", "image"]
DEFAULT_USER_AGENT = "Vidbush/1.0 (self-hosted documentary tool; +https://github.com/qwerfo/vidbush)"
USER_AGENT = DEFAULT_USER_AGENT
MAX_DOWNLOAD_BYTES = 400_000_000  # hard cap for any streamed download
SECTION_OVER_BYTES = 150_000_000  # videos above this are read as a section, never downloaded whole
MAX_IMAGE_BYTES = 60_000_000
RATE_LIMIT_RETRIES = 2
RATE_LIMIT_MAX_WAIT_S = 60.0
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


# ---------------------------------------------------------------- relevance

GENERIC_WORDS = {
    "footage", "archival", "archive", "film", "films", "video", "videos", "photo", "photos", "photograph",
    "photographs", "image", "images", "picture", "historic", "historical", "history", "old", "vintage",
    "black", "white", "aerial", "view", "shot", "scene", "documentary", "newsreel", "public", "domain",
    "color", "colour", "early", "rare", "real", "original", "the", "and", "with", "from", "for", "into",
}


def _stem(word: str) -> str:
    w = word.lower()
    for suf in ("ing", "ion", "es", "s"):
        if len(w) > 5 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_tokens(text: str) -> set[str]:
    """Stemmed words of 3+ letters that are not generic footage words."""
    return {_stem(w) for w in re.findall(r"[A-Za-z]{3,}", text or "") if w.lower() not in GENERIC_WORDS}


def relevance(query: str, raw: "RawResult") -> float:
    """Share of the query's specific words found in the result title, description and tags (0..1)."""
    want = content_tokens(query)
    if not want:
        return 1.0
    extra = raw.extra or {}
    have = content_tokens(" ".join([raw.title, str(extra.get("description", "")), " ".join(map(str, extra.get("tags", []) or []))]))
    return len(want & have) / len(want)


# ---------------------------------------------------------------- network seams

def user_agent_for(contact: str) -> str:
    """User-Agent with the operator contact (Wikimedia and loc.gov ask for one)."""
    contact = (contact or "").strip()
    return f"Vidbush/1.0 (self-hosted documentary tool; {contact})" if contact else DEFAULT_USER_AGENT


HTTP_PROXY = ""  # ffmpeg does not read proxy env vars; passed explicitly when set
YOUTUBE_COOKIES = ""  # path to a Netscape cookies.txt exported from a logged-in browser
YOUTUBE_COOKIES_FROM_BROWSER = ""  # or a browser name yt-dlp reads cookies from: chrome, firefox, edge


def configure(settings: Settings) -> None:
    """Set the User-Agent contact and the throttle overrides for this process."""
    global USER_AGENT, HTTP_PROXY
    USER_AGENT = user_agent_for(settings.key("VIDBUSH_CONTACT"))
    global YOUTUBE_COOKIES, YOUTUBE_COOKIES_FROM_BROWSER
    HTTP_PROXY = settings.key("HTTPS_PROXY") or settings.key("HTTP_PROXY")
    cookies = settings.key("YOUTUBE_COOKIES")
    YOUTUBE_COOKIES = str(settings.path(cookies)) if cookies else ""
    YOUTUBE_COOKIES_FROM_BROWSER = settings.key("YOUTUBE_COOKIES_FROM_BROWSER")
    throttle.configure(settings.sources)


def _rate_limited(call: Any, url: str) -> Any:
    """Run call() in a throttle slot for url's host; 429 waits Retry-After, retries, then cools the host."""
    host = throttle.host_of(url)
    for n in range(RATE_LIMIT_RETRIES + 1):
        try:
            with throttle.slot(host):
                return call()
        except RateLimitError as e:
            wait = e.retry_after if e.retry_after is not None else 5.0 * (2 ** n)
            throttle.cool_down(host, max(wait, 5.0))
            if n == RATE_LIMIT_RETRIES or wait > RATE_LIMIT_MAX_WAIT_S:
                raise
    raise AssertionError("unreachable")


def http_get_json(url: str, params: Optional[dict[str, Any]] = None, headers: Optional[dict[str, str]] = None,
                  timeout: float = 20.0) -> Any:
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    return _rate_limited(lambda: provider_http.get_json(url, params=params, headers=hdrs, timeout=timeout), url)


def http_get_bytes(url: str, headers: Optional[dict[str, str]] = None, timeout: float = 60.0) -> bytes:
    """Small downloads held in memory (kept for callers outside sourcing; sourcing streams to disk)."""
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    return _rate_limited(lambda: provider_http.get_bytes(url, headers=hdrs, timeout=max(timeout, 60.0)), url)


class DownloadTooLarge(TransportError):
    """The asset is over the size cap; size is the Content-Length or the bytes read so far."""

    def __init__(self, size: int) -> None:
        super().__init__(f"download too large ({size} bytes)")
        self.size = size


def http_download(url: str, dest: Path, headers: Optional[dict[str, str]] = None, timeout: float = 60.0,
                  max_bytes: int = MAX_DOWNLOAD_BYTES) -> Path:
    """Stream url into dest. Raises DownloadTooLarge (dest removed) when over max_bytes."""
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    return _rate_limited(lambda: _stream(url, dest, hdrs, max(timeout, 60.0), max_bytes), url)


def _stream(url: str, dest: Path, headers: dict[str, str], timeout: float, max_bytes: int) -> Path:
    import requests
    try:
        with requests.get(url, headers=headers, timeout=timeout, stream=True) as r:
            if r.status_code >= 400:
                raise http_error_to_exc(r.status_code, r.text[:300], r.headers.get("Retry-After"))
            length = int(r.headers.get("Content-Length") or 0)
            if length > max_bytes:
                raise DownloadTooLarge(length)
            got = 0
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 16):
                    got += len(chunk)
                    if got > max_bytes:
                        raise DownloadTooLarge(got)
                    fh.write(chunk)
        return dest
    except requests.RequestException as e:
        dest.unlink(missing_ok=True)
        raise TransportError(str(e)[:300]) from e
    except BaseException:
        dest.unlink(missing_ok=True)
        raise


def ffmpeg_section(url: str, start: float, length: float, out: Path) -> Path:
    """Copy one section of a remote (or local) video without downloading the rest."""
    proxy = ["-http_proxy", HTTP_PROXY] if HTTP_PROXY and "://" in url else []
    def call() -> Path:
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{length:.2f}",
             "-user_agent", USER_AGENT, *proxy, "-i", url, "-map", "0:v:0", "-c", "copy", "-avoid_negative_ts",
             "make_zero", str(out)], timeout=600)
        return out
    if "://" not in url:
        return call()
    with throttle.slot(throttle.host_of(url), max_wait=RATE_LIMIT_MAX_WAIT_S):
        return call()


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


def cached_download(url: str, ext: str, settings: Settings, headers: Optional[dict[str, str]] = None,
                    max_bytes: int = MAX_DOWNLOAD_BYTES) -> Path:
    """Stream url once into the shared cache; later calls reuse the file. DownloadTooLarge over max_bytes."""
    dest = cache_dir(settings) / f"{url_hash(url)}{ext}"
    with _lock_for(str(dest)):
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        tmp = dest.with_suffix(dest.suffix + ".part")
        http_download(url, tmp, headers=headers, timeout=settings.sources.timeout_s, max_bytes=max_bytes)
        tmp.replace(dest)
    return dest


def cached_section(url: str, start: float, length: float, settings: Settings, ext: str = ".mp4") -> Path:
    """ffmpeg_section through the shared cache, keyed by url and range."""
    dest = cache_dir(settings) / f"{url_hash(f'{url}#{start:.1f}+{length:.1f}')}{ext}"
    with _lock_for(str(dest)):
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        tmp = dest.with_name(dest.stem + ".part" + ext)
        try:
            ffmpeg_section(url, start, length, tmp)
        except ToolError:
            if "://" not in url:
                raise
            whole = cached_download(url, ext, settings, max_bytes=MAX_DOWNLOAD_BYTES)
            ffmpeg_section(str(whole), start, length, tmp)
        tmp.replace(dest)
    return dest


def section_window(duration: float, shot_duration: float) -> tuple[float, float]:
    """(start, length) of the part of a long video read as a section: skips the opening tenth."""
    length = clamp(shot_duration * 4, 40.0, 90.0)
    if duration <= 0:
        return 20.0, length
    return clamp(duration * 0.1, 0.0, max(0.0, duration - length)), min(length, duration)


def cached_video(url: str, ext: str, settings: Settings, shot_duration: float, duration: float = 0.0,
                 size: float = 0.0) -> Path:
    """A video in the cache: whole when small, else (size known, or found while streaming) a section."""
    if size <= SECTION_OVER_BYTES:
        try:
            return cached_download(url, ext, settings, max_bytes=SECTION_OVER_BYTES)
        except DownloadTooLarge:
            pass
    start, length = section_window(duration, shot_duration)
    return cached_section(url, start, length, settings, ext=ext)


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
