"""Operator footage library: a folder of video and image files, searched by keywords.

Each media file may have a sidecar json (<stem>.json or <file>.json) with
title, tags, licence, author, source_url, era, description. Search scores
the overlap between stemmed query tokens and the title, tags, era and
description, with the file name as a fallback. In offline mode, or when
nothing matches, a deterministic rotation of library items (hashed from
the shot id) fills the rest so every shot gets candidates; those results
carry the note "fixture rotation".
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..models import Candidate, Shot
from ..util import media_duration, video_size
from .base import BaseAdapter, Media, RawResult, link_into, make_candidates

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".ogv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
STOPWORDS = set("""a an the of and or in on at to for from by with into over under about as is are was were be been
being this that these those it its their there here very more most some any each other than then so such no not
only own same too can will just up down out off again once film footage video photo photograph image picture shot
clip stock hd 4k historical historic old new""".split())
FIELD_WEIGHTS = {"title": 3.0, "tags": 2.5, "era": 1.0, "description": 1.0}


def stem(word: str) -> str:
    """Tiny suffix stripper, good enough for keyword overlap."""
    w = word.lower()
    for suf in ("ings", "ing", "ies", "ers", "ed", "es", "er", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[: -len(suf)]
            return w + "y" if suf == "ies" else w
    return w


def tokens(text: str) -> set[str]:
    return {stem(t) for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in STOPWORDS and len(t) > 1}


class LibraryItem:
    """One media file plus its sidecar metadata and token index."""

    def __init__(self, path: Path, meta: dict[str, Any]) -> None:
        self.path = path
        self.meta = meta
        self.media: Media = "video" if path.suffix.lower() in VIDEO_EXT else "image"
        tags = meta.get("tags") or []
        self.fields = {
            "title": tokens(str(meta.get("title", ""))),
            "tags": tokens(" ".join(tags) if isinstance(tags, list) else str(tags)),
            "era": tokens(str(meta.get("era", ""))),
            "description": tokens(str(meta.get("description", ""))),
        }
        self.name_tokens = tokens(path.stem.replace("_", " "))
        self._probe: Optional[tuple[float, int, int]] = None

    def score(self, query: set[str]) -> float:
        """Weighted overlap in 0..1 against the query tokens."""
        if not query:
            return 0.0
        total = 0.0
        for tok in query:
            best = max((w for f, w in FIELD_WEIGHTS.items() if tok in self.fields[f]), default=0.0)
            if not best and tok in self.name_tokens:
                best = 0.8
            total += best
        return round(total / (3.0 * len(query)), 4)

    def probe(self) -> tuple[float, int, int]:
        """(duration, width, height), from the sidecar when present, else ffprobe or Pillow."""
        if self._probe is None:
            dur = float(self.meta.get("duration") or 0)
            w, h = int(self.meta.get("width") or 0), int(self.meta.get("height") or 0)
            if self.media == "video" and (not dur or not w):
                dur, (w, h) = media_duration(self.path), video_size(self.path)
            elif self.media == "image" and not w:
                from PIL import Image
                with Image.open(self.path) as im:
                    w, h = im.size
            self._probe = (dur, w, h)
        return self._probe


def _read_sidecar(path: Path) -> dict[str, Any]:
    for cand in (path.with_name(path.name + ".json"), path.with_suffix(".json")):
        if cand.exists():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except (OSError, ValueError):
                return {}
    return {}


_index_cache: dict[str, tuple[str, list[LibraryItem]]] = {}
_index_lock = threading.Lock()


def library_index(folder: Path) -> list[LibraryItem]:
    """All media files under folder (recursive), cached until the folder changes."""
    if not folder.is_dir():
        return []
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT | IMAGE_EXT)
    sig = hashlib.sha1("|".join(f"{p}:{p.stat().st_mtime_ns}" for p in files).encode()).hexdigest()
    key = str(folder.resolve())
    with _index_lock:
        hit = _index_cache.get(key)
        if hit and hit[0] == sig:
            return hit[1]
        items = [LibraryItem(p, _read_sidecar(p)) for p in files]
        _index_cache[key] = (sig, items)
        return items


def _hash_int(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode()).hexdigest()[:12], 16)


def rotation(items: list[LibraryItem], salt: str, n: int) -> list[LibraryItem]:
    """Deterministic rotation of items starting at a hash of salt (the shot id)."""
    if not items:
        return []
    ordered = sorted(items, key=lambda it: it.path.name)
    start = _hash_int(salt) % len(ordered)
    step = 1 + 2 * (_hash_int(salt, "step") % max(1, len(ordered) // 3))
    while len(ordered) > 1 and _gcd(step, len(ordered)) != 1:
        step += 1
    return [ordered[(start + i * step) % len(ordered)] for i in range(min(n, len(ordered)))]


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def clip_span(duration: float, shot: Shot, key: str) -> tuple[float, float]:
    """(in, out) covering the shot duration, offset deterministically per shot and file."""
    need = max(0.5, shot.duration)
    if duration <= need:
        return 0.0, round(duration, 3)
    slack = duration - need
    offset = (_hash_int(shot.id, key) % 1000) / 1000.0 * slack
    return round(offset, 3), round(offset + need, 3)


def is_bundled_library(lib: Path, settings: Settings) -> bool:
    return lib.resolve() == (Path(settings.repo_root) / "fixtures" / "sample_assets").resolve()


def real_judge(settings: Settings) -> bool:
    """True when the judge seat would call a real provider (not offline, a configured provider exists)."""
    if settings.offline:
        return False
    from ..providers.base import CostTracker
    from ..providers.seats import Seats
    return Seats(settings, CostTracker("availability")).is_real("judge")


class LocalAdapter(BaseAdapter):
    name = "local"
    media = {"video", "image"}

    def available(self, settings: Settings) -> bool:
        lib = settings.path(settings.sources.local_library)
        if not lib.is_dir():
            return False
        # the bundled synthetic library (labelled SAMPLE FOOTAGE) exists for mock runs only: it must never
        # reach a production whose judge is a real model
        return not (is_bundled_library(lib, settings) and real_judge(settings))

    def items(self, settings: Settings, media: Media) -> list[LibraryItem]:
        return [it for it in library_index(settings.path(settings.sources.local_library)) if it.media == media]

    def search(self, query: str, media: Media, n: int, settings: Settings,
               shot: Optional[Shot] = None) -> list[RawResult]:
        """Keyword matches first; rotation filler offline or when nothing matched."""
        pool = self.items(settings, media)
        q = tokens(query)
        salt = shot.id if shot else query
        # equal scores are ordered per shot so neighbouring shots see different items first
        scored = sorted(((it.score(q), it) for it in pool), key=lambda p: (-p[0], _hash_int(salt, p[1].path.name)))
        matches = [(s, it) for s, it in scored if s > 0.15][:n]
        out = [self._raw(it, s, "") for s, it in matches]
        if settings.offline or not out:
            seen = {it.path for _, it in matches}
            for it in rotation(pool, f"{salt}:{media}", len(pool)):
                if len(out) >= n:
                    break
                if it.path not in seen:
                    out.append(self._raw(it, 0.0, "fixture rotation"))
        return out

    def _raw(self, it: LibraryItem, score: float, note: str) -> RawResult:
        m = it.meta
        return RawResult(source="local", url=str(it.path), title=str(m.get("title") or it.path.stem.replace("_", " ")),
                         author=str(m.get("author", "")), licence=str(m.get("licence", "unknown")), media=it.media,
                         page_url=str(m.get("source_url", "")), download_url=str(it.path), score=score, note=note,
                         extra={"tags": m.get("tags", [])})

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Settings) -> list[Candidate]:
        src = Path(raw.download_url or raw.url)
        item = next((it for it in library_index(settings.path(settings.sources.local_library)) if it.path == src), None)
        dur, w, h = item.probe() if item else (0.0, 0, 0)
        raw = raw.model_copy(update={"width": w, "height": h, "duration": dur})
        name = f"local_{src.stem[:40]}_{_hash_int(str(src)) % 0xFFFFFF:06x}{src.suffix.lower()}"
        dst = link_into(src, out_dir, name)
        spans = [clip_span(dur, shot, src.name)] if raw.media == "video" else None
        return make_candidates(raw, dst, shot, settings, spans=spans)
