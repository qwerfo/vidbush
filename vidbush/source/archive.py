"""Internet Archive adapter: movies and images via advancedsearch and the metadata API.

Search: advancedsearch.php with mediatype movies or image, fields
identifier, title, creator, licenseurl, date. Fetch: /metadata/<id> picks
the best mp4 (or jpg), licence from licenseurl, with public domain
detection from rights fields and a few curated public domain collections
(news and user upload collections are not among them: their items need an
explicit licenseurl or rights statement). Long or large films are read as a
section with ffmpeg instead of a full download. File lengths may be seconds,
MM:SS or HH:MM:SS.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..models import Candidate, Shot
from . import base
from .base import BaseAdapter, Media, RawResult, cached_download, link_into, make_candidates

SEARCH = "https://archive.org/advancedsearch.php"
METADATA = "https://archive.org/metadata/"
DOWNLOAD = "https://archive.org/download/"
PD_COLLECTIONS = {"prelinger", "publicdomainmovies", "fedflix", "nasa", "usgovernmentdocuments"}


def _first(v: Any) -> str:
    if isinstance(v, list):
        return str(v[0]) if v else ""
    return str(v or "")


def parse_search(data: dict[str, Any], media: Media) -> list[RawResult]:
    """RawResults from an advancedsearch JSON response."""
    docs = (data.get("response") or {}).get("docs") or []
    out = []
    for d in docs:
        ident = d.get("identifier")
        if not ident:
            continue
        out.append(RawResult(source="archive", url=f"https://archive.org/details/{ident}", title=_first(d.get("title")),
                             author=_first(d.get("creator")), licence=_first(d.get("licenseurl")) or "unknown",
                             media=media, page_url=f"https://archive.org/details/{ident}", score=0.7,
                             extra={"identifier": ident, "date": _first(d.get("date"))}))
    return out


def detect_licence(meta: dict[str, Any], licenseurl: str) -> str:
    """Licence string from licenseurl, rights text or collection membership."""
    if licenseurl:
        return licenseurl
    rights = " ".join(_first(meta.get(k)) for k in ("rights", "possible-copyright-status", "licenseurl")).lower()
    if "public domain" in rights or "not_in_copyright" in rights or "not in copyright" in rights:
        return "Public domain"
    colls = meta.get("collection") or []
    colls = {c.lower() for c in (colls if isinstance(colls, list) else [colls])}
    if colls & PD_COLLECTIONS:
        return "Public domain"
    return "unknown (Internet Archive)"


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def parse_length(v: Any) -> float:
    """Seconds from an Archive length: '93.5', '01:33' or '1:02:03'."""
    text = str(v or "").strip()
    if ":" not in text:
        return _num(text)
    total = 0.0
    for part in text.split(":"):
        p = _num(part)
        if p == 0.0 and part.strip() not in {"0", "00", "0.0"}:
            return 0.0
        total = total * 60 + p
    return total


def pick_file(files: list[dict[str, Any]], media: Media) -> Optional[dict[str, Any]]:
    """Best mp4 (highest resolution, then size) or largest original jpg."""
    if media == "video":
        pool = [f for f in files if str(f.get("name", "")).lower().endswith(".mp4")]
        key = lambda f: (_num(f.get("height")), _num(f.get("size")))  # noqa: E731
    else:
        pool = [f for f in files if str(f.get("name", "")).lower().endswith((".jpg", ".jpeg"))
                and "thumb" not in str(f.get("format", "")).lower() and "_thumb" not in str(f.get("name", ""))]
        key = lambda f: (_num(f.get("width")) * _num(f.get("height")), _num(f.get("size")))  # noqa: E731
    return max(pool, key=key) if pool else None


def resolve(raw: RawResult, meta_doc: dict[str, Any]) -> Optional[RawResult]:
    """Fill download_url, size, duration and licence from a metadata response."""
    f = pick_file(meta_doc.get("files") or [], raw.media)
    if not f:
        return None
    ident = raw.extra["identifier"]
    meta = meta_doc.get("metadata") or {}
    return raw.model_copy(update={
        "download_url": f"{DOWNLOAD}{ident}/{f['name']}",
        "width": int(_num(f.get("width"))), "height": int(_num(f.get("height"))),
        "duration": parse_length(f.get("length")), "licence": detect_licence(meta, raw.licence if raw.licence != "unknown" else ""),
        "author": raw.author or _first(meta.get("creator")),
        "extra": {**raw.extra, "size": _num(f.get("size"))},
    })


class ArchiveAdapter(BaseAdapter):
    name = "archive"
    media = {"video", "image"}

    def search(self, query: str, media: Media, n: int, settings: Settings,
               shot: Optional[Shot] = None) -> list[RawResult]:
        mt = "movies" if media == "video" else "image"
        params = {"q": f"({query}) AND mediatype:({mt})", "fl[]": ["identifier", "title", "creator", "licenseurl", "date"],
                  "rows": max(n, 5), "page": 1, "output": "json", "sort[]": "downloads desc"}
        data = base.http_get_json(SEARCH, params=params, timeout=self.timeout(settings))
        return parse_search(data, media)[:n]

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Settings) -> list[Candidate]:
        meta_doc = base.http_get_json(METADATA + raw.extra["identifier"], timeout=self.timeout(settings))
        full = resolve(raw, meta_doc)
        if full is None:
            return []
        url = full.download_url
        if full.media == "image":
            src = cached_download(url, ".jpg", settings, max_bytes=base.MAX_IMAGE_BYTES)
            return make_candidates(full, link_into(src, out_dir, f"archive_{base.url_hash(url)}.jpg"), shot, settings)
        src = base.cached_video(url, ".mp4", settings, shot.duration, duration=full.duration,
                                size=full.extra.get("size", 0.0))
        dst = link_into(src, out_dir, f"archive_{base.url_hash(src.name)}.mp4")
        spans = base.cut_clips(base.detect_scenes(dst), shot.duration, max_clips=3)
        return make_candidates(full, dst, shot, settings, spans=spans)
