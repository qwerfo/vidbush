"""US National Archives catalogue adapter: photographs and films (moving images) through API v2.

Search: https://catalog.archives.gov/api/v2/records/search?q=...&limit=N with typeOfMaterials
"Photographs and other Graphic Materials" or "Moving Images" and availableOnline=true. The API needs
the x-api-key header (NARA_API_KEY): without it the same URL answers HTTP 200 with the catalogue's web
page instead of JSON, so the adapter is available only with the key.

Licence: only records whose use and access restrictions are both "Unrestricted" are returned, as
public domain federal records. "Restricted - Possibly" (copyright of a donor or contractor) and
"Undetermined" records are left out, because the licence code has no restricted string it flags.

Each record lists its digital objects (objectUrl, objectType, objectFileSize). A photo takes its
largest JPEG (or TIFF) within base.MAX_IMAGE_BYTES; a film takes its largest MP4, read whole when
small and as a section when large (base.cached_video, like loc.py). Film records whose only objects
are scanned shot list sheets (JPEG, PDF) are skipped for video.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..models import Candidate, Shot
from . import base
from .base import BaseAdapter, Media, RawResult, cached_download, link_into, make_candidates

SEARCH = "https://catalog.archives.gov/api/v2/records/search"
PHOTOS = "Photographs and other Graphic Materials"
FILMS = "Moving Images"
LICENCE = "Public domain (US National Archives, unrestricted federal record)"
IMAGE_TYPES = ("image (jpg)", "image (jpeg)", "image (tiff)")
MIN_VIDEO_BYTES = 1_000_000  # smaller MP4 objects are previews of a second or two
OVERFETCH = 3  # restricted records are dropped after the search, so ask for more


def _status(record: dict[str, Any], key: str) -> str:
    v = record.get(key) or {}
    return str(v.get("status") if isinstance(v, dict) else v or "").strip().lower()


def unrestricted(record: dict[str, Any]) -> bool:
    """True when both the use and the access restriction say Unrestricted."""
    return _status(record, "useRestriction") == "unrestricted" and _status(record, "accessRestriction") == "unrestricted"


def best_object(record: dict[str, Any], media: Media) -> Optional[dict[str, Any]]:
    """Largest usable digital object: a JPEG before a TIFF under the image cap, or the largest MP4."""
    objs = [o for o in record.get("digitalObjects") or [] if isinstance(o, dict) and o.get("objectUrl")]
    size = lambda o: int(o.get("objectFileSize") or 0)  # noqa: E731
    kind = lambda o: str(o.get("objectType") or "").lower()  # noqa: E731
    if media == "video":
        pool = [o for o in objs if "mp4" in kind(o) and size(o) >= MIN_VIDEO_BYTES]
        return max(pool, key=size, default=None)
    pool = [o for o in objs if kind(o) in IMAGE_TYPES and 0 < size(o) <= base.MAX_IMAGE_BYTES]
    return max(pool, key=lambda o: ("tiff" not in kind(o), size(o)), default=None)


def _year(record: dict[str, Any]) -> str:
    for key in ("productionDates", "coverageStartDate"):
        v = record.get(key)
        v = v[0] if isinstance(v, list) and v else v
        if isinstance(v, dict) and v.get("year"):
            return str(v["year"])
    return ""


def parse_search(data: dict[str, Any], media: Media) -> list[RawResult]:
    """RawResults from an API v2 records search: unrestricted records with a usable object only."""
    body = data.get("body", data) if isinstance(data, dict) else {}
    out = []
    for hit in ((body.get("hits") or {}).get("hits")) or []:
        record = (hit.get("_source") or {}).get("record") or {}
        obj = best_object(record, media)
        if not obj or not unrestricted(record):
            continue
        na_id = str(record.get("naId") or hit.get("_id") or "")
        page = f"https://catalog.archives.gov/id/{na_id}"
        year = _year(record)
        title = str(record.get("title") or f"National Archives {na_id}")
        out.append(RawResult(source="nara", url=page, page_url=page, title=f"{title} ({year})" if year else title,
                             author="US National Archives", licence=LICENCE, media=media,
                             download_url=str(obj["objectUrl"]), score=0.75,
                             extra={"naId": na_id, "size": float(obj.get("objectFileSize") or 0),
                                    "description": str(record.get("scopeAndContentNote") or "")[:400]}))
    return out


class NaraAdapter(BaseAdapter):
    name = "nara"
    media = {"video", "image"}
    key_name = "NARA_API_KEY"
    keyword_words = 3  # the catalogue search matches every word
    classes = frozenset({"archive"})
    matches_description = True  # the catalogue matches scope notes and descriptions

    def search(self, query: str, media: Media, n: int, settings: Settings,
               shot: Optional[Shot] = None) -> list[RawResult]:
        """Subject and year first (base.archive_queries), the subject alone only when that finds nothing."""
        headers = {"x-api-key": settings.key(self.key_name), "Accept": "application/json"}
        out: list[RawResult] = []
        for q in base.archive_queries(query, self.keyword_words):
            params = {"q": q, "limit": max(n, 5) * OVERFETCH,
                      "typeOfMaterials": FILMS if media == "video" else PHOTOS, "availableOnline": "true"}
            data = base.http_get_json(SEARCH, params=params, headers=headers, timeout=self.timeout(settings))
            out = parse_search(data, media)[:n]
            if out:
                break
        return out

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Settings) -> list[Candidate]:
        url = raw.download_url
        if raw.media == "video":
            src = base.cached_video(url, ".mp4", settings, shot.duration, size=raw.extra.get("size", 0.0),
                                    shot=shot, title=raw.title)
            dst = link_into(src, out_dir, f"nara_{base.url_hash(src.name)}.mp4")
            spans = base.cut_clips(base.focused_scenes(dst, shot, settings, url, raw.title), shot.duration, max_clips=3)
            return make_candidates(raw, dst, shot, settings, spans=spans)
        ext = base.ext_from_url(url, ".jpg")
        src = cached_download(url, ext, settings, max_bytes=base.MAX_IMAGE_BYTES)
        dst = link_into(src, out_dir, f"nara_{base.url_hash(url)}{ext}")
        return make_candidates(raw, dst, shot, settings)
