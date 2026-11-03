"""Wikimedia Commons adapter: stills and videos (webm, ogv) from the File namespace.

Search: action=query, generator=search in namespace 6, prop=imageinfo with
iiprop=url|size|mime|extmetadata and iiurlwidth=1920. Licence comes from
extmetadata LicenseShortName, author from Artist with HTML stripped.
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..models import Candidate, Shot
from . import base
from .base import BaseAdapter, Media, RawResult, cached_download, ext_from_url, link_into, make_candidates

API = "https://commons.wikimedia.org/w/api.php"
IMAGE_MIME = {"image/jpeg", "image/png", "image/webp", "image/tiff"}
VIDEO_MIME = {"video/webm", "application/ogg", "video/ogg", "video/mp4"}


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _meta(ext: dict[str, Any], key: str) -> str:
    v = ext.get(key) or {}
    return strip_html(str(v.get("value", ""))) if isinstance(v, dict) else ""


def parse_commons(data: dict[str, Any], media: Media) -> list[RawResult]:
    """RawResults from a Commons generator=search response."""
    pages = (data.get("query") or {}).get("pages") or {}
    items = pages.values() if isinstance(pages, dict) else pages
    out: list[tuple[int, RawResult]] = []
    for page in items:
        info = (page.get("imageinfo") or [{}])[0]
        mime = info.get("mime", "")
        if (media == "image" and mime not in IMAGE_MIME) or (media == "video" and mime not in VIDEO_MIME):
            continue
        ext = info.get("extmetadata") or {}
        title = _meta(ext, "ObjectName") or re.sub(r"^File:|\.[a-z0-9]+$", "", page.get("title", ""), flags=re.I)
        width, height = int(info.get("width") or 0), int(info.get("height") or 0)
        big = media == "image" and width > 1920 and info.get("thumburl")
        out.append((int(page.get("index", 0)), RawResult(
            source="wikimedia", url=info.get("url", ""), title=title, author=_meta(ext, "Artist"),
            licence=_meta(ext, "LicenseShortName") or _meta(ext, "License") or "unknown", media=media,
            width=int(info.get("thumbwidth") or width) if big else width,
            height=int(info.get("thumbheight") or height) if big else height,
            duration=float(info.get("duration") or 0.0),
            download_url=info.get("thumburl") if big else info.get("url", ""),
            page_url=info.get("descriptionurl", ""), score=0.8,
        )))
    return [r for _, r in sorted(out, key=lambda p: p[0])]


class WikimediaAdapter(BaseAdapter):
    name = "wikimedia"
    media = {"video", "image"}

    def search(self, query: str, media: Media, n: int, settings: Settings,
               shot: Optional[Shot] = None) -> list[RawResult]:
        search = f"{query} filetype:video" if media == "video" else f"{query} filetype:bitmap"
        params = {"action": "query", "format": "json", "generator": "search", "gsrsearch": search,
                  "gsrnamespace": 6, "gsrlimit": max(n, 5), "prop": "imageinfo",
                  "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1920}
        data = base.http_get_json(API, params=params, timeout=self.timeout(settings))
        return parse_commons(data, media)[:n]

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Settings) -> list[Candidate]:
        url = raw.download_url or raw.url
        ext = ext_from_url(url, ".webm" if raw.media == "video" else ".jpg")
        if raw.media == "video":
            src = base.cached_video(url, ext, settings, shot.duration, duration=raw.duration)
        else:
            src = cached_download(url, ext, settings, max_bytes=base.MAX_IMAGE_BYTES)
        dst = link_into(src, out_dir, f"wikimedia_{base.url_hash(src.name)}{ext}")
        spans = None
        if raw.media == "video":
            spans = base.cut_clips(base.detect_scenes(dst), shot.duration, max_clips=3)
        return make_candidates(raw, dst, shot, settings, spans=spans)
