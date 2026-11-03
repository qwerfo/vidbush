"""NASA Image and Video Library adapter (images-api.nasa.gov).

Search: /search?q=...&media_type=image|video. Fetch: the asset manifest
(/asset/<nasa_id>) gives every rendition. ~large is preferred (originals are
often broadcast masters of several GB or images over 100 MP); ~orig is used
only when no large or medium rendition exists. Videos over the full download
size are read as a section. Licence "Public domain (NASA)".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..models import Candidate, Shot
from . import base
from .base import BaseAdapter, Media, RawResult, cached_download, link_into, make_candidates

SEARCH = "https://images-api.nasa.gov/search"
ASSET = "https://images-api.nasa.gov/asset/"
LICENCE = "Public domain (NASA)"
IMAGE_PREF = ("~large.jpg", "~medium.jpg", "~orig.jpg")
VIDEO_PREF = ("~large.mp4", "~medium.mp4", "~orig.mp4", "~mobile.mp4")


def parse_search(data: dict[str, Any], media: Media) -> list[RawResult]:
    out = []
    for item in (data.get("collection") or {}).get("items") or []:
        d = (item.get("data") or [{}])[0]
        nasa_id = d.get("nasa_id")
        if not nasa_id or d.get("media_type") != media:
            continue
        author = d.get("photographer") or d.get("secondary_creator") or d.get("center") or "NASA"
        page = f"https://images.nasa.gov/details/{nasa_id}"
        out.append(RawResult(source="nasa", url=page, page_url=page, title=d.get("title", nasa_id), author=author,
                             licence=LICENCE, media=media, score=0.7, extra={"nasa_id": nasa_id}))
    return out


def manifest_urls(data: Any) -> list[str]:
    """Asset URLs from either an /asset response or a collection.json list."""
    if isinstance(data, list):
        return [str(u) for u in data]
    return [i.get("href", "") for i in (data.get("collection") or {}).get("items") or []]


def pick_asset(urls: list[str], media: Media) -> str:
    for suffix in IMAGE_PREF if media == "image" else VIDEO_PREF:
        for u in urls:
            if u.lower().endswith(suffix):
                return u.replace("http://", "https://", 1)
    return ""


class NasaAdapter(BaseAdapter):
    name = "nasa"
    media = {"video", "image"}

    def search(self, query: str, media: Media, n: int, settings: Settings,
               shot: Optional[Shot] = None) -> list[RawResult]:
        data = base.http_get_json(SEARCH, params={"q": query, "media_type": media, "page_size": max(n, 5)},
                                  timeout=self.timeout(settings))
        return parse_search(data, media)[:n]

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Settings) -> list[Candidate]:
        data = base.http_get_json(ASSET + raw.extra["nasa_id"], timeout=self.timeout(settings))
        url = pick_asset(manifest_urls(data), raw.media)
        if not url:
            return []
        raw = raw.model_copy(update={"download_url": url})
        if raw.media == "video":
            ext, src = ".mp4", base.cached_video(url, ".mp4", settings, shot.duration)
        else:
            ext, src = ".jpg", cached_download(url, ".jpg", settings, max_bytes=base.MAX_IMAGE_BYTES)
        dst = link_into(src, out_dir, f"nasa_{base.url_hash(src.name)}{ext}")
        spans = base.cut_clips(base.detect_scenes(dst), shot.duration, max_clips=3) if raw.media == "video" else None
        return make_candidates(raw, dst, shot, settings, spans=spans)
