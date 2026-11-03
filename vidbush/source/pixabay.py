"""Pixabay adapter: stock videos and photos (needs PIXABAY_API_KEY). Licence "Pixabay Content License"."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..models import Candidate, Shot
from . import base
from .base import BaseAdapter, Media, RawResult, cached_download, link_into, make_candidates
from .pexels import stock_spans

PHOTOS = "https://pixabay.com/api/"
VIDEOS = "https://pixabay.com/api/videos/"
LICENCE = "Pixabay Content License"


def _title(hit: dict[str, Any], kind: str) -> str:
    tags = [t.strip() for t in str(hit.get("tags", "")).split(",") if t.strip()]
    return (", ".join(tags[:3]).capitalize() if tags else f"Pixabay {kind} {hit.get('id')}")


def parse_videos(data: dict[str, Any], settings: Settings) -> list[RawResult]:
    out = []
    for hit in data.get("hits") or []:
        renditions = [v for v in (hit.get("videos") or {}).values() if isinstance(v, dict) and v.get("url")]
        ok = [v for v in renditions if settings.sources.youtube_min_height <= int(v.get("height") or 0)
              <= settings.sources.youtube_max_height]
        if not ok:
            continue
        best = max(ok, key=lambda v: int(v.get("height") or 0))
        out.append(RawResult(source="pixabay", url=hit.get("pageURL", ""), page_url=hit.get("pageURL", ""),
                             title=_title(hit, "video"), author=hit.get("user", ""), licence=LICENCE, media="video",
                             width=int(best.get("width") or 0), height=int(best.get("height") or 0),
                             duration=float(hit.get("duration") or 0), download_url=best["url"], score=0.6))
    return out


def parse_photos(data: dict[str, Any]) -> list[RawResult]:
    out = []
    for hit in data.get("hits") or []:
        url = hit.get("fullHDURL") or hit.get("largeImageURL")
        if not url:
            continue
        w, h = int(hit.get("imageWidth") or 0), int(hit.get("imageHeight") or 0)
        scale = min(1.0, 1920 / w) if w else 1.0
        out.append(RawResult(source="pixabay", url=hit.get("pageURL", ""), page_url=hit.get("pageURL", ""),
                             title=_title(hit, "photo"), author=hit.get("user", ""), licence=LICENCE, media="image",
                             width=int(w * scale), height=int(h * scale), download_url=url, score=0.6))
    return out


class PixabayAdapter(BaseAdapter):
    name = "pixabay"
    media = {"video", "image"}
    key_name = "PIXABAY_API_KEY"

    def search(self, query: str, media: Media, n: int, settings: Settings,
               shot: Optional[Shot] = None) -> list[RawResult]:
        params = {"key": settings.key(self.key_name), "q": query[:100], "per_page": max(n, 5), "safesearch": "true"}
        if media == "video":
            data = base.http_get_json(VIDEOS, params=params, timeout=self.timeout(settings))
            return parse_videos(data, settings)[:n]
        params.update({"image_type": "photo", "orientation": "horizontal"})
        data = base.http_get_json(PHOTOS, params=params, timeout=self.timeout(settings))
        return parse_photos(data)[:n]

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Settings) -> list[Candidate]:
        ext = ".mp4" if raw.media == "video" else ".jpg"
        src = cached_download(raw.download_url, ext, settings)
        dst = link_into(src, out_dir, f"pixabay_{base.url_hash(raw.download_url)}{ext}")
        spans = stock_spans(raw.duration, shot) if raw.media == "video" else None
        return make_candidates(raw, dst, shot, settings, spans=spans)
