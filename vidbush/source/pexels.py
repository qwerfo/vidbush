"""Pexels adapter: stock videos and photos (needs PEXELS_API_KEY). Licence "Pexels License"."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..models import Candidate, Shot
from . import base
from .base import BaseAdapter, Media, RawResult, cached_download, link_into, make_candidates

VIDEOS = "https://api.pexels.com/videos/search"
PHOTOS = "https://api.pexels.com/v1/search"
LICENCE = "Pexels License"


def best_video_file(files: list[dict[str, Any]], min_h: int, max_h: int) -> Optional[dict[str, Any]]:
    """Largest mp4 rendition within [min_h, max_h] lines; else the smallest one above max_h."""
    mp4 = [f for f in files if f.get("file_type") == "video/mp4" and f.get("link")]
    inside = [f for f in mp4 if min_h <= min(int(f.get("height") or 0), int(f.get("width") or 0)) <= max_h]
    if inside:
        return max(inside, key=lambda f: int(f.get("height") or 0) * int(f.get("width") or 0))
    above = [f for f in mp4 if min(int(f.get("height") or 0), int(f.get("width") or 0)) > max_h]
    return min(above, key=lambda f: int(f.get("height") or 0)) if above else None


def parse_videos(data: dict[str, Any], settings: Settings) -> list[RawResult]:
    out = []
    for v in data.get("videos") or []:
        f = best_video_file(v.get("video_files") or [], settings.sources.youtube_min_height,
                            settings.sources.youtube_max_height)
        if not f:
            continue
        user = v.get("user") or {}
        title = _title_from_url(v.get("url", "")) or f"Pexels video {v.get('id')}"
        out.append(RawResult(source="pexels", url=v.get("url", ""), page_url=v.get("url", ""), title=title,
                             author=user.get("name", ""), licence=LICENCE, media="video",
                             width=int(f.get("width") or 0), height=int(f.get("height") or 0),
                             duration=float(v.get("duration") or 0), download_url=f["link"], score=0.6))
    return out


def parse_photos(data: dict[str, Any]) -> list[RawResult]:
    out = []
    for p in data.get("photos") or []:
        src = p.get("src") or {}
        original = src.get("original", "")
        if not original:
            continue
        w, h = int(p.get("width") or 0), int(p.get("height") or 0)
        url = f"{original}?auto=compress&cs=tinysrgb&w=2400" if w > 2400 else original
        out.append(RawResult(source="pexels", url=p.get("url", ""), page_url=p.get("url", ""),
                             title=p.get("alt") or _title_from_url(p.get("url", "")) or f"Pexels photo {p.get('id')}",
                             author=p.get("photographer", ""), licence=LICENCE, media="image",
                             width=min(w, 2400), height=int(h * min(1.0, 2400 / w)) if w else h,
                             download_url=url, score=0.6))
    return out


def _title_from_url(url: str) -> str:
    """'https://www.pexels.com/video/ships-at-sea-123/' -> 'Ships at sea'."""
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    words = [w for w in slug.split("-") if not w.isdigit()]
    return " ".join(words).capitalize()


def stock_spans(duration: float, shot: Shot) -> list[tuple[float, float]]:
    """Clip points for a short single-take stock video."""
    if duration <= 0:
        return [(0.0, 0.0)]
    return base.cut_clips([(0.0, duration)], shot.duration, max_clips=2) or [(0.0, duration)]


class PexelsAdapter(BaseAdapter):
    name = "pexels"
    media = {"video", "image"}
    key_name = "PEXELS_API_KEY"

    def search(self, query: str, media: Media, n: int, settings: Settings,
               shot: Optional[Shot] = None) -> list[RawResult]:
        headers = {"Authorization": settings.key(self.key_name)}
        params = {"query": query, "per_page": max(n, 5), "orientation": "landscape"}
        if media == "video":
            data = base.http_get_json(VIDEOS, params={**params, "size": "medium"}, headers=headers,
                                      timeout=self.timeout(settings))
            return parse_videos(data, settings)[:n]
        data = base.http_get_json(PHOTOS, params=params, headers=headers, timeout=self.timeout(settings))
        return parse_photos(data)[:n]

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Settings) -> list[Candidate]:
        ext = ".mp4" if raw.media == "video" else ".jpg"
        cap = base.SECTION_OVER_BYTES if raw.media == "video" else base.MAX_IMAGE_BYTES
        src = cached_download(raw.download_url, ext, settings, max_bytes=cap)
        dst = link_into(src, out_dir, f"pexels_{base.url_hash(raw.download_url)}{ext}")
        spans = stock_spans(raw.duration, shot) if raw.media == "video" else None
        return make_candidates(raw, dst, shot, settings, spans=spans)
