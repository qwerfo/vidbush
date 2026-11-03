"""Library of Congress adapter: photos and films through the loc.gov JSON API.

Search: https://www.loc.gov/photos/?q=...&fo=json (or /film-and-videos/).
Fetch: the item JSON (<item url>?fo=json) lists resource files; the largest
jpeg (or an mp4) is downloaded. Rights statements become licence strings.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..models import Candidate, Shot
from . import base
from .base import BaseAdapter, Media, RawResult, cached_download, link_into, make_candidates

PHOTOS = "https://www.loc.gov/photos/"
FILMS = "https://www.loc.gov/film-and-videos/"


def rights_to_licence(rights: str) -> str:
    """Map an LoC rights statement to a licence string."""
    low = (rights or "").lower()
    if not low:
        return "unknown (Library of Congress)"
    if "no known restrictions" in low or "public domain" in low or "no known copyright" in low:
        return "Public domain (Library of Congress, no known restrictions)"
    if "creativecommons.org" in low or re.search(r"\bcc[\s-]*by", low):
        return rights
    return "Rights restricted (Library of Congress, see item page)"


def _text(v: Any) -> str:
    if isinstance(v, list):
        return "; ".join(str(x) for x in v[:2])
    return str(v or "")


def _largest_image_url(urls: list[str]) -> tuple[str, int, int]:
    """Pick the largest of LoC image_url entries ('...jpg#h=150&w=118')."""
    best = ("", 0, 0)
    for u in urls or []:
        m = re.search(r"#h=(\d+)&w=(\d+)", u)
        h, w = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        if w * h >= best[1] * best[2]:
            best = (u.split("#")[0], w, h)
    return best


def parse_search(data: dict[str, Any], media: Media) -> list[RawResult]:
    out = []
    for r in data.get("results") or []:
        page = r.get("url") or r.get("id") or ""
        if not page:
            continue
        img, w, h = _largest_image_url(r.get("image_url") or [])
        if media == "image" and not img:
            continue
        rights = _text(r.get("rights") or (r.get("item") or {}).get("rights_advisory"))
        out.append(RawResult(source="loc", url=page, page_url=page, title=_text(r.get("title")),
                             author=_text(r.get("contributor")), licence=rights_to_licence(rights), media=media,
                             width=w, height=h, download_url=img if media == "image" else "", score=0.75,
                             extra={"date": _text(r.get("date"))}))
    return out


def best_resource(item: dict[str, Any], media: Media) -> Optional[dict[str, Any]]:
    """Largest jpeg, or an mp4, from an item JSON's resources[].files[][]."""
    files: list[dict[str, Any]] = []
    for res in item.get("resources") or []:
        for group in res.get("files") or []:
            files.extend(f for f in (group if isinstance(group, list) else [group]) if isinstance(f, dict))
    want = "video/mp4" if media == "video" else "image/jpeg"
    pool = [f for f in files if f.get("mimetype") == want and f.get("url")]
    if not pool:
        return None
    return max(pool, key=lambda f: (int(f.get("width") or 0) * int(f.get("height") or 0), int(f.get("size") or 0)))


class LocAdapter(BaseAdapter):
    name = "loc"
    media = {"video", "image"}

    def search(self, query: str, media: Media, n: int, settings: Settings,
               shot: Optional[Shot] = None) -> list[RawResult]:
        url = FILMS if media == "video" else PHOTOS
        data = base.http_get_json(url, params={"q": query, "fo": "json", "c": max(n, 5)}, timeout=self.timeout(settings))
        return parse_search(data, media)[:n]

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Settings) -> list[Candidate]:
        item = base.http_get_json(raw.page_url.rstrip("/") + "/", params={"fo": "json"}, timeout=self.timeout(settings))
        f = best_resource(item, raw.media)
        if f:
            raw = raw.model_copy(update={"download_url": f["url"], "width": int(f.get("width") or raw.width),
                                         "height": int(f.get("height") or raw.height)})
        if not raw.download_url:
            return []
        ext = ".mp4" if raw.media == "video" else ".jpg"
        src = cached_download(raw.download_url, ext, settings)
        dst = link_into(src, out_dir, f"loc_{base.url_hash(raw.download_url)}{ext}")
        spans = base.cut_clips(base.detect_scenes(dst), shot.duration, max_clips=3) if raw.media == "video" else None
        return make_candidates(raw, dst, shot, settings, spans=spans)
