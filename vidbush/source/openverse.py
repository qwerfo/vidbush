"""Openverse adapter: openly licensed images (Creative Commons and public domain) indexed from Flickr,
Wikimedia, museums and other collections. Anonymous API, no key; the anonymous quota is small (about
100 requests a day, a few a minute), so the registry's search cache and budget matter here.
Every result is CC or public domain; mature content is filtered out at the API."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..models import Candidate, Shot
from . import base
from .base import BaseAdapter, Media, RawResult, cached_download, link_into, make_candidates

IMAGES = "https://api.openverse.org/v1/images/"


def licence_name(code: str, version: str) -> str:
    """Openverse licence code and version to the form normalise_licence knows ('by-sa', '2.0' -> 'CC BY-SA 2.0')."""
    code = (code or "").strip().lower()
    if code == "pdm":
        return "Public domain"
    if code == "cc0":
        return "CC0 1.0"
    if not code:
        return "unknown (Openverse)"
    return f"CC {code.upper()} {version}".strip()


def parse(data: dict[str, Any]) -> list[RawResult]:
    out = []
    for r in data.get("results") or []:
        url = r.get("url") or ""
        if not url or r.get("mature"):
            continue
        lic = licence_name(r.get("license", ""), r.get("license_version") or "")
        tags = [t.get("name", "") for t in r.get("tags") or [] if isinstance(t, dict) and t.get("name")]
        extra: dict[str, Any] = {"provider": r.get("provider", ""), "site": r.get("source", ""), "tags": tags,
                                 "openverse_id": r.get("id", "")}
        if r.get("attribution"):
            extra["attribution"] = r["attribution"]
        if r.get("description"):
            extra["description"] = r["description"]
        open_pd = lic.startswith(("Public domain", "CC0"))
        out.append(RawResult(source="openverse", url=url, download_url=url,
                             page_url=r.get("foreign_landing_url") or "", title=r.get("title") or "Untitled",
                             author=r.get("creator") or "", licence=lic, media="image",
                             width=int(r.get("width") or 0), height=int(r.get("height") or 0),
                             score=0.65 if open_pd else 0.6, extra=extra))
    return out


class OpenverseAdapter(BaseAdapter):
    name = "openverse"
    media = {"image"}
    classes = frozenset({"web_image", "encyclopedic", "stock", "archive"})
    matches_description = True

    def search(self, query: str, media: Media, n: int, settings: Settings,
               shot: Optional[Shot] = None) -> list[RawResult]:
        if media != "image":
            return []
        params = {"q": query, "page_size": max(n, 5), "mature": "false"}
        return parse(base.http_get_json(IMAGES, params=params, timeout=self.timeout(settings)))[:n]

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Settings) -> list[Candidate]:
        ext = base.ext_from_url(raw.download_url, ".jpg")
        ext = ext if ext in base.STILL_EXTS else ".jpg"
        src = cached_download(raw.download_url, ext, settings, max_bytes=base.MAX_IMAGE_BYTES)
        dst = link_into(src, out_dir, f"openverse_{base.url_hash(raw.download_url)}{ext}")
        return make_candidates(raw, dst, shot, settings)
