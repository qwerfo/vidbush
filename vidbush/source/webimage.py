"""General web image search: whatever pictures the open web has for a query (newspaper pages, museum
records, encyclopedia figures, news photos). Provider by the first key set: SERPER_API_KEY (Google
images via Serper), FIRECRAWL_API_KEY (Firecrawl search, images source), then the research search seat
(SEARCH_PROVIDER or the web_search seat type: serper, brave or tavily, key SEARCH_API_KEY).

The licence of a web image is not known: every result carries "unknown (web image search)" (flagged
licence_unknown, never hidden), the page's site as author and a credit line naming the page. Many
image hosts refuse hotlinks or answer with an HTML page, so fetch decodes the download with PIL and
rejects anything that is not a still image.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from ..config import Settings
from ..models import Candidate, Shot
from ..providers import http as provider_http
from . import base
from .base import BaseAdapter, Media, RawResult, cached_download, link_into, make_candidates

SERPER = "https://google.serper.dev/images"
FIRECRAWL = "https://api.firecrawl.dev/v2/search"
BRAVE = "https://api.search.brave.com/res/v1/images/search"
TAVILY = "https://api.tavily.com/search"
LICENCE = "unknown (web image search)"
MIN_LONG_SIDE = 300
SEAT_PROVIDERS = ("serper", "brave", "tavily")
IMAGE_ACCEPT = "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"
FORMAT_EXT = {"JPEG": ".jpg", "MPO": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tif"}


def http_post_json(url: str, payload: dict[str, Any], headers: Optional[dict[str, str]] = None,
                   timeout: float = 20.0) -> Any:
    """base.http_post_json, looked up at call time (the one network seam tests patch)."""
    return base.http_post_json(url, payload, headers=headers, timeout=timeout)


def pick_provider(settings: Settings) -> Optional[tuple[str, str]]:
    """(provider, key) for the first provider with a key set, else None."""
    for name, env in (("serper", "SERPER_API_KEY"), ("firecrawl", "FIRECRAWL_API_KEY")):
        if settings.key(env):
            return name, settings.key(env)
    seat = ((settings.seats or {}).get("providers") or {}).get("web_search") or {}
    kind = (settings.key("SEARCH_PROVIDER") or str(seat.get("type") or "")).strip().lower()
    key = settings.key(str(seat.get("key_env") or "SEARCH_API_KEY"))
    return (kind, key) if kind in SEAT_PROVIDERS and key else None


def site_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _usable(url: str) -> bool:
    return urlparse(url or "").scheme in ("http", "https") and "x-raw-image" not in url


def _result(provider: str, image: str, title: str, page: str, width: Any, height: Any, position: Any,
            description: str = "") -> Optional[RawResult]:
    """One RawResult, or None for a non-http(s) image or one known to be under MIN_LONG_SIDE."""
    if not _usable(image):
        return None
    w, h = int(width or 0), int(height or 0)
    if 0 < max(w, h) < MIN_LONG_SIDE:
        return None
    site = site_of(page or image)
    title = " ".join((title or "").split()) or site
    extra = {"provider": provider, "site": site, "position": int(position or 0),
             "attribution": f"{title} ({site}, via web image search)"}
    if description:
        extra["description"] = description
    return RawResult(source="webimage", url=image, download_url=image, page_url=page or "", title=title,
                     author=site, licence=LICENCE, media="image", width=w, height=h, score=0.6, extra=extra)


def parse_serper(data: dict[str, Any]) -> list[RawResult]:
    return [r for i in data.get("images") or [] if (r := _result(
        "serper", i.get("imageUrl", ""), i.get("title", ""), i.get("link", ""), i.get("imageWidth"),
        i.get("imageHeight"), i.get("position")))]


def parse_firecrawl(data: dict[str, Any]) -> list[RawResult]:
    return [r for i in (data.get("data") or {}).get("images") or [] if (r := _result(
        "firecrawl", i.get("imageUrl", ""), i.get("title", ""), i.get("url", ""), i.get("imageWidth"),
        i.get("imageHeight"), i.get("position")))]


def parse_brave(data: dict[str, Any]) -> list[RawResult]:
    out = []
    for n, i in enumerate(data.get("results") or [], 1):
        props = i.get("properties") or {}
        r = _result("brave", props.get("url", ""), i.get("title", ""), i.get("url", ""), props.get("width"),
                    props.get("height"), n)
        if r:
            out.append(r)
    return out


def parse_tavily(data: dict[str, Any]) -> list[RawResult]:
    """Tavily gives image URL and description only: no page, no size (checked at fetch)."""
    out = []
    for n, i in enumerate(data.get("images") or [], 1):
        url, desc = (i, "") if isinstance(i, str) else (i.get("url", ""), i.get("description") or "")
        r = _result("tavily", url, desc[:120], "", 0, 0, n, description=desc)
        if r:
            out.append(r)
    return out


class WebImageAdapter(BaseAdapter):
    name = "webimage"
    media = {"image"}
    # not encyclopedic: those queries are the bare name for entity lookup, and a bare name on the open web
    # returns its namesakes (a film, a product) before the thing
    classes = frozenset({"web_image", "newspaper"})
    matches_description = True

    def available(self, settings: Settings) -> bool:
        return not settings.offline and pick_provider(settings) is not None

    def search(self, query: str, media: Media, n: int, settings: Settings,
               shot: Optional[Shot] = None) -> list[RawResult]:
        picked = pick_provider(settings)
        if media != "image" or not picked:
            return []
        provider, key = picked
        t = self.timeout(settings)
        ask = max(n, 5)
        if provider == "serper":
            res = parse_serper(http_post_json(SERPER, {"q": query, "num": ask}, {"X-API-KEY": key}, timeout=t))
        elif provider == "firecrawl":
            body = {"query": query, "limit": ask, "sources": [{"type": "images"}]}
            res = parse_firecrawl(http_post_json(FIRECRAWL, body, {"Authorization": f"Bearer {key}"}, timeout=t))
        elif provider == "brave":
            res = parse_brave(base.http_get_json(BRAVE, params={"q": query, "count": ask}, timeout=t,
                                                 headers={"X-Subscription-Token": key, "Accept": "application/json"}))
        else:
            body = {"query": query, "max_results": ask, "include_images": True, "include_image_descriptions": True}
            res = parse_tavily(http_post_json(TAVILY, body, {"Authorization": f"Bearer {key}"}, timeout=t))
        seen: set[str] = set()
        return [r for r in res if not (r.url in seen or seen.add(r.url))][:n]

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Settings) -> list[Candidate]:
        from PIL import Image
        ext = base.ext_from_url(raw.download_url, ".jpg")
        ext = ext if ext in base.STILL_EXTS else ".jpg"
        headers = {"Accept": IMAGE_ACCEPT, **({"Referer": raw.page_url} if raw.page_url else {})}
        src = cached_download(raw.download_url, ext, settings, headers=headers, max_bytes=base.MAX_IMAGE_BYTES)
        try:
            with Image.open(src) as im:
                im.load()
                fmt, (w, h) = im.format or "", im.size
        except Image.DecompressionBombError:
            raise base.Rejected("image too large to decode")
        except Exception:  # noqa: BLE001 - HTML page, hotlink block or broken file
            src.unlink(missing_ok=True)
            raise base.Rejected("download is not an image (host refused the hotlink or sent a page)")
        if fmt not in FORMAT_EXT:
            raise base.Rejected(f"not a still image format ({fmt or 'unknown'})")
        if max(w, h) < MIN_LONG_SIDE:
            raise base.Rejected(f"image too small ({w}x{h})")
        raw = raw.model_copy(update={"width": w, "height": h})
        dst = link_into(src, out_dir, f"webimage_{base.url_hash(raw.download_url)}{FORMAT_EXT[fmt]}")
        return make_candidates(raw, dst, shot, settings)
