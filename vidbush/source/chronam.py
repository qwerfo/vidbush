"""Chronicling America adapter: historic US newspaper pages (Library of Congress, public domain).

Search: the loc.gov JSON API collection endpoint (the old chroniclingamerica.loc.gov API redirects there
since 2025), page level results, in the advanced search form the Library's own data-exploration
notebooks use: https://www.loc.gov/collections/chronicling-america/?dl=page&searchType=advanced
&ops=PHRASE&qs=<phrase>&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&fo=json. The year comes from the
query or from shot.real_thing.date. Each result is one newspaper page; its image_url list holds IIIF
image links on tile.loc.gov and the page's word coordinates service URL.
Fetch: the word coordinates (JSON: term -> boxes, in units of the page's own width and height) give
the boxes of the query words; the tightest cluster of them, padded to a 16:9 region, is requested from
IIIF as a percent region (the detail), then the whole page at 2000 px wide. The detail comes first.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..models import Candidate, Shot
from . import base
from .base import BaseAdapter, Media, RawResult, cached_download, link_into, make_candidates

SEARCH = "https://www.loc.gov/collections/chronicling-america/"
COORDS = "https://tile.loc.gov/text-services/word-coordinates-service"
LICENCE = "Public domain (Library of Congress, Chronicling America)"
PAGE_WIDTH = 2000  # px wide for the full page image
ASPECT = 16 / 9
MIN_REGION = 0.3  # the detail is at least this share of the page width (readable context around the words)
NEAR = 0.15  # query words further apart than this share of the page width are not one passage
TITLE_RE = re.compile(r"^Image (\d+) of (.+?),\s*([A-Z][a-z]+\.? \d{1,2}, \d{4})\s*$")


def year_span(query: str, shot: Optional[Shot] = None) -> Optional[tuple[int, int]]:
    """Years to search: a decade as written ('1890s'), else the year plus or minus one; from the query,
    else from the shot's real thing date."""
    m = base.YEAR_RE.search(query or "")
    if not m and shot is not None and shot.real_thing is not None:
        m = base.YEAR_RE.search(shot.real_thing.date or "")
    if not m:
        return None
    y = int(m.group(1))
    return (y, y + 9) if m.group(0).endswith("s") else (y - 1, y + 1)


def search_params(query: str, n: int, shot: Optional[Shot] = None, exact: bool = True) -> dict[str, Any]:
    """Search parameters: the query without its year, as an exact phrase when it is 2 to 5 words
    (exact=True), else its specific words all required."""
    rest = " ".join(re.sub(r"[^\w\s'\-]", " ", base.YEAR_RE.sub(" ", query or "")).split())
    phrase = exact and 2 <= len(rest.split()) <= 5
    params: dict[str, Any] = {"dl": "page", "searchType": "advanced", "ops": "PHRASE" if phrase else "AND",
                              "qs": rest if phrase else base.keyword_query(rest, 4), "c": max(n, 5), "fo": "json"}
    span = year_span(query, shot)
    if span:
        params.update(start_date=f"{span[0]}-01-01", end_date=f"{span[1]}-12-31")
    return params


def _first(v: Any) -> str:
    return str(v[0] if isinstance(v, list) and v else v or "")


def parse_search(data: dict[str, Any], words: list[str]) -> list[RawResult]:
    """One RawResult per newspaper page that has a IIIF image."""
    out = []
    for r in data.get("results") or []:
        urls = r.get("image_url") or []
        iiif = next((u.split("/full/")[0] for u in urls if "/image-services/iiif/" in u and "/full/" in u), "")
        page = (r.get("id") or r.get("url") or "").replace("http://", "https://", 1)
        if not iiif or not page:
            continue
        coords = next((u for u in urls if "word-coordinates-service" in u), "") or coords_url(iiif)
        raw_title = str(r.get("title") or "")
        m = TITLE_RE.match(raw_title)
        if m:
            paper, when = m.group(2), m.group(3)
            title = f"{paper}, {when}, Image {m.group(1)}"
        else:
            paper, when, title = _first(r.get("partof_title")) or raw_title, str(r.get("date") or ""), raw_title
        desc = r.get("description")
        out.append(RawResult(
            source="chronam", url=page, page_url=page, title=title, author=paper, licence=LICENCE, media="image",
            download_url=f"{iiif}/full/{PAGE_WIDTH},/0/default.jpg", score=0.8,
            extra={"date": str(r.get("date") or ""), "description": " ".join(desc) if isinstance(desc, list) else str(desc or ""),
                   "attribution": f"{paper}, {when}, Chronicling America, Library of Congress",
                   "iiif": iiif, "coords_url": coords, "match_words": words, "lccn": _first(r.get("number_lccn"))}))
    return out


def coords_url(iiif: str) -> str:
    """The word coordinates URL for a IIIF page id ('.../iiif/service:ndnp:...:1270' -> segment '/service/ndnp/.../1270.xml')."""
    sid = iiif.rsplit("/iiif/", 1)[-1]
    return f"{COORDS}?segment=/{sid.replace(':', '/')}.xml&format=alto_xml" if sid.startswith("service:") else ""


def match_words(query: str) -> list[str]:
    """The specific words of a query, lower case, as the coordinates service keys its terms."""
    return list(dict.fromkeys(w.lower() for w in re.findall(r"[A-Za-z]{3,}", base.YEAR_RE.sub(" ", query or ""))
                              if w.lower() not in base.SKIP_WORDS))


Box = tuple[float, float, float, float]  # x, y, w, h in the page's coordinate units


def _bbox(boxes: list[Box]) -> Box:
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _centre(b: Box) -> tuple[float, float]:
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def match_boxes(terms: dict[str, list[dict[str, Any]]], words: list[str], page_w: float) -> list[Box]:
    """Boxes of the query words in their tightest passage: around each occurrence, the nearest occurrence
    of every other word within NEAR of the page width; most words first, then the largest type (a headline
    reads on screen, body text in a detail rarely does)."""
    occ: dict[str, list[Box]] = {}
    for term, hits in terms.items():
        occ.setdefault(base._stem(term), []).extend(tuple(h["coordinates"][:4]) for h in hits if len(h.get("coordinates") or []) >= 4)
    found = [occ[s] for s in dict.fromkeys(base._stem(w) for w in words) if occ.get(s)]
    best: list[Box] = []
    best_key: tuple[int, float] = (0, 0.0)
    for i, group in enumerate(found):
        for anchor in group:
            ax, ay = _centre(anchor)
            chosen = [anchor]
            for j, other in enumerate(found):
                if j == i:
                    continue
                near = min(other, key=lambda b: math.dist(_centre(b), (ax, ay)))
                if math.dist(_centre(near), (ax, ay)) <= NEAR * page_w:
                    chosen.append(near)
            key = (len(chosen), sum(b[3] for b in chosen) / len(chosen))
            if not best or key > best_key:
                best, best_key = chosen, key
    return best


def crop_region(boxes: list[Box], page_w: float, page_h: float) -> Box:
    """A 16:9 region around the boxes: padded by their height, at least MIN_REGION of the page width,
    centred on them and kept inside the page."""
    x, y, w, h = _bbox(boxes)
    line = max(b[3] for b in boxes)
    w, h = max(w + 6 * line, MIN_REGION * page_w), h + 6 * line
    w, h = (h * ASPECT, h) if w / h < ASPECT else (w, w / ASPECT)
    if w > page_w:
        w, h = page_w, page_w / ASPECT
    if h > page_h:
        w, h = page_h * ASPECT, page_h
    cx, cy = _centre(_bbox(boxes))
    x0 = min(max(cx - w / 2, 0.0), page_w - w)
    y0 = min(max(cy - h / 2, 0.0), page_h - h)
    return round(x0), round(y0), round(w), round(h)


def region_url(iiif: str, region: Box, page_w: float, page_h: float) -> str:
    """IIIF percent region at native size (the region is a fraction of the page, so no image size is needed)."""
    x, y, w, h = region
    pct = ",".join(f"{v:.3f}" for v in (100 * x / page_w, 100 * y / page_h, 100 * w / page_w, 100 * h / page_h))
    return f"{iiif}/pct:{pct}/full/0/default.jpg"


def _dims(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size


class ChronamAdapter(BaseAdapter):
    name = "chronam"
    media = {"image"}
    classes = frozenset({"newspaper", "archive"})
    matches_description = True  # the engine searches the page OCR text, not titles
    uses_real_thing = True  # the search year can come from shot.real_thing.date: the cache is keyed on it

    def timeout(self, settings: Settings) -> float:
        """loc.gov search often takes 20 to 40 s; a short timeout loses most of its results."""
        return max(settings.sources.timeout_s, 60.0)

    def search(self, query: str, media: Media, n: int, settings: Settings,
               shot: Optional[Shot] = None) -> list[RawResult]:
        if media != "image":
            return []
        words = match_words(query)
        first = search_params(query, n, shot)
        tries = [first]
        loose = search_params(query, n, shot, exact=False)
        if loose["qs"] != first["qs"] or loose["ops"] != first["ops"]:
            tries.append(loose)  # sent only when the exact phrase finds nothing
        for params in tries:
            res = parse_search(base.http_get_json(SEARCH, params=params, headers={"Accept": "application/json"},
                                                  timeout=self.timeout(settings)), words)
            if res:
                return res[:n]
        return []

    def detail_url(self, raw: RawResult, settings: Settings) -> str:
        """IIIF URL of the region around the query words, or "" when the page has no match or no coordinates."""
        url = raw.extra.get("coords_url") or ""
        if not url or not raw.extra.get("match_words"):
            return ""
        try:
            data = base.http_get_json(url, timeout=self.timeout(settings))
        except Exception:  # noqa: BLE001 - the detail is optional; the full page is still fetched
            return ""
        seg = next(iter(data.values()), {}) if isinstance(data, dict) and data else {}
        page_w, page_h = float(seg.get("width") or 0), float(seg.get("height") or 0)
        if page_w <= 0 or page_h <= 0:
            return ""
        boxes = match_boxes(seg.get("coords") or {}, raw.extra["match_words"], page_w)
        return region_url(raw.extra["iiif"], crop_region(boxes, page_w, page_h), page_w, page_h) if boxes else ""

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Settings) -> list[Candidate]:
        if not raw.extra.get("iiif"):
            raise base.Rejected("no IIIF image for this newspaper page")
        jobs = []
        detail = self.detail_url(raw, settings)
        if detail:
            jobs.append((detail, "detail around the matched words", " (detail)"))
        jobs.append((raw.download_url, "full newspaper page", ""))
        out: list[Candidate] = []
        for url, note, suffix in jobs:
            try:
                src = cached_download(url, ".jpg", settings, max_bytes=base.MAX_IMAGE_BYTES)
            except Exception:  # noqa: BLE001 - a failed detail still leaves the full page
                if suffix:
                    continue
                raise
            dst = link_into(src, out_dir, f"chronam_{base.url_hash(url)}.jpg")
            w, h = _dims(dst)
            one = raw.model_copy(update={"download_url": url, "note": note, "title": raw.title + suffix,
                                         "width": w, "height": h})
            out += make_candidates(one, dst, shot, settings)
        return out
