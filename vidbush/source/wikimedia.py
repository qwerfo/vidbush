"""Wikimedia Commons adapter: stills and videos (webm, ogv) from the File namespace.

Search order: entity first, then text.
a) The real thing's name (shot.real_thing.name, else the query; a long name also tries its longest run
   of capitalised words) is resolved to a Wikidata item: wbsearchentities, then one wbgetentities for
   the hits. The best item matches the name by label or alias and, when known, agrees with
   real_thing.kind, date and place (a dog named Balto beats the 1995 film once a kind is given).
b) Its media: the P18 file(s) (score 0.9), then the files of its Commons category (P373, else the
   commonswiki sitelink when it is a Category: page; score 0.85), with one subcategory only when the
   category has few files; the P935 gallery only when there is no category. At most MAX_REQUESTS calls.
c) Text search (generator=search in namespace 6) when no entity or no file of the wanted media.
Every listing asks prop=imageinfo with iiprop=url|size|mime|extmetadata and iiurlwidth=1920. Licence
comes from extmetadata LicenseShortName, author from Artist with HTML stripped, description from
ImageDescription, date from DateTimeOriginal or DateTime.
"""
from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..models import Candidate, RealThing, Shot
from ..providers.base import ProviderError
from . import base
from .base import BaseAdapter, Media, RawResult, cached_download, ext_from_url, link_into, make_candidates

API = "https://commons.wikimedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
IMAGE_MIME = {"image/jpeg", "image/png", "image/webp", "image/tiff"}
VIDEO_MIME = {"video/webm", "application/ogg", "video/ogg", "video/mp4"}
IMAGEINFO = {"prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1920}
MAX_REQUESTS = 5  # per entity lookup; the text search fallback always gets its own two
CATEGORY_FILES = 20
FEW_FILES = 5  # a category with fewer files of the wanted media than this also lends its first subcategory
# P31 values that are never the thing itself: disambiguation, family and given names, categories, lists.
NOT_A_THING = {"Q4167410", "Q101352", "Q202444", "Q4167836", "Q13406463", "Q11266439"}
TIME_PROPS = ("P569", "P570", "P571", "P576", "P580", "P582", "P585", "P1619", "P729", "P730")
OPEN_LICENCES = ("public domain", "pd", "cc0", "cc by", "cc-by", "no restrictions")


def strip_html(text: str) -> str:
    """Visible text only: hidden elements (Commons hides 'date QS:P571,...' machine data) are dropped."""
    text = re.sub(r"<(\w+)[^>]*display:\s*none[^>]*>.*?</\1>", " ", text or "", flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", html.unescape(text)).strip()
    return re.sub(r"\s+([.,;:!?)])", r"\1", text)  # no gap where a tag closed before punctuation


def _clean_url(url: Any) -> Any:
    """Commons tracking parameters (?utm_source=...) dropped: the same file keeps one URL."""
    return re.sub(r"\?utm_[^#]*$", "", url) if isinstance(url, str) else url


def _meta(ext: dict[str, Any], key: str) -> str:
    v = ext.get(key) or {}
    return strip_html(str(v.get("value", ""))) if isinstance(v, dict) else ""


def parse_commons(data: dict[str, Any], media: Media, score: float = 0.8) -> list[RawResult]:
    """RawResults from a Commons query response with prop=imageinfo (search, category or titles)."""
    pages = (data.get("query") or {}).get("pages") or {}
    items = pages.values() if isinstance(pages, dict) else pages
    out: list[tuple[int, RawResult]] = []
    for page in items:
        info = (page.get("imageinfo") or [{}])[0]
        info = {k: _clean_url(v) if k in ("url", "thumburl") else v for k, v in info.items()}
        mime = info.get("mime", "")
        if (media == "image" and mime not in IMAGE_MIME) or (media == "video" and mime not in VIDEO_MIME):
            continue
        ext = info.get("extmetadata") or {}
        title = _meta(ext, "ObjectName") or re.sub(r"^File:|\.[a-z0-9]+$", "", page.get("title", ""), flags=re.I)
        width, height = int(info.get("width") or 0), int(info.get("height") or 0)
        big = media == "image" and width > 1920 and info.get("thumburl")
        licence = _meta(ext, "LicenseShortName") or _meta(ext, "License") or "unknown"
        extra = {k: v for k, v in (("description", _meta(ext, "ImageDescription")),
                                   ("date", _meta(ext, "DateTimeOriginal") or _meta(ext, "DateTime")),
                                   ("attribution", _meta(ext, "Attribution"))) if v}
        out.append((int(page.get("index", 0)), RawResult(
            source="wikimedia", url=info.get("url", ""), title=title, author=_meta(ext, "Artist"),
            licence=licence, media=media,
            width=int(info.get("thumbwidth") or width) if big else width,
            height=int(info.get("thumbheight") or height) if big else height,
            duration=float(info.get("duration") or 0.0),
            download_url=info.get("thumburl") if big else info.get("url", ""),
            page_url=info.get("descriptionurl", ""),
            score=score + (0.02 if licence.lower().startswith(OPEN_LICENCES) else 0.0),  # ties go to open licences
            extra=extra,
        )))
    return [r for _, r in sorted(out, key=lambda p: p[0])]


# ---------------------------------------------------------------- entity resolution

def _norm(text: str) -> str:
    """Lower case, accents and punctuation removed: 'Utqiaġvik' meets 'Utqiagvik'."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


LANGS = ("en", "mul")  # Wikidata keeps many names only under "mul" (the 1995 film Balto has no en label)


def _text(ent: dict[str, Any], key: str) -> str:
    """The en label or description of an item, else the mul one."""
    vals = ent.get(key) or {}
    return next((vals[lang].get("value", "") for lang in LANGS if lang in vals), "")


def proper_name_run(text: str) -> str:
    """The longest run of capitalised words in text (the first when tied); '' when there is none."""
    best: list[str] = []
    run: list[str] = []
    for w in re.findall(r"[A-Za-z][\w'\-]*", text or ""):
        if w[0].isupper():
            run.append(w)
            if len(run) > len(best):
                best = list(run)
        else:
            run = []
    return " ".join(best)


def entity_names(query: str, shot: Optional[Shot]) -> list[str]:
    """Names to resolve, best first: the real thing's name, else the query; then its proper-name run."""
    thing = shot.real_thing if shot is not None else None
    name = (thing.name if thing and thing.name else query).strip()
    names = [name] if name else []
    if len(name.split()) > 1:
        run = proper_name_run(name)
        if run and _norm(run) != _norm(name):
            names.append(run)
    return names


def _claims(ent: dict[str, Any], prop: str) -> list[Any]:
    out = []
    for c in (ent.get("claims") or {}).get(prop) or []:
        v = ((c.get("mainsnak") or {}).get("datavalue") or {}).get("value")
        if v is not None:
            out.append(v)
    return out


def _years(ent: dict[str, Any]) -> list[int]:
    years = []
    for p in TIME_PROPS:
        for v in _claims(ent, p):
            m = re.match(r"[+-]?(\d{1,4})-", str(v.get("time", "")) if isinstance(v, dict) else "")
            if m:
                years.append(int(m.group(1)))
    return years


def _label_match(name: str, ent: dict[str, Any]) -> float:
    """3 for a label or alias equal to the name, 2.5 for one equal to the part before a comma
    ('Coldfoot, Alaska'), 1 when one's words contain the other's, else 0."""
    want = _norm(name)
    head = _norm(name.split(",")[0])
    texts = [_text(ent, "labels")]
    texts += [a.get("value", "") for lang in LANGS for a in (ent.get("aliases") or {}).get(lang, [])]
    best = 0.0
    for t in map(_norm, texts):
        if not t:
            continue
        if t == want:
            return 3.0
        if t == head:
            best = max(best, 2.5)
        elif set(t.split()) <= set(want.split()) or set(want.split()) <= set(t.split()):
            best = max(best, 1.0)
    return best


def _is_work(ent: dict[str, Any]) -> bool:
    """A creative work named after the thing (film, book, album, song): it has a publication date,
    director, author or performer."""
    return any(_claims(ent, p) for p in ("P577", "P57", "P50", "P175"))


def _has_media(ent: dict[str, Any]) -> bool:
    return bool(_claims(ent, "P18") or _claims(ent, "P373") or _claims(ent, "P935")
                or (ent.get("sitelinks") or {}).get("commonswiki"))


def entity_score(name: str, ent: dict[str, Any], thing: Optional[RealThing]) -> float:
    """How well a Wikidata item fits the named real thing; 0 means it is not the thing."""
    match = _label_match(name, ent)
    p31 = {v.get("id") for v in _claims(ent, "P31") if isinstance(v, dict)}
    if not match or p31 & NOT_A_THING:
        return 0.0
    desc = _norm(_text(ent, "descriptions"))
    score = match + min(len(ent.get("sitelinks") or {}), 100) / 100.0  # well known items first
    if _has_media(ent):
        score += 0.5
    tail = _norm(name.split(",", 1)[1]) if "," in name else ""
    if tail and tail in desc:
        score += 0.5
    if thing is not None:
        kind = (thing.kind or "").lower()
        if kind and kind != "document" and _is_work(ent):
            score -= 2.0  # the 1995 film is not the dog
        if kind == "person" and "Q5" in p31:
            score += 1.0
        elif kind == "place" and _claims(ent, "P625"):
            score += 1.0
        elif kind == "event" and (_claims(ent, "P585") or _claims(ent, "P580")):
            score += 1.0
        m = base.YEAR_RE.search(thing.date or "")
        years = _years(ent)
        if m and years:
            y = int(m.group(1))
            score += 1.0 if min(years) - 1 <= y <= max(years) + 1 else -0.5
        if thing.place and set(_norm(thing.place).split()) & set(desc.split()):
            score += 0.5
    return max(score, 0.01)


class _Budget:
    """Counts requests so the entity path of one search never makes more than MAX_REQUESTS."""

    def __init__(self, timeout: float) -> None:
        self.left, self.timeout = MAX_REQUESTS, timeout

    def get(self, url: str, params: dict[str, Any]) -> Optional[dict[str, Any]]:
        if self.left <= 0:
            return None
        self.left -= 1
        return base.http_get_json(url, params={"format": "json", **params}, timeout=self.timeout)


def resolve_entity(names: list[str], thing: Optional[RealThing], budget: _Budget) -> Optional[dict[str, Any]]:
    """The best Wikidata item for the first name that finds one, with its claims; None when none fits."""
    for name in names:
        found = budget.get(WIKIDATA_API, {"action": "wbsearchentities", "search": name, "language": "en",
                                          "uselang": "en", "type": "item", "limit": 5})
        ids = [h["id"] for h in (found or {}).get("search") or [] if h.get("id")]
        if not ids:
            continue
        data = budget.get(WIKIDATA_API, {"action": "wbgetentities", "ids": "|".join(ids), "languages": "|".join(LANGS),
                                         "props": "claims|labels|aliases|descriptions|sitelinks"})
        ents = (data or {}).get("entities") or {}
        scored = [(entity_score(name, ents[i], thing), -n, ents[i]) for n, i in enumerate(ids) if i in ents]
        scored = [s for s in scored if s[0] > 0]
        if scored:
            return max(scored, key=lambda s: (s[0], s[1]))[2]
    return None


def _category(ent: dict[str, Any]) -> str:
    cats = [str(v) for v in _claims(ent, "P373")]
    if cats:
        return "Category:" + cats[0]
    site = ((ent.get("sitelinks") or {}).get("commonswiki") or {}).get("title", "")
    return site if site.startswith("Category:") else ""


def _gallery(ent: dict[str, Any]) -> str:
    pages = [str(v) for v in _claims(ent, "P935")]
    site = ((ent.get("sitelinks") or {}).get("commonswiki") or {}).get("title", "")
    return pages[0] if pages else ("" if site.startswith("Category:") else site)


def _members(cat: str, types: str, media: Media, budget: _Budget) -> tuple[list[RawResult], list[str]]:
    """Files of one category (score 0.85) and the titles of its subcategories, sorted."""
    data = budget.get(API, {"action": "query", "generator": "categorymembers", "gcmtitle": cat,
                            "gcmtype": types, "gcmlimit": CATEGORY_FILES, **IMAGEINFO}) or {}
    pages = (data.get("query") or {}).get("pages") or {}
    subs = sorted(p.get("title", "") for p in (pages.values() if isinstance(pages, dict) else pages)
                  if p.get("ns") == 14)
    return parse_commons(data, media, score=0.85), subs


def entity_media(ent: dict[str, Any], media: Media, budget: _Budget) -> list[RawResult]:
    """P18 files, then category files (one subcategory when the category is thin), else the gallery."""
    out: list[RawResult] = []
    files = ["File:" + str(v) for v in _claims(ent, "P18")]
    if files and media == "image":
        data = budget.get(API, {"action": "query", "titles": "|".join(files[:10]), **IMAGEINFO})
        out += parse_commons(data or {}, media, score=0.9)
    cat, gallery = _category(ent), _gallery(ent)
    if cat:
        found, subs = _members(cat, "file|subcat", media, budget)
        out += found
        if len(found) < FEW_FILES and subs:
            out += _members(subs[0], "file", media, budget)[0]
    elif gallery:
        data = budget.get(API, {"action": "query", "generator": "images", "titles": gallery,
                                "gimlimit": CATEGORY_FILES, **IMAGEINFO})
        out += parse_commons(data or {}, media, score=0.85)
    qid, label = ent.get("id", ""), _text(ent, "labels")
    seen: set[str] = set()
    unique = []
    for r in out:
        if r.url and r.url not in seen:  # the P18 file is usually in the category too
            seen.add(r.url)
            r.extra.update({"entity": qid, "entity_label": label})
            unique.append(r)
    return unique


# ---------------------------------------------------------------- video transcodes

TRANSCODE_SIZES = ("1080p", "720p")


def transcode_urls(url: str) -> list[str]:
    """Commons VP9 transcodes of an original video URL, largest first (originals can be gigabytes)."""
    clean = url.split("?")[0]
    m = re.match(r"(https://upload\.wikimedia\.org/wikipedia/commons)/([0-9a-f]/[0-9a-f]{2})/([^/]+)$", clean)
    if not m:
        return []
    root, shard, name = m.groups()
    return [f"{root}/transcoded/{shard}/{name}/{name}.{size}.vp9.webm" for size in TRANSCODE_SIZES]


def _video_source(url: str, ext: str, settings: Settings, shot_duration: float, duration: float,
                  shot: Optional[Shot] = None, title: str = "") -> Path:
    """Prefer a 1080p or 720p transcode; fall back to the original file."""
    for t_url in transcode_urls(url):
        try:
            return base.cached_video(t_url, ".webm", settings, shot_duration, duration=duration, shot=shot, title=title)
        except Exception:  # noqa: BLE001 - a missing transcode is normal, try the next
            continue
    return base.cached_video(url, ext, settings, shot_duration, duration=duration, shot=shot, title=title)


class WikimediaAdapter(BaseAdapter):
    name = "wikimedia"
    media = {"video", "image"}
    keyword_words = 3  # CirrusSearch matches every word
    classes = frozenset({"encyclopedic", "archive"})
    matches_description = True
    uses_real_thing = True  # entity search reads shot.real_thing: the registry keys its cache on it

    def search(self, query: str, media: Media, n: int, settings: Settings,
               shot: Optional[Shot] = None) -> list[RawResult]:
        budget = _Budget(self.timeout(settings))
        try:
            ent = resolve_entity(entity_names(query, shot), shot.real_thing if shot else None, budget)
            found = entity_media(ent, media, budget) if ent else []
        except ProviderError:  # Wikidata down or one lookup failed: the text search still runs
            found = []
        if found:
            return found[:n]
        budget.left = 2  # the text search fallback: the first archive query, the second only when it is empty
        for q in base.archive_queries(query, self.keyword_words):
            search = f"{q} filetype:video" if media == "video" else f"{q} filetype:bitmap"
            data = budget.get(API, {"action": "query", "generator": "search", "gsrsearch": search,
                                    "gsrnamespace": 6, "gsrlimit": max(n, 5), **IMAGEINFO})
            found = parse_commons(data or {}, media)
            if found:
                break
        return found[:n]

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Settings) -> list[Candidate]:
        url = raw.download_url or raw.url
        ext = ext_from_url(url, ".webm" if raw.media == "video" else ".jpg")
        if raw.media == "video":
            src = _video_source(url, ext, settings, shot.duration, raw.duration, shot=shot, title=raw.title)
            ext = src.suffix
        else:
            src = cached_download(url, ext, settings, max_bytes=base.MAX_IMAGE_BYTES)
        dst = link_into(src, out_dir, f"wikimedia_{base.url_hash(src.name)}{ext}")
        spans = None
        if raw.media == "video":
            spans = base.cut_clips(base.focused_scenes(dst, shot, settings, url, raw.title), shot.duration, max_clips=3)
        return make_candidates(raw, dst, shot, settings, spans=spans)
