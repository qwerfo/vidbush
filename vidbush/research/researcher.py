"""Research stage: sources to fact cards and visual anchors.

Online, Wikipedia in the brand language (search plus extracts) and the web
search seat (queries built from the brief; the top result pages are fetched,
not only snippets) feed page excerpts to the researcher seat (task
"research"). The prompt budget is shared fairly between sources so web pages
are not crowded out by long Wikipedia extracts. Offline, the seat answers from
canned fixtures in mock mode.

Contract with a real researcher seat (seats.is_real("researcher")):
- citations are tied to fetched URLs: a source_url that was not fetched is
  cleared and the card is capped at confidence 0.55 (general knowledge);
- fewer than MIN_FACTS cards triggers one follow-up request (task
  "research_more"); fewer than 5 anchors triggers one anchor request (task
  "research_anchors"); the result is trimmed to MAX_FACTS cards (verified and
  confident first) and 10 anchors (places with coordinates first);
- a seat that fails on every real provider raises ProviderError, so the
  stage fails and resume can retry it.
With no usable answer, minimal unverified cards are built from the fetched
text, else from the brief itself. The fact check pass then drops
contradicted cards. A count still under the minimum is logged as a warning.
"""
from __future__ import annotations

import html
import re
from typing import Any, Optional
from urllib.parse import urlparse

from ..config import Settings
from ..models import Brand, Brief, FactCard, Research, VideoFormat, VisualAnchor
from ..providers.base import ProviderError
from ..script import prompts
from ..util import JsonLogger, no_em_dash
from .factcheck import fact_check

MAX_PAGES = 4
MAX_WEB_PAGES = 3
MAX_EXCERPT_CHARS = 9000
MAX_PROMPT_CHARS = 30000
MIN_FACTS, MAX_FACTS = 25, 60
MIN_ANCHORS, MAX_ANCHORS = 5, 10
GENERAL_CONFIDENCE = 0.55
ANCHOR_KINDS = {"place", "date", "object", "person", "event", "number"}
UNIT_WORDS = ("miles", "mile", "feet", "foot", "ft", "km", "kilometers", "meters", "yards", "tons", "people",
              "workers", "deaths", "men", "lives", "percent", "dollars", "years", "acres", "gallons", "ships")
PAGE_USER_AGENT = "Vidbush/1.0 (self-hosted documentary research)"


def run_research(brief: Brief, brand: Brand, seats: Any, settings: Settings, log: JsonLogger) -> Research:
    """Fact cards, anchors and sources for a brief. Network failures are logged; real seat failures raise."""
    lang = prompts.language_code(brand.language)
    real = _is_real(seats, "researcher")
    excerpts = {} if settings.offline else gather_sources(brief, seats, log, lang)
    data = _ask(brief, brand, seats, excerpts)
    research = _from_seat(data, brief, excerpts) if data is not None else None
    if research is not None and (real or excerpts):
        tie_citations(research, excerpts)
    if research is not None and real and len(research.facts) < MIN_FACTS:
        _top_up(research, brief, brand, seats, excerpts, log)
    if research is None or not research.facts:
        log.warn("research seat gave no usable facts, using fallback cards", pages=len(excerpts))
        research = fallback_research(brief, excerpts)
    research.sources = _merge_sources(research, excerpts)
    if real and len(research.anchors) < MIN_ANCHORS:
        _ask_anchors(research, brief, brand, seats)
    _add_anchors(research, log)
    research = fact_check(research, seats, log, excerpts)
    research = enforce_bounds(research, log)
    log.info("research done", facts=len(research.facts), anchors=len(research.anchors),
             sources=len(research.sources))
    return research


def _is_real(seats: Any, role: str) -> bool:
    try:
        return bool(seats.is_real(role))
    except AttributeError:
        return False


# ---------------------------------------------------------------- sources

def gather_sources(brief: Brief, seats: Any, log: JsonLogger, lang: str = "en") -> dict[str, str]:
    """{url: text}: Wikipedia extracts (brand language) for the top results plus web pages and snippets."""
    from ..providers.search import wikipedia_extract, wikipedia_search
    out: dict[str, str] = {}
    titles: list[tuple[str, str]] = []
    for query in _queries(brief):
        try:
            for r in wikipedia_search(query, n=3, lang=lang):
                if r["url"] not in {u for u, _ in titles}:
                    titles.append((r["url"], r["title"]))
        except Exception as e:  # network errors are never fatal
            log.warn("wikipedia search failed", query=query, error=str(e)[:200])
    for url, title in titles[:MAX_PAGES]:
        try:
            text = wikipedia_extract(title, lang=lang)
        except Exception as e:
            log.warn("wikipedia extract failed", title=title, error=str(e)[:200])
            continue
        if text:
            out[url] = f"{title}\n{_trim(text)}"
    _web_sources(brief, seats, out, log)
    log.info("sources gathered", pages=len(out))
    return out


def _queries(brief: Brief) -> list[str]:
    qs = [brief.topic]
    if brief.angle:
        qs.append(f"{brief.topic} {brief.angle}")
    return qs


def web_queries(brief: Brief) -> list[str]:
    """Web search queries built from the brief: topic with angle, plus a format specific query."""
    first = f"{brief.topic} {brief.angle}".strip() if brief.angle else brief.topic
    second = f"{brief.topic} ranked list" if brief.format == VideoFormat.top10 else f"{brief.topic} facts and figures"
    return [first, second] if second != first else [first]


def _web_sources(brief: Brief, seats: Any, out: dict[str, str], log: JsonLogger) -> None:
    """Web search results: the top pages fetched in full, the rest kept as snippets."""
    if not _is_real(seats, "search"):
        return
    results: list[dict[str, str]] = []
    for q in web_queries(brief):
        results.extend(seats.web_search(q, stage="research"))  # ProviderError propagates (policy)
    pages = 0
    for r in results:
        url = r.get("url", "")
        if not url or url in out or _host(url).endswith("wikipedia.org"):
            continue
        text = ""
        if pages < MAX_WEB_PAGES:
            text = _safe_page(url, log)
            pages += 1 if text else 0
        text = text or r.get("snippet", "")
        if text:
            out[url] = f"{r.get('title', '') or _host(url)}\n{text}"


def _safe_page(url: str, log: JsonLogger) -> str:
    try:
        return fetch_page_text(url)
    except Exception as e:  # a page that will not load keeps its snippet
        log.info("web page fetch failed", url=url, error=str(e)[:200])
        return ""


def fetch_page_text(url: str, max_chars: int = MAX_EXCERPT_CHARS) -> str:
    """Readable text of an HTML page (network seam for tests); "" for non HTML."""
    from ..providers.http import get_bytes
    raw = get_bytes(url, headers={"User-Agent": PAGE_USER_AGENT}, timeout=20, max_bytes=3_000_000)
    if raw[:5] == b"%PDF-":
        return ""
    return html_to_text(raw.decode("utf-8", errors="replace"))[:max_chars]


def html_to_text(page: str) -> str:
    """Strip scripts, styles, navigation and tags; collapse whitespace."""
    page = re.sub(r"(?is)<(script|style|noscript|nav|header|footer|aside|form)[^>]*>.*?</\1>", " ", page)
    page = re.sub(r"(?i)<br\s*/?>|</p>|</h\d>|</li>", "\n", page)
    text = html.unescape(re.sub(r"<[^>]+>", " ", page))
    lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if len(ln.split()) >= 6)


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _trim(text: str) -> str:
    """Drop reference sections and cap the length of a page extract."""
    text = re.split(r"\n==\s*(See also|References|Notes|Further reading|External links|Bibliography)\s*==",
                    text)[0]
    return text[:MAX_EXCERPT_CHARS]


def excerpt_block(excerpts: dict[str, str], budget: int = MAX_PROMPT_CHARS) -> str:
    """Excerpts for a prompt, the budget shared fairly: short sources keep all, long ones split the rest."""
    items = sorted(excerpts.items(), key=lambda kv: len(kv[1]))
    shares: dict[str, int] = {}
    left = budget
    for i, (url, body) in enumerate(items):
        share = left // max(1, len(items) - i)
        shares[url] = min(len(body), share)
        left -= shares[url]
    parts = []
    for url, body in excerpts.items():
        title, _, rest = body.partition("\n")
        parts.append(f"URL: {url}\nTITLE: {title}\n{rest[:max(0, shares[url] - len(title))]}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- seat

def _ask(brief: Brief, brand: Brand, seats: Any, excerpts: dict[str, str]) -> Optional[Any]:
    user = prompts.RESEARCH_USER.format(topic=brief.topic, angle=brief.angle or "none",
                                        audience=brief.audience or "general", minutes=brief.target_minutes,
                                        language=prompts.language_line(brand.language),
                                        excerpts=excerpt_block(excerpts) or "none fetched, use well established "
                                                                            "knowledge only",
                                        min_facts=MIN_FACTS, max_facts=MAX_FACTS)
    return _seat_json(seats, "research", user)


def _seat_json(seats: Any, task: str, user: str) -> Optional[Any]:
    """Researcher seat JSON. ProviderError (every real provider failed) propagates; other errors give None."""
    try:
        return seats.llm_json("researcher", task, prompts.RESEARCH_SYSTEM, user, stage="research", temperature=0.2)
    except ProviderError:
        raise
    except Exception:  # a malformed answer falls back to minimal cards
        return None


def _from_seat(data: Any, brief: Brief, excerpts: dict[str, str]) -> Optional[Research]:
    """Validate seat JSON into Research with ids f01.. and clean text."""
    raw = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return None
    facts, id_map = parse_cards(raw, 0)
    anchors = [a for a in (_anchor(x, id_map) for x in data.get("anchors") or []) if a is not None]
    sources = [{"url": str(x.get("url", "")), "title": str(x.get("title", ""))}
               for x in data.get("sources") or [] if isinstance(x, dict) and x.get("url")]
    return Research(topic=brief.topic, facts=facts, anchors=anchors, sources=sources)


def parse_cards(raw: list[Any], start: int) -> tuple[list[FactCard], dict[str, str]]:
    """FactCards numbered f{start+1:02d}.. and a map from the seat's ids to ours."""
    facts: list[FactCard] = []
    id_map: dict[str, str] = {}
    for r in raw:
        if not isinstance(r, dict) or not str(r.get("claim", "")).strip():
            continue
        fid = f"f{start + len(facts) + 1:02d}"
        id_map[str(r.get("id", fid))] = fid
        facts.append(FactCard(
            id=fid, claim=no_em_dash(str(r["claim"])).strip(), source_url=str(r.get("source_url", "") or ""),
            source_title=no_em_dash(str(r.get("source_title", "") or "")), confidence=_conf(r.get("confidence")),
            date=str(r.get("date", "") or ""), status=_status(r.get("status")),
            numbers=[n for n in r.get("numbers") or [] if isinstance(n, dict)] or extract_numbers(str(r["claim"])),
        ))
    return facts, id_map


def _norm_url(url: str) -> str:
    u = urlparse(url.strip())
    return f"{(u.hostname or '').lower().removeprefix('www.')}{u.path.rstrip('/')}"


def tie_citations(research: Research, excerpts: dict[str, str]) -> int:
    """Point each card at a fetched URL or mark it general knowledge. Returns how many were cleared."""
    fetched = {_norm_url(u): u for u in excerpts}
    cleared = 0
    for f in research.facts:
        match = fetched.get(_norm_url(f.source_url)) if f.source_url else None
        if match:
            f.source_url = match
            f.source_title = f.source_title or excerpts[match].partition("\n")[0]
            continue
        if f.source_url:
            cleared += 1
        f.source_url, f.source_title = "", "general knowledge"
        f.confidence = min(f.confidence, GENERAL_CONFIDENCE)
    research.sources = [s for s in research.sources if _norm_url(s.get("url", "")) in fetched]
    return cleared


def _top_up(research: Research, brief: Brief, brand: Brand, seats: Any, excerpts: dict[str, str],
            log: JsonLogger) -> None:
    """One follow-up request when a real seat gave fewer than MIN_FACTS cards."""
    need = MIN_FACTS - len(research.facts) + 5
    known = "\n".join(f"{f.id}: {f.claim}" for f in research.facts)
    user = prompts.RESEARCH_MORE_USER.format(topic=brief.topic, angle=brief.angle or "none", known=known,
                                             language=prompts.language_line(brand.language),
                                             excerpts=excerpt_block(excerpts) or "none fetched",
                                             need=need, next_id=len(research.facts) + 1)
    data = _seat_json(seats, "research_more", user)
    raw = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return
    seen = {f.claim.lower() for f in research.facts}
    extra = Research(topic=research.topic, facts=[c for c in parse_cards(raw, len(research.facts))[0]
                                                  if c.claim.lower() not in seen])
    tie_citations(extra, excerpts)
    research.facts.extend(extra.facts)
    log.info("research top up", asked=need, got=len(extra.facts), facts=len(research.facts))


def _ask_anchors(research: Research, brief: Brief, brand: Brand, seats: Any) -> None:
    """One anchor request when a real seat left fewer than MIN_ANCHORS anchors."""
    facts = "\n".join(f"{f.id}: {f.claim}" for f in research.facts[:MAX_FACTS])
    user = prompts.RESEARCH_ANCHORS_USER.format(topic=brief.topic, facts=facts or "none",
                                                language=prompts.language_line(brand.language))
    data = _seat_json(seats, "research_anchors", user)
    rows = data.get("anchors") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return
    ids = {f.id: f.id for f in research.facts}
    have = {a.name.lower() for a in research.anchors}
    for a in (_anchor(x, ids) for x in rows):
        if a is not None and a.name.lower() not in have:
            research.anchors.append(a)
            have.add(a.name.lower())


def enforce_bounds(research: Research, log: JsonLogger) -> Research:
    """Trim to MAX_FACTS cards and MAX_ANCHORS anchors; warn when under the minimums."""
    if len(research.facts) > MAX_FACTS:
        rank = sorted(research.facts, key=lambda f: (f.status != "verified", -f.confidence))
        keep = {f.id for f in rank[:MAX_FACTS]}
        research.facts = [f for f in research.facts if f.id in keep]
    known = {f.id for f in research.facts}
    for a in research.anchors:
        a.fact_ids = [f for f in a.fact_ids if f in known]
    if len(research.anchors) > MAX_ANCHORS:
        order = sorted(range(len(research.anchors)), key=lambda i: (not _mappable(research.anchors[i]), i))
        keep_idx = set(order[:MAX_ANCHORS])
        research.anchors = [a for i, a in enumerate(research.anchors) if i in keep_idx]
    if len(research.facts) < MIN_FACTS or len(research.anchors) < MIN_ANCHORS:
        log.warn("research below minimum", facts=len(research.facts), min_facts=MIN_FACTS,
                 anchors=len(research.anchors), min_anchors=MIN_ANCHORS)
    return research


def _mappable(a: VisualAnchor) -> bool:
    return a.kind == "place" and a.lat is not None and a.lon is not None


def _conf(v: Any) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.6


def _status(v: Any) -> str:
    v = str(v or "unverified").lower()
    return v if v in {"verified", "unverified", "contradicted"} else "unverified"


def _anchor(x: Any, id_map: dict[str, str]) -> Optional[VisualAnchor]:
    if not isinstance(x, dict) or not x.get("name") or x.get("kind") not in ANCHOR_KINDS:
        return None
    try:
        lat = float(x["lat"]) if x.get("lat") is not None else None
        lon = float(x["lon"]) if x.get("lon") is not None else None
    except (TypeError, ValueError):
        lat = lon = None
    ids = [id_map.get(str(f), str(f)) for f in x.get("fact_ids") or []]
    return VisualAnchor(kind=x["kind"], name=no_em_dash(str(x["name"])), detail=no_em_dash(str(x.get("detail", ""))),
                        lat=lat, lon=lon, fact_ids=[f for f in ids if f in id_map.values()])


# ---------------------------------------------------------------- fallback

def fallback_research(brief: Brief, excerpts: dict[str, str]) -> Research:
    """Minimal unverified cards from fetched text (sentences with numbers first), else from the brief."""
    from ..script.parser import split_sentences
    facts: list[FactCard] = []
    for url, text in excerpts.items():
        title, _, body = text.partition("\n")
        picked = [s for s in split_sentences(body.replace("\n", " ")) if 8 <= len(s.split()) <= 40
                  and re.search(r"\d", s) and "=" not in s]
        for s in picked[: max(4, 30 // max(1, len(excerpts)))]:
            facts.append(FactCard(id=f"f{len(facts) + 1:02d}", claim=no_em_dash(s), source_url=url,
                                  source_title=title, confidence=0.4, date=_year(s), numbers=extract_numbers(s)))
    if not facts:
        facts = _brief_cards(brief)
    return Research(topic=brief.topic, facts=facts[:MAX_FACTS])


def _brief_cards(brief: Brief) -> list[FactCard]:
    lines = [f"This story is about {brief.topic}."]
    if brief.angle:
        lines.append(f"Its focus is {brief.angle.rstrip('.')}.")
    if brief.audience:
        lines.append(f"It is told for {brief.audience.rstrip('.')}.")
    return [FactCard(id=f"f{i + 1:02d}", claim=no_em_dash(t), source_title="brief", confidence=0.3)
            for i, t in enumerate(lines)]


def _year(text: str) -> str:
    m = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", text)
    return m.group(1) if m else ""


def extract_numbers(text: str) -> list[dict[str, Any]]:
    """[{value, unit, label}] for amounts with units, dollar figures and percentages in text."""
    out: list[dict[str, Any]] = []
    pat = rf"(\$)?(\d[\d,]*(?:\.\d+)?)\s*(million|billion|thousand)?\s*(%|percent|{'|'.join(UNIT_WORDS)})?"
    for m in re.finditer(pat, text, re.I):
        cur, num, scale, unit = m.groups()
        if not (cur or unit or scale):
            continue
        value = float(num.replace(",", ""))
        value *= {"million": 1e6, "billion": 1e9, "thousand": 1e3}.get((scale or "").lower(), 1)
        unit = "USD" if cur else (unit or "").lower().replace("%", "percent")
        out.append({"value": int(value) if value.is_integer() else value, "unit": unit,
                    "label": text[max(0, m.start() - 30):m.end() + 20].strip()})
    return out[:4]


# ---------------------------------------------------------------- anchors and sources

def _add_anchors(research: Research, log: JsonLogger) -> None:
    """Fill coordinates for place anchors and add date and place anchors when the seat gave none."""
    try:
        from ..mg.geo import find_places
    except Exception:  # another module; optional
        find_places = None
    for a in research.anchors:
        if a.kind == "place" and a.lat is None and find_places:
            hits = _safe_places(find_places, a.name, log)
            if hits:
                a.lat, a.lon = hits[0][1], hits[0][2]
    if len(research.anchors) >= MIN_ANCHORS:
        return
    have = {a.name.lower() for a in research.anchors}
    text = " ".join(f.claim for f in research.facts)
    if find_places:
        for name, lat, lon in _safe_places(find_places, text, log)[:5]:
            if name.lower() not in have:
                ids = [f.id for f in research.facts if name.lower() in f.claim.lower()]
                research.anchors.append(VisualAnchor(kind="place", name=name, lat=lat, lon=lon, fact_ids=ids))
                have.add(name.lower())
    for f in research.facts:
        y = _year(f.claim) or _year(f.date)
        if y and y not in have and len(research.anchors) < MAX_ANCHORS:
            research.anchors.append(VisualAnchor(kind="date", name=y, detail=f.claim[:80], fact_ids=[f.id]))
            have.add(y)
    _add_number_anchors(research, have)


def _add_number_anchors(research: Research, have: set[str]) -> None:
    """Number anchors (for number cards) from facts with figures while anchors are under the minimum."""
    for f in research.facts:
        if len(research.anchors) >= MIN_ANCHORS:
            return
        for n in f.numbers[:1]:
            name = f"{n.get('value')} {n.get('unit', '')}".strip()
            if n.get("value") is not None and name.lower() not in have:
                research.anchors.append(VisualAnchor(kind="number", name=name, detail=f.claim[:80], fact_ids=[f.id]))
                have.add(name.lower())


def _safe_places(fn: Any, text: str, log: JsonLogger) -> list[tuple[str, float, float]]:
    try:
        return [(str(n), float(la), float(lo)) for n, la, lo in fn(text)]
    except Exception as e:
        log.warn("place lookup failed", error=str(e)[:200])
        return []


def _merge_sources(research: Research, excerpts: dict[str, str]) -> list[dict[str, str]]:
    seen: dict[str, str] = {s.get("url", ""): s.get("title", "") for s in research.sources if s.get("url")}
    for f in research.facts:
        if f.source_url and f.source_url not in seen:
            seen[f.source_url] = f.source_title
    for url, text in excerpts.items():
        seen.setdefault(url, text.partition("\n")[0])
    return [{"url": u, "title": t} for u, t in seen.items()]
