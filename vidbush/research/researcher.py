"""Research stage: sources to fact cards and visual anchors.

Online, Wikipedia (search plus extracts) and the web search seat (when a key
exists) feed page excerpts to the researcher seat (task "research"). Offline,
or when the network fails, the seat still answers from canned fixtures in mock
mode. With no usable answer, minimal unverified cards are built from the
fetched text, else from the brief itself, so later stages always run. The
fact check pass then drops contradicted cards.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..config import Settings
from ..models import Brand, Brief, FactCard, Research, VisualAnchor
from ..script import prompts
from ..util import JsonLogger, no_em_dash
from .factcheck import fact_check

MAX_PAGES = 4
MAX_EXCERPT_CHARS = 9000
MAX_PROMPT_CHARS = 30000
MIN_FACTS, MAX_FACTS = 25, 60
ANCHOR_KINDS = {"place", "date", "object", "person", "event", "number"}
UNIT_WORDS = ("miles", "mile", "feet", "foot", "ft", "km", "kilometers", "meters", "yards", "tons", "people",
              "workers", "deaths", "men", "lives", "percent", "dollars", "years", "acres", "gallons", "ships")


def run_research(brief: Brief, brand: Brand, seats: Any, settings: Settings, log: JsonLogger) -> Research:
    """Fact cards, anchors and sources for a brief. Network failures are logged, never fatal."""
    excerpts = {} if settings.offline else gather_sources(brief, seats, log)
    data = _ask(brief, seats, excerpts, log)
    research = _from_seat(data, brief, excerpts) if data is not None else None
    if research is None or not research.facts:
        log.warn("research seat gave no usable facts, using fallback cards", pages=len(excerpts))
        research = fallback_research(brief, excerpts)
    research.sources = _merge_sources(research, excerpts)
    _add_anchors(research, log)
    research = fact_check(research, seats, log, excerpts)
    log.info("research done", facts=len(research.facts), anchors=len(research.anchors),
             sources=len(research.sources))
    return research


# ---------------------------------------------------------------- sources

def gather_sources(brief: Brief, seats: Any, log: JsonLogger) -> dict[str, str]:
    """{url: text}: Wikipedia extracts for the top results plus web search snippets."""
    from ..providers.search import wikipedia_extract, wikipedia_search
    out: dict[str, str] = {}
    titles: list[tuple[str, str]] = []
    for query in _queries(brief):
        try:
            for r in wikipedia_search(query, n=3):
                if r["url"] not in {u for u, _ in titles}:
                    titles.append((r["url"], r["title"]))
        except Exception as e:  # network errors are never fatal
            log.warn("wikipedia search failed", query=query, error=str(e)[:200])
    for url, title in titles[:MAX_PAGES]:
        try:
            text = wikipedia_extract(title)
        except Exception as e:
            log.warn("wikipedia extract failed", title=title, error=str(e)[:200])
            continue
        if text:
            out[url] = f"{title}\n{_trim(text)}"
    _web_snippets(brief, seats, out, log)
    log.info("sources gathered", pages=len(out))
    return out


def _queries(brief: Brief) -> list[str]:
    qs = [brief.topic]
    if brief.angle:
        qs.append(f"{brief.topic} {brief.angle}")
    return qs


def _web_snippets(brief: Brief, seats: Any, out: dict[str, str], log: JsonLogger) -> None:
    try:
        if not seats.is_real("search"):
            return
        results = seats.web_search(f"{brief.topic} history facts", stage="research")
    except Exception as e:
        log.warn("web search failed", error=str(e)[:200])
        return
    for r in results[:8]:
        url = r.get("url", "")
        if url and url not in out and r.get("snippet"):
            out[url] = f"{r.get('title', '')}\n{r['snippet']}"


def _trim(text: str) -> str:
    """Drop reference sections and cap the length of a page extract."""
    text = re.split(r"\n==\s*(See also|References|Notes|Further reading|External links|Bibliography)\s*==",
                    text)[0]
    return text[:MAX_EXCERPT_CHARS]


# ---------------------------------------------------------------- seat

def _ask(brief: Brief, seats: Any, excerpts: dict[str, str], log: JsonLogger) -> Optional[Any]:
    parts, budget = [], MAX_PROMPT_CHARS
    for url, body in excerpts.items():
        title, _, rest = body.partition("\n")
        chunk = f"URL: {url}\nTITLE: {title}\n{rest[:max(0, budget)]}"
        budget -= len(chunk)
        parts.append(chunk)
    text = "\n\n".join(parts)
    user = prompts.RESEARCH_USER.format(topic=brief.topic, angle=brief.angle or "none",
                                        audience=brief.audience or "general", minutes=brief.target_minutes,
                                        excerpts=text or "none fetched, use well established knowledge only",
                                        min_facts=MIN_FACTS, max_facts=MAX_FACTS)
    try:
        return seats.llm_json("researcher", "research", prompts.RESEARCH_SYSTEM, user, stage="research",
                              temperature=0.2)
    except Exception as e:  # a seat failure falls back to minimal cards
        log.warn("research seat failed", error=str(e)[:200])
        return None


def _from_seat(data: Any, brief: Brief, excerpts: dict[str, str]) -> Optional[Research]:
    """Validate seat JSON into Research with ids f01.. and clean text."""
    raw = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return None
    facts: list[FactCard] = []
    id_map: dict[str, str] = {}
    for r in raw:
        if not isinstance(r, dict) or not str(r.get("claim", "")).strip():
            continue
        fid = f"f{len(facts) + 1:02d}"
        id_map[str(r.get("id", fid))] = fid
        facts.append(FactCard(
            id=fid, claim=no_em_dash(str(r["claim"])).strip(), source_url=str(r.get("source_url", "")),
            source_title=no_em_dash(str(r.get("source_title", ""))), confidence=_conf(r.get("confidence")),
            date=str(r.get("date", "") or ""), status=_status(r.get("status")),
            numbers=[n for n in r.get("numbers") or [] if isinstance(n, dict)] or extract_numbers(str(r["claim"])),
        ))
    anchors = [a for a in (_anchor(x, id_map) for x in data.get("anchors") or []) if a is not None]
    sources = [{"url": str(x.get("url", "")), "title": str(x.get("title", ""))}
               for x in data.get("sources") or [] if isinstance(x, dict) and x.get("url")]
    return Research(topic=brief.topic, facts=facts, anchors=anchors, sources=sources)


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
    if len(research.anchors) >= 5:
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
        if y and y not in have and len(research.anchors) < 10:
            research.anchors.append(VisualAnchor(kind="date", name=y, detail=f.claim[:80], fact_ids=[f.id]))
            have.add(y)


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
