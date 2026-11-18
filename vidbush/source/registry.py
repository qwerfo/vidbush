"""Sourcing stage: find, download and keyframe candidates for footage and still shots.

source_all(shotlist, ctx, force=False) -> ShotList
    For every footage / footage_overlay shot (video) and image / image_overlay
    shot (still) without sourced candidates (or every one when force), run
    the shot's queries against the enabled adapters in parallel (one small
    pool per adapter, so a throttled source never starves the others),
    interleave
    sources for diversity, prefer assets the previous 3 shots do not use,
    download up to settings.sources.max_candidates per shot, write sidecars
    and keyframes. Network errors are logged, never fatal.
source_shot(shot, ctx, queries=None, exclude_ids=()) -> list[Candidate]
    Fresh candidates for one shot (director loop re-query). Assets already on
    the shot are skipped and ids never clash with existing ones.

Politeness: search responses are cached on disk under
<source_cache_folder>/search/<adapter>/ (SEARCH_CACHE_TTL_S) so a resume or
a re-run does not query again; each run has a query budget per adapter
(throttle.take_budget, keyed by run_id(ctx)), shared fairly by the shots that still need
candidates (queries go out in rounds, one series per adapter so a paced host never holds back
the others, and a shot with enough results leaves the pool, see _search_all); requests are rate limited per host and capped
globally in base/throttle. A throttled or over-budget search returns no
results and is logged once per adapter and run. Searches used per adapter and
run, the budget and whether it ran out are persisted in
work/source_notes.json (record_usage) so the QA report shows them after a
resume.

Query routing (v2): a shot's queries may arrive per source class (Shot.source_queries, or a
{real_thing, queries: {class: [...]}} structure passed to source_shot); each adapter gets the queries of
the classes it serves (adapter.classes), at most QUERIES_PER_SHOT. A flat list still goes to every source.
Every result keeps the query that found it, its rank in that source's answer and the query class
(RawResult.extra, then Candidate fields). The download cap counts distinct assets: a shot takes at most
one clip per asset until max_candidates assets are in, so its candidates come from as many different
assets as were found (up to the cap). source_pool sources a stretch of shots (a scene) into one shared
pool with every asset fetched once.

Search hints: an adapter that lists keyword names in search_hints gets them
from the brand (prefer_cc from brand.licence_policy.prefer_creative_commons,
language from brand.language); adapters without the attribute are called as
before. A Creative Commons first pass costs one extra search from the budget
and is dropped when the budget has only one search left. Hints are part of
the search cache key.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple, Optional, Union

from ..models import FOOTAGE_TYPES, GENERATED_SOURCES, Candidate, RealThing, Shot, ShotList, ShotType
from . import base, throttle
from .base import Media, RawResult, SourceAdapter
from .blocklist import Blocklist

if TYPE_CHECKING:
    from ..jobs.context import StageContext

STILL_SHOT_TYPES = {ShotType.image, ShotType.image_overlay}
QUERIES_PER_SHOT = 3
RECENT_SHOTS = 3
SEARCH_CACHE_TTL_S = 14 * 86400
# a cached search never outlives the download links it holds: Pixabay's image links expire after 24 hours
SEARCH_CACHE_TTL_BY_SOURCE = {"pixabay": 23 * 3600}
# One worker for a host whose token interval is longer than throttle.MAX_WAIT_S (Pexels: one every 22.5 s):
# its searches queue in the pool and each waits for its own token; a second worker would only be refused.
ADAPTER_WORKERS = {"youtube": 1, "loc": 1, "chronam": 1, "local": 2, "pexels": 1}  # chronam shares www.loc.gov
DEFAULT_ADAPTER_WORKERS = 3
UNCACHED = {"local"}  # in-memory index; its results change with the library
SOURCE_NOTES = "source_notes.json"  # in ctx.work: search usage per run, read by judge/qa.py
MAX_NOTE_RUNS = 10


def _factories() -> dict[str, Callable[[], SourceAdapter]]:
    """Adapter constructors by name, imported lazily."""
    def make(mod: str, cls: str) -> Callable[[], SourceAdapter]:
        def build() -> SourceAdapter:
            module = __import__(f"vidbush.source.{mod}", fromlist=[cls])
            return getattr(module, cls)()
        return build
    return {"local": make("local", "LocalAdapter"), "youtube": make("youtube", "YoutubeAdapter"),
            "wikimedia": make("wikimedia", "WikimediaAdapter"), "archive": make("archive", "ArchiveAdapter"),
            "loc": make("loc", "LocAdapter"), "nara": make("nara", "NaraAdapter"), "nasa": make("nasa", "NasaAdapter"),
            "pexels": make("pexels", "PexelsAdapter"), "pixabay": make("pixabay", "PixabayAdapter"),
            "webimage": make("webimage", "WebImageAdapter"), "openverse": make("openverse", "OpenverseAdapter"),
            "chronam": make("chronam", "ChronamAdapter")}


def active_adapters(ctx: "StageContext") -> list[SourceAdapter]:
    """Enabled and available adapters in settings.sources.enabled order."""
    cfg = ctx.settings.sources
    base.configure(ctx.settings)
    from . import segment
    segment.configure(getattr(ctx, "seats", None), getattr(ctx, "log", None))
    factories = _factories()
    out: list[SourceAdapter] = []
    for name in cfg.enabled:
        if name not in factories or (name == "youtube" and not cfg.allow_youtube):
            continue
        try:
            adapter = factories[name]()
        except ImportError as e:  # an adapter that cannot load is skipped, never the stage
            _note_once(ctx, f"import_{name}", "source adapter could not load, skipped", source=name, error=str(e)[:200])
            continue
        if adapter.available(ctx.settings):
            out.append(adapter)
    _setup_warnings(out, ctx)
    return out


def _setup_warnings(adapters: list[SourceAdapter], ctx: "StageContext") -> None:
    """Log, once per run, a missing YouTube JS runtime and a missing User-Agent contact."""
    names = {a.name for a in adapters}
    if "youtube" in names:
        from .youtube import runtime_warning
        msg = runtime_warning()
        if msg:
            _note_once(ctx, "youtube_runtime", msg)
    web = names - {"local"}
    if web and not ctx.settings.key("VIDBUSH_CONTACT"):
        _note_once(ctx, "contact", "VIDBUSH_CONTACT is not set: Wikimedia asks for a contact "
                                   "(email or URL) in the User-Agent and may throttle anonymous clients")


_noted: set[tuple[str, str]] = set()
_noted_lock = threading.Lock()


def run_id(ctx: "StageContext") -> str:
    """Key of one pipeline run (or one editor action): budgets and one-time warnings are per run."""
    extra = getattr(ctx, "extra", None)
    if not isinstance(extra, dict):
        return ctx.project.id
    with _noted_lock:
        return extra.setdefault("source_run_id", f"{ctx.project.id}:{uuid.uuid4().hex[:8]}")


def _note_once(ctx: "StageContext", key: str, msg: str, **fields: Any) -> None:
    rid = run_id(ctx)
    with _noted_lock:
        if (rid, key) in _noted:
            return
        _noted.add((rid, key))
    ctx.log.warn(msg, **fields)


def shot_media(shot: Shot) -> Optional[Media]:
    if shot.type in FOOTAGE_TYPES:
        return "video"
    if shot.type in STILL_SHOT_TYPES:
        return "image"
    return None


def asset_key(url: str) -> str:
    return url.strip()


def raw_key(raw: RawResult) -> str:
    """The asset a result is: its page, unless the adapter names the asset (extra asset_key: one article
    page holds many different web images)."""
    return asset_key(str((raw.extra or {}).get("asset_key") or "") or raw.page_url or raw.url)


def shot_queries(shot: Shot, queries: Optional[list[str]] = None) -> list[str]:
    qs = [q.strip() for q in (queries or shot.queries) if q and q.strip()]
    if not qs:
        qs = [" ".join(re.findall(r"[A-Za-z0-9']+", shot.intent)[:8])] if shot.intent else []
    return qs[:QUERIES_PER_SHOT]


class QueryPlan(NamedTuple):
    """A shot's queries: flat (every source) and per source class, plus the real thing it names."""

    flat: list[str]
    by_class: dict[str, list[str]]
    real_thing: Optional[RealThing]
    override: bool = False  # queries given for this search (a requery): sent as given, nothing prepended


QueriesIn = Union[None, list[str], dict[str, Any]]
# classes whose sources hold named real things: a real thing with no query for them gets its own query
THING_CLASSES = {"archive", "encyclopedic", "newspaper", "web_image"}


def _clean(qs: Any) -> list[str]:
    items = qs if isinstance(qs, (list, tuple)) else [qs]
    out: list[str] = []
    for q in items:
        q = " ".join(str(q or "").split())
        if q and q not in out:
            out.append(q)
    return out


def canonical_class(name: str) -> str:
    n = str(name or "").strip().lower()
    return base.CLASS_ALIASES.get(n, n)


def _by_class(src: Any) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for k, v in (src.items() if isinstance(src, dict) else []):
        c = canonical_class(k)
        if c in base.SOURCE_CLASSES:
            out[c] = out.get(c, []) + [q for q in _clean(v) if q not in out.get(c, [])]
    return {c: v for c, v in out.items() if v}


def _real_thing(v: Any) -> Optional[RealThing]:
    if isinstance(v, RealThing):
        return v
    if isinstance(v, dict) and str(v.get("name") or "").strip():
        return RealThing(**{k: str(v.get(k) or "") for k in ("name", "kind", "date", "place")})
    return None


def query_plan(shot: Shot, queries: QueriesIn = None) -> QueryPlan:
    """The shot's query plan. queries overrides the shot's own: a flat list (sent to every source), a
    {class: [...]} dict, or {real_thing: {name, kind, date, place}, queries: {class: [...]}}; class names
    are canonical (base.SOURCE_CLASSES) or aliases (web, video)."""
    rt = shot.real_thing
    override = queries is not None
    if isinstance(queries, dict) and "real_thing" in queries and not queries.get("queries"):
        rt = _real_thing(queries.get("real_thing")) or rt
        queries, override = None, False  # a real thing alone: the shot's own queries
    if queries is None:
        by = _by_class(shot.source_queries)
        flat = _clean(shot.queries) if by else shot_queries(shot)
    elif isinstance(queries, dict):
        inner: Any = queries
        if "queries" in queries or "real_thing" in queries:
            rt = _real_thing(queries.get("real_thing")) or rt
            inner = queries.get("queries") or {}
        by = {} if isinstance(inner, list) else _by_class(inner)
        flat = _clean(inner) if isinstance(inner, list) else []
    else:
        by, flat = {}, _clean(queries)
    if not flat and not by:
        flat = shot_queries(shot)
    return QueryPlan(flat, by, rt, override)


def thing_query(rt: RealThing) -> str:
    """Name plus the year of a real thing, the form archives and encyclopedias index it under."""
    year = base.YEAR_RE.search(rt.date or "")
    return f"{rt.name} {year.group(0)}" if year else rt.name


def adapter_queries(plan: QueryPlan, adapter: SourceAdapter) -> list[str]:
    """The queries one adapter sends for a shot: the classes it serves, round robin in class order, or
    the flat list when the shot has no per-class queries. A real thing named by the shot leads for the
    sources that hold named things when it has no query of its own there."""
    classes = getattr(adapter, "classes", base.SOURCE_CLASSES)
    if plan.by_class:
        cols = [plan.by_class[c] for c in base.SOURCE_CLASSES if c in plan.by_class and c in classes]
        qs: list[str] = []
        for i in range(max((len(c) for c in cols), default=0)):
            qs += [c[i] for c in cols if i < len(c) and c[i] not in qs]
        routed = {q for v in plan.by_class.values() for q in v}
        qs += [q for q in plan.flat if q not in routed and q not in qs]  # a flat query is for every source
        if not qs and plan.real_thing and set(classes) & THING_CLASSES:
            qs = [thing_query(plan.real_thing)]
        return qs[:QUERIES_PER_SHOT]
    qs = list(plan.flat)
    if plan.real_thing and not plan.override and set(classes) & THING_CLASSES and getattr(adapter, "name", "") != "local":
        tq = thing_query(plan.real_thing)
        qs = [tq] + [q for q in qs if q != tq]
    return qs[:QUERIES_PER_SHOT]


def query_class(plan: QueryPlan, adapter: SourceAdapter, query: str) -> str:
    """The class a query was written for ("thing" for the real thing's own query, "" for a flat query)."""
    classes = getattr(adapter, "classes", base.SOURCE_CLASSES)
    for c in base.SOURCE_CLASSES:
        if c in classes and query in plan.by_class.get(c, []):
            return c
    if plan.real_thing and query == thing_query(plan.real_thing):
        return "thing"
    return ""


def _with_class(results: list[RawResult], cls: str) -> list[RawResult]:
    if not cls:
        return results
    return [r.model_copy(update={"extra": {**r.extra, "query_class": cls}}) for r in results]


def _thing_shot(shot: Shot, plan: QueryPlan) -> Shot:
    """The shot as adapters see it: with the plan's real thing (entity search reads shot.real_thing)."""
    if plan.real_thing is shot.real_thing:
        return shot
    return shot.model_copy(update={"real_thing": plan.real_thing})


GENERATED = GENERATED_SOURCES  # never sourced, never replaced by a re-run of sourcing


def _has_sourced(shot: Shot) -> bool:
    """A shot holding only off-topic fillers is sourced again (with a fresh budget) on the next run."""
    return any(c.source not in GENERATED and "off_topic" not in c.flags for c in shot.candidates)


# ---------------------------------------------------------------- search

def search_hints(adapter: SourceAdapter, ctx: "StageContext") -> dict[str, Any]:
    """Keyword arguments the adapter declared in search_hints, filled from the brand."""
    wanted = set(getattr(adapter, "search_hints", ()) or ())
    brand = getattr(ctx, "brand", None)
    hints: dict[str, Any] = {}
    policy = getattr(brand, "licence_policy", None)
    if "prefer_cc" in wanted and getattr(policy, "prefer_creative_commons", False):
        hints["prefer_cc"] = True
    if "language" in wanted and getattr(brand, "language", ""):
        hints["language"] = str(brand.language)
    return hints


def _variant(hints: dict[str, Any]) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(hints.items()))


def search_variant(adapter: SourceAdapter, ctx: "StageContext", shot: Optional[Shot]) -> str:
    """Cache variant: the search hints, plus the real thing for an adapter whose search reads it
    (adapter.uses_real_thing, e.g. entity resolution), so two things with one query never share a search."""
    v = _variant(search_hints(adapter, ctx))
    rt = getattr(shot, "real_thing", None) if getattr(adapter, "uses_real_thing", False) else None
    if rt is not None:
        v = ",".join(x for x in (v, f"thing={rt.name}|{rt.kind}|{rt.date}|{rt.place}") if x)
    return v


def sent_query(adapter: SourceAdapter, query: str) -> str:
    """The query the adapter actually sends (its keyword form when the engine matches every word). The
    search cache is keyed on it, so long queries that shorten to the same request share one search."""
    if getattr(adapter, "uses_real_thing", False):
        return query  # entity lookup reads the whole query, so two queries never share one answer
    words = getattr(adapter, "keyword_words", 0)
    return base.archive_query(query, words) if words else query


def _search_one(adapter: SourceAdapter, query: str, media: Media, n: int, shot: Shot,
                ctx: "StageContext", unsent: Optional[set[tuple[str, str, str]]] = None) -> list[RawResult]:
    """One search: disk cache first, then the run budget (one per pass), then the adapter. Never raises.
    A search the budget deferred or the host throttled is added to unsent as (shot, source, query).
    The same request is never in flight twice: a second caller waits and reads the first one's cache."""
    if adapter.name in UNCACHED:
        return _search_sent(adapter, query, media, n, shot, ctx, unsent)
    variant = search_variant(adapter, ctx, shot)
    with throttle.exclusive(f"search:{adapter.name}", f"{sent_query(adapter, query)}|{media}|{n}|{variant}"):
        return _search_sent(adapter, query, media, n, shot, ctx, unsent)


def _search_sent(adapter: SourceAdapter, query: str, media: Media, n: int, shot: Shot,
                 ctx: "StageContext", unsent: Optional[set[tuple[str, str, str]]]) -> list[RawResult]:
    cacheable = adapter.name not in UNCACHED
    hints = search_hints(adapter, ctx)
    variant, key = search_variant(adapter, ctx, shot), sent_query(adapter, query)
    if cacheable:
        hit = cache_get(adapter.name, key, media, n, ctx.settings, variant)
        if hit is not None:
            return _screen(adapter, query, hit, shot, ctx)  # the cache holds raw results: same screen
        if not throttle.take_budget(run_id(ctx), adapter.name, shot.id):
            if unsent is not None:
                unsent.add((shot.id, adapter.name, query))
            if throttle.STATE.spent.get((run_id(ctx), adapter.name), 0) >= throttle.STATE.budget.get(
                    adapter.name, throttle.STATE.budget.get("*", 0)):
                _note_once(ctx, f"budget_{adapter.name}", "source query budget spent, skipping searches "
                           "(see qa_report.md)", source=adapter.name)
            return []  # otherwise this shot used its fair share; the rest stays for other shots
        if hints.get("prefer_cc") and not throttle.take_budget(run_id(ctx), adapter.name, shot.id):
            hints.pop("prefer_cc")  # the last search of the budget runs the normal pass only
    charged = (1 + bool(hints.get("prefer_cc"))) if cacheable else 0  # prefer_cc sends two passes
    try:
        results = adapter.search(query, media, n, ctx.settings, shot=shot, **hints)
    except throttle.Throttled as e:
        for _ in range(charged - min(charged, e.sent)):  # a search is charged only when it was sent
            throttle.refund_budget(run_id(ctx), adapter.name, shot.id)
        if unsent is not None:
            unsent.add((shot.id, adapter.name, query))
        _note_once(ctx, f"throttled_{adapter.name}", "source search throttled, some queries skipped",
                   source=adapter.name, error=str(e)[:200])
        return []
    except Exception as e:  # noqa: BLE001 - one failing source never stops sourcing
        if base.is_challenge(e):  # base cooled the host down; its searches are deferred until then
            if unsent is not None:
                unsent.add((shot.id, adapter.name, query))
            _note_once(ctx, f"challenge_{adapter.name}", "source answered with a bot check page; host paused "
                       "for its cool-down (the rest of this sourcing stage when the cool-down is longer)",
                       source=adapter.name, error=str(e)[:120])
            return []
        ctx.log.warn("source search failed", source=adapter.name, query=query, shot=shot.id, error=str(e)[:200])
        return []
    if cacheable:
        cache_put(adapter.name, key, media, n, results, ctx.settings, variant)
    return _screen(adapter, query, results, shot, ctx)


def _ranked(query: str, results: list[RawResult]) -> list[RawResult]:
    """Each result with the query that found it and its rank in the source's own answer (1 = first)."""
    return [r.model_copy(update={"extra": {**r.extra, "query": query, "rank": i + 1}}) for i, r in enumerate(results)]


def _screen(adapter: Any, query: str, results: list[RawResult], shot: Shot, ctx: "StageContext") -> list[RawResult]:
    """Record query and rank, then rank by relevance (off-topic results become fillers). There is no era
    rule: a recent upload of old film is still old film (run 3 dropped such uploads on the title year)."""
    return relevant(adapter, query, _ranked(query, results), ctx)


MIN_RELEVANCE = 1 / 3  # at least one in three of the query's specific words (2 of 6 counts)


def relevant(adapter: Any, query: str, results: list[RawResult], ctx: "StageContext") -> list[RawResult]:
    """Rank web results: one whose title, description and tags share fewer than MIN_RELEVANCE of the
    query's specific words becomes a filler (score 0) after every match. Nothing is removed: a title is
    weak evidence and the vision judge decides on real frames. adapter is an adapter or its name; an
    engine that matches descriptions (adapter.matches_description) is trusted with its own answer, since
    the words it matched are often in text the result does not carry (run 3: 2030 demotions)."""
    name = getattr(adapter, "name", adapter)
    if name == "local" or getattr(adapter, "matches_description", False):
        return results
    out = [r if r.score <= 0 or base.relevance(query, r) >= MIN_RELEVANCE - 1e-9
           else r.model_copy(update={"score": 0.0}) for r in results]
    demoted = sum(1 for a, b in zip(results, out) if a is not b)
    if demoted:
        ctx.log.info("off-topic results demoted to fillers", source=name, query=query, demoted=demoted)
    return sorted(out, key=lambda r: r.score <= 0)


_usage_lock = threading.Lock()


def record_usage(ctx: "StageContext") -> None:
    """Persist this run's searches per adapter ({used, budget, refused}) in work/source_notes.json."""
    from . import youtube_api
    youtube_api.flush_costs(ctx)  # paid YouTube downloads into the cost report
    usage = throttle.usage(run_id(ctx))
    try:
        path = Path(ctx.work) / SOURCE_NOTES
    except (AttributeError, TypeError):
        return
    if not usage:
        return
    with _usage_lock:
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            runs = data.get("runs") if isinstance(data, dict) and isinstance(data.get("runs"), dict) else {}
            runs[run_id(ctx)] = {"ts": round(time.time(), 3), "usage": usage}
            keep = sorted(runs, key=lambda k: float(runs[k].get("ts", 0)))[-MAX_NOTE_RUNS:]
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{threading.get_ident()}.part")
            tmp.write_text(json.dumps({"runs": {k: runs[k] for k in keep}}, indent=2), encoding="utf-8")
            tmp.replace(path)
        except (OSError, ValueError, TypeError, AttributeError) as e:
            ctx.log.warn("source usage note not written", error=str(e)[:200])


def _cache_path(adapter: str, query: str, media: str, n: int, settings: Any, variant: str = "") -> Path:
    text = f"{adapter}|{media}|{n}|{' '.join(query.lower().split())}" + (f"|{variant}" if variant else "")
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return base.cache_dir(settings) / "search" / adapter / f"{key[:24]}.json"


def cache_get(adapter: str, query: str, media: str, n: int, settings: Any,
              variant: str = "") -> Optional[list[RawResult]]:
    """Cached search results younger than the source's TTL, or None. variant holds the search hints."""
    path = _cache_path(adapter, query, media, n, settings, variant)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - float(data.get("ts", 0)) > SEARCH_CACHE_TTL_BY_SOURCE.get(adapter, SEARCH_CACHE_TTL_S):
            return None
        return [RawResult(**r) for r in data.get("results") or []]
    except (OSError, ValueError, TypeError):
        return None


def cache_put(adapter: str, query: str, media: str, n: int, results: list[RawResult], settings: Any,
              variant: str = "") -> None:
    path = _cache_path(adapter, query, media, n, settings, variant)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.part")
        tmp.write_text(json.dumps({"ts": time.time(), "query": query, "media": media,
                                   "results": [r.model_dump(mode="json") for r in results]}), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass  # the cache is an optimisation only


def _per_adapter_n(adapter: SourceAdapter, cap: int) -> int:
    """Local search is an in-memory index, so it returns a deep list for the diversity planner."""
    return max(cap * 3, 30) if adapter.name == "local" else max(3, cap // 2)


class SearchPools:
    """One small thread pool per adapter so a slow or throttled source never starves the others."""

    def __init__(self, adapters: list[SourceAdapter]) -> None:
        self.pools = {a.name: ThreadPoolExecutor(max_workers=ADAPTER_WORKERS.get(a.name, DEFAULT_ADAPTER_WORKERS))
                      for a in adapters}

    def submit(self, adapter: SourceAdapter, fn: Callable[..., Any], *args: Any) -> Future:
        return self.pools[adapter.name].submit(fn, *args)

    def __enter__(self) -> "SearchPools":
        return self

    def __exit__(self, *exc: Any) -> None:
        for pool in self.pools.values():
            pool.shutdown(wait=True)


def search_shot(shot: Shot, adapters: list[SourceAdapter], media: Media, queries: Union[list[str], QueryPlan],
                ctx: "StageContext", pools: SearchPools) -> dict[str, list[RawResult]]:
    """Run the shot's queries on every adapter (in its pool; a QueryPlan routes them per class, a list goes
    to every adapter); results merged per adapter. The shot rejoins the fair share pools for these
    searches and leaves them when done (a re-query of one shot)."""
    cap = ctx.settings.sources.max_candidates
    plan = queries if isinstance(queries, QueryPlan) else QueryPlan(list(queries), {}, None)
    seen_shot = _thing_shot(shot, plan)
    jobs = [(a, q, pools.submit(a, _search_one, a, q, media, _per_adapter_n(a, cap), seen_shot, ctx))
            for a in adapters if media in a.media for q in adapter_queries(plan, a)]
    merged: dict[str, list[RawResult]] = defaultdict(list)
    for a, q, fut in jobs:
        merged[a.name].extend(_with_class(fut.result(), query_class(plan, a, q)))
    throttle.release_shot(run_id(ctx), shot.id)
    return {name: _merge(results) for name, results in merged.items()}


def _merge(results: list[RawResult]) -> list[RawResult]:
    """Dedupe by asset (best score kept, best rank counted); matches before fillers, then by rank in the
    source's answer, so every query's first result comes before any query's second (the results of one
    source's queries arrive one query after another, and a vague query must not fill the source's slots:
    the Alaska proof run lost the Balto unveiling photo, rank 1 of its query, behind a bare-name query's
    film posters), then query order."""
    best: dict[str, list[Any]] = {}
    for i, r in enumerate(results):
        k = raw_key(r)
        rank = int((r.extra or {}).get("rank") or 0)
        if k not in best:
            best[k] = [i, rank, r]
            continue
        b = best[k]
        b[1] = min(b[1], rank) if b[1] and rank else (b[1] or rank)
        if r.score > b[2].score:
            b[2] = r
    ranked = sorted(best.values(), key=lambda b: (b[2].score <= 0, b[1], b[0]))
    return [b[2] for b in ranked]


def interleave(per_adapter: dict[str, list[RawResult]], order: list[str]) -> list[RawResult]:
    """Round-robin across sources: keyword matches first, rotation fillers last."""
    out: list[RawResult] = []
    for want_match in (True, False):
        lists = [[r for r in per_adapter.get(name, []) if (r.score > 0) == want_match] for name in order]
        for i in range(max((len(x) for x in lists), default=0)):
            out.extend(lst[i] for lst in lists if i < len(lst))
    return out


def plan_shot(raws: list[RawResult], blocklist: Blocklist, recent: set[str], skip: set[str], limit: int) -> list[RawResult]:
    """Filter blocked and skipped assets, then prefer assets not used by recent shots."""
    seen: set[str] = set()
    fresh: list[RawResult] = []
    reused: list[RawResult] = []
    for r in raws:
        k = raw_key(r)
        if k in seen or k in skip or blocklist.blocked(r.page_url, r.download_url, r.url):
            continue
        seen.add(k)
        (reused if k in recent else fresh).append(r)
    return (fresh + reused)[:limit]


# ---------------------------------------------------------------- fetch

def _fetch_raws(shot: Shot, plan: list[RawResult], adapters: dict[str, SourceAdapter], ctx: "StageContext",
                cap: int, media: Media) -> list[Candidate]:
    """Fetch the plan in order until cap distinct assets gave a file, then keep at most cap clips: the
    first clip of every asset before any asset's second (run 3: a clip cap let a few videos fill the 8
    slots, a median of 4 distinct assets per shot reached the judge)."""
    out_dir = Path(ctx.project.paths.sources) / shot.id
    per_asset: list[list[Candidate]] = []
    for raw in plan:
        if len(per_asset) >= cap:
            break
        t0 = time.monotonic()
        try:
            got = adapters[raw.source].fetch(raw, out_dir, shot, ctx.settings)
            if not got:
                raise base.Rejected("no usable file")
        except base.Rejected as e:  # every dropped result is in the run log, with its reason
            ctx.log.info("source result dropped", source=raw.source, shot=shot.id, url=raw.page_url or raw.url,
                         reason=str(e)[:200])
            continue
        except Exception as e:  # noqa: BLE001 - downloads fail all the time on the open web
            ctx.log.warn("source fetch failed", source=raw.source, shot=shot.id, url=raw.page_url or raw.url,
                         error=str(e)[:200])
            continue
        if media == "image":
            got = [_still_from_video(c, out_dir) if c.media == "video" else c for c in got]
        ctx.log.info("source asset fetched", source=raw.source, shot=shot.id, clips=len(got),
                     seconds=round(time.monotonic() - t0, 1))
        per_asset.append(got)
    return spread_clips(per_asset, cap)


def spread_clips(per_asset: list[list[Candidate]], cap: int) -> list[Candidate]:
    """At most cap clips, round robin over assets in plan order: every asset's first clip, then seconds."""
    out: list[Candidate] = []
    for i in range(max((len(g) for g in per_asset), default=0)):
        for g in per_asset:
            if i < len(g) and len(out) < cap:
                out.append(g[i])
    return out


def _still_from_video(c: Candidate, out_dir: Path) -> Candidate:
    """Turn a video candidate into a still (frame at the clip middle) for a still shot."""
    from ..util import run
    src = Path(c.local_path)
    t = (c.in_point + c.out_point) / 2 if c.out_point > c.in_point else 1.0
    out = out_dir / f"{src.stem}_still_{int(t * 10)}.jpg"
    if not out.exists():
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(src),
             "-frames:v", "1", "-q:v", "2", str(out)], timeout=60)
    side = src.with_name(src.name + ".json")
    if side.exists():
        out.with_name(out.name + ".json").write_text(side.read_text(encoding="utf-8"), encoding="utf-8")
    return c.model_copy(update={"local_path": str(out), "media": "image", "in_point": 0.0, "out_point": 0.0,
                                "flags": c.flags + ["frame_grab"]})


MAX_STILL_ASPECT = 2.6  # wider stills (panoramas) lose too much to a 16:9 cover crop


def finalize(shot: Shot, cands: list[Candidate], ctx: "StageContext", taken: Iterable[str] = ()) -> list[Candidate]:
    """Assign final ids f"{shot.id}_{source}_{n}", extract keyframes, flag low resolution.

    A public domain archival still under 1280 px is flagged small_archival instead of low_res: the judge
    accepts it down to judge.MIN_ARCHIVAL_WIDTH and the render frames it on a blurred fill, never upscaled.
    """
    from ..judge.judge import archival_still
    from .keyframes import extract_keyframes
    used = set(taken)
    counters: dict[str, int] = defaultdict(int)
    kf_dir = Path(ctx.project.paths.sources) / shot.id / "keyframes"
    out: list[Candidate] = []
    for c in cands:
        src = c.source.value
        while f"{shot.id}_{src}_{counters[src]}" in used:
            counters[src] += 1
        c.id = f"{shot.id}_{src}_{counters[src]}"
        used.add(c.id)
        try:
            extract_keyframes(c, kf_dir, play_s=shot.duration)  # frames from the seconds that will play
        except Exception as e:  # noqa: BLE001 - a broken file is dropped, not fatal
            ctx.log.warn("keyframes failed, candidate dropped", shot=shot.id, candidate=c.id, error=str(e)[:200])
            continue
        if c.media == "image" and c.width and c.height and c.width / c.height > MAX_STILL_ASPECT:
            ctx.log.info("still too thin for a 16:9 frame, dropped", shot=shot.id, candidate=c.id,
                         size=f"{c.width}x{c.height}")
            continue
        if (c.media == "video" and min(c.width, c.height) < 720) or (c.media == "image" and c.width < 1280):
            c.flags = sorted(set(c.flags) | {"small_archival" if archival_still(c) else "low_res"})
        twin = _same_still(c, out)
        if twin is not None:  # one photograph reached through two sources (Commons and a web search hit)
            keep = c if _licence_rank(c) < _licence_rank(twin) or (
                _licence_rank(c) == _licence_rank(twin) and c.width > twin.width) else twin
            ctx.log.info("same picture from two sources, one kept", shot=shot.id, kept=keep.id,
                         dropped=(twin if keep is c else c).id)
            if keep is c:
                out[out.index(twin)] = c
            continue
        out.append(c)
    return out


DUP_BITS = 6  # difference hash distance (of 64 bits) under which two stills are one picture


def _dhash(path: str) -> Optional[int]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            g = im.convert("L").resize((9, 8))
            px = list(g.tobytes())  # "L" mode: one byte per pixel
    except Exception:  # noqa: BLE001 - no hash, no dedupe
        return None
    h = sum(1 << i for i in range(64) if px[(i // 8) * 9 + i % 8] > px[(i // 8) * 9 + i % 8 + 1])
    # a flat or near-flat picture has no structure to compare: never call two of them the same
    return h if 8 <= bin(h).count("1") <= 56 else None


def _same_still(c: Candidate, kept: list[Candidate]) -> Optional[Candidate]:
    """A kept still showing the same picture as c (near-identical difference hash of the keyframes)."""
    if c.media != "image" or not c.keyframes:
        return None
    h = _dhash(c.keyframes[0])
    if h is None:
        return None
    for k in kept:
        if k.media == "image" and k.keyframes:
            hk = _dhash(k.keyframes[0])
            if hk is not None and bin(h ^ hk).count("1") <= DUP_BITS:
                return k
    return None


def _licence_rank(c: Candidate) -> int:
    """0 public domain or CC0, 1 another open licence, 2 unknown or standard: Adam prefers the open copy."""
    from .licence import is_public_domain, is_standard_youtube, is_unknown
    if is_public_domain(c.licence):
        return 0
    return 2 if is_unknown(c.licence) or is_standard_youtube(c.licence) else 1


def _web_filler(r: RawResult) -> bool:
    """An off-topic web result (relevant() set its score to 0); the local library's rotation is not one."""
    return r.score <= 0 and r.source not in UNCACHED


def _source_with_plan(shot: Shot, plan: list[RawResult], adapters: list[SourceAdapter], ctx: "StageContext",
                      media: Media, skip: set[str], taken: Iterable[str]) -> list[Candidate]:
    """Fetch the plan's matches; still shots fall back to frames from video sources when no matched
    still arrived. Only when no match gave a file, a few off-topic fillers (cap // 4, at least one)
    are fetched, flagged off_topic, so the judge sees frames without every slot being downloaded."""
    cap = ctx.settings.sources.max_candidates
    by_name = {a.name: a for a in adapters}
    fillers = [r for r in plan if _web_filler(r)]
    cands = _fetch_raws(shot, [r for r in plan if not _web_filler(r)], by_name, ctx, cap, media)
    if media == "image" and not cands:
        video_adapters = [a for a in adapters if "video" in a.media]
        with SearchPools(video_adapters) as pools:
            per = search_shot(shot, video_adapters, "video", query_plan(shot), ctx, pools)
        vplan = plan_shot(interleave(per, [a.name for a in video_adapters]), _blocklist(ctx), set(), skip, cap)
        fillers += [r for r in vplan if _web_filler(r)]
        cands = _fetch_raws(shot, [r for r in vplan if not _web_filler(r)], by_name, ctx, cap, "image")
    if not cands:
        cands = [c.model_copy(update={"flags": [*c.flags, "off_topic"]})
                 for c in _fetch_raws(shot, fillers, by_name, ctx, max(1, cap // 4), media)]
    return finalize(shot, cands, ctx, taken)


def _blocklist(ctx: "StageContext") -> Blocklist:
    return Blocklist.from_settings(ctx.settings.sources.blocklist, ctx.brand.blocklist)


def _workers() -> int:
    """Shots fetched at once: more than the YouTube download slots, so a slot freed while one shot
    runs scene detection is taken by another shot's download."""
    return max(2, min(8, os.cpu_count() or 2))


# ---------------------------------------------------------------- entry points

def source_all(shotlist: ShotList, ctx: "StageContext", force: bool = False) -> ShotList:
    """Fill candidates for every footage and still shot. Never raises on source failures."""
    adapters = active_adapters(ctx)
    targets = [s for s in shotlist.shots if shot_media(s) and (force or not _has_sourced(s))]
    ctx.log.info("sourcing", shots=len(targets), adapters=[a.name for a in adapters])
    if not targets or not adapters:
        return shotlist
    order = [a.name for a in adapters]
    cap = ctx.settings.sources.max_candidates
    blocklist = _blocklist(ctx)
    with SearchPools(adapters) as pools:
        found = _search_all(targets, adapters, ctx, pools)
    plans = _plan_all(shotlist, found, order, blocklist, cap, force)
    with ThreadPoolExecutor(max_workers=_workers()) as pool:
        futs = {s.id: pool.submit(_source_with_plan, s, plans[s.id], adapters, ctx, shot_media(s), set(), [])
                for s in targets}
        results = {sid: f.result() for sid, f in futs.items()}
    record_usage(ctx)  # still shots may have searched video sources again while fetching
    for s in targets:
        keep = [c for c in s.candidates if c.source in GENERATED or not (force or "off_topic" in c.flags)]
        s.candidates = keep + results[s.id]
    ctx.log.info("sourcing done", candidates=sum(len(v) for v in results.values()),
                 empty=[sid for sid, v in results.items() if not v])
    return shotlist


def enough_results(per_adapter: dict[str, list[RawResult]], cap: int) -> bool:
    """A shot has enough when its web sources (budgeted, unlike the local library) gave a full plan of
    matches; fillers (score 0) never release a shot early."""
    keys = {raw_key(r) for name, res in per_adapter.items() if name not in UNCACHED for r in res if r.score > 0}
    return len(keys) >= cap + cap // 2


def _search_all(targets: list[Shot], adapters: list[SourceAdapter], ctx: "StageContext",
                pools: SearchPools) -> dict[str, dict[str, list[RawResult]]]:
    """Searches for every target shot in query rounds that share each run budget fairly.

    Round k sends every still searching shot's k-th query to an adapter, all submitted before any
    result is awaited; each adapter keeps its own rounds, so a paced host never holds back the others.
    After an adapter's round a shot whose web results already fill its plan leaves that adapter's
    budget pool (throttle.release_shot), so its unused share goes to the shots that still need
    candidates. Each shot spends at most its fair share of a budget (throttle.take_budget), and only
    the budgets of sources that can serve its media.

    A second pass spends what the first left over, and only on the searches it deferred or the host
    throttled (a search that failed is not sent twice): shots still short send those again, round by
    round in timeline order, and a shot rejoins a pool only when it sends one. Each video source first
    holds back one search for every still shot without a match, because such a shot searches video
    sources for frames when it fetches (_source_with_plan).
    """
    cap = ctx.settings.sources.max_candidates
    rid = run_id(ctx)
    merged: dict[str, dict[str, list[RawResult]]] = {s.id: defaultdict(list) for s in targets}
    queries = {s.id: query_plan(s) for s in targets}
    unsent: set[tuple[str, str, str]] = set()
    _rounds(targets, adapters, queries, merged, ctx, pools, unsent)
    usage = throttle.usage(rid)
    no_match = sum(1 for s in targets if shot_media(s) == "image"
                   and not any(r.score > 0 for res in merged[s.id].values() for r in res))
    allowance = {a.name: usage[a.name]["budget"] - usage[a.name]["used"] - (no_match if "video" in a.media else 0)
                 for a in adapters if a.name not in UNCACHED and a.name in usage}
    left = [a for a in adapters if allowance.get(a.name, 0) > 0]
    short = [s for s in targets if not enough_results(merged[s.id], cap) and any(u[0] == s.id for u in unsent)]
    if left and short:
        _rounds(short, left, queries, merged, ctx, pools, set(), retry=unsent, allowance=allowance)
    found = {sid: {name: _merge(res) for name, res in per.items()} for sid, per in merged.items()}
    ctx.log.info("source searches", sent=throttle.spent(rid))
    record_usage(ctx)
    return found


def _rounds(shots: list[Shot], adapters: list[SourceAdapter], queries: dict[str, QueryPlan],
            merged: dict[str, dict[str, list[RawResult]]], ctx: "StageContext", pools: SearchPools,
            unsent: set[tuple[str, str, str]], retry: Optional[set[tuple[str, str, str]]] = None,
            allowance: Optional[dict[str, int]] = None) -> None:
    """One pass of query rounds over shots (timeline order); results are added to merged and searches
    not sent to unsent. Without retry every shot joins the pools first. With retry only those
    (shot, source, query) searches are sent, at most allowance[source] per source, and a shot joins a
    pool when it takes a search (throttle.take_budget).

    Each adapter runs its own rounds: a paced host (Pexels) still busy with round k never holds back
    another host's round k + 1, and its own next round goes only to shots still short on every result
    in by then. The pass returns when every adapter's rounds are done, so no result is lost.

    Stragglers never hold the stage: once at least half of the adapters with work in this pass are
    done, the ones still searching get straggler_grace(elapsed) more, so they at most double the search
    time (the 300 s floor when the others finished at once, as on a resume from the search cache), then
    their searches not yet started are cancelled: never charged, never put in unsent. A retry pass
    (retry set) gets STRAGGLER_MIN_S in all, whatever the adapter count. Run 3 evidence: Pexels at 200 an
    hour kept sourcing searching 45 minutes after every other source had finished."""
    lock = threading.Lock()  # merged is read by every adapter's rounds
    stop = threading.Event()
    t0 = time.monotonic()
    # an adapter with no shot of its media (or no retry entry) has no work and never counts as done
    busy = [a for a in adapters if any(shot_media(s) in a.media and adapter_queries(queries[s.id], a) and (
        retry is None or any((s.id, a.name, q) in retry for q in adapter_queries(queries[s.id], a))) for s in shots)]
    with ThreadPoolExecutor(max_workers=max(1, len(busy))) as drivers:
        futs = {drivers.submit(_adapter_rounds, a, shots, queries, merged, ctx, pools, unsent, retry, allowance,
                               lock, stop): a.name for a in busy}
        not_done = set(futs)
        try:
            while not_done:
                if retry is not None:  # a retry pass has one fixed allowance of time
                    rest = wait(not_done, timeout=STRAGGLER_MIN_S)
                elif len(not_done) <= len(futs) // 2:
                    rest = wait(not_done, timeout=straggler_grace(time.monotonic() - t0))
                else:
                    done, not_done = wait(not_done, return_when=FIRST_COMPLETED)
                    for f in done:
                        f.result()
                    continue
                if rest.not_done:
                    stop.set()
                    ctx.log.info("slow source stopped searching after the others finished; its later "
                                 "searches are skipped", sources=sorted(futs[f] for f in rest.not_done))
                for f in list(rest.done) + list(rest.not_done):
                    f.result()
                not_done = set()
        finally:
            stop.set()  # an error in one driver must not leave the others running every round


STRAGGLER_MIN_S = 300.0


def straggler_grace(elapsed: float) -> float:
    """How long sources still searching may run once most have finished: as long again as the time so far."""
    return max(STRAGGLER_MIN_S, elapsed)


def _adapter_rounds(a: SourceAdapter, shots: list[Shot], queries: dict[str, QueryPlan],
                    merged: dict[str, dict[str, list[RawResult]]], ctx: "StageContext", pools: SearchPools,
                    unsent: set[tuple[str, str, str]], retry: Optional[set[tuple[str, str, str]]],
                    allowance: Optional[dict[str, int]], lock: threading.Lock,
                    stop: Optional[threading.Event] = None) -> None:
    """The query rounds of one adapter (see _rounds); a shot leaves only this adapter's pool. When stop is
    set, searches not yet started are cancelled and no new round begins."""
    cap = ctx.settings.sources.max_candidates
    rid = run_id(ctx)
    routed = {s.id: adapter_queries(queries[s.id], a) for s in shots}  # this adapter's classes only
    # a shot shares only the budgets of sources that serve its media and get one of its queries
    pending = [s for s in shots if shot_media(s) in a.media and routed[s.id]]
    if retry is None:
        throttle.register_shots(rid, [a.name], [s.id for s in pending])
    asked: set[tuple[str, str]] = set()  # (shot, sent query): a shot never sends one request twice
    for k in range(QUERIES_PER_SHOT):
        if stop is not None and stop.is_set():
            break
        jobs: list[tuple[str, str, Future]] = []
        for s in pending:
            if k >= len(routed[s.id]):
                continue
            q = routed[s.id][k]
            if (retry is not None and (s.id, a.name, q) not in retry) or (s.id, sent_query(a, q)) in asked:
                continue
            asked.add((s.id, sent_query(a, q)))
            if allowance is not None:
                if allowance[a.name] <= 0:
                    continue
                allowance[a.name] -= 1 + bool(search_hints(a, ctx).get("prefer_cc"))
            jobs.append((s.id, q, pools.submit(a, _search_one, a, q, shot_media(s), _per_adapter_n(a, cap),
                                               _thing_shot(s, queries[s.id]), ctx, unsent)))
        for i, (sid, q, fut) in enumerate(jobs):
            if stop is not None and stop.is_set():
                for _, _, later in jobs[i:]:
                    later.cancel()  # all at once: the pool would start the next one before a one by one cancel
            if fut.cancelled():
                continue
            res = _with_class(fut.result(), query_class(queries[sid], a, q))
            with lock:
                merged[sid][a.name].extend(res)
        still: list[Shot] = []
        for s in pending:
            with lock:
                short = k + 1 < len(routed[s.id]) and not enough_results(merged[s.id], cap)
            if short:
                still.append(s)
            else:
                throttle.release_shot(rid, s.id, [a.name])
        pending = still
    for s in pending:
        throttle.release_shot(rid, s.id, [a.name])


def _plan_all(shotlist: ShotList, found: dict[str, dict[str, list[RawResult]]], order: list[str],
              blocklist: Blocklist, cap: int, force: bool) -> dict[str, list[RawResult]]:
    """Sequential plan in timeline order so adjacent shots avoid sharing assets."""
    used: list[set[str]] = []
    plans: dict[str, list[RawResult]] = {}
    for s in shotlist.shots:
        if s.id in found:
            recent = set().union(*used[-RECENT_SHOTS:]) if used else set()
            plans[s.id] = plan_shot(interleave(found[s.id], order), blocklist, recent, set(), cap + cap // 2)
            used.append({raw_key(r) for r in plans[s.id][:cap]})
        else:
            used.append({asset_key(c.url) for c in s.candidates if not force})
    return plans


def source_shot(shot: Shot, ctx: "StageContext", queries: QueriesIn = None,
                exclude_ids: Iterable[str] = ()) -> list[Candidate]:
    """New candidates for one shot; does not modify the shot. queries: see query_plan (a flat list, or
    the per-class structure with an optional real_thing)."""
    media = shot_media(shot) or "image"
    adapters = active_adapters(ctx)
    if not adapters:
        return []
    skip = {asset_key(c.url) for c in shot.candidates}  # every asset already on the shot, excluded ones included
    plan = query_plan(shot, queries)
    qs = plan.flat or [q for v in plan.by_class.values() for q in v]
    with SearchPools(adapters) as pools:
        per = search_shot(shot, adapters, media, plan, ctx, pools)
    cap = ctx.settings.sources.max_candidates
    plan_rows = plan_shot(interleave(per, [a.name for a in adapters]), _blocklist(ctx), set(), skip, cap + cap // 2)
    taken = {c.id for c in shot.candidates} | set(exclude_ids)
    cands = _source_with_plan(_thing_shot(shot, plan), plan_rows, adapters, ctx, media, skip, taken)
    record_usage(ctx)  # after the fetch: a still shot may search video sources again
    ctx.log.info("re-sourced shot", shot=shot.id, queries=qs, candidates=len(cands))
    return cands


def source_pool(shots: list[Shot], ctx: "StageContext") -> dict[str, list[Candidate]]:
    """Source a stretch of shots (one scene) into one shared pool: every shot searches with its own query
    plan, then assets are deduped across the stretch and each is fetched once, for the first shot (in
    timeline order) whose plan holds it, so an edit step can assign any pool picture to any shot.
    Returns candidates per shot id (the pool is their union); does not modify the shots."""
    adapters = active_adapters(ctx)
    targets = [s for s in shots if shot_media(s)]
    if not targets or not adapters:
        return {s.id: [] for s in shots}
    order = [a.name for a in adapters]
    cap = ctx.settings.sources.max_candidates
    with SearchPools(adapters) as pools:
        found = _search_all(targets, adapters, ctx, pools)
    plans = _plan_all(ShotList(project_id=ctx.project.id, duration=0.0, shots=targets), found, order,
                      _blocklist(ctx), cap, True)
    seen: set[str] = set()
    own: dict[str, list[RawResult]] = {}
    for s in targets:
        own[s.id] = [r for r in plans.get(s.id, []) if raw_key(r) not in seen]
        seen |= {raw_key(r) for r in own[s.id]}  # fetch may walk past cap when downloads fail
    with ThreadPoolExecutor(max_workers=_workers()) as pool:
        futs = {s.id: pool.submit(_source_with_plan, s, own[s.id], adapters, ctx, shot_media(s), set(),
                                  [c.id for c in s.candidates]) for s in targets}
        out = {sid: f.result() for sid, f in futs.items()}
    record_usage(ctx)
    ctx.log.info("sourced pool", shots=len(targets), candidates=sum(len(v) for v in out.values()),
                 assets=len({c.url for v in out.values() for c in v}))
    return {s.id: out.get(s.id, []) for s in shots}
