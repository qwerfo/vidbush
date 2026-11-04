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
(throttle.take_budget, keyed by run_id(ctx)); requests are rate limited per host and capped
globally in base/throttle. A throttled or over-budget search returns no
results and is logged once per adapter and run. Searches used per adapter and
run, the budget and whether it ran out are persisted in
work/source_notes.json (record_usage) so the QA report shows them after a
resume.

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
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

from ..models import FOOTAGE_TYPES, Candidate, Shot, ShotList, ShotType, SourceKind
from . import base, throttle
from .base import Media, RawResult, SourceAdapter
from .blocklist import Blocklist

if TYPE_CHECKING:
    from ..jobs.context import StageContext

STILL_SHOT_TYPES = {ShotType.image, ShotType.image_overlay}
QUERIES_PER_SHOT = 3
RECENT_SHOTS = 3
SEARCH_CACHE_TTL_S = 14 * 86400
ADAPTER_WORKERS = {"youtube": 1, "loc": 1, "local": 2}
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
            "loc": make("loc", "LocAdapter"), "nasa": make("nasa", "NasaAdapter"),
            "pexels": make("pexels", "PexelsAdapter"), "pixabay": make("pixabay", "PixabayAdapter")}


def active_adapters(ctx: "StageContext") -> list[SourceAdapter]:
    """Enabled and available adapters in settings.sources.enabled order."""
    cfg = ctx.settings.sources
    base.configure(ctx.settings)
    factories = _factories()
    out: list[SourceAdapter] = []
    for name in cfg.enabled:
        if name not in factories or (name == "youtube" and not cfg.allow_youtube):
            continue
        adapter = factories[name]()
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
        _note_once(ctx, "contact", "VIDBUSH_CONTACT is not set: Wikimedia and loc.gov ask for a contact "
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
    return asset_key(raw.page_url or raw.url)


def shot_queries(shot: Shot, queries: Optional[list[str]] = None) -> list[str]:
    qs = [q.strip() for q in (queries or shot.queries) if q and q.strip()]
    if not qs:
        qs = [" ".join(re.findall(r"[A-Za-z0-9']+", shot.intent)[:8])] if shot.intent else []
    return qs[:QUERIES_PER_SHOT]


GENERATED = {SourceKind.motion, SourceKind.ai_image, SourceKind.ai_video}


def _has_sourced(shot: Shot) -> bool:
    return any(c.source not in GENERATED for c in shot.candidates)


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


def _search_one(adapter: SourceAdapter, query: str, media: Media, n: int, shot: Shot,
                ctx: "StageContext") -> list[RawResult]:
    """One search: disk cache first, then the run budget (one per pass), then the adapter. Never raises."""
    cacheable = adapter.name not in UNCACHED
    hints = search_hints(adapter, ctx)
    if cacheable:
        hit = cache_get(adapter.name, query, media, n, ctx.settings, _variant(hints))
        if hit is not None:
            return hit
        if not throttle.take_budget(run_id(ctx), adapter.name):
            _note_once(ctx, f"budget_{adapter.name}", "source query budget spent, skipping searches "
                       "(see qa_report.md)", source=adapter.name)
            return []
        if hints.get("prefer_cc") and not throttle.take_budget(run_id(ctx), adapter.name):
            hints.pop("prefer_cc")  # the last search of the budget runs the normal pass only
    try:
        results = adapter.search(query, media, n, ctx.settings, shot=shot, **hints)
    except throttle.Throttled as e:
        _note_once(ctx, f"throttled_{adapter.name}", "source search throttled, some queries skipped",
                   source=adapter.name, error=str(e)[:200])
        return []
    except Exception as e:  # noqa: BLE001 - one failing source never stops sourcing
        ctx.log.warn("source search failed", source=adapter.name, query=query, shot=shot.id, error=str(e)[:200])
        return []
    if cacheable:
        cache_put(adapter.name, query, media, n, results, ctx.settings, _variant(hints))
    return results


_usage_lock = threading.Lock()


def record_usage(ctx: "StageContext") -> None:
    """Persist this run's searches per adapter ({used, budget, refused}) in work/source_notes.json."""
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
    """Cached search results younger than SEARCH_CACHE_TTL_S, or None. variant holds the search hints."""
    path = _cache_path(adapter, query, media, n, settings, variant)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - float(data.get("ts", 0)) > SEARCH_CACHE_TTL_S:
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


def search_shot(shot: Shot, adapters: list[SourceAdapter], media: Media, queries: list[str],
                ctx: "StageContext", pools: SearchPools) -> dict[str, list[RawResult]]:
    """Run every query on every adapter (in its pool); results merged per adapter."""
    cap = ctx.settings.sources.max_candidates
    jobs = [(a, q, pools.submit(a, _search_one, a, q, media, _per_adapter_n(a, cap), shot, ctx))
            for a in adapters if media in a.media for q in queries]
    merged: dict[str, list[RawResult]] = defaultdict(list)
    for a, _q, fut in jobs:
        merged[a.name].extend(fut.result())
    return {name: _merge(results) for name, results in merged.items()}


def _merge(results: list[RawResult]) -> list[RawResult]:
    """Dedupe by asset, keep best score, matches before fillers, query order otherwise."""
    best: dict[str, tuple[int, RawResult]] = {}
    for i, r in enumerate(results):
        k = raw_key(r)
        if k not in best or r.score > best[k][1].score:
            best[k] = (best[k][0] if k in best else i, r)
    ranked = sorted(best.values(), key=lambda p: (p[1].score <= 0, p[0]))
    return [r for _, r in ranked]


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
    out_dir = Path(ctx.project.paths.sources) / shot.id
    cands: list[Candidate] = []
    for raw in plan:
        if len(cands) >= cap:
            break
        try:
            got = adapters[raw.source].fetch(raw, out_dir, shot, ctx.settings)
        except Exception as e:  # noqa: BLE001 - downloads fail all the time on the open web
            ctx.log.warn("source fetch failed", source=raw.source, shot=shot.id, url=raw.page_url or raw.url,
                         error=str(e)[:200])
            continue
        if media == "image":
            got = [_still_from_video(c, out_dir) if c.media == "video" else c for c in got]
        cands.extend(got[: cap - len(cands)])
    return cands


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


def finalize(shot: Shot, cands: list[Candidate], ctx: "StageContext", taken: Iterable[str] = ()) -> list[Candidate]:
    """Assign final ids f"{shot.id}_{source}_{n}", extract keyframes, flag low resolution."""
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
            extract_keyframes(c, kf_dir)
        except Exception as e:  # noqa: BLE001 - a broken file is dropped, not fatal
            ctx.log.warn("keyframes failed, candidate dropped", shot=shot.id, candidate=c.id, error=str(e)[:200])
            continue
        if (c.media == "video" and min(c.width, c.height) < 720) or (c.media == "image" and c.width < 1280):
            c.flags = sorted(set(c.flags) | {"low_res"})
        out.append(c)
    return out


def _source_with_plan(shot: Shot, plan: list[RawResult], adapters: list[SourceAdapter], ctx: "StageContext",
                      media: Media, skip: set[str], taken: Iterable[str]) -> list[Candidate]:
    """Fetch the plan; still shots fall back to frames from video sources when no still arrived."""
    cap = ctx.settings.sources.max_candidates
    by_name = {a.name: a for a in adapters}
    cands = _fetch_raws(shot, plan, by_name, ctx, cap, media)
    if media == "image" and not cands:
        video_adapters = [a for a in adapters if "video" in a.media]
        with SearchPools(video_adapters) as pools:
            per = search_shot(shot, video_adapters, "video", shot_queries(shot), ctx, pools)
        vplan = plan_shot(interleave(per, [a.name for a in video_adapters]), _blocklist(ctx), set(), skip, cap)
        cands = _fetch_raws(shot, vplan, by_name, ctx, cap, "image")
    return finalize(shot, cands, ctx, taken)


def _blocklist(ctx: "StageContext") -> Blocklist:
    return Blocklist.from_settings(ctx.settings.sources.blocklist, ctx.brand.blocklist)


def _workers() -> int:
    return max(2, min(6, os.cpu_count() or 2))


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
    for s in targets:
        keep = [c for c in s.candidates if not force or c.source in GENERATED]
        s.candidates = keep + results[s.id]
    ctx.log.info("sourcing done", candidates=sum(len(v) for v in results.values()),
                 empty=[sid for sid, v in results.items() if not v])
    return shotlist


def _search_all(targets: list[Shot], adapters: list[SourceAdapter], ctx: "StageContext",
                pools: SearchPools) -> dict[str, dict[str, list[RawResult]]]:
    """Searches for every target shot, all submitted before any result is awaited."""
    cap = ctx.settings.sources.max_candidates
    jobs: dict[str, list[tuple[str, Future]]] = {}
    for s in targets:
        media = shot_media(s)
        jobs[s.id] = [(a.name, pools.submit(a, _search_one, a, q, media, _per_adapter_n(a, cap), s, ctx))
                      for a in adapters if media in a.media for q in shot_queries(s)]
    found: dict[str, dict[str, list[RawResult]]] = {}
    for sid, futs in jobs.items():
        merged: dict[str, list[RawResult]] = defaultdict(list)
        for name, fut in futs:
            merged[name].extend(fut.result())
        found[sid] = {name: _merge(res) for name, res in merged.items()}
    ctx.log.info("source searches", sent=throttle.spent(run_id(ctx)))
    record_usage(ctx)
    return found


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


def source_shot(shot: Shot, ctx: "StageContext", queries: Optional[list[str]] = None,
                exclude_ids: Iterable[str] = ()) -> list[Candidate]:
    """New candidates for one shot; does not modify the shot."""
    media = shot_media(shot) or "image"
    adapters = active_adapters(ctx)
    if not adapters:
        return []
    skip = {asset_key(c.url) for c in shot.candidates}  # every asset already on the shot, excluded ones included
    qs = shot_queries(shot, queries)
    with SearchPools(adapters) as pools:
        per = search_shot(shot, adapters, media, qs, ctx, pools)
    record_usage(ctx)
    cap = ctx.settings.sources.max_candidates
    plan = plan_shot(interleave(per, [a.name for a in adapters]), _blocklist(ctx), set(), skip, cap + cap // 2)
    taken = {c.id for c in shot.candidates} | set(exclude_ids)
    cands = _source_with_plan(shot, plan, adapters, ctx, media, skip, taken)
    ctx.log.info("re-sourced shot", shot=shot.id, queries=qs, candidates=len(cands))
    return cands
