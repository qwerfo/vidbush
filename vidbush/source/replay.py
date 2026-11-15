"""Offline replay of the sourcing search and planning from a run's search cache.

    python -m vidbush.source.replay <project dir> [--cache DIR] [--shots id,id | --empty-from run.jsonl]
                                    [--adapters a,b] [--json out.json]

Runs the real registry code (_search_all, interleave, plan_shot through _plan_all) on the project's
shots as they were before judging: candidates stripped, the first QUERIES_PER_SHOT queries (the judge
appends its requeries after them). No request leaves the machine: every adapter's search is replaced by
a stub that records the miss and fails, so only search cache hits give results. The stub keeps no
keyword form (adapter.keyword_words), so the cache is read under the query form the run sent. A shot the judge turned
into a generated still gets back the media its queries were cached under. Budgets and fair shares run
as in a live run, so a miss the budget would not have sent is told apart from a miss that was sent.

For every reported shot: cached results found, matches and fillers after the relevance screen, plan
length (matches and fillers), the searches that missed the cache, and for a still shot without a matched
still the video plan its frame fallback would fetch. With --drops-from
(a run log), the "off-topic results dropped" events of that run are replayed through today's screen.
"""
from __future__ import annotations

import argparse
import json
import threading
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from ..config import load_brand, load_settings
from ..models import ShotList
from . import registry, throttle
from .base import RawResult

REPLAY_ADAPTERS = ["youtube", "wikimedia", "archive", "loc", "nara", "nasa", "pexels", "pixabay"]


class CacheMiss(Exception):
    """A replayed search found nothing in the cache; live, it would have gone to the network."""


class Log:
    def __init__(self) -> None:
        self.lines: list[tuple[str, dict[str, Any]]] = []
        self.lock = threading.Lock()

    def info(self, msg: str, **kw: Any) -> None:
        with self.lock:
            self.lines.append((msg, kw))

    warn = info


class OfflineAdapter:
    """Wraps a real adapter: same name, media and search hints; search never touches the network."""

    def __init__(self, inner: Any) -> None:
        self.inner, self.name, self.media = inner, inner.name, set(inner.media)
        if hasattr(inner, "search_hints"):
            self.search_hints = inner.search_hints
        self.misses: list[tuple[str, str, str]] = []  # (shot, query, media), sent past the budget
        self.lock = threading.Lock()

    def available(self, settings: Any) -> bool:
        return True

    def search(self, query: str, media: str, n: int, settings: Any, shot: Any = None, **kw: Any) -> list[RawResult]:
        with self.lock:
            self.misses.append((getattr(shot, "id", ""), query, media))
        raise CacheMiss(f"{self.name}: not in the search cache")

    def fetch(self, *a: Any, **kw: Any) -> list:
        return []


def make_context(project_dir: Path, cache: Optional[str], brand_id: str = "") -> SimpleNamespace:
    settings = load_settings().model_copy(deep=True)
    if cache:
        settings.source_cache_folder = cache
    settings.offline = False
    project = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
    brand = load_brand(settings, brand_id or project.get("brand_id") or "default")
    pid = f"replay_{project.get('id') or project_dir.name}"
    return SimpleNamespace(settings=settings, brand=brand, log=Log(), work=None, extra={},
                           project=SimpleNamespace(id=pid, paths=SimpleNamespace(sources=str(project_dir / "sources"))))


def load_shots(project_dir: Path) -> ShotList:
    """The final shot list with candidates and selections stripped (shots as sourcing saw them)."""
    data = json.loads((project_dir / "shotlist.json").read_text(encoding="utf-8"))
    sl = ShotList(**data)
    for s in sl.shots:
        s.candidates, s.selected = [], None
    return sl


def cached_media(shot: Any, adapters: list[Any], ctx: Any) -> Optional[str]:
    """Media the shot's queries were cached under, None when none is cached. Image wins: only a still shot
    searches for images, while a still shot also searches video sources for frames when it fetches."""
    cap = ctx.settings.sources.max_candidates
    for media in ("image", "video"):
        for a in adapters:
            variant = registry.cache_variant(a, registry.search_hints(a, ctx))
            n = registry._per_adapter_n(a, cap)
            if media in a.media and any(registry.cache_get(a.name, q, media, n, ctx.settings, variant) is not None
                                        for q in registry.shot_queries(shot)):
                return media
    return None


def restore_types(sl: ShotList, adapters: list[Any], ctx: Any) -> dict[str, str]:
    """Shots the judge turned into AI stills get back a sourced type from the media of their cached queries."""
    from ..models import ShotType
    restored = {}
    for s in sl.shots:
        if registry.shot_media(s) or s.type == ShotType.motion_graphic:
            continue
        media = cached_media(s, adapters, ctx)
        if media:
            s.type = ShotType.footage if media == "video" else ShotType.image
            restored[s.id] = s.type.value
    return restored


def replay(project_dir: Path, cache: Optional[str] = None, shot_ids: Optional[list[str]] = None,
           adapter_names: Optional[list[str]] = None) -> dict[str, Any]:
    """Search and plan every sourced shot offline; per shot report for shot_ids (all when None)."""
    ctx = make_context(project_dir, cache)
    throttle.reset()
    registry.base.configure(ctx.settings)  # budgets and rates as a live run of these settings
    factories = registry._factories()
    names = [n for n in (adapter_names or REPLAY_ADAPTERS) if n in factories and n != "local"]
    adapters = [OfflineAdapter(factories[n]()) for n in names]
    sl = load_shots(project_dir)
    restored = restore_types(sl, adapters, ctx)
    targets = [s for s in sl.shots if registry.shot_media(s)]
    cap = ctx.settings.sources.max_candidates
    with registry.SearchPools(adapters) as pools:
        found = registry._search_all(targets, adapters, ctx, pools)
    plans = registry._plan_all(sl, found, names, registry._blocklist(ctx), cap, False)
    frames = _frame_fallbacks(sl, plans, adapters, ctx)
    misses: dict[str, list[str]] = defaultdict(list)
    for a in adapters:
        for sid, q, media in a.misses:
            misses[sid].append(f"{a.name}:{media}:{q}")
    unsent = Counter((kw.get("shot"), kw.get("source")) for msg, kw in ctx.log.lines if msg == "source search failed")
    want = set(shot_ids) if shot_ids else {s.id for s in targets}
    report: dict[str, Any] = {}
    for s in sl.shots:
        if s.id not in want:
            continue
        if s.id not in found:
            report[s.id] = {"type": s.type.value, "sourced": False, "reason": "no sourced media and no cached query"}
            continue
        per = found[s.id]
        plan = plans[s.id]
        report[s.id] = {
            "type": s.type.value, "restored_type": s.id in restored, "queries": registry.shot_queries(s),
            "results": {k: len(v) for k, v in per.items() if v},
            "matches": sum(1 for v in per.values() for r in v if r.score > 0),
            "fillers": sum(1 for v in per.values() for r in v if r.score <= 0),
            "plan": len(plan), "plan_matches": sum(1 for r in plan if r.score > 0),
            "plan_sources": dict(Counter(r.source for r in plan)),
            "cache_misses": misses.get(s.id, []),
        }
        if s.id in frames:
            report[s.id]["frame_fallback_plan"] = frames[s.id]
    deferred = Counter(kw.get("source") for msg, kw in ctx.log.lines if "budget" in msg)
    return {"shots": report, "usage": throttle.usage(registry.run_id(ctx)), "restored_types": restored,
            "missed_searches": sum(len(a.misses) for a in adapters), "budget_notes": dict(deferred),
            "failed": len(unsent)}


def _frame_fallbacks(sl: ShotList, plans: dict[str, list[RawResult]], adapters: list[Any],
                     ctx: Any) -> dict[str, dict[str, int]]:
    """A still shot without a matched still searches video sources for frames when it fetches
    (registry._source_with_plan); its video plan from the cache, as {plan, matches}."""
    video = [a for a in adapters if "video" in a.media]
    out: dict[str, dict[str, int]] = {}
    with registry.SearchPools(video) as pools:
        for s in sl.shots:
            if registry.shot_media(s) != "image" or any(r.score > 0 for r in plans.get(s.id, [])):
                continue
            per = registry.search_shot(s, video, "video", registry.shot_queries(s), ctx, pools)
            plan = registry.plan_shot(registry.interleave(per, [a.name for a in video]), registry._blocklist(ctx),
                                      set(), set(), ctx.settings.sources.max_candidates)
            out[s.id] = {"plan": len(plan), "matches": sum(1 for r in plan if r.score > 0)}
    return out


def replay_drops(log_path: Path, ctx: Any, stage: str = "sourcing") -> dict[str, Any]:
    """Replay the "off-topic results dropped" events a run logged in stage through today's screen
    (cache hits only): every result now survives, as a match or as a filler."""
    index = _cache_index(ctx.settings)
    out = Counter()
    events, current = 0, ""
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("msg") == "stage start":
            current = str(row.get("stage") or "")
        if row.get("msg") != "off-topic results dropped" or current != stage:
            continue
        events += 1
        hits = index.get((row.get("source"), " ".join(str(row.get("query", "")).lower().split())))
        if hits is None:
            out["events_not_cached"] += 1
            continue
        ranked = registry.relevant(row["source"], row["query"], hits, ctx)
        out["results"] += len(ranked)
        out["now_matches"] += sum(1 for r in ranked if r.score > 0)
        out["now_fillers"] += sum(1 for r in ranked if r.score <= 0)
    return {"events": events, **out}


def _cache_index(settings: Any) -> dict[tuple[str, str], list[RawResult]]:
    """(adapter, normalised query) to the cached raw results (the first cache file found)."""
    root = registry.base.cache_dir(settings) / "search"
    index: dict[tuple[str, str], list[RawResult]] = {}
    for d in root.iterdir() if root.exists() else ():
        for f in d.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                key = (d.name, " ".join(str(data.get("query", "")).lower().split()))
                index.setdefault(key, [RawResult(**r) for r in data.get("results") or []])
            except (OSError, ValueError, TypeError):
                continue
    return index


def empty_shots(log_path: Path) -> list[str]:
    """Shot ids the run's last "sourcing done" line lists as empty."""
    empty: list[str] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if '"sourcing done"' in line:
            empty = list(json.loads(line).get("empty") or [])
    return empty


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m vidbush.source.replay", description=__doc__.split("\n\n")[0])
    p.add_argument("project", type=Path, help="project folder (shotlist.json, project.json)")
    p.add_argument("--cache", help="source cache folder (default: settings.source_cache_folder)")
    p.add_argument("--shots", help="comma separated shot ids to report")
    p.add_argument("--empty-from", type=Path, help="run log (jsonl): report the shots its sourcing left empty")
    p.add_argument("--drops-from", type=Path, help="run log (jsonl): replay its relevance drop events")
    p.add_argument("--adapters", help=f"comma separated adapters (default {','.join(REPLAY_ADAPTERS)})")
    p.add_argument("--json", type=Path, help="write the full report here")
    args = p.parse_args(argv)
    ids = args.shots.split(",") if args.shots else (empty_shots(args.empty_from) if args.empty_from else None)
    rep = replay(args.project, args.cache, ids, args.adapters.split(",") if args.adapters else None)
    if args.drops_from:
        rep["drops"] = replay_drops(args.drops_from, make_context(args.project, args.cache))
    for sid, r in rep["shots"].items():
        if not r.get("sourced", True):
            print(f"{sid}  {r['type']:<16} not sourced: {r['reason']}")
            continue
        print(f"{sid}  {r['type']:<16} results {sum(r['results'].values()):>3}  matches {r['matches']:>3}  "
              f"fillers {r['fillers']:>3}  plan {r['plan']:>2} ({r['plan_matches']} matches)  "
              f"misses {len(r['cache_misses'])}"
              + (f"  frame fallback plan {r['frame_fallback_plan']['plan']} "
                 f"({r['frame_fallback_plan']['matches']} matches)" if "frame_fallback_plan" in r else ""))
    empty = [sid for sid, r in rep["shots"].items()
             if not r.get("plan") and not r.get("frame_fallback_plan", {}).get("plan")]
    print(f"shots {len(rep['shots'])}, empty plans {len(empty)}: {', '.join(empty) or 'none'}")
    print(f"searches that missed the cache: {rep['missed_searches']}; usage {json.dumps(rep['usage'])}")
    if "drops" in rep:
        print(f"relevance drops replayed: {json.dumps(rep['drops'])}")
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
