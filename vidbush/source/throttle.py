"""Politeness for web sources: per host token buckets, a global concurrency cap,
429 cooldowns and a per-run query budget.

slot(host) is a context manager around one network request: it takes a
token from the host's bucket (waiting at most MAX_WAIT_S, else Throttled),
holds one of the global concurrency slots, and serialises hosts listed in
SERIAL_HOSTS (YouTube). cool_down(host, seconds) pauses a host after a 429.
take_budget(run_id, adapter, shot_id) counts searches per run and adapter; usage(run_id)
reports used, budget and refusals so the run can surface a spent budget.

Fair shares: a run budget is shared by every shot that still needs candidates.
register_shots(run, adapters, shot_ids) puts shots in the pool; each active shot
may spend (budget - spent by shots that left) / active shots, so no early shot
can drain the budget before later shots search (live run 1 spent all 150 Pexels
searches in the first shots). release_shot(run, shot_id) takes a shot out of the
pool once it has enough candidates, and its unused share returns to the shots
still searching. A search turned away by the share (not by the budget) is counted
as deferred. A shot that searches again later (a re-query) rejoins the pool.
download_slot(host, workers) caps parallel downloads of a host on their own
semaphore (separate from the serial search slot), paced(host) takes the
host's rate token without its serial lock, and exclusive(host, key) makes
sure one asset (a video URL) is never downloaded twice at the same time.

Defaults follow the published limits: loc.gov about 1 search per 3 s,
Pexels 200 per hour, Pixabay 100 per minute, YouTube one yt-dlp call at a
time with a small gap. configure(settings) reads optional overrides from
settings.sources (rate_limits, max_concurrency, query_budget) when present.
"""
from __future__ import annotations

import contextlib
import threading
import time
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

MAX_WAIT_S = 15.0
# host: (requests per second, burst)
DEFAULT_RATES: dict[str, tuple[float, float]] = {
    "www.loc.gov": (1 / 3, 1), "loc.gov": (1 / 3, 1),
    "api.pexels.com": (200 / 3600, 40),
    "pixabay.com": (100 / 60, 20),
    "youtube": (0.5, 1),  # pseudo host for yt-dlp calls
    "commons.wikimedia.org": (5.0, 5), "archive.org": (2.0, 4), "images-api.nasa.gov": (2.0, 4),
}
SERIAL_HOSTS = {"youtube"}
DEFAULT_MAX_CONCURRENCY = 6
# searches per run (one pipeline run of one project) and adapter; "*" is every other adapter
DEFAULT_BUDGET: dict[str, int] = {"pexels": 150, "pixabay": 600, "loc": 300, "youtube": 250, "*": 800}


class Throttled(Exception):
    """A request was not sent: the host is cooling down, over its rate, or the budget is spent."""


class TokenBucket:
    """Classic token bucket. reserve() books the next token or refuses when the wait is too long."""

    def __init__(self, rate: float, burst: float) -> None:
        self.rate = max(rate, 1e-6)
        self.burst = max(burst, 1.0)
        self.tokens = self.burst
        self.stamp = time.monotonic()
        self.lock = threading.Lock()

    def reserve(self, max_wait: float) -> float:
        """Seconds to wait before sending; raises Throttled when that exceeds max_wait."""
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.burst, self.tokens + (now - self.stamp) * self.rate)
            self.stamp = now
            wait = 0.0 if self.tokens >= 1 else (1 - self.tokens) / self.rate
            if wait > max_wait:
                raise Throttled(f"rate limit: next slot in {wait:.0f}s")
            self.tokens -= 1
            return wait


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.rates: dict[str, tuple[float, float]] = dict(DEFAULT_RATES)
        self.buckets: dict[str, TokenBucket] = {}
        self.serial: dict[str, threading.Lock] = {h: threading.Lock() for h in SERIAL_HOSTS}
        self.cool_until: dict[str, float] = {}
        self.max_concurrency = DEFAULT_MAX_CONCURRENCY
        self.slots = threading.BoundedSemaphore(DEFAULT_MAX_CONCURRENCY)
        self.budget: dict[str, int] = dict(DEFAULT_BUDGET)
        self.spent: dict[tuple[str, str], int] = {}
        self.refused: set[tuple[str, str]] = set()
        # fair share pool per (run, adapter): active shots, each shot's spend, spend of shots that left
        self.active: dict[tuple[str, str], dict[str, None]] = {}  # insertion ordered: timeline order
        self.shot_spent: dict[tuple[str, str, str], int] = {}
        self.released: dict[tuple[str, str], set[str]] = {}
        self.released_spent: dict[tuple[str, str], int] = {}
        self.deferred: dict[tuple[str, str], int] = {}
        self.shots: dict[tuple[str, str], set[str]] = {}
        self.downloads: dict[str, tuple[int, threading.BoundedSemaphore]] = {}
        self.keyed: dict[tuple[str, str], threading.Lock] = {}


STATE = _State()


def host_of(url: str) -> str:
    return (urlparse(url).hostname or url).lower()


def configure(sources: Any) -> None:
    """Apply optional overrides from settings.sources (rate_limits, max_concurrency, query_budget)."""
    rates = getattr(sources, "rate_limits", None) or {}
    with STATE.lock:
        for host, value in rates.items():
            rate, burst = (value if isinstance(value, (list, tuple)) else (value, 1))[:2]
            STATE.rates[str(host).lower()] = (float(rate), float(burst))
            STATE.buckets.pop(str(host).lower(), None)
        cap = int(getattr(sources, "max_concurrency", 0) or DEFAULT_MAX_CONCURRENCY)
        if cap != STATE.max_concurrency:
            STATE.max_concurrency, STATE.slots = cap, threading.BoundedSemaphore(cap)
        STATE.budget.update({str(k): int(v) for k, v in (getattr(sources, "query_budget", None) or {}).items()})


def _bucket(host: str) -> Optional[TokenBucket]:
    with STATE.lock:
        if host not in STATE.rates:
            return None
        if host not in STATE.buckets:
            STATE.buckets[host] = TokenBucket(*STATE.rates[host])
        return STATE.buckets[host]


def cool_down(host: str, seconds: float) -> None:
    """Send nothing to host for seconds (after a 429)."""
    with STATE.lock:
        STATE.cool_until[host] = max(STATE.cool_until.get(host, 0.0), time.monotonic() + seconds)


def _wait_cooldown(host: str, max_wait: float) -> None:
    left = STATE.cool_until.get(host, 0.0) - time.monotonic()
    if left > max_wait:
        raise Throttled(f"{host} cooling down after a rate limit ({left:.0f}s left)")
    if left > 0:
        time.sleep(left)


@contextlib.contextmanager
def slot(host: str, max_wait: float = MAX_WAIT_S) -> Iterator[None]:
    """Rate limited, concurrency capped section for one request to host."""
    _wait_cooldown(host, max_wait)
    bucket = _bucket(host)
    serial = STATE.serial.get(host)
    with serial if serial is not None else contextlib.nullcontext():
        wait = bucket.reserve(max_wait) if bucket else 0.0
        if wait > 0:
            time.sleep(wait)
        with STATE.slots:
            yield


def _limit(adapter: str) -> int:
    return STATE.budget.get(adapter, STATE.budget.get("*", 0))


def register_shots(run_id: str, adapters: list[str], shot_ids: list[str]) -> None:
    """Put shots that need candidates into each adapter's fair share pool for the run."""
    with STATE.lock:
        for a in adapters:
            for sid in shot_ids:
                _activate((run_id, a), sid)


def _activate(key: tuple[str, str], shot_id: str) -> None:
    """Caller holds STATE.lock. A shot that left the pool and searches again brings its spend back."""
    act = STATE.active.setdefault(key, {})
    if shot_id in act:
        return
    gone = STATE.released.setdefault(key, set())
    if shot_id in gone:
        gone.discard(shot_id)
        STATE.released_spent[key] = STATE.released_spent.get(key, 0) - STATE.shot_spent.get((*key, shot_id), 0)
    act[shot_id] = None


def share(run_id: str, adapter: str) -> float:
    """Searches each active shot may spend now: what the shots that left did not use, split evenly."""
    with STATE.lock:
        return _share((run_id, adapter))


def _share(key: tuple[str, str]) -> float:
    n = len(STATE.active.get(key, ()))
    if n == 0:
        return float(_limit(key[1]))
    return (_limit(key[1]) - STATE.released_spent.get(key, 0)) / n


def _entitled(key: tuple[str, str], shot_id: str) -> int:
    """Whole searches this active shot may spend: the pool split evenly, the remainder one each to the
    first shots in timeline order, so the whole budget can be used and shares differ by at most one."""
    act = list(STATE.active.get(key, {}))
    if not act:
        return _limit(key[1])
    pool = max(0, _limit(key[1]) - STATE.released_spent.get(key, 0))
    base, extra = divmod(pool, len(act))
    rank = act.index(shot_id) if shot_id in act else len(act)
    return base + (1 if rank < extra else 0)


def take_budget(run_id: str, adapter: str, shot_id: str = "") -> bool:
    """Count one search for (run, adapter); False when the budget is spent or the shot used its share.

    Without a shot id the search only counts against the run budget (callers outside sourcing).
    A shot whose share is below one search may still take one while the budget lasts, so every shot
    gets a first search before any shot gets a second.
    """
    with STATE.lock:
        key = (run_id, adapter)
        if STATE.spent.get(key, 0) >= _limit(adapter):
            STATE.refused.add(key)
            return False
        if shot_id:
            _activate(key, shot_id)
            mine = STATE.shot_spent.get((run_id, adapter, shot_id), 0)
            if mine + 1 > max(1, _entitled(key, shot_id)):
                STATE.deferred[key] = STATE.deferred.get(key, 0) + 1
                return False
            STATE.shot_spent[(run_id, adapter, shot_id)] = mine + 1
            STATE.shots.setdefault(key, set()).add(shot_id)
        STATE.spent[key] = STATE.spent.get(key, 0) + 1
        return True


def release_shot(run_id: str, shot_id: str, adapters: Optional[list[str]] = None) -> None:
    """Take a shot out of the pools (it has enough candidates or finished searching); its unused share
    returns to the shots still searching."""
    with STATE.lock:
        for key in [k for k in STATE.active if k[0] == run_id and (adapters is None or k[1] in adapters)]:
            act = STATE.active[key]
            if shot_id in act:
                del act[shot_id]
                STATE.released.setdefault(key, set()).add(shot_id)
                STATE.released_spent[key] = STATE.released_spent.get(key, 0) + STATE.shot_spent.get(
                    (*key, shot_id), 0)


def spent(run_id: str) -> dict[str, int]:
    """Searches sent so far in a run, per adapter."""
    with STATE.lock:
        return {a: n for (r, a), n in STATE.spent.items() if r == run_id}


def usage(run_id: str) -> dict[str, dict[str, int]]:
    """Per adapter {"used", "budget", "refused", "shots", "deferred"} for a run. refused is 1 once the
    budget turned a search away; shots is how many shots searched; deferred counts searches a shot
    skipped because it had used its fair share."""
    with STATE.lock:
        names = ({a for (r, a) in STATE.spent if r == run_id} | {a for (r, a) in STATE.refused if r == run_id}
                 | {a for (r, a) in STATE.deferred if r == run_id})
        return {a: {"used": STATE.spent.get((run_id, a), 0), "budget": _limit(a),
                    "refused": int((run_id, a) in STATE.refused),
                    "shots": len(STATE.shots.get((run_id, a), ())),
                    "deferred": STATE.deferred.get((run_id, a), 0)} for a in sorted(names)}


def _download_sem(host: str, workers: int) -> threading.BoundedSemaphore:
    workers = max(1, int(workers))
    with STATE.lock:
        cur = STATE.downloads.get(host)
        if cur is None or cur[0] != workers:
            cur = (workers, threading.BoundedSemaphore(workers))
            STATE.downloads[host] = cur
        return cur[1]


@contextlib.contextmanager
def download_slot(host: str, workers: int, max_wait: float = MAX_WAIT_S) -> Iterator[None]:
    """One of at most workers parallel downloads from host (its own semaphore, not the serial lock).

    Raises Throttled when no download slot frees up within max_wait.
    """
    sem = _download_sem(host, workers)
    if not sem.acquire(timeout=max_wait):
        raise Throttled(f"{host}: no free download slot in {max_wait:.0f}s")
    try:
        yield
    finally:
        sem.release()


@contextlib.contextmanager
def paced(host: str, max_wait: float = MAX_WAIT_S) -> Iterator[None]:
    """Like slot() (cooldown, rate token, global cap) but never takes the host's serial lock."""
    _wait_cooldown(host, max_wait)
    bucket = _bucket(host)
    wait = bucket.reserve(max_wait) if bucket else 0.0
    if wait > 0:
        time.sleep(wait)
    with STATE.slots:
        yield


@contextlib.contextmanager
def exclusive(host: str, key: str) -> Iterator[None]:
    """Hold the lock of one asset of host (e.g. a video URL) so it is never fetched twice at once."""
    with STATE.lock:
        lock = STATE.keyed.setdefault((host, key), threading.Lock())
    with lock:
        yield


def reset() -> None:
    """Forget all buckets, cooldowns and budgets (tests)."""
    global STATE
    STATE = _State()
