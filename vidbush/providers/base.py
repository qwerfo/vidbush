"""Provider client: one wrapper for every external call.

Handles retries on transport errors, fallback on refusal or 4xx, provider down
marking with a 10 minute probe cycle, rate limit backoff (429 honours
Retry-After and never marks a provider down), per-provider concurrency
limits, timing, and cost recording.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..models import CostEntry, CostReport
from ..util import JsonLogger, NullLogger, write_json

PROBE_INTERVAL_S = 600.0
RATE_LIMIT_RETRIES = 6
RATE_LIMIT_MAX_WAIT_S = 60.0
# model names in config/seats.yaml that are placeholders, never real model ids
PLACEHOLDER_PREFIX = "your-"
# provider types whose requests name a model, so an empty model means not configured
MODEL_TYPES = {"openai_compat", "anthropic", "fal", "runware", "elevate", "algrow"}


class ProviderError(Exception):
    """Base provider failure."""


class TransportError(ProviderError):
    """Network, timeout or 5xx. Retried with backoff."""


class RateLimitError(TransportError):
    """HTTP 429. Retried after Retry-After (or exponential backoff); never marks a provider down."""

    def __init__(self, msg: str, retry_after: Optional[float] = None) -> None:
        super().__init__(msg)
        self.retry_after = retry_after


class RefusalError(ProviderError):
    """4xx, content refusal, bad request after adjustment. Falls back once, no retry."""


class NotConfigured(ProviderError):
    """Provider has no key or base URL."""


@dataclass
class ProviderSpec:
    """One entry in a role's ordered provider list."""

    name: str
    type: str  # openai_compat | anthropic | elevenlabs | algrow_tts | fal | runware | elevate | algrow | pexels | pixabay | tavily | serper | brave | mock
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        """True when the spec has what a real request needs. Placeholder models do not count."""
        if self.type == "mock":
            return True
        if is_placeholder_model(self.model):
            return False
        if self.type in MODEL_TYPES and not self.model:
            return False
        if self.type == "openai_compat":
            return bool(self.base_url and self.api_key)
        return bool(self.api_key)


def is_placeholder_model(model: str) -> bool:
    """A seats.yaml placeholder such as "your-strong-model"."""
    return model.strip().lower().startswith(PLACEHOLDER_PREFIX)


class CostTracker:
    """Collects CostEntry records for a project and writes cost_report.json."""

    def __init__(self, project_id: str, path: Optional[Path] = None) -> None:
        self.project_id = project_id
        self.path = path
        self._lock = threading.Lock()
        self.report = CostReport(project_id=project_id)
        if path and path.exists():
            try:
                self.report = CostReport.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass

    def add(self, entry: CostEntry) -> None:
        with self._lock:
            self.report.entries.append(entry)
            self.report.total_usd = round(sum(e.cost_usd for e in self.report.entries), 5)
            self.report.mock_calls = sum(1 for e in self.report.entries if e.mock)
            if self.path:
                write_json(self.path, self.report)

    @property
    def total_usd(self) -> float:
        """Running total of every recorded call (includes earlier runs of the project)."""
        with self._lock:
            return float(self.report.total_usd)

    def finalize(self, minutes: float) -> CostReport:
        with self._lock:
            self.report.per_minute_usd = round(self.report.total_usd / max(minutes, 0.01), 5)
            if self.path:
                write_json(self.path, self.report)
            return self.report


class HealthBoard:
    """Process-wide record of providers marked down, with probe timing."""

    def __init__(self) -> None:
        self._down: dict[str, float] = {}
        self._lock = threading.Lock()

    def is_down(self, key: str) -> bool:
        with self._lock:
            t = self._down.get(key)
            if t is None:
                return False
            if time.time() - t >= PROBE_INTERVAL_S:
                # probe window open: allow one request through
                self._down[key] = time.time()
                return False
            return True

    def mark_down(self, key: str) -> None:
        with self._lock:
            self._down[key] = time.time()

    def mark_up(self, key: str) -> None:
        with self._lock:
            self._down.pop(key, None)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._down)


HEALTH = HealthBoard()


class ConcurrencyLimiter:
    """Process-wide semaphores per provider, sized by options.max_concurrency."""

    def __init__(self) -> None:
        self._sems: dict[str, threading.BoundedSemaphore] = {}
        self._lock = threading.Lock()

    def get(self, spec: ProviderSpec) -> Optional[threading.BoundedSemaphore]:
        limit = int(spec.options.get("max_concurrency") or 0)
        if limit <= 0:
            return None
        key = f"{spec.name}:{limit}"
        with self._lock:
            if key not in self._sems:
                self._sems[key] = threading.BoundedSemaphore(limit)
            return self._sems[key]


LIMITS = ConcurrencyLimiter()


@dataclass
class CallResult:
    value: Any
    provider: ProviderSpec
    mock: bool
    cost_usd: float = 0.0
    units: float = 0.0
    unit_kind: str = ""
    credits: Optional[float] = None  # subscription credits the provider reported, recorded beside cost_usd


class ProviderClient:
    """Run a request against a role's ordered provider list."""

    def __init__(self, tracker: CostTracker, logger: Optional[JsonLogger] = None,
                 retries: int = 3, backoff_s: float = 1.5, sleep: Callable[[float], None] = time.sleep) -> None:
        self.tracker = tracker
        self.log = logger or NullLogger()
        self.retries = retries
        self.backoff_s = backoff_s
        self.sleep = sleep

    def call(self, role: str, stage: str, providers: list[ProviderSpec],
             fn: Callable[[ProviderSpec], CallResult], note: str = "") -> CallResult:
        """Try providers in order. Only this request falls back; the caller keeps its preference."""
        errors: list[str] = []
        for spec in providers:
            key = f"{spec.name}:{spec.model}"
            if not spec.configured:
                errors.append(f"{spec.name}: not configured")
                continue
            if spec.type != "mock" and HEALTH.is_down(key):
                errors.append(f"{spec.name}: marked down")
                continue
            res = self._try(role, stage, spec, fn, key, errors, note)
            if res is not None:
                return res
        raise ProviderError(f"all providers failed for {role}: " + "; ".join(errors))

    def _try(self, role: str, stage: str, spec: ProviderSpec, fn: Callable[[ProviderSpec], CallResult],
             key: str, errors: list[str], note: str) -> Optional[CallResult]:
        """Run fn on one provider with transport retries and rate limit backoff. None means move on."""
        transport_fails = rate_fails = 0
        while True:
            t0 = time.time()
            try:
                res = self._run(spec, fn)
            except RateLimitError as e:
                rate_fails += 1
                self._record(role, stage, spec, ok=False, latency=time.time() - t0, note=f"rate limit: {e}"[:300])
                if rate_fails < RATE_LIMIT_RETRIES:
                    self.sleep(self._rate_wait(e, rate_fails))
                    continue
                errors.append(f"{spec.name}: rate limited {e}")
                self.log.warn("provider rate limited", role=role, provider=spec.name, error=str(e)[:300])
                return None
            except TransportError as e:
                transport_fails += 1
                self._record(role, stage, spec, ok=False, latency=time.time() - t0, note=f"transport: {e}"[:300])
                if transport_fails < self.retries:
                    self.sleep(self.backoff_s * (2 ** (transport_fails - 1)))
                    continue
                HEALTH.mark_down(key)
                errors.append(f"{spec.name}: transport {e}")
                self.log.warn("provider marked down", role=role, provider=spec.name, error=str(e)[:300])
                return None
            except (RefusalError, NotConfigured) as e:
                self._record(role, stage, spec, ok=False, latency=time.time() - t0, note=f"refusal: {e}"[:300],
                             cost=float(getattr(e, "cost_usd", 0.0) or 0.0))
                errors.append(f"{spec.name}: {e}")
                return None
            except ProviderError as e:
                errors.append(f"{spec.name}: {e}")
                self._record(role, stage, spec, ok=False, latency=time.time() - t0, note=str(e)[:300])
                return None
            HEALTH.mark_up(key)
            self._record(role, stage, spec, ok=True, latency=time.time() - t0, note=note,
                         cost=res.cost_usd, units=res.units, unit_kind=res.unit_kind, mock=res.mock,
                         credits=res.credits)
            return res

    @staticmethod
    def _run(spec: ProviderSpec, fn: Callable[[ProviderSpec], CallResult]) -> CallResult:
        sem = LIMITS.get(spec)
        if sem is None:
            return fn(spec)
        with sem:
            return fn(spec)

    def _rate_wait(self, e: RateLimitError, n: int) -> float:
        if e.retry_after is not None and e.retry_after >= 0:
            return min(float(e.retry_after), RATE_LIMIT_MAX_WAIT_S)
        return min(self.backoff_s * (2 ** n), RATE_LIMIT_MAX_WAIT_S)

    def _record(self, role: str, stage: str, spec: ProviderSpec, ok: bool, latency: float, note: str = "",
                cost: float = 0.0, units: float = 0.0, unit_kind: str = "", mock: bool = False,
                credits: Optional[float] = None) -> None:
        self.tracker.add(CostEntry(role=role, provider=spec.name, model=spec.model, stage=stage, ok=ok,
                                   latency_s=round(latency, 3), note=note, cost_usd=round(cost, 6),
                                   units=units, unit_kind=unit_kind, mock=mock or spec.type == "mock",
                                   credits=credits))


def http_error_to_exc(status: int, body: str, retry_after: Optional[str] = None) -> ProviderError:
    """Map an HTTP status to the right exception class (429 to RateLimitError with Retry-After seconds)."""
    if status == 429:
        return RateLimitError(f"http 429: {body[:300]}", parse_retry_after(retry_after))
    if status in (408, 425) or status >= 500:
        return TransportError(f"http {status}: {body[:300]}")
    return RefusalError(f"http {status}: {body[:300]}")


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Retry-After as seconds (delta seconds or an HTTP date), or None."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None
