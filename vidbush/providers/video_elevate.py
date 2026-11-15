"""Elevate image to video adapter (provider type elevate_video), ported from RadioPlay's
gateway/adapters/elevate_video.py and ImageCast's imagecast/providers/elevate_video.py.

Elevate's catalogue (GET /v2/media/types, read 2026-11-15) lists one video model, veo_3_1 (Veo 3.1):
text to video and image to video, params.resolution 720p (1 credit) or 1080p (2 credits, the default).

    POST {base}/v2/media  {type: "video", prompt, model, operation: "image" | "prompt",
                           aspect_ratio: "landscape", source_image: <data URL>, params: {resolution}}
         -> {success, data: {id, status}}
    GET  {base}/v2/media/{id}?type=video -> {success, data: {id, status, result_url}}; poll until terminal
    GET  result_url -> the MP4 (hosted, no auth)
    GET  {base}/v2/media/usage  -> data.video {concurrent {used, limit}, hourly {used, limit, credits_used,
                                   credits_limit}}; the key's video lane is 1 at once, 40 an hour, 80 credits
    GET  {base}/v2/media/health -> data.video.status ("operational" when the lane works)

Rules, each for a reason:
- The request names the model: a POST without one answers INVALID_REQUEST (RadioPlay, 09/09/2026).
- No duration is sent: the contract has no such field and veo_3_1 renders 8 s whatever is asked
  (RadioPlay measured 8.0 s on every clip). The caller trims to the length it wants.
- The lane allows one task at a time per key. Before a submit the adapter waits while usage reports the
  lane full (up to options.capacity_wait_s), and a concurrent-limit 429 is waited out like the TTS lane's
  (tts_elevate._call); neither counts as a failure.
- Every accepted task id is written beside the output (<out>.task.json, with a hash of the request). A
  later call for the same output and request polls that task instead of submitting again, in this process
  or the next, so a timeout or a restart never pays twice for one clip. A dead task (failed, cancelled,
  unknown) is forgotten and one fresh submit is made.
- Generation time scales with load; Elevate asks clients not to abandon tasks on a fixed timeout. The
  poll deadline is options.timeout_s (900 s); a task still running then stays recorded for the retry.
- Subscription: every clip costs 0 USD and is recorded with its credits (1 at 720p, 2 at 1080p).
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

from .base import CallResult, ProviderSpec, RefusalError, TransportError
from .http import get_json
from .tts_elevate import _call, _json, unwrap

DEFAULT_BASE = "https://public-api.elevate.uno"
DEFAULT_MODEL = "veo_3_1"
RUNNER_UP = "veo_3_1 at 720p"  # the catalogue lists no second video model; the cheaper resolution is the fallback
CREDITS = {"720p": 1.0, "1080p": 2.0}
TIMEOUT_S = 900.0
POLL_S = 10.0
CAPACITY_WAIT_S = 1200.0
NATIVE_SECONDS = 8.0
_TERMINAL = {"completed", "failed", "cancelled", "canceled", "error"}


def _base(spec: ProviderSpec) -> str:
    return (spec.base_url or DEFAULT_BASE).rstrip("/")


def _opt(spec: ProviderSpec, key: str, default: float) -> float:
    try:
        v = float(spec.options.get(key) or default)
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


def resolution(spec: ProviderSpec) -> str:
    r = str(spec.options.get("resolution") or "1080p")
    return r if r in CREDITS else "1080p"


def _model(spec: ProviderSpec) -> str:
    m = (spec.model or "").strip()
    return m if m and not m.startswith("your-") else DEFAULT_MODEL


def _still_uri(still: Path) -> str:
    raw = still.read_bytes()
    mime = "image/png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def request_body(spec: ProviderSpec, prompt: str, still: Optional[Path]) -> dict[str, Any]:
    body: dict[str, Any] = {"type": "video", "prompt": prompt, "model": _model(spec), "aspect_ratio": "landscape",
                            "params": {"resolution": resolution(spec)}}
    if still is not None:
        body["operation"] = "image"
        body["source_image"] = _still_uri(still)
    else:
        body["operation"] = "prompt"
    return body


def _request_hash(body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def _sidecar(out_path: Path) -> Path:
    return out_path.with_suffix(".task.json")


def _remembered(out_path: Path, req: str) -> Optional[str]:
    p = _sidecar(out_path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return str(data.get("task_id")) if data.get("request") == req and data.get("task_id") else None


def _remember(out_path: Path, req: str, task_id: Optional[str]) -> None:
    p = _sidecar(out_path)
    if task_id is None:
        p.unlink(missing_ok=True)
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"task_id": task_id, "request": req, "ts": time.time()}), encoding="utf-8")


def lane(spec: ProviderSpec, timeout: float = 60.0) -> dict[str, Any]:
    """data.video of GET /v2/media/usage, {} when unreadable."""
    try:
        data = get_json(f"{_base(spec)}/v2/media/usage", headers={"Authorization": f"Bearer {spec.api_key}"},
                        timeout=timeout)
    except Exception:  # noqa: BLE001  a usage read never stops a request
        return {}
    return ((data or {}).get("data") or {}).get("video") or {} if isinstance(data, dict) else {}


def wait_for_slot(spec: ProviderSpec, sleep: Any = None, clock: Any = None) -> None:
    """Wait while the key's video lane reports every concurrent slot in use (shared with other work)."""
    sleep, clock = sleep or time.sleep, clock or time.monotonic
    deadline = clock() + _opt(spec, "capacity_wait_s", CAPACITY_WAIT_S)
    while True:
        conc = lane(spec).get("concurrent") or {}
        used, limit = conc.get("used"), conc.get("limit")
        if not isinstance(used, (int, float)) or not isinstance(limit, (int, float)) or used < limit:
            return
        if clock() >= deadline:
            raise TransportError(f"Elevate video lane still full ({used}/{limit}) after the capacity wait")
        sleep(_opt(spec, "poll_s", POLL_S))


def _submit(spec: ProviderSpec, body: dict[str, Any]) -> str:
    r = _call(spec, "POST", f"{_base(spec)}/v2/media", _opt(spec, "timeout_s", TIMEOUT_S), json_body=body)
    task = unwrap(_json(r), "create video")
    if not task.get("id"):
        raise TransportError(f"Elevate create video: reply has no 'id': {str(task)[:200]}")
    return str(task["id"])


def _poll(spec: ProviderSpec, task_id: str, sleep: Any = None, clock: Any = None) -> tuple[Optional[str], dict]:
    """(status, task) once terminal, ("unknown", {}) for a task the provider no longer knows, (None, {}) at
    the deadline (the task may still run)."""
    sleep, clock = sleep or time.sleep, clock or time.monotonic
    timeout = _opt(spec, "timeout_s", TIMEOUT_S)
    deadline = clock() + timeout
    while True:
        sleep(_opt(spec, "poll_s", POLL_S))
        try:
            r = _call(spec, "GET", f"{_base(spec)}/v2/media/{task_id}?type=video", 120.0, sleep=sleep, clock=clock)
            task = unwrap(_json(r), "get video task")
        except RefusalError as e:
            if "404" in str(e) or "not found" in str(e).lower():
                return "unknown", {}
            raise
        except TransportError:
            task = {"status": "polling_error"}  # a failed poll never resubmits: keep polling until the deadline
        status = str(task.get("status", "")).lower()
        if status in _TERMINAL:
            return status, task
        if clock() >= deadline:
            return None, {}


def generate_video(spec: ProviderSpec, prompt: str, still: Optional[Path], out_path: Path, seconds: float,
                   per_second_usd: float = 0.0) -> CallResult:
    """Animate a still (or a prompt alone when still is None) into an MP4 at out_path. value=out_path.

    seconds is what the caller will use; it is not sent (no duration field, veo renders 8 s)."""
    if still is not None and not Path(still).exists():
        raise RefusalError(f"still not found: {still}")
    body = request_body(spec, prompt, Path(still) if still is not None else None)
    req = _request_hash(body)
    task_id = _remembered(out_path, req)
    for attempt in range(2):
        if task_id is None:
            wait_for_slot(spec)
            task_id = _submit(spec, body)
            _remember(out_path, req, task_id)
        status, task = _poll(spec, task_id)
        if status is None:
            raise TransportError(f"Elevate video task {task_id} still running after "
                                 f"{_opt(spec, 'timeout_s', TIMEOUT_S):.0f} s; the retry resumes it")
        if status == "completed" and task.get("result_url"):
            break
        _remember(out_path, req, None)  # dead task: forget it; one fresh submit, as a first call would make
        error = str(task.get("error") or status)[:300]
        if any(w in error.lower() for w in ("policy", "safety", "moderat", "invalid", "refus")):
            raise RefusalError(f"Elevate video task {task_id} {status}: {error}")  # a refusal is never resubmitted
        if attempt == 1 or status not in ("unknown", "error", "failed"):
            raise TransportError(f"Elevate video task {task_id} {status}: {error}")
        task_id = None
    video = _call(spec, "GET", str(task["result_url"]), 300.0, auth=False).content
    if len(video) < 2048:
        raise TransportError(f"Elevate video download is only {len(video)} bytes")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(video)
    _remember(out_path, req, None)
    return CallResult(value=out_path, provider=spec, mock=False, cost_usd=0.0, units=CREDITS[resolution(spec)],
                      unit_kind="credits")


def probe(spec: ProviderSpec, timeout: float = 120.0) -> str:
    """GET /v2/media/usage with the key, then GET /v2/media/health: data.video.status must be operational.
    No clip is made."""
    auth = {"Authorization": f"Bearer {spec.api_key}"}
    data = unwrap(get_json(f"{_base(spec)}/v2/media/usage", headers=auth, timeout=timeout), "media usage")
    video = (data or {}).get("video") or {}
    health = unwrap(get_json(f"{_base(spec)}/v2/media/health", headers=auth, timeout=timeout), "media health")
    status = str(((health or {}).get("video") or {}).get("status", "unknown")).lower()
    if status != "operational":
        raise TransportError(f"Elevate video lane reports {status}")
    hourly = video.get("hourly") or {}
    return (f"auth ok; video lane {status}; model {_model(spec)} {resolution(spec)}; "
            f"hour {hourly.get('credits_used', '?')}/{hourly.get('credits_limit', '?')} credits")
