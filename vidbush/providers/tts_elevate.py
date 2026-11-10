"""Elevate TTS adapter (provider type elevate_tts), ported from RadioPlay's gateway/adapters/elevate_tts.py
and gateway/adapters/_elevate.py.

Elevate fronts ElevenLabs behind its media task API (billed against the Elevate subscription):

    POST {base}/v2/media             JSON {type: "tts", prompt, voice_id, model_id}, Bearer key
         -> {success, data: {id, status}}
    GET  {base}/v2/media/{id}?type=tts -> {success, data: {id, status, result_url}}; poll until terminal
    GET  result_url                  -> the MP3 bytes (hosted link, no auth)
    GET  {base}/v2/media/health      -> data.tts.status (the probe; "operational" when the lane works)

Errors arrive two ways: a real HTTP status, or HTTP 200 {success: false, error: {code, message}}.
The reply carries no word timings, so the result returns words=None and narration aligns the audio
against the known script (vidbush/audio/narration.py). Subscription: every call is recorded as 0 USD.

Rules carried over from RadioPlay, each for a reason:
- The request timeout is options.timeout_base_s + characters * options.timeout_per_char_s, capped at
  options.timeout_s (RadioPlay's working values 400 s + 0.002 s a character, cap 600 s), so a long chunk
  gets more time before it counts as stuck.
- A poll sleeps options.poll_s (18 s) before each GET: each poll depends on the task still running.
- An HTTP 429 that reports the concurrent-call limit (error.details used and limit, or "concurrent" in the
  message; DAILY_LIMIT_REACHED excluded) is waited out every poll_s for the whole request timeout, never
  counted as a rate limit retry. Any other 429 waits Retry-After, else 5 s doubling up to 60 s, at most
  options.rate_limit_retries (4) times.
- A task still running at its deadline keeps holding a lane slot server side, so it is remembered by
  (provider, text, voice, model): the next call for the same request (ProviderClient's transport retry)
  polls that task instead of paying for a second one. A resumed task that failed gets one fresh submit.
- A task that failed naming the voice is a refusal (the request names a voice Elevate does not carry),
  not an outage.
- voice_settings are not sent: RadioPlay sends them only for eleven_v3, and eleven_v4 takes none.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Optional

import requests

from .base import (POLICY_WORDS, CallResult, ProviderSpec, RateLimitError, RefusalError, SeatUnavailable,
                   TransportError, http_error_to_exc)
from .http import USER_AGENT, get_json
from .tts_algrow import resolve_model

DEFAULT_BASE = "https://public-api.elevate.uno"
TIMEOUT_S = 600.0
TIMEOUT_BASE_S = 400.0
TIMEOUT_PER_CHAR_S = 0.002
POLL_S = 18.0
RATE_LIMIT_RETRIES = 4
_TERMINAL = {"completed", "failed", "cancelled", "canceled", "error"}
_PENDING: dict[tuple[str, str, str, str], str] = {}  # (provider, text, voice, model) -> task id still running
_LOCK = threading.Lock()


def _base(spec: ProviderSpec) -> str:
    return (spec.base_url or DEFAULT_BASE).rstrip("/")


def _opt(spec: ProviderSpec, key: str, default: float) -> float:
    try:
        v = float(spec.options.get(key) or default)
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


def request_timeout(spec: ProviderSpec, chars: int) -> float:
    """base + characters * per_char, capped at timeout_s."""
    return min(_opt(spec, "timeout_base_s", TIMEOUT_BASE_S) + chars * _opt(spec, "timeout_per_char_s",
                                                                          TIMEOUT_PER_CHAR_S),
               _opt(spec, "timeout_s", TIMEOUT_S))


def _json(r: Any) -> Any:
    try:
        return r.json()
    except ValueError:
        return None


def is_capacity_error(r: Any) -> bool:
    """True for a 429 whose body reports the concurrent-call limit rather than a plain rate limit.

    Seen by RadioPlay on 05/09/2026: {"success": false, "error": {"code": "RATE_LIMIT_EXCEEDED", "message":
    "TTS concurrent limit reached (3/3)", "details": {"used": 3, "limit": 3}}}. A plain rate limit reuses the
    code without details, so classify on details.used and limit, or "concurrent" in the message.
    """
    payload = _json(r)
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict) or error.get("code") == "DAILY_LIMIT_REACHED":
        return False  # the daily cap carries used and limit too, but the day does not turn for hours
    details = error.get("details")
    if isinstance(details, dict) and "used" in details and "limit" in details:
        return True
    return "concurrent" in str(error.get("message", "")).lower()


def unwrap(payload: Any, operation: str) -> dict[str, Any]:
    """The data field of an Elevate envelope, or the classified error."""
    if not isinstance(payload, dict):
        raise TransportError(f"Elevate {operation} returned a non-object response")
    if payload.get("success") is False:
        error = payload.get("error") or {}
        code = str(error.get("code", "SERVICE_ERROR")).upper()
        message = str(error.get("message", "unknown Elevate error"))
        text = f"Elevate {operation} failed: {code}: {message}"[:400]
        if any(w in f"{code} {message}".lower() for w in POLICY_WORDS):
            raise RefusalError(text)
        if "RATE_LIMIT" in code:
            raise RateLimitError(text)
        if code in ("UNAUTHORIZED", "FORBIDDEN"):
            raise SeatUnavailable(text)
        if code in ("INVALID_REQUEST", "VALIDATION_ERROR") or message.lower().startswith("invalid"):
            raise RefusalError(text)
        raise TransportError(text)
    data = payload.get("data")
    if data is None:
        raise TransportError(f"Elevate {operation} returned no data")
    return data


def _call(spec: ProviderSpec, method: str, url: str, timeout: float, json_body: Optional[dict[str, Any]] = None,
          auth: bool = True, sleep: Any = None, clock: Any = None) -> Any:
    """One request with RadioPlay's 429 handling (see the module docstring). Returns the response."""
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json" if auth else "*/*"}
    if auth:
        headers["Authorization"] = f"Bearer {spec.api_key}"
    where = f"Elevate {method} {url.split('?')[0]}"
    attempt, capacity_deadline = 0, None
    while True:  # sequential on purpose: a 429 retry waits for the rate window or a lane slot to clear
        try:
            r = requests.request(method, url, json=json_body, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            raise TransportError(f"{where}: {str(e)[:300]}") from e
        if r.status_code == 429:
            if is_capacity_error(r):
                if capacity_deadline is None:
                    capacity_deadline = clock() + timeout
                if clock() >= capacity_deadline:
                    raise TransportError(f"{where} still at capacity after {timeout:.0f} s")
                sleep(_opt(spec, "poll_s", POLL_S))
                continue
            if attempt < int(_opt(spec, "rate_limit_retries", RATE_LIMIT_RETRIES)):
                attempt += 1
                ra = r.headers.get("Retry-After")
                sleep(float(ra) if ra and ra.isdigit() else min(60.0, 5.0 * 2 ** (attempt - 1)))
                continue
        if r.status_code >= 400:
            raise http_error_to_exc(r.status_code, f"{where}: {r.text}", r.headers.get("Retry-After"))
        return r


def _submit(spec: ProviderSpec, text: str, voice_id: str, model_id: str, timeout: float) -> str:
    body = {"type": "tts", "prompt": text, "voice_id": voice_id, "model_id": model_id}
    r = _call(spec, "POST", f"{_base(spec)}/v2/media", timeout, json_body=body)
    task = unwrap(_json(r), "create tts")
    if not task.get("id"):
        raise TransportError(f"Elevate create tts: reply has no 'id': {str(task)[:200]}")
    return str(task["id"])


def _poll(spec: ProviderSpec, task_id: str, timeout: float, sleep: Any = None,
          clock: Any = None) -> tuple[Optional[str], dict[str, Any]]:
    """(status, task) once terminal, or (None, {}) when the deadline passes first (the task may still run)."""
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    deadline = clock() + timeout
    while True:
        sleep(_opt(spec, "poll_s", POLL_S))
        r = _call(spec, "GET", f"{_base(spec)}/v2/media/{task_id}?type=tts", timeout, sleep=sleep, clock=clock)
        task = unwrap(_json(r), "get tts task")
        status = str(task.get("status", "")).lower()
        if status in _TERMINAL:
            return status, task
        if clock() >= deadline:
            return None, {}


def _run(spec: ProviderSpec, key: tuple[str, str, str, str], task_id: str, timeout: float) -> tuple[str, dict]:
    status, task = _poll(spec, task_id, timeout)
    if status is None:
        with _LOCK:
            _PENDING[key] = task_id  # still running and holding a lane slot: the retry resumes it
        raise TransportError(f"Elevate tts task {task_id} did not finish within {timeout:.0f} s")
    return status, task


def synthesize(spec: ProviderSpec, text: str, voice_id: str, model: str, out_path: Path,
               settings: Optional[dict[str, Any]] = None, cost_per_1k: float = 0.0, language: str = "",
               previous_text: str = "", next_text: str = "") -> CallResult:
    """Synthesize text to mp3. Returns value=(path, None, model_id): Elevate sends no word timings.

    settings, language, previous_text and next_text are accepted for the shared TTS interface and not
    sent: RadioPlay's working request carries only type, prompt, voice_id and model_id.
    """
    if not voice_id:
        raise RefusalError("elevate_tts needs a voice id (brand voice.voice_ids.elevate_tts or voice.voice_id)")
    model_id = resolve_model(spec, model)
    timeout = request_timeout(spec, len(text))
    key = (spec.name, text, voice_id, model_id)
    with _LOCK:
        task_id = _PENDING.pop(key, None)
    resumed = task_id is not None
    if not resumed:
        task_id = _submit(spec, text, voice_id, model_id, timeout)
    status, task = _run(spec, key, task_id, timeout)
    if resumed and (status != "completed" or not task.get("result_url")):
        # the resumed task is gone or failed server side: one fresh submit, as a first call would make
        task_id = _submit(spec, text, voice_id, model_id, timeout)
        status, task = _run(spec, key, task_id, timeout)
    if status != "completed" or not task.get("result_url"):
        error = str(task.get("error") or "no result_url")[:300]
        if "voice" in error.lower() or any(w in error.lower() for w in POLICY_WORDS):
            raise RefusalError(f"Elevate tts task {task_id} {status}: {error}")
        raise TransportError(f"Elevate tts task {task_id} {status}: {error}")
    audio = _call(spec, "GET", str(task["result_url"]), timeout, auth=False).content
    if len(audio) < 512:
        raise TransportError(f"Elevate audio download is only {len(audio)} bytes")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio)
    return CallResult(value=(out_path, None, model_id), provider=spec, mock=False, cost_usd=0.0,
                      units=len(text), unit_kind="characters")


def probe(spec: ProviderSpec, timeout: float = 120.0) -> str:
    """GET /v2/media/health and require data.tts.status operational (no audio is made).

    A degraded lane fails the probe (RadioPlay saw "degraded", success_rate 60.0, on 05/09/2026). A reply
    without a tts entry is judged on the HTTP status alone.
    """
    data = unwrap(get_json(f"{_base(spec)}/v2/media/health", headers={"Authorization": f"Bearer {spec.api_key}"},
                           timeout=timeout), "media health")
    tts = data.get("tts") if isinstance(data, dict) else None
    if not isinstance(tts, dict):
        return f"health ok (no tts entry); model {resolve_model(spec, 'auto')}"
    status = str(tts.get("status", "")).lower()
    if status != "operational":
        raise TransportError(f"Elevate tts {status or 'unknown'}: success_rate {tts.get('success_rate')}, "
                             f"queue {tts.get('queue')}")
    return f"tts lane {status}; model {resolve_model(spec, 'auto')}"


def reset() -> None:
    """Forget remembered running tasks (between runs and in tests)."""
    with _LOCK:
        _PENDING.clear()
