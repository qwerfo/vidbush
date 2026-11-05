"""Algrow TTS adapter (provider type algrow_tts), ported from RadioPlay's gateway/adapters/algrow_tts.py.

Algrow fronts ElevenLabs behind an asynchronous job API:

    POST {base}/api/generate-simple   multipart form {script, voice_id, model_id}, Bearer key
         -> {success, job_id, credits_used?}
    GET  {base}/api/job-status/{id}   -> {status: pending|processing|completed|failed, audio_url, credits?}
    GET  audio_url                    -> the MP3 bytes (hosted link, no auth)

The reply carries no word timings, so the result returns words=None and narration aligns the audio
against the known script (vidbush/audio/narration.py). Algrow is a subscription: every call is
recorded as 0 USD, with the credit count Algrow reports kept in the cost entry.

Rules carried over from RadioPlay, each for a reason:
- A script under MIN_SCRIPT_CHARS is padded with break tags, because Algrow answers HTTP 400
  "Text must be at least 200 characters." The original text stays the exact prefix.
- A job is polled until a terminal status or options.timeout_s (default 600 s), because RadioPlay
  measured about 31 s plus 1 ms per character and saw queue waits of several minutes.
- A refused model moves the whole run to options.fallback_model (sticky per provider, the same rule
  as the ElevenLabs adapter), so one video never mixes models. Quota and auth errors are not the
  model's fault and are raised as they are.
"""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import requests

from .base import CallResult, ProviderSpec, RefusalError, TransportError, http_error_to_exc
from .http import USER_AGENT, get_bytes, get_json

DEFAULT_BASE = "https://api.algrow.online"
DEFAULT_MODEL = "eleven_v4"
FALLBACK_MODEL = "eleven_multilingual_v2"
MIN_SCRIPT_CHARS = 200
PAD_TAG = '<break time="1.0s" />'
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_POLL_S = 3.0
_TERMINAL = {"completed", "failed", "error", "cancelled"}
_NOT_MODEL = ("quota", "credit", "unauthorized", "invalid_api_key", "http 401", "http 403", "insufficient")
_MODEL_CACHE: dict[str, str] = {}  # provider name -> model the run switched to after a refusal
_LOCK = threading.Lock()


def _base(spec: ProviderSpec) -> str:
    return (spec.base_url or DEFAULT_BASE).rstrip("/")


def _headers(spec: ProviderSpec) -> dict[str, str]:
    return {"Authorization": f"Bearer {spec.api_key}", "Accept": "application/json", "User-Agent": USER_AGENT}


def padded_script(text: str) -> str:
    """Text padded with break tags up to MIN_SCRIPT_CHARS; longer text is returned unchanged."""
    out = text
    while len(out) < MIN_SCRIPT_CHARS:
        out += f" {PAD_TAG}"
    return out


def resolve_model(spec: ProviderSpec, requested: str) -> str:
    """Brand model when it names one, else the run's sticky fallback, else the seat model, else eleven_v4."""
    if requested and requested != "auto":
        return requested
    with _LOCK:
        if spec.name in _MODEL_CACHE:
            return _MODEL_CACHE[spec.name]
    return spec.model if spec.model and spec.model != "auto" else DEFAULT_MODEL


def reset_models() -> None:
    """Forget sticky model switches (between runs and in tests)."""
    with _LOCK:
        _MODEL_CACHE.clear()


def _float(value: Any) -> Optional[float]:
    try:
        return None if value is None or isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return None


def _envelope_error(data: Any, where: str) -> None:
    """Raise for an HTTP 200 reply that says it failed ({success: false, error|message})."""
    if isinstance(data, dict) and data.get("success") is False:
        err = data.get("error") or data.get("message") or data
        msg = str(err.get("message", err) if isinstance(err, dict) else err)[:300]
        low = msg.lower()
        if "rate" in low and "limit" in low:
            raise http_error_to_exc(429, f"{where}: {msg}")
        if "retry" in low or "timeout" in low or "timed out" in low:
            raise TransportError(f"{where}: {msg}")
        raise RefusalError(f"{where}: {msg}")


def _post_form(url: str, fields: dict[str, str], headers: dict[str, str], timeout: float) -> Any:
    """Multipart form POST (Algrow's submit route takes a form, not JSON)."""
    files = {k: (None, str(v)) for k, v in fields.items()}  # (None, value) sends a plain form field
    try:
        r = requests.post(url, files=files, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise TransportError(str(e)[:300]) from e
    if r.status_code >= 400:
        raise http_error_to_exc(r.status_code, r.text, r.headers.get("Retry-After"))
    try:
        return r.json()
    except ValueError as e:
        raise TransportError(f"bad json from Algrow submit: {r.text[:200]}") from e


def _timeout(spec: ProviderSpec) -> float:
    try:
        t = float(spec.options.get("timeout_s") or DEFAULT_TIMEOUT_S)
    except (TypeError, ValueError):
        t = DEFAULT_TIMEOUT_S
    return t if t > 0 else DEFAULT_TIMEOUT_S


def _submit(spec: ProviderSpec, script: str, voice_id: str, model_id: str) -> dict[str, Any]:
    data = _post_form(f"{_base(spec)}/api/generate-simple",
                      {"script": script, "voice_id": voice_id, "model_id": model_id},
                      _headers(spec), timeout=60)
    _envelope_error(data, "Algrow generate-simple")
    if not isinstance(data, dict) or not data.get("job_id"):
        raise TransportError(f"Algrow generate-simple returned no job_id: {str(data)[:200]}")
    return data


def _poll(spec: ProviderSpec, job_id: str, sleep: Any = None, clock: Any = None) -> dict[str, Any]:
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    poll_s = float(spec.options.get("poll_s") or DEFAULT_POLL_S)
    limit = _timeout(spec)
    deadline = clock() + limit
    while True:
        sleep(poll_s)
        job = get_json(f"{_base(spec)}/api/job-status/{job_id}", headers=_headers(spec), timeout=60) or {}
        _envelope_error(job, f"Algrow job {job_id}")
        status = str(job.get("status", "")).lower()
        if status in _TERMINAL:
            if status != "completed":
                detail = str(job.get("error") or job.get("message") or "no detail")[:300]
                if any(w in detail.lower() for w in ("policy", "moderation", "not allowed", "invalid")):
                    raise RefusalError(f"Algrow job {job_id} {status}: {detail}")
                raise TransportError(f"Algrow job {job_id} {status}: {detail}")
            return job
        if clock() >= deadline:
            raise TransportError(f"Algrow job {job_id} did not finish within {limit:.0f} s")


def _is_model_refusal(err: Exception) -> bool:
    low = str(err).lower()
    return "model" in low and not any(w in low for w in _NOT_MODEL)


def synthesize(spec: ProviderSpec, text: str, voice_id: str, model: str, out_path: Path,
               settings: Optional[dict[str, Any]] = None, cost_per_1k: float = 0.0, language: str = "",
               previous_text: str = "", next_text: str = "") -> CallResult:
    """Synthesize text to mp3. Returns value=(path, None, model_id): Algrow sends no word timings.

    settings, language, previous_text and next_text are accepted for the shared TTS interface and not
    sent: generate-simple takes only script, voice_id and model_id.
    """
    if not voice_id:
        raise RefusalError("algrow_tts needs a voice id (brand voice.voice_ids.algrow_tts or voice.voice_id)")
    model_id = resolve_model(spec, model)
    script = padded_script(text)
    try:
        submitted = _submit(spec, script, voice_id, model_id)
    except RefusalError as e:
        fallback = str(spec.options.get("fallback_model") or FALLBACK_MODEL)
        if model not in ("", "auto") or model_id == fallback or not _is_model_refusal(e):
            raise
        with _LOCK:
            _MODEL_CACHE[spec.name] = fallback  # the rest of the run stays on the fallback model
        model_id = fallback
        submitted = _submit(spec, script, voice_id, model_id)
    job = _poll(spec, str(submitted["job_id"]))
    url = job.get("audio_url") or (job.get("audio_urls") or [None])[0]
    if not url:
        raise TransportError(f"Algrow job {submitted['job_id']} completed without an audio_url")
    audio = get_bytes(str(url), timeout=120)
    if len(audio) < 512:
        raise TransportError(f"Algrow audio download is only {len(audio)} bytes")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio)
    credits = _float(submitted.get("credits_used", submitted.get("credit_cost")))
    if credits is None:
        credits = _float(job.get("credits_used", job.get("credit_cost", job.get("credits"))))
    return CallResult(value=(out_path, None, model_id), provider=spec, mock=False, cost_usd=0.0,
                      units=len(text), unit_kind="characters", credits=credits)


def probe(spec: ProviderSpec, timeout: float = 120.0) -> str:
    """Cheapest authenticated GET: the account's credit balance (no audio is made)."""
    data = get_json(f"{_base(spec)}/api/credits", headers=_headers(spec), timeout=timeout)
    credits = (data or {}).get("credits") or {}
    if "remaining" not in credits:
        raise RefusalError(f"Algrow credits: {str(data)[:200]}")
    return f"auth ok; {credits['remaining']} credits left; model {resolve_model(spec, 'auto')}"
