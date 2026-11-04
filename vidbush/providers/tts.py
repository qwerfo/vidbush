"""ElevenLabs TTS adapter with character timestamps converted to word timings."""
from __future__ import annotations

import base64
import re
import threading
from pathlib import Path
from typing import Any, Optional

from .base import CallResult, ProviderSpec, RefusalError
from .http import get_json, post_json

DEFAULT_BASE = "https://api.elevenlabs.io"
# ElevenLabs premade voice "Brian" (deep, calm American narrator). Premade voices exist on every account,
# so an API key alone produces real narration. Brands override it with voice.voice_id.
DEFAULT_VOICE_ID = "nPczCjzI2devNBz1zQrb"
# preference when the brand model is "auto": consistent long-form narration first.
# Override per provider with options.model_preference in config/seats.yaml.
MODEL_PREFERENCE = ["eleven_multilingual_v2", "eleven_turbo_v2_5", "eleven_flash_v2_5"]
FALLBACK_MODEL = "eleven_multilingual_v2"
# models that accept language_code (others reject it with a 400)
LANGUAGE_MODELS = {"eleven_turbo_v2_5", "eleven_flash_v2_5", "eleven_v3"}
# models that take previous_text / next_text for prosody continuity between chunks
CONTEXT_MODELS = {"eleven_multilingual_v2", "eleven_turbo_v2_5", "eleven_flash_v2_5", "eleven_turbo_v2",
                  "eleven_monolingual_v1", "eleven_multilingual_v1", "eleven_v4"}
V3_STABILITY = (0.0, 0.5, 1.0)
V3_MAX_CHARS = 3000
_MODEL_CACHE: dict[str, str] = {}
_LOCK = threading.Lock()


def _base(spec: ProviderSpec) -> str:
    return (spec.base_url or DEFAULT_BASE).rstrip("/")


def pick_model(spec: ProviderSpec, requested: str) -> str:
    """Resolve "auto" to the first preferred TTS model the account lists. A failed lookup is not cached."""
    if requested and requested != "auto":
        return requested
    with _LOCK:
        if spec.name in _MODEL_CACHE:
            return _MODEL_CACHE[spec.name]
    try:
        models = get_json(_base(spec) + "/v1/models", headers={"xi-api-key": spec.api_key})
    except Exception:  # noqa: BLE001
        return FALLBACK_MODEL
    ids = [m.get("model_id", "") for m in models or [] if m.get("can_do_text_to_speech", True)]
    prefs = list(spec.options.get("model_preference") or MODEL_PREFERENCE)
    chosen = next((p for p in prefs if p in ids), FALLBACK_MODEL)
    with _LOCK:
        return _MODEL_CACHE.setdefault(spec.name, chosen)


def build_body(spec: ProviderSpec, text: str, model_id: str, settings: Optional[dict[str, Any]],
               language: str = "", previous_text: str = "", next_text: str = "") -> dict[str, Any]:
    """Request body adjusted to what the model accepts."""
    vs = dict(settings or {"stability": 0.45, "similarity_boost": 0.8})
    if model_id == "eleven_v3":
        if len(text) > V3_MAX_CHARS:
            raise RefusalError(f"text of {len(text)} chars exceeds the eleven_v3 limit of {V3_MAX_CHARS}")
        vs["stability"] = min(V3_STABILITY, key=lambda v: abs(v - float(vs.get("stability", 0.5))))
    body: dict[str, Any] = {"text": text, "model_id": model_id, "voice_settings": vs}
    lang_models = set(spec.options.get("language_models") or LANGUAGE_MODELS)
    if language and model_id in lang_models:
        body["language_code"] = language.split("-")[0].lower()
    if model_id in CONTEXT_MODELS:
        if previous_text:
            body["previous_text"] = previous_text
        if next_text:
            body["next_text"] = next_text
    return body


def synthesize(spec: ProviderSpec, text: str, voice_id: str, model: str, out_path: Path,
               settings: Optional[dict[str, Any]] = None, cost_per_1k: float = 0.18, language: str = "",
               previous_text: str = "", next_text: str = "") -> CallResult:
    """Synthesize text to mp3 with word timings. Returns value=(path, words)."""
    voice_id = voice_id or DEFAULT_VOICE_ID
    model_id = pick_model(spec, model)
    url = f"{_base(spec)}/v1/text-to-speech/{voice_id}/with-timestamps"
    body = build_body(spec, text, model_id, settings, language, previous_text, next_text)
    try:
        data = post_json(url, body, {"xi-api-key": spec.api_key}, timeout=300)
    except RefusalError as e:
        # an auto-picked model the API refuses switches to the fallback for the rest of the run (sticky,
        # so one video never mixes models); quota and auth errors are not the model's fault and re-raise
        low = str(e).lower()
        if model not in ("", "auto") or model_id == FALLBACK_MODEL or "model" not in low or any(
                w in low for w in ("quota", "credits", "unauthorized", "invalid_api_key", "http 401")):
            raise
        with _LOCK:
            _MODEL_CACHE[spec.name] = FALLBACK_MODEL
        model_id = FALLBACK_MODEL
        body = build_body(spec, text, model_id, settings, language, previous_text, next_text)
        data = post_json(url, body, {"xi-api-key": spec.api_key}, timeout=300)
    audio = data.get("audio_base64")
    if not audio:
        raise RefusalError(f"no audio in response: {str(data)[:200]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(audio))
    words = chars_to_words(data.get("alignment") or data.get("normalized_alignment") or {})
    cost = len(text) / 1000.0 * cost_per_1k
    return CallResult(value=(out_path, words), provider=spec, mock=False, cost_usd=cost,
                      units=len(text), unit_kind="characters")


def chars_to_words(alignment: dict[str, Any]) -> list[dict[str, Any]]:
    """Group ElevenLabs character alignment into [{text, start, end}] words."""
    chars = alignment.get("characters") or []
    starts = alignment.get("character_start_times_seconds") or []
    ends = alignment.get("character_end_times_seconds") or []
    words: list[dict[str, Any]] = []
    cur, cs, ce = "", None, None
    for ch, s, e in zip(chars, starts, ends):
        if re.match(r"\s", ch):
            if cur:
                words.append({"text": cur, "start": cs, "end": ce})
            cur, cs, ce = "", None, None
            continue
        if cs is None:
            cs = s
        cur += ch
        ce = e
    if cur:
        words.append({"text": cur, "start": cs, "end": ce})
    return words
