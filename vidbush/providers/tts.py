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
# preference order when picking "the newest eleven model" automatically
MODEL_PREFERENCE = ["eleven_v3", "eleven_multilingual_v3", "eleven_turbo_v3", "eleven_multilingual_v2",
                    "eleven_turbo_v2_5", "eleven_flash_v2_5", "eleven_monolingual_v1"]
_MODEL_CACHE: dict[str, str] = {}
_LOCK = threading.Lock()


def _base(spec: ProviderSpec) -> str:
    return (spec.base_url or DEFAULT_BASE).rstrip("/")


def pick_model(spec: ProviderSpec, requested: str) -> str:
    """Resolve "auto" to the newest TTS capable model the account lists."""
    if requested and requested != "auto":
        return requested
    with _LOCK:
        if spec.name in _MODEL_CACHE:
            return _MODEL_CACHE[spec.name]
    try:
        models = get_json(_base(spec) + "/v1/models", headers={"xi-api-key": spec.api_key})
    except Exception:  # noqa: BLE001
        models = []
    ids = [m.get("model_id", "") for m in models if m.get("can_do_text_to_speech", True)]
    chosen = next((p for p in MODEL_PREFERENCE if p in ids), None)
    if chosen is None:
        eleven = sorted((i for i in ids if i.startswith("eleven")), reverse=True)
        chosen = eleven[0] if eleven else "eleven_multilingual_v2"
    with _LOCK:
        _MODEL_CACHE[spec.name] = chosen
    return chosen


def synthesize(spec: ProviderSpec, text: str, voice_id: str, model: str, out_path: Path,
               settings: Optional[dict[str, Any]] = None, cost_per_1k: float = 0.18) -> CallResult:
    """Synthesize text to mp3 with word timings. Returns value=(path, words)."""
    if not voice_id:
        raise RefusalError("no voice_id configured for brand")
    model_id = pick_model(spec, model)
    url = f"{_base(spec)}/v1/text-to-speech/{voice_id}/with-timestamps"
    body = {"text": text, "model_id": model_id,
            "voice_settings": settings or {"stability": 0.45, "similarity_boost": 0.8}}
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
