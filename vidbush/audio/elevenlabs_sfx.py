"""ElevenLabs sound generation, ported from RadioPlay's gateway/adapters/elevenlabs_sfx.py.

POST {base}/v1/sound-generation with header xi-api-key and JSON {text, duration_seconds,
prompt_influence}; audio/mpeg bytes come back. The API renders 0.5 to 30 s and caps the text at
450 characters. Credits are read from the character-cost response header (about 10 a second;
RadioPlay measured 0.000182 USD a credit).
"""
from __future__ import annotations

import time
from typing import Any

import requests

DEFAULT_BASE_URL = "https://api.elevenlabs.io"
MAX_SECONDS = 30.0
MIN_SECONDS = 0.5
MAX_TEXT = 450
USD_PER_CREDIT = 0.000182


class SoundGenerationError(RuntimeError):
    """The sound could not be generated (offline, no key, bad request or provider failure)."""


def generate(text: str, seconds: float, prompt_influence: float = 0.5, settings: Any = None) -> tuple[bytes, float]:
    """Render one sound. Returns (mp3 bytes, credits charged; 0.0 when the header is missing)."""
    if settings is None:
        raise SoundGenerationError("ElevenLabs sound generation needs settings")
    if settings.offline:
        raise SoundGenerationError("ElevenLabs sound generation refused: settings are offline")
    key = settings.key("ELEVENLABS_API_KEY")
    if not key:
        raise SoundGenerationError("ElevenLabs sound generation refused: ELEVENLABS_API_KEY is not set")
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise SoundGenerationError(f"duration {seconds}s is outside the {MIN_SECONDS} to {MAX_SECONDS}s range")
    if len(text) > MAX_TEXT:
        raise SoundGenerationError(f"text_too_long: {len(text)} characters over the {MAX_TEXT} cap")
    base = (settings.key("ELEVENLABS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    body = {"text": text, "duration_seconds": float(seconds), "prompt_influence": float(prompt_influence)}
    for attempt in (1, 2):  # one retry on rate limit or server error
        try:
            r = requests.post(f"{base}/v1/sound-generation", json=body, timeout=120,
                              headers={"xi-api-key": key, "Accept": "audio/mpeg"})
        except requests.RequestException as e:
            if attempt == 2:
                raise SoundGenerationError(f"ElevenLabs sound generation failed: {e}") from e
            time.sleep(5)
            continue
        if (r.status_code == 429 or r.status_code >= 500) and attempt == 1:
            time.sleep(5)
            continue
        if r.status_code != 200:
            raise SoundGenerationError(f"ElevenLabs sound generation HTTP {r.status_code}: {r.text[:300]}")
        try:
            credits = float(r.headers.get("character-cost") or 0.0)
        except ValueError:
            credits = 0.0
        return r.content, credits
    raise SoundGenerationError("ElevenLabs sound generation failed after one retry")  # not reached
