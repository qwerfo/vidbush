"""Live provider probes: one tiny request per configured role.

Used by `vidbush doctor --probe` and the Settings page. Probes bypass the
ProviderClient (no retries, no cost entries, no down marking) and their last
result per role is cached in this process so the Settings page can show it.
"""
from __future__ import annotations

import base64
import io
import threading
import time
from typing import Any, Optional

from PIL import Image

from . import llm as llm_mod
from . import search as search_mod
from . import tts as tts_mod
from .base import ProviderError, ProviderSpec
from .http import get_json

PROBE_TIMEOUT_S = 30.0
_RESULTS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def _tiny_png_part() -> dict[str, Any]:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (40, 90, 160)).save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}}


def _probe_chat(spec: ProviderSpec, vision: bool) -> str:
    user: Any = 'Return exactly {"ok": true}'
    if vision:
        user = [{"type": "text", "text": 'Look at the image and return exactly {"ok": true}'}, _tiny_png_part()]
    llm_mod.chat(spec, "You reply with JSON only.", user, want_json=True, temperature=0.0, timeout=PROBE_TIMEOUT_S)
    return "chat ok"


def _probe_models(spec: ProviderSpec) -> str:
    data = get_json(spec.base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {spec.api_key}"},
                    timeout=PROBE_TIMEOUT_S)
    ids = [m.get("id", "") for m in (data.get("data") if isinstance(data, dict) else data) or []
           if isinstance(m, dict)]
    if ids and spec.model not in ids:
        return f"auth ok; model {spec.model} not in the provider model list (some relays omit image models)"
    return "auth ok"


def _probe_tts(spec: ProviderSpec, voice_id: str) -> str:
    base = (spec.base_url or tts_mod.DEFAULT_BASE).rstrip("/")
    v = get_json(f"{base}/v1/voices/{voice_id}", headers={"xi-api-key": spec.api_key}, timeout=PROBE_TIMEOUT_S)
    model = tts_mod.pick_model(spec, "auto")
    return f"voice {v.get('name', voice_id)} ok; auto model {model}"


def probe_spec(role: str, spec: ProviderSpec, voice_id: str = "") -> str:
    """Send one tiny request for this role; returns a note or raises ProviderError."""
    from .seats import LLM_ROLES, VISION_ROLES
    if spec.type == "elevenlabs":
        return _probe_tts(spec, voice_id or tts_mod.DEFAULT_VOICE_ID)
    if role == "search":
        search_mod.search(spec, "test", n=1)
        return "search ok"
    if spec.type == "openai_compat" and role in LLM_ROLES | VISION_ROLES:
        return _probe_chat(spec, vision=role in VISION_ROLES)
    if spec.type == "openai_compat":
        return _probe_models(spec)
    raise ProviderError(f"no free probe for {spec.type}; not checked")


def probe_seats(seats: Any, voice_id: str = "") -> dict[str, dict[str, Any]]:
    """Probe the first configured provider of every role. Roles without one are reported as mock."""
    from .seats import ROLES
    out: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        spec = next((p for p in seats.providers_for(role) if p.configured), None)
        out[role] = _probe_one(role, spec, voice_id)
    with _LOCK:
        _RESULTS.update(out)
    return out


def _probe_one(role: str, spec: Optional[ProviderSpec], voice_id: str) -> dict[str, Any]:
    if spec is None:
        return {"ok": None, "provider": None, "model": None, "note": "no configured provider: mock", "ts": time.time()}
    t0 = time.time()
    try:
        note, ok = probe_spec(role, spec, voice_id), True
    except ProviderError as e:
        note, ok = str(e)[:300], False
        if "no free probe" in note:
            ok = None
    except Exception as e:  # noqa: BLE001
        note, ok = f"{type(e).__name__}: {e}"[:300], False
    return {"ok": ok, "provider": spec.name, "model": spec.model, "note": note,
            "latency_s": round(time.time() - t0, 3), "ts": time.time()}


def last_results() -> dict[str, dict[str, Any]]:
    """Most recent probe result per role in this process (empty until a probe ran)."""
    with _LOCK:
        return {k: dict(v) for k, v in _RESULTS.items()}
