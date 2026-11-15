"""Elevate image generation with reference images, for the art layer.

Request shape ported from RadioPlay's gateway/adapters/elevate_image.py (live since 05/09/2026):

    POST {base}/v2/media {type: "image", prompt, model, aspect_ratio, params, reference_images: [data URIs]}
    GET  {base}/v2/media/{id}?type=image until completed, then download data.result_url (no auth)

reference_images is sent only when there is one (Elevate rejects an empty list). Envelope errors,
polling, the browser User-Agent and the 429 handling are the image seat's own
(vidbush/providers/image.py), so the art layer behaves exactly like the image seat on errors.

Rules, each for a reason:
- The raw picture is returned uncropped: a painted map is registered against its reference, and the
  seat's 16:9 fit (with its edge trim) would move the coast before the check could see it. The art
  layer trims and fits itself, with the reference widened by the same trim.
- A task still running at the deadline is remembered by (provider, model, prompt, references); the
  next call for the same request polls it instead of paying again (the TTS adapter's rule).
- usage() reads GET /v2/media/usage so callers can stay under an hourly credit budget shared with
  other work on the same key.
"""
from __future__ import annotations

import base64
import hashlib
import io
import threading
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image

from ..providers import image as image_mod
from ..providers.base import CallResult, ProviderSpec, RefusalError, TransportError
from ..providers.http import get_bytes, get_json, post_json

DEFAULT_BASE = "https://public-api.elevate.uno"
# Credits per image, from Elevate's /v2/media/types text (gpt-image) and measured from the usage counter
# on 2026-11-15 (nano_banana and gemini models; docs/PROVIDER_NOTES.md). Unknown models count as the
# dearest known, the safe side for a budget.
CREDITS = {"gpt-image-2.5-fast": 4.0, "gpt-image-2.5-thinking": 7.0}
REF_MODELS = ("nano_banana_pro", "nano_banana_2", "gemini_flash_image", "gpt-image-2.5-fast",
              "gpt-image-2.5-thinking")
_PENDING: dict[str, str] = {}
_LOCK = threading.Lock()


def credits(model: str) -> float:
    return CREDITS.get(model, max(CREDITS.values()))


def data_uri(img: Any) -> str:
    """PNG data URI from a path, PNG bytes or an RGB array."""
    if isinstance(img, (str, Path)):
        raw = Path(img).read_bytes()
    elif isinstance(img, np.ndarray):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "PNG")
        raw = buf.getvalue()
    else:
        raw = bytes(img)
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _base(spec: ProviderSpec) -> str:
    return (spec.base_url or DEFAULT_BASE).rstrip("/")


def _key(spec: ProviderSpec, model: str, prompt: str, refs: Sequence[str]) -> str:
    h = hashlib.sha256()
    for part in (spec.name, model, prompt, *refs):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def generate(spec: ProviderSpec, prompt: str, out_path: Path, model: str = "",
             references: Sequence[Any] = (), aspect_ratio: str = "landscape",
             params: Optional[dict[str, Any]] = None, seed: Optional[int] = None) -> CallResult:
    """One picture. value = out_path holding the provider's own PNG or JPEG bytes, uncropped."""
    model = model or spec.model
    refs = [data_uri(r) for r in references]
    headers = {"Authorization": f"Bearer {spec.api_key}"}
    key = _key(spec, model, prompt, refs)
    with _LOCK:
        task_id = _PENDING.pop(key, None)
    if task_id is None:
        body: dict[str, Any] = {"type": "image", "prompt": prompt, "model": model, "aspect_ratio": aspect_ratio,
                                "params": dict(params or {})}
        if refs:
            body["reference_images"] = refs
        if seed is not None:
            body["seed"] = int(seed) % 2147483647  # RadioPlay: a seed of 2**31 or more is an INTERNAL_ERROR
        task = image_mod._elevate_data(post_json(_base(spec) + "/v2/media", body, headers, timeout=90),
                                       "Elevate art submit")
        task_id = str(task.get("id") or "")
        if not task_id:
            raise TransportError(f"Elevate art submit returned no task id: {str(task)[:200]}")
    url = f"{_base(spec)}/v2/media/{task_id}?type=image"
    try:
        task = image_mod._poll(spec, url, headers, lambda r: image_mod._elevate_data(r, f"Elevate art task {task_id}"),
                               f"Elevate art task {task_id}", 420, 6)
    except RefusalError as e:
        if "not finished within" in str(e):
            with _LOCK:
                _PENDING[key] = task_id  # still running server side: the retry polls it, never pays twice
            raise TransportError(str(e)) from e
        raise
    if not task.get("result_url"):
        raise TransportError(f"Elevate art task {task_id} completed without result_url")
    raw = get_bytes(task["result_url"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)
    return CallResult(value=out_path, provider=spec, mock=False, cost_usd=0.0, units=credits(model),
                      unit_kind="credits")


def usage(spec: ProviderSpec) -> dict[str, Any]:
    """GET /v2/media/usage data ({image: {concurrent, hourly}, video: {...}}), {} on any error."""
    try:
        data = get_json(_base(spec) + "/v2/media/usage", headers={"Authorization": f"Bearer {spec.api_key}"},
                        timeout=60)
    except Exception:  # noqa: BLE001  a usage read never stops work
        return {}
    return (data or {}).get("data") or {} if isinstance(data, dict) else {}


def hour_credits(spec: ProviderSpec, kind: str = "image") -> tuple[float, float]:
    """(credits used this UTC hour on the key, hourly credit limit) for image or video; (0, 0) unknown."""
    hourly = (usage(spec).get(kind) or {}).get("hourly") or {}
    return float(hourly.get("credits_used") or 0), float(hourly.get("credits_limit") or 0)


def reset() -> None:
    with _LOCK:
        _PENDING.clear()
