"""Image generation adapters: OpenAI-compatible images API, fal.ai, Runware.

All adapters return a PNG on disk at least 1920x1080 (16:9), upscaled and
center cropped with Lanczos when the provider returns a smaller frame.
"""
from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from .base import CallResult, ProviderSpec, RefusalError, TransportError
from .http import get_bytes, post_json

SIZES_16x9 = ["1792x1024", "1536x1024", "1344x768", "1024x576", "1024x1024"]


def fit_16x9(raw: bytes, out_path: Path, width: int = 1920, height: int = 1080) -> Path:
    """Scale and center crop any image to width x height."""
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    scale = max(width / im.width, height / im.height)
    im = im.resize((max(width, round(im.width * scale)), max(height, round(im.height * scale))), Image.LANCZOS)
    left = (im.width - width) // 2
    top = (im.height - height) // 2
    im = im.crop((left, top, left + width, top + height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, "PNG")
    return out_path


def _cost(spec: ProviderSpec, default: float) -> float:
    return float(spec.options.get("cost_per_image", default))


def generate_openai(spec: ProviderSpec, prompt: str, out_path: Path, default_cost: float) -> CallResult:
    url = spec.base_url.rstrip("/") + "/images/generations"
    headers = {"Authorization": f"Bearer {spec.api_key}"}
    sizes = [spec.options["size"]] if spec.options.get("size") else SIZES_16x9
    last: Exception | None = None
    for size in sizes:
        body: dict[str, Any] = {"model": spec.model, "prompt": prompt, "n": 1, "size": size}
        body.update(spec.options.get("extra_body", {}))
        try:
            data = post_json(url, body, headers, timeout=240)
        except RefusalError as e:
            last = e
            if "size" in str(e).lower() or "dimension" in str(e).lower():
                continue
            raise
        return CallResult(value=_save(data, out_path), provider=spec, mock=False,
                          cost_usd=_cost(spec, default_cost), units=1, unit_kind="images")
    raise RefusalError(f"no accepted size for {spec.model}: {last}")


def _save(data: dict[str, Any], out_path: Path) -> Path:
    item = (data.get("data") or [{}])[0]
    if item.get("b64_json"):
        raw = base64.b64decode(item["b64_json"])
    elif item.get("url"):
        raw = get_bytes(item["url"])
    else:
        raise RefusalError(f"no image in response: {str(data)[:200]}")
    return fit_16x9(raw, out_path)


def generate_fal(spec: ProviderSpec, prompt: str, out_path: Path, default_cost: float) -> CallResult:
    base = spec.base_url or "https://fal.run"
    url = f"{base.rstrip('/')}/{spec.model}"
    body = {"prompt": prompt, "image_size": spec.options.get("image_size", "landscape_16_9"), "num_images": 1}
    data = post_json(url, body, {"Authorization": f"Key {spec.api_key}"}, timeout=240)
    images = data.get("images") or []
    if not images:
        raise RefusalError(f"fal returned no image: {str(data)[:200]}")
    raw = get_bytes(images[0]["url"])
    return CallResult(value=fit_16x9(raw, out_path), provider=spec, mock=False,
                      cost_usd=_cost(spec, default_cost), units=1, unit_kind="images")


def generate_runware(spec: ProviderSpec, prompt: str, out_path: Path, default_cost: float) -> CallResult:
    base = spec.base_url or "https://api.runware.ai/v1"
    task = [{"taskType": "imageInference", "taskUUID": str(uuid.uuid4()), "positivePrompt": prompt,
             "model": spec.model, "width": 1920, "height": 1088, "numberResults": 1, "outputType": "URL",
             "includeCost": True}]
    data = post_json(base, task, {"Authorization": f"Bearer {spec.api_key}"}, timeout=240)  # type: ignore[arg-type]
    items = data.get("data") or []
    if not items or not items[0].get("imageURL"):
        raise RefusalError(f"runware returned no image: {str(data)[:200]}")
    raw = get_bytes(items[0]["imageURL"])
    cost = float(items[0].get("cost") or _cost(spec, default_cost))
    return CallResult(value=fit_16x9(raw, out_path), provider=spec, mock=False,
                      cost_usd=cost, units=1, unit_kind="images")


ADAPTERS = {"openai_compat": generate_openai, "fal": generate_fal, "runware": generate_runware}


def generate(spec: ProviderSpec, prompt: str, out_path: Path, default_cost: float) -> CallResult:
    fn = ADAPTERS.get(spec.type)
    if fn is None:
        raise TransportError(f"no image adapter for type {spec.type}")
    return fn(spec, prompt, out_path, default_cost)
