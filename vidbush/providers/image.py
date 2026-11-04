"""Image generation adapters: OpenAI-compatible images API, OpenRouter style
chat image output (options.mode: chat), fal.ai, Runware, and async job APIs
(Elevate media tasks, Algrow jobs: submit, poll, download the hosted file).

All adapters return a PNG on disk at least 1920x1080 (16:9), upscaled and
center cropped with Lanczos when the provider returns a smaller frame.
"""
from __future__ import annotations

import base64
import io
import re
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from .base import CallResult, ProviderSpec, RateLimitError, RefusalError, TransportError
from .http import get_bytes, get_json, post_json

SIZES_16x9 = ["1792x1024", "1536x1024", "1344x768", "1024x576", "1024x1024"]
# per image USD at landscape size by quality; used for gpt-image models instead of the static estimate
GPT_IMAGE_COST = {
    "gpt-image-1-mini": {"low": 0.006, "medium": 0.015, "high": 0.052},
    "gpt-image-1": {"low": 0.016, "medium": 0.063, "high": 0.25},
}
# USD per 1M tokens (input, output) for usage based gpt-image cost
GPT_IMAGE_TOKENS = {"gpt-image-1-mini": (2.0, 8.0), "gpt-image-1": (5.0, 40.0)}
DEFAULT_GPT_IMAGE_QUALITY = "low"


def fit_16x9(raw: bytes, out_path: Path, width: int = 1920, height: int = 1080, trim: float = 0.0) -> Path:
    """Scale and center crop any image to width x height. trim crops that fraction from every edge
    first (providers that stamp a corner watermark set options.edge_trim)."""
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    if trim > 0:
        dx, dy = round(im.width * trim), round(im.height * trim)
        im = im.crop((dx, dy, im.width - dx, im.height - dy))
    scale = max(width / im.width, height / im.height)
    im = im.resize((max(width, round(im.width * scale)), max(height, round(im.height * scale))), Image.LANCZOS)
    left = (im.width - width) // 2
    top = (im.height - height) // 2
    im = im.crop((left, top, left + width, top + height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, "PNG")
    return out_path


def _trim(spec: ProviderSpec) -> float:
    return min(0.2, max(0.0, float(spec.options.get("edge_trim", 0.0))))


def _cost(spec: ProviderSpec, default: float) -> float:
    return float(spec.options.get("cost_per_image", default))


def _gpt_family(model: str) -> str:
    """The GPT_IMAGE_COST row for this model, or "" for models the table does not price (they use cost_per_image)."""
    low = model.lower()
    if not re.search(r"gpt-image-1(?![.\d])", low):
        return ""
    return "gpt-image-1-mini" if "mini" in low else "gpt-image-1"


def _quality(spec: ProviderSpec) -> str:
    q = str(spec.options.get("quality") or "")
    if not q and _gpt_family(spec.model):
        q = DEFAULT_GPT_IMAGE_QUALITY
    return q


def estimate_cost(spec: ProviderSpec, default: float) -> float:
    """Pre-call per image estimate. gpt-image models use the quality table, never the static estimate."""
    fam = _gpt_family(spec.model)
    if fam:
        table = GPT_IMAGE_COST[fam]
        return table.get(_quality(spec), table["high"])
    return _cost(spec, default)


def _actual_cost(spec: ProviderSpec, data: dict[str, Any], default: float) -> float:
    """Cost from the response: usage.cost, gpt-image token usage, else the estimate."""
    usage = data.get("usage") or {}
    if usage.get("cost"):
        return float(usage["cost"])
    fam = _gpt_family(spec.model)
    if fam and usage.get("output_tokens"):
        pin, pout = GPT_IMAGE_TOKENS[fam]
        return (float(usage.get("input_tokens") or 0) * pin + float(usage["output_tokens"]) * pout) / 1e6
    return estimate_cost(spec, default)


def _is_size_error(e: Exception) -> bool:
    low = str(e).lower()
    return "size" in low or "dimension" in low


def generate_openai(spec: ProviderSpec, prompt: str, out_path: Path, default_cost: float) -> CallResult:
    """POST /images/generations, trying 16:9 sizes, then one last request with no size key."""
    if spec.options.get("mode") == "chat":
        return generate_chat(spec, prompt, out_path, default_cost)
    url = spec.base_url.rstrip("/") + "/images/generations"
    headers = {"Authorization": f"Bearer {spec.api_key}"}
    sizes: list[str | None] = [spec.options["size"]] if spec.options.get("size") else list(SIZES_16x9)
    sizes.append(None)
    quality = _quality(spec)
    last: Exception | None = None
    for size in sizes:
        body: dict[str, Any] = {"model": spec.model, "prompt": prompt, "n": 1}
        if size:
            body["size"] = size
        if quality:
            body["quality"] = quality
        body.update(spec.options.get("extra_body", {}))
        try:
            data = post_json(url, body, headers, timeout=240)
        except RefusalError as e:
            last = e
            if _is_size_error(e):
                continue
            raise
        return CallResult(value=_save(data, out_path, _trim(spec)), provider=spec, mock=False,
                          cost_usd=_actual_cost(spec, data, default_cost), units=1, unit_kind="images")
    raise RefusalError(f"no accepted size for {spec.model}: {last}")


def generate_chat(spec: ProviderSpec, prompt: str, out_path: Path, default_cost: float) -> CallResult:
    """OpenRouter style: /chat/completions with modalities image+text, image returned as a data URL."""
    url = spec.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {spec.api_key}", "Content-Type": "application/json"}
    body: dict[str, Any] = {"model": spec.model, "messages": [{"role": "user", "content": prompt}],
                            "modalities": ["image", "text"],
                            "image_config": {"aspect_ratio": spec.options.get("aspect_ratio", "16:9")}}
    body.update(spec.options.get("extra_body", {}))
    data = post_json(url, body, headers, timeout=240)
    try:
        img_url = data["choices"][0]["message"]["images"][0]["image_url"]["url"]
    except (KeyError, IndexError, TypeError) as e:
        raise RefusalError(f"no image in chat response: {str(data)[:200]}") from e
    raw = _decode_image_url(img_url)
    return CallResult(value=fit_16x9(raw, out_path, trim=_trim(spec)), provider=spec, mock=False,
                      cost_usd=_actual_cost(spec, data, default_cost), units=1, unit_kind="images")


def _decode_image_url(url: str) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    return get_bytes(url)


def _save(data: dict[str, Any], out_path: Path, trim: float = 0.0) -> Path:
    item = (data.get("data") or [{}])[0]
    if item.get("b64_json"):
        raw = base64.b64decode(item["b64_json"])
    elif item.get("url"):
        raw = get_bytes(item["url"])
    else:
        raise RefusalError(f"no image in response: {str(data)[:200]}")
    return fit_16x9(raw, out_path, trim=trim)


def generate_fal(spec: ProviderSpec, prompt: str, out_path: Path, default_cost: float) -> CallResult:
    base = spec.base_url or "https://fal.run"
    url = f"{base.rstrip('/')}/{spec.model}"
    body = {"prompt": prompt, "image_size": spec.options.get("image_size", "landscape_16_9"), "num_images": 1,
            **spec.options.get("extra_body", {})}
    data = post_json(url, body, {"Authorization": f"Key {spec.api_key}"}, timeout=240)
    images = data.get("images") or []
    if not images:
        raise RefusalError(f"fal returned no image: {str(data)[:200]}")
    raw = get_bytes(images[0]["url"])
    return CallResult(value=fit_16x9(raw, out_path, trim=_trim(spec)), provider=spec, mock=False,
                      cost_usd=_cost(spec, default_cost), units=1, unit_kind="images")


def generate_runware(spec: ProviderSpec, prompt: str, out_path: Path, default_cost: float) -> CallResult:
    base = spec.base_url or "https://api.runware.ai/v1"
    task = [{"taskType": "imageInference", "taskUUID": str(uuid.uuid4()), "positivePrompt": prompt,
             "model": spec.model, "width": 1920, "height": 1088, "numberResults": 1, "outputType": "URL",
             "includeCost": True, **spec.options.get("extra_body", {})}]
    data = post_json(base, task, {"Authorization": f"Bearer {spec.api_key}"}, timeout=240)  # type: ignore[arg-type]
    items = data.get("data") or []
    if not items or not items[0].get("imageURL"):
        raise RefusalError(f"runware returned no image: {str(data)[:200]}")
    raw = get_bytes(items[0]["imageURL"])
    cost = float(items[0].get("cost") or _cost(spec, default_cost))
    return CallResult(value=fit_16x9(raw, out_path, trim=_trim(spec)), provider=spec, mock=False,
                      cost_usd=cost, units=1, unit_kind="images")


# ------------------------------------------------------------ async job APIs

_JOB_DONE = {"completed", "succeeded", "success", "done"}
_JOB_FAILED = {"failed", "error", "cancelled", "canceled"}
_RETRY_WORDS = ("retry", "timed out", "timeout", "temporarily", "unavailable", "overloaded", "busy")


def _job_failure(where: str, message: str) -> Exception:
    """A failed job is transient when the provider says to retry, else a refusal (moderation, bad prompt)."""
    low = message.lower()
    if any(w in low for w in _RETRY_WORDS):
        return TransportError(f"{where}: {message[:300]}")
    return RefusalError(f"{where}: {message[:300]}")


def _poll(spec: ProviderSpec, url: str, headers: dict[str, str], read: Any, where: str,
          default_timeout: float, default_poll: float) -> dict[str, Any]:
    """GET url until read(reply) gives a terminal status; returns the finished job object."""
    limit = float(spec.options.get("timeout_s", default_timeout))
    wait = float(spec.options.get("poll_s", default_poll))
    deadline = time.time() + limit
    empty = 0
    while True:
        # errors while polling never escape as transport errors: the client would resubmit a new paid job
        try:
            job = read(get_json(url, headers=headers, timeout=60))
        except RateLimitError as e:
            time.sleep(max(wait, e.retry_after or wait))
            continue
        except TransportError:
            job = {"status": "polling_error"}
        status = str(job.get("status", "")).lower()
        if status in _JOB_DONE:
            return job
        if status in _JOB_FAILED:
            raise _job_failure(where, str(job.get("error") or job.get("message") or status))
        empty = empty + 1 if not status else 0
        if empty >= 5:
            raise RefusalError(f"{where}: job replies carry no status")
        if time.time() >= deadline:
            # a stalled job: hand the request to the fallback provider rather than resubmitting here
            raise RefusalError(f"{where} not finished within {limit:.0f} s (status {status})")
        time.sleep(wait)


def _elevate_data(reply: Any, where: str) -> dict[str, Any]:
    """Elevate wraps every reply in {success, data, error}; errors can arrive with HTTP 200."""
    if not isinstance(reply, dict):
        raise TransportError(f"{where}: non-object reply")
    if reply.get("success") is False:
        err = reply.get("error") or {}
        code, msg = str(err.get("code", "")).upper(), str(err.get("message", "unknown error"))
        if "RATE_LIMIT" in code:
            raise RateLimitError(f"{where}: {code} {msg}"[:300], None)
        if code in ("INVALID_REQUEST", "VALIDATION_ERROR", "UNAUTHORIZED", "FORBIDDEN", "CONTENT_POLICY"):
            raise RefusalError(f"{where}: {code} {msg}"[:300])
        raise _job_failure(where, f"{code} {msg}")
    return reply.get("data") or {}


def generate_elevate(spec: ProviderSpec, prompt: str, out_path: Path, default_cost: float) -> CallResult:
    """Elevate media API: POST /v2/media, poll GET /v2/media/{id}?type=image, download result_url."""
    base = (spec.base_url or "https://public-api.elevate.uno").rstrip("/")
    headers = {"Authorization": f"Bearer {spec.api_key}"}
    body: dict[str, Any] = {"type": "image", "prompt": prompt, "model": spec.model,
                            "aspect_ratio": spec.options.get("aspect_ratio", "landscape"), "params": {}}
    body.update(spec.options.get("extra_body", {}))
    task = _elevate_data(post_json(base + "/v2/media", body, headers, timeout=60), "Elevate submit")
    task_id = task.get("id")
    if not task_id:
        raise TransportError(f"Elevate submit returned no task id: {str(task)[:200]}")
    if str(task.get("status", "")).lower() not in _JOB_DONE:
        task = _poll(spec, f"{base}/v2/media/{task_id}?type=image", headers,
                     lambda r: _elevate_data(r, f"Elevate task {task_id}"), f"Elevate task {task_id}", 300, 5)
    if not task.get("result_url"):
        raise TransportError(f"Elevate task {task_id} completed without result_url")
    raw = get_bytes(task["result_url"])
    return CallResult(value=fit_16x9(raw, out_path, trim=_trim(spec)), provider=spec, mock=False,
                      cost_usd=_cost(spec, default_cost), units=1, unit_kind="images")


def _algrow_url(job: dict[str, Any]) -> str:
    for key in ("image_url", "url", "result_url", "output_url"):
        v = job.get(key)
        if isinstance(v, str) and v.startswith("http"):
            return v
    for key in ("image_urls", "images", "outputs", "results"):
        v = job.get(key)
        if isinstance(v, list) and v:
            first = v[0]
            if isinstance(first, str) and first.startswith("http"):
                return first
            if isinstance(first, dict):
                return _algrow_url(first)
    return ""


def generate_algrow(spec: ProviderSpec, prompt: str, out_path: Path, default_cost: float) -> CallResult:
    """Algrow: POST /api/generate-image, poll GET /api/job-status/{job_id}, download the hosted image."""
    base = (spec.base_url or "https://api.algrow.online").rstrip("/")
    headers = {"Authorization": f"Bearer {spec.api_key}"}
    body: dict[str, Any] = {"prompt": prompt, "model": spec.model,
                            "aspect_ratio": spec.options.get("aspect_ratio", "16:9"),
                            "fast": bool(spec.options.get("fast", False))}
    body.update(spec.options.get("extra_body", {}))
    sub = post_json(base + "/api/generate-image", body, headers, timeout=60)
    if sub.get("success") is False or not sub.get("job_id"):
        msg = str(sub.get("error") or sub)[:300]
        if "rate" in msg.lower() or "concurrent" in msg.lower():
            raise RateLimitError(f"Algrow submit: {msg}", None)
        raise RefusalError(f"Algrow submit: {msg}")
    job_id = sub["job_id"]
    job = _poll(spec, f"{base}/api/job-status/{job_id}", headers, lambda r: r if isinstance(r, dict) else {},
                f"Algrow job {job_id}", 600, 3)
    url = _algrow_url(job)
    if not url:
        raise TransportError(f"Algrow job {job_id} completed without an image url")
    raw = get_bytes(url)
    return CallResult(value=fit_16x9(raw, out_path, trim=_trim(spec)), provider=spec, mock=False,
                      cost_usd=_cost(spec, default_cost), units=1, unit_kind="images")


ADAPTERS = {"openai_compat": generate_openai, "fal": generate_fal, "runware": generate_runware,
            "elevate": generate_elevate, "algrow": generate_algrow}



NEGATIVE_FIELD = {"fal": "negative_prompt", "runware": "negativePrompt"}


def with_negative(spec: ProviderSpec, prompt: str, negative: str) -> str:
    """Prompt text for providers without a negative field: the negatives become one prose sentence
    (GPT Image and similar models follow it; dropping it silently let text and logos into stills)."""
    if not negative or spec.type in NEGATIVE_FIELD:
        return prompt
    return f"{prompt} Do not include: {negative}."


def generate(spec: ProviderSpec, prompt: str, out_path: Path, default_cost: float, negative: str = "") -> CallResult:
    fn = ADAPTERS.get(spec.type)
    if fn is None:
        raise TransportError(f"no image adapter for type {spec.type}")
    if negative and spec.type in NEGATIVE_FIELD:
        spec = ProviderSpec(spec.name, spec.type, spec.model, spec.base_url, spec.api_key,
                            {**spec.options, "extra_body": {**spec.options.get("extra_body", {}),
                                                            NEGATIVE_FIELD[spec.type]: negative}})
    return fn(spec, with_negative(spec, prompt, negative), out_path, default_cost)
