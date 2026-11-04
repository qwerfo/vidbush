"""OpenAI-compatible chat adapter for LLM and vision roles.

Per-model request quirks are learned at runtime: when a model rejects a
parameter we drop it and retry before treating the model as failing. No
max_tokens cap is sent by default; prompts ask for concise output instead. A
cap is only sent when the operator sets options.max_tokens for the provider
(or role entry) in config/seats.yaml. A reply cut off by the length limit
(finish_reason "length" or "max_tokens") is continued once; a second cut, or
a joined reply that is still not valid JSON, raises RefusalError so the
request moves to the next provider without marking this one down.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from .base import CallResult, ProviderSpec, RefusalError
from .http import post_json

_LOG = logging.getLogger(__name__)
_QUIRKS: dict[tuple[str, str], set[str]] = {}
_QLOCK = threading.Lock()

# precise phrases per adjustable request feature; matched only on non policy 400s
_ADJUSTABLE = {
    "json_word": ("must contain the word 'json'", 'must contain the word "json"'),
    "response_format": ("response_format", "json_object", "json mode", "json_schema"),
    "temperature": ("'temperature'", '"temperature"', "temperature is not supported",
                    "unsupported parameter: temperature", "only the default (1) value"),
    "system_role": ("system role", "system message", "system_message", "role 'system'", 'role "system"', "role': 'system'",
                    "system prompt is not supported", "developer instruction"),
    # a model that wants the newer token cap name: rename max_tokens to it
    "max_completion_tokens": ("max_completion_tokens",),
}
_PARAM_QUIRK = {"response_format": "response_format", "temperature": "temperature",
                "max_tokens": "max_completion_tokens"}
# one try per adjustable quirk plus a final try with all of them applied
_MAX_TRIES = len(_ADJUSTABLE) + 1
# vision images are fit into this box before encoding: cost.vision_per_image_*_usd price a 640 px keyframe
VISION_MAX_SIDE = 640
# finish reasons that mean the reply was cut off by the output length limit
_TRUNCATED = {"length", "max_tokens"}
CONTINUE_PROMPT = ("Your previous reply was cut off. Continue exactly where it stopped, without repeating "
                   "anything and without any preamble, so that appending your text completes the reply.")
_POLICY_WORDS = ("safety", "content policy", "content_policy", "flagged", "moderation", "content_filter",
                 "violat", "prohibited")
_PARAM_RE = re.compile(r'"param"\s*:\s*"([^"]+)"')

REFUSAL_RE = re.compile(r"^\s*(i can('|no)t|i'm sorry|i am unable|i won't|sorry, i)", re.I)


def _qkey(spec: ProviderSpec) -> tuple[str, str]:
    return (spec.base_url.rstrip("/"), spec.model)


def _quirks(key: tuple[str, str]) -> set[str]:
    with _QLOCK:
        return set(_QUIRKS.get(key, set()))


def _add_quirk(key: tuple[str, str], q: str) -> None:
    with _QLOCK:
        _QUIRKS.setdefault(key, set()).add(q)


def reset_quirks() -> None:
    """Forget learned quirks (call between runs or in tests)."""
    with _QLOCK:
        _QUIRKS.clear()


def _messages(system: str, user: Any, quirks: set[str]) -> list[dict[str, Any]]:
    if "json_word" in quirks:
        system = f"{system}\n\nRespond in JSON."
    if "system_role" in quirks:
        if isinstance(user, str):
            return [{"role": "user", "content": f"{system}\n\n{user}"}]
        return [{"role": "user", "content": [{"type": "text", "text": system}] + list(user)}]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def max_tokens_option(spec: ProviderSpec) -> Optional[int]:
    """options.max_tokens as a positive int, else None (invalid values are logged and ignored)."""
    raw = spec.options.get("max_tokens")
    if raw is None or raw == "":
        return None
    try:
        val = int(raw) if not isinstance(raw, bool) and float(raw) == int(float(raw)) else 0
    except (TypeError, ValueError, OverflowError):
        val = 0
    if val <= 0:
        _LOG.warning("ignoring invalid max_tokens %r for provider %s", raw, spec.name)
        return None
    return val


def chat(spec: ProviderSpec, system: str, user: Any, want_json: bool, temperature: float = 0.6,
         timeout: float = 180.0) -> CallResult:
    """Send one chat completion, adjusting the request per model on 400s and continuing one cut off reply."""
    data = _post(spec, system, user, want_json, temperature, timeout, [])
    if not _truncated(data):
        return _result(spec, data, want_json)
    partial = _content(data)
    more = [{"role": "assistant", "content": partial}, {"role": "user", "content": CONTINUE_PROMPT}]
    data2 = _post(spec, system, user, False, temperature, timeout, more)
    if _truncated(data2):
        raise RefusalError(f"reply truncated twice by the length limit on {spec.model}")
    t1, c1 = _usage(data)
    t2, c2 = _usage(data2)
    return _build(spec, partial + _content(data2), want_json, t1 + t2, c1 + c2)


def _post(spec: ProviderSpec, system: str, user: Any, want_json: bool, temperature: float, timeout: float,
          extra: list[dict[str, Any]]) -> dict[str, Any]:
    """POST one request (extra messages appended), learning one quirk per 400 until every quirk could apply."""
    url = spec.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {spec.api_key}", "Content-Type": "application/json"}
    headers.update(spec.options.get("headers", {}))
    cap = max_tokens_option(spec)
    last: Optional[Exception] = None
    for _ in range(_MAX_TRIES):
        q = _quirks(_qkey(spec))
        body: dict[str, Any] = {"model": spec.model, "messages": _messages(system, user, q) + extra}
        if "temperature" not in q:
            body["temperature"] = temperature
        if want_json and "response_format" not in q:
            body["response_format"] = {"type": "json_object"}
        if cap is not None:
            body["max_completion_tokens" if "max_completion_tokens" in q else "max_tokens"] = cap
        body.update(spec.options.get("extra_body", {}))
        try:
            return post_json(url, body, headers, timeout=timeout)
        except RefusalError as e:
            last = e
            if not _learn(_qkey(spec), str(e), q):
                raise
    raise RefusalError(f"request could not be adjusted for {spec.model}: {last}")


def _truncated(data: dict[str, Any]) -> bool:
    """True when choices[0].finish_reason says the output length limit cut the reply."""
    try:
        reason = data["choices"][0].get("finish_reason")
    except (KeyError, IndexError, AttributeError, TypeError):
        return False
    return str(reason or "").lower() in _TRUNCATED


def _learn(key: tuple[str, str], err: str, have: set[str]) -> bool:
    """Learn one request adjustment from a 400. Content policy errors never teach a quirk."""
    low = err.lower()
    if any(w in low for w in _POLICY_WORDS):
        return False
    m = _PARAM_RE.search(err)
    if m and _PARAM_QUIRK.get(m.group(1)) and _PARAM_QUIRK[m.group(1)] not in have:
        _add_quirk(key, _PARAM_QUIRK[m.group(1)])
        return True
    for quirk, phrases in _ADJUSTABLE.items():
        if quirk not in have and any(p in low for p in phrases):
            _add_quirk(key, quirk)
            return True
    return False


def _content(data: dict[str, Any]) -> str:
    """Assistant text of choices[0], joining list content parts."""
    try:
        text = data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, AttributeError, TypeError) as e:
        raise RefusalError(f"no content in response: {str(data)[:200]}") from e
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    return str(text)


def _usage(data: dict[str, Any]) -> tuple[float, float]:
    """(total tokens, reported cost) of one response; zero when absent."""
    usage = data.get("usage") or {}
    return float(usage.get("total_tokens") or 0), float(usage.get("cost") or data.get("cost") or 0)


def _result(spec: ProviderSpec, data: dict[str, Any], want_json: bool) -> CallResult:
    text = _content(data)
    tokens, cost = _usage(data)
    return _build(spec, text, want_json, tokens, cost)


def _build(spec: ProviderSpec, text: str, want_json: bool, tokens: float, cost: float) -> CallResult:
    """CallResult from reply text; JSON replies must parse or RefusalError is raised."""
    value: Any = text
    if want_json:
        value = parse_json_loose(text)
        if value is None:
            if REFUSAL_RE.search(text):
                raise RefusalError(f"model refused: {text[:160]}")
            raise RefusalError(f"model did not return JSON: {text[:160]}")
    return CallResult(value=value, provider=spec, mock=False, cost_usd=cost, units=tokens, unit_kind="tokens")


def parse_json_loose(text: str) -> Optional[Any]:
    """Parse JSON from a model reply, tolerating code fences and leading prose."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = t.find(open_c), t.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(t[i : j + 1])
            except json.JSONDecodeError:
                continue
    return None


def image_part(path: str | Path, detail: str = "low") -> dict[str, Any]:
    """Build an OpenAI-style image content part from a local file, fit into VISION_MAX_SIDE."""
    p = Path(path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    data = _fit_image(p, mime)
    b64 = base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": detail}}


def _fit_image(path: Path, mime: str, max_side: int = VISION_MAX_SIDE) -> bytes:
    """File bytes, re-encoded smaller when either side exceeds max_side (unreadable images pass as is)."""
    raw = path.read_bytes()
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if max(im.size) <= max_side:
                return raw
            im.thumbnail((max_side, max_side))
            return _encode(im, mime)
    except (OSError, ValueError):
        return raw


def _encode(im: Image.Image, mime: str) -> bytes:
    buf = io.BytesIO()
    if mime == "image/png":
        im.save(buf, format="PNG")
    else:
        im.convert("RGB").save(buf, format="JPEG", quality=88)
    return buf.getvalue()
