"""OpenAI-compatible chat adapter for LLM and vision roles.

Per-model request quirks are learned at runtime: when a model rejects a
parameter we drop it and retry before treating the model as failing. No
max_tokens cap is ever sent; prompts ask for concise output instead.
"""
from __future__ import annotations

import base64
import json
import re
import threading
from pathlib import Path
from typing import Any, Optional

from .base import CallResult, ProviderSpec, RefusalError
from .http import post_json

_QUIRKS: dict[str, set[str]] = {}
_QLOCK = threading.Lock()

_ADJUSTABLE = {
    "response_format": ("response_format", "json_object", "json mode", "json_schema"),
    "temperature": ("temperature",),
    "system_role": ("system", "role"),
}

REFUSAL_RE = re.compile(r"^\s*(i can('|no)t|i'm sorry|i am unable|i won't|sorry, i)", re.I)


def _quirks(model: str) -> set[str]:
    with _QLOCK:
        return set(_QUIRKS.get(model, set()))


def _add_quirk(model: str, q: str) -> None:
    with _QLOCK:
        _QUIRKS.setdefault(model, set()).add(q)


def _messages(system: str, user: Any, quirks: set[str]) -> list[dict[str, Any]]:
    if "system_role" in quirks:
        if isinstance(user, str):
            return [{"role": "user", "content": f"{system}\n\n{user}"}]
        return [{"role": "user", "content": [{"type": "text", "text": system}] + list(user)}]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat(spec: ProviderSpec, system: str, user: Any, want_json: bool, temperature: float = 0.6,
         timeout: float = 180.0) -> CallResult:
    """Send one chat completion, adjusting the request per model on 400s."""
    url = spec.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {spec.api_key}", "Content-Type": "application/json"}
    headers.update(spec.options.get("headers", {}))
    last: Optional[Exception] = None
    for _ in range(4):
        q = _quirks(spec.model)
        body: dict[str, Any] = {"model": spec.model, "messages": _messages(system, user, q)}
        if "temperature" not in q:
            body["temperature"] = temperature
        if want_json and "response_format" not in q:
            body["response_format"] = {"type": "json_object"}
        body.update(spec.options.get("extra_body", {}))
        try:
            data = post_json(url, body, headers, timeout=timeout)
        except RefusalError as e:
            last = e
            if not _learn(spec.model, str(e), q):
                raise
            continue
        return _result(spec, data, want_json)
    raise RefusalError(f"request could not be adjusted for {spec.model}: {last}")


def _learn(model: str, err: str, have: set[str]) -> bool:
    low = err.lower()
    for quirk, words in _ADJUSTABLE.items():
        if quirk not in have and any(w in low for w in words):
            _add_quirk(model, quirk)
            return True
    return False


def _result(spec: ProviderSpec, data: dict[str, Any], want_json: bool) -> CallResult:
    try:
        text = data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError) as e:
        raise RefusalError(f"no content in response: {str(data)[:200]}") from e
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    usage = data.get("usage") or {}
    tokens = float(usage.get("total_tokens") or 0)
    cost = float(usage.get("cost") or data.get("cost") or 0)
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
    """Build an OpenAI-style image content part from a local file."""
    p = Path(path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": detail}}
