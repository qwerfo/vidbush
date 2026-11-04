"""OpenAI-compatible chat adapter for LLM and vision roles.

Per-model request quirks are learned at runtime: when a model rejects a
parameter we drop it and retry before treating the model as failing. No
max_tokens cap is sent by default; prompts ask for concise output instead.
A provider whose relay truncates without one sets options.max_tokens.
options.stream asks for a streamed reply (relays that cut idle connections).
When a reply carries no usage.cost the call is priced from the provider's own
GET /models price list (or options.price_per_mtok), cached per base URL.
"""
from __future__ import annotations

import base64
import json
import re
import threading
from pathlib import Path
from typing import Any, Optional

from .base import CallResult, ProviderSpec, RefusalError
from .http import get_json, post_json, post_stream

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
}
_PARAM_QUIRK = {"response_format": "response_format", "temperature": "temperature"}
_POLICY_WORDS = ("safety", "content policy", "content_policy", "flagged", "moderation", "content_filter",
                 "violat", "prohibited")
_PARAM_RE = re.compile(r'"param"\s*:\s*"([^"]+)"')

_CONTENT_RE = re.compile(r"<content>\s*(.*?)\s*(?:</content>|$)", re.S)
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


DEFAULT_TIMEOUT_S = 600.0


def chat(spec: ProviderSpec, system: str, user: Any, want_json: bool, temperature: float = 0.6,
         timeout: float = DEFAULT_TIMEOUT_S) -> CallResult:
    """Send one chat completion, adjusting the request per model on 400s."""
    url = spec.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {spec.api_key}", "Content-Type": "application/json"}
    headers.update(spec.options.get("headers", {}))
    last: Optional[Exception] = None
    for _ in range(4):
        q = _quirks(_qkey(spec))
        body: dict[str, Any] = {"model": spec.model, "messages": _messages(system, user, q)}
        if "temperature" not in q:
            body["temperature"] = temperature
        if want_json and "response_format" not in q:
            body["response_format"] = {"type": "json_object"}
        if spec.options.get("max_tokens"):
            body["max_tokens"] = int(spec.options["max_tokens"])
        body.update(spec.options.get("extra_body", {}))
        try:
            if spec.options.get("stream"):
                data = post_stream(url, body, headers, timeout=timeout)
            else:
                data = post_json(url, body, headers, timeout=timeout)
        except RefusalError as e:
            last = e
            if not _learn(_qkey(spec), str(e), q):
                raise
            continue
        return _result(spec, data, want_json)
    raise RefusalError(f"request could not be adjusted for {spec.model}: {last}")


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


def _result(spec: ProviderSpec, data: dict[str, Any], want_json: bool) -> CallResult:
    try:
        text = data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError) as e:
        raise RefusalError(f"no content in response: {str(data)[:200]}") from e
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    usage = data.get("usage") or {}
    tokens = float(usage.get("total_tokens") or 0)
    cost = float(usage.get("cost") or data.get("cost") or 0) or price_usage(spec, usage)
    value: Any = text
    if want_json:
        value = parse_json_loose(text)
        if value is None:
            if REFUSAL_RE.search(text):
                raise RefusalError(f"model refused: {text[:160]}")
            raise RefusalError(f"model did not return JSON: {text[:160]}")
    return CallResult(value=value, provider=spec, mock=False, cost_usd=cost, units=tokens, unit_kind="tokens")


_PRICES: dict[str, dict[str, dict[str, float]]] = {}
_PLOCK = threading.Lock()


def _per_token(entry: dict[str, Any]) -> dict[str, float]:
    """Catalogue pricing as USD per token. CloseRouter states per million, OpenRouter per token."""
    scale = 1e-6 if "million" in str(entry.get("unit", "")) else 1.0
    out = {}
    for k, v in entry.items():
        try:
            out[k] = float(v) * scale
        except (TypeError, ValueError):
            continue
    return out


def catalogue_prices(spec: ProviderSpec) -> dict[str, dict[str, float]]:
    """Model id to per token prices from GET {base}/models, fetched once per base URL. Empty on failure."""
    base = spec.base_url.rstrip("/")
    with _PLOCK:
        if base in _PRICES:
            return _PRICES[base]
    table: dict[str, dict[str, float]] = {}
    try:
        data = get_json(base + "/models", headers={"Authorization": f"Bearer {spec.api_key}"}, timeout=30)
        for m in (data.get("data") if isinstance(data, dict) else data) or []:
            if isinstance(m, dict) and isinstance(m.get("pricing"), dict):
                table[str(m.get("id"))] = _per_token(m["pricing"])
    except Exception:  # noqa: BLE001  a missing price list leaves the caller's estimate in charge
        pass
    with _PLOCK:
        _PRICES[base] = table
    return table


def price_usage(spec: ProviderSpec, usage: dict[str, Any]) -> float:
    """USD for one reply's usage, cache reads and writes priced at their own rates. 0 when unknown."""
    if not usage:
        return 0.0
    if spec.options.get("price_per_mtok"):
        p = _per_token({**spec.options["price_per_mtok"], "unit": "usd_per_million_tokens"})
    else:
        p = catalogue_prices(spec).get(spec.model, {})
    if not p.get("prompt") and not p.get("completion"):
        return 0.0
    details = usage.get("prompt_tokens_details") or {}
    prompt = float(usage.get("prompt_tokens") or 0)
    read = float(details.get("cached_tokens") or usage.get("cached_tokens") or 0)
    write = float(details.get("cached_creation_tokens") or details.get("cache_creation_tokens")
                  or usage.get("cache_write_tokens") or 0)
    fresh = max(0.0, prompt - read - write)
    out = float(usage.get("completion_tokens") or 0)
    pin = p.get("prompt", 0.0)
    return (fresh * pin + read * p.get("cache_read", pin) + write * p.get("cache_write_5m", pin)
            + out * p.get("completion", 0.0))


def parse_json_loose(text: str) -> Optional[Any]:
    """Parse JSON from a model reply, tolerating code fences and leading prose."""
    t = text.strip()
    # some relays inject an agent scaffold: "I'll write...<write_file><path>..</path><content>REPLY</content>"
    m = _CONTENT_RE.search(t)
    if m:
        t = m.group(1).strip()
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
