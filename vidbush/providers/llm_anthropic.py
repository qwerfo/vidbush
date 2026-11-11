"""Native Anthropic Messages API adapter (provider type anthropic) for LLM and vision roles.

POST {base}/v1/messages with x-api-key and anthropic-version headers. The system prompt is the
top-level system field (never a message). JSON is asked for in the prompt, because the Messages API
has no json_object switch: want_json appends JSON_INSTRUCTION to the system prompt and the reply is
read with the same loose parser as openai_compat. Vision parts in the OpenAI shape (image_url data
URIs, built by llm.image_part) are converted to base64 image blocks, so callers stay unchanged.

Behaviour matches openai_compat (vidbush/providers/llm.py), each for the same reason:
- Errors map through http_error_to_exc, so the ProviderClient retries transport errors, backs off on
  429 and falls back on refusals exactly as for every other provider.
- A 400 that names an adjustable request feature teaches a per (base URL, model) quirk and the request
  is sent again without it (effort, thinking, temperature, the API version header, a too large
  max_tokens). Content policy errors never teach a quirk.
- A reply cut off by max_tokens is continued once; a second cut raises RefusalError so the request
  moves to the next provider without marking this one down. A reply whose JSON does not parse is asked
  again once (BadJSONError). stop_reason refusal raises RefusalError.
- options.stream (default true) streams the reply over SSE: ping events are skipped, usage is taken
  from message_start and message_delta, and a stream that ends without message_stop or a stop reason
  is a TransportError (the partial reply is dropped, never used).
- Cost comes from a built-in price table per model (USD per million tokens, cache reads and writes at
  their own rates), overridable with options.price_per_mtok {prompt, completion, cache_read,
  cache_write_5m}.

The Messages API requires max_tokens, so options.max_tokens is sent when set, else 64000 streamed or
16000 not streamed (large enough that a normal reply never hits it). temperature is not sent: current
Claude models reject sampling parameters; options.send_temperature: true sends it for older models.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Optional

import requests

from .base import (POLICY_WORDS, CallResult, ProviderSpec, RateLimitError, RefusalError, TransportError,
                   http_error_to_exc)
from .http import USER_AGENT, get_json
from .llm import CONTINUE_PROMPT, JSON_RETRIES, REFUSAL_RE, BadJSONError, max_tokens_option, parse_json_loose

DEFAULT_BASE = "https://api.anthropic.com"
API_VERSION = "2026-10-21"
LEGACY_API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS_STREAM = 64000
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TIMEOUT_S = 600.0
JSON_INSTRUCTION = ("Reply with one JSON object only: no prose before or after it, no code fence, "
                    "and nothing but valid JSON.")

# USD per million tokens (Anthropic first party rates, cached 2026-10-06). cache_write_5m is 1.25x input.
# Prices change: override per provider or role entry with options.price_per_mtok.
PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-fable-5-1": {"prompt": 10.0, "completion": 50.0, "cache_read": 0.25, "cache_write_5m": 12.5},
    "claude-fable-5": {"prompt": 10.0, "completion": 50.0, "cache_read": 1.0, "cache_write_5m": 12.5},
    "claude-opus-5-5": {"prompt": 4.0, "completion": 20.0, "cache_read": 0.20, "cache_write_5m": 5.0},
    "claude-opus-5": {"prompt": 5.0, "completion": 25.0, "cache_read": 0.50, "cache_write_5m": 6.25},
    "claude-opus-4-8": {"prompt": 5.0, "completion": 25.0, "cache_read": 0.50, "cache_write_5m": 6.25},
    "claude-opus-4-7": {"prompt": 5.0, "completion": 25.0, "cache_read": 0.50, "cache_write_5m": 6.25},
    "claude-opus-4-6": {"prompt": 5.0, "completion": 25.0, "cache_read": 0.50, "cache_write_5m": 6.25},
    "claude-sonnet-5-5": {"prompt": 2.0, "completion": 10.0, "cache_read": 0.20, "cache_write_5m": 2.5},
    "claude-sonnet-5": {"prompt": 2.0, "completion": 10.0, "cache_read": 0.20, "cache_write_5m": 2.5},
    "claude-sonnet-4-6": {"prompt": 3.0, "completion": 15.0, "cache_read": 0.30, "cache_write_5m": 3.75},
    # Haiku 5.5 doubles past 100K prompt tokens; Vidbush prompts stay far below that
    "claude-haiku-5-5": {"prompt": 0.10, "completion": 0.50, "cache_read": 0.01, "cache_write_5m": 0.125},
    "claude-haiku-4-5": {"prompt": 1.0, "completion": 5.0, "cache_read": 0.10, "cache_write_5m": 1.25},
}

# adjustable request features: the 400 phrases that name each one
_ADJUSTABLE = {
    "effort": ("effort", "output_config"),
    "thinking": ("thinking",),
    "temperature": ("temperature", "top_p", "top_k", "sampling"),
    "legacy_version": ("anthropic-version",),
}
_NUM_RE = re.compile(r"\b(\d{3,7})\b")
_QUIRKS: dict[tuple[str, str], set[str]] = {}
_CAPS: dict[tuple[str, str], int] = {}
_QLOCK = threading.Lock()


def reset_quirks() -> None:
    """Forget learned quirks (between runs and in tests)."""
    with _QLOCK:
        _QUIRKS.clear()
        _CAPS.clear()


def _base(spec: ProviderSpec) -> str:
    base = (spec.base_url or DEFAULT_BASE).rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _qkey(spec: ProviderSpec) -> tuple[str, str]:
    return (_base(spec), spec.model)


def _quirks(spec: ProviderSpec) -> set[str]:
    with _QLOCK:
        return set(_QUIRKS.get(_qkey(spec), set()))


def headers(spec: ProviderSpec, quirks: Optional[set[str]] = None) -> dict[str, str]:
    version = LEGACY_API_VERSION if "legacy_version" in (quirks or set()) else str(
        spec.options.get("anthropic_version") or API_VERSION)
    h = {"x-api-key": spec.api_key, "anthropic-version": version, "content-type": "application/json",
         "User-Agent": USER_AGENT}
    betas = spec.options.get("betas") or []
    if betas:
        h["anthropic-beta"] = ",".join(str(b) for b in betas)
    h.update(spec.options.get("headers", {}))
    return h


# ---------------------------------------------------------------- request shape

def _image_block(part: dict[str, Any]) -> dict[str, Any]:
    url = str((part.get("image_url") or {}).get("url", ""))
    m = re.match(r"data:([^;]+);base64,(.*)$", url, re.S)
    if m:
        return {"type": "image", "source": {"type": "base64", "media_type": m.group(1), "data": m.group(2)}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def to_content(user: Any) -> Any:
    """A user message in the OpenAI shape (text or parts) as Anthropic content."""
    if isinstance(user, str):
        return user
    out: list[dict[str, Any]] = []
    for part in user or []:
        if not isinstance(part, dict):
            out.append({"type": "text", "text": str(part)})
        elif part.get("type") == "image_url":
            out.append(_image_block(part))
        elif part.get("type") == "text":
            out.append({"type": "text", "text": str(part.get("text", ""))})
        else:
            out.append(part)  # already an Anthropic block
    return out


def _cap(spec: ProviderSpec, stream: bool) -> int:
    with _QLOCK:
        learned = _CAPS.get(_qkey(spec))
    cap = max_tokens_option(spec) or (DEFAULT_MAX_TOKENS_STREAM if stream else DEFAULT_MAX_TOKENS)
    return min(cap, learned) if learned else cap


def build_body(spec: ProviderSpec, system: str, user: Any, want_json: bool, temperature: float,
               extra: list[dict[str, Any]], quirks: set[str], stream: bool) -> dict[str, Any]:
    sys_text = f"{system}\n\n{JSON_INSTRUCTION}" if want_json else system
    body: dict[str, Any] = {"model": spec.model, "max_tokens": _cap(spec, stream),
                            "messages": [{"role": "user", "content": to_content(user)}] + extra}
    if sys_text:
        body["system"] = sys_text
    effort = spec.options.get("effort")
    if effort and "effort" not in quirks:
        body["output_config"] = {"effort": str(effort)}
    thinking = spec.options.get("thinking")
    if isinstance(thinking, dict) and "thinking" not in quirks:
        body["thinking"] = thinking
    if spec.options.get("send_temperature") and "temperature" not in quirks:
        body["temperature"] = temperature
    body.update(spec.options.get("extra_body", {}))
    return body


# ---------------------------------------------------------------- transport

def _stream_on(spec: ProviderSpec) -> bool:
    raw = spec.options.get("stream", True)
    return raw if isinstance(raw, bool) else str(raw).strip().lower() not in ("0", "false", "no", "off")


def _error_exc(err: Any) -> Exception:
    """An error event or body ({type, message}) as the matching provider exception."""
    err = err if isinstance(err, dict) else {"message": str(err)}
    kind = str(err.get("type", ""))
    msg = f"anthropic {kind}: {str(err.get('message', err))[:300]}"
    if kind == "rate_limit_error":
        return RateLimitError(msg)
    if kind in ("overloaded_error", "api_error", "timeout_error"):
        return TransportError(msg)
    return RefusalError(msg)


def post_message(url: str, body: dict[str, Any], hdrs: dict[str, str], timeout: float) -> dict[str, Any]:
    try:
        r = requests.post(url, json=body, headers=hdrs, timeout=timeout)
    except requests.RequestException as e:
        raise TransportError(str(e)[:300]) from e
    if r.status_code >= 400:
        raise http_error_to_exc(r.status_code, r.text, r.headers.get("Retry-After"))
    try:
        data = r.json()
    except ValueError as e:
        raise TransportError(f"bad json: {r.text[:200]}") from e
    if isinstance(data, dict) and data.get("type") == "error":
        raise _error_exc(data.get("error"))
    return data


def post_message_stream(url: str, body: dict[str, Any], hdrs: dict[str, str], timeout: float) -> dict[str, Any]:
    """POST with stream on and fold the SSE events into one non-streamed message dict."""
    payload = {**body, "stream": True}
    blocks: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    stop_reason = None
    done = False
    deadline = time.monotonic() + timeout  # ping events must not extend the call forever
    try:
        with requests.post(url, json=payload, headers=hdrs, timeout=timeout, stream=True) as r:
            if r.status_code >= 400:
                raise http_error_to_exc(r.status_code, r.text, r.headers.get("Retry-After"))
            if "application/json" in (r.headers.get("Content-Type") or "").lower():
                data = r.json()  # a relay that ignored stream: the whole message in one body
                if isinstance(data, dict) and data.get("type") == "error":
                    raise _error_exc(data.get("error"))
                return data
            for raw in r.iter_lines(decode_unicode=True):
                if time.monotonic() > deadline:
                    raise TransportError(f"stream not finished within {timeout:.0f} s")
                if not raw or not raw.startswith("data:"):
                    continue  # event: lines, blank lines and : comments
                try:
                    ev = json.loads(raw[5:].strip())
                except ValueError:
                    continue
                kind = ev.get("type")
                if kind == "ping":
                    continue
                if kind == "error":
                    raise _error_exc(ev.get("error"))
                if kind == "message_start":
                    usage.update((ev.get("message") or {}).get("usage") or {})
                elif kind == "content_block_start":
                    blocks[int(ev.get("index", len(blocks)))] = dict(ev.get("content_block") or {})
                elif kind == "content_block_delta":
                    delta = ev.get("delta") or {}
                    blk = blocks.setdefault(int(ev.get("index", 0)), {"type": "text", "text": ""})
                    if delta.get("type") == "text_delta":
                        blk["text"] = str(blk.get("text") or "") + str(delta.get("text") or "")
                elif kind == "message_delta":
                    stop_reason = (ev.get("delta") or {}).get("stop_reason") or stop_reason
                    usage.update({k: v for k, v in (ev.get("usage") or {}).items() if v is not None})
                elif kind == "message_stop":
                    done = True
                    break
    except requests.RequestException as e:
        raise TransportError(str(e)[:300]) from e
    if not done and stop_reason is None:
        raise TransportError("stream ended before message_stop or a stop reason (partial reply dropped)")
    return {"content": [blocks[i] for i in sorted(blocks)], "stop_reason": stop_reason, "usage": usage}


def _learn(spec: ProviderSpec, err: str, have: set[str]) -> bool:
    """Learn one request adjustment from a 400. Content policy errors never teach a quirk."""
    low = err.lower()
    if any(w in low for w in POLICY_WORDS):
        return False
    if "max_tokens" in low and any(w in low for w in ("<=", "maximum", "exceed", "less than", "at most", ">")):
        # the model's limit is the smallest number the error names below what was sent
        sent = _cap(spec, _stream_on(spec))
        nums = [int(n) for n in _NUM_RE.findall(err) if 1000 <= int(n) < sent]  # status codes are not limits
        if nums:
            with _QLOCK:
                _CAPS[_qkey(spec)] = min(nums)
            return True
    for quirk, phrases in _ADJUSTABLE.items():
        if quirk not in have and any(p in low for p in phrases):
            with _QLOCK:
                _QUIRKS.setdefault(_qkey(spec), set()).add(quirk)
            return True
    return False


def _post(spec: ProviderSpec, system: str, user: Any, want_json: bool, temperature: float, timeout: float,
          extra: list[dict[str, Any]]) -> dict[str, Any]:
    url = _base(spec) + "/v1/messages"
    stream = _stream_on(spec)
    last: Optional[Exception] = None
    for _ in range(len(_ADJUSTABLE) + 2):
        q = _quirks(spec)
        body = build_body(spec, system, user, want_json, temperature, extra, q, stream)
        try:
            if stream:
                return post_message_stream(url, body, headers(spec, q), timeout)
            return post_message(url, body, headers(spec, q), timeout)
        except RefusalError as e:
            last = e
            if not _learn(spec, str(e), q):
                raise
    raise RefusalError(f"request could not be adjusted for {spec.model}: {last}")


# ---------------------------------------------------------------- reply

def _text(data: dict[str, Any]) -> str:
    content = data.get("content")
    if not isinstance(content, list):
        raise RefusalError(f"no content in response: {str(data)[:200]}")
    return "".join(str(b.get("text") or "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def _stop(data: dict[str, Any]) -> str:
    return str(data.get("stop_reason") or "")


def prices(spec: ProviderSpec) -> dict[str, float]:
    """USD per million tokens for the model: options.price_per_mtok, else the table (longest id prefix)."""
    if spec.options.get("price_per_mtok"):
        return {k: float(v) for k, v in spec.options["price_per_mtok"].items() if isinstance(v, (int, float))}
    model = spec.model.lower().split("/")[-1]
    best = max((k for k in PRICES_PER_MTOK if model == k or model.startswith(k + "-")), key=len, default="")
    return PRICES_PER_MTOK.get(best, {})


def usage_cost(spec: ProviderSpec, usage: dict[str, Any]) -> tuple[float, float]:
    """(total tokens, USD) of one reply. input_tokens excludes cache reads and writes in this API."""
    def n(k: str) -> float:
        try:
            return float(usage.get(k) or 0)
        except (TypeError, ValueError):
            return 0.0
    fresh, out = n("input_tokens"), n("output_tokens")
    read, write = n("cache_read_input_tokens"), n("cache_creation_input_tokens")
    p = prices(spec)
    pin = p.get("prompt", 0.0)
    cost = (fresh * pin + read * p.get("cache_read", pin * 0.1) + write * p.get("cache_write_5m", pin * 1.25)
            + out * p.get("completion", 0.0)) / 1e6
    return fresh + out + read + write, cost


def _build(spec: ProviderSpec, text: str, want_json: bool, tokens: float, cost: float, stop: str) -> CallResult:
    value: Any = text
    if want_json:
        value = parse_json_loose(text)
        if value is None:
            if REFUSAL_RE.search(text):
                raise RefusalError(f"model refused: {text[:160]}")
            raise BadJSONError(f"model did not return JSON (stop {stop or 'unknown'}, {len(text)} chars): "
                               f"{text[:80]} ... {text[-160:]}", cost)
    return CallResult(value=value, provider=spec, mock=False, cost_usd=cost, units=tokens, unit_kind="tokens")


def _chat_once(spec: ProviderSpec, system: str, user: Any, want_json: bool, temperature: float,
               timeout: float) -> CallResult:
    data = _post(spec, system, user, want_json, temperature, timeout, [])
    if _stop(data) == "refusal":
        raise RefusalError(f"model declined the request (stop_reason refusal) on {spec.model}")
    t1, c1 = usage_cost(spec, data.get("usage") or {})
    if _stop(data) != "max_tokens":
        return _build(spec, _text(data), want_json, t1, c1, _stop(data))
    partial = _text(data)
    more = [{"role": "assistant", "content": partial.rstrip() or "..."},
            {"role": "user", "content": CONTINUE_PROMPT}]
    data2 = _post(spec, system, user, False, temperature, timeout, more)
    if _stop(data2) in ("max_tokens", "refusal"):
        raise RefusalError(f"reply cut off twice ({_stop(data2)}) on {spec.model}")
    t2, c2 = usage_cost(spec, data2.get("usage") or {})
    try:
        return _build(spec, partial + _text(data2), want_json, t1 + t2, c1 + c2, _stop(data2))
    except BadJSONError as e:
        raise RefusalError(f"continued reply still not JSON on {spec.model}: {e}") from e


def chat(spec: ProviderSpec, system: str, user: Any, want_json: bool, temperature: float = 0.6,
         timeout: float = DEFAULT_TIMEOUT_S) -> CallResult:
    """One Messages API call with the openai_compat contract (see the module docstring)."""
    spent = 0.0
    for attempt in range(JSON_RETRIES + 1):
        try:
            res = _chat_once(spec, system, user, want_json, temperature, timeout)
        except BadJSONError as e:
            spent += e.cost_usd
            if attempt >= JSON_RETRIES:
                raise BadJSONError(str(e), spent) from e
            continue
        res.cost_usd += spent
        return res
    raise AssertionError("unreachable")


def list_models(spec: ProviderSpec, timeout: float = 30.0) -> list[str]:
    """Model ids from GET {base}/v1/models."""
    data = get_json(_base(spec) + "/v1/models", headers=headers(spec), timeout=timeout)
    rows = data.get("data") if isinstance(data, dict) else data
    return [str(m.get("id")) for m in rows or [] if isinstance(m, dict) and m.get("id")]
