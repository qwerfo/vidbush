"""OpenAI-compatible chat adapter for LLM and vision roles.

Per-model request quirks are learned at runtime: when a model rejects a
parameter we drop it and retry before treating the model as failing. No
max_tokens cap is sent by default; prompts ask for concise output instead. A
cap is only sent when the operator sets options.max_tokens for the provider
(or role entry) in config/seats.yaml. A reply cut off by the length limit
(finish_reason "length" or "max_tokens") is continued once; a second cut
raises RefusalError so the request moves to the next provider without marking
this one down. A reply whose JSON does not parse is asked again once on the
same provider (BadJSONError). Replies are streamed by default (options.stream,
default true; false per provider): Adam measured that Wellflow and CloseRouter
answer faster and stop timing out when streamed, because the relays send
keepalive pings while generating and drop silent connections. When a reply carries no usage.cost the
call is priced from the provider's own GET /models price list (or
options.price_per_mtok), cached per base URL.
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
from .http import get_json, post_json, post_stream

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
    # a relay that rejects stream_options: stream without asking for the usage chunk
    "no_stream_options": ("stream_options", "include_usage"),
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


def stream_enabled(spec: ProviderSpec) -> bool:
    """options.stream, default true. Accepts booleans and the usual strings (false, 0, no, off)."""
    raw = spec.options.get("stream", True)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() not in ("0", "false", "no", "off", "")


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


DEFAULT_TIMEOUT_S = 600.0
JSON_RETRIES = 1


class BadJSONError(RefusalError):
    """The reply was not a refusal but its JSON did not parse. cost_usd is what the reply cost."""

    def __init__(self, msg: str, cost_usd: float = 0.0) -> None:
        super().__init__(msg)
        self.cost_usd = cost_usd


def chat(spec: ProviderSpec, system: str, user: Any, want_json: bool, temperature: float = 0.6,
         timeout: float = DEFAULT_TIMEOUT_S) -> CallResult:
    """One chat completion: quirk learning per 400, one continuation of a cut off reply, and one
    retry on the same provider when the JSON does not parse (its cost is added to the kept reply)."""
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


def _chat_once(spec: ProviderSpec, system: str, user: Any, want_json: bool, temperature: float,
               timeout: float) -> CallResult:
    data = _post(spec, system, user, want_json, temperature, timeout, [])
    if not _truncated(data):
        return _result(spec, data, want_json)
    partial = _content(data)
    more = [{"role": "assistant", "content": partial}, {"role": "user", "content": CONTINUE_PROMPT}]
    data2 = _post(spec, system, user, False, temperature, timeout, more)
    if _truncated(data2):
        raise RefusalError(f"reply truncated twice by the length limit on {spec.model}")
    t1, c1 = _usage(spec, data)
    t2, c2 = _usage(spec, data2)
    try:
        return _build(spec, partial + _content(data2), want_json, t1 + t2, c1 + c2)
    except BadJSONError as e:
        # a continued reply that still does not parse goes to the next provider, not another retry here
        raise RefusalError(f"continued reply still not JSON on {spec.model}: {e}") from e


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
            if stream_enabled(spec):
                return post_stream(url, body, headers, timeout=timeout,
                                   include_usage="no_stream_options" not in q)
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


def _usage(spec: ProviderSpec, data: dict[str, Any]) -> tuple[float, float]:
    """(total tokens, cost) of one response: the reported cost, else priced from the provider's list."""
    usage = data.get("usage") or {}
    reported = float(usage.get("cost") or data.get("cost") or 0)
    return float(usage.get("total_tokens") or 0), reported or price_usage(spec, usage)


def _result(spec: ProviderSpec, data: dict[str, Any], want_json: bool) -> CallResult:
    tokens, cost = _usage(spec, data)
    try:
        finish = str((data.get("choices") or [{}])[0].get("finish_reason") or "")
    except (AttributeError, IndexError, TypeError):
        finish = ""
    return _build(spec, _content(data), want_json, tokens, cost, finish)


def _build(spec: ProviderSpec, text: str, want_json: bool, tokens: float, cost: float,
           finish: str = "") -> CallResult:
    """CallResult from reply text. A refusal raises RefusalError; unparseable JSON raises BadJSONError."""
    value: Any = text
    if want_json:
        value = parse_json_loose(text)
        if value is None:
            if REFUSAL_RE.search(text):
                raise RefusalError(f"model refused: {text[:160]}")
            raise BadJSONError(f"model did not return JSON (finish {finish or 'unknown'}, {len(text)} chars): "
                               f"{text[:80]} ... {text[-160:]}", cost)
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
    except Exception:  # noqa: BLE001  a missing price list leaves the caller's estimate in charge; retried next call
        return table
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
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
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
