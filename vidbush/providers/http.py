"""Thin HTTP helpers that map failures to provider exception classes."""
from __future__ import annotations

import json
import time
from typing import Any, Optional

import requests

from .base import TransportError, http_error_to_exc

DEFAULT_TIMEOUT = 120.0
# Some relays sit behind Cloudflare, which rejects the python-requests default agent (error 1010).
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/130.0 Safari/537.36 vidbush/0.1")


def _h(headers: Optional[dict[str, str]]) -> dict[str, str]:
    return {"User-Agent": USER_AGENT, **(headers or {})}


def post_json(url: str, payload: dict[str, Any], headers: Optional[dict[str, str]] = None,
              timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    try:
        r = requests.post(url, json=payload, headers=_h(headers), timeout=timeout)
    except requests.RequestException as e:
        raise TransportError(str(e)[:300]) from e
    if r.status_code >= 400:
        raise http_error_to_exc(r.status_code, r.text, r.headers.get("Retry-After"))
    try:
        return r.json()
    except ValueError as e:
        raise TransportError(f"bad json: {r.text[:200]}") from e


def get_json(url: str, params: Optional[dict[str, Any]] = None, headers: Optional[dict[str, str]] = None,
             timeout: float = 30.0) -> Any:
    try:
        r = requests.get(url, params=params, headers=_h(headers), timeout=timeout)
    except requests.RequestException as e:
        raise TransportError(str(e)[:300]) from e
    if r.status_code >= 400:
        raise http_error_to_exc(r.status_code, r.text, r.headers.get("Retry-After"))
    try:
        return r.json()
    except ValueError as e:
        raise TransportError(f"bad json: {r.text[:200]}") from e


def get_bytes(url: str, headers: Optional[dict[str, str]] = None, timeout: float = 60.0,
              max_bytes: int = 400_000_000) -> bytes:
    try:
        with requests.get(url, headers=_h(headers), timeout=timeout, stream=True) as r:
            if r.status_code >= 400:
                raise http_error_to_exc(r.status_code, r.text[:300], r.headers.get("Retry-After"))
            buf = bytearray()
            for chunk in r.iter_content(1 << 16):
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise TransportError("download too large")
            return bytes(buf)
    except requests.RequestException as e:
        raise TransportError(str(e)[:300]) from e


def post_bytes(url: str, payload: dict[str, Any], headers: Optional[dict[str, str]] = None,
               timeout: float = DEFAULT_TIMEOUT) -> bytes:
    try:
        r = requests.post(url, json=payload, headers=_h(headers), timeout=timeout)
    except requests.RequestException as e:
        raise TransportError(str(e)[:300]) from e
    if r.status_code >= 400:
        raise http_error_to_exc(r.status_code, r.text, r.headers.get("Retry-After"))
    return r.content


def post_stream(url: str, payload: dict[str, Any], headers: Optional[dict[str, str]] = None,
                timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """POST a chat request with stream on and fold the SSE chunks into one non-streamed reply."""
    body = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    text: list[str] = []
    usage: dict[str, Any] = {}
    finish = None
    done = False
    deadline = time.monotonic() + timeout  # keepalive pings must not extend the call forever
    try:
        with requests.post(url, json=body, headers=_h(headers), timeout=timeout, stream=True) as r:
            if r.status_code >= 400:
                raise http_error_to_exc(r.status_code, r.text, r.headers.get("Retry-After"))
            for raw in r.iter_lines(decode_unicode=True):
                if time.monotonic() > deadline:
                    raise TransportError(f"stream not finished within {timeout:.0f} s")
                if not raw or not raw.startswith("data:"):
                    continue  # blank lines and ": ping" keepalives
                chunk = raw[5:].strip()
                if chunk == "[DONE]":
                    done = True
                    break
                try:
                    obj = json.loads(chunk)
                except ValueError:
                    continue
                if obj.get("error"):
                    raise TransportError(f"stream error event: {str(obj['error'])[:300]}")
                usage = obj.get("usage") or usage
                for ch in obj.get("choices") or []:
                    text.append((ch.get("delta") or {}).get("content") or "")
                    finish = ch.get("finish_reason") or finish
    except requests.RequestException as e:
        raise TransportError(str(e)[:300]) from e
    if not done and finish is None:
        raise TransportError("stream ended before [DONE] or a finish reason (partial reply dropped)")
    return {"choices": [{"message": {"content": "".join(text)}, "finish_reason": finish}], "usage": usage}
