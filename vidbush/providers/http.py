"""Thin HTTP helpers that map failures to provider exception classes."""
from __future__ import annotations

from typing import Any, Optional

import requests

from .base import TransportError, http_error_to_exc

DEFAULT_TIMEOUT = 120.0


def post_json(url: str, payload: dict[str, Any], headers: Optional[dict[str, str]] = None,
              timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    try:
        r = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
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
        r = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
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
        with requests.get(url, headers=headers or {}, timeout=timeout, stream=True) as r:
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
        r = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
    except requests.RequestException as e:
        raise TransportError(str(e)[:300]) from e
    if r.status_code >= 400:
        raise http_error_to_exc(r.status_code, r.text, r.headers.get("Retry-After"))
    return r.content
