"""Resolve a seat's model id from the provider's own model list, by family and version.

A role entry may name `model_match: {family: sonnet, version: "5.5"}` instead of a model id. The id is
then read from GET {base}/models (openai_compat) or GET {base}/v1/models (anthropic) the first time the
seat is resolved, matched by family and version, and logged once. Reason: relays spell the same model
differently (Wellflow `claude-sonnet-5-5`, CloseRouter `anthropic/claude-sonnet-5.5`, Anthropic
`claude-sonnet-5-5`), and a guessed id fails only at the first real call.

Matching: an id matches when, after dropping a `vendor/` prefix and turning dots into dashes, it reads
`<prefix>-<family>-<version with dashes>` with nothing after it but an optional date suffix
(`-20261101`). The prefix defaults to `claude`; `prefix: ""` matches other families, e.g. `{family: grok,
version: "4.7", prefix: ""}` matches `x-ai/grok-4.7` and `grok-4.7`, and `{family: gemini, version:
"3.8-flash", prefix: ""}` matches `google/gemini-3.8-flash` but not `-lite` or `-preview`. Version "5"
therefore never matches `claude-sonnet-5-5`. An exact id beats a dated one.
A model_env pin that does not name the entry's own family and version is stale (pin_matches).
No match leaves the model empty, so the seat counts as not configured and the next provider serves.
A failed listing is retried after RETRY_S, because a relay that is down at startup may come back.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any, Optional

from .base import ProviderSpec
from .http import get_json

RETRY_S = 600.0
_CACHE: dict[tuple[str, str, str, str], tuple[float, str]] = {}  # key -> (time, id or "")
_LISTS: dict[tuple[str, str], tuple[float, list[str]]] = {}  # (provider, base) -> (time, ids), one GET each
_LOCK = threading.Lock()


def _norm(model_id: str) -> str:
    return model_id.strip().lower().split("/")[-1].replace(".", "-")


def _head(family: str, version: str, prefix: str = "claude") -> str:
    """Normalised `<prefix>-<family>-<version>` (no prefix when prefix is empty), regex escaped."""
    p = str(prefix).strip().lower()
    name = f"{p}-" if p else ""
    return re.escape(name + f"{str(family).strip().lower()}-{str(version).strip().lower().replace('.', '-')}")


def match_model(ids: list[str], family: str, version: str, prefix: str = "claude") -> str:
    """The best id for <prefix>-<family>-<version>, or "" (see the module docstring)."""
    pat = re.compile(rf"^{_head(family, version, prefix)}(-\d{{8}})?$")
    exact = [i for i in ids if pat.match(_norm(i)) and not pat.match(_norm(i)).group(1)]
    dated = sorted((i for i in ids if pat.match(_norm(i)) and pat.match(_norm(i)).group(1)), reverse=True)
    return (exact or dated or [""])[0]


def pin_matches(model: str, match: dict[str, Any]) -> bool:
    """True when a pinned id names the match's family and version. A date or a word may follow
    (`-20261101`, `-exact`), another version number may not (`claude-opus-5-5` is no Opus 5 pin)."""
    head = _head(match.get("family", ""), match.get("version", ""), match.get("prefix", "claude"))
    return bool(re.match(rf"^{head}(-\d{{8}}|-[a-z]|$)", _norm(model)))


def list_ids(spec: ProviderSpec, timeout: float = 30.0) -> list[str]:
    """Model ids the provider lists."""
    if spec.type == "anthropic":
        from .llm_anthropic import list_models
        return list_models(spec, timeout=timeout)
    data = get_json(spec.base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {spec.api_key}"},
                    timeout=timeout)
    rows = data.get("data") if isinstance(data, dict) else data
    return [str(m.get("id")) for m in rows or [] if isinstance(m, dict) and m.get("id")]


def _cached_ids(spec: ProviderSpec) -> list[str]:
    key = (spec.name, spec.base_url.rstrip("/"))
    with _LOCK:
        hit = _LISTS.get(key)
        if hit and time.time() - hit[0] < RETRY_S:
            return hit[1]
    ids = list_ids(spec)  # a failure raises and is not cached, so the next resolve asks again
    with _LOCK:
        _LISTS[key] = (time.time(), ids)
    return ids


def resolve(spec: ProviderSpec, match: dict[str, Any], log: Any) -> str:
    """Model id for spec (base URL and key set) matching {family, version}; "" when none. Cached."""
    family, version = str(match.get("family", "")), str(match.get("version", ""))
    prefix = str(match.get("prefix", "claude"))
    key = (spec.name, spec.base_url.rstrip("/"), f"{prefix.lower()}:{family.lower()}", version)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and (hit[1] or time.time() - hit[0] < RETRY_S):
            return hit[1]
    try:
        ids = _cached_ids(spec)
        found = match_model(ids, family, version, prefix)
        name = "-".join(x for x in (prefix, family, version) if x)
        err: Optional[str] = None if found else f"no listed id matches {name} ({len(ids)} ids)"
    except Exception as e:  # noqa: BLE001 - a failed listing leaves the seat unconfigured until retried
        found, err = "", f"model list failed: {str(e)[:200]}"
    with _LOCK:
        _CACHE[key] = (time.time(), found)
    if found:
        log.info("model resolved from the provider model list", provider=spec.name, family=family,
                 version=version, model=found)
    else:
        log.warn("model not resolved, seat skipped", provider=spec.name, family=family, version=version,
                 error=err)
    return found


def reset() -> None:
    with _LOCK:
        _CACHE.clear()
        _LISTS.clear()
