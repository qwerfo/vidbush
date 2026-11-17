"""Reply cache: every LLM and vision reply stored under a hash of role, model, prompt and images.

Principle: a model call whose inputs have not changed is answered from disk, so a replay of a run with
unchanged prompts costs nothing and returns the same decisions.

- Key: sha256 of role, model, system prompt, user prompt (text or the image parts, whose base64 data is
  part of the prompt) and the call options that change the reply (json mode, temperature).
- The n-th identical request in one process is answered by the reply stored for occurrence n (a map
  keyed by n, never a list, so a failed or empty occurrence leaves a gap instead of shifting the rest). A
  caller that retries the same prompt gets the second stored reply on replay, and a live call when none
  was stored, so retries never loop on one cached answer.
- fresh=True asks live and overwrites the stored reply for that occurrence.
- One writer per key at a time inside a process; two processes writing the same key at the same moment
  can lose one reply (shortcut: a thread lock only; add a file lock if runs ever share prompts in parallel).
- Only real replies are stored (never mock output, never None).
- Files live beside the source cache (settings.reply_cache_folder): <folder>/<role>/<key[:2]>/<key>.json.
- Switch: settings.reply_cache (config reply_cache, env VIDBUSH_REPLY_CACHE=0, CLI --no-reply-cache).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

VERSION = 1  # part of every key: bump only when the key itself changes, not the file format


def request_key(role: str, model: str, system: str, user: Any, options: dict[str, Any]) -> str:
    """Stable hash of everything that decides the reply."""
    blob = json.dumps({"v": VERSION, "role": role, "model": model, "system": system, "user": user,
                       "options": options}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ReplyCache:
    """Disk store of replies with a per-process occurrence counter per key."""

    def __init__(self, folder: Path, enabled: bool = True) -> None:
        self.folder = Path(folder)
        self.enabled = enabled
        self._lock = threading.Lock()
        self._served: dict[str, int] = {}  # key -> replies already handed out in this process
        self.hits = 0
        self.stores = 0

    def _path(self, role: str, key: str) -> Path:
        return self.folder / (role or "any") / key[:2] / f"{key}.json"

    def _read(self, role: str, key: str) -> dict[str, Any]:
        p = self._path(role, key)
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def take(self, role: str, key: str, fresh: bool = False) -> tuple[int, Optional[Any]]:
        """(occurrence index, stored reply or None). The index is reserved for this request.

        fresh=True never serves a stored reply (the caller wants a new one) but still reserves the index,
        so the live reply is stored as the next occurrence."""
        with self._lock:
            n = self._served.get(key, 0)
            self._served[key] = n + 1
        if not self.enabled or fresh:
            return n, None
        replies = _as_map(self._read(role, key).get("replies"))
        if str(n) in replies:
            with self._lock:
                self.hits += 1
            return n, replies[str(n)]
        return n, None

    def put(self, role: str, model: str, key: str, n: int, value: Any, overwrite: bool = False) -> None:
        """Store a live reply as occurrence n (a filled occurrence is kept unless overwrite)."""
        if not self.enabled or value is None:
            return
        p = self._path(role, key)
        with self._lock:
            data = self._read(role, key) or {"role": role, "model": model, "replies": {}}
            replies = data["replies"] = _as_map(data.get("replies"))
            if str(n) in replies and not overwrite:
                return
            replies[str(n)] = value
            data["ts"] = time.time()
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                tmp.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
            except (OSError, TypeError, ValueError):
                tmp.unlink(missing_ok=True)
                return
            for attempt in range(5):  # Windows: a reader holding the file makes os.replace fail briefly
                try:
                    os.replace(tmp, p)
                    self.stores += 1
                    return
                except PermissionError:
                    time.sleep(0.05 * (attempt + 1))
                except OSError:
                    break
            tmp.unlink(missing_ok=True)


def _as_map(replies: Any) -> dict[str, Any]:
    """Replies by occurrence index; files written before the map format hold a plain list."""
    if isinstance(replies, list):
        return {str(i): v for i, v in enumerate(replies)}
    return dict(replies) if isinstance(replies, dict) else {}


def default_folder(source_cache_folder: str) -> str:
    """Beside the source cache: <parent of source cache>/reply_cache."""
    return str(Path(source_cache_folder).parent / "reply_cache")
