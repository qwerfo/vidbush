"""Small shared helpers: JSON IO, logging, hashing, subprocess, media probing."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9_\-\.]{8,}|key=[A-Za-z0-9_\-]{8,})")


def scrub(text: str) -> str:
    """Remove anything that looks like a secret before it reaches a log."""
    return _SECRET_RE.sub("[redacted]", text)


# ---------------------------------------------------------------- json io

def write_json(path: Path | str, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, BaseModel):
        text = data.model_dump_json(indent=2)
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False, default=_json_default)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_model(path: Path | str, model: type[T]) -> T:
    return model.model_validate_json(Path(path).read_text(encoding="utf-8"))


def stable_hash(*parts: Any) -> str:
    """Deterministic hash of JSON-serialisable parts and files (by path + size + mtime)."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, BaseModel):
            p = p.model_dump(mode="json")
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:20]


def file_fingerprint(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    st = p.stat()
    return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"


# ---------------------------------------------------------------- logging

class JsonLogger:
    """Structured JSON-lines logger writing to a file and stdout."""

    def __init__(self, project_id: str, log_dir: Path, echo: bool = True) -> None:
        self.project_id = project_id
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{project_id}.jsonl"
        self.echo = echo
        self._lock = threading.Lock()

    def log(self, level: str, msg: str, **fields: Any) -> None:
        rec = {"ts": round(time.time(), 3), "level": level, "project": self.project_id, "msg": scrub(msg)}
        for k, v in fields.items():
            rec[k] = scrub(v) if isinstance(v, str) else v
        line = json.dumps(rec, default=str, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            if self.echo:
                print(line, file=sys.stdout, flush=True)

    def info(self, msg: str, **f: Any) -> None:
        self.log("info", msg, **f)

    def warn(self, msg: str, **f: Any) -> None:
        self.log("warn", msg, **f)

    def error(self, msg: str, **f: Any) -> None:
        self.log("error", msg, **f)

    def tail(self, n: int = 100) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return out


class NullLogger(JsonLogger):
    def __init__(self) -> None:  # noqa: D401
        self.project_id = "null"
        self.echo = False

    def log(self, level: str, msg: str, **fields: Any) -> None:
        logging.getLogger("vidbush").debug("%s %s %s", level, msg, fields)


# ---------------------------------------------------------------- subprocess and media

class ToolError(RuntimeError):
    pass


def run(cmd: list[str], timeout: Optional[float] = None, input_bytes: Optional[bytes] = None) -> subprocess.CompletedProcess:
    """Run a command, raising ToolError with stderr tail on failure."""
    try:
        cp = subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise ToolError(f"timeout after {timeout}s: {' '.join(cmd[:4])}") from e
    if cp.returncode != 0:
        tail = cp.stderr.decode(errors="replace")[-1500:]
        raise ToolError(f"{cmd[0]} failed ({cp.returncode}): {tail}")
    return cp


def have(binary: str) -> bool:
    return shutil.which(binary) is not None


def ffprobe(path: str | Path) -> dict[str, Any]:
    cp = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)], timeout=60)
    return json.loads(cp.stdout)


def media_duration(path: str | Path) -> float:
    info = ffprobe(path)
    try:
        return float(info["format"]["duration"])
    except (KeyError, ValueError):
        for s in info.get("streams", []):
            if "duration" in s:
                return float(s["duration"])
    return 0.0


def video_size(path: str | Path) -> tuple[int, int]:
    for s in ffprobe(path).get("streams", []):
        if s.get("codec_type") == "video":
            return int(s.get("width", 0)), int(s.get("height", 0))
    return 0, 0


def chunks(seq: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def slug(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return s[:maxlen] or "item"


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


EM_DASH = "\u2014"


def no_em_dash(text: str) -> str:
    """Replace em dashes (and the double hyphen used as one) with a comma pause."""
    text = text.replace(" " + EM_DASH + " ", ", ").replace(EM_DASH, ", ")
    return re.sub(r"\s--\s", ", ", text)
