"""Per-project lock file so the CLI and the API worker never run one project at once.

output/<id>/.lock holds {"pid", "host", "thread", "owner", "ts"}. It is taken
with an exclusive create, so two processes cannot both get it. The lock is
re-entrant inside the thread that holds it (apply_and_render takes it in
apply_ops and again in run_project). A lock whose pid no longer runs on this
host is stale and is taken over. A pid cannot be checked across hosts (a
container sharing the output volume), so a heartbeat thread touches the file
every HEARTBEAT_S seconds while it is held, and a lock from another host is
stale once it has not been touched for FOREIGN_STALE_S seconds.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

LOCK_NAME = ".lock"
HEARTBEAT_S = 30.0
FOREIGN_STALE_S = 300.0

_HELD: dict[str, list[Any]] = {}  # lock path -> [thread ident, depth]
_GUARD = threading.Lock()


class ProjectBusy(RuntimeError):
    """Another process (or thread) is working on this project."""


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _pid_alive_windows(pid: int) -> bool:
    """os.kill(pid, 0) sends CTRL_C_EVENT on Windows, so query the process handle instead."""
    import ctypes
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel32.CloseHandle(handle)
    return bool(ok) and code.value == 259  # STILL_ACTIVE


def read_lock(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _is_stale(path: Path, info: Optional[dict[str, Any]]) -> bool:
    if info is None:  # unreadable or half written: stale only once it is old
        try:
            return time.time() - path.stat().st_mtime > 30
        except OSError:
            return True
    if info.get("host") != socket.gethostname():
        try:
            return time.time() - path.stat().st_mtime > FOREIGN_STALE_S
        except OSError:
            return True
    if int(info.get("pid", 0)) == os.getpid():
        return str(path) not in _HELD  # held by a thread of ours, else a leak from an earlier run
    return not _pid_alive(int(info.get("pid", 0)))


def _busy_message(project_id: str, path: Path, info: Optional[dict[str, Any]]) -> str:
    info = info or {}
    since = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(info.get("ts", 0) or 0)))
    return (f"project {project_id} is busy: {info.get('owner', 'another run')} (pid {info.get('pid', '?')} on "
            f"{info.get('host', '?')}) has held it since {since}. Wait for it to finish, or delete {path} if "
            "no Vidbush process is running.")


def _try_create(path: Path, owner: str) -> bool:
    data = {"pid": os.getpid(), "host": socket.gethostname(), "thread": threading.get_ident(),
            "owner": owner, "ts": time.time()}
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return True


def acquire(root: Path, project_id: str, owner: str = "") -> bool:
    """Take the lock. True when newly taken, False when this thread already holds it."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCK_NAME
    key = str(path)
    me = threading.get_ident()
    with _GUARD:
        held = _HELD.get(key)
        if held and held[0] == me:
            held[1] += 1
            return False
        if held:
            raise ProjectBusy(_busy_message(project_id, path, read_lock(path)))
        for _ in range(3):
            if _try_create(path, owner or f"vidbush pid {os.getpid()}"):
                _HELD[key] = [me, 1]
                _start_heartbeat(path)
                return True
            info = read_lock(path)
            if not _is_stale(path, info):
                raise ProjectBusy(_busy_message(project_id, path, info))
            path.unlink(missing_ok=True)  # stale: the process that held it is gone
        raise ProjectBusy(_busy_message(project_id, path, read_lock(path)))


def _start_heartbeat(path: Path) -> None:
    """Touch the lock file every HEARTBEAT_S seconds until this process releases it."""
    key = str(path)
    entry = _HELD[key]

    def beat() -> None:
        while True:
            time.sleep(HEARTBEAT_S)
            if _HELD.get(key) is not entry:
                return
            touch(path.parent)

    threading.Thread(target=beat, name="vidbush-lock-heartbeat", daemon=True).start()


def release(root: Path) -> None:
    key = str(root / LOCK_NAME)
    with _GUARD:
        held = _HELD.get(key)
        if not held:
            return
        held[1] -= 1
        if held[1] > 0:
            return
        _HELD.pop(key, None)
        info = read_lock(Path(key))
        if info is None or int(info.get("pid", 0)) == os.getpid():
            Path(key).unlink(missing_ok=True)


def touch(root: Path) -> None:
    """Refresh the lock file time (a heartbeat for hosts that cannot check our pid)."""
    try:
        os.utime(root / LOCK_NAME)
    except OSError:
        pass


def check_free(root: Path, project_id: str) -> None:
    """Raise ProjectBusy when someone else holds a live lock on this project (no lock is taken)."""
    path = root / LOCK_NAME
    if not path.exists():
        return
    with _GUARD:
        held = _HELD.get(str(path))
        if held and held[0] == threading.get_ident():
            return
    info = read_lock(path)
    if not _is_stale(path, info):
        raise ProjectBusy(_busy_message(project_id, path, info))


@contextmanager
def project_lock(root: Path, project_id: str, owner: str = "") -> Iterator[None]:
    """Hold the project lock for the duration of the block (re-entrant in one thread)."""
    acquire(root, project_id, owner)
    try:
        yield
    finally:
        release(root)
