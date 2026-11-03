"""Per-project lock: exclusive between processes, re-entrant in a thread, stale pids taken over."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from vidbush import cli
from vidbush.jobs import lock
from vidbush.jobs.lock import ProjectBusy, project_lock
from vidbush.jobs.pipeline import create_project, run_project
from vidbush.models import STAGES, InputMode


def _foreign_lock(root: Path, pid: int, host: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / lock.LOCK_NAME
    path.write_text(json.dumps({"pid": pid, "host": host or socket.gethostname(), "owner": "vidbush serve",
                                "ts": time.time()}))
    return path


@pytest.fixture
def live_pid():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    yield proc.pid
    proc.kill()
    proc.wait()


def test_lock_is_reentrant_and_released(tmp_path):
    root = tmp_path / "p"
    with project_lock(root, "p"):
        with project_lock(root, "p"):
            assert (root / lock.LOCK_NAME).exists()
        assert (root / lock.LOCK_NAME).exists()
    assert not (root / lock.LOCK_NAME).exists()


def test_live_foreign_lock_is_busy_with_clear_message(tmp_path, live_pid):
    root = tmp_path / "p"
    _foreign_lock(root, live_pid)
    with pytest.raises(ProjectBusy) as e:
        with project_lock(root, "p"):
            pass
    assert "project p is busy" in str(e.value) and str(live_pid) in str(e.value) and ".lock" in str(e.value)
    assert (root / lock.LOCK_NAME).exists()  # someone else's lock is never removed


def test_stale_pid_lock_is_taken_over(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    root = tmp_path / "p"
    _foreign_lock(root, proc.pid)
    with project_lock(root, "p"):
        assert json.loads((root / lock.LOCK_NAME).read_text())["pid"] == os.getpid()


def test_foreign_host_lock_goes_stale_by_age(tmp_path):
    root = tmp_path / "p"
    path = _foreign_lock(root, 1, host="some-container")
    with pytest.raises(ProjectBusy):
        lock.acquire(root, "p")
    old = time.time() - lock.FOREIGN_STALE_S - 10
    os.utime(path, (old, old))
    with project_lock(root, "p"):
        pass


def test_other_thread_of_same_process_is_busy(tmp_path):
    root = tmp_path / "p"
    errors: list[Exception] = []
    with project_lock(root, "p"):
        def other():
            try:
                lock.acquire(root, "p")
            except ProjectBusy as e:
                errors.append(e)
        t = threading.Thread(target=other)
        t.start()
        t.join()
    assert errors


def _stubs():
    def make(name):
        def fn(ctx):
            p = ctx.work / f"{name}.done"
            p.write_text("ok")
            return [str(p)]
        return fn
    return {n: make(n) for n in STAGES}


def test_run_project_and_create_refuse_a_busy_project(settings, tmp_path, live_pid):
    script = tmp_path / "s.md"
    script.write_text("# T\n\nHello.")
    proj = create_project(settings, "busy", InputMode.script, script_path=str(script))
    _foreign_lock(Path(proj.paths.root), live_pid)
    with pytest.raises(ProjectBusy):
        run_project(settings, "busy", stages=_stubs(), echo=False)
    with pytest.raises(ProjectBusy):
        create_project(settings, "busy", InputMode.script, script_path=str(script))
    assert (Path(proj.paths.root) / "project.json").exists()


def test_cli_reports_busy_project(settings, tmp_path, live_pid, monkeypatch, capsys):
    script = tmp_path / "s.md"
    script.write_text("# T\n\nHello.")
    proj = create_project(settings, "busy", InputMode.script, script_path=str(script))
    _foreign_lock(Path(proj.paths.root), live_pid)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    assert cli.main(["resume", "busy", "--quiet"]) == 3
    assert "is busy" in capsys.readouterr().err
