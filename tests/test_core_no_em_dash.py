"""A10: no em dash character anywhere in the repository (text files)."""
from __future__ import annotations

import os
import subprocess

from vidbush.config import REPO_ROOT

SKIP_DIRS = {".git", "node_modules", "output", "data", "__pycache__", ".pytest_cache", ".venv", "venv"}
EM = chr(0x2014).encode()


def _repo_files() -> list[str]:
    """Files git tracks or would track (operator files git ignores, such as CLAUDE.local.md, are not the
    repository). Without git, every file outside SKIP_DIRS."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=REPO_ROOT,
                             capture_output=True, check=True).stdout.decode("utf-8")
        return [os.path.join(REPO_ROOT, f) for f in out.split("\0") if f]
    except (OSError, subprocess.CalledProcessError):
        files = []
        for root, dirs, names in os.walk(REPO_ROOT):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            files += [os.path.join(root, f) for f in names]
        return files


def test_no_em_dash_in_repo():
    bad = []
    for path in _repo_files():
        try:
            with open(path, "rb") as fh:
                if EM in fh.read():
                    bad.append(path)
        except OSError:
            continue
    assert not bad, bad
