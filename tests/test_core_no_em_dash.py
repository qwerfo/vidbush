"""A10: no em dash character anywhere in the repository (text files)."""
from __future__ import annotations

import os

from vidbush.config import REPO_ROOT

SKIP_DIRS = {".git", "node_modules", "output", "data", "__pycache__", ".pytest_cache"}
EM = chr(0x2014).encode()


def test_no_em_dash_in_repo():
    bad = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            path = os.path.join(root, f)
            try:
                with open(path, "rb") as fh:
                    if EM in fh.read():
                        bad.append(path)
            except OSError:
                continue
    assert not bad, bad
