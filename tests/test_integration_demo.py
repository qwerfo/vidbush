"""Integration: the demo in mock mode end to end, run twice to prove the segment cache.

Slow (renders a 4 minute video). Run alone with: python -m pytest tests/test_integration_demo.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vidbush.acceptance import check_project
from vidbush.cli import detect_input
from vidbush.config import REPO_ROOT
from vidbush.jobs.pipeline import create_project, run_project

pytestmark = pytest.mark.slow


def test_demo_end_to_end_and_cache(settings):
    mode, kw = detect_input(REPO_ROOT / "fixtures" / "demo_brief.md", None)
    create_project(settings, "demo", mode, seed=11, **kw)
    proj = run_project(settings, "demo", echo=False)
    assert proj.status == "done"
    res = check_project(settings, "demo")
    assert res["ok"], json.dumps(res, indent=2)
    root = Path(settings.output_folder) / "demo"
    first = json.loads((root / "work" / "render_stats.json").read_text())
    assert first["rendered"] > 0
    run_project(settings, "demo", force=["assembly"], echo=False)
    second = json.loads((root / "work" / "render_stats.json").read_text())
    assert second["rendered"] == 0 and second["cached"] == first["segments"]
