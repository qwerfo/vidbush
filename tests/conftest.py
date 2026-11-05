"""Shared test fixtures. Tests never touch the network: settings are offline."""
from __future__ import annotations

from pathlib import Path

import pytest

from vidbush.config import REPO_ROOT, load_settings
from vidbush.jobs.context import build_context
from vidbush.jobs.pipeline import create_project
from vidbush.models import Brief, InputMode, VideoFormat


@pytest.fixture
def settings(tmp_path: Path):
    """Offline settings writing output and data under tmp_path."""
    s = load_settings(REPO_ROOT, use_dotenv=False)
    s.offline = True
    s.output_folder = str(tmp_path / "output")
    s.data_folder = str(tmp_path / "data")
    s.source_cache_folder = str(tmp_path / "cache")
    s.fail_at_stage = None
    return s


@pytest.fixture
def make_ctx(settings):
    """Factory: make_ctx(mode='script', script_text=...) -> StageContext for a fresh project."""
    def _make(mode: str = "script", script_text: str = "", topic: str = "The Panama Canal",
              minutes: float = 4.0, fmt: str = "documentary", project_id: str = "t1"):
        kw = {}
        if mode == "brief":
            kw["brief"] = Brief(topic=topic, target_minutes=minutes, format=VideoFormat(fmt))
        elif mode == "script":
            p = Path(settings.output_folder).parent / "input_script.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(script_text or "# Title\n\nA short test script about the canal.", encoding="utf-8")
            kw["script_path"] = str(p)
        proj = create_project(settings, project_id, InputMode(mode), fmt=VideoFormat(fmt), seed=42,
                              target_minutes=minutes, **kw)
        return build_context(settings, proj, echo=False)
    return _make


@pytest.fixture
def generic_seats(settings):
    """Settings with the generic test seat map (tests/fixtures/seats_generic.yaml) instead of the
    operator's seat policy, so provider tests check mechanisms, not Adam's current provider choice."""
    import yaml
    settings.seats = yaml.safe_load((Path(__file__).parent / "fixtures" / "seats_generic.yaml").read_text())
    return settings
