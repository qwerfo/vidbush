"""Integration: the research, outline and script stage functions with the mock seats and fixtures."""
from __future__ import annotations

from pathlib import Path

from vidbush.config import REPO_ROOT
from vidbush.jobs.stages import stage_outline, stage_research, stage_script
from vidbush.models import Outline, Research, Script, VideoFormat


def test_brief_stages_use_fixtures(make_ctx):
    ctx = make_ctx(mode="brief", topic="The Panama Canal", minutes=4)
    stage_research(ctx)
    stage_outline(ctx)
    stage_script(ctx)
    assert len(ctx.load("research", Research).facts) >= 30
    assert len(ctx.load("outline", Outline).sections) == 7
    script = ctx.load("script", Script)
    assert len(script.beats()) == 35 and script.title.startswith("The Panama Canal")
    assert 560 <= script.word_count() <= 680


def test_script_stage_parses_demo_script(make_ctx):
    text = (Path(REPO_ROOT) / "fixtures" / "demo_script.md").read_text(encoding="utf-8")
    ctx = make_ctx(mode="script", script_text=text)
    stage_script(ctx)
    script = ctx.load("script", Script)
    assert script.format == VideoFormat.documentary and len(script.scenes) == 7
    assert ctx.project.title.startswith("Hoover Dam")
