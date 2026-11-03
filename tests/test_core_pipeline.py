"""Job runner contract: stage order, skip of done stages, injected failure, resume."""
from __future__ import annotations

from pathlib import Path

import pytest

from vidbush.cli import detect_input, parse_brief_file
from vidbush.jobs import pipeline
from vidbush.jobs.pipeline import InjectedFailure, create_project, load_project, run_project
from vidbush.models import STAGES, InputMode, StageStatus


def _stub_stages(calls: list[str]):
    def make(name):
        def fn(ctx):
            calls.append(name)
            p = ctx.work / f"{name}.done"
            p.write_text("ok")
            return [str(p)]
        return fn
    return {n: make(n) for n in STAGES}


def _project(settings, tmp_path: Path):
    script = tmp_path / "s.md"
    script.write_text("# T\n\nHello world.")
    return create_project(settings, "p1", InputMode.script, script_path=str(script), seed=1)


def test_runs_all_applicable_stages_in_order(settings, tmp_path):
    _project(settings, tmp_path)
    calls: list[str] = []
    proj = run_project(settings, "p1", stages=_stub_stages(calls), echo=False)
    assert proj.status == "done"
    assert calls == [s for s in STAGES if s not in ("research", "outline")]
    assert proj.stages["research"].status == StageStatus.skipped


def test_resume_after_injected_failure_skips_done_stages(settings, tmp_path):
    _project(settings, tmp_path)
    calls: list[str] = []
    settings.fail_at_stage = "judge"
    with pytest.raises(InjectedFailure):
        run_project(settings, "p1", stages=_stub_stages(calls), echo=False)
    assert load_project(settings, "p1").stages["judge"].status == StageStatus.failed
    settings.fail_at_stage = None
    calls.clear()
    proj = run_project(settings, "p1", stages=_stub_stages(calls), echo=False)
    assert calls[0] == "judge"
    assert "sourcing" not in calls and "intake" not in calls
    assert proj.status == "done"


def test_missing_output_reruns_stage(settings, tmp_path):
    proj = _project(settings, tmp_path)
    calls: list[str] = []
    run_project(settings, "p1", stages=_stub_stages(calls), echo=False)
    (Path(proj.paths.work) / "mix.done").unlink()
    calls.clear()
    run_project(settings, "p1", stages=_stub_stages(calls), echo=False)
    assert calls == ["mix"]


def test_invalidate_from(settings, tmp_path):
    _project(settings, tmp_path)
    run_project(settings, "p1", stages=_stub_stages([]), echo=False)
    proj = pipeline.invalidate_from(settings, "p1", "assembly")
    assert proj.stages["assembly"].status == StageStatus.pending
    assert proj.stages["mix"].status == StageStatus.done


def test_detect_input(tmp_path):
    brief = tmp_path / "b.md"
    brief.write_text("---\ntopic: The Panama Canal\ntarget_minutes: 4\nformat: documentary\n---\nNotes")
    mode, kw = detect_input(brief, None)
    assert mode == InputMode.brief and kw["brief"].target_minutes == 4
    script = tmp_path / "s.md"
    script.write_text("# Title\n\nText.")
    assert detect_input(script, None)[0] == InputMode.script
    audio = tmp_path / "v.mp3"
    audio.write_bytes(b"")
    (tmp_path / "v.md").write_text("words")
    mode, kw = detect_input(audio, None)
    assert mode == InputMode.voiceover and kw["script_path"].endswith("v.md")
    assert parse_brief_file(script) is None
