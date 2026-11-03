"""Pipeline runner: cascade on rerun, persisted edit force sets, needs_review, moved projects, briefs."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from vidbush.jobs import pipeline
from vidbush.jobs.context import build_context
from vidbush.jobs.paths import rebase_path, rebase_tree
from vidbush.jobs.pipeline import (InjectedFailure, create_project, downstream, load_project, mark_pending,
                                   run_project)
from vidbush.models import STAGES, Brief, InputMode, Narration, StageStatus, VideoFormat
from vidbush.util import write_json


def _stubs(calls: list[str], qa: dict | None = None):
    def make(name):
        def fn(ctx):
            calls.append(name)
            p = ctx.work / f"{name}.done"
            p.write_text("ok")
            if name == "packaging" and qa is not None:
                return [str(p), str(write_json(ctx.out_path("qa_report.json"), qa))]
            return [str(p)]
        return fn
    return {n: make(n) for n in STAGES}


def _script_project(settings, tmp_path: Path, pid: str = "p1"):
    script = tmp_path / "s.md"
    script.write_text("# T\n\nHello world.")
    return create_project(settings, pid, InputMode.script, script_path=str(script), seed=1)


def test_downstream_graph():
    assert downstream("motion") == {"assembly", "packaging"}
    assert downstream("narration") == {"shotlist", "sourcing", "judge", "visuals", "motion", "mix", "assembly",
                                       "packaging"}
    assert "mix" not in downstream("visuals")


def test_forced_stage_cascades_to_dependants(settings, tmp_path):
    _script_project(settings, tmp_path)
    run_project(settings, "p1", stages=_stubs([]), echo=False)
    calls: list[str] = []
    run_project(settings, "p1", stages=_stubs(calls), force=["narration"], echo=False)
    assert calls == ["narration", "shotlist", "sourcing", "judge", "visuals", "motion", "mix", "assembly",
                     "packaging"]


def test_failed_edit_render_then_resume_reruns_later_stages(settings, tmp_path):
    _script_project(settings, tmp_path)
    run_project(settings, "p1", stages=_stubs([]), echo=False)
    mark_pending(settings, "p1", ["motion", "assembly", "packaging"])
    proj = load_project(settings, "p1")
    assert proj.stages["assembly"].status == StageStatus.pending
    assert proj.stages["mix"].status == StageStatus.done
    settings.fail_at_stage = "motion"
    with pytest.raises(InjectedFailure):
        run_project(settings, "p1", stages=_stubs([]), force=["motion"], echo=False)
    settings.fail_at_stage = None
    calls: list[str] = []
    run_project(settings, "p1", stages=_stubs(calls), echo=False)  # plain resume, no force
    assert calls == ["motion", "assembly", "packaging"]


def test_needs_review_when_real_seats_and_mock_used(settings, tmp_path, monkeypatch):
    _script_project(settings, tmp_path)
    settings.offline = False
    monkeypatch.setattr("vidbush.providers.seats.Seats.status",
                        lambda self: {"writer": {"real": True}, "judge": {"real": False}})
    proj = run_project(settings, "p1", stages=_stubs([], qa={"mock_used": ["judge"]}), echo=False)
    assert proj.status == "needs_review"
    assert load_project(settings, "p1").status == "needs_review"


def test_done_when_offline_or_all_mock(settings, tmp_path, monkeypatch):
    _script_project(settings, tmp_path)
    proj = run_project(settings, "p1", stages=_stubs([], qa={"mock_used": ["judge"]}), echo=False)
    assert proj.status == "done"  # offline: mock is the operator's choice
    settings.offline = False
    monkeypatch.setattr("vidbush.providers.seats.Seats.status", lambda self: {"judge": {"real": False}})
    assert run_project(settings, "p1", stages=_stubs([]), force=["packaging"], echo=False).status == "done"


# ---------------------------------------------------------------- moved projects

def test_rebase_path_rules():
    new = "/app/output/demo"
    assert rebase_path("/home/u/vb/output/demo/work/a.json", "demo", new, ["/home/u/vb/output/demo"]) == \
        "/app/output/demo/work/a.json"
    # no known old root: the project id marker followed by a project folder part
    assert rebase_path("/srv/x/output/demo/sources/s1/f.mp4", "demo", new) == "/app/output/demo/sources/s1/f.mp4"
    assert rebase_path("/home/demo/output/demo/final.mp4", "demo", new) == "/app/output/demo/final.mp4"
    assert rebase_path("/srv/output/demo", "demo", new) == "/app/output/demo"
    assert rebase_path("C:\\vb\\output\\demo\\work\\x.wav", "demo", new) == "/app/output/demo/work/x.wav"
    # left alone: other projects, the shared cache, urls, relative paths, already current
    assert rebase_path("/srv/output/demo2/work/a.json", "demo", new) == "/srv/output/demo2/work/a.json"
    assert rebase_path("/cache/yt/abc.mp4", "demo", new) == "/cache/yt/abc.mp4"
    assert rebase_path("https://x.org/demo/work/a", "demo", new) == "https://x.org/demo/work/a"
    assert rebase_path("work/a.json", "demo", new) == "work/a.json"
    assert rebase_path("/app/output/demo/work/a", "demo", new) == "/app/output/demo/work/a"
    tree = rebase_tree({"a": ["/old/demo/work/x"], "n": 3}, "demo", new, ["/old/demo"])
    assert tree == {"a": ["/app/output/demo/work/x"], "n": 3}


def test_moved_project_loads_with_current_paths(settings, tmp_path):
    proj = _script_project(settings, tmp_path, "mv")
    old_root = Path(proj.paths.root)
    nar = Narration(audio_path=str(old_root / "work" / "narration" / "narration.wav"), duration=3.0, words=[])
    write_json(old_root / "work" / "narration.json", nar)
    new_out = tmp_path / "elsewhere" / "output"
    new_out.parent.mkdir(parents=True)
    shutil.move(str(old_root.parent), str(new_out))
    settings.output_folder = str(new_out)
    moved = load_project(settings, "mv")
    new_root = new_out / "mv"
    assert moved.paths.root == str(new_root) and moved.paths.work == str(new_root / "work")
    assert moved.script_path == str(new_root / "inputs" / "s.md")
    ctx = build_context(settings, moved, echo=False)
    assert ctx.load("narration", Narration).audio_path == str(new_root / "work" / "narration" / "narration.wav")
    ctx.save_project()  # after a save the stored root is the new one; the marker rule still rebases
    assert build_context(settings, load_project(settings, "mv"), echo=False).load(
        "narration", Narration).audio_path.startswith(str(new_root))


def test_resume_of_moved_project_writes_under_new_root(settings, tmp_path):
    proj = _script_project(settings, tmp_path, "mv2")
    run_project(settings, "mv2", stages=_stubs([]), echo=False)
    old_out = Path(proj.paths.root).parent
    new_out = tmp_path / "moved_output"
    shutil.move(str(old_out), str(new_out))
    settings.output_folder = str(new_out)
    calls: list[str] = []
    run_project(settings, "mv2", stages=_stubs(calls), echo=False)
    assert calls == []  # every stage output was found under the new root
    data = json.loads((new_out / "mv2" / "project.json").read_text())
    assert all(str(new_out) in o for r in data["stages"].values() for o in r["outputs"])


# ---------------------------------------------------------------- briefs and formats

def test_create_project_overrides_brief_format_and_minutes(settings):
    brief = Brief(topic="Wonders", target_minutes=8, format=VideoFormat.documentary)
    proj = create_project(settings, "b1", InputMode.brief, brief=brief, fmt=VideoFormat.top10, target_minutes=12)
    assert proj.brief.format == VideoFormat.top10 and proj.brief.target_minutes == 12
    assert proj.format == VideoFormat.top10 and proj.target_minutes == 12
    keep = create_project(settings, "b2", InputMode.brief, brief=brief)
    assert keep.brief.format == VideoFormat.documentary and keep.brief.target_minutes == 8


TOP10 = ("# Ancient Wonders\n\n## Number 3: The Colosseum\n\nThe arena held fifty thousand people.\n\n"
         "## Number 2: Petra\n\nCarved into rose red cliffs.\n\n## Number 1: Stonehenge\n\nStones older than writing.\n")


def test_stage_script_switches_to_top10_profile(make_ctx):
    from vidbush.jobs.stages import stage_intake, stage_script
    ctx = make_ctx(mode="script", script_text=TOP10, project_id="tt")
    stage_intake(ctx)
    assert not ctx.profile.top10
    stage_script(ctx)
    assert ctx.project.format == VideoFormat.top10
    assert ctx.profile.top10 and ctx.project.style_profile_id == ctx.profile.id
    assert load_project(ctx.settings, "tt").style_profile_id == ctx.profile.id


def test_stage_script_keeps_operator_profile(settings, tmp_path):
    from vidbush.jobs.stages import stage_script
    src = tmp_path / "top.md"
    src.write_text(TOP10)
    proj = create_project(settings, "op", InputMode.script, script_path=str(src), style_profile_id="documentary_default")
    ctx = build_context(settings, proj, echo=False)
    stage_script(ctx)
    assert ctx.project.format == VideoFormat.top10 and not ctx.profile.top10
    assert ctx.project.style_profile_id == "documentary_default"


def test_invalidate_from_still_marks_everything_after(settings, tmp_path):
    _script_project(settings, tmp_path)
    run_project(settings, "p1", stages=_stubs([]), echo=False)
    proj = pipeline.invalidate_from(settings, "p1", "mix")
    assert [n for n in STAGES if proj.stages[n].status == StageStatus.pending] == ["mix", "assembly", "packaging"]
