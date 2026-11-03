"""Edit robustness: atomic rewrite_beat, persisted force sets, recovered jobs, clip spans, motion splits."""
from __future__ import annotations

from pathlib import Path

import pytest

from test_editor_helpers import PID, assert_timeline, build_edit_project, load_shotlist, make_video
from vidbush.editor import ops as ops_mod
from vidbush.editor.ops import apply_and_render, apply_ops
from vidbush.editor.retime import EditError
from vidbush.jobs.pipeline import load_project
from vidbush.models import EditOp, EditOpName, Narration, Script, StageRecord, StageStatus
from vidbush.util import read_json, read_model, write_json


def op(name: str, **args) -> EditOp:
    return EditOp(op=EditOpName(name), args=args)


@pytest.fixture
def work(settings) -> Path:
    return build_edit_project(settings)


def _fake_renarrate(delta: float):
    """renarrate_beat stand-in that, like the real one, rewrites narration.wav in place."""
    def renarrate(narration, script, beat_id, new_text, brand, seats, settings_, work_dir, log):
        (Path(work_dir) / "narration" / "narration.wav").write_bytes(b"NEW AUDIO")
        return narration.model_copy(update={"duration": round(narration.duration + delta, 3)})
    return renarrate


def _snapshot(work: Path) -> dict[str, bytes]:
    return {n: (work / n).read_bytes() for n in ("script.json", "narration.json", "shotlist.json",
                                                 "narration/narration.wav")}


def test_rewrite_beat_failing_fit_restores_everything(settings, work, monkeypatch):
    (work / "narration" / "narration.wav").write_bytes(b"OLD AUDIO")
    before = _snapshot(work)
    monkeypatch.setattr("vidbush.audio.narration.renarrate_beat", _fake_renarrate(-4.4))

    def boom(*a, **k):
        raise EditError("the rewritten beat is too short to keep its shot")
    monkeypatch.setattr(ops_mod, "fit_beat_shots", boom)
    res = apply_ops(settings, PID, [op("rewrite_beat", beat_id="b001", instruction="shorter")])
    assert res.errors and not res.applied and res.force == []
    assert _snapshot(work) == before
    assert read_json(work / "edit_plan.json")["force"] == []
    assert all(s.duration > 0 for s in load_shotlist(work).shots)


def test_rewrite_beat_failing_timeline_check_rolls_back(settings, work, monkeypatch):
    (work / "narration" / "narration.wav").write_bytes(b"OLD AUDIO")
    before = _snapshot(work)
    monkeypatch.setattr("vidbush.audio.narration.renarrate_beat", _fake_renarrate(-1.0))
    calls = []

    def check(sl, total):
        calls.append(total)
        raise EditError(f"timeline ends at 0 s instead of {total} s")
    monkeypatch.setattr(ops_mod.retime, "check_timeline", check)
    res = apply_ops(settings, PID, [op("rewrite_beat", beat_id="b001", instruction="shorter")])
    assert res.errors and calls
    assert _snapshot(work) == before


def test_rewrite_beat_success_commits_and_removes_backup(settings, work, monkeypatch):
    monkeypatch.setattr("vidbush.audio.narration.renarrate_beat", _fake_renarrate(-1.0))
    res = apply_ops(settings, PID, [op("rewrite_beat", beat_id="b001", instruction="shorter")])
    assert not res.errors, res.errors
    assert read_model(work / "narration.json", Narration).duration == 28.0
    assert (work / "narration" / "narration.wav").read_bytes() == b"NEW AUDIO"
    assert not (work / "narration" / "narration.wav.edit_bak").exists()
    assert read_model(work / "script.json", Script).beats()[0].text == "The canal opened in 1914. It changed world trade."
    assert_timeline(load_shotlist(work), total=29.0)


def _done_stages(settings) -> None:
    proj = load_project(settings, PID)
    for name in proj.stages:
        proj.stages[name] = StageRecord(name=name, status=StageStatus.done)
    write_json(Path(proj.paths.root) / "project.json", proj)


def test_apply_ops_marks_forced_stages_pending(settings, work):
    _done_stages(settings)
    res = apply_ops(settings, PID, [op("set_duration", shot_id="shot_002", seconds=5)], job_id=41)
    assert res.force == ["motion", "assembly", "packaging"]
    stages = load_project(settings, PID).stages
    assert stages["motion"].status == stages["assembly"].status == StageStatus.pending
    assert stages["mix"].status == StageStatus.done
    assert read_json(work / "edit_plan.json")["job_id"] == 41


def test_recovered_job_renders_without_reapplying(settings, work, monkeypatch):
    rendered: list[list[str]] = []
    monkeypatch.setattr("vidbush.jobs.pipeline.run_project",
                        lambda s, pid, force=(), echo=True, **k: rendered.append(list(force)))
    split = [op("split_shot", shot_id="shot_007")]
    first = apply_and_render(settings, PID, split, echo=False, job_id=7)
    count = len(load_shotlist(work).shots)
    again = apply_and_render(settings, PID, split, echo=False, job_id=7)  # worker recovery requeued job 7
    assert len(load_shotlist(work).shots) == count == 9
    assert rendered == [first.force, first.force] and "recovered" in again.applied[-1]
    apply_and_render(settings, PID, split, echo=False, job_id=8)  # a new job applies again
    assert len(load_shotlist(work).shots) == 10


def test_split_motion_graphic_is_refused(settings, work):
    res = apply_ops(settings, PID, [op("split_shot", shot_id="shot_005")])
    assert res.errors and "motion graphic" in res.errors[0]
    assert len(load_shotlist(work).shots) == 8


def test_lengthened_footage_reads_further_into_its_file(settings, work):
    long_clip = make_video(work / "long.mp4", 6.0)
    sl = load_shotlist(work)
    shot = sl.by_id("shot_001")
    for c in shot.candidates:
        c.local_path, c.in_point, c.out_point = long_clip, 1.0, 4.0
    write_json(work / "shotlist.json", sl)
    res = apply_ops(settings, PID, [op("set_duration", shot_id="shot_001", seconds=4.5)])
    assert not res.errors, res.errors
    cand = load_shotlist(work).by_id("shot_001").selected_candidate()
    assert cand.in_point == 1.0 and cand.out_point == 5.5
    res = apply_ops(settings, PID, [op("set_duration", shot_id="shot_001", seconds=5.5)])
    assert not res.errors, res.errors
    cand = load_shotlist(work).by_id("shot_001").selected_candidate()
    assert abs(cand.out_point - 6.0) < 0.05 and abs(cand.in_point - 0.5) < 0.05  # end of file, so start earlier


def test_short_file_gets_a_note_not_an_error(settings, work):
    res = apply_ops(settings, PID, [op("set_duration", shot_id="shot_001", seconds=5.0)])
    assert not res.errors and "footage" in res.applied[0]
    cand = load_shotlist(work).by_id("shot_001").selected_candidate()
    assert cand.in_point == 0.0 and cand.out_point == 2.0
