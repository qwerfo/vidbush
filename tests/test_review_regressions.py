"""Review sheet regressions found by the review audit: snapshots after a fix, failed fixes, reused shot ids."""
from __future__ import annotations

import time

import pytest

from review_fixture import PID, build_review_project, build_shots, write_shots
from vidbush.jobs import runner
from vidbush.models import EditResult, ShotType
from vidbush.review import calibrate, export, fixes, store
from vidbush.review.store import ShotReview
from vidbush.util import write_json


@pytest.fixture
def root(settings):
    return build_review_project(settings)


def _repick(root, shot_id: str, cand_id: str, score: float) -> None:
    """Simulate a fix job that re-sourced shot_id: the pick and its score change."""
    shots = build_shots(root)
    for s in shots:
        if s.id == shot_id:
            s.selected, s.judge_score = cand_id, score
    write_shots(root, shots)


def test_comment_edit_after_fix_keeps_reviewed_snapshot(settings, root):
    store.update_shot(settings, PID, "shot_001", verdict="wrong_picture", comment="Modern boats.")
    p = fixes.plan(settings, PID)
    store.mark_fixed(settings, PID, p["ops"], None)
    _repick(root, "shot_001", "shot_001_archive_2", 95.0)
    time.sleep(0.01)
    _, e = store.update_shot(settings, PID, "shot_001", comment="Modern boats!")
    assert e.snapshot.selected == "shot_001_youtube_0" and e.snapshot.judge_score == 84
    assert fixes.plan(settings, PID)["ops"] == []
    assert calibrate.calibrate_project(settings, PID)["wrong_above_threshold"][0]["score"] == 84
    # a new verdict on the new pick reviews it: fresh snapshot, fix plannable again
    store.update_shot(settings, PID, "shot_001", verdict="good")
    _, e = store.update_shot(settings, PID, "shot_001", verdict="wrong_picture")
    assert e.snapshot.selected == "shot_001_archive_2"
    assert [o["args"]["shot_id"] for o in fixes.plan(settings, PID)["ops"]] == ["shot_001"]


def test_cli_marks_only_applied_fixes(settings, root, monkeypatch, capsys):
    from vidbush import cli
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    store.update_shot(settings, PID, "shot_001", verdict="wrong_picture", comment="I want canyon walls.")
    store.update_shot(settings, PID, "shot_003", verdict="wrong_picture", comment="I want the dam.")

    def fake(settings_, pid, ops, echo=True):
        return EditResult(ops=ops, applied=["change_type: shot_001 re-sourced"],
                          errors=["change_type: no usable candidate found for shot_003 as image"])
    monkeypatch.setattr("vidbush.editor.ops.apply_and_render", fake)
    assert cli.main(["review-fixes", PID, "--apply", "--quiet"]) == 1
    left = fixes.plan(settings, PID)
    assert [o["args"]["shot_id"] for o in left["ops"]] == ["shot_003"]
    assert [s["shot_id"] for s in left["skipped"]] == ["shot_001"]


def test_failed_fix_job_is_offered_again(settings, root):
    store.update_shot(settings, PID, "shot_001", verdict="wrong_picture")
    p = fixes.plan(settings, PID)
    job_id = runner.enqueue(settings, PID, "apply_ops", {"ops": p["ops"]})
    store.mark_fixed(settings, PID, p["ops"], job_id)
    assert fixes.plan(settings, PID)["ops"] == []
    write_json(root / "work" / "edit_plan.json",
               {"job_id": job_id, "errors": ["change_type: no usable candidate found for shot_001 as footage"]})
    assert [o["args"]["shot_id"] for o in fixes.plan(settings, PID)["ops"]] == ["shot_001"]


def test_reused_split_id_does_not_revive_old_review(settings, root):
    shots = build_shots(root)
    a = shots[2]
    b = a.model_copy(deep=True, update={"id": "shot_003b", "start": a.start + 2, "duration": a.duration - 2})
    a.duration, a.end = 2.0, a.start + 2
    write_shots(root, shots[:3] + [b] + shots[3:])
    store.update_shot(settings, PID, "shot_003b", verdict="wrong_picture", comment="Piece b shows a face")
    write_shots(root, build_shots(root))  # merged back
    assert store.load(settings, PID).reviewed == 0
    shots = build_shots(root)
    a = shots[2]
    b = a.model_copy(deep=True, update={"id": "shot_003b", "start": a.start + 1.5, "duration": a.duration - 1.5})
    a.duration, a.end = 1.5, a.start + 1.5
    write_shots(root, shots[:3] + [b] + shots[3:])
    r = store.load(settings, PID)
    assert r.entry("shot_003b").verdict is None and r.reviewed == 0
    old = [e for e in r.shots if e.shot_id.startswith("shot_003b~")]
    assert len(old) == 1 and old[0].comment == "Piece b shows a face" and not old[0].in_shotlist


def test_ai_video_stays_a_comment(root):
    shot = build_shots(root)[0].model_copy(update={"type": ShotType.ai_video})
    for verdict, rejects in (("wrong_picture", []), ("quality_problem", ["watermark_logo"])):
        op, why = fixes.op_for(shot, ShotReview(shot_id=shot.id, verdict=verdict, rejects=rejects, comment="x"))
        assert op is None and "AI video" in why


def test_export_uses_reviewed_number_and_time(settings, root):
    store.update_shot(settings, PID, "shot_004", verdict="quality_problem", rejects=["watermark_logo"], comment="logo")
    shots = build_shots(root)
    write_shots(root, shots[2:])  # two shots removed before shot_004: it is now shot 2 at a new time
    md = export.markdown(store.load(settings, PID))
    assert "### Shot 4 (shot_004) at 0:12.5" in md and "| 4 (shot_004) | 0:12.5 |" in md
