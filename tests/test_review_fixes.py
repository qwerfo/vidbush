"""Review verdicts to EditOps: the mapping table, comment queries, the plan, and the editor change_type path."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from review_fixture import PID, build_review_project
from vidbush.editor.ops import _review_notes
from vidbush.models import Candidate, EditOp, EditOpName, Shot, ShotType, SourceKind
from vidbush.review import fixes, store
from vidbush.review.store import ShotReview

EM = chr(0x2014)


def _shot(typ: ShotType) -> Shot:
    c = Candidate(id="s_pick", source=SourceKind.wikimedia, score=80)
    return Shot(id="shot_009", index=8, beat_id="b1", start=10, end=14, duration=4, type=typ, intent="dam",
                candidates=[] if typ == ShotType.motion_graphic else [c],
                selected=None if typ == ShotType.motion_graphic else "s_pick")


RESOURCE = ["footage", "footage_overlay", "image", "image_overlay"]
AI = ["ai_image", "ai_video"]
# (verdict, rejects) -> expected op per type group: change_type, regenerate_ai or None
CASES = [
    ("wrong_picture", [], {"resource": "change_type", "ai": "regenerate_ai", "motion": None}),
    ("quality_problem", ["watermark_logo"], {"resource": "change_type", "ai": "regenerate_ai", "motion": None}),
    ("quality_problem", [], {"resource": None, "ai": None, "motion": None}),
    ("wrong_timing", [], {"resource": None, "ai": None, "motion": None}),
    ("other", ["talking_head"], {"resource": None, "ai": None, "motion": None}),
    ("good", [], {"resource": None, "ai": None, "motion": None}),
    (None, [], {"resource": None, "ai": None, "motion": None}),
]


@pytest.mark.parametrize("verdict,rejects,expect", CASES)
@pytest.mark.parametrize("typ", [t.value for t in ShotType])
def test_op_mapping_table(typ, verdict, rejects, expect):
    group = "resource" if typ in RESOURCE else "ai" if typ in AI else "motion"
    shot = _shot(ShotType(typ))
    op, why = fixes.op_for(shot, ShotReview(shot_id=shot.id, verdict=verdict, rejects=rejects, comment="Wrong dam."))
    assert (op.op.value if op else None) == expect[group], why
    assert why
    if op is not None:
        assert op.args["shot_id"] == "shot_009" and op.reason.startswith("review: shot 9 ")
        if op.op == EditOpName.change_type:
            assert op.args["type"] == typ  # same type: the editor re-source path
            assert op.args["reject_selected"].startswith("operator review: ")
        else:
            assert op.args["prompt_delta"].startswith("Avoid: ")


def test_wrong_picture_resource_args():
    shot = _shot(ShotType.footage)
    e = ShotReview(shot_id=shot.id, verdict="wrong_picture",
                   comment=f"Too modern {EM} the boats have outboard motors. Show 1931 canyon walls.")
    op, _ = fixes.op_for(shot, e)
    assert op.args["add_must_avoid"] == ["Too modern, the boats have outboard motors. Show 1931 canyon walls."]
    assert op.args["add_queries"] == ["Too modern, the boats have outboard motors"]
    assert op.args["reject_selected"] == "operator review: wrong picture"
    assert EM not in op.reason


def test_wrong_picture_without_comment():
    op, _ = fixes.op_for(_shot(ShotType.image), ShotReview(shot_id="shot_009", verdict="wrong_picture"))
    assert op.args["add_must_avoid"] == ["the picture rejected in review (s_pick)"]
    assert "add_queries" not in op.args
    op, _ = fixes.op_for(_shot(ShotType.ai_image), ShotReview(shot_id="shot_009", verdict="wrong_picture"))
    assert op.args["prompt_delta"] == "Avoid: the previous image, it did not fit the narration"


def test_ai_avoid_note_is_capped():
    op, _ = fixes.op_for(_shot(ShotType.ai_image), ShotReview(shot_id="shot_009", verdict="wrong_picture",
                                                              comment="word " * 400))
    assert len(op.args["prompt_delta"]) <= len("Avoid: ") + fixes.AVOID_PROMPT_CHARS


def test_quality_problem_rejects_to_avoid():
    shot = _shot(ShotType.image_overlay)
    e = ShotReview(shot_id=shot.id, verdict="quality_problem", rejects=["burned_in_text", "watermark_logo"],
                   comment="ignored for the avoid list")
    op, why = fixes.op_for(shot, e)
    assert op.args["add_must_avoid"] == ["burned-in text, subtitles or captions", "watermark or channel logo"]
    assert "add_queries" not in op.args
    assert op.args["reject_selected"] == "operator review: burned-in text, watermark or logo"
    assert why == "review: shot 9 quality problem (burned-in text, watermark or logo)"
    ai, _ = fixes.op_for(_shot(ShotType.ai_video), e)
    assert ai.args["prompt_delta"] == "Avoid: burned-in text, subtitles or captions, watermark or channel logo"


@pytest.mark.parametrize("comment,query", [
    ("Too modern. Show 1931 canyon walls.", "Too modern"),
    ("Show the \"Six Companies\" trucks! Not cars.", "Show the Six Companies trucks"),
    ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen",
     "one two three four five six seven eight nine ten eleven twelve"),
    ("  multi\nline   spaced  text  ", "multi line spaced text"),
    ("Is this 1931? Looks later.", "Is this 1931"),
    ("", ""),
])
def test_comment_query(comment, query):
    assert fixes.comment_query(comment) == query
    assert len(fixes.comment_query(comment).split()) <= fixes.QUERY_WORDS


# ---------------------------------------------------------------- plan on the demo-like project

@pytest.fixture
def root(settings) -> Path:
    return build_review_project(settings)


def test_plan_groups(settings, root):
    store.update_shot(settings, PID, 1, verdict="wrong_picture", comment="Too modern.")  # footage
    store.update_shot(settings, PID, 2, verdict="quality_problem", rejects=["watermark_logo"])  # footage_overlay
    store.update_shot(settings, PID, 3, verdict="quality_problem")  # no rejects: comment
    store.update_shot(settings, PID, 4, verdict="wrong_timing")
    store.update_shot(settings, PID, 5, verdict="wrong_picture")  # ai_image
    store.update_shot(settings, PID, 6, verdict="wrong_picture", comment="needs a better map")  # motion graphic
    store.update_shot(settings, PID, 7, verdict="other", comment="hmm")
    store.update_shot(settings, PID, 8, verdict="good")
    p = fixes.plan(settings, PID)
    assert [(o["op"], o["args"]["shot_id"]) for o in p["ops"]] == [
        ("change_type", "shot_001"), ("change_type", "shot_002"), ("regenerate_ai", "shot_005")]
    assert p["ops"][1]["args"]["type"] == "footage_overlay"
    assert [t["number"] for t in p["timing"]] == [4]
    assert p["timing"][0]["duration"] == 6.0 and p["timing"][0]["why"] == "wrong timing: open the duration editor"
    assert [c["number"] for c in p["comments"]] == [3, 6, 7]
    assert p["skipped"] == []
    assert all(EditOp.model_validate(o) for o in p["ops"])
    text = fixes.plan_text(p)
    assert text.startswith("Planned edit ops: 3") and "shot 4 at 12.5 s, now 6.00 s" in text
    assert "Stays a comment: 3" in text


def test_plan_skips_queued_until_changed(settings, root):
    import time
    store.update_shot(settings, PID, 1, verdict="wrong_picture", comment="x")
    p = fixes.plan(settings, PID)
    store.mark_fixed(settings, PID, p["ops"], 12)
    again = fixes.plan(settings, PID)
    assert again["ops"] == [] and again["skipped"][0]["why"] == "fix already queued (change_type)"
    assert "Already queued: 1" in fixes.plan_text(again)
    time.sleep(0.01)
    store.update_shot(settings, PID, 1, comment="still wrong")
    assert len(fixes.plan(settings, PID)["ops"]) == 1


def test_plan_without_shotlist(settings, tmp_path):
    from vidbush.jobs.pipeline import create_project
    from vidbush.models import InputMode
    s = tmp_path / "s.md"
    s.write_text("# T\n\nText.")
    create_project(settings, "bare", InputMode.script, script_path=str(s), seed=1)
    with pytest.raises(store.ReviewError):
        fixes.plan(settings, "bare")


# ---------------------------------------------------------------- editor: _review_notes on a Shot

def _notes_shot() -> Shot:
    cands = [Candidate(id="pick", source=SourceKind.youtube, score=84, judge_reason="ok"),
             Candidate(id="other", source=SourceKind.archive, score=61, judge_reason="river"),
             Candidate(id="bad", source=SourceKind.youtube, score=0, reject_reason="channel_logo (prefilter: x)",
                       judge_reason="logo")]
    return Shot(id="shot_001", beat_id="b1", start=0, end=4, duration=4, type=ShotType.footage, intent="canyon",
                must_avoid=["modern cars"], queries=["black canyon 1931"], candidates=cands, selected="pick")


def test_review_notes_appends_and_rejects():
    shot = _notes_shot()
    _review_notes(shot, {"add_must_avoid": ["modern cars", f"outboard  motors {EM} boats"],
                         "add_queries": ["black canyon 1931", "Colorado river 1930s"],
                         "reject_selected": "operator review: wrong picture"})
    assert shot.must_avoid == ["modern cars", "outboard motors, boats"]
    assert shot.queries == ["black canyon 1931", "Colorado river 1930s"]
    pick, other, bad = shot.candidates
    assert pick.reject_reason == "operator review: wrong picture" and pick.score == 0.0
    assert other.score is None and other.judge_reason == "" and other.reject_reason is None
    assert bad.score == 0 and bad.reject_reason.startswith("channel_logo") and bad.judge_reason == "logo"


def test_review_notes_string_args_and_no_avoid_keeps_scores():
    shot = _notes_shot()
    _review_notes(shot, {"add_queries": "dam workers", "reject_selected": "operator review: anachronism"})
    assert shot.queries == ["black canyon 1931", "dam workers"] and shot.must_avoid == ["modern cars"]
    assert shot.candidates[0].score == 0.0
    assert shot.candidates[1].score == 61  # must_avoid did not grow: scores stay


def test_review_notes_never_rejects_motion_and_noop_without_args():
    shot = _notes_shot()
    shot.candidates.append(Candidate(id="mg", source=SourceKind.motion))
    shot.selected = "mg"
    _review_notes(shot, {"reject_selected": "operator review: wrong picture"})
    assert shot.candidates[-1].reject_reason is None
    before = _notes_shot()
    after = _notes_shot()
    _review_notes(after, {})
    assert after == before


# ---------------------------------------------------------------- editor: apply_ops with the planned op

def test_apply_planned_change_type_does_not_reselect_old_pick(settings, monkeypatch, tmp_path):
    from test_editor_helpers import PID as EPID, build_edit_project, load_shotlist
    from vidbush.editor.ops import apply_ops
    work = build_edit_project(settings)
    new_png = tmp_path / "new.png"
    Image.new("RGB", (640, 360), (30, 90, 160)).save(new_png)
    seen = {}

    def source_shot(shot, ctx, queries=None, exclude_ids=()):
        seen["must_avoid"], seen["queries"] = list(shot.must_avoid), list(shot.queries)
        return [Candidate(id=f"{shot.id}_local_9", source=SourceKind.local, local_path=str(new_png), media="image",
                          width=640, height=360, keyframes=[str(new_png)], url="local://new")]

    def judge(shotlist, ctx):  # pick the best candidate without a reject, like director_loop._ranked
        for s in shotlist.shots:
            for c in s.candidates:
                if c.score is None:
                    c.score = 90.0 if c.id.endswith("_local_9") else 60.0
            best = sorted(s.candidates, key=lambda c: (c.reject_reason is None, c.score or 0.0), reverse=True)[0]
            s.selected, s.judge_score = best.id, best.score
        return shotlist

    monkeypatch.setattr("vidbush.source.registry.source_shot", source_shot)
    monkeypatch.setattr("vidbush.judge.director_loop.judge_and_select", judge)
    before = load_shotlist(work).by_id("shot_002")
    old = before.selected
    shot = before.model_copy(deep=True)
    op, _ = fixes.op_for(shot, ShotReview(shot_id="shot_002", verdict="wrong_picture",
                                          comment="Not this dam. Show Hoover Dam concrete pour."))
    res = apply_ops(settings, EPID, [op])
    assert not res.errors, res.errors
    after = load_shotlist(work).by_id("shot_002")
    assert after.type == ShotType.image and after.selected != old
    assert after.selected == "shot_002_local_9"
    rejected = {c.id: c.reject_reason for c in after.candidates}
    assert rejected[old] == "operator review: wrong picture"
    assert "Not this dam. Show Hoover Dam concrete pour." in after.must_avoid
    assert "Not this dam" in after.queries
    assert seen["must_avoid"][-1] == "Not this dam. Show Hoover Dam concrete pour."
    assert "Not this dam" in seen["queries"]


def test_apply_planned_change_type_with_mock_judge(settings, monkeypatch, tmp_path):
    """The real offline judge: the rejected pick is ranked last and never selected again."""
    from test_editor_helpers import PID as EPID, build_edit_project, load_shotlist
    from vidbush.editor.ops import apply_ops
    work = build_edit_project(settings)
    new_png = tmp_path / "new.png"
    Image.new("RGB", (640, 360), (30, 90, 160)).save(new_png)
    monkeypatch.setattr("vidbush.source.registry.source_shot", lambda shot, ctx, queries=None, exclude_ids=(): [
        Candidate(id=f"{shot.id}_local_9", source=SourceKind.local, local_path=str(new_png), media="image",
                  width=640, height=360, keyframes=[str(new_png)], url="local://new")])
    old = load_shotlist(work).by_id("shot_002").selected
    shot = load_shotlist(work).by_id("shot_002")
    op, _ = fixes.op_for(shot, ShotReview(shot_id="shot_002", verdict="quality_problem", rejects=["watermark_logo"]))
    res = apply_ops(settings, EPID, [op])
    assert not res.errors, res.errors
    after = load_shotlist(work).by_id("shot_002")
    assert after.selected and after.selected != old
    assert next(c for c in after.candidates if c.id == old).reject_reason == "operator review: watermark or logo"
    assert "watermark or channel logo" in after.must_avoid
