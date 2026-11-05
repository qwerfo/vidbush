"""Judge calibration: disagreement counts, suggested threshold with ties, and missed hard rejects."""
from __future__ import annotations

import pytest

from review_fixture import PID, build_review_project
from vidbush.review import calibrate, store
from vidbush.review.calibrate import calibrate as calib, errors_at, suggest_threshold
from vidbush.review.store import Review, ShotReview, ShotSnapshot


def test_errors_at():
    pts = [(80.0, True), (60.0, False), (75.0, False), (65.0, True)]
    assert errors_at(pts, 70) == 2  # 75 wrong at or above, 65 good below
    assert errors_at(pts, 76) == 1  # only 65, a good shot below
    assert errors_at(pts, 0) == 2  # both wrongs are at or above 0
    assert errors_at(pts, 101) == 2  # both goods below
    assert errors_at([(70.0, False)], 70) == 1  # at the threshold counts as passing
    assert errors_at([(70.0, True)], 70) == 0
    assert errors_at([], 50) == 0


def test_suggest_threshold_separable():
    pts = [(90.0, True), (80.0, True), (60.0, False), (50.0, False)]
    # every t in 61..80 gives zero errors; closest to the current 70 is 70 itself
    assert suggest_threshold(pts, 70) == 70.0
    # current 95 is outside the zero range: the closest zero error t is 80
    assert suggest_threshold(pts, 95) == 80.0
    assert suggest_threshold(pts, 40) == 61.0


def test_suggest_threshold_tie_goes_higher():
    pts = [(80.0, True), (60.0, False)]
    # zero errors for 61..80; current 70.5 is 0.5 from both 70 and 71, the stricter wins
    assert suggest_threshold(pts, 70.5) == 71.0


def test_suggest_threshold_moves_up_for_wrong_high_scores():
    pts = [(84.0, False), (82.0, False), (90.0, True), (95.0, True), (66.0, True)]
    assert errors_at(pts, 70) == 3
    s = suggest_threshold(pts, 70)
    assert errors_at(pts, s) == 1 and s == 85.0  # 85..90 lose only the 66 good; 85 is closest to 70


def test_suggest_threshold_without_points_returns_current():
    assert suggest_threshold([], 70) == 70.0
    assert suggest_threshold([], 63.5) == 63.5


def _e(sid: str, i: int, verdict, score, rejects=(), selected_reject=None, reason="", flags=(), notes=""):
    sn = ShotSnapshot(type="footage", start=float(i * 4), end=float(i * 4 + 4), duration=4.0, judge_score=score,
                      judge_reason=reason, selected_reject=selected_reject, flags=list(flags), judge_notes=notes)
    return ShotReview(shot_id=sid, index=i, verdict=verdict, rejects=list(rejects), snapshot=sn, updated_at=1.0)


def test_calibrate_lists_and_skips():
    r = Review(project_id="p", shots=[
        _e("a", 0, "wrong_picture", 84, reason="looks right"),
        _e("b", 1, "quality_problem", 92, reason="sharp"),
        _e("c", 2, "good", 66, reason="faces small"),
        _e("d", 3, "good", 40, reason="dark"),
        _e("e", 4, "wrong_timing", 99),  # excluded: says nothing about the judge
        _e("f", 5, "other", 10),  # excluded
        _e("g", 6, "good", None),  # unscored: skipped
        _e("h", 7, "wrong_picture", 50),  # agrees with the judge
        _e("i", 8, None, 90),  # not reviewed
        _e("j", 9, "good", 95),
    ])
    r.shots.append(_e("k", 10, "wrong_picture", 99).model_copy(update={"in_shotlist": False}))
    c = calib(r, 70)
    assert c["labelled"] == 6 and c["good"] == 3 and c["wrong"] == 3 and c["skipped_unscored"] == 1
    assert [x["shot_id"] for x in c["wrong_above_threshold"]] == ["b", "a"]  # highest score first
    assert [x["shot_id"] for x in c["good_below_threshold"]] == ["d", "c"]  # lowest first
    assert c["wrong_above_threshold"][1] == {"shot_id": "a", "number": 1, "start": 0.0, "verdict": "wrong_picture",
                                             "score": 84, "type": "footage", "reason": "looks right", "rejects": [],
                                             "comment": ""}
    assert c["errors_now"] == 4
    assert c["errors_suggested"] == errors_at([(84, False), (92, False), (66, True), (40, True), (50, False),
                                               (95, True)], c["suggested_threshold"])
    assert c["errors_suggested"] <= c["errors_now"]
    assert c["mean_good"] == round((66 + 40 + 95) / 3, 1) and c["mean_wrong"] == round((84 + 92 + 50) / 3, 1)


def test_reason_prefers_selected_reject():
    r = Review(project_id="p", shots=[_e("a", 0, "wrong_picture", 75, selected_reject="best available accepted",
                                         reason="explosion")])
    assert calib(r, 70)["wrong_above_threshold"][0]["reason"] == "best available accepted"


def test_missed_rejects():
    r = Review(project_id="p", shots=[
        # caught: the pick's own reject reason carries the judge label
        _e("a", 0, "quality_problem", 50, rejects=["burned_in_text"],
           selected_reject="burned_in_captions (prefilter: 3 of 4 keyframes)"),
        # missed: nothing on the pick says text
        _e("b", 1, "quality_problem", 80, rejects=["burned_in_text", "watermark_logo"], reason="nice canyon"),
        # caught via the judge reason and via a flag
        _e("c", 2, "quality_problem", 80, rejects=["watermark_logo"], reason="small channel_logo top right"),
        _e("d", 3, "quality_problem", 80, rejects=["anachronism"], flags=["anachronism"]),
        # missed twice over
        _e("e", 4, "other", 80, rejects=["talking_head", "watermark_logo"]),
        _e("f", 5, "good", 80, rejects=["repeated_clip"], notes="repeats a used segment"),
    ])
    c = calib(r, 70)
    rows = {m["reject"]: m for m in c["missed_rejects"]}
    assert set(rows) == {"burned_in_text", "watermark_logo", "talking_head"}
    assert rows["watermark_logo"]["missed"] == 2 and rows["watermark_logo"]["ticked"] == 3
    assert rows["watermark_logo"]["shots"] == [2, 5]
    assert rows["burned_in_text"]["missed"] == 1 and rows["burned_in_text"]["ticked"] == 2
    assert rows["burned_in_text"]["shots"] == [2]
    assert rows["talking_head"]["tune"].startswith("judge.prefilter face")
    assert [m["reject"] for m in c["missed_rejects"]] == ["watermark_logo", "burned_in_text", "talking_head"]


def test_missed_reject_without_snapshot():
    e = ShotReview(shot_id="a", verdict="quality_problem", rejects=["anachronism"])
    c = calib(Review(project_id="p", shots=[e]), 70)
    assert c["missed_rejects"][0]["reject"] == "anachronism" and c["labelled"] == 0


def test_calibration_text():
    r = Review(project_id="p", shots=[_e("a", 0, "wrong_picture", 84, reason="looks right"),
                                      _e("b", 1, "good", 66, reason="faces small"),
                                      _e("c", 2, "quality_problem", 30, rejects=["watermark_logo"])])
    c = calib(r, 70)
    text = calibrate.calibration_text(c)
    assert text.startswith("Judge calibration for p (judge.threshold 70)")
    assert "  shot 1 score 84 (wrong_picture): looks right" in text
    assert "  shot 2 score 66: faces small" in text
    assert f"Suggested judge.threshold: {c['suggested_threshold']:.0f}" in text
    assert "watermark or logo: missed 1 of 1 (shots 3)" in text
    empty = calibrate.calibration_text(calib(Review(project_id="p"), 70))
    assert "Suggested judge.threshold: 70" in empty and empty.count("  none") == 3


@pytest.fixture
def root(settings):
    return build_review_project(settings)


def test_calibrate_project_on_demo(settings, root):
    store.update_shot(settings, PID, 1, verdict="wrong_picture")  # 84
    store.update_shot(settings, PID, 2, verdict="quality_problem", rejects=["watermark_logo"])  # 72, missed
    store.update_shot(settings, PID, 3, verdict="good")  # 66
    store.update_shot(settings, PID, 6, verdict="good")  # motion graphic, unscored
    store.update_shot(settings, PID, 7, verdict="quality_problem", rejects=["repeated_clip"])  # 58
    store.update_shot(settings, PID, 8, verdict="good")  # 91
    c = calibrate.calibrate_project(settings, PID)
    assert c["threshold"] == 70.0 and c["labelled"] == 5 and c["skipped_unscored"] == 1
    assert [r["number"] for r in c["wrong_above_threshold"]] == [1, 2]
    assert [r["number"] for r in c["good_below_threshold"]] == [3]
    assert c["suggested_threshold"] == 85.0  # 85..91 only lose the good 66
    assert [(m["reject"], m["shots"]) for m in c["missed_rejects"]] == [("watermark_logo", [2]),
                                                                       ("repeated_clip", [7])]
