"""Judge verdict parsing: none-like rejects, label mapping and score scales."""
from __future__ import annotations

import pytest

from vidbush.judge.judge import HARD_REJECTS, JUDGE_SYSTEM, normalize_rejects, parse_verdict
from vidbush.judge.qa import QA_REJECTS, QA_SYSTEM


@pytest.mark.parametrize("value", [["none"], ["N/A"], ["n/a"], ["No"], ["null"], ["-"], "none", [""], []])
def test_none_like_rejects_are_dropped(value) -> None:
    score, _reason, rejects = parse_verdict({"score": 82, "reason": "fine", "rejects": value})
    assert score == 82 and rejects == []


def test_reject_wording_maps_to_labels() -> None:
    labels, notes = normalize_rejects(["Burned-in subtitles", "big watermark", "TV channel logo", "talking head",
                                       "modern cars visible", "wrong subject", "slightly blurry"])
    assert labels == ["burned_in_captions", "watermark", "channel_logo", "talking_head", "anachronism",
                      "wrong_subject"]
    assert notes == ["slightly blurry"]


def test_unknown_labels_go_to_reason_not_rejects() -> None:
    score, reason, rejects = parse_verdict({"score": 75, "reason": "ok", "rejects": ["a bit dark"]})
    assert rejects == [] and "a bit dark" in reason


def test_scores_are_not_inflated() -> None:
    assert parse_verdict({"score": 5})[0] == 5.0
    assert parse_verdict({"score": 10})[0] == 10.0
    assert parse_verdict({"score": 0.85})[0] == pytest.approx(85.0)
    assert parse_verdict({"score": 140})[0] == 100.0


def test_prompts_name_the_labels_and_empty_list() -> None:
    for label in HARD_REJECTS:
        assert label in JUDGE_SYSTEM
    for label in QA_REJECTS:
        assert label in QA_SYSTEM
    assert "[] when no hard reject applies" in JUDGE_SYSTEM and "[] when no hard reject applies" in QA_SYSTEM


def test_qa_labels_include_frame_problems() -> None:
    _s, _r, rejects = parse_verdict({"score": 20, "rejects": ["black frame", "foreign subtitles"]}, QA_REJECTS)
    assert rejects == ["black_frame", "burned_in_captions"]
