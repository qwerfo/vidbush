"""Chat revision mapping: the acceptance phrase, common phrasings, seat output validation."""
from __future__ import annotations

import pytest

from test_editor_helpers import PID, build_edit_project
from vidbush.editor.revise import map_message, rule_ops, split_clauses
from vidbush.models import EditOpName
from vidbush.providers.seats import Seats


@pytest.fixture
def project(settings):
    return build_edit_project(settings)


def mapped(settings, message: str) -> list[tuple[str, dict]]:
    return [(o.op.value, o.args) for o in map_message(settings, PID, message)]


def test_acceptance_phrase(settings, project):
    ops = map_message(settings, PID, "make the first 30 seconds faster and replace shot 7 with an AI image")
    assert [o.op for o in ops] == [EditOpName.retime_section, EditOpName.change_type]
    assert ops[0].args["range"] == [0.0, 30.0] and abs(ops[0].args["factor"] - 0.7) < 1e-9
    assert ops[1].args == {"shot_id": "shot_007", "type": "ai_image"}


@pytest.mark.parametrize("message,expected", [
    ("make shot 7 an ai image", [("change_type", {"shot_id": "shot_007", "type": "ai_image"})]),
    ("swap shot 2", [("swap_shot", {"shot_id": "shot_002", "candidate_id": "shot_002_wikimedia_2"})]),
    ("use the next option for shot 3", [("swap_shot", {"shot_id": "shot_003", "candidate_id": "shot_003_local_2"})]),
    ("remove the overlay on shot 4", [("remove_overlay", {"shot_id": "shot_004"})]),
    ("make shot 5 6 seconds long", [("set_duration", {"shot_id": "shot_005", "seconds": 6.0})]),
    ("split shot 4 at 2 seconds", [("split_shot", {"shot_id": "shot_004", "at": 2.0})]),
    ("merge shots 4 and 5", [("merge_shots", {"a": "shot_004", "b": "shot_005"})]),
    ("move shot 3 to position 6", [("reorder", {"shot_id": "shot_003", "new_index": 5})]),
    ("use calmer music", [("change_music", {"track": "harbor_light.ogg"})]),
    ("change the music to Undertow", [("change_music", {"track": "undertow.ogg"})]),
    ("bigger captions", [("set_caption_style", {"delta": {"size": 77}})]),
    ("captions in yellow", [("set_caption_style", {"delta": {"color": "yellow"}})]),
    ("make the intro punchier", [("rewrite_beat", {"beat_id": "b001", "instruction": "make the intro punchier"})]),
    ("slow down the last 10 seconds", [("retime_section", {"factor": 1.4, "range": [20.0, 30.0]})]),
    ("speed up scene 2 a lot", [("retime_section", {"factor": 0.55, "scene_id": "s02"})]),
])
def test_phrasings(settings, project, message, expected):
    assert mapped(settings, message) == expected


def test_multiple_requests_and_joined_fragments(settings, project):
    got = mapped(settings, "captions in yellow and bigger, merge shots 1 and 2, then make shot 8 footage")
    assert got[0] == ("set_caption_style", {"delta": {"size": 77, "color": "yellow"}})
    assert got[1] == ("merge_shots", {"a": "shot_001", "b": "shot_002"})
    assert got[2] == ("change_type", {"shot_id": "shot_008", "type": "footage"})


def test_map_and_text_overlay(settings, project):
    got = mapped(settings, "add a map to shot 6")
    assert got[0][0] == "change_type" and got[0][1]["type"] == "motion_graphic" and got[0][1]["kind"] == "map"
    got = mapped(settings, 'add a lower third saying "Ferdinand de Lesseps" to shot 2')
    spec = got[0][1]["motion_spec"]
    assert got[0][0] == "set_overlay" and spec["kind"] == "lower_third"
    assert spec["payload"]["title"] == "Ferdinand de Lesseps"


def test_regenerate_and_swap_positions(settings, project):
    assert mapped(settings, "regenerate shot 2 with more dramatic lighting") == [
        ("regenerate_ai", {"shot_id": "shot_002", "prompt_delta": "more dramatic lighting"})]
    assert mapped(settings, "swap shots 2 and 6") == [("reorder", {"shot_id": "shot_002", "new_index": 5}),
                                                      ("reorder", {"shot_id": "shot_006", "new_index": 1})]


def test_unknown_message_maps_to_nothing(settings, project):
    assert mapped(settings, "hello there") == []
    assert mapped(settings, "swap shot 40") == []


def test_split_clauses():
    assert split_clauses("a and b, c; then d.") == ["a", "b", "c", "d"]


def test_seat_ops_are_validated_and_resolved(settings, project, monkeypatch):
    reply = {"ops": [{"op": "swap_shot", "args": {"shot_id": 2, "candidate_id": "next"}},
                     {"op": "explode", "args": {}},
                     {"op": "set_duration", "args": {"shot_id": "shot_003"}},
                     {"op": "retime_section", "args": {"range": [0, 10], "factor": 0.7}}]}
    seen = {}

    def llm_json(self, role, task, system, user, stage, temperature=0.6):
        seen.update(role=role, task=task, stage=stage, user=user)
        return reply

    monkeypatch.setattr(Seats, "llm_json", llm_json)
    ops = map_message(settings, PID, "whatever")
    assert seen["role"] == "editor" and seen["task"] == "revise" and seen["stage"] == "revise"
    assert "shot_007" in seen["user"] and "retime_section" in seen["user"] and "s02" in seen["user"]
    assert [o.op.value for o in ops] == ["swap_shot", "retime_section"]
    assert ops[0].args["shot_id"] == "shot_002"


def test_invalid_seat_reply_falls_back_to_rules(settings, project, monkeypatch):
    monkeypatch.setattr(Seats, "llm_json", lambda self, *a, **k: {"ops": [{"op": "nope"}]})
    assert mapped(settings, "remove the overlay on shot 3") == [("remove_overlay", {"shot_id": "shot_003"})]


def test_rule_ops_keeps_ids_from_before_edits(settings, project):
    from vidbush.editor.revise import RuleContext
    from test_editor_helpers import load_shotlist
    rc = RuleContext(shotlist=load_shotlist(project), script=None)
    ops = rule_ops("split shot 2 and make shot 3 an ai image", rc)
    assert ops[0].args["shot_id"] == "shot_002" and ops[1].args["shot_id"] == "shot_003"
