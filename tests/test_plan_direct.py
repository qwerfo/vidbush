"""Part 2 of the clause-first planner: the director reply is validated, retried with its problems, then repaired."""
import copy
import json

import pytest

from vidbush.plan import direct as D
from vidbush.plan import vocab as V
from vidbush.plan.clauses import segment

from test_plan_helpers import FakeSeats, build, seq


@pytest.fixture
def scene():
    script, nar = build([[seq("The ship sailed north from Seattle.", 0) + seq("It carried forty men to Alaska.", 3.0)
                          + seq("In nineteen fifty-nine Alaska became a state.", 6.0)]])
    cl = segment(script, nar)
    assert [c.id for c in cl] == ["c001", "c002", "c003"]
    return script, nar, cl


ANCHORS = {"c001": "ship", "c002": "forty men", "c003": "Alaska became a state"}


def valid() -> dict:
    return {"scene_purpose": "the voyage north",
            "clauses": [{"id": cid, "treatment": "picture", "subject": f"subject {cid}", "anchor": a,
                         "queries": {"stock": ["ship at sea"]}} for cid, a in ANCHORS.items()]}


def by_id(reply: dict, cid: str) -> dict:
    return next(p for p in reply["clauses"] if p["id"] == cid)


def test_valid_reply_has_no_errors(scene):
    script, nar, cl = scene
    plans, errors = D.validate(valid(), cl, nar)
    assert errors == []
    assert [p["treatment"] for p in plans] == ["picture"] * 3


def test_number_with_unit_and_year_alone_are_accepted(scene):
    script, nar, cl = scene
    r = valid()
    by_id(r, "c002").update(treatment="number_over_footage",
                            layout={"elements": [{"text": "40 men", "anchor": "forty men"}]})
    by_id(r, "c003").update(treatment="date_title", layout={"elements": [{"text": "1959", "anchor": "1959"}]})
    plans, errors = D.validate(r, cl, nar)
    assert errors == []
    assert plans[1]["layout"]["elements"][0]["at"] == 4.0  # "forty" is spoken at 4.0 s
    assert plans[2]["layout"]["elements"][0]["at"] == 6.5


def _missing(r):
    r["clauses"] = [p for p in r["clauses"] if p["id"] != "c002"]


def _unknown(r):
    by_id(r, "c002")["treatment"] = "dance"


def _hold_first(r):
    by_id(r, "c001").update(treatment="hold", reason="same ship")


def _bad_anchor(r):
    by_id(r, "c001")["anchor"] = "penguins"


def _element_not_spoken(r):
    by_id(r, "c002").update(treatment="number_over_footage",
                            layout={"elements": [{"text": "40 men", "anchor": "Seattle"}]})


def _no_unit(r):
    by_id(r, "c002").update(treatment="number_over_footage",
                            layout={"elements": [{"text": "40", "anchor": "forty men"}]})


def _long_hold(r):
    by_id(r, "c002").update(treatment="hold", reason="same ship")
    by_id(r, "c003").update(treatment="hold", reason="same ship")


def _route_no_presentation(r):
    by_id(r, "c001").update(treatment="route_map", layout={"elements": [{"text": "Seattle", "anchor": "ship"}]})


@pytest.mark.parametrize("mutate,problem", [
    (_missing, "These clauses are missing: c002"),
    (_unknown, "treatment 'dance' is not one of"),
    (_hold_first, "the first clause of a scene cannot be a hold"),
    (_bad_anchor, "anchor 'penguins' is not words of this clause"),
    (_element_not_spoken, "anchor 'Seattle' is not spoken in this shot"),
    (_no_unit, "the number '40' needs its unit"),
    (_long_hold, f"over the {V.HOLD_MAX_S:g} s limit"),
    (_route_no_presentation, "route_map needs presentation"),
])
def test_each_problem_is_reported(scene, mutate, problem):
    script, nar, cl = scene
    r = valid()
    mutate(r)
    _, errors = D.validate(r, cl, nar)
    assert any(problem in e for e in errors), errors


def run(script, nar, cl, replies):
    seats = FakeSeats(llm=replies)
    out = D.direct_scene(seats, script, script.scenes[0], cl, nar, [], 0, [])
    return seats, out


def test_valid_reply_one_call(scene):
    script, nar, cl = scene
    seats, out = run(script, nar, cl, [valid()])
    assert len(seats.llm_calls) == 1
    assert seats.llm_calls[0]["role"] == D.SEAT
    assert out["unresolved"] == [] and out["scene_purpose"] == "the voyage north"


def test_retry_carries_problems_and_previous_reply(scene):
    script, nar, cl = scene
    bad = valid()
    _missing(bad)
    seats, out = run(script, nar, cl, [copy.deepcopy(bad), valid(), valid()])
    assert len(seats.llm_calls) == 2  # stops at the first valid reply
    first, retry = seats.llm_calls[0]["user"], seats.llm_calls[1]["user"]
    assert retry.startswith(first)
    assert "These clauses are missing: c002" in retry
    assert json.dumps(bad, ensure_ascii=False) in retry
    assert out["unresolved"] == [] and len(out["attempts"]) == 2


def test_repair_after_retries_missing_becomes_hold_and_long_hold_splits(scene):
    script, nar, cl = scene
    bad = valid()
    _missing(bad)
    by_id(bad, "c003").update(treatment="hold", reason="same ship")
    seats, out = run(script, nar, cl, [copy.deepcopy(bad) for _ in range(D.RETRIES + 1)])
    assert len(seats.llm_calls) == D.RETRIES + 1
    assert out["unresolved"]
    plans = out["clauses"]
    assert [p["treatment"] for p in plans] == ["picture", "hold", "picture"]
    assert "missing" in plans[1]["repairs"]
    assert plans[2]["grammar"] == ["sequence"]
    assert plans[2]["subject"] == "subject c001"
    assert "hold over the limit split into a new picture" in plans[2]["repairs"]
    _, errors = D.validate({"scene_purpose": "x", "clauses": plans}, cl, nar)
    assert errors == []


def test_repair_missing_first_clause_becomes_picture(scene):
    script, nar, cl = scene
    bad = valid()
    bad["clauses"] = bad["clauses"][1:]
    plans, _ = D.validate(bad, cl, nar)
    fixed = D.repair(plans, cl, nar)
    assert fixed[0]["treatment"] == "picture"
    assert fixed[0]["subject"] == cl[0].text
    assert fixed[0]["anchor"] == cl[0].text
