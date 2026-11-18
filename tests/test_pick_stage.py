"""The live judge stage hands a clause-planned project to the picker; the plan round-trips from plan_v2.json."""
from __future__ import annotations

import json
from types import SimpleNamespace

from vidbush.judge import pick_stage as PS
from vidbush.judge.resource import shots_for
from vidbush.models import ShotType


def _ctx(tmp_path, planner="clause", files=True):
    work = tmp_path / "work"
    work.mkdir(parents=True)
    if files:
        (work / "clauses.json").write_text("[]", encoding="utf-8")
        (work / "plan_v2.json").write_text("{}", encoding="utf-8")
    return SimpleNamespace(project=SimpleNamespace(paths=SimpleNamespace(work=str(work))),
                           profile=SimpleNamespace(planner=planner))


def test_clause_project_needs_profile_and_plan_files(tmp_path):
    assert PS.is_clause_project(_ctx(tmp_path))
    assert not PS.is_clause_project(_ctx(tmp_path / "b", planner="slots"))
    assert not PS.is_clause_project(_ctx(tmp_path / "c", files=False))
    assert not PS.is_clause_project(SimpleNamespace())  # a test double without a project keeps the old path


def test_scene_plans_round_trip_holds_and_planned_treatment():
    plan = {"scenes": [{"scene": "s01", "scene_purpose": "p", "shots": [
        {"clauses": ["c001", "c002"], "start": 0, "end": 5, "treatment": "picture", "subject": "a", "edit": {}},
        {"clauses": ["c003"], "treatment": "route_map", "planned_treatment": "picture", "subject": "b"}]}]}
    sp = PS.scene_plans_from(json.loads(json.dumps(plan)))
    cl = sp[0]["clauses"]
    assert [c["id"] for c in cl] == ["c001", "c002", "c003"]
    assert cl[1] == {"id": "c002", "treatment": "hold"}
    assert cl[2]["treatment"] == "picture"  # an earlier edit's fallback is not the director's plan
    assert "edit" not in cl[0] and "start" not in cl[0]


def test_resource_shots_follow_medium_and_real_thing():
    shots = shots_for({"c001": {"queries": ["a", "b", "c", "d"], "medium": "still",
                                "real_thing": {"name": "Some Bridge", "kind": "place"}},
                       "c002": {"queries": ["x"], "medium": "moving"}, "c003": {"queries": []}}, "s01")
    assert [s.id for s in shots] == ["rs_c001", "rs_c002"]
    assert shots[0].type == ShotType.image and shots[0].queries == ["a", "b", "c"]
    assert shots[0].real_thing.name == "Some Bridge" and shots[1].real_thing is None
    assert shots[1].type == ShotType.footage
