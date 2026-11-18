"""The clause pipeline in a run: the ladder in code, and the demo end to end in mock mode with planner: clause."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vidbush.models import Candidate, SourceKind
from vidbush.plan import live as L
from vidbush.plan.clauses import Clause


def _clause(cid, start, end):
    return Clause(id=cid, scene="s01", beat="b1", start=start, end=end, words=(0, 1), text="x")


def _cand(cid, src=SourceKind.youtube):
    return Candidate(id=cid, source=src, media="video")


def test_ladder_order_map_then_text_then_ai_and_never_three_ai_in_a_row():
    cl = [_clause(f"c{i}", i * 3.0, i * 3.0 + 3.0) for i in range(5)]
    plans = [{"treatment": "picture", "real_thing": {"name": "Juneau", "kind": "place"}},  # map
             {"treatment": "picture"},  # text over related footage
             {"treatment": "picture"},  # AI still
             {"treatment": "picture"},  # AI still
             {"treatment": "picture"}]  # a third AI in a row is refused: words alone
    picks = [{"clause": "c0", "_cand": None, "fallback": "ai_still"},
             {"clause": "c1", "_cand": None, "fallback": "ai_still", "_fallback_cand": _cand("rel")},
             {"clause": "c2", "_cand": None, "fallback": "text_over_footage"},
             {"clause": "c3", "_cand": None, "fallback": "ai_still"},
             {"clause": "c4", "_cand": None, "fallback": "ai_still"}]
    edits = [{"scene": "s01", "picks": picks}]
    L.ladder(edits, [{"scene": "s01", "clauses": plans}], cl)
    assert [p["fallback"] for p in picks] == ["map", "text_over_footage", "ai_still", "ai_still",
                                              "text_over_footage"]
    assert picks[0]["map_place"] == "Juneau" and "ladder" in picks[0]["resolved"]


def test_ladder_never_ai_for_a_real_person_or_over_four_seconds():
    cl = [_clause("c0", 0.0, 3.0), _clause("c1", 3.0, 9.0)]
    plans = [{"treatment": "picture", "real_thing": {"name": "A Person", "kind": "person"}},
             {"treatment": "picture"}]
    picks = [{"clause": "c0", "_cand": None, "fallback": "ai_still"},
             {"clause": "c1", "_cand": None, "fallback": "ai_still"}]
    L.ladder([{"scene": "s01", "picks": picks}], [{"scene": "s01", "clauses": plans}], cl)
    assert [p["fallback"] for p in picks] == ["text_over_footage", "text_over_footage"]


@pytest.mark.slow
def test_demo_end_to_end_with_the_clause_planner(settings):
    from vidbush.cli import detect_input
    from vidbush.config import REPO_ROOT
    from vidbush.jobs.pipeline import create_project, run_project
    mode, kw = detect_input(REPO_ROOT / "fixtures" / "demo_brief.md", None)
    proj = create_project(settings, "democlause", mode, seed=11, **kw)
    work = Path(proj.paths.work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "profile_override.json").write_text(json.dumps({"planner": "clause"}), encoding="utf-8")
    proj = run_project(settings, "democlause", echo=False)
    assert proj.status in ("done", "needs_review"), proj.status_note
    root = Path(settings.output_folder) / "democlause"
    assert (root / "final.mp4").exists()
    for name in ("clauses.json", "direct.json", "plan_v2.json", "sourcing.shotlist.json", "judge.shotlist.json"):
        assert (root / "work" / name).exists(), name
