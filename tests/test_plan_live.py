"""The clause pipeline in a run: the ladder in code, and the demo end to end in mock mode with planner: clause."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vidbush.models import Shot, ShotList, ShotType
from vidbush.plan import live as L


def _shot(i, typ, dur=3.0):
    return Shot(id=f"s{i}", beat_id="b", scene_id="c", start=i * 3.0, end=i * 3.0 + dur, duration=dur, type=typ,
                intent="x")


def test_ai_rulings_are_measured_and_flagged_on_the_picked_list():
    ai, ft = ShotType.ai_image, ShotType.footage
    shots = [_shot(0, ai), _shot(1, ai), _shot(2, ai), _shot(3, ft), _shot(4, ai, 5.0)]
    sl = ShotList(project_id="p", duration=17.0, shots=shots)
    got = L.check_ai_rulings(sl)
    assert got["run_over_two"] == ["s2"] and got["over_4s"] == ["s4"] and got["real_thing"] == []
    assert "ai_ruling:run_over_two" in shots[2].flags


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
