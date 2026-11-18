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


def test_an_empty_shot_holds_the_previous_picture_or_takes_a_related_one_with_its_words(tmp_path):
    from vidbush.models import Candidate, SourceKind
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    vid = Candidate(id="v", source=SourceKind.youtube, media="video", in_point=0.0, out_point=8.0,
                    local_path=str(f), url="u1")
    other = Candidate(id="o", source=SourceKind.wikimedia, media="image", local_path=str(f), url="u2", score=80)
    a, b, c = _shot(0, ShotType.footage), _shot(1, ShotType.footage), _shot(2, ShotType.footage, 6.0)
    a.candidates, a.selected = [vid], "v"
    pool = ShotList(project_id="p", duration=12.0, shots=[a.model_copy(), b.model_copy(update={"candidates": [other]}),
                                                         c.model_copy()])
    sl = ShotList(project_id="p", duration=12.0, shots=[a, b, c])
    notes = L.fill_empty_shots(sl, lambda x, y: "the town was small", pool)
    assert b.selected == "v" and "filled:hold_previous" in b.flags  # 3 + 3 s inside the 8 s clip
    assert c.selected == "o" and c.type == ShotType.image_overlay and c.overlay.payload["title"] == "the town was small"
    assert len(notes) == 2


def test_a_document_without_its_page_becomes_its_words_over_related_footage(tmp_path):
    from vidbush.models import Candidate, LayoutElement, LayoutKind, LayoutSpec, SourceKind
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    rel = Candidate(id="r", source=SourceKind.youtube, media="video", in_point=0.0, out_point=8.0,
                    local_path=str(f), url="u", score=60)
    doc = _shot(0, ShotType.motion_graphic)
    doc.layout = LayoutSpec(kind=LayoutKind.document_focus, presentation="full",
                            elements=[LayoutElement(role="mark", text="an apology", anchor="an apology")])
    sl = ShotList(project_id="p", duration=3.0, shots=[doc])
    pool = ShotList(project_id="p", duration=3.0, shots=[doc.model_copy(update={"candidates": [rel], "layout": None})])
    assert L.drop_pageless_documents(sl)
    L.fill_empty_shots(sl, lambda a, b: "printed an apology", pool)
    assert doc.layout is None and doc.selected == "r" and doc.overlay.payload["title"] == "printed an apology"


def test_fill_uses_only_clean_described_pictures_ranked_by_the_words(tmp_path):
    from vidbush.models import Candidate, SourceKind
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    mk = lambda cid, sc: Candidate(id=cid, source=SourceKind.archive, media="video", in_point=0.0, out_point=8.0,
                                   local_path=str(f), url=cid, score=sc)
    slate, beach, plain = mk("slate", 90), mk("beach", 40), mk("plain", 70)
    recs = {"slate": {"visual": {"shows": "film leader", "text_in_frame": "END 1", "usable": True}},
            "beach": {"visual": {"shows": "soldiers waiting on the beach", "text_in_frame": "", "usable": True}},
            "plain": {"visual": {"shows": "a field of grass", "text_in_frame": "", "usable": True}}}
    s = _shot(0, ShotType.footage)
    pool = ShotList(project_id="p", duration=3.0, shots=[s.model_copy(update={"candidates": [slate, plain, beach]})])
    sl = ShotList(project_id="p", duration=3.0, shots=[s])
    L.fill_empty_shots(sl, lambda a, b: "men waited on the beach", pool, lambda c: recs[c.id])
    assert s.selected == "beach"
    assert not L.clean_record(None) and not L.clean_record(recs["slate"])
