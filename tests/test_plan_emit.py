"""Part 4 of the clause-first planner: clause plans and picks become the ShotList; and the build_shotlist switch."""
import json
import random
from pathlib import Path

import pytest
from PIL import Image

import vidbush.plan.planner as PL
from vidbush.models import (Candidate, LayoutElement, LayoutKind, LayoutSpec, ProjectPaths, Shot, ShotList,
                            ShotType, SourceKind)
from vidbush.plan import clauses as C
from vidbush.plan import emit as M
from vidbush.plan.edit import in_point
from vidbush.shots.director import build_shotlist

from test_plan_helpers import FakeSeats, build, seq
from test_shots_helpers import brand, doc_script, narrate, profile, project

# (sentence, first word time): the route map and the number clauses are long enough to show a layout whole
SENTENCES = [("The ship sailed north from Seattle.", 0.0), ("It carried forty men aboard.", 3.0),
             ("They crossed the open sea for many days.", 5.5), ("Forty men reached the coast at last.", 10.0),
             ("Juneau was a small town.", 14.0), ("The town still stands today.", 16.5)]


@pytest.fixture
def setup():
    script, nar = build([[sum((seq(s, t) for s, t in SENTENCES), [])]])
    cl = C.segment(script, nar)
    assert len(cl) == 6
    plans = [
        {"id": "c001", "treatment": "picture", "subject": "ship leaving Seattle", "medium": "moving",
         "queries": {"stock": ["ship at sea"], "archive": ["Seattle harbor 1897"]}, "grammar": ["hero"]},
        {"id": "c002", "treatment": "hold", "reason": "same ship"},
        {"id": "c003", "treatment": "route_map", "presentation": "full", "subject": "route north",
         "layout": {"title": "The route north", "elements": [{"text": "Seattle", "anchor": "sea", "at": 6.0}]}},
        {"id": "c004", "treatment": "number_over_footage", "subject": "men on a beach", "medium": "still",
         "layout": {"elements": [{"text": "40 men", "anchor": "forty men", "at": 10.0}]}},
        {"id": "c005", "treatment": "picture", "subject": "Juneau in 1900", "real_thing": {"name": "Juneau",
                                                                                          "kind": "place"}},
        {"id": "c006", "treatment": "picture", "subject": "Juneau today"},
    ]
    scene_plans = [{"scene": "s01", "scene_purpose": "the voyage", "clauses": plans}]
    return script, nar, cl, scene_plans


@pytest.fixture
def cands(tmp_path):
    kf = tmp_path / "k.jpg"
    Image.new("RGB", (32, 18)).save(kf)
    return {"video": Candidate(id="vid", source=SourceKind.youtube, media="video", in_point=10.0, out_point=30.0,
                               keyframes=[str(kf)] * 3),
            "image": Candidate(id="img", source=SourceKind.wikimedia, keyframes=[str(kf)]),
            "image2": Candidate(id="img2", source=SourceKind.wikimedia, keyframes=[str(kf)]),
            "ai": Candidate(id="ai", source=SourceKind.ai_image, keyframes=[str(kf)])}


def contiguous(sl, nar):
    assert sl.shots[0].start == 0.0
    for a, b in zip(sl.shots, sl.shots[1:]):
        assert a.end == b.start
    assert sl.shots[-1].end == round(nar.duration + 1.0, 3)


def test_emit_without_edits(setup):
    script, nar, cl, sp = setup
    plan, sl = M.emit("p1", script, nar, cl, sp, None)
    assert len(sl.shots) == len(cl) - 1  # one hold
    contiguous(sl, nar)
    assert [s.id for s in sl.shots] == [f"shot_{i:03d}" for i in range(1, 6)]
    assert [s.type for s in sl.shots] == [ShotType.footage, ShotType.motion_graphic, ShotType.image,
                                          ShotType.footage, ShotType.footage]
    first = sl.shots[0]
    assert first.end == cl[1].end and first.flags == ["hero"]
    assert first.queries == ["Seattle harbor 1897", "ship at sea"]
    route = sl.shots[1]
    assert route.layout.kind == "route_map" and route.layout.presentation == "full"
    assert route.motion is None  # no title_card placeholder: the layouts renderer draws the layout
    assert route.layout.title == "The route north"
    assert route.layout.elements[0].role == "place" and route.layout.elements[0].anchor == "sea"
    number = sl.shots[2]
    assert number.layout.kind == "number_over_footage" and number.motion is None
    assert sl.shots[3].must_show == ["Juneau"]
    assert plan["planner"] == "clause" and plan["scenes"][0]["shots"][0]["clauses"] == ["c001", "c002"]


def test_emit_with_edits(setup, cands):
    script, nar, cl, sp = setup
    picks = [{"_cand": cands["video"], "frame": "b"}, {}, {"_cand": cands["image"], "frame": "a"},
             {"_cand": cands["ai"]}, {"_cand": cands["image2"], "label": "Juneau today"}]
    plan, sl = M.emit("p1", script, nar, cl, sp, [{"scene": "s01", "picks": picks}])
    assert len(sl.shots) == 5
    contiguous(sl, nar)
    v = sl.shots[0]
    assert v.type == ShotType.footage and v.selected == "vid"
    assert (v.candidates[0].in_point, v.candidates[0].out_point) == in_point(cands["video"], "b", v.duration)
    assert v.candidates[0].in_point == 20.0
    assert sl.shots[1].type == ShotType.motion_graphic and sl.shots[1].layout.kind == "route_map"
    num = sl.shots[2]
    assert num.type == ShotType.image and num.selected == "img" and num.layout.kind == "number_over_footage"
    ai = sl.shots[3]
    assert ai.type == ShotType.ai_image and ai.ai_prompt == "Juneau in 1900" and ai.selected == "ai"
    assert sl.shots[4].type == ShotType.image and "label:Juneau today" in sl.shots[4].flags
    assert plan["scenes"][0]["shots"][0]["edit"]["pick"] == "vid"


def test_hold_previous_extends_the_previous_shot(setup, cands):
    script, nar, cl, sp = setup
    picks = [{"_cand": cands["video"]}, {}, {"_cand": cands["image"]}, {"_cand": cands["image2"]},
             {"pick": None, "fallback": "hold_previous"}]
    _, sl = M.emit("p1", script, nar, cl, sp, [{"scene": "s01", "picks": picks}])
    assert len(sl.shots) == 4
    contiguous(sl, nar)
    last = sl.shots[-1]
    assert last.selected == "img2" and last.start == cl[4].start
    assert "held_over:c006" in last.flags and last.duration == round(last.end - last.start, 3)


def test_extend_next_makes_the_next_shot_start_early(setup, cands):
    script, nar, cl, sp = setup
    picks = [{"_cand": cands["video"]}, {}, {"_cand": cands["image"]},
             {"pick": None, "fallback": "extend_next"}, {"_cand": cands["image2"]}]
    plan, sl = M.emit("p1", script, nar, cl, sp, [{"scene": "s01", "picks": picks}])
    assert len(sl.shots) == 4
    contiguous(sl, nar)
    last = sl.shots[-1]
    assert last.selected == "img2" and last.start == cl[4].start and last.duration == round(last.end - last.start, 3)
    assert plan["scenes"][0]["shots"][3]["note"] == "covered by the next shot starting early"
    assert all(s.overlay is None for s in sl.shots)


def test_the_pickers_alternates_ride_on_the_shot_and_the_ai_brief_has_the_setting(setup, cands):
    script, nar, cl, sp = setup
    sp[0]["clauses"][4]["setting"] = {"period": "1900", "season": "winter", "place": "Juneau",
                                      "circumstance": "miners in the snow"}
    picks = [{"_cand": cands["video"], "_fit": (cands["image2"], "a")}, {}, {"_cand": cands["image"]},
             {"_cand": cands["ai"], "_loose": (cands["image2"], "a")}, {}]
    _, sl = M.emit("p1", script, nar, cl, sp, [{"scene": "s01", "picks": picks}])
    first, ai = sl.shots[0], sl.shots[3]
    assert first.selected == "vid" and [c.flags for c in first.candidates if c.id == "img2"] == [["alt:fitting"]]
    assert M.alternate_of(ai).id == "img2" and "alt:loose" in M.alternate_of(ai).flags
    assert ai.ai_prompt == "Juneau in 1900; miners in the snow (Juneau, 1900, winter)"


def _lay_shot(sid, start, end, kind=LayoutKind.route_map, presentation="full", picture=None, alt=None):
    typ = ShotType.motion_graphic if picture is None else ShotType.footage
    lay = LayoutSpec(kind=kind, presentation=presentation if kind == LayoutKind.route_map else None,
                     elements=[LayoutElement(role="place", text="Nome", anchor="Nome")]
                     if kind == LayoutKind.route_map else [LayoutElement(role="number", text="40 KM", anchor="forty")])
    s = Shot(id=sid, beat_id="b", scene_id="s01", start=start, end=end, duration=end - start, type=typ, intent="x",
             layout=lay)
    if picture is not None:
        s.candidates.append(picture)
        s.selected = picture.id
    if alt is not None:
        s.candidates.append(alt.model_copy(update={"flags": ["alt:fitting"]}))
    return s


def test_keep_layouts_whole_drops_a_short_layout_and_keeps_the_picture(cands):
    s = _lay_shot("a", 0.0, 5.0, LayoutKind.number_over_footage, picture=cands["video"])
    notes = M.keep_layouts_whole([s], {"a": 2.5}, {})  # 2.75 s from its first word (minus the lead) to the cut
    assert s.layout is None and s.selected == "vid" and s.type == ShotType.footage
    assert "layout_dropped" in s.flags and notes and "2.8 s" in notes[0]
    s = _lay_shot("b", 0.0, 5.0, LayoutKind.number_over_footage, picture=cands["video"])
    assert M.keep_layouts_whole([s], {"b": 2.25}, {}) == [] and s.layout is not None  # exactly 3 s stays


def test_keep_layouts_whole_drops_a_short_full_layout_to_its_alternate(cands):
    s = _lay_shot("a", 0.0, 4.0, alt=cands["image"])
    M.keep_layouts_whole([s], {"a": 2.0}, {})
    assert s.layout is None and s.selected == "img" and s.type == ShotType.image
    s = _lay_shot("b", 0.0, 4.0)  # no picture at all: left for the live fill
    M.keep_layouts_whole([s], {"b": 2.0}, {})
    assert s.layout is None and s.selected is None


def test_a_dropped_layout_round_trips(cands):
    s = _lay_shot("a", 0.0, 4.0, alt=cands["image"])
    M.keep_layouts_whole([s], {"a": 2.0}, {})
    assert Shot.model_validate_json(s.model_dump_json()).layout_at == 0.0


def test_a_full_layout_with_late_words_is_shown_over_its_alternate_from_the_words(cands):
    s = _lay_shot("a", 0.0, 6.0, alt=cands["video"])
    notes = M.keep_layouts_whole([s], {"a": 1.5}, {})  # its words come 1.25 s after the cut
    assert s.layout is not None and s.selected == "vid" and s.type == ShotType.footage
    assert s.layout_at == 1.25 and "layout_over_picture_from_words" in s.flags and notes
    s = _lay_shot("b", 0.0, 6.0)  # no alternate: the layout is dropped rather than show bare ground
    M.keep_layouts_whole([s], {"b": 1.5}, {})
    assert s.layout is None and "layout_dropped" in s.flags
    s = _lay_shot("c", 0.0, 6.0, alt=cands["video"])  # words 0.5 s after the cut: the layout carries it
    M.keep_layouts_whole([s], {"c": 0.75}, {})
    assert s.layout is not None and s.selected is None and s.layout_at == 0.0


def test_flat_queries_interleave_classes_and_dedupe():
    q = {"stock": ["s1", "s2"], "encyclopedic": ["e1"], "archive": ["a1", "s1"]}
    assert M.flat_queries(q) == ["e1", "a1", "s1", "s2"]
    many = {k: [f"{k}{i}" for i in range(3)] for k in M.CLASS_ORDER}
    assert len(M.flat_queries(many)) == M.QUERY_CAP


def test_shotlist_round_trip(setup, cands):
    script, nar, cl, sp = setup
    picks = [{"_cand": cands["video"], "frame": "c"}, {}, {"_cand": cands["image"]}, {"_cand": cands["ai"]}, {}]
    _, sl = M.emit("p1", script, nar, cl, sp, [{"scene": "s01", "picks": picks}])
    back = ShotList.model_validate_json(sl.model_dump_json())
    assert back == sl
    assert back.shots[1].layout.kind == "route_map"


def _director_reply(clauses):
    return {"scene_purpose": "test", "clauses": [
        {"id": c.id, "treatment": "picture", "subject": c.text, "anchor": c.text, "queries": {"stock": ["x"]},
         "setting": {"period": "1925", "season": "winter", "place": "Nome, Alaska", "circumstance": "x"}}
        for c in clauses]}


def test_build_shotlist_clause_planner(tmp_path):
    script = doc_script(n_scenes=3, beats_per_scene=2, seed=1)
    nar = narrate(script)
    cl = C.segment(script, nar)
    replies = [_director_reply(C.scene_clauses(cl, s.id)) for s in script.scenes]
    seats = FakeSeats(llm=replies)
    proj = project()
    root = tmp_path / "p"
    proj.paths = ProjectPaths(root=str(root), work=str(root / "work"), sources=str(root / "s"),
                              segments=str(root / "g"), thumbnails=str(root / "t"))
    sl, dplan = build_shotlist(proj, script, nar, None, profile(planner="clause"), brand(), seats,
                               random.Random(1))
    assert len(seats.llm_calls) == len(script.scenes)  # every reply valid at the first attempt
    assert len(sl.shots) == len(cl)
    contiguous(sl, nar)
    assert sl.duration == round(nar.duration + 1.0, 3)
    work = Path(proj.paths.work)
    assert C.load(work / "clauses.json") == cl
    plan = json.loads((work / "plan_v2.json").read_text(encoding="utf-8"))
    assert plan["clauses"] == len(cl) and all(not s["director_unresolved"] for s in plan["scenes"])
    assert len(dplan.ideas) == len(sl.shots)


def test_build_shotlist_default_profile_keeps_old_planner(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("clause planner called")
    monkeypatch.setattr(PL, "plan_shotlist", boom)
    script = doc_script(n_scenes=2, beats_per_scene=2)
    nar = narrate(script)
    sl, _ = build_shotlist(project(), script, nar, None, profile(), brand(), None, random.Random(1))
    assert sl.shots
