"""The clause pipeline in a run: layouts kept whole, repeats, the empty-shot fill (never words), the AI
rulings, and the demo end to end in mock mode with planner: clause."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vidbush.models import Candidate, LayoutElement, LayoutKind, LayoutSpec, Shot, ShotList, ShotType, SourceKind
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
    assert got["opening"] == ["s0", "s1", "s2", "s4"]  # every AI shot starting inside the first 15 s
    assert "ai_ruling:run_over_two" in shots[2].flags and "ai_ruling:opening" in shots[0].flags
    assert "ai_ruling:opening" not in shots[3].flags


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


def _cand(tmp_path, cid, media="image", flags=(), **kw):
    f = tmp_path / f"{cid}.bin"
    f.write_bytes(b"x")
    return Candidate(id=cid, source=SourceKind.youtube if media == "video" else SourceKind.wikimedia, media=media,
                     local_path=str(f), url=cid, flags=list(flags), **kw)


def _showing(s, *cands, selected=True):
    s.candidates += list(cands)
    if selected:
        s.selected = cands[0].id
    return s


def _no_words(sl):
    return all(s.overlay is None and s.type not in (ShotType.footage_overlay, ShotType.image_overlay) for s in sl.shots)


def test_an_empty_shot_takes_its_own_alternate_first(tmp_path):
    a = _showing(_shot(0, ShotType.image), _cand(tmp_path, "A"))
    b = _showing(_shot(1, ShotType.footage), _cand(tmp_path, "X", flags=["alt:loose"]), selected=False)
    sl = ShotList(project_id="p", duration=6.0, shots=[a, b])
    notes = L.fill_empty_shots(sl)
    assert b.selected == "X" and b.type == ShotType.image and "filled:alternate" in b.flags and len(notes) == 1
    assert _no_words(sl)


def test_an_empty_shot_is_held_by_the_previous_picture_as_one_shot(tmp_path):
    a = _showing(_shot(0, ShotType.image), _cand(tmp_path, "A"))
    b = _showing(_shot(1, ShotType.footage), _cand(tmp_path, "U"), selected=False)  # an unchosen candidate is not used
    sl = ShotList(project_id="p", duration=6.0, shots=[a, b])
    L.fill_empty_shots(sl)
    assert sl.shots == [a] and a.end == 6.0 and a.duration == 6.0 and "held_over:s1" in a.flags
    assert _no_words(sl)


def test_an_alternate_shown_within_a_minute_is_not_used_by_the_fill(tmp_path):
    a = _showing(_shot(0, ShotType.image), _cand(tmp_path, "A"))
    b = _showing(_shot(1, ShotType.footage), _cand(tmp_path, "A", flags=["alt:fitting"]), selected=False)
    sl = ShotList(project_id="p", duration=6.0, shots=[a, b])
    L.fill_empty_shots(sl)
    assert sl.shots == [a] and "held_over:s1" in a.flags


def test_an_empty_shot_after_a_layout_is_covered_by_the_next_picture_starting_early(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "_file_s", lambda c: 8.0)
    lay = _shot(0, ShotType.motion_graphic)
    lay.layout = LayoutSpec(kind=LayoutKind.chart, presentation="full")
    b = _shot(1, ShotType.footage)
    c = _showing(_shot(2, ShotType.footage), _cand(tmp_path, "V", "video", in_point=5.0, out_point=8.0))
    sl = ShotList(project_id="p", duration=9.0, shots=[lay, b, c])
    L.fill_empty_shots(sl)
    assert sl.shots == [lay, c] and c.start == 3.0 and c.duration == 6.0 and "started_early_over:s1" in c.flags
    assert c.selected_candidate().in_point == 2.0  # 3 s earlier in the same clip
    assert _no_words(sl)


def test_an_empty_shot_with_nothing_to_hold_is_flagged_and_never_given_words(tmp_path):
    s = _showing(_shot(0, ShotType.footage), _cand(tmp_path, "U"), selected=False)
    sl = ShotList(project_id="p", duration=3.0, shots=[s])
    notes = L.fill_empty_shots(sl)
    assert s.selected is None and "filled:nothing" in s.flags and notes
    assert _no_words(sl)


def _split(tmp_path, right_picture):
    s = _showing(_shot(0, ShotType.motion_graphic), _cand(tmp_path, "L"), _cand(tmp_path, "R"),
                 _cand(tmp_path, "F", flags=["alt:fitting"]), selected=False)
    s.layout = LayoutSpec(kind=LayoutKind.split_comparison, elements=[
        LayoutElement(role="left", text="then", picture="L"),
        LayoutElement(role="right", text="now", picture="R" if right_picture else None)])
    return s


def test_a_one_sided_split_comparison_is_dropped_to_the_alternate(tmp_path):
    s = _split(tmp_path, right_picture=False)
    sl = ShotList(project_id="p", duration=3.0, shots=[s])
    notes = L.drop_incomplete_layouts(sl)
    assert s.layout is None and s.selected == "F" and s.type == ShotType.image and "layout_dropped" in s.flags
    assert notes and "missing real pictures" in notes[0]
    whole = _split(tmp_path, right_picture=True)
    assert L.drop_incomplete_layouts(ShotList(project_id="p", duration=3.0, shots=[whole])) == []
    assert whole.layout is not None


@pytest.mark.parametrize("elements, dropped", [
    ([("place", "Juneau, Alaska"), ("place", "Atlantis")], True),
    ([("route", "Juneau to Atlantis by sea")], True),
    ([], True),
    ([("place", "Juneau, Alaska"), ("route", "Juneau to Nome by sea")], False),
])
def test_a_route_map_with_a_place_off_the_map_is_dropped(tmp_path, monkeypatch, elements, dropped):
    from vidbush.judge import ladder
    monkeypatch.setattr(ladder, "place_resolves", lambda name: name.strip() != "Atlantis")
    s = _showing(_shot(0, ShotType.footage), _cand(tmp_path, "P"))
    s.layout = LayoutSpec(kind=LayoutKind.route_map, presentation="panel",
                          elements=[LayoutElement(role=r, text=t) for r, t in elements])
    L.drop_incomplete_layouts(ShotList(project_id="p", duration=3.0, shots=[s]))
    assert (s.layout is None) is dropped
    assert s.selected == "P"  # a panel's picture beneath stays either way


def test_a_document_without_its_page_is_never_drawn_and_the_shot_keeps_its_picture(tmp_path):
    s = _showing(_shot(0, ShotType.footage), _cand(tmp_path, "P", "video", in_point=0.0, out_point=8.0))
    s.layout = LayoutSpec(kind=LayoutKind.document_focus, presentation="panel",
                          elements=[LayoutElement(role="mark", text="an apology", anchor="an apology")])
    sl = ShotList(project_id="p", duration=3.0, shots=[s])
    assert L.drop_pictureless_layouts is L.drop_incomplete_layouts and L.drop_incomplete_layouts(sl)
    assert s.layout is None and s.selected == "P" and s.type == ShotType.footage and _no_words(sl)


def _run_of(tmp_path, n, dur=3.0, alt=None):
    shots = []
    for i in range(n):
        s = Shot(id=f"s{i}", beat_id="b", scene_id="c", start=i * dur, end=(i + 1) * dur, duration=dur,
                 type=ShotType.image, intent="x")
        _showing(s, _cand(tmp_path, "A"))
        shots.append(s)
    if alt is not None:
        shots[-1].candidates.append(_cand(tmp_path, alt, flags=["alt:fitting"]))
    return ShotList(project_id="p", duration=n * dur, shots=shots)


def test_a_third_consecutive_shot_of_one_asset_takes_its_alternate(tmp_path):
    sl = _run_of(tmp_path, 3, alt="F")
    notes = L.enforce_repeats(sl)
    assert [s.selected for s in sl.shots] == ["A", "A", "F"] and "repeat:alternate" in sl.shots[2].flags
    assert notes and "a third shot in a row" in notes[0]


def test_a_third_consecutive_shot_without_an_alternate_merges_into_the_previous(tmp_path):
    sl = _run_of(tmp_path, 3)
    L.enforce_repeats(sl)
    assert [s.id for s in sl.shots] == ["s0", "s1"] and sl.shots[1].end == 9.0 and "held_over:s2" in sl.shots[1].flags
    L.renumber(sl)
    assert [(s.id, s.index) for s in sl.shots] == [("shot_001", 0), ("shot_002", 1)]


def test_a_third_consecutive_shot_over_the_hold_limit_loses_its_picture(tmp_path):
    sl = _run_of(tmp_path, 3, dur=5.0)  # the second and third together would hold 10 s
    L.enforce_repeats(sl)
    assert len(sl.shots) == 3 and sl.shots[2].selected is None and "repeat:removed" in sl.shots[2].flags


def test_an_asset_again_within_a_minute_is_refused(tmp_path):
    a = _showing(_shot(0, ShotType.image), _cand(tmp_path, "A"))
    b = _showing(_shot(1, ShotType.image), _cand(tmp_path, "B"))
    soon = _showing(_shot(10, ShotType.image), _cand(tmp_path, "A"))  # 30 s, 27 s after A left
    late = _showing(_shot(25, ShotType.image), _cand(tmp_path, "A"))  # 75 s
    sl = ShotList(project_id="p", duration=78.0, shots=[a, b, soon, late])
    notes = L.enforce_repeats(sl)
    assert soon.selected is None and "repeat:removed" in soon.flags and "again within a minute" in notes[0]
    assert late.selected == "A" and len(notes) == 1


def test_pick_clause_drops_repeats_fills_and_renumbers(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from vidbush.judge import pick_stage
    a = _showing(_shot(0, ShotType.image), _cand(tmp_path, "A"))
    b = _shot(1, ShotType.footage)
    ai = _showing(_shot(2, ShotType.ai_image), Candidate(id="G", source=SourceKind.ai_image, local_path="g.png"))
    for name in (L.CLAUSES_FILE, L.PLAN_FILE):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pick_stage, "pick_shotlist",
                        lambda sourced, ctx: ShotList(project_id="p", duration=9.0, shots=[a, b, ai]))
    warns = []
    ctx = SimpleNamespace(work=str(tmp_path), log=SimpleNamespace(info=lambda *x, **k: None,
                                                                  warn=lambda msg, **k: warns.append(k)))
    sl = L.pick_clause(ShotList(project_id="p", duration=9.0, shots=[]), ctx)
    assert [s.id for s in sl.shots] == ["shot_001", "shot_002"] and sl.shots[0].end == 6.0
    plan = json.loads((tmp_path / L.PLAN_FILE).read_text(encoding="utf-8"))
    assert plan["ai_rulings"]["opening"] == ["shot_002"] and warns == [{"opening": ["shot_002"]}]


def test_a_panel_with_no_picture_beneath_is_shown_full_screen(tmp_path):
    s = _shot(0, ShotType.footage)
    s.layout = LayoutSpec(kind=LayoutKind.chart, presentation="panel")
    sl = ShotList(project_id="p", duration=3.0, shots=[s])
    L.fill_empty_shots(sl)
    assert s.layout.presentation == "full" and s.type == ShotType.motion_graphic and "filled:nothing" not in s.flags


def test_a_number_over_footage_with_no_footage_is_dropped_and_the_shot_filled(tmp_path):
    a = _showing(_shot(0, ShotType.image), _cand(tmp_path, "A"))
    b = _shot(1, ShotType.footage)
    b.layout = LayoutSpec(kind=LayoutKind.number_over_footage)
    sl = ShotList(project_id="p", duration=6.0, shots=[a, b])
    L.fill_empty_shots(sl)
    assert sl.shots == [a] and a.end == 6.0 and _no_words(sl)


def test_an_empty_shot_too_long_for_one_neighbour_is_split_between_both(tmp_path):
    a = _showing(_shot(0, ShotType.image, 6.0), _cand(tmp_path, "A"))
    b = _shot(2, ShotType.footage, 6.0)
    c = _showing(_shot(4, ShotType.image, 6.0), _cand(tmp_path, "C"))
    b.start, b.end = 6.0, 12.0
    c.start, c.end = 12.0, 18.0
    sl = ShotList(project_id="p", duration=18.0, shots=[a, b, c])
    L.fill_empty_shots(sl)
    assert sl.shots == [a, c] and a.end == c.start == 9.0 and a.duration == 9.0 and c.duration == 9.0


def test_a_next_clip_with_no_earlier_frames_plays_longer_from_its_start(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "_file_s", lambda c: 20.0)
    lay = _shot(0, ShotType.motion_graphic)
    lay.layout = LayoutSpec(kind=LayoutKind.chart, presentation="full")
    b = _shot(1, ShotType.footage)
    c = _showing(_shot(2, ShotType.footage), _cand(tmp_path, "V", "video", in_point=0.0, out_point=3.0))
    sl = ShotList(project_id="p", duration=9.0, shots=[lay, b, c])
    L.fill_empty_shots(sl)
    v = c.selected_candidate()
    assert sl.shots == [lay, c] and c.start == 3.0 and (v.in_point, v.out_point) == (0.0, 6.0)


def _rec(place, period, look="black_and_white", shows="a street"):
    return {"visual": {"place": place, "period": period, "look": look, "shows": shows, "quality": []}}


def test_an_empty_shot_takes_a_real_picture_of_the_same_place_and_period_from_the_film(tmp_path):
    s = _shot(0, ShotType.footage)
    s.flags.append("setting:Anchorage, Alaska|1958|summer")
    film = [(_cand(tmp_path, "page"), _rec("Anchorage newsroom", "1958", look="document")),
            (_cand(tmp_path, "modern"), _rec("Anchorage downtown", "Contemporary", look="modern_digital")),
            (_cand(tmp_path, "texas"), _rec("Austin, Texas", "1950s")),
            (_cand(tmp_path, "old"), _rec("A street in Anchorage, Alaska", "1950s, from the cars"))]
    sl = ShotList(project_id="p", duration=3.0, shots=[s])
    L.fill_empty_shots(sl, film=film)
    assert s.selected == "old" and "filled:setting" in s.flags and _no_words(sl)


def test_the_setting_fill_never_guesses_without_a_period(tmp_path):
    s = _shot(0, ShotType.footage)
    s.flags.append("setting:Anchorage, Alaska|1958|summer")
    sl = ShotList(project_id="p", duration=3.0, shots=[s])
    L.fill_empty_shots(sl, film=[(_cand(tmp_path, "x"), _rec("Anchorage, Alaska", ""))])
    assert s.selected is None and "filled:nothing" in s.flags


def test_periods_agree_within_ten_years_or_both_present_day():
    assert L.period_agrees("1958", "1950s") and L.period_agrees("present day", "Contemporary")
    assert not L.period_agrees("1958", "Contemporary") and not L.period_agrees("1958", "1920s")
