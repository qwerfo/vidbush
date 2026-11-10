"""Edit ops on a synthetic project: every op, timeline invariants, force sets, logs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from test_editor_helpers import PID, TOTAL, assert_timeline, build_edit_project, load_shotlist, make_video
from vidbush.editor import ops as ops_mod
from vidbush.editor.ops import apply_and_render, apply_ops, heuristic_rewrite
from vidbush.models import Candidate, EditOp, EditOpName, MotionKind, Narration, Script, ShotType, SourceKind
from vidbush.util import read_json, read_model


def op(name: str, **args) -> EditOp:
    return EditOp(op=EditOpName(name), args=args)


@pytest.fixture
def work(settings) -> Path:
    return build_edit_project(settings)


@pytest.fixture
def fake_ai(monkeypatch):
    """generate_ai_candidate stand-in writing a real PNG (another engineer owns the real one)."""
    calls: list = []

    def gen(shot, ctx, prompt=None, problem=""):
        n = sum(1 for c in shot.candidates if c.source == SourceKind.ai_image) + 1
        p = ctx.work / "ai" / f"{shot.id}_{n}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1920, 1080), (40, 60, 90)).save(p)
        calls.append(prompt)
        return Candidate(id=f"{shot.id}_ai_image_{n}", source=SourceKind.ai_image, local_path=str(p), media="image",
                         width=1920, height=1080, licence="AI generated", keyframes=[str(p)], flags=["mock"])

    monkeypatch.setattr("vidbush.visuals.ai_images.generate_ai_candidate", gen)
    return calls


def plan(work: Path) -> list[str]:
    return read_json(work / "edit_plan.json")["force"]


def run(settings, *edit_ops: EditOp):
    return apply_ops(settings, PID, list(edit_ops))


# ---------------------------------------------------------------- candidate ops

def test_swap_shot_explicit_and_next(settings, work):
    res = run(settings, op("swap_shot", shot_id="shot_002", candidate_id="shot_002_wikimedia_3"),
              op("swap_shot", shot_id=1, candidate_id="next"), op("swap_shot", shot_id="6", candidate_id="local_3"))
    assert not res.errors, res.errors
    sl = load_shotlist(work)
    assert sl.by_id("shot_002").selected == "shot_002_wikimedia_3"
    assert sl.by_id("shot_001").selected == "shot_001_local_2"
    assert sl.by_id("shot_006").selected == "shot_006_local_3"
    assert "operator_pick" in sl.by_id("shot_002").flags
    assert plan(work) == ["assembly", "packaging"]
    assert set(res.dirty_shots) == {"shot_001", "shot_002", "shot_006"}
    assert_timeline(sl)
    lines = (work / "edits.jsonl").read_text().splitlines()
    assert len(lines) == 3 and all(json.loads(x)["status"] == "applied" and json.loads(x)["ts"] for x in lines)


def test_swap_to_still_changes_shot_type(settings, work):
    sl = load_shotlist(work)
    shot = sl.by_id("shot_001")
    still = sl.by_id("shot_002").candidates[0].model_copy(update={"id": "shot_001_wikimedia_9"})
    shot.candidates.append(still)
    (work / "shotlist.json").write_text(sl.model_dump_json(), encoding="utf-8")
    assert not run(settings, op("swap_shot", shot_id="shot_001", candidate_id="shot_001_wikimedia_9")).errors
    assert load_shotlist(work).by_id("shot_001").type == ShotType.image


def test_swap_unknown_candidate_is_an_error(settings, work):
    res = run(settings, op("swap_shot", shot_id="shot_002", candidate_id="nope"))
    assert res.errors and not res.applied
    assert load_shotlist(work).by_id("shot_002").selected == "shot_002_wikimedia_1"


def test_upload_image_and_video(settings, work, tmp_path):
    img = tmp_path / "mine.jpg"
    Image.new("RGB", (800, 450), (200, 120, 30)).save(img)
    clip = make_video(tmp_path / "mine.mp4", 2.0)
    res = run(settings, op("upload_shot", shot_id="shot_002", path=str(img)),
              op("upload_shot", shot_id=3, path=clip))
    assert not res.errors, res.errors
    sl = load_shotlist(work)
    up = sl.by_id("shot_002").selected_candidate()
    assert up.id == "shot_002_upload_1" and up.source == SourceKind.upload and up.licence == "Operator supplied"
    assert Path(up.local_path).parent.name == "shot_002" and len(up.keyframes) == 1
    assert Path(up.local_path + ".json").exists()
    vid = sl.by_id("shot_003").selected_candidate()
    assert vid.media == "video" and len(vid.keyframes) == 3 and all(Path(k).exists() for k in vid.keyframes)
    assert sl.by_id("shot_003").type == ShotType.footage_overlay  # overlay kept
    assert plan(work) == ["motion", "assembly", "packaging"]


def test_upload_missing_file_errors(settings, work):
    res = run(settings, op("upload_shot", shot_id="shot_002", path="/nope/x.png"))
    assert res.errors and "not found" in res.errors[0]


def test_regenerate_ai_appends_prompt_delta(settings, work, fake_ai):
    res = run(settings, op("regenerate_ai", shot_id="shot_002", prompt_delta="golden hour light"))
    assert not res.errors, res.errors
    shot = load_shotlist(work).by_id("shot_002")
    assert shot.type == ShotType.ai_image and shot.selected_candidate().source == SourceKind.ai_image
    assert shot.judge_score is not None  # every take is judged, the best one is picked
    assert "golden hour light" in fake_ai[0] and fake_ai[0].startswith("The Panama Canal, view 2")
    assert "golden hour light" in shot.ai_prompt


def test_regenerate_ai_without_delta_always_takes_the_new_batch(settings, work, fake_ai, monkeypatch):
    """An operator reroll overrides the judge: the old pick may be wrong in a way the judge missed."""
    monkeypatch.setattr("vidbush.judge.judge.score_candidate", lambda shot, cand, ctx, narration="": (40.0, "x", None))
    old = load_shotlist(work).by_id("shot_002").selected
    assert not run(settings, op("regenerate_ai", shot_id="shot_002")).errors
    shot = load_shotlist(work).by_id("shot_002")
    new = [c for c in shot.candidates if c.source == SourceKind.ai_image]
    assert [c.score for c in new] == [40.0, 40.0, 40.0] and shot.selected == new[0].id
    assert old in {c.id for c in shot.candidates}  # the old pick stays pickable


# ---------------------------------------------------------------- change_type

def test_change_type_to_ai_image_keeps_overlay(settings, work, fake_ai):
    assert not run(settings, op("change_type", shot_id="shot_004", type="ai_image")).errors
    shot = load_shotlist(work).by_id("shot_004")
    assert shot.type == ShotType.ai_image and shot.overlay is not None
    assert shot.selected_candidate().source == SourceKind.ai_image
    assert plan(work) == ["motion", "assembly", "packaging"]


def test_change_type_to_motion_graphic_variants(settings, work):
    res = run(settings, op("change_type", shot_id="shot_002", type="motion_graphic", kind="map",
                           intent="add a map of Panama"),
              op("change_type", shot_id="shot_007", type="motion_graphic", intent="show 14,000 ships a year"),
              op("change_type", shot_id="shot_001", type="motion_graphic", kind="text_overlay",
                 text="A shortcut between oceans"))
    assert not res.errors, res.errors
    sl = load_shotlist(work)
    m = sl.by_id("shot_002").motion
    assert m.kind == MotionKind.map and abs(m.duration - 4.0) < 1e-6 and m.typed_payload().markers
    card = sl.by_id("shot_007").motion
    assert card.kind == MotionKind.number_card and card.typed_payload().number == "14,000"
    text = sl.by_id("shot_001").motion
    assert text.kind == MotionKind.text_overlay and text.background == "solid"
    assert text.typed_payload().title == "A shortcut between oceans"
    assert sl.by_id("shot_002").selected is None and sl.by_id("shot_002").type == ShotType.motion_graphic
    assert_timeline(sl)


def test_change_type_map_without_place_errors(settings, work):
    sl = load_shotlist(work)
    sl.by_id("shot_006").intent = "A quiet harbour at dawn"
    (work / "shotlist.json").write_text(sl.model_dump_json(), encoding="utf-8")
    res = run(settings, op("change_type", shot_id="shot_006", type="motion_graphic", kind="map", intent="a map"))
    assert res.errors and "place" in res.errors[0]


def test_change_type_to_footage_resources_and_judges(settings, work, monkeypatch):
    clip = str(work.parent / "sources" / "_assets" / "clip.mp4")
    seen = {}

    def source_shot(shot, ctx, queries=None, exclude_ids=()):
        seen["type"] = shot.type
        return [Candidate(id=f"{shot.id}_pexels_1", source=SourceKind.pexels, local_path=clip, media="video",
                          out_point=2.0, keyframes=[clip])]

    def judge(shotlist, ctx):
        for s in shotlist.shots:
            s.selected, s.judge_score = f"{s.id}_pexels_1", 81.0
        return shotlist

    monkeypatch.setattr("vidbush.source.registry.source_shot", source_shot)
    monkeypatch.setattr("vidbush.judge.director_loop.judge_and_select", judge)
    res = run(settings, op("change_type", shot_id="shot_007", type="footage"))
    assert not res.errors, res.errors
    shot = load_shotlist(work).by_id("shot_007")
    assert seen["type"] == ShotType.footage and shot.type == ShotType.footage
    assert shot.selected == "shot_007_pexels_1" and shot.judge_score == 81.0
    assert_timeline(load_shotlist(work))


# ---------------------------------------------------------------- overlays

def test_set_and_remove_overlay(settings, work):
    spec = {"kind": "callout", "payload": {"title": "1914", "position": "upper_left"}}
    res = run(settings, op("set_overlay", shot_id="shot_001", motion_spec=spec),
              op("remove_overlay", shot_id="shot_004"))
    assert not res.errors, res.errors
    sl = load_shotlist(work)
    s1, s4 = sl.by_id("shot_001"), sl.by_id("shot_004")
    assert s1.type == ShotType.footage_overlay and s1.overlay.background == "transparent"
    assert abs(s1.overlay.duration - s1.duration) < 1e-6
    assert s4.overlay is None and s4.overlay_path == "" and s4.type == ShotType.image
    assert plan(work) == ["motion", "assembly", "packaging"]


def test_overlay_errors(settings, work):
    res = run(settings, op("set_overlay", shot_id="shot_005", motion_spec={"kind": "callout",
                                                                         "payload": {"title": "x"}}),
              op("set_overlay", shot_id="shot_001", motion_spec={"kind": "map", "payload": {}}),
              op("remove_overlay", shot_id="shot_001"))
    assert len(res.errors) == 3 and not res.applied


# ---------------------------------------------------------------- timing

def test_set_duration_gives_time_to_next(settings, work):
    assert not run(settings, op("set_duration", shot_id="shot_004", seconds=3)).errors
    sl = load_shotlist(work)
    assert sl.by_id("shot_004").duration == 3.0 and sl.by_id("shot_005").duration == 6.0
    assert sl.by_id("shot_005").motion.duration == 6.0 and sl.by_id("shot_004").overlay.duration == 3.0
    assert_timeline(sl)


def test_set_duration_rules(settings, work):
    res = run(settings, op("set_duration", shot_id="shot_001", seconds=4.0),
              op("set_duration", shot_id="shot_008", seconds=5.0),
              op("set_duration", shot_id="shot_003", seconds=1.0))
    assert len(res.errors) == 1 and "1.5" in res.errors[0]
    sl = load_shotlist(work)
    assert sl.by_id("shot_001").duration == 4.0 and sl.by_id("shot_002").duration == 3.0
    assert sl.by_id("shot_008").duration == 5.0 and sl.by_id("shot_007").duration == 4.0
    assert_timeline(sl)


def test_set_duration_clamps_to_neighbour_minimum(settings, work):
    res = run(settings, op("set_duration", shot_id="shot_002", seconds=9.0))
    sl = load_shotlist(work)
    assert not res.errors and "limited" in res.applied[0]
    assert sl.by_id("shot_003").duration == 1.5 and sl.by_id("shot_002").duration == 4.5
    assert_timeline(sl)


def test_split_shot_uses_alternate_and_crop(settings, work):
    res = run(settings, op("split_shot", shot_id="shot_007", at=2.5), op("split_shot", shot_id="shot_004"))
    assert not res.errors, res.errors
    sl = load_shotlist(work)
    a, b = sl.by_id("shot_007"), sl.by_id("shot_007b")
    assert a.duration == 2.5 and b.duration == 3.5 and sl.shots.index(b) == sl.shots.index(a) + 1
    assert b.selected == "shot_007b_wikimedia_2" and a.selected == "shot_007_wikimedia_1"
    second = sl.by_id("shot_004b")
    assert second.overlay is None and second.type == ShotType.image  # overlay stays on the first half
    assert sl.by_id("shot_004").overlay is not None
    assert_timeline(sl)
    assert len(sl.shots) == 10


def test_split_without_alternates_derives_a_crop(settings, work):
    sl = load_shotlist(work)
    shot = sl.by_id("shot_007")
    shot.candidates = shot.candidates[:1]
    (work / "shotlist.json").write_text(sl.model_dump_json(), encoding="utf-8")
    assert not run(settings, op("split_shot", shot_id="shot_007", at=3)).errors
    piece = load_shotlist(work).by_id("shot_007b")
    pick = piece.selected_candidate()
    assert pick.crop and "derived" in pick.flags and pick.local_path == shot.candidates[0].local_path
    assert pick.id.startswith("shot_007b_") and piece.framing in ("tight", "detail")


def test_split_errors_on_short_piece(settings, work):
    res = run(settings, op("split_shot", shot_id="shot_003", at=1.0))
    assert res.errors and len(load_shotlist(work).shots) == 8


def test_merge_adjacent_only(settings, work):
    res = run(settings, op("merge_shots", a="shot_002", b="shot_001"), op("merge_shots", a=3, b=6))
    assert len(res.errors) == 1 and "adjacent" in res.errors[0]
    sl = load_shotlist(work)
    assert len(sl.shots) == 7 and sl.shots[0].id == "shot_001" and sl.shots[0].duration == 7.0
    assert_timeline(sl)


def test_reorder_moves_shot_with_its_duration(settings, work):
    res = run(settings, op("reorder", shot_id="shot_002", new_index=5), op("reorder", shot_id=8, position=1))
    assert not res.errors, res.errors
    sl = load_shotlist(work)
    ids = [s.id for s in sl.shots]
    assert ids == ["shot_008", "shot_001", "shot_003", "shot_004", "shot_005", "shot_006", "shot_002", "shot_007"]
    assert sl.by_id("shot_002").duration == 4.0
    assert_timeline(sl)


def test_retime_faster_splits_into_distinct_pieces(settings, work):
    res = run(settings, op("retime_section", range=[0, 14], factor=0.7))
    assert not res.errors, res.errors
    sl = load_shotlist(work)
    inside = [s for s in sl.shots if s.start < 14 - 0.05]
    assert len(inside) == 6  # 4 shots / 0.7 -> 6
    assert all(s.duration >= 1.5 for s in inside)
    for root in ("shot_002", "shot_004"):
        pieces = [s for s in inside if s.id.startswith(root)]
        assert len(pieces) == 2
        picks = {(c.local_path, tuple(c.crop or ()), c.in_point) for c in (p.selected_candidate() for p in pieces)}
        assert len(picks) == 2
    assert sl.by_id("shot_005").duration == 4.0  # outside the range untouched
    assert plan(work) == ["motion", "assembly", "packaging"]
    assert_timeline(sl)


def test_retime_never_cuts_motion_graphics_or_below_minimum(settings, work):
    res = run(settings, op("retime_section", scene_id="s02", factor=0.3))
    assert not res.errors, res.errors
    sl = load_shotlist(work)
    assert sl.by_id("shot_005").duration == 4.0 and not any(s.id.startswith("shot_005") and s.id != "shot_005"
                                                             for s in sl.shots)
    assert_timeline(sl)


def test_retime_slower_merges_neighbours(settings, work):
    res = run(settings, op("retime_section", range=[0, 30], factor=2.0))
    assert not res.errors, res.errors
    sl = load_shotlist(work)
    assert len(sl.shots) == 4
    assert_timeline(sl)


def test_retime_bad_args(settings, work):
    res = run(settings, op("retime_section", range=[0, 10], factor=1.0), op("retime_section", factor=0.7),
              op("retime_section", scene_id="s09", factor=0.7))
    assert len(res.errors) == 3


# ---------------------------------------------------------------- music and captions

def test_change_music(settings, work):
    res = run(settings, op("change_music", track="Harbor Light"), op("change_music", track="nonexistent"))
    assert len(res.errors) == 1
    assert read_json(work / "music_override.json") == {"track": "harbor_light.ogg"}
    assert plan(work) == ["mix", "assembly", "packaging"]
    assert not run(settings, op("change_music", track="auto")).errors
    assert not (work / "music_override.json").exists()


def test_set_caption_style_merges_and_validates(settings, work):
    res = run(settings, op("set_caption_style", delta={"size": 80}), op("set_caption_style", delta={"color": "yellow"}),
              op("set_caption_style", delta={"font_size": 3}), op("set_caption_style", delta={"weight": "Heavy"}))
    assert len(res.errors) == 2
    assert read_json(work / "profile_override.json") == {"captions": {"size": 80, "color": "#FFD84A"}}
    assert plan(work) == ["mix", "assembly", "packaging"]
    from vidbush.jobs.context import build_context
    from vidbush.jobs.pipeline import load_project
    prof = build_context(settings, load_project(settings, PID), echo=False).profile
    assert prof.captions.size == 80 and prof.captions.color == "#FFD84A"


# ---------------------------------------------------------------- rewrite_beat

def test_rewrite_beat_renarrates_and_shifts_shots(settings, work, monkeypatch):
    calls = {}

    def renarrate(narration, script, beat_id, new_text, brand, seats, settings_, work_dir, log):
        calls.update(beat=beat_id, text=new_text)
        delta = -2.5
        spans = {k: ([v[0], v[1] + delta] if k == beat_id else [v[0] + delta, v[1] + delta] if v[0] >= 14.0 else v)
                 for k, v in narration.beat_spans.items()}
        return narration.model_copy(update={"duration": narration.duration + delta, "beat_spans": spans})

    monkeypatch.setattr("vidbush.audio.narration.renarrate_beat", renarrate)
    res = run(settings, op("rewrite_beat", beat_id="b002", instruction="make it punchier"))
    assert not res.errors, res.errors
    assert res.needs_renarration
    assert calls["beat"] == "b002" and "worth noting" not in calls["text"] and "very" not in calls["text"]
    script = read_model(work / "script.json", Script)
    assert script.beats()[1].text == calls["text"]
    assert read_model(work / "narration.json", Narration).duration == 26.5
    sl = load_shotlist(work)
    assert_timeline(sl, total=27.5)
    assert abs(sl.by_id("shot_003").duration + sl.by_id("shot_004").duration - 4.5) < 1e-3
    assert sl.by_id("shot_005").start == 11.5 and sl.by_id("shot_005").duration == 4.0
    assert plan(work) == ["motion", "mix", "assembly", "packaging"]


def test_rewrite_shrinking_below_minimum_merges_cover(settings, work, monkeypatch):
    def renarrate(narration, script, beat_id, new_text, *a):
        return narration.model_copy(update={"duration": narration.duration - 5.0})

    monkeypatch.setattr("vidbush.audio.narration.renarrate_beat", renarrate)
    assert not run(settings, op("rewrite_beat", beat_id=2, instruction="much shorter")).errors
    sl = load_shotlist(work)
    assert "shot_004" not in [s.id for s in sl.shots] and sl.by_id("shot_003").duration == 2.0
    assert_timeline(sl, total=25.0)


def test_rewrite_unchanged_text_does_not_renarrate(settings, work, monkeypatch):
    monkeypatch.setattr("vidbush.audio.narration.renarrate_beat", lambda *a, **k: pytest.fail("renarrated"))
    res = run(settings, op("rewrite_beat", beat_id="b003", instruction="keep the facts"))
    assert not res.errors and "unchanged" in res.applied[0] and plan(work) == []


def test_heuristic_rewrite():
    assert heuristic_rewrite("One. Two three. Four five six.", "shorter") == "One. Two three."
    assert heuristic_rewrite("The canal opened in 1914, after a decade of brutal work.", "shorter") == \
        "The canal opened in 1914."
    out = heuristic_rewrite("It is worth noting that it was very hard, and many really suffered.", "punchier")
    assert out == "It was hard. Many suffered."
    assert heuristic_rewrite("Keep me.", "warmer") == "Keep me."
    dash = chr(0x2014)
    assert dash not in heuristic_rewrite(f"A {dash} B, and C.", "punchier")


# ---------------------------------------------------------------- driver

def test_failed_op_leaves_others_applied(settings, work, fake_ai):
    res = run(settings, op("merge_shots", a=1, b=5), op("change_type", shot_id="shot_007", type="ai_image"),
              op("swap_shot", shot_id=99))
    assert len(res.errors) == 2 and len(res.applied) == 1
    lines = [json.loads(x) for x in (work / "edits.jsonl").read_text().splitlines()]
    assert [x["status"] for x in lines] == ["error", "applied", "error"]
    assert load_shotlist(work).by_id("shot_007").type == ShotType.ai_image


def test_acceptance_ops_hit_intended_shot(settings, work, fake_ai):
    res = run(settings, op("retime_section", range=[0, 30], factor=0.7),
              op("change_type", shot_id="shot_007", type="ai_image"))
    assert not res.errors, res.errors
    sl = load_shotlist(work)
    assert sl.by_id("shot_007").type == ShotType.ai_image and len(sl.shots) > 8
    assert_timeline(sl)


def test_apply_and_render_forces_stages(settings, work, monkeypatch):
    seen = {}
    monkeypatch.setattr("vidbush.jobs.pipeline.run_project", lambda s, pid, force=(), echo=True, **k: seen.update(
        pid=pid, force=list(force)))
    res = apply_and_render(settings, PID, [op("change_music", track="undertow.ogg")], echo=False)
    assert not res.errors and seen == {"pid": PID, "force": ["mix", "assembly", "packaging"]}
    seen.clear()
    apply_and_render(settings, PID, [op("swap_shot", shot_id=99)], echo=False)
    assert seen == {}


def test_missing_shotlist_is_a_clear_error(settings):
    from vidbush.jobs.pipeline import create_project
    from vidbush.models import Brief, InputMode
    create_project(settings, "empty", InputMode.brief, brief=Brief(topic="X"))
    with pytest.raises(ops_mod.EditError):
        apply_ops(settings, "empty", [op("swap_shot", shot_id=1)])


def test_total_matches_narration_tail(settings, work):
    assert ops_mod.open_session(settings, PID).total == TOTAL
