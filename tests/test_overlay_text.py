"""On-screen text belongs to a spoken moment and never repeats the narration (run 2, overlays lane)."""
from __future__ import annotations

import json
import random

import pytest

from test_render_helpers import FPS, build_shotlist as render_shotlist, frame_at
from test_shots_helpers import brand, narrate, profile, project, six_minute_script
from vidbush.editor import retime
from vidbush.judge import qa
from vidbush.mg import builders as b
from vidbush.models import Beat, MotionKind, Scene, ScenePurpose, Shot, ShotList, ShotType, StyleProfile, Word
from vidbush.render.cache import frame_range
from vidbush.render.segment import render_segment
from vidbush.shots import ideas as I
from vidbush.shots import seat
from vidbush.shots.director import build_shotlist
from vidbush.shots.rules import time_overlays


def _words(text: str, t0: float = 0.0, step: float = 0.4) -> list[Word]:
    return [Word(text=w, start=round(t0 + k * step, 3), end=round(t0 + k * step + 0.3, 3))
            for k, w in enumerate(text.split())]


def _shot(i: int, start: float, end: float, typ: ShotType = ShotType.footage, overlay=None) -> Shot:
    return Shot(id=f"shot_{i:03d}", index=i, beat_id="b1", start=start, end=end, duration=end - start, type=typ,
                intent="x", overlay=overlay)


# ---------------------------------------------------------------- repeats_narration

SAID = "Starting out at about fifty below, one musher's hands froze to his sled near Nulato."


@pytest.mark.parametrize("text,repeats", [
    ("Starting out at about fifty below", True),  # verbatim clause
    ("Hands froze to his sled", True),  # 4 consecutive tokens
    ("musher hands froze sled", True),  # most content words are spoken
    ("50 below zero", False),  # 3 tokens: a label
    ("Nulato", False),  # a name
    ("Minus 46 C in the wind", False),  # a unit conversion the narrator does not say
    ("Frostbite ended his leg of the relay", False),  # a paraphrase that adds
])
def test_repeats_narration(text: str, repeats: bool) -> None:
    assert I.repeats_narration(text, SAID) is repeats


def test_spelled_numbers_match_digits() -> None:
    assert I.repeats_narration("about 50 below at the start", "they started at about fifty below at the start")


# ---------------------------------------------------------------- anchor timing

def test_anchor_time_finds_the_spoken_words_inside_the_window() -> None:
    words = _words("the serum left Nenana by train and reached Nome six days later")
    assert I.anchor_time("reached Nome", words, 0.0, 10.0) == pytest.approx(words[7].start)
    assert I.anchor_time("Nenana", words, 0.0, 10.0) == pytest.approx(words[3].start)
    assert I.anchor_time("reached Nome", words, 0.0, 2.0) is None  # spoken after the window
    assert I.anchor_time("Anchorage", words, 0.0, 10.0) is None


def test_overlay_starts_just_after_its_anchor_with_time_to_read() -> None:
    words = _words("the serum left Nenana by train and reached Nome six days later")
    spec = b.callout_spec("674 miles by dog sled").model_copy(update={"anchor": "reached Nome"})
    at = I.overlay_start(spec, words, 1.0, 6.0)
    assert at == pytest.approx(words[7].start + I.ANCHOR_LEAD_S - 1.0)
    assert I.overlay_start(spec, words, 1.0, 3.5) is None  # the shot ends before the words are spoken and read
    assert I.overlay_start(spec, words, 3.5, 8.0) is None  # the shot starts after the words


# ---------------------------------------------------------------- time_overlays

def test_overlay_moves_to_the_shot_that_holds_its_moment_or_is_dropped() -> None:
    words = _words("the serum left Nenana by train and reached Nome six days later")
    late = b.lower_third_spec("Nome", "Alaska")
    shots = [_shot(1, 0.0, 2.0, ShotType.footage_overlay, late), _shot(2, 2.0, 6.0), _shot(3, 6.0, 9.0)]
    time_overlays(shots, words)
    assert shots[0].type == ShotType.footage and shots[0].overlay is None
    assert shots[1].type == ShotType.footage_overlay and shots[1].overlay.payload["title"] == "Nome"
    assert shots[1].start + shots[1].overlay.start_at == pytest.approx(words[8].start + I.ANCHOR_LEAD_S)
    assert shots[1].overlay.anchor == "Nome" and shots[1].overlay.duration == pytest.approx(4.0)
    lost = [_shot(1, 0.0, 3.0, ShotType.footage_overlay, b.lower_third_spec("Anchorage"))]
    time_overlays(lost, words)
    assert lost[0].type == ShotType.footage and lost[0].overlay is None


def test_time_overlays_drops_text_that_repeats_the_narration() -> None:
    words = _words("the serum left Nenana by train and reached Nome six days later")
    copy = b.callout_spec("reached Nome six days later").model_copy(update={"anchor": "reached Nome"})
    shots = [_shot(1, 0.0, 6.0, ShotType.footage_overlay, copy)]
    time_overlays(shots, words)
    assert shots[0].overlay is None and shots[0].type == ShotType.footage


def test_insets_and_locked_overlays_keep_their_start() -> None:
    words = _words("ships crossed from Colon to Panama City")
    inset = b.route_map_spec([[("Colon", 9.36, -79.9), ("Panama City", 8.97, -79.54)]]).model_copy(
        update={"background": "transparent"})
    label = b.lower_third_spec("The Hoover Dam", "Number 3")
    shots = [_shot(1, 0.0, 3.0, ShotType.footage_overlay, inset), _shot(2, 3.0, 6.0, ShotType.footage_overlay, label)]
    time_overlays(shots, words, frozenset({"shot_002"}))
    assert shots[0].overlay.start_at == 0 and shots[1].overlay is label


def test_built_shotlist_text_overlays_follow_their_words() -> None:
    script = six_minute_script()
    nar = narrate(script)
    sl, _plan = build_shotlist(project(), script, nar, None, profile(), brand(), None, random.Random(3))
    timed = [s for s in sl.shots if s.overlay is not None and s.overlay.kind in I.TEXT_OVERLAY_KINDS]
    assert timed
    for s in timed:
        shown = s.start + s.overlay.start_at
        assert I.anchor_time(s.overlay.anchor, nar.words, shown - 0.5, shown) is not None, (s.id, shown)
        assert I.anchor_time(s.overlay.anchor, nar.words, s.start - 0.2, shown - 0.5) is None, (s.id, shown)
        assert s.overlay.start_at + I.read_seconds(s.overlay) <= s.duration + 1e-3
    kinds = {s.overlay.kind for s in sl.shots if s.overlay is not None}
    assert MotionKind.callout not in kinds  # the heuristic has no key phrase callouts any more
    assert not any(s.motion and s.motion.kind in (MotionKind.callout, MotionKind.text_overlay) for s in sl.shots)


# ---------------------------------------------------------------- heuristics and seat

def _bc(text: str) -> I.BeatContext:
    beat = Beat(id="b1", text=text)
    return I.BeatContext(beat=beat, scene=Scene(id="s1", title="Serum", purpose=ScenePurpose.act, beats=[beat]),
                         era=1925, topic="Serum run")


def test_heuristic_ideas_never_quote_the_beat() -> None:
    bc = _bc("In 1925 the mushers carried the serum through fifty below toward Nome.")
    made = I.ideas_for_beat(bc, None, "documentary_warm", random.Random(1))
    phrase = I.key_phrase(bc.beat.text)
    for spec in [i.overlay for i in made] + [i.motion for i in made]:
        if spec is not None and spec.kind in I.TEXT_OVERLAY_KINDS:
            assert spec.payload.get("title") != phrase
            assert not I.repeats_narration(I.overlay_title(spec), bc.beat.text)


def test_seat_text_that_repeats_its_beat_is_dropped_never_replaced() -> None:
    text = {"b001": "Nobody knew exactly how old she was, but the village loved her."}
    item = {"beat_id": "b001", "family": "footage", "intent": "Village", "queries": ["village snow"],
            "overlay": {"kind": "callout", "payload": {"title": "Nobody knew exactly how old she was"},
                        "anchor": "Nobody knew"}}
    idea = seat.validate_idea(item, {"b001"}, None, {"b001": 5.0}, text)
    assert idea is not None and idea.overlay is None
    item["overlay"]["payload"]["title"] = "Born about 1850"
    idea = seat.validate_idea(item, {"b001"}, None, {"b001": 5.0}, text)
    assert idea.overlay.payload["title"] == "Born about 1850" and idea.overlay.anchor == "Nobody knew"
    card = {"beat_id": "b001", "family": "motion", "intent": "Quote card",
            "motion": {"kind": "text_overlay", "payload": {"title": "the village loved her"}}}
    assert seat.validate_idea(card, {"b001"}, None, {"b001": 5.0}, text) is None


def test_seat_prompt_asks_for_anchor_and_new_words() -> None:
    from test_shots_helpers import doc_script
    script = doc_script(n_scenes=1, beats_per_scene=2)
    brief = json.loads(seat.director_prompt(script, {}, None, profile(), brand()))
    assert "anchor" in brief["motion_spec_schema"]["spec"]
    assert "anchor" in brief["output"]["ideas"][0]["overlay"]
    rules = " ".join(brief["rules"])
    assert "never repeats the narration" in rules and "captions already show" in rules
    assert "never the narration's own words" in brief["motion_spec_schema"]["kinds"]["callout"]["title"]


# ---------------------------------------------------------------- render and edit

def test_segment_overlay_appears_at_start_at(tmp_path, settings) -> None:
    shot = render_shotlist(tmp_path).shots[1].model_copy(deep=True)
    shot.transition_in = "cut"
    shot.overlay = b.callout_spec("x", duration=shot.duration).model_copy(update={"anchor": "x", "start_at": 0.6})
    out = render_segment(shot, StyleProfile(id="test"), [], settings, tmp_path / "seg", frame_range(shot, FPS))
    before, after = frame_at(out, 9), frame_at(out, 30)  # 0.3 s and 1.0 s into the shot
    assert not (before[500, 960, 0] > 200 and before[500, 960, 1] < 60), "no overlay before its anchor"
    r, g, bl = (int(c) for c in after[500, 960])
    assert r > 200 and g < 60 and bl < 60, "overlay drawn after its anchor"


def test_editor_retime_keeps_the_overlay_on_its_moment() -> None:
    spec = b.lower_third_spec("Nome").model_copy(update={"anchor": "Nome", "start_at": 2.0})
    sl = ShotList(project_id="p", duration=12.0, shots=[_shot(1, 0.0, 4.0), _shot(2, 4.0, 12.0,
                                                                                  ShotType.footage_overlay, spec)])
    sl.shots[0].duration = 5.0  # the shot before grows by 1 s; the narration does not move
    sl.shots[1].duration = 7.0
    retime.finalize(sl, 12.0)
    assert sl.shots[1].start == pytest.approx(5.0) and sl.shots[1].overlay.start_at == pytest.approx(1.0)
    assert sl.shots[1].overlay.duration == pytest.approx(7.0)


# ---------------------------------------------------------------- QA

def test_qa_reports_timing_error_and_verbatim_share() -> None:
    words = _words("the serum left Nenana by train and reached Nome six days later")
    timed = b.lower_third_spec("Nome").model_copy(update={"anchor": "Nome", "start_at": 1.4})
    early = b.callout_spec("674 miles by dog sled").model_copy(update={"anchor": "left Nenana"})
    card = b.text_overlay_spec("reached Nome six days later", background="solid")
    sl = ShotList(project_id="p", duration=9.0, shots=[
        _shot(1, 0.0, 2.0, ShotType.footage_overlay, early), _shot(2, 2.0, 5.0, ShotType.footage_overlay, timed),
        Shot(id="shot_003", index=2, beat_id="b1", start=5.0, end=9.0, duration=4.0, type=ShotType.motion_graphic,
             intent="card", motion=card)])
    stats = qa.overlay_text_stats(sl, words)
    gap_early = words[2].start  # shown at 0.0, spoken at 0.8
    gap_timed = 2.0 + 1.4 - words[8].start
    assert stats["text_items"] == 3 and stats["unanchored"] == 0
    assert stats["timing_error_s"] == pytest.approx((gap_early + gap_timed) / 2, abs=0.01)
    assert stats["verbatim_pct"] == pytest.approx(33.3)


# ---------------------------------------------------------------- review fixes

def test_possessive_and_hyphenated_words_still_anchor() -> None:
    words = _words("The train reached Nenana's rail yard where the dog-sled teams waited near Nome’s harbour")
    assert I.anchor_time("Nenana", words, 0.0, 10.0) == pytest.approx(words[3].start)
    assert I.anchor_time("dog sled", words, 0.0, 10.0) == pytest.approx(words[8].start)
    assert I.anchor_time("Nome", words, 0.0, 10.0) == pytest.approx(words[12].start)
    assert I.norm_tokens("Seppala's dog-sled") == ["seppala", "dog", "sled"]


@pytest.mark.parametrize("text,kind,repeats", [
    ("Diphtheria hits Nome", MotionKind.callout, True),  # a short clause of the narration is not a name
    ("diphtheria hits", MotionKind.text_overlay, True),
    ("Nome, Alaska", MotionKind.callout, False),  # a name
    ("January 1925", MotionKind.text_overlay, False),  # a date label
    ("diphtheria hits", MotionKind.lower_third, False),  # a lower third is a name by kind
])
def test_short_text_is_exempt_only_when_it_is_a_name(text: str, kind: MotionKind, repeats: bool) -> None:
    spec = b.callout_spec(text).model_copy(update={"kind": kind})
    assert I.spec_repeats_narration(spec, "In January diphtheria hits Nome hard") is repeats


def test_qa_measures_against_the_word_the_pipeline_anchored_to() -> None:
    words = [Word(text="Nome", start=2.0, end=2.4), Word(text="and", start=4.0, end=4.2),
             Word(text="Nome", start=7.0, end=7.4)]
    spec = b.lower_third_spec("Nome")
    at = I.overlay_start(spec, words, 6.0, 12.0)
    sl = ShotList(project_id="p", duration=12.0, shots=[
        _shot(1, 0.0, 6.0), _shot(2, 6.0, 12.0, ShotType.footage_overlay,
                                  spec.model_copy(update={"anchor": "Nome", "start_at": at}))])
    assert qa.overlay_text_stats(sl, words)["timing_error_s"] == pytest.approx(I.ANCHOR_LEAD_S)


def test_editor_split_moves_the_overlay_to_the_piece_that_holds_its_moment() -> None:
    spec = b.lower_third_spec("Nome").model_copy(update={"anchor": "Nome", "start_at": 4.2})
    sl = ShotList(project_id="p", duration=6.0, shots=[_shot(1, 0.0, 6.0, ShotType.footage_overlay, spec)])
    retime.split_into(sl, sl.shots[0], [3.0, 3.0])
    retime.finalize(sl, 6.0)
    first, second = sl.shots
    assert first.overlay is None and first.type == ShotType.footage
    assert second.type == ShotType.footage_overlay and second.start + second.overlay.start_at == pytest.approx(4.2)


def test_editor_drops_an_overlay_no_shot_can_show_on_its_moment() -> None:
    spec = b.lower_third_spec("Nome").model_copy(update={"anchor": "Nome", "start_at": 4.2})
    card = Shot(id="shot_002", index=1, beat_id="b1", start=6.0, end=10.0, duration=4.0,
                type=ShotType.motion_graphic, intent="card", motion=b.title_card_spec("Nome"))
    sl = ShotList(project_id="p", duration=10.0, shots=[_shot(1, 0.0, 6.0, ShotType.footage_overlay, spec), card])
    sl.shots[0].duration, sl.shots[1].duration = 3.0, 7.0  # the card now covers the moment
    retime.finalize(sl, 10.0)
    assert sl.shots[0].overlay is None and sl.shots[0].type == ShotType.footage and sl.shots[1].overlay is None


def test_cards_repeating_nearby_narration_of_another_beat_become_footage() -> None:
    from vidbush.shots.rules import demote_repeating_cards
    words = _words("the dogs rested at Nulato") + _words("the serum reached Nome six days later", t0=5.0)
    card = Shot(id="shot_001", index=0, beat_id="b1", start=0.0, end=4.0, duration=4.0,
                type=ShotType.motion_graphic, intent="card", motion=b.callout_spec("reached Nome six days later"))
    fresh = card.model_copy(update={"id": "shot_002", "motion": b.callout_spec("674 miles by dog sled")})
    shots = [card, fresh]
    rebound = []
    n = demote_repeating_cards(shots, words, [None], lambda s, i, k: rebound.append(s.id), lambda s: 0, lambda s: None)
    assert n == 1 and rebound == ["shot_001"] and shots[0].type != ShotType.motion_graphic
    assert shots[1].type == ShotType.motion_graphic
