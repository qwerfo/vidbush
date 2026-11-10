"""Full screen cards: hold caps, the excess as footage, cards on their cue, demoted cards keep their message,
spoken years, chapter titles as lower thirds, and the QA and acceptance measures by runtime."""
from __future__ import annotations

import random

import pytest

from test_shots_helpers import brand, doc_script, narrate, profile, project, top10_script
from vidbush.acceptance import _animation_checks
from vidbush.judge.qa import animation_share
from vidbush.mg import builders as b
from vidbush.models import (STATIC_CARD_KINDS, Beat, MotionKind, Scene, ScenePurpose, Shot, ShotList, ShotType,
                            VisualIdea, Word)
from vidbush.shots import ideas as I
from vidbush.shots import rules
from vidbush.shots.director import beat_spans, build_shotlist, card_exempt_ranges, structural_anchors

IDEAS = [VisualIdea(beat_id="b1", family="footage", intent="Ships in the locks", queries=["ships locks"])]


def _shots(spec: list[tuple[ShotType, float, object]], start: float = 100.0) -> list[Shot]:
    out, t = [], start
    for i, (typ, d, motion) in enumerate(spec):
        out.append(Shot(id=f"s{i}", beat_id="b1", start=t, end=t + d, duration=d, type=typ, intent=f"shot {i}",
                        motion=motion))
        t += d
    return out


def _rebind(shot: Shot, idea: VisualIdea, idx=None) -> None:
    shot.intent, shot.motion = idea.intent, None


def _contiguous(shots: list[Shot], total: float) -> None:
    assert all(abs(a.end - z.start) < 1e-6 for a, z in zip(shots, shots[1:]))
    assert round(sum(s.duration for s in shots), 6) == total


def _words(text: str, start: float, step: float = 0.4) -> list[Word]:
    return [Word(text=w, start=round(start + k * step, 3), end=round(start + k * step + 0.3, 3), beat_id="b1")
            for k, w in enumerate(text.split())]


# ---------------------------------------------------------------- hold caps

def test_long_number_card_gives_its_excess_to_the_neighbours() -> None:
    prof = profile()
    shots = _shots([(ShotType.footage, 4.0, None), (ShotType.motion_graphic, 9.0, b.number_card_spec("87", "years")),
                    (ShotType.footage, 4.0, None)])
    rules.cap_card_holds(shots, prof, IDEAS, _rebind, lambda s: 0, lambda s: None)
    assert shots[1].duration <= prof.pacing.card_hold_max + 1e-6
    _contiguous(shots, 17.0)


def test_card_excess_without_room_becomes_footage_of_its_beat() -> None:
    prof = profile()
    shots = _shots([(ShotType.motion_graphic, 7.0, b.title_card_spec("Nome")),
                    (ShotType.motion_graphic, 10.0, b.text_overlay_spec("Sorry for the black eye", background="solid")),
                    (ShotType.footage, 8.0, None)])
    rules.cap_card_holds(shots, prof, IDEAS, _rebind, lambda s: 0, lambda s: None)
    cards = [s for s in shots if s.type == ShotType.motion_graphic]
    assert all(s.duration <= prof.pacing.card_hold_max + 1e-6 for s in cards)
    assert len(shots) == 5 and shots[1].type == ShotType.footage and shots[1].intent == "Ships in the locks"
    assert shots[1].beat_id == "b1" and shots[1].transition_in == "cut"
    _contiguous(shots, 25.0)


def test_full_screen_map_holds_at_most_seven_seconds() -> None:
    spec = b.route_map_spec([[("Colon", 9.36, -79.9), ("Panama City", 8.97, -79.54)]], title="Across")
    shots = _shots([(ShotType.motion_graphic, 12.0, spec), (ShotType.motion_graphic, 5.0, b.number_card_spec("1"))])
    rules.cap_card_holds(shots, profile(), IDEAS, _rebind, lambda s: 0, lambda s: None)
    assert shots[0].duration == pytest.approx(rules.MG_HOLD_MAX_S)
    assert shots[1].type == ShotType.footage


def test_locked_cards_keep_their_length() -> None:
    shots = _shots([(ShotType.motion_graphic, 9.0, b.number_card_spec("#3")), (ShotType.footage, 4.0, None)])
    rules.cap_card_holds(shots, profile(), IDEAS, _rebind, lambda s: 0, lambda s: None, frozenset({"s0"}))
    assert shots[0].duration == 9.0


# ---------------------------------------------------------------- demoted cards keep their message

def test_demoted_title_and_number_cards_ride_on_footage() -> None:
    shots = _shots([(ShotType.motion_graphic, 4.0, b.title_card_spec("Gunnar Kaasen", "Musher")),
                    (ShotType.footage, 4.0, None),
                    (ShotType.motion_graphic, 4.0, b.number_card_spec("$77 million", "cost", count_up=True))])
    assert rules.demote_card(shots, 0, IDEAS, _rebind, lambda s: 0, lambda s: None)
    assert rules.demote_card(shots, 2, IDEAS, _rebind, lambda s: 0, lambda s: None)
    title, number = shots[0].overlay, shots[2].overlay
    assert shots[0].type == shots[2].type == ShotType.footage_overlay
    assert title.kind == MotionKind.lower_third and title.payload["title"] == "Gunnar Kaasen"
    assert number.kind == MotionKind.number_card and number.payload["number"] == "$77 million"
    assert number.background == "transparent" and shots[2].intent == "Ships in the locks"


# ---------------------------------------------------------------- cards on cue

def test_cue_times_read_digits_and_spelled_numbers() -> None:
    words = _words("In two thousand seven, about seventy-seven million dollars and 1,000 pounds.", 10.0)
    assert rules.cue_times(b.number_card_spec("2007"), words) == [10.4]
    assert rules.cue_times(b.number_card_spec("$77 million"), words) == [12.0]
    assert rules.cue_times(b.number_card_spec("1,000", "pounds"), words) == [13.6]
    assert rules.cue_times(b.title_card_spec("GUNNAR KAASEN"), _words("Then Gunnar's team ran.", 5.0)) == [5.4]


def test_early_card_moves_onto_its_spoken_number() -> None:
    prof = profile()
    shots = _shots([(ShotType.footage, 4.0, None), (ShotType.motion_graphic, 4.0, b.number_card_spec("87", "years")),
                    (ShotType.footage, 6.0, None)])
    words = _words("For more than eighty-seven years the race went on.", 106.0)  # 'eighty-seven' at 107.2
    n = rules.cards_on_cue(shots, words, {"b1": (100.0, 114.0)}, prof, IDEAS, _rebind, lambda s: 0, lambda s: None)
    assert n == 0 and shots[1].type == ShotType.motion_graphic
    assert abs(shots[1].start - (107.2 - rules.CUE_LEAD_S)) < 1e-6 and shots[1].duration == 4.0
    _contiguous(shots, 14.0)


def test_card_never_spoken_becomes_an_overlay_with_its_text() -> None:
    shots = _shots([(ShotType.footage, 4.0, None),
                    (ShotType.motion_graphic, 4.0, b.callout_spec("From TV story to turf", background="solid")),
                    (ShotType.footage, 5.0, None)])
    words = _words("The film made the dogs famous across the country.", 100.0)
    n = rules.cards_on_cue(shots, words, {"b1": (100.0, 113.0)}, profile(), IDEAS, _rebind, lambda s: 0,
                           lambda s: None)
    assert n == 1 and shots[1].type == ShotType.footage_overlay
    assert shots[1].overlay.payload["title"] == "From TV story to turf"


# ---------------------------------------------------------------- built videos

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_built_documentary_cards_are_short_and_on_cue(seed: int) -> None:
    script = doc_script(n_scenes=8, beats_per_scene=7)
    nar = narrate(script)
    prof = profile()
    sl, _plan = build_shotlist(project(), script, nar, None, prof, brand(), None, random.Random(seed))
    spans = beat_spans(script, nar)
    for s in sl.shots:
        if s.type != ShotType.motion_graphic or s.motion is None or s.motion.kind not in STATIC_CARD_KINDS:
            continue
        assert s.duration <= prof.pacing.card_hold_max + 1e-6, (seed, s.id)
        if s.motion.kind == MotionKind.title_card and s.motion.payload["title"] == "The Panama Canal":
            continue  # the structural main title at the setup
        a, z = spans[s.beat_id]
        cues = rules.cue_times(s.motion, [w for w in nar.words if a <= w.start <= z])
        assert any(s.start - rules.CUE_EARLY_S <= c <= s.start + rules.CUE_LATE_S for c in cues), (seed, s.id)


def test_top10_number_cards_are_short_and_only_top10_hooks_are_exempt() -> None:
    script = top10_script()
    nar = narrate(script)
    prof = profile(top10=True)
    sl, _plan = build_shotlist(project(), script, nar, None, prof, brand(), None, random.Random(1))
    numbers = [s for s in sl.shots if s.motion and s.motion.kind == MotionKind.number_card]
    assert len(numbers) == 10 and all(s.duration <= prof.pacing.card_hold_max + 1e-6 for s in numbers)
    spans = beat_spans(script, nar)
    assert (0.0, prof.pacing.hook_seconds) in card_exempt_ranges(script, spans, prof, [])
    doc = doc_script(n_scenes=4, beats_per_scene=3)
    doc_spans = beat_spans(doc, narrate(doc))
    anchors = structural_anchors(doc, doc_spans, profile(), False, 100.0, "x")
    assert anchors and anchors[0].duration == profile().pacing.card_hold_max
    assert card_exempt_ranges(doc, doc_spans, profile(), anchors) == []


# ---------------------------------------------------------------- ideas

def _bc(text: str, first: bool = False, purpose: ScenePurpose = ScenePurpose.act) -> I.BeatContext:
    beat = Beat(id="b1", text=text)
    scene = Scene(id="s1", title="The Serum Run", purpose=purpose, beats=[beat])
    return I.BeatContext(beat=beat, scene=scene, first_in_scene=first, era=1925, topic="Alaska")


@pytest.mark.parametrize("text,year", [
    ("In two thousand seven, the statue was cleaned and moved.", "2007"),
    ("By nineteen fifty-nine, the territory had become a state.", "1959"),
    ("In eighteen oh five, the first traders arrived.", "1805"),
])
def test_spoken_year_is_a_year_card_not_a_count(text: str, year: str) -> None:
    idea = I.number_idea(_bc(text), I.beat_entities(_bc(text)), None, "documentary_warm")
    assert idea is not None and idea.motion.payload["number"] == year
    assert idea.motion.payload["count_up"] is False


def test_spelled_counts_still_make_number_cards() -> None:
    text = "The team carried the serum six hundred seventy-four miles."
    idea = I.number_idea(_bc(text), I.beat_entities(_bc(text)), None, "documentary_warm")
    assert idea is not None and idea.motion.payload["number"].startswith("674")


def test_chapter_title_is_a_lower_third_on_footage_not_a_card() -> None:
    made = I.ideas_for_beat(_bc("Sled dogs raced across the frozen river.", first=True), None, "documentary_warm",
                            random.Random(1))
    assert not any(i.motion and i.motion.kind == MotionKind.title_card for i in made)
    foot = [i for i in made if i.family == "footage"]
    assert foot[0].overlay.kind == MotionKind.lower_third and foot[0].overlay.payload["title"] == "The Serum Run"


# ---------------------------------------------------------------- QA and acceptance

def _list(cards: list[tuple[float, MotionKind]], total: float = 60.0) -> ShotList:
    shots, t = [], 0.0
    specs = {MotionKind.number_card: b.number_card_spec("5"), MotionKind.title_card: b.title_card_spec("A")}
    for d, kind in cards:
        shots.append(Shot(id=f"c{t}", beat_id="b", start=t, end=t + d, duration=d, type=ShotType.motion_graphic,
                          intent="x", motion=specs[kind]))
        t += d
    shots.append(Shot(id="f", beat_id="b", start=t, end=total, duration=total - t, type=ShotType.footage_overlay,
                      intent="x", overlay=b.route_map_spec([[("Colon", 9.36, -79.9), ("Panama City", 8.97, -79.54)]])))
    return ShotList(project_id="p", duration=total, shots=shots)


def test_card_time_and_hold_checks() -> None:
    long = _list([(8.0, MotionKind.number_card), (8.0, MotionKind.title_card)])
    anim = animation_share(long)
    assert anim["full_screen_card_time"] == pytest.approx(26.7) and anim["max_card_hold_s"] == 8.0
    checks = _animation_checks(long, {})
    assert not checks["card_time_sparing"] and not checks["card_hold_short"]
    assert _animation_checks(long, {}, top10=True)["card_time_sparing"]  # structural top 10 cards
    short = _animation_checks(_list([(3.5, MotionKind.number_card)]), {})
    assert short["card_time_sparing"] and short["card_hold_short"]
