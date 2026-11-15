"""Run 3 timing lane, part A: every picture follows the words it illustrates."""
from __future__ import annotations

import random

from test_shots_helpers import profile
from vidbush.models import Beat, Scene, ScenePurpose, Script, Shot, ShotType, VideoFormat, VisualIdea, Word
from vidbush.shots import ideas as I
from vidbush.shots import seat
from vidbush.shots.assembler import _Assembler, picture_anchor_misses

TEXT = "the dog team ran across the frozen sea ice for hours until the serum finally reached the hospital in Nome"


def _words(text: str = TEXT, t0: float = 0.0, step: float = 0.5, beat: str = "b1") -> list[Word]:
    return [Word(text=w, start=round(t0 + k * step, 3), end=round(t0 + k * step + 0.4, 3), beat_id=beat)
            for k, w in enumerate(text.split())]


def _idea(intent: str, family: str = "footage", priority: float = 1.0, anchor: str = "", must_show=(),
          beat: str = "b1", safe: bool = True) -> VisualIdea:
    return VisualIdea(beat_id=beat, family=family, intent=intent, anchor=anchor, must_show=list(must_show),
                      priority=priority, opening_safe=safe)


def _ctx(ideas: list[VisualIdea], words: list[Word] | None = None) -> _Assembler:
    script = Script(title="Serum", format=VideoFormat.documentary, scenes=[
        Scene(id="s1", title="Run", purpose=ScenePurpose.act, beats=[Beat(id="b1", text=TEXT)])])
    words = words or _words()
    spans = {"b1": (0.0, words[-1].end)}
    bcs = I.beat_contexts(script.scenes, None, "serum")
    prof = profile()
    prof.pacing.hook_seconds = 0.0  # no opening rules in these unit tests
    return _Assembler(script, spans, bcs, ideas, prof, random.Random(0), words)


def _shot(start: float, end: float, typ: ShotType = ShotType.footage, intent: str = "x") -> Shot:
    return Shot(id=f"s{int(start)}", beat_id="b1", start=start, end=end, duration=end - start, type=typ,
                intent=intent)


# ---------------------------------------------------------------- idea_anchor_time

def test_idea_anchor_time_uses_anchor_then_must_show_then_intent() -> None:
    words, span = _words(), (0.0, 12.0)
    hospital = words[TEXT.split().index("hospital")].start
    assert I.idea_anchor_time(_idea("x", anchor="reached the hospital"), words, span) == words[15].start
    assert I.idea_anchor_time(_idea("x", must_show=["a hospital", "the dog team"]), words, span) == words[1].start
    assert I.idea_anchor_time(_idea("Archival footage of the hospital"), words, span) == hospital  # style words skipped
    assert I.idea_anchor_time(_idea("Archival footage of a glacier"), words, span) is None
    assert I.idea_anchor_time(_idea("x", anchor="hospital"), words, (0.0, 5.0)) is None  # only inside the beat span


# ---------------------------------------------------------------- pick

def test_picture_slot_takes_the_idea_whose_words_are_spoken_in_it() -> None:
    ideas = [_idea("Hospital in Nome", priority=1.0, anchor="hospital in Nome"),
             _idea("Dog team on sea ice", priority=0.6, anchor="dog team ran")]
    ctx = _ctx(ideas)
    assert ctx.pick("b1", "footage", 0.0, None, 4.0) == 1  # the dogs are spoken now, the hospital 8 s later
    assert ctx.pick("b1", "footage", 8.0, None, 4.0) == 0


def test_the_slot_family_wins_a_tie_and_the_slot_type_stays() -> None:
    ideas = [_idea("Photo of the dog team", family="still", priority=0.6, anchor="dog team"),
             _idea("Footage of the dog team", family="footage", priority=0.6, anchor="dog team"),
             _idea("Photo of the hospital", family="still", priority=1.0, anchor="hospital")]
    ctx = _ctx(ideas)
    assert ctx.pick("b1", "footage", 0.0, None, 4.0) == 1
    assert ctx.pick("b1", "still", 0.0, None, 4.0) == 0
    # a footage slot whose words only a still idea illustrates takes it, and stays a footage shot
    ctx = _ctx([_idea("Footage of the hospital", priority=0.6, anchor="hospital"),
                _idea("Photo of the dog team", family="still", priority=0.6, anchor="dog team")])
    from vidbush.shots.pacing import Slot
    shot = ctx.assign([Slot(start=0.0, end=4.0, type=ShotType.footage)])[0]
    assert shot.intent == "Photo of the dog team" and shot.type == ShotType.footage


def test_reuse_still_costs_more_than_any_timing_gap() -> None:
    ideas = [_idea("Hospital in Nome", priority=1.0, anchor="hospital"),
             _idea("Dog team on sea ice", priority=0.6, anchor="dog team")]
    ctx = _ctx(ideas)
    assert ctx.pick("b1", "footage", 0.0, None, 4.0) == 1
    assert ctx.pick("b1", "footage", 0.0, None, 4.0) == 0  # the anchored idea is used: the other one plays


def test_choose_is_the_same_scoring_with_hard_constraints() -> None:
    ideas = [_idea("Hospital in Nome", priority=1.0, anchor="hospital"),
             _idea("Dog team on sea ice", priority=0.6, anchor="dog team", safe=False),
             _idea("Musher on the ice", priority=0.5, anchor="frozen sea ice")]
    ctx = _ctx(ideas)
    shot = _shot(0.0, 4.0)
    assert ctx.choose(shot) == 1
    assert ctx.choose(shot, (1,)) == 2  # excluded (an adjacent shot shows it)
    assert ctx.footage_for(shot) == 1  # a demoted card takes the same anchor-aware choice
    assert ctx.choose(_shot(8.0, 12.0)) == 0


# ---------------------------------------------------------------- the measurement

def test_picture_anchor_misses_measures_pictures_off_their_words() -> None:
    ideas = [_idea("Dog team", priority=1.0, anchor="dog team"),
             _idea("Hospital", family="still", priority=1.0, anchor="hospital"),
             _idea("Glacier", priority=0.2)]
    words = _words()
    shots = [_shot(0.0, 4.0, intent="Hospital"),  # the hospital is said 4.5 s after this shot, the dogs in it
             _shot(1.0, 4.0, intent="Dog team"), _shot(8.0, 12.0, ShotType.image, intent="Hospital"),
             _shot(12.0, 13.0, intent="Glacier"),  # never said while anchored candidates are: a miss
             _shot(2.0, 6.0, ShotType.motion_graphic, intent="Hospital")]
    assert picture_anchor_misses(shots, ideas, words, {"b1": (0.0, 12.0)}) == ["s0", "s12"]


# ---------------------------------------------------------------- the seat

def test_seat_asks_every_picture_for_its_anchor_and_keeps_it() -> None:
    rules = " ".join(seat.DIRECTOR_RULES)
    assert "anchor: the exact narration words the picture illustrates" in rules
    idea = seat.validate_idea({"beat_id": "b1", "family": "footage", "intent": "Dog team on the ice",
                               "anchor": "dog team ran"}, {"b1"}, None)
    assert idea is not None and idea.anchor == "dog team ran"


def test_an_unanchored_picture_pays_the_full_anchor_cap() -> None:
    from vidbush.shots.assembler import ANCHOR_CAP_S, ANCHOR_W
    ideas = [_idea("Hospital in Nome", priority=0.6, anchor="hospital"),
             _idea("Archival footage of a glacier", priority=0.6)]  # its words are never spoken
    ctx = _ctx(ideas)
    assert ctx.anchor_cost(1, 6.0, 10.0) == ANCHOR_W * ANCHOR_CAP_S  # was 0 inside its own beat
    assert ctx.anchor_cost(0, 6.0, 10.0) == 0.0
    assert ctx.pick("b1", "footage", 6.0, None, 4.0) == 0  # the picture whose words are spoken wins


def test_a_null_anchor_keeps_the_idea() -> None:
    for anchor in (None, 5, ["x"]):
        idea = seat.validate_idea({"beat_id": "b1", "family": "footage", "intent": "dog sled on ice",
                                   "queries": ["dog sled"], "anchor": anchor}, {"b1"}, None)
        assert idea is not None and idea.anchor == ""
