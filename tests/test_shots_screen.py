"""Run 3 timing lane, parts B to D: maps and charts are timed like text, a map says only what it draws, and
labels claim only what the picture shows."""
from __future__ import annotations

import random

import pytest

from test_shots_helpers import profile
from vidbush.mg import builders as b
from vidbush.models import (Beat, MotionKind, MotionSpec, Scene, ScenePurpose, Script, Shot, ShotType, VideoFormat,
                            VisualIdea, Word)
from vidbush.shots import ideas as I
from vidbush.shots import rules, seat
from vidbush.shots.assembler import INSET_GAP_S, _Assembler, map_conflict
from vidbush.shots.director import structural_anchors

NOME, NENANA = ("Nome", 64.5011, -165.4064), ("Nenana", 64.5638, -149.0931)
IDEAS = [VisualIdea(beat_id="b1", family="footage", intent="Dog team on the ice", queries=["dog team"])]


def _words(text: str, t0: float = 0.0, step: float = 0.5) -> list[Word]:
    return [Word(text=w, start=round(t0 + k * step, 3), end=round(t0 + k * step + 0.4, 3), beat_id="b1")
            for k, w in enumerate(text.split())]


def _shot(k: int, start: float, end: float, typ: ShotType = ShotType.footage, overlay=None, motion=None,
          must_show=()) -> Shot:
    return Shot(id=f"s{k}", beat_id="b1", start=start, end=end, duration=round(end - start, 3), type=typ, intent="x",
                overlay=overlay, motion=motion, must_show=list(must_show))


def _inset(*places, start_at: float = 0.0) -> MotionSpec:
    return b.map_spec(list(places)).model_copy(update={"background": "transparent", "start_at": start_at})


def _rebind(shot: Shot, idea: VisualIdea, idx=None) -> None:
    shot.intent, shot.motion, shot.must_show = idea.intent, None, list(idea.must_show)


# ---------------------------------------------------------------- B: one timing test for every graphic

def test_spoken_times_reads_maps_charts_numbers_and_text() -> None:
    words = _words("the train left Anchorage for Nenana then teams ran thirteen thousand miles to Nome in "
                   "nineteen twenty-five")
    route = MotionSpec(kind=MotionKind.map, duration=6.0, payload={
        "markers": [], "shaded_regions": ["Yukon"],
        "routes": [{"from_place": "Nenana", "to_place": "Nome", "points": [[64.5, -149.1], [64.5, -165.4]]}]})
    assert I.spoken_times(route, words) == [words[5].start, words[13].start]  # either endpoint, in time order
    assert I.spoken_times(route.model_copy(update={"anchor": "Anchorage"}), words)[0] == words[3].start
    chart = b.bar_chart_spec(["Ship", "Rail"], [13000, 5200], "miles")
    assert I.spoken_times(chart, words) == [words[9].start]  # its number is said
    assert I.spoken_times(b.number_card_spec("1925", "serum run"), words) == [words[15].start]
    callout = b.callout_spec("674 miles by dog sled").model_copy(update={"anchor": "teams ran"})
    assert I.spoken_times(callout, words) == [words[7].start]
    assert I.spoken_times(_inset(("Juneau", 58.3, -134.4)), words) == []


def test_overlay_pool_keeps_only_insets_whose_moment_the_shot_holds() -> None:
    text = "the dog team ran across the frozen sea ice for hours until the serum finally reached the hospital in Nome"
    words = _words(text)  # Nome at 9.5 s
    script = Script(title="Serum", format=VideoFormat.documentary, scenes=[
        Scene(id="s1", title="Run", purpose=ScenePurpose.act, beats=[Beat(id="b1", text=text)])])
    ideas = [VisualIdea(beat_id="b1", family="motion", intent="Map of Nome", motion=b.map_spec([NOME])), *IDEAS]
    ctx = _Assembler(script, {"b1": (0.0, 10.0)}, I.beat_contexts(script.scenes, None, "serum"), ideas, profile(),
                     random.Random(0), words)
    holds = [o.kind for o in ctx.overlay_pool(_shot(1, 5.0, 14.0, ShotType.footage_overlay))]
    early = [o.kind for o in ctx.overlay_pool(_shot(2, 0.0, 6.0, ShotType.footage_overlay))]
    assert MotionKind.map in holds and MotionKind.map not in early


def test_time_overlays_places_an_inset_only_where_it_passes_map_conflict() -> None:
    words = _words("dogs ran from Nenana and later reached Nome")  # Nenana 1.5 s, Nome 3.5 s
    shots = [_shot(1, 0.0, 6.0, ShotType.footage_overlay, _inset(NENANA)),
             _shot(2, 6.0, 12.0), _shot(3, 12.0, 14.0, ShotType.footage_overlay, _inset(NOME))]
    words += _words("the town of Nome waited", 12.5)
    rules.time_overlays(shots, words)
    assert shots[0].overlay is not None and shots[0].overlay.start_at == pytest.approx(1.7)
    assert shots[2].overlay is None and shots[1].overlay is None  # within INSET_GAP_S of the first map: dropped


def test_a_full_screen_map_card_waits_for_its_place_or_becomes_an_inset() -> None:
    card = b.map_spec([NOME]).model_copy(update={"duration": 7.0})
    shots = [_shot(0, 100.0, 106.0), _shot(1, 106.0, 113.0, ShotType.motion_graphic, motion=card),
             _shot(2, 113.0, 120.0)]
    words = _words("the teams pushed on and on until Nome", 104.5)  # Nome at 108.0
    assert rules.cards_on_cue(shots, words, {"b1": (100.0, 120.0)}, profile(), IDEAS, _rebind, lambda s: 0,
                              lambda s: None) == 0
    assert shots[1].type == ShotType.motion_graphic and shots[1].start == pytest.approx(108.0 - rules.CUE_LEAD_S)
    silent = [_shot(0, 100.0, 106.0), _shot(1, 106.0, 113.0, ShotType.motion_graphic, motion=card),
              _shot(2, 113.0, 120.0)]
    assert rules.cards_on_cue(silent, _words("nobody names the town", 104.0), {"b1": (100.0, 120.0)}, profile(),
                              IDEAS, _rebind, lambda s: 0, lambda s: None) == 1
    assert silent[1].type == ShotType.footage_overlay and silent[1].overlay.kind == MotionKind.map


def test_map_conflict_is_one_predicate_for_spacing_novelty_and_repeats() -> None:
    both = b.map_spec([NOME, NENANA])
    history = [(100.0, both)]
    chart = b.bar_chart_spec(["A", "B"], [1, 2])
    assert map_conflict(chart, 100.0 + INSET_GAP_S - 1, history) and not map_conflict(chart, 100.0 + INSET_GAP_S,
                                                                                         history)
    assert map_conflict(both, 150.0, history)  # the same map again within INSET_REPEAT_S
    assert map_conflict(b.map_spec([NOME]), 130.0, history)  # no new place within KIND_WINDOW_S
    assert not map_conflict(b.map_spec([NOME]), 170.0, history)
    assert not map_conflict(b.map_spec([NOME, ("Unalakleet", 63.87, -160.79)]), 130.0, history)
    assert not map_conflict(b.lower_third_spec("Nome"), 100.0, history)


def test_a_demoted_map_card_rides_as_an_inset_only_with_room() -> None:
    shots = [_shot(0, 100.0, 106.0, ShotType.footage_overlay, _inset(NENANA)),
             _shot(1, 106.0, 113.0, ShotType.motion_graphic, motion=b.map_spec([NOME]))]
    assert rules.demote_card(shots, 1, IDEAS, _rebind, lambda s: 0, lambda s: None)
    assert shots[1].type == ShotType.footage and shots[1].overlay is None  # 6 s after another map


def test_screen_guard_checks_what_is_visible_in_time_order() -> None:
    late = _shot(0, 100.0, 108.0, ShotType.footage_overlay, _inset(NENANA, start_at=6.0))  # visible at 106
    early = _shot(1, 108.0, 112.0, ShotType.footage_overlay, _inset(("Unalakleet", 63.87, -160.79)))
    early.start, early.end = 103.0, 107.0  # visible at 103: it was on screen first
    card = _shot(2, 112.0, 118.0, ShotType.motion_graphic, motion=b.map_spec([NOME]))
    shots = [late, early, card]
    changed = rules.screen_guard(shots, IDEAS, _rebind, lambda s: 0)
    assert early.overlay is not None and late.overlay is None and late.type == ShotType.footage
    assert card.type == ShotType.footage and card.intent == "Dog team on the ice"  # crowded map card: footage
    assert set(changed) == {"s0", "s2"}


def test_screen_guard_drops_overlay_text_repeated_within_30_s() -> None:
    a = _shot(0, 100.0, 105.0, ShotType.footage_overlay, b.callout_spec("Minus 46 C"))
    c = _shot(1, 120.0, 125.0, ShotType.footage_overlay, b.callout_spec("Minus 46 C"))
    d = _shot(2, 140.0, 145.0, ShotType.footage_overlay, b.callout_spec("Minus 46 C"))
    rules.screen_guard([a, c, d], IDEAS, _rebind, lambda s: 0)
    assert a.overlay is not None and c.overlay is None and d.overlay is not None


# ---------------------------------------------------------------- C: a map says only what it draws

def test_drawn_title_names_only_what_is_drawn() -> None:
    route = {"routes": [{"from_place": "Nenana", "to_place": "Nome"}], "markers": [{"label": "Nenana"}]}
    assert b.drawn_title(route) == "Nenana to Nome"
    two = {"routes": [{"from_place": "A", "to_place": "B"}, {"from_place": "B", "to_place": "C"}],
           "markers": [{"label": n} for n in ("Nenana", "Ruby", "Nulato", "Nome")]}
    assert b.drawn_title(two) == "Nenana, Ruby and Nulato"
    assert b.drawn_title({"shaded_regions": ["Alaska", "Yukon"]}) == "Alaska and Yukon"
    assert b.drawn_title({}) == ""
    assert b.map_spec([NOME, NENANA]).payload["title"] == "Nome and Nenana"


def test_verify_map_rebuilds_the_title_from_resolved_places() -> None:
    spec = MotionSpec(kind=MotionKind.map, duration=6.0, payload={
        "center": [64.5, -160.0], "zoom": 4.0, "title": "Nome to Wrangell-St. Elias, 600 miles",
        "markers": [{"place": "Nome"}, {"place": "Nowhereville Flats"}]})
    out = seat.verify_map(spec)
    assert out is not None and out.payload["title"] == "Nome"


def test_seat_plans_maps_for_journeys_and_distances() -> None:
    rules_text = " ".join(seat.DIRECTOR_RULES)
    assert "journey or a distance between named places" in rules_text and "from_place and to_place" in rules_text
    assert "title" not in seat.MOTION_SCHEMA["kinds"]["map"]


# ---------------------------------------------------------------- D: labels claim only what the picture shows

def test_no_free_boxes_or_arrows() -> None:
    assert MotionKind.highlight_box not in seat.OVERLAY_KINDS and MotionKind.arrow not in seat.OVERLAY_KINDS
    assert "highlight_box" not in seat.MOTION_SCHEMA["kinds"] and "arrow" not in seat.MOTION_SCHEMA["overlay_kinds"]
    assert seat._overlay({"kind": "highlight_box", "duration": 4.0, "payload": {"x": 0.3, "y": 0.3, "w": 0.3,
                                                                                 "h": 0.3, "label": "Town"}}) is None
    bc = I.beat_contexts([Scene(id="s1", title="Town", purpose=ScenePurpose.act, beats=[
        Beat(id=f"b{k}", text="Steam shovels and workers dug the canal near the town.") for k in range(30)])],
        None, "canal")
    kinds = {o.kind for x in bc for o in I.overlay_options(x, I.beat_entities(x), "documentary_warm")}
    assert not kinds & {MotionKind.highlight_box, MotionKind.arrow}


def test_label_ok_needs_the_name_in_must_show() -> None:
    nome = b.lower_third_spec("Nome", "")
    assert I.label_ok(nome, ["the harbour of nome", "ships"]) and I.label_ok(nome, ["NOME"])
    assert not I.label_ok(nome, ["dog team on sea ice"])
    assert not I.label_ok(b.lower_third_spec("Fort Worth"), ["whole island of Manhattan"])  # run 2 shot_022
    assert not I.label_ok(b.lower_third_spec("Coast Guard"), ["coast", "tanker"])  # a word of it is not the name
    assert I.label_ok(b.lower_third_spec("Leonhard Seppala"), ["Seppala in parka"])  # the surname names him
    for person in ("Gunnar Kaasen", "Roald Amundsen", "Curtis Welch"):  # names no list knows
        assert not I.label_ok(b.lower_third_spec(person, "musher"), ["dog sled team"])
    assert I.label_ok(b.lower_third_spec("Gunnar Kaasen", "musher"), ["Gunnar Kaasen with his team"])
    assert I.label_ok(b.callout_spec("Minus 46 C"), [])


def test_a_lower_third_moves_only_to_a_shot_that_shows_its_name() -> None:
    words = _words("the serum left Nenana by train and reached Nome six days later")
    shots = [_shot(1, 0.0, 2.0, ShotType.footage_overlay, b.lower_third_spec("Nome")),
             _shot(2, 2.0, 6.0, must_show=("sled dogs",))]
    rules.time_overlays(shots, words)
    assert shots[0].overlay is None and shots[1].overlay is None


def test_a_demoted_title_card_keeps_a_place_lower_third_only_where_shown() -> None:
    shots = [_shot(0, 100.0, 106.0, ShotType.motion_graphic, motion=b.title_card_spec("Nome"))]
    assert rules.demote_card(shots, 0, IDEAS, _rebind, lambda s: 0, lambda s: None)
    assert shots[0].type == ShotType.footage and shots[0].overlay is None
    shown = [VisualIdea(beat_id="b1", family="footage", intent="Front Street, Nome", must_show=["Nome"])]
    shots = [_shot(0, 100.0, 106.0, ShotType.motion_graphic, motion=b.title_card_spec("Nome"))]
    assert rules.demote_card(shots, 0, shown, _rebind, lambda s: 0, lambda s: None)
    assert shots[0].overlay.kind == MotionKind.lower_third


def test_no_filler_subtitles_under_names() -> None:
    bc = I.beat_contexts([Scene(id="s1", title="Gold", purpose=ScenePurpose.act, beats=[
        Beat(id="b1", text="In 1925 the serum reached Nome."), Beat(id="b2", text="Miners came to Fairbanks.")]),
        Scene(id="s2", title="The Long Run", purpose=ScenePurpose.act, beats=[
            Beat(id="b3", text="In 1925 the teams set out from Nenana.")])], None, "serum")
    named = I._named_overlays(bc[0], I.beat_entities(bc[0]), "documentary_warm")
    assert named["Nome"].payload["subtitle"] == ""  # no country under a town
    chapter = I.ideas_for_beat(bc[2], None, "documentary_warm")[0].overlay
    assert chapter.payload == {**chapter.payload, "title": "The Long Run", "subtitle": ""}  # no era under a title
    shots = [_shot(k, 100.0 + 70 * k, 105.0 + 70 * k, ShotType.footage_overlay, b.lower_third_spec(name, sub),
                   must_show=(name,))
             for k, (name, sub) in enumerate([("Nome", "United States"), ("Fairbanks", "1950s"), ("Ruby", "2010S"),
                                              ("Leonhard Seppala", "Musher")])]
    rules.screen_guard(shots, IDEAS, _rebind, lambda s: 0)
    assert [s.overlay.payload["subtitle"] for s in shots] == ["", "", "", "Musher"]


def test_one_label_per_name_per_minute() -> None:
    shots = [_shot(k, t, t + 5.0, ShotType.footage_overlay, b.lower_third_spec(n), must_show=("Nome",))
             for k, (t, n) in enumerate([(100.0, "Nome"), (140.0, "NOME"), (170.0, "Nome")])]
    rules.screen_guard(shots, IDEAS, _rebind, lambda s: 0)
    assert [s.overlay is not None for s in shots] == [True, False, True]
    after_map = [_shot(0, 100.0, 107.0, ShotType.motion_graphic, motion=b.map_spec([NOME])),
                 _shot(1, 130.0, 135.0, ShotType.footage_overlay, b.lower_third_spec("Nome"), must_show=("Nome",))]
    rules.screen_guard(after_map, IDEAS, _rebind, lambda s: 0)
    assert after_map[1].overlay is None  # the map named Nome 23 s earlier


def test_on_screen_text_stays_at_least_read_min() -> None:
    words = _words("and then Nome")  # Nome at 1.0 s
    spec = b.lower_third_spec("Nome")
    assert I.READ_MIN_S == 2.5
    assert I.overlay_start(spec, words, 0.0, 3.6) is None  # 2.4 s left after it appears
    assert I.overlay_start(spec, words, 0.0, 3.8) == pytest.approx(1.2)


def test_a_number_needs_a_label_and_is_never_part_of_a_name() -> None:
    def card(text: str):
        bc = I.beat_contexts([Scene(id="s1", title="x", purpose=ScenePurpose.act, beats=[Beat(id="b1", text=text)])],
                             None, "x")[0]
        return I.number_idea(bc, I.beat_entities(bc), None, "documentary_warm")
    assert card("It has eighteen hundred named islands.") is None  # a bare 1,800 says nothing
    assert card("Alaska is about as wide as Interstate Ten is long.") is None
    assert I.in_name("11", "In July the Apollo 11 crew landed.") and not I.in_name("10", "Ten dogs pulled the sled.")
    assert card("The team carried the serum 674 miles.").motion.payload["number"] == "674"
    assert not I.in_name("586,000", "It is the size of Alaska's 586,000 square miles.")  # a possessive is no name
    for text, number in (("By then Togo's 260 miles were behind him.", "260"),
                         ("Seppala ran with Togo 91 miles.", "91")):  # a number with a unit is a quantity
        assert card(text).motion.payload["number"] == number, text
    bare = {"beat_id": "b1", "family": "motion", "intent": "Number card",
            "motion": {"kind": "number_card", "duration": 5.0, "payload": {"number": "10", "label": ""}}}
    assert seat.validate_idea(bare, {"b1"}, None) is None


def test_hook_title_card_is_dropped_when_it_repeats_the_opening_line() -> None:
    line = "Alaska is about as wide as Interstate Ten is long."

    def script(title: str) -> Script:
        return Script(title=title, format=VideoFormat.documentary, scenes=[
            Scene(id="s1", title="Hook", purpose=ScenePurpose.hook, beats=[Beat(id="b1", text=line)]),
            Scene(id="s2", title="Setup", purpose=ScenePurpose.setup, beats=[Beat(id="b2", text="It is big.")])])
    spans = {"b1": (0.0, 6.0), "b2": (6.0, 12.0)}
    assert not structural_anchors(script(line), spans, profile(), False, 13.0, "Alaska")
    assert [a.tag for a in structural_anchors(script("Alaska: The Last Frontier"), spans, profile(), False, 13.0,
                                              "Alaska")] == ["title_card"]


@pytest.mark.parametrize("title,line", [("The Serum Run: Race Against Death", "The serum run saved a town."),
                                        ("Race to Nome", "The race to Nome began in January.")])
def test_a_short_title_case_hook_title_repeats_the_opening_line_too(title: str, line: str) -> None:
    script = Script(title=title, format=VideoFormat.documentary, scenes=[
        Scene(id="s1", title="Hook", purpose=ScenePurpose.hook, beats=[Beat(id="b1", text=line)]),
        Scene(id="s2", title="Setup", purpose=ScenePurpose.setup, beats=[Beat(id="b2", text="It is big.")])])
    assert not structural_anchors(script, {"b1": (0.0, 6.0), "b2": (6.0, 12.0)}, profile(), False, 13.0, "Nome")
