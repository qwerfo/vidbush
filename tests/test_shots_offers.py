"""Overlays go to the words that say them, not to pre-typed overlay slots; heuristic maps only for movement
between places; the director seat is asked again when an answer has nothing usable (run 3 timing lane)."""
from __future__ import annotations

from vidbush.mg import builders as b
from vidbush.mg import geo
from vidbush.models import Beat, Scene, ScenePurpose, Script, Shot, ShotType, Word
from vidbush.shots import ideas as I
from vidbush.shots import seat
from vidbush.shots.rules import time_overlays


def _words(text: str, t0: float = 0.0, step: float = 0.4) -> list[Word]:
    return [Word(text=w, start=round(t0 + k * step, 3), end=round(t0 + k * step + 0.3, 3))
            for k, w in enumerate(text.split())]


def _shot(i: int, start: float, end: float, typ: ShotType = ShotType.footage) -> Shot:
    return Shot(id=f"shot_{i:03d}", index=i, beat_id="b1", start=start, end=end, duration=end - start, type=typ,
                intent="x")


def test_an_offer_goes_to_the_plain_shot_that_holds_its_words() -> None:
    words = _words("the serum went by train as far as Nenana and on to Nome by dog sled")
    callout = b.callout_spec("674 miles by dog sled").model_copy(update={"anchor": "on to Nome"})
    shots = [_shot(1, 0.0, 3.0), _shot(2, 3.0, 9.0), _shot(3, 9.0, 12.0)]
    time_overlays(shots, words, offers=[((0.0, 12.0), callout)], ceiling=5)
    assert shots[0].overlay is None and shots[2].overlay is None
    assert shots[1].type == ShotType.footage_overlay and shots[1].overlay.payload["title"] == "674 miles by dog sled"
    said = next(w.start for w in words if w.text == "Nome")  # stop words never anchor
    assert shots[1].start + shots[1].overlay.start_at == round(said + I.ANCHOR_LEAD_S, 3)


def test_offers_stop_at_the_overlay_ceiling_and_outside_their_beat() -> None:
    words = _words("the serum went by train as far as Nenana and on to Nome by dog sled")
    one = b.callout_spec("By rail first").model_copy(update={"anchor": "by train"})
    two = b.callout_spec("674 miles by dog sled").model_copy(update={"anchor": "on to Nome"})
    shots = [_shot(1, 0.0, 4.5), _shot(2, 4.5, 9.0)]
    time_overlays(shots, words, offers=[((0.0, 9.0), one), ((0.0, 9.0), two)], ceiling=1)
    assert [s.overlay is not None for s in shots] == [True, False]  # the ceiling is a cap, not a quota
    far = [_shot(1, 0.0, 4.0), _shot(2, 4.0, 9.0)]
    time_overlays(far, words, offers=[((60.0, 70.0), two)], ceiling=5)
    assert all(s.overlay is None for s in far)  # an offer only lands near its own beat


def _bc(text: str) -> I.BeatContext:
    beat = Beat(id="b1", text=text)
    return I.BeatContext(beat=beat, scene=Scene(id="s1", title="Story", purpose=ScenePurpose.act, beats=[beat]))


def test_heuristic_map_only_when_the_beat_moves_between_places() -> None:
    one = _bc("Fresh serum turned up in an Anchorage hospital that winter.")
    assert I.map_idea(one, I.beat_entities(one), "documentary_warm") is None  # one place is a lower third
    two = _bc("The serum went by train from Anchorage to Nenana.")
    idea = I.map_idea(two, I.beat_entities(two), "documentary_warm")
    assert idea is not None and {"Anchorage", "Nenana"} <= set(I.drawn_names(idea.motion))


def test_director_chunk_with_nothing_usable_is_asked_once_more() -> None:
    beat = Beat(id="b1", text="Nome had one doctor.")
    script = Script(title="T", scenes=[Scene(id="s1", title="Nome", purpose=ScenePurpose.act, beats=[beat])])
    answers = [{"ideas": []}, {"ideas": [{"beat_id": "b1", "family": "footage", "intent": "A doctor in Nome",
                                          "anchor": "one doctor"}]}]

    class Seats:
        calls = 0

        def llm_json(self, role, task, system, user, stage, **kw):
            if task == "script_summary":
                return {"summary": "A town and its doctor."}
            Seats.calls += 1
            return answers[Seats.calls - 1]

    from test_shots_helpers import brand, profile
    out = seat.director_ideas(script, {"b1": (0.0, 3.0)}, None, profile(), brand(), Seats(), _Log())
    assert Seats.calls == 2 and [i.anchor for i in out] == ["one doctor"]


class _Log:
    def info(self, *a, **k): ...
    def warn(self, *a, **k): ...


def test_near_only_breaks_ties_between_homonyms() -> None:
    plain = geo.lookup("Colon")
    assert plain is not None
    assert geo.lookup("Colon", near=[(9.1, -79.7)]).name == plain.name  # same town keeps its name form


def test_an_offer_takes_the_mention_inside_its_beat_over_an_earlier_one() -> None:
    words = [Word(text="Nome", start=10.0, end=10.3), Word(text="Nome", start=24.0, end=24.3)]
    callout = b.callout_spec("The last relay").model_copy(update={"anchor": "Nome"})
    shots = [_shot(1, 0.0, 9.0), _shot(2, 9.0, 14.0), _shot(3, 14.0, 20.0), _shot(4, 20.0, 28.0),
             _shot(5, 28.0, 30.0)]
    time_overlays(shots, words, offers=[((20.0, 30.0), callout)], ceiling=5)
    assert [s.id for s in shots if s.overlay is not None] == ["shot_004"]  # not 10 s early on the b4 mention


def test_a_timed_inset_gets_an_anchor_so_the_editor_keeps_it_on_its_moment() -> None:
    from vidbush.editor.retime import finalize
    from vidbush.models import ShotList
    words = _words("the serum reached Nenana by train and then the teams set out")
    inset = b.map_spec([("Nenana", 64.56, -149.09), ("Nome", 64.5, -165.4)]).model_copy(
        update={"background": "transparent"})
    shots = [_shot(1, 0.0, 7.4), _shot(2, 7.4, 12.0)]
    time_overlays(shots, words, offers=[((0.0, 12.0), inset)], ceiling=5)
    assert shots[0].overlay is not None and shots[0].overlay.anchor == "Nenana"
    assert shots[0].overlay.start_at > 0
    shots[0].duration, shots[0].end = 1.5, 1.5  # an editor op shortens the host below its moment
    shots[1].start, shots[1].duration = 1.5, 10.5
    sl = ShotList(project_id="p", duration=12.0, shots=shots)
    finalize(sl, 12.0)
    assert sl.shots[0].overlay is None  # moved or dropped, never left with a stale start_at


def test_a_small_town_ends_a_journey_only_beside_a_natural_earth_endpoint() -> None:
    assert I.travel_route("The standoff between the President and the Captain lasted weeks.") is None
    route = I.travel_route("The dogs ran from Nome to Shaktoolik.")
    assert route is not None and [p[0] for p in route] == ["Nome", "Shaktoolik"]
