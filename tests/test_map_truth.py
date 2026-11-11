"""Maps draw only what is true: gazetteer places, real route geometry, measured distances, real regions."""
from __future__ import annotations

import random

import numpy as np
import pytest

from test_shots_helpers import brand, narrate, profile, project
from vidbush.mg import builders as b
from vidbush.mg import geo
from vidbush.mg import routes as R
from vidbush.render.mg_python import build_comp
from vidbush.models import (Beat, FactCard, MotionKind, MotionSpec, Research, Scene, ScenePurpose, Script, Shot,
                            ShotType, VisualAnchor)
from vidbush.shots import ideas as I
from vidbush.shots.director import build_shotlist
from vidbush.shots.seat import checked_distance, validate_idea, verify_map


def _map(payload: dict, context: str = "") -> MotionSpec | None:
    return verify_map(MotionSpec(kind=MotionKind.map, duration=8.0, payload=payload), context)


# ---------------------------------------------------------------- routing over a line layer

def _fake_layer(monkeypatch: pytest.MonkeyPatch, lines: list[list[list[float]]]) -> None:
    layer = [({"scalerank": 1}, [np.asarray(ln, np.float64) for ln in lines])]
    monkeypatch.setattr("vidbush.mg.mapdraw.load_layer", lambda name: layer)


def test_ground_path_snaps_joins_and_measures(monkeypatch: pytest.MonkeyPatch) -> None:
    # lon, lat: two lines meeting at (0, 1), and a third whose end stops 0.5 km short of the second
    _fake_layer(monkeypatch, [[[0.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]], [[1.0, 1.0045], [1.0, 2.0]]])
    pts = R.ground_path((0.0, 0.02), (1.0, 0.99), "rail")
    assert pts[0] == pytest.approx([0.0, 0.0], abs=1e-3) and pts[-1] == pytest.approx([1.0, 0.99], abs=1e-3)
    assert [1.0, 0.0] in pts  # the corner at the junction
    assert R.path_miles(pts) == pytest.approx(69.1 + 68.4, rel=0.02)
    across = R.ground_path((0.0, 0.0), (2.0, 1.0), "rail")  # crosses the 0.5 km gap
    assert across is not None and across[-1] == pytest.approx([2.0, 1.0], abs=1e-3)
    assert R.ground_path((0.0, 0.0), (0.5, 0.5), "rail") is None  # 55 km from any line


def test_real_alaska_railroad_and_unreachable_places() -> None:
    talkeetna, hurricane = (62.3238, -150.1094), (63.003, -149.583)
    rail = R.route_points(talkeetna, hurricane, "rail")
    assert rail is not None and 45 < R.path_miles(rail) < 65  # the Hurricane Turn runs about 55 miles
    assert R.route_points((61.2181, -149.9003), (64.5061, -165.4064), "rail") is None  # no railroad to Nome
    assert R.route_points(talkeetna, hurricane, "trail") is None  # no trail data: never drawn
    arc = R.route_points((61.2181, -149.9003), (64.5061, -165.4064), "air")
    assert len(arc) > 10 and R.path_miles(arc) == pytest.approx(540, rel=0.03)


@pytest.mark.parametrize("texts,via", [
    (("Train", ""), "rail"), (("", "Driving the Dalton Highway north"), "road"), (("Dog sled",), "trail"),
    (("Upriver by barge",), "river"), (("The tanker sailed to Nome",), "sea"), (("Airmail flight",), "air"),
    (("Census route",), "line"),
])
def test_infer_via(texts: tuple[str, ...], via: str) -> None:
    assert R.infer_via(*texts) == via


# ---------------------------------------------------------------- verify_map

def test_model_route_points_are_never_drawn() -> None:
    """Run 1 shot_031: three invented points and a '58 miles' label borrowed from the next beat."""
    out = _map({"center": [62.5, -149.8], "zoom": 7.97, "title": "Hurricane Turn route",
                "markers": [{"lat": 62.32, "lon": -150.11, "label": "Talkeetna"}],
                "routes": [{"points": [[62.3, -150.4], [62.5, -149.8], [62.7, -149.2]], "label": "58 miles"}]})
    p = out.typed_payload()
    assert p.routes == [] and [m.label for m in p.markers] == ["Talkeetna"]
    assert out.source_note == b.MAP_SOURCE


def test_route_follows_rail_and_distance_label_is_measured() -> None:
    out = _map({"center": [62.0, -150.0], "zoom": 6, "title": "North by train",
                "routes": [{"from_place": "Anchorage", "to_place": "Talkeetna", "label": "100 miles"},
                           {"from_place": "Anchorage", "to_place": "Talkeetna", "via": "rail", "label": "58 miles"}]})
    first, second = out.typed_payload().routes
    assert first.via == "rail" and not first.dashed  # inferred from the title
    miles = R.path_miles(first.points)
    assert first.label == "100 miles" and 80 < miles < 130  # confirmed: the researched number stays
    assert second.label == ""  # more than 25 percent off: dropped
    assert {m.label for m in out.typed_payload().markers} == {"Anchorage", "Talkeetna"}


def test_unsnappable_route_keeps_endpoint_markers_only() -> None:
    p = _map({"center": [63.0, -157.0], "zoom": 4.5, "routes": [
        {"from_place": "Nenana", "to_place": "Nome", "via": "trail", "label": "Dog sled"}]}).typed_payload()
    assert p.routes == [] and {m.label for m in p.markers} == {"Nenana", "Nome"}
    assert all(b.in_view(p.center, p.zoom, (m.lat, m.lon)) for m in p.markers)


def test_distance_between_two_markers_is_checked() -> None:
    p = _map({"center": [63.0, -157.0], "zoom": 4.5, "title": "About 70 miles apart",
              "markers": [{"place": "Anchorage"}, {"place": "Nome", "label": "Nome, 540 miles away"}]}).typed_payload()
    assert p.title == "" and p.markers[1].label == "Nome, 540 miles away"
    assert checked_distance("Train", 47.0) == "Train"
    assert checked_distance("Panama Railroad, 47 miles", 42.0) == "Panama Railroad, 47 miles"
    assert checked_distance("fifty-eight miles", 47.0) == "fifty-eight miles"
    assert checked_distance("90 km", 50.0) == "90 km" and checked_distance("90 km", 30.0) == ""


def test_markers_are_gazetteer_places_in_their_own_view() -> None:
    p = _map({"center": [63.0, -157.0], "zoom": 4.5, "markers": [
        {"place": "Nome", "lat": 0, "lon": 0}, {"label": "Westernmost", "lat": 65.78, "lon": -169.05},
        {"label": "Toksook Bay", "lat": 60.2, "lon": 162.58}, {"place": "Fairbanks"},
        {"place": "Paris"}]}).typed_payload()
    nome = geo.lookup("Nome")
    assert [m.label for m in p.markers] == ["Nome", "Fairbanks"]  # descriptions, unknowns and Paris dropped
    assert (p.markers[0].lat, p.markers[0].lon) == (nome.lat, nome.lon)
    flipped = _map({"center": [61.0, 150.0], "zoom": 5, "markers": [{"place": "Anchorage"}]}).typed_payload()
    assert flipped.markers and b.in_view(flipped.center, flipped.zoom, (61.2181, -149.9003))


def test_map_with_nothing_true_left_is_rejected() -> None:
    raw = {"beat_id": "b1", "family": "motion", "intent": "Mark the census starting point",
           "motion": {"kind": "map", "payload": {"center": [60.2, -162.6], "zoom": 8, "title": "Census begins",
                                                  "markers": [{"lat": 60.2, "lon": 162.58, "label": "Toksook Bay"}],
                                                  "shaded_regions": ["Aleutian Islands"]}}}
    assert validate_idea(raw, {"b1"}, None) is None
    inset = {"beat_id": "b1", "family": "footage", "intent": "Village", "queries": ["a b"], "overlay": raw["motion"]}
    assert validate_idea(inset, {"b1"}, None).overlay is None


def test_shaded_regions_must_exist_and_arrows_are_dropped() -> None:
    out = _map({"center": [62.0, -150.0], "zoom": 3, "shaded_regions": ["alaska", "Aleutian Islands", "Canada"],
                "arrows": [{"start": [60, -150], "end": [61, -151]}]})
    p = out.typed_payload()
    assert p.shaded_regions == ["Alaska", "Canada"] and p.arrows == []


# ---------------------------------------------------------------- places named in narration

def test_a_name_given_to_an_animal_is_not_a_place() -> None:
    text = "The worst miles went to Leonhard Seppala and his lead dog, Togo. Togo covered about 260 miles."
    assert geo.creature_names(text) == {"togo"}
    beats = [Beat(id="b1", text="The worst miles went to Leonhard Seppala and his lead dog, Togo."),
             Beat(id="b2", text="Togo covered about two hundred sixty miles.")]
    later = Scene(id="s2", title="Glass case", purpose=ScenePurpose.act, beats=[Beat(id="b3", text="Togo ran on.")])
    bcs = I.beat_contexts([Scene(id="s1", title="Relay", purpose=ScenePurpose.act, beats=beats), later], None, "Alaska")
    assert all(I.beat_entities(bc).places == [] for bc in bcs)
    assert [h.name for h in geo.find_place_hits("Cotton went to Togo.")] == ["Togo"]  # still a country elsewhere


# ---------------------------------------------------------------- builders, renderer and repeats

def test_alaska_is_shaded_and_regional_map_renders() -> None:
    spec = b.map_spec([("Talkeetna", 62.3238, -150.1094)], shaded=["Alaska"], duration=1.0)
    assert spec.typed_payload().shaded_regions == ["Alaska"]
    comp = build_comp(spec, False)
    assert comp.base.shaded is not None and np.abs(comp.base.shaded.astype(int) - comp.base.plain).mean() > 1.0
    deep = b.map_spec([("Talkeetna", 62.3238, -150.1094), ("Anchorage", 61.2181, -149.9003)], route=True,
                      via="rail", duration=1.0)
    assert deep.typed_payload().zoom >= 5  # regional zoom: roads, railways and labelled rivers drawn
    assert build_comp(deep, False).render(0).shape == (1080, 1920, 3)


def test_the_same_map_is_not_shown_twice_in_the_repeat_span() -> None:
    beats = [Beat(id=f"b{i:03d}", text="Alaska is larger than Texas, California and Montana combined.")
             for i in range(1, 25)]
    script = Script(title="Alaska", scenes=[Scene(id="s1", title="Size", purpose=ScenePurpose.act, beats=beats)])
    nar = narrate(script)
    for seed in range(3):
        sl, _plan = build_shotlist(project(), script, nar, None, profile(), brand(), None, random.Random(seed))
        from vidbush.shots.assembler import INSET_REPEAT_S, motion_signature
        shown = [(s.start, motion_signature(sp)) for s in sl.shots for sp in (s.motion, s.overlay)
                 if sp is not None and sp.kind == MotionKind.map and s.type in (ShotType.motion_graphic,
                                                                                 ShotType.footage_overlay,
                                                                                 ShotType.image_overlay)]
        for i, (t, sig) in enumerate(shown):
            assert all(not (sig == s2 and abs(t - t2) < INSET_REPEAT_S) for t2, s2 in shown[i + 1:]), seed


# ---------------------------------------------------------------- homonyms, modes and repeats (review fixes)

def test_a_place_named_or_called_is_still_a_place() -> None:
    text = "Serum came by train to a railhead called Nenana. Later they reached a town called Nome."
    assert geo.creature_names(text) == set()
    assert geo.creature_names("a dog named Balto and his horse called Comanche") == {"balto", "comanche"}
    beats = [Beat(id="b1", text="Later they reached a town called Nome."),
             Beat(id="b2", text="The serum reached Nome.")]
    bcs = I.beat_contexts([Scene(id="s1", title="Relay", purpose=ScenePurpose.act, beats=beats)], None, "Alaska")
    assert [p[0] for p in I.beat_entities(bcs[1]).places] == ["Nome"]


def test_research_never_makes_a_named_animal_a_place_anchor() -> None:
    from vidbush.research.researcher import _add_anchors
    research = Research(topic="Serum Run", facts=[
        FactCard(id="f1", claim="Seppala's lead dog Togo ran 260 miles across Norton Sound to Nome.")],
        anchors=[VisualAnchor(kind="place", name="Togo", detail="")])
    _add_anchors(research, None)
    places = {a.name for a in research.anchors if a.kind == "place"}
    assert "Togo" not in places and "Nome" in places


def test_route_endpoints_pass_the_same_view_check_as_markers() -> None:
    """A Texas map: Paris, France is not a route endpoint and never refits the map to the Atlantic."""
    p = _map({"center": [33.3, -96.5], "zoom": 6.5, "markers": [{"place": "Dallas"}, {"place": "Paris"}],
              "routes": [{"from_place": "Dallas", "to_place": "Paris", "via": "road"}]}).typed_payload()
    assert p.routes == [] and [m.label for m in p.markers] == ["Dallas"]
    assert p.center == [33.3, -96.5] and p.zoom == 6.5


def test_a_state_sharing_a_country_name_is_chosen_by_the_map() -> None:
    """Georgia next to Atlanta and Savannah is the US state, shaded alone; with no such hint it is the country."""
    p = _map({"center": [33.0, -84.0], "zoom": 5.5, "shaded_regions": ["Georgia"],
              "markers": [{"place": "Atlanta"}, {"place": "Savannah"}]}).typed_payload()
    assert p.shaded_regions == ["Georgia, United States of America"]
    assert b.in_view(p.center, p.zoom, (33.75, -84.39)) and p.center[1] < -70
    assert geo.region_name("Georgia") == "Georgia" and geo.region_name("alaska") == "Alaska"
    assert geo.region_bbox("Georgia")[1] > 35  # the country's box, east of the Black Sea
    comp = build_comp(MotionSpec(kind=MotionKind.map, duration=1.0, payload=p.model_dump()), False)
    assert comp.base.shaded is not None and np.abs(comp.base.shaded.astype(int) - comp.base.plain).mean() > 1.0
    spec = b.map_spec([("Atlanta", 33.749, -84.388)], shaded=["Georgia"], duration=1.0)
    assert spec.typed_payload().shaded_regions == ["Georgia, United States of America"]


def test_an_unnamed_mode_follows_real_rail_or_is_dropped() -> None:
    """The Denali Star: no mode in the label, title or intent. The route is the real railroad (the narration says
    train, and even without it rail is tried first), never a straight dashed line over the Alaska Range."""
    raw = {"center": [63.0, -149.0], "zoom": 5, "title": "The Denali Star",
           "routes": [{"from_place": "Anchorage", "to_place": "Fairbanks", "label": "356 miles"}]}
    for context in (("Map tracing the route north",), ("Map tracing the route north", "The train runs north.")):
        out = verify_map(MotionSpec(kind=MotionKind.map, duration=8.0, payload=dict(raw)), *context)
        r = out.typed_payload().routes
        assert len(r) == 1 and r[0].via == "rail" and not r[0].dashed and len(r[0].points) > 20
        assert r[0].label == "356 miles"
    nowhere = _map({"center": [63.0, -157.0], "zoom": 4.5, "routes": [{"from_place": "Anchorage", "to_place": "Nome"}]})
    assert nowhere.typed_payload().routes == []  # no rail or road to Nome: markers only
    assert R.route_points((61.2181, -149.9003), (64.5061, -165.4064), "line") is None


def test_the_editor_map_draws_a_route_only_for_a_described_journey() -> None:
    from vidbush.editor import ops
    script = Script(title="Serum Run", scenes=[Scene(id="s1", title="Relay", purpose=ScenePurpose.act, beats=[
        Beat(id="b1", text="Seppala and his dog Togo raced toward Nome."),
        Beat(id="b2", text="The train ran from Anchorage to Fairbanks.")])])

    class Sess:
        def script(self) -> Script:
            return script

        def ctx(self):  # noqa: ANN202
            return type("Ctx", (), {"profile": profile()})()

    def shot(beat: str) -> Shot:
        return Shot(id="shot_001", index=0, beat_id=beat, start=0, end=6, duration=6.0,
                    type=ShotType.motion_graphic, intent="")
    togo = ops.build_motion_spec(Sess(), shot("b1"), {"kind": "map"}).typed_payload()
    assert togo.routes == [] and [m.label for m in togo.markers] == ["Nome"]
    rail = ops.build_motion_spec(Sess(), shot("b2"), {"kind": "map"}).typed_payload()
    assert len(rail.routes) == 1 and rail.routes[0].via == "rail" and not rail.routes[0].dashed


def test_an_idea_overlay_never_repeats_a_map_card() -> None:
    """A map card at 10 s, then at 70 s a footage idea whose own overlay is the same map: never offered."""
    from vidbush.models import VisualIdea
    from vidbush.shots.assembler import _Assembler
    beats = [Beat(id=f"b{i}", text=f"Beat {i} goes on about the long winter.") for i in range(1, 4)]
    script = Script(title="Alaska", scenes=[Scene(id="s1", title="Winter", purpose=ScenePurpose.act, beats=beats)])
    card = b.map_spec([("Anchorage", 61.2181, -149.9003), ("Fairbanks", 64.8378, -147.7164)], duration=6.0)
    inset = card.model_copy(update={"background": "transparent"})
    ideas = [VisualIdea(beat_id="b3", family="footage", intent="Snow", queries=["snow town"], overlay=inset)]
    ctx = _Assembler(script, {"b1": (0, 30), "b2": (30, 60), "b3": (60, 90)},
                     I.beat_contexts(script.scenes, None, "Alaska"), ideas, profile(), random.Random(0), [])
    shot = Shot(id="shot_009", index=8, beat_id="b3", start=70.0, end=75.0, duration=5.0,
                type=ShotType.footage_overlay, intent="")
    ctx.rebind(shot, ideas[0], 0)
    assert shot.overlay is inset  # nothing shown yet: the idea's own overlay is used
    ctx.card_log.append((10.0, card))
    assert all(sp.kind != MotionKind.map for sp in ctx.overlay_pool(shot))
    shot.overlay = None
    ctx.rebind(shot, ideas[0], 0)
    assert shot.overlay is None
