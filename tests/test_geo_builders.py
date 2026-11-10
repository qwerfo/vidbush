"""Motion spec builders: every kind validates, map zoom fits the points, timings retime."""
from __future__ import annotations

import pytest

from vidbush.mg import builders as b
from vidbush.models import MapPayload, MotionKind, MotionSpec

NY = ("New York", 40.71, -74.0)
SF = ("San Francisco", 37.77, -122.42)
GATUN = ("Gatun Locks", 9.272, -79.922)
MIRA = ("Miraflores Locks", 8.997, -79.591)


def _all_specs() -> list[MotionSpec]:
    return [
        b.map_spec([NY, SF], route=True, title="Coast to coast"),
        b.map_spec([], shaded=["Panama"], title="Panama"),
        b.route_map_spec([[NY, ("Cape Horn", -55.98, -67.27), SF], [NY, ("Colon", 9.36, -79.9), SF]],
                         labels=["Around the Horn", "Through Panama"]),
        b.bar_chart_spec(["Around Cape Horn", "Through Panama"], [13000, 5200], "miles", "Voyage", "Panama Canal", 0),
        b.line_chart_spec(["1914", "1950", "2000"], [1000, 5000, 13000], "ships", "Transits"),
        b.number_card_spec("22,000", "workers died", "French era", count_up=True),
        b.title_card_spec("The Panama Canal", "Ocean to Ocean"),
        b.lower_third_spec("George Goethals", "Chief engineer"),
        b.callout_spec("No pumps. Only gravity."),
        b.text_overlay_spec("1914", position="upper_left"),
        b.highlight_box_spec(0.2, 0.0, 0.3, 0.9, "Lock gate"),
        b.arrow_spec(0.1, 0.05, 0.6, 0.99, "Here"),
        b.recap_grid_spec(["Hoover Dam", "Erie Canal"]),
    ]


def test_every_builder_validates() -> None:
    specs = _all_specs()
    assert {s.kind for s in specs} == set(MotionKind)
    for s in specs:
        s.typed_payload()
        MotionSpec.model_validate_json(s.model_dump_json())
        if s.kind in (MotionKind.map, MotionKind.bar_chart, MotionKind.line_chart):
            assert s.source_note
        if s.kind in (MotionKind.lower_third, MotionKind.callout, MotionKind.text_overlay, MotionKind.highlight_box,
                      MotionKind.arrow):
            assert s.background == "transparent"


def test_source_notes() -> None:
    assert b.bar_chart_spec(["a", "b"], [1, 2], source_note="").source_note == "Illustrative"
    assert b.bar_chart_spec(["a", "b"], [1, 2], source_note="Panama Canal").source_note == "Source: Panama Canal"
    assert b.map_spec([NY]).source_note.startswith("Map data")


def test_safe_area_clamping() -> None:
    box = b.highlight_box_spec(0.2, 0.0, 0.3, 0.9).typed_payload()
    assert box.y >= 150 / 1080 - 1e-6 and box.y + box.h <= 930 / 1080 + 1e-6
    arrow = b.arrow_spec(0.1, 0.05, 0.6, 0.99).typed_payload()
    assert 150 / 1080 - 1e-6 <= arrow.y and arrow.to_y <= 930 / 1080 + 1e-6


def test_map_zoom_from_bbox() -> None:
    world = b.map_spec([NY, ("Sydney", -33.87, 151.21)]).typed_payload()
    usa = b.map_spec([NY, SF]).typed_payload()
    canal = b.map_spec([GATUN, MIRA]).typed_payload()
    assert world.zoom < usa.zoom < canal.zoom
    assert world.zoom <= 2.0 and canal.zoom >= 6.0
    assert usa.zoom_end > usa.zoom
    assert usa.center[0] == pytest.approx((40.71 + 37.77) / 2, abs=0.01)
    assert usa.center[1] == pytest.approx((-74.0 - 122.42) / 2, abs=0.01)
    pac = b.map_view([(21.3, -157.9), (35.7, 139.7)])[0]
    assert pac[1] > 150 or pac[1] < -150  # across the antimeridian, centred in the Pacific


def test_map_route_and_shading() -> None:
    p = b.map_spec([NY, ("Colon", 9.36, -79.9), SF], route=True, shaded=["Panama"], duration=8).typed_payload()
    r = p.routes[0]
    assert len(p.routes) == 1 and r.via == "line" and r.dashed  # a plain line claims no ground path
    assert r.points[0] == [40.71, -74.0] and r.points[-1] == [37.77, -122.42]
    assert any(abs(q[0] - 9.36) < 0.01 and abs(q[1] + 79.9) < 0.01 for q in r.points)  # through Colon
    assert p.shaded_regions == ["Panama"]
    times = [m.appear_at for m in p.markers]
    assert times == sorted(times) and times[-1] <= 8


def test_retime_scales_reveals() -> None:
    spec = b.map_spec([NY, SF], route=True, duration=8.0)
    r = b.retime_spec(spec, 4.0)
    p0, p1 = MapPayload(**spec.payload), MapPayload(**r.payload)
    assert r.duration == 4.0
    assert p1.routes[0].reveal_end == pytest.approx(p0.routes[0].reveal_end / 2, abs=0.01)
    assert all(m.appear_at <= 4.0 for m in p1.markers)
    card = b.retime_spec(b.title_card_spec("X"), 9.5)
    assert card.duration == 9.5 and card.payload["title"] == "X"


def test_format_number() -> None:
    assert b.format_number(375e6, "dollars") == "$375M"
    assert b.format_number(22000, "people") == "22,000"
    assert b.format_number(40, "percent") == "40%"
    assert b.format_number(1.5e9) == "1.5B"


def test_format_number_keeps_two_decimals() -> None:
    assert b.format_number(3.25e6, "cubic yards") == "3.25 million"
    assert b.format_number(2.75e9, "dollars") == "$2.75B"
    assert b.format_number(49e6, "dollars") == "$49M"


def test_recap_grid_carries_numbers() -> None:
    spec = b.recap_grid_spec(["A", "B", "C"], numbers=[3, 2, 1])
    assert spec.typed_payload().numbers == [3, 2, 1]
    assert b.recap_grid_spec(["A", "B"], numbers=[1]).typed_payload().numbers is None


def test_map_shades_states_from_admin_1() -> None:
    spec = b.map_spec([], shaded=["Nevada", "Arizona", "Atlantis"], title="Nevada and Arizona")
    p = spec.typed_payload()
    assert p.shaded_regions == ["Nevada", "Arizona"] and {lb.text for lb in p.labels} == {"NEVADA", "ARIZONA"}
    assert 30 < p.center[0] < 42 and -120 < p.center[1] < -108 and p.zoom > 3


def test_rail_map_follows_the_track_or_keeps_markers_only() -> None:
    anchorage, fairbanks = ("Anchorage", 61.2181, -149.9003), ("Fairbanks", 64.8378, -147.7164)
    p = b.map_spec([anchorage, fairbanks], route=True, via="rail").typed_payload()
    r = p.routes[0]
    assert r.via == "rail" and not r.dashed and len(r.points) > 20
    assert r.points[0] == pytest.approx([61.22, -149.9], abs=0.1)
    assert r.points[-1] == pytest.approx([64.84, -147.72], abs=0.1)
    nome = ("Nome", 64.5061, -165.4064)  # no railroad reaches Nome
    p = b.map_spec([anchorage, nome], route=True, via="rail").typed_payload()
    assert p.routes == [] and [m.label for m in p.markers] == ["Anchorage", "Nome"]


def test_in_view_matches_the_projection() -> None:
    assert b.in_view([62.5, -150.0], 8.0, (62.32, -150.11))
    assert not b.in_view([62.5, -150.0], 8.0, (63.6, -150.0))
    assert b.in_view([0.0, 179.0], 4.0, (0.0, -179.0))  # across the antimeridian
