"""Map layout: one collision set for all text and chrome, one name per river, a locator on close-up maps."""
from __future__ import annotations

from collections import Counter

from vidbush.mg.mapdraw import _box, _overlap, major_rivers, path_prefix
from vidbush.models import MotionKind, MotionSpec
from vidbush.render.mg_python import build_comp

W, H = 960, 540


def comp(payload: dict, background: str = "solid"):  # noqa: ANN201
    spec = MotionSpec(kind=MotionKind.map, duration=2.0, payload=payload, background=background,
                      source_note="Map data: Natural Earth", width=W, height=H, fps=10)
    return build_comp(spec, False)


def text_boxes(c) -> list[tuple[str, tuple[float, float, float, float]]]:  # noqa: ANN001
    """Every drawn text and chrome box at the final zoom."""
    out = [("credit", _box(c.note_xy, c._note))]
    if c.title is not None:
        out.append(("title", _box(c.title_xy, c.title)))
    if c.locator is not None:
        out.append(("locator", _box(c.locator_xy, c.locator)))
    for (_, uv), spr, (dx, dy) in zip(c.markers, c.marker_pills, c.offsets):
        if spr is not None:
            out.append(("marker", c._boxes(uv, spr, dx, dy)[-1]))
    for spr, (_, uv), (dx, dy) in zip(c.route_pills, c.routes, c.route_offs):
        if spr is not None:
            out.append(("route chip", c._boxes(path_prefix(uv, 0.5)[-1:], spr, dx, dy)[-1]))
    for spr, (_, uv), (dx, dy) in zip(c.arrow_pills, c.arrows, c.arrow_offs):
        if spr is not None:
            out.append(("arrow chip", c._boxes(uv[:1], spr, dx, dy)[-1]))
    for uv, spr, (dx, dy) in c.labels:
        out.append(("label", c._boxes(uv, spr, dx, dy)[-1]))
    for name, uv, spr, (dx, dy) in c.rivers:
        out.append((name, c._boxes(uv, spr, dx, dy)[-1]))
    return out


def test_no_text_overlaps_other_text_title_or_credit() -> None:
    # southeastern US at a regional zoom: many rivers, a shaded state, a label near the title corner
    c = comp({"center": [33.0, -92.0], "zoom": 4.4, "zoom_end": 4.7, "title": "Texas",
              "shaded_regions": ["Texas"], "markers": [{"lat": 29.76, "lon": -95.37, "label": "Houston"}],
              "labels": [{"lat": 31.0, "lon": -99.0, "text": "TEXAS", "size": "large"},
                         {"lat": 37.5, "lon": -104.0, "text": "COLORADO", "size": "large"}]})
    boxes = text_boxes(c)
    assert len(c.rivers) >= 3
    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            assert not _overlap(ba, bb), (a, b)


def test_route_chip_never_covers_a_marker_chip() -> None:
    # a marker chip near the route's midpoint, where the route chip sits: the marker chip moves around it
    c = comp({"center": [45.0, -100.0], "zoom": 4.6,
              "markers": [{"lat": 45.0, "lon": -100.0, "label": "Pierre"}, {"lat": 45.0, "lon": -90.0, "label": "Wausau"},
                          {"lat": 46.0, "lon": -95.5, "label": "Middle place long name"}],
              "routes": [{"points": [[45.0, -100.0], [45.0, -90.0]], "label": "Long route chip label here"}]})
    boxes = text_boxes(c)
    assert any(n == "route chip" for n, _ in boxes)
    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            assert not _overlap(ba, bb), (a, b)


def test_overlapping_region_labels_are_not_all_drawn() -> None:
    # four labels on one point: centre, above and below are the only candidates, so one is skipped
    same = [{"lat": 40.0, "lon": -100.0, "text": t, "size": "large"} for t in ("ONE", "TWO", "THREE", "FOUR")]
    c = comp({"center": [40.0, -100.0], "zoom": 3.0, "labels": same})
    assert len(c.labels) == 3
    boxes = [c._boxes(uv, spr, dx, dy)[-1] for uv, spr, (dx, dy) in c.labels]
    assert not any(_overlap(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:])


def test_each_river_named_once() -> None:
    split = Counter(str(p.get("name")) for p, _ in major_rivers())
    assert split["Tennessee"] > 1  # Natural Earth splits rivers into several features
    c = comp({"center": [33.0, -88.0], "zoom": 4.6})
    names = [r[0] for r in c.rivers]
    assert names and len(names) == len(set(names))


def test_title_chip_moves_off_a_marker() -> None:
    # a marker under the usual upper left title corner: the chip goes to another corner
    probe = comp({"center": [0.0, 0.0], "zoom": 3.0, "title": "A long enough title"})
    x, y = probe.title_xy[0] + 20, probe.title_xy[1] + probe.title.h / 2
    s = probe.proj.scale(3.0)
    lat, lon = -(y - H / 2) / s, (x - W / 2) / s
    c = comp({"center": [0.0, 0.0], "zoom": 3.0, "title": "A long enough title",
              "markers": [{"lat": lat, "lon": lon, "label": "Here"}]})
    assert c.title_xy != probe.title_xy
    mx, my = c.proj.px(c.markers[0][1], 3.0)[0]
    tb = _box(c.title_xy, c.title)
    assert not (tb[0] - 16 < mx < tb[2] + 16 and tb[1] - 16 < my < tb[3] + 16)


def test_locator_on_close_up_maps_only() -> None:
    close = {"center": [61.3, -159.5], "zoom": 6.0, "zoom_end": 6.3,
             "markers": [{"lat": 60.79, "lon": -161.76, "label": "Bethel"}]}
    assert comp(close).locator is not None
    assert comp(close, "transparent").inner.locator is not None  # inset panels carry it too
    assert comp({"center": [20.0, 0.0], "zoom": 1.5}).locator is None
    assert comp({"center": [40.0, -100.0], "zoom": 3.5, "zoom_end": 5.5}).locator is None  # wide at the start


def test_title_never_prints_over_a_place_name() -> None:
    """The title only repeats the drawn names: when no corner is clear of the places, it is left out."""
    corners = [(71.3, -156.8), (71.0, -60.0), (22.0, -150.0), (22.0, -66.0), (45.0, -110.0)]
    c = comp({"center": [47.0, -108.0], "zoom": 1.9, "title": "Five places",
              "markers": [{"lat": a, "lon": o, "label": f"Place {k}"} for k, (a, o) in enumerate(corners)]})
    boxes = text_boxes(c)
    title = [b for n, b in boxes if n == "title"]
    markers = [b for n, b in boxes if n == "marker"]
    assert len(markers) == 5
    assert not any(_overlap(t, m) for t in title for m in markers)
