"""Chronicling America adapter: year and phrase building, search parsing, word boxes to a 16:9 detail, fetch.

Fixtures: chronam_coords.json is a live tile.loc.gov word coordinates response (trimmed).
chronam_search.json follows the documented result shape from the LoC data-exploration notebook
(www.loc.gov answered a bot check from the dev laptop, so no search was recorded live).
"""
from __future__ import annotations

import json
from pathlib import Path

from test_source_adapters import canned, net, online, shot, tiny_jpg, tiny_mp4  # noqa: F401  (fixtures)

from vidbush.models import RealThing, Shot, ShotType
from vidbush.source import chronam
from vidbush.source.chronam import ChronamAdapter

PAGE = next(iter(canned("chronam_coords.json").values()))
W, H = float(PAGE["width"]), float(PAGE["height"])  # the service sends them as strings


def _shot(date: str) -> Shot:
    return Shot(id="s9", beat_id="b1", start=0, end=5, duration=5.0, type=ShotType.footage, intent="x",
                real_thing=RealThing(name="Woodrow Wilson", kind="person", date=date))


def test_year_span_from_query_or_real_thing() -> None:
    assert chronam.year_span("gold rush 1898") == (1897, 1899)
    assert chronam.year_span("the 1890s gold rush") == (1890, 1899)
    assert chronam.year_span("woodrow wilson", _shot("November 1924")) == (1923, 1925)
    assert chronam.year_span("gold rush 1898", _shot("1924")) == (1897, 1899)  # the query's year wins
    assert chronam.year_span("woodrow wilson") is None


def test_search_params_phrase_and_dates() -> None:
    p = chronam.search_params("Woodrow Wilson letters 1924", 5)
    assert p["ops"] == "PHRASE" and p["qs"] == "Woodrow Wilson letters" and p["dl"] == "page"
    assert p["start_date"] == "1923-01-01" and p["end_date"] == "1925-12-31" and p["fo"] == "json"
    long = chronam.search_params("when they were building the long pipeline over the mountains", 5)
    assert long["ops"] == "AND" and long["qs"] == "building long pipeline mountains" and "start_date" not in long
    assert chronam.search_params("Wilson", 5)["ops"] == "AND"
    assert chronam.search_params("Woodrow Wilson", 5, exact=False)["ops"] == "AND"


def test_search_parses_pages(net, online) -> None:  # noqa: F811
    routes, calls = net
    routes[chronam.SEARCH] = "chronam_search.json"
    res = ChronamAdapter().search("Woodrow Wilson love affairs 1924", "image", 5, online)
    assert len(calls) == 1 and calls[0][1]["qs"] == "Woodrow Wilson love affairs" and calls[0][2]["Accept"] == "application/json"
    assert [r.title for r in res] == ["Imperial Valley press (El Centro, Calif.), November 20, 1924, Image 6",
                                      "Evening star (Washington, D.C.), November 21, 1924, Image 3"]  # no image: skipped
    first = res[0]
    assert first.author == "Imperial Valley press (El Centro, Calif.)" and first.licence.startswith("Public domain")
    assert first.page_url == "https://www.loc.gov/resource/sn92070146/1924-11-20/ed-1/?sp=6"
    assert first.extra["date"] == "1924-11-20" and "LOVE AFFAIRS" in first.extra["description"]
    assert first.extra["attribution"] == ("Imperial Valley press (El Centro, Calif.), November 20, 1924, "
                                          "Chronicling America, Library of Congress")
    assert first.download_url.endswith(":1270/full/2000,/0/default.jpg")
    assert first.extra["match_words"] == ["woodrow", "wilson", "love", "affairs"]
    assert res[1].extra["coords_url"] == chronam.coords_url(res[1].extra["iiif"])  # derived when not listed
    assert first.extra["coords_url"] == chronam.coords_url(first.extra["iiif"])  # the derivation matches the listed one
    assert ChronamAdapter().search("x", "video", 5, online) == []


def test_match_boxes_pick_the_passage() -> None:
    boxes = chronam.match_boxes(PAGE["coords"], ["love", "affairs"], W)
    assert sorted(b[0] for b in boxes) == [2419.0, 3369.0]  # the headline pair, not scattered single hits
    assert chronam.match_boxes(PAGE["coords"], ["zebra"], W) == []
    assert len(chronam.match_boxes(PAGE["coords"], ["woodrow", "wilsons"], W)) == 2  # plural meets singular


def test_crop_region_is_16_9_inside_the_page() -> None:
    w, h = W, H
    boxes = chronam.match_boxes(PAGE["coords"], ["love", "affairs"], w)
    x, y, rw, rh = chronam.crop_region(boxes, w, h)
    assert abs(rw / rh - 16 / 9) < 0.01 and rw >= chronam.MIN_REGION * w - 1
    assert 0 <= x and 0 <= y and x + rw <= w and y + rh <= h
    bx, by, bw, bh = chronam._bbox(boxes)
    assert x <= bx and y <= by and bx + bw <= x + rw and by + bh <= y + rh
    edge = chronam.crop_region([(9900.0, 19950.0, 90.0, 40.0)], 10000.0, 20000.0)  # corner word: clamped in
    assert edge[0] + edge[2] == 10000 and edge[1] + edge[3] == 20000
    wide = chronam.crop_region([(0.0, 0.0, 10000.0, 100.0)], 10000.0, 20000.0)  # never wider than the page
    assert wide[2] == 10000 and abs(wide[3] - 5625) <= 1
    url = chronam.region_url("https://t/iiif/service:a", (1000, 2000, 5000, 2812), 10000.0, 20000.0)
    assert url == "https://t/iiif/service:a/pct:10.000,10.000,50.000,14.060/full/0/default.jpg"


def test_fetch_detail_then_full_page(net, online, shot, tmp_path) -> None:  # noqa: F811
    routes, calls = net
    routes[chronam.SEARCH] = "chronam_search.json"
    routes[chronam.COORDS] = "chronam_coords.json"
    raw = ChronamAdapter().search("love affairs 1924", "image", 5, online)[0]
    cands = ChronamAdapter().fetch(raw, tmp_path, shot, online)
    assert [c.title for c in cands] == [raw.title + " (detail)", raw.title]
    assert all(c.licence == chronam.LICENCE and c.media == "image" and c.width == 1920 for c in cands)
    assert cands[0].attribution == raw.extra["attribution"] and cands[0].date == "1924-11-20"
    side = [json.loads(Path(c.local_path + ".json").read_text(encoding="utf-8")) for c in cands]
    assert "/pct:" in side[0]["source_url"] and side[0]["note"] == "detail around the matched words"
    assert side[1]["source_url"] == raw.download_url and side[1]["note"] == "full newspaper page"
    assert cands[0].local_path != cands[1].local_path


def test_fetch_without_match_is_full_page_only(net, online, shot, tmp_path) -> None:  # noqa: F811
    routes, _ = net
    routes[chronam.SEARCH] = "chronam_search.json"
    routes[chronam.COORDS] = "chronam_coords.json"
    raw = ChronamAdapter().search("zebra crossing", "image", 5, online)[0]
    cands = ChronamAdapter().fetch(raw, tmp_path, shot, online)
    assert len(cands) == 1 and cands[0].title == raw.title
