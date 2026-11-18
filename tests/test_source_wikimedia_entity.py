"""Wikimedia by entity: Wikidata item first (P18, then its Commons category), text search as fallback.

Responses are recorded live (wbsearchentities "Balto", wbgetentities, the P18 imageinfo and the
Category:Balto members), trimmed, in fixtures/source_api/wikidata_balto.json.
"""
from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from vidbush.config import REPO_ROOT
from vidbush.models import RealThing, Shot, ShotType
from vidbush.providers.base import TransportError
from vidbush.source import base, wikimedia
from vidbush.source.wikimedia import WikimediaAdapter

API = REPO_ROOT / "fixtures" / "source_api"
BALTO = json.loads((API / "wikidata_balto.json").read_text(encoding="utf-8"))


def _canned(name: str) -> Any:
    return json.loads((API / name).read_text(encoding="utf-8"))


@pytest.fixture
def online(settings):
    settings.offline = False
    return settings


@pytest.fixture
def wiki(monkeypatch):
    """Routes Wikidata and Commons calls by their parameters; text search answers with the image fixture."""
    calls: list[dict] = []
    state = {"search": BALTO["search"], "text": "commons_search_image.json", "fail": False}

    def get_json(url: str, params=None, headers=None, timeout: float = 20.0):
        p = dict(params or {})
        calls.append({"url": url, **p})
        if url == wikimedia.WIKIDATA_API:
            if state["fail"]:
                raise TransportError("wikidata unreachable")
            if p["action"] == "wbsearchentities":
                return state["search"] if p["search"] == "Balto" else _canned("wikidata_search_empty.json")
            return copy.deepcopy(BALTO["entities"])
        assert url == wikimedia.API
        if "titles" in p:
            return copy.deepcopy(BALTO["p18"])
        if p.get("generator") == "categorymembers":
            return copy.deepcopy(BALTO["category"])
        assert p.get("generator") == "search"
        return _canned(state["text"])

    monkeypatch.setattr(base, "http_get_json", get_json)
    return calls, state


def _shot(name: str = "Balto", kind: str = "", date: str = "") -> Shot:
    return Shot(id="s001", beat_id="b001", start=0, end=5, duration=5.0, type=ShotType.footage, intent=name,
                real_thing=RealThing(name=name, kind=kind, date=date))


def test_entity_path_gives_p18_then_category_files(wiki, online) -> None:
    calls, _ = wiki
    res = WikimediaAdapter().search("sled dog hero of the serum run", "image", 10, online, shot=_shot(kind="object", date="1925"))
    titles = [r.title for r in res]
    assert titles[0] == "Gunnar Kaasen with Balto" and res[0].score > 0.9  # P18 first
    assert all(0.85 <= r.score < 0.9 for r in res[1:]) and len(res) == 4  # jpgs only: wav and webm skipped
    assert "Balto 1925 (cropped)" in titles
    assert {r.extra["entity"] for r in res} == {"Q805807"} and res[0].extra["entity_label"] == "Balto"
    assert res[0].extra["date"] == "circa 1925"  # hidden machine date data stripped
    assert all(r.licence == "Public domain" and r.media == "image" for r in res)
    assert not any(c.get("generator") == "search" for c in calls) and len(calls) == 4
    cat = next(c for c in calls if c.get("generator") == "categorymembers")
    assert cat["gcmtitle"] == "Category:Balto" and "extmetadata" in cat["iiprop"]


def test_results_keep_at_most_n(wiki, online) -> None:
    assert len(WikimediaAdapter().search("Balto", "image", 2, online, shot=_shot(kind="object"))) == 2


def test_kind_prefers_the_dog_over_the_film() -> None:
    ents = BALTO["entities"]["entities"]
    dog, film = ents["Q805807"], ents["Q764116"]
    thing = RealThing(name="Balto", kind="object", date="1925")
    assert wikimedia.entity_score("Balto", film, thing) < wikimedia.entity_score("Balto", film, None)
    assert wikimedia.entity_score("Balto", dog, thing) > wikimedia.entity_score("Balto", film, thing)
    assert wikimedia.entity_score("Balto", ents["Q30088844"], thing) == 0  # a family name is never the thing
    assert wikimedia.entity_score("Balto", ents["Q4853006"], thing) == 0  # Baltonsborough is not Balto


def test_resolve_picks_the_dog_and_long_names_try_their_proper_name(wiki, online) -> None:
    calls, _ = wiki
    res = WikimediaAdapter().search("Balto statue", "image", 20, online, shot=_shot("Balto statue", kind="object"))
    searched = [c["search"] for c in calls if c.get("action") == "wbsearchentities"]
    assert searched == ["Balto statue", "Balto"]
    # found through the fragment "Balto": the text search for the whole name leads, the dog's media follow
    assert any(r.extra.get("entity") == "Q805807" for r in res)
    assert not res[0].extra.get("entity") or len({r.extra.get("entity") for r in res}) == 1
    named = _shot("Balto sled dog in Nome", kind="object")
    assert wikimedia.entity_names("Balto sled dog in Nome", named) == ["Balto sled dog in Nome", "Balto"]
    assert wikimedia.entity_names("Balto sled dog in Nome", None) == []  # a description, no real thing
    assert wikimedia.proper_name_run("the Port of Nome Alaska and Balto") == "Nome Alaska"  # longest run
    assert wikimedia.proper_name_run("photo of Leonhard Seppala with dogs") == "Leonhard Seppala"


def test_no_entity_falls_back_to_text_search(wiki, online) -> None:
    calls, state = wiki
    state["search"] = _canned("wikidata_search_empty.json")
    res = WikimediaAdapter().search("Balto", "image", 5, online)
    assert [r.title for r in res] == ["Culebra Cut, Panama Canal, 1913", "Gatun Locks construction"]
    text = [c for c in calls if c.get("generator") == "search"]
    assert len(text) == 1 and text[0]["gsrsearch"] == "Balto filetype:bitmap" and text[0]["gsrnamespace"] == 6
    assert res[0].extra["description"] == "Steam shovels at work in the Culebra Cut."
    assert all(r.score == pytest.approx(0.82) for r in res)  # public domain and CC get the small tie bonus


def test_wikidata_failure_falls_back_to_text_search(wiki, online) -> None:
    calls, state = wiki
    state["fail"] = True
    assert WikimediaAdapter().search("Balto", "image", 5, online)
    assert calls[-1].get("generator") == "search"


def test_video_media_filters_mime(wiki, online) -> None:
    calls, state = wiki
    res = WikimediaAdapter().search("Balto", "video", 5, online, shot=_shot(kind="object"))
    assert [r.media for r in res] == ["video"] and res[0].url.endswith(".webm")
    assert res[0].extra["entity"] == "Q805807"
    assert not any("titles" in c for c in calls)  # P18 is a still: not fetched for video
    # an entity without videos falls back to the text search with filetype:video
    calls.clear()
    BALTO_NO_VIDEO = copy.deepcopy(BALTO["category"])
    pages = BALTO_NO_VIDEO["query"]["pages"]
    for k in [k for k, p in pages.items() if p["imageinfo"][0]["mime"].startswith("video")]:
        del pages[k]
    state["text"] = "commons_search_video.json"
    orig = BALTO["category"]
    BALTO["category"] = BALTO_NO_VIDEO
    try:
        res = WikimediaAdapter().search("Balto", "video", 5, online, shot=_shot(kind="object"))
    finally:
        BALTO["category"] = orig
    assert len(res) == 1 and res[0].duration == 94.5
    text = [c for c in calls if c.get("generator") == "search"]
    assert text and text[0]["gsrsearch"].endswith("filetype:video")


def test_strip_html_drops_hidden_data() -> None:
    raw = 'circa 1994<div style="display: none;">date QS:P,+1994-00-00T00:00:00Z/9</div>'
    assert wikimedia.strip_html(raw) == "circa 1994"


def test_adapter_classes() -> None:
    a = WikimediaAdapter()
    assert a.classes == frozenset({"encyclopedic", "archive"}) and a.matches_description


def test_thin_category_lends_its_first_subcategory(wiki, online) -> None:
    calls, _ = wiki
    cat = copy.deepcopy(BALTO["category"])
    cat["query"]["pages"]["-1"] = {"ns": 14, "title": "Category:Statue of Balto"}
    orig, BALTO["category"] = BALTO["category"], cat
    try:
        WikimediaAdapter().search("Balto", "image", 10, online, shot=_shot(kind="object"))
    finally:
        BALTO["category"] = orig
    members = [c for c in calls if c.get("generator") == "categorymembers"]
    assert [c["gcmtitle"] for c in members] == ["Category:Balto", "Category:Statue of Balto"]
    assert members[1]["gcmtype"] == "file" and len(calls) <= wikimedia.MAX_REQUESTS


def test_entity_names_try_the_query_before_name_fragments() -> None:
    from vidbush.models import RealThing, Shot, ShotType
    shot = Shot(id="s", beat_id="b", start=0, end=4, duration=4, type=ShotType.image, intent="x",
                real_thing=RealThing(name="abandoned British equipment on Dunkirk beach", kind="object"))
    assert wikimedia.entity_names("Dunkirk evacuation", shot)[:2] == [
        "abandoned British equipment on Dunkirk beach", "Dunkirk evacuation"]


def test_a_people_or_language_is_never_the_thing() -> None:
    ent = {"labels": {"en": {"value": "British"}}, "claims": {"P31": [
        {"mainsnak": {"datavalue": {"value": {"id": "Q41710"}}}}]}}
    assert wikimedia.entity_score("British", ent, None) == 0.0


def test_a_single_word_never_matches_a_longer_label() -> None:
    ent = {"labels": {"en": {"value": "British Indian Empire"}}, "claims": {}}
    assert wikimedia._label_match("British", ent) == 0.0
    assert wikimedia._label_match("Balto statue", {"labels": {"en": {"value": "Balto"}}, "claims": {}}) == 1.0
