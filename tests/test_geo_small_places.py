"""Small US places (GNIS) are opt-in (small=True) for explicit map names only, and near picks the homonym on
the map."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from vidbush.mg import geo

ALASKA = [(61.3, -150.1), (62.32, -150.11)]


@pytest.mark.parametrize("name", ["Coldfoot", "Deadhorse", "Toksook Bay", "Red Devil", "Nenana", "Bethel",
                                  "Talkeetna", "Wasilla", "Nome"])
def test_run2_alaska_names_resolve_near_alaska(name: str) -> None:
    hit = geo.lookup(name, near=ALASKA, small=True)
    assert hit is not None and hit.kind == "place" and hit.country == geo.US
    assert 54 < hit.lat < 72 and -170 < hit.lon < -140


def test_small_place_resolves_by_name() -> None:
    hit = geo.lookup("Coldfoot", small=True)
    assert (hit.name, hit.kind, hit.country, hit.state) == ("Coldfoot", "place", geo.US, "Alaska")
    assert any(h.state == "Alaska" for h in geo.candidates("Toksook Bay"))


def test_natural_earth_wins_without_near() -> None:
    assert geo.lookup("Paris").country == "France"
    assert geo.lookup("Victoria").country != geo.US
    assert any(h.state for h in geo.candidates("Paris"))  # small homonyms exist but do not win


def test_ambiguous_small_name_needs_near() -> None:
    assert geo.lookup("Ruby", small=True) is None  # dozens of US Rubys and no population to rank them
    assert geo.lookup("Ruby", near=ALASKA, small=True).state == "Alaska"


def test_near_picks_homonym_and_far_small_place_never_wins() -> None:
    assert geo.lookup("Hurricane", near=[(37.1, -113.3)], small=True).state == "Utah"
    assert geo.lookup("Hurricane", near=[(38.4, -82.0)], small=True).state == "West Virginia"
    assert geo.lookup("Hurricane", near=ALASKA, small=True) is None  # no Hurricane in Alaska; Utah is too far
    assert geo.lookup("Paris", near=[(33.6, -95.5)], small=True).state == "Texas"


@pytest.mark.parametrize("name,near", [("Canada", (45.0, -80.0)), ("Panama", (20.0, -85.0)),
                                       ("Mexico", (33.0, -92.0)), ("Cuba", (30.0, -85.0)), ("Peru", (40.0, -88.0))])
def test_small_place_never_beats_a_country_of_the_same_name(name: str, near: tuple) -> None:
    assert geo.lookup(name, near=[near], small=True).kind == "country"  # Canada, Ohio is never Canada


def test_free_text_never_uses_small_places() -> None:
    assert geo.find_places("Hope you enjoy this story about a Ruby and a Hurricane.") == []
    assert geo.find_places("They flew from Coldfoot to Deadhorse.") == []
    for word in ("Coldfoot", "President", "Captain", "Treaty", "Chicken"):
        assert geo.lookup(word) is None  # small places are opt-in: common words are GNIS towns
    assert geo.place_kind("Glacier") == ""
    assert geo.lookup("Chicken", small=True).state == "Alaska"


def test_small_places_load_lazily() -> None:
    code = ("from vidbush.mg import geo; geo.find_places('He sailed from Seattle to Nome.'); "
            "geo.lookup('Paris'); print(geo.small_places.cache_info().currsize)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parents[1])
    assert out.stdout.strip() == "0"
