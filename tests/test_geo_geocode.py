"""Offline geocoding and place extraction."""
from __future__ import annotations

import json

import pytest

from vidbush.mg import geo


@pytest.mark.parametrize("name,lat,lon", [
    ("Panama City", 8.97, -79.53), ("Colon", 9.36, -79.90), ("Colón", 9.36, -79.88), ("Cape Horn", -55.98, -67.27),
    ("Hoover Dam", 36.02, -114.74), ("Boulder Dam", 36.02, -114.74), ("New York", 40.75, -73.98),
    ("Buffalo", 42.89, -78.88), ("Albany", 42.65, -73.76), ("Gatun Locks", 9.27, -79.92),
    ("Miraflores Locks", 9.0, -79.59), ("the Alamo", 29.43, -98.49), ("Promontory Summit", 41.62, -112.55),
    ("Washington, D.C.", 38.91, -77.04), ("panama canal", 9.12, -79.75),
])
def test_geocode_known_places(name: str, lat: float, lon: float) -> None:
    got = geo.geocode(name)
    assert got is not None, name
    assert geo.distance_deg(got, (lat, lon)) < 0.3, (name, got)


def test_countries_and_unknowns() -> None:
    hit = geo.lookup("France")
    assert hit.kind == "country" and hit.iso_a3 == "FRA"
    assert geo.lookup("USA").name == "United States of America"
    assert geo.is_country("Panama") and not geo.is_country("Panama City")
    assert geo.geocode("Qwertyville Nowhere") is None
    box = geo.country_bbox("Panama")
    assert box and box[0] < 9 < box[2] and box[1] < -79.5 < box[3]


def test_us_bias_prefers_american_place() -> None:
    assert geo.lookup("Victoria").country != geo.US
    assert geo.lookup("Victoria", us_bias=True).country == geo.US


def test_landmarks_file_is_complete_and_valid() -> None:
    lms = json.loads((geo.GEO_DIR / "landmarks.json").read_text(encoding="utf-8"))
    assert len(lms) >= 80
    names = {lm["name"] for lm in lms}
    for required in ("Panama Canal", "Gatun Lake", "Culebra Cut", "Strait of Magellan", "Lake Mead", "Erie Canal",
                     "Suez Canal", "Golden Gate Bridge", "Pearl Harbor", "Kitty Hawk", "Cape Canaveral", "Yellowstone"):
        assert required in names
    for lm in lms:
        assert -90 <= lm["lat"] <= 90 and -180 <= lm["lon"] <= 180


def _names(text: str) -> list[str]:
    return [p[0] for p in geo.find_places(text)]


def test_find_places_sentences() -> None:
    assert _names("In 1914, a ship sailing from New York to San Francisco no longer had to round Cape Horn.") == [
        "New York", "San Francisco", "Cape Horn"]
    assert _names("Ships entered the Gatun Locks near Colon and crossed Gatun Lake toward the Culebra Cut.") == [
        "Gatun Locks", "Colon", "Gatun Lake", "Culebra Cut"]
    assert _names("Barges left Buffalo for Albany along the Erie Canal, then sailed down the Hudson River.") == [
        "Buffalo", "Albany", "Erie Canal", "Hudson River"]
    assert "South America" in _names("They fought their way around the stormy tip of South America.")


def test_find_places_skips_people_months_and_common_words() -> None:
    assert _names("In March, Theodore Roosevelt met Ferdinand de Lesseps.") == []
    assert _names("Lincoln spoke at Gettysburg in November.") == ["Gettysburg"]
    assert _names("The Hoover Dam rose in the Black Canyon.") == ["Hoover Dam", "Black Canyon"]
    assert _names("President Jimmy Carter signed the treaties with Panama.") == ["Panama"]
    assert _names("Panama, Panama and Panama again.") == ["Panama"]
