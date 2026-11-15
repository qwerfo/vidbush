"""US National Archives adapter: parsing recorded API v2 responses, licence gating, object choice, fetch."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vidbush.models import Shot, ShotType, SourceKind
from vidbush.source import base, nara, registry
from vidbush.source.nara import NaraAdapter

FIX = Path(__file__).parent / "fixtures"


def recorded(name: str) -> Any:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


@pytest.fixture
def online(settings):
    settings.offline = False
    settings.env["NARA_API_KEY"] = "test-key"
    return settings


def test_available_only_with_key(settings) -> None:
    """Without the key the API answers 200 with the catalogue web page, not JSON."""
    settings.offline = False
    assert not NaraAdapter().available(settings)
    settings.env["NARA_API_KEY"] = "k"
    assert NaraAdapter().available(settings)
    settings.offline = True
    assert not NaraAdapter().available(settings)


def test_photos_parse_with_public_domain_licence() -> None:
    out = nara.parse_search(recorded("nara_search_photos.json"), "image")
    assert [r.title.split(" (")[0] for r in out] == ["Resting sled dog", "Local resident with sled dog",
                                                     "Local resident working on dog sled harness"]
    r = out[0]
    assert r.source == "nara" and r.media == "image" and r.licence == nara.LICENCE
    assert r.page_url.startswith("https://catalog.archives.gov/id/") and r.download_url.endswith(".jpg")
    assert SourceKind(r.source) is SourceKind.nara


def test_films_keep_only_unrestricted_records_with_a_real_mp4() -> None:
    data = recorded("nara_search_films.json")
    statuses = {str(h["_source"]["record"]["naId"]): h["_source"]["record"]["useRestriction"]["status"]
                for h in data["body"]["hits"]["hits"]}
    assert {"Restricted - Possibly", "Undetermined", "Unrestricted"} <= set(statuses.values())
    out = nara.parse_search(data, "video")
    kept = {r.extra["naId"] for r in out}
    assert kept and all(statuses[k] == "Unrestricted" for k in kept)
    assert kept == {"78882"}  # 78857 is unrestricted too, but its only MP4 is a 77 kB preview
    assert out[0].download_url.endswith(".mp4") and out[0].extra["size"] > nara.MIN_VIDEO_BYTES


def test_film_record_with_only_shot_list_sheets_is_not_a_video() -> None:
    assert nara.parse_search(recorded("nara_search_film_sheets.json"), "video") == []


def test_best_object_prefers_jpeg_under_the_cap() -> None:
    rec = {"digitalObjects": [
        {"objectUrl": "https://x/a.tif", "objectType": "Image (TIFF)", "objectFileSize": 9_000_000},
        {"objectUrl": "https://x/b.jpg", "objectType": "Image (JPG)", "objectFileSize": 2_000_000},
        {"objectUrl": "https://x/c.jpg", "objectType": "Image (JPG)", "objectFileSize": base.MAX_IMAGE_BYTES + 1}]}
    assert nara.best_object(rec, "image")["objectUrl"] == "https://x/b.jpg"
    rec["digitalObjects"] = rec["digitalObjects"][:1]
    assert nara.best_object(rec, "image")["objectUrl"] == "https://x/a.tif"
    assert nara.best_object(rec, "video") is None


def test_search_sends_key_short_query_and_material_type(monkeypatch, online) -> None:
    calls: list[tuple[str, dict, dict]] = []

    def get_json(url: str, params=None, headers=None, timeout: float = 20.0) -> Any:
        calls.append((url, dict(params or {}), dict(headers or {})))
        return recorded("nara_search_photos.json")

    monkeypatch.setattr(base, "http_get_json", get_json)
    out = NaraAdapter().search("archival photo of a sled dog resting in the snow", "image", 2, online)
    assert len(out) == 2
    url, params, headers = calls[0]
    assert url == nara.SEARCH and headers["x-api-key"] == "test-key"
    assert params["q"] == "sled dog resting" and params["typeOfMaterials"] == nara.PHOTOS
    NaraAdapter().search("Alaska film", "video", 2, online)
    assert calls[1][1]["typeOfMaterials"] == nara.FILMS


def test_fetch_photo_and_section_of_large_film(monkeypatch, online, tmp_path) -> None:
    from PIL import Image
    shot = Shot(id="s001", beat_id="b001", start=0, end=5, duration=5.0, type=ShotType.image, intent="dog")

    def download(url: str, dest: Path, headers=None, timeout: float = 60.0, max_bytes: int = 0) -> Path:
        Image.new("RGB", (1600, 1200), (90, 90, 90)).save(dest, "JPEG")
        return dest

    monkeypatch.setattr(base, "http_download", download)
    photo = nara.parse_search(recorded("nara_search_photos.json"), "image")[0]
    cands = NaraAdapter().fetch(photo, tmp_path / "p", shot, online)
    assert len(cands) == 1 and cands[0].source is SourceKind.nara and cands[0].licence.startswith("Public domain")
    sidecar = json.loads(Path(cands[0].local_path + ".json").read_text(encoding="utf-8"))
    assert sidecar["page_url"] == photo.page_url and "US National Archives" in sidecar["attribution"]

    sections: list[tuple[str, float]] = []
    monkeypatch.setattr(base, "cached_video", lambda url, ext, s, d, duration=0.0, size=0.0:
                        sections.append((url, size)) or _tiny_video(tmp_path))
    monkeypatch.setattr(base, "detect_scenes", lambda p: [(0.0, 8.0)])
    film = nara.parse_search(recorded("nara_search_films.json"), "video")[0]
    cands = NaraAdapter().fetch(film, tmp_path / "f", shot.model_copy(update={"type": ShotType.footage}), online)
    assert sections == [(film.download_url, film.extra["size"])] and cands and cands[0].media == "video"


def _tiny_video(tmp: Path) -> Path:
    p = tmp / "tiny.mp4"
    p.write_bytes(b"\x00" * 16)
    return p


def test_registered_and_enabled_next_to_loc(settings) -> None:
    assert "nara" in registry._factories()
    enabled = settings.sources.enabled
    assert enabled.index("nara") == enabled.index("loc") + 1
    from vidbush.source import throttle
    assert throttle.DEFAULT_BUDGET["nara"] > 0 and "catalog.archives.gov" in throttle.DEFAULT_RATES
