"""loc.gov v2: archive query with a plain retry, Accept header, query classes."""
from __future__ import annotations

from test_source_adapters import canned, online, shot, tiny_jpg  # noqa: F401  (fixtures)

from vidbush.source import base
from vidbush.source.loc import LocAdapter


def _fake(monkeypatch, empty_when_year: bool) -> list:
    calls: list = []

    def get_json(url, params=None, headers=None, timeout=20.0):
        calls.append((url, dict(params or {}), dict(headers or {})))
        if url.startswith("https://www.loc.gov/item/"):
            return canned("loc_item.json")
        if empty_when_year and base.YEAR_RE.search(params["q"]):
            return {"results": []}
        return canned("loc_photos.json")

    monkeypatch.setattr(base, "http_get_json", get_json)
    return calls


def test_loc_classes() -> None:
    assert LocAdapter.classes == frozenset({"archive"}) and LocAdapter.matches_description


def test_loc_sends_archive_query_with_accept(monkeypatch, online) -> None:  # noqa: F811
    calls = _fake(monkeypatch, empty_when_year=False)
    res = LocAdapter().search("the Gatun locks under construction in 1913", "image", 5, online)
    assert len(res) == 2 and len(calls) == 1
    assert calls[0][1]["q"] == "Gatun locks construction 1913" and calls[0][2]["Accept"] == "application/json"


def test_loc_retries_without_year_only_when_empty(monkeypatch, online) -> None:  # noqa: F811
    calls = _fake(monkeypatch, empty_when_year=True)
    res = LocAdapter().search("Gatun locks 1913", "image", 5, online)
    assert [c[1]["q"] for c in calls] == ["Gatun locks 1913", "Gatun locks"] and len(res) == 2


def test_loc_item_fetch_sends_accept(monkeypatch, online, shot, tmp_path, tiny_jpg) -> None:  # noqa: F811
    calls = _fake(monkeypatch, empty_when_year=False)
    monkeypatch.setattr(base, "http_download", lambda url, dest, **kw: dest.write_bytes(tiny_jpg) and dest)
    raw = LocAdapter().search("gatun locks", "image", 5, online)[0]
    LocAdapter().fetch(raw, tmp_path, shot, online)
    item = [c for c in calls if c[0].startswith("https://www.loc.gov/item/")]
    assert item and item[0][2]["Accept"] == "application/json"

