"""Web image search (webimage) and Openverse adapters: parsing, provider choice and fetch checks.

Firecrawl and Openverse fixtures are recorded live responses; Serper, Brave and Tavily fixtures are
built from their documented response shapes (no working key for those here).
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from vidbush.config import REPO_ROOT
from vidbush.models import Shot, ShotType
from vidbush.source import base, openverse, webimage

API = REPO_ROOT / "fixtures" / "source_api"
KEYS = ("SERPER_API_KEY", "FIRECRAWL_API_KEY", "SEARCH_PROVIDER", "SEARCH_API_KEY")


def canned(name: str) -> Any:
    return json.loads((API / name).read_text(encoding="utf-8"))


def jpeg(w: int = 1920, h: int = 1080) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 120, 160)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def online(settings):
    settings.offline = False
    for k in KEYS:
        settings.env.pop(k, None)
    return settings


@pytest.fixture
def shot() -> Shot:
    return Shot(id="s001", beat_id="b001", start=0, end=5, duration=5.0, type=ShotType.footage, intent="crane")


@pytest.fixture
def net(monkeypatch):
    """Canned JSON by URL for GET and POST; downloads write state['body']. Records calls."""
    calls: list[dict] = []
    routes: dict[str, str] = {}
    state = {"body": jpeg()}

    def route(url: str) -> Any:
        for prefix, name in routes.items():
            if url.startswith(prefix):
                return canned(name)
        raise AssertionError(f"unexpected url {url}")

    def get_json(url, params=None, headers=None, timeout=20.0):
        calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        return route(url)

    def post_json(url, payload, headers=None, timeout=20.0):
        calls.append({"url": url, "payload": dict(payload), "headers": dict(headers or {})})
        return route(url)

    def download(url, dest: Path, headers=None, timeout=60.0, max_bytes=0) -> Path:
        calls.append({"url": url, "headers": dict(headers or {})})
        dest.write_bytes(state["body"])
        return dest

    monkeypatch.setattr(base, "http_get_json", get_json)
    monkeypatch.setattr(webimage, "http_post_json", post_json)
    monkeypatch.setattr(base, "http_download", download)
    return routes, calls, state


def test_provider_order_and_availability(online) -> None:
    a = webimage.WebImageAdapter()
    assert webimage.pick_provider(online) is None and not a.available(online)
    online.env["SEARCH_API_KEY"] = "s"
    assert webimage.pick_provider(online) == ("tavily", "s")  # the web_search seat type
    online.env["SEARCH_PROVIDER"] = "brave"
    assert webimage.pick_provider(online)[0] == "brave"
    online.env["SEARCH_PROVIDER"] = "bing"
    assert webimage.pick_provider(online) is None  # unsupported seat type
    online.env["FIRECRAWL_API_KEY"] = "f"
    assert webimage.pick_provider(online) == ("firecrawl", "f") and a.available(online)
    online.env["SERPER_API_KEY"] = "p"
    assert webimage.pick_provider(online) == ("serper", "p")
    online.offline = True
    assert not a.available(online)


def test_firecrawl_recorded(net, online) -> None:
    routes, calls, _ = net
    routes[webimage.FIRECRAWL] = "firecrawl_images_anchorage.json"
    online.env["FIRECRAWL_API_KEY"] = "f"
    res = webimage.WebImageAdapter().search("Anchorage Daily Times 1958 statehood", "image", 5, online)
    assert len(res) == 5 and calls[0]["headers"]["Authorization"] == "Bearer f"
    assert calls[0]["payload"]["sources"] == [{"type": "images"}]
    r = res[0]
    assert r.url == r.download_url == "https://vilda.alaska.edu/iiif/2/cdmg2:4837/full/720,/0/default.jpg"
    assert r.page_url == "https://vilda.alaska.edu/digital/collection/cdmg2/id/4837/"
    assert (r.width, r.height) == (720, 508) and r.title.startswith("Copy of newspaper headlines")
    assert r.author == "vilda.alaska.edu" and r.licence == "unknown (web image search)"
    assert r.extra["provider"] == "firecrawl" and r.extra["position"] == 1 and r.extra["site"] == "vilda.alaska.edu"
    assert r.extra["attribution"].endswith("(vilda.alaska.edu, via web image search)")
    assert webimage.WebImageAdapter().search("x", "video", 5, online) == []


def test_serper_documented_shape(net, online) -> None:
    routes, calls, _ = net
    routes[webimage.SERPER] = "serper_images.json"
    online.env["SERPER_API_KEY"] = "p"
    res = webimage.WebImageAdapter().search("harbour crane", "image", 5, online)
    assert [r.extra["position"] for r in res] == [1, 4]  # 128 px icon and x-raw-image dropped
    assert calls[0]["headers"]["X-API-KEY"] == "p" and calls[0]["payload"]["q"] == "harbour crane"
    assert res[0].extra["site"] == "museum.example.org"
    assert res[0].page_url == "https://www.museum.example.org/collection/crane-1950"


def test_brave_documented_shape(net, online) -> None:
    routes, calls, _ = net
    routes[webimage.BRAVE] = "brave_images.json"
    online.env.update({"SEARCH_PROVIDER": "brave", "SEARCH_API_KEY": "b"})
    res = webimage.WebImageAdapter().search("harbour crane", "image", 5, online)
    assert len(res) == 1  # data URI dropped
    assert res[0].download_url == "https://archive.example.net/files/crane.jpg"
    assert res[0].page_url == "https://archive.example.net/item/crane" and res[0].width == 0
    assert calls[0]["headers"]["X-Subscription-Token"] == "b" and calls[0]["params"]["q"] == "harbour crane"


def test_tavily_documented_shape(net, online) -> None:
    routes, calls, _ = net
    routes[webimage.TAVILY] = "tavily_images.json"
    online.env["SEARCH_API_KEY"] = "t"
    res = webimage.WebImageAdapter().search("harbour crane", "image", 5, online)
    assert len(res) == 2 and calls[0]["payload"]["include_image_descriptions"] is True
    assert res[0].extra["description"].startswith("A tall harbour crane") and res[0].page_url == ""
    assert res[0].author == "photos.example.org"


def test_fetch_rejects_non_image(net, online, shot, tmp_path) -> None:
    _, _, state = net
    state["body"] = b"<!doctype html><html><body>Hotlinking not allowed</body></html>"
    raw = webimage._result("serper", "https://img.example.com/a.jpg", "A", "https://example.com/p", 800, 600, 1)
    with pytest.raises(base.Rejected, match="not an image"):
        webimage.WebImageAdapter().fetch(raw, tmp_path, shot, online)
    state["body"] = jpeg(200, 150)
    small = webimage._result("serper", "https://img.example.com/b.jpg", "B", "", 0, 0, 2)
    with pytest.raises(base.Rejected, match="too small"):
        webimage.WebImageAdapter().fetch(small, tmp_path, shot, online)


def test_fetch_accepts_jpeg(net, online, shot, tmp_path) -> None:
    _, calls, _ = net
    raw = webimage._result("tavily", "https://img.example.com/photo?id=7", "Crane", "https://example.com/p", 0, 0, 1,
                           description="A crane at dusk")
    cands = webimage.WebImageAdapter().fetch(raw, tmp_path, shot, online)
    c = cands[0]
    assert calls[-1]["headers"]["Referer"] == "https://example.com/p" and "image/" in calls[-1]["headers"]["Accept"]
    assert Path(c.local_path).suffix == ".jpg" and (c.width, c.height) == (1920, 1080)
    assert c.licence == "unknown (web image search)" and "licence_unknown" in c.flags
    assert c.attribution == "Crane (example.com, via web image search)" and c.description == "A crane at dusk"
    assert c.page_url == "https://example.com/p"


def test_openverse_recorded(net, online, shot, tmp_path) -> None:
    routes, calls, _ = net
    routes[openverse.IMAGES] = "openverse_images.json"
    a = openverse.OpenverseAdapter()
    assert a.available(online) and a.matches_description
    res = a.search("sled dog race", "image", 5, online)
    assert calls[0]["params"]["mature"] == "false" and len(res) == 4
    r = res[0]
    assert r.title == "Sled Dog Race" and r.author == "misterbisson" and r.licence == "CC BY-NC-SA 2.0"
    assert r.page_url.startswith("https://www.flickr.com/photos/") and r.download_url.endswith(".jpg")
    assert r.extra["provider"] == "flickr" and r.extra["tags"] and all(isinstance(t, str) for t in r.extra["tags"])
    assert r.extra["attribution"].startswith('"Sled Dog Race" by misterbisson')
    assert res[1].licence == "CC BY-SA 4.0"
    cands = a.fetch(r, tmp_path, shot, online)
    assert cands[0].licence == "CC BY-NC-SA 2.0" and cands[0].attribution == r.extra["attribution"]
    assert a.search("x", "video", 5, online) == []


def test_openverse_licence_names() -> None:
    from vidbush.source.licence import normalise_licence
    assert openverse.licence_name("pdm", "1.0") == "Public domain"
    assert openverse.licence_name("cc0", "1.0") == "CC0 1.0"
    assert openverse.licence_name("by", "4.0") == "CC BY 4.0"
    for code, ver in (("pdm", "1.0"), ("cc0", "1.0"), ("by-sa", "2.0"), ("by-nc-nd", "3.0")):
        name = openverse.licence_name(code, ver)
        assert normalise_licence(name) == name
    pd = openverse.parse({"results": [{"url": "https://x.org/a.jpg", "license": "pdm", "title": "A"},
                                      {"url": "https://x.org/b.jpg", "license": "by", "license_version": "2.0"},
                                      {"url": "https://x.org/c.jpg", "license": "by", "mature": True}]})
    assert len(pd) == 2 and pd[0].score > pd[1].score  # public domain slightly preferred, nothing dropped but mature
