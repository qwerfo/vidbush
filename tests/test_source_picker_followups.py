"""Picker follow-ups at sourcing: stock agency hosts dropped, small stills dropped unless public domain
archival, and newspaper-class queries phrased for a newspaper page by web image search (flag newspaper_page)."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from vidbush.models import Candidate, Shot, ShotType, SourceKind
from vidbush.source import base, registry, webimage

KEYS = ("SERPER_API_KEY", "FIRECRAWL_API_KEY", "SEARCH_PROVIDER", "SEARCH_API_KEY")


def jpeg(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 120, 160)).save(buf, "JPEG")
    return buf.getvalue()


def _shot(**kw: Any) -> Shot:
    return Shot(id="s001", beat_id="b001", start=0, end=5, duration=5.0, type=ShotType.image, intent="a page", **kw)


@pytest.fixture
def online(settings):
    settings.offline = False
    for k in KEYS:
        settings.env.pop(k, None)
    settings.env["SERPER_API_KEY"] = "p"
    return settings


@pytest.fixture
def serper(monkeypatch):
    """Serper answers with state['images']; downloads write state['body']. Records POST payloads."""
    state: dict[str, Any] = {"images": [], "body": jpeg(1600, 1000), "payloads": []}

    def post_json(url, payload, headers=None, timeout=20.0):
        assert url == webimage.SERPER
        state["payloads"].append(dict(payload))
        images = state["images"]
        return {"images": images(payload) if callable(images) else images}

    def download(url, dest: Path, headers=None, timeout=60.0, max_bytes=0) -> Path:
        dest.write_bytes(state["body"])
        return dest

    monkeypatch.setattr(webimage, "http_post_json", post_json)
    monkeypatch.setattr(base, "http_download", download)
    return state


def _img(url: str, link: str, w: int = 1600, h: int = 1000, pos: int = 1) -> dict[str, Any]:
    return {"imageUrl": url, "link": link, "title": "t", "imageWidth": w, "imageHeight": h, "position": pos}


# ---------------------------------------------------------------- stock agencies

@pytest.mark.parametrize("url,stock", [
    ("https://www.shutterstock.com/image-photo/x", True),
    ("https://image.shutterstock.com/z/a.jpg", True),
    ("https://media.gettyimages.com/id/1/photo.jpg", True),
    ("https://c8.alamy.com/comp/A/a.jpg", True),
    ("https://t3.ftcdn.net/jpg/01/a.jpg", True),
    ("https://as1.ftcdn.net/v2/jpg/a.jpg", True),
    ("https://stock.adobe.com/images/a/1", True),
    ("https://img.freepik.com/free-photo/a.jpg", True),
    ("https://www.adobe.com/products/photoshop", False),
    ("https://notshutterstock.com/a.jpg", False),
    ("https://museum.example.org/a.jpg", False),
    ("", False),
])
def test_stock_host_matches_by_suffix(url: str, stock: bool) -> None:
    assert webimage.stock_host(url) is stock


def test_stock_page_or_image_is_dropped_at_search(serper, online) -> None:
    serper["images"] = [
        _img("https://museum.example.org/a.jpg", "https://museum.example.org/a", pos=1),
        _img("https://cdn.example.net/b.jpg", "https://www.alamy.com/stock-photo-b", pos=2),  # stock page
        _img("https://media.gettyimages.com/id/3/c.jpg", "https://news.example.com/c", pos=3),  # stock CDN
    ]
    res = webimage.WebImageAdapter().search("harbour crane", "image", 5, online)
    assert [r.extra["position"] for r in res] == [1]


# ---------------------------------------------------------------- small stills

def test_web_image_minimum_is_500_px_at_search_and_fetch(serper, online, tmp_path) -> None:
    assert webimage.MIN_LONG_SIDE == 500
    serper["images"] = [_img("https://a.example.org/1.jpg", "", 480, 320, 1),
                        _img("https://a.example.org/2.jpg", "", 500, 300, 2),
                        _img("https://a.example.org/3.jpg", "", 0, 0, 3)]  # size unknown: checked at fetch
    res = webimage.WebImageAdapter().search("q", "image", 5, online)
    assert [r.extra["position"] for r in res] == [2, 3]
    serper["body"] = jpeg(480, 300)
    with pytest.raises(base.Rejected, match="too small"):
        webimage.WebImageAdapter().fetch(res[1], tmp_path, _shot(), online)


def test_finalize_drops_small_stills_unless_public_domain_archival(make_ctx, monkeypatch, tmp_path) -> None:
    from vidbush.source import keyframes
    ctx = make_ctx(mode="script")
    monkeypatch.setattr(keyframes, "extract_keyframes",
                        lambda c, d, **kw: setattr(c, "keyframes", [c.local_path]) or c.keyframes)

    def cand(name: str, src: SourceKind, w: int, h: int, licence: str, file_size=(1600, 1000)) -> Candidate:
        p = tmp_path / f"{name}.png"
        Image.new("RGB", file_size, (len(name) * 40 % 255, 60, 90)).save(p)
        return Candidate(id=name, source=src, local_path=str(p), media="image", width=w, height=h, licence=licence)

    small_web = cand("web", SourceKind.webimage, 480, 320, "unknown (web image search)")
    small_cc = cand("cc", SourceKind.wikimedia, 400, 300, "CC BY-SA 4.0")
    small_pd = cand("pd", SourceKind.loc, 450, 300, "Public domain")
    unsized_small = cand("unsized", SourceKind.pexels, 0, 0, "Pexels License", file_size=(300, 200))
    big = cand("big", SourceKind.pexels, 1920, 1080, "Pexels License", file_size=(1920, 1080))
    out = registry.finalize(_shot(), [small_web, small_cc, small_pd, unsized_small, big], ctx)
    assert [c.source for c in out] == [SourceKind.loc, SourceKind.pexels]
    assert "small_archival" in out[0].flags and out[1].width == 1920
    own = cand("own", SourceKind.local, 400, 300, "own library")
    assert [c.source for c in registry.finalize(_shot(), [own], ctx)] == [SourceKind.local]  # the operator's own picture stays


# ---------------------------------------------------------------- newspaper pages

@pytest.mark.parametrize("q,sent", [
    ("statehood vote 1958", "statehood vote 1958 newspaper page"),
    ("Daily Times newspaper 1958", "Daily Times newspaper 1958"),
    ("strike front page 1934", "strike front page 1934"),
    ("Headline about the flood", "Headline about the flood"),
    ("Gazette notice 1802", "Gazette notice 1802"),
])
def test_newspaper_query_phrasing(q: str, sent: str) -> None:
    assert webimage.newspaper_query(q) == sent
    a = webimage.WebImageAdapter()
    assert a.class_query(q, "newspaper") == sent and a.class_query(q, "web_image") == q


def test_newspaper_class_is_phrased_recorded_and_flagged(serper, online, make_ctx, monkeypatch, tmp_path) -> None:
    ctx = make_ctx(mode="script")
    ctx.settings.offline = False
    ctx.settings.env.update(online.env)
    monkeypatch.setattr(registry, "cache_get", lambda *a, **k: None)
    monkeypatch.setattr(registry, "cache_put", lambda *a, **k: None)
    serper["images"] = lambda payload: [_img(f"https://news.example.org/{payload['q'].replace(' ', '_')}.jpg",
                                             "https://news.example.org/page")]
    a = webimage.WebImageAdapter()
    shot = _shot(source_queries={"newspaper": ["statehood vote 1958"], "web_image": ["statehood parade 1958"]})
    with registry.SearchPools([a]) as pools:
        found = registry._search_all([shot], [a], ctx, pools)
    assert sorted(p["q"] for p in serper["payloads"]) == ["statehood parade 1958",
                                                          "statehood vote 1958 newspaper page"]
    res = found[shot.id]["webimage"]
    by_class = {r.extra["query_class"]: r for r in res}
    news = by_class["newspaper"]
    assert news.extra["query"] == "statehood vote 1958"  # the planned query, as routed
    assert by_class["web_image"].extra["query"] == "statehood parade 1958"
    c = a.fetch(news, tmp_path, shot, ctx.settings)[0]
    assert "newspaper_page" in c.flags and c.found_by == "webimage:newspaper"
    side = json.loads(Path(c.local_path + ".json").read_text(encoding="utf-8"))
    assert "newspaper_page" in side["flags"]
    assert "newspaper_page" not in a.fetch(by_class["web_image"], tmp_path / "w", shot, ctx.settings)[0].flags
