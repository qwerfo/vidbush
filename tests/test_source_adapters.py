"""Every web adapter parses its canned API response and fetches through monkeypatched HTTP."""
from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from vidbush.config import REPO_ROOT
from vidbush.models import Shot, ShotType
from vidbush.source import base

API = REPO_ROOT / "fixtures" / "source_api"


def canned(name: str) -> Any:
    return json.loads((API / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tiny_mp4(tmp_path_factory) -> bytes:
    out = tmp_path_factory.mktemp("vid") / "tiny.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=4",
                    "-c:v", "libx264", "-preset", "ultrafast", "-g", "15", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out.read_bytes()


@pytest.fixture(scope="module")
def tiny_jpg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1920, 1080), (90, 120, 160)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def online(settings):
    settings.offline = False
    return settings


@pytest.fixture
def shot() -> Shot:
    return Shot(id="s001", beat_id="b001", start=0, end=5, duration=5.0, type=ShotType.footage, intent="ships")


@pytest.fixture
def net(monkeypatch, tiny_mp4: bytes, tiny_jpg: bytes):
    """Route JSON calls by URL to canned files, bytes calls to tiny media. Records calls."""
    calls: list[tuple[str, dict]] = []
    routes: dict[str, str] = {}

    def get_json(url: str, params=None, headers=None, timeout: float = 20.0):
        calls.append((url, dict(params or {}), dict(headers or {})))
        for prefix, name in routes.items():
            if url.startswith(prefix):
                return canned(name)
        raise AssertionError(f"unexpected url {url}")

    def download(url: str, dest: Path, headers=None, timeout: float = 60.0, max_bytes: int = 0) -> Path:
        calls.append((url, {}, {}))
        dest.write_bytes(tiny_mp4 if url.split("?")[0].endswith((".mp4", ".webm")) else tiny_jpg)
        return dest

    monkeypatch.setattr(base, "http_get_json", get_json)
    monkeypatch.setattr(base, "http_download", download)
    return routes, calls


def _sidecar(path: str) -> dict:
    return json.loads(Path(path + ".json").read_text(encoding="utf-8"))


def test_wikimedia_images(net, online, shot, tmp_path) -> None:
    from vidbush.source.wikimedia import WikimediaAdapter
    routes, calls = net
    routes["https://commons.wikimedia.org"] = "commons_search_image.json"
    res = WikimediaAdapter().search("culebra cut", "image", 5, online)
    assert [r.title for r in res] == ["Culebra Cut, Panama Canal, 1913", "Gatun Locks construction"]  # svg skipped
    first, second = res
    assert first.author == "Underwood & Underwood" and first.licence == "Public domain"
    assert first.download_url.endswith("1920px-Culebra_Cut_1913.jpg") and first.width == 1920
    assert first.page_url.startswith("https://commons.wikimedia.org/wiki/File:")
    assert calls[0][1]["gsrnamespace"] == 6 and "extmetadata" in calls[0][1]["iiprop"]
    cands = WikimediaAdapter().fetch(second, tmp_path, shot, online)
    assert cands[0].licence == "CC BY-NC-SA 2.0" and "nc_nd" in cands[0].flags
    assert cands[0].attribution == "Gatun Locks construction by Panama Canal Museum (Wikimedia Commons)"
    side = _sidecar(cands[0].local_path)
    assert side["licence"] == "CC BY-NC-SA 2.0" and side["page_url"] == second.page_url and "downloaded_at" in side


def test_wikimedia_video(net, online, shot, tmp_path) -> None:
    from vidbush.source.wikimedia import WikimediaAdapter
    routes, _ = net
    routes["https://commons.wikimedia.org"] = "commons_search_video.json"
    res = WikimediaAdapter().search("panama canal ship", "video", 5, online)
    assert len(res) == 1 and res[0].media == "video" and res[0].duration == 94.5
    assert res[0].download_url.endswith(".webm")
    cands = WikimediaAdapter().fetch(res[0], tmp_path, shot, online)
    assert cands and all(c.media == "video" and c.out_point > c.in_point for c in cands)


def test_archive_video(net, online, shot, tmp_path) -> None:
    from vidbush.source.archive import ArchiveAdapter
    routes, calls = net
    routes["https://archive.org/advancedsearch.php"] = "archive_advancedsearch.json"
    routes["https://archive.org/metadata/"] = "archive_metadata.json"
    res = ArchiveAdapter().search("panama canal construction", "video", 5, online)
    assert [r.extra["identifier"] for r in res] == ["PanamaCanal1913", "canal_newsreel_1914"]
    assert res[1].title == "The Big Ditch newsreel" and res[1].author == "Pathe News"
    assert "mediatype:(movies)" in calls[0][1]["q"]
    cands = ArchiveAdapter().fetch(res[0], tmp_path, shot, online)
    assert cands and cands[0].licence == "Public domain"
    side = _sidecar(cands[0].local_path)
    assert side["source_url"] == "https://archive.org/download/PanamaCanal1913/PanamaCanal1913.mp4"


def test_archive_licence_detection() -> None:
    from vidbush.source.archive import detect_licence, pick_file
    meta = canned("archive_metadata_image.json")
    assert detect_licence(meta["metadata"], "") == "Public domain"
    assert detect_licence({"collection": ["prelinger"]}, "") == "Public domain"
    assert detect_licence({}, "") == "unknown (Internet Archive)"
    assert detect_licence({}, "https://creativecommons.org/licenses/by/4.0/").endswith("/by/4.0/")
    assert pick_file(meta["files"], "image")["name"] == "gatun_locks_photo.jpg"


def test_loc_photos(net, online, shot, tmp_path) -> None:
    from vidbush.source.loc import LocAdapter
    routes, _ = net
    routes["https://www.loc.gov/photos/"] = "loc_photos.json"
    routes["https://www.loc.gov/item/"] = "loc_item.json"
    res = LocAdapter().search("gatun locks", "image", 5, online)
    assert len(res) == 2
    assert res[0].licence.startswith("Public domain (Library of Congress")
    assert res[0].download_url.endswith("3a00001v.jpg") and res[0].width == 1340
    assert res[1].licence.startswith("Rights restricted")
    cands = LocAdapter().fetch(res[0], tmp_path, shot, online)
    assert _sidecar(cands[0].local_path)["source_url"].endswith("3a00001v.jpg")
    assert cands[0].author == "Underwood & Underwood"


def test_loc_film_search(net, online) -> None:
    from vidbush.source.loc import LocAdapter
    routes, _ = net
    routes["https://www.loc.gov/film-and-videos/"] = "loc_film.json"
    res = LocAdapter().search("panama canal", "video", 5, online)
    assert res[0].media == "video" and res[0].licence.startswith("Public domain")


def test_nasa(net, online, shot, tmp_path) -> None:
    from vidbush.source.nasa import NasaAdapter, manifest_urls, pick_asset
    routes, _ = net
    routes["https://images-api.nasa.gov/search"] = "nasa_search.json"
    routes["https://images-api.nasa.gov/asset/"] = "nasa_asset.json"
    res = NasaAdapter().search("panama canal", "image", 5, online)
    assert len(res) == 1 and res[0].author == "JSC" and res[0].licence == "Public domain (NASA)"
    cands = NasaAdapter().fetch(res[0], tmp_path, shot, online)
    assert _sidecar(cands[0].local_path)["source_url"].endswith("~large.jpg")  # orig only as last resort
    vids = pick_asset(manifest_urls(canned("nasa_asset_video.json")), "video")
    assert vids == "https://images-assets.nasa.gov/video/panama_flyover/panama_flyover~orig.mp4"


def test_pexels(net, online, shot, tmp_path) -> None:
    from vidbush.source.pexels import PexelsAdapter
    routes, calls = net
    routes["https://api.pexels.com/videos/"] = "pexels_videos.json"
    routes["https://api.pexels.com/v1/"] = "pexels_photos.json"
    assert not PexelsAdapter().available(online)
    online.env["PEXELS_API_KEY"] = "test-key"
    assert PexelsAdapter().available(online)
    vids = PexelsAdapter().search("cargo ship", "video", 5, online)
    assert len(vids) == 1 and vids[0].height == 1080 and vids[0].title == "Cargo ship in a canal lock"
    assert calls[-1][2]["Authorization"] == "test-key"
    photos = PexelsAdapter().search("container ship", "image", 5, online)
    assert photos[0].author == "Tom Fisk" and "w=2400" in photos[0].download_url
    cands = PexelsAdapter().fetch(vids[0], tmp_path, shot, online)
    assert cands[0].licence == "Pexels License" and cands[0].media == "video"
    assert cands[0].out_point - cands[0].in_point >= shot.duration


def test_pixabay(net, online, shot, tmp_path) -> None:
    from vidbush.source.pixabay import PixabayAdapter
    routes, _ = net
    routes["https://pixabay.com/api/videos/"] = "pixabay_videos.json"
    routes["https://pixabay.com/api/"] = "pixabay_photos.json"
    online.env["PIXABAY_API_KEY"] = "k"
    vids = PixabayAdapter().search("ship", "video", 5, online)
    assert vids[0].height == 1080 and vids[0].download_url.endswith("31377-medium.mp4")
    photos = PixabayAdapter().search("ship", "image", 5, online)
    assert len(photos) == 1 and photos[0].title == "Ship, canal, lock" and photos[0].author == "Hans"
    cands = PixabayAdapter().fetch(photos[0], tmp_path, shot, online)
    assert cands[0].licence == "Pixabay Content License" and cands[0].media == "image"


def test_offline_or_missing_key_unavailable(settings) -> None:
    from vidbush.source.nasa import NasaAdapter
    from vidbush.source.youtube import YoutubeAdapter
    assert not NasaAdapter().available(settings)  # settings fixture is offline
    settings.offline = False
    assert NasaAdapter().available(settings)
    settings.sources.allow_youtube = False
    assert not YoutubeAdapter().available(settings)


def test_youtube_search_scoring(monkeypatch, online) -> None:
    from vidbush.source import youtube
    monkeypatch.setattr(youtube, "ydl_extract", lambda target, opts: canned("ytdlp_search.json"))
    res = youtube.YoutubeAdapter().search("panama canal construction archival", "video", 10, online)
    ids = [r.extra["id"] for r in res]
    assert ids[0] == "aaaaaaaaaa1"  # archival newsreel ranks first
    assert "dddddddddd4" not in ids  # shorts dropped
    assert "cccccccccc3" not in ids and "eeeeeeeeee5" not in ids  # reaction vlog and top 10 explained dropped
    assert res[0].title == "Building the Panama Canal (1913) Archival newsreel footage"
    assert youtube.YoutubeAdapter().search("x", "image", 5, online) == []


def test_youtube_fetch_cuts_clips(monkeypatch, online, shot, tmp_path, tiny_mp4: bytes) -> None:
    from vidbush.source import youtube
    infos = {"https://www.youtube.com/watch?v=aaaaaaaaaa1": canned("ytdlp_video.json"),
             "https://www.youtube.com/watch?v=bbbbbbbbbb2": canned("ytdlp_video_lowres.json")}
    monkeypatch.setattr(youtube, "ydl_extract", lambda target, opts: infos[target])
    seen: dict = {}

    def fake_download(url: str, opts: dict, stem: Path) -> Path:
        seen.update(opts)
        out = stem.with_suffix(".mp4")
        out.write_bytes(tiny_mp4)
        return out

    monkeypatch.setattr(youtube, "ydl_download", fake_download)
    monkeypatch.setattr(base, "detect_scenes", lambda p: [(0.0, 20.0), (20.0, 22.0), (22.0, 30.0)])
    raw = base.RawResult(source="youtube", url="https://www.youtube.com/watch?v=aaaaaaaaaa1", media="video",
                         page_url="https://www.youtube.com/watch?v=aaaaaaaaaa1", extra={"id": "aaaaaaaaaa1"})
    cands = youtube.YoutubeAdapter().fetch(raw, tmp_path, shot, online)
    assert 2 <= len(cands) <= 4
    assert all(3.0 <= c.out_point - c.in_point <= 12.0 for c in cands)
    assert cands[0].licence == "Creative Commons Attribution"
    assert cands[0].attribution == "Building the Panama Canal (1913) [HD] Archival newsreel footage by British Pathe (YouTube)"
    assert "height<=1080" in seen["format"] and "height>=720" in seen["format"] and "download_ranges" in seen
    low = base.RawResult(source="youtube", url="https://www.youtube.com/watch?v=bbbbbbbbbb2", media="video")
    assert youtube.YoutubeAdapter().fetch(low, tmp_path, shot, online) == []  # under 720p skipped


def test_youtube_section_and_licence() -> None:
    from vidbush.source.youtube import section_for, youtube_licence
    s = Shot(id="s", beat_id="b", start=0, end=5, duration=5, type=ShotType.footage, intent="")
    a, b = section_for(600.0, s)
    assert a >= 0 and b <= 600 and 40 <= b - a <= 100
    assert section_for(20.0, s) == (0.0, 20.0)
    assert youtube_licence({}) == "Standard YouTube License"


def test_cut_clips_bounds() -> None:
    clips = base.cut_clips([(0.0, 2.0), (2.0, 40.0), (40.0, 47.0)], 5.0, max_clips=4)
    assert clips and len(clips) <= 4
    assert all(3.0 <= b - a <= 12.0 for a, b in clips)
    assert base.cut_clips([(0.0, 1.0)], 5.0) == []
