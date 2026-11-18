"""The paid second YouTube route: taken on a bot check, same floor and licence, charged and budgeted."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vidbush.config import REPO_ROOT, Settings, load_settings
from vidbush.models import Shot, ShotType
from vidbush.source import base, youtube, youtube_api
from vidbush.source.base import RawResult


@pytest.fixture(autouse=True)
def fresh_route():
    youtube.ROUTE.name, youtube.ROUTE.counts = "free", {"free": 0, "api": 0}
    youtube.BOT_BLOCK.clear()
    youtube_api.LEDGER.__init__()
    yield
    youtube.ROUTE.name = "free"
    youtube.BOT_BLOCK.clear()


def _settings(tmp_path: Path, key: str = "k") -> Settings:
    s = load_settings(REPO_ROOT, use_dotenv=False)
    s.env = {"VIDEO_DOWNLOAD_API_KEY": key} if key else {}
    s.sources.youtube_api_budget_usd = 0.001
    return s


BOT = RuntimeError("ERROR: [youtube] x: Sign in to confirm you're not a bot")


def test_bot_check_switches_to_the_api_route_when_a_key_is_set(tmp_path):
    assert youtube.note_bot_block(BOT, _settings(tmp_path)) is True
    assert youtube.ROUTE.name == "api" and not youtube.BOT_BLOCK.is_set()


def test_without_a_key_a_bot_check_still_pauses(tmp_path):
    assert youtube.note_bot_block(BOT, _settings(tmp_path, key="")) is False
    assert youtube.ROUTE.name == "free" and youtube.BOT_BLOCK.is_set()


def test_fetch_retries_the_same_video_on_the_api_route(tmp_path, monkeypatch):
    ad = youtube.YoutubeAdapter()
    monkeypatch.setattr(ad, "_fetch", lambda *a: (_ for _ in ()).throw(BOT))
    monkeypatch.setattr(ad, "_fetch_api", lambda *a: ["api"])
    assert ad.fetch(RawResult(source="youtube", url="u", title="t"), tmp_path, None, _settings(tmp_path)) == ["api"]
    assert ad.fetch(RawResult(source="youtube", url="u2", title="t"), tmp_path, None, _settings(tmp_path)) == ["api"]


def test_api_route_keeps_the_height_floor_and_charges(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(base, "cache_dir", lambda s: tmp_path)

    class Client:
        def __init__(self, key):
            pass

        def download(self, url, dest):
            dest.write_bytes(b"x")
            youtube_api.LEDGER.charge("720", url)
            return dest, {}

    monkeypatch.setattr(youtube_api, "VdaClient", Client)
    monkeypatch.setattr("vidbush.util.video_size", lambda p: (854, 480))
    shot = Shot(id="s", beat_id="b", scene_id="c", start=0, end=4, duration=4, type=ShotType.footage, intent="x")
    with pytest.raises(base.Rejected, match="480p"):
        youtube.YoutubeAdapter()._fetch_api(RawResult(source="youtube", url="u", title="t"), tmp_path, shot, settings)
    assert youtube_api.LEDGER.spent == pytest.approx(0.0002) and youtube.ROUTE.counts["api"] == 1
    tracker = SimpleNamespace(entries=[], add=lambda e: tracker.entries.append(e))
    youtube_api.flush_costs(SimpleNamespace(tracker=tracker))
    assert [e.cost_usd for e in tracker.entries] == [0.0002] and tracker.entries[0].provider == "video_download_api"


def test_budget_stops_new_downloads(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(base, "cache_dir", lambda s: tmp_path)
    for i in range(5):
        youtube_api.LEDGER.charge("720", f"u{i}")
    with pytest.raises(base.Rejected, match="budget"):
        youtube.YoutubeAdapter()._fetch_api(RawResult(source="youtube", url="new", title="t"), tmp_path, None, settings)


def test_client_never_puts_the_key_in_errors():
    c = youtube_api.VdaClient("SECRETKEY")
    assert "SECRETKEY" not in c._scrub("http 401 for https://p.savenow.to/x?apikey=SECRETKEY")
