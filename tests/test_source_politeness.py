"""Sourcing politeness: token buckets, budgets, 429 backoff, search cache, streamed downloads, YouTube runtime."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from vidbush.models import Shot, ShotList, ShotType
from vidbush.providers.base import RateLimitError
from vidbush.source import base, registry, throttle
from vidbush.source.base import RawResult


@pytest.fixture(autouse=True)
def fresh_throttle():
    throttle.reset()
    yield
    throttle.reset()


def test_token_bucket_bursts_then_refuses_long_waits() -> None:
    b = throttle.TokenBucket(rate=1 / 3, burst=1)
    assert b.reserve(5.0) == 0.0
    wait = b.reserve(5.0)  # second request waits about 3 s
    assert 2.5 < wait <= 3.0
    with pytest.raises(throttle.Throttled):
        b.reserve(1.0)  # third would wait about 6 s


def test_default_rates_follow_published_limits() -> None:
    assert throttle.DEFAULT_RATES["www.loc.gov"][0] == pytest.approx(1 / 3)
    assert throttle.DEFAULT_RATES["api.pexels.com"][0] * 3600 == pytest.approx(200)
    assert throttle.DEFAULT_RATES["pixabay.com"][0] * 60 == pytest.approx(100)
    assert "youtube" in throttle.SERIAL_HOSTS


def test_cooldown_refuses_until_expired() -> None:
    throttle.cool_down("api.pexels.com", 100)
    with pytest.raises(throttle.Throttled):
        with throttle.slot("api.pexels.com", max_wait=1):
            pass


def test_budget_per_run_and_adapter() -> None:
    throttle.STATE.budget["pexels"] = 2
    assert throttle.take_budget("p1", "pexels") and throttle.take_budget("p1", "pexels")
    assert not throttle.take_budget("p1", "pexels")
    assert throttle.take_budget("p2", "pexels")  # another run has its own budget
    assert throttle.spent("p1") == {"pexels": 2}


def test_configure_overrides_from_sources() -> None:
    class Src:
        rate_limits = {"example.org": [2, 3]}
        max_concurrency = 2
        query_budget = {"loc": 5}
    throttle.configure(Src())
    assert throttle.STATE.rates["example.org"] == (2.0, 3.0)
    assert throttle.STATE.max_concurrency == 2 and throttle.STATE.budget["loc"] == 5


def test_429_backs_off_with_retry_after_then_succeeds(monkeypatch) -> None:
    calls: list[float] = []

    def get_json(url: str, params=None, headers=None, timeout: float = 20.0) -> Any:
        calls.append(time.monotonic())
        if len(calls) == 1:
            raise RateLimitError("429", retry_after=0.2)
        return {"ok": True}

    monkeypatch.setattr(base.provider_http, "get_json", get_json)
    assert base.http_get_json("https://example.org/api") == {"ok": True}
    assert len(calls) == 2 and calls[1] - calls[0] >= 0.15


def test_429_with_long_retry_after_gives_up_and_cools_host(monkeypatch) -> None:
    def get_json(*a: Any, **k: Any) -> Any:
        raise RateLimitError("429", retry_after=3600)

    monkeypatch.setattr(base.provider_http, "get_json", get_json)
    with pytest.raises(RateLimitError):
        base.http_get_json("https://api.pexels.com/v1/search")
    with pytest.raises(throttle.Throttled):  # the host is now cooling down: nothing is sent
        base.http_get_json("https://api.pexels.com/v1/search")


def test_user_agent_carries_contact(settings) -> None:
    settings.env["VIDBUSH_CONTACT"] = "ops@example.org"
    base.configure(settings)
    try:
        assert "ops@example.org" in base.user_agent_to("https://commons.wikimedia.org/w/api.php")
        assert "ops@example.org" in base.user_agent_to("https://upload.wikimedia.org/a.jpg")
        assert "ops@example.org" not in base.user_agent_to("https://api.pexels.com/v1/search")
        assert "ops@example.org" not in base.user_agent_to("https://notwikimedia.org/x")
    finally:
        settings.env.pop("VIDBUSH_CONTACT")
        base.configure(settings)
    assert "github.com/qwerfo/vidbush" in base.USER_AGENT  # default carries a contact URL


# ---------------------------------------------------------------- streamed downloads

class FakeResponse:
    def __init__(self, body: bytes, length: bool = True) -> None:
        self.body = body
        self.status_code = 200
        self.headers = {"Content-Length": str(len(body))} if length else {}
        self.text = ""

    def iter_content(self, size: int):
        for i in range(0, len(self.body), size):
            yield self.body[i:i + size]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *a: Any) -> None:
        return None


def test_download_streams_to_disk_and_caps_size(monkeypatch, tmp_path) -> None:
    import requests
    body = b"x" * 300_000
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(body))
    dest = base.http_download("https://example.org/a.mp4", tmp_path / "a.part", max_bytes=1_000_000)
    assert dest.read_bytes() == body
    with pytest.raises(base.DownloadTooLarge):  # Content-Length over the cap: nothing read
        base.http_download("https://example.org/b.mp4", tmp_path / "b.part", max_bytes=1000)
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(body, length=False))
    with pytest.raises(base.DownloadTooLarge):  # no length: aborted while streaming, file removed
        base.http_download("https://example.org/c.mp4", tmp_path / "c.part", max_bytes=100_000)
    assert not (tmp_path / "c.part").exists()


def test_large_video_is_read_as_a_section(monkeypatch, settings) -> None:
    sections: list[tuple[float, float]] = []

    def section(url: str, start: float, length: float, out: Path) -> Path:
        sections.append((start, length))
        out.write_bytes(b"v")
        return out

    def too_large(*a: Any, **k: Any) -> Path:
        raise base.DownloadTooLarge(10 ** 10)

    monkeypatch.setattr(base, "ffmpeg_section", section)
    monkeypatch.setattr(base, "http_download", too_large)
    p = base.cached_video("https://example.org/film.mp4", ".mp4", settings, 5.0, duration=3600)
    assert p.exists() and sections == [(360.0, 40.0)]
    p2 = base.cached_video("https://example.org/film2.mp4", ".mp4", settings, 5.0, duration=1000, size=9e9)
    assert p2.exists() and len(sections) == 2  # known size skips the download attempt


def test_archive_length_formats_and_pd_collections() -> None:
    from vidbush.source.archive import detect_licence, parse_length
    assert parse_length("93.5") == 93.5
    assert parse_length("01:33") == 93.0
    assert parse_length("1:02:03") == 3723.0
    assert parse_length("n/a") == 0.0
    assert detect_licence({"collection": ["newsandpublicaffairs"]}, "") == "unknown (Internet Archive)"
    assert detect_licence({"collection": ["prelinger"]}, "") == "Public domain"


def test_nasa_prefers_large_renditions() -> None:
    from vidbush.source.nasa import pick_asset
    urls = ["https://x/a~orig.jpg", "https://x/a~large.jpg", "https://x/a~orig.mp4", "https://x/a~medium.mp4"]
    assert pick_asset(urls, "image").endswith("~large.jpg")
    assert pick_asset(urls, "video").endswith("~medium.mp4")
    assert pick_asset(["https://x/a~orig.jpg"], "image").endswith("~orig.jpg")


# ---------------------------------------------------------------- registry cache and budget

class CountingAdapter:
    name = "pexels"
    media = {"video", "image"}

    def __init__(self) -> None:
        self.calls = 0

    def available(self, settings: Any) -> bool:
        return True

    def search(self, query: str, media: str, n: int, settings: Any, shot: Any = None) -> list[RawResult]:
        self.calls += 1
        return [RawResult(source="pexels", url=f"https://pexels.example/{query}", media=media, score=0.6)]

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Any) -> list:
        return []


def _shot(i: int) -> Shot:
    return Shot(id=f"s{i:03d}", index=i, beat_id="b001", start=0, end=4, duration=4.0, type=ShotType.footage,
                intent="ships", queries=["ships at sea"])


def test_search_cache_avoids_requery_on_resume(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    adapter = CountingAdapter()
    monkeypatch.setattr(registry, "active_adapters", lambda c: [adapter])
    registry.source_all(ShotList(project_id="t1", duration=4, shots=[_shot(0)]), ctx)
    assert adapter.calls == 1
    registry.source_all(ShotList(project_id="t1", duration=4, shots=[_shot(0)]), ctx, force=True)
    assert adapter.calls == 1  # served from <source_cache_folder>/search
    assert list((Path(ctx.settings.source_cache_folder) / "search" / "pexels").glob("*.json"))


def test_budget_spent_skips_searches(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    adapter = CountingAdapter()
    throttle.STATE.budget["pexels"] = 1
    monkeypatch.setattr(registry, "active_adapters", lambda c: [adapter])
    shots = [_shot(0), _shot(1)]
    shots[1].queries = ["steamship"]
    registry.source_all(ShotList(project_id="t1", duration=8, shots=shots), ctx)
    assert adapter.calls == 1


def test_youtube_runtime_detection(monkeypatch) -> None:
    from vidbush.source import youtube
    monkeypatch.setattr(youtube.shutil, "which", lambda n: "/usr/bin/deno" if n == "deno" else None)
    assert youtube.js_runtimes() == {"deno": {"path": "/usr/bin/deno"}}
    assert youtube._base_opts()["js_runtimes"] == {"deno": {"path": "/usr/bin/deno"}}
    monkeypatch.setattr(youtube.shutil, "which", lambda n: None)
    assert youtube.js_runtimes() == {} and "js_runtimes" not in youtube._base_opts()
    assert "deno" in youtube.runtime_warning()


def test_relevance_drops_off_topic_titles() -> None:
    from vidbush.source.base import RawResult, relevance
    q = "Panama Canal construction footage"
    assert relevance(q, RawResult(source="archive", url="u", title="Panama Canal Construction in 1912")) == 1.0
    assert relevance(q, RawResult(source="archive", url="u", title="SpanishPod Archive")) == 0.0
    assert relevance(q, RawResult(source="wikimedia", url="u", title="State of the Union Address 2010")) == 0.0


def test_commons_transcode_urls() -> None:
    from vidbush.source.wikimedia import transcode_urls
    urls = transcode_urls("https://upload.wikimedia.org/wikipedia/commons/7/78/A_b.webm?utm_source=x")
    assert urls[0].endswith("/transcoded/7/78/A_b.webm/A_b.webm.1080p.vp9.webm")
    assert transcode_urls("https://example.com/a.webm") == []


def test_youtube_bot_check_disables_youtube(monkeypatch) -> None:
    from vidbush.source import youtube
    youtube.BOT_BLOCK.clear()
    youtube.note_bot_block(RuntimeError("ERROR: [youtube] x: Sign in to confirm you're not a bot"))
    assert youtube.BOT_BLOCK.is_set()
    assert youtube.YoutubeAdapter().search("q", "video", 3, None) == []
    youtube.BOT_BLOCK.clear()


def test_era_mismatch_drops_modern_titles_for_historical_shots() -> None:
    from vidbush.models import Shot, ShotType
    from vidbush.source.base import RawResult
    from vidbush.source.registry import era_mismatch
    shot = Shot(id="s", beat_id="b", start=0, end=4, duration=4, type=ShotType.footage,
                intent="Archival footage of Thomas Jefferson and a canal in New York, 1810s.")
    assert era_mismatch(shot, RawResult(source="loc", url="u", title="Thomas Jefferson Building: 2019 Book Festival"))
    assert not era_mismatch(shot, RawResult(source="loc", url="u", title="Pilot boats in New York harbor 1903"))
