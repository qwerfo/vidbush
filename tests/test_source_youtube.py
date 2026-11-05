"""YouTube adapter: Creative Commons pass, licence flag, window selection, format, cookies, throughput, budget notes.

Everything is offline: yt-dlp is replaced by canned info dicts and fake seams.
"""
from __future__ import annotations

import contextlib
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from vidbush.judge import qa
from vidbush.judge.director_loop import _apply
from vidbush.models import Candidate, Shot, ShotList, ShotType, SourceKind
from vidbush.source import base, registry, throttle, youtube
from vidbush.source.base import RawResult

WATCH = "https://www.youtube.com/watch?v="


@pytest.fixture(autouse=True)
def fast_throttle():
    throttle.reset()
    throttle.STATE.rates["youtube"] = (1000.0, 1000.0)  # keep the serial rule, drop the gap for tests
    yield
    throttle.reset()


@pytest.fixture(scope="module")
def tiny_mp4(tmp_path_factory) -> bytes:
    out = tmp_path_factory.mktemp("yt") / "tiny.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=4",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out.read_bytes()


@pytest.fixture
def online(settings):
    settings.offline = False
    return settings


def make_shot(intent: str = "steam locomotive crossing a bridge", must_show: tuple[str, ...] = (),
              duration: float = 5.0) -> Shot:
    return Shot(id="s001", beat_id="b001", start=0, end=duration, duration=duration, type=ShotType.footage,
                intent=intent, must_show=list(must_show))


def entry(vid: str, title: str = "Archival newsreel footage") -> dict[str, Any]:
    return {"id": vid, "title": title, "channel": "Archive", "duration": 300, "url": WATCH + vid}


def info(vid: str, duration: float = 600.0, licence: str = "", **extra: Any) -> dict[str, Any]:
    return {"id": vid, "title": f"Film {vid}", "channel": "Archive", "duration": duration, "width": 1920,
            "height": 1080, "license": licence, "webpage_url": WATCH + vid,
            "formats": [{"height": 1080, "vcodec": "avc1.640028"}], **extra}


def raw_for(vid: str) -> RawResult:
    return RawResult(source="youtube", url=WATCH + vid, page_url=WATCH + vid, media="video", extra={"id": vid})


# ---------------------------------------------------------------- licence: Creative Commons pass

def test_cc_pass_runs_first_then_normal_merged_and_deduped(monkeypatch, online) -> None:
    targets: list[str] = []
    opts_seen: list[dict] = []

    def fake_extract(target: str, opts: dict) -> dict:
        targets.append(target)
        opts_seen.append(opts)
        if target.startswith("https://www.youtube.com/results"):
            return {"entries": [entry("cccccccccc1"), entry("ssssssssss2")]}
        return {"entries": [entry("ssssssssss2"), entry("nnnnnnnnnn3", "Documentary footage restored")]}

    monkeypatch.setattr(youtube, "ydl_extract", fake_extract)
    res = youtube.YoutubeAdapter().search("canal locks", "video", 10, online, prefer_cc=True)
    assert [r.extra["id"] for r in res] == ["cccccccccc1", "ssssssssss2", "nnnnnnnnnn3"]
    assert res[0].extra.get("cc_search") and res[1].extra.get("cc_search") and not res[2].extra.get("cc_search")
    assert res[0].licence == "Creative Commons Attribution" and res[2].licence == "Standard YouTube License"
    cc_url = targets[0]
    assert parse_qs(urlparse(cc_url).query) == {"search_query": ["canal locks"], "sp": ["EgIwAQ=="]}
    assert targets[1] == "ytsearch10:canal locks"
    assert opts_seen[0]["playlist_items"] == "1:10"  # the URL form has no built in limit
    targets.clear()
    youtube.YoutubeAdapter().search("canal locks", "video", 10, online)
    assert targets == ["ytsearch10:canal locks"]


def test_cc_url_is_accepted_by_the_installed_yt_dlp() -> None:
    from yt_dlp.extractor.youtube import YoutubeSearchURLIE
    assert YoutubeSearchURLIE.suitable(youtube.cc_search_url("panama canal 1914"))


class HintAdapter:
    name = "youtube"
    media = {"video"}
    search_hints = ("prefer_cc", "language")

    def __init__(self) -> None:
        self.kwargs: list[dict] = []

    def available(self, settings: Any) -> bool:
        return True

    def search(self, query: str, media: str, n: int, settings: Any, shot: Any = None, **kw: Any) -> list[RawResult]:
        self.kwargs.append(kw)
        return [RawResult(source="youtube", url=WATCH + query.replace(" ", "")[:11], media="video", score=0.6)]

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Any) -> list:
        return []


def test_registry_passes_policy_only_when_brand_asks_and_counts_each_pass(make_ctx) -> None:
    ctx = make_ctx(mode="script")
    adapter = HintAdapter()
    shot = make_shot()
    registry._search_one(adapter, "canal", "video", 5, shot, ctx)
    assert adapter.kwargs[-1] == {"language": "en"}
    assert throttle.spent(registry.run_id(ctx))["youtube"] == 1
    ctx.brand.licence_policy.prefer_creative_commons = True
    registry._search_one(adapter, "canal", "video", 5, shot, ctx)  # different hints: not served from cache
    assert adapter.kwargs[-1] == {"prefer_cc": True, "language": "en"}
    assert throttle.spent(registry.run_id(ctx))["youtube"] == 3
    throttle.STATE.budget["youtube"] = 4
    registry._search_one(adapter, "locks", "video", 5, shot, ctx)
    assert adapter.kwargs[-1] == {"language": "en"}  # one search left: the CC pass is dropped


def test_adapters_without_hints_are_called_as_before(make_ctx) -> None:
    ctx = make_ctx(mode="script")
    ctx.brand.licence_policy.prefer_creative_commons = True
    calls: list[tuple] = []

    class Plain(HintAdapter):
        name = "pexels"
        search_hints = ()

        def search(self, query: str, media: str, n: int, settings: Any, shot: Any = None) -> list[RawResult]:
            calls.append((query, n))
            return []

    registry._search_one(Plain(), "canal", "video", 5, make_shot(), ctx)
    assert calls == [("canal", 5)]


# ---------------------------------------------------------------- licence: Standard YouTube flag

def test_standard_licence_flag_on_candidates(online, tmp_path) -> None:
    shot = make_shot()
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"x")
    std = raw_for("aaaaaaaaaa1").model_copy(update={"licence": "Standard YouTube License"})
    cc = raw_for("aaaaaaaaaa1").model_copy(update={"licence": "Creative Commons Attribution"})
    assert "standard_youtube_licence" in base.make_candidates(std, clip, shot, online)[0].flags
    assert "standard_youtube_licence" not in base.make_candidates(cc, clip, shot, online)[0].flags
    online.sources.flag_standard_youtube = False
    assert "standard_youtube_licence" not in base.make_candidates(std, clip, shot, online)[0].flags


def _yt_shotlist(flags: list[str]) -> ShotList:
    cand = Candidate(id="s001_youtube_0", source=SourceKind.youtube, url=WATCH + "aaaaaaaaaa1", local_path="x.mp4",
                     media="video", licence="Standard YouTube License", title="Old film", attribution="Old film",
                     flags=flags)
    shot = make_shot()
    shot.candidates = [cand]
    _apply(shot, cand, False, [])
    return ShotList(project_id="t1", duration=5.0, shots=[shot])


def test_standard_licence_flag_reaches_shot_and_qa_report(make_ctx) -> None:
    ctx = make_ctx(mode="script")
    sl = _yt_shotlist(["standard_youtube_licence"])
    assert "standard_youtube_licence" in sl.shots[0].flags
    report = qa.make_report(sl, ctx, 70.0, {}, 1.0, 1.0, [])
    assert report.shots[0].flags == ["standard_youtube_licence"]
    assert len(report.standard_youtube_assets) == 1 and "Old film" in report.standard_youtube_assets[0]
    assert any("Standard YouTube Licence assets" in n for n in report.notes)
    md = qa.render_markdown(report)
    assert "Standard YouTube Licence assets: 1" in md and "## Standard YouTube Licence assets (1)" in md
    clean = qa.make_report(_yt_shotlist([]), ctx, 70.0, {}, 1.0, 1.0, [])
    assert clean.standard_youtube_assets == [] and "standard_youtube_licence" not in clean.shots[0].flags


# ---------------------------------------------------------------- window selection

def json3(cues: list[tuple[float, float, str]]) -> str:
    return json.dumps({"events": [{"tStartMs": int(a * 1000), "dDurationMs": int((b - a) * 1000),
                                   "segs": [{"utf8": t}]} for a, b, t in cues]})


def test_transcript_window_picks_matching_section(monkeypatch, online) -> None:
    cues = [(float(t), t + 5.0, "people cooking pasta in a kitchen") for t in range(0, 600, 5)]
    cues[60] = (300.0, 305.0, "the steam locomotive rolls onto the iron bridge")
    fetched: list[str] = []
    monkeypatch.setattr(youtube, "ydl_fetch_text", lambda url, opts: fetched.append(url) or json3(cues))
    data = info("aaaaaaaaaa1", automatic_captions={"en": [{"ext": "vtt", "url": "https://c/vtt"},
                                                          {"ext": "json3", "url": "https://c/json3"}]},
                subtitles={"de": [{"ext": "json3", "url": "https://c/de"}]})
    shot = make_shot()
    got = youtube.transcript_cues(data, online, "en")
    assert fetched == ["https://c/json3"] and len(got) == 120
    windows = youtube.choose_windows(data, shot, online, got)
    assert len(windows) == 1
    a, b = windows[0]
    assert a <= 300.0 and b >= 305.0 and b - a <= youtube.window_length(shot) + 2 * youtube.PADDING_S


def test_vtt_parsing() -> None:
    text = ("WEBVTT\n\n00:00:01.000 --> 00:00:03.500 align:start\nHello <c>world</c>\n\n"
            "01:02.000 --> 01:04.000\nsecond line\n")
    assert youtube.parse_vtt(text) == [(1.0, 3.5, "Hello world"), (62.0, 64.0, "second line")]


def test_chapter_selection_weights_must_show(online) -> None:
    chapters = [{"start_time": 0, "end_time": 120, "title": "Introduction"},
                {"start_time": 120, "end_time": 300, "title": "Bridge construction"},
                {"start_time": 300, "end_time": 500, "title": "Locomotive trials"},
                {"start_time": 500, "end_time": 600, "title": "Credits"}]
    data = info("aaaaaaaaaa1", chapters=chapters)
    plain = make_shot(intent="the bridge and a locomotive")
    (a, b), = youtube.choose_windows(data, plain, online)
    assert a == 110.0 and b <= 300 + youtube.PADDING_S  # first best chapter, padded
    weighted = make_shot(intent="the bridge and a locomotive", must_show=("locomotive",))
    (a, b), = youtube.choose_windows(data, weighted, online)
    assert a == 290.0 and b == 300 + youtube.window_length(weighted) + youtube.PADDING_S


def test_fallback_spreads_up_to_n_windows(online) -> None:
    data = info("aaaaaaaaaa1", duration=1200.0)
    shot = make_shot(intent="nothing in any transcript")
    for n in (1, 3, 5):
        online.sources.youtube_windows = n
        windows = youtube.choose_windows(data, shot, online)
        assert len(windows) == n
        assert all(0.05 * 1200 - youtube.PADDING_S <= a < b <= 0.95 * 1200 + youtube.PADDING_S for a, b in windows)
        assert windows == sorted(windows)
    online.sources.youtube_windows = 6
    short = youtube.choose_windows(info("x", duration=150.0), shot, online)
    assert 1 <= len(short) < 6  # overlapping windows merge


def test_padding_kept_and_clamped() -> None:
    assert youtube.pad_windows([(100.0, 130.0)], 600.0) == [(90.0, 140.0)]
    assert youtube.pad_windows([(3.0, 30.0)], 35.0) == [(0.0, 35.0)]
    assert youtube.pad_windows([(100.0, 120.0), (125.0, 140.0)], 600.0) == [(90.0, 150.0)]


def test_fetch_cuts_clips_per_window_with_local_in_points(monkeypatch, online, tmp_path, tiny_mp4) -> None:
    online.sources.youtube_windows = 3
    monkeypatch.setattr(youtube, "ydl_extract", lambda t, o: info("aaaaaaaaaa1"))
    stems: list[Path] = []

    def fake_download(url: str, opts: dict, stem: Path) -> Path:
        stems.append(stem)
        out = stem.with_suffix(".mp4")
        out.write_bytes(tiny_mp4)
        return out

    monkeypatch.setattr(youtube, "ydl_download", fake_download)
    monkeypatch.setattr(base, "detect_scenes", lambda p: [(0.0, 20.0), (20.0, 30.0)])
    cands = youtube.YoutubeAdapter().fetch(raw_for("aaaaaaaaaa1"), tmp_path, make_shot(), online)
    assert len(stems) == 3 and len({c.local_path for c in cands}) == 3
    assert 3 <= len(cands) <= youtube.MAX_CLIPS_PER_VIDEO
    assert all(0.0 <= c.in_point < c.out_point <= 30.0 for c in cands)  # positions inside each local file
    assert all("standard_youtube_licence" in c.flags for c in cands)
    youtube.YoutubeAdapter().fetch(raw_for("aaaaaaaaaa1"), tmp_path / "again", make_shot(), online)
    assert len(stems) == 3  # every window comes from the cache the second time


# ---------------------------------------------------------------- format and cookies

def test_format_prefers_resolution_then_h264(online) -> None:
    fmt = youtube.format_selector(online)
    assert fmt.startswith("bv*[height<=1080][height>=720][ext=mp4]/")
    assert fmt.endswith("b[height<=1080][height>=720]")
    opts = youtube.YoutubeAdapter._download_opts(10.0, 40.0, online)
    assert opts["format"] == fmt and "vcodec:h264" in opts["format_sort"]
    assert opts["format_sort"].index("res") < opts["format_sort"].index("vcodec:h264")


def test_browser_cookie_spec_parsing() -> None:
    assert youtube.browser_cookies("firefox") == ("firefox", None, None, None)
    assert youtube.browser_cookies("Chrome:Profile 1") == ("chrome", "Profile 1", None, None)
    assert youtube.browser_cookies("chromium+gnomekeyring:Default::work") == ("chromium", "Default", "GNOMEKEYRING",
                                                                              "work")
    assert youtube.browser_cookies("") is None


def test_cookies_passed_to_extract_and_download_only_when_set(monkeypatch, online, tmp_path, tiny_mp4) -> None:
    extract_opts: list[dict] = []
    download_opts: list[dict] = []
    monkeypatch.setattr(youtube, "ydl_extract", lambda t, o: extract_opts.append(o) or info("aaaaaaaaaa1", 50.0))

    def fake_download(url: str, opts: dict, stem: Path) -> Path:
        download_opts.append(opts)
        out = stem.with_suffix(".mp4")
        out.write_bytes(tiny_mp4)
        return out

    monkeypatch.setattr(youtube, "ydl_download", fake_download)
    monkeypatch.setattr(base, "detect_scenes", lambda p: [(0.0, 4.0)])
    youtube.YoutubeAdapter().fetch(raw_for("aaaaaaaaaa1"), tmp_path / "a", make_shot(), online)
    assert all("cookiesfrombrowser" not in o and "cookiefile" not in o for o in extract_opts + download_opts)
    online.sources.youtube_cookies_from_browser = "firefox"
    online.sources.youtube_cookie_file = "config/cookies.txt"
    online.sources.youtube_max_height = 1440  # new cache key, so the download runs again
    youtube.YoutubeAdapter().fetch(raw_for("aaaaaaaaaa1"), tmp_path / "b", make_shot(), online)
    youtube.YoutubeAdapter().search("canal", "video", 5, online)
    for o in (extract_opts[-2], extract_opts[-1], download_opts[-1]):
        assert o["cookiesfrombrowser"] == ("firefox", None, None, None)
        assert o["cookiefile"] == str(online.path("config/cookies.txt")) and Path(o["cookiefile"]).is_absolute()


# ---------------------------------------------------------------- throughput

class Gauge:
    """Counts concurrent entries per key and overall."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active: dict[str, int] = {}
        self.peak: dict[str, int] = {}

    def enter(self, *keys: str) -> None:
        with self.lock:
            for k in keys:
                self.active[k] = self.active.get(k, 0) + 1
                self.peak[k] = max(self.peak.get(k, 0), self.active[k])

    def leave(self, *keys: str) -> None:
        with self.lock:
            for k in keys:
                self.active[k] -= 1


def fake_ydl(gauge: Gauge, infos: dict[str, dict], tiny: bytes) -> type:
    class FakeYDL:
        def __init__(self, params: dict) -> None:
            self.params = params

        def __enter__(self) -> "FakeYDL":
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def extract_info(self, target: str, download: bool = False) -> dict:
            gauge.enter("extract")
            time.sleep(0.03)
            gauge.leave("extract")
            if target.startswith("ytsearch"):
                return {"entries": [entry("aaaaaaaaaa1")]}
            return infos[target]

        def sanitize_info(self, data: dict) -> dict:
            return data

        def download(self, urls: list[str]) -> None:
            gauge.enter("download", urls[0])
            time.sleep(0.15)
            Path(self.params["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(tiny)
            gauge.leave("download", urls[0])
    return FakeYDL


def test_parallel_downloads_capped_while_extraction_stays_serial(monkeypatch, online, tmp_path, tiny_mp4) -> None:
    import yt_dlp
    vids = [f"vid{i:08d}" for i in range(6)]
    gauge = Gauge()
    monkeypatch.setattr(yt_dlp, "YoutubeDL", fake_ydl(gauge, {WATCH + v: info(v, 50.0) for v in vids}, tiny_mp4))
    monkeypatch.setattr(base, "detect_scenes", lambda p: [(0.0, 4.0)])
    online.sources.youtube_download_workers = 3
    adapter = youtube.YoutubeAdapter()
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(adapter.fetch, raw_for(v), tmp_path / v, make_shot(), online) for v in vids]
        futs += [pool.submit(adapter.search, f"q{i}", "video", 5, online) for i in range(4)]
        results = [f.result() for f in futs]
    assert all(results[:6])
    assert gauge.peak["download"] == 3  # different videos overlap up to the worker count
    assert gauge.peak["extract"] == 1  # searches and full info extraction stay serial


def test_same_video_is_never_downloaded_twice_at_once(monkeypatch, online, tmp_path, tiny_mp4) -> None:
    import yt_dlp
    gauge = Gauge()
    url = WATCH + "aaaaaaaaaa1"
    counter: list[int] = []
    cls = fake_ydl(gauge, {url: info("aaaaaaaaaa1", 50.0)}, tiny_mp4)
    original = cls.download

    def counted(self: Any, urls: list[str]) -> None:
        counter.append(1)
        original(self, urls)

    cls.download = counted
    monkeypatch.setattr(yt_dlp, "YoutubeDL", cls)
    monkeypatch.setattr(base, "detect_scenes", lambda p: [(0.0, 4.0)])
    adapter = youtube.YoutubeAdapter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(adapter.fetch, raw_for("aaaaaaaaaa1"), tmp_path / f"s{i}", make_shot(), online)
                for i in range(4)]
        assert all(f.result() for f in futs)
    assert gauge.peak[url] == 1 and len(counter) == 1  # later callers reuse the cached file


def test_fetch_pool_runs_shots_in_parallel() -> None:
    assert registry._workers() >= 2 and registry.ADAPTER_WORKERS["youtube"] == 1


# ---------------------------------------------------------------- budget in the QA report

def _shots(n: int, word: str = "ships") -> list[Shot]:
    return [Shot(id=f"s{i:03d}", index=i, beat_id="b001", start=4 * i, end=4 * i + 4, duration=4.0,
                 type=ShotType.footage, intent=word, queries=[f"{word} {i}"]) for i in range(n)]


def test_search_usage_and_spent_budget_surface_in_qa_report(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    adapter = HintAdapter()
    monkeypatch.setattr(registry, "active_adapters", lambda c: [adapter])
    registry.source_all(ShotList(project_id="t1", duration=8, shots=_shots(2)), ctx)
    notes = qa.source_usage_notes(ctx)
    assert notes == [f"YouTube searches used: 2 of {throttle.DEFAULT_BUDGET['youtube']} across 2 shots"]
    ctx.extra.pop("source_run_id", None)  # a new run (resume) with a tiny budget
    throttle.STATE.budget["youtube"] = 1
    registry.source_all(ShotList(project_id="t1", duration=12, shots=_shots(3, "boats")), ctx)
    data = json.loads((ctx.work / registry.SOURCE_NOTES).read_text(encoding="utf-8"))
    assert len(data["runs"]) == 2
    report = qa.make_report(ShotList(project_id="t1", duration=12, shots=_shots(3)), ctx, 70.0, {}, 1.0, 1.0, [])
    warn = [n for n in report.notes if n.startswith("warning: YouTube search budget spent")]
    assert warn and "(1 of 1 searches across 1 shots)" in warn[0] and "run 2 of 2" in warn[0]
    assert data["runs"][registry.run_id(ctx)]["usage"]["youtube"] == {"used": 1, "budget": 1, "refused": 1,
                                                                      "shots": 1, "deferred": 0}
    assert any(n.startswith("YouTube searches used: 2 of") for n in report.notes)
    assert "warning: YouTube search budget spent" in qa.render_markdown(report)


class StillAdapter(HintAdapter):
    """An image source that finds nothing, so still shots fall back to video frames."""
    name = "pexels"
    media = {"image"}
    search_hints = ()

    def search(self, query: str, media: str, n: int, settings: Any, shot: Any = None) -> list[RawResult]:
        return []


def test_fallback_video_searches_of_still_shots_reach_the_usage_notes(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    video = HintAdapter()
    monkeypatch.setattr(registry, "active_adapters", lambda c: [StillAdapter(), video])
    still = Shot(id="s000", index=0, beat_id="b001", start=0, end=4, duration=4.0, type=ShotType.image,
                 intent="harbour", queries=["harbour cranes"])
    registry.source_all(ShotList(project_id="t1", duration=4, shots=[still]), ctx)
    assert video.kwargs  # the fallback searched the video source after the search phase
    assert qa.source_usage_notes(ctx) == [  # every budgeted source gets its line now, not only YouTube
        f"Pexels searches used: 1 of {throttle.DEFAULT_BUDGET['pexels']} across 1 shots",
        f"YouTube searches used: 1 of {throttle.DEFAULT_BUDGET['youtube']} across 1 shots"]
    throttle.reset()
    registry.source_shot(still, ctx)
    data = json.loads((ctx.work / registry.SOURCE_NOTES).read_text(encoding="utf-8"))
    assert data["runs"][registry.run_id(ctx)]["usage"]["youtube"]["used"] == 1


# ---------------------------------------------------------------- review fixes

def test_failed_cc_pass_still_runs_the_normal_pass(monkeypatch, online) -> None:
    def fake_extract(target: str, opts: dict) -> dict:
        if target.startswith("https://www.youtube.com/results"):
            raise RuntimeError("bot check")
        return {"entries": [entry("nnnnnnnnnn3")]}

    monkeypatch.setattr(youtube, "ydl_extract", fake_extract)
    res = youtube.YoutubeAdapter().search("canal locks", "video", 10, online, prefer_cc=True)
    assert [r.extra["id"] for r in res] == ["nnnnnnnnnn3"]

    def broken(target: str, opts: dict) -> dict:
        raise RuntimeError("down")

    monkeypatch.setattr(youtube, "ydl_extract", broken)
    with pytest.raises(RuntimeError):  # both passes failing still surfaces the error
        youtube.YoutubeAdapter().search("canal locks", "video", 10, online, prefer_cc=True)


def test_each_youtube_dl_gets_a_private_cookie_copy(monkeypatch, tmp_path) -> None:
    import yt_dlp
    operator = tmp_path / "cookies.txt"
    operator.write_text("# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tabc\n", encoding="utf-8")
    original = operator.read_text(encoding="utf-8")
    seen: list[Path] = []

    class SavingYDL:
        """Like yt-dlp: reads the cookiefile and truncates and rewrites it in place on close."""

        def __init__(self, params: dict) -> None:
            self.path = Path(params["cookiefile"])
            seen.append(self.path)
            self.loaded = self.path.read_text(encoding="utf-8") if self.path.exists() else None

        def __enter__(self) -> "SavingYDL":
            return self

        def __exit__(self, *a: Any) -> None:
            self.path.write_text("", encoding="utf-8")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", SavingYDL)
    with contextlib.ExitStack() as stack:  # overlapping instances, as with parallel downloads
        open_ydls = [stack.enter_context(youtube.youtube_dl({"cookiefile": str(operator)})) for _ in range(3)]
        assert all(y.loaded == original for y in open_ydls)
        with youtube.youtube_dl({"cookiefile": str(operator)}) as late:
            pass
        assert late.loaded == original and all(y.path.read_text(encoding="utf-8") == original for y in open_ydls)
    assert operator.read_text(encoding="utf-8") == original  # the operator's file is never written
    assert operator not in seen and len(set(seen)) == len(seen)
    assert not any(p.exists() for p in seen)  # copies are removed after each call
    with youtube.youtube_dl({"cookiefile": str(tmp_path / "missing.txt")}) as ydl:
        assert not ydl.path.exists() and ydl.path != tmp_path / "missing.txt"
    with youtube.private_cookies({"quiet": True}) as params:
        assert params == {"quiet": True}


def test_window_start_accounts_for_the_keyframe_lead(tmp_path, tiny_mp4) -> None:
    clip = tmp_path / "w.mp4"
    clip.write_bytes(tiny_mp4)  # 4 s long
    assert youtube.window_start(40.0, 42.0, clip) == pytest.approx(38.0, abs=0.05)  # 2 s before the cut
    assert youtube.window_start(40.0, 50.0, clip) == 40.0  # shorter than asked: no lead
    assert youtube.window_start(1.0, 2.0, clip) == 0.0  # never before the start of the video
    assert youtube.window_start(5.0, 6.0, tmp_path / "missing.mp4") == 5.0
    assert youtube.window_name("abc", 37.6, ".mp4") == "youtube_abc_38.mp4"


def test_window_file_name_carries_the_true_start(monkeypatch, online, tmp_path, tiny_mp4) -> None:
    from vidbush.judge.director_loop import _timeline
    online.sources.youtube_windows = 1
    monkeypatch.setattr(youtube, "ydl_extract", lambda t, o: info("aaaaaaaaaa1"))
    windows: list[tuple[float, float]] = []

    def fake_download(url: str, opts: dict, stem: Path) -> Path:
        out = stem.with_suffix(".mp4")
        out.write_bytes(tiny_mp4)
        return out

    def fake_windows(self: Any, url: str, wins: list, settings: Any) -> list:
        windows.extend(wins)
        return [(a, b, fake_download(url, {}, base.cache_dir(settings) / f"w{a:.0f}")) for a, b in wins]

    monkeypatch.setattr(youtube.YoutubeAdapter, "_download_windows", fake_windows)
    monkeypatch.setattr(base, "detect_scenes", lambda p: [(0.0, 4.0)])
    monkeypatch.setattr(youtube, "probe_duration", lambda p: windows[0][1] - windows[0][0] + 3.4)
    cands = youtube.YoutubeAdapter().fetch(raw_for("aaaaaaaaaa1"), tmp_path, make_shot(), online)
    true_start = windows[0][0] - 3.4
    family, offset = _timeline(cands[0])
    assert family == "youtube_aaaaaaaaaa1" and offset == round(true_start)
