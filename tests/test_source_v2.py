"""v2 sourcing (Part B): per-class query routing, archive queries with year and subject, provenance on every
candidate, distinct-asset download cap, no title-only gate for description engines, frame-chosen segments,
keyframes from played seconds, a shared scene pool."""
from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from vidbush.models import Candidate, RealThing, Shot, ShotType, SourceKind
from vidbush.source import base, registry, segment
from vidbush.source.base import RawResult
from vidbush.source.licence import licence_flags


def _shot(i: int = 0, typ: ShotType = ShotType.image, **kw: Any) -> Shot:
    return Shot(id=f"s{i:03d}", index=i, beat_id="b1", start=i * 3.0, end=i * 3.0 + 3.0, duration=3.0, type=typ,
                intent=kw.pop("intent", "a newspaper page"), **kw)


class _Ad:
    def __init__(self, name: str, classes: set[str], media=("image", "video"), desc: bool = False) -> None:
        self.name, self.classes, self.media, self.matches_description = name, frozenset(classes), set(media), desc
        self.sent: list[str] = []

    def available(self, settings: Any) -> bool:
        return True

    def search(self, query: str, media: str, n: int, settings: Any, shot: Any = None, **kw: Any) -> list[RawResult]:
        self.sent.append(query)
        return [RawResult(source="wikimedia", url=f"https://x/{self.name}/{query}/{i}", title=f"{query} {i}",
                          media=media) for i in range(2)]

    def fetch(self, *a: Any, **k: Any) -> list:
        return []


# ---------------------------------------------------------------- archive queries

@pytest.mark.parametrize("q,first", [
    ("Anchorage Daily Times apology editorial 1958", "Anchorage Daily Times 1958"),
    ("Leonhard Seppala with lead dog Togo 1925", "Leonhard Seppala Togo 1925"),
    ("dog sled team crossing ice in the 1920s", "dog sled team 1920s"),
])
def test_archive_query_keeps_subject_and_year(q: str, first: str) -> None:
    qs = base.archive_queries(q, 3)
    assert qs[0] == first
    assert qs[1] == " ".join(base.YEAR_RE.sub("", first).split())  # the plain form is the fallback


def test_archive_query_without_year_is_one_request() -> None:
    assert base.archive_queries("dog sled team on snow", 3) == ["dog sled team"]


# ---------------------------------------------------------------- query routing

STRUCT = {"real_thing": {"name": "Balto", "kind": "object", "date": "1925", "place": "Central Park"},
          "queries": {"archive": ["Balto statue unveiling 1925"], "stock": ["bronze dog statue park"],
                      "web": ["Balto statue Central Park 1925 photo"], "video": ["Balto statue Central Park"]}}


def test_structure_routes_each_class_to_its_sources() -> None:
    plan = registry.query_plan(_shot(), STRUCT)
    assert plan.real_thing.name == "Balto" and set(plan.by_class) == {"archive", "stock", "web_image", "web_video"}
    archive = _Ad("archive", {"archive", "web_video"})
    stock = _Ad("pexels", {"stock"})
    web = _Ad("webimage", {"web_image", "newspaper", "encyclopedic"})
    news = _Ad("chronam", {"newspaper"})
    assert registry.adapter_queries(plan, archive) == ["Balto statue unveiling 1925", "Balto statue Central Park"]
    assert registry.adapter_queries(plan, stock) == ["bronze dog statue park"]
    assert registry.adapter_queries(plan, web) == ["Balto statue Central Park 1925 photo"]
    # a class with no query of its own gets the real thing (name and year) when the source holds named things
    assert registry.adapter_queries(plan, news) == ["Balto 1925"]
    assert registry.query_class(plan, news, "Balto 1925") == "thing"
    assert registry.query_class(plan, archive, "Balto statue Central Park") == "web_video"


def test_old_flat_list_still_goes_to_every_source() -> None:
    shot = _shot(queries=["ships at sea", "steamship"])
    plan = registry.query_plan(shot)
    assert plan.by_class == {} and registry.adapter_queries(plan, _Ad("pexels", {"stock"})) == ["ships at sea", "steamship"]


def test_shot_source_queries_field_is_routed() -> None:
    shot = _shot(source_queries={"stock": ["plain harbour"], "encyclopedic": ["Port of Nome"]},
                 real_thing=RealThing(name="Port of Nome", kind="place"))
    plan = registry.query_plan(shot)
    assert registry.adapter_queries(plan, _Ad("wikimedia", {"encyclopedic", "archive"})) == ["Port of Nome"]
    assert registry.adapter_queries(plan, _Ad("pexels", {"stock"})) == ["plain harbour"]


def test_search_all_routes_and_records_class(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    monkeypatch.setattr(registry, "cache_get", lambda *a, **k: None)
    monkeypatch.setattr(registry, "cache_put", lambda *a, **k: None)
    stock, arch = _Ad("pexels", {"stock"}), _Ad("archive", {"archive"})
    shot = _shot(source_queries={"stock": ["bronze dog statue"], "archive": ["Balto statue 1925"]})
    with registry.SearchPools([stock, arch]) as pools:
        found = registry._search_all([shot], [stock, arch], ctx, pools)
    assert stock.sent == ["bronze dog statue"] and arch.sent == ["Balto statue 1925"]
    res = found[shot.id]["archive"]
    assert res[0].extra["query_class"] == "archive" and res[0].extra["rank"] == 1
    assert res[0].extra["query"] == "Balto statue 1925"


def test_a_source_with_no_query_for_the_shot_does_not_search(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    monkeypatch.setattr(registry, "cache_get", lambda *a, **k: None)
    monkeypatch.setattr(registry, "cache_put", lambda *a, **k: None)
    yt = _Ad("youtube", {"web_video"}, media=("video",))
    shot = _shot(typ=ShotType.footage, source_queries={"stock": ["harbour at dusk"]})
    with registry.SearchPools([yt]) as pools:
        registry._search_all([shot], [yt], ctx, pools)
    assert yt.sent == []


def test_real_thing_enters_the_cache_variant_of_entity_sources(make_ctx) -> None:
    ctx = make_ctx(mode="script")
    ent = _Ad("wikimedia", {"encyclopedic"})
    ent.uses_real_thing = True
    a = _shot(real_thing=RealThing(name="Balto", kind="object"))
    b = _shot(real_thing=RealThing(name="Balto", kind="person"))
    assert registry.search_variant(ent, ctx, a) != registry.search_variant(ent, ctx, b)
    assert registry.search_variant(_Ad("pexels", {"stock"}), ctx, a) == ""


# ---------------------------------------------------------------- relevance gate

def test_description_engines_are_never_demoted_on_title_words(make_ctx) -> None:
    ctx = make_ctx(mode="script")
    hits = [RawResult(source="archive", url="https://a/1", title="Reel 4", media="video")]
    assert registry.relevant(_Ad("archive", {"archive"}, desc=True), "serum run Nome 1925", hits, ctx)[0].score > 0
    assert registry.relevant(_Ad("pexels", {"stock"}), "serum run Nome 1925", hits, ctx)[0].score == 0


# ---------------------------------------------------------------- distinct asset cap

def _cand(asset: int, clip: int) -> Candidate:
    return Candidate(id=f"a{asset}c{clip}", source=SourceKind.archive, url=f"https://asset/{asset}", media="video")


def test_spread_clips_takes_every_assets_first_clip_first() -> None:
    per = [[_cand(a, c) for c in range(4)] for a in range(3)]
    got = registry.spread_clips(per, 8)
    assert [c.id for c in got[:3]] == ["a0c0", "a1c0", "a2c0"] and len(got) == 8


def test_cap_counts_assets_so_many_assets_reach_the_shot(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")

    class Many:
        name, media = "archive", {"video"}

        def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Any) -> list[Candidate]:
            i = int(raw.url.rsplit("/", 1)[1])
            return [_cand(i, c) for c in range(4)]  # a long film cut into 4 clips

    plan = [RawResult(source="archive", url=f"https://asset/{i}", media="video") for i in range(12)]
    got = registry._fetch_raws(_shot(typ=ShotType.footage), plan, {"archive": Many()}, ctx, 8, "video")
    assert len(got) == 8 and len({c.url for c in got}) == 8  # was: 2 assets x 4 clips


# ---------------------------------------------------------------- provenance and licence

def test_candidate_carries_what_found_it(settings, tmp_path) -> None:
    raw = RawResult(source="webimage", url="https://img/1.jpg", page_url="https://page/1", title="Balto unveiled",
                    licence="unknown (web image search)", extra={"query": "Balto statue 1925", "rank": 2,
                                                                  "query_class": "web_image",
                                                                  "snippet": "The statue was unveiled in 1925.",
                                                                  "date": "1925-12-15"})
    path = tmp_path / "x.jpg"
    Image.new("RGB", (64, 64)).save(path)
    c = base.make_candidates(raw, path, _shot(), settings)[0]
    assert (c.query, c.rank, c.found_by, c.date, c.page_url) == ("Balto statue 1925", 2, "webimage:web_image",
                                                                 "1925-12-15", "https://page/1")
    assert c.description.startswith("The statue") and "licence_unknown" in c.flags
    side = json.loads(path.with_name(path.name + ".json").read_text(encoding="utf-8"))
    assert side["query"] == "Balto statue 1925" and side["rank"] == 2


def test_unknown_licence_is_flagged_open_licences_are_not(settings) -> None:
    assert "licence_unknown" in licence_flags("unknown", settings)
    assert "licence_unknown" not in licence_flags("CC BY-SA 4.0", settings)
    assert "licence_unknown" not in licence_flags("Public domain", settings)


# ---------------------------------------------------------------- segment chooser

class _Seats:
    def __init__(self, reply: dict) -> None:
        self.reply, self.calls = reply, 0

    def is_real(self, role: str) -> bool:
        return True

    def vision_json(self, *a: Any, **k: Any) -> dict:
        self.calls += 1
        return dict(self.reply)


@pytest.fixture
def seats():
    s = _Seats({"frames": [3, 4], "best": 3})
    segment.configure(s)
    yield s
    segment.configure(None)


def _frames(n: int) -> list:
    return [(float(i * 10), Image.new("RGB", (160, 90), (i * 10, 0, 0))) for i in range(n)]


def test_choose_time_takes_the_best_frame_and_caches(settings, seats) -> None:
    shot = _shot(typ=ShotType.footage, real_thing=RealThing(name="Hurricane Turn train", kind="object"))
    t = segment.choose_time("vid1", lambda: _frames(8), shot, settings, "Alaska Railroad")
    assert t == 20.0 and seats.calls == 1
    assert segment.choose_time("vid1", lambda: _frames(8), shot, settings) == 20.0 and seats.calls == 1  # cached


def test_choose_time_none_when_no_frame_shows_it(settings) -> None:
    s = _Seats({"frames": [], "best": 0})
    segment.configure(s)
    try:
        assert segment.choose_time("vid2", lambda: _frames(6), _shot(typ=ShotType.footage), settings) is None
    finally:
        segment.configure(None)


def test_mock_seat_never_chooses(settings) -> None:
    s = _Seats({"frames": [1], "best": 1, "_mock": True})
    segment.configure(s)
    try:
        assert segment.choose_time("vid3", lambda: _frames(6), _shot(typ=ShotType.footage), settings) is None
    finally:
        segment.configure(None)


def test_storyboard_tiles_cover_the_video_in_order() -> None:
    fmt = {"rows": 2, "columns": 2, "fps": 0.5, "fragments": [{"url": "u0", "duration": 8.0}, {"url": "u1", "duration": 8.0}]}
    tiles = segment.storyboard_tiles(fmt, 14.0)
    assert [t for t, _, _ in tiles] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    assert tiles[4][1:] == (1, 0)


def test_youtube_frames_crop_sprite_tiles(monkeypatch) -> None:
    sheet = Image.new("RGB", (320, 180))
    for j in range(4):
        sheet.paste(Image.new("RGB", (160, 90), (60 * j, 0, 0)), ((j % 2) * 160, (j // 2) * 90))
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=95)
    monkeypatch.setattr(base, "http_get_bytes", lambda url, timeout=60: buf.getvalue())
    info = {"duration": 80.0, "formats": [{"format_note": "storyboard", "rows": 2, "columns": 2, "width": 160,
                                           "height": 90, "fragments": [{"url": f"u{i}", "duration": 20.0} for i in range(4)]}]}
    frames = segment.youtube_frames(info, n=4)
    assert len(frames) == 4 and all(im.size == (160, 90) for _, im in frames)


def test_youtube_window_centres_on_the_chosen_frame(settings, seats, monkeypatch) -> None:
    from vidbush.source import youtube
    monkeypatch.setattr(segment, "youtube_frames", lambda info, n=20: _frames(10))
    shot = _shot(typ=ShotType.footage, intent="the train at the flag stop")
    a, b = youtube.frame_window({"id": "abc", "duration": 600.0, "title": "t"}, shot, settings)
    assert a <= 20.0 <= b and b - a <= youtube.window_length(shot) + 2 * youtube.PADDING_S


# ---------------------------------------------------------------- keyframes from played seconds

def test_keyframes_come_from_the_seconds_that_play(tmp_path) -> None:
    from vidbush.source import keyframes
    src = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=20",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src)], check=True)
    times: list[float] = []
    real = keyframes._frame
    keyframes._frame = lambda path, t, out: (times.append(t), real(path, t, out))[1]
    try:
        c = Candidate(id="c", source=SourceKind.archive, local_path=str(src), media="video", in_point=0.0, out_point=20.0)
        keyframes.extract_keyframes(c, tmp_path / "k", play_s=5.0)
    finally:
        keyframes._frame = real
    assert max(times) <= 5.0


# ---------------------------------------------------------------- scene pool

def test_source_pool_fetches_each_asset_once(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    calls: list[str] = []

    class Shared:
        name, media, classes, matches_description = "archive", {"image"}, frozenset({"archive"}), True

        def available(self, settings: Any) -> bool:
            return True

        def search(self, query: str, media: str, n: int, settings: Any, shot: Any = None) -> list[RawResult]:
            return [RawResult(source="archive", url=f"https://same/{i}", title=f"Nome 1925 {i}", media="image")
                    for i in range(4)]

        def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Any) -> list[Candidate]:
            calls.append(raw.url)
            p = out_dir / (raw.url.rsplit("/", 1)[1] + ".jpg")
            p.parent.mkdir(parents=True, exist_ok=True)
            i = int(raw.url.rsplit("/", 1)[1])
            im = Image.new("RGB", (1920, 1080))
            im.paste(Image.new("RGB", (480, 1080), (255, 255, 255)), (480 * i, 0))  # four different pictures
            im.save(p)
            return base.make_candidates(raw, p, shot, settings)

    monkeypatch.setattr(registry, "active_adapters", lambda ctx: [Shared()])
    shots = [_shot(i, queries=["Nome 1925"]) for i in range(2)]
    out = registry.source_pool(shots, ctx)
    assert sorted(calls) == sorted(set(calls))  # no asset fetched twice across the scene
    assert len(out["s000"]) + len(out["s001"]) == 4


# ---------------------------------------------------------------- one picture through two sources

def test_same_still_from_two_sources_keeps_the_open_licence(make_ctx, monkeypatch, tmp_path) -> None:
    from vidbush.source import keyframes
    ctx = make_ctx(mode="script")
    img = Image.new("RGB", (1600, 900))
    for x in range(0, 1600, 100):
        img.paste(Image.new("RGB", (50, 900), (x % 255, 90, 160)), (x, 0))
    paths = []
    for name, size in (("web.jpg", (1600, 900)), ("commons.jpg", (1400, 788)), ("other.jpg", (1600, 900))):
        p = tmp_path / name
        (img.resize(size) if name != "other.jpg" else Image.new("RGB", size, (200, 30, 30))).save(p)
        paths.append(p)
    monkeypatch.setattr(keyframes, "extract_keyframes",
                        lambda c, d, **kw: setattr(c, "keyframes", [c.local_path]) or c.keyframes)
    web = Candidate(id="w", source=SourceKind.webimage, local_path=str(paths[0]), media="image", width=1600,
                    height=900, licence="unknown (web image search)")
    com = Candidate(id="c", source=SourceKind.wikimedia, local_path=str(paths[1]), media="image", width=1400,
                    height=788, licence="CC BY-SA 4.0")
    oth = Candidate(id="o", source=SourceKind.webimage, local_path=str(paths[2]), media="image", width=1600,
                    height=900, licence="unknown (web image search)")
    out = registry.finalize(_shot(), [web, com, oth], ctx)
    assert [c.source for c in out] == [SourceKind.wikimedia, SourceKind.webimage] and out[1].local_path == str(paths[2])


def test_one_sources_queries_interleave_by_rank() -> None:
    def hit(q: str, i: int) -> RawResult:
        return RawResult(source="webimage", url=f"https://w/{q}/{i}", title=q, extra={"query": q, "rank": i})
    merged = registry._merge([hit("vague", 1), hit("vague", 2), hit("vague", 3), hit("exact", 1), hit("exact", 2)])
    assert [r.url for r in merged[:3]] == ["https://w/vague/1", "https://w/exact/1", "https://w/vague/2"]
