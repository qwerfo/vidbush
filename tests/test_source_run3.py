"""Run 3 sourcing rules: short keyword queries for every-word engines, their cache form, Pixabay tags and
link lifetime, still repair, and the offline replay of a run's sourcing from its search cache."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from vidbush.models import Shot, ShotList, ShotType
from vidbush.source import base, registry, replay, throttle
from vidbush.source.base import RawResult, keyword_query


@pytest.fixture(autouse=True)
def fresh():
    throttle.reset()
    yield
    throttle.reset()


def test_keyword_query_keeps_the_first_specific_words_in_their_spelling() -> None:
    assert keyword_query("Leonhard Seppala and Togo photo", 3) == "Leonhard Seppala Togo"
    assert keyword_query("Alaska biplane 1920s skis snow photo", 3) == "Alaska biplane skis"
    assert keyword_query("Yup'ik dancing drumming school gym", 3) == "Yup'ik dancing drumming"
    assert keyword_query("archival footage 1950s", 3) == "archival footage 1950s"  # nothing specific: unchanged
    assert keyword_query("dog Dog dogs sled", 2) == "dog dogs"  # repeated spelling counted once


def test_keyword_query_skips_function_words_of_narration() -> None:
    assert keyword_query("When they were building the pipeline through Fairbanks", 3) == "building pipeline Fairbanks"
    assert keyword_query("Alaska is about as wide as Interstate Ten is long footage", 3) == "Alaska Interstate Ten"
    assert base.content_tokens("what this town has become") == {"town", "become"}


def test_every_word_engines_send_short_queries(monkeypatch, settings) -> None:
    from vidbush.source.archive import ArchiveAdapter
    from vidbush.source.loc import LocAdapter
    from vidbush.source.nasa import NasaAdapter
    from vidbush.source.wikimedia import WikimediaAdapter
    sent: list[str] = []

    def get_json(url: str, params=None, headers=None, timeout: float = 20.0) -> Any:
        sent.append(str(params.get("q") or params.get("gsrsearch")))
        return {}

    monkeypatch.setattr(base, "http_get_json", get_json)
    for a in (WikimediaAdapter(), ArchiveAdapter(), NasaAdapter(), LocAdapter()):
        a.search("vintage photo of a red sled dog team crossing ice", "image", 4, settings)
    assert sent[0] == "red sled dog filetype:bitmap" and sent[1].startswith("(red sled dog)")
    assert sent[2] == sent[3] == "red sled dog"


class _ShortAdapter:
    """An every-word engine: sends keyword_query(query, 3) and counts what it sent."""
    name, media, keyword_words = "wikimedia", {"image"}, 3

    def __init__(self) -> None:
        self.sent: list[str] = []

    def available(self, settings: Any) -> bool:
        return True

    def search(self, query: str, media: str, n: int, settings: Any, shot: Any = None, **kw: Any) -> list[RawResult]:
        self.sent.append(keyword_query(query, self.keyword_words))
        return [RawResult(source=self.name, url=f"https://c.example/{len(self.sent)}", media=media,
                          title="National Atlas United States map")]

    def fetch(self, *a: Any, **kw: Any) -> list:
        return []


def test_queries_that_shorten_alike_share_one_search(make_ctx) -> None:
    """The cache is keyed on the request sent: two long queries with one keyword form cost one search."""
    ctx = make_ctx(mode="script")
    a = _ShortAdapter()
    shot = Shot(id="s1", index=0, beat_id="b", start=0, end=4, duration=4, type=ShotType.image, intent="atlas")
    first = registry._search_one(a, "National Atlas of the United States map Alaska", "image", 4, shot, ctx)
    second = registry._search_one(a, "National Atlas United States 1970 page", "image", 4, shot, ctx)
    assert a.sent == ["National Atlas United"] and [r.url for r in first] == [r.url for r in second]
    assert throttle.spent(registry.run_id(ctx))["wikimedia"] == 1


def test_a_shot_never_sends_one_request_twice(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    a = _ShortAdapter()
    monkeypatch.setattr(registry, "cache_get", lambda *x, **k: None)  # only the in-shot check can stop the repeat
    monkeypatch.setattr(registry, "cache_put", lambda *x, **k: None)
    shot = Shot(id="s1", index=0, beat_id="b", start=0, end=4, duration=4, type=ShotType.image, intent="atlas",
                queries=["National Atlas of the United States map", "National Atlas United page", "Juneau harbor"])
    with registry.SearchPools([a]) as pools:
        registry._search_all([shot], [a], ctx, pools)
    assert a.sent == ["National Atlas United", "Juneau harbor"]


def test_a_slow_host_never_holds_back_another_hosts_rounds(make_ctx, monkeypatch) -> None:
    """A paced host (Pexels, one search per 22.5 s) still busy with round 1 does not stop a fast host
    from sending its round 2 and 3 queries, and the slow host's results still reach the shot."""
    import threading
    ctx = make_ctx(mode="script")
    monkeypatch.setattr(registry, "cache_get", lambda *x, **k: None)
    monkeypatch.setattr(registry, "cache_put", lambda *x, **k: None)
    fast_done = threading.Event()
    seen_by_slow: list[bool] = []

    class Fast(_ShortAdapter):
        name, keyword_words = "wikimedia", 0

        def search(self, query: str, *a: Any, **k: Any) -> list[RawResult]:
            self.sent.append(query)
            if len(self.sent) == 3:
                fast_done.set()
            return []

    class Slow(_ShortAdapter):
        name, keyword_words = "pexels", 0

        def search(self, query: str, media: str, *a: Any, **k: Any) -> list[RawResult]:
            seen_by_slow.append(fast_done.wait(2.0))  # a round barrier keeps the fast host waiting here
            return [RawResult(source="pexels", url=f"https://px/{query}", media=media, title=query)]

    fast, slow = Fast(), Slow()
    shot = Shot(id="s1", index=0, beat_id="b", start=0, end=4, duration=4, type=ShotType.image, intent="x",
                queries=["harbour cranes", "container ship", "dock workers"])
    with registry.SearchPools([fast, slow]) as pools:
        found = registry._search_all([shot], [fast, slow], ctx, pools)
    assert fast.sent == ["harbour cranes", "container ship", "dock workers"] and all(seen_by_slow)
    assert len(found["s1"]["pexels"]) == 3  # every slow result joined the shot


def test_pixabay_tags_count_for_relevance(settings) -> None:
    from vidbush.source import pixabay
    data = {"hits": [{"pageURL": "https://pixabay.com/photos/x-1/", "user": "u", "imageWidth": 4000,
                      "imageHeight": 3000, "largeImageURL": "https://pixabay.com/get/x.jpg",
                      "tags": "huskies, dogs, race, sled dog racing, snow, musher"}]}
    r = pixabay.parse_photos(data)[0]
    assert r.title == "Huskies, dogs, race" and "musher" in r.extra["tags"]
    assert base.relevance("musher snow", r) == 1.0  # both words are only in the tags past the title's three


def test_pixabay_search_cache_outlives_no_download_link(settings) -> None:
    registry.cache_put("pixabay", "q", "image", 4, [RawResult(source="pixabay", url="u")], settings)
    path = registry._cache_path("pixabay", "q", "image", 4, settings)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["ts"] = time.time() - 25 * 3600
    path.write_text(json.dumps(data), encoding="utf-8")
    assert registry.cache_get("pixabay", "q", "image", 4, settings) is None
    registry.cache_put("pexels", "q", "image", 4, [RawResult(source="pexels", url="u")], settings)
    p2 = registry._cache_path("pexels", "q", "image", 4, settings)
    d2 = json.loads(p2.read_text(encoding="utf-8"))
    d2["ts"] = time.time() - 25 * 3600
    p2.write_text(json.dumps(d2), encoding="utf-8")
    assert registry.cache_get("pexels", "q", "image", 4, settings) is not None


def test_padded_scan_is_repaired_on_download(monkeypatch, settings) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (10, 20, 30)).save(buf, "JPEG")
    padded = buf.getvalue()[:-60] + b"\x00" * 200  # end marker lost, zero padding instead

    def download(url: str, dest: Path, headers=None, timeout: float = 60.0, max_bytes: int = 0) -> Path:
        dest.write_bytes(padded)
        return dest

    monkeypatch.setattr(base, "http_download", download)
    path = base.cached_download("https://example.org/scan.jpg", ".jpg", settings)
    with Image.open(path) as im:
        im.load()  # strict decode works now
        assert im.size == (64, 48)


def test_still_too_large_to_check_is_kept_as_downloaded(monkeypatch, tmp_path) -> None:
    path = tmp_path / "huge.jpg"
    Image.new("RGB", (64, 48)).save(path, "JPEG")
    data = path.read_bytes()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)  # 64 x 48 counts as a decompression bomb now
    base.repair_still(path)  # no DecompressionBombError out of the download
    assert path.read_bytes() == data


# ---------------------------------------------------------------- offline replay

def _project(tmp: Path) -> Path:
    shots = [Shot(id="s001", index=0, beat_id="b", start=0, end=4, duration=4, type=ShotType.footage,
                  intent="harbour", queries=["harbour cranes", "container ship"]),
             Shot(id="s002", index=1, beat_id="b", start=4, end=8, duration=4, type=ShotType.ai_image,
                  intent="lighthouse", queries=["lighthouse storm"]),
             Shot(id="s003", index=2, beat_id="b", start=8, end=12, duration=4, type=ShotType.footage,
                  intent="nothing cached", queries=["never searched"]),
             Shot(id="s004", index=3, beat_id="b", start=12, end=16, duration=4, type=ShotType.image,
                  intent="foghorn", queries=["foghorn"])]
    proj = tmp / "proj"
    proj.mkdir()
    (proj / "shotlist.json").write_text(ShotList(project_id="p", duration=12, shots=shots).model_dump_json(),
                                        encoding="utf-8")
    (proj / "project.json").write_text(json.dumps({"id": "p", "brand_id": "default"}), encoding="utf-8")
    return proj


def test_replay_plans_from_cache_hits_only(monkeypatch, settings, tmp_path) -> None:
    monkeypatch.setattr(replay, "load_settings", lambda: settings)
    n = registry._per_adapter_n(type("A", (), {"name": "pexels"})(), settings.sources.max_candidates)
    on = RawResult(source="pexels", url="https://px/1", title="Harbour cranes at dusk", media="video")
    off = RawResult(source="pexels", url="https://px/2", title="Cat on a sofa", media="video")
    registry.cache_put("pexels", "harbour cranes", "video", n, [on, off], settings)
    still = RawResult(source="pexels", url="https://px/3", title="Lighthouse in a storm", media="image")
    registry.cache_put("pexels", "lighthouse storm", "image", n, [still], settings)
    registry.cache_put("pexels", "foghorn", "image", n, [], settings)
    horn = RawResult(source="pexels", url="https://px/4", title="Foghorn blowing", media="video")
    registry.cache_put("pexels", "foghorn", "video", n, [horn], settings)
    rep = replay.replay(_project(tmp_path), adapter_names=["pexels"])
    s1, s2, s3 = (rep["shots"][k] for k in ("s001", "s002", "s003"))
    assert s1["matches"] == 1 and s1["fillers"] == 1 and s1["plan"] == 2 and s1["plan_matches"] == 1
    assert s1["cache_misses"] == ["pexels:video:container ship"]  # sent past the budget, failed offline
    assert rep["restored_types"] == {"s002": "image"} and s2["plan"] == 1  # the AI still was a still shot
    assert s3["plan"] == 0 and s3["cache_misses"] == ["pexels:video:never searched"]
    assert rep["shots"]["s004"]["frame_fallback_plan"] == {"plan": 1, "matches": 1}  # no still: video frames
    assert "frame_fallback_plan" not in s2


def test_replay_reads_drop_events_of_one_stage(monkeypatch, settings, tmp_path) -> None:
    raws = [RawResult(source="pexels", url="https://px/1", title="Harbour cranes", media="video"),
            RawResult(source="pexels", url="https://px/2", title="Cat", media="video")]
    registry.cache_put("pexels", "harbour cranes", "video", 4, raws, settings)
    log = tmp_path / "run.jsonl"
    rows = [{"msg": "stage start", "stage": "sourcing"},
            {"msg": "off-topic results dropped", "source": "pexels", "query": "Harbour  cranes", "dropped": 1},
            {"msg": "stage start", "stage": "judge"},
            {"msg": "off-topic results dropped", "source": "pexels", "query": "harbour cranes", "dropped": 1}]
    log.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ctx = type("C", (), {"settings": settings, "log": replay.Log()})()
    assert replay.replay_drops(log, ctx) == {"events": 1, "results": 2, "now_matches": 1, "now_fillers": 1}
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n" + json.dumps(
        {"msg": "sourcing done", "empty": ["s003"]}), encoding="utf-8")
    assert replay.empty_shots(log) == ["s003"]


def test_a_straggler_stops_once_most_sources_are_done(make_ctx, monkeypatch) -> None:
    """Once most sources finished, one still searching gets straggler_grace more, then its searches not
    yet started are cancelled: it never holds the sourcing stage. The others' results all arrive."""
    import time as _t

    ctx = make_ctx(mode="script")
    monkeypatch.setattr(registry, "cache_get", lambda *x, **k: None)
    monkeypatch.setattr(registry, "cache_put", lambda *x, **k: None)
    monkeypatch.setattr(registry, "straggler_grace", lambda elapsed: 0.2)

    class Fast(_ShortAdapter):
        name, keyword_words = "wikimedia", 0

        def search(self, query: str, media: str, *a: Any, **k: Any) -> list[RawResult]:
            self.sent.append(query)
            return [RawResult(source=self.name, url=f"https://f/{query}", media=media, title=query)]

    class Fast2(Fast):
        name = "archive"

    class Slow(_ShortAdapter):
        name, keyword_words = "pexels", 0

        def search(self, query: str, *a: Any, **k: Any) -> list[RawResult]:
            self.sent.append(query)
            _t.sleep(0.5)
            return []

    fast, fast2, slow = Fast(), Fast2(), Slow()
    shots = [Shot(id=f"s{i}", index=i, beat_id="b", start=i, end=i + 1, duration=1, type=ShotType.image,
                  intent="x", queries=[f"harbour cranes {i}"]) for i in range(8)]
    t0 = _t.monotonic()
    with registry.SearchPools([fast, fast2, slow]) as pools:
        found = registry._search_all(shots, [fast, fast2, slow], ctx, pools)
    assert _t.monotonic() - t0 < 2.5  # 8 slow searches would take 4 s
    assert len(slow.sent) < 8
    assert all(found[s.id]["wikimedia"] and found[s.id]["archive"] for s in shots)
