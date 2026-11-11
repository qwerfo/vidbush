"""Run 2 sourcing losses: ranking instead of dropping, plain word forms, media pools, refunds, the
leftover pass and logged drops (fake adapters, offline)."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from vidbush.models import Candidate, Shot, ShotList, ShotType, SourceKind
from vidbush.source import base, registry, throttle, youtube
from vidbush.source.base import RawResult


@pytest.fixture(autouse=True)
def fresh():
    throttle.reset()
    throttle.STATE.rates.pop("api.pexels.com", None)
    yield
    throttle.reset()


class Log:
    def __init__(self) -> None:
        self.lines: list[tuple[str, dict]] = []

    def info(self, msg: str, **kw: Any) -> None:
        self.lines.append((msg, kw))

    warn = info


class Ctx:
    def __init__(self) -> None:
        self.log = Log()


def raw(title: str, n: int = 0, score: float = 0.7) -> RawResult:
    return RawResult(source="pexels", url=f"https://px/{n}", title=title, media="video", score=score)


class Fake:
    """A web source: one result per query titled with the query (on topic), unless throttle is set."""

    def __init__(self, name: str = "pexels", media: frozenset = frozenset({"video", "image"}),
                 throttle_all: bool = False) -> None:
        self.name, self.media, self.throttle_all = name, set(media), throttle_all
        self.calls: list[tuple[str, str]] = []
        self.shares: list[float] = []
        self.ctx: Any = None
        self.lock = threading.Lock()

    def available(self, settings: Any) -> bool:
        return True

    def search(self, query: str, media: str, n: int, settings: Any, shot: Any = None, **kw: Any) -> list[RawResult]:
        if self.throttle_all:
            raise throttle.Throttled("rate limit: next slot in 60s")
        with self.lock:
            self.calls.append((shot.id, query))
            if self.ctx is not None:
                self.shares.append(throttle.share(registry.run_id(self.ctx), self.name))
        return [RawResult(source=self.name, url=f"https://{self.name}/{shot.id}/{query.replace(' ', '_')}.mp4",
                          title=query, media="video", score=0.7)]

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Any) -> list:
        return []


def _shot(i: int, typ: ShotType = ShotType.footage) -> Shot:
    return Shot(id=f"s{i:03d}", index=i, beat_id="b001", start=4 * i, end=4 * i + 4, duration=4.0, type=typ,
                intent="harbour cranes", queries=[f"harbour cranes {i} a", f"harbour cranes {i} b",
                                                   f"harbour cranes {i} c"])


def _shotlist(shots: list[Shot]) -> ShotList:
    return ShotList(project_id="t1", duration=4.0 * len(shots), shots=shots)


# A. a relevance filter ranks results, it never removes the last ones

def test_relevance_demotes_off_topic_results_to_fillers_after_matches() -> None:
    ctx = Ctx()
    q = "Panama Canal construction footage"
    out = registry.relevant("archive", q, [raw("SpanishPod Archive", 1), raw("Panama Canal Construction in 1912", 2)],
                            ctx)
    assert [r.url for r in out] == ["https://px/2", "https://px/1"]  # nothing dropped, on topic first
    assert out[0].score == 0.7 and out[1].score == 0.0
    assert ctx.log.lines == [("off-topic results demoted to fillers",
                              {"source": "archive", "query": q, "demoted": 1})]


def test_one_in_three_specific_words_is_a_match() -> None:
    q = "sled dog team mushing trail race"  # 6 specific words
    hit = raw("Dog sled ride", 1)  # 2 of 6
    assert base.relevance(q, hit) == pytest.approx(1 / 3)
    assert registry.relevant("youtube", q, [hit], Ctx())[0].score == 0.7


def test_fillers_never_count_as_enough_results() -> None:
    fillers = {"pexels": [raw("x", i, score=0.0) for i in range(20)]}
    assert not registry.enough_results(fillers, 4)
    assert registry.enough_results({"pexels": [raw("x", i) for i in range(6)]}, 4)


def test_era_mismatch_stays_a_hard_drop_also_for_cached_results(make_ctx) -> None:
    ctx = make_ctx(mode="script")
    shot = _shot(0).model_copy(update={"intent": "gold rush 1898", "queries": ["gold rush 1898"]})
    adapter = Fake()
    hits = [RawResult(source="pexels", url="https://px/old", title="Gold rush 1898", media="video"),
            RawResult(source="pexels", url="https://px/new", title="Gold rush reenactment 2015", media="video")]
    registry.cache_put("pexels", "gold rush 1898", "video", 5, hits, ctx.settings)
    out = registry._search_one(adapter, "gold rush 1898", "video", 5, shot, ctx)
    assert [r.url for r in out] == ["https://px/old"] and not adapter.calls  # served from the cache, screened


# B. words match in their singular and plain forms

@pytest.mark.parametrize("a,b", [("dogs", "dog"), ("planes", "plane"), ("Alaskan", "Alaska"), ("Alaskans", "alaska"),
                                 ("cities", "city"), ("boxes", "box"), ("churches", "church"), ("mining", "mine"),
                                 ("constructions", "construction"), ("ships", "ship"), ("movies", "movie"),
                                 ("prairies", "prairie"), ("bushes", "bush"), ("dancing", "dance")])
def test_plural_and_plain_forms_share_a_stem(a: str, b: str) -> None:
    assert base._stem(a) == base._stem(b)


def test_words_ending_in_s_that_are_not_plurals_keep_it() -> None:
    assert base._stem("glass") == "glass" and base._stem("bus") == "bus" and base._stem("virus") == "virus"


@pytest.mark.parametrize("a,b", [("plane", "plan"), ("planes", "plan"), ("mine", "min"), ("notes", "not"),
                                 ("cane", "can"), ("here", "her")])
def test_a_stem_never_turns_a_word_into_another_common_word(a: str, b: str) -> None:
    assert base._stem(a) != base._stem(b)


def test_short_title_words_are_no_match_for_long_e_query_words() -> None:
    assert base.relevance("bush plane landing", raw("Alaska Travel Plan 2024")) == 0.0
    assert base.relevance("gold mine Alaska", raw("Gold price update in 5 min")) == pytest.approx(1 / 3)
    assert base.relevance("prairies Kansas", raw("prairie dog")) == 0.5


# C. a shot shares only the budgets of the sources it can search

def test_still_shots_hold_no_share_of_a_video_only_budget(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    yt = Fake("youtube", frozenset({"video"}))
    yt.ctx = ctx
    monkeypatch.setattr(registry, "active_adapters", lambda c: [yt])
    throttle.STATE.budget["youtube"] = 6
    shots = [_shot(0), _shot(1), *[_shot(i, ShotType.image) for i in range(2, 6)]]
    registry.source_all(_shotlist(shots), ctx)
    assert yt.shares[0] == 3.0  # 6 over the 2 video shots, not over all 6 shots
    assert {sid for sid, _q in yt.calls} == {"s000", "s001"} and len(yt.calls) == 6


# D. a search is charged only when it is sent

def test_refund_gives_back_a_search_that_was_not_sent() -> None:
    throttle.STATE.budget["pexels"] = 2
    assert throttle.take_budget("r", "pexels", "a")
    throttle.refund_budget("r", "pexels", "a")
    u = throttle.usage("r")["pexels"]
    assert u["used"] == 0 and u["shots"] == 0
    assert throttle.take_budget("r", "pexels", "a") and throttle.take_budget("r", "pexels", "a")


def test_throttled_searches_do_not_use_the_budget(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    adapter = Fake(throttle_all=True)
    monkeypatch.setattr(registry, "active_adapters", lambda c: [adapter])
    throttle.STATE.budget["pexels"] = 4
    registry.source_all(_shotlist([_shot(0), _shot(1)]), ctx)
    assert throttle.usage(registry.run_id(ctx))["pexels"]["used"] == 0


# E. budget left at the end is spent on the queries it deferred

def test_leftover_budget_runs_deferred_queries(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    adapter = Fake()
    monkeypatch.setattr(registry, "active_adapters", lambda c: [adapter])
    throttle.STATE.budget["pexels"] = 4  # 2 each, but s000 is answered from the cache and never spends
    shots = [_shot(0), _shot(1)]
    n = registry._per_adapter_n(adapter, ctx.settings.sources.max_candidates)
    for i, q in enumerate(shots[0].queries):
        registry.cache_put("pexels", q, "video", n, [raw(q, i)], ctx.settings)
    registry.source_all(_shotlist(shots), ctx)
    assert [q for sid, q in adapter.calls if sid == "s001"] == shots[1].queries  # its deferred third query ran
    assert all(sid == "s001" for sid, _q in adapter.calls)
    assert throttle.usage(registry.run_id(ctx))["pexels"]["used"] == 3


# F. every candidate drop is in the run log

def test_fetch_drops_are_logged_with_shot_and_reason(make_ctx) -> None:
    ctx = make_ctx(mode="script")
    log = Log()
    ctx.log = log

    class Floor(Fake):
        def fetch(self, raw, out_dir, shot, settings):
            raise base.Rejected("480p is below the 720p floor")

    shot = _shot(0)
    plan = [raw("a", 1), raw("b", 2).model_copy(update={"source": "nasa"})]
    out = registry._fetch_raws(shot, plan, {"pexels": Floor(), "nasa": Fake("nasa")}, ctx, 4, "video")
    assert out == []
    assert [(m, kw["shot"], kw["reason"]) for m, kw in log.lines] == [
        ("source result dropped", "s000", "480p is below the 720p floor"),
        ("source result dropped", "s000", "no usable file")]


# G. a search is charged only when it was sent: every pass, every 429 retry

def test_throttled_prefer_cc_search_refunds_both_passes(make_ctx) -> None:
    ctx = make_ctx(mode="script")
    ctx.brand.licence_policy.prefer_creative_commons = True

    class Cooling(Fake):
        search_hints = ("prefer_cc",)

        def search(self, query, media, n, settings, shot=None, **kw):
            assert kw.get("prefer_cc")
            raise throttle.Throttled("youtube cooling down after a rate limit (300s left)")

    registry._search_one(Cooling("youtube", frozenset({"video"})), "canal", "video", 5, _shot(0), ctx)
    assert throttle.usage(registry.run_id(ctx))["youtube"]["used"] == 0


def test_youtube_counts_the_creative_commons_pass_it_sent(monkeypatch) -> None:
    calls: list[str] = []

    def extract(target, opts):
        calls.append(target)
        if len(calls) == 2:
            raise throttle.Throttled("youtube: rate limit: next slot in 200s")
        return {"entries": []}

    def cooling(target, opts):
        raise throttle.Throttled("youtube cooling down")

    monkeypatch.setattr(youtube, "cookie_opts", lambda s: {})
    monkeypatch.setattr(youtube, "ydl_extract", extract)
    with pytest.raises(throttle.Throttled) as e:
        youtube.YoutubeAdapter().search("canal", "video", 5, None, prefer_cc=True)
    assert e.value.sent == 1  # the CC pass went out, the normal pass did not
    monkeypatch.setattr(youtube, "ydl_extract", cooling)
    with pytest.raises(throttle.Throttled) as e:  # the CC pass no longer swallows a throttle
        youtube.YoutubeAdapter().search("canal", "video", 5, None, prefer_cc=True)
    assert e.value.sent == 0


def test_throttled_retry_after_a_429_is_not_refunded(make_ctx) -> None:
    from vidbush.providers.base import RateLimitError

    def call():
        raise RateLimitError("429 Too Many Requests", retry_after=30)

    with pytest.raises(throttle.Throttled) as e:
        base._rate_limited(call, "https://api.example.test/search")
    assert e.value.sent == 1  # the first attempt reached the host; the 30 s cooldown stopped the retry

    ctx = make_ctx(mode="script")

    class Sent429(Fake):
        def search(self, query, media, n, settings, shot=None, **kw):
            err = throttle.Throttled("cooling")
            err.sent = 1
            raise err

    registry._search_one(Sent429(), "canal", "video", 5, _shot(0), ctx)
    assert throttle.usage(registry.run_id(ctx))["pexels"]["used"] == 1


# H. the leftover pass resends only what was not sent, and keeps the still shots' fallback search

class Down(Fake):
    def search(self, query, media, n, settings, shot=None, **kw):
        self.calls.append((shot.id, query))
        raise RuntimeError("HTTPSConnectionPool(host='archive.org'): Read timed out. (read timeout=30)")


def test_a_failed_search_is_not_sent_again_in_the_leftover_pass(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    down = Down("archive")
    monkeypatch.setattr(registry, "active_adapters", lambda c: [down])
    registry.source_all(_shotlist([_shot(i) for i in range(3)]), ctx)
    assert len(down.calls) == 9


class OffTopicStills(Fake):
    def search(self, query, media, n, settings, shot=None, **kw):
        return [RawResult(source=self.name, url=f"https://w/{shot.id}/{query.replace(' ', '_')}.jpg",
                          title="State of the Union Address 2010", media="image", score=0.8)]

    def fetch(self, raw, out_dir, shot, settings):
        self.calls.append((shot.id, raw.title))
        return [Candidate(id="x", source=SourceKind.wikimedia, url=raw.url, media="image", title=raw.title)]


class Frames(Fake):
    def fetch(self, raw, out_dir, shot, settings):
        return [Candidate(id="y", source=SourceKind.youtube, url=raw.url, media="video", title=raw.title)]


def _no_files(monkeypatch) -> None:
    monkeypatch.setattr(registry, "finalize", lambda shot, cands, ctx, taken=(): cands)
    monkeypatch.setattr(registry, "_still_from_video", lambda c, d: c.model_copy(update={"media": "image"}))


def test_leftover_pass_keeps_one_search_per_unmatched_still_shot(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    _no_files(monkeypatch)
    stills, yt = OffTopicStills("wikimedia", frozenset({"image"})), Frames("youtube", frozenset({"video"}))
    monkeypatch.setattr(registry, "active_adapters", lambda c: [stills, yt])
    throttle.STATE.budget["youtube"] = 3
    shots = [_shot(0), _shot(1), _shot(2, ShotType.image)]
    n = registry._per_adapter_n(yt, ctx.settings.sources.max_candidates)
    for i, q in enumerate(shots[0].queries):  # s000 is served from the cache: s001 gets 1 search, defers 2
        registry.cache_put("youtube", q, "video", n, [raw(q, i)], ctx.settings)
    registry.source_all(_shotlist(shots), ctx)
    assert [q for sid, q in yt.calls if sid == "s001"] == shots[1].queries[:2]  # one deferred query held back
    assert [sid for sid, _q in yt.calls].count("s002") == 1  # the still shot's frame search got it
    assert {c.source for c in shots[2].candidates} == {SourceKind.youtube}


# I. off-topic results never block a match and never fill a shot

def test_off_topic_stills_do_not_block_on_topic_video_frames(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    _no_files(monkeypatch)
    stills, yt = OffTopicStills("wikimedia", frozenset({"image"})), Frames("youtube", frozenset({"video"}))
    monkeypatch.setattr(registry, "active_adapters", lambda c: [stills, yt])
    shot = _shot(0, ShotType.image)
    registry.source_all(_shotlist([shot]), ctx)
    assert shot.candidates and all(c.source == SourceKind.youtube for c in shot.candidates)
    assert not stills.calls  # the off-topic still was never downloaded


class OffTopicVideos(Frames):
    def search(self, query, media, n, settings, shot=None, **kw):
        return [RawResult(source=self.name, url=f"https://y/{query.replace(' ', '_')}/{i}", title=t,
                          media="video", score=0.6)
                for i, t in enumerate(["SpanishPod Archive", "Lofi beats to study", "Minecraft gameplay"])]

    def fetch(self, raw, out_dir, shot, settings):
        self.calls.append((shot.id, raw.title))
        return super().fetch(raw, out_dir, shot, settings)


def test_a_shot_with_only_off_topic_results_gets_a_few_flagged_fillers(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    _no_files(monkeypatch)
    yt = OffTopicVideos("youtube", frozenset({"video"}))
    monkeypatch.setattr(registry, "active_adapters", lambda c: [yt])
    shot = _shot(0)
    registry.source_all(_shotlist([shot]), ctx)
    few = max(1, ctx.settings.sources.max_candidates // 4)
    assert len(yt.calls) == few and len(shot.candidates) == few
    assert all("off_topic" in c.flags for c in shot.candidates)
    assert not registry._has_sourced(shot)  # the next run sources it again with a fresh budget
