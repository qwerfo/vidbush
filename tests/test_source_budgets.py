"""Per run source budgets shared fairly by the shots that still need candidates (fake adapter, offline)."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from vidbush.judge import qa
from vidbush.models import Shot, ShotList, ShotType
from vidbush.source import registry, throttle
from vidbush.source.base import RawResult


@pytest.fixture(autouse=True)
def fresh():
    throttle.reset()
    throttle.STATE.rates.pop("api.pexels.com", None)
    yield
    throttle.reset()


class FakePexels:
    """Stock source with a run budget. rich shots get a full page of results on their first query."""
    name = "pexels"
    media = {"video", "image"}

    def __init__(self, rich: set[str] = frozenset(), per_query: int = 1) -> None:
        self.rich, self.per_query = set(rich), per_query
        self.calls: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def available(self, settings: Any) -> bool:
        return True

    def search(self, query: str, media: str, n: int, settings: Any, shot: Any = None, **kw: Any) -> list[RawResult]:
        with self.lock:
            self.calls.append((shot.id, query))
        count = 30 if shot.id in self.rich else self.per_query
        return [RawResult(source="pexels", url=f"https://px/{shot.id}/{query.replace(' ', '_')}/{i}.mp4",
                          title=query, media="video", score=0.7) for i in range(count)]

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Any) -> list:
        return []


def _shots(n: int) -> list[Shot]:
    return [Shot(id=f"s{i:03d}", index=i, beat_id="b001", start=4 * i, end=4 * i + 4, duration=4.0,
                 type=ShotType.footage, intent="harbour cranes",
                 queries=[f"harbour cranes {i} a", f"harbour cranes {i} b", f"harbour cranes {i} c"])
            for i in range(n)]


def _per_shot(adapter: FakePexels) -> dict[str, int]:
    out: dict[str, int] = {}
    for sid, _q in adapter.calls:
        out[sid] = out.get(sid, 0) + 1
    return out


def test_share_is_budget_over_active_shots_and_returns_on_release() -> None:
    throttle.STATE.budget["pexels"] = 10
    throttle.register_shots("r", ["pexels"], ["a", "b", "c", "d", "e"])
    assert throttle.share("r", "pexels") == 2.0
    took = [throttle.take_budget("r", "pexels", "a") for _ in range(4)]
    assert took == [True, True, False, False]  # a cannot drain the budget
    throttle.release_shot("r", "b")
    throttle.release_shot("r", "c")  # b and c found enough without searching: their shares return
    assert throttle.share("r", "pexels") == pytest.approx(10 / 3)
    # 10 over a, d, e: 3 each and the remainder to a (first in timeline order), so a may spend 4
    assert [throttle.take_budget("r", "pexels", "a") for _ in range(3)] == [True, True, False]
    u = throttle.usage("r")["pexels"]
    assert u["used"] == 4 and u["shots"] == 1 and u["deferred"] == 3 and u["refused"] == 0


def test_budget_smaller_than_the_shot_count_gives_every_shot_one_search_first() -> None:
    throttle.STATE.budget["pexels"] = 3
    throttle.register_shots("r", ["pexels"], ["a", "b", "c", "d"])
    assert throttle.take_budget("r", "pexels", "a")
    assert not throttle.take_budget("r", "pexels", "a")  # no second search while others wait
    assert throttle.take_budget("r", "pexels", "b") and throttle.take_budget("r", "pexels", "c")
    assert not throttle.take_budget("r", "pexels", "d")  # the budget itself is spent
    assert throttle.usage("r")["pexels"]["refused"] == 1


def test_a_shot_that_searches_again_rejoins_the_pool_with_its_spend() -> None:
    throttle.STATE.budget["pexels"] = 4
    throttle.register_shots("r", ["pexels"], ["a", "b"])
    assert throttle.take_budget("r", "pexels", "a")
    throttle.release_shot("r", "a")
    assert throttle.share("r", "pexels") == 3.0  # b alone shares what a left
    assert throttle.take_budget("r", "pexels", "a")  # a re-query: a is back, its first search counts
    assert throttle.share("r", "pexels") == 2.0
    assert not throttle.take_budget("r", "pexels", "a")


def test_run_one_scenario_later_shots_still_get_searches(make_ctx, monkeypatch) -> None:
    """Live run 1: 150 Pexels searches went to the first shots. A budget below 3 per shot is now spread."""
    ctx = make_ctx(mode="script")
    adapter = FakePexels()
    monkeypatch.setattr(registry, "active_adapters", lambda c: [adapter])
    throttle.STATE.budget["pexels"] = 12
    shots = _shots(8)
    registry.source_all(ShotList(project_id="t1", duration=32, shots=shots), ctx)
    per = _per_shot(adapter)
    assert set(per) == {s.id for s in shots}  # every shot searched, the last ones included
    assert len(adapter.calls) == 12 and max(per.values()) - min(per.values()) <= 1


def test_shots_that_find_enough_early_return_their_share(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    rich = {"s000", "s001"}
    adapter = FakePexels(rich=rich)
    monkeypatch.setattr(registry, "active_adapters", lambda c: [adapter])
    throttle.STATE.budget["pexels"] = 8  # 2 each for 4 shots, but the rich ones stop after one search
    registry.source_all(ShotList(project_id="t1", duration=16, shots=_shots(4)), ctx)
    per = _per_shot(adapter)
    assert per["s000"] == 1 and per["s001"] == 1
    assert per["s002"] == 3 and per["s003"] == 3  # their 2 each plus the share the rich shots returned
    notes = qa.source_usage_notes(ctx)
    assert notes == ["warning: Pexels search budget spent (8 of 8 searches across 4 shots): later queries were "
                     "skipped, raise settings.sources.query_budget or re-run to use the cache"]


def test_usage_line_for_every_budgeted_source_with_deferred_count(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    adapter = FakePexels()
    monkeypatch.setattr(registry, "active_adapters", lambda c: [adapter])
    throttle.STATE.budget["pexels"] = 5  # 2 shots want 3 each: 3 and 2, one search deferred
    registry.source_all(ShotList(project_id="t1", duration=8, shots=_shots(2)), ctx)
    usage = throttle.usage(registry.run_id(ctx))["pexels"]
    assert usage["used"] == len(adapter.calls) == 5 and usage["shots"] == 2
    assert _per_shot(adapter) == {"s000": 3, "s001": 2}
    assert qa.source_usage_notes(ctx)[0].startswith("warning: Pexels search budget spent (5 of 5 searches across 2 shots)")
    import json
    (ctx.work / registry.SOURCE_NOTES).write_text(json.dumps({"runs": {"r": {"ts": 1, "usage": {
        "loc": {"used": 40, "budget": 300, "refused": 0, "shots": 20, "deferred": 3}}}}}), encoding="utf-8")
    assert qa.source_usage_notes(ctx) == [
        "Library of Congress searches used: 40 of 300 across 20 shots, 3 deferred by fair shares"]


def test_requery_after_sourcing_uses_the_leftover_pool(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    adapter = FakePexels(rich={"s000", "s001"})
    monkeypatch.setattr(registry, "active_adapters", lambda c: [adapter])
    throttle.STATE.budget["pexels"] = 10
    shots = _shots(2)
    registry.source_all(ShotList(project_id="t1", duration=8, shots=shots), ctx)
    assert len(adapter.calls) == 2
    registry.source_shot(shots[0], ctx, queries=["harbour cranes night", "harbour cranes dawn"])
    assert len(adapter.calls) == 4  # the director loop's re-query draws on what is left
