"""Research and fact check tests: offline fixture path, fallbacks, network failures, place anchors."""
from __future__ import annotations

import sys
import types
from typing import Any

from vidbush.models import Brand, Brief, FactCard, Research
from vidbush.research import researcher
from vidbush.research.factcheck import fact_check, heuristic_status
from vidbush.research.researcher import extract_numbers, run_research
from vidbush.util import NullLogger


class FakeSeats:
    def __init__(self, answers: dict[str, Any] | None = None, real_search: bool = False) -> None:
        self.answers = answers or {}
        self.real_search = real_search
        self.users: dict[str, str] = {}

    def llm_json(self, role: str, task: str, system: str, user: str, stage: str, temperature: float = 0.6) -> Any:
        self.users[task] = user
        return self.answers.get(task)

    def is_real(self, role: str) -> bool:
        return self.real_search

    def web_search(self, query: str, stage: str) -> list[dict[str, str]]:
        return [{"url": "https://example.org/a", "title": "A", "snippet": "The canal opened in 1914 after ten years."}]


def test_offline_fixture_path(make_ctx):
    ctx = make_ctx(mode="brief", topic="The Panama Canal")
    r = run_research(ctx.project.brief, ctx.brand, ctx.seats, ctx.settings, ctx.log)
    assert len(r.facts) >= 30
    assert sum(1 for f in r.facts if f.status == "verified") >= 30
    assert any(f.status == "unverified" for f in r.facts)
    assert len(r.anchors) >= 5 and any(a.lat is not None for a in r.anchors if a.kind == "place")
    assert all(s["url"].startswith("https://en.wikipedia.org/wiki/") for s in r.sources)


def test_offline_unknown_topic_uses_brief_cards(make_ctx):
    ctx = make_ctx(mode="brief", topic="Lighthouses of Lake Superior")
    r = run_research(ctx.project.brief, ctx.brand, ctx.seats, ctx.settings, ctx.log)
    assert r.facts and all(f.status == "unverified" for f in r.facts)
    assert "Lighthouses of Lake Superior" in r.facts[0].claim


def test_online_network_failure_is_not_fatal(settings, monkeypatch):
    from vidbush.providers import search

    def boom(*a: Any, **k: Any) -> Any:
        raise ConnectionError("blocked")

    monkeypatch.setattr(search, "wikipedia_search", boom)
    monkeypatch.setattr(search, "wikipedia_extract", boom)
    settings.offline = False
    brief = Brief(topic="Erie Canal", target_minutes=4)
    r = run_research(brief, Brand(id="b"), FakeSeats(), settings, NullLogger())
    assert r.facts


def test_online_fallback_cards_from_fetched_text(settings, monkeypatch):
    from vidbush.providers import search
    text = ("The Erie Canal opened in 1825 and ran 363 miles from Albany to Buffalo. "
            "It had 83 locks and cost about 7 million dollars to build in total. "
            "Short line. ")
    monkeypatch.setattr(search, "wikipedia_search",
                        lambda q, n=5, lang="en": [{"url": "https://en.wikipedia.org/wiki/Erie_Canal",
                                                    "title": "Erie Canal"}])
    monkeypatch.setattr(search, "wikipedia_extract", lambda title, lang="en": text + "\n== References ==\nRef 1999 junk.")
    monkeypatch.setattr(researcher, "fetch_page_text", lambda url: "")
    settings.offline = False
    seats = FakeSeats(real_search=True)
    r = run_research(Brief(topic="Erie Canal", target_minutes=4), Brand(id="b"), seats, settings, NullLogger())
    claims = [f.claim for f in r.facts]
    assert any("363 miles" in c for c in claims)
    assert all("junk" not in c for c in claims)
    assert r.facts[0].source_url == "https://en.wikipedia.org/wiki/Erie_Canal"
    assert r.facts[0].status == "verified"  # heuristic: numbers found in its own source
    assert "363 miles" in seats.users["research"]
    urls = {s["url"] for s in r.sources}
    assert "https://example.org/a" in urls
    assert any(a.kind == "date" and a.name == "1825" for a in r.anchors)


def test_seat_answer_validated_and_contradicted_dropped(settings):
    answer = {"facts": [
        {"claim": "A true fact with an em dash \u2014 here.", "source_url": "u", "confidence": 3,
         "numbers": [{"value": 1, "unit": "x", "label": "y"}]},
        {"claim": "A false fact.", "confidence": "bad"},
        {"claim": ""},
        "junk"],
        "anchors": [{"kind": "place", "name": "Colon", "lat": "9.36", "lon": -79.9, "fact_ids": ["f01", "f09"]},
                    {"kind": "planet", "name": "Mars"}]}
    verdicts = {"verdicts": [{"id": "f01", "status": "verified"}, {"id": "f02", "status": "contradicted"}]}
    r = run_research(Brief(topic="X", target_minutes=4), Brand(id="b"),
                     FakeSeats({"research": answer, "factcheck": verdicts}), settings, NullLogger())
    assert [f.id for f in r.facts] == ["f01"]
    assert "\u2014" not in r.facts[0].claim and r.facts[0].confidence == 1.0
    assert r.facts[0].status == "verified"
    colon = [a for a in r.anchors if a.name == "Colon"][0]
    assert colon.lat == 9.36 and colon.fact_ids == ["f01"]
    assert all(a.name != "Mars" for a in r.anchors)


def test_place_anchors_from_geo_module(settings, monkeypatch):
    geo = types.ModuleType("vidbush.mg.geo")
    geo.find_places = lambda text: [("Buffalo", 42.886, -78.878)] if "Buffalo" in text else []
    monkeypatch.setitem(sys.modules, "vidbush.mg.geo", geo)
    answer = {"facts": [{"claim": "The canal reached Buffalo in 1825."}],
              "anchors": [{"kind": "place", "name": "Buffalo"}]}
    r = run_research(Brief(topic="X", target_minutes=4), Brand(id="b"), FakeSeats({"research": answer}),
                     settings, NullLogger())
    buf = [a for a in r.anchors if a.name == "Buffalo"][0]
    assert (buf.lat, buf.lon) == (42.886, -78.878)


def test_heuristic_fact_check_status():
    card = FactCard(id="f01", claim="It opened in 1914 at a cost of 375 million dollars.")
    assert heuristic_status(card, "Opened 1914. Cost: $375 million.") == "verified"
    assert heuristic_status(card, "Opened 1915.") is None
    r = Research(topic="t", facts=[card])
    out = fact_check(r, FakeSeats(), NullLogger(), {"": "irrelevant"})
    assert out.facts[0].status == "unverified"


def test_extract_numbers():
    nums = extract_numbers("It cost $375 million and ran 51 miles, cutting costs by 90 percent in 1914.")
    assert {"value": 375000000, "unit": "USD"}.items() <= nums[0].items()
    assert nums[1]["value"] == 51 and nums[1]["unit"] == "miles"
    assert nums[2]["unit"] == "percent"
    assert len(nums) == 3


def test_trim_drops_reference_sections():
    assert "Ref" not in researcher._trim("Body text.\n== References ==\nRef")
