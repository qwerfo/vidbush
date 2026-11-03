"""Research contract with a real seat: bounds, follow-ups, citations tied to fetched URLs, language, queries."""
from __future__ import annotations

from typing import Any, Callable

import pytest

from vidbush.models import Brand, Brief, VideoFormat
from vidbush.providers.base import ProviderError
from vidbush.research import researcher
from vidbush.research.researcher import MAX_FACTS, MIN_FACTS, excerpt_block, run_research, web_queries
from vidbush.util import NullLogger

WIKI = "https://en.wikipedia.org/wiki/Erie_Canal"


class RealSeats:
    """A seat that claims to be real; answers by task (value or callable(user))."""

    def __init__(self, answers: dict[str, Any], search: list[dict[str, str]] | None = None) -> None:
        self.answers = answers
        self.search = search or []
        self.calls: list[tuple[str, str]] = []
        self.queries: list[str] = []

    def is_real(self, role: str) -> bool:
        return role == "researcher" or (role == "search" and bool(self.search))

    def llm_json(self, role: str, task: str, system: str, user: str, stage: str, temperature: float = 0.6) -> Any:
        self.calls.append((task, user))
        v = self.answers.get(task)
        if isinstance(v, Exception):
            raise v
        return v(user) if callable(v) else v

    def web_search(self, query: str, stage: str) -> list[dict[str, str]]:
        self.queries.append(query)
        return self.search

    def tasks(self) -> list[str]:
        return [t for t, _ in self.calls]


def _tag(i: int) -> str:
    """Letters only, so the heuristic fact check (which matches digits) leaves statuses alone."""
    return "".join("abcdefghij"[int(d)] for d in str(i))


def _facts(n: int, start: int = 1, url: str = WIKI) -> list[dict[str, Any]]:
    return [{"id": f"f{i:02d}", "claim": f"Distinct fact {_tag(i)} about the canal.", "source_url": url,
             "confidence": 0.9 if i % 2 else 0.5} for i in range(start, start + n)]


@pytest.fixture
def online(settings, monkeypatch):
    from vidbush.providers import search
    seen: dict[str, Any] = {}

    def wsearch(q: str, n: int = 5, lang: str = "en") -> list[dict[str, str]]:
        seen["lang"] = lang
        return [{"url": f"https://{lang}.wikipedia.org/wiki/Erie_Canal", "title": "Erie Canal"}]

    monkeypatch.setattr(search, "wikipedia_search", wsearch)
    monkeypatch.setattr(search, "wikipedia_extract", lambda title, lang="en": "The canal opened in 1825 in Buffalo.")
    monkeypatch.setattr(researcher, "fetch_page_text", lambda url: f"Full page text of {url} with many words here.")
    settings.offline = False
    settings._seen = seen  # type: ignore[attr-defined]
    return settings


def _run(settings: Any, seats: RealSeats, brief: Brief | None = None, brand: Brand | None = None):
    return run_research(brief or Brief(topic="Erie Canal", target_minutes=8), brand or Brand(id="b"), seats,
                        settings, NullLogger())


def test_too_few_facts_triggers_one_follow_up(online) -> None:
    more: Callable[[str], Any] = lambda user: {"facts": _facts(30, start=9)}
    seats = RealSeats({"research": {"facts": _facts(8)}, "research_more": more})
    r = _run(online, seats)
    assert seats.tasks().count("research_more") == 1
    assert MIN_FACTS <= len(r.facts) <= MAX_FACTS
    assert len({f.id for f in r.facts}) == len(r.facts) and r.facts[8].id == "f09"
    follow_up = dict(seats.calls)["research_more"]
    assert "f01: Distinct fact b about" in follow_up and "f09" in follow_up


def test_too_many_facts_trimmed_confident_first(online) -> None:
    seats = RealSeats({"research": {"facts": _facts(120)}})
    r = _run(online, seats)
    assert len(r.facts) == MAX_FACTS
    assert all(f.confidence == 0.9 for f in r.facts)  # the 60 confident ones survive
    assert "research_more" not in seats.tasks()


def test_citations_tied_to_fetched_urls(online) -> None:
    facts = [{"claim": "Fetched source, other spelling.", "source_url": "http://en.wikipedia.org/wiki/Erie_Canal/",
              "confidence": 0.9},
             {"claim": "Invented citation.", "source_url": "https://madeup.example/page", "confidence": 0.95},
             {"claim": "General knowledge.", "source_url": "", "confidence": 0.9}]
    seats = RealSeats({"research": {"facts": facts + _facts(30, start=4)}})
    r = _run(online, seats)
    by_claim = {f.claim: f for f in r.facts}
    assert by_claim["Fetched source, other spelling."].source_url == WIKI
    invented = by_claim["Invented citation."]
    assert invented.source_url == "" and invented.confidence <= 0.55
    assert by_claim["General knowledge."].confidence <= 0.55
    assert all(s["url"] != "https://madeup.example/page" for s in r.sources)
    prompt = dict(seats.calls)["research"]
    assert '"id": "f01"' in prompt and "exactly one of the URLs listed" in prompt


def test_anchor_request_and_trim(online) -> None:
    anchors = [{"kind": "date", "name": str(1800 + i), "fact_ids": ["f01"]} for i in range(12)]
    anchors.append({"kind": "place", "name": "Buffalo", "lat": 42.9, "lon": -78.9, "fact_ids": ["f02"]})
    seats = RealSeats({"research": {"facts": _facts(30), "anchors": []},
                       "research_anchors": {"anchors": anchors}})
    r = _run(online, seats)
    assert seats.tasks().count("research_anchors") == 1
    assert len(r.anchors) == 10
    assert r.anchors[0].name == "Buffalo" or any(a.name == "Buffalo" for a in r.anchors)  # mappable place kept


def test_fallback_anchors_fill_from_numbers(settings) -> None:
    from vidbush.models import FactCard, Research
    r = Research(topic="t", facts=[FactCard(id=f"f{i:02d}", claim=f"It moved {i * 100} tons.",
                                            numbers=[{"value": i * 100, "unit": "tons"}]) for i in range(1, 8)])
    researcher._add_anchors(r, NullLogger())
    assert len(r.anchors) >= 5 and all(a.kind == "number" for a in r.anchors)


def test_web_queries_from_brief_and_pages_fetched(online) -> None:
    hits = [{"url": "https://history.example/erie", "title": "Erie history", "snippet": "short"}]
    seats = RealSeats({"research": {"facts": _facts(30)}}, search=hits)
    _run(online, seats, Brief(topic="Erie Canal", angle="the Irish labourers", target_minutes=8))
    assert seats.queries == ["Erie Canal the Irish labourers", "Erie Canal facts and figures"]
    assert "Full page text of https://history.example/erie" in dict(seats.calls)["research"]
    top10 = Brief(topic="Largest canals", format=VideoFormat.top10)
    assert web_queries(top10) == ["Largest canals", "Largest canals ranked list"]
    assert all("history facts" not in q for q in web_queries(top10))


def test_brand_language_reaches_wikipedia_and_prompts(online) -> None:
    seats = RealSeats({"research": {"facts": _facts(30, url="https://es.wikipedia.org/wiki/Erie_Canal")}})
    r = _run(online, seats, brand=Brand(id="b", language="es-MX"))
    assert online._seen["lang"] == "es"
    assert "Spanish" in dict(seats.calls)["research"]
    assert r.facts[0].source_url == "https://es.wikipedia.org/wiki/Erie_Canal"


def test_real_seat_failure_fails_the_stage(online) -> None:
    seats = RealSeats({"research": ProviderError("every provider failed")})
    with pytest.raises(ProviderError):
        _run(online, seats)


def test_excerpt_budget_is_shared_fairly() -> None:
    ex = {"https://a/1": "A\n" + "x" * 20000, "https://b/2": "B\n" + "y" * 20000, "https://c/3": "C\nshort snippet"}
    block = excerpt_block(ex, budget=6000)
    assert "short snippet" in block
    assert block.count("x") > 2500 and block.count("y") > 2500
    assert len(block) < 6400
