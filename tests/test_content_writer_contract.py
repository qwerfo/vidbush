"""Writer and outline contract: brand language, per-section context, real-mode length check, failure policy."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest

from vidbush.config import REPO_ROOT
from vidbush.models import Brand, Brief, Research, VideoFormat
from vidbush.providers.base import ProviderError
from vidbush.script import prompts
from vidbush.script.outline import heuristic_outline, make_outline
from vidbush.script.writer import ScriptTooShort, write_script
from vidbush.util import read_model

FIX = Path(REPO_ROOT) / "fixtures" / "llm" / "the_panama_canal"


class Seats:
    def __init__(self, answers: Optional[dict[str, Any]] = None, real: bool = False) -> None:
        self.answers = answers or {}
        self.real = real
        self.calls: list[tuple[str, str]] = []

    def is_real(self, role: str) -> bool:
        return self.real

    def llm_json(self, role: str, task: str, system: str, user: str, stage: str, temperature: float = 0.6) -> Any:
        self.calls.append((task, user))
        v = self.answers.get(task)
        if isinstance(v, Exception):
            raise v
        return v(user) if callable(v) else v


@pytest.fixture
def research() -> Research:
    return read_model(FIX / "research.json", Research)


def _brief(minutes: float) -> Brief:
    return Brief(topic="The Panama Canal", target_minutes=minutes, format=VideoFormat.documentary)


def _words(n: int) -> str:
    return " ".join(["canal"] * (n - 1)) + "."


def test_language_name_helpers() -> None:
    assert prompts.language_name("es-MX") == "Spanish" and prompts.language_name("") == "English"
    assert prompts.language_name("xx") == "xx"
    assert "German" in prompts.language_line("de")


def test_brand_language_in_outline_and_script_prompts(research: Research) -> None:
    brand = Brand(id="b", language="fr")
    seats = Seats()
    o = make_outline(_brief(4), research, brand, seats)
    write_script(o, research, _brief(4), brand, seats)
    tasks = dict(seats.calls)
    assert "French" in tasks["outline"] and "French" in tasks["script"]


def test_per_section_prompt_has_full_outline_and_used_facts(research: Research) -> None:
    brief = _brief(60)
    o = heuristic_outline(brief, research)
    answer = {"beats": [{"text": "A sentence long enough to count as one beat here.", "facts_used": ["f01"]}]}
    seats = Seats({"script": answer})
    write_script(o, research, brief, Brand(id="b"), seats)
    calls = [u for t, u in seats.calls if t == "script"]
    second = calls[1]
    assert all(s.title in second for s in o.sections)  # whole outline visible
    assert ">> " + o.sections[1].id in second  # current section marked
    assert "already used by earlier sections" in second and "f01" in second.split("already used")[1]
    assert "Do not re-introduce the topic" in second


def test_real_short_script_retried_then_fails(research: Research) -> None:
    brief = _brief(8)
    o = heuristic_outline(brief, research)
    short = {"title": "T", "scenes": [{"beats": [{"text": "Too short."}]} for _ in o.sections]}
    seats = Seats({"script": short}, real=True)
    with pytest.raises(ScriptTooShort):
        write_script(o, research, brief, Brand(id="b"), seats)
    calls = [u for t, u in seats.calls if t == "script"]
    assert len(calls) == 2 and "too short" in calls[1]


def test_real_short_script_recovers_on_retry(research: Research) -> None:
    brief = _brief(8)
    o = heuristic_outline(brief, research)
    state = {"n": 0}

    def answer(user: str) -> dict:
        state["n"] += 1
        n = 3 if state["n"] == 1 else 35
        return {"title": "T", "scenes": [{"beats": [{"text": _words(n)} for _ in range(max(1, s.target_words // 35))]}
                                         for s in o.sections]}

    s = write_script(o, research, brief, Brand(id="b"), Seats({"script": answer}, real=True))
    assert s.word_count() >= 0.5 * sum(sec.target_words for sec in o.sections)


def test_real_long_script_retries_only_short_sections(research: Research) -> None:
    brief = _brief(30)
    o = heuristic_outline(brief, research)
    asked: list[str] = []

    def answer(user: str) -> dict:
        asked.append(user)
        if "too short" in user:
            return {"beats": [{"text": _words(30)} for _ in range(40)]}
        return {"beats": [{"text": "Short."}]}

    s = write_script(o, research, brief, Brand(id="b"), Seats({"script": answer}, real=True))
    assert sum("too short" in u for u in asked) == len(o.sections)
    assert s.word_count() >= 0.5 * sum(sec.target_words for sec in o.sections)


def test_mock_short_script_is_not_failed(research: Research) -> None:
    brief = _brief(8)
    o = heuristic_outline(brief, research)
    short = {"title": "T", "scenes": [{"beats": [{"text": "Too short."}]} for _ in o.sections]}
    s = write_script(o, research, brief, Brand(id="b"), Seats({"script": short}, real=False))
    assert s.scenes


def test_provider_error_propagates(research: Research) -> None:
    brief = _brief(4)
    with pytest.raises(ProviderError):
        make_outline(brief, research, Brand(id="b"), Seats({"outline": ProviderError("down")}, real=True))
    o = heuristic_outline(brief, research)
    with pytest.raises(ProviderError):
        write_script(o, research, brief, Brand(id="b"), Seats({"script": ProviderError("down")}, real=True))
