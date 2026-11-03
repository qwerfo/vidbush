"""Outline sizing and script writer tests (seat answers faked, heuristic paths exercised)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest

from vidbush.config import REPO_ROOT
from vidbush.models import Brand, Brief, Research, ScenePurpose, VideoFormat
from vidbush.script.outline import heuristic_outline, make_outline, target_words
from vidbush.script.parser import split_sentences
from vidbush.script.writer import write_script
from vidbush.util import EM_DASH, read_model

FIX = Path(REPO_ROOT) / "fixtures" / "llm" / "the_panama_canal"


class FakeSeats:
    """Answers llm_json from a {task: value or callable(user)} table and records calls."""

    def __init__(self, answers: Optional[dict[str, Any]] = None) -> None:
        self.answers = answers or {}
        self.calls: list[tuple[str, str]] = []

    def llm_json(self, role: str, task: str, system: str, user: str, stage: str, temperature: float = 0.6) -> Any:
        self.calls.append((task, user))
        v = self.answers.get(task)
        return v(user) if callable(v) else v


@pytest.fixture
def research() -> Research:
    return read_model(FIX / "research.json", Research)


def _brief(minutes: float, fmt: str = "documentary") -> Brief:
    return Brief(topic="The Panama Canal", target_minutes=minutes, format=VideoFormat(fmt),
                 angle="the human cost and engineering triumph")


@pytest.mark.parametrize("minutes,acts", [(4, 3), (12, 4), (60, 5)])
def test_documentary_outline_sizing(research: Research, minutes: float, acts: int):
    o = make_outline(_brief(minutes), research, Brand(id="b"), FakeSeats())
    total = sum(s.target_words for s in o.sections)
    assert abs(total - minutes * 150) <= len(o.sections)
    purposes = [s.purpose for s in o.sections]
    assert purposes[0] == ScenePurpose.hook and purposes[1] == ScenePurpose.setup
    assert purposes[-1] == ScenePurpose.outro and ScenePurpose.climax in purposes
    assert purposes.count(ScenePurpose.turn) == 1
    assert purposes.count(ScenePurpose.act) + 1 == acts
    used = {f for s in o.sections for f in s.fact_ids}
    assert len(used) >= len(research.facts) - 1


def test_top10_outline_countdown(research: Research):
    o = heuristic_outline(_brief(12, "top10"), research)
    items = [s for s in o.sections if s.purpose == ScenePurpose.item]
    assert [s.item_number for s in items] == list(range(10, 0, -1))
    assert o.sections[0].purpose == ScenePurpose.hook and o.sections[-1].purpose == ScenePurpose.outro
    assert abs(sum(s.target_words for s in o.sections) - target_words(12)) <= len(o.sections)


def test_seat_outline_is_validated_and_resized(research: Research):
    answer = {"title": "T", "sections": [
        {"title": "Open", "purpose": "act", "target_words": 50, "fact_ids": ["f01", "nope"]},
        {"title": "Middle", "purpose": "bogus", "target_words": 100, "fact_ids": ["f04"]},
        {"title": "End", "purpose": "act", "target_words": 50}]}
    o = make_outline(_brief(8), research, Brand(id="b"), FakeSeats({"outline": answer}))
    assert [s.purpose for s in o.sections] == [ScenePurpose.hook, ScenePurpose.act, ScenePurpose.outro]
    assert o.sections[0].fact_ids == ["f01"]
    assert sum(s.target_words for s in o.sections) == 1200
    assert o.sections[1].target_words == 600


def test_bad_seat_outline_falls_back(research: Research):
    o = make_outline(_brief(4), research, Brand(id="b"), FakeSeats({"outline": {"sections": "x"}}))
    assert o.sections[0].id == "sec01" and len(o.sections) >= 6


def test_heuristic_writer(research: Research):
    brief = _brief(4)
    o = heuristic_outline(brief, research)
    s = write_script(o, research, brief, Brand(id="b"), FakeSeats())
    assert [sc.id for sc in s.scenes] == [f"s{i + 1:02d}" for i in range(len(o.sections))]
    beats = s.beats()
    assert [b.id for b in beats] == [f"b{i + 1:03d}" for i in range(len(beats))]
    assert s.word_count() > 300
    known = {f.id for f in research.facts}
    assert all(f in known for b in beats for f in b.facts_used)
    assert sum(1 for b in beats if b.facts_used) >= 30
    assert "subscribe" in s.scenes[-1].beats[-1].text
    assert all(EM_DASH not in b.text for b in beats)


def test_fixture_writer_single_call(research: Research):
    import json
    brief = _brief(4)
    seats = FakeSeats({"script": json.loads((FIX / "script.json").read_text())})
    o = make_outline(brief, research, Brand(id="b"),
                     FakeSeats({"outline": json.loads((FIX / "outline.json").read_text())}))
    s = write_script(o, research, brief, Brand(id="b"), seats)
    assert len([c for c in seats.calls if c[0] == "script"]) == 1
    assert len(s.scenes) == 7 and len(s.beats()) == 35
    assert s.beats()[0].id == "b001" and s.beats()[-1].id == "b035"


def test_long_script_written_per_section_with_continuity(research: Research):
    brief = _brief(60)
    o = heuristic_outline(brief, research)
    n = {"k": 0}

    def answer(user: str) -> dict:
        n["k"] += 1
        return {"beats": [{"text": f"Section {n['k']} opens here {EM_DASH} with a dash. And more words follow "
                                   "to make the beat long enough to keep. A third sentence. A fourth sentence.",
                           "facts_used": ["f01", "zz"], "emphasis": "high"}]}

    seats = FakeSeats({"script": answer})
    s = write_script(o, research, brief, Brand(id="b"), seats)
    calls = [c for c in seats.calls if c[0] == "script"]
    assert len(calls) == len(o.sections)
    first = s.scenes[0].beats
    assert "previous section ended" in calls[1][1] and first[-1].text in calls[1][1]
    assert "previous section ended" not in calls[0][1]
    assert EM_DASH not in first[0].text and "here, with a dash" in first[0].text
    assert len(first) == 2 and all(len(split_sentences(b.text)) <= 3 for b in first)
    assert first[0].facts_used == ["f01"]


def test_full_script_answer_to_section_calls_is_matched_by_position(research: Research):
    brief = _brief(60)
    o = heuristic_outline(brief, research)
    full = {"scenes": [{"title": f"S{i}", "beats": [{"text": f"Scene number {i} is told right here in full."}]}
                       for i in range(len(o.sections))]}
    s = write_script(o, research, brief, Brand(id="b"), FakeSeats({"script": full}))
    assert [sc.beats[0].text for sc in s.scenes][:3] == [f"Scene number {i} is told right here in full."
                                                       for i in range(3)]


def test_seat_exception_falls_back(research: Research):
    class Boom(FakeSeats):
        def llm_json(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("down")

    brief = _brief(4)
    o = make_outline(brief, research, Brand(id="b"), Boom())
    s = write_script(o, research, brief, Brand(id="b"), Boom())
    assert s.beats()
