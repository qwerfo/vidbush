"""Quiet degradation becomes loud: heuristic sections with a real provider are reported and need review."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_content_writer_contract import FIX, Seats as WriterSeats, _brief, _words
from test_director_context import RecordingSeats
from test_jobs_pipeline import _script_project, _stubs
from test_shots_helpers import brand, doc_script, narrate, profile
from vidbush.jobs.pipeline import load_project, run_project
from vidbush.jobs.stages import record_fallbacks
from vidbush.judge.qa import heuristic_fallbacks, render_markdown, review_notes
from vidbush.models import Brand, FallbackNote, QAReport, Research
from vidbush.providers.base import ProviderError
from vidbush.script.outline import heuristic_outline
from vidbush.script.writer import write_script
from vidbush.shots import seat
from vidbush.shots.director import beat_spans
from vidbush.util import NullLogger, read_model


class NotingWriterSeats(WriterSeats):
    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.notes: list[tuple[str, str, str, str]] = []

    def note_fallback(self, role: str, stage: str, scene_id: str, reason: str) -> None:
        self.notes.append((role, stage, scene_id, reason))


def test_writer_reports_each_heuristic_section_by_scene_id() -> None:
    research = read_model(FIX / "research.json", Research)
    brief = _brief(30)
    o = heuristic_outline(brief, research)
    broken = o.sections[1].id

    def answer(user: str):
        if ">> " + broken in user:
            return None  # the model keeps failing this one section
        return {"beats": [{"text": _words(30)} for _ in range(40)]}

    seats = NotingWriterSeats({"script": answer}, real=True)
    script = write_script(o, research, brief, Brand(id="b"), seats)
    assert [n[:3] for n in seats.notes] == [("writer", "script", "s02")]
    assert script.scenes[1].id == "s02" and script.scenes[1].beats  # heuristic beats still ship, but loudly


def test_writer_reports_nothing_when_every_section_is_written() -> None:
    research = read_model(FIX / "research.json", Research)
    brief = _brief(30)
    o = heuristic_outline(brief, research)
    seats = NotingWriterSeats({"script": lambda u: {"beats": [{"text": _words(30)} for _ in range(40)]}}, real=True)
    write_script(o, research, brief, Brand(id="b"), seats)
    assert seats.notes == []


class FlakyDirector(RecordingSeats):
    """Fails the director call for the chunk holding a given beat."""

    def __init__(self, fail_beat: str) -> None:
        super().__init__(real=True)
        self.fail_beat = fail_beat

    def llm_json(self, role, task, system, user, stage, temperature=0.6, timeout=180.0):
        if task == "director" and any(b["id"] == self.fail_beat for b in json.loads(user)["beats"]):
            raise ProviderError("upstream 502")
        return super().llm_json(role, task, system, user, stage, temperature, timeout)


def test_director_chunk_fallback_is_reported_with_its_scene() -> None:
    script = doc_script(n_scenes=4, beats_per_scene=5)
    target = script.scenes[2].beats[0].id
    seats = FlakyDirector(target)
    ideas = seat.director_ideas(script, beat_spans(script, narrate(script)), None, profile(), brand(), seats,
                                NullLogger())
    assert ideas  # the other chunks still came from the model
    assert [(r, st, sc) for r, st, sc, _ in seats.fallbacks] == [("director", "shotlist", script.scenes[2].id)]
    assert "ProviderError" in seats.fallbacks[0][3]


def test_real_seats_record_and_mock_roles_do_not(settings, monkeypatch) -> None:
    from vidbush.providers.base import CostTracker
    from vidbush.providers.seats import Seats
    seats = Seats(settings, CostTracker("p"))
    monkeypatch.setattr(Seats, "is_real", lambda self, role: role == "writer")
    seats.note_fallback("director", "shotlist", "s01", "mock answer")
    seats.note_fallback("writer", "script", "s03", "empty")
    assert seats.take_fallbacks("shotlist") == []
    assert seats.take_fallbacks("script") == [{"role": "writer", "stage": "script", "scene_id": "s03",
                                               "reason": "empty"}]
    assert seats.take_fallbacks("script") == []


def test_record_fallbacks_replaces_the_stage_entries(tmp_path) -> None:
    notes = {"script": [{"role": "writer", "stage": "script", "scene_id": "s02", "reason": "x"}]}
    seats = SimpleNamespace(take_fallbacks=lambda stage: notes.pop(stage, []))
    ctx = SimpleNamespace(work=tmp_path, seats=seats)
    record_fallbacks(ctx, "script")
    record_fallbacks(ctx, "shotlist")
    assert [n.scene_id for n in heuristic_fallbacks(ctx)] == ["s02"]
    record_fallbacks(ctx, "script")  # a rerun that succeeded clears the old notes
    assert heuristic_fallbacks(ctx) == []


def test_qa_lists_fallbacks_with_scene_ids() -> None:
    r = QAReport(project_id="p", threshold=70, shots=[], pct_above_threshold=100, pct_no_match_accepted=0,
                 mix_target={}, mix_achieved={}, pacing={}, caption_coverage=1, caption_word_match=1,
                 attribution_count=0, heuristic_fallbacks=[FallbackNote(role="director", stage="shotlist",
                                                                        scene_id="s04", reason="502")])
    ctx = SimpleNamespace(profile=profile())
    notes = review_notes(r, ctx)
    assert any("director s04" in n for n in notes)
    md = render_markdown(r)
    assert "## Heuristic fallbacks (needs review)" in md and "| s04 | director | shotlist |" in md


@pytest.mark.parametrize("fallbacks, status", [([], "done"), ([{"role": "writer", "stage": "script",
                                                                  "scene_id": "s02"}], "needs_review")])
def test_project_needs_review_when_qa_has_fallbacks(settings, tmp_path, fallbacks, status) -> None:
    _script_project(settings, tmp_path)
    settings.offline = False
    qa = {"mock_used": [], "heuristic_fallbacks": fallbacks}
    proj = run_project(settings, "p1", stages=_stubs([], qa=qa), echo=False)
    assert proj.status == status and load_project(settings, "p1").status == status
