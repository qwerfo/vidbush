"""Director context: every call carries the script summary, previous ideas, anchors so far and the next beat."""
from __future__ import annotations

import json
import random

from test_shots_helpers import brand, doc_script, narrate, profile, project
from vidbush.models import FactCard, Research, VisualAnchor
from vidbush.shots import seat
from vidbush.shots.director import beat_spans, build_shotlist
from vidbush.util import NullLogger


class RecordingSeats:
    """Answers the summary and director tasks and keeps every prompt it was sent."""

    def __init__(self, summary: str = "A canal is cut across Panama.", real: bool = False) -> None:
        self.summary = summary
        self.real = real
        self.calls: list[tuple[str, dict]] = []
        self.fallbacks: list[tuple[str, str, str, str]] = []

    def is_real(self, role: str) -> bool:
        return self.real

    def note_fallback(self, role: str, stage: str, scene_id: str, reason: str) -> None:
        self.fallbacks.append((role, stage, scene_id, reason))

    def llm_json(self, role, task, system, user, stage, temperature=0.6, timeout=180.0):
        if task == "script_summary":
            self.calls.append((task, json.loads(user)))
            return {"summary": self.summary} if self.summary else None
        body = json.loads(user)
        self.calls.append((task, body))
        return {"ideas": [{"beat_id": b["id"], "family": "footage", "intent": f"Wide view of Gatun Locks {b['id']}",
                           "must_show": ["Gatun Locks"], "queries": ["gatun locks"]} for b in body["beats"]]}

    def director_prompts(self) -> list[dict]:
        return [body for task, body in self.calls if task == "director"]


def _script():
    return doc_script(n_scenes=5, beats_per_scene=10)  # long scenes split, so there are many chunks


def test_every_director_call_carries_the_summary_and_rules() -> None:
    script = _script()
    seats = RecordingSeats()
    seat.director_ideas(script, beat_spans(script, narrate(script)), None, profile(), brand(), seats, NullLogger())
    prompts = seats.director_prompts()
    assert len(prompts) == len(seat.chunk_beats(script)) >= 6
    assert all(p["script_summary"] == "A canal is cut across Panama." for p in prompts)
    rules = " ".join(prompts[0]["rules"]).lower()
    for principle in ("keep a visual subject on screen", "vary framing", "first 60 seconds",
                      "narration moves between places", "two or more comparable numbers"):
        assert principle in rules
    assert prompts[0]["full_screen_card_spacing_s"] == 45


def test_previous_ideas_cover_the_last_two_chunks_only() -> None:
    script = _script()
    seats = RecordingSeats()
    seat.director_ideas(script, beat_spans(script, narrate(script)), None, profile(), brand(), seats, NullLogger())
    prompts = seats.director_prompts()
    chunk_beats = [[b["id"] for b in p["beats"]] for p in prompts]
    assert prompts[0]["previous_ideas"] == []
    for k in range(1, len(prompts)):
        want = [bid for ids in chunk_beats[max(0, k - 2):k] for bid in ids]
        got = prompts[k]["previous_ideas"]
        assert [i["beat_id"] for i in got] == want
        assert all(set(i) >= {"beat_id", "type", "intent"} and i["type"] == "footage" for i in got)
        assert all("queries" not in i and "must_show" not in i for i in got)  # intent and type only


def test_next_beat_and_running_anchors() -> None:
    script = _script()
    seats = RecordingSeats()
    seat.director_ideas(script, beat_spans(script, narrate(script)), None, profile(), brand(), seats, NullLogger())
    prompts = seats.director_prompts()
    beats = {b.id: b.text for b in script.beats()}
    for k in range(len(prompts) - 1):
        nb = prompts[k]["next_beat"]
        assert nb["id"] == prompts[k + 1]["beats"][0]["id"] and nb["text"] == beats[nb["id"]]
    assert prompts[-1]["next_beat"] is None
    assert prompts[0]["visual_anchors_so_far"] == []
    assert prompts[1]["visual_anchors_so_far"] == ["Gatun Locks"]


def test_anchors_so_far_uses_research_anchors_in_first_use_order() -> None:
    research = Research(topic="t", facts=[FactCard(id="f1", claim="c")], anchors=[
        VisualAnchor(kind="place", name="Culebra Cut"), VisualAnchor(kind="place", name="Gatun Locks"),
        VisualAnchor(kind="person", name="George Goethals")])
    from vidbush.models import VisualIdea
    ideas = [VisualIdea(beat_id="b1", family="footage", intent="Steam shovels in the Culebra Cut"),
             VisualIdea(beat_id="b2", family="still", intent="Ships", must_show=["Gatun Locks"])]
    assert seat.anchors_so_far(ideas, research) == ["Culebra Cut", "Gatun Locks"]


def test_summary_is_generated_once_and_cached_in_work(tmp_path) -> None:
    script = _script()
    seats = RecordingSeats(summary="First summary.")
    assert seat.script_summary(script, seats, NullLogger(), tmp_path) == "First summary."
    seats.summary = "Second summary."
    assert seat.script_summary(script, seats, NullLogger(), tmp_path) == "First summary."
    assert sum(1 for task, _ in seats.calls if task == "script_summary") == 1
    cached = json.loads((tmp_path / seat.SUMMARY_FILE).read_text())
    assert cached["source"] == "director" and cached["script_hash"]
    script.scenes[0].beats[0].text = "A different opening line."
    assert seat.script_summary(script, seats, NullLogger(), tmp_path) == "Second summary."  # script changed


def test_build_shotlist_caches_the_summary_in_the_project_work_folder(tmp_path) -> None:
    script = _script()
    proj = project()
    proj.paths.work = str(tmp_path)
    seats = RecordingSeats()
    build_shotlist(proj, script, narrate(script), None, profile(), brand(), seats, random.Random(1))
    assert (tmp_path / seat.SUMMARY_FILE).exists()


def test_summary_falls_back_to_scene_openings_and_reports_it_for_real_seats() -> None:
    script = _script()
    seats = RecordingSeats(summary="", real=True)
    text = seat.script_summary(script, seats, NullLogger())
    assert text.startswith(script.title) and len(text.split()) <= seat.SUMMARY_MAX_WORDS
    assert seats.fallbacks and seats.fallbacks[0][:3] == ("director", "shotlist", "all")


def test_map_and_chart_overlays_from_the_director_become_insets() -> None:
    item = {"beat_id": "b001", "family": "footage", "intent": "Ships", "queries": ["ships"],
            "overlay": {"kind": "map", "payload": {"center": [9, -79.7], "zoom": 6, "markers": [
                {"lat": 9.36, "lon": -79.9, "label": "Colon"}]}}}
    idea = seat.validate_idea(item, {"b001"}, None, {"b001": 5.0})
    assert idea.overlay.kind.value == "map" and idea.overlay.background == "transparent"
