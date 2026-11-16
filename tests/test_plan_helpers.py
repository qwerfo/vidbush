"""Hand-timed narrations and fake seats shared by the clause-first planner tests (no model calls)."""
from __future__ import annotations

from typing import Any

from vidbush.models import Beat, Narration, Scene, ScenePurpose, Script, Word


def seq(text: str, t0: float, step: float = 0.5, dur: float = 0.4) -> list[tuple[str, float, float]]:
    """(text, start, end) for each word of text, one word every step seconds from t0."""
    return [(w, round(t0 + i * step, 3), round(t0 + i * step + dur, 3)) for i, w in enumerate(text.split())]


def build(scenes: list[list[list[tuple[str, float, float]]]], tail: float = 0.2) -> tuple[Script, Narration]:
    """scenes -> beats -> timed words. Beat ids b001.., scene ids s01..; duration = last word end + tail."""
    words: list[Word] = []
    out_scenes = []
    spans: dict[str, list[float]] = {}
    k = 0
    for si, beats in enumerate(scenes):
        bs = []
        for timed in beats:
            k += 1
            bid = f"b{k:03d}"
            bs.append(Beat(id=bid, text=" ".join(t for t, _, _ in timed)))
            words.extend(Word(text=t, start=a, end=b, beat_id=bid) for t, a, b in timed)
            spans[bid] = [timed[0][1], timed[-1][2]]
        out_scenes.append(Scene(id=f"s{si + 1:02d}", title=f"Scene {si + 1}", purpose=ScenePurpose.act, beats=bs))
    nar = Narration(audio_path="n.wav", duration=round(words[-1].end + tail, 3), words=words, beat_spans=spans)
    return Script(title="Test", scenes=out_scenes), nar


class FakeSeats:
    """Canned replies in sequence; records every prompt it is sent."""

    def __init__(self, llm: list[Any] | None = None, vision: list[Any] | None = None):
        self.llm_replies = list(llm or [])
        self.vision_replies = list(vision or [])
        self.llm_calls: list[dict] = []
        self.vision_calls: list[dict] = []

    def llm_json(self, role, task, system, user, stage, **kw):
        self.llm_calls.append({"role": role, "task": task, "system": system, "user": user, "stage": stage, **kw})
        return self.llm_replies.pop(0)

    def vision_json(self, role, task, system, prompt, images, stage, **kw):
        self.vision_calls.append({"role": role, "task": task, "system": system, "prompt": prompt,
                                  "images": images, "stage": stage, **kw})
        return self.vision_replies.pop(0)
