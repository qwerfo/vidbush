"""Plan replay: anchor inside window, early and late counts, cut rate, screen time mix, uncovered beats."""
from __future__ import annotations

import json
from pathlib import Path

from vidbush.audit.plan_replay import plan_report, report_text
from vidbush.models import Candidate, Shot, ShotList, ShotType, SourceKind, VisualIdea
from vidbush.review_cli import REVIEW_COMMANDS


def _write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _shot(i: int, start: float, end: float, intent: str, typ: ShotType = ShotType.footage,
          source: SourceKind = SourceKind.youtube, media: str = "video") -> Shot:
    c = Candidate(id=f"c{i}", source=source, media=media)
    return Shot(id=f"shot_{i:03d}", index=i, beat_id="b1" if start < 6 else "b2", start=start, end=end,
                duration=end - start, type=typ, intent=intent, candidates=[c], selected=c.id)


def test_plan_replay_measures_a_stored_plan(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    words = [("dogs", 0.5), ("sled", 2.5), ("nome", 4.5), ("serum", 6.5), ("train", 8.5), ("harbour", 14.5)]
    _write(work / "narration.json", {"duration": 16.0, "words": [
        {"text": t, "start": s, "end": s + 0.4, "beat_id": "b1" if s < 6 else "b2"} for t, s in words],
        "beat_spans": {"b1": [0, 6], "b2": [6, 12], "b3": [12, 16]}})
    ideas = [VisualIdea(beat_id="b1", family="footage", intent="dogs running", anchor="dogs"),
             VisualIdea(beat_id="b1", family="still", intent="town of nome", anchor="nome"),
             VisualIdea(beat_id="b2", family="footage", intent="train arriving", anchor="train")]
    _write(work / "director_plan.json", {"ideas": [i.model_dump(mode="json") for i in ideas]})
    shots = [_shot(0, 0, 3, "dogs running"),                                                  # inside
             _shot(1, 3, 6, "town of nome", ShotType.image, SourceKind.ai_image, "image"),    # inside, AI
             _shot(2, 6, 8, "train arriving"),                                                # words later: early
             _shot(3, 8, 12, "a card", ShotType.motion_graphic, SourceKind.motion, "image")]  # not a picture shot
    _write(work / "shotlist.json", ShotList(project_id="p", duration=12, shots=shots).model_dump(mode="json"))
    r = plan_report(work)
    assert r["picture_shots"] == 3 and r["anchor"] == {"inside": 2, "early": 1, "late": 0, "no_anchor": 0,
                                                       "inside_pct": 66.7}
    assert r["cuts_per_minute"] == round(3 / (16 / 60), 1) and r["median_shot_s"] == 3.0
    assert r["screen_time_pct"] == {"real_footage": 41.7, "real_still": 0.0, "ai": 25.0, "designed_or_motion": 33.3}
    assert r["unit"] == "beat" and r["uncovered"] == ["b3"]
    _write(work / "clauses.json", [{"id": "c1", "start": 0, "end": 2}, {"id": "c2", "start": 9, "end": 11}])
    shots[0].end = 2.9
    _write(work / "shotlist.shotlist.json", ShotList(project_id="p", duration=12, shots=shots).model_dump(mode="json"))
    r = plan_report(tmp_path)  # a project folder works too; the stage snapshot wins over shotlist.json
    assert r["shotlist"].endswith("shotlist.shotlist.json") and r["unit"] == "clause" and r["uncovered"] == ["c2"]
    assert "anchor words inside their own shot window: 2 of 3" in report_text(r)
    assert "plan-replay" in REVIEW_COMMANDS
