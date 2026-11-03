"""rules.py: merging, motion minimums, opening density, repetition and type runs."""
from __future__ import annotations

from collections import Counter

from test_shots_helpers import profile
from vidbush.mg.builders import lower_third_spec, title_card_spec
from vidbush.models import Shot, ShotType, VisualIdea
from vidbush.shots import rules


def _shots(spec: list[tuple[str, float]], beat: str = "b1") -> list[Shot]:
    out, t = [], 0.0
    for i, (typ, d) in enumerate(spec):
        out.append(Shot(id=f"s{i}", beat_id=beat, start=t, end=t + d, duration=d, type=ShotType(typ),
                        intent=f"shot {i}", queries=[f"q{i}"]))
        t += d
    return out


def _contiguous(shots: list[Shot]) -> None:
    for a, b in zip(shots, shots[1:]):
        assert abs(a.end - b.start) < 1e-6


def _rebind(shot: Shot, idea: VisualIdea, idx, idea_of: dict, ideas: list) -> None:
    idea_of[shot.id] = idx if idx is not None else ideas.index(idea)
    shot.intent, shot.queries = idea.intent, list(idea.queries)


def test_merge_short_shots() -> None:
    shots = rules.merge_short(_shots([("footage", 4.0), ("image", 1.0), ("footage", 5.0), ("image", 0.8)]), 1.5)
    assert len(shots) == 2
    assert sum(s.duration for s in shots) == 10.8
    assert all(s.duration >= 1.5 for s in shots)
    _contiguous(shots)


def test_motion_graphic_grows_to_min_from_neighbours() -> None:
    prof = profile()
    shots = _shots([("footage", 6.5), ("motion_graphic", 4.0), ("image", 7.5)])
    for s in shots:
        s.start += 100.0
        s.end += 100.0
    rules._retime(shots)
    rules.enforce_motion_min(shots, prof)
    assert shots[1].duration >= prof.pacing.motion_graphic.min - 1e-6
    assert round(sum(s.duration for s in shots), 6) == 18.0
    _contiguous(shots)


def test_repeated_idea_gets_alternative_or_framing_cycle() -> None:
    ideas = [VisualIdea(beat_id="b1", family="footage", intent="only idea", queries=["q"])]
    shots = _shots([("footage", 4.0), ("footage", 4.0), ("footage", 4.0)])
    idea_of = {s.id: 0 for s in shots}
    rules.avoid_repeats(shots, ideas, idea_of, lambda s, i, k: _rebind(s, i, k, idea_of, ideas))
    assert [s.framing for s in shots] == ["full", "tight", "detail"]
    ideas.append(VisualIdea(beat_id="b1", family="footage", intent="second", queries=["other"]))
    shots = _shots([("footage", 4.0), ("footage", 4.0)])
    idea_of = {s.id: 0 for s in shots}
    rules.avoid_repeats(shots, ideas, idea_of, lambda s, i, k: _rebind(s, i, k, idea_of, ideas))
    assert idea_of["s1"] == 1 and shots[1].intent == "second"


def test_opening_prefers_opening_safe_footage() -> None:
    ideas = [VisualIdea(beat_id="b1", family="footage", intent="empty hills", opening_safe=False, priority=1.0),
             VisualIdea(beat_id="b1", family="footage", intent="workers digging", opening_safe=True, priority=0.5)]
    shots = _shots([("footage", 3.0), ("motion_graphic", 9.0), ("footage", 3.0)])
    shots[1].motion = title_card_spec("Title", duration=9.0)
    idea_of = {"s0": 0, "s2": 1}
    rules.densify_opening(shots, ideas, idea_of, profile(), lambda s, i, k: _rebind(s, i, k, idea_of, ideas))
    assert shots[0].intent == "workers digging"
    assert shots[1].duration <= profile().pacing.motion_graphic.min * 1.15 + 1e-6
    assert round(sum(s.duration for s in shots), 6) == 15.0
    _contiguous(shots)


def test_break_type_runs_preserves_counts() -> None:
    seq = [("footage", 4), ("footage", 4), ("footage", 4), ("motion_graphic", 7), ("image", 5),
           ("motion_graphic", 7), ("footage_overlay", 4), ("motion_graphic", 7)]
    shots = _shots(seq)
    before = Counter(s.type for s in shots)
    rules.break_type_runs(shots, lambda s: lower_third_spec("Panama"))
    after = Counter(s.type for s in shots)
    assert before == after
    types = [s.type for s in shots]
    assert not any(types[i] == types[i + 1] == types[i + 2] for i in range(len(types) - 2))
    for s in shots:
        assert (s.overlay is not None) == (s.type in (ShotType.footage_overlay, ShotType.image_overlay))
