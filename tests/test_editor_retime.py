"""Timeline helpers: shot references, piece planning, invariant checks."""
from __future__ import annotations

import pytest

from vidbush.editor import retime
from vidbush.editor.retime import EditError
from vidbush.models import MotionKind, MotionSpec, Shot, ShotList, ShotType


def _list(durs: list[float], types: list[ShotType] | None = None) -> ShotList:
    shots = [Shot(id=f"shot_{i + 1:03d}", beat_id="b001", start=0, end=0, duration=d,
                  type=(types or [ShotType.image] * len(durs))[i], intent="x") for i, d in enumerate(durs)]
    sl = ShotList(project_id="p", duration=0, shots=shots)
    sl.reindex()
    return sl


def test_resolve_shot_forms():
    sl = _list([2, 2, 2, 2, 2, 2, 2])
    for ref in ("shot_007", "7", 7, "#7", "shot 7", 7.0):
        assert retime.resolve_shot(sl, ref).id == "shot_007"
    assert retime.resolve_shot(sl, 1).index == 0
    for bad in (0, 8, "shot_099", "seven", None, True, 2.5):
        with pytest.raises(EditError):
            retime.resolve_shot(sl, bad)


def test_new_shot_id_is_unique():
    sl = _list([2, 2])
    assert retime.new_shot_id(sl, "shot_001") == "shot_001b"
    sl.shots[1].id = "shot_001b"
    assert retime.new_shot_id(sl, "shot_001") == "shot_001c"
    assert retime.new_shot_id(sl, "shot_001b") == "shot_001c"


def test_plan_pieces_targets_factor_and_minimum():
    sl = _list([6, 4, 2, 9], [ShotType.image, ShotType.footage, ShotType.image, ShotType.motion_graphic])
    plan = retime.plan_pieces(sl.shots, 0.5)
    assert plan["shot_004"] == 1  # motion graphics are never cut
    assert plan["shot_003"] == 1  # 2 s cannot be cut above 1.5 s
    assert sum(plan.values()) <= 8 and plan["shot_001"] >= 2
    assert all(sl.by_id(k).duration / v >= 1.5 for k, v in plan.items())


def test_even_split_sums_exactly():
    parts = retime.even_split(7.0, 3)
    assert len(parts) == 3 and abs(sum(parts) - 7.0) < 1e-9 and max(parts) - min(parts) < 0.01


def test_finalize_absorbs_drift_and_retimes_specs():
    sl = _list([3.3333, 3.3333, 3.3333])
    sl.shots[1].motion = MotionSpec(kind=MotionKind.title_card, duration=9, payload={"title": "x"})
    retime.finalize(sl, 10.0)
    retime.check_timeline(sl, 10.0)
    assert sl.duration == 10.0 and sl.shots[1].motion.duration == sl.shots[1].duration


def test_check_timeline_catches_problems():
    sl = _list([3, 1.0, 3])
    with pytest.raises(EditError, match="minimum"):
        retime.check_timeline(sl, 7.0)
    sl = _list([3, 3])
    with pytest.raises(EditError, match="ends"):
        retime.check_timeline(sl, 7.0)
    sl.shots[1].start = 3.5
    with pytest.raises(EditError, match="gap"):
        retime.check_timeline(sl, 6.0)


def test_slower_merges_within_scene_first():
    sl = _list([2, 2, 2, 2])
    for s, scene in zip(sl.shots, ["a", "a", "b", "b"]):
        s.scene_id = scene
    retime.slower(sl, list(sl.shots), 2.0)
    sl.reindex()
    assert [s.scene_id for s in sl.shots] == ["a", "b"] and [s.duration for s in sl.shots] == [4, 4]
