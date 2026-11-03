"""Pacing engine: mix, coverage, bounds, hook multiplier, determinism and word snapping."""
from __future__ import annotations

import random
import statistics

import pytest

from test_shots_helpers import narrate, profile, six_minute_script
from vidbush.models import ShotType
from vidbush.shots.pacing import Anchor, bounds, mix_targets, plan_slots, type_shares, word_gaps


@pytest.fixture(scope="module")
def nar():
    return narrate(six_minute_script())


def _plan(nar, seed: int, prof=None, **kw):
    prof = prof or profile()
    return plan_slots(nar.duration + 1.0, prof, random.Random(seed), nar.words, **kw)


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_mix_within_five_points(nar, seed: int) -> None:
    prof = profile()
    slots = _plan(nar, seed, prof, hard_times=[60.0, 120.0, 200.0, 280.0])
    assert len(slots) > 20
    shares = type_shares([s.type for s in slots])
    for t, share in mix_targets(prof).items():
        assert abs(shares.get(t.value, 0.0) - share * 100) <= 5.0, (t, shares)


def test_no_gaps_no_overlaps_full_cover(nar) -> None:
    slots = _plan(nar, 3)
    assert slots[0].start == 0.0
    assert slots[-1].end == pytest.approx(nar.duration + 1.0)
    for a, b in zip(slots, slots[1:]):
        assert a.end == b.start
        assert b.end > b.start


def test_durations_within_bounds(nar) -> None:
    prof = profile()
    for seed in range(5):
        slots = _plan(nar, seed, prof)
        bad = [s for s in slots if not bounds(prof, s.type, s.start)[0] - 1e-6 <= s.duration
               <= bounds(prof, s.type, s.start)[1] * 1.2 + 1e-6]
        assert len(bad) <= 1, [(s.type, s.duration) for s in bad]  # only the forced final tail may break bounds


def test_hook_multiplier_shortens_opening(nar) -> None:
    prof = profile()
    slots = _plan(nar, 5, prof)
    hook = [s.duration for s in slots if s.end <= prof.pacing.hook_seconds and s.type != ShotType.motion_graphic]
    rest = [s.duration for s in slots if s.start >= prof.pacing.hook_seconds and s.type != ShotType.motion_graphic]
    assert statistics.mean(hook) < 0.85 * statistics.mean(rest)
    fast = profile()
    fast.pacing.hook_multiplier = 1.0
    plain = [s.duration for s in _plan(nar, 5, fast) if s.end <= 60 and s.type != ShotType.motion_graphic]
    assert statistics.mean(hook) < statistics.mean(plain)


def test_opening_starts_on_footage(nar) -> None:
    for seed in range(4):
        assert _plan(nar, seed)[0].type in (ShotType.footage, ShotType.footage_overlay)


def test_deterministic_per_seed_and_varies_across_seeds(nar) -> None:
    a = [(s.type, s.start, s.end) for s in _plan(nar, 11)]
    b = [(s.type, s.start, s.end) for s in _plan(nar, 11)]
    c = [(s.type, s.start, s.end) for s in _plan(nar, 12)]
    assert a == b
    assert a != c


def test_boundaries_snap_to_word_gaps(nar) -> None:
    gaps = word_gaps(nar.words)
    slots = _plan(nar, 2)
    inner = [s.end for s in slots[:-1] if nar.words[0].start < s.end < nar.words[-1].end]

    def in_gap(t: float) -> bool:
        return any(g0 - 1e-3 <= t <= g1 + 1e-3 for g0, g1 in gaps)
    assert sum(in_gap(t) for t in inner) >= 0.85 * len(inner)


def test_ai_types_never_planned_by_default_but_honoured_when_asked(nar) -> None:
    assert not any(s.type in (ShotType.ai_image, ShotType.ai_video) for s in _plan(nar, 1))
    prof = profile()
    prof.visual_mix = {**prof.visual_mix, "ai_image": 15, "image": 0}
    shares = type_shares([s.type for s in _plan(nar, 1, prof)])
    assert abs(shares.get("ai_image", 0) - 100 * mix_targets(prof)[ShotType.ai_image]) <= 5.0
    assert "image" not in shares


def test_anchors_are_placed_exactly(nar) -> None:
    anc = Anchor(start=100.0, duration=7.0, type=ShotType.motion_graphic, tag="title_card")
    slots = _plan(nar, 4, anchors=[anc])
    fixed = [s for s in slots if s.fixed]
    assert len(fixed) == 1 and fixed[0].start == 100.0 and fixed[0].end == 107.0
    assert slots[-1].end == pytest.approx(nar.duration + 1.0)
