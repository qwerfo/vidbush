"""Pacing engine: typed, timed slots covering the whole timeline.

plan_slots() fills [0, total] with slots. Fixed anchors (number cards, title
cards, recap grids, item lower thirds) are placed first; the free time between
anchors and scene boundaries is filled with slots whose type is chosen by
deficit against profile.visual_mix (share of shot count) and whose duration is
drawn from profile.pacing (mean with bounded jitter, clamped to min and max,
times hook_multiplier inside the first hook_seconds). Internal boundaries are
snapped to gaps between words when that keeps both neighbours within bounds.
All randomness comes from the rng argument.
"""
from __future__ import annotations

import random
from typing import Optional, Sequence

from pydantic import BaseModel

from ..models import MotionSpec, ShotType, StyleProfile, Word

SNAP_TOLERANCE = 0.45  # seconds a boundary may move to land between words
NOISE = 0.7  # random tie breaking added to deficit scores (keeps mix exact, varies rhythm)
EPS = 1e-6


class Slot(BaseModel):
    """One planned shot: time range, type and optional fixed content."""

    start: float
    end: float
    type: ShotType
    tag: str = ""  # anchor tag such as "number_card", "item_lower_third", "recap", "title_card"
    fixed: bool = False
    motion: Optional[MotionSpec] = None
    overlay: Optional[MotionSpec] = None

    @property
    def duration(self) -> float:
        return self.end - self.start


class Anchor(BaseModel):
    """A slot that must exist at a given time with a given type."""

    start: float
    duration: float
    type: ShotType
    tag: str
    motion: Optional[MotionSpec] = None
    overlay: Optional[MotionSpec] = None


def mix_targets(profile: StyleProfile) -> dict[ShotType, float]:
    """visual_mix normalised to shares of shot count; zero or unknown types dropped."""
    raw = {ShotType(k): float(v) for k, v in profile.visual_mix.items() if k in ShotType.__members__ and v > 0}
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()} or {ShotType.footage: 1.0}


def bounds(profile: StyleProfile, t: ShotType, at: float) -> tuple[float, float, float]:
    """(min, max, mean) for a type at a timeline position, hook multiplier applied.

    Motion graphics never shrink below their pacing min; in the hook they stay close to it.
    """
    tp = profile.pacing.for_type(t)
    in_hook = at < profile.pacing.hook_seconds
    f = profile.pacing.hook_multiplier if in_hook else 1.0
    if t == ShotType.motion_graphic:
        if in_hook:
            return tp.min, max(tp.min, tp.min * 1.15), tp.min * 1.05
        return tp.min, tp.max, tp.mean
    return tp.min * f, tp.max * f, tp.mean * f


def sample_duration(profile: StyleProfile, t: ShotType, at: float, rng: random.Random) -> float:
    """Mean with bounded jitter, clamped to the type bounds."""
    lo, hi, mean = bounds(profile, t, at)
    j = profile.pacing.jitter
    return min(hi, max(lo, mean * (1.0 + rng.uniform(-j, j))))


def word_gaps(words: Sequence[Word]) -> list[tuple[float, float]]:
    """Silent intervals between consecutive words (zero length when words touch)."""
    ws = sorted(words, key=lambda w: w.start)
    return [(a.end, max(a.end, b.start)) for a, b in zip(ws, ws[1:])]


class _Planner:
    """Stateful fill: running type counts drive the deficit choice."""

    def __init__(self, profile: StyleProfile, rng: random.Random, gaps: list[tuple[float, float]]) -> None:
        self.profile = profile
        self.rng = rng
        self.gaps = gaps
        self.targets = mix_targets(profile)
        self.counts: dict[ShotType, int] = {t: 0 for t in self.targets}
        self.slots: list[Slot] = []

    # -------------------------------------------------------- type choice

    def choose(self, at: float, remaining: float, avoid_end: Optional[ShotType] = None) -> ShotType:
        """Type with the largest deficit, avoiding a third identical type in a row and types that cannot fit.

        avoid_end is the type of the anchor that follows this segment: it is penalised for the last slot.
        """
        n = sum(self.counts.values())
        last = [s.type for s in self.slots[-2:]]
        in_hook = at < self.profile.pacing.hook_seconds
        best, best_score = None, -1e9
        for t, share in self.targets.items():
            score = share * (n + 1) - self.counts.get(t, 0) + self.rng.uniform(0, NOISE)
            if len(last) == 2 and last[0] == last[1] == t:
                score -= 10.0
            elif last and last[-1] == t:
                score -= 0.25
            lo, hi, _ = bounds(self.profile, t, at)
            if lo > remaining + EPS:
                score -= 5.0
            if t == avoid_end and remaining <= hi + self._min_any(at):
                score -= 1.5
            if in_hook and t == ShotType.motion_graphic:
                score -= 1.2
            if not self.slots and t not in (ShotType.footage, ShotType.footage_overlay):
                score -= 20.0 if ShotType.footage in self.targets else 0.0
            if score > best_score:
                best, best_score = t, score
        return best or ShotType.footage

    def add(self, slot: Slot) -> None:
        self.slots.append(slot)
        self.counts[slot.type] = self.counts.get(slot.type, 0) + 1

    # -------------------------------------------------------- filling

    def fill(self, a: float, b: float, avoid_end: Optional[ShotType] = None) -> None:
        """Fill [a, b] with free slots."""
        t = a
        while b - t > EPS:
            remaining = b - t
            typ = self.choose(t, remaining, avoid_end)
            lo, hi, _ = bounds(self.profile, typ, t)
            d = sample_duration(self.profile, typ, t, self.rng)
            floor = self._min_any(t + d)
            if remaining < lo:
                if self._absorb(remaining, a):
                    return
                d = remaining
            elif remaining <= d + floor:
                d = remaining if remaining <= hi else max(lo, remaining - max(floor, d * 0.6))
            end = b if abs(b - (t + d)) < EPS else self.snap(t + d, t, typ, b)
            self.add(Slot(start=round(t, 3), end=round(end, 3), type=typ))
            t = end

    def _min_any(self, at: float) -> float:
        return min(bounds(self.profile, t, at)[0] for t in self.targets)

    def _absorb(self, extra: float, seg_start: float) -> bool:
        """Give a short tail to the previous free slot of this segment when it stays reasonable."""
        if not self.slots or self.slots[-1].fixed or self.slots[-1].start < seg_start - EPS:
            return False
        last = self.slots[-1]
        _lo, hi, _ = bounds(self.profile, last.type, last.start)
        if last.duration + extra > hi * 1.2:
            return False
        last.end = round(last.end + extra, 3)
        return True

    def snap(self, t: float, start: float, typ: ShotType, seg_end: float) -> float:
        """Move a boundary into the nearest word gap if both sides stay within bounds."""
        lo, hi, _ = bounds(self.profile, typ, start)
        floor = self._min_any(t)
        best, best_dist = t, SNAP_TOLERANCE + EPS
        for g0, g1 in self.gaps:
            if g1 < t - SNAP_TOLERANCE:
                continue
            if g0 > t + SNAP_TOLERANCE:
                break
            cand = min(max(t, g0), g1) if g1 - g0 > 0.02 else (g0 + g1) / 2
            if g1 - g0 > 0.08:
                cand = min(max(cand, g0 + 0.02), g1 - 0.02)
            dist = abs(cand - t)
            ok = lo - EPS <= cand - start <= hi + EPS and (seg_end - cand >= floor or seg_end - cand < EPS)
            if ok and dist < best_dist:
                best, best_dist = cand, dist
        return best


def plan_slots(total: float, profile: StyleProfile, rng: random.Random, words: Sequence[Word] = (),
               hard_times: Sequence[float] = (), anchors: Sequence[Anchor] = ()) -> list[Slot]:
    """Typed, timed slots covering [0, total] with no gaps or overlaps."""
    planner = _Planner(profile, rng, word_gaps(words))
    fixed = _place_anchors(anchors, total)
    cuts = sorted({0.0, total, *[h for h in hard_times if 0 < h < total]})
    cuts = _thin(cuts, 2.0, keep={0.0, total})
    t = 0.0
    for anc in fixed:
        _fill_range(planner, t, anc.start, cuts, anc.type)
        planner.add(Slot(start=round(anc.start, 3), end=round(anc.start + anc.duration, 3), type=anc.type,
                         tag=anc.tag, fixed=True, motion=anc.motion, overlay=anc.overlay))
        t = anc.start + anc.duration
    _fill_range(planner, t, total, cuts)
    return _close(planner.slots, total)


def _fill_range(planner: _Planner, a: float, b: float, cuts: list[float],
                avoid_end: Optional[ShotType] = None) -> None:
    if b - a < EPS:
        return
    if b - a < 1.5 and planner.slots:
        prev = planner.slots[-1]
        prev.end = round(b, 3)
        return
    inner = [c for c in cuts if a + 1.5 < c < b - 1.5]
    edges = [a, *inner, b]
    for x, y in zip(edges, edges[1:]):
        planner.fill(x, y, avoid_end if y == b else None)


def _place_anchors(anchors: Sequence[Anchor], total: float) -> list[Anchor]:
    """Sort anchors, clip them to the timeline and push overlapping ones later."""
    out: list[Anchor] = []
    t = 0.0
    for a in sorted(anchors, key=lambda x: x.start):
        start = max(a.start, t)
        dur = min(a.duration, total - start)
        if dur < 1.5:
            continue
        if out and start - t < 1.5:
            start = t  # avoid a sliver between two anchors
        out.append(a.model_copy(update={"start": start, "duration": dur}))
        t = start + dur
    return out


def _thin(cuts: list[float], min_gap: float, keep: set[float]) -> list[float]:
    out: list[float] = []
    for c in cuts:
        if out and c - out[-1] < min_gap and c not in keep:
            continue
        out.append(c)
    return out


def _close(slots: list[Slot], total: float) -> list[Slot]:
    """Make the slots exactly contiguous over [0, total]."""
    slots = [s for s in slots if s.end - s.start > EPS]
    for i, s in enumerate(slots):
        s.start = 0.0 if i == 0 else slots[i - 1].end
    if slots:
        slots[-1].end = round(total, 3)
    return slots


def type_shares(types: Sequence[ShotType | str]) -> dict[str, float]:
    """Achieved share of shot count per type, in percent."""
    n = len(types) or 1
    out: dict[str, float] = {}
    for t in types:
        k = ShotType(t).value
        out[k] = out.get(k, 0.0) + 100.0 / n
    return out
