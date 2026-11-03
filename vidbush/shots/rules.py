"""Shot list post-processing rules applied after slot assignment.

apply_rules() runs, in order: merge shots under 1.5 s into a neighbour, keep
motion graphics at their pacing min where the neighbours can give time, make
the opening dense (opening_safe footage, short motion graphics), avoid
consecutive shots from the same idea (framing full, tight, detail when an
idea must repeat) and break runs of three identical shot types with count
preserving swaps inside a family. Shot times stay contiguous throughout.
"""
from __future__ import annotations

from typing import Callable, Optional

from ..models import OVERLAY_TYPES, MotionSpec, Shot, ShotType, StyleProfile, VisualIdea
from .pacing import bounds

MIN_SHOT = 1.5
FRAMINGS = ("full", "tight", "detail")
TOGGLE = {ShotType.footage: ShotType.footage_overlay, ShotType.footage_overlay: ShotType.footage,
          ShotType.image: ShotType.image_overlay, ShotType.image_overlay: ShotType.image}
FAMILY_OF = {ShotType.footage: "footage", ShotType.footage_overlay: "footage", ShotType.ai_video: "footage",
             ShotType.image: "still", ShotType.image_overlay: "still", ShotType.ai_image: "still",
             ShotType.motion_graphic: "motion"}

Rebind = Callable[[Shot, VisualIdea, Optional[int]], None]
OverlayFor = Callable[[Shot], MotionSpec]


def apply_rules(shots: list[Shot], ideas: list[VisualIdea], idea_of: dict[str, int], profile: StyleProfile,
                rebind: Rebind, overlay_for: OverlayFor, locked: frozenset[str] = frozenset()) -> list[Shot]:
    """Run every rule; returns the polished list. Locked shot ids (structural anchors) keep type and content."""
    shots = merge_short(shots, MIN_SHOT)
    enforce_motion_min(shots, profile)
    densify_opening(shots, ideas, idea_of, profile, rebind, locked)
    avoid_repeats(shots, ideas, idea_of, rebind, locked)
    break_type_runs(shots, overlay_for, locked)
    _retime(shots)
    return shots


# ---------------------------------------------------------------- timing rules

def _retime(shots: list[Shot]) -> None:
    t = shots[0].start if shots else 0.0
    for s in shots:
        s.start = round(t, 3)
        t += s.duration
        s.end = round(t, 3)


def _move(shots: list[Shot], i: int, j: int, amount: float) -> None:
    """Transfer amount seconds from shot j to its neighbour shot i."""
    shots[j].duration = round(shots[j].duration - amount, 3)
    shots[i].duration = round(shots[i].duration + amount, 3)
    _retime(shots)


def merge_short(shots: list[Shot], min_len: float) -> list[Shot]:
    """Merge every shot shorter than min_len into its shorter neighbour (same beat preferred)."""
    shots = list(shots)
    while len(shots) > 1:
        short = next((k for k, s in enumerate(shots) if s.duration < min_len - 1e-6), None)
        if short is None:
            break
        cands = [k for k in (short - 1, short + 1) if 0 <= k < len(shots)]
        same = [k for k in cands if shots[k].beat_id == shots[short].beat_id]
        k = min(same or cands, key=lambda c: shots[c].duration)
        shots[k].duration = round(shots[k].duration + shots[short].duration, 3)
        del shots[short]
        _retime(shots)
    return shots


def _slack(shot: Shot, profile: StyleProfile) -> float:
    if shot.type == ShotType.motion_graphic:
        return 0.0
    lo = bounds(profile, shot.type, shot.start)[0]
    return max(0.0, shot.duration - max(lo, MIN_SHOT))


def enforce_motion_min(shots: list[Shot], profile: StyleProfile) -> None:
    """Grow motion graphics shorter than their pacing min using slack from neighbours."""
    mg_min = profile.pacing.motion_graphic.min
    for i, s in enumerate(shots):
        need = mg_min - s.duration
        for j in (i + 1, i - 1):
            if need <= 1e-6 or not 0 <= j < len(shots):
                continue
            give = min(need, _slack(shots[j], profile))
            if give > 0:
                _move(shots, i, j, give)
                need -= give


# ---------------------------------------------------------------- opening

def densify_opening(shots: list[Shot], ideas: list[VisualIdea], idea_of: dict[str, int], profile: StyleProfile,
                    rebind: Rebind, locked: frozenset[str] = frozenset()) -> None:
    """First hook_seconds: footage from opening_safe ideas, motion graphics trimmed toward their min."""
    hook = profile.pacing.hook_seconds
    for i, s in enumerate(shots):
        if s.start >= hook:
            break
        fam = FAMILY_OF[s.type]
        idx = idea_of.get(s.id)
        if fam == "footage" and idx is not None and not ideas[idx].opening_safe and s.id not in locked:
            alt = _alternative(shots, i, ideas, idea_of, need_safe=True)
            if alt is not None:
                rebind(s, ideas[alt], alt)
        if s.type == ShotType.motion_graphic:
            cap = bounds(profile, ShotType.motion_graphic, s.start)[1]
            excess = s.duration - cap
            for j in (i + 1, i - 1):
                if excess <= 1e-6 or not 0 <= j < len(shots) or shots[j].type == ShotType.motion_graphic:
                    continue
                room = bounds(profile, shots[j].type, shots[j].start)[1] * 1.15 - shots[j].duration
                give = min(excess, max(0.0, room))
                if give > 0:
                    _move(shots, j, i, give)
                    excess -= give


def _alternative(shots: list[Shot], i: int, ideas: list[VisualIdea], idea_of: dict[str, int],
                 need_safe: bool = False) -> Optional[int]:
    """Least used idea of the same family from the shot's beat or its neighbours, unused by adjacent shots."""
    s = shots[i]
    fam = FAMILY_OF[s.type]
    if fam == "motion":
        return None
    near = {shots[k].beat_id for k in range(max(0, i - 2), min(len(shots), i + 3))}
    adjacent = {idea_of.get(shots[k].id) for k in (i - 1, i, i + 1) if 0 <= k < len(shots)}
    used: dict[int, int] = {}
    for v in idea_of.values():
        used[v] = used.get(v, 0) + 1
    best, best_score = None, -1e9
    for k, idea in enumerate(ideas):
        if idea.family != fam or idea.beat_id not in near or k in adjacent:
            continue
        if need_safe and not idea.opening_safe:
            continue
        score = idea.priority - used.get(k, 0) + (0.5 if idea.beat_id == s.beat_id else 0.0)
        if score > best_score:
            best, best_score = k, score
    return best


# ---------------------------------------------------------------- repetition

def _same_look(a: Shot, b: Shot, ia: Optional[int], ib: Optional[int]) -> bool:
    if ia is not None and ia == ib:
        return True
    if a.type != ShotType.motion_graphic and a.intent.strip().lower() == b.intent.strip().lower():
        return True
    return a.type != ShotType.motion_graphic and bool(a.queries) and a.queries[:1] == b.queries[:1]


def avoid_repeats(shots: list[Shot], ideas: list[VisualIdea], idea_of: dict[str, int], rebind: Rebind,
                  locked: frozenset[str] = frozenset()) -> None:
    """No two consecutive shots from one idea; repeated ideas cycle framing full, tight, detail."""
    for i in range(1, len(shots)):
        a, b = shots[i - 1], shots[i]
        if b.id not in locked and _same_look(a, b, idea_of.get(a.id), idea_of.get(b.id)):
            alt = _alternative(shots, i, ideas, idea_of)
            if alt is not None:
                rebind(b, ideas[alt], alt)
    seen: dict[str, int] = {}
    for s in shots:
        idx = idea_of.get(s.id)
        if idx is None or s.type == ShotType.motion_graphic:
            continue
        sig = f"{ideas[idx].family}:{ideas[idx].intent.strip().lower()}"
        n = seen.get(sig, 0)
        s.framing = FRAMINGS[n % len(FRAMINGS)]
        seen[sig] = n + 1


def break_type_runs(shots: list[Shot], overlay_for: OverlayFor, locked: frozenset[str] = frozenset()) -> None:
    """Break runs of three identical types by swapping types with a later shot of the sibling type."""
    for i in range(2, len(shots)):
        t = shots[i].type
        if not (shots[i - 1].type == t == shots[i - 2].type) or t not in TOGGLE or shots[i].id in locked:
            continue
        alt = TOGGLE[t]
        if _creates_run(shots, i, alt):
            continue
        for j in range(len(shots) - 1, -1, -1):
            if abs(j - i) > 2 and shots[j].type == alt and shots[j].id not in locked and not _creates_run(shots, j, t):
                _set_type(shots[i], alt, overlay_for)
                _set_type(shots[j], t, overlay_for)
                break


def _creates_run(shots: list[Shot], k: int, new: ShotType) -> bool:
    types = [s.type for s in shots]
    types[k] = new
    lo, hi = max(0, k - 2), min(len(types), k + 3)
    window = types[lo:hi]
    return any(window[m] == window[m + 1] == window[m + 2] for m in range(len(window) - 2))


def _set_type(shot: Shot, new: ShotType, overlay_for: OverlayFor) -> None:
    shot.type = new
    if new in OVERLAY_TYPES:
        shot.overlay = shot.overlay or overlay_for(shot)
    else:
        shot.overlay = None
