"""Shot list post-processing rules applied after slot assignment.

apply_rules() runs, in order: merge shots under 1.5 s into a neighbour, keep
motion graphics at their pacing min where the neighbours can give time, make
the opening dense (opening_safe footage, short motion graphics), avoid
consecutive shots from the same idea (framing full, tight, detail when an
idea must repeat), break runs of three identical shot types with count
preserving swaps inside a family, cap how long a full screen card holds (the
excess becomes footage of the same beat), break runs of full screen cards
(more than two in a row or about 16 s) by turning one into footage, and drop
overlay text repeated within 30 s (another overlay, or the plain sibling
type). Shot times stay contiguous.
The director then runs, in order: cards_on_cue() and space_cards() (a text or
number card must start on the word that speaks it, and at most one free full
screen card per profile.pacing.card_spacing_s outside the exempt ranges; a
card that breaks either becomes footage carrying its message as an overlay: a
title as a lower third, a number or text on a transparent background, a map or
chart as an inset), demote_repeating_cards() (a callout or text card that
repeats the narration spoken near it becomes footage), and time_overlays()
(each text overlay shows just after the narrator says its anchor, moves to the
shot that holds that moment, or is dropped, and text that repeats the
narration is dropped).
"""
from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

from ..mg import builders as b
from ..models import (OVERLAY_TYPES, STATIC_CARD_KINDS, MotionKind, MotionSpec, Shot, ShotType, StyleProfile,
                      VisualIdea, Word)
from . import ideas as I
from .assembler import INSET_GAP_S
from .pacing import EPS, MAX_MG_RUN, MAX_MG_RUN_S, bounds, card_too_close

MIN_SHOT = 1.5
FRAMINGS = ("full", "tight", "detail")
TOGGLE = {ShotType.footage: ShotType.footage_overlay, ShotType.footage_overlay: ShotType.footage,
          ShotType.image: ShotType.image_overlay, ShotType.image_overlay: ShotType.image}
FAMILY_OF = {ShotType.footage: "footage", ShotType.footage_overlay: "footage", ShotType.ai_video: "footage",
             ShotType.image: "still", ShotType.image_overlay: "still", ShotType.ai_image: "still",
             ShotType.motion_graphic: "motion"}

PLAIN = {ShotType.footage_overlay: ShotType.footage, ShotType.image_overlay: ShotType.image}
TEXT_CARD_KINDS = {MotionKind.title_card, MotionKind.text_overlay, MotionKind.callout}
MG_HOLD_MAX_S = 7.0  # a full screen map, chart or other animated card holds at most this
OVERLAY_REPEAT_S = 30.0
OVERLAY_MOVE_S = 12.0  # a text overlay may move to a shot this close when its own shot misses its moment

Rebind = Callable[[Shot, VisualIdea, Optional[int]], None]
OverlayFor = Callable[[Shot], Optional[MotionSpec]]
OverlayPool = Callable[[Shot], list[MotionSpec]]
FootageFor = Callable[[Shot], Optional[int]]


def apply_rules(shots: list[Shot], ideas: list[VisualIdea], idea_of: dict[str, int], profile: StyleProfile,
                rebind: Rebind, overlay_for: OverlayFor, locked: frozenset[str] = frozenset(),
                overlay_pool: Optional[OverlayPool] = None, footage_for: Optional[FootageFor] = None) -> list[Shot]:
    """Run every rule; returns the polished list. Locked shot ids (structural anchors) keep type and content."""
    shots = merge_short(shots, MIN_SHOT)
    enforce_motion_min(shots, profile)
    densify_opening(shots, ideas, idea_of, profile, rebind, locked)
    avoid_repeats(shots, ideas, idea_of, rebind, locked)
    break_type_runs(shots, overlay_for, locked)
    cap_card_holds(shots, profile, ideas, rebind, footage_for, overlay_for, locked)
    if footage_for is not None:
        break_card_runs(shots, ideas, rebind, footage_for, overlay_for, locked)
    dedupe_overlays(shots, overlay_pool or (lambda sh: [o for o in [overlay_for(sh)] if o is not None]), locked)
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


def hold_cap(spec: Optional[MotionSpec], profile: StyleProfile) -> Optional[float]:
    """Longest hold of a full screen card: a static text or number card is read in card_hold_max, an animated
    map or chart plays out in MG_HOLD_MAX_S. None when the card has no spec yet."""
    if spec is None:
        return None
    return profile.pacing.card_hold_max if spec.kind in STATIC_CARD_KINDS else MG_HOLD_MAX_S


def enforce_motion_min(shots: list[Shot], profile: StyleProfile) -> None:
    """Grow motion graphics shorter than their pacing min using slack from neighbours, never past their hold cap."""
    mg_min = profile.pacing.motion_graphic.min
    for i, s in enumerate(shots):
        if s.type != ShotType.motion_graphic:
            continue  # footage and stills keep the length the planner drew for them
        need = min(mg_min, hold_cap(s.motion, profile) or mg_min) - s.duration
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
        if shot.overlay is None:
            shot.type = PLAIN[new]  # nothing honest to overlay
    else:
        shot.overlay = None


# ---------------------------------------------------------------- full screen cards

def cap_card_holds(shots: list[Shot], profile: StyleProfile, ideas: list[VisualIdea], rebind: Rebind,
                   footage_for: Optional[FootageFor], overlay_for: OverlayFor,
                   locked: frozenset[str] = frozenset()) -> None:
    """A free full screen card holds at most its hold_cap. The excess goes to a non motion neighbour that
    stays within 1.15 times its max; what is left becomes a footage shot of the same beat right after the
    card (at least MIN_SHOT long), so the beat stays on picture instead of on a frozen card."""
    k = 0
    while k < len(shots):
        s = shots[k]
        cap = hold_cap(s.motion, profile) if s.type == ShotType.motion_graphic and s.id not in locked else None
        excess = s.duration - cap if cap is not None else 0.0
        for j in (k + 1, k - 1):
            if excess <= 1e-6 or not 0 <= j < len(shots) or shots[j].type == ShotType.motion_graphic:
                continue
            room = bounds(profile, shots[j].type, shots[j].start)[1] * 1.15 - shots[j].duration
            give = min(excess, max(0.0, room))
            if give > 0:
                _move(shots, j, k, give)
                excess -= give
        if excess > 1e-6 and footage_for is not None:
            cut = round(max(excess, MIN_SHOT), 3)
            rest = s.model_copy(deep=True, update={"id": f"{s.id}_hold", "type": ShotType.footage, "motion": None,
                                                   "overlay": None, "transition_in": "cut", "duration": cut})
            s.duration = round(s.duration - cut, 3)
            shots.insert(k + 1, rest)
            _retime(shots)
            if not _to_footage(shots, k + 1, ideas, rebind, footage_for, overlay_for):
                del shots[k + 1]
                s.duration = round(s.duration + cut, 3)
                _retime(shots)
        k += 1


def _to_footage(shots: list[Shot], k: int, ideas: list[VisualIdea], rebind: Rebind, footage_for: FootageFor,
                overlay_for: OverlayFor) -> bool:
    """Make shot k footage of its beat, with an overlay unless that makes a run; False when no idea fits."""
    idx = footage_for(shots[k])
    if idx is None:
        return False
    shots[k].type = ShotType.footage
    rebind(shots[k], ideas[idx], idx)
    if not _creates_run(shots, k, ShotType.footage_overlay):
        _set_type(shots[k], ShotType.footage_overlay, overlay_for)
    return True


def _card_runs(shots: list[Shot]) -> list[list[int]]:
    runs, cur = [], []
    for k, s in enumerate(shots):
        if s.type == ShotType.motion_graphic:
            cur.append(k)
        else:
            if cur:
                runs.append(cur)
            cur = []
    return runs + ([cur] if cur else [])


def break_card_runs(shots: list[Shot], ideas: list[VisualIdea], rebind: Rebind, footage_for: FootageFor,
                    overlay_for: OverlayFor, locked: frozenset[str] = frozenset()) -> None:
    """No more than MAX_MG_RUN motion graphics in a row or MAX_MG_RUN_S seconds of them: a free card in the
    run (text cards first, from the middle) becomes footage of its beat."""
    for _ in range(len(shots)):
        bad = next((r for r in _card_runs(shots) if len(r) > MAX_MG_RUN
                    or sum(shots[k].duration for k in r) > MAX_MG_RUN_S + 1e-6), None)
        if bad is None:
            return
        free = [k for k in bad if shots[k].id not in locked]
        if not free:
            continue
        mid = (bad[0] + bad[-1]) / 2
        k = min(free, key=lambda x: (shots[x].motion is None or shots[x].motion.kind not in TEXT_CARD_KINDS,
                                     abs(x - mid)))
        if not _to_footage(shots, k, ideas, rebind, footage_for, overlay_for):
            return


# ---------------------------------------------------------------- overlay text

def _overlay_key(spec: MotionSpec) -> str:
    """What the viewer reads: the title or label; an untitled map or chart inset by what it draws."""
    p = spec.payload
    text = p.get("title") or p.get("label") or ""
    key = " ".join(str(text).lower().split())
    if not key and spec.kind in (MotionKind.map, MotionKind.bar_chart, MotionKind.line_chart):
        drawn = [m.get("label", "") for m in p.get("markers", [])] + list(p.get("labels", []) or [])
        key = f"{spec.kind.value}:{' '.join(sorted(str(d).lower() for d in drawn if isinstance(d, str)))}"
    return key


def dedupe_overlays(shots: list[Shot], overlay_pool: OverlayPool, locked: frozenset[str] = frozenset()) -> None:
    """An overlay whose text is on screen within OVERLAY_REPEAT_S (another overlay or a full screen text card)
    takes the next pool option, else the shot becomes its plain sibling type. Each such loss is then given
    back to a plain shot of the same family that has a fresh overlay, so the planned mix holds."""
    used: list[tuple[str, float, float]] = [(_overlay_key(s.motion), s.start, s.end) for s in shots
                                            if s.type == ShotType.motion_graphic and s.motion is not None
                                            and s.motion.kind in TEXT_CARD_KINDS]

    def taken(key: str, s: Shot) -> bool:
        return bool(key) and any(k == key and a - s.end < OVERLAY_REPEAT_S and s.start - b < OVERLAY_REPEAT_S
                                 for k, a, b in used)

    lost: dict[ShotType, int] = {}
    for s in shots:
        if s.type not in OVERLAY_TYPES:
            continue
        if s.id not in locked:
            pool = overlay_pool(s)  # empty on call to action beats: no overlay at all
            keep = s.overlay is not None and pool and (s.overlay.kind not in I.TEXT_OVERLAY_KINDS or s.overlay in pool)
            options = ([s.overlay] if keep else []) + pool  # a text overlay stays only where its words are spoken
            s.overlay = next((o for o in options if not taken(_overlay_key(o), s)), None)
            if s.overlay is None:
                lost[s.type] = lost.get(s.type, 0) + 1
                s.type = PLAIN[s.type]
                continue
        if s.overlay is not None:
            used.append((_overlay_key(s.overlay), s.start, s.end))
    for k, s in enumerate(shots):
        typ = next((t for t, plain in PLAIN.items() if plain == s.type and lost.get(t, 0) > 0), None)
        if typ is None or s.id in locked or _creates_run(shots, k, typ):
            continue
        spec = next((o for o in overlay_pool(s) if not taken(_overlay_key(o), s)), None)
        if spec is not None:
            s.type, s.overlay = typ, spec
            used.append((_overlay_key(spec), s.start, s.end))
            lost[typ] -= 1


def time_overlays(shots: list[Shot], words: list[Word], locked: frozenset[str] = frozenset()) -> None:
    """An overlay belongs to a spoken moment, not to a shot: each text overlay starts ANCHOR_LEAD_S after its
    anchor is spoken (start_at into its shot) and holds its reading time before the cut. When its shot misses
    that moment, it moves to a plain shot within OVERLAY_MOVE_S that holds it, else it is dropped; text that
    repeats the narration spoken around it is dropped too. A shot that loses its overlay becomes its plain
    sibling. Map and chart insets and locked structural overlays keep start 0."""
    for k, s in enumerate(shots):
        spec = s.overlay
        if s.type not in OVERLAY_TYPES or spec is None or spec.kind not in I.TEXT_OVERLAY_KINDS or s.id in locked:
            continue
        s.overlay, s.type = None, PLAIN[s.type]
        hosts = [k] + sorted((j for j, h in enumerate(shots) if j != k and h.type in PLAIN.values()
                              and h.id not in locked and h.end > s.start - OVERLAY_MOVE_S
                              and h.start < s.end + OVERLAY_MOVE_S), key=lambda j: abs(j - k))
        for j in hosts:
            h = shots[j]
            at = I.overlay_start(spec, words, h.start, h.end)
            if at is None or (j != k and _creates_run(shots, j, TOGGLE[h.type])):
                continue
            if I.spec_repeats_narration(spec, I.spoken_near(words, h.start + at)):
                break
            h.type = TOGGLE[h.type]
            h.overlay = spec.model_copy(update={"start_at": at, "anchor": spec.anchor or I.overlay_title(spec),
                                                "duration": h.duration})
            break


def drop_repeated_overlays(shots: list[Shot], locked: frozenset[str] = frozenset()) -> list[str]:
    """Last pass, after every step that places an overlay (card demotion, overlay timing): an overlay whose text
    is on screen within OVERLAY_REPEAT_S (an earlier overlay or a full screen text card) is dropped and its shot
    becomes its plain sibling. Returns the shot ids that lost one."""
    used = [(_overlay_key(s.motion), s.start, s.end) for s in shots if s.type == ShotType.motion_graphic
            and s.motion is not None and s.motion.kind in TEXT_CARD_KINDS]
    dropped: list[str] = []
    for s in shots:
        if s.type not in OVERLAY_TYPES or s.overlay is None:
            continue
        key = _overlay_key(s.overlay)
        if (key and s.id not in locked and any(k == key and a - s.end < OVERLAY_REPEAT_S
                                               and s.start - b < OVERLAY_REPEAT_S for k, a, b in used)):
            s.overlay, s.type = None, PLAIN[s.type]
            dropped.append(s.id)
            continue
        used.append((key, s.start, s.end))
    return dropped


# ---------------------------------------------------------------- card spacing

INSET_CARD_KINDS = {MotionKind.map, MotionKind.bar_chart, MotionKind.line_chart}


def space_cards(shots: list[Shot], profile: StyleProfile, exempt: list[tuple[float, float]], ideas: list[VisualIdea],
                rebind: Rebind, footage_for: FootageFor, overlay_for: OverlayFor,
                locked: frozenset[str] = frozenset()) -> int:
    """At most one full screen card per card_spacing_s outside the exempt ranges; returns cards demoted.

    Locked cards (number cards, title card, recap) always stay and count against their neighbours.
    """
    spacing = profile.pacing.card_spacing_s
    if spacing <= 0:
        return 0
    kept = [s.start for s in shots if s.type == ShotType.motion_graphic and s.id in locked]
    demoted = 0
    for k, s in enumerate(shots):
        if s.type != ShotType.motion_graphic or s.id in locked:
            continue
        if not card_too_close(s.start, kept, spacing, exempt):
            kept.append(s.start)
            continue
        if demote_card(shots, k, ideas, rebind, footage_for, overlay_for):
            demoted += 1
        else:
            kept.append(s.start)
    return demoted


def demote_repeating_cards(shots: list[Shot], words: list[Word], ideas: list[VisualIdea], rebind: Rebind,
                           footage_for: FootageFor, overlay_for: OverlayFor,
                           locked: frozenset[str] = frozenset()) -> int:
    """A free callout or text card that repeats the narration spoken within REPEAT_WINDOW_S of it (the window
    QA and acceptance use, not only its own beat) becomes footage; returns cards demoted."""
    demoted = 0
    for k, s in enumerate(shots):
        if (s.type == ShotType.motion_graphic and s.id not in locked and s.motion is not None
                and s.motion.kind in (MotionKind.callout, MotionKind.text_overlay)
                and I.spec_repeats_narration(s.motion, I.spoken_near(words, s.start))):
            demoted += demote_card(shots, k, ideas, rebind, footage_for, overlay_for)
    return demoted


def demote_card(shots: list[Shot], k: int, ideas: list[VisualIdea], rebind: Rebind, footage_for: FootageFor,
                overlay_for: OverlayFor) -> bool:
    """Turn a free card into footage of its beat that keeps the card's message as an overlay (card_overlay)."""
    shot = shots[k]
    spec = shot.motion
    idx = footage_for(shot)
    if idx is None:
        return False
    shot.type = ShotType.footage
    rebind(shot, ideas[idx], idx)
    keep = card_overlay(spec) if spec is not None else None
    if keep is not None and keep.kind in INSET_CARD_KINDS and any(
            o is not shot and o.overlay is not None and o.overlay.kind in INSET_CARD_KINDS
            and abs(o.start - shot.start) < INSET_GAP_S for o in shots):
        keep = None  # an inset needs room to be read (assembler.insets)
    if keep is not None:
        shot.type, shot.overlay = ShotType.footage_overlay, keep
    elif not _creates_run(shots, k, ShotType.footage_overlay):
        _set_type(shot, ShotType.footage_overlay, overlay_for)
    return True


def card_overlay(spec: MotionSpec) -> Optional[MotionSpec]:
    """A full screen card's message as an overlay on footage: a title as a lower third, a map or chart as an
    inset, a number, text or callout card on a transparent background. None for other kinds."""
    if spec.kind == MotionKind.title_card:
        p = spec.payload
        return b.lower_third_spec(str(p.get("title", "")), str(p.get("subtitle", "")), duration=spec.duration,
                                  palette=spec.style.palette)
    if spec.kind in INSET_CARD_KINDS or spec.kind in STATIC_CARD_KINDS:
        return spec.model_copy(update={"background": "transparent"})
    return None


# ---------------------------------------------------------------- cards on cue

CUE_EARLY_S = 0.5  # the cue word may start this long before the card
CUE_LATE_S = 1.5  # or at most this far into it
CUE_LEAD_S = 0.3  # a card moved to its cue starts this long before the word
_SAID_NUMBER_RE = re.compile(r"\$?(\d[\d,]*(?:\.\d+)?)\s*(?:%|percent)?")


def _norm(word: str) -> str:
    return re.sub(r"'s$", "", word.lower().strip(".,;:!?\"'()[]"))


def cue_times(spec: MotionSpec, words: Sequence[Word]) -> list[float]:
    """Start times of the words that speak a card's key text: a number card's number (digits, a year read
    aloud or a spelled out count, up to four words), else the first content word of the card's title."""
    p = spec.payload
    if spec.kind != MotionKind.number_card:
        key = next((t for t in map(_norm, str(p.get("title", "")).split()) if len(t) > 2 and t not in I.STOP), "")
        return [w.start for w in words if key and _norm(w.text) == key]
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(p.get("number", "")))
    if m is None:
        return []
    key = float(m.group(0).replace(",", ""))
    out: list[float] = []
    for i, w in enumerate(words):
        for n in range(1, 5):
            said = " ".join(_norm(x.text) for x in words[i:i + n])
            said = _SAID_NUMBER_RE.fullmatch(I.spelled_to_digits(I.spoken_years_to_digits(said)))
            if said and float(said.group(1).replace(",", "")) == key:
                out.append(w.start)
                break
    return out


def _shift(shots: list[Shot], k: int, delta: float, profile: StyleProfile, locked: frozenset[str]) -> bool:
    """Move card k by delta seconds, keeping its length: one neighbour grows and the other shrinks. Both must
    be free non card shots that stay within their min and 1.15 times their max."""
    if k == 0 or k + 1 >= len(shots):
        return False
    for sh, new in ((shots[k - 1], shots[k - 1].duration + delta), (shots[k + 1], shots[k + 1].duration - delta)):
        lo, hi, _ = bounds(profile, sh.type, sh.start)
        if sh.type == ShotType.motion_graphic or sh.id in locked or not max(lo, MIN_SHOT) - EPS <= new <= hi * 1.15:
            return False
    shots[k - 1].duration = round(shots[k - 1].duration + delta, 3)
    shots[k + 1].duration = round(shots[k + 1].duration - delta, 3)
    _retime(shots)
    return True


def cards_on_cue(shots: list[Shot], words: Sequence[Word], spans: dict[str, tuple[float, float]],
                 profile: StyleProfile, ideas: list[VisualIdea], rebind: Rebind, footage_for: FootageFor,
                 overlay_for: OverlayFor, locked: frozenset[str] = frozenset()) -> int:
    """A free text or number card appears on its cue: a word of its beat that speaks its number or name
    (cue_times) starts at most CUE_EARLY_S before the card or CUE_LATE_S into it. Otherwise the card moves to
    start CUE_LEAD_S before the nearest cue (_shift); when it cannot move, or the beat never says it, it
    becomes footage carrying its message (demote_card). Returns cards demoted."""
    demoted = 0
    for k, s in enumerate(shots):
        if s.type != ShotType.motion_graphic or s.id in locked or s.motion is None \
                or s.motion.kind not in STATIC_CARD_KINDS:
            continue
        a, z = spans.get(s.beat_id, (s.start, s.end))
        cues = cue_times(s.motion, [w for w in words if a - EPS <= w.start <= z + EPS])
        if any(s.start - CUE_EARLY_S <= c <= s.start + CUE_LATE_S for c in cues):
            continue
        cue = min(cues, key=lambda c: abs(c - s.start)) if cues else None
        if cue is not None and _shift(shots, k, cue - CUE_LEAD_S - s.start, profile, locked):
            continue
        if demote_card(shots, k, ideas, rebind, footage_for, overlay_for):
            demoted += 1
    return demoted
