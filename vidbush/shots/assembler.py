"""Layer two of the director: pacing slots to shots by picking ideas.

_Assembler.pick() scores every idea of the slot's family. Footage and stills
may come from nearby beats (small distance penalty). Motion graphics only come
from the beat the slot plays over, or the adjacent beat when the slot spans it,
so a map or card never shows content the narration has not reached. Reuse is
charged by content signature (maps by view and labels, cards by text), the
same motion kind within 60 s costs a decaying penalty, and maps are capped at
about MAP_SHARE of motion slots unless they show a new place, and the same map
is never shown twice within INSET_REPEAT_S (card or inset). Overlay shots
draw first on the beat's own map and chart ideas as insets over the picture,
since reference channels move most of their animation over footage.
"""
from __future__ import annotations

import bisect
import random
import re
from typing import Optional

from ..models import (OVERLAY_TYPES, MapPayload, MotionKind, MotionSpec, Script, Shot, ShotType, StyleProfile,
                      VisualIdea)
from ..util import no_em_dash
from . import ideas as I
from ..mg.inset import INSET_KINDS
from .pacing import Slot

FAMILY = {ShotType.footage: "footage", ShotType.footage_overlay: "footage", ShotType.image: "still",
          ShotType.image_overlay: "still", ShotType.ai_image: "still", ShotType.ai_video: "footage",
          ShotType.motion_graphic: "motion"}
TEXT_KINDS = {MotionKind.title_card, MotionKind.text_overlay, MotionKind.callout}
PLAIN_OF = {ShotType.footage_overlay: ShotType.footage, ShotType.image_overlay: ShotType.image}
KIND_WINDOW_S = 60.0
KIND_PENALTY = 0.5
MAP_SHARE = 0.35
REUSE_PENALTY = 0.9
INSET_GAP_S = 15.0  # seconds between two map or chart insets
INSET_REPEAT_S = 180.0  # the same inset is not shown again within this


class _BeatLocator:
    """Beat at a timeline position: the latest beat that started at or before it."""

    def __init__(self, script: Script, spans: dict[str, tuple[float, float]]) -> None:
        pairs = sorted((spans[bt.id][0], bt.id) for bt in script.beats() if bt.id in spans)
        self.starts = [p[0] for p in pairs]
        self.ids = [p[1] for p in pairs]

    def at(self, t: float) -> str:
        i = bisect.bisect_right(self.starts, t) - 1
        return self.ids[max(0, i)] if self.ids else ""


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Seconds two time ranges share."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


class _Assembler:
    """Turns pacing slots into shots by picking ideas; also serves rules.py callbacks."""

    def __init__(self, script: Script, spans: dict[str, tuple[float, float]], bcs: list[I.BeatContext],
                 ideas: list[VisualIdea], profile: StyleProfile, rng: random.Random) -> None:
        self.script, self.spans, self.profile, self.rng = script, spans, profile, rng
        self.ideas = ideas
        self.bc = {bc.beat.id: bc for bc in bcs}
        self.order = [bc.beat.id for bc in bcs]
        self.pos = {bid: i for i, bid in enumerate(self.order)}
        self.locate = _BeatLocator(script, spans)
        self.idea_of: dict[str, int] = {}
        self.locked: set[str] = set()
        self.avoid_kind: set[MotionKind] = set()
        self.usage: dict[int, int] = {}
        self.sig_usage: dict[str, int] = {}
        self.motion_log: list[tuple[float, MotionKind]] = []
        self.card_log: list[tuple[float, MotionSpec]] = []
        self.inset_log: list[float] = []
        self.map_places: set[str] = set()
        self.by_key: dict[tuple[str, str], list[int]] = {}
        for i, idea in enumerate(ideas):
            self.by_key.setdefault((idea.beat_id, idea.family), []).append(i)

    def assign(self, slots: list[Slot]) -> list[Shot]:
        shots: list[Shot] = []
        prev_scene = self.bc[self.order[0]].scene.id if self.order else ""
        fades = 0
        for k, slot in enumerate(slots):
            beat_id = self.locate.at((slot.start + slot.end) / 2) or self.order[0]
            nxt = slots[k + 1] if k + 1 < len(slots) else None
            self.avoid_kind = {sp.kind for sp in (nxt.motion if nxt and nxt.fixed else None,
                                                  shots[-1].motion if shots else None) if sp is not None}
            idx = self._slot_idea(slot, beat_id, shots)
            shot = self._make_shot(k, slot, beat_id, idx)
            if slot.fixed:
                self.locked.add(shot.id)
            scene_id = shot.scene_id
            if k > 0 and scene_id != prev_scene:
                p_fade = 1.0 - self.profile.transitions.cut_probability
                if fades == 0 or self.rng.random() < p_fade:
                    shot.transition_in = "fade"
                    fades += 1
            prev_scene = scene_id
            shots.append(shot)
        return shots

    def _slot_idea(self, slot: Slot, beat_id: str, shots: list[Shot]) -> int:
        family = FAMILY[slot.type]
        if slot.fixed and slot.motion is not None:
            idea = VisualIdea(beat_id=beat_id, family="motion", intent=_anchor_intent(slot), motion=slot.motion,
                              priority=2.0, opening_safe=False)
            self.ideas.append(idea)
            return len(self.ideas) - 1
        prev = self.idea_of.get(shots[-1].id) if shots else None
        return self.pick(beat_id, family, slot.start, prev, slot.duration)

    def _beats_near(self, beat_id: str, family: str, at: float, duration: float) -> list[tuple[int, str]]:
        """(distance, beat id) an idea may come from: up to 3 beats away, motion only from the beats it plays over."""
        bpos = self.pos.get(beat_id, 0)
        out: list[tuple[int, str]] = []
        for d in (0, -1, 1, -2, 2, -3, 3):
            j = bpos + d
            if not 0 <= j < len(self.order):
                continue
            bid = self.order[j]
            if family == "motion" and d != 0:
                span = self.spans.get(bid, (0.0, 0.0))
                if abs(d) > 1 or overlap(span, (at, at + duration)) < min(2.0, 0.3 * max(duration, 0.1)):
                    continue
            out.append((d, bid))
        return out

    def pick(self, beat_id: str, family: str, at: float, prev: Optional[int], duration: float = 0.0,
             exclude: tuple[int, ...] = ()) -> int:
        """Best idea of a family near a beat: priority minus reuse, distance and context penalties."""
        hook = at < self.profile.pacing.hook_seconds
        scene = self.bc[beat_id].scene.id if beat_id in self.bc else ""
        scene_start = self.spans.get(self.bc[beat_id].scene.beats[0].id, (0, 0))[0] if beat_id in self.bc else 0.0
        best, best_score = -1, -1e9
        for d, bid in self._beats_near(beat_id, family, at, duration):
            for i in self.by_key.get((bid, family), []):
                idea = self.ideas[i]
                if idea.motion is not None and self._repeated_map(idea.motion, at):
                    continue
                reuse = max(self.usage.get(i, 0), self.sig_usage.get(signature(idea), 0))
                score = idea.priority - REUSE_PENALTY * reuse - 0.25 * abs(d)
                score -= 0.8 if self.bc[bid].scene.id != scene else 0.0
                score -= 5.0 if i == prev or i in exclude else 0.0
                if hook and family == "footage" and not idea.opening_safe:
                    score -= 0.5
                if idea.motion is not None:
                    score -= self._motion_penalty(idea.motion, at, scene_start, duration)
                if score > best_score:
                    best, best_score = i, score
        if best < 0:
            best = self._fallback_idea(beat_id, family, at)
        self._count(best, 1)
        if family == "motion" and self.ideas[best].motion is not None:
            self._log_motion(self.ideas[best].motion, at)
        return best

    def _count(self, idx: int, delta: int) -> None:
        sig = signature(self.ideas[idx])
        self.usage[idx] = max(0, self.usage.get(idx, 0) + delta)
        self.sig_usage[sig] = max(0, self.sig_usage.get(sig, 0) + delta)

    def _log_motion(self, spec: MotionSpec, at: float) -> None:
        self.motion_log.append((at, spec.kind))
        self.card_log.append((at, spec))
        self.map_places |= map_places(spec)

    def _motion_penalty(self, spec: MotionSpec, at: float, scene_start: float, duration: float) -> float:
        mg = self.profile.pacing.motion_graphic
        p = 1.0 if spec.kind in self.avoid_kind else 0.0
        if spec.kind == MotionKind.title_card and at - scene_start > 8.0:
            p += 1.0
        if spec.kind in TEXT_KINDS and duration > mg.min + 1.5:
            p += 1.5
        if spec.kind == MotionKind.number_card and duration > 9.0:
            p += 0.3
        if spec.kind == MotionKind.map and duration and duration < 6.0:
            p += 0.3
        recent = [at - t for t, k in self.motion_log if k == spec.kind and 0 <= at - t < KIND_WINDOW_S]
        if recent:
            p += KIND_PENALTY * (1.0 - min(recent) / KIND_WINDOW_S)
        if spec.kind == MotionKind.map:
            maps = sum(1 for _t, k in self.motion_log if k == MotionKind.map)
            if maps >= MAP_SHARE * (len(self.motion_log) + 1) and not (map_places(spec) - self.map_places):
                p += 1.0
        return p

    def _repeated_map(self, spec: MotionSpec, at: float) -> bool:
        """True for a map already on screen (full screen card or inset) within INSET_REPEAT_S."""
        if spec.kind != MotionKind.map:
            return False
        sig = motion_signature(spec)
        return any(sp.kind == MotionKind.map and abs(at - t) < INSET_REPEAT_S and motion_signature(sp) == sig
                   for t, sp in self.card_log)

    def _fallback_idea(self, beat_id: str, family: str, at: float = 0.0) -> int:
        bc = self.bc[beat_id]
        made = [x for x in I.ideas_for_beat(bc, None, self.profile.palette, random.Random(beat_id))
                if x.family == family]
        idea = next((x for x in made if x.motion is None or not self._repeated_map(x.motion, at)), made[0])
        self.ideas.append(idea)
        self.by_key.setdefault((beat_id, family), []).append(len(self.ideas) - 1)
        return len(self.ideas) - 1

    def _make_shot(self, k: int, slot: Slot, beat_id: str, idx: int) -> Shot:
        bc = self.bc[beat_id]
        shot = Shot(id=f"shot_{k + 1:03d}", index=k, beat_id=beat_id, scene_id=bc.scene.id,
                    start=round(slot.start, 3), end=round(slot.end, 3), duration=round(slot.end - slot.start, 3),
                    type=slot.type, intent="")
        self.rebind(shot, self.ideas[idx], idx)
        if slot.type == ShotType.motion_graphic:
            shot.motion = slot.motion or shot.motion
        if slot.type in OVERLAY_TYPES:
            shot.overlay = slot.overlay or self.overlay_for(shot)
            if shot.overlay is None:
                shot.type = PLAIN_OF[shot.type]  # nothing honest to overlay (call to action beat)
            elif shot.overlay.kind in INSET_KINDS:
                self.card_log.append((slot.start, shot.overlay))
                self.inset_log.append(slot.start)
        return shot

    # -------------------------------------------------------- callbacks for rules.py

    def rebind(self, shot: Shot, idea: VisualIdea, idx: Optional[int] = None) -> None:
        """Fill a shot's content fields from an idea."""
        if idx is None:
            idx = self.ideas.index(idea)
        old = self.idea_of.get(shot.id)
        if old is not None and old != idx:
            self._count(old, -1)
            self._count(idx, 1)
        self.idea_of[shot.id] = idx
        shot.intent = idea.intent
        shot.must_show = list(idea.must_show)
        shot.must_avoid = list(idea.must_avoid)
        shot.queries = list(idea.queries)
        shot.motion = idea.motion if shot.type == ShotType.motion_graphic else None
        if shot.type in OVERLAY_TYPES:
            shot.overlay = idea.overlay or shot.overlay
        else:
            shot.overlay = None
        shot.ai_prompt = "" if shot.type == ShotType.motion_graphic else ai_prompt(idea, self.bc.get(shot.beat_id))

    def overlay_pool(self, shot: Shot) -> list[MotionSpec]:
        """Overlays a shot may carry, best first: map and chart insets of its beat (animation moves over the
        picture), its idea's overlay, then lower thirds for what it shows and the beat's generic overlays.
        Empty on call-to-action beats."""
        bc = self.bc.get(shot.beat_id)
        if bc is None or I.CTA_RE.search(bc.beat.text):
            return []
        idx = self.idea_of.get(shot.id)
        own = [self.ideas[idx].overlay] if idx is not None and self.ideas[idx].overlay is not None else []
        return self.insets(shot.beat_id, shot.start) + own + I.overlay_choices(bc, self.profile.palette,
                                                                              shot.must_show)

    def insets(self, beat_id: str, at: float) -> list[MotionSpec]:
        """The beat's map and chart ideas as transparent insets.

        An inset is information the viewer has to read, so it needs room: none within INSET_GAP_S of
        another inset, the same map or chart not again within INSET_REPEAT_S, and a map only when it shows
        a place no map showed in the last KIND_WINDOW_S (a map earns its place when the narration moves
        between places, not on every beat that names one)."""
        if any(0 < abs(at - t) < INSET_GAP_S for t in self.inset_log):
            return []
        recent = [sp for t, sp in self.card_log if 0 <= at - t < KIND_WINDOW_S]
        shown = {motion_signature(sp) for t, sp in self.card_log if 0 <= at - t < INSET_REPEAT_S}
        places = set().union(*[map_places(sp) for sp in recent]) if recent else set()
        out: list[MotionSpec] = []
        for i in self.by_key.get((beat_id, "motion"), []):
            spec = self.ideas[i].motion
            if spec is None or spec.kind not in INSET_KINDS or motion_signature(spec) in shown:
                continue
            if spec.kind == MotionKind.map and not (map_places(spec) - places):
                continue
            out.append(spec.model_copy(update={"background": "transparent"}))
        return out

    def overlay_for(self, shot: Shot) -> Optional[MotionSpec]:
        """First overlay for an overlay shot, or None when nothing fits."""
        pool = self.overlay_pool(shot)
        return pool[0] if pool else None

    def footage_for(self, shot: Shot) -> Optional[int]:
        """A footage idea for a shot that must stop being a full screen card."""
        return self.pick(shot.beat_id, "footage", shot.start, self.idea_of.get(shot.id), shot.duration)


def map_places(spec: MotionSpec) -> set[str]:
    """Marker and label names a map shows (empty for other kinds)."""
    if spec.kind != MotionKind.map:
        return set()
    p = spec.payload
    names = [m.get("label", "") for m in p.get("markers", [])] + [lb.get("text", "") for lb in p.get("labels", [])]
    return {n.strip().lower() for n in names if n and n.strip()}


def signature(idea: VisualIdea) -> str:
    """Content signature: footage and stills by intent; motion graphics by what is drawn on screen."""
    if idea.family == "motion" and idea.motion is not None:
        return motion_signature(idea.motion)
    return f"{idea.family}:{re.sub(r'[^a-z0-9]+', ' ', idea.intent.lower()).strip()}"


def motion_signature(spec: MotionSpec) -> str:
    """Maps by rounded view (0.5 degree, 0.5 zoom) and sorted labels; cards and charts by their text."""
    p = spec.payload
    if spec.kind == MotionKind.map:
        try:
            mp = MapPayload(**p)
        except (TypeError, ValueError):
            return f"map:{sorted(map_places(spec))}"
        view = (round(mp.center[0] * 2) / 2, round(mp.center[1] * 2) / 2, round(mp.zoom * 2) / 2) if mp.center else ()
        routes = sorted(r.label or "route" for r in mp.routes)
        return f"map:{view}:{sorted(map_places(spec))}:{routes}"
    text = p.get("number") or p.get("title") or ""
    extra = p.get("label") or p.get("labels") or ""
    return f"{spec.kind.value}:{re.sub(r'[^a-z0-9]+', ' ', f'{text} {extra}'.lower()).strip()}"


def _anchor_intent(slot: Slot) -> str:
    p = slot.motion.payload if slot.motion else {}
    label = p.get("title") or p.get("label") or ""
    return no_em_dash(f"{slot.tag.replace('_', ' ').capitalize()}: {p.get('number', '')} {label}".replace("  ", " "))


# ---------------------------------------------------------------- AI image prompt

_STYLE_PREFIX = re.compile(r"^(black and white |vintage |historical |archival |cinematic )*(archival footage|footage|"
                           r"historical photograph|documentary photograph|photograph|wide establishing archival "
                           r"footage|wide establishing footage|close detail photograph|portrait photograph|image|"
                           r"shot|still)\s+of\s+", re.I)
_PURPOSE = re.compile(r"(?:,\s+|\s+(?=to (?:show|illustrate|convey|suggest|emphasi[sz]e)\b))(to show|to illustrate|"
                      r"to convey|to suggest|to emphasi[sz]e|showing (?:how|that|why)|which shows|that shows|"
                      r"illustrating|for the|as a symbol|symboli[sz]ing|representing)\b.*$", re.I)
_COLOURS = re.compile(r"\b(green|golden|gold-lit|blue|red|yellow|orange|purple|pink|turquoise|emerald|colou?rful|"
                      r"vivid|sunlit golden|lush green)\s+", re.I)
_LOWER_FIRST = I.STOP | {"wide", "close", "aerial", "old", "early", "huge", "giant", "dramatic", "busy", "crowded",
                         "steam", "rows", "lines", "dozens", "hundreds", "thousands"}


def prompt_subject(intent: str, era: Optional[int]) -> str:
    """Subject of an image prompt: no style prefix, no purpose clause, lowercase article, era-safe wording."""
    subject = _STYLE_PREFIX.sub("", intent.strip().rstrip("."))
    subject = _PURPOSE.sub("", subject)
    subject = re.sub(r",?\s*(in the )?\d{4}s$", "", subject).strip(" ,")
    first = subject.split(" ", 1)[0] if subject else ""
    if first and (first.lower() in _LOWER_FIRST or first.lower() in I.CONCRETE):
        subject = subject[:1].lower() + subject[1:]
    if era and era < 1935:
        subject = _COLOURS.sub("", subject)
    if era and era < 1930:
        subject = re.sub(r"\b(an )?aerial (view|shot|photograph)\b", "an elevated view from a hillside", subject,
                         flags=re.I)
        subject = re.sub(r"\baerial\b", "elevated", subject, flags=re.I)
    return " ".join(subject.split())


def ai_prompt(idea: VisualIdea, bc: Optional[I.BeatContext]) -> str:
    """Positive, era-correct image prompt: style, subject, era, what it shows, light and lens (no negations)."""
    era = bc.era if bc else None
    if era and era < 1935:
        style = "Black and white archival documentary photograph"
    elif era and era < 1975:
        style = "Vintage color documentary photograph on period film stock"
    else:
        style = "Cinematic documentary photograph"
    when = f", {I.era_label(era)}" if era else ""
    subject = prompt_subject(idea.intent, era) or "a documentary scene"
    shows = f" Clearly showing {', '.join(idea.must_show[:4])}." if idea.must_show else ""
    guard = " Period-accurate clothing, vehicles and architecture." if era and era < 1990 else ""
    return no_em_dash(f"{style} of {subject}{when}.{shows}{guard} Natural light, realistic detail, wide 16:9 "
                      "composition, 35mm lens.")
