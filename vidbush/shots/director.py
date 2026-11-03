"""Director: script plus narration timing to a typed, timed ShotList.

Layer one gathers visual ideas per beat from the director seat (validated:
pydantic, map coordinates checked against the gazetteer, chart numbers
checked against fact cards) merged with heuristic ideas so every beat has
footage, still and motion ideas. Layer two is the pacing engine, which plans
typed slots over [0, narration.duration + 1.0]; each slot takes the best
unused idea of its family from the beat at its midpoint. rules.py then
polishes the sequence.
"""
from __future__ import annotations

import bisect
import json
import random
import re
from typing import Any, Optional

from pydantic import ValidationError

from ..mg import builders as b
from ..models import (OVERLAY_TYPES, Brand, ChartPayload, DirectorPlan, MapPayload, MotionKind, MotionSpec, Narration,
                      Project, Research, ScenePurpose, Script, Shot, ShotList, ShotType, StyleProfile, VideoFormat,
                      VisualIdea)
from ..util import NullLogger, no_em_dash
from . import ideas as I
from .pacing import Anchor, Slot, plan_slots

FAMILY = {ShotType.footage: "footage", ShotType.footage_overlay: "footage", ShotType.image: "still",
          ShotType.image_overlay: "still", ShotType.ai_image: "still", ShotType.ai_video: "footage",
          ShotType.motion_graphic: "motion"}
OVERLAY_KINDS = {MotionKind.lower_third, MotionKind.callout, MotionKind.text_overlay, MotionKind.highlight_box,
                 MotionKind.arrow}
MAP_TOLERANCE_DEG = 1.0

SYSTEM = (
    "You are the director of a long-form faceless YouTube documentary. For every narration beat propose "
    "concrete visual ideas a researcher can find or build. Reply with strict JSON only, no prose, no markdown. "
    "Keep every string short and concrete."
)


# ---------------------------------------------------------------- entry point

def build_shotlist(project: Project, script: Script, narration: Narration, research: Optional[Research],
                   profile: StyleProfile, brand: Brand, seats: Any,
                   rng: random.Random) -> tuple[ShotList, DirectorPlan]:
    """Ideas per beat (seat plus heuristics), pacing slots, rules; returns the ShotList and the merged plan."""
    from .rules import apply_rules
    log = getattr(seats, "log", None) or NullLogger()
    total = round(narration.duration + 1.0, 3)
    spans = beat_spans(script, narration)
    topic = I.topic_of(script.title or project.title, research)
    bcs = I.beat_contexts(script.scenes, research, topic)
    llm = director_ideas(script, spans, research, profile, brand, seats, log)
    ideas = merge_ideas(llm, bcs, research, profile.palette, rng)
    top10 = profile.top10 or script.format == VideoFormat.top10
    anchors = structural_anchors(script, spans, profile, top10, total, topic)
    slots = plan_slots(total, profile, rng, narration.words, scene_starts(script, spans), anchors)
    ctx = _Assembler(script, spans, bcs, ideas, profile, rng)
    shots = ctx.assign(slots)
    shots = apply_rules(shots, ctx.ideas, ctx.idea_of, profile, ctx.rebind, ctx.overlay_for, frozenset(ctx.locked))
    sl = ShotList(project_id=project.id, duration=total, shots=shots)
    _finalize(sl, profile)
    log.info("shot list built", shots=len(sl.shots), ideas=len(ctx.ideas), llm_ideas=len(llm),
             duration=sl.duration, top10=top10)
    return sl, DirectorPlan(ideas=ctx.ideas)


# ---------------------------------------------------------------- timing helpers

def beat_spans(script: Script, narration: Narration) -> dict[str, tuple[float, float]]:
    """beat_id -> (start, end): narration spans, else word beat ids, else proportional to word count."""
    beats = script.beats()
    spans = {k: (float(v[0]), float(v[1])) for k, v in narration.beat_spans.items() if len(v) >= 2}
    if beats and all(bt.id in spans for bt in beats):
        return spans
    by_words: dict[str, list[float]] = {}
    for w in narration.words:
        if w.beat_id:
            s = by_words.setdefault(w.beat_id, [w.start, w.end])
            s[0], s[1] = min(s[0], w.start), max(s[1], w.end)
    if beats and all(bt.id in by_words for bt in beats):
        return {k: (v[0], v[1]) for k, v in by_words.items()}
    counts = [max(1, len(bt.text.split())) for bt in beats]
    total_words = sum(counts) or 1
    out, t = {}, 0.0
    for bt, c in zip(beats, counts):
        d = narration.duration * c / total_words
        out[bt.id] = (t, t + d)
        t += d
    return out


def scene_starts(script: Script, spans: dict[str, tuple[float, float]]) -> list[float]:
    """Start time of every scene after the first."""
    out = []
    for sc in script.scenes[1:]:
        if sc.beats and sc.beats[0].id in spans:
            out.append(spans[sc.beats[0].id][0])
    return out


class _BeatLocator:
    """Beat at a timeline position: the latest beat that started at or before it."""

    def __init__(self, script: Script, spans: dict[str, tuple[float, float]]) -> None:
        pairs = sorted((spans[bt.id][0], bt.id) for bt in script.beats() if bt.id in spans)
        self.starts = [p[0] for p in pairs]
        self.ids = [p[1] for p in pairs]

    def at(self, t: float) -> str:
        i = bisect.bisect_right(self.starts, t) - 1
        return self.ids[max(0, i)] if self.ids else ""


# ---------------------------------------------------------------- layer one: ideas

def director_ideas(script: Script, spans: dict[str, tuple[float, float]], research: Optional[Research],
                   profile: StyleProfile, brand: Brand, seats: Any, log: Any) -> list[VisualIdea]:
    """Ask the director seat for a DirectorPlan and keep only ideas that validate."""
    if seats is None:
        return []
    try:
        raw = seats.llm_json("director", "director", SYSTEM, director_prompt(script, spans, research, profile, brand),
                             "shotlist")
    except Exception as e:  # noqa: BLE001 - any seat failure falls back to heuristics
        log.warn("director seat failed, using heuristics", error=str(e)[:300])
        return []
    items = raw.get("ideas") if isinstance(raw, dict) else raw if isinstance(raw, list) else None
    if not isinstance(items, list):
        return []
    beat_ids = {bt.id for bt in script.beats()}
    out = [idea for idea in (validate_idea(it, beat_ids, research) for it in items) if idea]
    log.info("director ideas validated", received=len(items), kept=len(out))
    return out


def director_prompt(script: Script, spans: dict[str, tuple[float, float]], research: Optional[Research],
                    profile: StyleProfile, brand: Brand) -> str:
    """User message for the director seat."""
    beats = [{"id": bt.id, "scene": sc.id, "scene_title": sc.title, "purpose": sc.purpose.value, "text": bt.text,
              "start": round(spans.get(bt.id, (0, 0))[0], 2), "end": round(spans.get(bt.id, (0, 0))[1], 2),
              "facts": bt.facts_used} for sc in script.scenes for bt in sc.beats]
    facts = [{"id": f.id, "claim": f.claim[:240], "numbers": f.numbers, "source": f.source_title, "date": f.date}
             for f in (research.facts if research else [])]
    anchors = [a.model_dump(exclude_defaults=True) for a in (research.anchors if research else [])]
    brief = {
        "title": script.title, "format": script.format.value, "visual_mix_percent": profile.visual_mix,
        "channel_notes": brand.niche_notes, "beats": beats, "facts": facts, "anchors": anchors,
        "rules": [
            "For every beat give at least one idea of each family: footage, still, motion.",
            "footage and still ideas: intent, must_show, must_avoid, 2 to 4 search queries, most specific first, "
            "era correct. Optional overlay MotionSpec (lower_third, text_overlay, callout, highlight_box, arrow).",
            "motion ideas carry a MotionSpec: map (real lat/lon), bar_chart or line_chart (only numbers from the "
            "facts, with source_note), number_card, title_card, text_overlay, callout.",
            "opening_safe true only for footage with real activity, people or movement.",
            "No em dashes. Concise strings.",
        ],
        "output": {"ideas": [{"beat_id": "b001", "family": "footage|still|motion", "intent": "...",
                              "must_show": [], "must_avoid": [], "queries": [], "overlay": None, "motion": None,
                              "priority": 1.0, "opening_safe": True}]},
    }
    return json.dumps(brief, ensure_ascii=False)


def validate_idea(item: Any, beat_ids: set[str], research: Optional[Research]) -> Optional[VisualIdea]:
    """Pydantic validation plus content checks; None when the idea is unusable."""
    try:
        idea = VisualIdea.model_validate(item)
    except (ValidationError, TypeError, ValueError):
        return None
    if idea.beat_id not in beat_ids:
        return None
    idea = _clean_text(idea)
    if idea.overlay is not None and (idea.overlay.kind not in OVERLAY_KINDS or not _payload_ok(idea.overlay)):
        idea.overlay = None
    if idea.family != "motion":
        idea.motion = None
        return idea
    if idea.motion is None or not _payload_ok(idea.motion):
        return None
    if idea.motion.kind == MotionKind.map:
        idea.motion = fix_map(idea.motion)
    if idea.motion.kind in (MotionKind.bar_chart, MotionKind.line_chart):
        chart = check_chart(idea.motion, research)
        if chart is None:
            return None
        idea.motion = chart
    return idea


def _payload_ok(spec: MotionSpec) -> bool:
    try:
        spec.typed_payload()
    except (ValidationError, TypeError, ValueError, KeyError):
        return False
    return True


def _clean_text(idea: VisualIdea) -> VisualIdea:
    """Strip em dashes from every string the idea carries."""
    data = idea.model_dump(mode="json")
    return VisualIdea.model_validate(_strip(data))


def _strip(v: Any) -> Any:
    if isinstance(v, str):
        return no_em_dash(v)
    if isinstance(v, list):
        return [_strip(x) for x in v]
    if isinstance(v, dict):
        return {k: _strip(x) for k, x in v.items()}
    return v


def fix_map(spec: MotionSpec) -> MotionSpec:
    """Move markers and labels whose coordinates are off by more than about 1 degree to the geocoded place."""
    from ..mg import geo
    mp = MapPayload(**spec.payload)
    moved: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for item, text in [(m, m.label) for m in mp.markers] + [(lb, lb.text) for lb in mp.labels]:
        hit = geo.lookup(text.title() if text.isupper() else text) if text else None
        if hit is None:
            continue
        old = (item.lat, item.lon)
        if geo.distance_deg(old, (hit.lat, hit.lon)) > MAP_TOLERANCE_DEG:
            moved.append((old, (hit.lat, hit.lon)))
            item.lat, item.lon = hit.lat, hit.lon
    mp.markers = [m for m in mp.markers if -90 <= m.lat <= 90 and -180 <= m.lon <= 180]
    if not moved:
        return spec.model_copy(update={"payload": mp.model_dump()})
    for r in mp.routes:
        for p in r.points:
            for old, new in moved:
                if geo.distance_deg((p[0], p[1]), old) < 0.3:
                    p[0], p[1] = new
    pts = [(m.lat, m.lon) for m in mp.markers] + [(p[0], p[1]) for r in mp.routes for p in r.points]
    delta = (mp.zoom_end - mp.zoom) if mp.zoom_end is not None else None
    mp.center, mp.zoom = b.map_view(pts)
    mp.zoom_end = round(mp.zoom + delta, 2) if delta is not None else None
    return spec.model_copy(update={"payload": mp.model_dump()})


def check_chart(spec: MotionSpec, research: Optional[Research]) -> Optional[MotionSpec]:
    """With research, every charted value must appear in a fact card; fills a missing source_note."""
    chart = ChartPayload(**spec.payload)
    facts = research.facts if research else []
    values = [v for s in chart.series for v in s.values]
    if not values:
        return None
    if not facts:
        return spec if spec.source_note else spec.model_copy(update={"source_note": "Illustrative"})
    cited = [I.number_is_cited(v, facts) for v in values]
    if not all(cited):
        return None
    if spec.source_note:
        return spec
    return spec.model_copy(update={"source_note": b.source_note_for(cited[0].source_title)})


def merge_ideas(llm: list[VisualIdea], bcs: list[I.BeatContext], research: Optional[Research], palette: str,
                rng: random.Random) -> list[VisualIdea]:
    """Model ideas first; heuristic ideas fill every missing family and stay as lower priority reserves."""
    out = list(llm)
    have = {(i.beat_id, i.family) for i in llm}
    for bc in bcs:
        for idea in I.ideas_for_beat(bc, research, palette, rng):
            if (bc.beat.id, idea.family) in have:
                idea.priority = round(idea.priority * 0.5, 3)
            out.append(idea)
    return out


# ---------------------------------------------------------------- structural anchors

def item_title(title: str) -> str:
    """'Number 3: The Hoover Dam' -> 'The Hoover Dam'."""
    t = re.sub(r"^\s*(#\s*\d+|no\.?\s*\d+|number\s+\w+|\d+)\s*[:.)\-]\s*", "", title, flags=re.I).strip()
    return t or title.strip()


def structural_anchors(script: Script, spans: dict[str, tuple[float, float]], profile: StyleProfile,
                       top10: bool, total: float, topic: str) -> list[Anchor]:
    """Top 10: number card and item lower third per item, recap grid in the outro. Documentary: main title."""
    mg = profile.pacing.motion_graphic
    pal = profile.palette
    out: list[Anchor] = []
    starts = {sc.id: spans[sc.beats[0].id][0] for sc in script.scenes if sc.beats and sc.beats[0].id in spans}
    ends = {sc.id: spans[sc.beats[-1].id][1] for sc in script.scenes if sc.beats and sc.beats[-1].id in spans}
    if top10:
        items = [sc for sc in script.scenes if sc.purpose == ScenePurpose.item or sc.item_number is not None]
        for k, sc in enumerate(items):
            if sc.id not in starts:
                continue
            n = sc.item_number if sc.item_number is not None else len(items) - k
            title = item_title(sc.title)
            nc = min(mg.max, max(mg.min, mg.mean * 0.8))
            s0, length = starts[sc.id], ends[sc.id] - starts[sc.id]
            out.append(Anchor(start=s0, duration=nc, type=ShotType.motion_graphic, tag="number_card",
                              motion=b.number_card_spec(f"#{n}", label=title, palette=pal)))
            fo = profile.pacing.footage_overlay.mean
            if length >= nc + fo + 1.5:
                out.append(Anchor(start=s0 + nc, duration=fo, type=ShotType.footage_overlay, tag="item_lower_third",
                                  overlay=b.lower_third_spec(title, f"Number {n}", palette=pal)))
        out += _recap_anchor(script, starts, [item_title(sc.title) for sc in items], profile, total)
        return out
    setup = next((sc for sc in script.scenes if sc.purpose == ScenePurpose.setup and sc.id in starts), None)
    if setup is not None and starts[setup.id] >= 5.0:
        main, _, sub = (script.title or topic).partition(":")
        out.append(Anchor(start=starts[setup.id], duration=mg.min, type=ShotType.motion_graphic, tag="title_card",
                          motion=b.title_card_spec(main.strip() or topic, sub.strip(), palette=pal)))
    return out


def _recap_anchor(script: Script, starts: dict[str, float], items: list[str], profile: StyleProfile,
                  total: float) -> list[Anchor]:
    if not items:
        return []
    mg = profile.pacing.motion_graphic
    dur = min(mg.max, max(mg.min, 3.0 + 0.8 * len(items)))
    outro = next((sc for sc in script.scenes if sc.purpose == ScenePurpose.outro and sc.id in starts), None)
    start = starts[outro.id] if outro else max(0.0, total - dur - 1.0)
    return [Anchor(start=start, duration=dur, type=ShotType.motion_graphic, tag="recap",
                   motion=b.recap_grid_spec(items, title="The Full Countdown", palette=profile.palette))]


# ---------------------------------------------------------------- layer two: slot to shot

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

    def pick(self, beat_id: str, family: str, at: float, prev: Optional[int], duration: float = 0.0,
             exclude: tuple[int, ...] = ()) -> int:
        """Best idea of a family near a beat: priority minus reuse, distance and context penalties."""
        hook = at < self.profile.pacing.hook_seconds
        bpos = self.pos.get(beat_id, 0)
        scene = self.bc[beat_id].scene.id if beat_id in self.bc else ""
        scene_start = self.spans.get(self.bc[beat_id].scene.beats[0].id, (0, 0))[0] if beat_id in self.bc else 0.0
        best, best_score = -1, -1e9
        for d in (0, -1, 1, -2, 2, -3, 3):
            j = bpos + d
            if not 0 <= j < len(self.order):
                continue
            bid = self.order[j]
            for i in self.by_key.get((bid, family), []):
                idea = self.ideas[i]
                reuse = max(self.usage.get(i, 0), self.sig_usage.get(signature(idea), 0))
                score = idea.priority - 0.9 * reuse - 0.25 * abs(d)
                score -= 0.8 if self.bc[bid].scene.id != scene else 0.0
                score -= 5.0 if i == prev or i in exclude else 0.0
                if hook and family == "footage" and not idea.opening_safe:
                    score -= 0.5
                if idea.motion is not None:
                    score -= self._motion_penalty(idea.motion, at, scene_start, duration)
                if score > best_score:
                    best, best_score = i, score
        if best < 0:
            best = self._fallback_idea(beat_id, family)
        self._count(best, 1)
        return best

    def _count(self, idx: int, delta: int) -> None:
        sig = signature(self.ideas[idx])
        self.usage[idx] = max(0, self.usage.get(idx, 0) + delta)
        self.sig_usage[sig] = max(0, self.sig_usage.get(sig, 0) + delta)

    def _motion_penalty(self, spec: MotionSpec, at: float, scene_start: float, duration: float) -> float:
        p = 1.0 if spec.kind in self.avoid_kind else 0.0
        if spec.kind == MotionKind.title_card and at - scene_start > 8.0:
            p += 1.0
        if spec.kind in (MotionKind.title_card, MotionKind.text_overlay, MotionKind.callout) and duration > 8.0:
            p += 0.5
        if spec.kind == MotionKind.number_card and duration > 9.0:
            p += 0.3
        if spec.kind == MotionKind.map and duration and duration < 6.0:
            p += 0.3
        return p

    def _fallback_idea(self, beat_id: str, family: str) -> int:
        bc = self.bc[beat_id]
        made = I.ideas_for_beat(bc, None, self.profile.palette, random.Random(beat_id))
        idea = next(x for x in made if x.family == family)
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
            shot.overlay = slot.overlay or shot.overlay or self.overlay_for(shot)
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

    def overlay_for(self, shot: Shot) -> MotionSpec:
        """Overlay for an overlay shot: its idea's overlay, else a default from the beat."""
        idx = self.idea_of.get(shot.id)
        if idx is not None and self.ideas[idx].overlay is not None:
            return self.ideas[idx].overlay
        return I.default_overlay(self.bc[shot.beat_id], self.profile.palette)


def signature(idea: VisualIdea) -> str:
    """Content signature: ideas with the same family and intent look the same on screen."""
    return f"{idea.family}:{re.sub(r'[^a-z0-9]+', ' ', idea.intent.lower()).strip()}"


def _anchor_intent(slot: Slot) -> str:
    p = slot.motion.payload if slot.motion else {}
    label = p.get("title") or p.get("label") or ""
    return no_em_dash(f"{slot.tag.replace('_', ' ').capitalize()}: {p.get('number', '')} {label}".replace("  ", " "))


def ai_prompt(idea: VisualIdea, bc: Optional[I.BeatContext]) -> str:
    """Photographic, era-correct image prompt from the idea's intent and must_show."""
    era = bc.era if bc else None
    if era and era < 1935:
        style = "Black and white archival documentary photograph"
    elif era and era < 1975:
        style = "Vintage color documentary photograph on period film stock"
    else:
        style = "Cinematic documentary photograph"
    when = f", {I.era_label(era)}" if era else ""
    subject = re.sub(r"^(archival footage|footage|historical photograph|documentary photograph|wide establishing "
                     r"archival footage|wide establishing footage|close detail photograph|portrait photograph)\s+of\s+",
                     "", idea.intent.rstrip("."), flags=re.I)
    subject = re.sub(r",?\s*(in the )?\d{4}s$", "", subject)
    shows = f" Clearly showing {', '.join(idea.must_show[:4])}." if idea.must_show else ""
    guard = " Period-accurate clothing, vehicles and architecture." if era and era < 1990 else ""
    return no_em_dash(f"{style} of {subject}{when}.{shows}{guard} Natural light, realistic detail, wide 16:9 "
                      "composition, no text, no captions, no watermark, no logos.")


# ---------------------------------------------------------------- finalize

def _finalize(sl: ShotList, profile: StyleProfile) -> None:
    """Ids, indices and times; every spec retimed to its shot and painted with the profile palette."""
    for i, s in enumerate(sl.shots):
        s.id = f"shot_{i + 1:03d}"
        s.duration = round(s.duration, 3)
    drift = round(sl.duration - sum(s.duration for s in sl.shots), 3)
    if sl.shots and drift:
        sl.shots[-1].duration = round(sl.shots[-1].duration + drift, 3)
    total = sl.duration
    sl.reindex()
    sl.duration = total
    if sl.shots:
        sl.shots[-1].end = total
    for s in sl.shots:
        if s.type == ShotType.motion_graphic and s.motion is not None:
            spec = b.with_palette(b.retime_spec(s.motion, s.duration), profile.palette)
            s.motion = spec if spec.background != "transparent" else spec.model_copy(update={"background": "solid"})
        else:
            s.motion = None
        if s.type in OVERLAY_TYPES and s.overlay is not None:
            spec = b.with_palette(b.retime_spec(s.overlay, s.duration), profile.palette)
            s.overlay = spec.model_copy(update={"background": "transparent"})
        elif s.type not in OVERLAY_TYPES:
            s.overlay = None
