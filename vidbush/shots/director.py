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

import random
import re
from typing import Any, Optional

from ..mg import builders as b
from ..models import (OVERLAY_TYPES, Brand, DirectorPlan, MotionKind, Narration, Project, Research, Scene,
                      ScenePurpose, Script, Shot, ShotList, ShotType, StyleProfile, VideoFormat, VisualIdea)
from ..util import NullLogger
from . import ideas as I
from .assembler import FAMILY, _Assembler, ai_prompt, overlap, prompt_subject, signature  # noqa: F401 re-exported
from .pacing import Anchor, plan_slots
from .seat import (MAP_TOLERANCE_DEG, OVERLAY_KINDS, SYSTEM, check_chart, director_ideas,  # noqa: F401 re-exported
                   director_prompt, fix_map, validate_idea)


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
    shots = apply_rules(shots, ctx.ideas, ctx.idea_of, profile, ctx.rebind, ctx.overlay_for, frozenset(ctx.locked),
                        overlay_pool=ctx.overlay_pool, footage_for=ctx.footage_for)
    ensure_chart(shots, ctx, spans, log)
    sl = ShotList(project_id=project.id, duration=total, shots=shots)
    _finalize(sl, profile)
    log.info("shot list built", shots=len(sl.shots), ideas=len(ctx.ideas), llm_ideas=len(llm),
             duration=sl.duration, top10=top10)
    return sl, DirectorPlan(ideas=ctx.ideas)


CHART_KINDS = (MotionKind.bar_chart, MotionKind.line_chart)


def ensure_chart(shots: list[Shot], ctx: _Assembler, spans: dict[str, tuple[float, float]], log: Any) -> None:
    """Guarantee one data chart when any beat has chartable numbers, shown while that beat is narrated.

    Rebinds a free motion shot over the chart's beat, else turns a long enough free shot over it into one.
    """
    if any(s.motion and s.motion.kind in CHART_KINDS for s in shots):
        return
    charts = [i for i in ctx.ideas if i.family == "motion" and i.motion and i.motion.kind in CHART_KINDS]
    mg_min = ctx.profile.pacing.motion_graphic.min
    best: Optional[tuple[tuple[int, float], Shot, VisualIdea]] = None
    for idea in charts:
        span = spans.get(idea.beat_id, (0.0, 0.0))
        for k, sh in enumerate(shots):
            shared = overlap(span, (sh.start, sh.end))
            if sh.id in ctx.locked or shared < min(2.0, 0.3 * sh.duration):
                continue
            is_mg = sh.type == ShotType.motion_graphic
            if not is_mg and (sh.duration < mg_min or _mg_neighbours(shots, k) >= 1):
                continue
            rank = (1 if is_mg else 0, shared)
            if best is None or rank > best[0]:
                best = (rank, sh, idea)
    if best is None:
        return
    _, shot, idea = best
    if shot.type != ShotType.motion_graphic:
        shot.type = ShotType.motion_graphic
    ctx.rebind(shot, idea)
    log.info("chart guaranteed", shot=shot.id, beat=idea.beat_id)


def _mg_neighbours(shots: list[Shot], k: int) -> int:
    return sum(1 for j in (k - 1, k + 1) if 0 <= j < len(shots) and shots[j].type == ShotType.motion_graphic)


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


# ---------------------------------------------------------------- layer one: ideas

CTA_MOTION_BLOCK = {MotionKind.map, MotionKind.number_card, MotionKind.lower_third, MotionKind.bar_chart,
                    MotionKind.line_chart}


def merge_ideas(llm: list[VisualIdea], bcs: list[I.BeatContext], research: Optional[Research], palette: str,
                rng: random.Random) -> list[VisualIdea]:
    """Model ideas first; heuristic ideas fill every missing family and stay as lower priority reserves.

    On call-to-action beats model overlays are dropped, and so are model maps, charts and number cards.
    """
    cta = {bc.beat.id for bc in bcs if I.CTA_RE.search(bc.beat.text)}
    out: list[VisualIdea] = []
    for idea in llm:
        if idea.beat_id in cta:
            if idea.motion is not None and idea.motion.kind in CTA_MOTION_BLOCK:
                continue
            idea.overlay = None
        out.append(idea)
    have = {(i.beat_id, i.family) for i in out}
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
    """Top 10: number card and item lower third per item, recap grid on the Recap scene (else the outro).
    Documentary: main title at the setup."""
    mg = profile.pacing.motion_graphic
    pal = profile.palette
    out: list[Anchor] = []
    starts = {sc.id: spans[sc.beats[0].id][0] for sc in script.scenes if sc.beats and sc.beats[0].id in spans}
    ends = {sc.id: spans[sc.beats[-1].id][1] for sc in script.scenes if sc.beats and sc.beats[-1].id in spans}
    if top10:
        items = [sc for sc in script.scenes if sc.purpose == ScenePurpose.item or sc.item_number is not None]
        numbers = [sc.item_number if sc.item_number is not None else len(items) - k for k, sc in enumerate(items)]
        for sc, n in zip(items, numbers):
            if sc.id in starts:
                out += _item_anchors(sc, n, starts[sc.id], ends[sc.id] - starts[sc.id], profile)
        out += _recap_anchor(script, starts, ends, items, numbers, profile, total)
        return out
    setup = next((sc for sc in script.scenes if sc.purpose == ScenePurpose.setup and sc.id in starts), None)
    if setup is not None and starts[setup.id] >= 5.0:
        main, _, sub = (script.title or topic).partition(":")
        out.append(Anchor(start=starts[setup.id], duration=mg.min, type=ShotType.motion_graphic, tag="title_card",
                          motion=b.title_card_spec(main.strip() or topic, sub.strip(), palette=pal)))
    return out


def _item_anchors(sc: Scene, n: int, s0: float, length: float, profile: StyleProfile) -> list[Anchor]:
    """Number card pinned to the item start (shortened for short items) and the item lower third when it fits."""
    mg, fo_pace = profile.pacing.motion_graphic, profile.pacing.footage_overlay
    title = item_title(sc.title)
    nc = min(mg.max, max(mg.min, mg.mean * 0.8))
    nc = max(mg.min * 0.6, min(nc, length - 1.0))
    out = [Anchor(start=s0, duration=nc, type=ShotType.motion_graphic, tag="number_card", pinned=True,
                  motion=b.number_card_spec(f"#{n}", label=title, palette=profile.palette))]
    fo = min(fo_pace.mean, length - nc - 1.0)
    if fo >= fo_pace.min:
        out.append(Anchor(start=s0 + nc, duration=fo, type=ShotType.footage_overlay, tag="item_lower_third",
                          overlay=b.lower_third_spec(title, f"Number {n}", palette=profile.palette)))
    return out


def recap_scene(script: Script, items: list[Scene], starts: dict[str, float]) -> Optional[Scene]:
    """The scene the recap grid plays over: a scene titled Recap after the items, else the outro."""
    last = max((script.scenes.index(sc) for sc in items), default=-1)
    later = [sc for sc in script.scenes[last + 1:] if sc.id in starts]
    titled = [sc for sc in later if re.search(r"\brecap\b", sc.title, re.I)]
    titled.sort(key=lambda sc: sc.purpose != ScenePurpose.resolution)
    if titled:
        return titled[0]
    return next((sc for sc in later if sc.purpose == ScenePurpose.outro), None) or next(
        (sc for sc in script.scenes if sc.purpose == ScenePurpose.outro and sc.id in starts), None)


def _recap_anchor(script: Script, starts: dict[str, float], ends: dict[str, float], items: list[Scene],
                  numbers: list[int], profile: StyleProfile, total: float) -> list[Anchor]:
    """Recap grid with item numbers, clamped to its scene's narration and kept inside the timeline."""
    if not items:
        return []
    mg = profile.pacing.motion_graphic
    dur = min(mg.max, max(mg.min, 3.0 + 0.8 * len(items)))
    scene = recap_scene(script, items, starts)
    if scene is not None:
        dur = max(mg.min, min(dur, ends[scene.id] - starts[scene.id]))
        start = starts[scene.id]
    else:
        start = total - dur - 1.0
    start = max(0.0, min(start, total - dur - 0.5))
    spec = b.recap_grid_spec([item_title(sc.title) for sc in items], title="The Full Countdown",
                             palette=profile.palette, numbers=numbers)
    return [Anchor(start=start, duration=dur, type=ShotType.motion_graphic, tag="recap", motion=spec)]


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
