"""Director seat: model visual ideas per beat, one call per scene (or batch of beats), validated.

director_ideas() splits the script into chunks of at most CHUNK_BEATS beats
(scene boundaries first) and sends them one after another, so every call can
see the story so far. Each call carries a compact MotionSpec schema plus the
continuity context: a one paragraph summary of the whole script (written
once by the director seat and cached in work/script_summary.json), the ideas
already chosen for the previous two chunks (intent and type only), the
visual anchors used so far and the narration of the next beat. Only ideas
that validate are kept (pydantic, map coordinates checked against the
gazetteer, chart numbers checked against fact cards). Ideas for beats outside
a chunk are ignored, so a canned mock fixture that holds the whole plan still
works chunk by chunk. A chunk (or the summary) that falls back to heuristics
while a real director provider is configured is reported through
seats.note_fallback so QA shows it and the project ends needs_review.
"""
from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from ..mg import builders as b
from ..models import (Brand, ChartPayload, MapPayload, MotionKind, MotionSpec, Research, Scene, Script, StyleProfile,
                      VisualIdea)
from ..util import no_em_dash, read_json, stable_hash, write_json
from . import ideas as I

CHUNK_BEATS = 8
SEAT_TIMEOUT_S = 600.0
MAP_TOLERANCE_DEG = 1.0
PREVIOUS_CHUNKS = 2
SUMMARY_FILE = "script_summary.json"
SUMMARY_MAX_WORDS = 160
OVERLAY_KINDS = {MotionKind.lower_third, MotionKind.callout, MotionKind.text_overlay, MotionKind.highlight_box,
                 MotionKind.arrow, MotionKind.map, MotionKind.bar_chart, MotionKind.line_chart}

SUMMARY_SYSTEM = (
    "You summarise documentary scripts for the visual director. Reply with strict JSON only: "
    '{"summary": "..."}. One paragraph of at most 120 words: the story arc from start to end, the main '
    "places, people, objects and periods, and where the story moves between places. No em dashes."
)

# Visual rules sent with every director call, stated as principles with the reason behind each.
DIRECTOR_RULES = [
    "For every beat give exactly 1 footage idea, 1 still idea and at most 1 motion idea.",
    "footage and still ideas: intent (what is on screen, no purpose clauses), must_show, must_avoid, 2 to 4 "
    "search queries, most specific first, era correct. Optional overlay: a MotionSpec of an overlay kind, only "
    "naming or drawing something the narration of that beat supports.",
    "Animation moves over the picture: reference channels show most maps, charts, callouts and arrows as "
    "overlays on footage or stills, and use full screen cards sparingly (at most one per "
    "full_screen_card_spacing_s seconds, the opening included, each on screen a few seconds as its words are "
    "spoken), because a run of cards reads as a slideshow and loses the viewer. Prefer a map or chart as an "
    "overlay, and a name, number or short phrase as a lower third or callout on the picture; keep motion ideas "
    "for the moments that need the whole frame.",
    "Keep a visual subject on screen across consecutive beats about it rather than cutting to something new "
    "every beat: the viewer follows one thing while the narration develops it. Read previous_ideas and "
    "visual_anchors_so_far and continue the subjects the story is still on.",
    "Vary framing between consecutive shots (wide, medium, close detail) and say it in the intent: two shots "
    "in a row with the same framing read as a jump cut, while a change of framing on the same subject reads "
    "as continuity.",
    "In the first 60 seconds prefer footage with real activity (people working, machines, water, traffic): "
    "movement holds attention while the hook is made. opening_safe true only for footage with real activity, "
    "people or movement.",
    "Use a map when the narration moves between places, so the viewer can follow the geography; a beat that "
    "only names one place gets a lower third, not a map.",
    "Use a chart only when two or more comparable numbers appear in the beat or its facts, and only with "
    "numbers from the facts with a source_note: a single number is a number card or a callout, not a chart.",
    "next_beat is the narration that follows this chunk: plan the last beat's visuals so the cut into it is "
    "natural (same subject, or a matching place or movement).",
    "No maps, lower thirds or number cards on subscribe or call to action beats.",
    "No em dashes. Concise strings. Use payload keys exactly as in motion_spec_schema.",
]


@dataclass
class DirectorContext:
    """What one director call knows beyond its own beats: the story, recent choices and what comes next."""

    summary: str = ""
    previous: list[dict[str, str]] = field(default_factory=list)
    anchors: list[str] = field(default_factory=list)
    next_beat: Optional[dict[str, str]] = None

SYSTEM = (
    "You are the director of a long-form faceless YouTube documentary. For every narration beat propose "
    "concrete visual ideas a researcher can find or build. Reply with strict JSON only, no prose, no markdown. "
    "Keep every string short and concrete."
)

# Compact MotionSpec schema shown to the model: exact payload keys per kind, duration optional.
MOTION_SCHEMA: dict[str, Any] = {
    "spec": {"kind": "<kind>", "payload": {}, "source_note": "required for map and charts",
             "duration": "optional seconds, omit to fit the shot"},
    "kinds": {
        "map": {"center": [9.1, -79.7], "zoom": 6, "title": "Colon to Panama City",
                "markers": [{"lat": 9.36, "lon": -79.9, "label": "Colon"}],
                "routes": [{"points": [[9.36, -79.9], [8.97, -79.54]], "label": ""}],
                "shaded_regions": ["Panama"], "_note": "real lat/lon; zoom 1 world, +1 doubles scale"},
        "bar_chart": {"title": "Distance by sea", "labels": ["Before", "After"],
                      "series": [{"name": "miles", "values": [13000, 5200]}], "units": "miles",
                      "_note": "only numbers that appear in the facts"},
        "line_chart": {"title": "...", "labels": ["1900", "1910", "1920"], "series": [{"values": [1, 2, 3]}]},
        "number_card": {"number": "5,600", "label": "workers died", "sublabel": "1904 to 1914"},
        "title_card": {"title": "Chapter title", "subtitle": ""},
        "lower_third": {"title": "Name or place", "subtitle": "role or country"},
        "callout": {"title": "Short key phrase", "subtitle": ""},
        "text_overlay": {"title": "1914", "subtitle": "", "position": "upper_left"},
        "highlight_box": {"x": 0.3, "y": 0.3, "w": 0.4, "h": 0.3, "label": "Gatun Locks"},
        "arrow": {"x": 0.2, "y": 0.3, "to_x": 0.5, "to_y": 0.5, "label": ""},
    },
    "motion_kinds": ["map", "bar_chart", "line_chart", "number_card", "title_card", "text_overlay", "callout"],
    "overlay_kinds": ["map", "bar_chart", "line_chart", "lower_third", "callout", "text_overlay", "highlight_box",
                      "arrow"],
    "_overlay_note": "a map or chart overlay is drawn as an inset panel over the footage or still",
}


# ---------------------------------------------------------------- chunks

def chunk_beats(script: Script, size: int = CHUNK_BEATS) -> list[list[tuple[Scene, Any]]]:
    """(scene, beat) chunks: one per scene, long scenes split into batches of about size beats."""
    out: list[list[tuple[Scene, Any]]] = []
    for sc in script.scenes:
        beats = list(sc.beats)
        n = max(1, -(-len(beats) // size))
        step = -(-len(beats) // n) if beats else 0
        for k in range(0, len(beats), max(1, step)):
            out.append([(sc, bt) for bt in beats[k:k + step]])
    return [c for c in out if c]


# ---------------------------------------------------------------- calls

def director_ideas(script: Script, spans: dict[str, tuple[float, float]], research: Optional[Research],
                   profile: StyleProfile, brand: Brand, seats: Any, log: Any,
                   work_dir: Optional[Path] = None) -> list[VisualIdea]:
    """Model ideas for every chunk, in order, each call carrying the continuity context.

    Chunks run one after another because each call needs the ideas chosen before it. A failed chunk keeps
    heuristic ideas for its beats. When every chunk fails with a ProviderError (real providers configured
    and all down) the error is raised so the stage fails and resume can retry.
    """
    if seats is None:
        return []
    chunks = chunk_beats(script)
    if not chunks:
        return []
    summary = script_summary(script, seats, log, work_dir)
    history: list[list[VisualIdea]] = []
    errors: list[Exception] = []
    out: list[VisualIdea] = []
    for k, chunk in enumerate(chunks):
        context = DirectorContext(summary=summary, previous=previous_ideas(history[-PREVIOUS_CHUNKS:]),
                                  anchors=anchors_so_far(out, research), next_beat=next_beat(chunks, k))
        try:
            ideas = _chunk_ideas(chunk, script, spans, research, profile, brand, seats, log, context)
        except Exception as e:  # noqa: BLE001 - one failed chunk keeps heuristic ideas for its beats
            log.warn("director seat failed for a chunk, using heuristics", beats=len(chunk), error=str(e)[:300])
            errors.append(e)
            ideas = []
            _note_chunk(seats, chunk, f"director call failed: {type(e).__name__}: {str(e)[:160]}")
        else:
            if not ideas:
                _note_chunk(seats, chunk, "director answer had no usable ideas")
        history.append(ideas)
        out += ideas
    if errors and len(errors) == len(chunks) and all(_is_provider_error(e) for e in errors):
        raise errors[0]
    log.info("director ideas validated", chunks=len(chunks), failed=len(errors), kept=len(out))
    return out


def _note_chunk(seats: Any, chunk: list[tuple[Scene, Any]], reason: str) -> None:
    """Report a heuristic fallback for every scene of a chunk (only counted when the seat is real)."""
    for scene_id in dict.fromkeys(sc.id for sc, _bt in chunk):
        note_fallback(seats, "director", "shotlist", scene_id, reason)


def note_fallback(seats: Any, role: str, stage: str, scene_id: str, reason: str) -> None:
    """seats.note_fallback when the seat object has it (test doubles may not)."""
    fn = getattr(seats, "note_fallback", None)
    if callable(fn):
        fn(role, stage, scene_id, reason)


def next_beat(chunks: list[list[tuple[Scene, Any]]], k: int) -> Optional[dict[str, str]]:
    """The first beat after chunk k (id, scene and narration), or None at the end of the script."""
    if k + 1 >= len(chunks) or not chunks[k + 1]:
        return None
    sc, bt = chunks[k + 1][0]
    return {"id": bt.id, "scene_title": sc.title, "text": bt.text}


def previous_ideas(chunks: list[list[VisualIdea]]) -> list[dict[str, str]]:
    """Intent and type of ideas already chosen, oldest first: type is the family, or the motion kind."""
    out = []
    for ideas in chunks:
        for idea in ideas:
            kind = idea.motion.kind.value if idea.family == "motion" and idea.motion else idea.family
            item = {"beat_id": idea.beat_id, "type": kind, "intent": idea.intent[:140]}
            if idea.overlay is not None:
                item["overlay"] = idea.overlay.kind.value
            out.append(item)
    return out


def _idea_text(idea: VisualIdea) -> str:
    parts = [idea.intent, *idea.must_show, *idea.queries]
    for spec in (idea.motion, idea.overlay):
        if spec is not None:
            parts.append(json.dumps(spec.payload, ensure_ascii=False))
    return " ".join(parts).lower()


def anchors_so_far(ideas: list[VisualIdea], research: Optional[Research], limit: int = 16) -> list[str]:
    """Visual anchors already on screen, in first-use order: research anchors the ideas mention, else the
    must_show subjects of the ideas (most recent last)."""
    if research and research.anchors:
        blob = " ".join(_idea_text(i) for i in ideas)
        named = [a.name for a in research.anchors if a.name and re.search(rf"\b{re.escape(a.name.lower())}\b", blob)]
        return named[:limit]
    seen: dict[str, None] = {}
    for idea in ideas:
        for m in idea.must_show:
            key = m.strip()
            if key:
                seen.pop(key, None)
                seen[key] = None
    return list(seen)[-limit:]


# ---------------------------------------------------------------- script summary

def script_summary(script: Script, seats: Any, log: Any, work_dir: Optional[Path] = None) -> str:
    """One paragraph summary of the whole script, written once by the director seat and cached in
    work/script_summary.json (keyed by a hash of the script). Falls back to the scene openings."""
    key = stable_hash(script)
    cache = Path(work_dir) / SUMMARY_FILE if work_dir else None
    if cache is not None and cache.exists():
        try:
            data = read_json(cache)
            if data.get("script_hash") == key and data.get("summary"):
                return str(data["summary"])
        except (OSError, ValueError, AttributeError):
            pass
    text, source = _ask_summary(script, seats, log), "director"
    if not text:
        text, source = heuristic_summary(script), "heuristic"
        note_fallback(seats, "director", "shotlist", "all", "script summary fell back to scene openings")
    if cache is not None and cache.parent.exists():
        write_json(cache, {"script_hash": key, "source": source, "summary": text})
    return text


def _ask_summary(script: Script, seats: Any, log: Any) -> str:
    body = {"title": script.title, "format": script.format.value,
            "scenes": [{"title": sc.title, "purpose": sc.purpose.value, "text": " ".join(b.text for b in sc.beats)}
                       for sc in script.scenes]}
    try:
        raw = seats.llm_json("director", "script_summary", SUMMARY_SYSTEM, json.dumps(body, ensure_ascii=False),
                             "shotlist")
    except Exception as e:  # noqa: BLE001 - the summary is context, never a reason to fail the stage
        log.warn("script summary failed, using scene openings", error=str(e)[:300])
        return ""
    text = raw.get("summary") if isinstance(raw, dict) else raw if isinstance(raw, str) else ""
    return _clip_words(no_em_dash(str(text or "")).strip(), SUMMARY_MAX_WORDS)


def heuristic_summary(script: Script) -> str:
    """Title plus the first sentence of every scene, clipped to SUMMARY_MAX_WORDS."""
    from ..script.parser import split_sentences
    parts = [script.title.strip().rstrip(".") + "." if script.title else ""]
    for sc in script.scenes:
        if sc.beats:
            first = split_sentences(sc.beats[0].text)
            parts.append(first[0] if first else sc.beats[0].text)
    return _clip_words(no_em_dash(" ".join(p for p in parts if p)), SUMMARY_MAX_WORDS)


def _clip_words(text: str, n: int) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n]).rstrip(",;") + "."


def _is_provider_error(e: Exception) -> bool:
    try:
        from ..providers.base import ProviderError
    except ImportError:
        return False
    return isinstance(e, ProviderError)


def _chunk_ideas(chunk: list[tuple[Scene, Any]], script: Script, spans: dict[str, tuple[float, float]],
                 research: Optional[Research], profile: StyleProfile, brand: Brand, seats: Any,
                 log: Any, context: Optional[DirectorContext] = None) -> list[VisualIdea]:
    user = director_prompt(script, spans, research, profile, brand, chunk, context)
    kwargs: dict[str, Any] = {}
    if "timeout" in _params(seats.llm_json):
        kwargs["timeout"] = SEAT_TIMEOUT_S
    raw = seats.llm_json("director", "director", SYSTEM, user, "shotlist", **kwargs)
    items = raw.get("ideas") if isinstance(raw, dict) else raw if isinstance(raw, list) else None
    if not isinstance(items, list):
        return []
    beat_ids = {bt.id for _sc, bt in chunk}
    lengths = {bid: max(0.5, spans.get(bid, (0.0, 6.0))[1] - spans.get(bid, (0.0, 6.0))[0]) for bid in beat_ids}
    mine = [it for it in items if isinstance(it, dict) and it.get("beat_id") in beat_ids]
    return [idea for idea in (validate_idea(it, beat_ids, research, lengths) for it in mine) if idea]


def _params(fn: Any) -> set[str]:
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


def director_prompt(script: Script, spans: dict[str, tuple[float, float]], research: Optional[Research],
                    profile: StyleProfile, brand: Brand, chunk: Optional[list[tuple[Scene, Any]]] = None,
                    context: Optional[DirectorContext] = None) -> str:
    """User message for one director call: the chunk's beats, the continuity context (summary, previous
    ideas, anchors so far, next beat), facts they cite, research anchors, schema and rules."""
    pairs = chunk if chunk is not None else [(sc, bt) for sc in script.scenes for bt in sc.beats]
    ctx = context or DirectorContext()
    beats = [{"id": bt.id, "scene": sc.id, "scene_title": sc.title, "purpose": sc.purpose.value, "text": bt.text,
              "start": round(spans.get(bt.id, (0, 0))[0], 2), "end": round(spans.get(bt.id, (0, 0))[1], 2),
              "facts": bt.facts_used} for sc, bt in pairs]
    cited = {f for _sc, bt in pairs for f in bt.facts_used}
    facts = [{"id": f.id, "claim": f.claim[:240], "numbers": f.numbers, "source": f.source_title, "date": f.date}
             for f in (research.facts if research else []) if f.id in cited or not cited]
    anchors = [a.model_dump(exclude_defaults=True) for a in (research.anchors if research else [])]
    brief = {
        "title": script.title, "format": script.format.value, "channel_notes": brand.niche_notes,
        "script_summary": ctx.summary, "previous_ideas": ctx.previous, "visual_anchors_so_far": ctx.anchors,
        "beats": beats, "next_beat": ctx.next_beat, "facts": facts[:40], "anchors": anchors[:12],
        "full_screen_card_spacing_s": profile.pacing.card_spacing_s, "motion_spec_schema": MOTION_SCHEMA,
        "rules": DIRECTOR_RULES,
        "output": {"ideas": [{"beat_id": "b001", "family": "footage|still|motion", "intent": "...",
                              "must_show": [], "must_avoid": [], "queries": [],
                              "overlay": {"kind": "lower_third", "payload": {"title": "Colon", "subtitle": "Panama"}},
                              "motion": None, "priority": 1.0, "opening_safe": True}]},
    }
    return json.dumps(brief, ensure_ascii=False)


# ---------------------------------------------------------------- validation

def _with_duration(spec: Any, seconds: float) -> Any:
    """A model spec without duration gets the beat length (retimed to the shot later)."""
    if isinstance(spec, dict) and spec.get("duration") in (None, "", 0):
        return {**spec, "duration": round(seconds, 2)}
    return spec


def _overlay(raw: Any, research: Optional[Research] = None) -> Optional[MotionSpec]:
    """A valid overlay spec; a map or chart overlay gets the motion checks and a transparent background
    (it renders as an inset over the picture)."""
    try:
        spec = MotionSpec.model_validate(raw)
    except (ValidationError, TypeError, ValueError):
        return None
    if spec.kind not in OVERLAY_KINDS or not _payload_ok(spec):
        return None
    if spec.kind == MotionKind.map:
        mp = MapPayload(**spec.payload)
        if not (mp.markers or mp.routes or mp.arrows):  # an inset map must show where, or it is decoration
            return None
        spec = fix_map(spec)
    if spec.kind in (MotionKind.bar_chart, MotionKind.line_chart):
        spec = check_chart(spec, research)
    return spec.model_copy(update={"background": "transparent"}) if spec is not None else None


def validate_idea(item: Any, beat_ids: set[str], research: Optional[Research],
                  lengths: Optional[dict[str, float]] = None) -> Optional[VisualIdea]:
    """Pydantic validation plus content checks; None when the idea is unusable.

    A bad overlay only clears the overlay; a missing spec duration takes the beat length.
    """
    if not isinstance(item, dict):
        return None
    seconds = (lengths or {}).get(str(item.get("beat_id")), 6.0)
    raw_overlay = _with_duration(item.get("overlay"), seconds)
    body = {**item, "overlay": None, "motion": _with_duration(item.get("motion"), seconds)}
    try:
        idea = VisualIdea.model_validate(body)
    except (ValidationError, TypeError, ValueError):
        return None
    if idea.beat_id not in beat_ids:
        return None
    idea.overlay = _overlay(raw_overlay, research) if raw_overlay else None
    idea = _clean_text(idea)
    if idea.family != "motion":
        idea.motion = None
        return idea
    if idea.motion is None or not _payload_ok(idea.motion):
        return None
    idea.overlay = None
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
