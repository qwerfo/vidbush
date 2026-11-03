"""Director seat: model visual ideas per beat, one call per scene (or batch of beats), validated.

director_ideas() splits the script into chunks of at most CHUNK_BEATS beats
(scene boundaries first), sends the chunks in parallel with a compact
MotionSpec schema in the prompt, and keeps only ideas that validate
(pydantic, map coordinates checked against the gazetteer, chart numbers
checked against fact cards). Ideas for beats outside a chunk are ignored, so
a canned mock fixture that holds the whole plan still works chunk by chunk.
"""
from __future__ import annotations

import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from pydantic import ValidationError

from ..mg import builders as b
from ..models import (Brand, ChartPayload, MapPayload, MotionKind, MotionSpec, Research, Scene, Script, StyleProfile,
                      VisualIdea)
from ..util import no_em_dash
from . import ideas as I

CHUNK_BEATS = 8
SEAT_WORKERS = 4
SEAT_TIMEOUT_S = 600.0
MAP_TOLERANCE_DEG = 1.0
OVERLAY_KINDS = {MotionKind.lower_third, MotionKind.callout, MotionKind.text_overlay, MotionKind.highlight_box,
                 MotionKind.arrow}

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
    "overlay_kinds": ["lower_third", "callout", "text_overlay", "highlight_box", "arrow"],
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
                   profile: StyleProfile, brand: Brand, seats: Any, log: Any) -> list[VisualIdea]:
    """Model ideas for every chunk, in parallel; failed chunks fall back to heuristics.

    When every chunk fails with a ProviderError (real providers configured and all down) the error is
    raised so the stage fails and resume can retry.
    """
    if seats is None:
        return []
    chunks = chunk_beats(script)
    if not chunks:
        return []

    def one(chunk: list[tuple[Scene, Any]]) -> tuple[list[VisualIdea], Optional[Exception]]:
        try:
            return _chunk_ideas(chunk, script, spans, research, profile, brand, seats, log), None
        except Exception as e:  # noqa: BLE001 - one failed chunk keeps heuristic ideas for its beats
            log.warn("director seat failed for a chunk, using heuristics", beats=len(chunk), error=str(e)[:300])
            return [], e

    with ThreadPoolExecutor(max_workers=max(1, min(SEAT_WORKERS, len(chunks)))) as pool:
        results = list(pool.map(one, chunks))
    errors = [e for _ideas, e in results if e is not None]
    if errors and len(errors) == len(chunks) and all(_is_provider_error(e) for e in errors):
        raise errors[0]
    out = [idea for ideas, _e in results for idea in ideas]
    log.info("director ideas validated", chunks=len(chunks), failed=len(errors), kept=len(out))
    return out


def _is_provider_error(e: Exception) -> bool:
    try:
        from ..providers.base import ProviderError
    except ImportError:
        return False
    return isinstance(e, ProviderError)


def _chunk_ideas(chunk: list[tuple[Scene, Any]], script: Script, spans: dict[str, tuple[float, float]],
                 research: Optional[Research], profile: StyleProfile, brand: Brand, seats: Any,
                 log: Any) -> list[VisualIdea]:
    user = director_prompt(script, spans, research, profile, brand, chunk)
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
                    profile: StyleProfile, brand: Brand, chunk: Optional[list[tuple[Scene, Any]]] = None) -> str:
    """User message for one director call: the chunk's beats, facts they cite, anchors, schema and rules."""
    pairs = chunk if chunk is not None else [(sc, bt) for sc in script.scenes for bt in sc.beats]
    beats = [{"id": bt.id, "scene": sc.id, "scene_title": sc.title, "purpose": sc.purpose.value, "text": bt.text,
              "start": round(spans.get(bt.id, (0, 0))[0], 2), "end": round(spans.get(bt.id, (0, 0))[1], 2),
              "facts": bt.facts_used} for sc, bt in pairs]
    cited = {f for _sc, bt in pairs for f in bt.facts_used}
    facts = [{"id": f.id, "claim": f.claim[:240], "numbers": f.numbers, "source": f.source_title, "date": f.date}
             for f in (research.facts if research else []) if f.id in cited or not cited]
    anchors = [a.model_dump(exclude_defaults=True) for a in (research.anchors if research else [])]
    brief = {
        "title": script.title, "format": script.format.value, "channel_notes": brand.niche_notes,
        "beats": beats, "facts": facts[:40], "anchors": anchors[:12], "motion_spec_schema": MOTION_SCHEMA,
        "rules": [
            "For every beat give exactly 1 footage idea, 1 still idea and at most 1 motion idea.",
            "footage and still ideas: intent (what is on screen, no purpose clauses), must_show, must_avoid, "
            "2 to 4 search queries, most specific first, era correct. Optional overlay: a MotionSpec of an "
            "overlay kind, only naming something the shot shows.",
            "motion ideas carry motion: a MotionSpec of a motion kind, built only from that beat's narration. "
            "Charts only use numbers from the facts and carry source_note.",
            "No maps, lower thirds or number cards on subscribe or call to action beats.",
            "opening_safe true only for footage with real activity, people or movement.",
            "No em dashes. Concise strings. Use payload keys exactly as in motion_spec_schema.",
        ],
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


def _overlay(raw: Any) -> Optional[MotionSpec]:
    try:
        spec = MotionSpec.model_validate(raw)
    except (ValidationError, TypeError, ValueError):
        return None
    return spec if spec.kind in OVERLAY_KINDS and _payload_ok(spec) else None


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
    idea.overlay = _overlay(raw_overlay) if raw_overlay else None
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
