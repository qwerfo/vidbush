"""Sound plan: what plays under the narration, when, at what level, and why.

Four layers sit under the voice, each planned from what the story and the picture say:

  music     planned per scene from the scene's stated purpose (work/plan_v2.json scene_purpose, else the
            script purpose). A change at a scene boundary is an equal power crossfade of 1 to 2 s (length
            chosen by the incoming mood) moved onto the outgoing track's phrase boundary when the track has
            a clear pulse. Inside a scene the music changes only at a quiet, where it has already fallen away.
  quiet     on a clause the planner labelled quiet the music fades out over QUIET_FADE_S before the clause
            and returns after the line has landed. Ambience, or room tone when no place is shown, holds
            underneath so the track never goes dead.
  ambience  a very low bed under real footage and stills of a place, chosen from the shot description,
            held steady across the shots of one place and eased at a change of place. None under full
            frame layouts unless the same place continues on both sides.
  effects   one soft family of sounds on the things the house style animates (a map line drawn, a number
            landing, a card settling, a document mark sweeping), timed to the graphic's word timings, off
            for any graphic whose moment is quiet.

Levels are dB relative to the narration loudness (LUFS), so they hold whatever the final normalisation does.
The plan is pure data (no audio); vidbush/audio/mixer.py renders it and sound_qa.py measures the result.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..models import Narration, Script, ShotList

QUIET_FADE_S = 1.5  # music fades out over this before a quiet clause
LAND_S = 0.6  # after the quiet line ends, the music waits this long before returning
RETURN_FADE_S = 2.0
OPEN_GAP_S = 3.0  # a speech pause at least this long lets the music rise to its open level
MIN_SECTION_S = 20.0
MAX_SECTION_S = 300.0
MAX_SNAP_S = 4.0  # a scene change crossfade moves at most this far to reach a phrase boundary
MIN_PULSE_CONFIDENCE = 0.3
XFADE_BY_MOOD = {"calm": 2.0, "mysterious": 2.0, "uplifting": 1.5, "tense": 1.0}
PLACE_EASE_S = 1.5  # ambience crossfade at a change of place
BRIDGE_S = 8.0  # a gap of non place shots this short between shots of one place keeps the bed
MIN_PLACE_S = 8.0  # a place shown for less than this gets no bed of its own (a bed per 4 s shot flickers)
EFFECT_SPACING_S = 3.0
QUIET_GUARD_S = 1.0  # a graphic this close to a quiet clause gets no effect

# ---------------------------------------------------------------- mood from a stated purpose

MOOD_WORDS: dict[str, tuple[str, ...]] = {
    "tense": ("desperat", "danger", "threat", "fear", "urgent", "race ", "struggl", "harsh", "deadly", "crisis",
              "frostbit", "surviv", "risk", "tension", "dread", "storm", "blizzard", "dying", "death", "epidemic",
              "against the clock", "stakes", "isolat", "cut off", "trapped", "hardship"),
    "mysterious": ("myster", "puzzl", "guess", "question", "secret", "strange", "unknown", "hidden", "curious",
                   "riddle", "surpris", "odd", "trick", "counterintuit", "reveal"),
    "uplifting": ("triumph", "pride", "proud", "joy", "celebrat", "payoff", "victor", "huge", "scale", "awe",
                  "arriv", "hope", "rescue", "achiev", "funny", "humor", "humour", "playful", "delight", "amaz",
                  "wonder", "grand", "vast", "record", "first game", "saved", "heroic", "win "),
    "calm": ("reflect", "gentle", "everyday", "ordinary", "explain", "settle", "calm", "home", "peace", "routine",
             "rest", "daily life", "community", "close", "warm", "farewell", "summar", "recap", "subscribe",
             "call to action", "quiet"),
}
MOOD_ENERGY = {"tense": 0.6, "mysterious": 0.4, "uplifting": 0.65, "calm": 0.3}
# script purposes (the enum every script has) when no purpose text is available
PURPOSE_MOOD = {"hook": "uplifting", "setup": "calm", "act": "mysterious", "item": "uplifting", "turn": "tense",
                "climax": "uplifting", "resolution": "calm", "outro": "calm"}
TURN_RE = re.compile(r"\b(then|until|but|before|lifts? into|gives way to|turns? to|and finally)\b", re.I)


def mood_of(text: str, fallback: str = "calm") -> tuple[str, float, list[str]]:
    """(mood, energy, matched words) for a stated purpose; the mood whose words appear most wins, ties go to
    the earlier mood in MOOD_WORDS order (tense, mysterious, uplifting, calm)."""
    low = f" {text.lower()} "
    hits = {m: [w.strip() for w in words if w in low] for m, words in MOOD_WORDS.items()}
    best = max(MOOD_WORDS, key=lambda m: len(hits[m]))
    if not hits[best]:
        return fallback, MOOD_ENERGY.get(fallback, 0.4), []
    energy = MOOD_ENERGY[best] + min(0.15, 0.05 * (len(hits[best]) - 1))
    return best, round(energy, 2), hits[best]


def turn_text(text: str) -> str:
    """The part of a purpose or note after its turn word ("... then lifts into arrival" -> "arrival"), or ""."""
    m = None
    for m in TURN_RE.finditer(text or ""):
        pass
    return text[m.end():].strip(" ,.;:") if m else ""


# ---------------------------------------------------------------- inputs

@dataclass
class SceneInfo:
    id: str
    start: float
    end: float  # last word end
    purpose: str  # stated purpose text, or the script purpose name
    script_purpose: str
    mood: str = "calm"
    energy: float = 0.4
    words: list[str] = field(default_factory=list)


@dataclass
class QuietSpan:
    start: float
    end: float
    scene: str
    clause: str = ""
    note: str = ""


def load_plan_v2(work_dir: Path) -> dict[str, Any]:
    p = Path(work_dir) / "plan_v2.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def scene_infos(script: Script, narration: Narration, plan: dict[str, Any]) -> list[SceneInfo]:
    """Each scene's time span (from the narration beat spans) and its stated purpose and mood."""
    stated = {s.get("scene"): str(s.get("scene_purpose") or "") for s in plan.get("scenes", []) or []}
    out = []
    for sc in script.scenes:
        spans = [narration.beat_spans[b.id] for b in sc.beats if b.id in narration.beat_spans]
        if not spans:
            continue
        sp = sc.purpose.value
        text = stated.get(sc.id) or ""
        mood, energy, words = mood_of(text, PURPOSE_MOOD.get(sp, "calm")) if text else \
            (PURPOSE_MOOD.get(sp, "calm"), MOOD_ENERGY[PURPOSE_MOOD.get(sp, "calm")], [])
        out.append(SceneInfo(sc.id, min(s[0] for s in spans), max(s[1] for s in spans), text or sp, sp,
                             mood, energy, words))
    out.sort(key=lambda s: s.start)
    return out


def quiet_spans(shotlist: ShotList, plan: dict[str, Any]) -> list[QuietSpan]:
    """Quiet clauses: shots the planner labelled quiet (plan_v2.json grammar, or Shot.flags), merged."""
    spans: list[QuietSpan] = []
    for sc in plan.get("scenes", []) or []:
        for sh in sc.get("shots", []) or []:
            if "quiet" in (sh.get("grammar") or []):
                spans.append(QuietSpan(float(sh["start"]), float(sh["end"]), sc.get("scene", ""),
                                       ",".join(sh.get("clauses") or []), str(sh.get("grammar_note") or "")))
    if not spans:
        for s in shotlist.shots:
            if "quiet" in s.flags:
                spans.append(QuietSpan(float(s.start), float(s.end), s.scene_id or "", s.id, ""))
    spans.sort(key=lambda q: q.start)
    merged: list[QuietSpan] = []
    for q in spans:
        if merged and q.start - merged[-1].end < QUIET_FADE_S + RETURN_FADE_S:
            merged[-1].end = max(merged[-1].end, q.end)
            merged[-1].note = (merged[-1].note + " " + q.note).strip()
        else:
            merged.append(q)
    return merged


def speech_spans(narration: Narration, gap: float = OPEN_GAP_S) -> list[tuple[float, float]]:
    """Word spans merged across pauses shorter than gap: inside them the music holds one level."""
    out: list[list[float]] = []
    for w in narration.words:
        if out and w.start - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], w.end)
        else:
            out.append([w.start, w.end])
    return [(a, b) for a, b in out]


# ---------------------------------------------------------------- music

@dataclass
class MusicSection:
    track: str  # file name in the music folder
    start: float  # nominal: the crossfade midpoint (or the quiet return) that begins this section
    end: float
    offset: float  # position in the track at start
    mood: str
    xfade_in: float  # equal power crossfade length at start (0: the section starts out of a quiet or at 0)
    reason: str
    after_quiet: bool = False


@dataclass
class MusicChange:
    t: float
    kind: str  # crossfade | quiet_return
    from_track: str
    to_track: str
    xfade_s: float
    snapped: bool
    reason: str


def _choose_track(mood: str, energy: float, tracks: list[dict[str, Any]], prev: Optional[str],
                  used: dict[str, int]) -> dict[str, Any]:
    def score(t: dict[str, Any]) -> float:
        m = 2.0 if t.get("mood") == mood else 0.0
        return m - 2.0 * abs(float(t.get("energy", 0.5)) - energy) - (3.0 if t["file"] == prev else 0.0) \
            - 0.6 * used.get(t["file"], 0)
    return max(tracks, key=score)


def snap_to_phrase(t: float, track: dict[str, Any], section_start: float, offset: float,
                   lo: float, hi: float) -> tuple[float, bool]:
    """Move a change time onto the outgoing track's nearest phrase boundary inside [lo, hi] (at most MAX_SNAP_S
    away). A track with no clear pulse does not allow it: the time is kept."""
    phrase = float(track.get("phrase_s") or 0.0)
    if phrase <= 0 or float(track.get("bpm_confidence", 0.0)) < MIN_PULSE_CONFIDENCE:
        return t, False
    first = float(track.get("first_downbeat_s") or 0.0)
    pos = offset + (t - section_start)
    k = round((pos - first) / phrase)
    best, found = t, False
    for kk in (k - 1, k, k + 1):
        cand = section_start + (first + kk * phrase - offset)
        if lo <= cand <= hi and abs(cand - t) <= MAX_SNAP_S and (not found or abs(cand - t) < abs(best - t)):
            best, found = cand, True
    return round(best, 3), found


def plan_music(scenes: list[SceneInfo], quiets: list[QuietSpan], tracks: list[dict[str, Any]],
               duration: float) -> tuple[list[MusicSection], list[MusicChange]]:
    """Music sections: one per run of scenes that share a mood, changing at scene boundaries (crossfade on a
    phrase) or at a quiet inside a scene whose stated purpose or quiet note turns to another mood."""
    if not tracks or not scenes:
        return [], []
    by_file = {t["file"]: t for t in tracks}
    # candidate change points: (time, mood, energy, kind, reason, window lo, window hi)
    points: list[tuple[float, str, float, str, str, float, float]] = []
    for i, sc in enumerate(scenes):
        if i > 0:
            prev_end = scenes[i - 1].end
            points.append(((prev_end + sc.start) / 2, sc.mood, sc.energy, "crossfade",
                           f"scene {sc.id}: {sc.mood} for '{_short(sc.purpose)}'", prev_end - 3.0, sc.start + 3.0))
        for q in quiets:
            if not (sc.start - 0.5 <= q.start <= sc.end + 0.5):
                continue
            after = turn_text(q.note) or (turn_text(sc.purpose) if sc.purpose else "")
            if not after:
                continue
            mood, energy, words = mood_of(after, sc.mood)
            if words:
                points.append((q.end + LAND_S, mood, energy, "quiet_return",
                               f"scene {sc.id} turns at the quiet: '{_short(after)}' reads {mood}", q.end, q.end))
    points.sort()
    first = scenes[0]
    used: dict[str, int] = {}
    tr = _choose_track(first.mood, first.energy, tracks, None, used)
    used[tr["file"]] = 1
    sections = [MusicSection(tr["file"], 0.0, duration, 0.0, first.mood, 0.0,
                             f"opening scene {first.id}: {first.mood} for '{_short(first.purpose)}'")]
    changes: list[MusicChange] = []
    for t, mood, energy, kind, reason, lo, hi in points:
        cur = sections[-1]
        if kind == "crossfade":
            q = next((q for q in quiets if q.start - QUIET_FADE_S - 2.0 <= t <= q.end + LAND_S + RETURN_FADE_S),
                     None)
            if q is not None:  # a scene change inside a quiet happens at its return, where the music is away
                kind, t = "quiet_return", q.end + LAND_S
                reason += "; at the quiet, where the music is already away"
        min_len = MIN_SECTION_S if kind == "crossfade" else MIN_SECTION_S / 2
        long_enough = t - cur.start >= min_len and duration - t >= MIN_SECTION_S * 0.7
        rotate = t - cur.start >= MAX_SECTION_S
        if not long_enough or (mood == cur.mood and not rotate):
            continue
        new = _choose_track(mood, energy, tracks, cur.track, used)
        if new["file"] == cur.track:
            continue
        snapped = False
        xf = XFADE_BY_MOOD.get(mood, 1.5) if kind == "crossfade" else 0.0
        if kind == "crossfade":
            t, snapped = snap_to_phrase(t, by_file[cur.track], cur.start, cur.offset, lo, hi)
        used[new["file"]] = used.get(new["file"], 0) + 1
        cur.end = round(t, 3)
        off = max(0.0, float(new.get("first_downbeat_s") or 0.0) - xf / 2)
        sections.append(MusicSection(new["file"], round(t, 3), duration, round(off, 3), mood, xf,
                                     reason + ("" if mood != cur.mood else "; same mood, new track for variety"),
                                     after_quiet=kind == "quiet_return"))
        changes.append(MusicChange(round(t, 3), kind, cur.track, new["file"], xf, snapped,
                                   reason + (", on a phrase boundary" if snapped else "")))
    return sections, changes


def music_gain_points(speech: list[tuple[float, float]], duration: float, under_db: float,
                      open_db: float) -> list[tuple[float, float]]:
    """Music level (dB relative to narration) as (time, dB) breakpoints, joined by raised cosine ramps.
    One steady level under speech (speech spans already merge pauses under OPEN_GAP_S); the open level only
    before the first word, after the last and inside long pauses. Every ramp lasts at least 1 s.
    Quiets are a separate multiplier (quiet_gain_points)."""
    if not speech:
        return [(0.0, open_db), (duration, open_db)]
    first = speech[0][0]
    pts = [(0.0, open_db), (first - 1.7, open_db), (first - 0.2, under_db)] if first > 2.5 else [(0.0, under_db)]
    for (a, b), (na, _) in zip(speech, speech[1:]):
        pts += [(b + 0.3, under_db), (b + 1.3, open_db), (na - 1.2, open_db), (na - 0.2, under_db)]
    last = speech[-1][1]
    if duration - last > 3.0:
        pts += [(last + 0.5, under_db), (last + 2.5, open_db)]
    pts.append((duration, pts[-1][1]))
    return [(round(t, 3), v) for t, v in pts]


def quiet_gain_points(quiets: list[QuietSpan], duration: float) -> list[tuple[float, float]]:
    """Music multiplier (0..1) breakpoints: fades to nothing over QUIET_FADE_S ending as the quiet clause
    starts, holds through it, waits LAND_S after it ends, then returns over RETURN_FADE_S."""
    pts = [(0.0, 1.0)]
    for q in quiets:
        down = max(pts[-1][0], q.start - QUIET_FADE_S)
        back = min(duration, q.end + LAND_S)
        pts += [(down, 1.0), (q.start, 0.0), (back, 0.0)]
        if back + RETURN_FADE_S <= duration:
            pts.append((back + RETURN_FADE_S, 1.0))
    pts.append((duration, pts[-1][1]))
    return [(round(t, 3), v) for t, v in pts]


def _short(s: str, n: int = 70) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


# ---------------------------------------------------------------- ambience

# place key -> words in a shot description that show it; checked in this order (most specific first)
PLACE_WORDS: list[tuple[str, tuple[str, ...]]] = [
    ("rain", ("in the rain", "rainy", "rain falling", "downpour", "rainstorm")),
    ("snow_wind", ("blizzard", "sea ice", "snowstorm", "snowfield", "snow field", "ice floe", "frozen sea",
                   "frozen bering", "frozen river", "ice road", "across the ice", "snowy strip", "deep snow",
                   "snow-covered", "snow covered", "dog team", "sled dog", "snowmobile", "frozen tundra", "icebreaker")),
    ("coast", ("harbor", "harbour", "port of", "dock", "pier", "bay", "shore", "coast", "beach", "waves", "ocean",
               "seaside", "fishing boat", "sea ")),
    ("river", ("river bank", "riverbank", "river valley", "braided river", "along the river", "creek", "stream",
               "rapids", "river")),
    ("forest", ("forest", "woods", "spruce", "pine trees", "trees", "trail")),
    ("city_street", ("city street", "downtown", "crosswalk", "manhattan", "skyline", "busy street", "traffic light")),
    ("road", ("highway", "interstate", "gravel road", "two-lane", "road ", "truck", "driving", "windshield",
              "traffic")),
    ("town_street", ("main street", "front street", "small town", "town street", "village street")),
    ("interior_public", ("gym", "hall", "classroom", "church", "courtroom", "cafe", "restaurant", "station",
                         "depot", "platform", "crowd", "market")),
    ("interior_quiet", ("kitchen", "clinic", "office", "interior", "inside", "cabin", "room", "hangar", "desk",
                        "table", "library")),
    ("open_ground", ("tundra", "plain", "prairie", "grassland", "field", "mountain", "valley", "glacier",
                     "landscape", "aerial", "village", "open land", "lakes", "ponds", "wilderness")),
]
HUMAN_PLACES = {"city_street", "road", "town_street", "interior_public"}
NOT_A_PLACE = ("portrait", "newspaper", "headline", "front page", "document", "letter", "census form", "map",
               "atlas", "chart", "diagram", "license plate", "licence plate", "jukebox", "vials", "gauge",
               "dashboard", "flag", "sign listing", "printed", "clipboard", "notepad", "ruler", "scan of")
PERIOD_WORDS = ("archival", "black and white", "historical", "historic", "vintage", "1900s", "1910s", "1920s",
                "1930s", "1940s", "1950s", "1960s", "old photo", "sepia")
PLACE_TYPES = {"footage", "footage_overlay", "image", "image_overlay", "ai_image", "ai_video"}


def place_of(text: str) -> tuple[Optional[str], str]:
    """(place key, reason) a viewer could hear from a shot description, or (None, reason).
    A close view of an object, a document or a map is not a place. A period picture (archival, black and
    white, a past decade) never gets a modern human bed (traffic, a street, a crowd); natural beds stay."""
    low = f" {(text or '').lower()} "
    period = any(w in low for w in PERIOD_WORDS)
    if low.lstrip().startswith(("close", "extreme close", "macro")) and any(_word_re(n).search(low)
                                                                         for n in NOT_A_PLACE):
        return None, "a close view of an object, document or map"
    for key, words in PLACE_WORDS:
        found = next(((w, m) for w in words for m in [_word_re(w).search(low)] if m), None)
        if not found:
            continue
        hit, m = found
        if key in HUMAN_PLACES and period:
            continue
        lead = low[: m.start()]
        if any(n in lead for n in NOT_A_PLACE) and ("close" in lead or "detail" in lead or "view of" in lead):
            return None, f"a close view of an object ('{hit.strip()}' is only its setting)"
        return key, f"'{hit.strip()}' in the shot description"
    if any(n in low for n in NOT_A_PLACE):
        return None, "an object, document or map, not a place"
    return None, "no place a viewer could hear"


def _word_re(w: str) -> "re.Pattern[str]":
    """Whole words only: 'road' must not match 'railroad'."""
    return re.compile(r"\b" + re.escape(w.strip()) + r"\b")


@dataclass
class AmbienceBed:
    place: str
    start: float
    end: float
    shots: list[str]
    reason: str
    ease_in: float = PLACE_EASE_S
    ease_out: float = PLACE_EASE_S


def plan_ambience(shotlist: ShotList, quiets: list[QuietSpan],
                  classify: Callable[[str], tuple[Optional[str], str]] = place_of) -> list[AmbienceBed]:
    """Beds under runs of shots of one place; a full frame layout or a short non place shot between two shots
    of the same place keeps the bed (the place continues behind it)."""
    rows = []
    for s in shotlist.shots:
        kind = s.type.value
        if kind not in PLACE_TYPES:
            rows.append((s, None, "a full frame layout"))
            continue
        place, why = classify(" ".join([s.intent or "", " ".join(s.must_show or [])]))
        rows.append((s, place, why))
    beds: list[AmbienceBed] = []
    for s, place, why in rows:
        if place is None:
            continue
        last = beds[-1] if beds else None
        if last and last.place == place and s.start - last.end <= BRIDGE_S:
            last.end, last.shots = float(s.end), last.shots + [s.id]
            continue
        beds.append(AmbienceBed(place, float(s.start), float(s.end), [s.id], f"{s.id}: {why}"))
    beds = [b for b in beds if b.end - b.start >= MIN_PLACE_S]
    # adjacent beds of different places meet in a crossfade centred on the cut
    for a, b in zip(beds, beds[1:]):
        if b.start - a.end < PLACE_EASE_S:
            mid = (a.end + b.start) / 2
            a.end, b.start = mid, mid
    return beds


def room_tone_spans(quiets: list[QuietSpan], beds: list[AmbienceBed], duration: float) -> list[tuple[float, float]]:
    """Under each quiet: room tone from the start of the music fade to the end of its return, so the track
    never goes dead. It plays under a place bed too: a bed with gaps in it (waves, gusts) keeps a floor."""
    out = []
    for q in quiets:
        a, b = max(0.0, q.start - QUIET_FADE_S - 0.5), min(duration, q.end + LAND_S + RETURN_FADE_S + 0.5)
        out.append((round(a, 3), round(b, 3)))
    return out


# ---------------------------------------------------------------- effects

LAYOUT_EFFECT = {"route_map": "line_draw", "number_over_footage": "number_land", "name_card": "card_settle",
                 "date_title": "card_settle", "equals_card": "card_settle", "split_comparison": "card_settle",
                 "document_focus": "mark_sweep"}
OVERLAY_EFFECT = {"callout": "card_settle", "lower_third": "card_settle", "number_card": "number_land",
                  "title_card": "card_settle"}


@dataclass
class EffectEvent:
    t: float
    kind: str
    shot_id: str
    reason: str


def _layout_events(shot: Any, words: list[Any]) -> list[tuple[float, str, str]]:
    lay = shot.layout
    kind = lay.kind.value if hasattr(lay.kind, "value") else str(lay.kind)
    effect = LAYOUT_EFFECT.get(kind)
    if effect is None or (kind == "document_focus" and getattr(lay, "purpose", None) == "period"):
        return []
    try:
        from ..mg.layouts.timing import cue_times
        cues = cue_times(lay.elements, words, float(shot.start), float(shot.end))
    except Exception:  # timing module missing or a malformed element: the graphic simply gets no sound
        return []
    found = [(c, lay.elements[c.index]) for c in cues if getattr(c, "found", False)]
    if not found:
        return []
    if effect == "line_draw":
        moving = [c for c, el in found if el.role not in ("label", "title")] or [c for c, _ in found]
        return [(float(shot.start) + moving[0].word, effect, f"{kind}: the route line starts drawing")]
    if effect == "mark_sweep":
        return [(float(shot.start) + c.word, effect, f"document mark on '{_short(el.anchor, 30)}'")
                for c, el in found[:3]]
    if effect == "number_land":
        num = [c for c, el in found if el.role in ("number", "value", "figure")] or [c for c, _ in found]
        return [(float(shot.start) + num[0].word + 0.1, effect, f"{kind}: the number lands")]
    return [(float(shot.start) + found[0][0].appear + 0.3, effect, f"{kind}: the card settles")]


def _overlay_events(shot: Any) -> list[tuple[float, str, str]]:
    ov = shot.overlay
    kind = ov.kind.value if hasattr(ov.kind, "value") else str(ov.kind)
    t = float(shot.start) + float(getattr(ov, "start_at", 0.0) or 0.0)
    if kind == "map":
        payload = ov.payload.model_dump() if hasattr(ov.payload, "model_dump") else dict(ov.payload or {})
        if payload.get("routes"):
            return [(t + 0.5, "line_draw", "map overlay: a route line draws")]
        return []
    effect = OVERLAY_EFFECT.get(kind)
    return [(t + 0.3, effect, f"{kind} overlay settles")] if effect else []


def plan_effects(shotlist: ShotList, narration: Narration, quiets: list[QuietSpan]) -> tuple[list[EffectEvent],
                                                                                                 list[dict[str, Any]]]:
    """Effect events and the ones skipped (with why)."""
    raw: list[tuple[float, str, str, str]] = []
    for s in shotlist.shots:
        evs = []
        if getattr(s, "layout", None) is not None:
            evs = _layout_events(s, narration.words)
        elif s.overlay is not None:
            evs = _overlay_events(s)
        elif s.motion is not None and s.type.value == "motion_graphic":
            k = s.motion.kind.value
            if k in OVERLAY_EFFECT:
                evs = [(float(s.start) + 0.3, OVERLAY_EFFECT[k], f"{k} card settles")]
        raw += [(t, k, s.id, r) for t, k, r in evs]
    raw.sort()
    out: list[EffectEvent] = []
    skipped: list[dict[str, Any]] = []
    for t, kind, sid, reason in raw:
        if any(q.start - QUIET_GUARD_S - QUIET_FADE_S <= t <= q.end + LAND_S for q in quiets):
            skipped.append({"t": round(t, 3), "kind": kind, "shot_id": sid, "why": "its moment is quiet"})
            continue
        if out and t - out[-1].t < EFFECT_SPACING_S:
            skipped.append({"t": round(t, 3), "kind": kind, "shot_id": sid,
                            "why": f"within {EFFECT_SPACING_S} s of the previous effect"})
            continue
        out.append(EffectEvent(round(t, 3), kind, sid, reason))
    return out, skipped


def as_dicts(items: list[Any]) -> list[dict[str, Any]]:
    return [asdict(i) for i in items]
