"""Script writer: outline plus research to a beat-level Script.

Short scripts are written in one writer seat call (task "script"). Long
scripts are written one section per call; each call sees the whole outline
(titles and summaries), the facts earlier sections already used and the
previous section's last beat, so acts do not repeat facts or re-introduce
the topic, and a 60 minute script works. Every prompt carries the brand
language. Model output is cleaned (no em dashes, no stage directions),
over-long beats are re-split, and ids are renumbered: scenes s01.., beats
b001.. in order. When the seat returns nothing usable for a section, beats
are assembled from the fact claims.

Real mode (seats.is_real("writer")): a script under MIN_LENGTH_RATIO of the
outline's target words is retried once (whole script, or each short
section, with a "too short" note); still short raises ScriptTooShort so the
stage fails. A writer that fails on every real provider raises ProviderError.
A section that still ships from fact-claim heuristics is reported through
seats.note_fallback with its scene id, so QA lists it and the project ends
needs_review instead of shipping it silently.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..models import (Beat, Brand, Brief, Emphasis, FactCard, Outline, OutlineSection, Research, Scene,
                      ScenePurpose, Script, VideoFormat)
from ..providers.base import ProviderError
from ..util import no_em_dash
from . import prompts
from .parser import BEAT_MAX_WORDS, clean_narration, make_beat_texts, split_sentences

SINGLE_CALL_MAX_WORDS = 1800  # about 12 minutes; longer scripts are written per section
SINGLE_CALL_MAX_SECTIONS = 9
MIN_LENGTH_RATIO = 0.5


class ScriptTooShort(RuntimeError):
    """A real writer produced under MIN_LENGTH_RATIO of the target words after one retry."""


def write_script(outline: Outline, research: Research, brief: Brief, brand: Brand, seats: Any) -> Script:
    """Write the narration script. Falls back per section on unusable answers; see the module contract."""
    total = sum(s.target_words for s in outline.sections)
    facts = {f.id: f for f in research.facts if f.status != "contradicted"}
    whole = total <= SINGLE_CALL_MAX_WORDS and len(outline.sections) <= SINGLE_CALL_MAX_SECTIONS
    real = _is_real(seats)
    heuristic: set[int] = set()
    title, scenes = _draft(outline, facts, brief, brand, seats, whole, "", heuristic)
    if real and script_words(scenes) < MIN_LENGTH_RATIO * total:
        note = prompts.TOO_SHORT.format(got=script_words(scenes), want=total)
        if whole:
            heuristic.clear()
            title, scenes = _draft(outline, facts, brief, brand, seats, True, note, heuristic)
        else:
            scenes = _retry_short_sections(outline, scenes, facts, brief, brand, seats, heuristic)
        if script_words(scenes) < MIN_LENGTH_RATIO * total:
            raise ScriptTooShort(f"script has {script_words(scenes)} words for a target of {total} "
                                 f"(under {int(MIN_LENGTH_RATIO * 100)} percent after one retry)")
    for i in sorted(heuristic):
        _note_fallback(seats, f"s{i + 1:02d}", "writer answer for this section was missing or unusable; "
                                               "beats were assembled from fact claims")
    return renumber(Script(title=no_em_dash(title or outline.title or brief.topic).strip(), format=outline.format,
                           scenes=scenes))


def script_words(scenes: list[Scene]) -> int:
    return sum(len(b.text.split()) for sc in scenes for b in sc.beats)


def _note_fallback(seats: Any, scene_id: str, reason: str) -> None:
    fn = getattr(seats, "note_fallback", None)
    if callable(fn):
        fn("writer", "script", scene_id, reason)


def _is_real(seats: Any) -> bool:
    try:
        return bool(seats.is_real("writer"))
    except AttributeError:
        return False


def _draft(outline: Outline, facts: dict[str, FactCard], brief: Brief, brand: Brand, seats: Any, whole: bool,
           note: str, heuristic: Optional[set[int]] = None) -> tuple[str, list[Scene]]:
    """One pass over the outline: (title, scenes). Indices of sections built from heuristics go in heuristic."""
    title = outline.title or brief.topic
    raw_scenes: list[Optional[dict]] = [None] * len(outline.sections)
    if whole:
        data = _ask(seats, brief, brand, outline, outline.sections, facts, "", note)
        title = _title(data) or title
        raw_scenes = _match_scenes(data, outline.sections, 0)
    scenes: list[Scene] = []
    for i, sec in enumerate(outline.sections):
        raw = raw_scenes[i]
        if raw is None and not whole:
            raw = _ask_section(seats, brief, brand, outline, i, facts, scenes, note)
        scenes.append(_scene(sec, raw, facts, brief, i, outline))
        if heuristic is not None and (raw is None or not _beats_from_raw(raw, facts)):
            heuristic.add(i)
    return title, scenes


def _ask_section(seats: Any, brief: Brief, brand: Brand, outline: Outline, i: int, facts: dict[str, FactCard],
                 previous: list[Scene], note: str) -> Optional[dict]:
    """Per-section request with the whole outline, the facts used so far and the last beat."""
    last = previous[-1].beats[-1].text if previous and previous[-1].beats else ""
    used = sorted({f for sc in previous for b in sc.beats for f in b.facts_used})
    context = prompts.FULL_OUTLINE.format(
        outline="\n".join(f"{'>> ' if k == i else ''}{s.id} {s.title}: {s.summary}"
                          for k, s in enumerate(outline.sections)),
        used=", ".join(used) or "none")
    extra = "\n".join(x for x in (context, note) if x)
    data = _ask(seats, brief, brand, outline, [outline.sections[i]], facts, last, extra)
    return _match_scenes(data, [outline.sections[i]], i)[0]


def _scene(sec: OutlineSection, raw: Optional[dict], facts: dict[str, FactCard], brief: Brief, i: int,
           outline: Outline) -> Scene:
    beats = _beats_from_raw(raw, facts) if raw is not None else []
    if not beats:
        beats = heuristic_beats(sec, facts, brief, i, len(outline.sections))
    name = no_em_dash(str(raw.get("title") or sec.title)).strip() if raw is not None else sec.title
    return Scene(id="", title=name, purpose=sec.purpose, beats=beats,
                 item_number=sec.item_number if outline.format == VideoFormat.top10 else None)


def _retry_short_sections(outline: Outline, scenes: list[Scene], facts: dict[str, FactCard], brief: Brief,
                          brand: Brand, seats: Any, heuristic: Optional[set[int]] = None) -> list[Scene]:
    """Rewrite, once, every section under MIN_LENGTH_RATIO of its own target."""
    out = list(scenes)
    for i, sec in enumerate(outline.sections):
        got = sum(len(b.text.split()) for b in out[i].beats)
        if got >= MIN_LENGTH_RATIO * sec.target_words:
            continue
        note = prompts.TOO_SHORT.format(got=got, want=sec.target_words)
        raw = _ask_section(seats, brief, brand, outline, i, facts, out[:i], note)
        if raw is not None:
            out[i] = _scene(sec, raw, facts, brief, i, outline)
            if heuristic is not None and _beats_from_raw(raw, facts):
                heuristic.discard(i)
    return out


def renumber(script: Script) -> Script:
    """Scene ids s01.., beat ids b001.. globally unique and in order."""
    n = 0
    for i, sc in enumerate(script.scenes):
        sc.id = f"s{i + 1:02d}"
        for b in sc.beats:
            n += 1
            b.id = f"b{n:03d}"
    return script


# ---------------------------------------------------------------- seat

def _ask(seats: Any, brief: Brief, brand: Brand, outline: Outline, sections: list[OutlineSection],
         facts: dict[str, FactCard], last_beat: str, note: str = "") -> Optional[Any]:
    wanted = {f for s in sections for f in s.fact_ids}
    fact_lines = [f"{fid}: {facts[fid].claim}" for fid in facts if fid in wanted or len(sections) > 1]
    sec_lines = []
    for s in sections:
        item = f", number {s.item_number}" if s.item_number else ""
        sec_lines.append(prompts.SECTION_LINE.format(id=s.id, purpose=s.purpose.value, item=item, title=s.title,
                                                     summary=s.summary, words=s.target_words,
                                                     fact_ids=", ".join(s.fact_ids) or "none"))
    continuity = prompts.CONTINUITY.format(last=last_beat) if last_beat else ""
    continuity = "\n".join(x for x in (continuity, note) if x)
    user = prompts.SCRIPT_USER.format(topic=brief.topic, angle=brief.angle or "none",
                                      audience=brief.audience or "older US viewers", notes=brand.niche_notes or "none",
                                      format=outline.format.value, sections="\n".join(sec_lines),
                                      language=prompts.language_line(brand.language),
                                      facts="\n".join(fact_lines[:150]) or "none", continuity=continuity)
    try:
        return seats.llm_json("writer", "script", prompts.SCRIPT_SYSTEM, user, stage="script")
    except ProviderError:
        raise  # every real provider failed: the stage fails, resume retries
    except Exception:  # a malformed answer falls back to the heuristic writer
        return None


def _title(data: Any) -> str:
    return str(data.get("title", "")).strip() if isinstance(data, dict) else ""


def _match_scenes(data: Any, sections: list[OutlineSection], offset: int) -> list[Optional[dict]]:
    """Map a seat answer onto sections. Accepts {"scenes": [...]} or {"beats": [...]} for one section.

    A full-script answer to a per-section request (a canned fixture) is matched by
    section position, so offset is the index of the first section in the whole outline.
    """
    out: list[Optional[dict]] = [None] * len(sections)
    if not isinstance(data, dict):
        return out
    if isinstance(data.get("beats"), list) and len(sections) == 1:
        return [data]
    scenes = data.get("scenes")
    if not isinstance(scenes, list):
        return out
    for k in range(len(sections)):
        idx = k + offset if len(scenes) > len(sections) else k
        if idx < len(scenes) and isinstance(scenes[idx], dict):
            out[k] = scenes[idx]
    return out


def _beats_from_raw(raw: dict, facts: dict[str, FactCard]) -> list[Beat]:
    """Clean seat beats: strip em dashes and directions, re-split long beats, keep known fact ids."""
    out: list[Beat] = []
    for b in raw.get("beats") or []:
        if isinstance(b, str):
            b = {"text": b}
        if not isinstance(b, dict):
            continue
        text = clean_narration(no_em_dash(str(b.get("text", ""))))
        if not text:
            continue
        used = [f for f in b.get("facts_used") or [] if f in facts]
        emph = _emphasis(b.get("emphasis"))
        sentences = split_sentences(text)
        parts = [text] if len(sentences) <= 3 and len(text.split()) <= BEAT_MAX_WORDS else make_beat_texts(sentences)
        for k, part in enumerate(parts):
            out.append(Beat(id="", text=part, facts_used=used, emphasis=emph if k == 0 else Emphasis.normal))
    return out


def _emphasis(value: Any) -> Emphasis:
    try:
        return Emphasis(str(value).lower())
    except ValueError:
        return Emphasis.normal


# ---------------------------------------------------------------- heuristic

CONNECTIVES = ["", "Then, ", "At the same time, ", "", "Even so, ", "Meanwhile, ", "", "In the end, "]
LOWER_OK = {"The", "A", "An", "It", "In", "By", "For", "At", "On", "This", "That", "These", "Those", "There",
            "Its", "Many", "Most", "Some", "Each", "Every", "One", "Over", "After", "Before", "During", "When"}


def heuristic_beats(sec: OutlineSection, facts: dict[str, FactCard], brief: Brief, index: int,
                    count: int) -> list[Beat]:
    """Assemble beats from the section's fact claims with simple connective sentences."""
    claims = [(fid, _sentence(facts[fid].claim)) for fid in sec.fact_ids
              if fid in facts and facts[fid].source_title != "brief"]
    lines: list[tuple[str, list[str]]] = []
    for k, (fid, claim) in enumerate(claims):
        lines.append((_connect(claim, CONNECTIVES[k % len(CONNECTIVES)] if k else ""), [fid]))
    opener = _opener(sec, brief)
    if opener and sec.purpose == ScenePurpose.hook:
        lines.append((opener, []))
    elif opener:
        lines.insert(0, (opener, []))
    closer = _closer(sec, brief, index, count)
    if closer:
        lines.append((closer, []))
    if not lines and sec.summary and sec.purpose != ScenePurpose.outro:
        lines.append((_sentence(sec.summary), []))
    if not lines:
        lines.append((_sentence(sec.title), []))
    beats = []
    for k, (text, used) in enumerate(lines):
        emph = Emphasis.high if k == 0 and sec.purpose in (ScenePurpose.hook, ScenePurpose.climax) else Emphasis.normal
        if sec.purpose == ScenePurpose.outro and "subscribe" in text.lower():
            emph = Emphasis.low
        beats.append(Beat(id="", text=no_em_dash(text), facts_used=used, emphasis=emph))
    return beats


def _mid(topic: str) -> str:
    """Topic as written mid sentence: "The Panama Canal" becomes "the Panama Canal"."""
    return "the" + topic[3:] if topic.startswith("The ") else topic


def _opener(sec: OutlineSection, brief: Brief) -> str:
    topic = _mid(brief.topic)
    if sec.purpose == ScenePurpose.hook:
        angle = f" It is a story of {brief.angle.rstrip('.')}." if brief.angle else ""
        return f"This is the story of {topic}.{angle}"
    if sec.purpose == ScenePurpose.item and sec.item_number:
        named = sec.title and sec.title != f"Number {sec.item_number}"
        return f"Number {sec.item_number}. {sec.title}." if named else f"Number {sec.item_number}."
    if sec.purpose == ScenePurpose.turn:
        return "And then, everything changed."
    if sec.purpose == ScenePurpose.climax:
        return "Now came the hardest part of all."
    if sec.purpose == ScenePurpose.resolution and brief.format == VideoFormat.top10:
        return "Let us look back at the list."
    return ""


def _closer(sec: OutlineSection, brief: Brief, index: int, count: int) -> str:
    if sec.purpose == ScenePurpose.outro or index == count - 1:
        return f"That is the story of {_mid(brief.topic)}. If you enjoyed it, subscribe for more stories like this one."
    return ""


def _sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", no_em_dash(text)).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text[:1].upper() + text[1:]


def _connect(claim: str, connective: str) -> str:
    if not connective:
        return claim
    first = claim.split(" ", 1)[0]
    body = claim[0].lower() + claim[1:] if first in LOWER_OK else claim
    return connective + body
