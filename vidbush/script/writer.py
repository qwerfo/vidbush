"""Script writer: outline plus research to a beat-level Script.

Short scripts are written in one writer seat call (task "script"). Long
scripts are written one section per call, passing the previous section's last
beat for continuity, so a 60 minute script works. Model output is cleaned
(no em dashes, no stage directions), over-long beats are re-split, and ids
are renumbered: scenes s01.., beats b001.. in order. When the seat returns
nothing usable for a section, beats are assembled from the fact claims.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..models import (Beat, Brand, Brief, Emphasis, FactCard, Outline, OutlineSection, Research, Scene,
                      ScenePurpose, Script, VideoFormat)
from ..util import no_em_dash
from . import prompts
from .parser import BEAT_MAX_WORDS, clean_narration, make_beat_texts, split_sentences

SINGLE_CALL_MAX_WORDS = 1800  # about 12 minutes; longer scripts are written per section
SINGLE_CALL_MAX_SECTIONS = 9


def write_script(outline: Outline, research: Research, brief: Brief, brand: Brand, seats: Any) -> Script:
    """Write the narration script. Never raises on seat failures; falls back per section."""
    total = sum(s.target_words for s in outline.sections)
    facts = {f.id: f for f in research.facts if f.status != "contradicted"}
    whole = total <= SINGLE_CALL_MAX_WORDS and len(outline.sections) <= SINGLE_CALL_MAX_SECTIONS
    title = outline.title or brief.topic
    raw_scenes: list[Optional[dict]] = [None] * len(outline.sections)
    if whole:
        data = _ask(seats, brief, brand, outline, outline.sections, facts, "")
        title = _title(data) or title
        raw_scenes = _match_scenes(data, outline.sections, 0)
    scenes: list[Scene] = []
    for i, sec in enumerate(outline.sections):
        raw = raw_scenes[i]
        if raw is None and not whole:
            last = scenes[-1].beats[-1].text if scenes and scenes[-1].beats else ""
            data = _ask(seats, brief, brand, outline, [sec], facts, last)
            raw = _match_scenes(data, [sec], i)[0]
        beats = _beats_from_raw(raw, facts) if raw is not None else []
        if not beats:
            beats = heuristic_beats(sec, facts, brief, i, len(outline.sections))
        name = no_em_dash(str(raw.get("title") or sec.title)).strip() if raw is not None else sec.title
        scenes.append(Scene(id="", title=name, purpose=sec.purpose, beats=beats,
                            item_number=sec.item_number if outline.format == VideoFormat.top10 else None))
    return renumber(Script(title=no_em_dash(title).strip(), format=outline.format, scenes=scenes))


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
         facts: dict[str, FactCard], last_beat: str) -> Optional[Any]:
    wanted = {f for s in sections for f in s.fact_ids}
    fact_lines = [f"{fid}: {facts[fid].claim}" for fid in facts if fid in wanted or len(sections) > 1]
    sec_lines = []
    for s in sections:
        item = f", number {s.item_number}" if s.item_number else ""
        sec_lines.append(prompts.SECTION_LINE.format(id=s.id, purpose=s.purpose.value, item=item, title=s.title,
                                                     summary=s.summary, words=s.target_words,
                                                     fact_ids=", ".join(s.fact_ids) or "none"))
    continuity = prompts.CONTINUITY.format(last=last_beat) if last_beat else ""
    user = prompts.SCRIPT_USER.format(topic=brief.topic, angle=brief.angle or "none",
                                      audience=brief.audience or "older US viewers", notes=brand.niche_notes or "none",
                                      format=outline.format.value, sections="\n".join(sec_lines),
                                      facts="\n".join(fact_lines[:150]) or "none", continuity=continuity)
    try:
        return seats.llm_json("writer", "script", prompts.SCRIPT_SYSTEM, user, stage="script")
    except Exception:  # seat failures fall back to the heuristic writer
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
