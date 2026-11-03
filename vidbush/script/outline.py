"""Outline builder: documentary or top10 skeleton sized at about 150 words per minute.

make_outline asks the writer seat (task "outline", in brand.language) and
validates the answer. When the seat returns nothing usable, a deterministic
skeleton is built and the research facts are distributed over it in
chronological order. A real seat that fails on every provider raises
ProviderError so the stage fails and resume can retry it.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..models import Brand, Brief, FactCard, Outline, OutlineSection, Research, ScenePurpose, VideoFormat
from ..providers.base import ProviderError
from ..util import no_em_dash
from . import prompts

WORDS_PER_MINUTE = 150

# (purpose, share of total words, default title)
DOC_HEAD = [(ScenePurpose.hook, 0.07, "The Hook"), (ScenePurpose.setup, 0.12, "Background")]
DOC_TAIL = [(ScenePurpose.climax, 0.15, "The Climax"), (ScenePurpose.resolution, 0.08, "Aftermath"),
            (ScenePurpose.outro, 0.05, "Legacy")]


def target_words(minutes: float) -> int:
    """Total narration words for a target length."""
    return max(60, int(round(minutes * WORDS_PER_MINUTE)))


def act_count(minutes: float) -> int:
    """Three acts up to 8 minutes, four up to 20, five beyond."""
    return 3 if minutes <= 8 else 4 if minutes <= 20 else 5


def make_outline(brief: Brief, research: Research, brand: Brand, seats: Any) -> Outline:
    """Outline from the writer seat, validated and resized; heuristic skeleton on failure."""
    words = target_words(brief.target_minutes)
    data = _ask_seat(brief, research, brand, seats, words)
    outline = _from_seat(data, brief, research, words) if data is not None else None
    if outline is None:
        outline = heuristic_outline(brief, research)
    return outline


def _ask_seat(brief: Brief, research: Research, brand: Brand, seats: Any, words: int) -> Optional[Any]:
    if brief.format == VideoFormat.top10:
        structure = prompts.TOP10_STRUCTURE
    else:
        structure = prompts.DOC_STRUCTURE.format(acts=act_count(brief.target_minutes))
    facts = "\n".join(f"{f.id}: {f.claim}" for f in research.facts[:120])
    user = prompts.OUTLINE_USER.format(format=brief.format.value, topic=brief.topic, angle=brief.angle or "none",
                                       audience=brief.audience or "general", minutes=brief.target_minutes,
                                       words=words, notes=brand.niche_notes or "none", structure=structure,
                                       facts=facts or "none", language=prompts.language_line(brand.language))
    try:
        return seats.llm_json("writer", "outline", prompts.OUTLINE_SYSTEM, user, stage="outline")
    except ProviderError:
        raise  # every real provider failed: the stage fails, resume retries
    except Exception:  # a malformed answer falls back to the heuristic skeleton
        return None


# ---------------------------------------------------------------- validation

def _from_seat(data: Any, brief: Brief, research: Research, words: int) -> Optional[Outline]:
    """Validate a seat answer: known purposes, known fact ids, sane structure, resized to target."""
    raw = data.get("sections") if isinstance(data, dict) else None
    if not isinstance(raw, list) or len(raw) < 3:
        return None
    known = {f.id for f in research.facts}
    sections: list[OutlineSection] = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict) or not str(r.get("title", "")).strip():
            return None
        sections.append(OutlineSection(
            id=f"sec{i + 1:02d}", title=no_em_dash(str(r["title"])).strip(), purpose=_purpose(r.get("purpose")),
            summary=no_em_dash(str(r.get("summary", ""))).strip(),
            target_words=max(20, _int(r.get("target_words"), 0)),
            fact_ids=[f for f in r.get("fact_ids") or [] if f in known],
            item_number=_int(r.get("item_number"), None),
        ))
    _fix_structure(sections, brief.format)
    resize(sections, words)
    title = no_em_dash(str(data.get("title") or brief.topic)).strip()
    return Outline(format=brief.format, title=title, sections=sections)


def _purpose(value: Any) -> ScenePurpose:
    try:
        return ScenePurpose(str(value).strip().lower())
    except ValueError:
        return ScenePurpose.act


def _int(value: Any, default: Optional[int]) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fix_structure(sections: list[OutlineSection], fmt: VideoFormat) -> None:
    """First section is the hook, last the outro; top10 items count down 10 to 1."""
    sections[0].purpose = ScenePurpose.hook
    sections[-1].purpose = ScenePurpose.outro
    if fmt != VideoFormat.top10:
        for s in sections:
            s.item_number = None
            if s.purpose == ScenePurpose.item:
                s.purpose = ScenePurpose.act
        return
    items = [s for s in sections if s.purpose == ScenePurpose.item or s.item_number is not None]
    for k, s in enumerate(items):
        s.purpose = ScenePurpose.item
        s.item_number = len(items) - k


def resize(sections: list[OutlineSection], words: int) -> None:
    """Scale target_words so the total matches words, keeping proportions."""
    total = sum(s.target_words for s in sections) or 1
    for s in sections:
        s.target_words = max(20, int(round(s.target_words * words / total)))


# ---------------------------------------------------------------- heuristic

def heuristic_outline(brief: Brief, research: Research) -> Outline:
    """Deterministic skeleton with facts distributed in chronological order."""
    words = target_words(brief.target_minutes)
    if brief.format == VideoFormat.top10:
        skeleton = _top10_skeleton()
    else:
        skeleton = _doc_skeleton(brief.target_minutes)
    sections = []
    for i, (purpose, share, title, item) in enumerate(skeleton):
        sections.append(OutlineSection(id=f"sec{i + 1:02d}", title=title, purpose=purpose, summary="",
                                       target_words=max(20, int(round(words * share))), item_number=item))
    resize(sections, words)
    _distribute_facts(sections, research)
    _name_sections(sections, research, brief)
    return Outline(format=brief.format, title=brief.topic, sections=sections)


def _doc_skeleton(minutes: float) -> list[tuple[ScenePurpose, float, str, Optional[int]]]:
    n = act_count(minutes)
    tail = DOC_TAIL if minutes >= 6 else [DOC_TAIL[0], (ScenePurpose.outro, 0.10, "Legacy")]
    body_share = 1.0 - sum(s for _, s, _ in DOC_HEAD) - sum(s for _, s, _ in tail)
    turn_at = max(1, round(n * 0.55)) - 1 if n > 3 else 1
    acts = [(ScenePurpose.turn if k == turn_at else ScenePurpose.act, body_share / n, f"Part {k + 1}")
            for k in range(n)]
    return [(p, s, t, None) for p, s, t in DOC_HEAD + acts + tail]


def _top10_skeleton() -> list[tuple[ScenePurpose, float, str, Optional[int]]]:
    out: list[tuple[ScenePurpose, float, str, Optional[int]]] = [(ScenePurpose.hook, 0.06, "Introduction", None)]
    out += [(ScenePurpose.item, 0.083, f"Number {n}", n) for n in range(10, 0, -1)]
    out += [(ScenePurpose.resolution, 0.05, "Recap", None), (ScenePurpose.outro, 0.03, "Final Thoughts", None)]
    return out


def fact_year(fact: FactCard) -> float:
    """Sort key: first four digit year in the date field or claim, else infinity."""
    m = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", fact.date) or re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", fact.claim)
    return float(m.group(1)) if m else float("inf")


def _distribute_facts(sections: list[OutlineSection], research: Research) -> None:
    """Hook gets the most striking fact; body sections get chronological slices weighted by words."""
    facts = [f for f in research.facts if f.status != "contradicted"]
    if not facts:
        return
    striking = max(facts, key=lambda f: (len(f.numbers), f.confidence))
    sections[0].fact_ids = [striking.id]
    body = [s for s in sections if s.purpose not in (ScenePurpose.hook, ScenePurpose.outro)] or sections
    ordered = sorted(facts, key=lambda f: (fact_year(f), f.id))
    total = sum(s.target_words for s in body) or 1
    pos = 0
    for k, s in enumerate(body):
        take = len(ordered) - pos if k == len(body) - 1 else round(len(ordered) * s.target_words / total)
        s.fact_ids = [f.id for f in ordered[pos:pos + take]]
        pos += take
    outro = [s for s in sections if s.purpose == ScenePurpose.outro]
    if outro and ordered:
        outro[-1].fact_ids = [ordered[-1].id]


def _name_sections(sections: list[OutlineSection], research: Research, brief: Brief) -> None:
    """Give body sections a title from their first fact and a speakable summary."""
    from .parser import title_from_text
    by_id = {f.id: f for f in research.facts}
    used = {brief.topic.lower()}
    for s in sections:
        claims = [by_id[f].claim for f in s.fact_ids if f in by_id and by_id[f].source_title != "brief"]
        if s.purpose in (ScenePurpose.act, ScenePurpose.turn, ScenePurpose.climax) and claims:
            name = title_from_text(" ".join(claims), exclude=used)
            if name:
                s.title = name
                used.add(name.lower())
        if s.purpose == ScenePurpose.item and claims:
            s.title = title_from_text(claims[0], exclude=used) or s.title
            used.add(s.title.lower())
        s.summary = claims[0] if claims else _empty_summary(s, brief.topic)


def _empty_summary(s: OutlineSection, topic: str) -> str:
    """Speakable summary for a section that has no facts."""
    topic = "the" + topic[3:] if topic.startswith("The ") else topic
    if s.purpose == ScenePurpose.item:
        return f"Number {s.item_number} on our list."
    return {
        ScenePurpose.hook: f"Few stories are as remarkable as {topic}.",
        ScenePurpose.setup: f"To understand {topic}, we have to start at the beginning.",
        ScenePurpose.turn: "Then the story took an unexpected turn.",
        ScenePurpose.climax: "Everything led to this moment.",
        ScenePurpose.resolution: "What followed shaped everything that came after.",
        ScenePurpose.outro: f"That is the story of {topic}.",
    }.get(s.purpose, f"The story of {topic} moved forward one step at a time.")
