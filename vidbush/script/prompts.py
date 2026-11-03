"""Prompt text for the research, outline and script seats.

Every prompt asks for concise output and strict JSON. None of them may contain
the em dash character, and generated text is cleaned with util.no_em_dash.
Research, outline and script prompts carry a {language} line built by
language_line(brand.language) so a non English brand gets non English text.
"""
from __future__ import annotations

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese",
    "nl": "Dutch", "pl": "Polish", "sv": "Swedish", "da": "Danish", "no": "Norwegian", "fi": "Finnish",
    "ru": "Russian", "uk": "Ukrainian", "tr": "Turkish", "ar": "Arabic", "hi": "Hindi", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese", "id": "Indonesian", "vi": "Vietnamese", "cs": "Czech", "ro": "Romanian",
    "el": "Greek", "hu": "Hungarian", "he": "Hebrew", "th": "Thai",
}


def language_code(code: str) -> str:
    """'es-MX' -> 'es'; empty -> 'en'."""
    return (code or "en").strip().lower().split("-")[0].split("_")[0] or "en"


def language_name(code: str) -> str:
    """Display name for a language code; unknown codes pass through unchanged."""
    return LANGUAGE_NAMES.get(language_code(code), (code or "English").strip())


def language_line(code: str) -> str:
    """The instruction every content prompt carries."""
    return f"Language: write every claim, title, summary and line of narration in {language_name(code)}."

STYLE_RULES = (
    "Never use the em dash character or a double hyphen as punctuation; use commas or periods instead. "
    "Do not invent facts, quotes, numbers or dates. Mark approximations with words like about or roughly."
)

RESEARCH_SYSTEM = (
    "You are a meticulous documentary researcher. You read source excerpts and extract accurate, "
    "specific, checkable facts for a long-form YouTube documentary. Prefer concrete numbers, dates, "
    "names and places. Only use facts supported by the excerpts or by well established knowledge; "
    "set confidence below 0.6 for anything uncertain. " + STYLE_RULES + " Reply with strict JSON only."
)

RESEARCH_USER = """Topic: {topic}
Angle: {angle}
Audience: {audience}
Target length: {minutes} minutes
{language}

Source excerpts (each starts with its URL and title):
{excerpts}

Return JSON with exactly this shape:
{{"facts": [{{"id": "f01", "claim": "one sentence", "source_url": "...", "source_title": "...",
  "confidence": 0.0-1.0, "date": "year or era", "numbers": [{{"value": 0, "unit": "...", "label": "..."}}]}}],
 "anchors": [{{"kind": "place|date|object|person|event|number", "name": "...", "detail": "...",
  "lat": null, "lon": null, "fact_ids": ["f01"]}}]}}
Give {min_facts} to {max_facts} facts with ids f01, f02 and so on, and 5 to 10 visual anchors (things a
viewer can see on screen) whose fact_ids use those ids. source_url must be exactly one of the URLs listed
above, or "" for well established general knowledge (then set confidence below 0.6).
Give lat and lon for places when you know them. Keep each claim under 35 words."""

RESEARCH_MORE_USER = """Topic: {topic}
Angle: {angle}
{language}

You already gave these facts (id: claim):
{known}

Source excerpts (each starts with its URL and title):
{excerpts}

Give {need} more distinct, specific facts that do not repeat the ones above, with ids continuing from
f{next_id:02d}. source_url must be exactly one of the URLs listed above, or "" for well established general
knowledge (then set confidence below 0.6). Keep each claim under 35 words.
Return JSON: {{"facts": [{{"id": "f{next_id:02d}", "claim": "...", "source_url": "...", "source_title": "...",
  "confidence": 0.0-1.0, "date": "...", "numbers": []}}]}}"""

RESEARCH_ANCHORS_USER = """Topic: {topic}
{language}

Facts (id: claim):
{facts}

Give 5 to 10 visual anchors: places (with lat and lon), dates, objects, people, events or numbers a viewer
can see on screen, each tied to the facts above by id.
Return JSON: {{"anchors": [{{"kind": "place|date|object|person|event|number", "name": "...", "detail": "...",
  "lat": null, "lon": null, "fact_ids": ["f01"]}}]}}"""

FACTCHECK_SYSTEM = (
    "You are a strict fact checker for a documentary. For each fact card decide if it is verified "
    "(supported by the excerpts or by well established knowledge), unverified (plausible but not "
    "supported) or contradicted (the excerpts or established knowledge say otherwise). Be concise. "
    "Reply with strict JSON only."
)

FACTCHECK_USER = """Fact cards:
{cards}

Source excerpts:
{excerpts}

Return JSON: {{"verdicts": [{{"id": "f01", "status": "verified|unverified|contradicted", "note": "short reason"}}]}}"""

OUTLINE_SYSTEM = (
    "You are a story editor for long-form faceless YouTube documentaries and Top 10 videos. You build "
    "outlines with a strong hook, clear escalation and payoff, sized to the target length at about 150 "
    "spoken words per minute. " + STYLE_RULES + " Reply with strict JSON only."
)

OUTLINE_USER = """Format: {format}
Topic: {topic}
Angle: {angle}
Audience: {audience}
Target: {minutes} minutes, about {words} words in total.
Channel notes: {notes}
{language}

Structure to follow:
{structure}

Facts (id: claim):
{facts}

Return JSON: {{"title": "working title", "sections": [{{"id": "sec01", "title": "short chapter title",
 "purpose": "hook|setup|act|turn|climax|resolution|item|outro", "summary": "one or two sentences",
 "target_words": 0, "fact_ids": ["f01"], "item_number": null}}]}}
The target_words values must add up to about {words}. Use every important fact at least once."""

DOC_STRUCTURE = (
    "hook (about 7 percent, open on the most striking fact or image), setup (context), {acts} acts that "
    "build tension, one of them a turn near 55 percent where the story changes direction, a climax near "
    "80 percent, a resolution, and a short outro."
)

TOP10_STRUCTURE = (
    "hook (promise the list and tease number one), then exactly ten items counted down from item_number 10 "
    "to item_number 1, escalating so number 1 is the most impressive, each with a mini arc (setup, detail, "
    "payoff), then a short recap (purpose resolution) and an outro."
)

SCRIPT_SYSTEM = (
    "You are an experienced narration writer for long-form documentary YouTube channels with an older US "
    "audience. Write in a calm, clear, authoritative spoken register: short concrete sentences, active "
    "verbs, vivid but plain words, no filler, no hype, no cliches, no rhetorical questions in a row. "
    "Every line must be easy to say aloud. Spell out how numbers are said when it helps a narrator. "
    "Each beat is one to three sentences (about 12 to 40 words) and covers one idea that can be shown "
    "on screen. Tag each beat with the fact ids it uses. " + STYLE_RULES + " Reply with strict JSON only."
)

SCRIPT_USER = """Topic: {topic}
Angle: {angle}
Audience: {audience}
Channel notes: {notes}
Format: {format}
{language}

{sections}

Facts you may use (id: claim):
{facts}

{continuity}
Return JSON: {{"title": "video title", "scenes": [{{"id": "s01", "title": "...", "purpose": "...",
 "item_number": null, "beats": [{{"text": "...", "facts_used": ["f01"], "emphasis": "low|normal|high"}}]}}]}}
Write one scene per section above, in the same order, and hit each section's word target."""

SECTION_LINE = "Section {id} ({purpose}{item}): {title}. {summary} Target {words} words. Facts: {fact_ids}."

CONTINUITY = "The previous section ended with this beat, continue naturally from it: \"{last}\""

FULL_OUTLINE = """You are writing one section of a longer script. The whole outline, for context (write only the
section listed above):
{outline}
Facts already used by earlier sections (do not repeat them, refer back briefly at most): {used}
Do not re-introduce the topic; set up the next section."""

TOO_SHORT = """Your previous draft was too short: {got} words for a target of {want}. Write the full length this
time, with more concrete detail from the facts, not filler."""
