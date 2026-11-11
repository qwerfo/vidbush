"""AI image candidates through the image seat.

generate_ai_candidate(shot, ctx, prompt=None) -> Candidate writes
work/ai/<shot_id>_<n>.png and returns Candidate(source=ai_image). The seat
handles the sticky provider and the per image cost ceiling. Prompts are
positive only (diffusion models have no negation); unwanted content travels
as a separate negative prompt.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from ..models import Candidate, Shot, SourceKind
from ..util import no_em_dash

HOUSE_STYLE = ("Photographic documentary still, natural light, realistic detail, era-correct clothing, vehicles "
               "and architecture, wide 16:9 composition, 35mm lens. People appear only as anonymous figures, never "
               "as the likeness of a named real person.")
BASE_NEGATIVE = ["text", "captions", "watermark", "logos", "letters", "signature", "newspaper headlines",
                 "news chyrons or TV station branding", "the likeness of a named real person"]
_LEADING = re.compile(r"^(archival footage|footage|stock footage|historical photograph|photograph|photo|image|"
                      r"documentary photograph|wide establishing (archival )?footage|close detail photograph)"
                      r"\s+of\s+", re.I)
# negations inside a positive prompt: "no text, no captions", "Do not include ...", "without logos"
_NEGATION = re.compile(r"(?:,\s*|\.\s*|^)(?:no|without|do not include|avoid)\s+[^.]*", re.I)
_FILM_WORDS = re.compile(r"\b(footage|film|filmed|video|clip|reel)\b", re.I)


def build_prompt(shot: Shot, prompt: Optional[str] = None) -> str:
    """Positive image prompt: subject, era, setting, light and lens only. Unwanted things go to negative_prompt."""
    base = (prompt or shot.ai_prompt or "").strip()
    if not base:
        subject = _LEADING.sub("", shot.intent.strip().rstrip("."))
        base = subject[:1].upper() + subject[1:] + "." if subject else "Documentary scene."
        if shot.must_show:
            base += f" Clearly showing {', '.join(shot.must_show)}."
    base = _NEGATION.sub(".", base)
    base = re.sub(r"(\.\s*){2,}", ". ", base).strip(" .") + "."
    if "natural light" not in base.lower():
        base = f"{base} {HOUSE_STYLE}"
    return no_em_dash(" ".join(base.split()))


def negative_prompt(shot: Shot, prompt: Optional[str] = None) -> str:
    """Negative prompt for diffusion models: house negatives, negations found in the prompt and must_avoid.

    must_avoid entries that talk about footage or film make no sense for a still and are left out.
    """
    terms = list(BASE_NEGATIVE)
    for m in _NEGATION.finditer(prompt or shot.ai_prompt or ""):
        body = re.sub(r"^[,.\s]*(no|without|do not include|avoid)\s+", "", m.group(0), flags=re.I)
        terms += [t for t in re.split(r",\s*(?:no\s+)?|\s+or\s+|\s+and\s+", body) if t]
    terms += [t for t in shot.must_avoid if not _FILM_WORDS.search(t)]
    out: list[str] = []
    for t in terms:
        t = " ".join(t.strip(" .").split()).lower()
        if t and t not in out:
            out.append(t)
    return no_em_dash(", ".join(out))


def next_ai_index(shot: Shot, work: Path) -> int:
    """First n not used by an existing ai candidate id or file for this shot."""
    used = {c.id for c in shot.candidates}
    n = 1
    while f"{shot.id}_ai_image_{n}" in used or (work / "ai" / f"{shot.id}_{n}.png").exists():
        n += 1
    return n


SAFE_SYSTEM = ("You rewrite prompts for a photographic image model used in documentaries. Change wording, never "
               "content: keep every subject with its count, every action and every relationship (who pulls, rides, "
               "carries or holds what), every Must keep item, and the scene, place, era, mood and composition. "
               "Remove every named real person (describe an anonymous person of the right age and role instead), "
               "every brand, company, channel or station name, and anything that asks for readable text: "
               "headlines, signs, captions, labels, numbers, charts, maps with words. The narration is context: "
               "use it to keep the subject true to what is said, never copy its names or text. A Problem line says "
               "what the last image got wrong: write the prompt so the picture shows the Must keep items instead, "
               "and never describe the wrong picture. Return JSON "
               '{"prompt": "<one paragraph, under 90 words>"}.')
# Text artifacts as noun phrases, never single words: 'a letter home', 'document the war' or 'a record label' are
# scenes, while 'a newspaper headline' or 'a map of the trail' are text an image model would invent.
TEXT_SUBJECT_RE = re.compile(
    r"\b(?:headlines?|front pages?|newspaper (?:pages?|articles?|clippings?|columns?)|"
    r"(?:labell?ed|annotated|lettered|printed|road|street|survey) maps?|maps? (?:of|showing|with|marking|labell?ed)|"
    r"(?:text|wording|words|pages?) of (?:an? |the )?(?:letter|telegram|document|report|diary|treaty|law|song|book)|"
    r"document pages?|web ?pages?|websites?|sheet music|musical scores?|(?:bar|line|pie|data) (?:charts?|graphs?)|"
    r"(?:charts?|graphs?) (?:of|showing))\b", re.I)
_KEEP_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*\d+|\d+|[A-Za-z][A-Za-z'-]+")  # 'I-80' is one code
# Stems of what the rewrite removes on purpose (text bearing objects), so dropping them is not lost content.
_TEXT_OBJECT_STEMS = {"sign", "poster", "banner", "plaque", "placard", "billboard", "label", "headline", "caption",
                      "ticket", "text", "lettering", "wording", "word", "read", "say", "written"}
# Counts are content although STOP lists some of them as filler.
_COUNT_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                "twenty", "dozen", "dozens", "pair", "team", "both", "single", "lone", "hundreds", "thousands"}
# Count words that say the same count: 'single' rewritten 'one' keeps the count.
_SAME_COUNT = [{"one", "single", "lone", "sole", "1"}, {"two", "pair", "both", "couple", "twin", "2"}]
# Pronouns, auxiliaries and generic verbs carry no subject, count, action or relationship of their own.
_FILLER = {"him", "himself", "herself", "themselves", "we", "us", "our", "you", "your", "me", "my", "someone",
           "something", "am", "being", "do", "does", "did", "doing", "done", "get", "gets", "getting", "got",
           "make", "makes", "making", "made", "take", "takes", "taking", "taken", "took", "set", "sets", "setting",
           "live", "lives", "living", "lit", "show", "shows", "showing", "shown", "seen", "appear", "appears"}
_APOS = str.maketrans("’‘", "''")


def needs_readable_text(shot: Shot) -> bool:
    """True when the shot's subject is text to be read (a labelled map, a headline, a document, sheet music)."""
    return bool(TEXT_SUBJECT_RE.search(" ".join([shot.intent, *shot.must_show])))


def _stem(word: str) -> str:
    """Plain form for a prefix match: possessive, plural, -ing and -ed go, and a doubled last consonant is
    undoubled ('hitting' meets 'hit', 'flipping' meets 'flips', 'tiled' meets 'tile', "musher's" meets 'musher')."""
    word = re.sub(r"'s$", "", word.translate(_APOS))
    if (len(word) > 5 and word.endswith("ing")) or (len(word) > 4 and word.endswith("ed")):
        word = word[:-3] if word.endswith("ing") else word[:-2]
        return word[:-1] if len(word) > 2 and word[-1] == word[-2] and word[-1] not in "aeiouslz" else word
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _mid_sentence(word: str, text: str) -> bool:
    """True when text writes word capitalised after another word of the same sentence."""
    for m in re.finditer(rf"\b{re.escape(word)}\b", text):
        before = text[:m.start()].rstrip()
        if before and before[-1] not in ".!?:;\"":
            return True
    return False


def _keep_words(keep: list[str], texts: list[str]) -> tuple[list[str], set[str]]:
    """(content words, names) of the keep items. A capitalised word is a name when it is a possessive, follows
    another word of its item, is followed by a capitalised word, or the texts (prompt, narration, keep items)
    capitalise it mid-sentence and never write it lowercase ('Sled dogs' is content; 'Leonhard Seppala',
    'Seppala's team' and 'Togo' are names). Acronyms and abbreviations ('US', 'I-' of 'I-80') are neither."""
    from ..shots.ideas import STOP
    texts = [t.translate(_APOS) for t in texts]
    words: list[str] = []
    names: set[str] = set()
    for item in keep:
        found = _KEEP_WORD.findall(item.translate(_APOS))
        for i, w in enumerate(found):
            base = re.sub(r"'s$", "", w)
            low = base.lower()
            if any(c.isupper() for c in base[1:]) or w.endswith("-") or (re.search(r"\d", w) and not w.isdigit()):
                continue  # acronym or code ('US', 'TV', 'I-80')
            if base[0].isupper() and (base != w or i > 0 or (len(found) > 1 and found[1][0].isupper())
                                      or (any(_mid_sentence(base, t) for t in texts)
                                          and not any(re.search(rf"\b{re.escape(low)}\b", t) for t in texts))):
                names.add(base)
                continue
            if (low in STOP and low not in _COUNT_WORDS) or low in _FILLER or _stem(low) in _TEXT_OBJECT_STEMS:
                continue
            words.append(w)
    return words, names


def lost_keep_words(old: str, new: str, keep: list[str], narration: str = "") -> list[str]:
    """Content words and counts of keep items that the old prompt had and the new one lost. Names, acronyms,
    pronouns, auxiliaries, generic verbs and text bearing objects never count; a count matches its synonyms."""
    words, _ = _keep_words(keep, [old, narration, *keep])
    old, new = old.translate(_APOS), new.translate(_APOS)

    def has(text: str, w: str) -> bool:
        low = w.lower()
        forms = next((g for g in _SAME_COUNT if low in g), {low})
        return any(re.search(rf"\b{re.escape(_stem(f))}" + (r"\b" if f.isdigit() else ""), text, re.I)
                   for f in forms)

    return list(dict.fromkeys(w for w in words if has(old, w) and not has(new, w)))


def without_names(text: str, names: set[str]) -> str:
    """text with every run of capitalised words that holds one of names dropped ('Leonhard Seppala's team'
    becomes 'team'), so a prompt kept as it was never sends a name to the image model.
    shortcut: drops only names found in the keep items; a name only the prompt has stays, add the shot's
    people list if one is ever modelled."""
    if not names:
        return text

    def drop(m: re.Match) -> str:
        return "" if any(re.sub(r"['’]s$", "", t) in names for t in m.group(0).split()) else m.group(0)

    out = re.sub(r"\b[A-Z][\w'’]*(?:\s+[A-Z][\w'’]*)*", drop, text)
    out = re.sub(r"\s+([,.;:])", r"\1", out)
    out = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", out)
    out = " ".join(re.sub(r"^[\s,;:.]+", "", out).split())
    return out[:1].upper() + out[1:]


def safe_prompt(text: str, ctx: Any, keep: Optional[list[str]] = None, narration: str = "",
                problem: str = "") -> str:
    """The prompt with named people, brands and readable text rewritten out by the editor seat. Image models
    render names as fake likenesses and text as invented, often wrong, facts. Unchanged when no model answers
    or when the rewrite dropped content of a keep item (a rewrite may change wording, never content).
    A rewrite that lost keep words is asked once more with them named; the original prompt kept in any
    fallback goes out without the keep items' names, because sending names is what the rewrite exists to stop.
    problem (the judge's reason on a retry) is context only: it describes the wrong picture, so its words never
    go to the image model as prompt text."""
    keep = keep or []
    original = without_names(text, _keep_words(keep, [text, narration, *keep])[1])
    user = "\n".join([f"Prompt: {text}"] + ([f"Must keep: {'; '.join(keep)}"] if keep else [])
                     + ([f"Narration: {narration}"] if narration else [])
                     + ([f"Problem: {problem}"] if problem else []))
    lost: list[str] = []
    for _ in range(2):
        ask = user + (f"\nMust keep these words the last rewrite lost: {', '.join(lost)}" if lost else "")
        try:
            raw = ctx.seats.llm_json("editor", "ai_prompt_safety", SAFE_SYSTEM, ask, ctx.stage or "judge",
                                     temperature=0.2)
        except Exception as e:  # noqa: BLE001 - the original prompt still goes out; the negatives still apply
            ctx.log.warn("ai prompt rewrite failed, original prompt kept without names", error=str(e)[:200])
            return original
        new = raw.get("prompt") if isinstance(raw, dict) else None
        if not (isinstance(new, str) and len(new) > 20):
            return original
        new = no_em_dash(" ".join(new.split()))
        lost = lost_keep_words(text, new, keep, narration)
        if not lost:
            return new
    ctx.log.warn("ai prompt rewrite dropped content, original prompt kept without names", lost=lost,
                 rewrite=new[:300])
    return original


def generate_ai_candidate(shot: Shot, ctx: Any, prompt: Optional[str] = None, problem: str = "") -> Candidate:
    """Generate one still for the shot and return it as an ai_image candidate (not appended). The rewrite is
    grounded in the beat's narration; every still after a shot's first is a retry in the cost report.
    A retry's problem (the judge's reason) adds a positive "Must clearly show" line and goes to the rewrite as
    context only. The negative prompt comes from the base prompt alone, so a reason such as 'no sled' never
    turns the sled into a negative."""
    from ..judge.judge import narration_by_beat
    text = build_prompt(shot, prompt)
    if problem:
        text = build_prompt(shot, f"{text} Must clearly show: {', '.join(shot.must_show) or shot.intent}.")
    text = safe_prompt(text, ctx, keep=shot.must_show, narration=narration_by_beat(ctx).get(shot.beat_id, ""),
                       problem=problem)
    n = next_ai_index(shot, ctx.work)
    out = ctx.work / "ai" / f"{shot.id}_{n}.png"
    path, is_mock = _call_image(ctx, text, negative_prompt(shot, prompt), out, "image" if n == 1 else "image retry")
    width, height = _size(path)
    ctx.log.info("ai image generated", shot=shot.id, mock=is_mock, path=str(path), prompt=text[:300])
    return Candidate(id=f"{shot.id}_ai_image_{n}", source=SourceKind.ai_image, local_path=str(path), media="image",
                     width=width, height=height, licence="AI generated", attribution="AI generated image",
                     title="AI generated still", keyframes=[str(path)], flags=["mock"] if is_mock else [])


def _call_image(ctx: Any, text: str, negative: str, out: Path, note: str = "image") -> tuple[Path, bool]:
    """Image seat call; the negative prompt (fal and Runware use it) and the cost note go when the seat takes them."""
    try:
        params = inspect.signature(ctx.seats.image).parameters
    except (TypeError, ValueError):
        params = {}
    extra = {k: v for k, v in (("negative_prompt", negative), ("note", note)) if k in params}
    return ctx.seats.image(text, out, ctx.stage or "judge", **extra)


def _size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as im:
            return im.size
    except (OSError, ValueError):
        return 0, 0
