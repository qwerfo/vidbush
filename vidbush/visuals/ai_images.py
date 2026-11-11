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
               '{"prompt": "<one paragraph, under 90 words>", "removed": ["<every person name, brand, company, '
               'channel or station name and readable text you removed, as written in the prompt>"]}.')
# Text artifacts as noun phrases, never single words: 'a letter home', 'document the war' or 'a record label' are
# scenes, while 'a newspaper headline' or 'a map of the trail' are text an image model would invent.
TEXT_SUBJECT_RE = re.compile(
    r"\b(?:headlines?|front pages?|newspaper (?:pages?|articles?|clippings?|columns?)|"
    r"(?:labell?ed|annotated|lettered|printed|road|street|survey) maps?|maps? (?:of|showing|with|marking|labell?ed)|"
    r"(?:text|wording|words|pages?) of (?:an? |the )?(?:letter|telegram|document|report|diary|treaty|law|song|book)|"
    r"document pages?|web ?pages?|websites?|sheet music|musical scores?|(?:bar|line|pie|data) (?:charts?|graphs?)|"
    r"(?:charts?|graphs?) (?:of|showing))\b", re.I)
_WORD = re.compile(r"[\w'-]+")  # unicode words; hyphens split later, so 'sea-ice' meets 'sea ice'
_NAME_WORD = re.compile(r"[^\W\d_][\w'-]*")  # a word that starts with a letter of any alphabet
# Stems of what the rewrite removes on purpose (text bearing objects), so dropping them is not lost content.
_TEXT_OBJECT_STEMS = {"sign", "poster", "banner", "plaque", "placard", "billboard", "label", "headline", "caption",
                      "ticket", "text", "lettering", "wording", "word", "read", "say", "written"}
# Counts are content although STOP lists some of them as filler.
_COUNT_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                "twenty", "dozen", "dozens", "pair", "team", "both", "single", "lone", "hundreds", "thousands"}
# Words that say the same count: 'single' rewritten 'one' and 'three' rewritten '3' keep the count.
_SAME_COUNT = [{w, str(i + 1)} for i, w in enumerate(["one", "two", "three", "four", "five", "six", "seven", "eight",
                                                       "nine", "ten", "eleven", "twelve"])]
_SAME_COUNT[0] |= {"single", "lone", "sole"}
_SAME_COUNT[1] |= {"pair", "both", "couple", "twin"}
# Pronouns, auxiliaries and generic verbs carry no subject, count, action or relationship of their own.
_FILLER = {"him", "himself", "herself", "themselves", "we", "us", "our", "you", "your", "me", "my", "someone",
           "something", "am", "being", "do", "does", "did", "doing", "done", "get", "gets", "getting", "got",
           "make", "makes", "making", "made", "take", "takes", "taking", "taken", "took", "set", "sets", "setting",
           "show", "shows", "showing", "shown", "seen", "appear", "appears"}
_APOS = str.maketrans("’‘", "''")
_POSSESSIVE = re.compile(r"'s$|(?<=s)'$")  # "musher's", "sleds'"


def needs_readable_text(shot: Shot) -> bool:
    """True when the shot's subject is text to be read (a labelled map, a headline, a document, sheet music)."""
    return bool(TEXT_SUBJECT_RE.search(" ".join([shot.intent, *shot.must_show])))


def _stem(word: str) -> str:
    """Plain form for a prefix match: possessive, plural, -ing and -ed go, and a doubled last consonant is
    undoubled ('hitting' meets 'hit', 'flipping' meets 'flips', 'tiled' meets 'tile', "musher's" meets 'musher')."""
    word = _POSSESSIVE.sub("", word.translate(_APOS))
    if (len(word) > 5 and word.endswith("ing")) or (len(word) > 4 and word.endswith("ed")):
        word = word[:-3] if word.endswith("ing") else word[:-2]
        return word[:-1] if len(word) > 2 and word[-1] == word[-2] and word[-1] not in "aeiouslz" else word
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _keep_words(keep: list[str], names: Any = ()) -> list[str]:
    """Content words and counts of the keep items. Words of names (what the editor says it removed) never
    count, nor do codes ('I-80'), short acronyms ('US', 'TV'), pronouns, auxiliaries, generic verbs and text
    bearing objects. No capital letter rule: the editor that removes names is the one that names them."""
    from ..shots.ideas import STOP
    named = {_POSSESSIVE.sub("", w).casefold() for n in names for raw in _WORD.findall(n.translate(_APOS))
             for w in raw.split("-") if w}
    words: list[str] = []
    for raw in _WORD.findall(" ".join(keep).translate(_APOS)):
        if re.search(r"\d", raw) and not raw.isdigit():
            continue  # a code such as 'I-80' or 'B-29' names a thing, it is not a count
        for w in filter(None, raw.split("-")):
            base = _POSSESSIVE.sub("", w).strip("'")
            low = base.casefold()
            if (not base or low in named or (base.isupper() and 2 <= len(base) <= 3)
                    or (len(base) < 2 and not base.isdigit())):
                continue
            if (low in STOP and low not in _COUNT_WORDS) or low in _FILLER or _stem(low) in _TEXT_OBJECT_STEMS:
                continue
            words.append(base)
    return words


def lost_keep_words(old: str, new: str, keep: list[str], names: Any = ()) -> list[str]:
    """Content words and counts of keep items that the old prompt had and the new one lost; a count matches
    its synonyms and digits, and a word its reworded forms."""
    old, new = old.translate(_APOS), new.translate(_APOS)

    def has(text: str, w: str) -> bool:
        low = w.casefold()
        forms = next((g for g in _SAME_COUNT if low in g), {low})
        return any(re.search(rf"\b{re.escape(_stem(f))}" + (r"\b" if f.isdigit() else ""), text, re.I)
                   for f in forms)

    return list(dict.fromkeys(w for w in _keep_words(keep, names) if has(old, w) and not has(new, w)))


def _contains(text: str, item: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(item.translate(_APOS))}(?!\w)", text.translate(_APOS)) is not None


def _name_forms(names: list[str]) -> list[str]:
    """names plus the surname of each two or three word personal name ('Leonhard Seppala' adds 'Seppala'),
    because a prompt that names someone in full often names them again by surname ("Seppala's sled")."""
    out = list(names)
    for n in names:
        ws = n.split()
        if 2 <= len(ws) <= 3 and all(w[0].isupper() and any(c.islower() for c in w) for w in ws):
            out.append(ws[-1])
    return list(dict.fromkeys(out))


def guess_names(text: str, keep: Optional[list[str]] = None) -> list[str]:
    """Heuristic names, only for when the editor names none (no usable answer, or the old answer shape).
    A run of capitalised words is a name, except one word opening a sentence that has no inner capital
    ('McDonald', "O'Brien") and is neither capitalised mid-sentence elsewhere nor a name in a keep item; all caps words of 4+ letters are names; 2 and 3 letter
    acronyms ('US', 'TV') stay."""
    text = text.translate(_APOS)
    toks = list(_NAME_WORD.finditer(text))

    def base(w: str) -> str:
        return _POSSESSIVE.sub("", w)

    def opens(m: re.Match) -> bool:
        before = text[:m.start()].rstrip(" \"'(")
        return not before or before[-1] in ".!?:;"

    mid = {base(m.group(0)) for m in toks if m.group(0)[0].isupper() and not opens(m)}
    kept: set[str] = set()
    for item in keep or []:
        ws = _NAME_WORD.findall(item.translate(_APOS))
        kept |= {base(w) for i, w in enumerate(ws)
                 if w[0].isupper() and (i > 0 or len(ws) == 1 or ws[1][0].isupper())}
    names: list[str] = []

    def flush(run: list[re.Match]) -> None:
        words = [m.group(0) for m in run]
        if not words:
            return
        if (opens(run[0]) and len(words) == 1 and base(words[0]) not in mid | kept
                and not any(c.isupper() for c in words[0][1:])):
            return  # an opening capital is a sentence start, not a name
        if len(words) == 1 and words[0].isupper() and len(words[0]) <= 3:
            return  # a short acronym
        names.append(" ".join(words))

    run: list[re.Match] = []
    for m in toks:
        if m.group(0)[0].isupper():
            if run and not text[run[-1].end():m.start()].isspace():
                flush(run)
                run = []
            run.append(m)
        else:
            flush(run)
            run = []
    flush(run)
    names += [m.group(0) for m in toks if m.group(0).isupper() and len(m.group(0)) >= 4]
    return list(dict.fromkeys(base(n) for n in names))


_DANGLING = r"(?:and|or|with|of|by|beside|from|to|for|as|near)"


def without_names(text: str, names: Any) -> str:
    """text with every name (and its possessive) deleted, then whitespace and punctuation tidied, so a prompt
    kept as it was never sends a name to the image model."""
    if not names:
        return text
    out = text.translate(_APOS)
    for n in sorted(names, key=len, reverse=True):
        out = re.sub(rf"(?<!\w){re.escape(n.translate(_APOS))}(?:'s|')?(?!\w)", " ", out, flags=re.I)  # the editor may list a name in other casing
    for _ in range(3):
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)
        out = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", out)
        out = re.sub(rf"\b{_DANGLING}\s*(?=[,.;:!?]|$)", "", out, flags=re.I)
        out = re.sub(rf"(^|[.!?]\s*)\s*(?:{_DANGLING}\b|[,;:.])\s*", r"\1", out, flags=re.I)
    out = " ".join(out.split())
    return re.sub(r"(^|[.!?]\s+)(\w)", lambda m: m.group(1) + m.group(2).upper(), out)


def _content_count(text: str) -> int:
    from ..shots.ideas import STOP
    return sum(1 for w in re.findall(r"[^\W\d_]{3,}", text) if w.lower() not in STOP)


def _answer(raw: Any) -> tuple[Optional[str], Optional[list[str]]]:
    """(prompt, removed). A bare string or {"prompt"} without "removed" is the old answer shape: removed None."""
    if isinstance(raw, str):
        return raw, None
    if not isinstance(raw, dict):
        return None, None
    removed = raw.get("removed")
    return raw.get("prompt"), ([str(r) for r in removed if str(r).strip()] if isinstance(removed, list) else None)


def safe_prompt(text: str, ctx: Any, keep: Optional[list[str]] = None, narration: str = "",
                problem: str = "") -> str:
    """The prompt with named people, brands and readable text rewritten out by the editor seat. Image models
    render names as fake likenesses and text as invented, often wrong, facts. The editor lists what it removed;
    those are names to the keep check (never lost content) and must not appear in the rewrite. A rewrite that
    lost keep words or kept a removed item is asked once more with them named; after that the original prompt
    goes out with the removed items deleted (the rewrite instead when too little is left), so no fallback ever
    sends a name. Names are guessed by a heuristic only when the editor names none.
    problem (the judge's reason on a retry) is context only: it describes the wrong picture, so its words never
    go to the image model as prompt text."""
    keep = keep or []
    user = "\n".join([f"Prompt: {text}"] + ([f"Must keep: {'; '.join(keep)}"] if keep else [])
                     + ([f"Narration: {narration}"] if narration else [])
                     + ([f"Problem: {problem}"] if problem else []))
    names: list[str] = []
    lost: list[str] = []
    leaked: list[str] = []
    clean: Optional[str] = None  # last rewrite with no name in it
    new = ""
    for _ in range(2):
        ask = "\n".join([user] + ([f"Must keep these words the last rewrite lost: {', '.join(lost)}"] if lost else [])
                        + ([f"Remove what the last rewrite still named: {', '.join(leaked)}"] if leaked else []))
        try:
            raw = ctx.seats.llm_json("editor", "ai_prompt_safety", SAFE_SYSTEM, ask, ctx.stage or "judge",
                                     temperature=0.2)
        except Exception as e:  # noqa: BLE001 - the original prompt still goes out without names
            ctx.log.warn("ai prompt rewrite failed", error=str(e)[:200])
            break
        answer, removed = _answer(raw)
        if not (isinstance(answer, str) and len(answer) > 20):
            break
        new = no_em_dash(" ".join(answer.split()))
        if removed is None:
            removed = guess_names(text, keep)
            ctx.log.warn("ai prompt rewrite named no removals, names guessed by heuristic", names=removed)
        names = _name_forms(names + removed)
        leaked = [n for n in names if _contains(new, n)]
        lost = lost_keep_words(text, new, keep, names)
        if not leaked and not lost:
            return new
        if not leaked:
            clean = new
    if not new:
        names = _name_forms(guess_names(text, keep))
        ctx.log.warn("ai prompt rewrite gave no answer, names guessed by heuristic", names=names)
    out = no_em_dash(without_names(text, names))
    if clean and _content_count(out) < 3:
        ctx.log.warn("ai prompt kept without names was too short, rewrite used", lost=lost, rewrite=clean[:300])
        return clean
    if new:
        ctx.log.warn("ai prompt rewrite dropped content or kept a name, original prompt kept without names",
                     lost=lost, leaked=leaked, rewrite=new[:300])
    return out


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
