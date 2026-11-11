"""Heuristic visual ideas from beat text: the deterministic fallback for the director seat.

ideas_for_beat(bc, palette, rng) returns at least one footage and one still
VisualIdea for a beat, plus the motion ideas its concrete nouns, people,
places, era and numbers support (a call to action beat gets no motion idea).
Text helpers here are also used by the director to validate model output
(numbers cited by charts, places on maps) and to time on-screen text to the
narration and keep it from repeating it.
"""
from __future__ import annotations

import math
import random
import re
from typing import Optional

from pydantic import BaseModel, Field

from ..models import Beat, FactCard, MotionKind, MotionSpec, Research, Scene, ScenePurpose, VisualIdea, Word
from ..util import no_em_dash

# ---------------------------------------------------------------- vocabulary

ACTIVITY_NOUNS = {
    "worker", "workers", "laborer", "laborers", "men", "women", "crowd", "crowds", "soldier", "soldiers", "ship",
    "ships", "steamship", "steamer", "boat", "boats", "train", "trains", "locomotive", "railroad", "shovel",
    "shovels", "excavator", "dredge", "dredges", "crane", "cranes", "explosion", "dynamite", "construction",
    "factory", "factories", "engine", "engines", "parade", "march", "battle", "troops", "army", "sailors",
    "doctor", "doctors", "nurse", "nurses", "hospital", "patients", "children", "families", "farmers", "miners",
    "engineers", "engineer", "people", "president", "car", "cars", "truck", "trucks", "aircraft", "plane",
    "planes", "rocket", "launch", "tugboat", "tugboats", "locks", "workforce", "gang", "crew", "crews",
    "fire", "flood", "storm", "waves", "riot", "protest", "ceremony", "celebration", "mule", "mules",
    "horse", "horses", "wagon", "wagons", "steam", "drill", "drills", "machine", "machines", "mosquito",
    "mosquitoes", "fumigation", "fumigating", "spraying", "digging", "dig", "blast", "blasting", "landslide",
    "landslides", "mudslide", "mudslides", "passengers", "immigrants", "tanker", "tankers", "freighter", "war",
}
SCENIC_NOUNS = {
    "jungle", "jungles", "mountain", "mountains", "river", "rivers", "ocean", "oceans", "sea", "seas", "lake",
    "lakes", "valley", "desert", "forest", "forests", "rainforest", "swamp", "swamps", "coast", "coastline",
    "island", "islands", "canyon", "cliffs", "hills", "landscape", "isthmus", "beach", "rock", "rocks", "sky",
    "sunset", "sunrise", "harbor", "harbour", "port", "bay", "channel", "waterway", "canal", "dam", "bridge",
    "city", "town", "village", "tunnel", "map", "ruins", "cathedral", "fort", "fortress", "building",
    "buildings", "skyline", "gate", "gates", "lock", "spillway", "reservoir", "tropics", "rain", "mud",
    "grave", "graves", "cemetery", "office", "offices", "treaty", "document", "documents", "flag", "flags",
    "money", "gold", "coins", "bank", "stock", "shares", "newspaper", "newspapers", "headline", "headlines",
    "telegram", "letter", "letters", "blueprint", "blueprints", "plans", "model", "dirt", "earth",
}
CONCRETE = ACTIVITY_NOUNS | SCENIC_NOUNS
TITLE_WORDS = {"president", "king", "queen", "general", "doctor", "dr", "colonel", "captain", "emperor", "sir",
               "lord", "senator", "governor", "admiral", "major", "lieutenant", "chief", "engineer", "secretary"}
UNIT_ALIASES = {
    "mile": "miles", "miles": "miles", "km": "km", "kilometers": "km", "kilometres": "km", "kilometer": "km",
    "feet": "feet", "foot": "feet", "ft": "feet", "meters": "meters", "metres": "meters", "meter": "meters",
    "tons": "tons", "ton": "tons", "tonnes": "tons", "dollars": "dollars", "dollar": "dollars", "$": "dollars",
    "usd": "dollars", "percent": "percent", "%": "percent", "workers": "people", "worker": "people",
    "people": "people", "men": "people", "lives": "people", "deaths": "people", "laborers": "people",
    "employees": "people", "ships": "ships", "vessels": "ships", "years": "years", "days": "days",
    "hours": "hours", "acres": "acres", "gallons": "gallons", "cubic": "cubic yards", "yards": "yards",
    "locomotives": "locomotives", "shovels": "shovels", "francs": "francs",
}
TIME_UNITS = {"years", "days", "hours", "months", "weeks", "decades", "centuries", "minutes", "seconds"}
_MULT = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "hundred": 1e2}
_NUM_RE = re.compile(r"(\$)?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(thousand|million|billion|hundred)?\s*(%|percent)?",
                     re.I)
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})(s)?\b")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "by", "for", "with", "from", "that", "this",
        "it", "its", "was", "were", "is", "are", "be", "been", "had", "has", "have", "their", "they", "them",
        "he", "she", "his", "her", "about", "roughly", "nearly", "almost", "around", "over", "more", "than",
        "some", "into", "would", "could", "which", "who", "what", "when", "then", "just", "only", "also", "so",
        "but", "not", "no", "as", "one", "all", "most", "after", "before", "while", "every", "each", "there",
        "these", "those", "again", "still", "even", "very", "much", "many", "such", "own", "suddenly", "during",
        "across", "through", "between", "under", "within", "without", "against", "toward", "towards", "up",
        "down", "out", "off", "until", "since", "because", "though", "although", "among", "per", "will"}
DISPLAY_NAMES = {"United States of America": "United States", "United Kingdom": "Britain"}


class NumberMention(BaseModel):
    """A number found in text with its unit and a short label from context."""

    value: float
    unit: str = ""
    raw: str
    label: str = ""
    after: str = ""
    is_year: bool = False


class BeatContext(BaseModel):
    """Everything the heuristics know about one beat."""

    beat: Beat
    scene: Scene
    first_in_scene: bool = False
    era: Optional[int] = None
    facts: list[FactCard] = Field(default_factory=list)
    topic: str = ""
    scene_index: int = 0
    not_places: list[str] = Field(default_factory=list)  # names the script gives to people or animals


# ---------------------------------------------------------------- text helpers

def topic_of(title: str, research: Optional[Research]) -> str:
    """Short topic: research topic or the script title before a colon."""
    topic = research.topic if research and research.topic else re.split(r"[:|]", title or "")[0]
    return re.sub(r"^the\s+", "", topic.strip(), flags=re.I)


def extract_years(text: str) -> list[int]:
    """Four digit years (1000 to 2099) that are not quantities with units; spoken years count too."""
    text = spoken_years(text)
    out = []
    for m in _YEAR_RE.finditer(text):
        nxt = _WORD_RE.match(text[m.end():].lstrip())
        if nxt and nxt.group(0).lower() in UNIT_ALIASES and nxt.group(0).lower() not in TIME_UNITS:
            continue
        if text[max(0, m.start() - 1):m.start()] == "$":
            continue
        out.append(int(m.group(1)))
    return out


def _context_words(text: str, start: int, end: int, n: int = 3) -> tuple[str, str]:
    pre = re.split(r"[,.;:!?\d$]", text[max(0, start - 60):start])[-1]
    before = [w for w in _WORD_RE.findall(pre) if w.lower() not in STOP][-n:]
    seg = re.split(r"[,.;:!?\d$]", text[end:end + 60])[0]
    after = [w for w in _WORD_RE.findall(seg)][:4]
    while after and after[-1].lower() in STOP:
        after.pop()
    return " ".join(before), " ".join(after)


_SMALL = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
_SPELLED_RE = re.compile(r"\b(?:(?:%s)(?:[\s-]+|\s+and\s+)?)+\b" % "|".join(
    sorted(list(_SMALL) + list(_TENS) + list(_SCALES), key=len, reverse=True)), re.I)


def _words_value(words: list[str]) -> Optional[int]:
    total, cur = 0, 0
    for w in words:
        if w in _SMALL:
            cur += _SMALL[w]
        elif w in _TENS:
            cur += _TENS[w]
        elif w == "hundred":
            cur = max(cur, 1) * 100
        elif w in _SCALES:
            total += max(cur, 1) * _SCALES[w]
            cur = 0
    return total + cur if words else None


_UNIT_W = "one|two|three|four|five|six|seven|eight|nine"
_TEEN_W = "ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
_TENS_W = "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
_DECADES = {"twenties": 20, "thirties": 30, "forties": 40, "fifties": 50, "sixties": 60, "seventies": 70,
            "eighties": 80, "nineties": 90}
_NO_SCALE = rf"(?![\s-]+(?:hundred|thousand|million|billion|{_UNIT_W}|{_TEEN_W}|{_TENS_W})\b)"  # whole numbers only
# 'nineteen twenty-five', 'eighteen oh five', 'nineteen twelve', 'the nineteen twenties'. The century word
# starts at thirteen, so clock times ('ten thirty', 'eleven forty') are never read as years.
_SPOKEN_YEAR_RE = re.compile(
    rf"\b(thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)[\s-]+(?:oh[\s-]+({_UNIT_W})|"
    rf"({'|'.join(_DECADES)})|({_TEEN_W})|({_TENS_W})(?:[\s-]+({_UNIT_W}))?)\b{_NO_SCALE}", re.I)
# 'two thousand seven', 'two thousand and twelve'
_TWO_THOUSAND_RE = re.compile(
    rf"\btwo thousand(?:\s+and)?\s+(?:({_TEEN_W})|({_TENS_W})(?:[\s-]+({_UNIT_W}))?|({_UNIT_W}))\b{_NO_SCALE}",
    re.I)


# words after a number that make it a quantity or a range, never a year: 'fifteen twenty minutes', 'twenty thirty
# miles a day', 'sixteen seventeen year olds' (extract_years skips digits before a unit the same way)
_QUANTITY_WORDS = set(UNIT_ALIASES) | TIME_UNITS | {u[:-1] for u in TIME_UNITS}


def _quantity_follows(m: re.Match) -> bool:
    nxt = _WORD_RE.match(m.string[m.end():].lstrip())
    return bool(nxt) and nxt.group(0).lower().split("-")[0] in _QUANTITY_WORDS


def spoken_years(text: str) -> str:
    """Years written the way narration says them become digits: 'nineteen twenty-five' -> '1925',
    'the nineteen twenties' -> 'the 1920s', 'two thousand seven' -> '2007'. Other number words stay, and so
    does a pair followed by a unit ('fifteen twenty minutes later' is a range)."""
    def year(m: re.Match) -> str:
        if _quantity_follows(m):
            return m.group(0)
        cent, oh, decade, teen, tens, unit = (g.lower() if g else "" for g in m.groups())
        head = _words_value([cent]) * 100
        if decade:
            return f"{head + _DECADES[decade]}s"
        return str(head + (_words_value([w for w in (oh, teen, tens, unit) if w]) or 0))

    def two_k(m: re.Match) -> str:
        if _quantity_follows(m):
            return m.group(0)
        return str(2000 + (_words_value([g.lower() for g in m.groups() if g]) or 0))

    return _TWO_THOUSAND_RE.sub(two_k, _SPOKEN_YEAR_RE.sub(year, text))


def spelled_to_digits(text: str) -> str:
    """'one hundred dollars' -> '$100', 'ninety percent' -> '90 percent'. Single 'one' and 'zero' stay words.
    Spoken years are read as years first ('nineteen twenty-five' is 1925, not 19 + 25)."""
    text = spoken_years(text)

    def repl(m: re.Match) -> str:
        raw = m.group(0)
        words = [w for w in re.split(r"[\s-]+", raw.lower()) if w and w != "and"]
        if len(words) == 1 and words[0] in ("one", "zero", "hundred", "thousand", "million", "billion"):
            return raw
        val = _words_value(words)
        tail = " " if raw.endswith(" ") else ""
        return f"{val:,}{tail}" if val is not None else raw
    out = _SPELLED_RE.sub(repl, text)
    return re.sub(r"\b([\d,]+) dollars\b", r"$\1", out)


def extract_numbers(text: str) -> list[NumberMention]:
    """Quantities in text with normalised units; bare numbers inherit the unit used earlier in the sentence."""
    text = spelled_to_digits(text)
    out: list[NumberMention] = []
    years = set(extract_years(text))
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        last_unit = ""
        for m in _NUM_RE.finditer(sentence):
            out_m = _mention(sentence, m, years, last_unit)
            if out_m is None:
                continue
            last_unit = out_m.unit or last_unit
            out.append(out_m)
    return out


def _mention(sentence: str, m: re.Match, years: set[int], last_unit: str) -> Optional[NumberMention]:
    dollar, digits, mult, pct = m.group(1), m.group(2), m.group(3), m.group(4)
    if re.match(r"[A-Za-z]", sentence[m.end(2):m.end(2) + 1] or " "):
        return None  # part of a word, such as 1880s or 3rd
    value = float(digits.replace(",", ""))
    is_year = not dollar and not mult and not pct and "," not in digits and int(value) in years
    end = m.start() + len(m.group(0).rstrip())
    nxt = _WORD_RE.match(sentence[end:].lstrip())
    word = nxt.group(0).lower() if nxt and sentence[end:end + 2].strip()[:1].isalpha() else ""
    unit = "dollars" if dollar else "percent" if pct else UNIT_ALIASES.get(word, "")
    if not unit and not is_year and re.fullmatch(r"[a-z]{3,}[^s]s", word or ""):
        unit = word
    if not unit and not is_year:
        unit = last_unit
    value *= _MULT.get((mult or "").lower(), 1.0)
    label, after = _context_words(sentence, m.start(), end)
    return NumberMention(value=value, unit=unit, raw=m.group(0).strip(), label=label, after=after, is_year=is_year)


def fact_numbers(facts: list[FactCard]) -> list[tuple[NumberMention, FactCard]]:
    """Numbers carried by fact cards: structured numbers first, then numbers in the claim text."""
    out: list[tuple[NumberMention, FactCard]] = []
    for f in facts:
        for n in f.numbers:
            try:
                v = float(str(n.get("value", "")).replace(",", ""))
            except ValueError:
                continue
            unit = UNIT_ALIASES.get(str(n.get("unit", "")).lower(), str(n.get("unit", "")).lower())
            out.append((NumberMention(value=v, unit=unit, raw=str(n.get("value")), label=str(n.get("label", ""))), f))
        out.extend((m, f) for m in extract_numbers(f.claim) if not m.is_year)
    return out


def number_is_cited(value: float, facts: list[FactCard]) -> Optional[FactCard]:
    """Fact card whose numbers include value (1 percent tolerance, any thousand scale), else None."""
    for m, f in fact_numbers(facts):
        for scale in (1.0, 1e3, 1e6, 1e9):
            target = m.value / scale
            if abs(value - target) <= max(0.011 * abs(target), 1e-9):
                return f
    return None


def capitalised_names(text: str) -> list[str]:
    """People mentioned in text: runs that start with a known first name or follow a title word."""
    from ..mg import geo
    toks = geo.tokens(text)
    out: list[str] = []
    for start, words, _initial in geo.cap_runs(toks):
        if not words:
            continue
        prev = toks[start - 1].lower().rstrip(".") if start > 0 else ""
        first = words[0].lower().rstrip(".")
        if first in TITLE_WORDS and len(words) >= 2:
            words = words[1:]
        elif not (first in geo.FIRST_NAMES and len(words) >= 2) and prev not in TITLE_WORDS:
            continue
        name = " ".join(words).strip()
        if name and geo.lookup(name) is None and name not in out:
            out.append(name)
    return out


COMPOUND_HEADS = {"steam", "yellow", "cargo", "container", "sailing", "railroad", "construction", "rock", "dirt",
                  "concrete", "steel", "lock"}


def visual_nouns(text: str) -> list[str]:
    """Concrete visual nouns in order of appearance, with compound pairs like 'steam shovels'."""
    raw = _WORD_RE.findall(text)
    words = [w.lower() for w in raw]
    out: list[str] = []
    for i, w in enumerate(words):
        if w not in CONCRETE or (w in TITLE_WORDS and raw[i][0].isupper()):
            continue
        prev = words[i - 1] if i > 0 else ""
        if prev in COMPOUND_HEADS:
            if prev in out:
                out.remove(prev)
            w = f"{prev} {w}"
        if w not in out:
            out.append(w)
    return out


def content_words(text: str, n: int = 4) -> list[str]:
    """Longest non-stopword words, used when no concrete noun is found."""
    words = [w for w in _WORD_RE.findall(text) if w.lower() not in STOP and len(w) > 3
             and not w.lower().endswith("ly")]
    ranked = sorted(dict.fromkeys(words), key=lambda w: (-len(w), words.index(w)))
    return [w.lower() for w in ranked[:n]]


_TIME_PREFIX = (r"(in|by|on|during|at|since|until|from)\s+(the\s+)?((early|late|mid|summer|spring|autumn|fall|"
                r"winter)\s+of\s+)?((?:%s)\s+)?(\d{1,2}(st|nd|rd|th)?,?\s+)?\d{4}s?" % "|".join(
                    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
                     "october", "november", "december"]))
_TIME_ONLY_RE = re.compile(r"^(%s|\d{4}s?)$" % _TIME_PREFIX, re.I)
_TIME_LEAD_RE = re.compile(r"^%s\s*,?\s*" % _TIME_PREFIX, re.I)


def is_time_only(clause: str) -> bool:
    """True for a clause that is only a time expression: 'In November 1932', 'In the summer of 1931'."""
    return bool(_TIME_ONLY_RE.match(clause.strip(" ,.")))


def _has_substance(clause: str) -> bool:
    """A clause worth quoting carries a verb or a concrete noun."""
    words = [w.lower() for w in _WORD_RE.findall(clause)]
    return any(w in CONCRETE or w in _VERBS or (len(w) > 4 and w.endswith("ed")) for w in words)


def key_phrase(text: str, max_words: int = 9) -> str:
    """The most quotable short clause of a beat, ideally 4 to max_words words; never a bare date."""
    clauses = [c.strip(" \"'") for c in re.split(r"[;:.!?](?!\d)|,(?!\d{3})", text) if c.strip()]
    clauses = [re.sub(r"^(and|but|so|yet|or|then)\s+", "", c, flags=re.I) for c in clauses]
    clauses = [c for c in clauses if not is_time_only(c)]
    clauses = [_TIME_LEAD_RE.sub("", c) for c in clauses]
    clauses = [c for c in clauses if c.strip()]

    def score(c: str) -> float:
        n = len(c.split())
        s = -5.0 if n < 3 else 2.0 if n <= max_words else 2.0 - (n - max_words) * 0.4
        s += 0.8 if re.search(r"\d", c) else 0.0
        s += 0.4 * sum(w.lower().strip(",") in CONCRETE for w in c.split())
        s -= 0.0 if _has_substance(c) else 1.5
        return s - abs(n - 6) * 0.1
    best = max(clauses, key=score) if clauses else ""
    words = best.split()
    if len(words) > max_words + 3:
        words = words[:max_words]
        while words and words[-1].lower() in STOP:
            words.pop()
    phrase = " ".join(words).strip(" ,")
    return no_em_dash(phrase[:1].upper() + phrase[1:]) if phrase else ""


def era_label(year: Optional[int]) -> str:
    """'1910s' for 1914, '' when unknown."""
    return f"{year // 10 * 10}s" if year else ""


def era_avoid(year: Optional[int]) -> list[str]:
    """Anachronism guards for a historical era."""
    if not year or year >= 1990:
        return []
    out = ["modern vehicles", "modern buildings", "modern clothing", "smartphones or screens"]
    if year < 1935:
        out.append("color footage")
    return out


def era_from_facts(facts: list[FactCard]) -> Optional[int]:
    """First year mentioned in fact dates."""
    for f in facts:
        ys = extract_years(f.date or "")
        if ys:
            return ys[0]
    return None


# ---------------------------------------------------------------- entities

class BeatEntities(BaseModel):
    """What a beat is about, extracted once."""

    places: list[tuple[str, float, float]] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    areas: list[str] = Field(default_factory=list)  # countries, states, continents, seas: no single point
    people: list[str] = Field(default_factory=list)
    nouns: list[str] = Field(default_factory=list)
    years: list[int] = Field(default_factory=list)
    numbers: list[NumberMention] = Field(default_factory=list)
    phrase: str = ""


def beat_entities(bc: BeatContext) -> BeatEntities:
    """Places, people, nouns, years, numbers and key phrase of a beat (scene title adds places).

    Places named only in a comparison ('enough to pave a highway from X to Y') are not places of the beat.
    """
    from ..mg import geo
    text = bc.beat.text
    literal = " ".join(x for x in re.split(r"(?<=[.!?])\s+", text) if not COMPARE_RE.search(x))
    skip = set(bc.not_places)
    hits = geo.find_place_hits(literal, skip) or (geo.find_place_hits(bc.scene.title, skip) if literal == text else [])
    scene_text = " ".join(b.text for b in bc.scene.beats)
    nouns = (visual_nouns(text) or visual_nouns(bc.scene.title) or visual_nouns(scene_text)[:2]
             or content_words(bc.topic, 2))
    return BeatEntities(places=[(DISPLAY_NAMES.get(h.name, h.name), h.lat, h.lon) for h in hits],
                        countries=[DISPLAY_NAMES.get(h.name, h.name) for h in hits if h.kind == "country"],
                        areas=[DISPLAY_NAMES.get(h.name, h.name) for h in hits if geo.is_area(h.name)],
                        people=capitalised_names(text), nouns=nouns, years=extract_years(text),
                        numbers=[n for n in extract_numbers(text) if not n.is_year], phrase=key_phrase(text))


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _queries(parts: list[list[str]]) -> list[str]:
    out: list[str] = []
    for p in parts:
        q = re.sub(r"\s+", " ", " ".join(dict.fromkeys(x for x in p if x))).strip()
        if q and len(q.split()) >= 2 and q.lower() not in (o.lower() for o in out):
            out.append(q)
    return out[:5]


CTA_RE = re.compile(r"\b(subscribe|enjoyed this|leave a comment|like this video|this channel|next video)\b", re.I)
BASE_AVOID = ["people talking to camera", "on-screen text", "watermarks or channel logos"]


# ---------------------------------------------------------------- footage and still ideas

MASS_NOUNS = {"jungle", "rock", "rocks", "swamp", "mud", "rain", "steam", "dirt", "earth", "gold", "money",
              "fire", "dynamite", "construction", "people", "men", "women", "children", "workforce", "sky"}


def noun_phrase(noun: str) -> str:
    """'a ship', 'workers', 'jungle': adds an article to singular count nouns."""
    head = noun.split()[-1]
    if noun[:1].isupper() or head.endswith("s") or head in MASS_NOUNS:
        return noun
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def footage_ideas(bc: BeatContext, ent: BeatEntities) -> list[VisualIdea]:
    """Two footage ideas: the main subject, then a secondary subject or a wide establishing shot."""
    era = era_label(bc.era)
    old = bool(bc.era and bc.era < 1990)
    kind = "archival footage" if old else "footage"
    place = next((p[0] for p in ent.places if p[0] not in ent.areas), ent.places[0][0] if ent.places else "")
    main = (ent.people[:1] + ent.nouns[:2])[:2] or [bc.topic]
    rest = [n for n in ent.nouns if n not in main][:2]
    variants: list[tuple[list[str], bool]] = [(main, False)]
    variants.append((rest, False) if rest else ([place or bc.topic], True))
    out: list[VisualIdea] = []
    for k, (subj, wide) in enumerate(variants):
        prep = _prep(place)
        where = f" {prep} {place}" if place and place not in subj else ""
        when = f", {era}" if era else ""
        what = _join([noun_phrase(x) for x in subj])
        intent = (f"Wide establishing {kind} of {what}{when}." if wide
                  else f"{kind.capitalize()} of {what}{where}{when}.")
        queries = _queries([[place if place not in subj else "", *subj[:2], era, kind], [*subj[:2], era, kind],
                            [place, subj[0]], [bc.topic, subj[0], "film" if old else "footage"],
                            [bc.topic, kind]])
        activity = not wide and (bool(ent.people) or any(w.split()[-1] in ACTIVITY_NOUNS for w in subj))
        out.append(VisualIdea(beat_id=bc.beat.id, family="footage", intent=no_em_dash(intent),
                              must_show=subj[:2] + ([place] if place and k == 0 and place not in subj else []),
                              must_avoid=BASE_AVOID + era_avoid(bc.era), queries=queries,
                              priority=0.6 - 0.15 * k, opening_safe=activity))
    return out


def still_ideas(bc: BeatContext, ent: BeatEntities) -> list[VisualIdea]:
    """Two still ideas: a period photograph (or painting) and a detail or portrait."""
    era = era_label(bc.era)
    medium = "painting" if bc.era and bc.era < 1840 else "photograph"
    place = next((p[0] for p in ent.places if p[0] not in ent.areas), ent.places[0][0] if ent.places else "")
    out: list[VisualIdea] = []
    subjects = ent.nouns[:2] or [bc.topic]
    if ent.people:
        person = ent.people[0]
        out.append(VisualIdea(beat_id=bc.beat.id, family="still", intent=f"Portrait {medium} of {person}.",
                              must_show=[person], must_avoid=["on-screen text", "watermarks", "modern reproductions"],
                              queries=_queries([[person, "portrait", medium], [person, era], [person]]),
                              priority=0.62, opening_safe=True))
    main = _join([noun_phrase(x) for x in subjects])
    lead = "Historical" if bc.era and bc.era < 1990 else "Documentary"
    prep = f" {_prep(place)} "
    intent = f"{lead} {medium} of {main}{prep + place if place else ''}{', ' + era if era else ''}."
    out.append(VisualIdea(beat_id=bc.beat.id, family="still", intent=no_em_dash(intent),
                          must_show=subjects + ([place] if place else []),
                          must_avoid=["on-screen text", "watermarks or channel logos"] + era_avoid(bc.era),
                          queries=_queries([[place, *subjects, era, medium], [*subjects, era, medium],
                                            [place, subjects[0], medium], [bc.topic, subjects[0]],
                                            [bc.topic, "historical", medium]]),
                          priority=0.55, opening_safe=bool(ent.people)))
    if len(out) < 2:
        detail = (ent.nouns[1:2] or ent.nouns[:1] or [bc.topic])[0]
        out.append(VisualIdea(beat_id=bc.beat.id, family="still",
                              intent=f"Close detail {medium} of {noun_phrase(detail)}.",
                              must_show=[detail], must_avoid=["on-screen text", "watermarks"] + era_avoid(bc.era),
                              queries=_queries([[bc.topic, detail, medium], [detail, era], [detail, "close up"]]),
                              priority=0.4, opening_safe=False))
    return out


def person_role(name: str, text: str) -> str:
    """Role for a person lower third: a title word before the name ('President'), else the appositive after it
    ('Frank Crowe, a blunt, tireless engineer' -> 'Engineer'), else ''."""
    pre = re.search(r"\b([A-Za-z]+)\s+" + re.escape(name), text)
    if pre and pre.group(1).lower() in TITLE_WORDS:
        return pre.group(1).capitalize()
    m = re.search(re.escape(name) + r",\s+(?:a|an|the)\s+([^.;:!?]+)", text)
    if not m:
        return ""
    body = re.split(r",?\s+(?:who|which|that|was|had|is|has|would|did)\b", m.group(1))[0]
    parts = [p.strip() for p in body.split(",") if p.strip()]
    while len(parts) > 1 and len(parts[0].split()) == 1 and " of " not in parts[0]:
        parts.pop(0)  # a lone adjective before the next comma: 'a blunt, tireless engineer'
    phrase = parts[0] if parts else ""
    words = phrase.split()
    if " of " in f" {phrase} ":
        k = words.index("of")
        role = words[k - 1:k] + words[k:k + 5] if k > 0 else []
    else:
        role = words[-1:]
    out = " ".join(role).strip(" ,")
    return out[:1].upper() + out[1:] if out and len(out) <= 40 else ""


def _named_overlays(bc: BeatContext, ent: BeatEntities, palette: str) -> dict[str, MotionSpec]:
    """Lower thirds keyed by the name they show: people, and cities or landmarks (never areas)."""
    from ..mg import builders as b
    out: dict[str, MotionSpec] = {}
    for person in ent.people[:2]:
        out[person] = b.lower_third_spec(person, person_role(person, bc.beat.text), palette=palette)
    for name, _la, _lo in ent.places:
        if name not in ent.areas and name not in out:
            out[name] = b.lower_third_spec(name, _place_sub(name), palette=palette)
    return out


def overlay_options(bc: BeatContext, ent: BeatEntities, palette: str, rng: random.Random) -> list[MotionSpec]:
    """Generic overlays any footage or still of the beat can carry, best first (no named lower thirds).

    Labels only (a year, a noun): a heuristic cannot paraphrase, and a clause of the narration on screen
    repeats what the captions already show."""
    from ..mg import builders as b
    out: list[MotionSpec] = []
    if CTA_RE.search(bc.beat.text):
        return out
    for y in ent.years[:1]:
        sub = next((p[0] for p in ent.places if p[0] not in ent.areas), "")
        out.append(b.text_overlay_spec(str(y), position="upper_left", subtitle=sub, palette=palette))
    roll = rng.random()
    if roll < 0.12 and ent.nouns:
        out.append(b.highlight_box_spec(0.3 + rng.random() * 0.2, 0.3, 0.32, 0.3, ent.nouns[0].title(),
                                        palette=palette))
    elif roll < 0.2 and ent.nouns:
        out.append(b.arrow_spec(0.2, 0.3, 0.45 + rng.random() * 0.15, 0.5, ent.nouns[0].title(), palette=palette))
    return out


def overlay_choices(bc: BeatContext, palette: str, must_show: list[str]) -> list[MotionSpec]:
    """Overlays for one shot, best first: lower thirds naming what it shows, then the generic options.

    Empty on call-to-action beats, where an overlay shot should become a plain shot.
    """
    if CTA_RE.search(bc.beat.text):
        return []
    ent = beat_entities(bc)
    named = _named_overlays(bc, ent, palette)
    shown = {m.lower() for m in must_show}
    first = [spec for name, spec in named.items() if name.lower() in shown]
    return first + overlay_options(bc, ent, palette, random.Random(bc.beat.id))


PREPOSITIONS = {
    "at the": {"canal", "locks", "dam", "bridge", "building", "railroad"},
    "at": {"monument", "fort", "battlefield", "site", "port", "spaceport", "camp", "settlement", "ruins",
           "landmark", "waterfall", "geyser", "beach", "harbor"},
    "on the": {"river"},
    "on": {"lake"},
    "in the": {"ocean", "sea", "gulf", "strait", "bay", "isthmus", "canyon", "valley"},
}


def _prep(place: str) -> str:
    """Natural preposition for a place: 'at the' Gatun Locks, 'on the' Chagres River, 'in' Panama."""
    from ..mg import geo
    hit = geo.lookup(place)
    if hit is None or hit.kind != "landmark":
        return "in"
    kind = next((lm.get("kind", "") for lm in geo.landmarks() if lm["name"] == hit.name), "")
    prep = next((p for p, kinds in PREPOSITIONS.items() if kind in kinds), "in")
    return prep.replace(" the", "") if place.lower().startswith("the ") else prep


def _place_sub(name: str) -> str:
    from ..mg import geo
    hit = geo.lookup(name)
    if hit is None or not hit.country or hit.country == name:
        return ""
    return "United States" if hit.country == geo.US else hit.country


# ---------------------------------------------------------------- on-screen text against the narration

TEXT_OVERLAY_KINDS = {MotionKind.callout, MotionKind.text_overlay, MotionKind.lower_third, MotionKind.highlight_box,
                      MotionKind.arrow}
ANCHOR_LEAD_S = 0.2  # a text overlay appears this long after its anchor's first word starts
READ_MIN_S = 1.5
READ_PER_WORD_S = 0.3
REPEAT_WINDOW_S = 8.0  # narration this close to on-screen text counts when checking for repeats
NAME_TOKENS = 3  # texts this short (names, places, years, labels) never count as repeating the narration


def norm_tokens(text: str) -> list[str]:
    """Lowercase word and number tokens, spelled numbers as digits, thousands commas and possessive 's
    dropped (Nome's is Nome), hyphenated words split (dog-sled is dog sled)."""
    low = re.sub(r"['\u2019]s\b", "", spelled_to_digits(text).lower())
    return [re.sub(r"[,']", "", t) for t in re.findall(r"[a-z0-9]+(?:[,'.][a-z0-9]+)*", low)]


def reads_as_name(text: str) -> bool:
    """Every word is capitalised, a number or a stop word (Nome, Gulf of Alaska, 1925): a name or label."""
    ws = re.findall(r"[^\W_][\w'\u2019.-]*", text)
    return any(w.lower() not in STOP for w in ws) and all(w[0].isupper() or w[0].isdigit() or w.lower() in STOP
                                                          for w in ws)


def repeats_narration(text: str, narration: str, name: bool = False) -> bool:
    """True when on-screen text copies the narration, which the captions already show: min(4, its length)
    consecutive tokens of it are spoken, or (above NAME_TOKENS tokens) 60 percent or more of its content words
    are. Names of NAME_TOKENS tokens or fewer never count (name: the caller knows it is one, else
    reads_as_name), so names, places and years pass; a short clause of the narration does not."""
    toks, said = norm_tokens(text), norm_tokens(narration)
    if len(toks) < 2 or (len(toks) <= NAME_TOKENS and (name or reads_as_name(text))):
        return False
    n = min(4, len(toks))
    grams = {tuple(said[i:i + n]) for i in range(len(said) - n + 1)}
    if any(tuple(toks[i:i + n]) in grams for i in range(len(toks) - n + 1)):
        return True
    content = {t for t in toks if t not in STOP}
    return len(toks) > NAME_TOKENS and bool(content) and len(content & set(said)) >= 0.6 * len(content)


def spec_repeats_narration(spec: Optional[MotionSpec], narration: str) -> bool:
    """A text overlay or card whose title or subtitle repeats the narration (a lower third is a name)."""
    return spec is not None and spec.kind in TEXT_OVERLAY_KINDS and any(
        repeats_narration(str(t), narration, spec.kind == MotionKind.lower_third)
        for t in (overlay_title(spec), spec.payload.get("subtitle")) if t)


def overlay_title(spec: MotionSpec) -> str:
    """The main text a viewer reads on an overlay: title, else label."""
    return str(spec.payload.get("title") or spec.payload.get("label") or "")


def read_seconds(spec: MotionSpec) -> float:
    """Reading time of an overlay's text: at least READ_MIN_S, about READ_PER_WORD_S per word."""
    text = f"{overlay_title(spec)} {spec.payload.get('subtitle') or ''}"
    return max(READ_MIN_S, READ_PER_WORD_S * len(text.split()))


def anchor_time(anchor: str, words: list[Word], lo: float, hi: float) -> Optional[float]:
    """Start of the narration word where the anchor is spoken, among words starting in [lo, hi]; None when
    it is not. The match holds the most of the anchor's content words (at least half) within its length
    plus two tokens; the earliest best match wins. A word is matched by its own tokens (Nenana's, dog-sled)."""
    toks = norm_tokens(anchor)
    want = {t for t in toks if t not in STOP} or set(toks)
    if not want:
        return None
    span = [(w.start, t) for w in words if lo <= w.start <= hi for t in norm_tokens(w.text)]
    best: Optional[tuple[int, float]] = None
    for i, (start, tok) in enumerate(span):
        if tok not in want:
            continue
        got = len(want & {t for _s, t in span[i:i + len(toks) + 2]})
        if got >= max(1, math.ceil(0.5 * len(want))) and (best is None or got > best[0]):
            best = (got, start)
    return best[1] if best else None


def overlay_start(spec: MotionSpec, words: list[Word], start: float, end: float) -> Optional[float]:
    """Seconds into the shot [start, end] where a text overlay appears: ANCHOR_LEAD_S after its anchor (the
    seat's anchor, else its title) is spoken, with its reading time left before the cut. None when the shot
    does not hold that moment."""
    t = anchor_time(spec.anchor or overlay_title(spec), words, start - ANCHOR_LEAD_S,
                    end - read_seconds(spec) - ANCHOR_LEAD_S)
    return None if t is None else round(max(0.0, t + ANCHOR_LEAD_S - start), 3)


def spoken_near(words: list[Word], t: float, window: float = REPEAT_WINDOW_S) -> str:
    """Narration spoken within window seconds of t."""
    return " ".join(w.text for w in words if abs(w.start - t) <= window)


# ---------------------------------------------------------------- motion ideas

def _motion_idea(bc: BeatContext, spec: MotionSpec, intent: str, priority: float) -> VisualIdea:
    return VisualIdea(beat_id=bc.beat.id, family="motion", intent=no_em_dash(intent), motion=spec,
                      priority=priority, opening_safe=spec.kind.value in {"map", "number_card"})


TRAVEL_RE = re.compile(r"\b(sail|sailed|sailing|sails|travel|travell?ed|travels|ship|shipped|shipping|carry|"
                       r"carried|carries|haul|hauled|move|moved|moving|flew|fly|flight|marched|march|steamed|"
                       r"crossed|cross|voyage|journey|journeys|route|routes|ran|run|runs|drove|drive|rode|"
                       r"railroad|railway|road|highway|pipeline|trek|trekked|migrated|fled|delivered|sent)\b", re.I)
COMPARE_RE = re.compile(r"\b(enough(?:\s+\w+){0,3}\s+to|as long as|as far as|as wide as|as tall as|like a|longer than|"
                        r"the distance from|equivalent|the length of)\b", re.I)
_NAME = r"((?:the\s+)?[A-Z][\w'.-]*(?:\s+(?:of\s+)?[A-Z][\w'.-]*){0,3})"
_FROM_TO_RE = re.compile(r"\bfrom\s+" + _NAME + r"\s+to\s+" + _NAME)
_BETWEEN_RE = re.compile(r"\bbetween\s+" + _NAME + r"\s+and\s+" + _NAME)


def _point(name: str) -> Optional[tuple[str, float, float]]:
    """A city or landmark with one location (never a country, state, continent or sea)."""
    from ..mg import geo
    name = re.sub(r"^the\s+", "", name.strip(), flags=re.I)
    hit = geo.lookup(name)
    if hit is None or geo.is_area(hit.name):
        return None
    return DISPLAY_NAMES.get(hit.name, hit.name), hit.lat, hit.lon


def travel_route(text: str) -> Optional[list[tuple[str, float, float]]]:
    """Endpoints of a real journey: 'from X to Y' with a movement word, or 'between X and Y'.

    Comparisons ('enough to pave a highway from X to Y') and areas as endpoints give None.
    """
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if COMPARE_RE.search(sent):
            continue
        for rx, need_verb in ((_FROM_TO_RE, True), (_BETWEEN_RE, False)):
            m = rx.search(sent)
            if not m or (need_verb and not TRAVEL_RE.search(sent)):
                continue
            a, b = _point(m.group(1)), _point(m.group(2))
            if a and b and a[0] != b[0]:
                return [a, b]
    return None


def map_idea(bc: BeatContext, ent: BeatEntities, palette: str) -> Optional[VisualIdea]:
    """Route map for a real journey between two points (real geometry for the travel mode the beat names, else
    markers only), else markers on the named points with areas labelled."""
    from ..mg import builders as b
    from ..mg.routes import infer_via
    if not ent.places:
        return None
    route = travel_route(bc.beat.text)
    points = route or [p for p in ent.places if p[0] not in ent.areas]
    shaded = [a for a in ent.areas][:3]
    if route:
        title = f"{route[0][0]} to {route[-1][0]}"
        intent = f"Map tracing the route from {route[0][0]} to {route[-1][0]}."
    else:
        title = points[0][0] if points else _join(shaded)
        if points:
            intent = f"Map pushing in on {title}" + (f", with {_join(shaded)} shaded." if shaded else ".")
        else:
            intent = f"Map pushing in with {title} shaded."
    via = infer_via(bc.beat.text) if route else "line"
    spec = b.map_spec(points[:6], route=bool(route), shaded=shaded, title=title, palette=palette, via=via)
    return _motion_idea(bc, spec, intent, 0.95)


def _same_unit_groups(mentions: list[NumberMention]) -> dict[str, list[NumberMention]]:
    groups: dict[str, list[NumberMention]] = {}
    for m in mentions:
        if m.unit and m.unit not in TIME_UNITS:
            g = groups.setdefault(m.unit, [])
            if all(abs(o.value - m.value) > 1e-9 for o in g):
                g.append(m)
    return groups


_VERBS = {"is", "was", "were", "are", "rose", "rises", "stands", "stood", "reached", "cost", "costs", "had",
          "has", "took", "measured", "measures", "weighed", "weighs", "grew", "fell", "carried", "held", "holds",
          "employed", "killed", "claimed", "spans", "spanned", "covered", "covers", "stretches", "stretched"}


def _sentence_of(raw: str, text: str) -> str:
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if raw in sent:
            return sent
    return ""


_LEAD_RE = re.compile(r"^(Before|After|On|With|By|In|Under|Through|Via)\s+([^,]{3,40}),")


def _subject(sentence: str) -> str:
    """A leading phrase ('Before the canal'), else words up to the first verb, as a short chart label."""
    lead = _LEAD_RE.match(sentence.strip())
    if lead and len(lead.group(0).split()) <= 5:
        return lead.group(0).rstrip(",")
    words: list[str] = []
    for w in _WORD_RE.findall(sentence):
        if w.lower() in _VERBS or re.match(r"\d", w):
            break
        words.append(w)
    return " ".join(words[:5])


def _subject_labels(group: list[NumberMention], text: str) -> Optional[list[str]]:
    sents = [_sentence_of(m.raw, text) for m in group]
    if len(set(sents)) != len(sents) or not all(sents):
        return None
    labels = [_subject(s) for s in sents]
    if not all(labels) or len(set(labels)) != len(labels):
        return None
    return [lab[:1].upper() + lab[1:] for lab in labels]


def _chart_labels(group: list[NumberMention], text: str) -> list[str]:
    from ..mg.builders import format_number
    text = spelled_to_digits(text)
    subjects = _subject_labels(group, text)
    if subjects:
        return subjects
    same = len({_sentence_of(m.raw, text) for m in group}) == 1
    if len(group) == 2 and same and re.search(r"\b(shrunk|shrank|fell|dropped|cut|rose|grew|from\b.*\bto)\b", text, re.I):
        return ["Before", "After"]
    labels = []
    for m in group:
        after = " ".join(w for w in m.after.split() if UNIT_ALIASES.get(w.lower(), w.lower()) != m.unit)
        words = (after or m.label or format_number(m.value, m.unit)).split()[:5]
        while len(words) > 1 and words[-1].lower() in STOP:
            words.pop()
        lab = " ".join(words)[:32].strip()
        labels.append(lab[:1].upper() + lab[1:])
    return labels if len(set(labels)) == len(labels) else [format_number(m.value, m.unit) for m in group]


CHART_SKIP = {"approximate", "approx", "estimated", "estimate", "total", "number", "figure", "average"}


def _chart_title(group: list[NumberMention], unit: str) -> str:
    counts: dict[str, int] = {}
    for m in group:
        for w in set(_WORD_RE.findall(m.label.lower())) - STOP - CHART_SKIP:
            counts[w] = counts.get(w, 0) + 1
    common = [w for w, c in sorted(counts.items(), key=lambda kv: -kv[1]) if c >= 2 and len(w) > 3]
    return f"{(common[0] if common else unit).capitalize()} compared"


def chart_idea(bc: BeatContext, ent: BeatEntities, research: Optional[Research], palette: str) -> Optional[VisualIdea]:
    """Bar chart when the beat (or its facts) has two or more comparable numbers with the same unit."""
    from ..mg import builders as b
    facts = research.facts if research else []
    mentions = list(ent.numbers) + [m for m, _f in fact_numbers(bc.facts)]
    if facts:
        mentions = [m for m in mentions if number_is_cited(m.value, facts)]
    groups = {u: g for u, g in _same_unit_groups(mentions).items() if len(g) >= 2}
    if not groups:
        return None
    unit, group = max(groups.items(), key=lambda kv: len(kv[1]))
    group = group[:6]
    src = number_is_cited(group[0].value, facts) if facts else None
    values = [m.value for m in group]
    title = ent.phrase if re.search(r"\d", ent.phrase) else _chart_title(group, unit)
    spec = b.bar_chart_spec(_chart_labels(group, bc.beat.text), values, units=unit, title=title,
                            source_note=src.source_title if src else "",
                            highlight_index=values.index(max(values)), palette=palette)
    return _motion_idea(bc, spec, f"Bar chart comparing {unit}: {title}.", 1.0)


def line_idea(bc: BeatContext, ent: BeatEntities, research: Optional[Research], palette: str) -> Optional[VisualIdea]:
    """Line chart for a series of one unit over three or more years found in fact cards."""
    from ..mg import builders as b
    if not research:
        return None
    units = {m.unit for m in ent.numbers} | {m.unit for m, _f in fact_numbers(bc.facts)}
    series: dict[tuple[str, str], dict[int, tuple[float, FactCard]]] = {}
    for m, f in fact_numbers(research.facts):
        ys = extract_years(m.label) or (extract_years(f.date)[:1] if len(extract_years(f.date)) == 1 else [])
        key = (m.unit, _YEAR_RE.sub("", m.label.lower()).strip(" ,in"))
        if m.unit in units - {""} and ys and key[1]:
            series.setdefault(key, {}).setdefault(ys[0], (m.value, f))
    best: list[tuple[int, float, FactCard]] = []
    best_unit = ""
    for (unit, _label), pts in series.items():
        if len(pts) >= 3 and len(pts) > len(best):
            best = sorted((y, v, f) for y, (v, f) in pts.items())
            best_unit = unit
    if not best:
        return None
    spec = b.line_chart_spec([str(y) for y, _v, _f in best], [v for _y, v, _f in best], units=best_unit,
                             title=f"{best_unit.title()} by year", source_note=best[0][2].source_title,
                             palette=palette)
    return _motion_idea(bc, spec, f"Line chart of {best_unit} over the years.", 1.0)


_PLACE_AFTER_RE = re.compile(r"^(from|to)\s+((?:the\s+)?[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,3})")
_MONEY_RE = re.compile(r"\b(cost|costs|spent|price|budget|paid|bill)\b", re.I)


def _tail_after(m: NumberMention, sentence: str) -> str:
    i = sentence.find(m.raw)
    return sentence[i + len(m.raw):].strip() if i >= 0 else m.after


def number_label(m: NumberMention, text: str) -> str:
    """Card label: the unit as written plus a short noun phrase ('miles to Las Vegas', 'cubic yards of concrete').

    Prepositions and articles end the phrase; money and percent fall back to 'cost' or nothing.
    """
    from ..mg import geo
    text = spelled_to_digits(text)
    words = _WORD_RE.findall(_tail_after(m, _sentence_of(m.raw, text) or text))
    unit_words: list[str] = []
    while words and (UNIT_ALIASES.get(words[0].lower()) == m.unit or words[0].lower() in m.unit.split()):
        unit_words.append(words.pop(0))
    head = " ".join(unit_words) or ("" if m.unit in ("dollars", "percent") else m.unit)
    rest = " ".join(words)
    place = _PLACE_AFTER_RE.match(rest)
    if place and head:
        hit = geo.lookup(re.sub(r"^the\s+", "", place.group(2)))
        if hit is not None:
            return f"{head} {place.group(1)} {DISPLAY_NAMES.get(hit.name, hit.name)}"
    if head and len(words) > 1 and words[0].lower() == "of" and _nounish(words[1]):
        noun = words[1:3] if len(words) > 2 and _nounish(words[2]) else words[1:2]
        return f"{head} of {' '.join(noun)}"
    if not head and m.unit == "dollars" and _MONEY_RE.search(_sentence_of(m.raw, text)):
        return "cost"
    return head


def _nounish(word: str) -> bool:
    w = word.lower()
    return w not in STOP and w not in _VERBS and not (len(w) > 4 and w.endswith("ed"))


def number_sublabel(m: NumberMention, text: str) -> str:
    """A year from the same sentence as the number, else empty."""
    years = [y for y in extract_years(_sentence_of(m.raw, spelled_to_digits(text))) if y != int(m.value)]
    return str(years[0]) if years else ""


_UNIT_W = "one|two|three|four|five|six|seven|eight|nine"
_TEEN_W = "ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
_TENS_W = "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
_SPOKEN_YEAR_RE = re.compile(
    r"\b(?:two[\s-]+thousand(?:\s+and)?[\s-]+(?P<t>(?:%s)(?:[\s-]+(?:%s))?|%s|%s)"
    r"|(?P<h>%s|twenty)[\s-]+(?P<r>oh[\s-]+(?:%s)|(?:%s)(?:[\s-]+(?:%s))?|%s))\b"
    r"(?![\s-]+(?:hundred|thousand|million|billion))" % (_TENS_W, _UNIT_W, _TEEN_W, _UNIT_W, _TEEN_W, _UNIT_W,
                                                          _TENS_W, _UNIT_W, _TEEN_W), re.I)


def spoken_years_to_digits(text: str) -> str:
    """'two thousand seven' -> '2007', 'nineteen fifty-nine' -> '1959', 'eighteen oh five' -> '1805': a year
    read aloud is a year, not the count spelled_to_digits would make of it ('2,007', '78'). '<teen> hundred'
    stays a count ('fifteen hundred soldiers'): that is how a quantity from 1,100 to 1,999 is said."""
    def value(words: str) -> int:
        return _words_value([w for w in re.split(r"[\s-]+", words.lower()) if w not in ("oh", "and")]) or 0

    def repl(m: re.Match) -> str:
        if m.group("t"):
            return str(2000 + value(m.group("t")))
        return str(value(m.group("h")) * 100 + value(m.group("r")))
    return _SPOKEN_YEAR_RE.sub(repl, text)


def number_idea(bc: BeatContext, ent: BeatEntities, research: Optional[Research], palette: str) -> Optional[VisualIdea]:
    """Number card for the most striking number (count up) or, failing that, the year. Years read aloud count
    as years (spoken_years_to_digits), never as a quantity."""
    from ..mg import builders as b
    facts = research.facts if research else []
    text = spoken_years_to_digits(bc.beat.text)
    numbers = [n for n in extract_numbers(text) if not n.is_year]
    years = extract_years(text)
    if numbers:
        cited = [m for m in numbers if not facts or number_is_cited(m.value, facts)]
        pool = cited or numbers
        m = max(pool, key=lambda x: (x.unit not in TIME_UNITS, x.value))
        src = number_is_cited(m.value, facts) if facts else None
        label = number_label(m, text)[:40]
        spec = b.number_card_spec(b.format_number(m.value, m.unit), label=label,
                                  sublabel=number_sublabel(m, text), count_up=True, palette=palette,
                                  source_note=b.source_note_for(src.source_title) if src else "")
        return _motion_idea(bc, spec, f"Number card: {m.raw} {label}.".replace("  ", " "), 0.85)
    if years:
        words = ent.phrase.split()[:6]
        while words and words[-1].lower() in STOP:
            words.pop()
        spec = b.number_card_spec(str(years[0]), label=" ".join(words), count_up=False, palette=palette)
        return _motion_idea(bc, spec, f"Year card: {years[0]}.", 0.6)
    return None


# ---------------------------------------------------------------- per beat

def _attach_overlays(bc: BeatContext, ent: BeatEntities, ideas: list[VisualIdea], palette: str,
                     rng: random.Random) -> None:
    """Each overlay goes to at most one footage and one still idea; named lower thirds only where shown."""
    named = _named_overlays(bc, ent, palette)
    generic = overlay_options(bc, ent, palette, rng)
    for family in ("footage", "still"):
        used: set[int] = set()
        for idea in (i for i in ideas if i.family == family):
            shown = {m.lower() for m in idea.must_show}
            pool = [spec for name, spec in named.items() if name.lower() in shown] + generic
            spec = next((sp for sp in pool if id(sp) not in used), None)
            if spec is not None:
                used.add(id(spec))
                idea.overlay = spec


def ideas_for_beat(bc: BeatContext, research: Optional[Research], palette: str,
                   rng: random.Random) -> list[VisualIdea]:
    """All heuristic ideas for one beat; always includes footage and still, motion when the beat supports one.

    Call-to-action beats get plain footage and stills only: no overlay and no card. The first beat of a
    scene after the hook carries the chapter title as a lower third on its first footage idea, because a
    full screen chapter card stops the story for a title.
    """
    from ..mg import builders as b
    ent = beat_entities(bc)
    foot = footage_ideas(bc, ent)
    still = still_ideas(bc, ent)
    if CTA_RE.search(bc.beat.text):
        return foot + still
    _attach_overlays(bc, ent, foot + still, palette, rng)
    if bc.first_in_scene and bc.scene.purpose != ScenePurpose.hook and foot:
        foot[0].overlay = b.lower_third_spec(bc.scene.title, era_label(bc.era) if bc.era else "", palette=palette)
    motion = [x for x in (chart_idea(bc, ent, research, palette), line_idea(bc, ent, research, palette),
                          map_idea(bc, ent, palette), number_idea(bc, ent, research, palette)) if x]
    return foot + still + motion


def beat_contexts(script_scenes: list[Scene], research: Optional[Research], topic: str) -> list[BeatContext]:
    """BeatContext for every beat with the era carried forward from earlier beats."""
    facts = {f.id: f for f in (research.facts if research else [])}
    out: list[BeatContext] = []
    era: Optional[int] = None
    from ..mg import geo
    not_places = sorted(geo.creature_names(" ".join(b.text for sc in script_scenes for b in sc.beats)))
    for si, scene in enumerate(script_scenes):
        for bi, beat in enumerate(scene.beats):
            bfacts = [facts[f] for f in beat.facts_used if f in facts]
            years = extract_years(beat.text)
            era = years[0] if years else (era_from_facts(bfacts) or era)
            out.append(BeatContext(beat=beat, scene=scene, first_in_scene=bi == 0, era=era, facts=bfacts,
                                   topic=topic, scene_index=si, not_places=not_places))
    later: Optional[int] = None
    for bc in reversed(out):  # beats before the first dated beat take the next known era
        later = bc.era or later
        bc.era = bc.era or later
    return out
