"""Parse a pasted script of any length into a Script.

Handles markdown titles and headings, ALL CAPS headings, "Scene 3:" and
"Part II" headings, top10 markers ("#10", "Number 10:", "10.", "No. 10"),
plain paragraphs with no headings, YAML front matter and stage directions.
Beats are one to three sentences of roughly 12 to 45 words.

Public helpers reused by the writer: split_sentences, make_beat_texts,
beats_from_paragraphs, clean_narration, infer_purposes, title_from_text.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Optional

import yaml

from ..models import Beat, Emphasis, Scene, ScenePurpose, Script, VideoFormat
from ..util import no_em_dash

BEAT_MIN_WORDS = 12
BEAT_MAX_WORDS = 45
BEAT_MAX_SENTENCES = 3
SCENE_MIN_WORDS = 120
SCENE_MAX_WORDS = 250
SCENE_TARGET_WORDS = 200

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "jr", "sr", "gen", "col", "lt", "capt", "cpt", "sgt", "maj", "adm",
    "gov", "sen", "rep", "pres", "prof", "rev", "hon", "mt", "ft", "no", "nos", "vs", "etc", "e.g", "i.e",
    "inc", "co", "corp", "ltd", "bros", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "approx", "ave", "blvd", "rd", "dept", "est", "fig", "vol", "op", "cf", "al",
    "a.m", "p.m", "u.s", "u.k", "u.n", "d.c", "u.s.a", "s.s", "h.m.s", "u.s.s", "ca", "c",
}

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
         "xi": 11, "xii": 12}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
                "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20}

DIRECTION_WORDS = (
    "music", "sfx", "sound", "pause", "beat", "cut", "fade", "b-roll", "broll", "b roll", "visual", "on screen",
    "onscreen", "footage", "narrator", "vo", "v.o", "cue", "transition", "image", "graphic", "map", "insert",
    "text", "title", "lower third", "zoom", "pan", "shot", "scene", "intro", "outro", "stock", "clip", "archival",
    "photo", "animation", "show", "sting", "swell", "silence", "ambient", "drone", "timestamp", "chapter",
)
DIRECTION_LINE = re.compile(
    r"^\s*(visuals?|b[- ]?roll|sfx|sound|music|on[- ]?screen|text on screen|graphics?|image|footage|shot|"
    r"cut to|fade (in|out|to)|note|director'?s? note|editor'?s? note|camera|transition|lower third|caption|"
    r"thumbnail|title card|map|animation|overlay)\s*[:\-]", re.I)
SPEAKER_PREFIX = re.compile(r"^\s*(narrator|narration|vo|v\.o\.|voice ?over|host)\s*(\([^)]*\))?\s*:\s*", re.I)
TIMESTAMP = r"\(?\[?\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\]?\)?"
LEAD_TIMESTAMP = re.compile(
    rf"^\s*{TIMESTAMP}(?:\s*(?:-|to|–)\s*{TIMESTAMP})?(?:\s*[:\-|.]\s*|\s+(?=[A-Z\"“(\[])|\s*$)")
TOP10_MARKER = re.compile(
    r"^\s*(?:#\s?(?P<a>\d{1,2})\b|(?:number|num\.?|no\.?)\s*(?P<b>\d{1,2}|[a-z]+)\b|(?P<c>\d{1,2})[.)](?=\s|$))"
    r"\s*[:.)\-–]?\s*(?P<rest>.*)$", re.I)
SCENE_HEADING = re.compile(
    r"^\s*(scene|part|chapter|act|section|segment|episode)\s+(\d{1,3}|[ivxlc]+|[a-z]+)\b\s*[:.\-–]?\s*(?P<rest>.*)$",
    re.I)


@dataclass
class _Section:
    title: str = ""
    item_number: Optional[int] = None
    paragraphs: list[str] = field(default_factory=list)
    explicit: bool = False  # came from a heading

    def words(self) -> int:
        return sum(len(p.split()) for p in self.paragraphs)


# ---------------------------------------------------------------- entry point

def parse_script(text: str, fmt: Optional[VideoFormat | str] = None) -> Script:
    """Parse pasted script text into a Script. fmt="top10" forces top10; top10 markers also set it."""
    meta, body = _front_matter(text)
    title, sections, top10 = _sections(body)
    is_top10 = top10 or _wanted_format(meta.get("format") or fmt) == VideoFormat.top10
    if not any(s.explicit for s in sections):
        sections = _group_paragraphs(sections)
    sections = [s for s in sections if s.paragraphs]
    if not sections:
        sections = [_Section(title="Introduction", paragraphs=[clean_narration(body) or "No narration found."])]
    title = str(meta.get("title") or title or _title_from_first_sentence(sections))
    if is_top10 and not any(s.item_number for s in sections):
        _assign_items(sections)
    if is_top10:
        sections = _split_closing(sections)
    if sections and not sections[0].title and any(s.explicit for s in sections[1:]):
        sections[0].title = "Introduction"
    scenes = _build_scenes(sections, is_top10, title)
    return Script(title=no_em_dash(title).strip(), format=VideoFormat.top10 if is_top10 else VideoFormat.documentary,
                  scenes=scenes)


def _wanted_format(value: object) -> VideoFormat:
    try:
        return VideoFormat(value) if value else VideoFormat.documentary
    except ValueError:
        return VideoFormat.documentary


def _assign_items(sections: list[_Section]) -> None:
    """Forced top10 without markers: middle sections become a countdown ending at 1."""
    n = len(sections)
    lo = 1 if n > 2 else 0
    hi = n - 1 if n > 3 else n
    mids = sections[lo:hi]
    for k, sec in enumerate(mids):
        sec.item_number = len(mids) - k


CLOSING = re.compile(r"\b(that'?s our list|that is our list|thanks for watching|thank you for watching|subscribe|"
                     r"to recap|in summary|let us know|in the comments|which one|until next time)\b", re.I)


def _split_closing(sections: list[_Section]) -> list[_Section]:
    """Top10: trailing closing paragraphs of the last item become an outro section."""
    items = [i for i, s in enumerate(sections) if s.item_number is not None]
    if not items or items[-1] != len(sections) - 1:
        return sections
    last = sections[-1]
    k = len(last.paragraphs)
    while k > 1 and CLOSING.search(last.paragraphs[k - 1]):
        k -= 1
    if k == len(last.paragraphs):
        return sections
    outro = _Section(title="Final Thoughts", paragraphs=last.paragraphs[k:], explicit=True)
    last.paragraphs = last.paragraphs[:k]
    return sections + [outro]


def _front_matter(text: str) -> tuple[dict, str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    m = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)(.*)$", text, re.S)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), m.group(2)


# ---------------------------------------------------------------- line classification

def _sections(body: str) -> tuple[str, list[_Section], bool]:
    """Split body into sections by headings. Returns (title, sections, is_top10)."""
    lines = _strip_block_noise(body).split("\n")
    markers = _top10_marker_lines(lines)
    title = ""
    sections: list[_Section] = [_Section()]
    para: list[str] = []

    def flush() -> None:
        if para:
            txt = clean_narration("\n".join(para))
            if txt:
                sections[-1].paragraphs.append(txt)
            para.clear()

    for i, raw in enumerate(lines):
        line = raw.strip()
        h1 = re.match(r"^#\s+(.+)$", line)
        if h1 and not title and not markers.get(i):
            flush()
            title = _clean_heading(h1.group(1))
            continue
        head = _heading(line, markers.get(i))
        if head is not None:
            flush()
            htitle, num, rest = head
            sections.append(_Section(title=htitle, item_number=num, explicit=True))
            if rest:
                para.append(rest)
            continue
        if not line or re.fullmatch(r"[-*_=]{3,}", line):
            flush()
            continue
        para.append(line)
    flush()
    return title, sections, bool(markers)


def _strip_block_noise(body: str) -> str:
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.S)
    body = re.sub(r"```.*?```", " ", body, flags=re.S)
    return body


def _top10_marker_lines(lines: list[str]) -> dict[int, tuple[int, str]]:
    """Line index -> (item number, rest) when the script has a real countdown (3+ distinct items)."""
    found: dict[int, tuple[int, str]] = {}
    for i, raw in enumerate(lines):
        line = re.sub(r"^#{2,6}\s+", "", raw.strip())
        line = re.sub(r"^\*\*(.+?)\*\*", r"\1", line)
        m = TOP10_MARKER.match(line)
        if not m:
            continue
        num = _marker_number(m)
        if num is None or not 1 <= num <= 20:
            continue
        found[i] = (num, m.group("rest").strip())
    nums = [n for n, _ in found.values()]
    if len(set(nums)) < 3 or len(set(nums)) != len(nums):
        return {}
    return found


def _marker_number(m: re.Match) -> Optional[int]:
    for g in ("a", "b", "c"):
        v = m.group(g)
        if v:
            v = v.lower()
            if v.isdigit():
                return int(v)
            return NUMBER_WORDS.get(v)
    return None


def _heading(line: str, marker: Optional[tuple[int, str]]) -> Optional[tuple[str, Optional[int], str]]:
    """(title, item_number, leftover narration) when the line is a heading, else None."""
    if marker:
        num, rest = marker
        rest = _clean_heading(rest)
        if len(rest.split()) <= 12 and not re.search(r"[.!?]\s+\S", rest):
            return rest.rstrip(".") or f"Number {num}", num, ""
        lead = re.split(r"[,.;:!?]|\s(?:is|was|which|who|that|where)\s", rest, maxsplit=1)[0]
        return " ".join(lead.split()[:6]) or f"Number {num}", num, rest
    md = re.match(r"^#{1,6}\s+(.+)$", line)
    if md:
        return _scene_heading_title(_clean_heading(md.group(1))), None, ""
    if not line or len(line.split()) > 12:
        return None
    sm = SCENE_HEADING.match(_clean_heading(line))
    if sm and (not sm.group("rest") or len(sm.group("rest").split()) <= 10) and _numberish(sm.group(2)):
        return _scene_heading_title(_clean_heading(line)), None, ""
    if _is_caps_heading(line):
        return _scene_heading_title(_clean_heading(line)), None, ""
    return None


def _numberish(tok: str) -> bool:
    t = tok.lower()
    return t.isdigit() or t in ROMAN or t in NUMBER_WORDS or t in {"first", "second", "third", "final", "last"}


def _scene_heading_title(text: str) -> str:
    """Drop the "Scene 3:" or "Part II -" prefix when a real title follows it; title-case ALL CAPS."""
    sm = SCENE_HEADING.match(text)
    if sm and sm.group("rest").strip() and _numberish(sm.group(2)):
        text = sm.group("rest").strip()
    if not text.isupper():
        return text
    words = [w.upper() if w.lower().strip(":.,") in ROMAN else w for w in string.capwords(text).split()]
    return " ".join(words)


def _is_caps_heading(line: str) -> bool:
    s = _clean_heading(line)
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 3 or s.endswith((",", ";")):
        return False
    if SPEAKER_PREFIX.match(s) or DIRECTION_LINE.match(s):
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) > 0.9 and len(s.split()) <= 10


def _clean_heading(text: str) -> str:
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"\s*#+\s*$", "", text)
    text = no_em_dash(text).strip().strip(":").strip()
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------- cleaning

def clean_narration(text: str) -> str:
    """Strip markdown, stage directions and timestamps, leaving speakable narration."""
    out_lines = []
    for line in text.split("\n"):
        line = line.strip()
        if DIRECTION_LINE.match(line):
            continue
        line = LEAD_TIMESTAMP.sub("", line)
        line = SPEAKER_PREFIX.sub("", line)
        line = re.sub(r"^\s*(>+|[-*+•]\s+)", "", line)
        out_lines.append(line)
    text = " ".join(out_lines)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\((?:https?://|/)[^)]*\)", r"\1", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\(([^()]*)\)", _paren_direction, text)
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r"(\*\*|__|\*|`)", "", text)
    text = re.sub(r"(?<!\w)_(\S[^_]*\S|\S)_(?!\w)", r"\1", text)
    text = no_em_dash(text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",\s*,", ",", text)
    return re.sub(r"\s+", " ", text).strip()


def _paren_direction(m: re.Match) -> str:
    inner = m.group(1).strip()
    low = inner.lower()
    if not inner or re.fullmatch(r"[\d:.,\s\-]+", inner):
        return " "
    if any(re.match(rf"{re.escape(w)}s?\b", low) for w in DIRECTION_WORDS):
        return " "
    if inner.isupper() and len(inner) > 2:
        return " "
    return m.group(0)


# ---------------------------------------------------------------- sentences and beats

def split_sentences(text: str) -> list[str]:
    """Split at sentence boundaries without breaking abbreviations, initials or decimals."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out, start = [], 0
    for m in re.finditer(r"[.!?]+[\"'”’)\]]*\s+(?=[\"'“‘(\[]?[A-Z0-9])", text):
        end = m.start() + len(m.group(0).rstrip())
        if _is_abbreviation(text[start:m.start() + 1], text[m.end():]):
            continue
        out.append(text[start:end].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


def _is_abbreviation(before: str, after: str) -> bool:
    """True when the period ending `before` belongs to an abbreviation, not a sentence end."""
    if not before.endswith("."):
        return False
    tok = before.split()[-1] if before.split() else ""
    word = tok.rstrip(".").lstrip("(\"'“").lower()
    if word in ABBREVIATIONS:
        if word in {"etc", "inc", "co", "corp", "ltd", "est", "al"}:
            return False
        if word in {"no", "nos", "op", "vol", "fig"}:
            return bool(re.match(r"\d", after))
        return True
    if re.fullmatch(r"[A-HJ-Z]", tok.rstrip(".")):
        return True  # initials: John D. Rockefeller
    return bool(re.fullmatch(r"(?:[A-Za-z]\.){2,}", tok))


def make_beat_texts(sentences: list[str], min_words: int = BEAT_MIN_WORDS, max_words: int = BEAT_MAX_WORDS,
                    breaks: Optional[set[int]] = None) -> list[str]:
    """Group sentences into beats of 1 to 3 sentences and about min..max words.

    breaks holds indices of sentences that start a new paragraph; a beat ends
    there when it already has min_words.
    """
    breaks = breaks or set()
    pieces: list[tuple[str, bool]] = []
    for i, s in enumerate(sentences):
        for k, part in enumerate(_split_long_sentence(s, max_words)):
            pieces.append((part, k == 0 and i in breaks))
    beats: list[list[str]] = []
    cur: list[str] = []
    for s, para_start in pieces:
        cw = sum(len(x.split()) for x in cur)
        sw = len(s.split())
        full = cw + sw > max_words or len(cur) >= BEAT_MAX_SENTENCES or (cw >= 22 and cw + sw > 34)
        if cur and (full or (para_start and cw >= min_words)):
            beats.append(cur)
            cur = []
        cur.append(s)
    if cur:
        beats.append(cur)
    return [" ".join(b) for b in _merge_short(beats, min_words, max_words)]


def beats_from_paragraphs(paragraphs: list[str]) -> list[str]:
    """Beat texts for a scene, preferring paragraph boundaries as beat boundaries."""
    sentences: list[str] = []
    breaks: set[int] = set()
    for p in paragraphs:
        breaks.add(len(sentences))
        sentences.extend(split_sentences(p))
    return make_beat_texts(sentences, breaks=breaks)


def _merge_short(beats: list[list[str]], min_words: int, max_words: int) -> list[list[str]]:
    out: list[list[str]] = []
    for b in beats:
        bw = sum(len(x.split()) for x in b)
        if out:
            pw = sum(len(x.split()) for x in out[-1])
            short = bw < min_words or pw < min_words
            if short and pw + bw <= max_words and len(out[-1]) + len(b) <= BEAT_MAX_SENTENCES:
                out[-1] = out[-1] + b
                continue
        out.append(b)
    return out


def _split_long_sentence(sentence: str, max_words: int) -> list[str]:
    """Split a sentence above max_words at the clause break nearest its middle."""
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    mid = len(words) / 2
    best, best_d = None, None
    for i in range(BEAT_MIN_WORDS, len(words) - BEAT_MIN_WORDS + 1):
        w = words[i - 1]
        nxt = words[i].lower() if i < len(words) else ""
        if w.endswith((",", ";", ":")) or nxt in {"and", "but", "while", "which", "because", "although", "so"}:
            d = abs(i - mid)
            if best_d is None or d < best_d:
                best, best_d = i, d
    if best is None:
        best = int(mid)
    left = " ".join(words[:best]).rstrip(",;:")
    right = " ".join(words[best:])
    return _split_long_sentence(left, max_words) + _split_long_sentence(right, max_words)


# ---------------------------------------------------------------- grouping and scenes

def _group_paragraphs(sections: list[_Section]) -> list[_Section]:
    """No headings: group paragraphs into scenes of roughly 120 to 250 words."""
    paras: list[str] = []
    for s in sections:
        for p in s.paragraphs:
            paras.extend(_split_paragraph(p))
    groups: list[list[str]] = []
    cur: list[str] = []
    for p in paras:
        cw = sum(len(x.split()) for x in cur)
        pw = len(p.split())
        if cur and ((cw >= SCENE_MIN_WORDS and cw + pw > SCENE_TARGET_WORDS) or cw + pw > SCENE_MAX_WORDS):
            groups.append(cur)
            cur = []
        cur.append(p)
    if cur:
        if groups and sum(len(x.split()) for x in cur) < SCENE_MIN_WORDS / 2:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    return [_Section(paragraphs=g) for g in groups]


def _split_paragraph(p: str) -> list[str]:
    """Break a paragraph longer than the scene cap into sentence chunks of about 180 words."""
    if len(p.split()) <= SCENE_MAX_WORDS:
        return [p]
    chunks, cur = [], []
    for s in split_sentences(p):
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= 180:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _build_scenes(sections: list[_Section], top10: bool, script_title: str = "") -> list[Scene]:
    weights = [s.words() for s in sections]
    titles = _scene_titles(sections, script_title)
    if top10:
        purposes = _top10_purposes(sections)
    else:
        purposes = infer_purposes(weights, titles)
    scenes, n = [], 0
    for i, (sec, purpose) in enumerate(zip(sections, purposes, strict=True)):
        beats = []
        for txt in beats_from_paragraphs(sec.paragraphs):
            n += 1
            beats.append(Beat(id=f"b{n:03d}", text=txt, emphasis=_emphasis(txt, purpose, not beats)))
        scenes.append(Scene(id=f"s{i + 1:02d}", title=titles[i], purpose=purpose, beats=beats,
                            item_number=sec.item_number if top10 else None))
    return scenes


def _emphasis(text: str, purpose: ScenePurpose, first: bool) -> Emphasis:
    low = text.lower()
    if purpose == ScenePurpose.outro and re.search(r"subscribe|thanks for watching|see you", low):
        return Emphasis.low
    if first and purpose in (ScenePurpose.hook, ScenePurpose.climax, ScenePurpose.turn, ScenePurpose.item):
        return Emphasis.high
    return Emphasis.high if "!" in text else Emphasis.normal


def _scene_titles(sections: list[_Section], script_title: str = "") -> list[str]:
    used: set[str] = {script_title.lower()} if script_title else set()
    out = []
    for i, s in enumerate(sections):
        t = s.title or title_from_text(" ".join(s.paragraphs), exclude=used) or f"Part {i + 1}"
        used.add(t.lower())
        out.append(t)
    return out


# ---------------------------------------------------------------- purposes

PURPOSE_KEYWORDS: list[tuple[str, ScenePurpose]] = [
    (r"\b(intro|introduction|cold open|hook|prologue|opening)\b", ScenePurpose.hook),
    (r"\b(outro|conclusion|closing|final thoughts|epilogue|wrap|recap|summary|in summary)\b", ScenePurpose.outro),
    (r"\b(legacy|aftermath|afterward|today)\b", ScenePurpose.resolution),
    (r"\b(background|origins?|beginnings?)\b", ScenePurpose.setup),
]


def infer_purposes(weights: list[int], titles: Optional[list[str]] = None) -> list[ScenePurpose]:
    """Documentary purposes: hook, setup, acts, a turn near 55%, climax near 80%, resolution, outro."""
    n = len(weights)
    if n == 0:
        return []
    out: list[Optional[ScenePurpose]] = [None] * n
    out[0] = ScenePurpose.hook
    if n >= 2:
        out[-1] = ScenePurpose.outro
    if n >= 5:
        out[1] = ScenePurpose.setup
    _keyword_purposes(out, titles or [])
    if n >= 7 and out[n - 2] is None and ScenePurpose.resolution not in out:
        out[n - 2] = ScenePurpose.resolution
    frac = _midpoints(weights)
    free = [i for i in range(n) if out[i] is None]
    if len(free) >= 2:
        climax = min(free, key=lambda i: abs(frac[i] - 0.8))
        out[climax] = ScenePurpose.climax
        free.remove(climax)
    if len(free) >= 2:
        turn = min(free, key=lambda i: abs(frac[i] - 0.55))
        out[turn] = ScenePurpose.turn
        free.remove(turn)
    return [p or ScenePurpose.act for p in out]


def _keyword_purposes(out: list[Optional[ScenePurpose]], titles: list[str]) -> None:
    n = len(out)
    for i, t in enumerate(titles):
        for pat, purpose in PURPOSE_KEYWORDS:
            if not re.search(pat, t.lower()):
                continue
            if purpose == ScenePurpose.hook and i > 1:
                continue
            if purpose in (ScenePurpose.setup, ScenePurpose.resolution) and i in (0, n - 1):
                continue
            if purpose == ScenePurpose.outro and i < n - 2:
                continue
            if purpose == ScenePurpose.outro and i == n - 2 and out[-1] == ScenePurpose.outro:
                out[i] = ScenePurpose.resolution
                break
            out[i] = purpose
            break


def _midpoints(weights: list[int]) -> list[float]:
    total = max(1, sum(weights))
    acc, out = 0, []
    for w in weights:
        out.append((acc + w / 2) / total)
        acc += w
    return out


def _top10_purposes(sections: list[_Section]) -> list[ScenePurpose]:
    idx = [i for i, s in enumerate(sections) if s.item_number is not None]
    first, last = (idx[0], idx[-1]) if idx else (len(sections), -1)
    out = []
    for i, s in enumerate(sections):
        if s.item_number is not None:
            out.append(ScenePurpose.item)
        elif i < first:
            out.append(ScenePurpose.hook if i == 0 else ScenePurpose.setup)
        elif i > last:
            out.append(ScenePurpose.outro)
        else:
            out.append(ScenePurpose.act)
    return out


# ---------------------------------------------------------------- titles

STOP_STARTS = {"The", "A", "An", "In", "On", "At", "By", "For", "But", "And", "It", "This", "That", "These",
               "Those", "There", "Then", "When", "If", "As", "Of", "To", "From", "With", "We", "You", "They", "He",
               "She", "His", "Her", "Our", "Its", "So", "Yet", "Now", "Today", "Every", "One", "Some", "Many",
               "Most", "No", "Not", "Number"}


MONTHS = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}


def title_from_text(text: str, exclude: Optional[set[str]] = None, max_words: int = 5) -> str:
    """Short title from text: its most frequent new proper noun phrase (multiword first), else opening words."""
    exclude = {e.lower() for e in (exclude or set())}
    counts: dict[str, int] = {}
    order: list[str] = []
    for m in re.finditer(r"\b([A-Z][A-Za-z'’]+(?:\s+(?:of|the|de|and|del|von|van)?\s*[A-Z][A-Za-z'’]+)*)", text):
        words = [w for w in m.group(1).split() if w not in MONTHS]
        while words and (words[0] in STOP_STARTS or words[0].lower() in {"of", "the", "and"}):
            words = words[1:]
        phrase = re.sub(r"['’]s$", "", " ".join(words[:max_words]))
        low = phrase.lower()
        if len(phrase) < 4 or any(re.search(rf"\b{re.escape(low)}\b", e) for e in exclude):
            continue
        if phrase not in counts:
            order.append(phrase)
        counts[phrase] = counts.get(phrase, 0) + 1
    if order:
        return max(order, key=lambda ph: (len(ph.split()) > 1, counts[ph], -order.index(ph)))
    first = split_sentences(text)[:1]
    words = (first[0] if first else text).split()[:max_words]
    return re.sub(r"[,.;:!?]+$", "", " ".join(words)).strip()


def _title_from_first_sentence(sections: list[_Section]) -> str:
    for s in sections:
        for p in s.paragraphs:
            sents = split_sentences(p)
            if sents:
                words = sents[0].split()
                t = " ".join(words[:10])
                return re.sub(r"[,.;:!?]+$", "", t)
    return "Untitled"
