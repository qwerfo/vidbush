"""Packaging: chapters.txt, description.md (titles, description, tags, chapters, credits) and thumbnails.

package(shotlist, ctx) -> list[Path]. Chapters start at the first narrated
word of each scene (plus the intro offset from work/assembly.json), the
first one at 00:00, each at least MIN_CHAPTER_S long, at least three when
the video is long enough. Titles, description and tags come from the writer
seat with deterministic heuristic fallbacks. Every generated text goes
through util.no_em_dash.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..models import Narration, Scene, Script, ShotList, SourceKind, VideoFormat
from ..util import no_em_dash, read_json

MIN_CHAPTER_S = 10.0
MAX_TITLE_CHARS = 70
DESC_MIN_WORDS, DESC_MAX_WORDS = 150, 250
TAGS_MAX_CHARS = 450
CLOSING = ("This is a long-form documentary built from published sources. If you enjoy deep dives like this one, "
           "subscribe for more, and tell us in the comments which part of the story surprised you most.")
GENERIC_HOOK = {"", "hook", "intro", "introduction", "cold open", "opening", "teaser", "prologue"}
SYSTEM = ("You write YouTube metadata for a long-form faceless documentary channel. Be accurate and "
          "curiosity driven, never clickbait, no emojis, no em dashes. Reply with strict JSON only, concise.")
LANGUAGE_NAMES = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
                  "pt": "Portuguese", "nl": "Dutch", "pl": "Polish", "sv": "Swedish", "ja": "Japanese",
                  "ko": "Korean", "zh": "Chinese", "ar": "Arabic", "hi": "Hindi", "ru": "Russian", "tr": "Turkish"}
YEAR_RE = re.compile(r"^(1[0-9]{3}|20[0-9]{2})$")


@dataclass
class Chapter:
    start: float
    title: str


def package(shotlist: ShotList, ctx: Any) -> list[Path]:
    """Write chapters.txt and description.md in ctx.root, then thumbnails. Returns all paths."""
    from .thumbnails import make_thumbnails
    script = ctx.maybe_load("script", Script)
    narration = ctx.maybe_load("narration", Narration)
    video_title = clean((script.title if script else "") or ctx.project.title or "Untitled")
    asm = ctx.work / "assembly.json"
    intro = float(read_json(asm).get("intro_duration", 0.0)) if asm.exists() else 0.0
    body = narration.duration if narration else shotlist.duration
    total = intro + max(body, shotlist.duration)
    chapters = build_chapters(script, narration, total, intro, video_title)
    chapters_txt = "\n".join(f"{fmt_ts(c.start, total)} {c.title}" for c in chapters) + "\n"
    ch_path = ctx.root / "chapters.txt"
    ch_path.write_text(chapters_txt, encoding="utf-8")
    titles, tags = make_titles(script, video_title, ctx)
    description = make_description(script, video_title, ctx)
    md = render_markdown(titles, description, chapters_txt, tags, credits_block(shotlist))
    desc_path = ctx.root / "description.md"
    desc_path.write_text(no_em_dash(md), encoding="utf-8")
    ctx.log.info("package written", chapters=len(chapters), titles=len(titles),
                 description_words=len(description.split()))
    return [ch_path, desc_path] + make_thumbnails(shotlist, titles, ctx)


def language_name(code: str) -> str:
    """Display name for a brand language code ('en' -> 'English'); unknown codes pass through."""
    code = (code or "en").strip()
    return LANGUAGE_NAMES.get(code.lower().split("-")[0], code)


def system_prompt(base: str, ctx: Any) -> str:
    """base plus the brand language instruction (every packaging prompt goes through this)."""
    brand = getattr(ctx, "brand", None)
    lang = language_name(getattr(brand, "language", "en") or "en")
    return f"{base} Write every text in {lang}."


def clean(text: str) -> str:
    """Single-line text with em dashes removed."""
    return re.sub(r"\s+", " ", no_em_dash(str(text))).strip()


# ---------------------------------------------------------------- chapters

def fmt_ts(t: float, total: float) -> str:
    """YouTube timestamp: MM:SS, or H:MM:SS for videos of an hour or more."""
    s = int(max(t, 0.0))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if total >= 3600 else f"{m:02d}:{sec:02d}"


def scene_title(scene: Scene, is_first: bool, video_title: str) -> str:
    """Chapter title for a scene: hook falls back to the video title, countdown items get their number."""
    title = clean(scene.title)
    if is_first and title.lower() in GENERIC_HOOK:
        title = video_title.split(":")[0].strip() or video_title
    if scene.item_number is not None and not re.search(rf"\b{scene.item_number}\b", title):
        title = f"#{scene.item_number} {title}".strip()
    return (title or "Introduction")[:80]


def _beat_start(beat_id: str, narration: Narration) -> Optional[float]:
    starts = [w.start for w in narration.words if w.beat_id == beat_id]
    if starts:
        return min(starts)
    span = narration.beat_spans.get(beat_id)
    return float(span[0]) if span else None


def scene_points(script: Script, narration: Optional[Narration], video_title: str) -> list[Chapter]:
    """(start, title) per scene from the first word of its first beat; scenes with no timing are skipped."""
    out: list[Chapter] = []
    for i, scene in enumerate(script.scenes):
        start = 0.0 if i == 0 else None
        if i and narration is not None and scene.beats:
            start = _beat_start(scene.beats[0].id, narration)
        if start is not None:
            out.append(Chapter(start, scene_title(scene, i == 0, video_title)))
    return out


def build_chapters(script: Optional[Script], narration: Optional[Narration], total: float, intro: float,
                   video_title: str) -> list[Chapter]:
    """YouTube chapters: first at 00:00, each at least MIN_CHAPTER_S, three or more when possible."""
    points = scene_points(script, narration, video_title) if script and script.scenes else []
    points = [Chapter(0.0 if i == 0 else p.start + intro, p.title) for i, p in enumerate(points)]
    if not points:
        points = [Chapter(0.0, video_title)]
    kept = [points[0]]
    for p in points[1:]:
        if p.start - kept[-1].start >= MIN_CHAPTER_S:
            kept.append(p)
    while len(kept) > 1 and total - kept[-1].start < MIN_CHAPTER_S:
        kept.pop()
    while len(kept) < 3 and total >= 3 * MIN_CHAPTER_S:
        if not _split_longest(kept, script, narration, total, intro):
            break
    return kept


def _split_longest(kept: list[Chapter], script: Optional[Script], narration: Optional[Narration], total: float,
                   intro: float) -> bool:
    """Add a chapter at the beat nearest the middle of the longest chapter. False when impossible."""
    ends = [c.start for c in kept[1:]] + [total]
    i = max(range(len(kept)), key=lambda k: ends[k] - kept[k].start)
    lo, hi = kept[i].start + MIN_CHAPTER_S, ends[i] - MIN_CHAPTER_S
    if hi < lo:
        return False
    mid = (kept[i].start + ends[i]) / 2
    best: Optional[Chapter] = None
    for beat in (script.beats() if script else []):
        t = _beat_start(beat.id, narration) if narration else None
        if t is None or not lo <= t + intro <= hi:
            continue
        if best is None or abs(t + intro - mid) < abs(best.start - mid):
            best = Chapter(t + intro, phrase_title(beat.text))
    if best is None:
        best = Chapter((lo + hi) / 2, f"{kept[i].title}, Part 2")
    kept.insert(i + 1, best)
    return True


def phrase_title(text: str, words: int = 5) -> str:
    """Short title-cased phrase from the first clause of a sentence."""
    first = re.split(r"[.,;:!?]", clean(text))[0].split()[:words]
    return " ".join(w if w[:1].isupper() else w.capitalize() for w in first) or "Continued"


# ---------------------------------------------------------------- titles and tags

def _context(script: Optional[Script], video_title: str) -> str:
    scenes = [s.title for s in script.scenes] if script else []
    hook = " ".join(b.text for b in script.scenes[0].beats[:3]) if script and script.scenes else ""
    fmt = script.format.value if script else "documentary"
    return (f"Video title: {video_title}\nFormat: {fmt}\nChapters: {'; '.join(scenes)}\n"
            f"Opening narration: {hook[:700]}")


def make_titles(script: Optional[Script], video_title: str, ctx: Any) -> tuple[list[str], list[str]]:
    """Three title options under MAX_TITLE_CHARS and a tag list (writer seat, heuristic fallback)."""
    user = (_context(script, video_title) + '\nReturn {"titles": [3 distinct titles under 70 characters], '
            '"tags": [10 to 15 lowercase search tags]}.')
    data = ctx.seats.llm_json("writer", "titles", system_prompt(SYSTEM, ctx), user, stage="packaging")
    titles: list[str] = []
    tags: list[str] = []
    if isinstance(data, dict):
        titles = [t for t in (fit_title(clean(x)) for x in data.get("titles") or [] if isinstance(x, str)) if t]
        tags = [clean(x).lower() for x in data.get("tags") or [] if isinstance(x, str) and x.strip()]
    for t in heuristic_titles(script, video_title):
        if len(titles) >= 3:
            break
        if t.lower() not in {x.lower() for x in titles}:
            titles.append(t)
    return titles[:3], limit_tags(tags or heuristic_tags(script, video_title))


def fit_title(title: str) -> str:
    """Title within MAX_TITLE_CHARS: cut after a colon or at a word boundary, or '' when hopeless."""
    if len(title) <= MAX_TITLE_CHARS:
        return title
    head = title.split(":")[0].strip()
    if 10 <= len(head) <= MAX_TITLE_CHARS:
        return head
    cut = title[:MAX_TITLE_CHARS].rsplit(" ", 1)[0].rstrip(",;:")
    return cut if len(cut) >= 20 else ""


def key_phrases(script: Optional[Script]) -> list[str]:
    """Recurring proper names (two or more capitalised words) and notable numbers, most frequent first."""
    if not script:
        return []
    text = " ".join(b.text for b in script.beats())
    names = re.findall(r"(?<![.!?]\s)\b([A-Z][a-z]+(?:\s(?:of\s|de\s)?[A-Z][a-z]+)+)", text)
    counts: dict[str, int] = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    return sorted(counts, key=lambda k: (-counts[k], text.index(k)))[:6]


def inner_topic(topic: str) -> str:
    """Topic for use inside a title: a leading 'The' becomes 'the'."""
    return re.sub(r"^The\s+", "the ", topic)


def number_phrase(script: Optional[Script]) -> str:
    """Largest non-year number in the script with the word after it, title cased ('22,000 Workers')."""
    if not script:
        return ""
    best: tuple[float, str] = (0.0, "")
    text = " ".join(b.text for b in script.beats())
    for m in re.finditer(r"\b(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s+([a-z][a-z-]+)\b", text):
        num, word = m.group(1), m.group(2)
        if YEAR_RE.match(num) or word in {"and", "or", "the", "a", "an", "to", "of", "in", "on", "at", "by"}:
            continue
        value = float(num.replace(",", ""))
        if value >= 10 and value > best[0]:
            best = (value, f"{num} {word.capitalize()}")
    return best[1]


def heuristic_titles(script: Optional[Script], video_title: str) -> list[str]:
    """Honest, fact-led title variants from the script title and its biggest number."""
    topic = video_title.split(":")[0].strip() or video_title
    inner = inner_topic(topic)
    fact = number_phrase(script)
    if script and script.format == VideoFormat.top10:
        opts = [video_title, f"{topic}, Ranked", f"{topic}: The Full Countdown"]
    else:
        opts = [video_title]
        if fact:
            opts.append(f"{fact}: The Story of {inner}")
        opts += [f"{topic}: How It Really Happened", f"Why {inner} Still Matters"]
    out: list[str] = []
    for t in (fit_title(clean(o)) for o in opts):
        if t and t.lower() not in {x.lower() for x in out}:
            out.append(t)
    return out


def heuristic_tags(script: Optional[Script], video_title: str) -> list[str]:
    topic = video_title.split(":")[0].strip().lower()
    tags = [topic, f"{topic} history", f"{topic} documentary"]
    tags += [n.lower() for n in key_phrases(script)]
    tags += ["top 10", "countdown"] if script and script.format == VideoFormat.top10 else ["history documentary",
                                                                                          "documentary"]
    return tags


def limit_tags(tags: list[str]) -> list[str]:
    """Unique tags whose comma joined length stays under the YouTube limit."""
    out: list[str] = []
    used = 0
    for t in tags:
        t = re.sub(r"[<>,#]", "", t).strip()
        if not t or t in out or used + len(t) + 2 > TAGS_MAX_CHARS:
            continue
        out.append(t)
        used += len(t) + 2
    return out


# ---------------------------------------------------------------- description

def make_description(script: Optional[Script], video_title: str, ctx: Any) -> str:
    """150 to 250 word description (writer seat, heuristic fallback, then fitted to the range)."""
    user = (_context(script, video_title) + '\nReturn {"description": "150 to 250 words in 2 or 3 short '
            'paragraphs: open with the most intriguing fact, then what the video covers. No hashtags, no links."}')
    data = ctx.seats.llm_json("writer", "description", system_prompt(SYSTEM, ctx), user, stage="packaging")
    text = data.get("description", "") if isinstance(data, dict) else ""
    text = no_em_dash(text.strip()) if isinstance(text, str) else ""
    if not text:
        text = heuristic_description(script, video_title)
    return fit_words(text, script)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean(text)) if s.strip()]


def heuristic_description(script: Optional[Script], video_title: str) -> str:
    """Hook sentences, a sentence naming the chapters, and the closing thought."""
    if not script or not script.scenes:
        return f"{video_title}. A long-form documentary."
    hook = " ".join(_sentences(" ".join(b.text for b in script.scenes[0].beats))[:3])
    names = name_vocab(script)
    titles = [inline_title(clean(s.title), names) for s in script.scenes[1:]
              if clean(s.title).lower() not in GENERIC_HOOK]
    if script.format == VideoFormat.top10:
        covers = "We count down " + _join(titles) + "." if titles else ""
    else:
        covers = f"We follow {_join(titles, ', then ')}." if titles else ""
    last = script.scenes[-1]
    closing = " ".join(_sentences(" ".join(b.text for b in last.beats))[-2:]) if len(script.scenes) > 1 else ""
    return "\n\n".join(p for p in (hook, covers, closing) if p)


def _join(items: list[str], last: str = " and ") -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + last + items[-1]


def name_vocab(script: Script) -> set[str]:
    """Words the narration capitalises mid-sentence (names), lower cased."""
    text = " ".join(b.text for b in script.beats())
    return {m.group(1).lower() for m in re.finditer(r"(?<![.!?])\s([A-Z][A-Za-z'-]+)", text)}


def inline_title(title: str, names: set[str]) -> str:
    """Scene title for use mid-sentence: words lower cased except names and acronyms."""
    out = []
    for w in title.split():
        core = re.sub(r"[^A-Za-z'-]", "", w)
        keep = not core or core.isupper() or core.lower() in names
        out.append(w if keep else w.lower())
    return " ".join(out).rstrip(".")


def fit_words(text: str, script: Optional[Script]) -> str:
    """Trim at a sentence end to DESC_MAX_WORDS, or extend with scene sentences to DESC_MIN_WORDS."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: list[str] = []
    count = 0
    for para in paras:
        kept = []
        for s in _sentences(para):
            n = len(s.split())
            if count + n > DESC_MAX_WORDS:
                break
            kept.append(s)
            count += n
        if kept:
            out.append(" ".join(kept))
    scenes = script.scenes if script else []
    extra = [s for sc in scenes for b in sc.beats[:2] for s in _sentences(b.text)]
    extra += [s for sc in scenes for b in sc.beats[2:] for s in _sentences(b.text)]
    added: list[str] = []
    for s in extra:
        if count >= DESC_MIN_WORDS:
            break
        if s in " ".join(out) or count + len(s.split()) > DESC_MAX_WORDS:
            continue
        added.append(s)
        count += len(s.split())
    if added:
        out.append(" ".join(added))
    if count < DESC_MIN_WORDS:
        out.append(CLOSING)
    return "\n\n".join(out)


# ---------------------------------------------------------------- credits and markdown

def credits_block(shotlist: ShotList) -> str:
    """Attribution section from source.attribution, or a local block from selected candidates."""
    try:
        from ..source.attribution import attribution_block
        return attribution_block(shotlist)
    except ImportError:
        pass
    lines: list[str] = []
    for shot in shotlist.shots:
        c = shot.selected_candidate()
        if c is None or c.source in {SourceKind.motion, SourceKind.ai_image, SourceKind.ai_video}:
            continue
        head = f"{c.title or 'Untitled'} by {c.author}" if c.author else (c.title or "Untitled")
        line = c.attribution or ", ".join(p for p in (head, c.licence, c.url) if p and p != "unknown")
        if line not in lines:
            lines.append(line)
    body = [f"- {ln}" for ln in lines] or ["- No third party footage or images."]
    return "\n".join(["## Footage and image credits", "", *body]) + "\n"


def render_markdown(titles: list[str], description: str, chapters_txt: str, tags: list[str], credits: str) -> str:
    opts = "\n".join(f"{i}. {t}" for i, t in enumerate(titles, 1))
    return (f"# {titles[0] if titles else 'Untitled'}\n\n## Title options\n\n{opts}\n\n## Description\n\n"
            f"{description}\n\n## Chapters\n\n{chapters_txt}\n## Tags\n\n{', '.join(tags)}\n\n{credits}")

