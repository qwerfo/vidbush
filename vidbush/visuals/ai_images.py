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
               "use it to keep the subject true to what is said, never copy its names or text. Return JSON "
               '{"prompt": "<one paragraph, under 90 words>"}.')
# Subjects whose meaning is readable text: an image model invents the words, so these are never generated stills.
TEXT_SUBJECT_RE = re.compile(r"\b(maps?|headlines?|newspapers?|front pages?|documents?|letters?|telegrams?|"
                             r"statements?|web ?pages?|websites?|sheet music|musical scores?|charts?|graphs?|"
                             r"labels?)\b", re.I)
_KEEP_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


def needs_readable_text(shot: Shot) -> bool:
    """True when the shot's subject is text to be read (a labelled map, a headline, a document, sheet music)."""
    return bool(TEXT_SUBJECT_RE.search(" ".join([shot.intent, *shot.must_show])))


def _stem(word: str) -> str:
    if len(word) > 5 and word.endswith(("ing", "ed")):
        return word[:-3] if word.endswith("ing") else word[:-2]
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def lost_keep_words(old: str, new: str, keep: list[str]) -> list[str]:
    """Content words of keep items that the old prompt had and the new one lost. Capitalised words are names,
    which the rewrite removes on purpose."""
    from ..shots.ideas import STOP
    old_l, new_l = old.lower(), new.lower()
    words = [w for item in keep for w in _KEEP_WORD.findall(item) if not w[0].isupper() and w.lower() not in STOP]
    return list(dict.fromkeys(w for w in words if _stem(w.lower()) in old_l and _stem(w.lower()) not in new_l))


def safe_prompt(text: str, ctx: Any, keep: Optional[list[str]] = None, narration: str = "") -> str:
    """The prompt with named people, brands and readable text rewritten out by the editor seat. Image models
    render names as fake likenesses and text as invented, often wrong, facts. Unchanged when no model answers
    or when the rewrite dropped content of a keep item (a rewrite may change wording, never content)."""
    keep = keep or []
    user = "\n".join([f"Prompt: {text}"] + ([f"Must keep: {'; '.join(keep)}"] if keep else [])
                     + ([f"Narration: {narration}"] if narration else []))
    try:
        raw = ctx.seats.llm_json("editor", "ai_prompt_safety", SAFE_SYSTEM, user, ctx.stage or "judge", temperature=0.2)
    except Exception as e:  # noqa: BLE001 - the original prompt still goes out; the negatives still apply
        ctx.log.warn("ai prompt rewrite failed, original prompt kept", error=str(e)[:200])
        return text
    new = raw.get("prompt") if isinstance(raw, dict) else None
    if not (isinstance(new, str) and len(new) > 20):
        return text
    new = no_em_dash(" ".join(new.split()))
    lost = lost_keep_words(text, new, keep)
    if lost:
        ctx.log.warn("ai prompt rewrite dropped content, original prompt kept", lost=lost, rewrite=new[:300])
        return text
    return new


def generate_ai_candidate(shot: Shot, ctx: Any, prompt: Optional[str] = None) -> Candidate:
    """Generate one still for the shot and return it as an ai_image candidate (not appended). The rewrite is
    grounded in the beat's narration; every still after a shot's first is a retry in the cost report."""
    from ..judge.judge import narration_by_beat
    text = safe_prompt(build_prompt(shot, prompt), ctx, keep=shot.must_show,
                       narration=narration_by_beat(ctx).get(shot.beat_id, ""))
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
