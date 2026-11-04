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
            base += f" Clearly showing {', '.join(shot.must_show[:4])}."
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


def generate_ai_candidate(shot: Shot, ctx: Any, prompt: Optional[str] = None) -> Candidate:
    """Generate one still for the shot and return it as an ai_image candidate (not appended)."""
    text = build_prompt(shot, prompt)
    n = next_ai_index(shot, ctx.work)
    out = ctx.work / "ai" / f"{shot.id}_{n}.png"
    path, is_mock = _call_image(ctx, text, negative_prompt(shot, prompt), out)
    width, height = _size(path)
    ctx.log.info("ai image generated", shot=shot.id, mock=is_mock, path=str(path))
    return Candidate(id=f"{shot.id}_ai_image_{n}", source=SourceKind.ai_image, local_path=str(path), media="image",
                     width=width, height=height, licence="AI generated", attribution="AI generated image",
                     title=text[:200], keyframes=[str(path)], flags=["mock"] if is_mock else [])


def _call_image(ctx: Any, text: str, negative: str, out: Path) -> tuple[Path, bool]:
    """Image seat call; the negative prompt is passed when the seat accepts it (fal and Runware use it)."""
    try:
        accepts = "negative_prompt" in inspect.signature(ctx.seats.image).parameters
    except (TypeError, ValueError):
        accepts = False
    if accepts:
        return ctx.seats.image(text, out, ctx.stage or "judge", negative_prompt=negative)
    return ctx.seats.image(text, out, ctx.stage or "judge")


def _size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as im:
            return im.size
    except (OSError, ValueError):
        return 0, 0
