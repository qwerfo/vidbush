"""AI image candidates through the image seat.

generate_ai_candidate(shot, ctx, prompt=None) -> Candidate writes
work/ai/<shot_id>_<n>.png and returns Candidate(source=ai_image). The seat
handles the sticky provider and the per image cost ceiling.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from ..models import Candidate, Shot, SourceKind
from ..util import no_em_dash

HOUSE_STYLE = ("Photographic documentary still, natural light, realistic detail, era-correct clothing, vehicles "
               "and architecture, wide 16:9 composition, no text, no captions, no watermark, no logos.")
_LEADING = re.compile(r"^(archival footage|footage|stock footage|historical photograph|photograph|photo|image|"
                      r"documentary photograph|wide establishing (archival )?footage|close detail photograph)"
                      r"\s+of\s+", re.I)


def build_prompt(shot: Shot, prompt: Optional[str] = None) -> str:
    """Image prompt: explicit prompt or shot.ai_prompt or intent plus must_show, house style, plain negatives."""
    base = (prompt or shot.ai_prompt or "").strip()
    if not base:
        subject = _LEADING.sub("", shot.intent.strip().rstrip("."))
        base = subject[:1].upper() + subject[1:] + "." if subject else "Documentary scene."
        if shot.must_show:
            base += f" Clearly showing {', '.join(shot.must_show[:4])}."
    if "no text" not in base.lower():
        base = f"{base.rstrip()} {HOUSE_STYLE}"
    if shot.must_avoid:
        base += f" Do not include {', '.join(shot.must_avoid[:6])}."
    return no_em_dash(" ".join(base.split()))


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
    path, is_mock = ctx.seats.image(text, out, ctx.stage or "judge")
    width, height = _size(path)
    ctx.log.info("ai image generated", shot=shot.id, mock=is_mock, path=str(path))
    return Candidate(id=f"{shot.id}_ai_image_{n}", source=SourceKind.ai_image, local_path=str(path), media="image",
                     width=width, height=height, licence="AI generated", attribution="AI generated image",
                     title=text[:200], keyframes=[str(path)], flags=["mock"] if is_mock else [])


def _size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as im:
            return im.size
    except (OSError, ValueError):
        return 0, 0
