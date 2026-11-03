"""Credits for the description: one line per unique sourced asset that is on screen.

attribution_lines(shotlist) lists selected candidates except motion graphics
and AI images, formatted "<title> by <author>, <licence>, <url>".
attribution_block(shotlist) wraps them in a markdown section with summary
lines for AI generated images and motion graphics.
"""
from __future__ import annotations

from ..models import Candidate, ShotList, SourceKind

GENERATED = {SourceKind.motion, SourceKind.ai_image, SourceKind.ai_video}


def format_line(c: Candidate) -> str:
    """'<title> by <author>, <licence>, <url>' with empty parts dropped."""
    head = c.title or "Untitled"
    if c.author:
        head = f"{head} by {c.author}"
    parts = [head]
    if c.licence and c.licence != "unknown":
        parts.append(c.licence)
    if c.url.startswith(("http://", "https://")):
        parts.append(c.url)
    return ", ".join(parts)


def _selected(shotlist: ShotList) -> list[Candidate]:
    out = []
    for shot in shotlist.shots:
        c = shot.selected_candidate()
        if c is not None:
            out.append(c)
    return out


def attribution_lines(shotlist: ShotList) -> list[str]:
    """Unique credit lines in order of first appearance."""
    seen: set[str] = set()
    lines: list[str] = []
    for c in _selected(shotlist):
        if c.source in GENERATED:
            continue
        key = c.url if c.url.startswith("http") else (c.local_path or c.title)
        line = format_line(c)
        if key in seen or line in lines:
            continue
        seen.add(key)
        lines.append(line)
    return lines


def attribution_block(shotlist: ShotList) -> str:
    """Markdown 'Footage and image credits' section."""
    lines = attribution_lines(shotlist)
    selected = _selected(shotlist)
    ai = sum(1 for c in selected if c.source in {SourceKind.ai_image, SourceKind.ai_video})
    motion = sum(1 for c in selected if c.source == SourceKind.motion)
    out = ["## Footage and image credits", ""]
    out += [f"- {ln}" for ln in lines] or ["- No third party footage or images."]
    out += ["", f"AI generated images: {ai} shots"]
    out.append(f"Motion graphics: generated ({motion} shots)" if motion else "Motion graphics: generated")
    return "\n".join(out) + "\n"
