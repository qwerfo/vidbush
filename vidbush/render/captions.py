"""Captions: one line, min_words to max_words, at a fixed position in the bottom band.

group_words chooses line breaks with a small dynamic program (prefer sentence
and clause ends and real pauses, avoid dangling function words, never leave a
one or two word orphan). caption_geometry places the line in the letterbox band
(or a virtual band of the same height when letterbox is off) and sizes the font
so a line never wraps. ASS output uses {\\an5\\pos(x,y)} on every event, so the
vertical position is identical on every line, and WrapStyle 2 (no wrapping).

Font sizes are ASS font sizes: libass scales a font so that its Windows ascent
plus descent equals the size, so Pillow measures at the matching em size.
"""
from __future__ import annotations

import functools
import re
import struct
from pathlib import Path
from typing import Any, Optional

from PIL import ImageFont
from pydantic import BaseModel, Field

from ..config import Settings
from ..models import CaptionStyle, Narration, StyleProfile, Word
from ..util import read_json, write_json

HOLD_GAP_S = 0.3  # a line stays up until the next one when the gap is shorter than this
SIDE_MARGIN = 0.06  # fraction of width kept clear left and right
WEAK_END = {"the", "a", "an", "of", "to", "and", "in", "on", "at", "for", "with", "by", "from", "as", "or",
            "but", "that", "its", "their", "his", "her", "our", "is", "was", "were", "into", "than", "this",
            "it", "they", "we", "he", "she", "i", "you", "not", "be", "has", "had", "have", "will", "would",
            "could", "should", "very", "most", "more", "so", "if", "when", "who", "which", "where"}
_SENT_END = re.compile(r"[.!?]['\")\]]*$")
_CLAUSE_END = re.compile(r"[,;:]['\")\]]*$")


class CaptionLine(BaseModel):
    """One caption event: display text, timing and the narration words it covers."""

    text: str
    start: float
    end: float
    words: list[Word] = Field(default_factory=list)


# ---------------------------------------------------------------- grouping

def _line_cost(words: list[Word], i: int, j: int, n_total: int) -> float:
    """Cost of a line made of words[i:j]."""
    n = j - i
    cost = abs(n - 4) * 0.15
    last = words[j - 1].text
    if j < n_total:
        cost += 0.0 if _SENT_END.search(last) else 0.35 if _CLAUSE_END.search(last) else 1.0
        if words[j].start - words[j - 1].end >= 0.35:
            cost -= 0.5
        if re.sub(r"\W", "", last.lower()) in WEAK_END and not _CLAUSE_END.search(last):
            cost += 0.6
    for k in range(i, j - 1):
        t = words[k].text
        if _SENT_END.search(t):
            cost += 1.2
        elif _CLAUSE_END.search(t):
            cost += 0.3
        if words[k + 1].start - words[k].end >= 0.5:
            cost += 2.0
        if words[k + 1].beat_id != words[k].beat_id:
            cost += 0.8
    dur = words[j - 1].end - words[i].start
    return cost + max(0.0, dur - 3.2) * 0.5


def _partition(words: list[Word], lo: int, hi: int) -> Optional[list[tuple[int, int]]]:
    """Best split of words into runs of lo..hi words, or None if impossible."""
    n = len(words)
    best: list[float] = [0.0] + [float("inf")] * n
    back: list[int] = [-1] * (n + 1)
    for j in range(1, n + 1):
        for size in range(lo, hi + 1):
            i = j - size
            if i < 0 or best[i] == float("inf"):
                continue
            c = best[i] + _line_cost(words, i, j, n)
            if c < best[j]:
                best[j], back[j] = c, i
    if best[n] == float("inf"):
        return None
    runs, j = [], n
    while j > 0:
        runs.append((back[j], j))
        j = back[j]
    return runs[::-1]


def display_text(words: list[Word], uppercase: bool = False) -> str:
    """Caption text: words joined, trailing commas and periods dropped (? and ! kept)."""
    text = " ".join(w.text for w in words).strip()
    text = re.sub(r"[,.;:]+(['\")\]]*)$", r"\1", text)
    return text.upper() if uppercase else text


def group_words(words: list[Word], style: CaptionStyle) -> list[CaptionLine]:
    """Caption lines of style.min_words to style.max_words words (shorter only for tiny narrations)."""
    words = [w for w in words if w.text.strip()]
    if not words:
        return []
    lo, hi = max(1, style.min_words), max(style.min_words, style.max_words)
    if len(words) < lo:
        runs: Optional[list[tuple[int, int]]] = [(0, len(words))]
    else:
        runs = None
        while runs is None and lo >= 1:
            runs = _partition(words, lo, hi)
            lo -= 1
    lines = [CaptionLine(text=display_text(words[a:b], style.uppercase), start=round(words[a].start, 3),
                         end=round(words[b - 1].end, 3), words=words[a:b]) for a, b in runs or []]
    for cur, nxt in zip(lines, lines[1:]):
        if nxt.start - cur.end < HOLD_GAP_S:
            cur.end = nxt.start
    return lines


# ---------------------------------------------------------------- geometry and fonts

def font_path(settings_root: Optional[str], style: CaptionStyle) -> Path:
    root = Path(settings_root) if settings_root else Path(__file__).resolve().parents[2]
    return root / "assets" / "fonts" / f"Inter-{style.weight}.otf"


@functools.lru_cache(maxsize=16)
def _em_ratio(path: str) -> float:
    """unitsPerEm / (usWinAscent + usWinDescent): ASS size to em size for this font file."""
    data = Path(path).read_bytes()
    tables = {}
    for i in range(struct.unpack(">H", data[4:6])[0]):
        tag, _cs, off, _ln = struct.unpack(">4sIII", data[12 + 16 * i: 28 + 16 * i])
        tables[tag] = off
    upem = struct.unpack(">H", data[tables[b"head"] + 18: tables[b"head"] + 20])[0]
    win_asc, win_desc = struct.unpack(">HH", data[tables[b"OS/2"] + 74: tables[b"OS/2"] + 78])
    return upem / float(win_asc + win_desc)


def text_width(text: str, style: CaptionStyle, font_px: int, root: Optional[str] = None) -> float:
    """Rendered width in px of one caption line at ASS size font_px, outline included."""
    path = font_path(root, style)
    em = max(1, int(round(font_px * _em_ratio(str(path)))))
    font = ImageFont.truetype(str(path), em)
    return font.getlength(text) + 2 * style.outline


def caption_geometry(profile: StyleProfile, w: int = 1920, h: int = 1080) -> dict[str, Any]:
    """Band and baseline font: {"x", "y", "font_px", "band_top", "band_bottom", "band_px", "max_width"}.

    Letterbox on: the bottom bar of the 2.39 picture is the band. Letterbox off:
    a virtual band of the same height at the bottom of the frame.
    """
    ratio = profile.letterbox.ratio if profile.letterbox.ratio > 0 else 2.39
    band = max(int(round((h - w / ratio) / 2)), int(round(h * 0.1)))
    band_top = h - band
    style = profile.captions
    y = int(round(band_top + band * style.band_height))
    room = 2 * min(y - band_top, h - y) - 2 * style.outline - 4
    font_px = int(max(12, min(round(style.size * h / 1080.0), room)))
    return {"x": w // 2, "y": y, "font_px": font_px, "band_top": band_top, "band_bottom": h, "band_px": band,
            "max_width": int(w * (1 - 2 * SIDE_MARGIN))}


def line_font_px(text: str, profile: StyleProfile, geo: dict[str, Any], root: Optional[str] = None) -> int:
    """Font size for one line: the band size, reduced until the line fits on one row."""
    size = int(geo["font_px"])
    width = text_width(text, profile.captions, size, root)
    if width <= geo["max_width"]:
        return size
    size = int(size * geo["max_width"] / width)
    while size > 12 and text_width(text, profile.captions, size, root) > geo["max_width"]:
        size -= 1
    return size


# ---------------------------------------------------------------- writers

def _ass_colour(hex_rgb: str, alpha: int = 0) -> str:
    """#RRGGBB to ASS &HAABBGGRR."""
    h = hex_rgb.lstrip("#")
    h = (h + "000000")[:6] if len(h) != 6 else h
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _ass_font(style: CaptionStyle) -> tuple[str, int]:
    """(font family name, bold flag) matching the bundled Inter files."""
    if style.weight == "Regular":
        return style.font, 0
    if style.weight == "Bold":
        return style.font, -1
    return f"{style.font} {style.weight}", 0


def _ass_time(t: float) -> str:
    cs = int(round(max(t, 0.0) * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _srt_time(t: float) -> str:
    ms = int(round(max(t, 0.0) * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ass_header(profile: StyleProfile, geo: dict[str, Any], w: int, h: int) -> str:
    st = profile.captions
    name, bold = _ass_font(st)
    style = (f"Style: Default,{name},{geo['font_px']},{_ass_colour(st.color)},{_ass_colour(st.highlight_color)},"
             f"{_ass_colour(st.outline_color)},{_ass_colour(st.outline_color, 0x80)},{bold},0,0,0,100,100,0,0,1,"
             f"{st.outline},{st.shadow},5,0,0,0,1")
    return "\n".join([
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "WrapStyle: 2",
        "ScaledBorderAndShadow: yes", "YCbCr Matrix: TV.709", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        style, "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text", ""])


def _ass_event(line: CaptionLine, start: float, end: float, profile: StyleProfile, geo: dict[str, Any],
               root: Optional[str]) -> str:
    text = line.text.replace("\\", "/").replace("{", "(").replace("}", ")")
    size = line_font_px(text, profile, geo, root)
    fs = f"\\fs{size}" if size != geo["font_px"] else ""
    return (f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,"
            f"{{\\an5\\pos({geo['x']},{geo['y']}){fs}}}{text}")


def write_ass(lines: list[CaptionLine], profile: StyleProfile, path: Path, w: int = 1920, h: int = 1080,
              root: Optional[str] = None) -> Path:
    geo = caption_geometry(profile, w, h)
    events = [_ass_event(ln, ln.start, ln.end, profile, geo, root) for ln in lines]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ass_header(profile, geo, w, h) + "\n".join(events) + "\n", encoding="utf-8")
    return path


def write_ass_window(lines: list[CaptionLine], profile: StyleProfile, t0: float, t1: float, path: Path,
                     w: int = 1920, h: int = 1080) -> Path:
    """ASS holding only lines overlapping [t0, t1], shifted by -t0 and clipped to [0, t1 - t0]."""
    geo = caption_geometry(profile, w, h)
    span = t1 - t0
    events = []
    for ln in lines:
        if ln.end <= t0 + 1e-3 or ln.start >= t1 - 1e-3:
            continue
        s, e = max(ln.start - t0, 0.0), min(ln.end - t0, span)
        if e - s >= 0.01:
            events.append(_ass_event(ln, s, e, profile, geo, None))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ass_header(profile, geo, w, h) + "\n".join(events) + "\n", encoding="utf-8")
    return path


def write_srt(lines: list[CaptionLine], path: Path) -> Path:
    blocks = [f"{i}\n{_srt_time(ln.start)} --> {_srt_time(ln.end)}\n{ln.text}\n" for i, ln in enumerate(lines, 1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def load_caption_lines(path: Path | str) -> list[CaptionLine]:
    """Caption lines from work/captions.json (a {"lines": [...]} object or a bare list)."""
    data = read_json(path)
    items = data.get("lines", []) if isinstance(data, dict) else data
    return [CaptionLine(**d) for d in items]


def build_captions(narration: Narration, profile: StyleProfile, out_root: Path, settings: Settings) -> list[Path]:
    """Write captions.srt and captions.ass in out_root and work/captions.json. Returns the three paths."""
    out_root = Path(out_root)
    w, h = settings.render.width, settings.render.height
    lines = group_words(narration.words, profile.captions)
    srt = write_srt(lines, out_root / "captions.srt")
    ass = write_ass(lines, profile, out_root / "captions.ass", w, h, settings.repo_root)
    geo = caption_geometry(profile, w, h)
    js = write_json(out_root / "work" / "captions.json",
                    {"lines": [ln.model_dump(mode="json") for ln in lines], "geometry": geo,
                     "style": profile.captions.model_dump(mode="json"), "enabled": profile.captions.enabled})
    return [srt, ass, js]
