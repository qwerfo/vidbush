"""Element timing from the narration word timings.

Every layout element names its anchor: the exact narration words it belongs to. anchor_span finds those
words (case and punctuation insensitive, hyphenated numbers matched either joined or split) nearest the
shot, and cue_times turns each element into a Cue: when it starts entering (LEAD_S before its first word,
never more than MAX_LEAD_S), when the word starts and ends. Times are seconds relative to a clip start.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..house import LEAD_S, MAX_LEAD_S

NEVER = 1e9  # cue time of an element that is not shown

_TOKEN = re.compile(r"[a-z0-9']+")


def tokens(text: str) -> list[str]:
    """Lower case word tokens; hyphens and other punctuation split words ("fifty-nine" -> fifty, nine)."""
    return _TOKEN.findall(text.lower().replace("’", "'"))


def _word_tokens(words: Sequence[Any]) -> list[tuple[str, int]]:
    """(token, word index) for every token of every narration word."""
    out = []
    for i, w in enumerate(words):
        for tk in tokens(_text(w)):
            out.append((tk, i))
    return out


def _text(w: Any) -> str:
    return w["text"] if isinstance(w, dict) else w.text


def _start(w: Any) -> float:
    return float(w["start"] if isinstance(w, dict) else w.start)


def _end(w: Any) -> float:
    return float(w["end"] if isinstance(w, dict) else w.end)


def anchor_span(anchor: str, words: Sequence[Any], near: float = 0.0, lo: float = float("-inf"),
                hi: float = float("inf")) -> Optional[tuple[float, float]]:
    """(start, end) seconds of the narration words matching anchor, or None.

    Among several matches the one whose start lies in [lo, hi] and is nearest `near` wins; with none in
    [lo, hi], the nearest overall."""
    want = tokens(anchor)
    if not want or not words:
        return None
    toks = _word_tokens(words)
    seq = [t for t, _ in toks]
    hits = []
    n = len(want)
    for k in range(len(seq) - n + 1):
        if seq[k:k + n] == want:
            a, b = toks[k][1], toks[k + n - 1][1]
            hits.append((_start(words[a]), _end(words[b])))
    if not hits:
        return None
    inside = [h for h in hits if lo <= h[0] <= hi]
    return min(inside or hits, key=lambda h: abs(h[0] - near))


@dataclass
class Cue:
    """When one element shows, in seconds from the clip start."""

    index: int  # element index in the spec
    appear: float  # starts entering
    word: float  # its first anchor word starts (the element is fully readable shortly after)
    word_end: float  # its last anchor word ends
    found: bool  # the anchor was found in the narration (else it appears with the layout)
    stage: int = 1
    outside: bool = False  # its words are spoken, but outside this clip: it is not shown


def cue_times(elements: Sequence[Any], words: Sequence[Any], clip_start: float, clip_end: float,
              lead: float = LEAD_S) -> list[Cue]:
    """One Cue per element, relative to clip_start.

    Anchors are searched near the clip and in spoken order (each from the previous element's word on).
    An element without an anchor, or whose anchor is not spoken, appears with the first element (or at
    0 when none is anchored). Appear times are never earlier than MAX_LEAD_S before the word."""
    lead = min(lead, MAX_LEAD_S)
    cues: list[Cue] = []
    cursor = clip_start - 2.0
    for i, el in enumerate(elements):
        anchor = el.anchor if hasattr(el, "anchor") else el.get("anchor", "")
        stage = int(getattr(el, "stage", 1) if not isinstance(el, dict) else el.get("stage", 1))
        span = anchor_span(anchor, words, near=max(cursor, clip_start), lo=cursor - 0.01, hi=clip_end + 1.0) \
            if anchor else None
        if span is None:
            cues.append(Cue(i, 0.0, 0.0, 0.0, False, stage))
            continue
        if span[0] > clip_end + 0.5 or span[1] < clip_start - 4.0:
            # spoken outside this clip: never shown here (outside=True lets the caller note it)
            cues.append(Cue(i, NEVER, NEVER, NEVER, False, stage, outside=True))
            continue
        cursor = span[0]
        w0, w1 = span[0] - clip_start, span[1] - clip_start
        cues.append(Cue(i, max(0.0, w0 - lead), w0, w1, True, stage))
    first = min((c.appear for c in cues if c.found), default=0.0)
    for c in cues:
        if not c.found and not c.outside:
            c.appear = c.word = c.word_end = first
    return cues
