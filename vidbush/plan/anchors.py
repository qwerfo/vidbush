"""Find the exact narration words a picture or a layout element belongs to, and when they are spoken.

Spoken numbers are folded to digits first, so an anchor "1959" or "29 people" finds "nineteen fifty-nine" or
"twenty-nine people" in the word timings.
"""
from __future__ import annotations

import math
import re
from typing import Optional, Sequence

from ..models import Word
from . import numbers as N

_TOK = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
STOP = {"the", "a", "an", "of", "to", "in", "on", "at", "and", "or", "is", "was", "it", "its", "it's", "that",
        "this", "for", "by", "with", "as", "be", "are", "were", "from", "but", "so", "then", "just"}


def _toks(text: str) -> list[str]:
    return [t.replace("'", "") for t in _TOK.findall(text.lower().replace(",", ""))]


def folded(words: Sequence[Word]) -> list[tuple[str, int]]:
    """(token, word index) pairs with spoken number runs replaced by one digit token at the run start."""
    texts = [w.text for w in words]
    runs = {lo: (hi, digits) for lo, hi, digits in N.spoken_numbers(texts)}
    out: list[tuple[str, int]] = []
    i = 0
    while i < len(words):
        if i in runs:
            hi, digits = runs[i]
            out.append((N.normalise_number(digits), i))
            # keep the words of the run too, so "two" or "hundred" can still match as words
            for j in range(i, hi + 1):
                out.extend((t, j) for t in _toks(words[j].text))
            i = hi + 1
            continue
        out.extend((t, i) for t in _toks(words[i].text))
        i += 1
    return out


def _phrase_toks(phrase: str) -> list[str]:
    out = []
    for t in _toks(phrase):
        if any(c.isdigit() for c in t):
            t = N.normalise_number(t)
        out.append(t)
    return out


def find(words: Sequence[Word], phrase: str) -> Optional[int]:
    """Index (into words) of the first word of the best match of phrase, or None.

    A match needs at least 60 percent of the phrase's content tokens, in a window no longer than the phrase
    plus two tokens. Ties go to the earliest window.
    """
    ptoks = _phrase_toks(phrase)
    content = [t for t in ptoks if t not in STOP] or ptoks
    if not content:
        return None
    seq = folded(words)
    need = max(1, math.ceil(0.6 * len(content)))
    width = len(ptoks) + 2  # in words: a spoken number adds a digit token beside its words
    best: Optional[tuple[int, int]] = None  # (-hits, word index)
    for s in range(len(seq)):
        if seq[s][0] not in content:
            continue
        w0 = seq[s][1]
        window = [t for t, wi in seq[s:] if wi < w0 + width]
        hits = sum(1 for t in set(content) if t in window)
        if hits >= need and (best is None or -hits < best[0]):
            best = (-hits, seq[s][1])
    return None if best is None else best[1]


def time_of(words: Sequence[Word], phrase: str) -> Optional[float]:
    i = find(words, phrase)
    return None if i is None else words[i].start
