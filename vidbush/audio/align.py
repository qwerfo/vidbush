"""Word alignment without a provider: silence based alignment and transcript matching.

silence_align: find speech regions with FFmpeg silencedetect, then distribute
script tokens across them weighted by syllables, anchoring sentence and clause
breaks to the pauses the speaker actually made. match_words: map words heard by
a recognizer (whisper) onto the script tokens with a sequence alignment, so the
script text and beat ids are kept while timings come from the audio.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Optional

import numpy as np

from ..models import Script, Word
from ..util import run
from .loudness import read_audio

Region = tuple[float, float]
_SENT_RE = re.compile(r"[.!?]['\")\]]*$")
_CLAUSE_RE = re.compile(r"[,;:]['\")\]]*$")
WORD_GAP = 0.35  # weight units of inter word spacing
MIN_ANCHOR_GAP = 0.18  # pauses shorter than this are not used as anchors


class Token(Word):
    """A script word waiting for timings (start and end are filled by alignment)."""

    start: float = 0.0
    end: float = 0.0


def script_tokens(script: Script) -> list[Token]:
    """All script words in order, each tagged with its beat id."""
    return [Token(text=t, beat_id=b.id) for b in script.beats() for t in b.text.split()]


def text_tokens(text: str, beat_id: str = "") -> list[Token]:
    return [Token(text=t, beat_id=beat_id) for t in text.split()]


def syllables(word: str) -> int:
    """Rough English syllable count; years read as four syllables, other digits about 1.5 each."""
    digits = re.sub(r"\D", "", word)
    w = re.sub(r"[^a-z]", "", word.lower())
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    if len(digits) == 4 and 1000 <= int(digits) < 2100:
        n += 4
    elif digits:
        n += int(round(len(digits) * 1.5))
    return max(1, n)


def break_strength(token: str) -> float:
    """How strongly a pause is expected after this token (sentence 1, clause 0.6, none 0)."""
    if _SENT_RE.search(token):
        return 1.0
    if _CLAUSE_RE.search(token):
        return 0.6
    return 0.0


# ---------------------------------------------------------------- speech regions

def noise_threshold_db(samples: np.ndarray, sr: int) -> float:
    """Silence threshold from the level statistics: above the noise floor, below speech."""
    frame = max(1, int(sr * 0.05))
    n = len(samples) // frame
    if n < 4:
        return -40.0
    blocks = samples[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(np.square(blocks, dtype=np.float64), axis=1)) + 1e-9
    db = 20 * np.log10(rms)
    floor, speech = float(np.percentile(db, 8)), float(np.percentile(db, 75))
    return float(np.clip(max(floor + 10.0, speech - 28.0), -60.0, -20.0))


def speech_regions(path: Path | str, min_silence: float = 0.12,
                   noise_db: Optional[float] = None) -> tuple[list[Region], float]:
    """Speech regions [(start, end)] and total duration, from FFmpeg silencedetect."""
    samples = read_audio(path, 16000, 1)[:, 0]
    duration = len(samples) / 16000.0
    thr = noise_db if noise_db is not None else noise_threshold_db(samples, 16000)
    cp = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
              f"silencedetect=n={thr:.1f}dB:d={min_silence}", "-f", "null", "-"], timeout=1800)
    log = cp.stderr.decode(errors="replace")
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: (-?[\d.]+)", log)]
    silences = [(max(0.0, s), ends[i] if i < len(ends) else duration) for i, s in enumerate(starts)]
    return regions_from_silences(silences, duration), duration


def regions_from_silences(silences: list[Region], duration: float, min_len: float = 0.04) -> list[Region]:
    """Complement of the silence intervals inside [0, duration]."""
    out: list[Region] = []
    t = 0.0
    for s, e in sorted(silences):
        if s - t >= min_len:
            out.append((round(t, 3), round(s, 3)))
        t = max(t, e)
    if duration - t >= min_len:
        out.append((round(t, 3), round(duration, 3)))
    return out


def clip_regions(regions: list[Region], t0: float, t1: float) -> list[Region]:
    out = [(max(s, t0), min(e, t1)) for s, e in regions if e > t0 and s < t1]
    out = [(s, e) for s, e in out if e - s >= 0.04]
    return out or [(t0, t1)]


# ---------------------------------------------------------------- silence alignment

def silence_align(tokens: list[Token], regions: list[Region], t0: float, t1: float) -> list[Word]:
    """Time the tokens inside [t0, t1] using the speech regions found there."""
    if not tokens:
        return []
    regs = clip_regions(regions, t0, t1)
    weights = np.array([syllables(t.text) + WORD_GAP for t in tokens], dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(weights)])
    out: list[Word] = []
    for (ta, tb), (ra, rb) in _segments(tokens, regs, cum):
        out.extend(_distribute(tokens[ta:tb], weights[ta:tb], regs[ra:rb]))
    return out


def _segments(tokens: list[Token], regs: list[Region], cum: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Anchor token breaks to pauses. Returns [((tok_a, tok_b), (reg_a, reg_b))]."""
    n, durs = len(tokens), np.array([e - s for s, e in regs])
    segs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    tok0, reg0 = 0, 0
    for k in range(len(regs) - 1):
        if regs[k + 1][0] - regs[k][1] < MIN_ANCHOR_GAP or n - tok0 < 2:
            continue
        j = _best_break(tokens, cum, tok0, float(durs[reg0:k + 1].sum()), float(durs[reg0:].sum()))
        if j is None:
            continue
        segs.append(((tok0, j), (reg0, k + 1)))
        tok0, reg0 = j, k + 1
    segs.append(((tok0, n), (reg0, len(regs))))
    return segs


def _best_break(tokens: list[Token], cum: np.ndarray, tok0: int, seg_s: float, rem_s: float) -> Optional[int]:
    """Token index j so tokens[tok0:j] best fill seg_s seconds of speech, or None."""
    n = len(tokens)
    w_rem = cum[n] - cum[tok0]
    rate = w_rem / max(rem_s, 1e-3)
    expected = seg_s * rate
    tol = min(4.0, 0.45 + 0.12 * seg_s)
    best, best_score = None, 1e9
    for j in range(tok0 + 1, n):
        cw = cum[j] - cum[tok0]
        err_s = abs(cw - expected) / rate
        if cw > expected and err_s > tol:
            break
        if err_s > tol:
            continue
        score = err_s / tol - 0.9 * break_strength(tokens[j - 1].text)
        if score < best_score:
            best, best_score = j, score
    return best


def _distribute(tokens: list[Token], weights: np.ndarray, regs: list[Region]) -> list[Word]:
    """Spread tokens over the concatenated speech time of regs, never across a pause."""
    durs = np.array([e - s for s, e in regs])
    axis = np.concatenate([[0.0], np.cumsum(durs)])
    scale = axis[-1] / max(float(weights.sum()), 1e-6)
    out: list[Word] = []
    x = 0.0
    for tok, w in zip(tokens, weights):
        a, b = x, x + (w - WORD_GAP * 0.6) * scale
        x += w * scale
        ka = int(np.clip(np.searchsorted(axis, a, side="right") - 1, 0, len(regs) - 1))
        kb = int(np.clip(np.searchsorted(axis, b, side="left") - 1, 0, len(regs) - 1))
        if kb != ka:  # crosses a pause: keep the word in the region holding most of it
            if axis[ka + 1] - a >= b - axis[kb]:
                b = axis[ka + 1]
            else:
                a, ka = axis[kb], kb
        s = regs[ka][0] + (a - axis[ka])
        e = regs[ka][0] + (min(b, axis[ka + 1]) - axis[ka])
        out.append(Word(text=tok.text, beat_id=tok.beat_id, start=round(s, 3), end=round(max(e, s + 0.02), 3)))
    return out


# ---------------------------------------------------------------- transcript matching

def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def match_words(tokens: list[Token], heard: list[Word], t0: float = 0.0,
                t1: Optional[float] = None) -> list[Word]:
    """Script tokens timed from recognized words via sequence alignment.

    Matched words take the heard timing; replaced runs share the heard span by
    syllables; script words the recognizer missed squeeze into the gap.
    """
    if not tokens:
        return []
    if not heard:
        end = t1 if t1 is not None else t0 + len(tokens) * 0.4
        return silence_align(tokens, [(t0, end)], t0, end)
    a, b = [_norm(t.text) for t in tokens], [_norm(w.text) for w in heard]
    times: list[Optional[tuple[float, float]]] = [None] * len(tokens)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                times[i1 + k] = (heard[j1 + k].start, heard[j1 + k].end)
        elif tag == "replace":
            span = (heard[j1].start, heard[j2 - 1].end)
            for k, w in enumerate(silence_align(tokens[i1:i2], [span], *span)):
                times[i1 + k] = (w.start, w.end)
    return _fill_missing(tokens, times, t0, t1 if t1 is not None else heard[-1].end)


def _fill_missing(tokens: list[Token], times: list[Optional[tuple[float, float]]], t0: float,
                  t1: float) -> list[Word]:
    """Give untimed tokens slots between their timed neighbours, weighted by syllables."""
    out: list[Word] = []
    i, n = 0, len(tokens)
    while i < n:
        if times[i] is not None:
            s, e = times[i]
            out.append(Word(text=tokens[i].text, beat_id=tokens[i].beat_id, start=round(s, 3), end=round(e, 3)))
            i += 1
            continue
        j = i
        while j < n and times[j] is None:
            j += 1
        lo = out[-1].end if out else t0
        hi = times[j][0] if j < n else t1
        if hi - lo < 0.04 * (j - i):  # no room: borrow a little time around the gap
            lo, hi = lo - 0.02 * (j - i), max(hi, lo) + 0.02 * (j - i)
            lo = max(lo, out[-1].start + 0.01 if out else t0)
        out.extend(silence_align(tokens[i:j], [(lo, hi)], lo, hi))
        i = j
    return out


# ---------------------------------------------------------------- beat spans

def beat_spans(words: list[Word], beat_ids: list[str], duration: float,
               fixed: Optional[dict[str, float]] = None) -> dict[str, list[float]]:
    """Contiguous [start, end] per beat covering [0, duration].

    fixed maps beat_id -> known start time (chunk joins). Otherwise a beat starts
    0.25 s before its first word, but never before the previous beat's last word.
    A beat without words gets a zero length span where the next beat starts.
    """
    fixed = fixed or {}
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    for w in words:
        first.setdefault(w.beat_id, w.start)
        last[w.beat_id] = w.end
    starts: list[Optional[float]] = []
    floor, prev_end = 0.0, 0.0
    for i, bid in enumerate(beat_ids):
        if bid in fixed or (bid in first and i > 0):
            t = fixed.get(bid, max(prev_end, first.get(bid, 0.0) - 0.25))
            floor = min(max(t, floor), duration)
            starts.append(floor)
        else:
            starts.append(0.0 if i == 0 else None)
        prev_end = last.get(bid, prev_end)
    nxt = duration
    for i in range(len(starts) - 1, -1, -1):
        if starts[i] is None:
            starts[i] = nxt
        nxt = starts[i]
    st = [round(float(x), 3) for x in starts]
    ends = st[1:] + [round(duration, 3)]
    return {bid: [s, e] for bid, s, e in zip(beat_ids, st, ends)}
