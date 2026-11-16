"""Part 1 of the clause-first planner: split narration into clauses from punctuation and word timings.

A clause is a stretch of narration that names one thing a viewer could be shown. Rules (deterministic):
- split at sentence ends, beat ends and scene ends;
- inside a sentence split at semicolons, colons, commas and before coordinating conjunctions when both
  sides last at least SPLIT_MIN_S;
- each item of a spoken list ("a, b, and c") is its own clause when it lasts at least LIST_ITEM_MIN_S;
- a piece under MERGE_UNDER_S joins the neighbour it belongs to grammatically;
- a long clause with no internal break stays whole.
A clause's cut is the start of its first word minus CUT_LEAD_S; clauses tile the timeline.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from ..models import Narration, Script

SPLIT_MIN_S = 1.8
LIST_ITEM_MIN_S = 1.0
MERGE_UNDER_S = 1.5
CUT_LEAD_S = 0.1
TAIL_S = 1.0  # the timeline runs this far past the last word, as the old planner does
LIST_ITEM_MAX_WORDS = 4
CONJUNCTIONS = {"and", "but", "or", "so", "yet", "nor"}
_SENT_END = re.compile(r"[.!?][\"')\]]*$")
_STRIP = re.compile(r"^[\"'(\[]+|[\"')\]]+$")

# break kinds, in the order they are tried
STRONG, LIST, COMMA, CONJ = "strong", "list", "comma", "conj"
_ORDER = (STRONG, LIST, COMMA, CONJ)


@dataclass
class Clause:
    id: str
    beat: str
    scene: str
    text: str
    start: float  # cut time: first word start minus CUT_LEAD_S
    end: float  # next clause's cut (or the timeline end)
    words: list[int]  # word indices, first and last inclusive
    speech_start: float = 0.0
    speech_end: float = 0.0
    list_item: bool = False

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


@dataclass
class _Piece:
    lo: int
    hi: int  # inclusive word indices
    sentence: int
    list_item: bool = False
    extra: dict = field(default_factory=dict)


def _bare(tok: str) -> str:
    return _STRIP.sub("", tok).lower()


def _sentences(words, scene_of: dict[str, str]) -> list[tuple[int, int]]:
    """(lo, hi) word ranges: split at sentence punctuation and at every beat change."""
    out, lo = [], 0
    for i, w in enumerate(words):
        last = i == len(words) - 1
        nxt_beat = None if last else words[i + 1].beat_id
        if last or _SENT_END.search(w.text.strip()) or nxt_beat != w.beat_id:
            out.append((lo, i))
            lo = i + 1
    return out


def _list_breaks(words, lo: int, hi: int) -> set[int]:
    """Break points (after word index b) between the items of a spoken list inside one sentence."""
    segs, s = [], lo
    for i in range(lo, hi + 1):
        if i == hi or words[i].text.rstrip().endswith(","):
            segs.append((s, i))
            s = i + 1
    breaks: set[int] = set()
    n = len(segs)
    k = 0
    while k < n:
        # a run of short comma segments; the first may carry lead-in words
        j = k + 1
        while j < n and (segs[j][1] - segs[j][0] + 1) <= LIST_ITEM_MAX_WORDS + 1:
            j += 1
        run = segs[k:j]
        last_lo, last_hi = run[-1]
        conj_at = next((x for x in range(last_lo, last_hi + 1) if _bare(words[x].text) in ("and", "or")), None)
        items = len(run) + (1 if conj_at is not None and conj_at > last_lo else 0)
        ends_in_conj = conj_at is not None
        if len(run) >= 2 and ends_in_conj and items >= 3:
            for a, b in run[:-1]:
                breaks.add(b)
            if conj_at > last_lo:
                breaks.add(conj_at - 1)
        k = max(j, k + 1)
    return breaks


def _candidates(words, lo: int, hi: int) -> list[tuple[str, int]]:
    lst = _list_breaks(words, lo, hi)
    out = []
    for b in range(lo, hi):
        tok = words[b].text.rstrip()
        nxt = _bare(words[b + 1].text)
        if b in lst:
            out.append((LIST, b))
        elif tok.endswith((";", ":")):
            out.append((STRONG, b))
        elif tok.endswith(","):
            out.append((COMMA, b))
        elif nxt in CONJUNCTIONS:
            out.append((CONJ, b))
    out.sort(key=lambda kb: (_ORDER.index(kb[0]), kb[1]))
    return out


def _span(words, lo: int, hi: int) -> float:
    return words[hi].end - words[lo].start


def _split_sentence(words, lo: int, hi: int) -> list[tuple[int, int, bool]]:
    cuts: list[int] = []  # accepted break points b (a new piece starts at b + 1)
    list_cuts: set[int] = set()
    for kind, b in _candidates(words, lo, hi):
        bounds = [lo - 1] + sorted(cuts) + [hi]
        seg_lo = max(x for x in bounds if x < b) + 1
        seg_hi = min(x for x in bounds if x >= b + 1) if any(x >= b + 1 for x in bounds) else hi
        thr = LIST_ITEM_MIN_S if kind == LIST else SPLIT_MIN_S
        if _span(words, seg_lo, b) >= thr and _span(words, b + 1, seg_hi) >= thr:
            cuts.append(b)
            if kind == LIST:
                list_cuts.add(b)
    pieces, s = [], lo
    for b in sorted(cuts) + [hi]:
        is_item = (s - 1) in list_cuts or b in list_cuts
        pieces.append((s, b, is_item))
        s = b + 1
    return pieces


def _merge_short(pieces: list[_Piece], words, scene_of_beat) -> list[_Piece]:
    def scene(p: _Piece) -> str:
        return scene_of_beat.get(words[p.lo].beat_id, "")

    changed = True
    while changed:
        changed = False
        for i, p in enumerate(pieces):
            dur = _span(words, p.lo, p.hi)
            floor = LIST_ITEM_MIN_S if p.list_item else MERGE_UNDER_S
            if dur >= floor or len(pieces) == 1:
                continue
            prev = pieces[i - 1] if i > 0 and scene(pieces[i - 1]) == scene(p) else None
            nxt = pieces[i + 1] if i + 1 < len(pieces) and scene(pieces[i + 1]) == scene(p) else None
            lead_in = words[p.hi].text.rstrip().endswith((",", ":", ";"))
            same_sent_next = nxt is not None and nxt.sentence == p.sentence
            same_sent_prev = prev is not None and prev.sentence == p.sentence
            if lead_in and nxt is not None:  # a comma ends no sentence: only a beat change split it
                target = i + 1
            elif same_sent_prev:
                target = i - 1
            elif same_sent_next:
                target = i + 1
            elif prev is not None:
                target = i - 1  # a short sentence follows from the one before it
            elif nxt is not None:
                target = i + 1
            else:
                continue
            t = pieces[target]
            merged = _Piece(min(t.lo, p.lo), max(t.hi, p.hi), min(t.sentence, p.sentence), False)
            lo_i, hi_i = sorted((i, target))
            pieces[lo_i:hi_i + 1] = [merged]
            changed = True
            break
    return pieces


def segment(script: Script, narration: Narration) -> list[Clause]:
    """Clauses for the whole narration, in order, tiling 0 to narration.duration + TAIL_S."""
    words = narration.words
    if not words:
        return []
    scene_of_beat = {b.id: s.id for s in script.scenes for b in s.beats}
    pieces: list[_Piece] = []
    for si, (lo, hi) in enumerate(_sentences(words, scene_of_beat)):
        for a, b, item in _split_sentence(words, lo, hi):
            pieces.append(_Piece(a, b, si, item))
    pieces = _merge_short(pieces, words, scene_of_beat)
    total = round(narration.duration + TAIL_S, 3)
    clauses: list[Clause] = []
    for n, p in enumerate(pieces):
        w0, w1 = words[p.lo], words[p.hi]
        cut = 0.0 if n == 0 else round(max(0.0, w0.start - CUT_LEAD_S), 3)
        clauses.append(Clause(
            id=f"c{n + 1:03d}", beat=w0.beat_id, scene=scene_of_beat.get(w0.beat_id, ""),
            text=" ".join(w.text for w in words[p.lo:p.hi + 1]), start=cut, end=0.0,
            words=[p.lo, p.hi], speech_start=w0.start, speech_end=w1.end, list_item=p.list_item))
    for a, b in zip(clauses, clauses[1:]):
        b.start = max(b.start, a.start)
        a.end = b.start
    clauses[-1].end = max(total, clauses[-1].speech_end)
    return clauses


def clause_words(clause: Clause, narration: Narration):
    return narration.words[clause.words[0]:clause.words[1] + 1]


def save(clauses: list[Clause], path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in clauses:
        d = asdict(c)
        d["duration"] = c.duration
        rows.append(d)
    p.write_text(json.dumps({"clauses": rows}, indent=1), encoding="utf-8")
    return p


def load(path: Path | str) -> list[Clause]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["clauses"]
    return [Clause(**{k: v for k, v in r.items() if k != "duration"}) for r in rows]


def stats(clauses: list[Clause]) -> dict:
    durs = sorted(c.duration for c in clauses)
    total = clauses[-1].end if clauses else 0.0
    return {"clauses": len(clauses), "per_minute": round(len(clauses) / (total / 60.0), 2) if total else 0.0,
            "median_s": durs[len(durs) // 2] if durs else 0.0, "min_s": durs[0] if durs else 0.0,
            "max_s": durs[-1] if durs else 0.0, "over_6s": sum(d > 6.0 for d in durs),
            "under_1_5s": sum(d < 1.5 for d in durs)}


def scene_clauses(clauses: list[Clause], scene_id: str) -> list[Clause]:
    return [c for c in clauses if c.scene == scene_id]


def find(clauses: list[Clause], cid: str) -> Optional[Clause]:
    return next((c for c in clauses if c.id == cid), None)
