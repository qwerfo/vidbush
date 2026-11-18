"""document_focus: a real document presented so it carries the story, never a still page filling the screen.

Reading the page: the prefilter's OCR engine (RapidOCR) on the full resolution still, one whole page pass
(headlines that cross tile seams) plus overlapping tiles (small type), lines kept as quads so marks follow the
page's skew. A passage is found by fuzzy token matching over 1 to 3 lines in reading order. Results are
cached per path and mtime.

Purpose (LayoutSpec.purpose, None reads as evidence):
  evidence  the page is seen whole long enough to be recognised, then the camera moves to the passage (the
            quote, then any marks), the rest dims, and the passage is marked as it is read. A source label
            (publication and date) is shown.
  period    atmosphere: no marks and no claim about the text, a slow push on the page (usually a panel over
            footage).
  subject   the narration is about the document: the camera explores it, moving to each named part (mark
            elements) as it is spoken and marking it, the rest dimmed less than for evidence.
Every target (the quote and each mark) moves the camera to it and is marked from its first word on, at the
pace of the narration words (job.extra words). Earlier marks stay, quieter. Between moves the page drifts,
so nothing is ever still.

Mark form by the passage: a ring drawn on around a figure, a date or a signature; an underline for a short
phrase (up to 4 words); a highlighter sweep for a line or two; a highlight block with a margin bar for a
longer passage. House colours: amber marks, ink chips, paper type.

Readable or not: a mark is only made where the marked text reaches the type floor (34 px at 1080p) on screen
at no more than MAX_UP times the scan's own resolution. In a panel too small for that (panel_share sizes the
panel so it is, up to about half the frame) the document is shown only for its look, with a note. When the
quoted words are not on the page: the page with a slow push and the quotation set beside it (full frame) or
the page for its look (panel). No real document (missing, a clip, or generated, which make_job refuses): the
quotation alone on the ink ground.
"""
from __future__ import annotations

import difflib
import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from ...models import LayoutKind
from .. import easing as ez
from ..house import MIN_BODY_PX, MIN_LABEL_PX
from ..textdraw import blit, rect, scaled
from . import register
from .base import LayoutComp, _still, drift, framed
from .timing import NEVER, anchor_span, tokens

TILE = 1024  # OCR tile side (full resolution pixels); the engine detects at 640, so text is read at 1.6x down
OVERLAP = 192
MAX_SIDE = 4096  # larger scans are reduced to this first
MIN_CONF = 0.55
MATCH_MIN = 0.78  # character similarity of the passage and the best page window
SHORT_MATCH_MIN = 0.9  # one or two words: stricter, a near miss is another word
MAX_LINES = 3
MAX_UP = 2.0  # a scan is never shown larger than this times its own pixels (beyond it the type is mush)
MOVE_S = 1.3  # a move from one part of the page to the next
DRIFT = 0.018  # slow push between moves, share of scale per second
KEEP = {"evidence": 0.34, "subject": 0.55}  # brightness kept on the page outside the current passage
QUIET = 0.45  # opacity of an earlier mark once the next one starts
SLACK = 1.05  # marked text a little over the floor, never just at it
CONTEXT = 2.2  # a passage at rest is at most this times the floor tall, so the page around it stays in view

Box = tuple[float, float, float, float]
Quad = tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]  # tl tr br bl
Line = tuple[Box, str, float, Quad]

_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE_RE = re.compile(rf"\b{_MONTH}\s+\d{{1,2}}(st|nd|rd|th)?\b|\b\d{{1,2}}(st|nd|rd|th)?\s+(of\s+)?{_MONTH}|"
                     r"\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b", re.I)


# ---------------------------------------------------------------- OCR

def _quad_box(q: Quad) -> Box:
    xs, ys = [p[0] for p in q], [p[1] for p in q]
    return min(xs), min(ys), max(xs), max(ys)


@lru_cache(maxsize=64)
def _ocr_cached(path: str, mtime: float) -> tuple[Line, ...]:
    from ...judge.prefilter import ocr_engine
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.cvtColor(_still(path), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    f = min(1.0, MAX_SIDE / max(h, w))
    if f < 1.0:
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    engine = ocr_engine()
    found: list[Line] = []
    step = TILE - OVERLAP
    # one pass over the whole page reads headlines that cross tile seams; tiles read the small type
    origins = [(0, 0, img)] if max(h, w) > TILE else []
    origins += [(x, y, img[y:y + TILE, x:x + TILE]) for y in range(0, max(1, h - OVERLAP), step)
                for x in range(0, max(1, w - OVERLAP), step)]
    for x, y, tile in origins:
        result, _ = engine(np.ascontiguousarray(tile))
        for pts, text, conf in result or []:
            if float(conf) < MIN_CONF or not str(text).strip():
                continue
            q = tuple((float(p[0] + x) / f, float(p[1] + y) / f) for p in pts[:4])
            found.append((_quad_box(q), str(text).strip(), float(conf), q))  # type: ignore[arg-type]
    return tuple(_dedupe(found))


def _dedupe(lines: list[Line]) -> list[Line]:
    """Lines read twice (whole page pass and tiles, tile overlaps): the longer reading, then the more confident."""
    out: list[Line] = []
    for ln in sorted(lines, key=lambda r: (-len(r[1]), -r[2])):
        b = ln[0]
        area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        if any(_inter(b, o[0]) > 0.5 * min(area, (o[0][2] - o[0][0]) * (o[0][3] - o[0][1])) for o in out):
            continue
        out.append(ln)
    return _join_rows(sorted(out, key=lambda r: (r[0][1], r[0][0])))


def _join_rows(lines: list[Line]) -> list[Line]:
    """Fragments of one printed line (split at a tile seam) joined: same row, touching or overlapping. Words
    are cut at the middle of the overlap, so a word read by both fragments is kept once."""
    lines = sorted(lines, key=lambda r: r[0][0])
    changed = True
    while changed:
        changed = False
        for i, (a, ta, ca, qa) in enumerate(lines):
            for j, (b, tb, cb, qb) in enumerate(lines):
                if i == j or b[0] < a[0]:
                    continue
                ha, hb = a[3] - a[1], b[3] - b[1]
                vo = min(a[3], b[3]) - max(a[1], b[1])
                if vo < 0.6 * min(ha, hb) or b[0] - a[2] > 0.8 * min(ha, hb) or b[2] <= a[2]:
                    continue
                cut = (max(a[0], b[0]) + min(a[2], b[2])) / 2 if b[0] < a[2] else a[2]
                keep = [w for w, s, e in _spans(ta, a) if (s + e) / 2 < cut] + \
                       [w for w, s, e in _spans(tb, b) if (s + e) / 2 >= cut]
                q = (qa[0], qb[1], qb[2], qa[3])
                merged = ((a[0], min(a[1], b[1]), b[2], max(a[3], b[3])), " ".join(keep), min(ca, cb), q)
                lines = [ln for k, ln in enumerate(lines) if k not in (i, j)] + [merged]
                changed = True
                break
            if changed:
                break
    return sorted(lines, key=lambda r: (r[0][1], r[0][0]))


def _spans(text: str, box: Box) -> list[tuple[str, float, float]]:
    """(word, x start, x end) of each space separated word of a line, by character share of its box."""
    n, out, pos = max(1, len(text)), [], 0
    for raw in text.split():
        k = text.find(raw, pos)
        pos = k + len(raw)
        out.append((raw, box[0] + k / n * (box[2] - box[0]), box[0] + pos / n * (box[2] - box[0])))
    return out


def _inter(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def ocr_lines(path: str) -> list[Line]:
    """(box, text, confidence, quad tl tr br bl) of every line read on the still, full resolution pixels,
    cached per path and mtime. [] when the OCR engine is unavailable."""
    try:
        return list(_ocr_cached(os.path.abspath(path), os.path.getmtime(path)))
    except ImportError:
        return []


# ---------------------------------------------------------------- phrase matching

def _next_line(lines: Sequence[Line], i: int) -> Optional[int]:
    """The line read after line i: just below it in the same column."""
    b = lines[i][0]
    h = b[3] - b[1]
    best, gap = None, 2.2 * h
    for j, ln in enumerate(lines):
        c = ln[0]
        if j == i or c[1] <= b[1] + 0.5 * h:
            continue
        ov = min(b[2], c[2]) - max(b[0], c[0])
        if ov < 0.3 * min(b[2] - b[0], c[2] - c[0]):
            continue
        d = c[1] - b[3]
        if -0.3 * h <= d < gap:
            best, gap = j, d
    return best


def _words(text: str) -> list[tuple[str, float, float]]:
    """(token, start share, end share) of each word of an OCR line, positions by character share."""
    out = []
    n = max(1, len(text))
    pos = 0
    for raw in text.split():
        k = text.find(raw, pos)
        pos = k + len(raw)
        tks = tokens(raw)
        for m, tk in enumerate(tks):  # '64-20' is two tokens: each takes its share of the word
            out.append((tk, (k + len(raw) * m / len(tks)) / n, (k + len(raw) * (m + 1) / len(tks)) / n))
    return out


def _sub_quad(q: Quad, a: float, b: float) -> Quad:
    """The part of a line quad between character shares a and b (follows the line's skew)."""
    (tl, tr, br, bl) = [np.asarray(p, np.float64) for p in q]
    top = lambda s: tuple(tl + s * (tr - tl))  # noqa: E731
    bot = lambda s: tuple(bl + s * (br - bl))  # noqa: E731
    return top(a), top(b), bot(b), bot(a)  # type: ignore[return-value]


def _line_quad(ln: Sequence[Any]) -> Quad:
    if len(ln) > 3:
        return ln[3]
    x0, y0, x1, y1 = ln[0]
    return (x0, y0), (x1, y0), (x1, y1), (x0, y1)


def find_phrase(lines: Sequence[Sequence[Any]], quote: str,
                min_ratio: float = MATCH_MIN) -> Optional[dict[str, Any]]:
    """Where the words are on the page, or None. {"ratio", "text", "words": [axis box per matched word],
    "quads": [quad per matched word], "line_of": [line number per word], "lines": [box per line used]}.
    Tokens are matched fuzzily (OCR misreads, hyphenation) over windows of 1 to MAX_LINES lines in reading
    order. One or two words need a closer match (SHORT_MATCH_MIN)."""
    q = tokens(quote)
    if not q or not lines:
        return None
    if len(q) <= 2:
        min_ratio = max(min_ratio, SHORT_MATCH_MIN)
    qs = " ".join(q)
    best = None
    for i in range(len(lines)):
        chain = [i]
        while len(chain) < MAX_LINES:
            nx = _next_line(lines, chain[-1])
            if nx is None or nx in chain:
                break
            chain.append(nx)
        for n in range(1, len(chain) + 1):
            ws = [(tk, li, a, b) for li in chain[:n] for tk, a, b in _words(lines[li][1])]
            if not ws:
                continue
            for L in range(max(1, len(q) - 2), len(q) + 3):
                for s in range(0, max(1, len(ws) - L + 1)):
                    win = ws[s:s + L]
                    if n > 1 and win[-1][1] != chain[n - 1]:
                        continue  # a longer chain must use its last line, else the shorter one says it
                    r = difflib.SequenceMatcher(None, qs, " ".join(w[0] for w in win), autojunk=False).ratio()
                    if best is None or r > best[0] + 1e-9:
                        best = (r, win)
    if best is None or best[0] < min_ratio:
        return None
    r, win = best
    quads = [_sub_quad(_line_quad(lines[li]), a, b) for _, li, a, b in win]
    used = list(dict.fromkeys(li for _, li, _, _ in win))
    return {"ratio": r, "words": [_quad_box(qd) for qd in quads], "quads": quads,
            "line_of": [used.index(li) for _, li, _, _ in win], "lines": [lines[li][0] for li in used],
            "text": " ".join(w[0] for w in win)}


# ---------------------------------------------------------------- mark form and timing

def is_figure(text: str) -> bool:
    """A figure, a date or a signature: ringed, not underlined."""
    if "signature" in text.lower() or DATE_RE.search(text):
        return True
    chars = [ch for ch in text if ch.isalnum()]
    return bool(chars) and sum(ch.isdigit() for ch in chars) >= 0.5 * len(chars)


def mark_form(text: str, n_words: int, n_lines: int) -> str:
    """ring | underline | highlight | block, chosen by the passage."""
    if is_figure(text):
        return "ring"
    if n_words <= 4 and n_lines == 1:
        return "underline"
    return "highlight" if n_lines <= 2 else "block"


def word_times(job: Any, cue: Any) -> list[tuple[float, float]]:
    """(start, end) clip seconds of each narration word of the cue's anchor: from job.extra "words" (narration
    words, start/end seconds) and "clip_start", else the anchor span split evenly by its tokens."""
    el = job.spec.elements[cue.index]
    words, c0 = job.extra.get("words"), job.extra.get("clip_start")
    if words and c0 is not None and cue.found:
        span = anchor_span(el.anchor, words, near=c0 + cue.word)
        if span is not None and abs(span[0] - c0 - cue.word) < 0.05:
            got = [(float(_w(w, "start")) - c0, float(_w(w, "end")) - c0) for w in words
                   if span[0] - 1e-3 <= float(_w(w, "start")) <= span[1] + 1e-3]
            if got:
                return got
    n = max(1, len(tokens(el.anchor or el.text)))
    d = max(0.3, cue.word_end - cue.word) / n
    return [(cue.word + k * d, cue.word + (k + 1) * d) for k in range(n)]


def _w(w: Any, k: str) -> Any:
    return w[k] if isinstance(w, dict) else getattr(w, k)


def read_progress(t: float, times: Sequence[tuple[float, float]]) -> float:
    """0..1 share of the passage read at t: whole words done plus the share of the word being spoken."""
    if not times:
        return 0.0
    done = 0.0
    for a, b in times:
        if t >= b:
            done += 1
        elif t > a:
            done += (t - a) / max(1e-3, b - a)
            break
        else:
            break
    return done / len(times)


@dataclass
class Target:
    """One passage on the page the camera moves to and marks."""

    index: int  # element index
    text: str
    form: str
    hit: dict
    times: list[tuple[float, float]]
    box: Box  # page pixels (a signature: the name with the space above it)
    lh: float  # line height, page pixels
    s: float = 1.0  # camera scale at rest on it
    c: np.ndarray = field(default_factory=lambda: np.zeros(2))
    t0: float = 0.0  # move starts
    t1: float = 0.0  # camera at rest on it

    @property
    def start(self) -> float:
        return self.times[0][0]

    @property
    def end(self) -> float:
        return max(self.times[-1][1], self.start + (0.7 if self.form == "ring" else 0.3))


def _line_height(hit: dict) -> float:
    return float(np.median([b[3] - b[1] for b in hit["lines"]]))


def _hit_box(hit: dict) -> Box:
    q = np.array([p for qd in hit["quads"] for p in qd], np.float64)
    return float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())


def locate(lines: Sequence[Line], text: str) -> Optional[tuple[dict, Box, float]]:
    """(hit, box, line height) of a passage or a signature on the page. A signature is found by the name typed
    under it ('signature Albert Einstein'), and ringed with the space above the name.
    shortcut: assumes the hand signature sits just above the typed name, as in a typed letter; a page where it
    stands elsewhere is ringed at the name; upgrade with a stroke detector if signatures matter."""
    sig = "signature" in text.lower()
    words = re.sub(r"(?i)\bsignature\b|[:,]", " ", text).strip() if sig else text
    hit = find_phrase(lines, words) if words else None
    if hit is None:
        return None
    lh = _line_height(hit)
    x0, y0, x1, y1 = _hit_box(hit)
    if sig:
        y0 -= 2.4 * lh
    return hit, (x0, y0, x1, y1), lh


def crowded_below(lines: Sequence[Line], box: Box, lh: float) -> bool:
    """Another line starts within half a line height under the box and overlaps it sideways."""
    for ln in lines:
        b = ln[0]
        if b[1] > box[1] + 0.5 * lh and b[1] - box[3] < 0.5 * lh and min(b[2], box[2]) - max(b[0], box[0]) > 0:
            return True
    return False


def standalone(lines: Sequence[Line], hit: dict) -> bool:
    """The passage is all its line says (a label, a heading), not words inside running text."""
    n = len(hit["quads"])
    for b in hit["lines"]:
        ln = next((x for x in lines if x[0] == b), None)
        if ln is not None and len(tokens(ln[1])) > n + 1:
            return False
    return True


def headline(lines: Sequence[Line]) -> Optional[Box]:
    """The box of the largest type read on the page (masthead, headline, letterhead), None when nothing is read."""
    got = [ln for ln in lines if ln[2] >= 0.7 and len(ln[1]) >= 3]
    return max(got, key=lambda ln: ln[0][3] - ln[0][1])[0] if got else None


def _doc_path(spec: Any, extra: Optional[dict]) -> Optional[str]:
    """The document still's path before the job exists: extra["pictures"] (element index -> path or PictureRef)
    when the caller passes it, else the element's picture when it is a file."""
    for i, e in enumerate(spec.elements):
        if e.role != "document":
            continue
        got = ((extra or {}).get("pictures") or {}).get(i)
        got = getattr(got, "path", got)
        if got and Path(str(got)).is_file():
            return str(got)
        if e.picture and Path(e.picture).is_file():
            return e.picture
        return None
    return None


# ---------------------------------------------------------------- layout

@register(LayoutKind.document_focus)
class DocumentFocus(LayoutComp):
    """A real document: recognised whole, explored, its passages marked as they are spoken."""

    panel_size = (0.36, 0.46)

    @classmethod
    def panel_share(cls, spec: Any, extra: Optional[dict] = None) -> tuple[float, float]:
        """(w, h) share of a 1920x1080 frame. Sized so every passage to be marked is on screen at the type floor
        when the camera rests on it (its OCR line height times the scale at least 34 px, the passage inside the
        panel box), at no more than MAX_UP times the scan's pixels. When that needs more than 0.5 x 0.6 of the
        frame, or nothing is to be marked (period, or no passage found), the panel is sized for the page's look
        (the comp then shows it for its look only and notes it)."""
        W, H = 1920, 1080
        path = _doc_path(spec, extra)
        if path is None:
            return cls.panel_size
        from PIL import Image
        with Image.open(path) as im:
            iw, ih = im.size

        def look() -> tuple[float, float]:
            h = 0.46
            m = max(18.0, 0.06 * h * H)
            w = (h * H - 2 * m) * iw / ih + 2 * m
            return round(float(np.clip(w / W, 0.26, 0.5)), 3), h

        if (spec.purpose or "evidence") == "period":
            return look()
        lines = ocr_lines(path)
        need_w = need_h = 0.0
        for e in spec.elements:
            if e.role not in ("quote", "mark") or not e.text.strip():
                continue
            got = locate(lines, e.text)
            if got is None:
                continue
            _, (x0, y0, x1, y1), lh = got
            s = MIN_LABEL_PX * SLACK / lh
            if s > MAX_UP:
                return look()  # too small in the scan to read on a television at any size
            need_w, need_h = max(need_w, (x1 - x0) * s / 0.92), max(need_h, (y1 - y0) * s / 0.6)
        if need_w <= 0:
            return look()
        w, h = need_w, max(need_h, 0.36 * H)
        for _ in range(3):  # the panel margin grows with the panel
            m = max(18.0, 0.06 * min(w, h))
            w, h = need_w + 2 * m, max(need_h + 2 * m, 0.36 * H)
        w = max(w, 0.30 * W)
        if w > 0.5 * W or h > 0.6 * H:
            return look()
        return round(w / W, 3), round(h / H, 3)

    def setup(self) -> None:
        job = self.job
        self.purpose = self.spec.purpose or "evidence"
        doc, quote, src = job.first("document"), job.first("quote"), job.first("source")
        self.quote = quote if quote and quote[2].appear < NEVER / 10 else None
        self.qtext = self.quote[1].text.strip() if self.quote else ""
        self.times = word_times(job, self.quote[2]) if self.quote else []
        self.floor = MIN_LABEL_PX * self.OH / 1080.0
        self.fit_ok = True
        self.targets: list[Target] = []
        self.src = None
        if src and src[1].text.strip() and src[2].appear < NEVER / 10:
            self.src = (self.label(src[1].text, 30, "paper", "Medium", caps=False,
                                   max_w=self.bw - 2 * self.px(14), lines=2, floor=MIN_LABEL_PX), src[2])
        ref = job.pictures.get(doc[0]) if doc else None
        self.mode = "quote_only"
        if doc is None or ref is None:
            job.notes.append("document_focus: no real document picture (missing or generated): the quotation alone")
        elif ref.media != "image":
            job.notes.append("document_focus: the document is a clip, not a still: the quotation alone")
        else:
            self.img = _still(ref.path)
            self._setup_page(ref.path)
        if self.mode in ("quote_only", "beside"):
            self._setup_set_quote()
        elif self.purpose == "evidence" and self.src is None:
            job.notes.append("document_focus: evidence shown without a source label (none given)")

    # ------------------------------------------------------------ the page

    def _setup_page(self, path: str) -> None:
        job = self.job
        ih, iw = self.img.shape[:2]
        lines = self.lines = ocr_lines(path)
        wanted = [(i, e, c) for i, e, c in job.els("quote") + job.els("mark") if e.text.strip()]
        wanted.sort(key=lambda r: (r[2].word, r[0]))
        if self.purpose == "period":
            if wanted:
                job.notes.append("document_focus: period (atmosphere): no marks, no claim about the text")
            wanted = []
        x0, y0, x1, y1 = self.box
        self.s0 = min(self.bw / iw, self.bh / ih)  # whole page in the box
        small = []
        for i, e, c in wanted:
            if c.appear >= NEVER / 10:
                continue  # spoken outside this clip (make_job noted it)
            got = locate(lines, e.text)
            if got is None:
                job.notes.append(f"document_focus: '{e.text}' is not legible on the page: not marked")
                continue
            hit, box, lh = got
            pw, ph = box[2] - box[0], box[3] - box[1]
            s_max = min(self.bw * 0.92 / pw, self.bh * 0.6 / ph, MAX_UP)
            if lh * s_max < self.floor:
                small.append(e.text)
                continue
            # the passage large in the box, but never past CONTEXT times the floor: the page around it stays seen
            s = min(self.bw * 0.7 / pw, self.bh * 0.4 / ph, MAX_UP, self.floor * CONTEXT / lh)
            s = float(np.clip(s, self.floor * SLACK / lh, s_max))
            s = max(s, self.s0)
            n_lines = len(hit["lines"])
            form = mark_form(e.text, len(hit["quads"]), n_lines)
            if form == "underline" and (standalone(lines, hit) or crowded_below(lines, box, lh)):
                # a label on its own (a map name, a heading) or type just beneath: an underline would run through
                # what lies under it (OCR does not read every small figure), a ring does not
                form = "ring"
            t = Target(i, e.text, form, hit, word_times(job, c), box, lh, s)
            t.c = self._clamp(np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]), s)
            self.targets.append(t)
            job.notes.append(f"document_focus: '{e.text}' found on the page (match {hit['ratio']:.2f}), {form}")
        if small:
            if self.panel:
                job.notes.append("document_focus: the text is too small to read at this panel size: shown for its "
                                 "look only, no marks")
                self.targets, self.fit_ok = [], False
            else:
                job.notes.append(f"document_focus: too small to read even at {MAX_UP:g}x the scan: "
                                 + ", ".join(f"'{s}'" for s in small) + " not marked")
        if not self.targets and self.qtext and self.purpose == "evidence" and self.fit_ok:
            if self.panel:
                job.notes.append("document_focus: the quoted words are not legible on the page: shown for its "
                                 "look only in the panel")
            else:
                self.mode = "beside"
                job.notes.append("document_focus: the quoted words are not legible on the page: the document "
                                 "with the quotation beside it")
                return
        self.mode = "page"
        self._plan_camera()
        self._build_levels()
        self._build_masks()
        self.src_corner = self._source_corner()

    def _source_corner(self) -> tuple[float, float]:
        """The box corner for the source chip that covers the fewest marked passages (screen boxes at rest),
        lower left first; a period page avoids the largest type it pushes toward."""
        if self.src is None:
            return self.box[0], self.box[1]
        pad = self.px(14)
        w, h = self.src[0].w + 2 * pad, self.src[0].h + 2 * pad
        x0, y0, x1, y1 = self.box
        corners = [(x0, y1 - h), (x0, y0), (x1 - w, y1 - h), (x1 - w, y0)]
        bx, by = (x0 + x1) / 2, (y0 + y1) / 2
        rests = [(t.box, t.s, t.c, t.lh) for t in self.targets]
        if not self.targets:
            hb = headline(self.lines)
            if hb:
                rests = [(hb, self.s0 * 1.1, self.push_c, hb[3] - hb[1])]

        def cover(cx: float, cy: float) -> float:
            tot = 0.0
            for b, s, c, lh in rests:
                m = 0.8 * lh  # the ring and the bright area around the passage
                r = (bx + (b[0] - m - c[0]) * s, by + (b[1] - m - c[1]) * s, bx + (b[2] + m - c[0]) * s,
                     by + (b[3] + m - c[1]) * s)
                tot += _inter(r, (cx, cy, cx + w, cy + h))
            return tot
        return min(corners, key=lambda xy: cover(*xy))

    def _clamp(self, c: np.ndarray, s: float) -> np.ndarray:
        """A view centre that leaves no empty ground inside the box when the page is larger than it."""
        ih, iw = self.img.shape[:2]
        out = []
        for v, size, half in ((c[0], iw, self.bw / 2 / s), (c[1], ih, self.bh / 2 / s)):
            out.append(float(np.clip(v, half, size - half)) if size >= 2 * half else size / 2)
        return np.array(out)

    def _plan_camera(self) -> None:
        """Move windows: the page is held whole first (recognised), then each target is reached as its first
        word is spoken; a period page pushes slowly toward its upper part (masthead, letterhead)."""
        ih, iw = self.img.shape[:2]
        self.c0 = np.array([iw / 2, ih / 2])
        prev = 0.0
        for k, t in enumerate(self.targets):
            t.t1 = max(t.start - 0.1, prev + 0.35)
            if k == 0:
                t.t0 = max(0.3, min(t.t1 - 1.4, t.t1 * 0.5)) if self.purpose == "evidence" else \
                    max(0.0, t.t1 - MOVE_S)
            else:
                t.t0 = max(prev + 0.25, t.t1 - MOVE_S)
            t.t0 = min(t.t0, t.t1 - 0.3)
            prev = t.t1
        if not self.targets:  # period or look only: the slow push toward the largest type (masthead, headline)
            hb = headline(self.lines)
            to = np.array([(hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2]) if hb else np.array([iw / 2, ih * 0.42])
            self.push_c = self._clamp(self.c0 + (to - self.c0) * 0.6, self.s0 * 1.1)

    def camera(self, t: float) -> tuple[float, np.ndarray]:
        """(scale, page point at the box centre) at t."""
        if not self.targets:
            u = ez.ease_in_out_sine(ez.clamp01(t / max(self.dur, 1e-3)))
            return self.s0 * (1 + 0.1 * u), self.c0 + (self.push_c - self.c0) * u
        s, c, rest = self.s0, self.c0, 0.0
        for tg in self.targets:
            if t < tg.t0:
                break
            s_from = s * (1 + DRIFT * (tg.t0 - rest))
            u = ez.ease_in_out_cubic(ez.progress(t, tg.t0, tg.t1))
            s, c = s_from * (tg.s / s_from) ** u, c + (tg.c - c) * u
            if t < tg.t1:
                return s, c
            s, c, rest = tg.s, tg.c, tg.t1
        return s * (1 + DRIFT * (t - rest)), c

    def _build_levels(self) -> None:
        """The page at halving sizes, so every frame is one warp from a level at or just above the scale."""
        self.levels = [(1.0, self.img)]
        sc, img = 1.0, self.img
        while sc / 2 >= max(self.s0 * 0.9, 1e-3) and min(img.shape[:2]) > 64:
            img = cv2.resize(img, (max(1, img.shape[1] // 2), max(1, img.shape[0] // 2)), interpolation=cv2.INTER_AREA)
            sc /= 2
            self.levels.append((sc, img))
        self.levels.sort(key=lambda r: r[0])

    def _level(self, s: float) -> tuple[float, np.ndarray]:
        for sc, img in self.levels:
            if sc >= s * 0.98:
                return sc, img
        return self.levels[-1]

    def _build_masks(self) -> None:
        """Per target, where the page stays bright (low resolution, warped with the page each frame)."""
        ih, iw = self.img.shape[:2]
        self.mf = min(1.0, 640 / max(ih, iw))
        mh, mw = max(1, int(ih * self.mf)), max(1, int(iw * self.mf))
        self.masks = []
        for t in self.targets:
            m = np.zeros((mh, mw), np.float32)
            pad = 0.45 * t.lh * self.mf
            for qd in t.hit["quads"]:
                pts = np.array(qd, np.float64) * self.mf
                cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1.0)
            if t.form == "ring" or "signature" in t.text.lower():
                x0, y0, x1, y1 = (np.array(t.box) * self.mf).astype(int)
                cv2.rectangle(m, (x0, y0), (x1, y1), 1.0, -1)
            m = cv2.dilate(m, np.ones((max(1, int(pad * 2)),) * 2, np.uint8))
            self.masks.append(cv2.GaussianBlur(m, (0, 0), max(1.0, pad)))
        self.ones = np.ones((mh, mw), np.float32)

    def _mask(self, t: float) -> Optional[np.ndarray]:
        if not self.targets or self.purpose not in KEEP:
            return None
        m = None
        for k, tg in enumerate(self.targets):
            if t < tg.t0:
                break
            u = ez.ease_in_out_cubic(ez.progress(t, tg.t0 + 0.3 * (tg.t1 - tg.t0), tg.t1 + 0.2))
            prev = self.masks[k - 1] if k else self.ones
            m = self.masks[k] if u >= 1 else prev * (1 - u) + self.masks[k] * u
        return m

    def _page(self, frame: np.ndarray, t: float) -> tuple[float, np.ndarray]:
        s, c = self.camera(t)
        bx, by = (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2
        ih, iw = self.img.shape[:2]
        X0, Y0, X1, Y1 = bx - c[0] * s, by - c[1] * s, bx + (iw - c[0]) * s, by + (ih - c[1]) * s
        b = max(2, int(round(self.house.frame_px * self.OH / 1080)))
        o = self.px(10)
        sx0, sy0 = max(0, int(X0 - b + o)), max(0, int(Y0 - b + o))
        sx1, sy1 = min(self.W, int(X1 + b + o)), min(self.H, int(Y1 + b + o))
        if sx1 > sx0 and sy1 > sy0:  # hard drop shadow, then the paper border
            frame[sy0:sy1, sx0:sx1] >>= 1
        cv2.rectangle(frame, (int(X0 - b), int(Y0 - b)), (int(X1 + b), int(Y1 + b)), self.house.rgb("paper"), -1)
        rx0, ry0 = max(0, int(math.ceil(X0))), max(0, int(math.ceil(Y0)))
        rx1, ry1 = min(self.W, int(X1)), min(self.H, int(Y1))
        if rx1 <= rx0 or ry1 <= ry0:
            return s, c
        sc, lvl = self._level(s)
        k = s / sc
        # warp only the visible page rectangle (an offset matrix into the ROI)
        m = np.array([[k, 0, bx - c[0] * s - rx0], [0, k, by - c[1] * s - ry0]], np.float64)
        roi = cv2.warpAffine(lvl, m, (rx1 - rx0, ry1 - ry0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = self._mask(t)
        if mask is not None:
            km = s / self.mf
            mm = np.array([[km, 0, bx - c[0] * s - rx0], [0, km, by - c[1] * s - ry0]], np.float64)
            w = cv2.warpAffine(mask, mm, (rx1 - rx0, ry1 - ry0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            keep = KEEP[self.purpose]
            f = cv2.convertScaleAbs(w, alpha=255 * (1 - keep), beta=255 * keep)  # uint8 brightness kept
            roi = cv2.multiply(roi, cv2.cvtColor(f, cv2.COLOR_GRAY2RGB), scale=1 / 255.0)
        frame[ry0:ry1, rx0:rx1] = roi
        return s, c

    # ------------------------------------------------------------ marks

    def _to_frame(self, pts: Any, s: float, c: np.ndarray) -> np.ndarray:
        bx, by = (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2
        p = np.asarray(pts, np.float64)
        return np.stack([bx + (p[..., 0] - c[0]) * s, by + (p[..., 1] - c[1]) * s], -1)

    def _marks(self, frame: np.ndarray, t: float, s: float, c: np.ndarray) -> None:
        for k, tg in enumerate(self.targets):
            if t < tg.start:
                break  # later marks are later still: never before their own first word
            p = read_progress(t, tg.times) if tg.form != "ring" else \
                ez.ease_in_out_sine(ez.progress(t, tg.start, tg.end))
            if p <= 0:
                continue
            nxt = self.targets[k + 1].start if k + 1 < len(self.targets) else NEVER
            op = 1 - (1 - QUIET) * ez.progress(t, nxt, nxt + 0.4)
            lh = tg.lh * s
            getattr(self, f"_{tg.form}")(frame, tg, p, op, s, c, lh)

    def _word_spans(self, tg: Target, p: float, s: float, c: np.ndarray) -> list[np.ndarray]:
        """Frame quads of the words read so far (the last one partly), each reaching the next word on its line."""
        qs = [self._to_frame(q, s, c) for q in tg.hit["quads"]]
        lo = tg.hit["line_of"]
        n, out = len(qs), []
        for j, q in enumerate(qs):
            share = ez.clamp01(p * n - j)
            if share <= 0:
                break
            tl, tr, br, bl = q
            if share >= 1 and j + 1 < n and lo[j + 1] == lo[j]:
                tr, br = qs[j + 1][0], qs[j + 1][3]
            out.append(np.array([tl, tl + (tr - tl) * share, bl + (br - bl) * share, bl]))
        return out

    def _ink(self, frame: np.ndarray, polys: list[np.ndarray], alpha: float, mode: str = "fill",
             thick: int = 1, color: Optional[tuple] = None) -> None:
        """Draw anti-aliased polygons or polylines on a region of the frame at alpha (amber by default);
        mode "multiply" tints the paper like a highlighter while dark type stays dark."""
        if not polys:
            return
        allp = np.concatenate([np.asarray(q).reshape(-1, 2) for q in polys])
        pad = thick + 4
        x0, y0 = max(0, int(allp[:, 0].min()) - pad), max(0, int(allp[:, 1].min()) - pad)
        x1, y1 = min(self.W, int(allp[:, 0].max()) + pad + 1), min(self.H, int(allp[:, 1].max()) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            return
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        sh = 4  # sub-pixel bits
        for q in polys:
            pts = np.round((np.asarray(q).reshape(-1, 2) - [x0, y0]) * (1 << sh)).astype(np.int32)
            if mode == "line":
                cv2.polylines(m, [pts], False, 255, max(1, thick), cv2.LINE_AA, sh)
            else:
                cv2.fillPoly(m, [pts], 255, cv2.LINE_AA, sh)
        a = (m.astype(np.float32) / 255.0 * alpha)[..., None]
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        col = np.array(color or self.house.rgb("accent"), np.float32)
        top = roi * (col / 255.0) if mode == "multiply" else col
        frame[y0:y1, x0:x1] = (roi * (1 - a) + top * a).astype(np.uint8)

    def _thick(self, lh: float, share: float, lo: float, hi: float) -> int:
        k = self.OH / 1080.0
        return int(round(float(np.clip(share * lh, lo * k, hi * k))))

    def _underline(self, frame, tg, p, op, s, c, lh) -> None:  # noqa: ANN001
        th = self._thick(lh, 0.09, 4, 10)
        lines = []
        for q in self._word_spans(tg, p, s, c):
            tl, tr, br, bl = q
            down = (bl - tl) / max(1e-3, np.linalg.norm(bl - tl))
            off = down * (0.12 * lh + th / 2)
            lines.append(np.array([bl + off, br + off]))
        self._ink(frame, lines, 0.55 * op, "line", th + 4, self.house.rgb("ink"))
        self._ink(frame, lines, op, "line", th)

    def _highlight(self, frame, tg, p, op, s, c, lh, alpha: float = 0.9) -> None:  # noqa: ANN001
        polys = []
        for q in self._word_spans(tg, p, s, c):
            tl, tr, br, bl = q
            down = (bl - tl) / max(1e-3, np.linalg.norm(bl - tl))
            e = down * 0.14 * lh
            polys.append(np.array([tl - e, tr - e, br + e, bl + e]))
        self._ink(frame, polys, alpha * op, "multiply")

    def _block(self, frame, tg, p, op, s, c, lh) -> None:  # noqa: ANN001
        self._highlight(frame, tg, p, op, s, c, lh, alpha=0.5)
        x0, y0, x1, y1 = tg.box
        th = self._thick(lh, 0.14, 5, 12)
        a = self._to_frame([[x0 - 0.6 * tg.lh, y0], [x0 - 0.6 * tg.lh, y1]], s, c)
        bar = np.array([a[0], a[0] + (a[1] - a[0]) * p])
        self._ink(frame, [bar], op, "line", th)

    def _ring(self, frame, tg, p, op, s, c, lh) -> None:  # noqa: ANN001
        """A hand-drawn feeling ellipse drawn on around the figure: starts upper left, slight wobble, closes
        past its start."""
        x0, y0, x1, y1 = tg.box
        quad = tg.hit["quads"][0]
        ang = math.atan2(quad[1][1] - quad[0][1], quad[1][0] - quad[0][0])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = (x1 - x0) / 2 + 0.45 * tg.lh, (y1 - y0) / 2 + 0.55 * tg.lh
        n = 96
        th0 = math.radians(-160)
        sweep = 2 * math.pi * 1.1
        u = np.linspace(0, p, max(2, int(n * p)))
        a = th0 + sweep * u
        r = (1 + 0.035 * np.sin(2 * a + 1.3)) * (1 + 0.06 * u)
        ex, ey = rx * r * np.cos(a), ry * r * np.sin(a)
        px = cx + ex * math.cos(ang) - ey * math.sin(ang)
        py = cy + ex * math.sin(ang) + ey * math.cos(ang)
        pts = self._to_frame(np.stack([px, py], -1), s, c)
        th = self._thick(lh, 0.1, 5, 11)
        self._ink(frame, [pts], 0.5 * op, "line", th + 4, self.house.rgb("ink"))
        self._ink(frame, [pts], op, "line", th)

    # ------------------------------------------------------------ not found / no document: the quotation set

    def _setup_set_quote(self) -> None:
        beside = self.mode == "beside"
        tw = self.bw * (0.46 if beside else 0.8)
        text = f"“{self.qtext}”" if self.qtext else ""
        self.qspr = self.label(text, 96, "paper", "SemiBold", caps=False, max_w=tw, lines=4, floor=MIN_BODY_PX,
                               align="left" if beside else "center") if text else None
        if beside:
            dw, dh = self.bw * 0.46, self.bh * 0.96
            self.doc = framed(self.img, dw, dh, self.house, self.k, fit=True)

    def _set_quote(self, frame: np.ndarray, t: float) -> None:
        beside = self.mode == "beside"
        x0, y0, x1, y1 = self.box
        tx = x0
        if beside:
            o = ez.ease_out_cubic(ez.progress(t, 0.0, 0.5))
            s = 1.0 + 0.05 * ez.ease_in_out_sine(ez.clamp01(t / max(self.dur, 1e-3)))  # the slow push on the page
            d = scaled(self.doc, s)
            blit(frame, d, x0 + (self.bw * 0.48 - d.w) / 2, y0 + (self.bh - d.h) / 2, o)
            tx = x0 + self.bw * 0.53
        q = self.quote[2] if self.quote else None
        total = (self.qspr.h if self.qspr is not None else 0) + (self.src[0].h + self.px(40) if self.src else 0)
        ty = y0 + (self.bh - total) / 2
        if self.qspr is not None and q is not None:
            o = self.settle(t, q)
            x = tx if beside else (self.W - self.qspr.w) / 2
            blit(frame, self.qspr, x, ty + self.px(16) * (1 - o), o)
            p = read_progress(t, self.times)  # the amber rule under the quotation grows as it is read
            if p > 0:
                blit(frame, rect(self.qspr.w * p, max(4, int(5 * self.OH / 1080)), self.house.rgb("accent")),
                     x, ty + self.qspr.h + self.px(14))
        if self.src is not None:
            spr, sc = self.src
            x = tx if beside else (self.W - spr.w) / 2
            blit(frame, spr, x, ty + total - spr.h, self.settle(t, sc))

    # ------------------------------------------------------------ frame

    def fits(self) -> bool:
        """False when this size is too small to read a passage that is to be marked (the stage keeps a larger panel)."""
        return self.fit_ok

    def draw(self, frame: np.ndarray, t: float) -> None:
        if self.mode != "page":
            self._set_quote(frame, t)
            if self.mode == "quote_only":
                frame[:] = drift(frame, t, self.dur, self.house.drift)  # never still
            return
        s, c = self._page(frame, t)
        self._marks(frame, t, s, c)
        if self.src is not None:
            spr, sc = self.src
            o = self.settle(t, sc)
            pad = self.px(14)
            x, y = self.src_corner
            blit(frame, rect(spr.w + 2 * pad, spr.h + 2 * pad, self.house.rgb("ink"), 0.85 * o), x, y)
            blit(frame, spr, x + pad, y + pad, o)
