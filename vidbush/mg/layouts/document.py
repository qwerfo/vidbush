"""document_focus: a real document shown whole, then a move into the sentence being quoted.

The quoted words (role "quote") are found on the page with the prefilter's OCR engine (RapidOCR) run on the
full resolution still in overlapping tiles, then matched fuzzily against the OCR lines, a quote allowed to
run across up to three lines. Found: the page is shown whole, moves in on the phrase as the narration
reaches it, the rest dims, and an amber underline runs under the words as they are read (following the
narration word timings across the quote's anchor when job.extra carries "words" and "clip_start", else
spread evenly over the anchor). Not found: a slow push on the document with the quotation set in house type
beside it and the source label. No document, or a generated one (a layout never shows a generated
document as a real one): the quotation on the ink ground with the source label, noted.
"""
from __future__ import annotations

import difflib
import math
import os
from functools import lru_cache
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from ...models import LayoutKind
from .. import easing as ez
from ..house import MIN_BODY_PX, MIN_LABEL_PX
from ..textdraw import blit, rect
from . import register
from .base import LayoutComp, _still, framed
from .timing import anchor_span, tokens

TILE = 1024  # OCR tile side (full resolution pixels); the engine detects at 640, so text is read at 1.6x down
OVERLAP = 192
MAX_SIDE = 4096  # larger scans are reduced to this first
MIN_CONF = 0.55
MATCH_MIN = 0.78  # character similarity of the quote and the best page window
MAX_LINES = 3

Box = tuple[float, float, float, float]


# ---------------------------------------------------------------- OCR

@lru_cache(maxsize=64)
def _ocr_cached(path: str, mtime: float) -> tuple[tuple[Box, str, float], ...]:
    from ...judge.prefilter import ocr_engine
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        rgb = _still(path)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    f = min(1.0, MAX_SIDE / max(h, w))
    if f < 1.0:
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    engine = ocr_engine()
    found: list[tuple[Box, str, float]] = []
    step = TILE - OVERLAP
    # one pass over the whole page reads headlines that cross tile seams; tiles read the small type.
    # shortcut: small type lines longer than OVERLAP that cross a seam come back as two fragments (each still
    # matchable alone); merge same-row fragments if quotes in body text start failing.
    origins = [(0, 0, img)] if max(h, w) > TILE else []
    origins += [(x, y, img[y:y + TILE, x:x + TILE]) for y in range(0, max(1, h - OVERLAP), step)
                for x in range(0, max(1, w - OVERLAP), step)]
    for x, y, tile in origins:
        result, _ = engine(np.ascontiguousarray(tile))
        for pts, text, conf in result or []:
            if float(conf) < MIN_CONF or not str(text).strip():
                continue
            xs, ys = [p[0] + x for p in pts], [p[1] + y for p in pts]
            found.append(((min(xs) / f, min(ys) / f, max(xs) / f, max(ys) / f), str(text).strip(), float(conf)))
    return tuple(_dedupe(found))


def _dedupe(lines: list[tuple[Box, str, float]]) -> list[tuple[Box, str, float]]:
    """Lines read twice in the tile overlap: the longer reading (then the more confident) is kept."""
    out: list[tuple[Box, str, float]] = []
    for ln in sorted(lines, key=lambda r: (-len(r[1]), -r[2])):
        b = ln[0]
        area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        if any(_inter(b, o[0]) > 0.5 * min(area, (o[0][2] - o[0][0]) * (o[0][3] - o[0][1])) for o in out):
            continue
        out.append(ln)
    return _join_rows(sorted(out, key=lambda r: (r[0][1], r[0][0])))


def _join_rows(lines: list[tuple[Box, str, float]]) -> list[tuple[Box, str, float]]:
    """Fragments of one printed line (split at a tile seam) joined: same row, touching or overlapping. Words
    are cut at the middle of the overlap, so a word read by both fragments is kept once."""
    lines = sorted(lines, key=lambda r: r[0][0])
    changed = True
    while changed:
        changed = False
        for i, (a, ta, ca) in enumerate(lines):
            for j, (b, tb, cb) in enumerate(lines):
                if i == j or b[0] < a[0]:
                    continue
                ha, hb = a[3] - a[1], b[3] - b[1]
                vo = min(a[3], b[3]) - max(a[1], b[1])
                if vo < 0.6 * min(ha, hb) or b[0] - a[2] > 0.8 * min(ha, hb) or b[2] <= a[2]:
                    continue
                cut = (max(a[0], b[0]) + min(a[2], b[2])) / 2 if b[0] < a[2] else a[2]
                keep = [w for w, s, e in _spans(ta, a) if (s + e) / 2 < cut] + \
                       [w for w, s, e in _spans(tb, b) if (s + e) / 2 >= cut]
                merged = ((a[0], min(a[1], b[1]), b[2], max(a[3], b[3])), " ".join(keep), min(ca, cb))
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


def ocr_lines(path: str) -> list[tuple[Box, str, float]]:
    """(box in full resolution pixels, text, confidence) of every line read on the still, cached per path and
    mtime. [] when the OCR engine is unavailable."""
    try:
        return list(_ocr_cached(os.path.abspath(path), os.path.getmtime(path)))
    except ImportError:
        return []


# ---------------------------------------------------------------- phrase matching

def _next_line(lines: Sequence[tuple[Box, str, float]], i: int) -> Optional[int]:
    """The line read after line i: just below it in the same column."""
    b = lines[i][0]
    h = b[3] - b[1]
    best, gap = None, 2.2 * h
    for j, (c, _, _) in enumerate(lines):
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


def find_phrase(lines: Sequence[tuple[Box, str, float]], quote: str,
                min_ratio: float = MATCH_MIN) -> Optional[dict[str, Any]]:
    """Where the quote is on the page: {"ratio", "words": [(box of each matched page word)], "lines": [boxes]}
    or None. Tokens are matched fuzzily (OCR misreads, hyphenation) over windows of 1 to MAX_LINES lines
    in reading order."""
    q = tokens(quote)
    if not q or not lines:
        return None
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
    boxes = []
    for _, li, a, b in win:
        lb = lines[li][0]
        boxes.append((lb[0] + a * (lb[2] - lb[0]), lb[1], lb[0] + b * (lb[2] - lb[0]), lb[3]))
    used = list(dict.fromkeys(li for _, li, _, _ in win))
    return {"ratio": r, "words": boxes, "lines": [lines[li][0] for li in used], "text": " ".join(w[0] for w in win)}


# ---------------------------------------------------------------- word timing

def word_times(job: Any, cue: Any) -> list[tuple[float, float]]:
    """(start, end) clip seconds of each narration word of the cue's anchor: from job.extra "words" (narration
    words, start/end seconds) and "clip_start" when given, else the anchor span split evenly by its tokens."""
    el = job.spec.elements[cue.index]
    words, c0 = job.extra.get("words"), job.extra.get("clip_start")
    if words and c0 is not None and cue.found:
        span = anchor_span(el.anchor, words, near=c0 + cue.word)
        if span is not None:
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
    """0..1 share of the quote read at t: whole words done plus the share of the word being spoken."""
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


# ---------------------------------------------------------------- layout

@register(LayoutKind.document_focus)
class DocumentFocus(LayoutComp):
    """Whole page, a move into the quoted sentence, the rest dimmed, an amber underline as it is read."""

    def setup(self) -> None:
        job = self.job
        doc, quote, src = job.first("document"), job.first("quote"), job.first("source")
        self.quote = quote
        self.qtext = quote[1].text.strip() if quote else ""
        self.times = word_times(job, quote[2]) if quote else []
        self.src = (self.label(src[1].text, 30, "muted", "Medium", caps=False, max_w=self.bw * 0.6, floor=MIN_LABEL_PX),
                    src[2]) if src and src[1].text.strip() else None
        ref = job.pictures.get(doc[0]) if doc else None
        self.mode = "quote_only"
        if doc is None or ref is None:
            job.notes.append("document_focus: no real document picture (missing or generated): the quotation alone")
        elif ref.media != "image":
            job.notes.append("document_focus: the document is a clip, not a still: the quotation alone")
        else:
            self.img = _still(ref.path)
            hit = find_phrase(ocr_lines(ref.path), self.qtext) if self.qtext else None
            if hit is not None:
                self.mode = "focus"
                job.notes.append(f"document_focus: quote found on the page (match {hit['ratio']:.2f}: '{hit['text']}')")
                self._setup_focus(hit)
            else:
                self.mode = "beside"
                job.notes.append("document_focus: the quoted words are not legible on the page: document with the "
                                 "quotation beside it")
        if self.mode != "focus":
            self._setup_set_quote()

    # ------------------------------------------------------------ found: move into the phrase

    def _setup_focus(self, hit: dict) -> None:
        img = self.img
        ih, iw = img.shape[:2]
        x0, y0, x1, y1 = self.box
        self.s0 = min(self.bw / iw, self.bh / ih)  # whole page in the box
        lb = np.array(hit["lines"], np.float64)
        px0, py0, px1, py1 = lb[:, 0].min(), lb[:, 1].min(), lb[:, 2].max(), lb[:, 3].max()
        pw, ph = px1 - px0, py1 - py0
        # the phrase fills about 70 percent of the box width (or 40 percent of its height), never past 1.6x native
        self.s1 = max(self.s0, min(self.bw * 0.7 / max(pw, 1), self.bh * 0.4 / max(ph, 1), 1.6))
        self.c0 = np.array([iw / 2, ih / 2])
        # the phrase is centred unless that would leave empty ground inside the box: a page larger than the box
        # keeps its edges outside it (a headline at the foot of the page sits low, the page fills the frame)
        c1 = []
        for v, size, half in ((px0 + px1) / 2, iw, self.bw / 2 / self.s1), ((py0 + py1) / 2, ih, self.bh / 2 / self.s1):
            c1.append(float(np.clip(v, half, size - half)) if size >= 2 * half else size / 2)
        self.c1 = np.array(c1)
        # working copy at the deepest scale (or native), so each frame is one downscaling warp
        self.ws = min(self.s1, 1.0) if self.s1 <= 1.0 else 1.0
        work = cv2.resize(img, (max(1, int(iw * self.ws)), max(1, int(ih * self.ws))), interpolation=cv2.INTER_AREA) \
            if self.ws < 1.0 else img
        mask = np.zeros(work.shape[:2], np.float32)
        pad = 0.35 * float(np.median(lb[:, 3] - lb[:, 1])) * self.ws
        for b in hit["lines"]:
            cv2.rectangle(mask, (int(b[0] * self.ws - pad), int(b[1] * self.ws - pad)),
                          (int(b[2] * self.ws + pad), int(b[3] * self.ws + pad)), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, pad))
        self.work = work
        self.dim = (work.astype(np.float32) * (0.32 + 0.68 * mask[..., None])).astype(np.uint8)
        self.words = hit["words"]
        q = self.quote[2]
        self.m1 = max(0.6, q.word - 0.1)  # the move ends as the quote starts being read
        self.m0 = max(0.3, min(self.m1 - 1.4, self.m1 * 0.5))
        self.bpad = max(2, int(round(self.house.frame_px * self.OH / 1080)))
        # where the phrase ends up (frame y of its foot): a phrase in the lower third sends the source label up
        foot = (self.box[1] + self.box[3]) / 2 + (py1 - self.c1[1]) * self.s1
        self.phrase_low = foot > self.box[1] + self.bh * 0.62

    def _focus(self, frame: np.ndarray, t: float) -> None:
        u = ez.ease_in_out_cubic(ez.progress(t, self.m0, self.m1))
        s = self.s0 * (self.s1 / self.s0) ** u  # zoom feels even when it is geometric
        c = self.c0 + (self.c1 - self.c0) * u
        bx = (self.box[0] + self.box[2]) / 2
        by = (self.box[1] + self.box[3]) / 2
        k = s / self.ws  # work pixels to frame pixels
        m = np.array([[k, 0, bx - c[0] * s], [0, k, by - c[1] * s]], np.float64)
        ih, iw = self.img.shape[:2]
        # paper border and shadow under the page (frame coordinates)
        X0, Y0, X1, Y1 = bx - c[0] * s, by - c[1] * s, bx + (iw - c[0]) * s, by + (ih - c[1]) * s
        b = self.bpad
        o = self.px(10)
        sx0, sy0 = max(0, int(X0 - b + o)), max(0, int(Y0 - b + o))
        sx1, sy1 = min(self.W, int(X1 + b + o)), min(self.H, int(Y1 + b + o))
        if sx1 > sx0 and sy1 > sy0:  # hard drop shadow, then the paper border
            frame[sy0:sy1, sx0:sx1] = (frame[sy0:sy1, sx0:sx1] * 0.5).astype(np.uint8)
        cv2.rectangle(frame, (int(X0 - b), int(Y0 - b)), (int(X1 + b), int(Y1 + b)), self.house.rgb("paper"), -1)
        d = ez.ease_in_out_cubic(ez.progress(t, self.m0 + 0.3 * (self.m1 - self.m0), self.m1 + 0.2))
        src = self.work if d <= 0.001 else self.dim if d >= 0.999 else \
            cv2.addWeighted(self.work, 1 - d, self.dim, d, 0)
        mask = np.zeros(frame.shape[:2], np.uint8)
        page = cv2.warpAffine(src, m, (self.W, self.H), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        cv2.rectangle(mask, (int(math.ceil(X0)), int(math.ceil(Y0))), (int(X1) - 1, int(Y1) - 1), 255, -1)
        np.copyto(frame, page, where=mask[..., None].astype(bool))
        # underline under the words read so far
        p = read_progress(t, self.times)
        if p <= 0:
            return
        n = len(self.words)
        th = max(4, int(round(5 * self.OH / 1080)))
        col = self.house.rgb("accent")
        for j, (a0, a1, a2, a3) in enumerate(self.words):
            share = ez.clamp01(p * n - j)
            if share <= 0:
                break
            x0, x1 = bx + (a0 - c[0]) * s, bx + (a2 - c[0]) * s
            y = by + (a3 - c[1]) * s + th * 0.6
            nxt = self.words[j + 1] if j + 1 < n else None
            if nxt is not None and abs(nxt[3] - a3) < 0.5 * (a3 - a1):  # same line: bridge the word gap
                x1 = bx + (nxt[0] - c[0]) * s if share >= 1 else x1
            blit(frame, rect(max(1, (x1 - x0) * share), th, col), x0, y)

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
        if beside:
            o = ez.ease_out_cubic(ez.progress(t, 0.0, 0.5))
            s = 1.0 + 0.05 * ez.ease_in_out_sine(ez.clamp01(t / max(self.dur, 1e-3)))  # the slow push on the page
            from ..textdraw import scaled
            d = scaled(self.doc, s)
            dx = x0 + (self.bw * 0.48 - d.w) / 2
            blit(frame, d, dx, y0 + (self.bh - d.h) / 2, o)
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
                rw = self.qspr.w * p
                blit(frame, rect(rw, max(4, int(5 * self.OH / 1080)), self.house.rgb("accent")),
                     x, ty + self.qspr.h + self.px(14))
        if self.src is not None:
            spr, sc = self.src
            o = self.settle(t, sc)
            x = tx if beside else (self.W - spr.w) / 2
            blit(frame, spr, x, ty + total - spr.h, o)

    # ------------------------------------------------------------ frame

    def draw(self, frame: np.ndarray, t: float) -> None:
        if self.mode == "focus":
            self._focus(frame, t)
            if self.src is not None:
                spr, sc = self.src
                o = self.settle(t, sc)
                # the source on an ink chip in a box corner the quoted phrase does not use
                pad = self.px(14)
                y = self.box[1] if self.phrase_low else self.box[3] - spr.h - 2 * pad
                blit(frame, rect(spr.w + 2 * pad, spr.h + 2 * pad, self.house.rgb("ink"), 0.85 * o), self.box[0], y)
                blit(frame, spr, self.box[0] + pad, y + pad, o)
        else:
            self._set_quote(frame, t)
