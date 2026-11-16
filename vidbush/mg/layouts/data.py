"""Data layouts: chart (bars or a line), table and timeline, full frame or as a panel over the picture.

Each follows the pictures.py pattern: setup() measures and builds sprites once, draw() places them from
the cues, nothing shows before the first cue. Type goes through fs() at the house floors (cells and labels
40 px, titles 56 px full frame and 40 px in a panel), so a panel is as readable as the full layout.
setup() also records `need`, the content size (w, h) it needs at those floors, and `ok` (no text had to
be cut), which panel_share() uses to choose a panel size that fits its own content.

Values are never false at a readable moment: a bar's value shows only once the bar has reached its true
length, a point's value only once the line has reached it. The largest bar, or the turning point of a
line, is amber from the moment it is spoken; the rest are paper fading to muted.
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional, Sequence

import numpy as np

from ...models import LayoutKind, LayoutSpec
from .. import easing as ez
from ..house import MIN_BODY_PX, MIN_HEAD_PX, MIN_LABEL_PX, SETTLE_S
from ..textdraw import ELLIPSIS, Sprite, blit, disc, fit_text, font, polyline, text_block, text_sprite, text_width, wrap
from . import register
from .base import LayoutComp, LayoutJob
from .panel import NEVER
from .timing import Cue

SLACK = 1.08  # panels get this much room over their measured need, so content breathes
HIDDEN = NEVER / 2  # a cue at or after this is never shown (stage 2 while a panel): left out of the layout
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_YEAR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
_NUMERIC_CELL = re.compile(r"^[~<>]?\s*[$£€]?\s*-?\d")


# ---------------------------------------------------------------- parsing and helpers

def parse_value(text: str) -> tuple[str, str, Optional[float]]:
    """'No road: 82%' -> ('No road', '82%', 82.0). No colon: the whole text is the label."""
    label, sep, val = (text or "").rpartition(":")
    if not sep:
        label, val = text or "", ""
    m = _NUM.search(val)
    return label.strip(), val.strip(), float(m.group().replace(",", "")) if m else None


def parse_event(text: str) -> tuple[str, str, Optional[int]]:
    """'1925: Diphtheria in Nome' -> ('1925', 'Diphtheria in Nome', 1925)."""
    date, sep, label = (text or "").partition(":")
    if not sep:
        date, label = "", text or ""
    m = _YEAR.search(date)
    return date.strip(), label.strip(), int(m.group(1)) if m else None


def chart_emphasis(values: Sequence[float], line: bool) -> int:
    """Index to emphasise: the largest bar; on a line the point with the largest change of slope (the peak
    when there are fewer than three points or the line is straight). -1 when empty."""
    if not values:
        return -1
    peak = int(np.argmax(values))
    if not line or len(values) < 3:
        return peak
    d = [abs((values[i + 1] - values[i]) - (values[i] - values[i - 1])) for i in range(1, len(values) - 1)]
    k = int(np.argmax(d))
    return k + 1 if d[k] > 1e-9 else peak


def timeline_history(previous_specs: Sequence[LayoutSpec], spec: LayoutSpec) -> list[str]:
    """Event texts of earlier timeline specs with the same id, in order, without repeats or the spec's own."""
    if not spec.id:
        return []
    mine = {e.text for e in spec.elements if e.role == "event"}
    out: list[str] = []
    for s in previous_specs:
        if s is spec or s.kind != LayoutKind.timeline or s.id != spec.id:
            continue
        for e in s.elements:
            if e.role == "event" and e.text and e.text not in out and e.text not in mine:
                out.append(e.text)
    return out


def _cells(text: str) -> list[str]:
    return [c.strip() for c in (text or "").split("|")]


def _min2(text: str, f: Any, tr: float = 0.0) -> float:
    """Narrowest width at which text wraps to at most two lines."""
    ws = text.split()
    if len(ws) < 2:
        return text_width(text, f, tr)
    return min(max(text_width(" ".join(ws[:k]), f, tr), text_width(" ".join(ws[k:]), f, tr))
               for k in range(1, len(ws)))


def _fill(frame: np.ndarray, x0: float, y0: float, x1: float, y1: float, col: Sequence[int], a: float = 1.0) -> None:
    """Axis aligned rectangle, opaque or blended."""
    H, W = frame.shape[:2]
    X0, Y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    X1, Y1 = min(W, int(round(x1))), min(H, int(round(y1)))
    if X1 <= X0 or Y1 <= Y0 or a <= 0.003:
        return
    roi = frame[Y0:Y1, X0:X1]
    c = np.array(col[:roi.shape[2]], np.float32)
    if a >= 0.997:
        roi[...] = c.astype(np.uint8)
    else:
        roi[...] = (roi.astype(np.float32) * (1 - a) + c * a).astype(np.uint8)


def _mix(a: Sequence[int], b: Sequence[int], u: float) -> tuple[int, int, int]:
    return tuple(int(round(x + (y - x) * u)) for x, y in zip(a, b))  # type: ignore[return-value]


def _live(c: Cue) -> bool:
    return c.appear < HIDDEN


def _ease_in(t: float, start: float, length: float = SETTLE_S) -> float:
    return ez.ease_out_cubic(ez.progress(t, start, start + length))


# ---------------------------------------------------------------- shared base

class DataComp(LayoutComp):
    """Title with an amber rule, a slow internal drift, floor-aware text and the panel size search."""

    share_start = (0.32, 0.30)
    share_max = (0.62, 0.62)  # a panel larger than this is a full layout; 0.62 high still fits the letterbox

    @classmethod
    def panel_share(cls, spec: LayoutSpec, extra: Optional[dict[str, Any]] = None) -> tuple[float, float]:
        """(width, height) share of a 1920x1080 frame this content needs as a panel, type at the floors.

        Found by laying the content out at growing sizes until its measured need fits (width first, then
        height). panel_then_expand sizes the panel for its stage 1 content only. extra carries the job's
        kind inputs (timeline "history")."""
        W, H = 1920, 1080
        hide = spec.presentation == "panel_then_expand"
        cues = [Cue(i, NEVER if hide and e.stage == 2 else 0.0, NEVER if hide and e.stage == 2 else 0.0,
                    NEVER if hide and e.stage == 2 else 0.0, True, e.stage) for i, e in enumerate(spec.elements)]
        job = LayoutJob(spec=spec, cues=cues, duration=1.0, width=W, height=H, extra=dict(extra or {}))
        (w, h), (mw, mh) = cls.share_start, cls.share_max

        def make(a: float, b: float) -> "DataComp":
            return cls(job, int(W * a) // 2 * 2, int(H * b) // 2 * 2, True)

        c = make(w, mh)
        for _ in range(8):
            if c.need[0] * SLACK <= c.bw + 0.5 or w >= mw:
                break
            w = min(mw, w + (c.need[0] * SLACK - c.bw) / W + 0.004)
            c = make(w, mh)
        c = make(w, h)
        for _ in range(8):
            if c.need[1] * SLACK <= c.bh + 0.5 or h >= mh:
                break
            h = min(mh, h + (c.need[1] * SLACK - c.bh) / H + 0.004)
            c = make(w, h)
        return round(w, 3), round(h, 3)

    def fits(self) -> bool:
        """All text at or above its floor, nothing cut, content inside the box."""
        return self.ok and self.need[0] <= self.bw + 0.5 and self.need[1] <= self.bh + 0.5

    def setup(self) -> None:
        self.ok = True
        self.need = [0.0, 0.0]
        self.t0 = min((c.appear for c in self.job.cues if _live(c)), default=NEVER)
        self.c_accent, self.c_paper, self.c_muted = (self.house.rgb(k) for k in ("accent", "paper", "muted"))
        self.title_spr: Optional[Sprite] = None
        self.title_h = 0.0
        self.pre_build()
        if self.spec.title.strip():
            floor = MIN_BODY_PX if self.panel else MIN_HEAD_PX
            t = self.spec.title.strip()
            self.title_spr = self.txt(t, 60, floor, "SemiBold", True, self.bw, 2, tracking_em=0.08)
            fl = self.fs(floor, floor)
            self.need_w(_min2(t.upper(), font("SemiBold", fl), 0.08 * fl))
            self.rule_h, self.rule_gap = max(3, self.fs(5, 4)), self.fs(14, 10)
            self.title_h = self.title_spr.h + self.rule_gap + self.rule_h + self.fs(40, 26)
        self.build()

    def pre_build(self) -> None:
        pass

    def build(self) -> None:
        raise NotImplementedError

    def need_w(self, v: float) -> None:
        self.need[0] = max(self.need[0], float(v))

    def txt(self, text: str, base: float, floor: float, weight: str = "Medium", caps: bool = False,
            max_w: Optional[float] = None, lines: int = 1, color: str = "paper", tracking_em: float = 0.0,
            align: str = "left") -> Sprite:
        """Text fitted to max_w: shrink to the floor, wrap to `lines`; a cut (ellipsis) marks the layout not ok."""
        t = text.upper() if caps else text
        f, ln = fit_text(t, weight, self.fs(base, floor), max_w or self.bw, lines, self.fs(floor, floor), tracking_em)
        if ELLIPSIS not in t and any(ELLIPSIS in s for s in ln):
            self.ok = False
        tr = tracking_em * f.size
        col = self.house.rgb(color)
        return text_sprite(ln[0], f, col, tr) if len(ln) == 1 else text_block(ln, f, col, align, 1.12, tr)

    def title_at(self) -> float:
        return self.t0

    def off(self, t: float) -> tuple[float, float]:
        """The slow drift of the whole content over the clip (centred, well inside the margins)."""
        p = ez.ease_in_out_sine(ez.clamp01(t / max(self.dur, 1e-3))) - 0.5
        return 0.014 * self.bw * p, 0.008 * self.bh * p

    def draw_title(self, frame: np.ndarray, t: float, ox: float, oy: float) -> None:
        if self.title_spr is None:
            return
        ta = self.title_at()
        o = 1.0 if ta <= 0 and t <= 0 else _ease_in(t, ta)
        if t < ta or o <= 0:
            return
        x, y = self.box[0] + ox, self.box[1] + oy
        blit(frame, self.title_spr, x, y + self.fs(14, 8) * (1 - o), o)
        rw = min(self.title_spr.w, max(self.fs(140, 60), 0.12 * self.bw)) * o
        ry = y + self.title_spr.h + self.rule_gap
        _fill(frame, x, ry, x + rw, ry + self.rule_h, self.c_accent)


# ---------------------------------------------------------------- chart

@register(LayoutKind.chart)
class Chart(DataComp):
    """Bars from a zero baseline growing on their words, or a line drawn point by point."""

    share_start = (0.32, 0.34)

    def build(self) -> None:
        bars = [x for x in self.job.els("bar") if _live(x[2])]
        pts = [x for x in self.job.els("point") if _live(x[2])]
        self.line = not bars and bool(pts)
        self.items: list[dict[str, Any]] = []
        for _i, e, c in (bars or pts):
            lab, shown, num = parse_value(e.text)
            self.items.append({"lab": lab, "shown": shown, "v": num if num is not None else 0.0, "cue": c})
        self.emph = chart_emphasis([it["v"] for it in self.items], self.line)
        if self.items:
            (self._build_line if self.line else self._build_bars)()

    def _value_sprites(self, base: float, floor: float) -> float:
        f = font("Bold", self.fs(base, floor))
        for k, it in enumerate(self.items):
            col = self.c_accent if k == self.emph else self.c_paper
            it["vs"] = text_sprite(it["shown"], f, col) if it["shown"] else None
        return max((it["vs"].w for it in self.items if it["vs"] is not None), default=0)

    # bars
    def _build_bars(self) -> None:
        """Up to three bars: label above each bar. Four or more: labels in a column left of the bars, so
        the chart stays short enough for a panel."""
        n = len(self.items)
        inline = n >= 4
        fl = font("Medium", self.fs(MIN_BODY_PX, MIN_BODY_PX))
        ind = self.fs(14, 10)
        lab_max = 0.45 * self.bw if inline else self.bw - ind
        for it in self.items:
            it["ls"] = self.txt(it["lab"], 54, MIN_BODY_PX, "Medium", max_w=lab_max, lines=2)
        vw = self._value_sprites(96, 48)
        self.bt = self.fs(84, 30)
        self.lgap, self.vgap = self.fs(12, 8), self.fs(20, 14)
        self.base_w = max(2, self.fs(4, 3))
        labw = max(it["ls"].w for it in self.items)
        lab_need = max(_min2(it["lab"], fl) for it in self.items)
        if inline:
            rows = [max(it["ls"].h, self.bt) for it in self.items]
            self.x0 = self.box[0] + labw + self.vgap + self.base_w
            nat = max(text_width(it["lab"], fl) for it in self.items)  # one line labels: the panel widens
            self.need_w(max(nat / 0.45, nat + self.vgap + vw * 3.2 + self.vgap + self.base_w))
        else:
            rows = [it["ls"].h + self.lgap + self.bt for it in self.items]
            self.x0 = self.box[0] + self.base_w
            self.need_w(lab_need + ind + self.base_w)
            self.need_w(max(vw * 3.2 + self.vgap, labw + ind) + self.base_w)
        mingap, maxgap = self.fs(30, 20 if inline else 24), self.fs(110, 30)
        top = self.box[1] + self.title_h
        avail = self.box[3] - top
        gap = min(maxgap, max(mingap, (avail - sum(rows)) / n)) if n > 1 else 0.0
        block = sum(rows) + gap * (n - 1)
        y = top + max(0.0, (avail - block) / 2)
        for it, rh in zip(self.items, rows):
            if inline:
                it["lx"], it["ly"], it["by"] = self.box[0], y + (rh - it["ls"].h) / 2, y + (rh - self.bt) / 2
            else:
                it["lx"], it["ly"], it["by"] = self.x0 + ind, y, y + it["ls"].h + self.lgap
            y += rh + gap
        self.L = self.box[2] - self.x0 - vw - self.vgap - 2
        self.vmax = max(max(it["v"] for it in self.items), 1e-9)
        self.need[1] = self.title_h + sum(rows) + mingap * (n - 1)
        self.axis = (self.items[0]["by"] - self.fs(10, 6), self.items[-1]["by"] + self.bt + self.fs(10, 6))

    def _draw_bars(self, frame: np.ndarray, t: float, ox: float, oy: float) -> None:
        a0, a1 = self.axis
        ga = _ease_in(t, self.t0, 0.4)
        _fill(frame, self.x0 - self.base_w + ox, a0 + oy, self.x0 + ox, a0 + (a1 - a0) * ga + oy, self.c_paper, 0.9)
        ce = self.items[self.emph]["cue"]
        u = ez.progress(t, ce.appear, ce.appear + 0.4)
        for k, it in enumerate(self.items):
            c = it["cue"]
            if t < c.appear:
                continue
            o = _ease_in(t, c.appear)
            blit(frame, it["ls"], it["lx"] + ox, it["ly"] + oy + self.fs(12, 8) * (1 - o), o)
            end = max(c.word + 0.35, c.appear + 0.5)
            full = self.L * max(0.0, it["v"]) / self.vmax
            ln = full * ez.ease_out_cubic(ez.progress(t, c.appear, end))
            col = self.c_accent if k == self.emph else _mix(self.c_paper, self.c_muted, u)
            _fill(frame, self.x0 + ox, it["by"] + oy, self.x0 + ln + ox, it["by"] + self.bt + oy, col)
            va = ez.progress(t, end, end + 0.15)  # the true value, only once the bar has its true length
            if va > 0 and it["vs"] is not None:
                blit(frame, it["vs"], self.x0 + full + self.vgap + ox, it["by"] + (self.bt - it["vs"].h) / 2 + oy, va)

    # line
    def _build_line(self) -> None:
        n = len(self.items)
        gx = self.fs(24, 16)
        slot = self.bw / n
        fl = font("Medium", self.fs(MIN_LABEL_PX, MIN_LABEL_PX))
        for it in self.items:
            it["ls"] = self.txt(it["lab"], 44, MIN_LABEL_PX, "Medium", max_w=slot - gx, lines=2, align="center")
        vw = self._value_sprites(64, MIN_BODY_PX)
        lab_min = max(_min2(it["lab"], fl) for it in self.items)
        self.need_w(n * (max(lab_min, vw) + gx))
        xl_h = max(it["ls"].h for it in self.items)
        v_h = max((it["vs"].h for it in self.items if it["vs"] is not None), default=0)
        g = self.fs(18, 12)
        self.top = self.box[1] + self.title_h + v_h + g
        self.bottom = self.box[3] - xl_h - g
        self.need[1] = self.title_h + v_h + g + self.fs(200, 140) + g + xl_h
        vals = [it["v"] for it in self.items]
        lo, hi = min(0.0, min(vals)), max(vals)
        hi = hi if hi > lo else lo + 1.0
        for k, it in enumerate(self.items):
            it["x"] = self.box[0] + slot * (k + 0.5)
            it["y"] = self.bottom - (it["v"] - lo) / (hi - lo) * (self.bottom - self.top)
        self.lw = max(3, self.fs(6, 4))
        self.r = self.fs(11, 8)

    def _reach(self) -> list[tuple[float, float]]:
        """(start, end) of the line's travel to each point; point 0 pops on its own cue."""
        out, prev = [], 0.0
        for k, it in enumerate(self.items):
            c = it["cue"]
            if k == 0:
                s, e = c.appear, max(c.word, c.appear + 0.2)
            else:
                s = max(c.appear, prev)
                e = max(c.word + 0.25, s + 0.3)
            out.append((s, e))
            prev = e
        return out

    def _draw_line(self, frame: np.ndarray, t: float, ox: float, oy: float) -> None:
        ga = _ease_in(t, self.t0, 0.5)
        _fill(frame, self.box[0] + ox, self.bottom + oy, self.box[0] + self.bw * ga + ox,
              self.bottom + max(2, self.fs(3, 2)) + oy, self.c_paper, 0.8)
        P = np.array([(it["x"] + ox, it["y"] + oy) for it in self.items], np.float64)
        reach = self._reach()
        path = []
        for k, (s, e) in enumerate(reach):
            if t < s:
                break
            if k == 0:
                path.append(P[0])
                continue
            q = ez.ease_in_out_sine(ez.progress(t, s, e))
            path.append(P[k - 1] + (P[k] - P[k - 1]) * q)
            if q < 1:
                break
        if len(path) > 1:
            polyline(frame, np.array(path), self.c_paper, self.lw)
        for k, it in enumerate(self.items):
            c, (s, e) = it["cue"], reach[k]
            if t < c.appear:
                continue
            o = _ease_in(t, c.appear)
            blit(frame, it["ls"], it["x"] - it["ls"].w / 2 + ox, self.bottom + self.fs(18, 12) + oy + self.fs(10, 6) * (1 - o), o)
            if t < e - (0.2 if k == 0 else 0.05):
                continue
            pop = ez.ease_out_back(ez.progress(t, e - (0.2 if k == 0 else 0.05), e + 0.3), 2.0)
            emph = k == self.emph
            if emph:
                disc(frame, P[k][0], P[k][1], self.r * 2.1 * pop, self.c_accent, max(2, self.fs(3, 2)))
            disc(frame, P[k][0], P[k][1], self.r * (1.4 if emph else 1.0) * pop, self.c_accent if emph else self.c_paper)
            va = ez.progress(t, e, e + 0.15)
            if va > 0 and it["vs"] is not None:
                vy = P[k][1] - self.r * 2.4 - it["vs"].h
                blit(frame, it["vs"], P[k][0] - it["vs"].w / 2, vy, va)

    def draw(self, frame: np.ndarray, t: float) -> None:
        ox, oy = self.off(t)
        self.draw_title(frame, t, ox, oy)
        if not self.items or t < self.t0:
            return
        (self._draw_line if self.line else self._draw_bars)(frame, t, ox, oy)


# ---------------------------------------------------------------- table

@register(LayoutKind.table)
class Table(DataComp):
    """Rows appear as they are named; the row being spoken carries an amber left rule and band."""

    share_start = (0.36, 0.32)

    def build(self) -> None:
        hd = [x for x in self.job.els("header") if _live(x[2])]
        self.header = (_cells(hd[0][1].text), hd[0][2]) if hd else None
        self.rows = [(_cells(e.text), c) for _i, e, c in self.job.els("row") if _live(c)]
        allrows = [r for r, _ in self.rows] + ([self.header[0]] if self.header else [])
        if not allrows:
            self.cols = 0
            return
        nc = self.cols = max(len(r) for r in allrows)
        for r in allrows:
            r.extend([""] * (nc - len(r)))
        body = [r for r, _ in self.rows]
        self.right = [bool(any(r[c] for r in body)) and all(_NUMERIC_CELL.match(r[c]) for r in body if r[c])
                      for c in range(nc)]
        self.rule, self.padx, self.pady = max(3, self.fs(6, 5)), self.fs(22, 16), self.fs(14, 10)
        gapc = self.fs(44, 30)
        lead = self.rule + self.padx
        fixed = lead + self.padx + gapc * (nc - 1)
        avail = self.bw - fixed
        trh = 0.06
        hdr = self.header[0] if self.header else [""] * nc

        def natural(sz: int) -> list[float]:
            fb, fh = font("Medium", sz), font("SemiBold", sz)
            return [max([text_width(r[c], fb) for r in body] + [text_width(hdr[c].upper(), fh, trh * sz)])
                    for c in range(nc)]

        floor_sz = self.fs(MIN_BODY_PX, MIN_BODY_PX)
        size = self.fs(64, MIN_BODY_PX)
        nat = natural(size)
        while sum(nat) > avail and size > floor_sz:
            size = max(floor_sz, int(size * 0.94))
            nat = natural(size)
        fb, fh = font("Medium", size), font("SemiBold", size)
        m2 = [max([_min2(r[c], fb) for r in body] + [_min2(hdr[c].upper(), fh, trh * size)]) for c in range(nc)]
        widths = list(nat)
        deficit = sum(widths) - avail
        for c in sorted(range(nc), key=lambda c: -widths[c]):  # wrap the widest columns to two lines first
            if deficit <= 0:
                break
            cut = min(deficit, widths[c] - m2[c])
            widths[c] -= cut
            deficit -= cut
        nat_floor = natural(floor_sz)
        self.need_w(fixed + sum(nat_floor))  # one line per cell at the floor: the panel size to aim for
        if deficit > 0:
            self.ok = False
        extra = avail - sum(widths)
        if nc > 1 and extra > 0:
            gapc += min(extra / (nc - 1), gapc)
        self.widths, self.gapc = widths, gapc
        self.table_w = lead + sum(widths) + gapc * (nc - 1) + self.padx
        xs, x = [], self.box[0] + lead
        for w in widths:
            xs.append(x)
            x += w + gapc
        self.xs = xs

        def cell(text: str, c: int, weight: str, f: Any, color: Sequence[int], tr: float) -> Optional[Sprite]:
            if not text:
                return None
            ln = wrap(text, f, widths[c] + 0.5, tr)
            if len(ln) > 2 or any(text_width(s, f, tr) > widths[c] + 1 for s in ln):
                _f, ln = fit_text(text, weight, f.size, widths[c], 2, f.size, tr / f.size)
                self.ok = False
            al = "right" if self.right[c] else "left"
            return text_sprite(ln[0], f, color, tr) if len(ln) == 1 else text_block(ln, f, color, al, 1.12, tr)

        self.hcells = [cell(hdr[c].upper(), c, "SemiBold", fh, self.c_muted, trh * size) for c in range(nc)] if self.header else []
        self.bcells = [[cell(r[c], c, "Medium", fb, self.c_paper, 0.0) for c in range(nc)] for r in body]
        self.hh = (max((s.h for s in self.hcells if s), default=0) + 2 * self.pady) if self.header else 0.0
        self.rh = [max((s.h for s in row if s), default=fb.size) + 2 * self.pady for row in self.bcells]
        top = self.box[1] + self.title_h
        block = self.hh + sum(self.rh)
        self.need[1] = self.title_h + block
        self.y0 = top + max(0.0, self.box[3] - top - block) * 0.4
        self.line_h = max(2, self.fs(2, 2))

    def _cell_x(self, c: int, spr: Sprite) -> float:
        return self.xs[c] + (self.widths[c] - spr.w if self.right[c] else 0)

    def draw(self, frame: np.ndarray, t: float) -> None:
        ox, oy = self.off(t)
        self.draw_title(frame, t, ox, oy)
        if not self.cols or t < self.t0:
            return
        bx = self.box[0] + ox
        y = self.y0 + oy
        rise = self.fs(14, 8)
        if self.header:
            c = self.header[1]
            if t >= c.appear:
                o = _ease_in(t, c.appear)
                for k, s in enumerate(self.hcells):
                    if s is not None:
                        blit(frame, s, self._cell_x(k, s) + ox, y + self.pady + rise * (1 - o), o)
                _fill(frame, bx, y + self.hh - self.line_h, bx + self.table_w * o, y + self.hh, self.c_paper, 0.55 * o)
            y += self.hh
        for r, ((_cells_, c), row, rh) in enumerate(zip(self.rows, self.bcells, self.rh)):
            if t >= c.appear:
                o = _ease_in(t, c.appear)
                nxt = self.rows[r + 1][1].appear if r + 1 < len(self.rows) else NEVER
                e = ez.progress(t, c.appear, c.appear + 0.25) * (1 - ez.progress(t, nxt, nxt + 0.25))
                _fill(frame, bx, y, bx + self.table_w, y + rh, self.c_accent, 0.13 * e)
                _fill(frame, bx, y, bx + self.rule, y + rh, self.c_accent, e)
                for k, s in enumerate(row):
                    if s is not None:
                        blit(frame, s, self._cell_x(k, s) + ox, y + self.pady + rise * (1 - o), o)
                _fill(frame, bx, y + rh - self.line_h, bx + self.table_w * o, y + rh, self.c_muted, 0.35 * o)
            y += rh


# ---------------------------------------------------------------- timeline

@register(LayoutKind.timeline)
class Timeline(DataComp):
    """A line that gains each event as it is spoken: it extends, a dot pops, date and label settle in.
    job.extra["history"] holds earlier event texts of the same timeline id, shown from the first frame."""

    share_start = (0.46, 0.26)

    def pre_build(self) -> None:
        new = [(e.text, c) for _i, e, c in self.job.els("event") if _live(c)]
        own = {s for s, _ in new}
        hist: list[str] = []
        for s in self.job.extra.get("history") or []:
            if isinstance(s, str) and s and s not in own and s not in hist:
                hist.append(s)
        self.hist = hist

    def title_at(self) -> float:
        return 0.0 if self.hist else self.t0

    def build(self) -> None:
        new = [(e.text, c) for _i, e, c in self.job.els("event") if _live(c)]
        evs = [{"text": s, "cue": None} for s in self.hist] + [{"text": s, "cue": c} for s, c in new]
        self.evs = evs
        if not evs:
            return
        for k, ev in enumerate(evs):
            ev["date"], ev["lab"], ev["year"] = parse_event(ev["text"])
            ev["k"] = k
        dated = all(ev["year"] is not None for ev in evs)
        order = sorted(evs, key=lambda ev: (ev["year"], ev["k"])) if dated else list(evs)
        n = len(evs)
        m = math.ceil(n / 2) if n > 1 else 1
        gap = self.fs(40, 28)
        wl = min(0.42 * self.bw, (self.bw - gap * (m - 1)) / m)
        fd = font("SemiBold", self.fs(MIN_BODY_PX, MIN_BODY_PX))
        fl = font("Medium", self.fs(MIN_BODY_PX, MIN_BODY_PX))
        wmin = max(max(text_width(ev["date"].upper(), fd, 0.06 * fd.size), _min2(ev["lab"], fl)) for ev in evs)
        wnat = max(max(text_width(ev["date"].upper(), fd, 0.06 * fd.size), text_width(ev["lab"], fl)) for ev in evs)
        wone = wnat if wnat < 0.42 * 1180 else wmin  # short labels stay on one line: the panel widens for them
        self.need_w(max(wone / 0.42, m * wone + gap * (m - 1)))
        for ev in evs:
            hist = ev["cue"] is None
            if ev["date"]:
                ev["d_acc"] = self.txt(ev["date"], 60, MIN_BODY_PX, "SemiBold", True, wl, 1, "accent", 0.06)
                ev["d_pap"] = self.txt(ev["date"], 60, MIN_BODY_PX, "SemiBold", True, wl, 1,
                                       "muted" if hist else "paper", 0.06)
            else:
                ev["d_acc"] = ev["d_pap"] = None
            ev["ls"] = self.txt(ev["lab"], 54, MIN_BODY_PX, "Medium", False, wl, 2, "muted" if hist else "paper",
                                align="center") if ev["lab"] else None
            dh = ev["d_pap"].h if ev["d_pap"] is not None else 0
            ev["dl"] = self.fs(6, 4) if dh and ev["ls"] is not None else 0
            ev["bh"] = dh + ev["dl"] + (ev["ls"].h if ev["ls"] is not None else 0)
        for rank, ev in enumerate(order):
            ev["above"] = rank % 2 == 0
        self.stem, self.r = self.fs(22, 16), self.fs(11, 8)
        above = max((ev["bh"] for ev in evs if ev["above"]), default=0)
        below = max((ev["bh"] for ev in evs if not ev["above"]), default=0)
        content = above + self.stem + 2 * self.r + (self.stem + below if below else 0)
        top = self.box[1] + self.title_h
        self.need[1] = self.title_h + content
        self.axis_y = top + max(0.0, self.box[3] - top - content) * 0.5 + above + self.stem + self.r
        lo, hi = self.box[0] + wl / 2, self.box[2] - wl / 2
        years = [ev["year"] for ev in order]
        if n == 1:
            xs = [(lo + hi) / 2]
        elif dated and len(set(years)) > 1:
            xs = [lo + (y - years[0]) / (years[-1] - years[0]) * (hi - lo) for y in years]
        else:
            xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        mind = (wl + gap) / 2  # alternate sides: a neighbour on the same side is two steps away
        for i in range(1, n):
            xs[i] = max(xs[i], xs[i - 1] + mind)
        if n > 1 and xs[-1] > hi:
            xs[-1] = hi
            for i in range(n - 2, -1, -1):
                xs[i] = min(xs[i], xs[i + 1] - mind)
            if xs[0] < lo:
                xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        for ev, x in zip(order, xs):
            ev["x"] = x
        self.tail = self.fs(60, 30)
        self.left = self.box[0]
        self.hist_ext = max((ev["x"] + self.tail for ev in evs if ev["cue"] is None), default=None)
        self.new = [ev for ev in evs if ev["cue"] is not None]
        self.th = max(2, self.fs(4, 3))

    def _extent(self, t: float) -> Optional[float]:
        e = self.hist_ext
        last = 0.0
        for ev in self.new:
            c = ev["cue"]
            if t < c.appear:
                break
            start = e if e is not None else self.left
            tgt = max(start, ev["x"] + self.tail)
            e = ez.lerp(start, tgt, ez.ease_out_cubic(ez.progress(t, c.appear, c.word + 0.1)))
            last = c.word + 0.1
        if e is None:
            return None
        return min(self.box[2], e + self.fs(8, 4) * max(0.0, t - last))  # keeps creeping on toward the future

    def _event(self, frame: np.ndarray, ev: dict[str, Any], o: float, pop: float, hot: float, ox: float, oy: float) -> None:
        x, ay = ev["x"] + ox, self.axis_y + oy
        sgn = -1 if ev["above"] else 1
        s0 = ay + sgn * self.r
        s1 = s0 + sgn * self.stem * o
        _fill(frame, x - 1, min(s0, s1), x + 1, max(s0, s1), self.c_paper, 0.6 * o)
        if pop > 0:
            col = _mix(self.c_muted if ev["cue"] is None else self.c_paper, self.c_accent, hot)
            disc(frame, x, ay, self.r * (1 + 0.3 * hot) * pop, col)
        y = ay - self.r - self.stem - ev["bh"] if ev["above"] else ay + self.r + self.stem
        y += self.fs(12, 8) * (1 - o) * sgn
        if ev["d_pap"] is not None:
            d = ev["d_pap"]
            dx = self._clamp_x(x - d.w / 2, d.w)
            blit(frame, ev["d_acc"], dx, y, o * hot)
            blit(frame, d, dx, y, o * (1 - hot))
            y += d.h + ev["dl"]
        if ev["ls"] is not None:
            s = ev["ls"]
            blit(frame, s, self._clamp_x(x - s.w / 2, s.w), y, o)

    def _clamp_x(self, x: float, w: float) -> float:
        return min(max(x, self.box[0]), self.box[2] - w)

    def draw(self, frame: np.ndarray, t: float) -> None:
        ox, oy = self.off(t)
        self.draw_title(frame, t, ox, oy)
        if not self.evs:
            return
        ext = self._extent(t)
        if ext is None:
            return
        _fill(frame, self.left + ox, self.axis_y - self.th / 2 + oy, ext + ox, self.axis_y + self.th / 2 + oy,
              self.c_paper, 0.85)
        for ev in self.evs:
            if ev["cue"] is None:
                self._event(frame, ev, 1.0, 1.0, 0.0, ox, oy)
        for j, ev in enumerate(self.new):
            c = ev["cue"]
            if t < c.appear:
                continue
            s = max(c.appear, c.word - 0.1)
            pop = ez.ease_out_back(ez.progress(t, s, c.word + 0.25), 2.0)
            o = _ease_in(t, max(c.appear, c.word - 0.05))
            nxt = self.new[j + 1]["cue"].appear if j + 1 < len(self.new) else NEVER
            hot = ez.progress(t, s, s + 0.2) * (1 - ez.progress(t, nxt, nxt + 0.3))
            self._event(frame, ev, o, pop, hot, ox, oy)


__all__ = ["Chart", "Table", "Timeline", "chart_emphasis", "parse_value", "parse_event", "timeline_history"]
