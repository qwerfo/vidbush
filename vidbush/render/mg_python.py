"""Built-in Python motion graphics renderer (the default behind the MotionSpec contract).

render_spec_python(spec, out_path, alpha, settings) renders exactly
round(spec.duration * spec.fps) frames at spec.width x spec.height and pipes
them as raw RGB (or straight RGBA) to ffmpeg: H.264 yuv420p mp4 when alpha is
False, qtrle argb mov when alpha is True. Frames are drawn premultiplied and
converted to straight alpha at the end, so overlays composite without dark
halos. Full-frame graphics fade in and out over 0.3 s; overlays animate in and
out themselves (wipes, slides, traces).

This module holds the text compositions (title_card, number_card,
lower_third, callout, text_overlay, highlight_box, arrow, recap_grid); maps
and charts live in vidbush/mg/mapdraw.py and vidbush/mg/chartdraw.py.
"""
from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from pydantic import ValidationError

from ..mg import easing as ez
from ..mg.palettes import get_palette, mix
from ..mg.textdraw import (
    CompBase,
    GlowLayer,
    Sprite,
    arrow_head,
    blit,
    disc,
    fade,
    fill_poly,
    fit_text,
    font,
    glow_sprite,
    on_colour,
    path_prefix,
    pill,
    polyline,
    rect,
    round_rect,
    scaled,
    text_block,
    text_sprite,
    text_width,
    unpremultiply,
    with_shadow,
)
from ..models import MotionKind, MotionSpec, TextPayload
from ..util import ToolError

WHITE = (255, 255, 255)
TNUM = ["tnum"]


# ---------------------------------------------------------------- entry point

def render_spec_python(spec: MotionSpec, out_path: Path, alpha: bool, settings: Any) -> Path:
    """Render spec to out_path (mp4 or alpha mov). Invalid payloads fall back to a text_overlay."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comp = build_comp(spec, alpha)
    n = max(1, round(spec.duration * spec.fps))
    preset = getattr(getattr(settings, "render", None), "preset", "veryfast")
    full_fade = not comp.transparent or not comp.own_exit
    with FrameSink(out_path, spec.width, spec.height, spec.fps, alpha, preset) as sink:
        for i in range(n):
            frame = comp.render(i)
            if full_fade:
                frame = fade(frame, ez.fade_factor(i / spec.fps, spec.duration, 0.3))
            sink.write(frame)
    return out_path


def fallback_spec(spec: MotionSpec, text: str) -> MotionSpec:
    """A text_overlay carrying `text`, same size, timing and palette as spec."""
    payload = TextPayload(title=(text or spec.kind.value.replace("_", " ")).strip()[:160], position="center")
    return spec.model_copy(update={"kind": MotionKind.text_overlay, "payload": payload.model_dump()})


def build_comp(spec: MotionSpec, alpha: bool) -> CompBase:
    """Composition object for spec; a payload that does not validate becomes a text_overlay."""
    try:
        payload = spec.typed_payload()
    except (ValidationError, TypeError, ValueError):
        raw = spec.payload if isinstance(spec.payload, dict) else {}
        title = str(raw.get("title") or raw.get("label") or "")
        spec = fallback_spec(spec, title)
        payload = spec.typed_payload()
    return _registry()[spec.kind](spec, payload, get_palette(spec.style.palette), alpha)


def _registry() -> dict[MotionKind, Callable[..., CompBase]]:
    from ..mg.chartdraw import BarChart, LineChart
    from ..mg.mapdraw import MapComp
    return {
        MotionKind.map: MapComp, MotionKind.bar_chart: BarChart, MotionKind.line_chart: LineChart,
        MotionKind.number_card: NumberCard, MotionKind.title_card: TitleCard, MotionKind.lower_third: LowerThird,
        MotionKind.callout: Callout, MotionKind.highlight_box: HighlightBox, MotionKind.text_overlay: TextOverlay,
        MotionKind.arrow: Arrow, MotionKind.recap_grid: RecapGrid,
    }


class FrameSink:
    """ffmpeg process fed raw frames on stdin; writes to a temp file and renames on success."""

    def __init__(self, out: Path, w: int, h: int, fps: int, alpha: bool, preset: str) -> None:
        self.out, self.alpha = out, alpha
        self.tmp = out.with_name(out.stem + ".part" + out.suffix)
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba" if alpha else "rgb24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
        if alpha:
            cmd += ["-c:v", "qtrle", "-pix_fmt", "argb"]
        else:
            cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv", "-c:v", "libx264", "-preset", preset,
                    "-crf", "18", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
                    "-color_trc", "bt709", "-movflags", "+faststart"]
        self.cmd = cmd + [str(self.tmp)]
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "FrameSink":
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        return self

    def write(self, frame: np.ndarray) -> None:
        data = unpremultiply(frame) if self.alpha else frame
        try:
            self.proc.stdin.write(np.ascontiguousarray(data).tobytes())
        except BrokenPipeError as e:
            raise ToolError(f"ffmpeg closed the pipe: {self.proc.stderr.read().decode(errors='replace')[-800:]}") from e

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        code = self.proc.wait()
        if exc_type is None and code != 0:
            raise ToolError(f"ffmpeg failed ({code}): {err[-800:]}")
        if exc_type is None:
            self.tmp.replace(self.out)
        else:
            self.tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------- title card

class TitleCard(CompBase):
    """Cinematic title: tracking settles from wide to tight, thin rules grow, subtitle follows."""

    def __init__(self, spec: MotionSpec, payload: TextPayload, pal, alpha: bool) -> None:  # noqa: ANN001
        super().__init__(spec, payload, pal, alpha)
        self.col = WHITE if self.transparent else pal.rgb("text")
        max_w = self.W - 2.6 * self.margin
        self.f, self.lines = fit_text(payload.title.upper(), "Bold", self.px(120), max_w, 2, self.px(44), 0.08)
        self.tr1 = 0.08 * self.f.size
        widest = max(text_width(ln, self.f) for ln in self.lines)
        chars = max(len(ln) for ln in self.lines)
        room = (self.W * 0.94 - widest) / max(1, chars - 1)
        self.tr0 = max(self.tr1, min(0.45 * self.f.size, room))
        self.final = self._title(self.tr1)
        self.sub: Optional[Sprite] = None
        if payload.subtitle:
            sf, sl = fit_text(payload.subtitle.upper(), "Medium", self.px(34), max_w * 0.9, 2, self.px(20), 0.26)
            self.sub = text_block(sl, sf, pal.rgb("accent"), "center", 1.3, 0.26 * sf.size, self._shadow())
        gap = self.px(34)
        block = self.final.h + (gap + self.sub.h if self.sub else 0)
        self.y0 = (self.H - block) / 2 - self.px(10)
        self.rule_w = min(self.W * 0.5, max(self.px(260), self.final.w * 0.55))

    def _shadow(self) -> float:
        return 0.6 if self.transparent else 0.0

    def _title(self, tracking: float) -> Sprite:
        return text_block(self.lines, self.f, self.col, "center", 1.04, tracking, self._shadow())

    def draw(self, frame: np.ndarray, t: float) -> None:
        settle = ez.ease_out_expo(ez.progress(t, 0.0, min(2.4, self.dur * 0.6)))
        spr = self.final if settle > 0.995 else self._title(ez.lerp(self.tr0, self.tr1, settle))
        op = ez.ease_out_cubic(ez.progress(t, 0.05, 0.9))
        blit(frame, spr, (self.W - spr.w) / 2, self.y0 + self.px(16) * (1 - op), op)
        grow = ez.ease_in_out_cubic(ez.progress(t, 0.3, 1.4))
        w = self.rule_w * grow
        accent = self.pal.rgb("accent")
        top = self.y0 - self.px(40)
        bottom = self.y0 + self.final.h + (self.px(34) + self.sub.h if self.sub else 0) + self.px(40)
        for y in (top, bottom):
            if w > 1:
                blit(frame, rect(w, max(1, self.px(2)), accent), (self.W - w) / 2, y, 0.9)
        d = self.px(7) * ez.ease_out_back(ez.progress(t, 0.9, 1.4))
        if d > 0.5:
            cx = self.W / 2
            fill_poly(frame, [np.array([[cx, top - d], [cx + d, top + 1], [cx, top + d + 2], [cx - d, top + 1]])],
                      accent)
        if self.sub is not None:
            so = ez.ease_out_cubic(ez.progress(t, 0.8, 1.6))
            blit(frame, self.sub, (self.W - self.sub.w) / 2,
                 self.y0 + self.final.h + self.px(34) + self.px(10) * (1 - so), so)


# ---------------------------------------------------------------- number card

NUM_RE = re.compile(r"^(?P<pre>[^\d\-]*?)(?P<num>-?\d[\d,]*(?:\.\d+)?)(?P<suf>.*)$")


def parse_number(text: str) -> Optional[tuple[str, float, str, int, bool]]:
    """'$375M' -> ('$', 375.0, 'M', 0, False); None when there is no number to count."""
    m = NUM_RE.match(text.strip())
    if not m:
        return None
    num = m.group("num")
    try:
        value = float(num.replace(",", ""))
    except ValueError:
        return None
    decimals = len(num.split(".")[1]) if "." in num else 0
    return m.group("pre"), value, m.group("suf"), decimals, "," in num


def format_count(value: float, decimals: int, commas: bool) -> str:
    """Number formatted like the target (same decimals, thousands separators when the target had them)."""
    return f"{value:,.{decimals}f}" if commas else f"{value:.{decimals}f}"


def count_start(value: float, pre: str, suf: str, commas: bool, decimals: int) -> float:
    """Where a count up starts: years count up from a few decades earlier, everything else from zero."""
    if not pre and not suf.strip() and not commas and decimals == 0 and 1000 <= value <= 2100:
        return value - 60
    return 0.0


class NumberCard(CompBase):
    """Huge number (counting up when numeric and count_up), accent rule, label and sublabel."""

    def __init__(self, spec: MotionSpec, payload, pal, alpha: bool) -> None:  # noqa: ANN001
        super().__init__(spec, payload, pal, alpha)
        p = payload
        self.parsed = parse_number(p.number) if p.count_up else None
        if self.parsed and self.parsed[0].strip() == "#":
            self.parsed = None
        dark = pal.dark or self.transparent
        self.col = pal.rgb("accent") if pal.name == "top10_bold" else (WHITE if self.transparent else pal.rgb("text"))
        self.glow_col = pal.rgb("accent") if dark else None
        self.nf, _ = fit_text(p.number, "Black", self.px(300), self.W - 2 * self.margin, 1, self.px(90))
        self.shadow = 0.55 if self.transparent else 0.0
        self.final = self._num(p.number)
        tc = WHITE if self.transparent else pal.rgb("text")
        self.label = self._line(p.label.upper(), "SemiBold", 52, 0.7, tc, 0.12) if p.label else None
        sc = (220, 220, 220) if self.transparent else pal.rgb("muted")
        self.sub = self._line(p.sublabel, "Regular", 34, 0.62, sc, 0.0) if p.sublabel else None
        self.gap = self.px(30)
        hs = [self.final.h, self.px(6)] + [s.h for s in (self.label, self.sub) if s is not None]
        total = sum(hs) + self.gap * (len(hs) - 1)
        self.y0 = max(self.safe_top, (self.H - total) / 2)
        self._cache: tuple[str, Sprite, Optional[Sprite]] = ("", self.final, None)

    def _line(self, text: str, weight: str, size: int, wf: float, col, tr_em: float) -> Sprite:  # noqa: ANN001
        f, lines = fit_text(text, weight, self.px(size), self.W * wf, 2, self.px(size * 0.5), tr_em)
        return text_block(lines, f, col, "center", 1.2, tr_em * f.size, self.shadow)

    def _num(self, text: str) -> Sprite:
        return text_sprite(text, self.nf, self.col, shadow=self.shadow, features=TNUM)

    def _number_text(self, t: float) -> str:
        if not self.parsed:
            return self.p.number
        pre, value, suf, dec, commas = self.parsed
        start = count_start(value, pre, suf, commas, dec)
        p = ez.ease_out_expo(ez.progress(t, 0.2, max(0.6, min(self.dur * 0.6, 2.6))))
        v = start + (value - start) * p
        return pre + format_count(v if p < 1 else value, dec, commas) + suf

    def draw(self, frame: np.ndarray, t: float) -> None:
        text = self._number_text(t)
        if text != self._cache[0]:
            spr = self._num(text)
            glow = glow_sprite(spr, self.glow_col, self.px(40), 0.35) if self.glow_col else None
            self._cache = (text, spr, glow)
        _, spr, glow = self._cache
        pop = ez.ease_out_back(ez.progress(t, 0.0, 0.7), 1.2)
        op = ez.ease_out_cubic(ez.progress(t, 0.0, 0.45))
        s = 0.9 + 0.1 * pop
        cx, y = self.W / 2, self.y0
        if glow is not None:
            g = scaled(glow, s)
            blit(frame, g, cx - g.w / 2, y + self.final.h * (1 - s) / 2, op * 0.9)
        sp = scaled(spr, s)
        blit(frame, sp, cx - sp.w / 2, y + self.final.h * (1 - s) / 2, op)
        y += self.final.h + self.gap
        rw = self.px(130) * ez.ease_in_out_cubic(ez.progress(t, 0.35, 1.0))
        if rw > 1:
            blit(frame, rect(rw, self.px(6), self.pal.rgb("accent")), cx - rw / 2, y)
        y += self.px(6) + self.gap
        for k, spr2 in enumerate(s2 for s2 in (self.label, self.sub) if s2 is not None):
            o = ez.ease_out_cubic(ez.progress(t, 0.55 + 0.2 * k, 1.15 + 0.2 * k))
            blit(frame, spr2, cx - spr2.w / 2, y + self.px(14) * (1 - o), o)
            y += spr2.h + self.gap


# ---------------------------------------------------------------- panel overlays (lower third, callout)

class PanelOverlay(CompBase):
    """Accent bar plus a panel that wipes open, with text items fading in; reverses on exit."""

    own_exit = True
    axis = "x"  # wipe direction

    def setup(self, items: list[tuple[Sprite, float, float]], pad: tuple[float, float], position: str,
              inset: float) -> None:
        """items: (sprite, x, y) relative to the panel's inner origin; builds panel geometry."""
        self.items = items
        self.right = position.endswith("right")
        iw = max(s.w + x for s, x, _ in items)
        ih = max(s.h + y for s, _, y in items)
        self.pw, self.ph = int(iw + 2 * pad[0]), int(ih + 2 * pad[1])
        self.bar = max(3, self.px(8))
        self.pad = pad
        self.x0, self.y0 = self.anchor(self.pw + self.bar, self.ph, position, inset)
        self.panel = round_rect(self.pw, self.ph, self.px(4), self.pal.rgb("panel"), 0.93)

    def _layer(self, e_in: float, e_out: float) -> np.ndarray:
        s = self.px(30)
        layer = np.zeros((self.ph + 2 * s, self.pw + 2 * s, 4), np.uint8)
        blit(layer, self.panel, s, s)
        for k, (spr, x, y) in enumerate(self.items):
            o = ez.ease_out_cubic(ez.progress(e_in, 0.35 + 0.12 * k, 0.8 + 0.12 * k))
            o *= 1 - ez.progress(e_out, 0, 0.45)
            dx = self.px(26) * (1 - o) * (1 if self.right else -1) if self.axis == "x" else 0
            dy = self.px(18) * (1 - o) if self.axis == "y" else 0
            blit(layer, spr, s + self.pad[0] + x + dx, s + self.pad[1] + y + dy, o)
        return with_shadow(layer, radius=self.px(18), offset=self.px(6), opacity=0.35)

    def draw(self, frame: np.ndarray, t: float) -> None:
        e_in, e_out = ez.envelope(t, self.dur, 1.0, 0.7)
        wipe = ez.ease_out_cubic(ez.progress(e_in, 0.12, 0.7))
        wipe *= 1 - ez.ease_in_out_cubic(ez.progress(e_out, 0.25, 0.95))
        bar = ez.ease_out_cubic(ez.progress(e_in, 0.0, 0.35)) * (1 - ez.ease_in_cubic(ez.progress(e_out, 0.7, 1.0)))
        s = self.px(30)
        layer = self._layer(e_in, e_out)
        px_x = self.x0 + (0 if self.right else self.bar)
        if self.axis == "x":
            keep = int(s + self.pw * wipe)
            if self.right:
                layer[:, : max(0, layer.shape[1] - keep)] = 0
            else:
                layer[:, keep:] = 0
        else:
            layer[int(s + self.ph * wipe):] = 0
        if wipe > 0.002:
            blit(frame, layer, px_x - s, self.y0 - s)
        if bar > 0.002:
            bh = self.ph * bar
            by = self.y0 + (self.ph - bh) / 2 if self.axis == "x" else self.y0
            bx = self.x0 + self.pw if self.right else self.x0
            blit(frame, rect(self.bar, bh, self.pal.rgb("accent")), bx, by)


class LowerThird(PanelOverlay):
    """Name or place strap above the caption band; bar slides in, panel wipes from the bar."""

    def __init__(self, spec: MotionSpec, payload: TextPayload, pal, alpha: bool) -> None:  # noqa: ANN001
        super().__init__(spec, payload, pal, alpha)
        pos = payload.position if payload.position != "center" else "lower_left"
        tf, tl = fit_text(payload.title, "SemiBold", self.px(50), self.W * 0.42, 1, self.px(28))
        title = text_sprite(tl[0], tf, pal.rgb("panel_text"))
        items = [(title, 0.0, 0.0)]
        if payload.subtitle:
            sf, sl = fit_text(payload.subtitle.upper(), "Medium", self.px(25), self.W * 0.42, 1, self.px(16), 0.16)
            items.append((text_sprite(sl[0], sf, pal.rgb("accent"), 0.16 * sf.size), 0.0, title.h + self.px(4)))
        self.setup(items, (self.px(30), self.px(18)), pos, 40)


class Callout(PanelOverlay):
    """Key phrase box with an accent bar; unfolds from the top, lines fade up in turn."""

    axis = "y"

    def __init__(self, spec: MotionSpec, payload: TextPayload, pal, alpha: bool) -> None:  # noqa: ANN001
        super().__init__(spec, payload, pal, alpha)
        tf, tl = fit_text(payload.title, "Bold", self.px(56), self.W * 0.4, 3, self.px(30))
        step = int(tf.size * 1.12)
        items = [(text_sprite(ln, tf, pal.rgb("panel_text")), 0.0, float(i * step)) for i, ln in enumerate(tl)]
        y = items[-1][2] + items[-1][0].h
        if payload.subtitle:
            sf, sl = fit_text(payload.subtitle, "Regular", self.px(30), self.W * 0.4, 2, self.px(18))
            items.append((text_block(sl, sf, pal.rgb("muted"), "left", 1.2), 0.0, y + self.px(14)))
        self.setup(items, (self.px(36), self.px(28)), payload.position, 30)


# ---------------------------------------------------------------- text overlay

def _is_stamp(text: str) -> bool:
    words = text.split()
    return 0 < len(words) <= 5 and any(ch.isdigit() for ch in text)


class TextOverlay(CompBase):
    """Kinetic text (words rise in one by one) or a date stamp (letters type on behind an accent rule)."""

    own_exit = True

    def __init__(self, spec: MotionSpec, payload: TextPayload, pal, alpha: bool) -> None:  # noqa: ANN001
        super().__init__(spec, payload, pal, alpha)
        self.pos = payload.position
        self.stamp = _is_stamp(payload.title) and self.transparent
        self.col = WHITE if self.transparent else pal.rgb("text")
        self.shadow = 0.65 if self.transparent else 0.0
        self.align = "left" if self.pos.endswith("left") else "right" if self.pos.endswith("right") else "center"
        if self.stamp:
            self._layout_stamp(payload)
        else:
            self._layout_words(payload)

    def _layout_words(self, p: TextPayload) -> None:
        big = not self.transparent or self.pos == "center"
        max_w = self.W * (0.78 if big else 0.5)
        f, lines = fit_text(p.title, "Bold", self.px(84 if big else 66), max_w, 3, self.px(30))
        step = int(f.size * 1.14)
        asc, desc = f.getmetrics()
        bw = max(text_width(ln, f) for ln in lines)
        sub = None
        if p.subtitle:
            sf, sl = fit_text(p.subtitle.upper(), "Medium", self.px(28), max_w, 2, self.px(16), 0.2)
            sub = text_block(sl, sf, self.pal.rgb("accent") if not self.transparent else (235, 235, 235),
                             self.align, 1.25, 0.2 * sf.size, self.shadow)
        bh = step * (len(lines) - 1) + asc + desc + (self.px(28) + sub.h if sub else 0)
        x0, y0 = self.anchor(max(bw, sub.w if sub else 0), bh, self.pos)
        self.words: list[tuple[Sprite, float, float]] = []
        space = f.getlength(" ")
        for li, ln in enumerate(lines):
            lw = text_width(ln, f)
            lx = x0 + {"left": 0, "center": (bw - lw) / 2, "right": bw - lw}[self.align]
            for w in ln.split():
                self.words.append((text_sprite(w, f, self.col, shadow=self.shadow), lx, y0 + li * step))
                lx += f.getlength(w) + space
        self.sub = sub
        self.sub_xy = (x0 + {"left": 0, "center": (bw - (sub.w if sub else 0)) / 2,
                             "right": bw - (sub.w if sub else 0)}[self.align],
                       y0 + step * (len(lines) - 1) + asc + desc + self.px(28))

    def _layout_stamp(self, p: TextPayload) -> None:
        f, lines = fit_text(p.title.upper(), "SemiBold", self.px(46), self.W * 0.45, 1, self.px(24), 0.18)
        self.text, self.f, self.tr = lines[0], f, 0.18 * f.size
        self.full = text_sprite(self.text, f, self.col, self.tr, self.shadow)
        self.sub = None
        if p.subtitle:
            sf, sl = fit_text(p.subtitle, "Regular", self.px(28), self.W * 0.45, 1, self.px(16))
            self.sub = text_sprite(sl[0], sf, (230, 230, 230), shadow=self.shadow)
        h = self.full.h + self.px(18) + (self.sub.h + self.px(10) if self.sub else 0)
        self.x0, self.y0 = self.anchor(max(self.full.w, self.sub.w if self.sub else 0), h, self.pos)

    def draw(self, frame: np.ndarray, t: float) -> None:
        if self.stamp:
            self._draw_stamp(frame, t)
        else:
            n = len(self.words)
            stg = min(0.09, 0.9 / max(1, n))
            for i, (spr, x, y) in enumerate(self.words):
                o = ez.ease_out_cubic(ez.progress(t, 0.1 + i * stg, 0.55 + i * stg))
                blit(frame, spr, x, y + self.px(28) * (1 - o), o)
            if self.sub is not None:
                o = ez.ease_out_cubic(ez.progress(t, 0.35 + n * stg, 0.9 + n * stg))
                blit(frame, self.sub, self.sub_xy[0], self.sub_xy[1] + self.px(12) * (1 - o), o)
        out = ez.ease_in_cubic(ez.progress(t, self.dur - min(0.45, self.dur * 0.2), self.dur))
        if out > 0:
            frame[...] = fade(frame, 1 - out)

    def _draw_stamp(self, frame: np.ndarray, t: float) -> None:
        rule = ez.ease_out_cubic(ez.progress(t, 0.0, 0.5))
        rw = (self.full.w + self.px(10)) * rule
        accent = self.pal.rgb("accent")
        y_rule = self.y0 + self.full.h + self.px(6)
        rx = self.x0 if self.align != "right" else self.x0 + self.full.w - rw
        if rw > 1:
            blit(frame, rect(rw, max(2, self.px(4)), accent), rx, y_rule)
        n = len(self.text)
        shown = int(round(n * ez.progress(t, 0.25, 0.25 + min(1.2, 0.06 * n))))
        if shown > 0:
            part = self.text[:shown]
            spr = self.full if shown == n else text_sprite(part, self.f, self.col, self.tr, self.shadow)
            blit(frame, spr, self.x0, self.y0)
        if self.sub is not None:
            o = ez.ease_out_cubic(ez.progress(t, 0.9, 1.4))
            blit(frame, self.sub, self.x0, y_rule + self.px(16) + self.px(10) * (1 - o), o)


# ---------------------------------------------------------------- highlight box and arrow

class HighlightBox(CompBase):
    """Dims the frame outside a region, traces an accent stroke around it and tags it."""

    own_exit = True

    def __init__(self, spec: MotionSpec, payload, pal, alpha: bool) -> None:  # noqa: ANN001
        super().__init__(spec, payload, pal, alpha)
        p = payload
        x0, y0 = max(0.0, p.x) * self.W, max(0.0, p.y) * self.H
        x1, y1 = min(1.0, p.x + max(0.02, p.w)) * self.W, min(1.0, p.y + max(0.02, p.h)) * self.H
        self.box = (x0, y0, x1, y1)
        self.perim = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], np.float64)
        self.accent = pal.rgb("accent")
        self.glow = GlowLayer(self.W, self.H)
        self.tag = None
        if p.label:
            f, ln = fit_text(p.label.upper(), "SemiBold", self.px(26), self.W * 0.4, 1, self.px(16), 0.1)
            self.tag = pill(ln[0], f, on_colour(self.accent), self.accent, self.px(16), self.px(9), 0.1 * f.size)
            above = y0 - self.tag.h - self.px(10)
            self.tag_xy = (x0, above if above >= self.safe_top else y0 + self.px(12))

    def draw(self, frame: np.ndarray, t: float) -> None:
        x0, y0, x1, y1 = self.box
        dim = 0.38 * ez.ease_out_cubic(ez.progress(t, 0.0, 0.6))
        if self.transparent and dim > 0 and frame.shape[2] == 4:
            frame[..., 3] = int(255 * dim)
            frame[int(y0):int(y1), int(x0):int(x1), 3] = 0
        trace = ez.ease_in_out_cubic(ez.progress(t, 0.15, 1.05))
        pts = path_prefix(self.perim, trace)
        th = max(2, self.px(5))
        self.glow.clear()
        if len(pts) > 1:
            polyline(self.glow.buf, pts / self.glow.f, self.accent, max(1, th // 2 + 1))
            self.glow.dirty = True
            self.glow.composite(frame, sigma=3.0, gain=0.6)
            polyline(frame, pts, self.accent, th)
        if self.tag is not None:
            o = ez.ease_out_cubic(ez.progress(t, 0.8, 1.3))
            blit(frame, self.tag, self.tag_xy[0] + self.px(20) * (1 - o), self.tag_xy[1], o)
        out = ez.ease_in_cubic(ez.progress(t, self.dur - min(0.45, self.dur * 0.2), self.dur))
        if out > 0:
            frame[...] = fade(frame, 1 - out)


class Arrow(CompBase):
    """Curved arrow drawn from (x, y) to (to_x, to_y) with a soft shadow, head pop and tail label."""

    own_exit = True

    def __init__(self, spec: MotionSpec, payload, pal, alpha: bool) -> None:  # noqa: ANN001
        super().__init__(spec, payload, pal, alpha)
        p = payload
        sx, sy = p.x * self.W, p.y * self.H
        tx = p.to_x if p.to_x is not None else (p.x + p.w if p.w > 0 else p.x + 0.18)
        ty = p.to_y if p.to_y is not None else (p.y + p.h if p.h > 0 else p.y + 0.12)
        ex, ey = min(1.0, max(0.0, tx)) * self.W, min(1.0, max(0.0, ty)) * self.H
        self.pts = _curve((sx, sy), (ex, ey), 0.16)
        self.accent = pal.rgb("accent")
        self.glow = GlowLayer(self.W, self.H)
        self.label = None
        if p.label:
            f, ln = fit_text(p.label, "SemiBold", self.px(30), self.W * 0.35, 1, self.px(18))
            self.label = pill(ln[0], f, pal.rgb("panel_text"), pal.rgb("panel"), self.px(18), self.px(10),
                              accent=self.accent)
            lx = sx - self.label.w - self.px(24) if ex >= sx else sx + self.px(24)
            lx = min(max(self.px(20), lx), self.W - self.label.w - self.px(20))
            ly = min(max(self.safe_top, sy - self.label.h / 2), self.safe_bottom - self.label.h)
            self.label_xy = (lx, ly)

    def draw(self, frame: np.ndarray, t: float) -> None:
        th = max(2, self.px(7))
        p = ez.ease_in_out_cubic(ez.progress(t, 0.2, 1.1))
        pts = path_prefix(self.pts, p)
        self.glow.clear()
        if len(pts) > 1:
            polyline(self.glow.buf, pts / self.glow.f, (0, 0, 0), max(2, th // 2 + 2))
            self.glow.dirty = True
            self.glow.composite(frame, sigma=2.5, gain=0.55)
            polyline(frame, pts, self.accent, th)
        disc(frame, *self.pts[0], self.px(9) * ez.ease_out_back(ez.progress(t, 0.05, 0.4)), self.accent)
        head = ez.ease_out_back(ez.progress(t, 0.95, 1.3), 2.0)
        if head > 0.01:
            fill_poly(frame, [arrow_head(self.pts, self.px(34) * head)], self.accent)
        if self.label is not None:
            o = ez.ease_out_cubic(ez.progress(t, 0.0, 0.5))
            blit(frame, self.label, self.label_xy[0], self.label_xy[1] + self.px(12) * (1 - o), o)
        out = ez.ease_in_cubic(ez.progress(t, self.dur - min(0.45, self.dur * 0.2), self.dur))
        if out > 0:
            frame[...] = fade(frame, 1 - out)


def _curve(a: tuple[float, float], b: tuple[float, float], bend: float, n: int = 64) -> np.ndarray:
    """Quadratic bezier from a to b bowing upward by `bend` of its length."""
    ax, ay = a
    bx, by = b
    mx, my = (ax + bx) / 2, (ay + by) / 2
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    if ny > 0:
        nx, ny = -nx, -ny
    cx, cy = mx + nx * length * bend, my + ny * length * bend
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 2 * np.array([ax, ay]) + 2 * (1 - u) * u * np.array([cx, cy]) + u ** 2 * np.array([bx, by])


# ---------------------------------------------------------------- recap grid

ITEM_NUM_RE = re.compile(r"^\s*#?(\d{1,3})[\.\):\s]+(.*)$")


def recap_numbers(items: list[str], countdown: bool) -> list[tuple[int, str]]:
    """(number, label) per item: explicit leading numbers win, else countdown n..1 or 1..n."""
    out = []
    n = len(items)
    for i, it in enumerate(items):
        m = ITEM_NUM_RE.match(it)
        if m and m.group(2).strip():
            out.append((int(m.group(1)), m.group(2).strip()))
        else:
            out.append((n - i if countdown else i + 1, it.strip()))
    return out


class RecapGrid(CompBase):
    """Title with accent rule and a grid of numbered cards that pop in with a stagger."""

    def __init__(self, spec: MotionSpec, payload, pal, alpha: bool) -> None:  # noqa: ANN001
        super().__init__(spec, payload, pal, alpha)
        countdown = pal.name == "top10_bold" or "top" in payload.title.lower()
        items = recap_numbers(payload.items or ["Recap"], countdown)
        n = len(items)
        cols = n if n <= 4 else min(5, math.ceil(n / 2)) if n <= 10 else 5
        rows = math.ceil(n / cols)
        tf, tl = fit_text(payload.title.upper(), "Bold", self.px(54), self.W * 0.8, 1, self.px(28), 0.08)
        self.title = text_sprite(tl[0], tf, WHITE if self.transparent else pal.rgb("text"), 0.08 * tf.size,
                                 0.6 if self.transparent else 0.0)
        top = self.safe_top + self.px(30) + self.title.h + self.px(46)
        gap = self.px(22)
        area_w = self.W - 2 * self.margin
        area_h = self.safe_bottom - self.px(20) - top
        cw = (area_w - gap * (cols - 1)) / cols
        ch = min(self.px(300), (area_h - gap * (rows - 1)) / rows)
        y_start = top + (area_h - (ch * rows + gap * (rows - 1))) / 2
        best = min(nu for nu, _ in items)
        self.cards = []
        for i, (num, label) in enumerate(items):
            r, c = divmod(i, cols)
            in_row = min(cols, n - r * cols)
            x = self.margin + (area_w - (in_row * cw + (in_row - 1) * gap)) / 2 + c * (cw + gap)
            self.cards.append((self._card(int(cw), int(ch), num, label, num == best), x, y_start + r * (ch + gap)))

    def _card(self, w: int, h: int, num: int, label: str, top: bool) -> Sprite:
        pal = self.pal
        base = mix(pal.rgb("bg"), (255, 255, 255), 0.07 if pal.dark else 0.4)
        card = np.zeros((h, w, 4), np.uint8)
        blit(card, round_rect(w, h, self.px(10), pal.rgb("accent") if top else mix(base, pal.rgb("text"), 0.12)), 0, 0)
        b = max(2, self.px(3)) if top else max(1, self.px(1))
        blit(card, round_rect(w - 2 * b, h - 2 * b, self.px(8), pal.rgb("panel") if pal.dark else base, 0.96), b, b)
        nf = font("Black", int(min(h * 0.36, self.px(76))))
        ns = text_sprite(f"#{num}" if pal.name == "top10_bold" else f"{num:02d}", nf, pal.rgb("accent"),
                         features=TNUM)
        pad = self.px(22)
        blit(card, ns, pad, pad * 0.7)
        lf, ll = fit_text(label, "SemiBold", self.px(30), w - 2 * pad, 3, self.px(16))
        lbl = text_block(ll, lf, pal.rgb("panel_text") if pal.dark else pal.rgb("text"), "left", 1.15)
        blit(card, lbl, pad, min(h - lbl.h - pad * 0.8, pad * 0.7 + ns.h + self.px(6)))
        return Sprite(with_shadow(np.pad(card, ((12, 12), (12, 12), (0, 0))), radius=self.px(14), offset=self.px(5),
                                  opacity=0.35 if not pal.dark else 0.6), 12)

    def draw(self, frame: np.ndarray, t: float) -> None:
        o = ez.ease_out_cubic(ez.progress(t, 0.05, 0.7))
        tx, ty = self.margin, self.safe_top + self.px(30)
        blit(frame, self.title, tx, ty + self.px(14) * (1 - o), o)
        rw = self.px(110) * ez.ease_in_out_cubic(ez.progress(t, 0.3, 0.9))
        if rw > 1:
            blit(frame, rect(rw, self.px(5), self.pal.rgb("accent")), tx, ty + self.title.h + self.px(14))
        n = len(self.cards)
        span = min(self.dur * 0.4, 0.16 * n)
        for i, (spr, x, y) in enumerate(self.cards):
            a, b = ez.stagger(i, n, 0.45, span, 0.5)
            p = ez.ease_out_back(ez.progress(t, a, b), 1.4)
            if p <= 0.001:
                continue
            s = 0.88 + 0.12 * p
            sp = scaled(spr, s)
            blit(frame, sp, x + spr.w * (1 - s) / 2, y + spr.h * (1 - s) / 2 + self.px(20) * (1 - p),
                 ez.clamp01(p * 1.6))
