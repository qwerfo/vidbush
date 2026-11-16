"""Layouts built from real pictures and short text.

number_over_footage is the reference implementation of the pattern every kind follows: setup() builds
sprites once, draw() places them from the cues (nothing shows before its cue), and pictures come from
job.pictures (real candidates only; a generated picture is refused).
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Optional

import cv2
import numpy as np

from ...models import LayoutKind
from .. import easing as ez
from ..house import MIN_BODY_PX, MIN_HEAD_PX, MIN_HERO_PX, MIN_LABEL_PX
from ..textdraw import ELLIPSIS, Sprite, blit, fit_text, font, rect, round_rect, scaled, text_sprite, with_shadow
from . import register
from .base import Footage, LayoutComp, LayoutJob, darken, framed, ink_ground

NUMBER_RE = re.compile(r"^\s*(?P<pre>[$£€]?)(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<suf>%|[a-zA-Z]{0,2}\b)?\s*(?P<unit>.*)$")
ROLL_S = 0.35  # the digits roll only in the last moments before the spoken number, blurred: never readable


def split_number(text: str) -> tuple[str, str]:
    """'29 people' -> ('29', 'people'); '$77 million' -> ('$77', 'million'); '82%' -> ('82%', '')."""
    m = NUMBER_RE.match(text or "")
    if not m:
        return text.strip(), ""
    head = f"{m.group('pre')}{m.group('num')}{m.group('suf') or ''}"
    return head, m.group("unit").strip()


class PictureGround(LayoutComp):
    """A layout over a darkened, moving real picture (the 'background' element) instead of the ink ground."""

    keep = None  # brightness kept on the picture; None uses the house shade
    blur = 2.0

    def make_ground(self):  # noqa: ANN201
        return None

    def setup_ground(self) -> None:
        bg = self.job.first("background")
        self.foot: Optional[Footage] = None
        if bg is not None and bg[0] in self.job.pictures and not self.job.pictures[bg[0]].generated:
            self.foot = Footage(self.job.pictures[bg[0]], self.W, self.H, self.fps)
        elif bg is not None:
            self.job.notes.append("background picture missing or generated: house ground used")

    def canvas(self, i: int) -> np.ndarray:
        if self.foot is None:
            from .base import ink_ground
            return ink_ground(self.W, self.H, self.house).copy()
        return darken(self.foot.frame(i), self.keep if self.keep is not None else self.house.shade, self.blur * self.k)

    def close(self) -> None:
        if getattr(self, "foot", None) is not None:
            self.foot.close()


@register(LayoutKind.number_over_footage)
class NumberOverFootage(PictureGround):
    """A large number with its unit over darkened moving footage, landing on the spoken number."""

    def setup(self) -> None:
        self.setup_ground()
        num = self.job.first("number")
        self.num_cue = num[2] if num else None
        head, unit = split_number(num[1].text if num else "")
        self.head = head
        hero = self.fs(320, MIN_HERO_PX)
        f = font("Black", hero)
        while f.getlength(head) > self.bw * 0.9 and f.size > self.fs(MIN_HERO_PX, MIN_HERO_PX):
            f = font("Black", int(f.size * 0.92))
        self.nf = f
        self.num = text_sprite(head, f, self.house.rgb("paper"), -0.02 * f.size, shadow=0.5)
        self.unit = self.label(unit, 72, "accent", max_w=self.bw * 0.8, floor=MIN_BODY_PX, tracking_em=0.22,
                               shadow=0.5) if unit else None
        lab = self.job.first("label")
        self.lab = (self.label(lab[1].text, 44, "paper", "Medium", caps=False, max_w=self.bw * 0.7, lines=2,
                               floor=MIN_BODY_PX, shadow=0.5, align="center"), lab[2]) if lab and lab[1].text else None
        self.digits = re.sub(r"[^\d]", "", head)

    def _rolling(self, t: float) -> Optional[Sprite]:
        """During the roll, a vertically blurred, faded copy with changing digits (unreadable by design)."""
        c = self.num_cue
        if c is None or not self.digits or not (c.word - ROLL_S <= t < c.word):
            return None
        rng = np.random.default_rng(int(t * 1000))
        fake = "".join(str(rng.integers(0, 10)) if ch.isdigit() else ch for ch in self.head)
        spr = text_sprite(fake, self.nf, self.house.rgb("paper"), -0.02 * self.nf.size)
        k = max(3, int(self.nf.size * 0.9)) | 1  # smeared well past legibility
        arr = cv2.GaussianBlur(spr.arr, (1, k), 0)
        return Sprite(arr, spr.pad)

    def draw(self, frame: np.ndarray, t: float) -> None:
        cx = self.W / 2
        total = self.num.h + (self.unit.h + self.px(18) if self.unit else 0)
        y = self.box[1] + (self.bh - total) / 2 - (self.lab[0].h / 2 if self.lab else 0)
        c = self.num_cue
        if c is not None and t >= c.word - ROLL_S:
            rolling = self._rolling(t)
            if rolling is not None:
                blit(frame, rolling, cx - rolling.w / 2, y - self.px(30) * (1 - ez.progress(t, c.word - ROLL_S, c.word)),
                     0.4)
            else:  # landed: the true value only, settling in
                p = ez.ease_out_back(ez.progress(t, c.word, c.word + 0.35), 1.3)
                s = 0.92 + 0.08 * p
                from ..textdraw import scaled
                sp = scaled(self.num, s)
                blit(frame, sp, cx - sp.w / 2, y + self.num.h * (1 - s) / 2, ez.clamp01(p * 1.5))
        if self.unit is not None and c is not None:
            o = self.settle(t, c.__class__(c.index, c.word + 0.1, c.word, c.word_end, True))
            uy = y + self.num.h + self.px(18)
            rw = (self.unit.w + self.px(40)) * o
            if rw > 1:
                blit(frame, rect(rw, max(3, self.px(5)), self.house.rgb("accent")), cx - rw / 2, uy - self.px(6))
            blit(frame, self.unit, cx - self.unit.w / 2, uy + self.px(14) + self.px(16) * (1 - o), o)
        if self.lab is not None:
            spr, lc = self.lab
            o = self.settle(t, lc)
            blit(frame, spr, cx - spr.w / 2, self.box[3] - spr.h - self.px(10) + self.px(14) * (1 - o), o)


# ---------------------------------------------------------------- shared helpers for the picture kinds

SLIDE_S = 0.6  # a picture sliding or stepping to its new place


def still_of(job: LayoutJob, i: int, w: float, h: float) -> Optional[np.ndarray]:
    """One still of element i's real picture, covering w x h (a clip gives a frame just after its in point);
    None when the element has no usable picture."""
    ref = job.pictures.get(i)
    if ref is None or ref.generated:
        return None
    if ref.media == "video":
        ref = replace(ref, in_point=ref.in_point + 0.5)  # past any fade in at the cut point
    foot = Footage(ref, max(8, int(w)), max(8, int(h)), job.fps)
    try:
        return foot.frame(0).copy()
    finally:
        foot.close()


def wipe(spr: Sprite, frac: float) -> Optional[Sprite]:
    """The sprite revealed left to right up to frac of its content width (None when nothing shows)."""
    if frac <= 0.0:
        return None
    if frac >= 1.0:
        return spr
    return Sprite(spr.arr[:, :spr.pad + max(1, int(spr.w * frac))], spr.pad)


def tinted(img: np.ndarray, color: tuple[int, int, int], amount: float) -> np.ndarray:
    return (img.astype(np.float32) * (1 - amount) + np.array(color, np.float32) * amount).clip(0, 255).astype(np.uint8)


class PictureComp(LayoutComp):
    """A layout of framed real pictures: frames are built once per element; a missing picture becomes a
    quiet text card (and a note) so the layout still reads."""

    def framed_or_card(self, i: int, w: float, h: float, text: str,
                       focus: tuple[float, float] = (0.5, 0.45)) -> Sprite:
        img = still_of(self.job, i, w, h)
        if img is not None:
            return framed(img, w, h, self.house, self.k, focus)
        self.job.notes.append(f"element {i}: no real picture, shown as a text card")
        card = np.zeros((int(h), int(w), 4), np.uint8)
        blit(card, round_rect(int(w), int(h), self.px(self.house.radius_px), self.house.rgb("ink2")), 0, 0)
        if text:
            t = self.label(text, 52, "muted", max_w=w * 0.84, lines=2, floor=MIN_BODY_PX, align="center")
            blit(card, t, (w - t.w) / 2, (h - t.h) / 2)
        return Sprite(card, 0)

    def balanced(self, text: str, base: float, color: str, max_w: float, floor: float, tracking_em: float,
                 weight: str = "SemiBold", caps: bool = True) -> Sprite:
        """A label of at most two lines; when it must wrap, the narrowest width that keeps the same size and
        two lines, so the lines are balanced (no orphan word on line two)."""
        t = text.upper() if caps else text
        size, fl, tr = self.fs(base, floor), self.fs(floor, floor), tracking_em if caps else 0.0
        f, ln = fit_text(t, weight, size, max_w, 2, fl, tr)
        if len(ln) == 2 and ELLIPSIS not in ln[-1]:
            lo, hi = max_w / 2, max_w
            for _ in range(12):
                mid = (lo + hi) / 2
                f2, ln2 = fit_text(t, weight, size, mid, 2, fl, tr)
                if f2.size == f.size and len(ln2) == 2 and ELLIPSIS not in ln2[-1]:
                    hi = mid
                else:
                    lo = mid
            max_w = hi + 1
        return self.label(text, base, color, weight, caps, max_w, 2, floor, tracking_em)


# ---------------------------------------------------------------- split comparison

@register(LayoutKind.split_comparison)
class SplitComparison(PictureComp):
    """Two framed real pictures side by side, each with a caption; each half fills only on its own word and
    an amber rule sits under the side being spoken."""

    def make_ground(self):  # noqa: ANN201
        g = ink_ground(self.W, self.H, self.house).copy()
        mid = self.W // 2
        g[:, :mid] = tinted(g[:, :mid], self.house.rgb("ink"), 0.35)
        light = tuple(min(255, int(c * 1.18) + 4) for c in self.house.rgb("ink2"))
        g[:, mid:] = tinted(g[:, mid:], light, 0.4)
        return g

    def setup(self) -> None:
        self.sides = []
        half = self.bw / 2
        pw = min(0.40 * self.W, half * 0.96)  # each picture about 40% of the frame width
        caps = []
        for role in ("left", "right"):
            got = self.job.first(role)
            if got is None:
                continue
            i, el, cue = got
            cap = self.label(el.text, 52, max_w=pw, lines=2, floor=MIN_BODY_PX, align="center") if el.text else None
            caps.append((i, el, cue, cap))
        ch = max((c[3].h for c in caps if c[3] is not None), default=0)
        self.gap = self.px(26)
        self.rule_h = max(3, self.px(5))
        ph = min(pw / 1.5, self.bh - ch - 2 * self.gap - self.rule_h)
        pw = ph * 1.5
        top = self.box[1] + (self.bh - ph - ch - 2 * self.gap - self.rule_h) / 2
        for i, el, cue, cap in caps:
            side = 0 if el.role == "left" else 1
            cx = self.box[0] + half * (side + 0.5)
            pic = self.framed_or_card(i, pw, ph, "" if cap else el.text)
            self.sides.append(dict(side=side, cue=cue, pic=pic, cap=cap, cx=cx, top=top, ph=ph))

    def draw(self, frame: np.ndarray, t: float) -> None:
        for s in self.sides:
            c = s["cue"]
            if t < c.appear:
                continue
            p = self.settle(t, c, 0.55)
            out = 1 if s["side"] == 1 else -1
            dx = out * self.px(14) * ez.ease_in_out_sine(ez.progress(t, c.appear, self.dur))  # slow outward drift
            pic = s["pic"]
            blit(frame, pic, s["cx"] - pic.w / 2 + dx, s["top"] + self.px(36) * (1 - p), p)
            cap = s["cap"]
            y = s["top"] + s["ph"] + self.gap
            if cap is not None:
                w = wipe(cap, ez.ease_out_cubic(ez.progress(t, c.appear + 0.12, c.appear + 0.62)))
                if w is not None:
                    blit(frame, w, s["cx"] - cap.w / 2 + dx, y + self.rule_h + self.gap)
            # the amber rule grows under the side being spoken and gives way when the other side is spoken
            later = [o["cue"].appear for o in self.sides if o["cue"].appear > c.appear and t >= o["cue"].appear]
            g = ez.ease_out_cubic(ez.progress(t, max(c.appear, c.word - 0.1), c.word + 0.35))
            if later:
                g *= 1 - ez.ease_in_out_cubic(ez.progress(t, later[0], later[0] + 0.35))
            rw = (cap.w if cap is not None else pic.w * 0.5) * g
            if rw > 2:
                blit(frame, rect(rw, self.rule_h, self.house.rgb("accent")), s["cx"] - rw / 2 + dx, y)


# ---------------------------------------------------------------- name card and its lower third fallback

@register(LayoutKind.name_card)
class NameCard(PictureComp):
    """A real portrait with the person's name and role beside it. Without a real portrait the pipeline shows
    NameLowerThird over the shot's own picture instead (fallback_needed); rendered anyway, the card shows the
    name and role alone and notes it."""

    @staticmethod
    def fallback_needed(job: LayoutJob) -> bool:
        got = job.first("portrait")
        if got is None:
            return True
        ref = job.pictures.get(got[0])
        return ref is None or ref.generated

    def setup(self) -> None:
        name, role, por = self.job.first("name"), self.job.first("role"), self.job.first("portrait")
        no_pic = self.fallback_needed(self.job)
        if no_pic:
            self.job.notes.append("name_card: no real portrait, use NameLowerThird over the shot's picture")
        ph = self.bh * 0.78  # the portrait fills most of the box height
        pw = ph * 0.8
        self.gap = self.px(64)
        tw = self.bw * 0.86 if no_pic else min(0.45 * self.W, self.bw - pw - self.gap)
        if not no_pic and tw < 0.3 * self.W:  # a narrow frame: the portrait gives way to the text column
            tw = 0.3 * self.W
            pw = self.bw - tw - self.gap
            ph = pw / 0.8
        self.name = (self.balanced(name[1].text, 120, "paper", tw, MIN_HEAD_PX, 0.0, "Bold", caps=False),
                     name[2]) if name and name[1].text else None
        self.role = (self.balanced(role[1].text, 40, "muted", tw, MIN_BODY_PX, 0.14), role[2]) \
            if role and role[1].text else None
        if self.role is not None and self.name is not None:  # the role never comes before the name
            c = self.role[1]
            self.role = (self.role[0], replace(c, appear=max(c.appear, self.name[1].appear + 0.3)))
        self.pic = None
        if not no_pic:
            img = still_of(self.job, por[0], pw, ph)
            self.pic = (framed(img, pw, ph, self.house, self.k, (0.5, 0.35)), por[2])
        textw = max([s[0].w for s in (self.name, self.role) if s is not None] or [0])
        self.rule_h = max(3, self.px(6))
        self.th = (self.name[0].h + self.px(52) if self.name else 0) + (self.role[0].h if self.role else 0)
        total = (pw + self.gap if self.pic else 0) + textw
        self.x0 = self.box[0] + (self.bw - total) / 2
        self.tx = self.x0 + (pw + self.gap if self.pic else 0)
        self.pw, self.ph = pw, ph
        self.cy = self.box[1] + self.bh / 2

    def draw(self, frame: np.ndarray, t: float) -> None:
        if self.pic is not None and t >= self.pic[1].appear:
            spr, c = self.pic
            p = self.settle(t, c, 0.6)
            dy = -self.px(12) * ez.ease_in_out_sine(ez.progress(t, c.appear, self.dur))
            blit(frame, spr, self.x0 - self.px(40) * (1 - p), self.cy - self.ph / 2 + dy, p)
        y = self.cy - self.th / 2
        if self.name is not None:
            spr, c = self.name
            if t >= c.appear:
                w = wipe(spr, ez.ease_out_cubic(ez.progress(t, c.appear, c.appear + 0.55)))
                drift_x = self.px(10) * ez.progress(t, c.appear, self.dur)
                if w is not None:
                    blit(frame, w, self.tx + drift_x, y)
                rw = min(spr.w, self.px(220)) * ez.ease_out_cubic(ez.progress(t, c.word, c.word + 0.5))
                if rw > 2:
                    blit(frame, rect(rw, self.rule_h, self.house.rgb("accent")), self.tx + drift_x,
                         y + spr.h + self.px(20))
            y += spr.h + self.px(52)
        if self.role is not None:
            spr, c = self.role
            if t >= c.appear:
                p = self.settle(t, c)
                x = self.tx + self.px(10) * ez.progress(t, c.appear, self.dur)
                blit(frame, spr, x, y + self.px(14) * (1 - p), p)


class NameLowerThird(LayoutComp):
    """The name card's fallback: a lower third (name, role) over the shot's own picture, lower left inside the
    visible area and above the caption band. render_rgba(i) is a premultiplied full-frame overlay. Not a
    registered kind: the caller picks it when NameCard.fallback_needed(job)."""

    def make_ground(self):  # noqa: ANN201
        return None

    def setup(self) -> None:
        name, role = self.job.first("name"), self.job.first("role")
        mw = self.bw * 0.6
        self.name = (self.label(name[1].text, 64, "paper", "Bold", caps=False, max_w=mw, floor=MIN_HEAD_PX),
                     name[2]) if name and name[1].text else None
        self.role = (self.label(role[1].text, 36, "muted", max_w=mw, floor=MIN_LABEL_PX, tracking_em=0.14),
                     role[2]) if role and role[1].text else None
        if self.role is not None and self.name is not None:
            c = self.role[1]
            self.role = (self.role[0], replace(c, appear=max(c.appear, self.name[1].appear + 0.25)))
        self.padx, self.pady, self.bar = self.px(34), self.px(22), max(4, self.px(8))
        self.lgap = self.px(8)
        tw = max([s[0].w for s in (self.name, self.role) if s is not None] or [0])
        th = sum(s[0].h for s in (self.name, self.role) if s is not None) + (self.lgap if self.role else 0)
        self.pw, self.ph = int(tw + 2 * self.padx + self.bar), int(th + 2 * self.pady)
        sp = self.px(26) + 2
        body = np.zeros((self.ph + 2 * sp, self.pw + 2 * sp, 4), np.uint8)
        blit(body, round_rect(self.pw, self.ph, self.px(self.house.radius_px), self.house.rgb("ink"), 0.9), sp, sp)
        blit(body, rect(self.bar, self.ph, self.house.rgb("accent")), sp, sp)
        self.panel = Sprite(with_shadow(body, self.px(16) + 1, max(1, self.px(6)), 0.45), sp)
        self.x, self.y = self.box[0], self.box[3] - self.ph
        self.start = min([s[1].appear for s in (self.name, self.role) if s is not None] or [0.0])

    def draw(self, frame: np.ndarray, t: float) -> None:
        if t < self.start:
            return
        out = 1.0 - ez.ease_in_cubic(ez.progress(t, self.dur - 0.35, self.dur))
        w = wipe(self.panel, ez.ease_out_cubic(ez.progress(t, self.start, self.start + 0.45)))
        if w is not None:
            blit(frame, w, self.x, self.y, out)
        y = self.y + self.pady
        for s in (self.name, self.role):
            if s is None:
                continue
            spr, c = s
            if t >= c.appear:
                p = self.settle(t, c, 0.4)
                blit(frame, spr, self.x + self.bar + self.padx + self.px(18) * (1 - p), y, p * out)
            y += spr.h + self.lgap

    def render_rgba(self, i: int) -> np.ndarray:
        frame = np.zeros((self.H, self.W, 4), np.uint8)
        self.draw(frame, i / self.fps)
        return frame


# ---------------------------------------------------------------- list slideshow

@register(LayoutKind.list_slideshow)
class ListSlideshow(PictureComp):
    """One framed real picture per spoken item. Each arrives on its word, sliding in from the right as the
    hero; the earlier ones step back to a column on the left, newest on top (at most KEEP kept)."""

    KEEP = 3

    def make_ground(self):  # noqa: ANN201
        g = ink_ground(self.W, self.H, self.house).copy()
        step, d, o = max(6, self.px(30)), max(1, self.px(3)), max(3, self.px(30)) // 2
        paper = np.array(self.house.rgb("paper"), np.float32)
        for oy in range(d):  # a faint dot grid: our own ground, not graph paper
            for ox in range(d):
                v = g[o + oy::step, o + ox::step].astype(np.float32)
                g[o + oy::step, o + ox::step] = (v * 0.9 + paper * 0.1).astype(np.uint8)
        return g

    def setup(self) -> None:
        items = self.job.els("item")
        n = len(items)
        m = min(self.KEEP, max(0, n - 1))
        self.m = m
        col = self.bw * 0.27 if m else 0.0
        gap = self.px(56) if m else 0.0
        caps = [self.label(el.text, 52, max_w=self.bw - col - gap, floor=MIN_BODY_PX) if el.text else None
                for _, el, _ in items]
        ch = max((c.h for c in caps if c is not None), default=0)
        self.cgap = self.px(22)
        avail = self.bh - ch - self.cgap
        hw = min(0.53 * self.W, self.bw - col - gap, avail * 1.78)  # the hero about half the frame width
        hh = min(hw / 1.5, avail)
        a = hw / hh
        x0 = self.box[0] + (self.bw - col - gap - hw) / 2  # column and hero centred as one group
        self.hero = (x0 + col + gap, self.box[1] + (self.bh - hh - ch - self.cgap) / 2)
        self.hw, self.hh = hw, hh
        smalls = [self.label(el.text, MIN_LABEL_PX, max_w=col, floor=MIN_LABEL_PX) if el.text and m else None
                  for _, el, _ in items]
        sch = max((c.h for c in smalls if c is not None), default=0)
        self.sgap = self.px(10)
        if m:
            vgap = self.px(20)
            slot = (self.bh - (m - 1) * vgap) / m
            th = min(slot - sch - self.sgap, col / a)
            self.s = th / hh
            self.slots = [(x0 + (col - th * a) / 2,
                           self.box[1] + r * (slot + vgap) + (slot - th - sch - self.sgap) / 2) for r in range(m)]
        else:
            self.s, self.slots = 1.0, []
        self.items = []
        for (i, el, cue), cap, small in zip(items, caps, smalls):
            pic = self.framed_or_card(i, hw, hh, "" if cap else el.text)
            self.items.append(dict(cue=cue, pic=pic, cap=cap, small=small))

    def _target(self, k: int, t: float) -> dict[int, tuple[float, ...]]:
        """Resting pose of every arrived item once k items have arrived:
        (x, y, scale, opacity, hero caption, small caption)."""
        poses = {}
        for j in range(k):
            if j == k - 1:
                c = self.items[j]["cue"]
                dx = -self.px(18) * ez.ease_in_out_sine(ez.progress(t, c.appear, self.dur))
                poses[j] = (self.hero[0] + dx, self.hero[1], 1.0, 1.0, 1.0, 0.0)
            else:
                r = k - 2 - j
                x, y = self.slots[min(r, self.m - 1)]
                gone = r >= self.m
                poses[j] = (x, y + (self.px(40) if gone else 0), self.s, 0.0 if gone else 1.0, 0.0, 1.0)
        return poses

    def _poses(self, k: int, t: float) -> dict[int, tuple[float, ...]]:
        """Poses at time t with k items arrived: the last arrival's move eases from wherever things stood
        when it began, so arrivals closer together than SLIDE_S never jump."""
        if k == 0:
            return {}
        start = self.items[k - 1]["cue"].appear
        before = self._poses(k - 1, start)
        before[k - 1] = (self.W + self.px(60), self.hero[1], 1.0, 1.0, 0.0, 0.0)
        u = ez.ease_in_out_cubic(ez.progress(t, start, start + SLIDE_S))
        after = self._target(k, t)
        return {j: tuple(ez.lerp(a, b, u) for a, b in zip(before[j], after[j])) for j in after}

    def draw(self, frame: np.ndarray, t: float) -> None:
        k = sum(1 for it in self.items if t >= it["cue"].appear)
        poses = self._poses(k, t)
        for j in sorted(poses):  # newest drawn last, on top
            x, y, s, op, hc, sc = poses[j]
            if op <= 0.003:
                continue
            it = self.items[j]
            if abs(s - self.s) < 0.002:  # at rest in the column: the small frame is made once
                if "pic_s" not in it:
                    it["pic_s"] = scaled(it["pic"], self.s)
                spr = it["pic_s"]
            else:
                spr = scaled(it["pic"], s)
            blit(frame, spr, x, y, op)
            cap = it["cap"]
            if cap is not None and hc > 0.5:
                blit(frame, cap, x + self.hw * s / 2 - cap.w / 2, y + self.hh * s + self.cgap, ((hc - 0.5) * 2) ** 2)
            small = it["small"]
            if small is not None and sc > 0.92:  # only once the picture has (nearly) settled in its slot
                blit(frame, small, x + self.hw * s / 2 - small.w / 2, y + self.hh * s + self.sgap,
                     op * (sc - 0.92) / 0.08)


# ---------------------------------------------------------------- date title

DATE_RE = re.compile(r"^(?P<head>.*?)[\s,]*(?P<year>\d{4}s?)\s*$")


@register(LayoutKind.date_title)
class DateTitle(PictureGround):
    """A date over darkened related footage: the month large in Inter Bold, an amber rule drawing out from the
    centre, the year in tracked SemiBold beneath, an optional label under it; all on the date's word."""

    def setup(self) -> None:
        self.setup_ground()
        d = self.job.first("date")
        self.cue_d = d[2] if d else None
        text = d[1].text.strip() if d else ""
        m = DATE_RE.match(text)
        head, year = (m.group("head").strip(), m.group("year")) if m and m.group("head").strip() else (text, "")
        self.head = self.label(head, 210, "paper", "Bold", caps=False, max_w=self.bw * 0.9, floor=MIN_HEAD_PX,
                               shadow=0.45) if head else None
        self.year = self.label(year, 76, "paper", max_w=self.bw * 0.9, floor=MIN_BODY_PX, tracking_em=0.45,
                               shadow=0.45) if year else None
        lab = self.job.first("label")
        self.lab = None
        if lab and lab[1].text:
            c = lab[2]
            if self.cue_d is not None:  # the label follows the date, never leads it
                c = replace(c, appear=max(c.appear, self.cue_d.appear + 0.5))
            self.lab = (self.label(lab[1].text, 38, "muted", max_w=self.bw * 0.8, floor=MIN_LABEL_PX,
                                   tracking_em=0.16, shadow=0.5), c)
        self.rule_h = max(3, self.px(5))
        self.rgap = self.px(30)
        parts = [s.h for s in (self.head, self.year) if s is not None]
        self.total = sum(parts) + (2 * self.rgap + self.rule_h if len(parts) == 2 else 0) + \
            (self.lab[0].h + self.px(40) if self.lab else 0)
        self.rule_w = max([s.w for s in (self.head, self.year) if s is not None] or [0]) * 0.55

    def draw(self, frame: np.ndarray, t: float) -> None:
        c = self.cue_d
        if c is None or t < c.appear:
            return
        cx = self.box[0] + self.bw / 2
        y = self.box[1] + (self.bh - self.total) / 2
        if self.head is not None:
            p = self.settle(t, c, 0.6)
            s = 1.0 + 0.04 * ez.progress(t, c.appear, self.dur)  # the month grows a touch over its hold
            spr = scaled(self.head, s)
            blit(frame, spr, cx - spr.w / 2, y - (spr.h - self.head.h) / 2 + self.px(30) * (1 - p), p)
            y += self.head.h + self.rgap
        if self.head is not None and self.year is not None:
            rw = self.rule_w * ez.ease_out_cubic(ez.progress(t, max(c.appear, c.word - 0.05), c.word + 0.5))
            if rw > 2:
                blit(frame, rect(rw, self.rule_h, self.house.rgb("accent")), cx - rw / 2, y)
            y += self.rule_h + self.rgap
        if self.year is not None:
            p = ez.ease_out_cubic(ez.progress(t, c.appear + 0.2, c.appear + 0.7))
            blit(frame, self.year, cx - self.year.w / 2, y - self.px(20) * (1 - p), p)
            y += self.year.h + self.px(40)
        if self.lab is not None and t >= self.lab[1].appear:
            spr, lc = self.lab
            p = self.settle(t, lc)
            blit(frame, spr, cx - spr.w / 2, y + self.px(12) * (1 - p), p)


# ---------------------------------------------------------------- equals card

@register(LayoutKind.equals_card)
class EqualsCard(LayoutComp):
    """Two quantities joined by an equals sign on the ink ground: the left on its words, the equals sign
    drawing on just before the right, the right (in the comparison teal) on its own words."""

    def setup(self) -> None:
        left, right = self.job.first("left"), self.job.first("right")
        self.eqw, self.gx = self.px(96), self.px(60)
        self.mw = (self.bw - self.eqw - 2 * self.gx) / 2
        texts = [e[1].text.upper() for e in (left, right) if e and e[1].text]
        big, floor = self.fs(168, MIN_HEAD_PX), self.fs(MIN_HEAD_PX, MIN_HEAD_PX)
        # one size for both sides (the smaller fit), so the two quantities read as equals
        size = min([fit_text(tx, "Black", big, self.mw, 2, floor, 0.02)[0].size for tx in texts] or [big])
        self.sides = []
        for n, (e, col) in enumerate(((left, "paper"), (right, "accent2"))):
            if e and e[1].text:
                spr = self.label(e[1].text, size / self.k, col, "Black", max_w=self.mw, lines=2, floor=MIN_HEAD_PX,
                                 tracking_em=0.02, align="center")
                self.sides.append((n, spr, e[2]))
        self.cy = self.box[1] + self.bh / 2
        self.bar_h = max(4, self.px(16))
        self.eq_t = None
        if left is not None and right is not None:
            t0 = max(left[2].appear + 0.3, right[2].appear - 0.35)
            self.eq_t = (t0, t0 + 0.35)

    def draw(self, frame: np.ndarray, t: float) -> None:
        cx = self.box[0] + self.bw / 2
        for n, spr, c in self.sides:
            if t < c.appear:
                continue
            p = self.settle(t, c, 0.55)
            sign = -1 if n == 0 else 1
            inward = -sign * self.px(16) * ez.ease_in_out_sine(ez.progress(t, c.appear, self.dur))
            x = cx + sign * (self.eqw / 2 + self.gx + self.mw / 2) - spr.w / 2 + inward
            dx, dy = (0.0, self.px(30) * (1 - p)) if n == 0 else (self.px(60) * (1 - p), 0.0)
            blit(frame, spr, x + dx, self.cy - spr.h / 2 + dy, p)
        if self.eq_t is not None and t >= self.eq_t[0]:
            amber = self.house.rgb("accent")
            for b, yoff in enumerate((-1, 1)):
                u = ez.ease_out_cubic(ez.progress(t, self.eq_t[0] + 0.08 * b, self.eq_t[1] + 0.08 * b))
                if self.eqw * u > 1:
                    blit(frame, rect(self.eqw * u, self.bar_h, amber), cx - self.eqw / 2,
                         self.cy + yoff * self.px(20) - self.bar_h / 2)
