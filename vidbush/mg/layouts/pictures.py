"""Layouts built from real pictures and short text.

number_over_footage is the reference implementation of the pattern every kind follows: setup() builds
sprites once, draw() places them from the cues (nothing shows before its cue), and pictures come from
job.pictures (real candidates only; a generated picture is refused).
"""
from __future__ import annotations

import re
from typing import Optional

import cv2
import numpy as np

from ...models import LayoutKind
from .. import easing as ez
from ..house import MIN_BODY_PX, MIN_HERO_PX
from ..textdraw import Sprite, blit, font, rect, text_sprite
from . import register
from .base import Footage, LayoutComp, darken

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
        hero = self.fs(250, MIN_HERO_PX)
        f = font("Black", hero)
        while f.getlength(head) > self.bw * 0.9 and f.size > self.fs(MIN_HERO_PX, MIN_HERO_PX):
            f = font("Black", int(f.size * 0.92))
        self.nf = f
        self.num = text_sprite(head, f, self.house.rgb("paper"), -0.02 * f.size, shadow=0.5)
        self.unit = self.label(unit, 64, "accent", max_w=self.bw * 0.8, floor=MIN_BODY_PX, tracking_em=0.22,
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
