"""The house style: one type, colour and motion set for every designed layout, panel, label and map frame.

Defined once here, overridable per brand (Brand.house_style), so every graphic in a video reads as one set.
The look is our own, not the benchmark's: a deep blue-black ink ground with a faint paper grain, warm paper
white type, one amber signal accent (it ties to the amber of the documentary_warm map palette, so maps and
layouts share a colour), a muted sea teal as the second voice of a comparison, Inter throughout (Black for
hero numbers, SemiBold tracked capitals for labels, Medium for sentences), thin paper-white picture frames
with a soft drop shadow, and one motion vocabulary: things settle in (a short rise with ease out), labels
wipe on behind a growing accent rule, lines draw, and the whole layout drifts slowly so nothing is static.

Type floors (output pixels at 1080p, scaled with frame height), chosen for an older viewer on a television
at living room distance: at 3 m from a 55 inch set one 1080p pixel is about 0.64 mm, so 34 px Inter (cap
height about 25 px, 16 mm) is the smallest label we allow anywhere, panels included (run 3 panel type was
about 20 px, under 10 mm cap height, which reviewers could not read). Sentences and captions under a
picture use 40 px, headline labels 56 px, hero numbers 150 px and up.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Optional

from .palettes import RGB, hex_rgb

# minimum type sizes in px at a 1080 px tall frame (see module docstring for why)
MIN_LABEL_PX = 34  # any text anywhere, panel chips included
MIN_BODY_PX = 40  # sentences, captions under pictures, table cells
MIN_HEAD_PX = 56  # a layout's title or name
MIN_HERO_PX = 150  # the number of a number layout

LEAD_S = 0.25  # an element starts entering this long before its first anchor word (rule: never more than 0.3 s)
MAX_LEAD_S = 0.3
SETTLE_S = 0.45  # entry animation length


@dataclass(frozen=True)
class House:
    ink: str = "#0E1319"  # layout ground
    ink2: str = "#1A222C"  # ground gradient partner, panel body
    paper: str = "#F3EEE4"  # type and picture frames
    muted: str = "#A7AFB8"  # secondary type
    accent: str = "#E8A33D"  # amber: the one signal colour (rules, the spoken row, the line being drawn)
    accent2: str = "#5FB3A6"  # sea teal: the second side of a comparison, second series
    shade: float = 0.42  # brightness kept on footage under type (darkened background)
    frame_px: int = 4  # picture frame thickness at 1080p
    radius_px: int = 6  # picture and panel corner radius at 1080p
    grain: float = 3.5  # paper grain strength on the ink ground
    drift: float = 0.035  # whole-layout slow push over its hold (share of frame)
    font: str = "Inter"

    def rgb(self, key: str) -> RGB:
        return hex_rgb(getattr(self, key))


DEFAULT = House()


def house_style(overrides: Optional[dict[str, Any]] = None) -> House:
    """The house style with a brand's overrides (unknown keys ignored)."""
    if not overrides:
        return DEFAULT
    names = {f.name for f in fields(House)}
    return replace(DEFAULT, **{k: v for k, v in overrides.items() if k in names})


def type_px(base: float, floor: float, height: int) -> int:
    """Type size in output pixels: base design px (1080p) and the floor, both scaled to the frame height."""
    s = height / 1080.0
    return int(round(max(base, floor) * s))
