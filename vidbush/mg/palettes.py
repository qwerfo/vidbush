"""Motion graphics palettes and bundled fonts.

Three palettes: documentary_warm (aged paper, ink, amber accent),
documentary_cold (steel blue, slate, cyan accent) and top10_bold (near black,
white, strong red with gold). Unknown names fall back to documentary_warm.
Fonts are the Inter weights bundled in assets/fonts.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..config import REPO_ROOT

FONT_DIR = Path(REPO_ROOT) / "assets" / "fonts"
Weight = Literal["Regular", "Medium", "SemiBold", "Bold", "ExtraBold", "Black"]
RGB = tuple[int, int, int]


class Palette(BaseModel):
    """Named colours as #RRGGBB strings. `dark` tells compositions which way contrast goes."""

    name: str
    bg: str  # background base
    bg2: str  # background gradient partner (edges)
    land: str
    land_border: str
    water: str
    text: str
    muted: str
    accent: str
    accent2: str
    route: str
    shade: str  # shaded region fill
    panel: str  # overlay panel behind text (lower thirds, callouts, pills)
    panel_text: str
    grid: str  # chart gridlines and map graticule
    dark: bool = False

    def rgb(self, key: str) -> RGB:
        """Colour by field name as an (r, g, b) tuple."""
        return hex_rgb(getattr(self, key))


PALETTES: dict[str, Palette] = {
    "documentary_warm": Palette(
        name="documentary_warm", bg="#EAE0CB", bg2="#CDBB98", land="#E9DCBD", land_border="#9B8763",
        water="#A9BAB3", text="#2B2118", muted="#76664F", accent="#C47A1C", accent2="#8C3A2B",
        route="#9E2F23", shade="#D49533", panel="#F2EADA", panel_text="#2B2118", grid="#B9A887", dark=False),
    "documentary_cold": Palette(
        name="documentary_cold", bg="#122030", bg2="#070E17", land="#2B3D51", land_border="#5A7290",
        water="#0C1724", text="#EAF1F7", muted="#8DA0B6", accent="#3CCFE4", accent2="#86AEDD",
        route="#3CCFE4", shade="#2C93A8", panel="#0E1A27", panel_text="#EAF1F7", grid="#2A3B4F", dark=True),
    "top10_bold": Palette(
        name="top10_bold", bg="#141418", bg2="#050506", land="#26262C", land_border="#4A4A55",
        water="#0D0D10", text="#FFFFFF", muted="#A2A2AD", accent="#E5252C", accent2="#F6B800",
        route="#F6B800", shade="#C81F26", panel="#0E0E11", panel_text="#FFFFFF", grid="#2C2C33", dark=True),
}


def hex_rgb(value: str) -> RGB:
    """'#RRGGBB' to (r, g, b)."""
    v = value.lstrip("#")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def mix(a: RGB, b: RGB, p: float) -> RGB:
    """Linear blend from a to b."""
    return tuple(int(round(x + (y - x) * p)) for x, y in zip(a, b))  # type: ignore[return-value]


def get_palette(name: str) -> Palette:
    """Palette by name, documentary_warm when unknown."""
    return PALETTES.get(name or "", PALETTES["documentary_warm"])


@lru_cache(maxsize=None)
def font_path(weight: str = "Regular") -> Path:
    """Path of the bundled Inter font for a weight (Regular when the weight is unknown)."""
    p = FONT_DIR / f"Inter-{weight}.otf"
    return p if p.exists() else FONT_DIR / "Inter-Regular.otf"
