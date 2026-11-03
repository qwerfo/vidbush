"""Ken Burns parameters for still shots.

kenburns_params(shot, intensity) -> {zoom0, zoom1, x0, y0, x1, y1}. Zooms lie
in [1.0, 1.0 + intensity]; x and y are crop centers in 0..1 of the source
(or of the derived crop). Deterministic from shot.id with bounded random
pan and zoom. Centers stay within [0.5/zoom, 1 - 0.5/zoom], so the crop
never leaves the image at either end, and because that bound is concave in
zoom it also holds at every point of a linear interpolation. shot.kenburns
overrides any key; overrides are clamped to the same bounds.
"""
from __future__ import annotations

import random

from ..models import Shot
from ..util import clamp, stable_hash

MAX_INTENSITY = 0.5
PAN_SHARE = 0.85  # fraction of the allowed travel a pan may use
FRAMING_START = {"full": 0.0, "tight": 0.5, "detail": 0.7}  # share of intensity tight/detail start from
KEYS = ("zoom0", "zoom1", "x0", "y0", "x1", "y1")


def center_bound(zoom: float) -> float:
    """Largest allowed |center - 0.5| at this zoom."""
    return max(0.0, 0.5 - 0.5 / max(zoom, 1.0))


def _zooms(rng: random.Random, framing: str, intensity: float) -> tuple[float, float]:
    hi = 1.0 + intensity
    start = 1.0 + intensity * FRAMING_START.get(framing, 0.0)
    if framing == "full":
        lo, high = 1.0, 1.0 + intensity * rng.uniform(0.6, 1.0)
        return (lo, high) if rng.random() < 0.6 else (high, lo)
    if rng.random() < 0.5:  # push further in from an already tight start
        return start + (hi - start) * rng.uniform(0.0, 0.3), hi
    return hi, start  # ease out to the framing's minimum


def _center(rng: random.Random, zoom: float) -> float:
    return 0.5 + rng.uniform(-1.0, 1.0) * center_bound(zoom) * PAN_SHARE


def kenburns_params(shot: Shot, intensity: float) -> dict[str, float]:
    """Deterministic bounded pan and zoom for a still shot; shot.kenburns overrides."""
    intensity = clamp(float(intensity), 0.0, MAX_INTENSITY)
    rng = random.Random(stable_hash("kenburns", shot.id))
    zoom0, zoom1 = _zooms(rng, shot.framing, intensity)
    params = {"zoom0": zoom0, "zoom1": zoom1,
              "x0": _center(rng, zoom0), "y0": _center(rng, zoom0),
              "x1": _center(rng, zoom1), "y1": _center(rng, zoom1)}
    for k, v in (shot.kenburns or {}).items():
        if k in KEYS and v is not None:
            params[k] = float(v)
    return _clamped(params)


def _clamped(p: dict[str, float]) -> dict[str, float]:
    out = {"zoom0": max(1.0, p["zoom0"]), "zoom1": max(1.0, p["zoom1"])}
    for c, z in (("x0", "zoom0"), ("y0", "zoom0"), ("x1", "zoom1"), ("y1", "zoom1")):
        b = center_bound(out[z])
        out[c] = clamp(p[c], 0.5 - b, 0.5 + b)
    return {k: round(out[k], 5) for k in KEYS}
