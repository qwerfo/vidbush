"""Easing curves and time helpers for motion graphics.

Every curve maps progress p in [0, 1] to [0, 1] (ease_out_back overshoots
slightly above 1 by design). Inputs outside [0, 1] are clamped.
"""
from __future__ import annotations

import math


def clamp01(p: float) -> float:
    """Clamp to [0, 1]."""
    return 0.0 if p <= 0.0 else 1.0 if p >= 1.0 else p


def progress(t: float, start: float, end: float) -> float:
    """Linear progress of t through [start, end], clamped. A zero length window is a step."""
    if end <= start:
        return 1.0 if t >= start else 0.0
    return clamp01((t - start) / (end - start))


def linear(p: float) -> float:
    return clamp01(p)


def ease_in_cubic(p: float) -> float:
    p = clamp01(p)
    return p * p * p


def ease_out_cubic(p: float) -> float:
    p = clamp01(p)
    return 1.0 - (1.0 - p) ** 3


def ease_in_out_cubic(p: float) -> float:
    p = clamp01(p)
    return 4 * p * p * p if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def ease_out_quart(p: float) -> float:
    p = clamp01(p)
    return 1.0 - (1.0 - p) ** 4


def ease_out_expo(p: float) -> float:
    p = clamp01(p)
    return 1.0 if p >= 1.0 else 1.0 - 2 ** (-10 * p)


def ease_in_out_sine(p: float) -> float:
    p = clamp01(p)
    return -(math.cos(math.pi * p) - 1) / 2


def ease_out_back(p: float, overshoot: float = 1.70158) -> float:
    """Pop with a small overshoot (used for markers and cards)."""
    p = clamp01(p)
    c3 = overshoot + 1
    return 1 + c3 * (p - 1) ** 3 + overshoot * (p - 1) ** 2


def smoothstep(p: float) -> float:
    p = clamp01(p)
    return p * p * (3 - 2 * p)


def envelope(t: float, duration: float, t_in: float = 0.5, t_out: float = 0.4) -> tuple[float, float]:
    """(entrance, exit) progress: entrance goes 0 to 1 over the first t_in seconds,
    exit goes 0 to 1 over the last t_out seconds. Both scale down for short clips."""
    k = min(1.0, duration / max(1e-6, (t_in + t_out) * 1.6))
    t_in, t_out = t_in * k, t_out * k
    return progress(t, 0.0, t_in), progress(t, duration - t_out, duration)


def fade_factor(t: float, duration: float, fade: float = 0.3) -> float:
    """Opacity for a full-frame fade in and fade out of `fade` seconds (shortened for short clips)."""
    fade = min(fade, duration / 4)
    if fade <= 0:
        return 1.0
    return min(progress(t, 0.0, fade), 1.0 - progress(t, duration - fade, duration))


def stagger(i: int, n: int, start: float, span: float, each: float) -> tuple[float, float]:
    """Window [a, b] for item i of n: items start evenly within `span` and each lasts `each`."""
    a = start + (span * i / max(1, n - 1) if n > 1 else 0.0)
    return a, a + each


def lerp(a: float, b: float, p: float) -> float:
    return a + (b - a) * p
