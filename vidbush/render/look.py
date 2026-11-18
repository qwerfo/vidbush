"""Matched look: adjacent sources brought toward one exposure, white balance and saturation.

Eight providers side by side look assembled by a machine. With StyleProfile.colour.match_look on, every real
source is measured once (a still, or three frames of the clip from its in point): mean luminance, the grey
world colour cast and the mean chroma. The video's target is the median of its colour sources. Each shot is
moved match_strength of the way toward it (a mild correction, clamped), as one ffmpeg filter stored in
Shot.grade by the motion stage. Black and white archival keeps its look: only its exposure moves. An
optional named house grade (colour.brand_grade) is applied after the match, to every real source alike.
Motion graphics and layouts are never graded (they are already in the house colours).
"""
from __future__ import annotations

import math
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..models import GENERATED_SOURCES, ShotList, SourceKind, StyleProfile

BW_CHROMA = 0.035  # mean chroma below this reads as black and white
GAMMA_LIMIT = (0.82, 1.22)
GAIN_LIMIT = 0.12  # at most 12 percent per channel
SAT_LIMIT = (0.8, 1.2)
GRADES = {  # named house grades, mild, applied after the match
    "house_warm": "colorbalance=rs=0.025:bs=-0.03:rm=0.015:bm=-0.02,eq=saturation=0.96:contrast=1.03",
    "house_cool": "colorbalance=rs=-0.02:bs=0.03:rm=-0.01:bm=0.02,eq=saturation=0.94:contrast=1.03",
}


@lru_cache(maxsize=1024)
def _measure(path: str, media: str, in_point: float, mtime: float) -> Optional[tuple[float, float, float, float, float]]:
    """(luma, r, g, b, chroma) means in 0..1 of a still or of three clip frames, or None."""
    import cv2
    frames: list[np.ndarray] = []
    if media == "image":
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    else:
        for dt in (0.3, 1.5, 3.0):
            p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{in_point + dt:.2f}", "-i", path, "-frames:v", "1",
                                "-vf", "scale=160:90", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                               capture_output=True, timeout=60)
            if len(p.stdout) == 160 * 90 * 3:
                frames.append(np.frombuffer(p.stdout, np.uint8).reshape(90, 160, 3))
    if not frames:
        return None
    px = np.concatenate([cv2.resize(f, (160, 90), interpolation=cv2.INTER_AREA).reshape(-1, 3) for f in frames])
    px = px.astype(np.float32) / 255.0
    b, g, r = px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean()
    luma = float((0.2126 * px[:, 2] + 0.7152 * px[:, 1] + 0.0722 * px[:, 0]).mean())
    chroma = float((px.max(axis=1) - px.min(axis=1)).mean())
    return luma, float(r), float(g), float(b), chroma


def measure(cand: Any) -> Optional[tuple[float, float, float, float, float]]:
    p = Path(cand.local_path or "")
    if not p.is_file():
        return None
    return _measure(str(p), cand.media, float(cand.in_point or 0.0), p.stat().st_mtime)


def correction(m: tuple[float, float, float, float, float], target: tuple[float, float, float, float, float],
               strength: float) -> str:
    """ffmpeg filter moving measure m strength of the way toward target (empty when nothing to do)."""
    luma, r, g, b, chroma = m
    tl, tr, tg, tb, tc = target
    parts = []
    if 0.02 < luma < 0.98 and 0.02 < tl < 0.98:
        full = math.log(tl) / math.log(luma)  # luma ** full == tl
        gam = 1.0 / math.exp(strength * math.log(full))
        gam = min(max(gam, GAMMA_LIMIT[0]), GAMMA_LIMIT[1])
        if abs(gam - 1.0) > 0.01:
            parts.append(f"eq=gamma={gam:.3f}")
    if chroma >= BW_CHROMA:  # colour source: white balance and saturation too
        grey, tgrey = (r + g + b) / 3, (tr + tg + tb) / 3
        gains = []
        for c, tc_ in ((r, tr), (g, tg), (b, tb)):
            want = (tc_ / max(tgrey, 1e-3)) / max(c / max(grey, 1e-3), 1e-3)
            gains.append(min(max(want ** strength, 1 - GAIN_LIMIT), 1 + GAIN_LIMIT))
        if max(abs(x - 1) for x in gains) > 0.01:
            parts.append("colorchannelmixer=rr={:.3f}:gg={:.3f}:bb={:.3f}".format(*gains))
        sat = min(max((tc / max(chroma, 1e-3)) ** strength, SAT_LIMIT[0]), SAT_LIMIT[1])
        if abs(sat - 1.0) > 0.02:
            parts.append(f"eq=saturation={sat:.3f}")
    return ",".join(parts)


def assign_grades(shotlist: ShotList, profile: StyleProfile) -> int:
    """Set Shot.grade for every real source shot; returns how many shots were graded."""
    col = profile.colour
    brand = GRADES.get(col.brand_grade, "") if col.brand_grade else ""
    measured = {}
    for s in shotlist.shots:
        s.grade = ""
        c = s.selected_candidate()
        if c is None or c.source == SourceKind.motion or c.source in GENERATED_SOURCES:
            continue
        if col.match_look:
            m = measure(c)
            if m is not None:
                measured[s.id] = m
        elif brand:
            s.grade = brand
    if not col.match_look:
        return sum(1 for s in shotlist.shots if s.grade)
    colour = [m for m in measured.values() if m[4] >= BW_CHROMA] or list(measured.values())
    if not colour:
        return 0
    target = tuple(float(np.median([m[i] for m in colour])) for i in range(5))
    n = 0
    for s in shotlist.shots:
        if s.id in measured:
            f = correction(measured[s.id], target, col.match_strength)  # type: ignore[arg-type]
            s.grade = ",".join(x for x in (f, brand) if x)
            n += bool(s.grade)
    return n
