"""Designed layouts and the panel system (house style, word timed, drawn by our own renderer).

Entry points:
- comp_class(kind): the LayoutComp subclass drawing a LayoutKind.
- render_full(job, out, alpha): a full-frame clip, opaque (H.264 mp4) or an alpha overlay (qtrle mov).
- render_panel(job, plan, out): a panel presentation as an alpha overlay clip (qtrle mov).
Each returns the seconds the render took, so the cost of each kind can be reported.

Kinds live in pictures.py (split_comparison, number_over_footage, name_card, list_slideshow, date_title,
equals_card), data.py (chart, table, timeline), routemap.py (route_map) and document.py (document_focus).
"""
from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ...models import LayoutKind
from .. import easing as ez
from .base import LayoutComp, LayoutJob, drift
from .panel import IN_S, PanelComp, PanelPlan

REGISTRY: dict[LayoutKind, type[LayoutComp]] = {}
_MODULES = ("pictures", "data", "routemap", "document")


def register(kind: LayoutKind) -> Callable[[type[LayoutComp]], type[LayoutComp]]:
    def deco(cls: type[LayoutComp]) -> type[LayoutComp]:
        REGISTRY[kind] = cls
        return cls
    return deco


def comp_class(kind: LayoutKind) -> type[LayoutComp]:
    if kind not in REGISTRY:
        for m in _MODULES:
            try:
                importlib.import_module(f"{__name__}.{m}")
            except ModuleNotFoundError as e:
                if e.name != f"{__name__}.{m}":
                    raise
    return REGISTRY[LayoutKind(kind)]


def _sink(out: Path, job: LayoutJob, alpha: bool):  # noqa: ANN202
    from ...render.mg_python import FrameSink
    return FrameSink(Path(out), job.width, job.height, job.fps, alpha, "veryfast")


def render_full(job: LayoutJob, out: Path, alpha: bool = False) -> float:
    """Full-frame layout clip; returns render seconds. Opaque from the cut, or with alpha an overlay that
    fades in over the shot's own picture (a layout whose first words come after the cut). Nothing is written
    when the composition finds the layout incomplete (job.missing): it is shown whole or not at all."""
    t0 = time.perf_counter()
    comp = comp_class(job.spec.kind)(job, job.width, job.height, False)
    if job.missing:
        comp.close()
        return time.perf_counter() - t0
    n = max(1, round(job.duration * job.fps))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    try:
        with _sink(out, job, alpha) as sink:
            for i in range(n):
                f = drift(comp.render(i), i / job.fps, job.duration, job.house.drift)
                if alpha:  # premultiplied: the picture beneath shows only while the layout fades in
                    a = ez.ease_out_cubic(ez.progress(i / job.fps, 0.0, IN_S))
                    f = np.dstack([cv2.convertScaleAbs(f, alpha=a), np.full(f.shape[:2], round(255 * a), np.uint8)])
                sink.write(f)
    finally:
        comp.close()
    return time.perf_counter() - t0


def render_panel(job: LayoutJob, plan: PanelPlan, out: Path) -> float:
    """Panel presentation as a premultiplied alpha overlay clip; returns render seconds. Nothing is written
    when the layout is incomplete (job.missing)."""
    t0 = time.perf_counter()
    cls = comp_class(job.spec.kind)
    pc = PanelComp(job, lambda j, w, h, panel: cls(j, w, h, panel), plan)
    if job.missing:
        pc.close()
        return time.perf_counter() - t0
    n = max(1, round(job.duration * job.fps))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    try:
        with _sink(out, job, True) as sink:
            for i in range(n):
                sink.write(pc.render(i))
    finally:
        pc.close()
    return time.perf_counter() - t0


__all__ = ["REGISTRY", "register", "comp_class", "render_full", "render_panel", "LayoutJob", "LayoutComp",
           "PanelPlan"]
