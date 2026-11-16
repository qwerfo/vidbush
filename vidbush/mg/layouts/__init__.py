"""Designed layouts and the panel system (house style, word timed, drawn by our own renderer).

Entry points:
- comp_class(kind): the LayoutComp subclass drawing a LayoutKind.
- render_full(job, out): an opaque full-frame clip (H.264 mp4).
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

from ...models import LayoutKind
from .base import LayoutComp, LayoutJob, drift
from .panel import PanelComp, PanelPlan

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


def render_full(job: LayoutJob, out: Path) -> float:
    """Opaque full-frame layout clip; returns render seconds."""
    t0 = time.perf_counter()
    comp = comp_class(job.spec.kind)(job, job.width, job.height, False)
    n = max(1, round(job.duration * job.fps))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    try:
        with _sink(out, job, False) as sink:
            for i in range(n):
                sink.write(drift(comp.render(i), i / job.fps, job.duration, job.house.drift))
    finally:
        comp.close()
    return time.perf_counter() - t0


def render_panel(job: LayoutJob, plan: PanelPlan, out: Path) -> float:
    """Panel presentation as a premultiplied alpha overlay clip; returns render seconds."""
    t0 = time.perf_counter()
    cls = comp_class(job.spec.kind)
    pc = PanelComp(job, lambda j, w, h, panel: cls(j, w, h, panel), plan)
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
