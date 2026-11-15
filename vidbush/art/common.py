"""Shared helpers of the art layer: narration windows, text detection, image seat access, budgets."""
from __future__ import annotations

import dataclasses
import json
import re
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from ..models import Narration, Shot
from ..providers.base import CallResult, ProviderSpec

ART_DIR = "art"
TEXT_MIN_CONF = 0.6  # OCR confidence a box needs to count as text
TEXT_MIN_HEIGHT = 0.012  # and a height of at least this share of the frame (texture is not text)
_LOCK = threading.Lock()


def art_dir(ctx: Any) -> Path:
    d = Path(ctx.work) / ART_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def narration(ctx: Any) -> Optional[Narration]:
    try:
        return ctx.maybe_load("narration", Narration)
    except Exception:  # noqa: BLE001  a context without narration judges on the intent alone
        return None


def window_text(narr: Optional[Narration], start: float, end: float, pad: float = 0.3) -> str:
    """The narration words spoken while the shot is on screen (pad seconds either side)."""
    if narr is None:
        return ""
    words = [w.text for w in narr.words if w.end >= start - pad and w.start <= end + pad]
    return " ".join(words).strip()


def shot_text(narr: Optional[Narration], shot: Shot, context_s: float = 0.0) -> str:
    return window_text(narr, shot.start - context_s, shot.end + context_s)


def find_text(path: str | Path) -> list[str]:
    """Readable text the OCR finds anywhere in a picture (the judge's own engine), [] when none or when the
    engine is not installed. A designed visual carries no text of its own: every word on screen is drawn
    by the renderer, where it is checked."""
    try:
        from ..judge.prefilter import ocr_engine
        engine = ocr_engine()
    except ImportError:
        return []
    im = Image.open(path).convert("RGB")
    if im.width > 1280:
        im = im.resize((1280, round(im.height * 1280 / im.width)))
    arr = np.asarray(im)
    result, _ = engine(np.ascontiguousarray(arr))
    out = []
    for pts, text, conf in result or []:
        ys = [p[1] for p in pts]
        if float(conf) >= TEXT_MIN_CONF and (max(ys) - min(ys)) >= TEXT_MIN_HEIGHT * arr.shape[0] \
                and re.search(r"[A-Za-z0-9]{2,}", str(text)):
            out.append(str(text))
    return out


def has_bars(path: str | Path, band: float = 0.06, max_std: float = 6.0) -> bool:
    """True when a picture has flat bands painted along its top and bottom (or left and right) edges: a model
    that heard "letterbox" paints bars, which then stack on the film's own letterbox."""
    a = np.asarray(Image.open(path).convert("L").resize((320, 180)), np.float32)
    h, w = a.shape
    bh, bw = max(2, int(band * h)), max(2, int(band * w))
    flat = lambda x: float(x.std()) < max_std  # noqa: E731
    if flat(a[bh:-bh, bw:-bw]):
        return False  # a flat picture has no bars, only a flat picture
    return (flat(a[:bh]) and flat(a[-bh:])) or (flat(a[:, :bw]) and flat(a[:, -bw:]))


def elevate_spec(ctx: Any) -> Optional[ProviderSpec]:
    """The first configured Elevate provider of the image seat (the art layer needs reference images, which
    only Elevate's media API takes here), or None: no art without it, never a mock."""
    if getattr(ctx.seats, "force_mock", False):
        return None
    for p in ctx.seats.providers_for("image"):
        if p.configured and p.type == "elevate":
            return p
    return None


def call_image(ctx: Any, spec: ProviderSpec, fn: Any, note: str, model: str = "") -> CallResult:
    """Run one art image request through the shared provider client, so it is costed, logged and health
    tracked like any image seat call (no fallback to another provider: a reference image is Elevate only).
    The spec carries the art model, so health is tracked per model: one model's outage (Elevate answered
    "temporarily unavailable" for gemini_flash_image three times on 2026-11-15) never marks the other art
    model or the seat's own stills model down."""
    if model and model != spec.model:
        spec = dataclasses.replace(spec, model=model)
    return ctx.seats.client.call("image", ctx.stage or "visuals", [spec], fn, note)


def over_hour_cap(ctx: Any, spec: ProviderSpec) -> bool:
    """True while the key's image credits used this UTC hour reach the profile's hour_credit_cap."""
    cap = float(getattr(ctx.profile.art, "hour_credit_cap", 0) or 0)
    if cap <= 0:
        return False
    from .elevate import hour_credits
    used, _limit = hour_credits(spec)
    return used >= cap


def write_report(ctx: Any, name: str, data: Any) -> Path:
    p = art_dir(ctx) / name
    with _LOCK:
        p.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    return p


def strip_names(text: str, names: list[str]) -> str:
    """Remove the given names (places, people) from a prompt so the image model is never invited to letter
    them on the picture."""
    for n in sorted({n for n in names if n and len(n) > 1}, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(n)}\b", "", text, flags=re.I)
    return re.sub(r"\s{2,}", " ", re.sub(r"\s+([,.;:])", r"\1", text)).strip()
