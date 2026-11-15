"""Illustrated maps that are still true (Part B).

Principle: the geography comes from our renderer, the paint from the model. The exact map view is
rendered as a text free reference (mapref.reference) and sent as the reference image; the prompt asks
for an illustrated rendering of that same view, built from the narration at that moment and the map's
own places, never from a fixed template, and always forbids text, labels and borders drawn by the model.
Several variants are painted; each must pass registration (mapref.register: the painted coast within
tolerance of the true one) and carry no text (OCR). Among those, a vision call on appeal and fit to the
narration picks the best. If none passes, the map stays plain: a map slot is never empty and never shows
wrong geography. Labels, markers, routes, title and credit are drawn by the renderer on top, so they pass
the same verify_map check as on a plain map.

When to paint: a plain map when the narration makes a precise point (a route, a distance, a border, an
arrow, a shaded region); an illustrated one when it establishes a place or describes what a region is
like (paint_reason). Shares and repeat windows are applied by the planner (plan.py).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ..models import MapPayload, MotionSpec
from . import common, mapref
from .elevate import generate as elevate_generate

PRECISE_WORDS = re.compile(
    r"\b(miles?|kilomet(er|re)s?|km|border(s|ed)?|boundary|route|distance|far(ther|thest)?|apart|across|"
    r"from \w+ to|between|north of|south of|east of|west of|times (larger|bigger|the size))\b", re.I)

PAINT_SYSTEM = (
    "You art-direct one illustrated map in an animation-heavy geography documentary. You get the narration "
    "spoken while the map is on screen, the places the map marks and the region it shows. Describe what the "
    "painter should show on THIS land so the map reads as the place the narration describes: terrain relief, "
    "water that looks like water, vegetation, ice or desert, and at most two small illustrated touches the "
    "narration makes relevant (an animal herd, boats, a dog team on a frozen river, a railway), placed where "
    "they belong. Never write any proper name, place name, person name or number: the painter must not letter "
    "anything. Reply with strict JSON only: "
    '{"land": "one sentence", "water": "one sentence", "touches": ["short phrase"], "light": "short phrase"}'
)

# Phrased as a repaint of the given image: models keep the outlines of an image they edit far better than of
# one they take as loose inspiration (run 3 probe: regional views were redrawn when asked for "a map of").
PROMPT_VERSION = "p3"  # in variant file names: a changed prompt never reuses paintings of the old one
LEAD = ("Repaint this exact map image. It is a map seen from directly above: the tan shapes are land, the blue "
        "shapes are sea, lakes and rivers. Trace every outline exactly where it is in the image, same framing and "
        "scale; do not move, add or remove any coast, island, bay, lake or river.")
FIXED_RULES = ("No text, no letters, no labels, no numbers, no place names, no borders or dividing lines, no legend, "
               "no compass rose, no frame, no map grid.")
MAP_STYLE = ("An illustrated, hand-painted relief map in the style of an animation-heavy geography documentary: "
             "painterly shaded relief, rich but natural colour, soft cinematic light; clearly a painting.")


def view_text(p: MapPayload) -> str:
    """Places on the map, for the art director (never sent to the image model)."""
    names = [m.label for m in p.markers if m.label] + [lb.text for lb in p.labels] + list(p.shaded_regions)
    if p.title:
        names.append(p.title)
    lat, lon = (list(p.center) + [0.0, 0.0])[:2]
    return f"Places: {', '.join(dict.fromkeys(names)) or 'none named'}. Center {lat:.1f}, {lon:.1f}, zoom {p.zoom:.1f}."


def paint_reason(spec: MotionSpec, window: str) -> Optional[str]:
    """None when the map may be painted; else why it stays plain (a precise point needs the plain map)."""
    try:
        p = spec.typed_payload()
    except Exception:  # noqa: BLE001
        return "payload unreadable"
    if not isinstance(p, MapPayload):
        return "not a map"
    if p.routes:
        return "route"
    if p.arrows:
        return "arrow"
    if p.shaded_regions:
        return "shaded region (a border point)"
    if PRECISE_WORDS.search(f"{window} {p.title}"):
        return "narration makes a precise point"
    if not mapref.paintable(mapref.land_mask(p, 384, 216)):
        return "too little coastline to hold a painting to the view"
    return None


def _clean(s: Any, names: list[str]) -> str:
    text = " ".join(str(s or "").split())
    text = re.sub(r"\d", "", text)
    return common.strip_names(text, names)[:300]


def map_prompt(p: MapPayload, window: str, ctx: Any, style: str = "") -> tuple[str, bool]:
    """(prompt, from_llm). The art director writes the land, water and touches from the narration and the map's
    places; the code adds the fixed rules. Names are stripped so nothing invites lettering."""
    names = [m.label for m in p.markers] + [lb.text for lb in p.labels] + list(p.shaded_regions) + [p.title]
    names += re.findall(r"\b[A-Z][a-z]+(?:[- ][A-Z][a-z]+)*\b", window)
    raw = None
    try:
        raw = ctx.seats.llm_json("director", "art_map_prompt", PAINT_SYSTEM,
                                 f"Narration: {window or '(none)'}\n{view_text(p)}", ctx.stage or "visuals",
                                 temperature=0.7)
    except Exception as e:  # noqa: BLE001  a failed art director falls back to a neutral description
        ctx.log.warn("art map prompt failed", error=str(e)[:200])
    parts: list[str] = []
    ok = isinstance(raw, dict) and bool(raw.get("land"))
    if ok:
        parts.append(f"Land: {_clean(raw.get('land'), names)}")
        if raw.get("water"):
            parts.append(f"Water: {_clean(raw.get('water'), names)}")
        touches = [t for t in (_clean(t, names) for t in (raw.get("touches") or [])[:2]) if t]
        if touches:
            parts.append("Small illustrated touches where they belong: " + "; ".join(touches) + ".")
        if raw.get("light"):
            parts.append(f"Light: {_clean(raw.get('light'), names)}")
    else:
        parts.append("Land: natural shaded relief with the vegetation, rock, snow and rivers of this latitude. "
                     "Water: deep water with soft painted texture, lighter near the shore.")
    lead = (style.strip() + " ") if style else ""
    return f"{LEAD} {lead}{MAP_STYLE} {' '.join(parts)} {FIXED_RULES}".strip(), ok


APPEAL_SYSTEM = (
    "You are the art director of a geography documentary. Several illustrated map paintings of the same view are "
    "attached, in order. For each, score appeal (beautiful, cinematic, reads clearly as land and water) and fit "
    "(shows what the narration describes) from 0 to 100, and answer two yes or no questions: text (any letters, "
    "numbers or labels painted in) and lines (border lines, grid lines or a frame painted in). Reply with strict "
    'JSON only: {"maps": [{"appeal": n, "fit": n, "text": false, "lines": false}]}'
)


def _appeal(paths: list[str], window: str, ctx: Any) -> list[dict[str, Any]]:
    if not paths:
        return []
    try:
        raw = ctx.seats.vision_json("judge", "art_map_appeal", APPEAL_SYSTEM, f"Narration: {window or '(none)'}",
                                    paths, ctx.stage or "visuals", details=["low"] * len(paths))
    except Exception as e:  # noqa: BLE001
        ctx.log.warn("art map appeal failed", error=str(e)[:200])
        return [{} for _ in paths]
    rows = raw.get("maps") if isinstance(raw, dict) else None
    if raw.get("_mock") or not isinstance(rows, list):
        return [{"mock": True} for _ in paths]
    return [r if isinstance(r, dict) else {} for r in (rows + [{}] * len(paths))[:len(paths)]]


def _seed(key: str, i: int) -> int:
    return int(hashlib.sha256(f"{key}:{i}".encode()).hexdigest()[:8], 16)


def save_base(raw_path: Path, p: MapPayload, reg: dict[str, Any], out: Path, meta: dict[str, Any]) -> Path:
    """The painting warped onto the true view (the small refit registration found), at 1920x1080, with a
    sidecar naming the view it belongs to; the renderer uses it only for that view."""
    img = cv2.cvtColor(cv2.imread(str(raw_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    warp = np.asarray(reg["warp"], np.float32)
    aligned = mapref.warp_to_ref(img, warp, (1920, 1080))
    cv2.imwrite(str(out), cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
    from .hooks import sidecar
    side = {"ok": True, "view": mapref.view_key(p), **meta,
            "registration": {k: reg[k] for k in ("iou", "worst_patch", "drift_p90", "invent_p90")}}
    sidecar(out).write_text(json.dumps(side, indent=1), encoding="utf-8")
    return out


def paint_map(p: MapPayload, window: str, ctx: Any, out_dir: Path, tag: str, variants: int = 0) -> dict[str, Any]:
    """Paint one view. Returns a report with every variant's checks and `base` (the chosen aligned painting) or
    None when no variant passed."""
    spec = common.elevate_spec(ctx)
    if spec is None:
        return {"skipped": "no Elevate image seat", "base": None}
    art = ctx.profile.art
    n = variants or max(1, int(art.map_variants))
    view = mapref.view_key(p)
    key = hashlib.sha256(json.dumps(view, sort_keys=True).encode()).hexdigest()[:10]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"map_{key}_base.png"
    from .hooks import _matches
    if out.exists() and _matches(p, str(out)):  # resume: a finished painting of this view is never redone
        return {"tag": tag, "view": view, "window": window, "variants": [], "base": str(out), "reused": True}
    ref = mapref.reference(p)
    ref_path = out_dir / f"map_{key}_ref.png"
    cv2.imwrite(str(ref_path), cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
    cache = out_dir / f"map_{key}_{tag}_prompt_{PROMPT_VERSION}.json"
    if cache.exists():  # a resumed stage reuses the art director's prompt, so the paintings on disk still match
        prompt, from_llm = json.loads(cache.read_text(encoding="utf-8"))
    else:
        # maps keep the documentary map style, not the brand's illustration line: the probe's plain documentary
        # map style was richer, and a map reads as a map in any set
        prompt, from_llm = map_prompt(p, window, ctx)
        cache.write_text(json.dumps([prompt, from_llm]), encoding="utf-8")
    report: dict[str, Any] = {"tag": tag, "view": view, "window": window, "prompt": prompt, "prompt_from_llm": from_llm,
                              "reference": str(ref_path), "variants": [], "base": None}
    # the last variant is the fallback model: a second painter registers where the first redraws (run 3 probe)
    fb = art.map_fallback_model if art.map_fallback_model and art.map_fallback_model != art.map_model else ""
    models = [art.map_model] * (n - 1) + [fb or art.map_model] if n > 1 else [art.map_model]
    passed: list[dict[str, Any]] = []
    for i, model in enumerate(models):
        name = f"map_{key}_{PROMPT_VERSION}_{'v' if model == art.map_model else 'fb'}{i}.png"
        row = _variant(spec, p, prompt, ref, model, out_dir / name, _seed(key + tag, i), ctx)
        report["variants"].append(row)
        if row.get("ok"):
            passed.append(row)
    if not passed:
        ctx.log.info("illustrated map: no variant passed, plain map kept", view=view)
        return report
    scores = _appeal([r["path"] for r in passed], window, ctx)
    for r, sc in zip(passed, scores):
        r["appeal"] = sc
        if sc.get("text") or sc.get("lines"):  # lettering or painted border or grid lines: never on screen
            r["ok"] = False
            r["reason"] = "vision: " + ("text painted in" if sc.get("text") else "lines painted in")
        r["rank"] = (float(sc.get("appeal") or 50) + float(sc.get("fit") or 50), -float(r["registration"]["worst_patch"]))
    passed = [r for r in passed if r["ok"]]
    if not passed:
        ctx.log.info("illustrated map: every registered variant had text or lines, plain map kept", view=view)
        return report
    best = max(passed, key=lambda r: r["rank"])
    save_base(Path(best["path"]), p, best["registration"], out,
              {"model": best["model"], "prompt": prompt, "source": best["path"]})
    report["base"] = str(out)
    report["chosen"] = best["path"]
    return report


def _variant(spec: Any, p: MapPayload, prompt: str, ref: np.ndarray, model: str, path: Path, seed: int,
             ctx: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path), "model": model, "seed": seed, "ok": False}
    try:
        if not path.exists():
            common.call_image(ctx, spec, model=model, fn=lambda s: elevate_generate(s, prompt, path, model=model, references=[ref],
                                                                    seed=seed), note="art map")
        img = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    except Exception as e:  # noqa: BLE001  one failed variant never stops the map
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        ctx.log.warn("art map variant failed", model=model, error=row["error"])
        return row
    reg = mapref.register(img, p)
    row["registration"] = {k: reg[k] for k in ("ok", "iou", "worst_patch", "drift_p90", "invent_p90", "reason")}
    row["registration"]["warp"] = reg["warp"]
    text = common.find_text(path)
    row["text"] = text
    row["ok"] = bool(reg["ok"]) and not text
    if not reg["ok"]:
        row["reason"] = reg["reason"]
    elif text:
        row["reason"] = f"text in picture: {text[:3]}"
    return row
