"""Contact sheet, numbered clips and yield numbers from a proving run's reports.

    python -m vidbush.art.sheet --review <review dir>

Reads <review>/art_project/work/art/art_report*.json. The best twelve stills overall (illustrated maps as the
renderer finally draws them, with markers and title, and illustrations), at least four of each when there are,
ordered by time, go on one numbered sheet with each item's timestamp and the narration line it serves. Every
clip that passed its checks is copied beside it as clip_NN.mp4 (by time), with clips.md naming each one's
timestamp and line. yield.json counts what was generated, what passed and what was picked.
"""
from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ..mg.palettes import font_path
from .illustrate import TEXT_ARTIFACT

TILE_W, TILE_H, CAP_H, PAD, COLS = 640, 360, 150, 24, 4


def _items(art: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in sorted(art.glob("art_report*.json")):
        if "attempt" in f.name:
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        out += data if isinstance(data, list) else []
    return out


def mmss(t: Any) -> str:
    t = float(t or 0)
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _line(text: str, n: int = 150) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "..."


def stills(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for it in items:
        if it.get("type") == "illustrated_map" and it.get("ok") and it.get("frame") and Path(it["frame"]).exists():
            chosen = next((v for v in it["report"]["variants"] if v.get("path") == it["report"].get("chosen")), {})
            ap = chosen.get("appeal") or {}
            score = (float(ap.get("appeal") or 60) + float(ap.get("fit") or 60)) / 2
            out.append({"kind": "illustrated map", "path": it["frame"], "start": it["start"], "line": it["window"],
                        "score": score})
        elif it.get("type") == "illustration" and it.get("ok") and it.get("still") and Path(it["still"]).exists()                 and not TEXT_ARTIFACT.search(str(it.get("report", {}).get("brief", "")))                 and not it.get("scene"):  # the picker's own rule; a split-panel picture is not one scene
            out.append({"kind": "illustration", "path": it["still"], "start": it["start"], "line": it["window"],
                        "score": float(it.get("score") or 0)})
    seen: dict[str, dict[str, Any]] = {}  # one item per picture: several processes may report the same one
    for c in out:
        if c["path"] not in seen or c["score"] > seen[c["path"]]["score"]:
            seen[c["path"]] = c
    return list(seen.values())


def pick12(cands: list[dict[str, Any]], n: int = 12, floor: int = 4) -> list[dict[str, Any]]:
    best: dict[tuple, dict[str, Any]] = {}  # one item per moment and kind: the best scored
    for c in cands:
        k = (c["kind"], round(float(c["start"])))
        if k not in best or c["score"] > best[k]["score"]:
            best[k] = c
    cands = list(best.values())
    by = {k: sorted([c for c in cands if c["kind"] == k], key=lambda c: -c["score"])
          for k in ("illustrated map", "illustration")}
    chosen = [c for k in by for c in by[k][:floor]]
    rest = sorted([c for c in cands if c not in chosen], key=lambda c: -c["score"])
    chosen += rest[:max(0, n - len(chosen))]
    return sorted(chosen[:n], key=lambda c: c["start"])


def contact_sheet(chosen: list[dict[str, Any]], out: Path) -> Path:
    rows = (len(chosen) + COLS - 1) // COLS
    W = COLS * TILE_W + (COLS + 1) * PAD
    H = rows * (TILE_H + CAP_H) + (rows + 1) * PAD
    sheet = Image.new("RGB", (W, H), (22, 24, 28))
    d = ImageDraw.Draw(sheet)
    big = ImageFont.truetype(str(font_path("Bold")), 44)
    mid = ImageFont.truetype(str(font_path("SemiBold")), 22)
    small = ImageFont.truetype(str(font_path("Regular")), 19)
    for i, c in enumerate(chosen):
        x = PAD + (i % COLS) * (TILE_W + PAD)
        y = PAD + (i // COLS) * (TILE_H + CAP_H + PAD)
        im = Image.open(c["path"]).convert("RGB")
        im = im.resize((TILE_W, round(im.height * TILE_W / im.width)))
        top = max(0, (im.height - TILE_H) // 2)
        sheet.paste(im.crop((0, top, TILE_W, top + TILE_H)), (x, y))
        d.rectangle((x, y, x + 74, y + 60), fill=(0, 0, 0))
        d.text((x + 10, y + 6), str(i + 1), font=big, fill=(255, 216, 74))
        d.text((x, y + TILE_H + 8), f"{mmss(c['start'])}   {c['kind']}", font=mid, fill=(255, 255, 255))
        for k, ln in enumerate(textwrap.wrap(_line(c["line"]), 58)[:4]):
            d.text((x, y + TILE_H + 40 + k * 24), ln, font=small, fill=(200, 200, 200))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def failed_stills(art: Path) -> set[str]:
    """Stills whose clip failed any check in any report, earlier attempts included."""
    out: set[str] = set()
    for f in art.glob("art_report*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        for it in data if isinstance(data, list) else []:
            if it.get("type") == "clip" and it.get("check") and not it.get("ok"):
                out.add(str(it.get("still")))
    return out


def number_clips(items: list[dict[str, Any]], review: Path, failed: set[str] = frozenset()) -> list[dict[str, Any]]:
    """Clips whose every check passed: a clip that failed once and passed on a later check is not counted, because
    the vision check is not deterministic and the later pass would be a re-roll."""
    ok, seen = [], set()
    for it in sorted(items, key=lambda it: float(it.get("start") or 0)):
        if it.get("type") == "clip" and it.get("ok") and it.get("final") and Path(it["final"]).exists() \
                and str(it.get("still")) not in failed and str(it.get("still")) not in seen:
            seen.add(str(it.get("still")))
            ok.append(it)
    for old in review.glob("clip_*.mp4"):
        old.unlink()
    lines = ["# Clips", "", "| clip | time | made from | narration line |", "|---|---|---|---|"]
    out = []
    for i, it in enumerate(ok, 1):
        dst = review / f"clip_{i:02d}.mp4"
        shutil.copy2(it["final"], dst)
        kind = {"illustrated_map": "illustrated map", "illustration": "illustration",
                "ai_image": "AI still (photo style)"}.get(it["kind"], it["kind"])
        lines.append(f"| {i} | {mmss(it.get('start'))} | {kind} | {_line(it.get('window'), 200)} |")
        out.append({"n": i, "file": str(dst), "kind": it["kind"], "start": it.get("start")})
    (review / "clips.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def yield_numbers(items: list[dict[str, Any]]) -> dict[str, Any]:
    maps = [it for it in items if it.get("type") == "illustrated_map"]
    ills = [it for it in items if it.get("type") == "illustration"]
    clips = [it for it in items if it.get("type") == "clip"]
    mv = [v for it in maps for v in it["report"].get("variants", [])]
    iv = [v for it in ills for v in it["report"].get("variants", [])]
    return {
        "maps": {"views": len(maps), "paintings": sum(1 for v in mv if not v.get("error")),
                 "passed_registration": sum(1 for v in mv if (v.get("registration") or {}).get("ok")),
                 "passed_all": sum(1 for v in mv if v.get("ok")), "views_with_a_base": sum(1 for it in maps if it.get("ok"))},
        "illustrations": {"shots": len(ills), "images": len(iv),
                          "text_found": sum(1 for v in iv if v.get("text")),
                          "passed_judge": sum(1 for v in iv if v.get("score") is not None and not v.get("reject")
                                              and not v.get("text") and float(v.get("score") or 0) >= 70),
                          "shots_with_one": sum(1 for it in ills if it.get("ok"))},
        "clips": {"generated": sum(1 for c in clips if c.get("raw")), "passed": sum(1 for c in clips if c.get("ok")),
                  "failed_reasons": [r for c in clips if c.get("raw") and not c.get("ok")
                                     for r in ((c.get("check") or {}).get("reasons") or [])]},
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="contact sheet and numbered clips of a proving run")
    ap.add_argument("--review", required=True)
    args = ap.parse_args(argv)
    review = Path(args.review)
    items = _items(review / "art_project" / "work" / "art")
    chosen = pick12(stills(items))
    contact_sheet(chosen, review / "art_contact_sheet.png")
    bad = failed_stills(review / "art_project" / "work" / "art")
    bad |= {str(it["still"]) for it in items if it.get("type") == "illustration"
            and (it.get("scene") or TEXT_ARTIFACT.search(str(it.get("report", {}).get("brief", ""))))}  # its clip too
    clips = number_clips(items, review, bad)
    y = yield_numbers(items)
    (review / "yield.json").write_text(json.dumps({"yield": y, "sheet": chosen, "clips": clips}, indent=1),
                                       encoding="utf-8")
    print(json.dumps(y, indent=1))


if __name__ == "__main__":
    main()
