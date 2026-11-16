"""Review sheets: one row per clause with its words, the new treatment and picked frame, and the old planner's
picture on screen at the same moment.

    python -m vidbush.plan.compare <project_dir> <replay_out_dir> <scene_id> <out_dir>
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

from ..models import Shot, ShotList, ShotType
from . import clauses as C
from .sheets import _fit, _font

THUMB_W, THUMB_H = 240, 135
TEXT_W = 560
ROW_H = THUMB_H + 10
PAGE_ROWS = 14
HEAD_H = 52


def _at(sl: ShotList, t: float) -> Optional[Shot]:
    return next((s for s in sl.shots if s.start <= t < s.end), None)


def _frame_of(shot: Optional[Shot]) -> tuple[Optional[str], str]:
    if shot is None:
        return None, "no shot"
    c = shot.selected_candidate()
    if c is None or not c.keyframes:
        if shot.type == ShotType.motion_graphic:
            return None, "layout"
        return None, "no pick"
    # the old render plays from in_point; keyframe a (20 percent in) is the nearest stored frame
    return c.keyframes[0], c.source.value


def _new_frame(shot: Optional[Shot], plan_rows: dict) -> tuple[Optional[str], str]:
    if shot is None:
        return None, "no shot"
    c = shot.selected_candidate()
    if c is None or shot.type == ShotType.motion_graphic:
        lay = shot.layout
        return None, (f"LAYOUT {lay.kind}" if lay else "nothing fits")
    row = plan_rows.get(shot.id, {})
    letter = (row.get("edit") or {}).get("frame") or "a"
    idx = "abc".index(letter) if letter in "abc" else 0
    frames = c.keyframes
    return (frames[min(idx, len(frames) - 1)] if frames else None), c.source.value


def build(project: Path, replay: Path, scene: str, out_dir: Path) -> list[str]:
    clauses = C.load(replay / "work" / "clauses.json")
    plan = json.loads((replay / "work" / "plan_v2.json").read_text(encoding="utf-8"))
    new = ShotList.model_validate_json((replay / "work" / "shotlist.json").read_text(encoding="utf-8"))
    sl_path = project / "work" / "shotlist.json"
    old = ShotList.model_validate_json(sl_path.read_text(encoding="utf-8"))
    rows_by_clause = {}
    for sc in plan["scenes"]:
        for r in sc["shots"]:
            for cid in r["clauses"]:
                rows_by_clause[cid] = r
    shot_rows = {}
    for s in new.shots:
        head = next((c for c in clauses if abs(c.start - s.start) < 1e-6), None)
        if head and head.id in rows_by_clause:
            shot_rows[s.id] = rows_by_clause[head.id]
    cl = [c for c in clauses if c.scene == scene]
    purpose = next((sc.get("scene_purpose", "") for sc in plan["scenes"] if sc["scene"] == scene), "")
    out_dir.mkdir(parents=True, exist_ok=True)
    f_txt, f_small, f_head = _font(15), _font(13), _font(17)
    paths = []
    for page in range(0, len(cl), PAGE_ROWS):
        chunk = cl[page:page + PAGE_ROWS]
        W = TEXT_W + 2 * (THUMB_W + 10) + 10
        img = Image.new("RGB", (W, HEAD_H + ROW_H * len(chunk)), (250, 250, 247))
        d = ImageDraw.Draw(img)
        d.text((8, 6), f"{scene}  {purpose[:110]}", fill=(0, 0, 0), font=f_head)
        d.text((TEXT_W + 10, HEAD_H - 16), "NEW (clause-first)", fill=(0, 90, 0), font=f_small)
        d.text((TEXT_W + THUMB_W + 20, HEAD_H - 16), "OLD (run 3, same moment)", fill=(140, 0, 0), font=f_small)
        for r, c in enumerate(chunk):
            y = HEAD_H + r * ROW_H
            mid = (c.speech_start + c.speech_end) / 2
            ns, os_ = _at(new, mid), _at(old, mid)
            row = rows_by_clause.get(c.id, {})
            held = row and row["clauses"][0] != c.id
            treat = "hold" if held else row.get("treatment", "?")
            if row.get("presentation") and not held:
                treat += f" / {row['presentation']}"
            g = ",".join(row.get("grammar", [])) if not held else ""
            d.line([(0, y), (W, y)], fill=(200, 200, 200))
            d.text((8, y + 4), f"{c.id}  {c.start:6.1f}s  {c.duration:.1f}s  [{treat}] {g}", fill=(20, 20, 120), font=f_small)
            for k, line in enumerate(textwrap.wrap(c.text, 62)[:4]):
                d.text((8, y + 22 + k * 18), line, fill=(0, 0, 0), font=f_txt)
            if not held:
                subj = (row.get("subject") or "")[:150]
                for k, line in enumerate(textwrap.wrap("subject: " + subj, 76)[:2]):
                    d.text((8, y + 98 + k * 16), line, fill=(90, 90, 90), font=f_small)
            for col, (path, tag) in enumerate((_new_frame(ns, shot_rows), _frame_of(os_))):
                x = TEXT_W + 10 + col * (THUMB_W + 10)
                if path and Path(path).exists():
                    img.paste(_fit(path, THUMB_W, THUMB_H), (x, y + 4))
                else:
                    d.rectangle([x, y + 4, x + THUMB_W, y + 4 + THUMB_H], fill=(60, 60, 70))
                    lay = ns.layout if (col == 0 and ns is not None) else None
                    txt = tag if lay is None else f"{lay.kind}\n" + "\n".join(e.text for e in lay.elements[:4])
                    d.multiline_text((x + 8, y + 12), txt[:120], fill=(255, 255, 255), font=f_small)
                d.rectangle([x, y + 4 + THUMB_H - 18, x + THUMB_W, y + 4 + THUMB_H], fill=(0, 0, 0))
                d.text((x + 4, y + 4 + THUMB_H - 17), tag[:30], fill=(255, 255, 0), font=f_small)
        p = out_dir / f"sheet_{scene}_{page // PAGE_ROWS + 1}.jpg"
        img.save(p, quality=90)
        paths.append(str(p))
    return paths


if __name__ == "__main__":
    a = sys.argv[1:]
    print(build(Path(a[0]), Path(a[1]), a[2], Path(a[3])))
