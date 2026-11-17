"""Calibration sheet (FABLE_REVIEW 6.2): 40 tiles for a person to mark, and the agreement check.

The page never shows the audit's verdict; the verdict goes to a separate key file.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

from . import core

TRUSTED_AT = 34  # of 40 (85%)


def pick(rows: list[dict[str, Any]], per_side: int = 20) -> list[dict[str, Any]]:
    """per_side flagged and per_side unflagged tiles, each side spread evenly over the video, in time order."""
    valid = [r for r in rows if not r.get("error") and "noticeable" in r]
    yes = core.spread([r for r in valid if r["noticeable"]], per_side)
    no = core.spread([r for r in valid if not r["noticeable"]], per_side)
    return sorted(yes + no, key=lambda r: r["t"])


def tile_id(r: dict[str, Any], vhash: str) -> str:
    """Tile id carrying the video's short hash, so marks of another render can never match a key."""
    return f"{vhash}-{core.frame_name(r['t'])[:-4]}"


class CalibrationError(ValueError):
    pass


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Tile calibration: {title}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:0 auto;padding:16px;background:#f4f4f4;color:#111}}
header{{position:sticky;top:0;background:#f4f4f4;padding:8px 0;border-bottom:1px solid #ccc;z-index:1}}
.tile{{background:#fff;border:1px solid #ccc;border-radius:6px;padding:12px;margin:16px 0}}
.tile img{{width:640px;max-width:100%;display:block}}
.meta{{font-weight:600;margin:8px 0 4px}} .ctx{{color:#555;font-size:14px}}
.words{{font-size:17px;margin:4px 0 10px}}
button{{font-size:15px;padding:8px 16px;margin-right:8px;border:2px solid #888;background:#fff;border-radius:4px;cursor:pointer}}
button.sel-notice{{background:#c62828;color:#fff;border-color:#c62828}}
button.sel-fine{{background:#2e7d32;color:#fff;border-color:#2e7d32}}
button:focus{{outline:3px solid #1565c0}}
textarea{{width:100%;min-height:60px;margin-top:8px;font:inherit}}
</style></head><body>
<header><h1 style="margin:0;font-size:20px">Would an ordinary viewer notice a problem in this frame?</h1>
<p style="margin:4px 0">Video: {title}. Judge the picture against the words spoken at that moment.
Marked: <span id="progress">0</span> of {n}. <button id="export" type="button">Export</button></p></header>
{tiles}
<script>
const VIDEO = {video_json};
const IDS = {ids_json};
const TIMES = {times_json};
const marks = {{}};
function setMark(id, mark) {{
  marks[id] = marks[id] === mark ? null : mark;
  for (const m of ["notice", "fine"]) {{
    const b = document.getElementById(id + "-" + m);
    b.className = marks[id] === m ? "sel-" + m : "";
    b.setAttribute("aria-pressed", marks[id] === m ? "true" : "false");
  }}
  document.getElementById("progress").textContent = IDS.filter(i => marks[i]).length;
}}
function exportMarks() {{
  const out = {{video: VIDEO, marks: IDS.map(id => ({{id: id, t: TIMES[id], mark: marks[id] || null,
    comment: document.getElementById(id + "-comment").value}}))}};
  const blob = new Blob([JSON.stringify(out, null, 2)], {{type: "application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "calibration_marks.json";
  document.body.appendChild(a); a.click(); a.remove();
}}
document.getElementById("export").addEventListener("click", exportMarks);
for (const id of IDS) for (const m of ["notice", "fine"])
  document.getElementById(id + "-" + m).addEventListener("click", () => setMark(id, m));
</script></body></html>
"""

TILE = """<section class="tile" id="{id}">
<img alt="frame at {ts}" src="data:image/jpeg;base64,{b64}">
<div class="meta">{num}. {ts}</div>
<div class="ctx">Before: {before}</div>
<div class="words">Spoken now: <b>{words}</b></div>
<div class="ctx">After: {after}</div>
<p><button type="button" id="{id}-notice" aria-pressed="false">Would notice</button>
<button type="button" id="{id}-fine" aria-pressed="false">Fine</button></p>
<label>Comment<textarea id="{id}-comment"></textarea></label>
</section>"""


def build_sheet(rows: list[dict[str, Any]], tiles_dir: Path, video: str, vhash: str, html_path: Path,
                key_path: Path, per_side: int = 20) -> list[dict[str, Any]]:
    chosen = pick(rows, per_side)
    parts = []
    for i, r in enumerate(chosen, 1):
        frame = tiles_dir / core.frame_name(r["t"])
        b64 = base64.b64encode(frame.read_bytes()).decode("ascii")
        parts.append(TILE.format(id=tile_id(r, vhash), ts=core.mmss(r["frame_t"]), b64=b64, num=i,
                                 words=html.escape(r.get("words", "")), before=html.escape(r.get("before", "")),
                                 after=html.escape(r.get("after", ""))))
    ids = [tile_id(r, vhash) for r in chosen]
    page = PAGE.format(title=html.escape(Path(video).name), n=len(chosen), tiles="\n".join(parts),
                       video_json=json.dumps(video), ids_json=json.dumps(ids),
                       times_json=json.dumps({tile_id(r, vhash): r["t"] for r in chosen}))
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")
    key = {"video": video, "video_hash": vhash, "key": {tile_id(r, vhash): bool(r["noticeable"]) for r in chosen},
           "defects": {tile_id(r, vhash): r.get("defect", "") for r in chosen}}
    key_path.write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    return chosen


def agreement(marks: dict[str, Any], key: dict[str, Any]) -> dict[str, Any]:
    """Agreement of the person's marks with the audit's noticeable verdicts (unmarked tiles are not counted).
    Marks made on another video than the key's are refused."""
    if marks.get("video") != key.get("video"):
        raise CalibrationError(f"marks are for {marks.get('video')!r}, the key is for {key.get('video')!r}")
    k = key["key"]
    if not any(m.get("id") in k for m in marks.get("marks", [])):
        raise CalibrationError("no marked tile id is in the key (another render of the video?)")
    agree, disagree, unmarked = 0, [], []
    for m in marks.get("marks", []):
        if m.get("id") not in k:
            continue
        if m.get("mark") not in ("notice", "fine"):
            unmarked.append(m["id"])
            continue
        person = m["mark"] == "notice"
        if person == k[m["id"]]:
            agree += 1
        else:
            disagree.append({"id": m["id"], "t": m.get("t"), "person": m["mark"],
                             "audit": "notice" if k[m["id"]] else "fine", "comment": m.get("comment", ""),
                             "audit_defect": key.get("defects", {}).get(m["id"], "")})
    marked = agree + len(disagree)
    return {"agree": agree, "marked": marked, "total": len(k), "unmarked": unmarked, "disagreements": disagree,
            "trusted": marked == len(k) and agree >= round(TRUSTED_AT * len(k) / 40)}
