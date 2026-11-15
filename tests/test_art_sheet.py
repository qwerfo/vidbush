"""Contact sheet selection, numbering and yield (no network)."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from vidbush.art import sheet


def _img(path, colour):
    Image.fromarray(np.full((108, 192, 3), colour, np.uint8)).save(path)
    return str(path)


def test_pick12_keeps_four_of_each_and_orders_by_time():
    cands = [{"kind": "illustration", "score": 90 - i, "start": 100 - i, "path": "", "line": ""} for i in range(10)]
    cands += [{"kind": "illustrated map", "score": 50 - i, "start": 5 * i, "path": "", "line": ""} for i in range(6)]
    got = sheet.pick12(cands)
    assert len(got) == 12
    assert sum(c["kind"] == "illustrated map" for c in got) >= 4
    assert [c["start"] for c in got] == sorted(c["start"] for c in got)


def test_sheet_clips_and_yield(tmp_path):
    art = tmp_path / "art_project" / "work" / "art"
    art.mkdir(parents=True)
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x" * 4096)
    items = [
        {"type": "illustration", "ok": True, "still": _img(tmp_path / "a.png", 200), "start": 12, "window": "A line.",
         "score": 80, "report": {"variants": [{"score": 80}, {"score": 40, "text": ["AB"]}]}},
        {"type": "illustrated_map", "ok": True, "frame": _img(tmp_path / "m.png", 90), "start": 3, "window": "Map.",
         "report": {"chosen": "v0", "variants": [{"path": "v0", "ok": True, "registration": {"ok": True},
                                                  "appeal": {"appeal": 80, "fit": 70}},
                                                 {"path": "v1", "ok": False, "registration": {"ok": False}}]}},
        {"type": "clip", "kind": "illustration", "ok": True, "raw": "r", "final": str(clip), "start": 12, "window": "A"},
        {"type": "clip", "kind": "ai_image", "ok": False, "raw": "r", "check": {"reasons": ["morphing"]}, "start": 1},
    ]
    (art / "art_report.json").write_text(json.dumps(items), encoding="utf-8")
    (art / "art_report_attempt1.json").write_text(json.dumps(items * 3), encoding="utf-8")  # ignored
    sheet.main(["--review", str(tmp_path)])
    assert (tmp_path / "art_contact_sheet.png").exists()
    assert (tmp_path / "clip_01.mp4").exists() and not (tmp_path / "clip_02.mp4").exists()
    y = json.loads((tmp_path / "yield.json").read_text(encoding="utf-8"))["yield"]
    assert y["maps"] == {"views": 1, "paintings": 2, "passed_registration": 1, "passed_all": 1, "views_with_a_base": 1}
    assert y["illustrations"]["text_found"] == 1 and y["illustrations"]["passed_judge"] == 1
    assert y["clips"] == {"generated": 2, "passed": 1, "failed_reasons": ["morphing"]}
    assert "0:12" in (tmp_path / "clips.md").read_text(encoding="utf-8")
