"""Tile audit (vidbush/audit): offline, with a fake vision seat and a fake frame extractor."""
from __future__ import annotations

import html.parser
import json
from pathlib import Path

from vidbush.audit import auditor, calibrate, core

WORDS = [("Alaska", 0.0, 0.5), ("is", 0.6, 0.8), ("big.", 0.9, 1.4), ("Texas", 2.5, 3.0), ("is", 3.1, 3.3),
         ("smaller.", 3.4, 4.0), ("Maps", 9.0, 9.5), ("lie.", 9.6, 10.0)]


def test_word_window_now_before_after() -> None:
    w = core.tile_words(WORDS, 2.0)  # window [1, 4]
    assert w["words"] == "big. Texas is smaller."
    assert w["before"] == "Alaska is"
    assert w["after"] == "Maps lie."  # starts at 9.0, inside the context after 4.0
    assert core.tile_words(WORDS, 0.0)["before"] == ""
    assert core.tile_words(WORDS, 14.0) == {"words": "", "before": "is smaller. Maps lie.", "after": ""}


def test_both_timing_formats(tmp_path: Path) -> None:
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "narration.json").write_text(json.dumps(
        {"words": [{"text": "Alaska", "start": 0.0, "end": 0.56}, {"text": "is", "start": 0.64, "end": 0.72}]}))
    tr = tmp_path / "transcript.json"
    tr.write_text(json.dumps({"words": [{"word": " is", "startMs": 640, "endMs": 720},
                                        {"word": "Alaska", "startMs": 0, "endMs": 560}]}))
    assert core.load_words(tmp_path) == core.load_words(tr) == [("Alaska", 0.0, 0.56), ("is", 0.64, 0.72)]


def test_tile_times_and_spread() -> None:
    assert core.tile_times(7.0) == [0, 2, 4]  # the frame at 7.0 would be past the end
    assert core.spread(list(range(10)), 3) == [0, 3, 6]
    assert core.spread([1, 2], 5) == [1, 2]


def _row(t: int, kind: str = "real_footage", match: str = "on_words", noticeable: bool = False,
         overlay: str = "", defect: str = "") -> dict:
    return {"t": t, "kind": kind, "match": match, "noticeable": noticeable, "overlay": overlay, "defect": defect}


def test_summary_math() -> None:
    rows = [_row(0), _row(2, "ai_image", "early", True, "LABEL"), _row(4, "map", "related", False, "X"),
            _row(6, "ai_video", "wrong", True), _row(8, "document"), {"t": 10, "error": "timeout"}]
    s = core.summarize(rows)
    assert s["tiles"] == 5 and s["overlay_tiles"] == 2
    assert s["real_share"] == 40.0 and s["ai_share"] == 40.0 and s["designed_share"] == 20.0
    assert s["on_words"] == 40.0 and s["noticeable"] == 40.0 and s["overlay_noticeable"] == 50.0
    assert core.summarize([])["noticeable"] == 0.0


def test_worst_stretches_grouping() -> None:
    rows = [_row(0, noticeable=True, defect="a"), _row(2, noticeable=True, defect="a"), _row(4),
            _row(6, noticeable=True, defect="b"), _row(8, noticeable=True, defect="c"),
            _row(10, noticeable=True, defect="b"), _row(20, noticeable=True)]
    w = core.worst_stretches(rows)
    assert [(x["start"], x["tiles"]) for x in w] == [(6, 3), (0, 2), (20, 1)]
    assert w[0]["range"] == "0:06-0:12" and w[0]["defects"] == ["b", "c"]
    assert core.worst_stretches(rows, top=1)[0]["start"] == 6


def test_compare_joins_on_t() -> None:
    audit = [_row(0), _row(2, noticeable=True), _row(4, match="generic")]
    ref = [_row(0), _row(2), _row(6)]
    res = core.compare(audit, ref)
    assert res["agreement"]["tiles"] == 2
    assert res["agreement"]["noticeable"] == 50.0 and res["agreement"]["match"] == 100.0
    assert res["diff"]["noticeable"] == 50.0 and res["agreement"]["noticeable_audit_only"] == 1


def _fake_extract(video: Path, t: float, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"\xff\xd8\xff\xd9")
    return out


def test_run_audit_with_fake_seat(tmp_path: Path) -> None:
    calls = []

    def vision(system: str, user: str, image: str) -> dict:
        calls.append(user)
        if "0:01" in user:
            return {"bad": 1}  # unusable twice: recorded as an error row
        return {"kind": "AI_IMAGE", "match": "late", "noticeable": "true", "shows": "x", "overlay": "",
                "overlay_true": True, "defect": "d"}

    shots = [{"id": "shot_001", "type": "footage", "start": 0.0, "end": 2.5},
             {"id": "shot_002", "type": "image", "start": 2.5, "end": 9.0}]
    rows = auditor.run_audit(Path("v.mp4"), WORDS, tmp_path, vision, 6.0, shots=shots, workers=2,
                             extract=_fake_extract, log=lambda m: None)
    assert [r["t"] for r in rows] == [0, 2, 4]
    assert rows[0]["error"] and rows[0]["shot_id"] == "shot_001"
    assert rows[1]["kind"] == "ai_image" and rows[1]["noticeable"] is True and rows[1]["overlay_true"] is None
    assert rows[1]["shot_id"] == "shot_002" and rows[1]["frame_t"] == 3
    assert "Texas is smaller." in calls[-1] or any("Texas" in c for c in calls)
    assert len(calls) == 4
    # a rerun keeps valid rows from the same prompt and only retries the failed tile
    calls.clear()
    auditor.run_audit(Path("v.mp4"), WORDS, tmp_path, vision, 6.0, extract=_fake_extract, log=lambda m: None)
    assert len(calls) == 2
    summary = auditor.write_reports(rows, tmp_path, Path("v.mp4"), "fake/model", 0.01)
    assert set(core.SUMMARY_KEYS) <= set(summary) and summary["tiles"] == 2 and summary["ai_share"] == 100.0
    assert "Ten worst stretches" in (tmp_path / "audit.md").read_text(encoding="utf-8")


def test_mock_reply_is_an_error(tmp_path: Path) -> None:
    row = auditor.audit_tile(lambda s, u, i: {"score": 50, "_mock": True},
                             {"t": 0, "frame_t": 1, "words": "", "before": "", "after": ""}, tmp_path / "f.jpg")
    assert "mock" in row["error"]


def test_prompt_is_general() -> None:
    low = auditor.SYSTEM.lower()
    for word in ("alaska", "texas", "nome", "sled", "balto", "vidrush"):
        assert word not in low


class _Parser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.imgs, self.buttons, self.textareas, self.script = 0, 0, 0, ""
        self._in_script = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self.imgs += tag == "img"
        self.buttons += tag == "button"
        self.textareas += tag == "textarea"
        self._in_script = tag == "script"

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self.script += data


def test_calibration_sheet_and_agreement(tmp_path: Path) -> None:
    rows = [{**_row(t, noticeable=t % 4 == 0, defect="secret defect"), "frame_t": t + 1, "words": "w",
             "before": "", "after": ""} for t in range(0, 200, 2)]
    tiles = tmp_path / "tiles"
    for r in rows:
        _fake_extract(Path(""), 0, tiles / f"{calibrate.tile_id(r)}.jpg")
    chosen = calibrate.build_sheet(rows, tiles, "v.mp4", tmp_path / "c.html", tmp_path / "k.json")
    assert len(chosen) == 40 and sum(r["noticeable"] for r in chosen) == 20
    page = (tmp_path / "c.html").read_text(encoding="utf-8")
    assert "secret defect" not in page and "noticeable" not in page.lower()
    p = _Parser()
    p.feed(page)
    assert p.imgs == 40 and p.textareas == 40 and p.buttons == 81
    assert "function exportMarks" in p.script and "new Blob" in p.script
    key = json.loads((tmp_path / "k.json").read_text(encoding="utf-8"))
    marks = {"video": "v.mp4", "marks": [{"id": i, "t": 0, "mark": "notice" if v else "fine", "comment": ""}
                                         for i, v in key["key"].items()]}
    for m in marks["marks"][:5]:
        m["mark"] = "fine" if m["mark"] == "notice" else "notice"
    res = calibrate.agreement(marks, key)
    assert res["agree"] == 35 and res["trusted"] and len(res["disagreements"]) == 5
    marks["marks"][5]["mark"] = "fine" if marks["marks"][5]["mark"] == "notice" else "notice"
    marks["marks"][6]["mark"] = None
    res = calibrate.agreement(marks, key)
    assert res["agree"] == 33 and not res["trusted"] and res["unmarked"] == [marks["marks"][6]["id"]]


def test_shot_list_provenance_sets_ai_kind(tmp_path: Path) -> None:
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "shotlist.json").write_text(json.dumps({"shots": [
        {"id": "s1", "type": "footage", "start": 0, "end": 3, "selected": "c1",
         "candidates": [{"id": "c0", "source": "pexels"}, {"id": "c1", "source": "ai_image"}]},
        {"id": "s2", "type": "footage", "start": 3, "end": 9, "selected": "c2",
         "candidates": [{"id": "c2", "source": "youtube"}]}]}))
    shots = core.load_shots(tmp_path)
    assert [s["source"] for s in shots] == ["ai_image", "youtube"]
    users = []

    def vision(system: str, user: str, image: str) -> dict:
        users.append(user)
        return {"kind": "real_footage", "match": "on_words", "noticeable": False}

    rows = auditor.run_audit(Path("v.mp4"), WORDS, tmp_path, vision, 6.0, shots=shots, workers=1,
                             extract=_fake_extract, log=lambda m: None)
    assert rows[0]["kind"] == "ai_image" and rows[0]["kind_seen"] == "real_footage"
    assert rows[1]["kind"] == "real_footage" and "kind_seen" not in rows[1]
    assert sum("image generator" in u for u in users) == 1
