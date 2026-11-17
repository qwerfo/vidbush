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
    empty = core.summarize([_row(0)])  # no overlay tile: the overlay share is unknown, not 0
    assert core.summarize([])["noticeable"] is None and empty["overlay_noticeable"] is None


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
        _fake_extract(Path(""), 0, tiles / core.frame_name(r["t"]))
    chosen = calibrate.build_sheet(rows, tiles, "v.mp4", "abcd1234", tmp_path / "c.html", tmp_path / "k.json")
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


GOOD = {"kind": "real_footage", "match": "on_words", "noticeable": False, "shows": "x", "overlay": "", "defect": ""}


def _good(system: str, user: str, image: str) -> dict:
    return dict(GOOD)


def test_failed_frame_extraction_is_one_error_row(tmp_path: Path) -> None:
    from vidbush.util import ToolError

    def extract(video: Path, t: float, out: Path) -> Path:
        if t == 3:
            raise ToolError("ffmpeg failed (1): https://host/x.mp4?sig=SECRET123 token=abc")
        return _fake_extract(video, t, out)

    rows = auditor.run_audit(Path("v.mp4"), WORDS, tmp_path, _good, 6.0, workers=2, extract=extract,
                             log=lambda m: None)
    assert [bool(r.get("error")) for r in rows] == [False, True, False]
    err = rows[1]["error"]
    assert err.startswith("ToolError: ") and "https://host/x.mp4" in err
    assert "SECRET123" not in err and "abc" not in err
    assert len(core.latest_rows(tmp_path / "audit.jsonl")) == 3


def test_rows_are_on_disk_before_a_crash(tmp_path: Path) -> None:
    calls = []

    def vision(system: str, user: str, image: str) -> dict:
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt  # not an Exception: the run stops here
        return dict(GOOD)

    try:
        auditor.run_audit(Path("v.mp4"), WORDS, tmp_path, vision, 10.0, workers=1, extract=_fake_extract,
                          log=lambda m: None)
    except KeyboardInterrupt:
        pass
    assert [r["t"] for r in core.latest_rows(tmp_path / "audit.jsonl")] == [0, 2]  # paid verdicts survived
    again = []

    def counting(system: str, user: str, image: str) -> dict:
        again.append(1)
        return dict(GOOD)

    auditor.run_audit(Path("v.mp4"), WORDS, tmp_path, counting, 10.0, workers=1, extract=_fake_extract,
                      log=lambda m: None)
    assert len(again) == 3  # only the three tiles never audited
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["t"] for x in lines] == [0, 2, 4, 6, 8]  # compacted, one row per tile


def test_torn_last_line_is_skipped(tmp_path: Path) -> None:
    p = tmp_path / "audit.jsonl"
    p.write_text(json.dumps(_row(0)) + "\n" + json.dumps(_row(0, noticeable=True)) + "\n{\"t\": 2, \"ki",
                 encoding="utf-8")
    assert core.latest_rows(p) == [_row(0, noticeable=True)]


def test_cache_key_covers_model_context_source_and_video(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"one")
    calls = []

    def vision(system: str, user: str, image: str) -> dict:
        calls.append(user)
        return dict(GOOD)

    more = WORDS + [("Later.", 13.0, 13.5)]

    def go(words=WORDS, model="m1", shots=None, extract=_fake_extract) -> int:
        calls.clear()
        auditor.run_audit(video, words, tmp_path / "out", vision, 6.0, shots=shots, workers=1,
                          extract=extract, model=model, log=lambda m: None)
        return len(calls)

    assert go() == 3
    assert go() == 0
    assert go(model="m2") == 3
    assert go(model="m2", words=more) == 2  # only the AFTER words of tiles 2 and 4 changed
    ai = [{"id": "s", "type": "image", "start": 0.0, "end": 2.0, "source": "ai_image"}]
    assert go(model="m2", words=more, shots=ai) == 1  # the source line of tile 0 changed
    assert (tmp_path / "out" / "tiles" / core.frame_name(0)).exists()
    video.write_bytes(b"a re-render")  # new identity: every tile again, the old frames dropped first
    seen = []

    def extract(v: Path, t: float, out: Path) -> Path:
        seen.append(out.exists())
        return _fake_extract(v, t, out)

    assert go(model="m2", words=more, shots=ai, extract=extract) == 3
    assert seen == [False, False, False]


def test_frame_names_do_not_collide_below_one_second() -> None:
    times = core.tile_times(3.0, 0.5)
    assert len({core.frame_name(t) for t in times}) == len(times) and core.frame_name(1.5) == "t00001500.jpg"
    assert calibrate.tile_id({"t": 0.5}, "ab") != calibrate.tile_id({"t": 0}, "ab")


def test_designed_kind_survives_the_ai_provenance_override(tmp_path: Path) -> None:
    tile = {"t": 0, "frame_t": 1, "words": "", "before": "", "after": "", "source": "ai_image"}
    for seen, want in (("map", "map"), ("designed_card", "designed_card"), ("other", "ai_image"),
                       ("real_still", "ai_image"), ("ai_video", "ai_video")):
        row = auditor.audit_tile(lambda s, u, i, k=seen: {**GOOD, "kind": k}, tile, tmp_path / "f.jpg")
        assert row["kind"] == want


def test_shift_moves_words_and_shots() -> None:
    assert core.shift([("a", 0.0, 1.0)], 5.0) == [("a", 5.0, 6.0)]
    assert core.shift([{"id": "s", "start": 0.0, "end": 2.0}], 5.0)[0]["start"] == 5.0


def test_calibration_refuses_marks_of_another_video() -> None:
    import pytest
    key = {"video": "a.mp4", "key": {"h1-t00000000": True}}
    with pytest.raises(calibrate.CalibrationError):
        calibrate.agreement({"video": "b.mp4", "marks": [{"id": "h1-t00000000", "mark": "notice"}]}, key)
    with pytest.raises(calibrate.CalibrationError):  # same name, another render: ids carry the video hash
        calibrate.agreement({"video": "a.mp4", "marks": [{"id": "h2-t00000000", "mark": "notice"}]}, key)


class _Settings:
    def __init__(self, root: Path) -> None:
        self.output_folder = str(root / "output")

    def path(self, p: str) -> Path:
        return Path(p)


class _FakeSeats:
    def __init__(self, settings: object, tracker: object) -> None:
        pass

    def providers_for(self, role: str) -> list:
        return []


def _patch_cli(monkeypatch, duration: float, vision) -> None:
    import vidbush.util
    from vidbush.providers import seats as seats_mod
    monkeypatch.setattr(seats_mod, "Seats", _FakeSeats)
    monkeypatch.setattr(auditor, "seats_vision", lambda seats: vision)
    monkeypatch.setattr(vidbush.util, "media_duration", lambda v: duration)
    monkeypatch.setattr(auditor, "extract_frame", _fake_extract)


def test_cli_audit_reports_all_rows_and_own_cost_file(tmp_path: Path, monkeypatch) -> None:
    from argparse import Namespace
    from vidbush.audit import cli

    root = tmp_path / "output" / "proj"
    (root / "work").mkdir(parents=True)
    (root / "final.mp4").write_bytes(b"v")
    (root / "work" / "narration.json").write_text(json.dumps({"words": [
        {"text": w, "start": a, "end": b} for w, a, b in WORDS]}))
    (root / "work" / "assembly.json").write_text(json.dumps({"intro_duration": 4.0}))
    (root / "work" / "shotlist.json").write_text(json.dumps({"shots": [
        {"id": "s1", "type": "image", "start": 0.0, "end": 3.0},
        {"id": "s2", "type": "footage", "start": 3.0, "end": 30.0}]}))
    _patch_cli(monkeypatch, 12.0, _good)
    args = Namespace(target=str(root), script=None, out=None, every=2.0, limit=2, workers=1)
    assert cli.cmd_audit(_Settings(tmp_path), args) == 0
    out = root / "audit"
    assert (out / "audit_cost_report.json").exists() and not (out / "cost_report.json").exists()
    args.limit = 0
    assert cli.cmd_audit(_Settings(tmp_path), args) == 0
    tile4 = next(r for r in core.latest_rows(out / "audit.jsonl") if r["t"] == 4)
    # the 4 s intro shifts words and shots: the frame at 5 s is body time 1 s, in shot s1
    assert tile4["shot_id"] == "s1" and tile4["words"].startswith("Alaska")
    args.limit = 1  # a slice rerun after the full audit: the reports still cover every tile
    assert cli.cmd_audit(_Settings(tmp_path), args) == 0
    assert json.loads((out / "audit_summary.json").read_text())["tiles"] == 6


def test_cli_audit_with_no_valid_tile_fails_without_summary(tmp_path: Path, monkeypatch) -> None:
    from argparse import Namespace
    from vidbush.audit import cli

    video = tmp_path / "v.mp4"
    video.write_bytes(b"v")
    tr = tmp_path / "tr.json"
    tr.write_text(json.dumps({"words": [{"word": "a", "startMs": 0, "endMs": 500}]}))
    _patch_cli(monkeypatch, 6.0, lambda s, u, i: {"_mock": True})
    out = tmp_path / "out"
    args = Namespace(target=str(video), script=str(tr), out=str(out), every=2.0, limit=0, workers=1)
    assert cli.cmd_audit(_Settings(tmp_path), args) == 1
    assert not (out / "audit_summary.json").exists() and not (out / "audit.md").exists()


def test_cli_calibrate_finds_the_key_by_video(tmp_path: Path) -> None:
    from argparse import Namespace
    from vidbush.audit import cli
    aroot = tmp_path / "audit"
    aroot.mkdir()
    (aroot / "calibration_a_key.json").write_text(json.dumps({"video": "a.mp4", "key": {"h-t00000000": True}}))
    (aroot / "calibration_b_key.json").write_text(json.dumps({"video": "b.mp4", "key": {"h-t00000000": False}}))
    marks = tmp_path / "m.json"
    marks.write_text(json.dumps({"video": "b.mp4", "marks": [{"id": "h-t00000000", "t": 0, "mark": "fine",
                                                               "comment": ""}]}))
    assert cli.cmd_audit_calibrate(_Settings(tmp_path), Namespace(marks=str(marks), key=None)) == 0
    marks.write_text(json.dumps({"video": "c.mp4", "marks": []}))
    assert cli.cmd_audit_calibrate(_Settings(tmp_path), Namespace(marks=str(marks), key=None)) == 2
