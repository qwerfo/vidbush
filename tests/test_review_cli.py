"""Review subcommands: review-set, review-comment, review-summary, review-export, review-fixes, review-calibrate."""
from __future__ import annotations

import io
import json

import pytest

from review_fixture import PID, build_review_project
from vidbush import cli
from vidbush.models import EditResult
from vidbush.review import store
from vidbush.util import read_json


@pytest.fixture
def root(settings):
    return build_review_project(settings)


@pytest.fixture
def run_cli(settings, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    def _run(*argv: str) -> tuple[int, str, str]:
        code = cli.main(list(argv))
        cap = capsys.readouterr()
        return code, cap.out, cap.err
    return _run


def _entry(settings, n: int) -> store.ShotReview:
    return store.load(settings, PID).current()[n - 1]


def test_review_set(run_cli, settings, root):
    code, out, _ = run_cli("review-set", PID, "2", "--verdict", "quality_problem", "--comment", "Logo top right",
                           "--reject", "watermark_logo", "--reject", "burned_in_text")
    assert code == 0
    assert out.splitlines() == ["shot 2 (shot_002): quality_problem, rejects burned_in_text, watermark_logo, "
                                "comment 14 chars", "reviewed 1 of 8"]
    e = _entry(settings, 2)
    assert e.verdict == "quality_problem" and e.rejects == ["burned_in_text", "watermark_logo"]
    assert e.comment == "Logo top right"
    code, out, _ = run_cli("review-set", PID, "shot_002", "--clear-rejects")
    assert code == 0 and _entry(settings, 2).rejects == [] and _entry(settings, 2).comment == "Logo top right"
    code, out, _ = run_cli("review-set", PID, "2", "--verdict", "none")
    assert code == 0 and _entry(settings, 2).verdict is None and "reviewed 0 of 8" in out


def test_review_set_comment_from_stdin(run_cli, settings, root, monkeypatch):
    text = "Line one.\n" + "y" * 20000 + "\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(text))
    code, _, _ = run_cli("review-set", PID, "3", "--comment", "-")
    assert code == 0 and _entry(settings, 3).comment == text


def test_review_set_errors(run_cli, settings, root):
    code, _, err = run_cli("review-set", PID, "3")
    assert code == 2 and "nothing to change" in err
    code, _, err = run_cli("review-set", PID, "42", "--verdict", "good")
    assert code == 2 and "out of range" in err
    with pytest.raises(SystemExit) as ex:
        run_cli("review-set", PID, "1", "--verdict", "awful")
    assert ex.value.code == 2


def test_review_comment(run_cli, settings, root):
    code, out, _ = run_cli("review-comment", PID, "Music too loud at 2:10.")
    assert code == 0 and out.strip() == "project comment saved (23 chars)"
    assert read_json(root / "review.json")["project_comment"] == "Music too loud at 2:10."


def test_review_summary(run_cli, settings, root):
    run_cli("review-set", PID, "1", "--verdict", "good")
    run_cli("review-set", PID, "4", "--verdict", "quality_problem", "--reject", "anachronism")
    code, out, _ = run_cli("review-summary", PID)
    assert code == 0
    assert out.splitlines()[0] == f"Review of {PID}: reviewed 2 of 8 shots"
    assert "  shot 4 at 0:12.5: anachronism" in out
    code, out, _ = run_cli("review-summary", PID, "--json")
    d = json.loads(out)
    assert code == 0 and d["reviewed"] == 2 and d["verdicts"]["quality_problem"] == 1
    assert d["flagged"][0]["number"] == 4


def test_review_export(run_cli, settings, root):
    run_cli("review-set", PID, "3", "--verdict", "wrong_picture", "--comment", "No tents.")
    code, out, _ = run_cli("review-export", PID)
    md = root / "review.md"
    assert code == 0 and md.exists() and f"wrote {md}" in out
    assert "> No tents." in md.read_text(encoding="utf-8")
    md.unlink()
    code, out, _ = run_cli("review-export", PID, "--json")
    d = json.loads(out)
    assert code == 0 and d["path"] == str(md) and md.exists() and d["comments"] == 1


def test_review_fixes_plan_only(run_cli, settings, root, monkeypatch):
    called = []
    monkeypatch.setattr("vidbush.editor.ops.apply_and_render", lambda *a, **k: called.append(a))
    run_cli("review-set", PID, "1", "--verdict", "wrong_picture", "--comment", "Too modern. Show 1931 walls.")
    run_cli("review-set", PID, "4", "--verdict", "wrong_timing")
    code, out, _ = run_cli("review-fixes", PID)
    assert code == 0 and not called
    assert out.startswith("Planned edit ops: 1")
    assert "change_type shot_id=shot_001, type=footage" in out
    assert "shot 4 at 12.5 s, now 6.00 s" in out
    assert _entry(settings, 1).fix is None  # nothing marked without --apply
    code, out, _ = run_cli("review-fixes", PID, "--json")
    d = json.loads(out)
    assert code == 0 and [o["op"] for o in d["ops"]] == ["change_type"] and d["timing"][0]["number"] == 4


def test_review_fixes_apply(run_cli, settings, root, monkeypatch):
    seen = []

    def fake(settings, project_id, ops, echo=True, job_id=None):
        seen.append([o.model_dump(mode="json") for o in ops])
        return EditResult(ops=ops, applied=["change_type: ok"])
    monkeypatch.setattr("vidbush.editor.ops.apply_and_render", fake)
    run_cli("review-set", PID, "1", "--verdict", "wrong_picture", "--comment", "x")
    code, out, _ = run_cli("review-fixes", PID, "--apply", "--quiet")
    assert code == 0 and len(seen) == 1 and seen[0][0]["args"]["shot_id"] == "shot_001"
    assert "applied change_type: ok" in out
    assert _entry(settings, 1).fix.op == "change_type"
    code, out, _ = run_cli("review-fixes", PID, "--apply")
    assert code == 0 and len(seen) == 1 and "Already queued: 1" in out


def test_review_calibrate(run_cli, settings, root):
    run_cli("review-set", PID, "1", "--verdict", "wrong_picture")
    run_cli("review-set", PID, "3", "--verdict", "good")
    code, out, _ = run_cli("review-calibrate", PID)
    assert code == 0 and out.startswith(f"Judge calibration for {PID} (judge.threshold 70)")
    assert "Suggested judge.threshold:" in out
    code, out, _ = run_cli("review-calibrate", PID, "--json")
    d = json.loads(out)
    assert code == 0 and d["labelled"] == 2 and d["wrong_above_threshold"][0]["number"] == 1


@pytest.mark.parametrize("argv", [("review-set", "nope", "1", "--verdict", "good"), ("review-comment", "nope", "x"),
                                  ("review-summary", "nope"), ("review-export", "nope"), ("review-fixes", "nope"),
                                  ("review-calibrate", "nope")])
def test_unknown_project_exit_2(run_cli, root, argv):
    code, _, err = run_cli(*argv)
    assert code == 2 and "project not found: nope" in err


def test_no_shotlist_exit_2(run_cli, settings, tmp_path):
    from vidbush.jobs.pipeline import create_project
    from vidbush.models import InputMode
    s = tmp_path / "s.md"
    s.write_text("# T\n\nText.")
    create_project(settings, "bare", InputMode.script, script_path=str(s), seed=1)
    code, _, err = run_cli("review-set", "bare", "1", "--verdict", "good")
    assert code == 2 and "no shot list" in err
    code, _, err = run_cli("review-fixes", "bare")
    assert code == 2 and "no shot list" in err
