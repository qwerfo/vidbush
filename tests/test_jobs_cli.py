"""CLI: subcommands for every UI action, raw ops, listings, and run overrides reaching the brief."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vidbush import cli
from vidbush.jobs.pipeline import create_project, load_project
from vidbush.models import EditResult, InputMode, VideoFormat


@pytest.fixture
def run_cli(settings, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    def _run(*argv: str) -> tuple[int, str]:
        code = cli.main(list(argv))
        return code, capsys.readouterr().out
    return _run


@pytest.fixture
def seen_ops(monkeypatch):
    seen: list = []

    def fake(settings, project_id, ops, echo=True, job_id=None):
        seen.append((project_id, [o.model_dump(mode="json") for o in ops]))
        return EditResult(ops=ops, applied=["ok"])
    monkeypatch.setattr("vidbush.editor.ops.apply_and_render", fake)
    return seen


def test_parse_op_args():
    assert cli.parse_op_args(["shot_id=7", "factor=0.7", "range=[0,30]", "track=undertow.ogg"]) == \
        {"shot_id": 7, "factor": 0.7, "range": [0, 30], "track": "undertow.ogg"}
    with pytest.raises(ValueError):
        cli.parse_op_args(["nokey"])


def test_shot_shortcuts_map_to_ui_ops(run_cli, seen_ops, tmp_path):
    img = tmp_path / "mine.png"
    img.write_bytes(b"x")
    assert run_cli("regenerate", "demo", "7", "--prompt", "at dusk", "--quiet")[0] == 0
    assert run_cli("duration", "demo", "shot_003", "5", "--quiet")[0] == 0
    assert run_cli("remove-overlay", "demo", "4", "--quiet")[0] == 0
    assert run_cli("upload", "demo", "2", str(img), "--quiet")[0] == 0
    got = [(o[0]["op"], o[0]["args"]) for _, o in seen_ops]
    assert got == [("regenerate_ai", {"shot_id": "7", "prompt_delta": "at dusk"}),
                   ("set_duration", {"shot_id": "shot_003", "seconds": 5.0}),
                   ("remove_overlay", {"shot_id": "4"}),
                   ("upload_shot", {"shot_id": "2", "path": str(img)})]


def test_op_and_ops_commands(run_cli, seen_ops, tmp_path):
    assert run_cli("op", "demo", "retime_section", "range=[0,30]", "factor=0.7", "--quiet")[0] == 0
    assert seen_ops[-1][1][0] == {"op": "retime_section", "args": {"range": [0, 30], "factor": 0.7},
                                  "reason": "cli"}
    f = tmp_path / "ops.json"
    f.write_text(json.dumps({"ops": [{"op": "swap_shot", "args": {"shot_id": 1, "candidate_id": "next"}},
                                     {"op": "change_music", "args": {"track": "auto"}}]}))
    assert run_cli("ops", "demo", str(f), "--quiet")[0] == 0
    assert [o["op"] for o in seen_ops[-1][1]] == ["swap_shot", "change_music"]
    code, out = run_cli("op", "demo", "swap_shot", "shot_id=1", "--dry-run")
    assert code == 0 and len(seen_ops) == 2 and json.loads(out)[0]["op"] == "swap_shot"
    assert run_cli("op", "demo", "not_an_op")[0] == 2


def test_listings(run_cli, settings, tmp_path):
    script = tmp_path / "s.md"
    script.write_text("# T\n\nHello.")
    create_project(settings, "lp", InputMode.script, script_path=str(script))
    code, out = run_cli("projects")
    assert code == 0 and [r["id"] for r in json.loads(out)] == ["lp"]
    code, out = run_cli("brands")
    data = json.loads(out)
    assert code == 0 and any(b["id"] == "default" for b in data["brands"]) and "top10_default" in data["style_profiles"]
    code, out = run_cli("seats")
    data = json.loads(out)
    assert code == 0 and "writer" in data["seats"] and data["offline"] is True and "down" in data


def test_run_format_and_minutes_reach_the_brief(run_cli, settings, tmp_path, monkeypatch):
    brief = tmp_path / "wonders.md"
    brief.write_text("---\ntopic: Ancient Wonders\ntarget_minutes: 8\nformat: documentary\n---\nNotes")
    monkeypatch.setattr("vidbush.jobs.pipeline.run_project",
                        lambda s, pid, **k: load_project(s, pid))
    run_cli("run", str(brief), "--format", "top10", "--minutes", "12", "--quiet")
    proj = load_project(settings, "wonders")
    assert proj.brief.format == VideoFormat.top10 and proj.brief.target_minutes == 12
    assert proj.format == VideoFormat.top10 and Path(proj.paths.root).name == "wonders"
