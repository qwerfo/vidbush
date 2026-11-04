"""Batch queue: discovery, cost cap, report, pause and resume, CLI (pipeline stages stubbed)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vidbush import cli
from vidbush.jobs import batch, pipeline
from vidbush.jobs.batch_report import report_markdown
from vidbush.jobs.costcap import read_cap
from vidbush.jobs.pipeline import create_project, load_project, run_project
from vidbush.models import STAGES, CostEntry, InputMode, StageStatus
from vidbush.util import write_json

BRIEF = "---\ntopic: The Erie Canal\ntarget_minutes: 5\n---\n"


def _stages(calls: list, cost: dict | None = None, fail: set | None = None, qa: float = 92.5):
    """Stub stage map. cost: stage -> usd added to the tracker; fail: project ids whose shotlist raises."""
    def make(name):
        def fn(ctx):
            calls.append((ctx.project.id, name))
            if fail and ctx.project.id in fail and name == "shotlist":
                raise RuntimeError("boom in shotlist")
            if cost and name in cost:
                ctx.tracker.add(CostEntry(role="tts", provider="stub", stage=name, cost_usd=cost[name]))
            p = ctx.work / f"{name}.done"
            p.write_text("ok")
            if name == "packaging":
                write_json(ctx.out_path("shotlist.json"), {"duration": 120.0})
                write_json(ctx.out_path("qa_report.json"), {"pct_above_threshold": qa, "mock_used": []})
            return [str(p)]
        return fn
    return {n: make(n) for n in STAGES}


@pytest.fixture
def stub_runs(monkeypatch):
    """Every run_project call uses stub stages; returns (calls, config dict to set cost and fail)."""
    calls: list = []
    conf: dict = {"cost": None, "fail": None, "costs": {}}
    real = pipeline.run_project

    def fake(settings, project_id, **kw):
        cost = conf["costs"].get(project_id, conf["cost"])
        return real(settings, project_id, stages=_stages(calls, cost, conf["fail"]), **kw)
    monkeypatch.setattr(pipeline, "run_project", fake)
    return calls, conf


def _folder(tmp_path: Path) -> Path:
    d = tmp_path / "in"
    d.mkdir()
    (d / "a_brief.md").write_text(BRIEF)
    (d / "b_script.md").write_text("# Hoover Dam\n\nConcrete in the desert.")
    (d / "c_notes.txt").write_text("A plain text script.")
    (d / "d_voice.mp3").write_bytes(b"ID3")
    (d / "d_voice.md").write_text("# Sidecar\n\nThe script of the voiceover.")
    (d / "e_image.png").write_bytes(b"x")
    (d / ".hidden.md").write_text("x")
    return d


def test_discover_folder_modes_and_sidecar(tmp_path):
    items = batch.plan_items(_folder(tmp_path))
    assert [(Path(i.input).name, i.mode) for i in items] == [
        ("a_brief.md", "brief"), ("b_script.md", "script"), ("c_notes.txt", "script"), ("d_voice.mp3", "voiceover")]
    assert [i.project_id for i in items] == ["a_brief", "b_script", "c_notes", "d_voice"]
    assert all(i.status == "queued" for i in items)


def test_discover_list_file(tmp_path):
    d = _folder(tmp_path)
    lst = tmp_path / "queue.txt"
    lst.write_text("# tonight\n\nin/d_voice.mp3\nin/d_voice.md\n  in/a_brief.md  \nmissing.md\n"
                   f"{d / 'b_script.md'}\nin/e_image.png\n")
    items = batch.plan_items(lst, id_prefix="x-")
    assert [(Path(i.input).name, i.status) for i in items] == [
        ("d_voice.mp3", "queued"), ("a_brief.md", "queued"), ("missing.md", "skipped"),
        ("b_script.md", "queued"), ("e_image.png", "skipped")]
    assert items[0].project_id == "x-d_voice" and items[2].note == "file not found"


def test_project_ids_unique_within_batch(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    (d / "canal story.md").write_text("# A\n\nOne.")
    (d / "canal story.txt").write_text("Two.")
    items = batch.plan_items(d)
    assert [i.project_id for i in items] == ["canal_story", "canal_story_2"]


def _script_project(settings, tmp_path, pid="p1"):
    s = tmp_path / f"{pid}.md"
    s.write_text("# T\n\nHello world.")
    return create_project(settings, pid, InputMode.script, script_path=str(s), seed=1)


def test_cost_cap_stops_run_and_resume_finishes(settings, tmp_path):
    _script_project(settings, tmp_path)
    calls: list = []
    cost = {"narration": 3.0, "shotlist": 3.0}
    proj = run_project(settings, "p1", stages=_stages(calls, cost), echo=False, cost_cap=5.0)
    assert [c[1] for c in calls][-1] == "shotlist" and "sourcing" not in [c[1] for c in calls]
    assert proj.status == "needs_review" and proj.status_note.startswith("cost cap reached")
    on_disk = load_project(settings, "p1")
    assert on_disk.status == "needs_review" and "resume continues at sourcing" in on_disk.status_note
    assert on_disk.stages["sourcing"].status == StageStatus.pending
    assert read_cap(Path(on_disk.paths.work))["stage"] == "shotlist"
    log = (Path(settings.data_folder) / "logs" / "p1.jsonl").read_text()
    assert "cost cap reached" in log
    calls.clear()
    proj = run_project(settings, "p1", stages=_stages(calls), echo=False)  # resume without a cap
    assert [c[1] for c in calls][0] == "sourcing" and proj.status == "done"
    assert proj.status_note == "" and read_cap(Path(proj.paths.work)) is None


def test_no_cap_and_cap_not_crossed_run_to_the_end(settings, tmp_path):
    _script_project(settings, tmp_path)
    proj = run_project(settings, "p1", stages=_stages([], {"narration": 1.0}), echo=False, cost_cap=5.0)
    assert proj.status == "done"
    proj = run_project(settings, "p1", stages=_stages([], {"narration": 50.0}), force=["narration"], echo=False)
    assert proj.status == "done"


def test_batch_runs_all_records_failure_cap_and_report(settings, tmp_path, stub_runs):
    calls, conf = stub_runs
    conf["fail"] = {"b_script"}
    conf["costs"] = {"c_notes": {"narration": 4.0, "shotlist": 4.0}, "a_brief": {"narration": 0.5}}
    state = batch.create_batch(settings, _folder(tmp_path), brand="default", cost_cap=5.0, batch_id="b1")
    assert state.cost_cap_usd == 5.0 and state.status == "queued"
    state = batch.run_batch(settings, "b1", echo=False)
    by_id = {i.project_id: i for i in state.items}
    assert state.status == "done"
    assert by_id["a_brief"].status == "done" and by_id["d_voice"].status == "done"
    assert by_id["b_script"].status == "failed" and "boom in shotlist" in by_id["b_script"].note
    assert by_id["c_notes"].status == "capped" and "cost cap reached" in by_id["c_notes"].note
    assert load_project(settings, "c_notes").status == "needs_review"
    a = by_id["a_brief"]
    assert (a.duration_s, a.cost_usd, a.cost_per_min_usd, a.qa_pct) == (120.0, 0.5, 0.25, 92.5)
    folder = batch.batch_dir(settings, "b1")
    rep = json.loads((folder / "batch_report.json").read_text())
    assert [r["status"] for r in rep["rows"]] == ["done", "failed", "capped", "done"]
    assert rep["rows"][0]["mode"] == "brief" and rep["rows"][3]["mode"] == "voiceover"
    md = (folder / "batch_report.md").read_text()
    assert "| a_brief | a_brief.md | brief | done | 2:00 | $0.50 | $0.25 | 92.5% |" in md
    assert "cost cap reached" in md and "boom in shotlist" in md
    assert json.loads((folder / "state.json").read_text())["paused"] is False
    proj = load_project(settings, "d_voice")
    assert proj.input_mode == InputMode.voiceover and Path(proj.script_path).name == "d_voice.md"


def test_pause_between_projects_then_resume(settings, tmp_path, stub_runs):
    calls, _ = stub_runs
    batch.create_batch(settings, _folder(tmp_path), batch_id="b2")

    def pause_after_first(state, item):
        if item.index == 0:
            batch.pause_batch(settings, "b2")
    state = batch.run_batch(settings, "b2", echo=False, on_item=pause_after_first)
    assert state.status == "paused" and state.paused
    assert [i.status for i in state.items] == ["done", "queued", "queued", "queued"]
    assert {c[0] for c in calls} == {"a_brief"}
    state = batch.resume_batch(settings, "b2", echo=False)
    assert state.status == "done" and not state.paused
    assert all(i.status == "done" for i in state.items)
    assert [c[0] for c in calls if c[1] == "intake"] == ["a_brief", "b_script", "c_notes", "d_voice"]


def test_interrupted_item_resumes_its_project(settings, tmp_path, monkeypatch):
    real = pipeline.run_project
    calls: list = []
    state = {"crash": True}

    def fake(settings_, project_id, **kw):
        stages = _stages(calls)
        if state["crash"]:
            def crash(ctx):
                raise KeyboardInterrupt
            stages["sourcing"] = crash
        return real(settings_, project_id, stages=stages, **kw)
    monkeypatch.setattr(pipeline, "run_project", fake)
    d = tmp_path / "one"
    d.mkdir()
    (d / "x.md").write_text("# X\n\nOne line.")
    batch.create_batch(settings, d, batch_id="b3")
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(settings, "b3", echo=False)
    st = batch.load_batch(settings, "b3")
    assert st.items[0].status == "running" and batch.effective_status(settings, st) == "interrupted"
    created = load_project(settings, "x").created_at
    state["crash"], calls[:] = False, []
    st = batch.resume_batch(settings, "b3", echo=False)
    assert st.items[0].status == "done" and load_project(settings, "x").created_at == created
    assert [c[1] for c in calls][0] == "sourcing"


def test_retry_failed_with_higher_cap(settings, tmp_path, stub_runs):
    _, conf = stub_runs
    d = tmp_path / "one"
    d.mkdir()
    (d / "x.md").write_text("# X\n\nOne line.")
    conf["cost"] = {"narration": 6.0}
    st = batch.run_batch(settings, batch.create_batch(settings, d, batch_id="b4").id, echo=False)
    assert st.items[0].status == "capped"
    conf["cost"] = None
    st = batch.resume_batch(settings, "b4", echo=False, retry_failed=True, cost_cap=20.0)
    assert st.items[0].status == "done" and st.cost_cap_usd == 20.0


def test_report_markdown_for_empty_values(settings, tmp_path):
    st = batch.create_batch(settings, _folder(tmp_path), batch_id="b5", cost_cap=0)
    md = report_markdown(st)
    assert "Cost cap per project run: none" in md and "| a_brief | a_brief.md | brief | queued | - |" in md


@pytest.fixture
def run_cli(settings, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    def _run(*argv: str) -> tuple[int, str, str]:
        code = cli.main(list(argv))
        out = capsys.readouterr()
        return code, out.out, out.err
    return _run


def test_cli_batch_status_pause_resume(settings, tmp_path, stub_runs, run_cli):
    d = _folder(tmp_path)
    code, out, _ = run_cli("batch", str(d), "--brand", "default", "--batch-id", "c1", "--plan", "--quiet")
    assert code == 0 and "batch_report.md" in out and "| d_voice |" in out
    code, out, _ = run_cli("batch-pause", "c1")
    assert code == 0 and json.loads(out)["paused"] is True
    code, out, _ = run_cli("batch-status")
    assert json.loads(out)[0]["id"] == "c1" and json.loads(out)[0]["status"] == "paused"
    code, out, err = run_cli("batch-resume", "c1", "--quiet")
    assert code == 0 and "| a_brief | a_brief.md | brief | done |" in out
    code, out, _ = run_cli("batch-status", "c1", "--json")
    assert [r["status"] for r in json.loads(out)["rows"]] == ["done"] * 4
    code, out, err = run_cli("batch", str(d), "--brand", "default", "--batch-id", "c2", "--quiet",
                             "--cost-cap", "2.5")
    assert code == 0 and "report:" in out and json.loads((batch.batch_dir(settings, "c2") / "state.json")
                                                         .read_text())["cost_cap_usd"] == 2.5
    assert run_cli("batch", str(d), "--brand", "nope")[0] == 2
    assert run_cli("batch", str(tmp_path / "missing"), "--brand", "default")[0] == 2
    assert run_cli("batch-status", "nope")[0] == 2


def test_cli_batch_defaults_cap_from_config(settings, tmp_path, run_cli):
    assert settings.batch.cost_cap_usd == 5.0
    run_cli("batch", str(_folder(tmp_path)), "--batch-id", "c3", "--plan")
    assert batch.load_batch(settings, "c3").cost_cap_usd == 5.0
