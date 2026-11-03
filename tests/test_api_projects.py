"""API: project creation, detail, logs, jobs, resume and the job handler (pipeline stubbed)."""
from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from vidbush.api.app import create_app
from vidbush.api.jobs_handler import handle_job
from vidbush.jobs import pipeline, runner
from vidbush.models import EditResult, StageStatus

SCRIPT = "# Canal Story\n\n## The dream\nPeople wanted a shortcut between two oceans for centuries.\n"


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, start_workers=False)) as c:
        yield c


def _create_script(client: TestClient, pid: str = "p1") -> dict:
    r = client.post("/api/projects", data={"mode": "script", "script_text": SCRIPT, "project_id": pid,
                                           "target_minutes": "6", "format": "documentary"})
    assert r.status_code == 200, r.text
    return r.json()


def _noop_stages() -> dict:
    return {name: (lambda ctx: []) for name in pipeline.STAGES}


def _stub_pipeline(monkeypatch, calls: list) -> None:
    """run_project runs the real runner with no-op stages, so on_progress fires for every stage."""
    real = pipeline.run_project

    def fake(settings, project_id, **kw):
        calls.append(kw)
        return real(settings, project_id, stages=_noop_stages(), **kw)
    monkeypatch.setattr(pipeline, "run_project", fake)


def test_create_script_project_enqueues_run(client, settings):
    body = _create_script(client)
    root = Path(settings.output_folder) / "p1"
    proj = json.loads((root / "project.json").read_text())
    assert proj["input_mode"] == "script" and proj["target_minutes"] == 6
    assert Path(proj["script_path"]).name == "p1.md"
    assert (root / "inputs" / "p1.md").read_text() == SCRIPT
    jobs = runner.list_jobs(settings, "p1")
    assert [j.kind for j in jobs] == ["run"] and jobs[0].id == body["job_id"]
    listed = client.get("/api/projects").json()
    assert listed[0]["id"] == "p1" and listed[0]["status"] == "created"
    assert listed[0]["active_job"]["kind"] == "run"


def test_create_from_uploaded_script_keeps_stem_for_fixture_key(client, settings):
    files = {"script": ("demo_script.md", SCRIPT.encode(), "text/markdown")}
    r = client.post("/api/projects", data={"mode": "script"}, files=files)
    assert r.status_code == 200, r.text
    proj = r.json()["project"]
    assert proj["id"].startswith("demo_script_")
    assert Path(proj["script_path"]).name == "demo_script.md"


def test_create_brief_and_voiceover(client, settings):
    r = client.post("/api/projects", data={"mode": "brief", "topic": "The Panama Canal", "target_minutes": "10",
                                           "format": "top10", "audience": "history fans", "angle": "engineering"})
    assert r.status_code == 200, r.text
    proj = r.json()["project"]
    assert proj["brief"]["topic"] == "The Panama Canal" and proj["format"] == "top10"
    assert proj["id"].startswith("the_panama_canal_")
    audio = {"voiceover": ("my take.mp3", b"ID3fake", "audio/mpeg")}
    r = client.post("/api/projects", data={"mode": "voiceover", "script_text": SCRIPT, "project_id": "vo1"},
                    files=audio)
    assert r.status_code == 200, r.text
    proj = r.json()["project"]
    assert Path(proj["voiceover_path"]).name == "my_take.mp3" and Path(proj["script_path"]).exists()


@pytest.mark.parametrize("data,code", [
    ({"mode": "brief"}, 422),
    ({"mode": "script"}, 422),
    ({"mode": "voiceover", "script_text": "x"}, 422),
    ({"mode": "novel", "topic": "x"}, 422),
    ({"mode": "brief", "topic": "x", "target_minutes": "500"}, 422),
    ({"mode": "brief", "topic": "x", "brand": "nope"}, 422),
    ({"mode": "brief", "topic": "x", "project_id": "../evil"}, 400),
])
def test_create_validation(client, data, code):
    assert client.post("/api/projects", data=data).status_code == code


def test_create_rejects_bad_upload_type_and_duplicate_id(client):
    r = client.post("/api/projects", data={"mode": "script"}, files={"script": ("x.exe", b"MZ", "application/x")})
    assert r.status_code == 422
    _create_script(client, "dup")
    r = client.post("/api/projects", data={"mode": "script", "script_text": SCRIPT, "project_id": "dup"})
    assert r.status_code == 409


def test_handler_runs_stubbed_pipeline_and_updates_row(client, settings, monkeypatch):
    calls: list = []
    _stub_pipeline(monkeypatch, calls)
    _create_script(client)
    job = runner.list_jobs(settings, "p1")[0]
    handle_job(settings, job)
    assert calls and calls[0]["echo"] is False and callable(calls[0]["on_progress"])
    with Session(runner.engine_for(settings)) as s:
        row = s.get(runner.ProjectRow, "p1")
    assert row.status == "done" and row.current_stage == ""
    detail = client.get("/api/projects/p1").json()
    assert detail["status"] == "done" and detail["final_url"] is None
    assert detail["stages"]["research"]["status"] == StageStatus.skipped.value
    assert detail["stages"]["packaging"]["status"] == "done" and len(detail["stage_order"]) == 13


def test_detail_final_url_is_cache_busted(client, settings):
    _create_script(client)
    (Path(settings.output_folder) / "p1" / "final.mp4").write_bytes(b"\x00" * 64)
    (Path(settings.output_folder) / "p1" / "description.md").write_text("Hello")
    d = client.get("/api/projects/p1").json()
    assert d["final_url"].startswith("/api/projects/p1/media/final.mp4?v=")
    assert d["outputs"]["description.md"].startswith("/api/projects/p1/media/description.md")
    assert client.get("/api/projects").json()[0]["has_final"] is True
    assert client.get("/api/projects/missing").status_code == 404


def test_log_tail_and_jobs(client, settings):
    _create_script(client)
    log = Path(settings.data_folder) / "logs" / "p1.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"ts": i, "level": "info", "msg": f"line {i}"}) for i in range(5)]
    log.write_text("\n".join(lines[:3] + ["not json"] + lines[3:]) + "\n")
    tail = client.get("/api/projects/p1/log?n=3").json()
    assert [r["msg"] for r in tail] == ["line 3", "line 4"]
    assert len(client.get("/api/projects/p1/log").json()) == 5
    jobs = client.get("/api/projects/p1/jobs").json()
    assert jobs[0]["kind"] == "run" and jobs[0]["payload"] == {}
    assert client.get(f"/api/projects/p1/jobs/{jobs[0]['id']}").json()["status"] == "queued"
    assert client.get("/api/projects/p1/jobs/9999").status_code == 404


def test_resume_refuses_while_a_run_is_queued(client, settings):
    _create_script(client)
    assert client.post("/api/projects/p1/resume").status_code == 409
    runner._finish(settings, runner.list_jobs(settings, "p1")[0].id, True)
    r = client.post("/api/projects/p1/resume")
    assert r.status_code == 200
    assert runner.list_jobs(settings, "p1")[0].kind == "resume"
    assert client.post("/api/projects/nope/resume").status_code == 404


def _fake_ops_module(monkeypatch, fn) -> None:
    mod = types.ModuleType("vidbush.editor.ops")
    mod.apply_and_render = fn
    monkeypatch.setitem(sys.modules, "vidbush.editor.ops", mod)


def test_handler_apply_ops_writes_last_edit(client, settings, monkeypatch):
    _create_script(client)
    seen = {}

    def apply_and_render(s, project_id, ops, echo=True, job_id=None):
        seen.update(project_id=project_id, ops=ops, echo=echo, job_id=job_id)
        return EditResult(ops=ops, applied=[o.op.value for o in ops], dirty_shots=["s1"])
    _fake_ops_module(monkeypatch, apply_and_render)
    ops = [{"op": "set_duration", "args": {"shot_id": "s1", "seconds": 5}}]
    jid = runner.enqueue(settings, "p1", "apply_ops", {"ops": ops})
    handle_job(settings, next(j for j in runner.list_jobs(settings, "p1") if j.id == jid))
    assert seen["project_id"] == "p1" and seen["ops"][0].args["seconds"] == 5 and seen["echo"] is False
    assert seen["job_id"] == jid
    last = json.loads((Path(settings.output_folder) / "p1" / "work" / "last_edit.json").read_text())
    assert last["job_id"] == jid and last["ok"] is True and last["result"]["dirty_shots"] == ["s1"]
    assert client.get("/api/projects/p1").json()["last_edit"]["job_id"] == jid


def test_handler_apply_ops_failure_records_and_raises(client, settings, monkeypatch):
    _create_script(client)

    def boom(*a, **k):
        raise RuntimeError("render exploded")
    _fake_ops_module(monkeypatch, boom)
    jid = runner.enqueue(settings, "p1", "apply_ops", {"ops": []})
    with pytest.raises(RuntimeError):
        handle_job(settings, next(j for j in runner.list_jobs(settings, "p1") if j.id == jid))
    last = json.loads((Path(settings.output_folder) / "p1" / "work" / "last_edit.json").read_text())
    assert last["ok"] is False and "render exploded" in last["error"]


def test_handler_rejects_unknown_kind(settings):
    with pytest.raises(ValueError):
        handle_job(settings, runner.Job(project_id="x", kind="mystery"))


def test_worker_pool_processes_jobs_on_startup(settings, monkeypatch):
    calls: list = []
    _stub_pipeline(monkeypatch, calls)
    with TestClient(create_app(settings, start_workers=True, workers=1)) as c:
        assert c.get("/api/health").json()["workers"] is True
        jid = _create_script(c, "live")["job_id"]
        deadline = time.time() + 15
        while time.time() < deadline:
            job = c.get(f"/api/projects/live/jobs/{jid}").json()
            if job["status"] in ("done", "failed"):
                break
            time.sleep(0.2)
        assert job["status"] == "done", job["error"]
        assert c.get("/api/projects/live").json()["status"] == "done"
