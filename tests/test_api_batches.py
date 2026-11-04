"""API: batch create, list, detail, pause and resume, and the batch worker job (pipeline stubbed)."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vidbush.api.app import create_app
from vidbush.api.jobs_handler import handle_job
from vidbush.jobs import batch, pipeline, runner
from vidbush.models import STAGES


@pytest.fixture
def client(settings, monkeypatch):
    real = pipeline.run_project

    def fake(settings_, project_id, **kw):
        return real(settings_, project_id, stages={n: (lambda ctx: []) for n in STAGES}, **kw)
    monkeypatch.setattr(pipeline, "run_project", fake)
    with TestClient(create_app(settings, start_workers=False)) as c:
        yield c


def _folder(tmp_path: Path) -> Path:
    d = tmp_path / "briefs"
    d.mkdir()
    (d / "one.md").write_text("---\ntopic: The Erie Canal\n---\n")
    (d / "two.md").write_text("# Two\n\nA script.")
    return d


def _run_queued_job(settings) -> runner.Job:
    job = runner._claim(settings, set())
    assert job is not None
    handle_job(settings, job)
    runner._finish(settings, int(job.id), True)
    return job


def test_create_enqueues_batch_job_and_worker_runs_it(client, settings, tmp_path):
    r = client.post("/api/batches", json={"source": str(_folder(tmp_path)), "brand": "default",
                                          "batch_id": "q1", "cost_cap_usd": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["batch_id"] == "q1" and body["cost_cap_usd"] == 3 and body["job_id"]
    assert [row["status"] for row in body["rows"]] == ["queued", "queued"]
    assert body["report_path"].endswith("batch_report.md")
    jobs = runner.list_jobs(settings, "batch:q1")
    assert [j.kind for j in jobs] == ["batch"]
    again = client.post("/api/batches/q1/resume").json()  # an active job exists: no second job
    assert again["job_id"] == body["job_id"] and len(runner.list_jobs(settings, "batch:q1")) == 1
    job = _run_queued_job(settings)
    assert job.project_id == "batch:q1"
    detail = client.get("/api/batches/q1").json()
    assert detail["status"] == "done" and [row["status"] for row in detail["rows"]] == ["done", "done"]
    assert [row["mode"] for row in detail["rows"]] == ["brief", "script"]
    listed = client.get("/api/batches").json()
    assert listed[0]["id"] == "q1" and listed[0]["counts"] == {"done": 2}
    assert client.get("/api/projects/one").status_code == 200


def test_pause_and_resume_routes(client, settings, tmp_path):
    client.post("/api/batches", json={"source": str(_folder(tmp_path)), "batch_id": "q2", "start": False})
    assert runner.list_jobs(settings, "batch:q2") == []
    p = client.post("/api/batches/q2/pause").json()
    assert p["paused"] is True and p["status"] == "paused"
    state = batch.run_batch(settings, "q2", echo=False)  # a runner that starts while paused does nothing
    assert [i.status for i in state.items] == ["queued", "queued"]
    r = client.post("/api/batches/q2/resume", json={"retry_failed": False}).json()
    assert r["paused"] is False and r["job_id"] and r["status"] == "queued"
    _run_queued_job(settings)
    assert client.get("/api/batches/q2").json()["status"] == "done"


def test_errors(client, tmp_path):
    assert client.post("/api/batches", json={"source": str(tmp_path / "nope")}).status_code == 422
    assert client.post("/api/batches", json={"source": str(tmp_path), "brand": "nope"}).status_code == 422
    assert client.get("/api/batches/unknown").status_code == 404
    assert client.get("/api/batches/bad..id").status_code == 400
    assert client.post("/api/batches/unknown/pause").status_code == 404
    d = _folder(tmp_path)
    assert client.post("/api/batches", json={"source": str(d), "batch_id": "dup", "start": False}).status_code == 200
    assert client.post("/api/batches", json={"source": str(d), "batch_id": "dup"}).status_code == 409
