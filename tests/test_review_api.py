"""Review API: review.json via GET and PUT, progress on the project, summary, export, fixes and calibration."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from review_fixture import PID, build_review_project, build_shots, write_shots
from vidbush.api.app import create_app
from vidbush.jobs import runner
from vidbush.jobs.pipeline import create_project
from vidbush.models import InputMode
from vidbush.util import read_json

BASE = f"/api/projects/{PID}/review"


@pytest.fixture
def root(settings) -> Path:
    return build_review_project(settings)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, start_workers=False)) as c:
        yield c


def put(client, shot, **body):
    return client.put(f"{BASE}/shots/{shot}", json=body)


def test_get_before_any_save(client, root):
    r = client.get(BASE)
    assert r.status_code == 200
    d = r.json()
    assert d["exists"] is False and d["total"] == 8 and d["reviewed"] == 0
    assert [e["shot_id"] for e in d["shots"]] == [f"shot_{i:03d}" for i in range(1, 9)]
    assert all(e["verdict"] is None and e["comment"] == "" and e["rejects"] == [] for e in d["shots"])
    assert [v["key"] for v in d["verdicts"]] == ["good", "wrong_picture", "wrong_timing", "quality_problem", "other"]
    assert [x["key"] for x in d["rejects"]] == ["burned_in_text", "watermark_logo", "talking_head", "anachronism",
                                                "repeated_clip"]
    assert d["threshold"] == 70.0 and d["export_url"] is None
    assert not (root / "review.json").exists()


def test_put_verdict_comment_rejects(client, root):
    r = put(client, "shot_002", verdict="quality_problem", comment="Logo in the corner", rejects=["watermark_logo"])
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["reviewed"] == 1 and d["total"] == 8
    s = d["shot"]
    assert s["verdict"] == "quality_problem" and s["comment"] == "Logo in the corner"
    assert s["rejects"] == ["watermark_logo"] and s["index"] == 1
    g = client.get(BASE).json()
    assert g["exists"] is True and g["reviewed"] == 1
    assert g["shots"][1]["verdict"] == "quality_problem"


def test_partial_put_keeps_other_fields(client, root):
    put(client, "shot_001", verdict="wrong_picture", comment="first", rejects=["anachronism", "burned_in_text"])
    s = put(client, "shot_001", comment="second").json()["shot"]
    assert s["verdict"] == "wrong_picture" and s["comment"] == "second"
    assert s["rejects"] == ["burned_in_text", "anachronism"]  # checklist order
    s = put(client, "shot_001", verdict="other").json()["shot"]
    assert s["comment"] == "second" and s["rejects"] == ["burned_in_text", "anachronism"]
    s = put(client, "shot_001", rejects=[]).json()["shot"]
    assert s["rejects"] == [] and s["verdict"] == "other"


def test_clear_verdict(client, root):
    put(client, "shot_003", verdict="good", comment="keep")
    d = put(client, "shot_003", clear_verdict=True).json()
    assert d["shot"]["verdict"] is None and d["shot"]["comment"] == "keep" and d["reviewed"] == 0


def test_shot_ref_by_number(client, root):
    d = put(client, "4", verdict="good").json()
    assert d["shot"]["shot_id"] == "shot_004"


@pytest.mark.parametrize("body", [{"verdict": "terrible"}, {"rejects": ["blurry"]}])
def test_unknown_verdict_or_reject_is_422(client, root, body):
    assert put(client, "shot_001", **body).status_code == 422
    assert not (root / "review.json").exists()


def test_404s(client, settings, root, tmp_path):
    assert put(client, "shot_999", verdict="good").status_code == 404
    assert put(client, "99", verdict="good").status_code == 404
    assert client.get("/api/projects/nope/review").status_code == 404
    assert client.put("/api/projects/nope/review/shots/1", json={"verdict": "good"}).status_code == 404
    script = tmp_path / "s.md"
    script.write_text("# T\n\nText.")
    create_project(settings, "bare", InputMode.script, script_path=str(script), seed=1)
    for path in ("", "/summary", "/fixes", "/calibration"):
        assert client.get(f"/api/projects/bare/review{path}").status_code == 404
    assert client.put("/api/projects/bare/review/shots/1", json={"verdict": "good"}).status_code == 404
    assert client.put("/api/projects/bare/review/comment", json={"comment": "x"}).status_code == 404


def test_long_comment_round_trips(client, root):
    text = ("Line one of a long note.\n\n" + "x" * 50000 + "\nend | with pipe")
    assert len(text) > 50000
    assert put(client, "shot_005", comment=text).json()["shot"]["comment"] == text
    assert client.get(BASE).json()["shots"][4]["comment"] == text
    assert read_json(root / "review.json")["shots"][4]["comment"] == text
    long_project = "P" * 60000
    assert client.put(f"{BASE}/comment", json={"comment": long_project}).json()["project_comment"] == long_project


def test_project_comment(client, root):
    r = client.put(f"{BASE}/comment", json={"comment": "Pacing is good.\nMusic too loud at 2:10."})
    assert r.status_code == 200
    d = r.json()
    assert d["project_comment"] == "Pacing is good.\nMusic too loud at 2:10." and d["reviewed"] == 0
    g = client.get(BASE).json()
    assert g["project_comment"] == "Pacing is good.\nMusic too loud at 2:10."
    assert g["project_comment_updated_at"] is not None


def test_project_page_review_key(client, root):
    assert client.get(f"/api/projects/{PID}").json()["review"] is None
    put(client, "shot_001", verdict="good")
    put(client, "shot_002", comment="no verdict yet")
    assert client.get(f"/api/projects/{PID}").json()["review"] == {"reviewed": 1, "total": 8}


def test_timestamps(client, root):
    first = put(client, "shot_001", verdict="good").json()["shot"]
    assert first["created_at"] and first["created_at"] == first["updated_at"]
    time.sleep(0.01)
    second = put(client, "shot_001", comment="still good").json()["shot"]
    assert second["created_at"] == first["created_at"]
    assert second["updated_at"] > first["updated_at"]
    other = client.get(BASE).json()["shots"][1]
    assert other["created_at"] is None and other["updated_at"] is None


def test_snapshot_content(client, root):
    s = put(client, "shot_001", verdict="good").json()["shot"]["snapshot"]
    assert s["type"] == "footage" and s["start"] == 0.0 and s["duration"] == 4.0
    assert s["selected"] == "shot_001_youtube_0" and s["source"] == "youtube" and s["site"] == "www.youtube.com"
    assert s["licence"] == "CC BY 3.0" and s["judge_score"] == 84 and s["judge_reason"] == "canyon walls, period look"
    assert [c["id"] for c in s["rejected"]] == ["shot_001_youtube_1"]
    assert s["rejected"][0]["reason"].startswith("burned_in_captions (prefilter")
    s7 = put(client, "shot_007", verdict="wrong_picture").json()["shot"]["snapshot"]
    assert s7["site"] == "www.pexels.com" and s7["selected_reject"] == "best available accepted"
    assert "shot_007_pexels_0" not in [c["id"] for c in s7["rejected"]]  # the selected one is never listed
    assert s7["flags"] == ["below_threshold", "no_match_accepted"]
    s5 = put(client, "shot_005", verdict="good").json()["shot"]["snapshot"]
    assert s5["site"] == "ai_image" and s5["url"] == ""  # no url falls back to the source kind
    s6 = put(client, "shot_006", verdict="good").json()["shot"]["snapshot"]
    assert s6["selected"] is None and s6["judge_score"] is None and s6["site"] == ""


def test_review_json_on_disk(client, root):
    put(client, "shot_003", verdict="wrong_picture")
    client.put(f"{BASE}/comment", json={"comment": "overall fine"})
    data = read_json(root / "review.json")
    assert data["project_id"] == PID and data["project_comment"] == "overall fine"
    assert len(data["shots"]) == 8 and len({e["shot_id"] for e in data["shots"]}) == 8
    assert data["created_at"] <= data["updated_at"]
    e = data["shots"][2]
    assert e["shot_id"] == "shot_003" and e["verdict"] == "wrong_picture" and e["created_at"] and e["updated_at"]


def test_removed_shots_are_kept_but_not_counted(client, root):
    put(client, "shot_008", verdict="good", comment="last shot")
    put(client, "shot_001", verdict="good")
    shots = build_shots(root)[:-1]
    write_shots(root, shots)
    d = client.get(BASE).json()
    assert d["total"] == 7 and d["reviewed"] == 1
    gone = [e for e in d["shots"] if e["shot_id"] == "shot_008"]
    assert gone and gone[0]["in_shotlist"] is False and gone[0]["comment"] == "last shot"
    assert d["shots"][-1]["shot_id"] == "shot_008"
    assert client.get(f"/api/projects/{PID}").json()["review"] == {"reviewed": 1, "total": 7}
    summ = client.get(f"{BASE}/summary").json()
    assert summ["total"] == 7 and summ["comments"] == 0


def test_summary_endpoint(client, root):
    put(client, "shot_001", verdict="good")
    put(client, "shot_002", verdict="quality_problem", rejects=["watermark_logo"], comment="logo")
    d = client.get(f"{BASE}/summary").json()
    assert d["reviewed"] == 2 and d["total"] == 8 and d["unreviewed"] == 6
    assert d["verdicts"]["good"] == 1 and d["verdicts"]["quality_problem"] == 1
    assert d["rejects"]["watermark_logo"] == 1
    assert d["flagged"] == [{"shot_id": "shot_002", "number": 2, "start": 4.0, "time": "0:04.0",
                             "rejects": ["watermark_logo"], "verdict": "quality_problem"}]
    assert "reviewed 2 of 8" in d["text"]


def test_export_endpoint(client, root):
    put(client, "shot_003", verdict="wrong_picture", comment="No tents, show the shacks.")
    r = client.post(f"{BASE}/export")
    assert r.status_code == 200
    d = r.json()
    md = root / "review.md"
    assert Path(d["path"]) == md and md.exists()
    assert d["markdown"] == md.read_text(encoding="utf-8")
    assert "No tents, show the shacks." in d["markdown"]
    assert d["url"]
    assert client.get(BASE).json()["export_url"] == d["url"]


def test_fixes_plan_apply_and_skip(client, settings, root):
    put(client, "shot_001", verdict="wrong_picture", comment="Too modern. Show 1931 canyon walls.")
    put(client, "shot_002", verdict="wrong_timing")
    put(client, "shot_003", verdict="other", comment="hmm")
    plan = client.get(f"{BASE}/fixes").json()
    assert [o["op"] for o in plan["ops"]] == ["change_type"]
    assert plan["ops"][0]["args"]["shot_id"] == "shot_001"
    assert [t["shot_id"] for t in plan["timing"]] == ["shot_002"]
    assert [c["shot_id"] for c in plan["comments"]] == ["shot_003"]
    assert plan["skipped"] == []
    r = client.post(f"{BASE}/fixes", json={"ops": plan["ops"]})
    assert r.status_code == 200, r.text
    jobs = runner.list_jobs(settings, PID)
    assert len(jobs) == 1 and jobs[0].kind == "apply_ops" and jobs[0].id == r.json()["job_id"]
    assert json.loads(jobs[0].payload)["ops"] == plan["ops"]
    fix = client.get(BASE).json()["shots"][0]["fix"]
    assert fix["op"] == "change_type" and fix["job_id"] == jobs[0].id
    again = client.get(f"{BASE}/fixes").json()
    assert again["ops"] == [] and [s["shot_id"] for s in again["skipped"]] == ["shot_001"]
    time.sleep(0.01)
    put(client, "shot_001", comment="Still wrong: no dam workers visible.")
    third = client.get(f"{BASE}/fixes").json()
    assert [o["args"]["shot_id"] for o in third["ops"]] == ["shot_001"] and third["skipped"] == []


def test_fixes_post_needs_ops(client, settings, root):
    assert client.post(f"{BASE}/fixes", json={"ops": []}).status_code == 422
    assert runner.list_jobs(settings, PID) == []


def test_calibration_endpoint(client, root):
    put(client, "shot_001", verdict="wrong_picture")  # 84, wrong above
    put(client, "shot_003", verdict="good")  # 66, good below
    put(client, "shot_008", verdict="good")  # 91
    d = client.get(f"{BASE}/calibration").json()
    assert d["threshold"] == 70.0 and d["labelled"] == 3
    assert [r["shot_id"] for r in d["wrong_above_threshold"]] == ["shot_001"]
    assert [r["shot_id"] for r in d["good_below_threshold"]] == ["shot_003"]
    assert "Suggested judge.threshold" in d["text"]
