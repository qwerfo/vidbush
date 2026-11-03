"""API: shot list, thumbnails, media serving (traversal guard, ranges), shot actions, ops and chat."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vidbush.api.app import create_app
from vidbush.jobs import runner
from vidbush.jobs.pipeline import create_project
from vidbush.models import (Candidate, EditOp, EditOpName, InputMode, MotionKind, MotionSpec, Shot, ShotList,
                            ShotType, SourceKind)
from vidbush.util import run, write_json

PID = "shots1"


def _make_video(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=1",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", str(path)], timeout=60)
    return path


@pytest.fixture
def project(settings, tmp_path):
    """A project with a synthetic 3-shot shotlist, one image, one video clip and one rendered segment."""
    script = tmp_path / "s.md"
    script.write_text("# T\n\nText.")
    proj = create_project(settings, PID, InputMode.script, script_path=str(script), seed=1)
    root = Path(proj.paths.root)
    img = root / "sources" / "s0" / "still.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 360), (200, 120, 40)).save(img)
    clip = _make_video(root / "sources" / "s1" / "clip.mp4")
    seg = _make_video(root / "work" / "segments" / "abc.mp4")
    overlay = MotionSpec(kind=MotionKind.lower_third, duration=4, background="transparent",
                         payload={"title": "Gatun Locks"})
    shots = [
        Shot(id="s0", beat_id="b0", start=0, end=4, duration=4, type=ShotType.image_overlay, intent="Locks at dawn",
             overlay=overlay, overlay_path=str(root / "work" / "mg" / "ov.mov"),
             candidates=[Candidate(id="s0_wikimedia_0", source=SourceKind.wikimedia, local_path=str(img),
                                   keyframes=[str(img)], score=82, licence="CC BY 4.0"),
                         Candidate(id="s0_local_1", source=SourceKind.local, media="video", local_path=str(clip),
                                   keyframes=[], out_point=1.0, score=40, reject_reason="off topic")],
             selected="s0_wikimedia_0", judge_score=82),
        Shot(id="s1", beat_id="b1", start=4, end=9, duration=5, type=ShotType.footage, intent="Ship in lock",
             candidates=[Candidate(id="s1_local_0", source=SourceKind.local, media="video", local_path=str(clip),
                                   out_point=1.0, score=55)],
             selected="s1_local_0", judge_score=55, flags=["below_threshold"]),
        Shot(id="s2", beat_id="b2", start=9, end=12, duration=3, type=ShotType.motion_graphic, intent="Map",
             render_segment_path=str(seg)),
    ]
    write_json(root / "work" / "shotlist.json", ShotList(project_id=PID, duration=12, shots=shots))
    return root


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, start_workers=False)) as c:
        yield c


def _last_job(settings) -> runner.Job:
    return runner.list_jobs(settings, PID)[0]


def test_shots_endpoint_shape(client, project):
    data = client.get(f"/api/projects/{PID}/shots").json()
    assert data["duration"] == 12 and len(data["shots"]) == 3
    s0, s1, s2 = data["shots"]
    assert s0["overlay"] == "lower_third" and s0["selected"] == "s0_wikimedia_0" and s0["judge_score"] == 82
    assert {"index", "id", "type", "start", "end", "duration", "intent", "flags", "thumb_url"} <= set(s0)
    c = s0["candidates"][1]
    assert c["reject_reason"] == "off topic" and c["licence"] == "unknown" and c["thumb_url"]
    assert s0["candidates"][0]["selected"] is True and s0["candidates"][0]["licence"] == "CC BY 4.0"
    assert s1["flags"] == ["below_threshold"] and s2["candidates"] == [] and s2["segment_url"]


def test_shots_404_without_shotlist(client, settings, tmp_path):
    script = tmp_path / "x.md"
    script.write_text("x")
    create_project(settings, "empty", InputMode.script, script_path=str(script))
    assert client.get("/api/projects/empty/shots").status_code == 404


def _is_jpeg(r) -> bool:
    return r.status_code == 200 and r.content[:2] == b"\xff\xd8"


def test_shot_thumbs_from_keyframe_video_and_segment(client, project):
    shots = client.get(f"/api/projects/{PID}/shots").json()["shots"]
    for s in shots:
        r = client.get(s["thumb_url"])
        assert _is_jpeg(r), (s["id"], r.status_code, r.text[:200])
    assert len(list((project / "work" / "thumbs").glob("*.jpg"))) == 3
    cand = shots[0]["candidates"][1]
    assert _is_jpeg(client.get(cand["thumb_url"]))
    media = client.get(cand["media_url"])
    assert media.status_code == 200 and media.headers["content-type"] == "video/mp4"
    assert client.get(f"/api/projects/{PID}/shots/nope/thumb").status_code == 404


def test_media_serves_files_and_ranges(client, project):
    full = client.get(f"/api/projects/{PID}/media/work/segments/abc.mp4")
    assert full.status_code == 200 and full.headers["accept-ranges"] == "bytes"
    r = client.get(f"/api/projects/{PID}/media/work/segments/abc.mp4", headers={"Range": "bytes=10-99"})
    assert r.status_code == 206 and len(r.content) == 90
    assert r.headers["content-range"] == f"bytes 10-99/{len(full.content)}"
    assert r.content == full.content[10:100]
    assert client.get(f"/api/projects/{PID}/media/project.json").json()["id"] == PID


@pytest.mark.parametrize("path,codes", [
    ("../../etc/passwd", (404,)),  # the client normalises this one before sending
    ("..%2F..%2Fetc%2Fpasswd", (403,)),
    ("%2e%2e/%2e%2e/etc/passwd", (403,)),
    ("work/..%2F..%2F..%2Fproject.json", (403,)),
    ("%2Fetc%2Fpasswd", (403,)),
    ("work/%00x", (403,)),
])
def test_media_blocks_traversal(client, project, path, codes):
    r = client.get(f"/api/projects/{PID}/media/{path}")
    assert r.status_code in codes and "root:" not in r.text


def test_safe_join_guard(tmp_path):
    from fastapi import HTTPException
    from vidbush.api.media import safe_join
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f.txt").write_text("ok")
    assert safe_join(tmp_path, "a/f.txt") == (tmp_path / "a" / "f.txt").resolve()
    for bad in ("../x", "a/../../x", "/etc/passwd", "C:/x", "a\\..\\..\\x", ""):
        with pytest.raises(HTTPException) as e:
            safe_join(tmp_path / "a", bad)
        assert e.value.status_code == 403, bad


def test_media_blocks_symlink_escape(client, project, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    os.symlink(secret, project / "link.txt")
    assert client.get(f"/api/projects/{PID}/media/link.txt").status_code == 403
    assert client.get(f"/api/projects/{PID}/media/missing.txt").status_code == 404
    assert client.get("/api/projects/..%2F..%2Fetc/media/passwd").status_code in (400, 404)


def test_shot_actions_enqueue_apply_ops(client, settings, project):
    r = client.post(f"/api/projects/{PID}/shots/s0/swap", json={"candidate_id": "s0_local_1"})
    assert r.status_code == 200, r.text
    job = _last_job(settings)
    assert job.kind == "apply_ops" and job.id == r.json()["job_id"]
    assert json.loads(job.payload)["ops"] == [{"op": "swap_shot", "args": {"shot_id": "s0",
                                               "candidate_id": "s0_local_1"}, "reason": "operator picked a candidate"}]
    r = client.post(f"/api/projects/{PID}/shots/s1/regenerate", json={"prompt_delta": " at dusk "})
    assert r.json()["ops"][0] == {"op": "regenerate_ai", "args": {"shot_id": "s1", "prompt_delta": "at dusk"},
                                  "reason": "operator asked for an AI image"}
    r = client.post(f"/api/projects/{PID}/shots/s1/duration", json={"seconds": 6.25})
    assert r.json()["ops"][0]["args"] == {"shot_id": "s1", "seconds": 6.25}
    r = client.post(f"/api/projects/{PID}/shots/s0/remove_overlay")
    assert r.json()["ops"][0]["op"] == "remove_overlay"


@pytest.mark.parametrize("url,body,code", [
    ("shots/s0/swap", {"candidate_id": "nope"}, 404),
    ("shots/zz/swap", {"candidate_id": "x"}, 404),
    ("shots/s1/duration", {"seconds": 0.1}, 422),
    ("shots/s1/duration", {"seconds": 600}, 422),
    ("shots/s1/remove_overlay", None, 422),
    ("ops", {"ops": [{"op": "explode", "args": {}}]}, 422),
    ("ops", {"ops": []}, 422),
])
def test_shot_action_validation(client, settings, project, url, body, code):
    r = client.post(f"/api/projects/{PID}/{url}", json=body)
    assert r.status_code == code, r.text
    assert runner.list_jobs(settings, PID) == []


def test_upload_saves_file_and_enqueues(client, settings, project):
    r = client.post(f"/api/projects/{PID}/shots/s1/upload", files={"file": ("My Shot.PNG", b"\x89PNG data", "image/png")})
    assert r.status_code == 200, r.text
    path = Path(r.json()["ops"][0]["args"]["path"])
    assert path.parent == project / "inputs" / "uploads" and path.suffix == ".png" and path.read_bytes() == b"\x89PNG data"
    assert json.loads(_last_job(settings).payload)["ops"][0]["op"] == "upload_shot"
    bad = client.post(f"/api/projects/{PID}/shots/s1/upload", files={"file": ("x.sh", b"rm", "text/plain")})
    assert bad.status_code == 422


def test_ops_endpoint(client, settings, project):
    ops = [{"op": "change_music", "args": {"track": "undertow.ogg"}},
           {"op": "set_caption_style", "args": {"delta": {"color": "#FFD84A"}}}]
    r = client.post(f"/api/projects/{PID}/ops", json={"ops": ops})
    assert r.status_code == 200
    assert [o["op"] for o in json.loads(_last_job(settings).payload)["ops"]] == ["change_music", "set_caption_style"]


def test_chat_maps_and_optionally_applies(client, settings, project, monkeypatch):
    seen = {}

    def map_message(s, project_id, message):
        seen.update(project_id=project_id, message=message)
        return [EditOp(op=EditOpName.set_duration, args={"shot_id": "s1", "seconds": 7}, reason="longer")]
    mod = types.ModuleType("vidbush.editor.revise")
    mod.map_message = map_message
    monkeypatch.setitem(sys.modules, "vidbush.editor.revise", mod)
    r = client.post(f"/api/projects/{PID}/chat", json={"message": "make shot 2 longer"})
    assert r.status_code == 200 and r.json()["job_id"] is None
    assert r.json()["ops"][0]["args"]["seconds"] == 7 and seen == {"project_id": PID, "message": "make shot 2 longer"}
    assert runner.list_jobs(settings, PID) == []
    r = client.post(f"/api/projects/{PID}/chat", json={"message": "make shot 2 longer", "apply": True})
    job = _last_job(settings)
    assert r.json()["job_id"] == job.id and job.kind == "apply_ops"
    assert client.post(f"/api/projects/{PID}/chat", json={"message": ""}).status_code == 422
