"""Elevate image to video adapter (provider type elevate_video), its registry line and probe route.
All HTTP is mocked: no live network call."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from vidbush.providers import probe as probe_mod, seats as seats_mod, tts_elevate, video as video_mod
from vidbush.providers import video_elevate as V
from vidbush.providers.base import CallResult, ProviderSpec, RefusalError, TransportError

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4096
BASE = "https://public-api.elevate.uno"


class Resp:
    def __init__(self, body: Any = None, status: int = 200, raw: bytes = b"", headers: Optional[dict] = None) -> None:
        self.status_code = status
        self.headers = headers or {}
        self._body = body
        self.content = raw or json.dumps(body).encode()
        self.text = self.content.decode("utf-8", "replace")

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no json")
        return self._body


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def sleep(self, s: float) -> None:
        self.t += float(s)

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(V.time, "sleep", c.sleep)
    monkeypatch.setattr(V.time, "monotonic", c)
    monkeypatch.setattr(V, "get_json", lambda *a, **k: {"success": True, "data": {"video": {
        "concurrent": {"used": 0, "limit": 1}}}})
    return c


class Server:
    """Elevate fake: submits return task ids in order; each task id serves a list of poll replies (last repeats)."""

    def __init__(self, monkeypatch, polls: dict[str, list[Any]], submits: Optional[list[Any]] = None) -> None:
        self.polls = polls
        self.submits = list(submits if submits is not None else polls.keys())
        self.sent: list[dict] = []
        monkeypatch.setattr(tts_elevate.requests, "request", self.request)

    def count(self, method: str, part: str = "") -> int:
        return sum(1 for s in self.sent if s["method"] == method and part in s["url"])

    def request(self, method, url, json=None, headers=None, timeout=None):
        self.sent.append({"method": method, "url": url, "json": json, "headers": headers or {}})
        if method == "POST":
            r = self.submits.pop(0)
            return r if isinstance(r, Resp) else Resp({"success": True, "data": {"id": r, "status": "pending"}})
        if "/v2/media/" in url:
            tid = url.split("/v2/media/")[1].split("?")[0]
            seq = self.polls[tid]
            r = seq.pop(0) if len(seq) > 1 else seq[0]
            return r if isinstance(r, Resp) else Resp({"success": True, "data": r})
        return Resp(raw=MP4)  # result_url download


def _spec(**opts) -> ProviderSpec:
    return ProviderSpec("elevate_video", "elevate_video", "veo_3_1", api_key="k", options=opts)


def _done(tid: str) -> dict:
    return {"id": tid, "status": "completed", "result_url": f"https://cdn.example/{tid}.mp4"}


def _still(tmp_path: Path) -> Path:
    p = tmp_path / "still.png"
    p.write_bytes(PNG)
    return p


# ---------------------------------------------------------------- request body

@pytest.mark.parametrize("model", ["", "your-image-to-video-model"])
def test_model_defaults_to_veo(model, tmp_path):
    spec = ProviderSpec("v", "elevate_video", model, api_key="k")
    assert V.request_body(spec, "p", None)["model"] == "veo_3_1"


def test_image_operation_with_png_data_url(tmp_path):
    body = V.request_body(_spec(), "move", _still(tmp_path))
    assert body["operation"] == "image" and body["type"] == "video" and body["aspect_ratio"] == "landscape"
    assert body["source_image"].startswith("data:image/png;base64,")
    assert base64.b64decode(body["source_image"].split(",", 1)[1]) == PNG
    jpg = tmp_path / "s.jpg"
    jpg.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
    assert V.request_body(_spec(), "m", jpg)["source_image"].startswith("data:image/jpeg;base64,")


def test_prompt_operation_without_still():
    body = V.request_body(_spec(), "p", None)
    assert body["operation"] == "prompt" and "source_image" not in body


def test_resolution_option():
    assert V.request_body(_spec(), "p", None)["params"]["resolution"] == "1080p"
    assert V.request_body(_spec(resolution="720p"), "p", None)["params"]["resolution"] == "720p"
    assert V.request_body(_spec(resolution="4k"), "p", None)["params"]["resolution"] == "1080p"


# ---------------------------------------------------------------- generate

@pytest.mark.parametrize("res,credits", [("1080p", 2.0), ("720p", 1.0)])
def test_happy_path(monkeypatch, clock, tmp_path, res, credits):
    srv = Server(monkeypatch, {"t1": [{"status": "pending"}, {"status": "processing"}, _done("t1")]})
    out = tmp_path / "clip.mp4"
    r = V.generate_video(_spec(resolution=res), "p", _still(tmp_path), out, 5.0)
    assert isinstance(r, CallResult) and r.value == out and out.read_bytes() == MP4
    assert r.units == credits and r.unit_kind == "credits" and r.cost_usd == 0.0
    assert srv.count("POST") == 1 and srv.count("GET", "/v2/media/t1") == 3
    download = srv.sent[-1]
    assert download["url"].startswith("https://cdn.example/") and "Authorization" not in download["headers"]
    assert "duration" not in srv.sent[0]["json"]
    assert not V._sidecar(out).exists()


def test_resume_from_sidecar_polls_without_submit(monkeypatch, clock, tmp_path):
    still, out = _still(tmp_path), tmp_path / "clip.mp4"
    req = V._request_hash(V.request_body(_spec(), "p", still))
    V._remember(out, req, "old")
    srv = Server(monkeypatch, {"old": [_done("old")]}, submits=[])
    V.generate_video(_spec(), "p", still, out, 4.0)
    assert srv.count("POST") == 0 and out.read_bytes() == MP4


def test_sidecar_for_other_request_ignored(monkeypatch, clock, tmp_path):
    still, out = _still(tmp_path), tmp_path / "clip.mp4"
    V._remember(out, "different-hash", "old")
    srv = Server(monkeypatch, {"new": [_done("new")]})
    V.generate_video(_spec(), "p", still, out, 4.0)
    assert srv.count("POST") == 1 and srv.count("GET", "/v2/media/old") == 0


def test_deadline_keeps_sidecar(monkeypatch, clock, tmp_path):
    out = tmp_path / "clip.mp4"
    Server(monkeypatch, {"slow": [{"status": "processing"}]})
    with pytest.raises(TransportError, match="still running"):
        V.generate_video(_spec(timeout_s=60, poll_s=10), "p", _still(tmp_path), out, 4.0)
    assert json.loads(V._sidecar(out).read_text())["task_id"] == "slow"
    assert not out.exists()


def test_dead_task_one_fresh_submit(monkeypatch, clock, tmp_path):
    srv = Server(monkeypatch, {"a": [{"status": "failed", "error": "worker crashed"}], "b": [_done("b")]})
    V.generate_video(_spec(), "p", _still(tmp_path), tmp_path / "c.mp4", 4.0)
    assert srv.count("POST") == 2


def test_dead_twice_raises_after_two_submits(monkeypatch, clock, tmp_path):
    srv = Server(monkeypatch, {"a": [{"status": "failed"}], "b": [{"status": "failed"}]})
    with pytest.raises(TransportError):
        V.generate_video(_spec(), "p", _still(tmp_path), tmp_path / "c.mp4", 4.0)
    assert srv.count("POST") == 2


def test_policy_failure_is_refusal_without_resubmit(monkeypatch, clock, tmp_path):
    srv = Server(monkeypatch, {"a": [{"status": "failed", "error": "blocked by safety policy"}]}, submits=["a", "b"])
    with pytest.raises(RefusalError):
        V.generate_video(_spec(), "p", _still(tmp_path), tmp_path / "c.mp4", 4.0)
    assert srv.count("POST") == 1


def test_404_poll_is_unknown_and_resubmits_once(monkeypatch, clock, tmp_path):
    srv = Server(monkeypatch, {"gone": [Resp({"detail": "Not Found"}, status=404)], "b": [_done("b")]})
    V.generate_video(_spec(), "p", _still(tmp_path), tmp_path / "c.mp4", 4.0)
    assert srv.count("POST") == 2


def test_capacity_429_on_submit_waited_out(monkeypatch, clock, tmp_path):
    busy = Resp({"success": False, "error": {"code": "RATE_LIMIT_EXCEEDED", "message": "Video concurrent limit reached",
                                             "details": {"used": 1, "limit": 1}}}, status=429)
    srv = Server(monkeypatch, {"t": [_done("t")]}, submits=[busy, busy, "t"])
    V.generate_video(_spec(), "p", _still(tmp_path), tmp_path / "c.mp4", 4.0)
    assert srv.count("POST") == 3 and clock.t > 0


def test_missing_still_refused(tmp_path):
    with pytest.raises(RefusalError):
        V.generate_video(_spec(), "p", tmp_path / "nope.png", tmp_path / "c.mp4", 4.0)


# ---------------------------------------------------------------- lane wait

def _usage(used: int, limit: int = 1) -> dict:
    return {"success": True, "data": {"video": {"concurrent": {"used": used, "limit": limit}}}}


def test_wait_for_slot_waits_until_free(monkeypatch):
    replies = [_usage(1), _usage(1), _usage(0)]
    monkeypatch.setattr(V, "get_json", lambda *a, **k: replies.pop(0))
    c = Clock()
    V.wait_for_slot(_spec(poll_s=10), sleep=c.sleep, clock=c)
    assert c.t == 20 and not replies


def test_wait_for_slot_gives_up(monkeypatch):
    monkeypatch.setattr(V, "get_json", lambda *a, **k: _usage(1))
    c = Clock()
    with pytest.raises(TransportError, match="still full"):
        V.wait_for_slot(_spec(poll_s=10, capacity_wait_s=50), sleep=c.sleep, clock=c)


def test_wait_for_slot_unreadable_usage_proceeds(monkeypatch):
    def boom(*a, **k):
        raise TransportError("down")
    monkeypatch.setattr(V, "get_json", boom)
    V.wait_for_slot(_spec(), sleep=lambda s: pytest.fail("must not wait"), clock=lambda: 0.0)


# ---------------------------------------------------------------- probe

def test_probe_ok(monkeypatch):
    def gj(url, **k):
        if url.endswith("/usage"):
            return {"success": True, "data": {"video": {"hourly": {"credits_used": 3, "credits_limit": 80}}}}
        return {"success": True, "data": {"video": {"status": "operational"}}}
    monkeypatch.setattr(V, "get_json", gj)
    note = V.probe(_spec())
    assert "operational" in note and "3/80" in note and "veo_3_1" in note


def test_probe_degraded_raises(monkeypatch):
    def gj(url, **k):
        if url.endswith("/usage"):
            return {"success": True, "data": {"video": {}}}
        return {"success": True, "data": {"video": {"status": "degraded"}}}
    monkeypatch.setattr(V, "get_json", gj)
    with pytest.raises(TransportError, match="degraded"):
        V.probe(_spec())


# ---------------------------------------------------------------- registry and probe route

def _seats(spec: ProviderSpec) -> Any:
    s = seats_mod.Seats.__new__(seats_mod.Seats)
    s.settings = SimpleNamespace(cost=SimpleNamespace(video_per_second_usd=0.1))
    s._call = lambda role, stage, fn, mock_fn, note="": fn(spec)
    return s


def test_seat_dispatches_elevate_video(monkeypatch, tmp_path):
    seen = []

    def fake(spec, prompt, still, out, seconds, per_s):
        seen.append("elevate")
        return CallResult(value=out, provider=spec, mock=False)
    monkeypatch.setattr(V, "generate_video", fake)
    monkeypatch.setattr(video_mod, "generate_video", lambda *a: pytest.fail("fal path used"))
    out = tmp_path / "o.mp4"
    assert _seats(_spec()).video("p", None, out, 4.0, "visuals") == out and seen == ["elevate"]


def test_seat_other_types_use_video_py(monkeypatch, tmp_path):
    seen = []

    def fake(spec, prompt, still, out, seconds, per_s):
        seen.append(per_s)
        return CallResult(value=out, provider=spec, mock=False)
    monkeypatch.setattr(video_mod, "generate_video", fake)
    monkeypatch.setattr(V, "generate_video", lambda *a: pytest.fail("elevate path used"))
    fal = ProviderSpec("video_primary", "fal", "m", api_key="k")
    _seats(fal).video("p", None, tmp_path / "o.mp4", 4.0, "visuals")
    assert seen == [0.1]


def test_probe_spec_routes_elevate_video(monkeypatch):
    monkeypatch.setattr(V, "probe", lambda spec, timeout=0: "video probed")
    assert probe_mod.probe_spec("video", _spec()) == "video probed"
