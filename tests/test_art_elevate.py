"""Elevate image calls with reference images for the art layer (vidbush/art/elevate.py). No live network."""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from vidbush.art import elevate as E
from vidbush.providers import image as image_mod
from vidbush.providers.base import ProviderSpec, TransportError

RAW = b"\x89PNG\r\n\x1a\n" + b"raw-uncropped" * 20


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    E.reset()
    monkeypatch.setattr(image_mod.time, "sleep", lambda s: None)
    yield
    E.reset()


def _spec(**opts) -> ProviderSpec:
    return ProviderSpec("image_primary", "elevate", "nano_banana_pro", api_key="k", options=opts)


class Wire:
    def __init__(self, monkeypatch, polls=None) -> None:
        self.bodies: list[dict] = []
        self.polls = polls or [{"success": True, "data": {"status": "completed", "result_url": "https://cdn/x.png"}}]
        self.gets = 0
        monkeypatch.setattr(E, "post_json", self.post)
        monkeypatch.setattr(image_mod, "get_json", self.get)
        monkeypatch.setattr(E, "get_bytes", lambda url, *a, **k: RAW)

    def post(self, url, body, headers, timeout=None):
        self.bodies.append(body)
        return {"success": True, "data": {"id": f"t{len(self.bodies)}", "status": "pending"}}

    def get(self, url, headers=None, timeout=None):
        self.gets += 1
        return self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]


def test_data_uri_sources(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(RAW)
    for src in (p, str(p), RAW):
        uri = E.data_uri(src)
        assert uri.startswith("data:image/png;base64,") and base64.b64decode(uri.split(",", 1)[1]) == RAW
    arr = np.zeros((4, 6, 3), np.uint8)
    arr[..., 2] = 200
    im = Image.open(io.BytesIO(base64.b64decode(E.data_uri(arr).split(",", 1)[1])))
    assert im.size == (6, 4) and np.asarray(im)[0, 0, 2] == 200


def test_body_and_raw_bytes(monkeypatch, tmp_path):
    w = Wire(monkeypatch)
    out = tmp_path / "o.png"
    r = E.generate(_spec(), "paint", out, model="gpt-image-2.5-fast", references=[RAW], seed=2 ** 31 + 5)
    b = w.bodies[0]
    assert b["type"] == "image" and b["model"] == "gpt-image-2.5-fast" and b["aspect_ratio"] == "landscape"
    assert len(b["reference_images"]) == 1 and b["reference_images"][0].startswith("data:image/png;base64,")
    assert 0 <= b["seed"] < 2 ** 31
    assert out.read_bytes() == RAW  # uncropped, exactly as served
    assert r.units == 4.0 and r.unit_kind == "credits" and r.cost_usd == 0.0


def test_no_reference_key_without_references(monkeypatch, tmp_path):
    w = Wire(monkeypatch)
    r = E.generate(_spec(), "paint", tmp_path / "o.png")
    assert "reference_images" not in w.bodies[0] and "seed" not in w.bodies[0]
    assert w.bodies[0]["model"] == "nano_banana_pro"
    assert r.units == 5.0  # measured nano_banana_pro credits
    assert E.credits("some_new_model") == max(E.CREDITS.values())  # unknown cost counts as the dearest known


def test_poll_timeout_resumes_without_new_submit(monkeypatch, tmp_path):
    clock = iter(range(0, 10 ** 6, 1000))
    monkeypatch.setattr(image_mod.time, "time", lambda: next(clock))
    w = Wire(monkeypatch, polls=[{"success": True, "data": {"status": "processing"}}])
    with pytest.raises(TransportError, match="not finished"):
        E.generate(_spec(timeout_s=1), "paint", tmp_path / "o.png", references=[RAW])
    assert len(w.bodies) == 1 and len(E._PENDING) == 1
    w.polls = [{"success": True, "data": {"status": "completed", "result_url": "https://cdn/x.png"}}]
    E.generate(_spec(timeout_s=1), "paint", tmp_path / "o.png", references=[RAW])
    assert len(w.bodies) == 1 and not E._PENDING
    assert (tmp_path / "o.png").read_bytes() == RAW


def test_usage_error_is_empty(monkeypatch):
    def boom(*a, **k):
        raise TransportError("down")
    monkeypatch.setattr(E, "get_json", boom)
    assert E.usage(_spec()) == {}
    assert E.hour_credits(_spec()) == (0.0, 0.0)


def test_hour_credits(monkeypatch):
    data = {"success": True, "data": {
        "image": {"hourly": {"credits_used": 22, "credits_limit": 800}},
        "video": {"hourly": {"credits_used": 3, "credits_limit": 80}}}}
    monkeypatch.setattr(E, "get_json", lambda *a, **k: data)
    assert E.hour_credits(_spec()) == (22.0, 800.0)
    assert E.hour_credits(_spec(), "video") == (3.0, 80.0)


def test_failed_retry_task_drops_its_sidecar(monkeypatch, tmp_path):
    from vidbush.providers.base import TransportError
    out = tmp_path / "o.png"
    out.with_suffix(".task.json").write_text('{"key": "k", "task_id": "t1"}', encoding="utf-8")
    monkeypatch.setattr(E, "_key", lambda *a, **k: "k")
    monkeypatch.setattr(E.image_mod, "_poll", lambda *a, **k: (_ for _ in ()).throw(TransportError("Processing failed, please retry")))
    with pytest.raises(TransportError):
        E.generate(_spec(), "p", out)
    assert not out.with_suffix(".task.json").exists()  # the next call submits afresh instead of re-polling a dead task
