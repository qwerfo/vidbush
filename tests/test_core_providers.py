"""Provider layer contract: retries, fallback per request, down marking, cost records, mock mode."""
from __future__ import annotations

from pathlib import Path

import pytest

from vidbush.models import VoiceConfig
from vidbush.providers import base
from vidbush.providers.base import (CallResult, CostTracker, ProviderClient, ProviderError, ProviderSpec,
                                    RefusalError, TransportError)
from vidbush.providers.llm import parse_json_loose
from vidbush.providers.seats import Seats


def _spec(name: str) -> ProviderSpec:
    return ProviderSpec(name=name, type="openai_compat", model="m", base_url="http://x", api_key="k")


@pytest.fixture(autouse=True)
def _clean_health():
    base.HEALTH._down.clear()
    yield
    base.HEALTH._down.clear()


def test_transport_retries_then_falls_back():
    tracker = CostTracker("p")
    client = ProviderClient(tracker, sleep=lambda s: None)
    calls = {"a": 0, "b": 0}

    def fn(spec):
        calls[spec.name] += 1
        if spec.name == "a":
            raise TransportError("boom")
        return CallResult(value="ok", provider=spec, mock=False, cost_usd=0.01)

    res = client.call("writer", "script", [_spec("a"), _spec("b")], fn)
    assert res.value == "ok" and res.provider.name == "b"
    assert calls["a"] == 3 and calls["b"] == 1
    assert base.HEALTH.is_down("a:m")
    assert tracker.report.total_usd == pytest.approx(0.01)
    assert len(tracker.report.entries) == 4


def test_refusal_falls_back_without_retry_and_does_not_mark_down():
    client = ProviderClient(CostTracker("p"), sleep=lambda s: None)
    calls = {"a": 0}

    def fn(spec):
        if spec.name == "a":
            calls["a"] += 1
            raise RefusalError("content policy")
        return CallResult(value=1, provider=spec, mock=False)

    assert client.call("image", "visuals", [_spec("a"), _spec("b")], fn).value == 1
    assert calls["a"] == 1
    assert not base.HEALTH.is_down("a:m")


def test_down_provider_is_skipped_until_probe_window(monkeypatch):
    base.HEALTH.mark_down("a:m")
    client = ProviderClient(CostTracker("p"), sleep=lambda s: None)
    used = []

    def fn(spec):
        used.append(spec.name)
        return CallResult(value=1, provider=spec, mock=False)

    client.call("writer", "s", [_spec("a"), _spec("b")], fn)
    assert used == ["b"]
    base.HEALTH._down["a:m"] -= base.PROBE_INTERVAL_S + 1
    used.clear()
    client.call("writer", "s", [_spec("a"), _spec("b")], fn)
    assert used == ["a"] and not base.HEALTH.is_down("a:m")


def test_all_fail_raises():
    client = ProviderClient(CostTracker("p"), sleep=lambda s: None)
    with pytest.raises(ProviderError):
        client.call("x", "s", [_spec("a")], lambda spec: (_ for _ in ()).throw(RefusalError("no")))


def test_parse_json_loose():
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose('Here you go: {"a": [1,2]} thanks') == {"a": [1, 2]}
    assert parse_json_loose("nope") is None


def test_seats_mock_mode(settings, tmp_path: Path):
    tracker = CostTracker("p", tmp_path / "cost.json")
    seats = Seats(settings, tracker, fixture_key="the_panama_canal")
    assert seats.status()["writer"]["real"] is False
    img, is_mock = seats.image("A steam shovel in the Culebra Cut, 1913", tmp_path / "a.png", "visuals")
    assert is_mock and img.exists()
    score = seats.vision_json("judge", "judge", "sys", "prompt", [str(img)], "judge", intent="steam shovel")
    assert 0 <= score["score"] <= 100 and score["_mock"] is True
    audio, words, mock, engine = seats.tts("Hello there, this is a test of narration.", VoiceConfig(), tmp_path / "n.wav", "narration")
    assert mock and audio.exists() and len(words) == 8
    assert words[0]["start"] < words[-1]["end"]
    assert tracker.report.mock_calls >= 3
    assert (tmp_path / "cost.json").exists()


def test_image_sticky_provider(settings, tmp_path, monkeypatch):
    settings.offline = False
    settings.env.update({"IMAGE_BASE_URL": "http://a", "IMAGE_API_KEY": "k", "IMAGE_FALLBACK_BASE_URL": "http://b",
                         "IMAGE_FALLBACK_API_KEY": "k"})
    seats = Seats(settings, CostTracker("p"))
    from vidbush.providers import image as image_mod
    order = []

    def fake(spec, prompt, out_path, cost):
        order.append(spec.name)
        if spec.name == "image_primary":
            raise TransportError("down")
        out_path.write_bytes(b"x")
        return CallResult(value=out_path, provider=spec, mock=False, cost_usd=0.02)

    monkeypatch.setattr(image_mod, "generate", fake)
    seats.client.sleep = lambda s: None
    seats.image("p", tmp_path / "1.png", "visuals")
    assert seats._sticky["image"] == "image_fallback"
    order.clear()
    seats.image("p", tmp_path / "2.png", "visuals")
    assert order[0] == "image_fallback"
