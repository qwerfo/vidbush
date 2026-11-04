"""Provider fixes: mock fallback policy, 429 backoff, cost estimates, placeholders, adapters, probes."""
from __future__ import annotations

import base64
import io
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from vidbush.models import VoiceConfig
from vidbush.providers import base, image, llm, probe, tts
from vidbush.providers.base import (CallResult, CostTracker, ProviderClient, ProviderError, ProviderSpec,
                                    RateLimitError, RefusalError, TransportError, http_error_to_exc)
from vidbush.providers.seats import Seats


@pytest.fixture(autouse=True)
def _clean():
    base.HEALTH._down.clear()
    llm.reset_quirks()
    tts._MODEL_CACHE.clear()
    yield
    base.HEALTH._down.clear()
    llm.reset_quirks()
    tts._MODEL_CACHE.clear()


def _spec(name: str = "a", **kw) -> ProviderSpec:
    return ProviderSpec(name=name, type=kw.pop("type", "openai_compat"), model=kw.pop("model", "m"),
                        base_url="http://x", api_key="k", **kw)


def _llm_env(settings) -> None:
    settings.offline = False
    settings.env.update({"LLM_PRIMARY_BASE_URL": "http://a", "LLM_PRIMARY_API_KEY": "k", "LLM_PRIMARY_MODEL": "m1",
                         "LLM_WRITER_MODEL": "w1"})


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 36), (10, 20, 30)).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- mock fallback policy

def test_real_role_failure_raises_instead_of_mock(settings, monkeypatch):
    _llm_env(settings)
    seats = Seats(settings, CostTracker("p"))
    seats.client.sleep = lambda s: None
    monkeypatch.setattr(llm, "chat", lambda *a, **k: (_ for _ in ()).throw(RefusalError("http 404: no model")))
    with pytest.raises(ProviderError):
        seats.llm_json("writer", "script", "s", "u", "script")
    assert "writer" not in seats.mock_roles
    assert not any(e.mock for e in seats.tracker.report.entries)


def test_allow_mock_fallback_uses_mock_and_flags_review(settings, monkeypatch):
    _llm_env(settings)
    settings.allow_mock_fallback = True
    seats = Seats(settings, CostTracker("p"))
    monkeypatch.setattr(llm, "chat", lambda *a, **k: (_ for _ in ()).throw(RefusalError("no")))
    seats.llm_json("writer", "script", "s", "u", "script")
    assert "writer" in seats.mock_roles and "writer" in seats.fallback_roles
    assert seats.tracker.report.mock_calls == 1
    assert seats.used_mock_with_real_seats()


def test_unconfigured_role_uses_mock_without_error(settings, tmp_path):
    _llm_env(settings)
    seats = Seats(settings, CostTracker("p"))
    img, is_mock = seats.image("x", tmp_path / "a.png", "visuals")
    assert is_mock and "image" in seats.mock_roles and "image" not in seats.fallback_roles
    assert seats.used_mock_with_real_seats()  # writer is real, image used mock: needs_review


def test_offline_mock_is_not_review(settings, tmp_path):
    seats = Seats(settings, CostTracker("p"))
    seats.image("x", tmp_path / "a.png", "visuals")
    assert not seats.used_mock_with_real_seats()
    assert seats.mock_policy()["offline"] is True


# ---------------------------------------------------------------- placeholders

def test_placeholder_models_do_not_count_as_configured(settings):
    settings.offline = False
    settings.env.update({"LLM_PRIMARY_BASE_URL": "http://a", "LLM_PRIMARY_API_KEY": "k"})
    seats = Seats(settings, CostTracker("p"))
    assert not seats.is_real("writer")
    assert all(p.model == "" for p in seats.providers_for("writer"))
    assert seats.status()["writer"]["providers"][0]["down_since"] is None
    assert not ProviderSpec(name="x", type="openai_compat", model="your-model", base_url="u", api_key="k").configured
    assert ProviderSpec(name="t", type="elevenlabs", model="auto", api_key="k").configured
    assert ProviderSpec(name="s", type="tavily", model="", api_key="k").configured


# ---------------------------------------------------------------- 429 handling

def test_rate_limit_honours_retry_after_and_never_marks_down():
    waits: list[float] = []
    client = ProviderClient(CostTracker("p"), sleep=waits.append)
    n = {"c": 0}

    def fn(spec):
        n["c"] += 1
        if n["c"] < 5:
            raise RateLimitError("http 429", retry_after=2.0)
        return CallResult(value="ok", provider=spec, mock=False)

    assert client.call("tts", "narration", [_spec()], fn).value == "ok"
    assert waits == [2.0, 2.0, 2.0, 2.0]
    assert not base.HEALTH.is_down("a:m")


def test_rate_limit_exhausted_moves_on_without_down():
    client = ProviderClient(CostTracker("p"), sleep=lambda s: None)
    with pytest.raises(ProviderError):
        client.call("tts", "n", [_spec()], lambda s: (_ for _ in ()).throw(RateLimitError("429")))
    assert not base.HEALTH.is_down("a:m")
    assert len(client.tracker.report.entries) == base.RATE_LIMIT_RETRIES


def test_rate_limit_backoff_caps_at_max():
    client = ProviderClient(CostTracker("p"), backoff_s=10.0, sleep=lambda s: None)
    assert client._rate_wait(RateLimitError("x"), 5) == base.RATE_LIMIT_MAX_WAIT_S
    assert client._rate_wait(RateLimitError("x", retry_after=500), 1) == base.RATE_LIMIT_MAX_WAIT_S


def test_http_429_maps_to_rate_limit_with_retry_after():
    e = http_error_to_exc(429, "slow down", "7")
    assert isinstance(e, RateLimitError) and e.retry_after == 7.0
    assert isinstance(http_error_to_exc(503, ""), TransportError)
    assert not isinstance(http_error_to_exc(503, ""), RateLimitError)
    assert base.parse_retry_after("soon") is None


def test_max_concurrency_limits_parallel_calls():
    client = ProviderClient(CostTracker("p"), sleep=lambda s: None)
    spec = _spec(options={"max_concurrency": 2})
    live = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def fn(s):
        with lock:
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
        time.sleep(0.05)
        with lock:
            live["now"] -= 1
        return CallResult(value=1, provider=s, mock=False)

    threads = [threading.Thread(target=client.call, args=("tts", "n", [spec], fn)) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert live["peak"] == 2


# ---------------------------------------------------------------- cost estimates

def test_llm_and_vision_estimates_reach_cost_entry(settings, monkeypatch, tmp_path):
    _llm_env(settings)
    settings.env.update({"VISION_BASE_URL": "http://v", "VISION_API_KEY": "k", "VISION_MODEL": "vm"})
    seats = Seats(settings, CostTracker("p"))
    monkeypatch.setattr(llm, "chat", lambda spec, *a, **k: CallResult(value={"score": 80}, provider=spec, mock=False,
                                                                      units=10000, unit_kind="tokens"))
    seats.llm_json("writer", "script", "s", "u", "script")
    e = seats.tracker.report.entries[-1]
    assert e.cost_usd == pytest.approx(10 * settings.cost.llm_per_1k_tokens_usd)
    img = tmp_path / "i.png"
    img.write_bytes(_png())
    monkeypatch.setattr(llm, "chat", lambda spec, *a, **k: CallResult(value={"score": 80}, provider=spec, mock=False))
    seats.vision_json("judge", "judge", "s", "p", [str(img), str(img)], "judge")
    assert seats.tracker.report.entries[-1].cost_usd == pytest.approx(2 * settings.cost.vision_per_image_usd)


# ---------------------------------------------------------------- image adapters

def test_image_retries_without_size(monkeypatch, tmp_path):
    bodies = []

    def fake_post(url, body, headers, timeout=0):
        bodies.append(dict(body))
        if "size" in body:
            raise RefusalError("http 400: Unrecognized request argument supplied: size")
        return {"data": [{"b64_json": base64.b64encode(_png()).decode()}]}

    monkeypatch.setattr(image, "post_json", fake_post)
    res = image.generate_openai(_spec(model="flux"), "p", tmp_path / "o.png", 0.02)
    assert "size" not in bodies[-1] and len(bodies) == len(image.SIZES_16x9) + 1
    assert Image.open(res.value).size == (1920, 1080)


def test_gpt_image_quality_and_cost(monkeypatch, tmp_path):
    spec = _spec(model="gpt-image-1", options={"cost_per_image": 0.02})
    assert image.estimate_cost(spec, 0.02) == image.GPT_IMAGE_COST["gpt-image-1"]["low"]
    assert image.estimate_cost(_spec(model="gpt-image-1", options={"quality": "high"}), 0.02) > 0.05
    sent = {}

    def fake_post(url, body, headers, timeout=0):
        sent.update(body)
        return {"data": [{"b64_json": base64.b64encode(_png()).decode()}],
                "usage": {"input_tokens": 100, "output_tokens": 1000}}

    monkeypatch.setattr(image, "post_json", fake_post)
    res = image.generate_openai(spec, "p", tmp_path / "o.png", 0.02)
    assert sent["quality"] == "low"
    assert res.cost_usd == pytest.approx((100 * 5 + 1000 * 40) / 1e6)


def test_image_ceiling_uses_quality_table(settings, tmp_path, monkeypatch):
    settings.offline = False
    settings.env.update({"IMAGE_BASE_URL": "http://a", "IMAGE_API_KEY": "k", "IMAGE_MODEL": "gpt-image-1"})
    settings.seats["providers"]["image_primary"]["options"] = {"quality": "high", "cost_per_image": 0.02}
    seats = Seats(settings, CostTracker("p"))
    monkeypatch.setattr(image, "generate", lambda *a, **k: pytest.fail("must not call above ceiling"))
    with pytest.raises(ProviderError):
        seats.image("p", tmp_path / "x.png", "visuals")


def test_openrouter_chat_image_mode(monkeypatch, tmp_path):
    sent = {}
    data_url = "data:image/png;base64," + base64.b64encode(_png()).decode()

    def fake_post(url, body, headers, timeout=0):
        sent["url"], sent["body"] = url, body
        return {"choices": [{"message": {"content": "", "images": [{"image_url": {"url": data_url}}]}}],
                "usage": {"cost": 0.03}}

    monkeypatch.setattr(image, "post_json", fake_post)
    res = image.generate_openai(_spec(model="google/gemini-image", options={"mode": "chat"}), "p",
                                tmp_path / "o.png", 0.02)
    assert sent["url"].endswith("/chat/completions") and sent["body"]["modalities"] == ["image", "text"]
    assert sent["body"]["image_config"]["aspect_ratio"] == "16:9"
    assert res.cost_usd == 0.03 and Image.open(res.value).size == (1920, 1080)


# ---------------------------------------------------------------- tts

def test_tts_default_voice_language_and_model(settings, monkeypatch, tmp_path):
    settings.offline = False
    settings.env.update({"ELEVENLABS_API_KEY": "k"})
    # the module default preference, independent of what config/seats.yaml ships
    settings.seats["providers"]["elevenlabs"].setdefault("options", {})["model_preference"] = tts.MODEL_PREFERENCE
    tts._MODEL_CACHE.clear()
    seats = Seats(settings, CostTracker("p"))
    assert seats.voice_id(VoiceConfig()) == tts.DEFAULT_VOICE_ID
    settings.env["ELEVENLABS_VOICE_ID"] = "envvoice"
    assert seats.voice_id(VoiceConfig()) == "envvoice"
    assert seats.voice_id(VoiceConfig(voice_id="brandvoice")) == "brandvoice"
    sent = {}

    def fake_post(url, body, headers, timeout=0):
        sent["url"], sent["body"] = url, body
        return {"audio_base64": base64.b64encode(b"mp3").decode(),
                "alignment": {"characters": list("hi"), "character_start_times_seconds": [0, 0.1],
                              "character_end_times_seconds": [0.1, 0.2]}}

    monkeypatch.setattr(tts, "post_json", fake_post)
    monkeypatch.setattr(tts, "get_json", lambda *a, **k: [{"model_id": "eleven_v3"}, {"model_id": "eleven_turbo_v2_5"}])
    _, words, is_mock, _ = seats.tts("hi", VoiceConfig(), tmp_path / "c.wav", "narration", language="en")
    assert not is_mock and words[0]["text"] == "hi" and "/envvoice/" in sent["url"]
    assert sent["body"]["model_id"] == "eleven_turbo_v2_5" and sent["body"]["language_code"] == "en"


def test_tts_body_rules():
    spec = ProviderSpec(name="e", type="elevenlabs", api_key="k")
    b = tts.build_body(spec, "x", "eleven_multilingual_v2", None, language="en", previous_text="p")
    assert "language_code" not in b and b["previous_text"] == "p"
    b = tts.build_body(spec, "x", "eleven_v3", {"stability": 0.45, "similarity_boost": 0.8}, previous_text="p")
    assert b["voice_settings"]["stability"] == 0.5 and "previous_text" not in b
    with pytest.raises(RefusalError):
        tts.build_body(spec, "x" * (tts.V3_MAX_CHARS + 1), "eleven_v3", None)


def test_tts_failed_model_lookup_is_not_cached(monkeypatch):
    spec = ProviderSpec(name="e", type="elevenlabs", api_key="k")
    monkeypatch.setattr(tts, "get_json", lambda *a, **k: (_ for _ in ()).throw(TransportError("x")))
    assert tts.pick_model(spec, "auto") == tts.FALLBACK_MODEL
    assert "e" not in tts._MODEL_CACHE


def test_default_brand_has_premade_voice(settings):
    from vidbush.config import load_brand
    assert load_brand(settings, "default").voice.voice_id == tts.DEFAULT_VOICE_ID


# ---------------------------------------------------------------- llm quirks

def test_quirk_learner_ignores_policy_errors_and_is_keyed_by_base_url():
    key = ("http://x", "m")
    assert not llm._learn(key, "http 400: flagged by our safety system for role content", set())
    assert llm._learn(key, 'http 400: {"error": {"param": "temperature", "message": "bad"}}', set())
    assert llm._quirks(key) == {"temperature"}
    assert llm._quirks(("http://other", "m")) == set()
    assert llm._learn(key, "'messages' must contain the word 'json' in some form, to use 'response_format'", set())
    assert "json_word" in llm._quirks(key) and "response_format" not in llm._quirks(key)
    assert not llm._learn(key, "invalid value for messages[0].content", set())


def test_json_word_quirk_adds_instruction():
    msgs = llm._messages("sys", "u", {"json_word"})
    assert "JSON" in msgs[0]["content"]


# ---------------------------------------------------------------- probe and doctor

def test_probe_reports_each_role(settings, monkeypatch):
    _llm_env(settings)
    seats = Seats(settings, CostTracker("p"))
    monkeypatch.setattr(llm, "chat", lambda *a, **k: CallResult(value={"ok": True}, provider=_spec(), mock=False))
    out = probe.probe_seats(seats)
    assert out["writer"]["ok"] is True and out["writer"]["model"] == "w1"
    assert out["image"]["ok"] is None and "mock" in out["image"]["note"]
    assert probe.last_results()["writer"]["ok"] is True
    monkeypatch.setattr(llm, "chat", lambda *a, **k: (_ for _ in ()).throw(RefusalError("http 401")))
    assert probe.probe_seats(seats)["writer"]["ok"] is False


def test_doctor_probe_skipped_offline(settings):
    from vidbush.doctor import doctor
    d = doctor(settings, probe=True)
    assert d["probe"] == "skipped: offline" and d["mock_policy"]["allow_mock_fallback"] is False
    assert d["tts_voice_id"] == tts.DEFAULT_VOICE_ID


def test_settings_route_reports_policy_and_probe(settings, tmp_path):
    import shutil

    from fastapi.testclient import TestClient

    from vidbush.api.app import create_app
    from vidbush.config import REPO_ROOT
    cfg = tmp_path / "config"
    shutil.copytree(Path(REPO_ROOT) / "config", cfg)
    settings.config_folder = str(cfg)
    with TestClient(create_app(settings, start_workers=False)) as c:
        s = c.get("/api/settings", params={"probe": "true"}).json()
    assert s["mock_policy"]["allow_mock_fallback"] is False and s["mock_policy"]["offline"] is True
    assert s["seats"]["tts"]["configured"] is False and "probe" in s["seats"]["tts"]
    assert s["seats"]["writer"]["providers"] == []  # offline resolves no real providers
    assert s["doctor"]["probe"] == "skipped: offline"
