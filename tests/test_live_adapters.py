"""Adapters and quirks added on the first live run (offline, canned replies)."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from vidbush.jobs.paths import rebase_path
from vidbush.providers import image as image_mod
from vidbush.providers import llm as llm_mod
from vidbush.providers import tts as tts_mod
from vidbush.providers.base import ProviderSpec, RefusalError, TransportError

pytestmark = pytest.mark.usefixtures("generic_seats")


def _png(w: int = 1376, h: int = 768) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 120, 160)).save(buf, "PNG")
    return buf.getvalue()


def test_elevate_submit_poll_download(tmp_path: Path, monkeypatch) -> None:
    replies = iter([{"success": True, "data": {"id": 7, "status": "processing"}},
                    {"success": True, "data": {"id": 7, "status": "completed", "result_url": "https://x/r.png"}}])
    sent: dict = {}
    monkeypatch.setattr(image_mod, "post_json", lambda url, body, headers, timeout: sent.update(body) or next(replies))
    monkeypatch.setattr(image_mod, "get_json", lambda url, headers, timeout: next(replies))
    monkeypatch.setattr(image_mod, "get_bytes", lambda url: _png())
    monkeypatch.setattr(image_mod.time, "sleep", lambda s: None)
    spec = ProviderSpec("e", "elevate", "gpt-image-2.5-fast", api_key="k", options={"cost_per_image": 0.0, "edge_trim": 0.05})
    r = image_mod.generate(spec, "trucks", tmp_path / "a.png", 0.02)
    assert sent["type"] == "image" and sent["aspect_ratio"] == "landscape"
    assert Image.open(r.value).size == (1920, 1080) and r.cost_usd == 0.0


def test_elevate_envelope_errors(monkeypatch) -> None:
    with pytest.raises(RefusalError):
        image_mod._elevate_data({"success": False, "error": {"code": "INVALID_REQUEST", "message": "bad"}}, "w")
    with pytest.raises(TransportError):
        image_mod._elevate_data({"success": False, "error": {"code": "X", "message": "Request timed out, please retry"}}, "w")


def test_algrow_job_image_urls(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(image_mod, "post_json", lambda *a, **k: {"success": True, "job_id": "j1", "status": "pending"})
    jobs = iter([{"status": "processing"}, {"status": "completed", "image_urls": ["https://x/i.png"]}])
    monkeypatch.setattr(image_mod, "get_json", lambda *a, **k: next(jobs))
    monkeypatch.setattr(image_mod, "get_bytes", lambda url: _png(1672, 941))
    monkeypatch.setattr(image_mod.time, "sleep", lambda s: None)
    spec = ProviderSpec("a", "algrow", "gpt-image-2.5", api_key="k", options={"cost_per_image": 0.01085})
    r = image_mod.generate(spec, "trucks", tmp_path / "b.png", 0.02)
    assert r.cost_usd == 0.01085 and Image.open(r.value).size == (1920, 1080)


def test_gpt_image_table_only_prices_known_models() -> None:
    assert image_mod._gpt_family("gpt-image-1-mini") == "gpt-image-1-mini"
    assert image_mod._gpt_family("gpt-image-2.5-fast") == ""
    spec = ProviderSpec("e", "elevate", "gpt-image-2.5-fast", api_key="k", options={"cost_per_image": 0.0})
    assert image_mod.estimate_cost(spec, 0.02) == 0.0


def test_price_usage_from_catalogue_with_cache_rates(monkeypatch) -> None:
    llm_mod._PRICES.clear()
    monkeypatch.setattr(llm_mod, "get_json", lambda *a, **k: {"data": [{"id": "anthropic/claude-sonnet-5", "pricing": {
        "prompt": "1.0", "completion": "5.0", "cache_read": "0.1", "cache_write_5m": "1.25",
        "unit": "usd_per_million_tokens"}}]})
    spec = ProviderSpec("p", "openai_compat", "anthropic/claude-sonnet-5", base_url="https://r/v1", api_key="k")
    usage = {"prompt_tokens": 10_000, "completion_tokens": 1_000,
             "prompt_tokens_details": {"cached_tokens": 8_000, "cached_creation_tokens": 1_000}}
    # 1000 fresh at 1.0, 8000 read at 0.1, 1000 written at 1.25, 1000 out at 5.0 (per million)
    assert llm_mod.price_usage(spec, usage) == pytest.approx((1000 * 1.0 + 8000 * 0.1 + 1000 * 1.25 + 1000 * 5.0) / 1e6)
    llm_mod._PRICES.clear()


def test_json_inside_agent_scaffold() -> None:
    text = "I'll write {it}.\n<write_file><path>a.json</path><content>{\"a\": [1]}</content></write_file>"
    assert llm_mod.parse_json_loose(text) == {"a": [1]}


def test_tts_auto_model_refusal_switches_to_fallback_for_the_run(tmp_path: Path, monkeypatch) -> None:
    tts_mod._MODEL_CACHE.clear()
    spec = ProviderSpec("el", "elevenlabs", api_key="k", options={"model_preference": ["eleven_v4"]})
    monkeypatch.setattr(tts_mod, "get_json", lambda *a, **k: [{"model_id": "eleven_v4"}])
    models: list[str] = []

    def post(url, body, headers, timeout):
        models.append(body["model_id"])
        if body["model_id"] == "eleven_v4":
            raise RefusalError("http 400: text too long for this model")
        return {"audio_base64": "AAAA", "alignment": {}}

    monkeypatch.setattr(tts_mod, "post_json", post)
    tts_mod.synthesize(spec, "Hello there.", "v", "auto", tmp_path / "a.mp3")
    tts_mod.synthesize(spec, "Again.", "v", "auto", tmp_path / "b.mp3")
    assert models == ["eleven_v4", "eleven_multilingual_v2", "eleven_multilingual_v2"]
    tts_mod._MODEL_CACHE.clear()


def test_rebase_keeps_target_separator_style() -> None:
    assert rebase_path("C:\\vb\\output\\demo\\work\\x.wav", "demo", "/app/output/demo") == "/app/output/demo/work/x.wav"
    assert rebase_path("/app/output/demo/work/x.wav", "demo", "C:\\vidbush\\output\\demo") == \
        "C:\\vidbush\\output\\demo\\work\\x.wav"


def test_bundled_sample_library_never_feeds_a_real_judge(settings) -> None:
    from vidbush.source.local import LocalAdapter
    settings.sources.local_library = "fixtures/sample_assets"
    assert LocalAdapter().available(settings)  # offline: mock run, fixtures allowed
    settings.offline = False
    settings.env.update({"VISION_BASE_URL": "https://r/v1", "VISION_API_KEY": "k", "VISION_MODEL": "m"})
    assert not LocalAdapter().available(settings)


def test_bot_check_page_is_a_challenge() -> None:
    from vidbush.source.base import is_challenge
    assert is_challenge(RefusalError("http 403: <!DOCTYPE html><title>Just a moment...</title>"))
    assert not is_challenge(RefusalError("http 403: forbidden"))


def test_malformed_json_is_asked_again_on_the_same_provider(monkeypatch) -> None:
    replies = iter([{"choices": [{"message": {"content": '{"ideas": [{"a": 1}'}, "finish_reason": "stop"}],
                     "usage": {"cost": 0.002}},
                    {"choices": [{"message": {"content": '{"ideas": []}'}}], "usage": {"cost": 0.003}}])
    monkeypatch.setattr(llm_mod, "post_json", lambda *a, **k: next(replies))
    spec = ProviderSpec("p", "openai_compat", "m", base_url="https://r/v1", api_key="k", options={"stream": False})
    res = llm_mod.chat(spec, "s", "u", want_json=True)
    assert res.value == {"ideas": []} and res.cost_usd == pytest.approx(0.005)
    broken = {"choices": [{"message": {"content": "{broken"}, "finish_reason": "stop"}], "usage": {}}
    monkeypatch.setattr(llm_mod, "post_json", lambda *a, **k: broken)
    with pytest.raises(llm_mod.BadJSONError, match="finish stop"):
        llm_mod.chat(spec, "s", "u", want_json=True)
    cut = {"choices": [{"message": {"content": "{broken"}, "finish_reason": "length"}], "usage": {}}
    monkeypatch.setattr(llm_mod, "post_json", lambda *a, **k: cut)
    with pytest.raises(RefusalError, match="truncated twice"):  # continued once, then fails over
        llm_mod.chat(spec, "s", "u", want_json=True)


def test_stalled_image_job_falls_back_instead_of_resubmitting(tmp_path: Path, monkeypatch) -> None:
    submits = []
    monkeypatch.setattr(image_mod, "post_json", lambda *a, **k: submits.append(1) or {"success": True, "job_id": "j"})
    monkeypatch.setattr(image_mod, "get_json", lambda *a, **k: {"status": "processing"})
    monkeypatch.setattr(image_mod.time, "sleep", lambda s: None)
    spec = ProviderSpec("a", "algrow", "gpt-image-2.5", api_key="k", options={"timeout_s": 0})
    with pytest.raises(RefusalError, match="not finished"):
        image_mod.generate(spec, "x", tmp_path / "c.png", 0.02)
    assert submits == [1]


def test_stream_cut_before_done_is_a_transport_error(monkeypatch) -> None:
    from vidbush.providers import http as http_mod

    class R:
        status_code = 200
        headers: dict = {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_lines(self, decode_unicode=True):
            yield 'data: {"choices": [{"delta": {"content": "half a"}}]}'

    monkeypatch.setattr(http_mod.requests, "post", lambda *a, **k: R())
    with pytest.raises(TransportError, match="before"):
        http_mod.post_stream("https://r/v1/chat/completions", {"model": "m"})


def test_negatives_become_prose_for_providers_without_a_field(tmp_path: Path, monkeypatch) -> None:
    sent: dict = {}
    monkeypatch.setattr(image_mod, "post_json", lambda url, body, headers, timeout: sent.update(body) or
                        {"success": True, "data": {"id": 1, "status": "completed", "result_url": "u"}})
    monkeypatch.setattr(image_mod, "get_bytes", lambda url: _png())
    spec = ProviderSpec("e", "elevate", "gpt-image-2.5-fast", api_key="k", options={"cost_per_image": 0.0})
    image_mod.generate(spec, "A dog sled.", tmp_path / "n.png", 0.02, negative="text, logos")
    assert sent["prompt"] == "A dog sled. Do not include: text, logos."
    fal = ProviderSpec("f", "fal", "m", api_key="k")
    monkeypatch.setattr(image_mod, "post_json", lambda url, body, headers, timeout: sent.update(body) or
                        {"images": [{"url": "u"}]})
    image_mod.generate(fal, "A dog sled.", tmp_path / "f.png", 0.02, negative="text")
    assert sent["prompt"] == "A dog sled." and sent["negative_prompt"] == "text"


def test_ai_prompt_is_rewritten_without_named_people(make_ctx, monkeypatch) -> None:
    from vidbush.visuals import ai_images
    ctx = make_ctx()
    seen = {}

    def fake_llm(role, task, system, user, stage, temperature=0.6, timeout=0):
        seen["role"], seen["user"] = role, user
        return {"prompt": "An anonymous football mother in a Florida living room holding fundraising flyers."}

    monkeypatch.setattr(ctx.seats, "llm_json", fake_llm)
    assert "anonymous" in ai_images.safe_prompt("Cathy Parker speaks on WJXT news.", ctx)
    assert seen["role"] == "editor" and "Cathy Parker" in seen["user"]
    monkeypatch.setattr(ctx.seats, "llm_json", lambda *a, **k: None)
    assert ai_images.safe_prompt("A dog sled at dawn.", ctx) == "A dog sled at dawn."
