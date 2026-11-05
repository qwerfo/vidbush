"""Native Anthropic Messages adapter, offline: canned JSON replies and canned SSE streams."""
from __future__ import annotations

import json
from typing import Any

import pytest

from vidbush.providers import llm_anthropic as an
from vidbush.providers import seats as seats_mod
from vidbush.providers.base import ProviderSpec, RateLimitError, RefusalError, TransportError
from vidbush.providers.llm import BadJSONError, image_part


@pytest.fixture(autouse=True)
def _fresh():
    an.reset_quirks()
    yield
    an.reset_quirks()


def _spec(**opts: Any) -> ProviderSpec:
    return ProviderSpec("anthropic_direct", "anthropic", "claude-sonnet-5-5", api_key="k", options=opts)


def _msg(text: str, stop: str = "end_turn", **usage: Any) -> dict[str, Any]:
    return {"type": "message", "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": text}],
            "stop_reason": stop, "usage": {"input_tokens": 100, "output_tokens": 50, **usage}}


class FakeResp:
    def __init__(self, status: int = 200, body: Any = None, lines: list[str] | None = None,
                 ctype: str = "text/event-stream") -> None:
        self.status_code, self._body, self._lines = status, body, lines or []
        self.headers = {"Content-Type": ctype}
        self.text = json.dumps(body) if body is not None else ""

    def json(self) -> Any:
        return self._body

    def iter_lines(self, decode_unicode: bool = True):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _sse(*events: dict[str, Any], ping: bool = True) -> list[str]:
    out = []
    for i, ev in enumerate(events):
        if ping and i == 1:
            out += ["event: ping", 'data: {"type": "ping"}', ""]
        out += [f"event: {ev['type']}", "data: " + json.dumps(ev), ""]
    return out


def _stream_events(text: str, stop: str = "end_turn", out_tokens: int = 40, done: bool = True) -> list[str]:
    evs = [{"type": "message_start", "message": {"usage": {"input_tokens": 1000, "output_tokens": 1,
                                                           "cache_read_input_tokens": 2000}}},
           {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
           {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
           {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}]
    half = len(text) // 2
    evs += [{"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text[:half]}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text[half:]}}]
    if done:
        evs += [{"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": out_tokens}},
                {"type": "message_stop"}]
    return _sse(*evs)


def test_non_streamed_json_request_shape_and_cost(monkeypatch) -> None:
    sent: dict = {}

    def post(url, json=None, headers=None, timeout=None, **kw):
        sent.update(url=url, body=json, headers=headers)
        return FakeResp(body=_msg('{"ok": true}', cache_read_input_tokens=1000), ctype="application/json")
    monkeypatch.setattr(an.requests, "post", post)
    r = an.chat(_spec(stream=False, effort="low"), "Be brief.", "hello", want_json=True)
    assert r.value == {"ok": True}
    body, h = sent["body"], sent["headers"]
    assert sent["url"] == "https://api.anthropic.com/v1/messages"
    assert body["system"].startswith("Be brief.") and an.JSON_INSTRUCTION in body["system"]
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert "temperature" not in body and "response_format" not in body and body["max_tokens"] == 16000
    assert body["output_config"] == {"effort": "low"}
    assert h["x-api-key"] == "k" and h["anthropic-version"] == an.API_VERSION
    # sonnet 5.5: 100 in at 2.00, 50 out at 10.00, 1000 cache reads at 0.20 per million
    assert r.cost_usd == pytest.approx((100 * 2.0 + 50 * 10.0 + 1000 * 0.20) / 1e6)
    assert r.units == 1150 and r.unit_kind == "tokens"


def test_stream_with_pings_folds_text_and_usage(monkeypatch) -> None:
    sent: dict = {}

    def post(url, json=None, headers=None, timeout=None, stream=False):
        sent.update(body=json, stream=stream)
        return FakeResp(lines=_stream_events('{"a": 1}'))
    monkeypatch.setattr(an.requests, "post", post)
    r = an.chat(_spec(), "s", "u", want_json=True)
    assert r.value == {"a": 1} and sent["body"]["stream"] is True and sent["stream"] is True
    assert sent["body"]["max_tokens"] == an.DEFAULT_MAX_TOKENS_STREAM
    assert r.cost_usd == pytest.approx((1000 * 2.0 + 40 * 10.0 + 2000 * 0.20) / 1e6)


def test_partial_stream_is_a_transport_error(monkeypatch) -> None:
    monkeypatch.setattr(an.requests, "post", lambda *a, **k: FakeResp(lines=_stream_events("abc", done=False)))
    with pytest.raises(TransportError, match="partial"):
        an.chat(_spec(), "s", "u", want_json=False)


def test_stream_error_events_map_to_provider_errors(monkeypatch) -> None:
    for kind, exc in (("overloaded_error", TransportError), ("rate_limit_error", RateLimitError),
                      ("invalid_request_error", RefusalError)):
        lines = _sse({"type": "message_start", "message": {"usage": {}}},
                     {"type": "error", "error": {"type": kind, "message": "x"}}, ping=False)
        monkeypatch.setattr(an.requests, "post", lambda *a, _l=lines, **k: FakeResp(lines=_l))
        with pytest.raises(exc):
            an.chat(_spec(), "s", "u", want_json=False)


def test_truncated_reply_is_continued_once(monkeypatch) -> None:
    replies = iter([_msg('{"text": "the canal', stop="max_tokens"), _msg(' opened"}')])
    bodies: list = []

    def post(url, json=None, headers=None, timeout=None, **kw):
        bodies.append(json)
        return FakeResp(body=next(replies), ctype="application/json")
    monkeypatch.setattr(an.requests, "post", post)
    r = an.chat(_spec(stream=False), "s", "u", want_json=True)
    assert r.value == {"text": "the canal opened"}
    assert [m["role"] for m in bodies[1]["messages"]] == ["user", "assistant", "user"]
    assert r.units == 300


def test_refusal_and_bad_json(monkeypatch) -> None:
    monkeypatch.setattr(an.requests, "post", lambda *a, **k: FakeResp(body=_msg("", stop="refusal"),
                                                                      ctype="application/json"))
    with pytest.raises(RefusalError, match="refusal"):
        an.chat(_spec(stream=False), "s", "u", want_json=True)
    calls: list = []
    monkeypatch.setattr(an.requests, "post", lambda *a, **k: calls.append(1) or FakeResp(
        body=_msg("not json at all"), ctype="application/json"))
    with pytest.raises(BadJSONError) as e:
        an.chat(_spec(stream=False), "s", "u", want_json=True)
    assert len(calls) == 2 and e.value.cost_usd > 0  # asked again once, both replies paid


def test_quirks_learned_from_400s(monkeypatch) -> None:
    bodies: list = []
    replies = iter([FakeResp(400, {"error": {"type": "invalid_request_error",
                                             "message": "output_config.effort: unsupported value"}}),
                    FakeResp(400, {"error": {"message": "max_tokens: 64000 > 32000, which is the maximum"}}),
                    FakeResp(body=_msg("plain"), ctype="application/json")])

    def post(url, json=None, headers=None, timeout=None, **kw):
        bodies.append(json)
        return next(replies)
    monkeypatch.setattr(an.requests, "post", post)
    r = an.chat(_spec(stream=False, effort="max", max_tokens=64000), "s", "u", want_json=False)
    assert r.value == "plain"
    assert "output_config" in bodies[0] and "output_config" not in bodies[1]
    assert bodies[2]["max_tokens"] == 32000


def test_policy_400_never_teaches_a_quirk(monkeypatch) -> None:
    monkeypatch.setattr(an.requests, "post", lambda *a, **k: FakeResp(
        400, {"error": {"message": "request flagged by content policy (temperature)"}}))
    with pytest.raises(RefusalError):
        an.chat(_spec(stream=False), "s", "u", want_json=False)
    assert an._quirks(_spec()) == set()


def test_vision_parts_become_image_blocks(tmp_path) -> None:
    from PIL import Image
    p = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(p)
    content = an.to_content([{"type": "text", "text": "score"}, image_part(p)])
    assert content[0] == {"type": "text", "text": "score"}
    assert content[1]["type"] == "image" and content[1]["source"]["type"] == "base64"
    assert content[1]["source"]["media_type"] == "image/png"


def test_price_table_and_override() -> None:
    assert an.prices(_spec())["prompt"] == 2.0
    opus = ProviderSpec("a", "anthropic", "claude-opus-5-5", api_key="k")
    assert an.prices(opus) == an.PRICES_PER_MTOK["claude-opus-5-5"]
    dated = ProviderSpec("a", "anthropic", "claude-opus-5-20260101", api_key="k")
    assert an.prices(dated) == an.PRICES_PER_MTOK["claude-opus-5"]
    assert an.prices(_spec(price_per_mtok={"prompt": 1.0, "completion": 2.0}))["completion"] == 2.0
    assert an.prices(ProviderSpec("a", "anthropic", "unknown", api_key="k")) == {}


def test_seats_dispatch_anthropic_type(monkeypatch) -> None:
    called: list = []
    monkeypatch.setattr(an, "chat", lambda spec, *a, **k: called.append(spec.type))
    seats_mod.chat_fn(_spec())(_spec(), "s", "u", want_json=True)
    assert called == ["anthropic"]
    assert _spec().configured and not ProviderSpec("a", "anthropic", "", api_key="k").configured
