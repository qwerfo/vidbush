"""Streaming is the default for openai_compat providers; post_stream folds what live relays send."""
from __future__ import annotations

import json
from typing import Any

import pytest

from vidbush.providers import http as http_mod
from vidbush.providers import llm as llm_mod
from vidbush.providers.base import ProviderSpec, RefusalError, TransportError


class R:
    def __init__(self, lines: list[str] | None = None, status: int = 200, ctype: str = "text/event-stream",
                 body: Any = None) -> None:
        self.lines, self.status_code, self.body = lines or [], status, body
        self.headers = {"Content-Type": ctype}
        self.text = json.dumps(body) if body is not None else ""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self, decode_unicode=True):
        yield from self.lines

    def json(self):
        return self.body


def _chunk(content: Any = None, finish: Any = None, usage: Any = None, choices: bool = True) -> str:
    obj: dict[str, Any] = {"choices": [{"delta": {"content": content} if content is not None else {},
                                        "finish_reason": finish}] if choices else []}
    if usage is not None:
        obj["usage"] = usage
    return "data: " + json.dumps(obj)


def test_pings_and_usage_in_the_final_chunk(monkeypatch) -> None:
    sent: dict = {}
    lines = [": ping", "", _chunk('{"a"'), ": ping", "", _chunk(": 1}"), _chunk(finish="stop"),
             _chunk(usage={"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12, "cost": 0.01},
                    choices=False), "data: [DONE]"]
    monkeypatch.setattr(http_mod.requests, "post", lambda url, json=None, **k: sent.update(json) or R(lines))
    out = http_mod.post_stream("https://r/v1/chat/completions", {"model": "m"})
    assert out["choices"][0]["message"]["content"] == '{"a": 1}' and out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["total_tokens"] == 12
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}


def test_finish_without_done_is_complete_but_no_finish_is_partial(monkeypatch) -> None:
    monkeypatch.setattr(http_mod.requests, "post", lambda *a, **k: R([_chunk("ok"), _chunk(finish="stop")]))
    assert http_mod.post_stream("u", {})["choices"][0]["message"]["content"] == "ok"
    monkeypatch.setattr(http_mod.requests, "post", lambda *a, **k: R([_chunk("half"), ": ping"]))
    with pytest.raises(TransportError, match="partial"):
        http_mod.post_stream("u", {})


def test_error_event_and_plain_json_reply(monkeypatch) -> None:
    monkeypatch.setattr(http_mod.requests, "post",
                        lambda *a, **k: R([_chunk("x"), 'data: {"error": {"message": "upstream 502"}}']))
    with pytest.raises(TransportError, match="stream error"):
        http_mod.post_stream("u", {})
    body = {"choices": [{"message": {"content": "whole"}, "finish_reason": "stop"}], "usage": {}}
    monkeypatch.setattr(http_mod.requests, "post", lambda *a, **k: R(ctype="application/json", body=body))
    assert http_mod.post_stream("u", {}) == body


def test_list_content_deltas_and_usage_on_every_chunk(monkeypatch) -> None:
    lines = [_chunk([{"type": "text", "text": "a"}], usage=None), _chunk("b", usage={"total_tokens": 1}),
             _chunk(finish="stop", usage={"total_tokens": 5}), "data: [DONE]"]
    monkeypatch.setattr(http_mod.requests, "post", lambda *a, **k: R(lines))
    out = http_mod.post_stream("u", {}, include_usage=False)
    assert out["choices"][0]["message"]["content"] == "ab" and out["usage"]["total_tokens"] == 5


def _spec(**opts: Any) -> ProviderSpec:
    return ProviderSpec("p", "openai_compat", "m", base_url="https://r/v1", api_key="k", options=opts)


def test_stream_is_the_default_and_false_is_honoured(monkeypatch) -> None:
    used: list = []
    reply = {"choices": [{"message": {"content": '{"ok": 1}'}, "finish_reason": "stop"}], "usage": {}}
    monkeypatch.setattr(llm_mod, "post_stream", lambda *a, **k: used.append("stream") or reply)
    monkeypatch.setattr(llm_mod, "post_json", lambda *a, **k: used.append("json") or reply)
    llm_mod.chat(_spec(), "s", "u", want_json=True)
    llm_mod.chat(_spec(stream=False), "s", "u", want_json=True)
    llm_mod.chat(_spec(stream="false"), "s", "u", want_json=True)
    assert used == ["stream", "json", "json"]


def test_relay_rejecting_stream_options_is_retried_without_it(monkeypatch) -> None:
    llm_mod.reset_quirks()
    seen: list = []
    reply = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {}}

    def post(url, body, headers=None, timeout=0.0, include_usage=True):
        seen.append(include_usage)
        if include_usage:
            raise RefusalError('http 400: {"error": {"message": "Unknown parameter: stream_options"}}')
        return reply
    monkeypatch.setattr(llm_mod, "post_stream", post)
    assert llm_mod.chat(_spec(), "s", "u", want_json=False).value == "hi"
    assert seen == [True, False]
    llm_mod.reset_quirks()
