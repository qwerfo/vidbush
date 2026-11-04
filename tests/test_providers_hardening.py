"""LLM provider hardening: optional max_tokens, per role timeouts, one continuation of truncated replies."""
from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from vidbush.providers import base, llm
from vidbush.providers.base import CostTracker, ProviderError, ProviderSpec, RefusalError
from vidbush.providers.seats import Seats


@pytest.fixture(autouse=True)
def _clean():
    base.HEALTH._down.clear()
    llm.reset_quirks()
    yield
    base.HEALTH._down.clear()
    llm.reset_quirks()


def _env(settings, fallback: bool = False) -> None:
    settings.offline = False
    settings.seats = copy.deepcopy(settings.seats)
    settings.env.update({"LLM_PRIMARY_BASE_URL": "http://a", "LLM_PRIMARY_API_KEY": "k", "LLM_PRIMARY_MODEL": "m1",
                         "LLM_WRITER_MODEL": "w1", "LLM_CHEAP_MODEL": "c1"})
    if fallback:
        settings.env.update({"LLM_FALLBACK_BASE_URL": "http://b", "LLM_FALLBACK_API_KEY": "k",
                             "LLM_FALLBACK_MODEL": "f1"})


def _set_opts(settings, provider: str, **opts: Any) -> None:
    settings.seats["providers"][provider]["options"] = dict(opts)


def _reply(content: str, finish: str = "stop", tokens: int = 10, cost: float = 0.0) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"total_tokens": tokens, "cost": cost}}


class _Fake:
    """Canned post_json: pops replies (dict or exception) in order and records every request."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url, body, headers=None, timeout=0.0):
        self.calls.append({"url": url, "body": copy.deepcopy(body), "timeout": timeout})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _install(monkeypatch, replies: list[Any]) -> _Fake:
    fake = _Fake(replies)
    monkeypatch.setattr(llm, "post_json", fake)
    monkeypatch.setattr(llm, "post_stream", fake)  # providers with options.stream (the fallback) post here
    return fake


def _spec(**opts: Any) -> ProviderSpec:
    return ProviderSpec(name="p", type="openai_compat", model="m", base_url="http://x", api_key="k", options=opts)


# ---------------------------------------------------------------- max_tokens

def test_max_tokens_absent_by_default(settings, monkeypatch):
    _env(settings)
    fake = _install(monkeypatch, [_reply('{"a": 1}')])
    assert Seats(settings, CostTracker("p")).llm_json("writer", "script", "s", "u", "script") == {"a": 1}
    body = fake.calls[0]["body"]
    assert "max_tokens" not in body and "max_completion_tokens" not in body


def test_max_tokens_sent_when_set_on_provider(settings, monkeypatch):
    _env(settings)
    _set_opts(settings, "primary_llm", max_tokens=4096)
    fake = _install(monkeypatch, [_reply('{"a": 1}')])
    Seats(settings, CostTracker("p")).llm_json("researcher", "t", "s", "u", "research")
    assert fake.calls[0]["body"]["max_tokens"] == 4096


def test_role_entry_option_overrides_provider_max_tokens(settings, monkeypatch):
    _env(settings)
    _set_opts(settings, "primary_llm", max_tokens=1000)
    settings.seats["roles"]["writer"][0] = {**settings.seats["roles"]["writer"][0], "options": {"max_tokens": 16000}}
    fake = _install(monkeypatch, [_reply('{"a": 1}'), _reply('{"b": 2}')])
    seats = Seats(settings, CostTracker("p"))
    seats.llm_json("writer", "script", "s", "u", "script")
    seats.llm_json("researcher", "t", "s", "u", "research")
    assert fake.calls[0]["body"]["max_tokens"] == 16000
    assert fake.calls[1]["body"]["max_tokens"] == 1000


@pytest.mark.parametrize("bad", [0, -5, "lots", 1.5, True, [3]])
def test_invalid_max_tokens_ignored(bad, monkeypatch):
    fake = _install(monkeypatch, [_reply("hello")])
    res = llm.chat(_spec(max_tokens=bad), "s", "u", want_json=False)
    assert res.value == "hello"
    assert "max_tokens" not in fake.calls[0]["body"]
    assert llm.max_tokens_option(_spec(max_tokens="512")) == 512


def test_max_tokens_renamed_on_unsupported_400(monkeypatch):
    err = RefusalError('http 400: {"error": {"message": "Unsupported parameter: \'max_tokens\' is not supported '
                       'with this model. Use \'max_completion_tokens\' instead.", "param": "max_tokens"}}')
    fake = _install(monkeypatch, [err, _reply('{"ok": true}'), _reply('{"ok": 2}')])
    spec = _spec(max_tokens=800)
    assert llm.chat(spec, "s", "u", want_json=True).value == {"ok": True}
    second = fake.calls[1]["body"]
    assert second["max_completion_tokens"] == 800 and "max_tokens" not in second
    llm.chat(spec, "s", "u", want_json=True)  # learned: no second 400
    assert len(fake.calls) == 3 and fake.calls[2]["body"]["max_completion_tokens"] == 800


def test_max_tokens_rename_learned_from_message_without_param(monkeypatch):
    err = RefusalError("http 400: max_tokens is not supported, use max_completion_tokens")
    fake = _install(monkeypatch, [err, _reply("fine")])
    assert llm.chat(_spec(max_tokens=50), "s", "u", want_json=False).value == "fine"
    assert fake.calls[1]["body"]["max_completion_tokens"] == 50


def test_model_needing_every_quirk_succeeds_on_first_request(monkeypatch):
    # one 400 per adjustable feature: system role, temperature, response_format, then max_tokens
    errs = [RefusalError("http 400: system role is not supported"),
            RefusalError('http 400: {"error": {"message": "bad", "param": "temperature"}}'),
            RefusalError('http 400: {"error": {"message": "bad", "param": "response_format"}}'),
            RefusalError('http 400: {"error": {"message": "bad", "param": "max_tokens"}}')]
    fake = _install(monkeypatch, errs + [_reply('{"a": 1}')])
    assert llm.chat(_spec(max_tokens=1000), "s", "u", want_json=True).value == {"a": 1}
    last = fake.calls[-1]["body"]
    assert len(fake.calls) == 5 and last["max_completion_tokens"] == 1000
    assert not {"temperature", "response_format", "max_tokens"} & set(last)
    assert [m["role"] for m in last["messages"]] == ["user"]


def test_unlearnable_400_still_stops_retrying(monkeypatch):
    fake = _install(monkeypatch, [RefusalError("http 400: something else entirely")])
    with pytest.raises(RefusalError):
        llm.chat(_spec(), "s", "u", want_json=True)
    assert len(fake.calls) == 1


# ---------------------------------------------------------------- timeouts

def test_every_text_role_gets_600s(settings, monkeypatch):
    """Live run 1: slow relays ran past 180 s on several roles, so 600 s is the default for all."""
    _env(settings)
    fake = _install(monkeypatch, [_reply('{"a": 1}'), _reply('{"a": 1}'), _reply("t"), _reply('{"a": 1}')])
    seats = Seats(settings, CostTracker("p"))
    seats.llm_json("writer", "script", "s", "u", "script")
    seats.llm_json("researcher", "t", "s", "u", "research")
    seats.llm_text("writer", "t", "s", "u", "packaging")
    seats.llm_json("editor", "revise", "s", "u", "revise")
    assert [c["timeout"] for c in fake.calls] == [600.0, 600.0, 600.0, 600.0]


def test_caller_longer_timeout_wins_and_shorter_does_not(settings, monkeypatch):
    _env(settings)
    fake = _install(monkeypatch, [_reply('{"a": 1}'), _reply('{"a": 1}')])
    seats = Seats(settings, CostTracker("p"))
    seats.llm_json("director", "director", "s", "u", "shotlist", timeout=600.0)
    seats.llm_json("writer", "script", "s", "u", "script", timeout=60.0)
    assert [c["timeout"] for c in fake.calls] == [600.0, 600.0]


def test_provider_timeout_option_overrides(settings, monkeypatch):
    _env(settings)
    _set_opts(settings, "primary_llm", timeout_s=900)
    fake = _install(monkeypatch, [_reply('{"a": 1}'), _reply('{"a": 1}')])
    seats = Seats(settings, CostTracker("p"))
    seats.llm_json("researcher", "t", "s", "u", "research")
    _set_opts(settings, "primary_llm", timeout_s="soon")
    seats.llm_json("writer", "script", "s", "u", "script")
    assert [c["timeout"] for c in fake.calls] == [900.0, 600.0]


# ---------------------------------------------------------------- truncation

def test_truncated_json_continued_parses_and_sums_usage(settings, monkeypatch):
    _env(settings)
    full = json.dumps({"beats": [{"text": "one"}, {"text": "two"}]})
    cut = len(full) // 2
    fake = _install(monkeypatch, [_reply(full[:cut], "length", 100, 0.01), _reply(full[cut:], "stop", 40, 0.002)])
    seats = Seats(settings, CostTracker("p"))
    assert seats.llm_json("writer", "script", "s", "u", "script") == json.loads(full)
    msgs = fake.calls[1]["body"]["messages"]
    assert msgs[-2] == {"role": "assistant", "content": full[:cut]}
    assert msgs[-1]["role"] == "user" and "continue" in msgs[-1]["content"].lower()
    assert "response_format" not in fake.calls[1]["body"]
    ok = [e for e in seats.tracker.report.entries if e.ok]
    assert ok[0].units == 140 and ok[0].cost_usd == pytest.approx(0.012)


def test_truncated_twice_fails_over_to_fallback(settings, monkeypatch):
    _env(settings, fallback=True)
    fake = _install(monkeypatch, [_reply('{"a": [1, ', "length"), _reply("2, ", "max_tokens"),
                                  _reply('{"from": "fallback"}')])
    seats = Seats(settings, CostTracker("p"))
    assert seats.llm_json("writer", "script", "s", "u", "script") == {"from": "fallback"}
    assert [c["url"] for c in fake.calls] == ["http://a/chat/completions"] * 2 + ["http://b/chat/completions"]
    assert not base.HEALTH.snapshot()  # a truncation never marks the provider down


def test_truncated_twice_on_every_provider_raises(settings, monkeypatch):
    _env(settings)
    _install(monkeypatch, [_reply("{", "length"), _reply('"a"', "length")])
    with pytest.raises(ProviderError):
        Seats(settings, CostTracker("p")).llm_json("writer", "script", "s", "u", "script")


def test_continuation_that_still_does_not_parse_raises_refusal(monkeypatch):
    _install(monkeypatch, [_reply('{"a": ', "length"), _reply("oops", "stop")])
    with pytest.raises(RefusalError):
        llm.chat(_spec(), "s", "u", want_json=True)


def test_truncated_text_is_continued(settings, monkeypatch):
    _env(settings)
    fake = _install(monkeypatch, [_reply("The canal opened in ", "length", 7), _reply("1914.", "stop", 3)])
    text = Seats(settings, CostTracker("p")).llm_text("writer", "t", "s", "u", "script")
    assert text == "The canal opened in 1914."
    assert len(fake.calls) == 2


def test_finished_reply_is_not_continued(monkeypatch):
    fake = _install(monkeypatch, [_reply("done", "stop")])
    assert llm.chat(_spec(), "s", "u", want_json=False).value == "done"
    assert len(fake.calls) == 1
