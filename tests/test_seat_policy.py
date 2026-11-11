"""Seat policy in config/seats.yaml (Adam, 2026-11-11): exact ids, model ids resolved from /models, stale
pins, prices per entry, seat level errors, primary switch, tts order."""
from __future__ import annotations

from typing import Any

import pytest

from vidbush.providers import model_resolve, seats as seats_mod
from vidbush.providers.base import (HEALTH, CallResult, CostTracker, ProviderClient, ProviderError, ProviderSpec,
                                    RateLimitError, RefusalError, SeatUnavailable, http_error_to_exc)

WELLFLOW_IDS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-sonnet-5", "claude-opus-5", "gemini-3.8-flash",
                "grok-4.7"]
CLOSEROUTER_IDS = ["anthropic/claude-sonnet-5.5", "anthropic/claude-opus-5.5", "anthropic/claude-sonnet-5",
                   "google/gemini-3.8-flash", "x-ai/grok-4.7"]
OPENLUX_IDS = ["claude-sonnet-5-5", "claude-opus-5-5", "grok-4.7-fast", "grok-4.7", "gemini-3.8-flash-lite",
               "gemini-3.8-flash-preview", "gemini-3.8-flash"]
ANTHROPIC_IDS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]


class Log:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str, dict]] = []

    def info(self, msg: str, **kw: Any) -> None:
        self.lines.append(("info", msg, kw))

    def warn(self, msg: str, **kw: Any) -> None:
        self.lines.append(("warn", msg, kw))


@pytest.fixture
def live(settings, monkeypatch):
    model_resolve.reset()
    seats_mod._UNKNOWN_LOGGED.clear()
    settings.offline = False
    settings.env = {"WELLFLOW_CHEAP": "c", "WELLFLOW_API_KEY": "w", "CLOSEROUTER_API_KEY": "r",
                    "OPENLUX_API_KEY": "o", "ANTHROPIC_KEY": "d", "ALGROW_API_KEY": "a", "ELEVATE_API_KEY": "e",
                    "ELEVENLABS_API_KEY": "x"}
    calls: list[str] = []

    def get_json(url, headers=None, timeout=0, **kw):
        calls.append(url)
        if "closerouter" in url:
            return {"data": [{"id": i} for i in CLOSEROUTER_IDS]}
        if "openlux" in url:
            return {"data": [{"id": i} for i in OPENLUX_IDS]}
        return {"data": [{"id": i} for i in WELLFLOW_IDS]}
    monkeypatch.setattr(model_resolve, "get_json", get_json)
    monkeypatch.setattr("vidbush.providers.llm_anthropic.get_json",
                        lambda url, headers=None, timeout=0: calls.append(url) or {"data": [{"id": i} for i in ANTHROPIC_IDS]})
    yield settings, calls
    model_resolve.reset()


def _rows(s: seats_mod.Seats, role: str) -> list[tuple[str, str]]:
    return [(p.name, p.model) for p in s.providers_for(role)]


def test_match_by_family_and_version() -> None:
    assert model_resolve.match_model(WELLFLOW_IDS, "sonnet", "5.5") == "claude-sonnet-5-5"
    assert model_resolve.match_model(WELLFLOW_IDS, "sonnet", "5") == "claude-sonnet-5"
    assert model_resolve.match_model(CLOSEROUTER_IDS, "opus", "5.5") == "anthropic/claude-opus-5.5"
    assert model_resolve.match_model(["claude-opus-5-5-20261101", "claude-opus-5-5"], "opus", "5.5") == "claude-opus-5-5"
    assert model_resolve.match_model(["claude-opus-5-5-20261101"], "opus", "5.5") == "claude-opus-5-5-20261101"
    assert model_resolve.match_model(["claude-opus-5-5"], "sonnet", "5.5") == ""


def test_match_other_families_with_an_empty_prefix() -> None:
    assert model_resolve.match_model(["x-ai/grok-4.7-fast", "x-ai/grok-4.7"], "grok", "4.7", prefix="") == "x-ai/grok-4.7"
    assert model_resolve.match_model(["grok-4.7"], "grok", "4.7", prefix="") == "grok-4.7"
    assert model_resolve.match_model(["google/gemini-3.8-flash-lite", "google/gemini-3.8-flash"], "gemini",
                                     "3.8-flash", prefix="") == "google/gemini-3.8-flash"
    assert model_resolve.match_model(["gemini-3.8-flash-lite", "gemini-3.8-flash-preview"], "gemini", "3.8-flash",
                                     prefix="") == ""
    assert model_resolve.match_model(["x-ai/grok-4.7"], "grok", "4.7") == ""  # the default prefix is claude


def test_pin_matches_family_and_version_only() -> None:
    m = {"family": "opus", "version": "5.5"}
    assert model_resolve.pin_matches("claude-opus-5-5", m)
    assert model_resolve.pin_matches("anthropic/claude-opus-5.5", m)
    assert model_resolve.pin_matches("claude-opus-5-5-exact", m)
    assert model_resolve.pin_matches("claude-opus-5-5-20261101", m)
    assert not model_resolve.pin_matches("anthropic/claude-opus-5", m)
    assert not model_resolve.pin_matches("claude-sonnet-5-5", m)
    assert not model_resolve.pin_matches("claude-opus-5-5", {"family": "opus", "version": "5"})
    assert model_resolve.pin_matches("x-ai/grok-4.7", {"family": "grok", "version": "4.7", "prefix": ""})


def test_writer_exact_ids_in_policy_order_with_prices(live) -> None:
    settings, calls = live
    s = seats_mod.Seats(settings, CostTracker("p"), Log())
    writer = s.providers_for("writer")
    assert _rows(s, "writer") == [("wellflow_cheap", "claude-opus-5-5"), ("closerouter", "anthropic/claude-opus-5.5"),
                                  ("anthropic_direct", "claude-opus-5-5")]
    assert all(p.configured and p.options["stream"] is True for p in writer)
    assert writer[0].api_key == "c" and writer[2].api_key == "d" and writer[2].type == "anthropic"
    assert writer[0].options["price_per_mtok"] == {"prompt": 0.30, "completion": 0.30}
    assert writer[1].options["price_per_mtok"] == {"prompt": 0.205866, "completion": 1.029332}
    assert calls == []  # exact ids: no model listing


def test_legacy_writer_pin_is_not_read(live) -> None:
    settings, _ = live
    settings.env["LLM_WRITER_MODEL"] = "anthropic/claude-opus-5"  # run 1's CloseRouter Opus 5 id
    assert seats_mod.Seats(settings, CostTracker("p"), Log()).providers_for("writer")[0].model == "claude-opus-5-5"


def test_sonnet_roles_closerouter_resolved_then_openlux_then_anthropic(live) -> None:
    settings, calls = live
    log = Log()
    s = seats_mod.Seats(settings, CostTracker("p"), log)
    for role in ("director", "researcher"):
        assert _rows(s, role) == [("closerouter", "anthropic/claude-sonnet-5.5"), ("openlux", "claude-sonnet-5-5"),
                                  ("anthropic_direct", "claude-sonnet-5-5")], role
        openlux = s.providers_for(role)[1]
        assert openlux.base_url == "https://api.openlux.ai/v1" and openlux.api_key == "o"
        assert openlux.options["stream"] is True
        assert openlux.options["price_per_mtok"] == {"prompt": 0.1765, "completion": 0.8824}
    resolved = [ln for ln in log.lines if ln[1].startswith("model resolved")]
    assert len([r for r in resolved if r[2]["family"] == "sonnet"]) == 1  # logged once, cached
    assert len([c for c in calls if "closerouter" in c]) == 1


def test_stale_pin_is_ignored_with_one_warning_and_a_matching_pin_wins(live) -> None:
    settings, _ = live
    settings.env["LLM_DIRECTOR_MODEL"] = "anthropic/claude-sonnet-5"
    log = Log()
    s = seats_mod.Seats(settings, CostTracker("p"), log)
    for _ in range(3):
        assert s.providers_for("director")[0].model == "anthropic/claude-sonnet-5.5"
    assert len([ln for ln in log.lines if "does not match model_match" in ln[1]]) == 1
    settings.env["LLM_DIRECTOR_MODEL"] = "anthropic/claude-sonnet-5.5-exact"
    assert s.providers_for("director")[0].model == "anthropic/claude-sonnet-5.5-exact"


def test_unlisted_family_leaves_the_seat_unconfigured(live, monkeypatch) -> None:
    settings, _ = live
    monkeypatch.setattr(model_resolve, "get_json", lambda *a, **k: {"data": [{"id": "gemini-3.8-flash"}]})
    log = Log()
    s = seats_mod.Seats(settings, CostTracker("p"), log)
    d = s.providers_for("director")[0]
    assert d.model == "" and not d.configured
    assert any(ln[0] == "warn" and "not resolved" in ln[1] for ln in log.lines)


def test_editor_is_grok_on_closerouter_then_openlux_then_wellflow(live) -> None:
    settings, _ = live
    s = seats_mod.Seats(settings, CostTracker("p"), Log())
    assert _rows(s, "editor") == [("closerouter", "x-ai/grok-4.7"), ("openlux", "grok-4.7"), ("wellflow", "grok-4.7")]
    provs = s.providers_for("editor")
    assert all(p.configured and p.options.get("stream") is True for p in provs)
    assert provs[2].api_key == "w"  # the normal Wellflow key, never the cheap one


def test_vision_roles_gemini_flash_never_lite_or_preview(live) -> None:
    settings, _ = live
    s = seats_mod.Seats(settings, CostTracker("p"), Log())
    for role in ("judge", "vision", "cheap_classifier"):
        assert _rows(s, role) == [("closerouter", "google/gemini-3.8-flash"), ("openlux", "gemini-3.8-flash"),
                                  ("wellflow", "gemini-3.8-flash")], role
        assert all(p.configured and p.options.get("stream") is True for p in s.providers_for(role))


def test_anthropic_direct_preferred_until_the_date(live) -> None:
    settings, calls = live
    settings.env.update({"VIDBUSH_PREFER_PROVIDER": "anthropic_direct", "VIDBUSH_PREFER_UNTIL": "2999-01-01T00:00:00Z"})
    s = seats_mod.Seats(settings, CostTracker("p"), Log())
    first = s.providers_for("writer")[0]
    assert (first.name, first.type, first.model) == ("anthropic_direct", "anthropic", "claude-opus-5-5")
    assert [p.name for p in s.providers_for("editor")] == ["closerouter", "openlux", "wellflow"]  # not listed there
    settings.env["VIDBUSH_PREFER_UNTIL"] = "2026-01-01"
    log = Log()
    s2 = seats_mod.Seats(settings, CostTracker("p"), log)
    assert s2.providers_for("writer")[0].name == "wellflow_cheap"
    s2.providers_for("writer")
    assert len([ln for ln in log.lines if "has passed" in ln[1]]) == 1


def test_tts_order_elevenlabs_elevate_algrow_all_registered(live) -> None:
    settings, _ = live
    log = Log()
    s = seats_mod.Seats(settings, CostTracker("p"), log)
    assert _rows(s, "tts") == [("elevenlabs", "eleven_v4"), ("elevate_tts", "eleven_v4"), ("algrow_tts", "eleven_v4")]
    assert all(p.configured for p in s.providers_for("tts"))
    assert not [ln for ln in log.lines if "not registered" in ln[1]]


def test_images_unchanged(live) -> None:
    settings, _ = live
    s = seats_mod.Seats(settings, CostTracker("p"), Log())
    assert [(p.name, p.type) for p in s.providers_for("image")] == [("image_primary", "elevate"),
                                                                    ("image_fallback", "algrow")]


def test_offline_never_lists_models(live) -> None:
    settings, calls = live
    settings.offline = True
    assert seats_mod.Seats(settings, CostTracker("p"), Log()).providers_for("director") == []
    assert calls == []


# ---------------------------------------------------------------- seat level errors

def test_seat_errors_are_told_from_request_errors() -> None:
    for status, body in [(401, "invalid api key"), (402, "payment required"), (403, "forbidden"),
                         (403, '{"error": {"code": "model_not_in_channel"}}'),
                         (400, "You have reached your specified API usage limits until 2026-12-01"),
                         (400, "Your credit balance is too low"),
                         (429, '{"error": {"code": "insufficient_quota", "message": "You exceeded your current quota"}}'),
                         (429, "You have reached your specified API usage limits")]:
        assert isinstance(http_error_to_exc(status, body), SeatUnavailable), (status, body)
    for status, body in [(400, "request violates the content policy"), (403, "blocked by moderation"),
                         (403, "request violates our usage guidelines"), (403, "Input flagged: prohibited content"),
                         (400, "bad request: messages missing"), (404, "not found")]:
        e = http_error_to_exc(status, body)
        assert isinstance(e, RefusalError) and not isinstance(e, SeatUnavailable), (status, body)
    # a rate window, including a per minute quota, is waited out and never marks the seat down
    for body in ["Rate limit reached for requests", "RESOURCE_EXHAUSTED: Quota exceeded for metric per minute"]:
        assert type(http_error_to_exc(429, body)) is RateLimitError, body


def test_every_adapter_shares_one_policy_word_list() -> None:
    from vidbush.providers import base, llm, llm_anthropic, tts_elevate
    assert llm.POLICY_WORDS is llm_anthropic.POLICY_WORDS is tts_elevate.POLICY_WORDS is base.POLICY_WORDS


def test_tts_engine_is_the_first_seat_not_held_down(settings, monkeypatch) -> None:
    settings.offline = False
    settings.seats = {"roles": {"tts": [{"provider": "elevenlabs", "model": "eleven_v4"},
                                        {"provider": "elevate_tts", "model": "eleven_v4"}]},
                      "providers": {"elevenlabs": {"type": "elevenlabs", "key_env": "ELEVENLABS_API_KEY"},
                                    "elevate_tts": {"type": "elevate_tts", "key_env": "ELEVATE_API_KEY"}}}
    settings.env = {"ELEVENLABS_API_KEY": "k", "ELEVATE_API_KEY": "k"}
    seats = seats_mod.Seats(settings, CostTracker("p"), Log())
    try:
        assert seats.tts_engine() == "elevenlabs"
        HEALTH.mark_down("elevenlabs:eleven_v4")
        assert seats.tts_engine() == "elevate_tts" and seats.tts_engine() == "elevate_tts"  # no probe used
        monkeypatch.setattr("vidbush.providers.base.time.time", lambda: 1e12)  # the probe window opens
        assert seats.tts_engine() == "elevenlabs"
    finally:
        HEALTH.mark_up("elevenlabs:eleven_v4")


def test_seat_error_marks_the_seat_down_and_a_request_error_does_not() -> None:
    a = ProviderSpec("seat_a", "openai_compat", "m1", base_url="http://a", api_key="k")
    b = ProviderSpec("seat_b", "openai_compat", "m1", base_url="http://b", api_key="k")
    hits: list[str] = []

    def fn(err: Exception):
        def call(spec: ProviderSpec) -> CallResult:
            hits.append(spec.name)
            if spec.name == "seat_a":
                raise err
            return CallResult(value="ok", provider=spec, mock=False)
        return call

    log = Log()
    client = ProviderClient(CostTracker("p"), log, sleep=lambda s: None)
    try:
        seat_err = http_error_to_exc(403, '{"error": {"code": "model_not_in_channel"}}')
        for _ in range(2):
            assert client.call("writer", "script", [a, b], fn(seat_err)).value == "ok"
        assert hits == ["seat_a", "seat_b", "seat_b"]  # the second request skips seat_a
        assert len([ln for ln in log.lines if "seat unavailable" in ln[1]]) == 1
        HEALTH.mark_up("seat_a:m1")
        hits.clear()
        for _ in range(2):
            client.call("writer", "script", [a, b], fn(http_error_to_exc(400, "content policy")))
        assert hits == ["seat_a", "seat_b", "seat_a", "seat_b"]  # one request's refusal never marks down
        with pytest.raises(ProviderError):
            client.call("writer", "script", [a], fn(http_error_to_exc(402, "no credit")))
    finally:
        HEALTH.mark_up("seat_a:m1")
