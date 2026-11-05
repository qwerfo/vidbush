"""Seat policy in config/seats.yaml: model ids resolved from /models, primary switch, tts order."""
from __future__ import annotations

from typing import Any

import pytest

from vidbush.providers import model_resolve, seats as seats_mod
from vidbush.providers.base import CostTracker

WELLFLOW_IDS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-sonnet-5", "claude-opus-5", "gemini-3.8-flash"]
CLOSEROUTER_IDS = ["anthropic/claude-sonnet-5.5", "anthropic/claude-opus-5.5", "anthropic/claude-sonnet-5",
                   "google/gemini-3.8-flash"]
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
                    "ALGROW_API_KEY": "a", "ELEVATE_API_KEY": "e"}
    calls: list[str] = []

    def get_json(url, headers=None, timeout=0, **kw):
        calls.append(url)
        if "closerouter" in url:
            return {"data": [{"id": i} for i in CLOSEROUTER_IDS]}
        return {"data": [{"id": i} for i in WELLFLOW_IDS]}
    monkeypatch.setattr(model_resolve, "get_json", get_json)
    monkeypatch.setattr("vidbush.providers.llm_anthropic.get_json",
                        lambda url, headers=None, timeout=0: calls.append(url) or {"data": [{"id": i} for i in ANTHROPIC_IDS]})
    yield settings, calls
    model_resolve.reset()


def test_match_by_family_and_version() -> None:
    assert model_resolve.match_model(WELLFLOW_IDS, "sonnet", "5.5") == "claude-sonnet-5-5"
    assert model_resolve.match_model(WELLFLOW_IDS, "sonnet", "5") == "claude-sonnet-5"
    assert model_resolve.match_model(CLOSEROUTER_IDS, "opus", "5.5") == "anthropic/claude-opus-5.5"
    assert model_resolve.match_model(["claude-opus-5-5-20261101", "claude-opus-5-5"], "opus", "5.5") == "claude-opus-5-5"
    assert model_resolve.match_model(["claude-opus-5-5-20261101"], "opus", "5.5") == "claude-opus-5-5-20261101"
    assert model_resolve.match_model(["claude-opus-5-5"], "sonnet", "5.5") == ""


def test_anthropic_roles_resolve_on_wellflow_cheap_and_log_once(live) -> None:
    settings, calls = live
    log = Log()
    s = seats_mod.Seats(settings, CostTracker("p"), log)
    writer = s.providers_for("writer")
    assert [(p.name, p.model) for p in writer][:1] == [("wellflow_cheap", "claude-opus-5-5")]
    assert writer[0].configured and writer[0].options["stream"] is True
    assert [p.model for p in s.providers_for("director")][0] == "claude-sonnet-5-5"
    assert [p.model for p in s.providers_for("researcher")][0] == "claude-sonnet-5-5"
    closer = [p for p in writer if p.name == "closerouter"][0]
    assert closer.model == "anthropic/claude-opus-5.5"
    assert not [p for p in writer if p.name == "anthropic_direct"][0].configured  # no key
    for _ in range(3):
        s.providers_for("writer")
    resolved = [ln for ln in log.lines if ln[1].startswith("model resolved")]
    assert len([r for r in resolved if r[2]["provider"] == "wellflow_cheap" and r[2]["family"] == "opus"]) == 1
    assert len([c for c in calls if "wellflow" in c]) == 1  # one listing, cached


def test_pinned_model_env_wins_over_resolution(live) -> None:
    settings, calls = live
    settings.env["LLM_WRITER_MODEL"] = "claude-opus-5-5-exact"
    s = seats_mod.Seats(settings, CostTracker("p"), Log())
    assert s.providers_for("writer")[0].model == "claude-opus-5-5-exact"


def test_unlisted_family_leaves_the_seat_unconfigured(live, monkeypatch) -> None:
    settings, _ = live
    monkeypatch.setattr(model_resolve, "get_json", lambda *a, **k: {"data": [{"id": "gemini-3.8-flash"}]})
    log = Log()
    s = seats_mod.Seats(settings, CostTracker("p"), log)
    w = s.providers_for("writer")[0]
    assert w.model == "" and not w.configured
    assert any(ln[0] == "warn" and "not resolved" in ln[1] for ln in log.lines)


def test_anthropic_direct_preferred_until_the_date(live) -> None:
    settings, calls = live
    settings.env.update({"ANTHROPIC_DIRECT_API_KEY": "d", "VIDBUSH_PREFER_PROVIDER": "anthropic_direct",
                         "VIDBUSH_PREFER_UNTIL": "2999-01-01T00:00:00Z"})
    s = seats_mod.Seats(settings, CostTracker("p"), Log())
    first = s.providers_for("writer")[0]
    assert (first.name, first.type, first.model) == ("anthropic_direct", "anthropic", "claude-opus-5-5")
    assert any("api.anthropic.com/v1/models" in c for c in calls)
    assert [p.name for p in s.providers_for("editor")] == ["closerouter", "wellflow"]  # not listed there
    settings.env["VIDBUSH_PREFER_UNTIL"] = "2026-01-01"
    log = Log()
    s2 = seats_mod.Seats(settings, CostTracker("p"), log)
    assert s2.providers_for("writer")[0].name == "wellflow_cheap"
    s2.providers_for("writer")
    assert len([ln for ln in log.lines if "has passed" in ln[1]]) == 1


def test_non_anthropic_roles_closerouter_then_wellflow_streamed(live) -> None:
    settings, _ = live
    s = seats_mod.Seats(settings, CostTracker("p"), Log())
    for role in ("editor", "judge", "vision", "cheap_classifier"):
        provs = s.providers_for(role)
        assert [p.name for p in provs] == ["closerouter", "wellflow"], role
        assert all(p.configured and p.options.get("stream") is True for p in provs)
        assert provs[1].api_key == "w"  # the normal Wellflow key, never the cheap one


def test_tts_order_skips_unregistered_elevate_with_one_warning(live) -> None:
    settings, _ = live
    log = Log()
    s = seats_mod.Seats(settings, CostTracker("p"), log)
    names = [p.name for p in s.providers_for("tts")]
    s.providers_for("tts")
    if "elevate_tts" in seats_mod.TTS_TYPES:  # the laptop lane's adapter is merged
        assert names == ["elevate_tts", "algrow_tts", "elevenlabs"]
    else:
        assert names == ["algrow_tts", "elevenlabs"]
        assert len([ln for ln in log.lines if "not registered" in ln[1]]) == 1
    assert s.is_real("tts")


def test_images_unchanged(live) -> None:
    settings, _ = live
    s = seats_mod.Seats(settings, CostTracker("p"), Log())
    assert [(p.name, p.type) for p in s.providers_for("image")] == [("image_primary", "elevate"),
                                                                    ("image_fallback", "algrow")]


def test_offline_never_lists_models(live) -> None:
    settings, calls = live
    settings.offline = True
    assert seats_mod.Seats(settings, CostTracker("p"), Log()).providers_for("writer") == []
    assert calls == []
