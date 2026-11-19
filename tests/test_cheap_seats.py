"""Cheapest provider rule for describer, cheap_classifier and auditor (providers/seats.py), CheapSeatDown at the
runner and the audit command, describer and picker in the role lists, and price warnings across vendor prefixes."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vidbush.models import CostEntry, InputMode, StageStatus
from vidbush.providers import seats as seats_mod
from vidbush.providers.base import (HEALTH, CallResult, CheapSeatDown, CostTracker, TransportError, price_warnings)


def _entry(name: str, price=None, model: str = "m") -> dict:
    return {"provider": name, "model": model, **({"options": {"usd_per_call": price}} if price is not None else {})}


@pytest.fixture
def cheap(settings, monkeypatch):
    """Seats over fake providers a, b, c of a registered fake type; down names raise transport errors."""
    HEALTH._down.clear()
    seats_mod._UNKNOWN_LOGGED.clear()
    settings.offline = False
    settings.reply_cache = False
    settings.env = {"KEY": "k"}
    settings.seats = {"providers": {n: {"type": "fake", "key_env": "KEY"} for n in "abc"}, "roles": {}}
    calls: list[str] = []
    down: set[str] = set()

    def chat(spec, system, user, want_json=True, temperature=0.0, **kw):
        calls.append(spec.name)
        if spec.name in down:
            raise TransportError("http 503")
        return CallResult(value={"ok": True}, provider=spec, mock=False, cost_usd=0.001, units=1, unit_kind="tokens")
    monkeypatch.setitem(seats_mod.LLM_TYPES, "fake", SimpleNamespace(chat=chat))

    def make(role: str, entries: list[dict]) -> seats_mod.Seats:
        settings.seats["roles"][role] = entries
        s = seats_mod.Seats(settings, CostTracker("p"))
        s.client.sleep = lambda t: None
        return s
    yield make, calls, down
    HEALTH._down.clear()


def test_classifier_runs_only_on_the_cheapest_and_stops_when_it_is_down(cheap) -> None:
    make, calls, down = cheap
    s = make("cheap_classifier", [_entry("a", 0.0002), _entry("b", 0.01), _entry("c", 0.01)])
    s.vision_json("cheap_classifier", "segment", "sys", "p1", [], "sourcing")
    assert calls == ["a"]
    down.add("a")
    with pytest.raises(CheapSeatDown) as e:
        s.vision_json("cheap_classifier", "segment", "sys", "p2", [], "sourcing")
    assert set(calls) == {"a"}, "the dearer providers are never called"
    assert "cheap_classifier" in str(e.value) and "b, c" in str(e.value) and str(e.value).startswith("cheap seat down")
    assert not issubclass(CheapSeatDown, Exception)  # `except Exception` fallbacks cannot swallow it


def test_auditor_runs_cheapest_first_within_five_times_and_skips_the_dear_one(cheap) -> None:
    make, calls, down = cheap
    s = make("auditor", [_entry("a", 0.0009), _entry("b", 0.0006), _entry("c", 0.007)])
    s.vision_json("auditor", "audit", "sys", "p1", [], "audit")
    assert calls == ["b"]
    down.add("b")
    s.vision_json("auditor", "audit", "sys", "p2", [], "audit")
    assert calls[-1] == "a" and "c" not in calls
    down.add("a")
    with pytest.raises(CheapSeatDown):
        s.vision_json("auditor", "audit", "sys", "p3", [], "audit")
    assert "c" not in calls


def test_measured_history_beats_the_configured_price_and_unpriced_entries_are_skipped(cheap) -> None:
    make, calls, _ = cheap
    s = make("describer", [_entry("a", 0.0002), _entry("b"), _entry("c", 0.02)])
    for _ in range(3):  # c really costs less than a here: three paid calls are enough to trust it
        s.tracker.add(CostEntry(role="describer", provider="c", model="m", cost_usd=0.0001, units=1, unit_kind="tokens"))
    s.tracker.add(CostEntry(role="describer", provider="a", model="m", cost_usd=0.0, unit_kind="reply_cache"))
    assert s.unit_prices("describer", s.providers_for("describer")) == [0.0002, None, pytest.approx(0.0001)]
    cheap_provs, dear = s.cheapest("describer", s.providers_for("describer"))
    assert [p.name for p in cheap_provs] == ["c", "a"] and [p.name for p in dear] == ["b"]
    s.vision_json("describer", "describe_asset", "sys", "p1", [], "judge")
    assert calls == ["c"]


def test_a_seat_with_no_priced_provider_stops_and_other_roles_keep_their_order(cheap) -> None:
    make, calls, _ = cheap
    s = make("describer", [_entry("a"), _entry("b")])
    with pytest.raises(CheapSeatDown, match="none has a price"):
        s.vision_json("describer", "describe_asset", "sys", "p1", [], "judge")
    assert calls == []
    s = make("judge", [_entry("c", 0.05), _entry("a", 0.0001)])
    s.vision_json("judge", "score", "sys", "p1", [], "judge")
    assert calls == ["c"], "the rule only covers describe, classify and audit calls"


def test_describer_and_picker_are_listed() -> None:
    assert {"describer", "picker"} <= set(seats_mod.ROLES) and {"describer", "picker"} <= seats_mod.VISION_ROLES


def test_policy_prices_every_cheap_role_entry(settings) -> None:
    for role in seats_mod.CHEAPEST_ROLES:
        for entry in settings.seats["roles"][role]:
            assert isinstance(entry.get("options", {}).get("usd_per_call"), float), (role, entry["provider"])


def test_price_warnings_compare_across_vendor_prefixes() -> None:
    e = dict(role="cheap_classifier", unit_kind="tokens", units=1000)
    out = price_warnings([CostEntry(provider="closerouter", model="google/gemini-x", cost_usd=0.0002, **e),
                          CostEntry(provider="openlux", model="gemini-x", cost_usd=0.01, **e),
                          CostEntry(provider="openlux", model="gemini-x", role="judge", unit_kind="tokens",
                                    units=1000, cost_usd=0.5)])
    assert len(out) == 1 and "openlux charged 50.0 times closerouter" in out[0] and "cheap_classifier" in out[0]


# ---------------------------------------------------------------- reporting

def test_runner_stops_like_a_cap_and_resume_reruns_the_stage(settings, tmp_path) -> None:
    from vidbush.jobs.pipeline import create_project, load_project, run_project
    from vidbush.models import STAGES
    script = tmp_path / "p.md"
    script.write_text("# T\n\nHello world.")
    create_project(settings, "p1", InputMode.script, script_path=str(script), seed=1)
    state = {"down": True}

    def make(name):
        def fn(ctx):
            if name == "sourcing" and state["down"]:
                raise CheapSeatDown("cheap_classifier", "sourcing", ["closerouter"], ["openlux", "wellflow"])
            p = ctx.work / f"{name}.done"
            p.write_text("ok")
            return [str(p)]
        return fn
    stages = {n: make(n) for n in STAGES}
    proj = run_project(settings, "p1", stages=stages, echo=False)
    assert proj.status == "needs_review" and proj.status_note.startswith("cheap seat down")
    disk = load_project(settings, "p1")
    assert disk.stages["sourcing"].status == StageStatus.pending
    assert disk.stages["sourcing"].history[-1].status == "stopped"
    assert not (Path(disk.paths.work) / "cost_cap.json").exists()
    state["down"] = False
    proj = run_project(settings, "p1", stages=stages, echo=False)
    assert proj.status == "done" and proj.status_note == ""


def test_audit_command_returns_1_and_keeps_its_cost_file(tmp_path, monkeypatch) -> None:
    import vidbush.util
    from vidbush.audit import auditor, cli

    class _Seats:
        def __init__(self, settings, tracker):
            pass

        def providers_for(self, role):
            return []

    def stop(*a, **kw):
        raise CheapSeatDown("auditor", "audit", ["openlux", "closerouter"], ["anthropic_direct"])
    root = tmp_path / "output" / "proj"
    (root / "work").mkdir(parents=True)
    (root / "final.mp4").write_bytes(b"v")
    (root / "work" / "narration.json").write_text('{"words": []}')
    monkeypatch.setattr(seats_mod, "Seats", _Seats)
    monkeypatch.setattr(auditor, "seats_vision", lambda seats: None)
    monkeypatch.setattr(auditor, "run_audit", stop)
    monkeypatch.setattr(vidbush.util, "media_duration", lambda v: 10.0)
    settings = SimpleNamespace(output_folder=str(tmp_path / "output"), path=Path)
    args = Namespace(target=str(root), script=None, out=None, every=2.0, limit=0, workers=1)
    assert cli.cmd_audit(settings, args) == 1
    assert (root / "audit" / cli.COST_FILE).exists()
