"""V2 measurement lane: reply cache, cost cap inside a stage, stage snapshots, run provenance, run flags."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from vidbush import cli
from vidbush.jobs.context import SHOTLIST_STAGES, snapshot_path
from vidbush.jobs.costcap import read_cap
from vidbush.jobs.pipeline import create_project, load_project, mark_pending, run_project
from vidbush.models import (STAGES, Candidate, InputMode, Shot, ShotList, SourceKind, StageStatus)
from vidbush.providers import base, llm
from vidbush.providers.base import CallResult, CostCapReached, CostTracker
from vidbush.providers.seats import Seats
from vidbush.util import read_json, write_json


@pytest.fixture(autouse=True)
def _clean():
    base.HEALTH._down.clear()
    llm.reset_quirks()
    yield
    base.HEALTH._down.clear()


def _live(settings) -> None:
    settings.offline = False
    settings.env.update({"LLM_PRIMARY_BASE_URL": "http://a", "LLM_PRIMARY_API_KEY": "k", "LLM_PRIMARY_MODEL": "m1",
                         "VISION_BASE_URL": "http://v", "VISION_API_KEY": "k", "VISION_MODEL": "vm"})


def _fake_chat(monkeypatch, calls: list, cost: float = 0.01):
    def chat(spec, system, user, want_json=True, temperature=0.6, timeout=0):
        calls.append(user)
        return CallResult(value={"n": len(calls)}, provider=spec, mock=False, cost_usd=cost)
    monkeypatch.setattr(llm, "chat", chat)


def _png(path: Path, colour) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (32, 18), colour).save(buf, "PNG")
    path.write_bytes(buf.getvalue())
    return str(path)


# ---------------------------------------------------------------- reply cache

def test_replay_with_unchanged_prompts_costs_nothing(generic_seats, monkeypatch):
    settings = generic_seats
    _live(settings)
    calls: list = []
    _fake_chat(monkeypatch, calls)
    first = Seats(settings, CostTracker("p"))
    a = first.llm_json("director", "plan", "sys", "user prompt", "shotlist")
    b = first.llm_json("director", "plan", "sys", "user prompt", "shotlist")  # a retry in the same run: live
    assert len(calls) == 2 and a != b
    replay = Seats(settings, CostTracker("p2"))
    assert replay.llm_json("director", "plan", "sys", "user prompt", "shotlist") == a  # n-th ask, n-th reply
    assert replay.llm_json("director", "plan", "sys", "user prompt", "shotlist") == b
    assert len(calls) == 2 and replay.tracker.total_usd == 0.0
    assert replay.tracker.report.entries[-1].unit_kind == "reply_cache"
    assert replay.llm_json("director", "plan", "sys", "another prompt", "shotlist") == {"n": 3}


def test_fresh_and_switch_off_ask_live(generic_seats, monkeypatch):
    settings = generic_seats
    _live(settings)
    calls: list = []
    _fake_chat(monkeypatch, calls)
    Seats(settings, CostTracker("p")).llm_json("director", "plan", "s", "u", "shotlist")
    assert Seats(settings, CostTracker("p")).llm_json("director", "plan", "s", "u", "shotlist", fresh=True) == {"n": 2}
    settings.reply_cache = False
    assert Seats(settings, CostTracker("p")).llm_json("director", "plan", "s", "u", "shotlist") == {"n": 3}


def test_vision_key_includes_the_images(generic_seats, monkeypatch, tmp_path):
    settings = generic_seats
    _live(settings)
    calls: list = []
    _fake_chat(monkeypatch, calls)
    red, blue = _png(tmp_path / "r.png", (200, 0, 0)), _png(tmp_path / "b.png", (0, 0, 200))
    Seats(settings, CostTracker("p")).vision_json("judge", "judge", "s", "p", [red], "judge")
    replay = Seats(settings, CostTracker("p"))
    assert replay.vision_json("judge", "judge", "s", "p", [red], "judge")["n"] == 1
    assert replay.vision_json("judge", "judge", "s", "p", [blue], "judge")["n"] == 2
    assert Path(settings.source_cache_folder).parent.joinpath("reply_cache").is_dir()  # beside the source cache


def test_mock_replies_are_never_stored(settings):
    seats = Seats(settings, CostTracker("p"))  # offline: every role is mock
    seats.llm_json("director", "plan", "s", "u", "shotlist")
    assert not seats.reply_cache.enabled and seats.reply_cache.stores == 0


# ---------------------------------------------------------------- cost cap inside a stage

def _script_project(settings, tmp_path: Path, pid: str = "p1"):
    s = tmp_path / f"{pid}.md"
    s.write_text("# T\n\nHello world.")
    return create_project(settings, pid, InputMode.script, script_path=str(s), seed=1)


def _stubs(calls: list, inside=None):
    def make(name):
        def fn(ctx):
            calls.append(name)
            if inside and name in inside:
                inside[name](ctx)
            p = ctx.work / f"{name}.done"
            p.write_text("ok")
            return [str(p)]
        return fn
    return {n: make(n) for n in STAGES}


def test_cap_is_checked_inside_a_stage_and_resume_reruns_it(generic_seats, monkeypatch, tmp_path):
    settings = generic_seats
    _live(settings)
    calls_llm: list = []
    _fake_chat(monkeypatch, calls_llm, cost=1.0)

    def judge_many(ctx):  # ten paid calls inside one stage
        for i in range(10):
            ctx.seats.llm_json("director", "requery", "s", f"shot {i}", "judge")

    _script_project(settings, tmp_path)
    calls: list = []
    proj = run_project(settings, "p1", stages=_stubs(calls, {"judge": judge_many}), echo=False, cost_cap=3.0)
    assert len(calls_llm) == 3 and calls[-1] == "judge"  # stopped before the fourth paid call
    assert proj.status == "needs_review" and "inside judge" in proj.status_note
    disk = load_project(settings, "p1")
    assert disk.stages["judge"].status == StageStatus.pending
    assert disk.stages["judge"].history[-1].status == "capped"
    cap = read_cap(Path(disk.paths.work))
    assert cap["inside_stage"] and cap["stage"] == "judge"
    calls.clear()
    proj = run_project(settings, "p1", stages=_stubs(calls, {"judge": judge_many}), echo=False)
    assert calls[0] == "judge" and proj.status == "done"
    assert len(calls_llm) == 10  # the three paid replies came back from the reply cache


def test_cost_cap_is_a_base_exception():
    assert not issubclass(CostCapReached, Exception)  # `except Exception` fallbacks cannot swallow it


# ---------------------------------------------------------------- stage snapshots

def _cand(i: int, score=None) -> Candidate:
    return Candidate(id=f"c{i}", source=SourceKind.pexels, url=f"http://x/{i}", score=score)


def _shot_stages(seen: dict):
    """Fake shot list stages: each records what it read and adds its mark."""
    def make(name):
        def fn(ctx):
            if name not in SHOTLIST_STAGES:
                p = ctx.work / f"{name}.done"
                p.write_text("ok")
                return [str(p)]
            if name == "shotlist":
                sl = ShotList(project_id="p1", duration=4.0,
                              shots=[Shot(id="s1", index=0, beat_id="b1", intent="x", start=0, end=4, duration=4,
                                         type="footage")])
            else:
                sl = ctx.load("shotlist", ShotList)
            seen[name] = [c.model_copy() for c in sl.shots[0].candidates], sl.shots[0].judge_notes
            if name == "sourcing":
                sl.shots[0].candidates = [_cand(1), _cand(2)]
            if name == "judge":
                for c in sl.shots[0].candidates:
                    c.score = 80.0
                sl.shots[0].judge_notes = "judged"
            if name in SHOTLIST_STAGES:
                return [str(ctx.save("shotlist", sl))]
            p = ctx.work / f"{name}.done"
            p.write_text("ok")
            return [str(p)]
        return fn
    return {n: make(n) for n in STAGES}


def test_each_stage_writes_its_own_snapshot_and_force_judge_rescores(settings, tmp_path):
    _script_project(settings, tmp_path)
    seen: dict = {}
    run_project(settings, "p1", stages=_shot_stages(seen), echo=False)
    work = Path(load_project(settings, "p1").paths.work)
    for st in SHOTLIST_STAGES:
        assert snapshot_path(work, st).exists()
    assert read_json(snapshot_path(work, "sourcing"))["shots"][0]["candidates"][0]["score"] is None
    seen.clear()
    run_project(settings, "p1", stages=_shot_stages(seen), force=["judge"], echo=False)
    cands, notes = seen["judge"]
    assert [c.score for c in cands] == [None, None] and notes == ""  # judge read the sourcing snapshot


def test_force_sourcing_starts_from_the_director_list(settings, tmp_path):
    _script_project(settings, tmp_path)
    seen: dict = {}
    run_project(settings, "p1", stages=_shot_stages(seen), echo=False)
    seen.clear()
    run_project(settings, "p1", stages=_shot_stages(seen), force=["sourcing"], echo=False)
    assert seen["sourcing"][0] == []  # not the judged list with its candidates


def test_an_edit_to_the_working_copy_wins_over_an_older_snapshot(settings, tmp_path):
    _script_project(settings, tmp_path)
    seen: dict = {}
    run_project(settings, "p1", stages=_shot_stages(seen), echo=False)
    work = Path(load_project(settings, "p1").paths.work)
    data = read_json(work / "shotlist.json")
    data["shots"][0]["judge_notes"] = "operator edit"
    write_json(work / "shotlist.json", data)
    mark_pending(settings, "p1", ["visuals"])  # what editor ops do after changing a shot
    seen.clear()
    run_project(settings, "p1", stages=_shot_stages(seen), echo=False)
    assert seen["visuals"][1] == "operator edit"


# ---------------------------------------------------------------- provenance

def test_run_records_commit_settings_and_attempt_times(settings, tmp_path):
    _script_project(settings, tmp_path)
    run_project(settings, "p1", stages=_stubs([]), echo=False, cost_cap=9.0)
    run_project(settings, "p1", stages=_stubs([]), force=["mix"], echo=False)
    proj = load_project(settings, "p1")
    assert len(proj.runs) == 2 and proj.runs[0].cost_cap == 9.0 and proj.runs[1].force == ["mix"]
    assert proj.runs[0].finished_at and proj.runs[0].status == "done"
    assert "seats" in proj.runs[0].settings and "config_sha" in proj.runs[0].settings
    mix = proj.stages["mix"]
    assert [a.attempt for a in mix.history] == [1, 2] and all(a.seconds is not None for a in mix.history)
    assert mix.history[0].commit == proj.runs[0].commit


# ---------------------------------------------------------------- run flags

@pytest.fixture
def run_cli(settings, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    seen: dict = {}

    def fake_run(s, pid, **k):
        seen.update(k)
        return load_project(s, pid)
    monkeypatch.setattr("vidbush.jobs.pipeline.run_project", fake_run)

    def _run(*argv: str):
        code = cli.main(list(argv))
        return code, capsys.readouterr()
    return _run, seen


def test_run_refuses_an_existing_project_without_fresh(run_cli, settings, tmp_path):
    run, seen = run_cli
    script = tmp_path / "story.md"
    script.write_text("# Story\n\nOne line.")
    assert run("run", str(script), "--id", "s1", "--cost-cap", "4.5", "--quiet")[0] != 2
    assert seen["cost_cap"] == 4.5
    marker = Path(load_project(settings, "s1").paths.root) / "keep.txt"
    marker.write_text("x")
    code, out = run("run", str(script), "--id", "s1", "--quiet")
    assert code == 2 and "--fresh" in out.err and marker.exists()
    assert run("run", str(script), "--id", "s1", "--fresh", "--quiet")[0] != 2 and not marker.exists()


# ---------------------------------------------------------------- review fixes

def test_a_failed_occurrence_leaves_a_gap_not_a_shift(tmp_path):
    from vidbush.providers.reply_cache import ReplyCache
    first = ReplyCache(tmp_path)
    n0, _ = first.take("judge", "k")  # occurrence 0 failed: nothing stored
    n1, _ = first.take("judge", "k")
    first.put("judge", "m", "k", n1, {"v": 1})
    replay = ReplyCache(tmp_path)
    assert replay.take("judge", "k") == (0, None)  # occurrence 0 asks live again
    assert replay.take("judge", "k") == (1, {"v": 1})


def test_cap_inside_a_cascaded_stage_leaves_later_stages_pending(generic_seats, monkeypatch, tmp_path):
    settings = generic_seats
    _live(settings)
    _fake_chat(monkeypatch, [], cost=1.0)
    _script_project(settings, tmp_path)
    run_project(settings, "p1", stages=_stubs([]), echo=False)

    def judge_many(ctx):
        for i in range(10):
            ctx.seats.llm_json("director", "requery", "s", f"shot {i}", "judge")

    run_project(settings, "p1", stages=_stubs([], {"judge": judge_many}), force=["sourcing"], echo=False,
                cost_cap=3.0)
    disk = load_project(settings, "p1")
    assert all(disk.stages[s].status == StageStatus.pending for s in ("judge", "visuals", "motion", "assembly"))


def test_forced_stage_keeps_an_operator_edit_before_starting_from_its_snapshot(settings, tmp_path):
    _script_project(settings, tmp_path)
    seen: dict = {}
    run_project(settings, "p1", stages=_shot_stages(seen), echo=False)
    work = Path(load_project(settings, "p1").paths.work)
    data = read_json(work / "shotlist.json")
    data["shots"][0]["judge_notes"] = "operator edit"
    write_json(work / "shotlist.json", data)
    run_project(settings, "p1", stages=_shot_stages(seen), force=["judge"], echo=False)
    kept = list(work.glob("operator_edit_*.shotlist.json"))
    assert len(kept) == 1 and read_json(kept[0])["shots"][0]["judge_notes"] == "operator edit"
