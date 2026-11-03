"""Judge and director loop: threshold selection, requery, AI conversion, never empty, diversity."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Optional

import pytest

from vidbush.config import REPO_ROOT
from vidbush.judge import director_loop as dl
from vidbush.judge.judge import parse_verdict, score_candidate
from vidbush.models import Candidate, MotionKind, MotionSpec, Shot, ShotList, ShotType, SourceKind

ASSETS = REPO_ROOT / "fixtures" / "sample_assets"


def img(n: int) -> str:
    return str(ASSETS / f"sample_image_{n:02d}.jpg")


def cand(shot_id: str, n: int, image: int, media: str = "image", **kw: Any) -> Candidate:
    base = dict(id=f"{shot_id}_local_{n}", source=SourceKind.local, local_path=img(image), media=media,
                width=2400, height=1350, keyframes=[img(image)], attribution="Vidbush fixtures")
    base.update(kw)
    return Candidate(**base)


def shot(i: int, typ: ShotType = ShotType.image, cands: Optional[list[Candidate]] = None, **kw: Any) -> Shot:
    sid = f"shot_{i:03d}"
    return Shot(id=sid, index=i, beat_id="b1", scene_id="s1", start=float(i), end=float(i + 1), duration=1.0,
                type=typ, intent="Archival footage of steam shovels digging the Culebra Cut in 1910",
                must_show=["steam shovel"], queries=["vintage steam shovel 1910s archival"],
                candidates=cands or [], **kw)


def shotlist(shots: list[Shot]) -> ShotList:
    return ShotList(project_id="t1", duration=float(len(shots)), shots=shots)


def fake_vision(scores: dict[str, float], ai_score: float = 85.0, rejects: Optional[dict[str, list[str]]] = None):
    """vision_json replacement: score by keyframe file name, AI images (work/ai) get ai_score."""
    calls: list[str] = []

    def vision_json(role: str, task: str, system: str, prompt: str, images: list[str], stage: str,
                    intent: str = "") -> dict[str, Any]:
        name = Path(images[0]).name
        calls.append(name)
        score = ai_score if "/ai/" in images[0] else scores.get(name, 50.0)
        return {"score": score, "reason": f"scored {name}", "rejects": (rejects or {}).get(name, []), "_mock": False}

    vision_json.calls = calls  # type: ignore[attr-defined]
    return vision_json


@pytest.fixture
def ctx(make_ctx):
    c = make_ctx()
    c.stage = "judge"
    return c


@pytest.fixture
def registry(monkeypatch):
    """Fake source registry: records calls and returns what the test queues."""
    state: dict[str, Any] = {"calls": [], "result": []}
    mod = types.ModuleType("vidbush.source.registry")

    def source_shot(s: Shot, c: Any, queries: Optional[list[str]] = None, exclude_ids: Any = ()) -> list[Candidate]:
        state["calls"].append({"shot": s.id, "queries": list(queries or []), "exclude": tuple(exclude_ids)})
        return list(state["result"])

    mod.source_shot = source_shot  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vidbush.source.registry", mod)
    return state


def test_selects_top_candidate_at_or_above_threshold(ctx, monkeypatch, registry):
    s = shot(0, cands=[cand("shot_000", 1, 1), cand("shot_000", 2, 2), cand("shot_000", 3, 3)])
    monkeypatch.setattr(ctx.seats, "vision_json", fake_vision(
        {"sample_image_01.jpg": 60, "sample_image_02.jpg": 70, "sample_image_03.jpg": 65}))
    out = dl.judge_and_select(shotlist([s]), ctx).shots[0]
    assert out.selected == "shot_000_local_2"  # exactly at threshold passes
    assert out.judge_score == 70
    assert out.attempts == 1 and "no_match_accepted" not in out.flags
    assert registry["calls"] == []


def test_rejected_candidate_never_selected_while_alternative_passes(ctx, monkeypatch, registry):
    s = shot(0, cands=[cand("shot_000", 1, 1), cand("shot_000", 2, 2)])
    monkeypatch.setattr(ctx.seats, "vision_json", fake_vision(
        {"sample_image_01.jpg": 95, "sample_image_02.jpg": 75},
        rejects={"sample_image_01.jpg": ["burned in subtitles"]}))
    out = dl.judge_and_select(shotlist([s]), ctx).shots[0]
    assert out.selected == "shot_000_local_2"
    assert out.candidates[0].reject_reason == "burned_in_captions"


def test_requery_then_ai_conversion_keeps_overlay(ctx, monkeypatch, registry):
    overlay = MotionSpec(kind=MotionKind.lower_third, duration=1.0, background="transparent",
                         payload={"title": "Culebra Cut"})
    s = shot(0, ShotType.image_overlay, [cand("shot_000", 1, 1)], overlay=overlay)
    registry["result"] = [cand("shot_000", 2, 2)]
    monkeypatch.setattr(ctx.seats, "vision_json", fake_vision(
        {"sample_image_01.jpg": 40, "sample_image_02.jpg": 55}, ai_score=82))
    out = dl.judge_and_select(shotlist([s]), ctx).shots[0]
    assert len(registry["calls"]) == 1
    qs = registry["calls"][0]["queries"]
    assert qs and all("1910" not in q and "vintage" not in q.lower() for q in qs)
    assert "steam shovel" in qs
    assert out.type == ShotType.ai_image and out.overlay is not None
    sel = out.selected_candidate()
    assert sel is not None and sel.source == SourceKind.ai_image and Path(sel.local_path).exists()
    assert sel.licence == "AI generated" and "mock" in sel.flags
    assert out.attempts == 3 and out.judge_score == 82
    assert "mock" in out.flags and "no_match_accepted" not in out.flags
    assert {c.id for c in out.candidates} >= {"shot_000_local_1", "shot_000_local_2"}


def test_never_empty_without_any_candidates(ctx, monkeypatch, registry):
    ctx.settings.judge.max_attempts = 1
    s = shot(0, ShotType.footage)
    monkeypatch.setattr(ctx.seats, "vision_json", fake_vision({}, ai_score=40))
    out = dl.judge_and_select(shotlist([s]), ctx).shots[0]
    sel = out.selected_candidate()
    assert sel is not None and sel.source == SourceKind.ai_image
    assert "no_match_accepted" in out.flags


def test_best_available_accepted_and_type_restored(ctx, monkeypatch, registry):
    s = shot(0, ShotType.image, [cand("shot_000", 1, 1)])
    monkeypatch.setattr(ctx.seats, "vision_json", fake_vision({"sample_image_01.jpg": 62}, ai_score=30))
    out = dl.judge_and_select(shotlist([s]), ctx).shots[0]
    assert out.selected == "shot_000_local_1"
    assert out.type == ShotType.image
    assert "no_match_accepted" in out.flags and out.attempts == 3


def test_adjacent_shots_avoid_same_asset_within_five_points(ctx, monkeypatch, registry):
    a0, b0 = cand("shot_000", 1, 1), cand("shot_000", 2, 2)
    a1, b1 = cand("shot_001", 1, 1), cand("shot_001", 2, 2)
    monkeypatch.setattr(ctx.seats, "vision_json", fake_vision({"sample_image_01.jpg": 90, "sample_image_02.jpg": 86}))
    sl = dl.judge_and_select(shotlist([shot(0, cands=[a0, b0]), shot(1, cands=[a1, b1])]), ctx)
    assert sl.shots[0].selected == "shot_000_local_1"
    assert sl.shots[1].selected == "shot_001_local_2"


def test_adjacent_same_asset_kept_when_alternative_much_worse(ctx, monkeypatch, registry):
    monkeypatch.setattr(ctx.seats, "vision_json", fake_vision({"sample_image_01.jpg": 92, "sample_image_02.jpg": 75}))
    sl = dl.judge_and_select(shotlist([shot(0, cands=[cand("shot_000", 1, 1)]),
                                       shot(1, cands=[cand("shot_001", 1, 1), cand("shot_001", 2, 2)])]), ctx)
    assert sl.shots[1].selected == "shot_001_local_1"


def test_motion_graphic_skipped_and_ai_image_planned_gets_candidate(ctx, monkeypatch, registry):
    mg = shot(0, ShotType.motion_graphic)
    planned = shot(1, ShotType.ai_image, ai_prompt="Black and white photograph of the Gatun locks. No text.")
    monkeypatch.setattr(ctx.seats, "vision_json", fake_vision({}, ai_score=80))
    sl = dl.judge_and_select(shotlist([mg, planned]), ctx)
    assert sl.shots[0].selected is None and sl.shots[0].candidates == []
    assert sl.shots[1].selected_candidate().source == SourceKind.ai_image
    assert registry["calls"] == []


def test_ai_video_falls_back_to_ai_image_without_real_seat(ctx, monkeypatch, registry):
    monkeypatch.setattr(ctx.seats, "vision_json", fake_vision({}, ai_score=80))
    sl = dl.judge_and_select(shotlist([shot(0, ShotType.ai_video)]), ctx)
    assert sl.shots[0].type == ShotType.ai_image
    assert sl.shots[0].selected_candidate().media == "image"


def test_mock_vision_marks_mock_flag(ctx, registry):
    s = shot(0, cands=[cand("shot_000", 1, 7), cand("shot_000", 2, 10)])
    out = dl.judge_and_select(shotlist([s]), ctx).shots[0]
    assert out.selected is not None and "mock" in out.flags
    assert "judge" in ctx.seats.mock_roles


def test_local_rejects_low_res_and_missing_keyframes(ctx, monkeypatch):
    s = shot(0)
    vision = fake_vision({})
    monkeypatch.setattr(ctx.seats, "vision_json", vision)
    low = cand("shot_000", 1, 1, media="video", width=854, height=480)
    assert score_candidate(s, low, ctx)[2].startswith("resolution below 720p")
    small = cand("shot_000", 2, 1, width=800, height=450)
    assert "too small" in score_candidate(s, small, ctx)[2]
    missing = cand("shot_000", 3, 1, keyframes=["/nonexistent.jpg"])
    assert score_candidate(s, missing, ctx)[2] == "missing keyframes"
    assert vision.calls == []


def test_parse_verdict_shapes():
    assert parse_verdict({"score": 7.5, "reason": "ok"})[0] == 7.5  # a poor 0..100 score stays poor
    assert parse_verdict({"score": 1})[0] == 1.0
    assert parse_verdict({"score": 0.8})[0] == pytest.approx(80.0)
    assert parse_verdict({"score": "88", "rejects": "watermark"})[2] == ["watermark"]
    assert parse_verdict({"reason": "no score"}) is None
    assert parse_verdict(None) is None


def test_heuristic_queries_broaden():
    s = shot(0)
    qs = dl.heuristic_queries(s, "Digging the Culebra Cut")
    assert "steam shovel" in qs
    assert all(q.lower() not in {x.lower() for x in s.queries} for q in qs)
    assert any("Culebra" in q for q in qs)
    assert all("1910" not in q for q in qs)


def test_add_candidates_renumbers_and_dedupes():
    s = shot(0, cands=[cand("shot_000", 1, 1)])
    added = dl.add_candidates(s, [cand("shot_000", 1, 2), cand("shot_000", 5, 1)])
    assert [c.id for c in added] == ["shot_000_local_2"]
    assert len(s.candidates) == 2


def test_reuse_penalised_across_motion_graphics_and_far_apart(ctx, monkeypatch, registry):
    monkeypatch.setattr(ctx.seats, "vision_json", fake_vision({"sample_image_01.jpg": 90, "sample_image_02.jpg": 87}))
    shots = [shot(0, cands=[cand("shot_000", 1, 1)]), shot(1, ShotType.motion_graphic),
             shot(2, cands=[cand("shot_002", 1, 1), cand("shot_002", 2, 2)])]
    sl = dl.judge_and_select(shotlist(shots), ctx)
    assert sl.shots[2].selected == "shot_002_local_2"  # the motion graphic no longer resets diversity
    far = dl._Usage()
    far.add(cand("x", 1, 1), 0.0)
    assert far.penalty(cand("y", 1, 1), 300.0) == dl.DIVERSITY_GAP  # reuse costs something anywhere
    assert far.penalty(cand("y", 1, 1), 30.0) == dl.DIVERSITY_GAP + dl.RECENT_PENALTY


def test_reused_video_needs_a_different_segment():
    usage = dl._Usage()
    usage.add(cand("a", 1, 1, media="video", url="http://x/v.mp4", in_point=10.0), 0.0)
    same = cand("b", 1, 1, media="video", url="http://x/v.mp4", in_point=11.0)
    other = cand("b", 2, 1, media="video", url="http://x/v.mp4", in_point=20.0)
    assert usage.penalty(same, 200.0) > usage.penalty(other, 200.0)
