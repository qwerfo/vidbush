"""Judge image detail: first keyframe high, the rest low; every other vision caller stays low."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from vidbush.judge import judge as judge_mod
from vidbush.judge.qa import score_post_render
from vidbush.learn.style_learner import classify_shots
from vidbush.models import Candidate, Shot, ShotList, ShotType, SourceKind
from vidbush.providers import base, llm, seats as seats_mod
from vidbush.providers.base import CallResult, CostTracker
from vidbush.providers.seats import Seats
from vidbush.render import thumbnails as th
from vidbush.util import NullLogger


@pytest.fixture(autouse=True)
def _clean():
    base.HEALTH._down.clear()
    llm.reset_quirks()
    yield
    base.HEALTH._down.clear()
    llm.reset_quirks()


def _real_vision(settings) -> None:
    settings.offline = False
    settings.env.update({"VISION_BASE_URL": "http://v", "VISION_API_KEY": "k", "VISION_MODEL": "vm",
                         "VISION_CHEAP_MODEL": "cm"})


def _capture(monkeypatch, value: Any = None, cost: float = 0.0) -> list[list[str]]:
    """Patch llm.chat; each call appends the image detail list it was sent."""
    sent: list[list[str]] = []

    def chat(spec, system, user, **kw):  # noqa: ANN001
        sent.append([p["image_url"]["detail"] for p in user if p.get("type") == "image_url"])
        return CallResult(value=value or {"score": 80, "reason": "ok", "rejects": []}, provider=spec,
                          mock=False, cost_usd=cost)
    monkeypatch.setattr(llm, "chat", chat)
    return sent


def _frames(tmp_path: Path, n: int = 3) -> list[str]:
    out = []
    for i in range(n):
        p = tmp_path / f"k{i}.jpg"
        Image.new("RGB", (1280, 720), (40 * i, 90, 120)).save(p)
        out.append(str(p))
    return out


def _judge(settings, tmp_path: Path, n: int = 3) -> SimpleNamespace:
    shot = Shot(id="s0", beat_id="b0", start=0, end=4, duration=4, type=ShotType.footage, intent="A harbour")
    cand = Candidate(id="c0", source=SourceKind.local, media="video", local_path="x.mp4",
                     keyframes=_frames(tmp_path, n), width=1920, height=1080)
    ctx = SimpleNamespace(settings=settings, seats=Seats(settings, CostTracker("p")), stage="judge",
                          log=NullLogger())
    judge_mod.score_candidate(shot, cand, ctx)
    return ctx


def test_judge_first_keyframe_high_rest_low(settings, monkeypatch, tmp_path):
    _real_vision(settings)
    sent = _capture(monkeypatch)
    _judge(settings, tmp_path)
    assert sent == [["high", "low", "low"]]


def test_config_can_switch_first_to_low(settings, monkeypatch, tmp_path):
    _real_vision(settings)
    settings.judge.first_keyframe_detail = "low"
    sent = _capture(monkeypatch)
    _judge(settings, tmp_path)
    assert sent == [["low", "low", "low"]]


def test_invalid_detail_falls_back_to_low_and_logs_once(settings, monkeypatch, tmp_path):
    _real_vision(settings)
    settings.judge.first_keyframe_detail = "ultra_sharp_test_value"
    seats_mod._BAD_DETAILS_LOGGED.discard("ultra_sharp_test_value")
    sent = _capture(monkeypatch)
    warns: list[str] = []
    s = Seats(settings, CostTracker("p"))
    monkeypatch.setattr(s.log, "warn", lambda msg, **f: warns.append(msg))
    imgs = _frames(tmp_path, 2)
    for _ in range(2):
        s.vision_json("judge", "judge", "s", "p", imgs, "judge",
                      details=judge_mod.keyframe_details(2, settings))
    assert sent == [["low", "low"], ["low", "low"]]
    assert warns.count("unknown image detail, using low") == 1


def test_auto_is_sent_as_auto(settings, monkeypatch, tmp_path):
    _real_vision(settings)
    sent = _capture(monkeypatch)
    Seats(settings, CostTracker("p")).vision_json("judge", "judge", "s", "p", _frames(tmp_path, 2), "judge",
                                                  details=["AUTO"])
    assert sent == [["auto", "low"]]


def test_default_vision_json_and_learn_style_send_low(settings, monkeypatch, tmp_path):
    _real_vision(settings)
    sent = _capture(monkeypatch, value={"type": "b_roll"})
    s = Seats(settings, CostTracker("p"))
    s.vision_json("vision", "x", "s", "p", _frames(tmp_path, 3), "x")
    classify_shots([_frames(tmp_path, 2)], s)
    assert sent == [["low", "low", "low"], ["low", "low"]]


def test_qa_pass_sends_all_low(settings, monkeypatch, tmp_path):
    _real_vision(settings)
    sent = _capture(monkeypatch)
    shot = Shot(id="s0", beat_id="b0", start=0, end=4, duration=4, type=ShotType.footage, intent="A harbour")
    ctx = SimpleNamespace(settings=settings, seats=Seats(settings, CostTracker("p")), stage="packaging",
                          log=NullLogger())
    out = score_post_render(ShotList(project_id="p", duration=4, shots=[shot]), {"s0": _frames(tmp_path, 3)}, ctx)
    assert "s0" in out
    assert sent and all(d and set(d) == {"low"} for d in sent)


def test_thumbnail_scoring_sends_low(settings, monkeypatch, tmp_path, make_ctx):
    _real_vision(settings)
    sent = _capture(monkeypatch)
    ctx = make_ctx(mode="brief")
    img = _frames(tmp_path, 1)[0]
    shot = Shot(id="s0", beat_id="b0", start=0, end=4, duration=4, type=ShotType.footage, intent="A harbour",
                candidates=[Candidate(id="c0", source=SourceKind.wikimedia, local_path=img, keyframes=[img])],
                selected="c0")
    th.best_frames(ShotList(project_id="p", duration=4, shots=[shot]), ctx, 1)
    assert sent == [["low"]]


def test_cost_estimate_is_one_high_plus_two_low(settings, monkeypatch, tmp_path):
    _real_vision(settings)
    _capture(monkeypatch)
    ctx = _judge(settings, tmp_path)
    c = settings.cost
    assert ctx.seats.tracker.report.entries[-1].cost_usd == pytest.approx(
        c.vision_per_image_high_usd + 2 * c.vision_per_image_usd)


def test_cost_uses_provider_cost_when_given(settings, monkeypatch, tmp_path):
    _real_vision(settings)
    _capture(monkeypatch, cost=0.0123)
    ctx = _judge(settings, tmp_path)
    assert ctx.seats.tracker.report.entries[-1].cost_usd == pytest.approx(0.0123)


def test_high_price_default_matches_tile_ratio(settings):
    # a 640 x 360 keyframe at high detail is 2 tiles: 85 + 170 * 2 = 425 tokens versus 85 at low
    c = settings.cost
    assert c.vision_per_image_high_usd == pytest.approx(c.vision_per_image_usd * 425 / 85)


def test_mock_vision_ignores_detail(settings, tmp_path):
    s = Seats(settings, CostTracker("p"))
    imgs = _frames(tmp_path, 3)
    a = s.vision_json("judge", "judge", "s", "p", imgs, "judge", intent="harbour", details=["high", "low", "low"])
    b = s.vision_json("judge", "judge", "s", "p", imgs, "judge", intent="harbour")
    assert a == b and a["_mock"] is True


def _sent_size(part: dict[str, Any]) -> tuple[int, int]:
    b64 = part["image_url"]["url"].split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(b64))) as im:
        return im.size


def test_full_size_first_keyframe_is_downscaled_before_high_detail(settings, monkeypatch, tmp_path):
    # an AI image candidate's keyframe is the 1920 x 1080 file itself; it must go out at keyframe size
    _real_vision(settings)
    big = tmp_path / "gen.png"
    Image.new("RGB", (1920, 1080), (10, 20, 30)).save(big)
    sizes: list[tuple[int, int]] = []

    def chat(spec, system, user, **kw):  # noqa: ANN001
        sizes.extend(_sent_size(p) for p in user if p.get("type") == "image_url")
        return CallResult(value={"score": 80, "reason": "ok", "rejects": []}, provider=spec, mock=False)
    monkeypatch.setattr(llm, "chat", chat)
    shot = Shot(id="s0", beat_id="b0", start=0, end=4, duration=4, type=ShotType.ai_image, intent="A harbour")
    cand = Candidate(id="c0", source=SourceKind.local, media="image", local_path=str(big),
                     keyframes=[str(big)], width=1920, height=1080)
    ctx = SimpleNamespace(settings=settings, seats=Seats(settings, CostTracker("p")), stage="judge",
                          log=NullLogger())
    judge_mod.score_candidate(shot, cand, ctx)
    assert sizes == [(640, 360)]


def test_small_image_bytes_are_sent_unchanged(tmp_path):
    p = tmp_path / "k.jpg"
    Image.new("RGB", (640, 360), (1, 2, 3)).save(p)
    part = llm.image_part(p, detail="high")
    assert base64.b64decode(part["image_url"]["url"].split(",", 1)[1]) == p.read_bytes()


def test_unreadable_image_is_sent_as_is(tmp_path):
    p = tmp_path / "broken.jpg"
    p.write_bytes(b"not an image")
    assert llm.image_part(p)["image_url"]["url"].endswith("bm90IGFuIGltYWdl")
