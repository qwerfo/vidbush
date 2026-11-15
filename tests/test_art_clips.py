"""Part D: clip planning, prompts, checks, composition (no network)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vidbush.art import clips as C
from vidbush.models import ArtProfile, Shot, ShotList, ShotType, SourceKind
from vidbush.util import media_duration, video_size


def _shot(i: int, start: float, dur: float) -> Shot:
    return Shot(id=f"s{i}", index=i, beat_id="b", start=start, end=start + dur, duration=dur,
                type=ShotType.ai_image, intent=f"Wide view of a frozen river valley {i}")


def _list(n: int = 40, dur: float = 5.0) -> ShotList:
    return ShotList(project_id="p", shots=[_shot(i, i * dur, dur) for i in range(n)], duration=n * dur)


# ---------------------------------------------------------------- plan_clips

def test_plan_budget_opening_and_spread():
    sl = _list(80, 5.0)  # 400 s
    elig = {s.id: "ai_image" for s in sl.shots}
    elig["s3"] = "illustration"
    picks = C.plan_clips(sl, ArtProfile(clip_every_s=90, clip_opening_s=60), elig)
    assert len(picks) == 4
    assert picks[0] == "s3"  # opening first, illustration preferred
    starts = sorted(next(s.start for s in sl.shots if s.id == p) for p in picks[1:])
    step = (400 - 60) / 3
    for k, st in enumerate(starts):
        assert 60 + k * step <= st < 60 + (k + 1) * step


def test_plan_off_min_one_and_ai_flag_and_min_length():
    sl = _list(10, 5.0)  # 50 s, under one budget period
    elig = {s.id: "ai_image" for s in sl.shots}
    assert C.plan_clips(sl, ArtProfile(clip_every_s=0), elig) == []
    assert len(C.plan_clips(sl, ArtProfile(clip_every_s=90), elig)) == 1
    assert C.plan_clips(sl, ArtProfile(clip_every_s=90, clip_from_ai_stills=False), elig) == []
    assert C.plan_clips(sl, ArtProfile(clip_every_s=90, clip_min_s=6), elig) == []


def test_plan_prefers_designed_then_longer():
    sl = ShotList(project_id="p", shots=[_shot(0, 0, 4), _shot(1, 4, 6), _shot(2, 10, 3.5)], duration=13.5)
    assert C.plan_clips(sl, ArtProfile(clip_every_s=90), {"s0": "ai_image", "s1": "ai_image",
                                                           "s2": "illustrated_map"}) == ["s2"]
    assert C.plan_clips(sl, ArtProfile(clip_every_s=90), {"s0": "ai_image", "s1": "ai_image"}) == ["s1"]


# ---------------------------------------------------------------- motion_prompt

def test_motion_prompt_rules_no_names_deterministic():
    p = C.motion_prompt("illustrated_map", "Ferry crossing from Juneau toward Sitka at dawn")
    assert p == C.motion_prompt("illustrated_map", "Ferry crossing from Juneau toward Sitka at dawn")
    assert p.startswith("Animate this exact picture with subtle natural motion.")
    assert "coastlines" in p and "No text, letters or logos" in p and "No cut, no zoom" in p
    assert "Juneau" not in p and "Sitka" not in p
    assert chr(0x2014) not in p
    assert "painted style" in C.motion_prompt("illustration", "a cabin")
    assert len(C.motion_prompt("ai_image", "word " * 100).split()) < 120


# ---------------------------------------------------------------- media fixtures

@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    still = np.zeros((180, 320, 3), np.uint8)
    cv2.rectangle(still, (40, 30), (200, 150), (200, 120, 40), -1)
    cv2.circle(still, (260, 60), 30, (40, 200, 220), -1)
    sp = d / "still.png"
    cv2.imwrite(str(sp), still)
    raw = d / "raw.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-i", str(sp),
                    "-vf", "noise=alls=4:allf=t,format=yuv420p", "-t", "8", "-r", "24", "-c:v", "libx264",
                    str(raw)], check=True)
    other = np.full((180, 320, 3), 30, np.uint8)
    cv2.line(other, (0, 179), (319, 0), (255, 255, 255), 25)
    op = d / "other.png"
    cv2.imwrite(str(op), other)
    return SimpleNamespace(dir=d, still=sp, raw=raw, other=op)


class _Log:
    def __init__(self):
        self.lines = []

    def warn(self, msg, **kw):
        self.lines.append(("warn", msg, kw))

    def info(self, msg, **kw):
        self.lines.append(("info", msg, kw))


GOOD = {"same_subject": True, "same_layout": True, "new_objects": False, "text": False, "morphing": False}


def _ctx(answers, video_src=None):
    calls = []

    def video(prompt, still, out, seconds, stage):
        calls.append(prompt)
        out.write_bytes(Path(video_src).read_bytes())
        return out

    seats = SimpleNamespace(vision_json=lambda *a, **k: dict(answers), video=video)
    return SimpleNamespace(seats=seats, log=_Log(), stage="visuals", profile=SimpleNamespace(art=ArtProfile()),
                           calls=calls)


def _copy(media, name):
    p = media.dir / name
    p.write_bytes(media.raw.read_bytes())
    return p


# ---------------------------------------------------------------- check

def test_check_pass(media):
    r = C.check(_copy(media, "c_pass.mp4"), media.still, "illustration", _ctx(GOOD))
    assert r["ok"], r["reasons"]
    assert r["first_corr"] >= C.FIRST_FRAME_MIN and len(r["frames"]) == 3


@pytest.mark.parametrize("key", list(GOOD))
def test_check_each_failing_answer(media, key):
    bad = dict(GOOD, **{key: not GOOD[key]})
    r = C.check(_copy(media, f"c_{key}.mp4"), media.still, "illustration", _ctx(bad))
    assert not r["ok"] and any(key in x for x in r["reasons"])


def test_check_missing_answer_and_mock(media):
    r = C.check(_copy(media, "c_missing.mp4"), media.still, "illustration", _ctx({}))
    assert not r["ok"]
    r = C.check(_copy(media, "c_mock.mp4"), media.still, "illustration", _ctx(dict(GOOD, _mock=True)))
    assert not r["ok"] and any("mock" in x for x in r["reasons"])


def test_check_first_frame_mismatch(media):
    r = C.check(_copy(media, "c_mis.mp4"), media.other, "illustration", _ctx(GOOD))
    assert not r["ok"] and any("first frame" in x for x in r["reasons"])


# ---------------------------------------------------------------- compose

@pytest.mark.parametrize("duration", [7.2, 2.4])
def test_compose_lengths(media, duration):
    out = C.compose(media.raw, media.still, duration, media.dir / f"comp_{duration}.mp4", 3.0, 5.0)
    assert abs(media_duration(out) - duration) <= 0.1
    assert video_size(out) == (1920, 1080)


# ---------------------------------------------------------------- make_clip

def test_make_clip_end_to_end(media):
    shot = _shot(1, 0, 6.0)
    shot.judge_score = 7.5
    out = media.dir / "e2e"
    out.mkdir()
    ctx = _ctx(GOOD, media.raw)
    r = C.make_clip(shot, media.still, "illustrated_map", ctx, out)
    assert r["ok"] and Path(r["final"]).exists()
    cand = r["candidate"]
    assert cand.source == SourceKind.art_video and cand.media == "video" and cand.score == 7.5
    assert abs(media_duration(cand.local_path) - 6.0) <= 0.1
    C.apply_to_shot(shot, cand)
    assert shot.selected == cand.id and shot.type == ShotType.ai_video
    # resume: the raw clip on disk is reused, no second seat call
    C.make_clip(shot, media.still, "illustrated_map", ctx, out)
    assert len(ctx.calls) == 1


def test_make_clip_failure_keeps_still(media):
    shot = _shot(2, 0, 4.0)
    out = media.dir / "fail"
    out.mkdir()
    r = C.make_clip(shot, media.still, "ai_image", _ctx(dict(GOOD, morphing=True), media.raw), out)
    assert not r["ok"] and r["candidate"] is None

    def boom(*a, **k):
        raise RuntimeError("lane down")
    ctx = _ctx(GOOD)
    ctx.seats.video = boom
    r = C.make_clip(_shot(3, 0, 4.0), media.still, "ai_image", ctx, out)
    assert r["raw"] is None and not r["ok"]


def test_ai_still_clip_is_ai_video(media):
    out = media.dir / "ai"
    out.mkdir()
    r = C.make_clip(_shot(4, 0, 4.0), media.still, "ai_image", _ctx(GOOD, media.raw), out)
    assert r["candidate"].source == SourceKind.ai_video
    assert r["candidate"].attribution == "AI generated video"
