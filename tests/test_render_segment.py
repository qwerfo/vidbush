"""Segment rendering: exact frames, letterbox, overlay, captions, Ken Burns smoothness, cache keys."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from test_render_helpers import FPS, build_shotlist, caption_lines, count_frames, frame_at
from vidbush.models import StyleProfile
from vidbush.render.cache import frame_range, kenburns_for, lines_in_window, segment_hash
from vidbush.render.segment import kb_progress, render_segment, still_frames
from vidbush.util import ffprobe


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    return build_shotlist(tmp_path_factory.mktemp("media"))


def _profile(**kw) -> StyleProfile:
    p = StyleProfile(id="test")
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_footage_segment_exact_frames_letterbox_and_format(media, settings, tmp_path) -> None:
    shot = media.shots[0]
    fr = frame_range(shot, FPS)
    out = render_segment(shot, _profile(), caption_lines(), settings, tmp_path, fr)
    assert count_frames(out) == fr[1] - fr[0] == 36
    streams = ffprobe(out)["streams"]
    assert len(streams) == 1 and streams[0]["codec_type"] == "video"
    v = streams[0]
    assert (v["codec_name"], v["profile"], v["pix_fmt"], v["r_frame_rate"]) == ("h264", "High", "yuv420p", "30/1")
    assert (v["width"], v["height"]) == (1920, 1080)
    img = frame_at(out, 20)
    assert img[:128].max() <= 4, "top bar is black"
    assert img[128:138].mean() < 8, "only codec bleed at the bar edge"
    assert img[946:, :300].max() <= 6 and img[946:, 1700:].max() <= 6, "bottom bar outside captions is black"
    assert img[200:900].mean() > 40, "picture area is not black"
    assert img[942:, 600:1320].max() > 200, "caption burned into the bottom band"


def test_still_segment_overlay_and_fade(media, settings, tmp_path) -> None:
    shot = media.shots[1]
    fr = frame_range(shot, FPS)
    out = render_segment(shot, _profile(), [], settings, tmp_path, fr)
    assert count_frames(out) == fr[1] - fr[0] == 39
    first, mid = frame_at(out, 0), frame_at(out, 20)
    assert first.mean() < 8, "starts from black (fade)"
    r, g, b = (int(c) for c in mid[500, 960])
    assert r > 200 and g < 60 and b < 60, "red overlay box drawn over the still"


def test_ken_burns_motion_is_smooth() -> None:
    """Consecutive frame differences of a slow pan and zoom never stall or spike (no rounding jitter)."""
    yy, xx = np.mgrid[0:600, 0:1000]
    img = np.stack([(np.sin(xx / 9.0) * 120 + 128), (np.cos(yy / 7.0) * 120 + 128),
                    ((xx + yy) % 256)], -1).astype(np.uint8)
    kb = {"zoom0": 1.0, "zoom1": 1.12, "x0": 0.5, "y0": 0.5, "x1": 0.53, "y1": 0.48}
    frames = [np.frombuffer(f, np.uint8).reshape(270, 480, 3).astype(np.float32)
              for f in still_frames(img, kb, 60, 480, 270)]
    diffs = np.array([np.abs(a - b).mean() for a, b in zip(frames, frames[1:])])
    core = diffs[5:-5]
    assert core.min() > 0.2, "no frozen frames"
    assert core.max() / core.min() < 2.0, "steady motion, no jitter"


def test_kb_progress_monotonic_and_bounded() -> None:
    vals = [kb_progress(i, 50) for i in range(50)]
    assert vals[0] == 0.0 and abs(vals[-1] - 1.0) < 1e-9
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_motion_segment_and_cache_reuse(media, settings, tmp_path) -> None:
    shot = media.shots[2]
    fr = frame_range(shot, FPS)
    out = render_segment(shot, _profile(), caption_lines(), settings, tmp_path, fr)
    assert count_frames(out) == fr[1] - fr[0] == 34
    mtime = out.stat().st_mtime_ns
    again = render_segment(shot, _profile(), caption_lines(), settings, tmp_path, fr)
    assert again == out and again.stat().st_mtime_ns == mtime


def test_missing_media_renders_slate(media, settings, tmp_path) -> None:
    shot = media.shots[0].model_copy(deep=True)
    shot.candidates[0].local_path = str(tmp_path / "gone.mp4")
    out = render_segment(shot, _profile(), [], settings, tmp_path, (0, 15))
    assert count_frames(out) == 15


def test_hash_changes_only_with_relevant_inputs(media, settings) -> None:
    prof = _profile()
    shot = media.shots[1]
    lines = lines_in_window(caption_lines(), shot.start, shot.end)
    base = segment_hash(shot, prof, lines, settings)
    assert segment_hash(shot, prof, lines, settings) == base
    moved = shot.model_copy(deep=True)
    moved.candidates[0].crop = [0.1, 0.1, 0.8, 0.8]
    assert segment_hash(moved, prof, lines, settings) != base
    edited = [dict(d) for d in lines]
    edited[0]["text"] = "changed words here"
    assert segment_hash(shot, prof, edited, settings) != base
    assert segment_hash(shot, _profile(), lines, settings.model_copy(update={})) == base
    settings2 = settings.model_copy(deep=True)
    settings2.render.crf = 30
    assert segment_hash(shot, prof, lines, settings2) != base
    assert segment_hash(shot, prof, lines, settings, fade_out=True) != base


def test_kenburns_fallback_when_module_missing(media, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "vidbush.visuals.kenburns", None)
    kb = kenburns_for(media.shots[1], _profile())
    assert set(kb) == {"zoom0", "zoom1", "x0", "y0", "x1", "y1"}
    assert 1.0 <= min(kb["zoom0"], kb["zoom1"]) and max(kb["zoom0"], kb["zoom1"]) <= 1.13
    assert kb == kenburns_for(media.shots[1], _profile())


def test_colour_lut_applied_to_footage(media, settings, tmp_path) -> None:
    cube = tmp_path / "warm.cube"
    rows = [f"{r:.3f} 0.000 {b:.3f}" for b in (0.0, 1.0) for _g in (0.0, 1.0) for r in (0.0, 1.0)]
    cube.write_text("LUT_3D_SIZE 2\n" + "\n".join(rows) + "\n", encoding="utf-8")
    prof = _profile()
    prof.colour.lut_path = str(cube)
    shot = media.shots[0]
    plain = segment_hash(shot, _profile(), [], settings)
    assert segment_hash(shot, prof, [], settings) != plain
    out = render_segment(shot, prof, [], settings, tmp_path, (0, 6))
    img = frame_at(out, 3)
    assert img[200:900, :, 1].max() < 30, "LUT zeroes the green channel"
