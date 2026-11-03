"""Thumbnails: three distinct 1280x720 compositions, frame ranking, fallbacks, readable text layout."""
from __future__ import annotations

import numpy as np
from PIL import Image

from test_render_helpers import build_shotlist, make_still
from vidbush.render.thumbnails import best_frames, fit_text, font_path, make_thumbnails, subject_box


def test_three_distinct_thumbnails(make_ctx) -> None:
    ctx = make_ctx(mode="brief")
    shotlist = build_shotlist(ctx.work / "media")
    paths = make_thumbnails(shotlist, ["The Panama Canal: The Shortcut That Cost a Fortune"], ctx)
    assert [p.name for p in paths] == ["thumb_1.png", "thumb_2.png", "thumb_3.png"]
    arrays = [np.asarray(Image.open(p).convert("RGB"), dtype=np.int16) for p in paths]
    assert all(a.shape == (720, 1280, 3) for a in arrays)
    assert np.abs(arrays[0] - arrays[1]).mean() > 10 and np.abs(arrays[1] - arrays[2]).mean() > 10
    for a in arrays:
        assert (a.min(axis=2) > 235).mean() > 0.01 or (a.max(axis=2) < 30).mean() > 0.01, "bold text present"


def test_best_frames_ranked_by_score_motion_last(make_ctx) -> None:
    ctx = make_ctx(mode="brief")
    shotlist = build_shotlist(ctx.work / "media")
    other = make_still(ctx.work / "media" / "other.jpg", hue=80)
    shotlist.shots[0].candidates[0].keyframes = [str(other)]
    frames = best_frames(shotlist, ctx, 5)
    assert frames[0].name == "still.jpg" and frames[1].name == "other.jpg"
    assert len(frames) == 3 and frames[-1].suffix == ".png", "motion clip frame grabbed and ranked last"


def test_no_frames_falls_back_to_ai_image(make_ctx) -> None:
    ctx = make_ctx(mode="script")
    shotlist = build_shotlist(ctx.work / "media")
    for s in shotlist.shots:
        s.candidates, s.selected = [], None
    paths = make_thumbnails(shotlist, [], ctx)
    assert len(paths) == 3 and all(Image.open(p).size == (1280, 720) for p in paths)


def test_fit_text_two_lines_max_and_large(settings) -> None:
    lines, font = fit_text("13,000 TO 5,200 MILES", font_path(settings), 690, 520)
    assert 1 <= len(lines) <= 2 and font.size >= 90
    assert max(font.getlength(ln) for ln in lines) <= 690


def test_subject_box_finds_detail(tmp_path) -> None:
    im = Image.new("RGB", (1600, 900), (20, 20, 20))
    detail = np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    im.paste(Image.fromarray(detail), (1100, 500))
    x, y, w, h = subject_box(im)
    assert x + w > 0.8 and y + h > 0.8


def test_text_length_limits() -> None:
    from vidbush.render.thumbnails import heuristic_text, _short
    assert _short("one two three four five") == "" and _short("a very long wordlengthsss here") == ""
    assert 0 < len(heuristic_text("Extraordinarily Underappreciated Transcontinental Engineering")) <= 24
