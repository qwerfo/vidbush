"""Local pre-filter: burned-in text, corner logos and talking heads are rejected before the vision judge.

Every frame is generated here (PIL and numpy); OCR and the haar cascade run for real and offline.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from vidbush.config import REPO_ROOT, PrefilterConfig
from vidbush.judge import director_loop as dl
from vidbush.judge import prefilter as pf
from vidbush.models import Candidate, Shot, ShotList, ShotType, SourceKind

W, H = 640, 360
FONT = REPO_ROOT / "assets" / "fonts" / "Inter-Bold.otf"
LINES = ["The canal opened to ships in nineteen fourteen", "Thousands of workers dug through the hills",
         "Disease killed more men than the machines"]


# ---------------------------------------------------------------- synthetic frames

def scene(seed: int) -> Image.Image:
    """Textured moving content: blurred colour noise, different for every seed."""
    rng = np.random.default_rng(seed)
    base = (rng.random((H // 8, W // 8, 3)) * 200 + 30).astype(np.uint8)
    return Image.fromarray(base).resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(2))


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT), size)


def subtitle(im: Image.Image, text: str) -> Image.Image:
    ImageDraw.Draw(im).text((W // 2, int(H * 0.86)), text, font=font(24), fill="white", stroke_width=3,
                            stroke_fill="black", anchor="mm")
    return im


def corner_word(im: Image.Image) -> Image.Image:
    ImageDraw.Draw(im).text((W - 40, H - 22), "NEWS", font=font(16), fill="white", stroke_width=2,
                            stroke_fill="black", anchor="mm")
    return im


def badge(im: Image.Image) -> Image.Image:
    """An opaque channel bug in the top right corner: identical on every keyframe."""
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([W - 84, 14, W - 18, 46], radius=6, fill=(245, 245, 245), outline=(20, 20, 20), width=2)
    d.text((W - 51, 30), "TV9", font=font(18), fill=(200, 30, 30), anchor="mm")
    return im


def face(im: Image.Image, cx: int, cy: int, fh: int, blur: float = 3) -> Image.Image:
    """Shaded frontal face: hair, skin ellipse, eye sockets, brows, nose shadow, mouth, then a blur."""
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    fw = int(fh * 0.75)
    x0, y0 = cx - fw // 2, cy - fh // 2
    d.ellipse([x0 - fw * 0.08, y0 - fh * 0.12, x0 + fw * 1.08, y0 + fh * 0.6], fill=(40, 28, 20, 255))
    d.ellipse([x0, y0, x0 + fw, y0 + fh], fill=(205, 160, 130, 255))
    ey = y0 + fh * 0.40
    for ex in (cx - fw * 0.2, cx + fw * 0.2):
        d.ellipse([ex - fw * 0.13, ey - fh * 0.06, ex + fw * 0.13, ey + fh * 0.06], fill=(110, 75, 60, 255))
        d.ellipse([ex - fw * 0.06, ey - fh * 0.03, ex + fw * 0.06, ey + fh * 0.03], fill=(30, 20, 20, 255))
        d.rectangle([ex - fw * 0.14, ey - fh * 0.12, ex + fw * 0.14, ey - fh * 0.09], fill=(50, 35, 25, 255))
    d.polygon([(cx, ey), (cx - fw * 0.08, y0 + fh * 0.65), (cx + fw * 0.08, y0 + fh * 0.65)], fill=(170, 125, 100, 255))
    d.ellipse([cx - fw * 0.07, y0 + fh * 0.62, cx + fw * 0.07, y0 + fh * 0.68], fill=(120, 80, 65, 255))
    d.ellipse([cx - fw * 0.18, y0 + fh * 0.76, cx + fw * 0.18, y0 + fh * 0.83], fill=(130, 60, 60, 255))
    out = im.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB").filter(ImageFilter.GaussianBlur(blur))


def plain(seed: int) -> Image.Image:
    return Image.new("RGB", (W, H), (70 + seed * 3, 80, 95))


def save(tmp: Path, name: str, frames: list[Image.Image]) -> list[str]:
    out = []
    for i, im in enumerate(frames):
        p = tmp / f"{name}_k{i + 1}.jpg"
        im.save(p, quality=90)
        out.append(str(p))
    return out


def video(keyframes: list[str], cid: str = "c1", source: SourceKind = SourceKind.pexels, **kw: Any) -> Candidate:
    base = dict(id=cid, source=source, media="video", local_path=keyframes[0], width=1920, height=1080,
                in_point=0.0, out_point=10.0, keyframes=keyframes)
    base.update(kw)
    return Candidate(**base)


def check(tmp_path: Path, name: str, frames: list[Image.Image], cfg: Optional[PrefilterConfig] = None,
          media: str = "video") -> Optional[str]:
    return pf.prefilter_candidate(video(save(tmp_path, name, frames), media=media), cfg or PrefilterConfig())


# ---------------------------------------------------------------- text

def test_rapidocr_reads_synthetic_subtitle(tmp_path):
    frame = pf.load_frames(save(tmp_path, "ocr", [subtitle(scene(1), LINES[0])]))[0]
    boxes = pf.read_text(frame, PrefilterConfig())
    assert boxes and max(conf for _, conf in boxes) >= 0.6
    assert pf.subtitle_lines(boxes, W, H, PrefilterConfig())


def test_subtitles_on_two_keyframes_rejected(tmp_path):
    frames = [subtitle(scene(1), LINES[0]), subtitle(scene(2), LINES[1]), scene(3)]
    reason = check(tmp_path, "sub2", frames)
    assert reason is not None and reason.startswith("burned_in_captions (prefilter:")
    assert "2 of 3" in reason


def test_subtitle_on_one_keyframe_accepted(tmp_path):
    assert check(tmp_path, "sub1", [subtitle(scene(1), LINES[0]), scene(2), scene(3)]) is None


def test_small_corner_word_is_not_a_subtitle(tmp_path):
    frames = [corner_word(scene(i)) for i in (1, 2, 3)]
    keyframes = save(tmp_path, "word", frames)
    loaded = pf.load_frames(keyframes)
    assert any(pf.read_text(f, PrefilterConfig()) for f in loaded)  # the word is read ...
    assert pf.prefilter_candidate(video(keyframes), PrefilterConfig()) is None  # ... but is no subtitle


def test_subtitle_lines_merge_boxes_on_one_line():
    cfg = PrefilterConfig()
    words = [((100.0, 300.0, 160.0, 320.0), 0.9), ((170.0, 301.0, 230.0, 321.0), 0.9),
             ((240.0, 299.0, 300.0, 319.0), 0.9)]
    lines = pf.subtitle_lines(words, W, H, cfg)
    assert len(lines) == 1 and (lines[0][0], lines[0][2]) == (100.0, 300.0)
    assert pf.subtitle_lines(words[:1], W, H, cfg) == []  # one word is too narrow
    high = [((100.0, 50.0, 500.0, 80.0), 0.9)]
    assert pf.subtitle_lines(high, W, H, cfg) == []  # a title at the top is not in the band
    low_conf = [((100.0, 300.0, 500.0, 320.0), 0.3)]
    assert pf.subtitle_lines(low_conf, W, H, cfg) == []


def test_single_still_never_text_rejected(tmp_path):
    reason = check(tmp_path, "still", [subtitle(scene(1), LINES[0])], media="image")
    assert reason is None


# ---------------------------------------------------------------- logo

def test_corner_badge_over_moving_content_rejected(tmp_path):
    reason = check(tmp_path, "logo", [badge(scene(i)) for i in (1, 2, 3)])
    assert reason is not None and reason.startswith("channel_logo (prefilter: static top right")


def test_fully_static_clip_is_not_a_logo(tmp_path):
    frames = [badge(scene(7)) for _ in range(3)]
    assert check(tmp_path, "static", frames) is None


def test_flat_dark_corner_is_not_a_logo(tmp_path):
    frames = []
    for i in (1, 2, 3):
        im = scene(i)
        ImageDraw.Draw(im).rectangle([W - 110, 0, W, 70], fill=(8, 8, 8))
        frames.append(im)
    assert check(tmp_path, "dark", frames) is None


def test_letterbox_bars_are_not_a_logo(tmp_path):
    frames = []
    for i in (1, 2, 3):
        im = scene(i)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, W, 44], fill=(0, 0, 0))
        d.rectangle([0, H - 44, W, H], fill=(0, 0, 0))
        frames.append(im)
    assert check(tmp_path, "bars", frames) is None


def test_logo_needs_two_keyframes_and_video(tmp_path):
    assert check(tmp_path, "one", [badge(scene(1))]) is None
    assert check(tmp_path, "img", [badge(scene(i)) for i in (1, 2, 3)], media="image") is None


# ---------------------------------------------------------------- face

def test_synthetic_face_is_detected_by_real_cascade(tmp_path):
    frame = pf.load_frames(save(tmp_path, "f", [face(plain(0), W // 2, H // 2, 200)]))[0]
    boxes = pf.large_faces(frame, PrefilterConfig())
    assert len(boxes) == 1 and boxes[0][2] >= 150


def test_large_centred_face_rejected_as_talking_head(tmp_path):
    frames = [face(plain(i), W // 2 + 6 * i, H // 2, 200) for i in range(3)]
    reason = check(tmp_path, "head", frames)
    assert reason is not None and reason.startswith("talking_head (prefilter:")
    assert "3 of 3" in reason


def test_small_faces_accepted(tmp_path):
    frames = []
    for i in range(3):
        im = plain(i)
        for cx in (90, 240, 400, 550):
            im = face(im, cx, H // 2, 56, blur=1.5)
        frames.append(im)
    assert check(tmp_path, "crowd", frames) is None


def test_face_on_one_of_three_keyframes_accepted(tmp_path):
    frames = [face(plain(0), W // 2, H // 2, 200), plain(1), plain(2)]
    assert check(tmp_path, "cut", frames) is None


def test_off_centre_large_face_accepted():
    cfg = PrefilterConfig()
    assert pf._centred((250, 90, 150, 150), W, H, cfg)
    assert not pf._centred((480, 90, 150, 150), W, H, cfg)  # centre in the right quarter
    assert not pf._centred((300, 150, 40, 40), W, H, cfg)  # too small


def test_still_portrait_is_not_a_talking_head(tmp_path):
    assert check(tmp_path, "portrait", [face(plain(0), W // 2, H // 2, 200)], media="image") is None


# ---------------------------------------------------------------- clean, switches, trust, dependencies

def test_clean_frames_accepted(tmp_path):
    assert check(tmp_path, "clean", [scene(i) for i in (1, 2, 3)]) is None


@pytest.mark.parametrize("field,frames", [
    ("text", lambda: [subtitle(scene(i), LINES[i - 1]) for i in (1, 2, 3)]),
    ("logo", lambda: [badge(scene(i)) for i in (1, 2, 3)]),
    ("face", lambda: [face(plain(i), W // 2, H // 2, 200) for i in range(3)]),
    ("enabled", lambda: [badge(subtitle(scene(i), LINES[i - 1])) for i in (1, 2, 3)]),
])
def test_switches_turn_each_check_off(tmp_path, field, frames):
    made = frames()
    assert check(tmp_path, f"on_{field}", made) is not None
    assert check(tmp_path, f"off_{field}", made, PrefilterConfig(**{field: False})) is None


def test_operator_and_generated_media_are_trusted(tmp_path):
    keyframes = save(tmp_path, "trusted", [badge(scene(i)) for i in (1, 2, 3)])
    for src in (SourceKind.local, SourceKind.upload, SourceKind.ai_image):
        assert pf.prefilter_candidate(video(keyframes, source=src), PrefilterConfig()) is None


class Log:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str, dict[str, Any]]] = []

    def info(self, msg: str, **f: Any) -> None:
        self.lines.append(("info", msg, f))

    def warn(self, msg: str, **f: Any) -> None:
        self.lines.append(("warn", msg, f))


def test_missing_dependency_skips_check_and_warns_once(tmp_path, monkeypatch):
    def broken() -> Any:
        raise ImportError("no rapidocr")

    monkeypatch.setattr(pf, "ocr_engine", broken)
    monkeypatch.setattr(pf, "_WARNED", set())
    log = Log()
    frames = [subtitle(scene(i), LINES[i - 1]) for i in (1, 2, 3)]
    cands = [video(save(tmp_path, f"dep{n}", frames), cid=f"c{n}") for n in range(2)]
    assert pf.apply_prefilter(cands, None, log) == []
    assert [ln for ln in log.lines if ln[0] == "warn"] == [
        ("warn", "prefilter check skipped, dependency missing", {"check": "text", "error": "no rapidocr"})]
    assert all(c.score is None and c.reject_reason is None for c in cands)


def test_apply_prefilter_marks_hard_reject_and_logs_counts(tmp_path):
    log = Log()
    bad = video(save(tmp_path, "bad", [badge(scene(i)) for i in (1, 2, 3)]), cid="bad")
    good = video(save(tmp_path, "good", [scene(i) for i in (4, 5, 6)]), cid="good")
    assert pf.apply_prefilter([bad, good], None, log) == [bad]
    assert bad.score == 0.0 and bad.reject_reason.startswith("channel_logo (prefilter:")
    assert bad.judge_reason.startswith("prefilter:")
    assert good.score is None and good.reject_reason is None
    info = [ln for ln in log.lines if ln[0] == "info"]
    assert len(info) == 1 and info[0][2]["channel_logo"] == 1 and info[0][2]["checked"] == 2


# ---------------------------------------------------------------- director loop: rejects never reach the vision seat

@pytest.fixture
def ctx(make_ctx):
    c = make_ctx()
    c.stage = "judge"
    return c


def vision_recorder(score: float = 85.0):
    calls: list[list[str]] = []

    def vision_json(role: str, task: str, system: str, prompt: str, images: list[str], stage: str,
                    intent: str = "") -> dict[str, Any]:
        calls.append(list(images))
        return {"score": score, "reason": "fine", "rejects": [], "_mock": False}

    vision_json.calls = calls  # type: ignore[attr-defined]
    return vision_json


def make_shot(cands: list[Candidate]) -> Shot:
    return Shot(id="shot_000", index=0, beat_id="b1", scene_id="s1", start=0.0, end=4.0, duration=4.0,
                type=ShotType.footage, intent="Ships passing through the canal locks", queries=["canal locks ship"],
                candidates=cands)


def test_director_loop_never_sends_prefiltered_candidates_to_vision(ctx, monkeypatch, tmp_path):
    sub = video(save(tmp_path, "sub", [subtitle(scene(i), LINES[i - 1]) for i in (1, 2, 3)]), cid="shot_000_pexels_1")
    logo = video(save(tmp_path, "logo", [badge(scene(i)) for i in (4, 5, 6)]), cid="shot_000_pexels_2")
    clean = video(save(tmp_path, "clean", [scene(i) for i in (7, 8, 9)]), cid="shot_000_pexels_3")
    rec = vision_recorder()
    monkeypatch.setattr(ctx.seats, "vision_json", rec)
    out = dl.judge_and_select(ShotList(project_id="t1", duration=4.0, shots=[make_shot([sub, logo, clean])]), ctx)
    s = out.shots[0]
    sent = {Path(p).name for call in rec.calls for p in call}
    assert sent == {Path(p).name for p in clean.keyframes}
    assert s.selected == "shot_000_pexels_3"
    by_id = {c.id: c for c in s.candidates}
    assert by_id["shot_000_pexels_1"].reject_reason.startswith("burned_in_captions (prefilter:")
    assert by_id["shot_000_pexels_2"].reject_reason.startswith("channel_logo (prefilter:")
    assert by_id["shot_000_pexels_1"].score == 0.0


def test_requery_prefilters_new_candidates(ctx, monkeypatch, tmp_path):
    ctx.settings.judge.max_attempts = 2
    first = video(save(tmp_path, "first", [scene(i) for i in (1, 2, 3)]), cid="shot_000_pexels_1")
    logo = video(save(tmp_path, "relogo", [badge(scene(i)) for i in (4, 5, 6)]), cid="shot_000_pexels_9",
                 url="http://example.invalid/logo.mp4")
    mod = types.ModuleType("vidbush.source.registry")
    mod.source_shot = lambda s, c, queries=None, exclude_ids=(): [logo]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vidbush.source.registry", mod)
    rec = vision_recorder(score=40.0)
    monkeypatch.setattr(ctx.seats, "vision_json", rec)
    s = dl.judge_and_select(ShotList(project_id="t1", duration=4.0, shots=[make_shot([first])]), ctx).shots[0]
    sent = {Path(p).name for call in rec.calls for p in call}
    assert not sent & {Path(p).name for p in logo.keyframes}
    added = next(c for c in s.candidates if c.id == "shot_000_pexels_9")
    assert added.reject_reason.startswith("channel_logo (prefilter:") and added.score == 0.0
    assert s.selected == "shot_000_pexels_1"  # the rejected re-query result is never picked
