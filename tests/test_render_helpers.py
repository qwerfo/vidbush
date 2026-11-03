"""Synthetic media and shot lists for render and package tests, plus ffmpeg_cmds unit tests."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vidbush.models import Candidate, Shot, ShotList
from vidbush.render import ffmpeg_cmds as fc

FPS = 30


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", *args], check=True)


def make_clip(path: Path, seconds: float, size: str = "640x360", rate: int = 25) -> Path:
    """testsrc2 clip at a different rate and size than the output."""
    ff("-f", "lavfi", "-i", f"testsrc2=s={size}:r={rate}:d={seconds}", "-pix_fmt", "yuv420p", str(path))
    return path


def make_still(path: Path, size: tuple[int, int] = (1600, 1000), hue: int = 0) -> Path:
    """Structured still: diagonal stripes and a disc (good for motion checks)."""
    im = Image.new("RGB", size, (40 + hue, 60, 90))
    d = ImageDraw.Draw(im)
    for x in range(-400, size[0], 70):
        d.line([(x, 0), (x + 400, size[1])], fill=(230, 200, 120), width=6)
    cx, cy = size[0] // 2, size[1] // 2
    d.ellipse([cx - 200, cy - 200, cx + 200, cy + 200], fill=(200, 40, 40))
    im.save(path)
    return path


def make_overlay(path: Path, seconds: float) -> Path:
    """qtrle argb overlay: transparent frame with an opaque red box at x 800..1120, y 400..600 (1920 space)."""
    png = path.with_suffix(".png")
    im = Image.new("RGBA", (960, 540), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle([400, 200, 559, 299], fill=(255, 0, 0, 255))
    im.save(png)
    ff("-loop", "1", "-framerate", "30", "-i", str(png), "-t", str(seconds), "-c:v", "qtrle", "-pix_fmt", "argb",
       str(path))
    return path


def make_wav(path: Path, seconds: float, freq: int = 440, rate: int = 48000) -> Path:
    ff("-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate={rate}:duration={seconds}", "-ac", "2",
       "-af", "volume=-12dB", str(path))
    return path


def make_bumper(path: Path, seconds: float) -> Path:
    """Intro or outro clip with its own audio, 4:3 so it gets pillarboxed."""
    ff("-f", "lavfi", "-i", f"testsrc=s=640x480:r=24:d={seconds}", "-f", "lavfi", "-i",
       f"sine=frequency=880:sample_rate=44100:duration={seconds}", "-pix_fmt", "yuv420p", "-shortest", str(path))
    return path


def build_shotlist(media: Path, project_id: str = "t1") -> ShotList:
    """3 shots: short footage clip (slowed and held), still with overlay and fade in, motion clip."""
    media.mkdir(parents=True, exist_ok=True)
    clip = make_clip(media / "clip.mp4", 1.0)
    still = make_still(media / "still.jpg")
    mg = make_clip(media / "mg.mp4", 1.2, size="1920x1080", rate=30)
    ov = make_overlay(media / "ov.mov", 1.3)
    shots = [
        Shot(id="s1", index=0, beat_id="b1", scene_id="sc1", start=0.0, end=1.2, duration=1.2, type="footage",
             intent="canal footage", selected="s1_local_1",
             candidates=[Candidate(id="s1_local_1", source="local", local_path=str(clip), media="video",
                                   in_point=0.1, title="Test clip", author="Tester", licence="CC BY 4.0",
                                   url="https://example.org/clip", score=81.0,
                                   keyframes=[str(still)])]),
        Shot(id="s2", index=1, beat_id="b2", scene_id="sc2", start=1.2, end=2.5, duration=1.3,
             type="image_overlay", intent="map still", transition_in="fade", overlay_path=str(ov),
             selected="s2_local_1",
             candidates=[Candidate(id="s2_local_1", source="wikimedia", local_path=str(still), media="image",
                                   title="Canal photo", author="Photographer", licence="CC BY-SA 4.0",
                                   url="https://commons.example.org/photo", score=88.0, keyframes=[str(still)])]),
        Shot(id="s3", index=2, beat_id="b3", scene_id="sc3", start=2.5, end=3.633, duration=1.133,
             type="motion_graphic", intent="number card", selected="s3_motion_1",
             candidates=[Candidate(id="s3_motion_1", source="motion", local_path=str(mg), media="video")]),
    ]
    return ShotList(project_id=project_id, duration=3.633, shots=shots)


def caption_lines() -> list[dict]:
    return [{"text": "The canal opened in", "start": 0.1, "end": 1.0},
            {"text": "nineteen fourteen after decades", "start": 1.0, "end": 2.2},
            {"text": "of work and loss", "start": 2.3, "end": 3.4}]


def write_captions(work: Path) -> Path:
    p = work / "captions.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"lines": caption_lines()}), encoding="utf-8")
    return p


def count_frames(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                         check=True)
    return int(out.stdout.strip())


def frame_at(path: Path, n: int, size: tuple[int, int] = (1920, 1080)) -> np.ndarray:
    """Decode frame n as an RGB array."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{n})", "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(size[1], size[0], 3)


# ---------------------------------------------------------------- ffmpeg_cmds unit tests

def test_bar_height_matches_letterbox_rule() -> None:
    assert fc.bar_height(1920, 1080, 2.39) == 138
    assert fc.bar_height(1920, 1080, 16 / 9) == 0


def test_escape_filter_value_quotes_colons_and_quotes() -> None:
    assert fc.escape_filter_value("C:\\dir/it's.ass") == "'C\\:/dir/it'\\''s.ass'"


def test_fades_clamped_to_half_segment() -> None:
    assert fc.fades(0.4, 0.4, 0.0) == "fade=t=in:st=0:d=0.2000"
    assert "fade=t=out:st=0.8000:d=0.2000" in fc.fades(1.0, 0.0, 0.2)
    assert fc.fades(1.0, 0.0, 0.0) == ""


def test_crop_fraction_ignores_full_and_bad_crops() -> None:
    assert fc.crop_fraction(None) == ""
    assert fc.crop_fraction([0, 0, 1, 1]) == ""
    assert fc.crop_fraction([0.1, 0.2, 0.5, 0.5]).startswith("crop=w=iw*0.50000")


def test_encoder_args_identical_and_constant_rate(settings) -> None:
    a, b = fc.encoder_args(settings), fc.encoder_args(settings)
    assert a == b
    assert "-r" in a and a[a.index("-r") + 1] == "30" and "high" in a and "-an" in a
