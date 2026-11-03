"""Assembly: final.mp4 streams and duration, segment cache reuse, one-shot re-render, intro and outro."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from test_render_helpers import build_shotlist, count_frames, make_bumper, make_wav, write_captions
from vidbush.render.assemble import assemble
from vidbush.util import ffprobe, read_json


@pytest.fixture
def ready_ctx(make_ctx):
    ctx = make_ctx(mode="script")
    shotlist = build_shotlist(ctx.work / "media")
    write_captions(ctx.work)
    make_wav(ctx.work / "mix.wav", shotlist.duration)
    return ctx, shotlist


def _streams(path: Path) -> dict[str, dict]:
    return {s["codec_type"]: s for s in ffprobe(path)["streams"]}


def test_assemble_final_and_cache(ready_ctx) -> None:
    ctx, shotlist = ready_ctx
    final, shotlist = assemble(shotlist, ctx)
    assert final == ctx.root / "final.mp4" and final.exists()
    st = _streams(final)
    v, a = st["video"], st["audio"]
    assert (v["codec_name"], v["profile"], v["width"], v["height"], v["r_frame_rate"]) == \
        ("h264", "High", 1920, 1080, "30/1")
    assert (a["codec_name"], a["sample_rate"], a["channels"]) == ("aac", "48000", 2)
    assert 180_000 <= int(a.get("bit_rate", 192_000)) <= 200_000
    assert count_frames(final) == round(shotlist.duration * 30) == 109
    report = read_json(ctx.work / "assembly.json")
    assert report["durations_match"] and abs(report["video_duration"] - report["audio_duration"]) <= 1 / 30
    assert report["intro_duration"] == 0.0 and report["outro_duration"] == 0.0
    stats = read_json(ctx.work / "render_stats.json")
    assert (stats["rendered"], stats["cached"], stats["segments"]) == (3, 0, 3) and "seconds" in stats
    for shot in shotlist.shots:
        assert shot.render_hash and Path(shot.render_segment_path).name == f"{shot.render_hash}.mp4"

    _, shotlist = assemble(shotlist, ctx)
    stats = read_json(ctx.work / "render_stats.json")
    assert (stats["rendered"], stats["cached"]) == (0, 3)

    old = {s.id: s.render_hash for s in shotlist.shots}
    shotlist.shots[2].candidates[0].in_point = 0.1
    _, shotlist = assemble(shotlist, ctx)
    stats = read_json(ctx.work / "render_stats.json")
    assert (stats["rendered"], stats["cached"]) == (1, 2)
    changed = [s.id for s in shotlist.shots if s.render_hash != old[s.id]]
    assert changed == ["s3"]


def _loudness(path: Path) -> float:
    cp = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
                         "loudnorm=I=-14:TP=-1:print_format=json", "-f", "null", "-"], capture_output=True, text=True)
    return float(json.loads(re.findall(r"\{[^{}]*\}", cp.stderr)[-1])["input_i"])


def test_assemble_with_intro_and_outro(ready_ctx) -> None:
    ctx, shotlist = ready_ctx
    ctx.brand.intro_path = str(make_bumper(ctx.work / "intro.mp4", 1.0))
    ctx.brand.outro_path = str(make_bumper(ctx.work / "outro.mp4", 0.5))
    final, _ = assemble(shotlist, ctx)
    report = read_json(ctx.work / "assembly.json")
    assert report["intro_duration"] == pytest.approx(1.0, abs=0.04)
    assert report["outro_duration"] == pytest.approx(0.5, abs=0.04)
    assert count_frames(final) == 109 + 30 + 15
    assert report["durations_match"]
    assert _loudness(final) == pytest.approx(-14.0, abs=1.5)


def test_missing_intro_is_skipped(ready_ctx) -> None:
    ctx, shotlist = ready_ctx
    ctx.brand.intro_path = "does/not/exist.mp4"
    final, _ = assemble(shotlist, ctx)
    assert read_json(ctx.work / "assembly.json")["intro_duration"] == 0.0
    assert count_frames(final) == 109
