"""Assembly: final.mp4 streams and duration, segment cache reuse, one-shot re-render, intro and outro."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

import numpy as np

from test_render_helpers import build_shotlist, count_frames, make_bumper, write_captions
from vidbush.audio.loudness import normalize, read_audio, write_audio
from vidbush.models import Narration, Word
from vidbush.render.assemble import assemble, verify
from vidbush.util import NullLogger, ToolError, ffprobe, read_json


def speech_wav(path: Path, seconds: float, seed: int = 1, bed: float = 0.0) -> Path:
    """Stereo 48 kHz tone bursts of uneven length: an envelope the sound gate can follow (a steady sine has none);
    bed adds a steady tone under it, as music sits under the narration in a mix."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * 48000))
    gate, i, on = np.zeros(n, np.float32), 0, True
    while i < n:
        step = int(rng.uniform(0.08, 0.35) * 48000)
        gate[i:i + step] = 1.0 if on else 0.0
        i, on = i + step, not on
    t = np.arange(n) / 48000
    sig = 0.25 * np.sin(2 * np.pi * 330 * t) * gate + bed * np.sin(2 * np.pi * 523 * t)
    return write_audio(path, np.stack([sig, sig], axis=1), 48000)


def save_narration(ctx, seconds: float, spoken: float) -> Path:
    """work/narration.json and its wav, speaking from 0 to `spoken` seconds."""
    wav = speech_wav(ctx.work / "narration" / "narration.wav", seconds)
    ctx.save("narration", Narration(audio_path=str(wav), duration=seconds,
                                    words=[Word(text="the", start=0.05, end=spoken)]))
    return wav


@pytest.fixture
def ready_ctx(make_ctx):
    ctx = make_ctx(mode="script")
    shotlist = build_shotlist(ctx.work / "media")
    write_captions(ctx.work)
    save_narration(ctx, shotlist.duration, shotlist.duration - 0.1)
    speech_wav(ctx.work / "mix.wav", shotlist.duration, bed=0.1)  # the narration with a music bed under it
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


def _master_mix(ctx, seconds: float) -> None:
    """mix.wav as the mix stage leaves it: the narration, already at -14 LUFS."""
    normalize(Path(read_json(ctx.work / "narration.json")["audio_path"]), ctx.work / "mix.wav", -14.0, -1.0, 48000,
              2, exact_samples=round(seconds * 48000))


def _srt_starts(path: Path) -> list[float]:
    out = []
    for m in re.finditer(r"(\d\d):(\d\d):(\d\d),(\d{3}) -->", path.read_text(encoding="utf-8")):
        h, mi, se, ms = (int(g) for g in m.groups())
        out.append(h * 3600 + mi * 60 + se + ms / 1000)
    return out


def test_assemble_with_intro_and_outro(ready_ctx) -> None:
    ctx, shotlist = ready_ctx
    _master_mix(ctx, shotlist.duration)
    ctx.brand.intro_path = str(make_bumper(ctx.work / "intro.mp4", 1.0))
    ctx.brand.outro_path = str(make_bumper(ctx.work / "outro.mp4", 0.5))
    final, _ = assemble(shotlist, ctx)
    report = read_json(ctx.work / "assembly.json")
    assert report["intro_duration"] == pytest.approx(1.0, abs=0.04)
    assert report["outro_duration"] == pytest.approx(0.5, abs=0.04)
    assert count_frames(final) == 109 + 30 + 15
    assert report["durations_match"] and report["frames"] == report["expected_frames"] == 154
    assert _loudness(final) == pytest.approx(-14.0, abs=1.5)
    starts = _srt_starts(ctx.root / "captions.srt")
    assert starts and starts[0] >= report["intro_duration"] - 1e-3, "sidecar captions on the final timeline"
    assert starts[0] == pytest.approx(0.1 + report["intro_duration"], abs=0.002)
    assert "0:00:01.10" in (ctx.root / "captions.ass").read_text(encoding="utf-8")
    assert read_json(ctx.work / "captions.json")["lines"][0]["start"] == 0.1, "work captions stay in body time"


def test_body_mix_is_not_renormalized_with_a_hot_intro(ready_ctx) -> None:
    """A mastered intro peaking near 0 dBTP must not push the body into loudnorm gain riding."""
    ctx, shotlist = ready_ctx
    _master_mix(ctx, shotlist.duration)
    hot = ctx.work / "hot_intro.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=640x360:r=30:d=1",
                    "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=1", "-af", "volume=-0.1dB",
                    "-pix_fmt", "yuv420p", "-shortest", str(hot)], check=True)
    ctx.brand.intro_path = str(hot)
    assemble(shotlist, ctx)
    body = read_audio(ctx.work / "final_audio.wav")[48000:48000 + round(shotlist.duration * 48000), 0]
    mix = read_audio(ctx.work / "mix.wav")[:len(body), 0]
    assert np.allclose(body, mix, atol=2e-3), "body samples pass through untouched"
    intro = read_audio(ctx.work / "final_audio.wav")[:48000, 0]
    assert np.abs(intro).max() < 10 ** (-0.9 / 20), "intro limited under the true peak ceiling"


def test_intro_with_longer_audio_holds_last_frame(ready_ctx) -> None:
    """Intro with 1 s of picture and 2 s of sound: picture is held so the body stays in sync."""
    ctx, shotlist = ready_ctx
    intro = ctx.work / "long_audio_intro.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=640x360:r=30:d=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
                    "-pix_fmt", "yuv420p", str(intro)], check=True)
    ctx.brand.intro_path = str(intro)
    final, _ = assemble(shotlist, ctx)
    report = read_json(ctx.work / "assembly.json")
    assert report["intro_duration"] == pytest.approx(2.0, abs=0.04)
    assert count_frames(final) == 109 + 60 == report["frames"]
    assert report["durations_match"]


def test_verify_raises_on_av_drift(tmp_path) -> None:
    clip = tmp_path / "drift.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=320x180:r=30:d=1",
                    "-f", "lavfi", "-i", "sine=duration=2", "-pix_fmt", "yuv420p", "-c:a", "aac", str(clip)],
                   check=True)
    with pytest.raises(ToolError):
        verify(clip, 30, NullLogger())


def test_sound_gate_keeps_a_final_without_narration_off_disk(ready_ctx) -> None:
    ctx, shotlist = ready_ctx
    final, _ = assemble(shotlist, ctx)
    good = final.read_bytes()
    assert read_json(ctx.work / "sound_check.json")["ok"]
    write_audio(ctx.work / "mix.wav", np.zeros((round(shotlist.duration * 48000), 2), np.float32), 48000)
    with pytest.raises(ToolError, match="sound check failed"):
        assemble(shotlist, ctx)
    assert final.read_bytes() == good, "the silent file never replaced final.mp4"
    report = read_json(ctx.work / "sound_check.json")
    assert not report["ok"] and report["faults"]


def test_sound_gate_fails_when_the_mix_cut_the_narration(ready_ctx) -> None:
    ctx, shotlist = ready_ctx
    save_narration(ctx, shotlist.duration + 2.0, shotlist.duration + 1.5)
    with pytest.raises(ToolError, match="the mix cut it"):
        assemble(shotlist, ctx)
    (ctx.work / "trim.json").write_text(json.dumps({"until": 2.0}))  # an excerpt cuts it on purpose
    assemble(shotlist, ctx)
    assert read_json(ctx.work / "sound_check.json")["runtime_s"] == pytest.approx(shotlist.duration)


def test_missing_intro_is_skipped(ready_ctx) -> None:
    ctx, shotlist = ready_ctx
    ctx.brand.intro_path = "does/not/exist.mp4"
    final, _ = assemble(shotlist, ctx)
    assert read_json(ctx.work / "assembly.json")["intro_duration"] == 0.0
    assert count_frames(final) == 109
