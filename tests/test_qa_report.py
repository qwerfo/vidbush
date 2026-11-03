"""QA report on a synthetic project with a tiny generated final.mp4."""
from __future__ import annotations

import json
import subprocess

import pytest

from vidbush.config import REPO_ROOT
from vidbush.judge.qa import build_qa, caption_check, mmss
from vidbush.models import Candidate, Narration, QAReport, Shot, ShotList, ShotType, SourceKind, Word
from vidbush.util import read_model, write_json

IMG = str(REPO_ROOT / "fixtures" / "sample_assets" / "sample_image_07.jpg")


def make_video(path, seconds: float = 3.0) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc2=size=320x180:rate=30:duration={seconds}", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                    "-preset", "ultrafast", str(path)], check=True)


def make_shotlist() -> ShotList:
    def c(sid: str, src: SourceKind, **kw) -> Candidate:
        return Candidate(id=f"{sid}_{src.value}_1", source=src, local_path=IMG, keyframes=[IMG], **kw)

    s1 = Shot(id="shot_001", index=0, beat_id="b1", start=0, end=1, duration=1, type=ShotType.footage,
              intent="Ships in the locks", judge_score=88, attempts=1, flags=["mock"],
              candidates=[c("shot_001", SourceKind.wikimedia, attribution="Photo by A, CC BY-NC-ND",
                            flags=["nc_nd"], licence="CC BY-NC-ND 4.0", title="Locks", url="https://x/locks")])
    s2 = Shot(id="shot_002", index=1, beat_id="b1", start=1, end=2, duration=1, type=ShotType.ai_image,
              intent="Workers digging", judge_score=55, attempts=3, flags=["mock", "no_match_accepted"],
              candidates=[c("shot_002", SourceKind.ai_image, attribution="AI generated image")])
    s3 = Shot(id="shot_003", index=2, beat_id="b2", start=2, end=3, duration=1, type=ShotType.motion_graphic,
              intent="Map of the route", candidates=[c("shot_003", SourceKind.motion, attribution="")])
    for s in (s1, s2, s3):
        s.selected = s.candidates[0].id
    return ShotList(project_id="t1", duration=3.0, shots=[s1, s2, s3])


def write_audio_side(ctx) -> None:
    words = [Word(text=t, start=0.2 + 0.4 * i, end=0.5 + 0.4 * i) for i, t in
             enumerate("The canal opened in nineteen fourteen.".split())]
    ctx.save("narration", Narration(audio_path="x.wav", duration=2.6, words=words))
    lines = [{"text": "The canal opened", "start": 0.2, "end": 1.3},
             {"text": "in nineteen", "start": 1.4, "end": 2.1}]
    write_json(ctx.work / "captions.json", {"lines": lines})


@pytest.fixture
def qa_ctx(make_ctx):
    ctx = make_ctx()
    ctx.stage = "packaging"
    ctx.settings.judge.post_render_every_s = 1.0
    make_video(ctx.root / "final.mp4")
    write_audio_side(ctx)
    return ctx


def test_build_qa_report(qa_ctx):
    paths = build_qa(make_shotlist(), qa_ctx)
    assert [p.name for p in paths] == ["qa_report.json", "qa_report.md"]
    rep = read_model(paths[0], QAReport)
    assert len(rep.shots) == 3 and rep.threshold == 70
    assert all(s.post_render_score is not None for s in rep.shots)
    assert len(list((qa_ctx.work / "qa_frames").glob("f_*.jpg"))) == 3
    assert rep.pct_above_threshold == 50.0 and rep.pct_no_match_accepted == 50.0
    assert rep.attribution_count == 1  # ai and motion excluded
    assert len(rep.nc_nd_assets) == 1 and "shot_001" in rep.nc_nd_assets[0]
    assert "vision" in rep.mock_used
    assert rep.mix_achieved["footage"] == pytest.approx(33.3, abs=0.1)
    assert rep.pacing["ai_image"]["count"] == 1 and rep.pacing["footage"]["mean"] == 1.0
    flags = {s.shot_id: s.flags for s in rep.shots}
    assert "nc_nd" in flags["shot_001"] and "no_match_accepted" in flags["shot_002"]
    assert rep.caption_word_match == pytest.approx(5 / 6, abs=1e-3)
    assert 0.6 < rep.caption_coverage < 0.95
    assert any("above the 1.0% target" in n for n in rep.notes)
    md = paths[1].read_text()
    assert "| 00:01 | shot_002 | ai_image |" in md and "Flagged shots" in md
    assert chr(0x2014) not in md + paths[0].read_text()


def test_intro_offset_shifts_sampling_and_timestamps(qa_ctx):
    make_video(qa_ctx.root / "final.mp4", seconds=5.0)
    qa_ctx.brand.intro_path = "intro.mp4"
    write_json(qa_ctx.work / "assembly.json", {"intro_duration": 2.0})
    paths = build_qa(make_shotlist(), qa_ctx)
    assert "| 00:03 | shot_002 |" in paths[1].read_text()
    rep = read_model(paths[0], QAReport)
    assert all(s.post_render_score is not None for s in rep.shots)


def test_missing_final_is_noted(make_ctx):
    ctx = make_ctx()
    rep = read_model(build_qa(make_shotlist(), ctx)[0], QAReport)
    assert any("final.mp4 not found" in n for n in rep.notes)
    assert all(s.post_render_score is None for s in rep.shots)
    assert rep.caption_coverage == 0.0


def test_caption_check_full_match(make_ctx):
    ctx = make_ctx()
    words = [Word(text=t, start=i * 0.5, end=i * 0.5 + 0.4) for i, t in enumerate(["Hello,", "World!", "again"])]
    ctx.save("narration", Narration(audio_path="x.wav", duration=2, words=words))
    (ctx.work / "captions.json").write_text(json.dumps({"lines": [{"text": "HELLO WORLD AGAIN", "start": 0,
                                                                     "end": 1.5}]}))
    assert caption_check(ctx) == (1.0, 1.0)


def test_mmss():
    assert mmss(0) == "00:00" and mmss(61.4) == "01:01" and mmss(3599.6) == "60:00"


def test_mix_headline_is_shot_count_with_screen_time_secondary():
    from vidbush.judge.qa import mix_and_pacing
    from vidbush.models import Shot, ShotList, ShotType
    shots = [Shot(id=f"s{i}", beat_id="b", start=0, end=d, duration=d, type=t, intent="x")
             for i, (t, d) in enumerate([(ShotType.footage, 3.0), (ShotType.motion_graphic, 9.0)])]
    mix, pacing = mix_and_pacing(ShotList(project_id="p", duration=12.0, shots=shots))
    assert mix["motion_graphic"] == 50.0 and mix["footage"] == 50.0
    assert pacing["motion_graphic"]["time_pct"] == 75.0 and pacing["motion_graphic"]["count_pct"] == 50.0
