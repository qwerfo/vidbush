"""Synthetic edit project shared by the editor tests (no tests here).

build_edit_project(settings) writes script.json, narration.json and
shotlist.json for an 8 shot, 30 s project (29 s narration + 1 s tail) with
real image and video files so keyframes and crops work.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from vidbush.jobs.pipeline import create_project
from vidbush.models import (Beat, Candidate, InputMode, MotionKind, MotionSpec, Narration, Scene, ScenePurpose, Script,
                            Shot, ShotList, ShotType, SourceKind, Word)
from vidbush.util import read_model, write_json

PID = "edit1"
TOTAL = 30.0
DURATIONS = [3.0, 4.0, 2.0, 5.0, 4.0, 3.0, 6.0, 3.0]
TYPES = [ShotType.footage, ShotType.image, ShotType.footage_overlay, ShotType.image_overlay, ShotType.motion_graphic,
         ShotType.footage, ShotType.image, ShotType.image]
BEATS = {"b001": "The canal opened in 1914. It changed world trade. Ships saved weeks at sea.",
         "b002": "It is worth noting that the work was very hard, and thousands of workers died in Panama.",
         "b003": "The locks lift every ship twenty six metres above the sea.",
         "b004": "Today the canal moves more than 14,000 ships a year."}
BEAT_OF_SHOT = ["b001", "b001", "b002", "b002", "b003", "b003", "b004", "b004"]
SPANS = {"b001": [0.0, 7.0], "b002": [7.0, 14.0], "b003": [14.0, 21.0], "b004": [21.0, 29.0]}


def _image(path: Path, hue: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (640, 360), (hue % 255, (hue * 3) % 255, (hue * 7) % 255))
    d = ImageDraw.Draw(im)
    for k in range(6):
        d.ellipse((40 + k * 90, 80 + (k % 2) * 60, 120 + k * 90, 160 + (k % 2) * 60), fill=(255 - hue % 255, 200, 40))
    im.save(path)
    return str(path)


def make_video(path: Path, seconds: float = 2.0) -> str:
    """Tiny test clip (testsrc2, 320x180)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc2=size=320x180:rate=30:duration={seconds}", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def _candidates(shot_id: str, kind: ShotType, assets: Path, video: str) -> list[Candidate]:
    out = []
    for n in range(1, 4):
        if kind in (ShotType.footage, ShotType.footage_overlay):
            out.append(Candidate(id=f"{shot_id}_local_{n}", source=SourceKind.local, local_path=video, media="video",
                                 in_point=0.0, out_point=2.0, width=320, height=180, score=90 - n * 5,
                                 keyframes=[video], url=f"local://{shot_id}/{n}"))
        else:
            p = _image(assets / f"{shot_id}_{n}.png", int(shot_id[-3:]) * 37 + n * 61)
            out.append(Candidate(id=f"{shot_id}_wikimedia_{n}", source=SourceKind.wikimedia, local_path=p,
                                 media="image", width=640, height=360, score=88 - n * 4, keyframes=[p],
                                 url=f"https://example.org/{shot_id}/{n}"))
    return out


def _shot(i: int, start: float, assets: Path, video: str) -> Shot:
    sid = f"shot_{i + 1:03d}"
    kind, dur = TYPES[i], DURATIONS[i]
    shot = Shot(id=sid, index=i, beat_id=BEAT_OF_SHOT[i], scene_id="s01" if i < 4 else "s02", start=start,
                end=start + dur, duration=dur, type=kind, intent=f"Footage of the Panama Canal, view {i + 1}")
    if kind == ShotType.motion_graphic:
        shot.motion = MotionSpec(kind=MotionKind.title_card, duration=dur, payload={"title": "The Locks"})
        return shot
    shot.candidates = _candidates(sid, kind, assets, video)
    shot.selected, shot.judge_score = shot.candidates[0].id, shot.candidates[0].score
    if kind in (ShotType.footage_overlay, ShotType.image_overlay):
        shot.overlay = MotionSpec(kind=MotionKind.lower_third, duration=dur, background="transparent",
                                  payload={"title": "Panama", "position": "lower_left"})
    return shot


def _narration(work: Path) -> Narration:
    words = []
    for bid, (s, e) in SPANS.items():
        toks = BEATS[bid].split()
        step = (e - s - 0.4) / len(toks)
        words += [Word(text=t, start=round(s + k * step, 3), end=round(s + (k + 0.8) * step, 3), beat_id=bid)
                  for k, t in enumerate(toks)]
    audio = work / "narration" / "narration.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"")
    return Narration(audio_path=str(audio), duration=TOTAL - 1.0, words=words, beat_spans=SPANS)


def build_edit_project(settings, pid: str = PID) -> Path:
    """Create the project and its work files; returns the work folder."""
    src = Path(settings.output_folder).parent / "edit_script.md"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("# Canal\n\nA script.", encoding="utf-8")
    proj = create_project(settings, pid, InputMode.script, script_path=str(src), seed=7)
    work = Path(proj.paths.work)
    assets = Path(proj.paths.sources) / "_assets"
    video = make_video(assets / "clip.mp4")
    script = Script(title="Canal", scenes=[
        Scene(id="s01", title="Opening", purpose=ScenePurpose.hook,
              beats=[Beat(id="b001", text=BEATS["b001"]), Beat(id="b002", text=BEATS["b002"])]),
        Scene(id="s02", title="The locks", purpose=ScenePurpose.act,
              beats=[Beat(id="b003", text=BEATS["b003"]), Beat(id="b004", text=BEATS["b004"])])])
    shots, t = [], 0.0
    for i in range(len(DURATIONS)):
        shots.append(_shot(i, t, assets, video))
        t += DURATIONS[i]
    write_json(work / "script.json", script)
    write_json(work / "narration.json", _narration(work))
    write_json(work / "shotlist.json", ShotList(project_id=pid, duration=TOTAL, shots=shots))
    return work


def load_shotlist(work: Path) -> ShotList:
    return read_model(work / "shotlist.json", ShotList)


def assert_timeline(sl: ShotList, total: float = TOTAL) -> None:
    """Contiguous from 0, no shot under 1.5 s, ends at total, motion specs match shot durations."""
    t = 0.0
    for i, s in enumerate(sl.shots):
        assert s.index == i
        assert abs(s.start - t) < 2e-3, (s.id, s.start, t)
        assert abs(s.end - s.start - s.duration) < 2e-3
        assert s.duration >= 1.5 - 1e-3, (s.id, s.duration)
        for spec in (s.motion, s.overlay):
            if spec is not None:
                assert abs(spec.duration - s.duration) < 2e-3
        t = s.end
    assert abs(t - total) < 2e-3
    assert abs(sl.duration - total) < 2e-3
    assert len({s.id for s in sl.shots}) == len(sl.shots)
