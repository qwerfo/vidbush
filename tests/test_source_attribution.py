"""Credits lines and block, plus keyframe extraction."""
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image

from vidbush.models import Candidate, Shot, ShotList, ShotType, SourceKind
from vidbush.source.attribution import attribution_block, attribution_lines
from vidbush.source.keyframes import extract_keyframes


def _shot(i: int, cand: Candidate) -> Shot:
    return Shot(id=f"s{i}", beat_id="b", start=i, end=i + 1, duration=1, type=ShotType.footage, intent="",
                candidates=[cand], selected=cand.id)


def test_attribution_lines_and_block() -> None:
    wiki = Candidate(id="a", source=SourceKind.wikimedia, title="Culebra Cut", author="Underwood",
                     licence="Public domain", url="https://commons.wikimedia.org/wiki/File:C.jpg")
    local = Candidate(id="b", source=SourceKind.local, title="Ship clip", author="Vidbush fixtures",
                      licence="CC0 (Vidbush generated fixture)", url="/lib/ship.mp4", local_path="/lib/ship.mp4")
    ai = Candidate(id="c", source=SourceKind.ai_image, title="gen")
    mg = Candidate(id="d", source=SourceKind.motion, title="map")
    shots = [_shot(0, wiki), _shot(1, local), _shot(2, ai), _shot(3, mg), _shot(4, wiki.model_copy(update={"id": "e"}))]
    sl = ShotList(project_id="p", duration=5, shots=shots)
    lines = attribution_lines(sl)
    assert lines == ["Culebra Cut by Underwood, Public domain, https://commons.wikimedia.org/wiki/File:C.jpg",
                     "Ship clip by Vidbush fixtures, CC0 (Vidbush generated fixture)"]
    block = attribution_block(sl)
    assert block.startswith("## Footage and image credits")
    assert "- Culebra Cut by Underwood" in block
    assert "AI generated images: 1 shots" in block and "Motion graphics: generated" in block


def test_attribution_block_empty() -> None:
    block = attribution_block(ShotList(project_id="p", duration=0, shots=[]))
    assert "No third party footage" in block and "AI generated images: 0 shots" in block


def test_extract_keyframes_video_and_image(tmp_path: Path) -> None:
    vid = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=3",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(vid)], check=True)
    c = Candidate(id="s1_local_0", source=SourceKind.local, local_path=str(vid), media="video",
                  in_point=0.5, out_point=2.5)
    frames = extract_keyframes(c, tmp_path / "kf")
    assert len(frames) == 3 and c.keyframes == frames and (c.width, c.height) == (1280, 720)
    with Image.open(frames[0]) as im:
        assert im.width == 640
    open_ended = Candidate(id="s1_local_1", source=SourceKind.local, local_path=str(vid), media="video")
    extract_keyframes(open_ended, tmp_path / "kf")
    assert 2.9 <= open_ended.out_point <= 3.1

    img = tmp_path / "i.png"
    Image.new("RGB", (2400, 1350), (10, 20, 30)).save(img)
    s = Candidate(id="s2_wikimedia_0", source=SourceKind.wikimedia, local_path=str(img), media="image")
    frames = extract_keyframes(s, tmp_path / "kf")
    assert len(frames) == 1 and (s.width, s.height) == (2400, 1350)
    with Image.open(frames[0]) as im:
        assert im.size == (640, 360)
