"""Demo-like project for the review tests (no tests here).

build_review_project(settings) creates project "demo" from fixtures/demo_script.md and writes
work/shotlist.json with 8 shots covering every shot type, real looking source urls and licences,
judge scores around the threshold of 70 and rejected candidates whose reasons look like the real
pre-filter and judge ones. Keyframes are small PNGs so a mock judge could read them.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from vidbush.config import REPO_ROOT
from vidbush.jobs.pipeline import create_project
from vidbush.models import (Candidate, InputMode, MotionKind, MotionSpec, Shot, ShotList, ShotType, SourceKind)
from vidbush.util import write_json

PID = "demo"
DURATIONS = [4.0, 5.0, 3.5, 6.0, 4.5, 5.0, 3.0, 4.0]
TOTAL = sum(DURATIONS)


def png(path: Path, rgb: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 36), rgb).save(path)
    return str(path)


def _cand(root: Path, cid: str, source: SourceKind, url: str, licence: str, score, media: str = "image",
          reject: str | None = None, reason: str = "", title: str = "", flags=None, rgb=(120, 100, 80)) -> Candidate:
    kf = png(root / "sources" / "kf" / f"{cid}.png", rgb)
    return Candidate(id=cid, source=source, url=url, local_path=kf, media=media, licence=licence, title=title,
                     keyframes=[kf], score=score, judge_reason=reason, reject_reason=reject, flags=flags or [],
                     out_point=4.0 if media == "video" else 0.0, width=1920, height=1080)


def build_shots(root: Path) -> list[Shot]:
    lt = MotionSpec(kind=MotionKind.lower_third, duration=5.0, background="transparent",
                    payload={"title": "Black Canyon"})
    shots: list[Shot] = []
    t = 0.0

    def add(sid: str, typ: ShotType, intent: str, cands: list[Candidate], selected, score, **kw) -> None:
        nonlocal t
        i = len(shots)
        d = DURATIONS[i]
        shots.append(Shot(id=sid, index=i, beat_id=f"b{i // 2:03d}", start=t, end=t + d, duration=d, type=typ,
                          intent=intent, candidates=cands, selected=selected, judge_score=score, **kw))
        t = round(t + d, 3)

    add("shot_001", ShotType.footage, "Black Canyon in summer heat, 1931", [
        _cand(root, "shot_001_youtube_0", SourceKind.youtube, "https://www.youtube.com/watch?v=abc123XYZ",
              "CC BY 3.0", 84, "video", reason="canyon walls, period look", title="Black Canyon 1931"),
        _cand(root, "shot_001_youtube_1", SourceKind.youtube, "https://www.youtube.com/watch?v=def456UVW",
              "CC BY 3.0", 0, "video", reject="burned_in_captions (prefilter: 3 of 4 keyframes)"),
        _cand(root, "shot_001_archive_2", SourceKind.archive, "https://archive.org/details/hoover_dam_reel",
              "Public domain", 61, "video", reason="river, but modern boats"),
    ], "shot_001_youtube_0", 84)
    add("shot_002", ShotType.footage_overlay, "The Colorado River flooding farms", [
        _cand(root, "shot_002_archive_0", SourceKind.archive, "https://archive.org/details/colorado_flood_1920",
              "Public domain", 72, "video", reason="flood water over fields"),
        _cand(root, "shot_002_youtube_1", SourceKind.youtube, "https://www.youtube.com/watch?v=ghi789RST",
              "CC BY 3.0", 0, "video", reject="channel_logo (prefilter: corner mark on 4 of 4 keyframes)"),
    ], "shot_002_archive_0", 72, overlay=lt)
    add("shot_003", ShotType.image, "Jobless men arriving at Ragtown", [
        _cand(root, "shot_003_wikimedia_0", SourceKind.wikimedia,
              "https://commons.wikimedia.org/wiki/File:Ragtown_1931.jpg", "Public domain", 66,
              reason="tents in desert, faces small", flags=["below_threshold"]),
        _cand(root, "shot_003_wikimedia_1", SourceKind.wikimedia,
              "https://commons.wikimedia.org/wiki/File:Boulder_City_1932.jpg", "CC BY-SA 4.0", 0,
              reject="talking_head (prefilter: face covers 41% of frame)"),
    ], "shot_003_wikimedia_0", 66, flags=["below_threshold"])
    add("shot_004", ShotType.image_overlay, "Boulder City under construction", [
        _cand(root, "shot_004_loc_0", SourceKind.loc, "https://www.loc.gov/item/2017878798/",
              "No known restrictions", 77, reason="street of new houses"),
        _cand(root, "shot_004_wikimedia_1", SourceKind.wikimedia,
              "https://commons.wikimedia.org/wiki/File:Las_Vegas_1930.jpg", "CC BY 4.0", 12,
              reject="watermark; anachronism"),
    ], "shot_004_loc_0", 77, overlay=lt.model_copy(update={"duration": 6.0}))
    add("shot_005", ShotType.ai_image, "Frank Crowe studying plans by lamplight", [
        _cand(root, "shot_005_ai_image_1", SourceKind.ai_image, "", "AI generated", 70,
              reason="period office, plausible", flags=["mock"]),
    ], "shot_005_ai_image_1", 70, ai_prompt="Frank Crowe at a desk, 1931, lamplight")
    add("shot_006", ShotType.motion_graphic, "Map of the four diversion tunnels", [], None, None,
        motion=MotionSpec(kind=MotionKind.title_card, duration=5.0, payload={"title": "Four tunnels"}))
    add("shot_007", ShotType.footage, "Blasting the diversion tunnels", [
        _cand(root, "shot_007_pexels_0", SourceKind.pexels, "https://www.pexels.com/video/explosion-123/",
              "Pexels licence", 58, "video", reason="explosion, no tunnel, modern quarry",
              reject="best available accepted", flags=["below_threshold"]),
        _cand(root, "shot_007_archive_1", SourceKind.archive, "https://archive.org/details/tunnel_blast",
              "Public domain", 40, "video", reject="segment_repeat: repeats a used segment"),
    ], "shot_007_pexels_0", 58, flags=["below_threshold", "no_match_accepted"])
    add("shot_008", ShotType.image, "Workers pouring the first concrete", [
        _cand(root, "shot_008_nasa_0", SourceKind.nasa, "https://images.nasa.gov/details/hoover-1935",
              "Public domain", 91, reason="concrete buckets, workers"),
    ], "shot_008_nasa_0", 91)
    return shots


def build_review_project(settings, pid: str = PID) -> Path:
    """Project pid with the 8 shot demo-like shot list; returns the project root."""
    proj = create_project(settings, pid, InputMode.script, script_path=str(REPO_ROOT / "fixtures" / "demo_script.md"),
                          seed=1)
    root = Path(proj.paths.root)
    write_json(root / "work" / "shotlist.json", ShotList(project_id=pid, duration=TOTAL, shots=build_shots(root)))
    return root


def write_shots(root: Path, shots: list[Shot], pid: str = PID) -> None:
    for i, s in enumerate(shots):
        s.index = i
    write_json(root / "work" / "shotlist.json",
               ShotList(project_id=pid, duration=sum(s.duration for s in shots), shots=shots))
