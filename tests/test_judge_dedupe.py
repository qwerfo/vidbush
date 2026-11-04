"""Segment dedupe rule in the selection scorer: banned within a scene and next shot, penalised across scenes."""
from __future__ import annotations

from typing import Any

import pytest

from vidbush.config import REPO_ROOT
from vidbush.judge import director_loop as dl
from vidbush.models import Candidate, Shot, ShotList, ShotType, SourceKind

ASSETS = REPO_ROOT / "fixtures" / "sample_assets"
URL = "https://www.youtube.com/watch?v=abc123"


def vid(cid: str, score: float, in_point: float, url: str = URL, path: str = "/w/youtube_abc123_40.mp4",
        source: SourceKind = SourceKind.youtube, **kw: Any) -> Candidate:
    return Candidate(id=cid, source=source, url=url, local_path=path, media="video", in_point=in_point,
                     out_point=in_point + 4.0, width=1920, height=1080, score=score, **kw)


def still(cid: str, score: float, n: int = 1) -> Candidate:
    p = str(ASSETS / f"sample_image_{n:02d}.jpg")
    return Candidate(id=cid, source=SourceKind.local, local_path=p, media="image", width=2400, height=1350,
                     keyframes=[p], score=score)


def shot(i: int, scene: str, cands: list[Candidate], start: float | None = None,
         typ: ShotType = ShotType.footage) -> Shot:
    at = float(i * 5) if start is None else start
    return Shot(id=f"shot_{i:03d}", index=i, beat_id="b1", scene_id=scene, start=at, end=at + 5.0, duration=5.0,
                type=typ, intent="Archival footage", candidates=cands)


class _Log:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def info(self, msg: str, **f: Any) -> None:
        pass

    def warn(self, msg: str, **f: Any) -> None:
        self.warnings.append(msg)


class _Ctx:
    def __init__(self) -> None:
        self.log = _Log()


def select(shots: list[Shot], threshold: float = 60.0) -> tuple[list[Shot], _Ctx]:
    ctx = _Ctx()
    sl = ShotList(project_id="t1", duration=float(len(shots) * 5), shots=shots)
    dl._select_all(sl, ctx, threshold, {})
    return sl.shots, ctx


def test_same_segment_not_picked_twice_in_one_scene_even_when_best():
    seg = lambda sid: vid(f"{sid}_yt_0", 95, 10.0)  # noqa: E731
    shots = [shot(0, "s1", [seg("a")]),
             shot(1, "s1", [vid("b_px", 80, 0.0, url="https://pexels/1", path="/w/pexels_ff.mp4",
                                source=SourceKind.pexels)]),
             shot(2, "s1", [seg("c"), vid("c_px", 70, 0.0, url="https://pexels/2", path="/w/pexels_ee.mp4",
                                          source=SourceKind.pexels)])]
    out, ctx = select(shots)
    assert out[2].selected == "c_px"
    assert ctx.log.warnings == []


def test_never_in_consecutive_shots_even_across_a_scene_boundary():
    other = vid("b_px", 60, 0.0, url="https://pexels/1", path="/w/pexels_ff.mp4", source=SourceKind.pexels)
    out, _ = select([shot(0, "s1", [vid("a", 99, 10.0)]), shot(1, "s2", [vid("b", 99, 11.0), other])])
    assert out[1].selected == "b_px"


def test_motion_graphic_between_does_not_break_adjacency():
    other = vid("c_px", 60, 0.0, url="https://pexels/1", path="/w/pexels_ff.mp4", source=SourceKind.pexels)
    shots = [shot(0, "s1", [vid("a", 99, 10.0)]), shot(1, "s2", [], typ=ShotType.motion_graphic),
             shot(2, "s2", [vid("c", 99, 10.0), other])]
    out, _ = select(shots)
    assert out[2].selected == "c_px"


def _across(reused: float, alt: float, alt_only: bool = False) -> str:
    """Segment used in scene s1, then two unrelated shots, then offered again in s2 far later."""
    filler = [shot(i, "s1", [vid(f"f{i}", 90, 0.0, url=f"https://pexels/{i}", path=f"/w/pexels_{i}.mp4",
                                 source=SourceKind.pexels)]) for i in (1, 2)]
    cands = [vid("again", reused, 10.0)]
    if not alt_only:
        cands.append(vid("alt", alt, 0.0, url="https://pexels/9", path="/w/pexels_9.mp4", source=SourceKind.pexels))
    later = shot(3, "s2", cands, start=200.0)
    out, _ = select([shot(0, "s1", [vid("first", 95, 10.0)])] + filler + [later])
    return out[3].selected


def test_across_scenes_strong_penalty_lets_slightly_lower_alternative_win():
    assert _across(95, 88) == "alt"
    assert dl.CROSS_SCENE_SEGMENT_PENALTY > 15.0  # stronger than the old same segment penalty


def test_across_scenes_reuse_still_wins_when_far_better_or_only_option():
    out, _ = select([shot(0, "s1", [vid("first", 95, 10.0)]),
                     shot(1, "s1", [vid("f", 90, 0.0, url="https://pexels/1", path="/w/pexels_1.mp4",
                                        source=SourceKind.pexels)]),
                     shot(2, "s2", [vid("again", 99, 10.0),
                                    vid("alt", 48, 0.0, url="https://pexels/9", path="/w/pexels_9.mp4",
                                        source=SourceKind.pexels)], start=200.0)], threshold=40.0)
    assert out[2].selected == "again"  # 99 - 45 beats 48
    assert _across(70, 0, alt_only=True) == "again"


def test_different_segments_of_one_video_allowed_in_the_same_scene():
    shots = [shot(0, "s1", [vid("a", 95, 10.0)]),
             shot(1, "s1", [vid("b_px", 85, 0.0, url="https://pexels/1", path="/w/pexels_1.mp4",
                                source=SourceKind.pexels)]),
             shot(2, "s1", [vid("c", 95, 30.0), vid("c_px", 70, 0.0, url="https://pexels/2", path="/w/pexels_2.mp4",
                                                    source=SourceKind.pexels)])]
    out, _ = select(shots)
    assert out[2].selected == "c"


def test_youtube_windows_compare_on_absolute_source_time():
    a = vid("a", 90, 2.0, path="/w/s1/youtube_abc123_40.mp4")
    overlap = vid("b", 90, 0.5, path="/w/s2/youtube_abc123_42.mp4")  # 42.5 s vs 42 s: same footage
    apart = vid("c", 90, 2.0, path="/w/s2/youtube_abc123_120.mp4")  # 122 s: another part of the video
    copy = vid("d", 90, 2.5, path="/w/s9/youtube_abc123_40.mp4")  # per shot copy of the same window
    assert dl.same_segment(a, overlap) and dl.same_segment(a, copy)
    assert not dl.same_segment(a, apart)


def test_unprovable_files_of_one_url_count_as_the_same_segment():
    a = vid("a", 90, 2.0, path="/w/clip_x.mp4", source=SourceKind.archive)
    b = vid("b", 90, 50.0, path="/w/clip_y.mp4", source=SourceKind.archive)
    assert dl.same_segment(a, b)
    c = vid("c", 90, 50.0, path="/w/clip_x.mp4", source=SourceKind.archive)
    assert not dl.same_segment(a, c)


def test_ranges_overlap_even_when_starts_are_far_apart():
    a = vid("a", 90, 0.0).model_copy(update={"out_point": 12.0})
    b = vid("b", 90, 8.0)
    assert dl.same_segment(a, b)


def test_stills_keep_repeating_with_derived_framings():
    shots = [shot(0, "s1", [still("a", 92)], typ=ShotType.image),
             shot(1, "s1", [still("b", 92), still("b2", 75, n=2)], typ=ShotType.image)]
    out, ctx = select(shots)
    assert out[1].selected == "b"  # 92 - 15 beats 75: a still is not a segment, only the reuse penalty applies
    assert ctx.log.warnings == []
    assert not dl.same_segment(still("x", 1), still("y", 1))


def test_never_empty_when_every_option_is_banned():
    shots = [shot(0, "s1", [vid("a", 95, 10.0)]),
             shot(1, "s1", [vid("b", 95, 30.0)]),
             shot(2, "s1", [vid("c_same_prev", 95, 30.0), vid("c_same_scene", 80, 11.0)])]
    out, ctx = select(shots)
    assert out[2].selected == "c_same_scene"  # not the previous shot's segment wins first
    assert ctx.log.warnings
    lone, ctx2 = select([shot(0, "s1", [vid("a", 95, 10.0)]), shot(1, "s1", [vid("b", 95, 10.0)])])
    assert lone[1].selected == "b" and ctx2.log.warnings


def test_pinned_upload_kept_even_when_banned():
    up = vid("up", 50, 10.0, source=SourceKind.upload)
    s1 = shot(1, "s1", [up, vid("alt", 90, 0.0, url="https://pexels/1", path="/w/pexels_1.mp4",
                                source=SourceKind.pexels)])
    s1.selected = "up"
    out, _ = select([shot(0, "s1", [vid("a", 95, 10.0)]), s1])
    assert out[1].selected == "up"


@pytest.mark.parametrize("scene", ["s1", "s2"])
def test_usage_records_scene_and_index(scene):
    usage = dl._Usage()
    usage.add(vid("a", 90, 10.0), 0.0, scene, 4)
    assert usage.picks[-1].scene_id == scene and usage.picks[-1].index == 4
    assert usage.banned(vid("b", 90, 11.0), "other")  # previous shot always banned
