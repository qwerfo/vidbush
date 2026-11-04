"""Segment dedupe rule in the selection scorer: banned within a scene and next shot, penalised across scenes."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from test_editor_helpers import PID, build_edit_project, load_shotlist
from vidbush.config import REPO_ROOT
from vidbush.editor.ops import apply_ops
from vidbush.judge import director_loop as dl
from vidbush.models import Candidate, EditOp, EditOpName, Shot, ShotList, ShotType, SourceKind
from vidbush.visuals.derive import vary_repeated_video

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
    def __init__(self, threshold: float = 60.0) -> None:
        self.log = _Log()
        self.settings = SimpleNamespace(judge=SimpleNamespace(threshold=threshold))


@pytest.fixture(autouse=True)
def no_ai(monkeypatch):
    """Offline: AI still generation adds nothing unless a test swaps in a generator."""
    calls: list[str] = []
    monkeypatch.setattr(dl, "_add_ai", lambda shot, ctx: calls.append(shot.id))
    return calls


def pexels(cid: str, score: float, n: int = 1) -> Candidate:
    return vid(cid, score, 0.0, url=f"https://pexels/{n}", path=f"/w/pexels_{n}.mp4", source=SourceKind.pexels)


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


def test_banned_passing_segment_loses_to_allowed_sub_threshold_candidate(no_ai):
    shots = [shot(0, "s1", [vid("a", 95, 10.0)]), shot(1, "s1", [pexels("b_px", 80)]),
             shot(2, "s1", [vid("c", 95, 10.0), pexels("c_px", 55, 2)])]
    out, ctx = select(shots)
    assert out[2].selected == "c_px" and "no_match_accepted" in out[2].flags
    assert ctx.log.warnings == [] and no_ai == []


def test_consecutive_banned_passing_segment_loses_to_sub_threshold_alternative():
    out, _ = select([shot(0, "s1", [vid("a", 95, 10.0)]), shot(1, "s1", [vid("b", 95, 10.5), pexels("alt", 50)])])
    assert out[1].selected == "alt"


def test_ai_still_generated_when_only_banned_segments_remain(monkeypatch):
    def add_ai(s: Shot, ctx: Any) -> None:
        s.candidates.append(still("ai", 72).model_copy(update={"source": SourceKind.ai_image}))

    monkeypatch.setattr(dl, "_add_ai", add_ai)
    out, _ = select([shot(0, "s1", [vid("a", 95, 10.0)]), shot(1, "s1", [vid("b", 95, 10.0)])])
    assert out[1].selected == "ai" and out[1].flags == []


def test_kept_repeat_is_flagged_for_qa(no_ai):
    out, ctx = select([shot(0, "s1", [vid("a", 95, 10.0)]), shot(1, "s1", [vid("b", 95, 10.0)])])
    assert out[1].selected == "b" and no_ai == ["shot_001"]
    assert {"no_match_accepted", dl.SEGMENT_REPEAT_FLAG} <= set(out[1].flags) and ctx.log.warnings
    dl._apply(out[1], out[1].candidates[0], False, [])
    assert dl.SEGMENT_REPEAT_FLAG not in out[1].flags


def test_enforce_segment_rule_bans_the_next_shot_and_the_scene():
    nxt = shot(2, "s1", [vid("n", 95, 10.0)])
    nxt.selected = "n"
    edited = shot(1, "s1", [vid("e", 95, 10.5), pexels("e_px", 70)])
    edited.selected = "e"
    sl = ShotList(project_id="t1", duration=15.0, shots=[shot(0, "s0", [pexels("p", 90, 3)]), edited, nxt])
    sl.shots[0].selected = "p"
    assert dl.enforce_segment_rule(sl, edited, _Ctx())
    assert edited.selected == "e_px"
    assert not dl.enforce_segment_rule(sl, edited, _Ctx())  # now clear: left alone


def test_editor_resource_does_not_repeat_the_previous_shots_segment(settings, monkeypatch):
    work = build_edit_project(settings)
    prev = load_shotlist(work).by_id("shot_006").selected_candidate()

    def source_shot(s: Shot, ctx: Any, queries: Any = None, exclude_ids: Any = ()) -> list[Candidate]:
        alt = prev.model_copy(update={"id": f"{s.id}_pexels_1", "url": "https://pexels/77",
                                      "source": SourceKind.pexels})
        return [prev.model_copy(update={"id": f"{s.id}_local_9"}), alt]

    def judge(sl: ShotList, ctx: Any) -> ShotList:
        for s in sl.shots:
            scores = {f"{s.id}_local_9": 95.0, f"{s.id}_pexels_1": 75.0}
            for c in s.candidates:
                c.score = scores.get(c.id, 10.0)
            s.selected = f"{s.id}_local_9"  # a one-shot selection cannot see shot_006
        return sl

    monkeypatch.setattr("vidbush.source.registry.source_shot", source_shot)
    monkeypatch.setattr("vidbush.judge.director_loop.judge_and_select", judge)
    res = apply_ops(settings, PID, [EditOp(op=EditOpName("change_type"), args={"shot_id": "shot_007",
                                                                               "type": "footage"})])
    assert not res.errors, res.errors
    assert load_shotlist(work).by_id("shot_007").selected == "shot_007_pexels_1"


def test_vary_repeated_video_never_shifts_onto_a_banned_segment(monkeypatch):
    monkeypatch.setattr("vidbush.visuals.derive._clip_length", lambda c, cache: 40.0)
    clip = lambda cid, t: vid(cid, 90, t, path="/w/clip.mp4", source=SourceKind.local)  # noqa: E731
    shots = [shot(0, "A", [clip("s0", 0.0)]), shot(1, "A", [pexels("s1", 90)]),
             shot(2, "B", [clip("s2", 5.0)]), shot(3, "B", [clip("s3", 0.0)])]
    for s in shots:
        s.selected = s.candidates[0].id
        s.candidates[0].out_point = s.candidates[0].in_point + 5.0
    sl = ShotList(project_id="t1", duration=20.0, shots=shots)
    assert vary_repeated_video(sl, None) == 1
    s3 = sl.shots[3].selected_candidate()
    assert not dl.same_segment(s3, sl.shots[2].selected_candidate())
    assert s3.in_point == 10.0 and s3.out_point == 15.0
