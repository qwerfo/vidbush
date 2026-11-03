"""Source edge cases in segment rendering: portrait, 16-bit stills, HDR, short clips, interlace, content keys."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from test_render_helpers import FPS, count_frames, frame_at, make_clip, make_still
from vidbush.models import Candidate, Shot, StyleProfile
from vidbush.render import ffmpeg_cmds as fc
from vidbush.render.cache import content_fingerprint, segment_hash
from vidbush.render.segment import (_prepare_still, footage_plan, render_segment, to_8bit_rgb, tonemap_chain,
                                    _video_input)


def _shot(path: Path, media: str, dur: float, **cand) -> Shot:
    c = Candidate(id="c1", source="local", local_path=str(path), media=media, **cand)
    return Shot(id="x1", beat_id="b1", start=0.0, end=dur, duration=dur, type="footage" if media == "video"
                else "image", intent="test", candidates=[c], selected="c1")


def test_content_fingerprint_ignores_mtime_and_name(tmp_path) -> None:
    a = make_still(tmp_path / "a.png")
    b = tmp_path / "copy_b.png"
    shutil.copyfile(a, b)
    os.utime(b, (1_000_000, 1_000_000))
    assert content_fingerprint(a) == content_fingerprint(b)
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (3 << 20))
    fp = content_fingerprint(big)
    with open(big, "r+b") as f:  # same size, last byte changes, mtime kept the same
        st = big.stat()
        f.seek((3 << 20) - 1)
        f.write(b"y")
    os.utime(big, ns=(st.st_atime_ns, st.st_mtime_ns))
    from vidbush.render import cache
    cache._FP_MEMO.clear()
    assert content_fingerprint(big) != fp
    assert content_fingerprint(tmp_path / "nope") == "missing"


def test_segment_hash_survives_copy_with_new_mtime(tmp_path, settings) -> None:
    src = make_still(tmp_path / "s.jpg")
    copy = tmp_path / "restored" / "s.jpg"
    copy.parent.mkdir()
    shutil.copyfile(src, copy)
    os.utime(copy, (2_000_000, 2_000_000))
    prof = StyleProfile(id="t")
    assert segment_hash(_shot(src, "image", 2.0), prof, [], settings) == \
        segment_hash(_shot(copy, "image", 2.0), prof, [], settings)


def test_sixteen_bit_grey_still_is_not_white(tmp_path) -> None:
    a = (np.tile(np.linspace(0, 65535, 256), (64, 1))).astype(np.uint16)
    p = tmp_path / "scan16.png"
    Image.fromarray(a).save(p)
    rgb = np.asarray(to_8bit_rgb(Image.open(p)))
    assert rgb.shape[2] == 3 and 100 < rgb.mean() < 155 and (rgb == 255).mean() < 0.05


def test_portrait_still_is_blur_filled(tmp_path, settings) -> None:
    p = make_still(tmp_path / "tall.png", size=(600, 1000))
    kb = {"zoom0": 1.0, "zoom1": 1.0, "x0": 0.5, "y0": 0.5, "x1": 0.5, "y1": 0.5}
    img = _prepare_still(p, None, kb, settings, tmp_path / "cache", 803)
    assert img.shape[:2] == (1080, 1920)
    centre = img[138:941, 960]
    assert (np.abs(centre.astype(int) - [200, 40, 40]).sum(axis=1) < 60).mean() > 0.2, "whole disc visible"


def test_portrait_footage_keeps_full_height(tmp_path, settings) -> None:
    clip = make_clip(tmp_path / "tall.mp4", 1.0, size="360x640", rate=30)
    shot = _shot(clip, "video", 0.5)
    out = render_segment(shot, StyleProfile(id="t"), [], settings, tmp_path, (0, 15))
    assert count_frames(out) == 15
    img = frame_at(out, 5).astype(int)
    left, mid = img[300:800, 100:400], img[300:800, 860:1060]
    assert np.abs(np.diff(left, axis=1)).mean() < np.abs(np.diff(mid, axis=1)).mean(), "sides blurred, centre sharp"


def test_short_clip_extends_into_rest_of_file(tmp_path) -> None:
    clip = make_clip(tmp_path / "c.mp4", 6.0, rate=30)
    cand = Candidate(id="c", source="local", local_path=str(clip), media="video", in_point=1.0, out_point=2.5)
    plan = footage_plan(cand, clip, 4.0, FPS)
    assert plan.start == 1.0 and plan.slow == 1.0 and plan.hold == 0.0
    late = cand.model_copy(update={"in_point": 4.0, "out_point": 5.5})
    plan = footage_plan(late, clip, 4.0, FPS)
    assert plan.start == 2.0 and plan.hold == 0.0, "starts earlier when the file ends first"
    tiny = make_clip(tmp_path / "t.mp4", 1.5, rate=30)
    plan = footage_plan(cand.model_copy(update={"in_point": 0.0, "out_point": 1.5}), tiny, 4.0, FPS)
    assert plan.slow == 1.25 and 2.0 < plan.hold < 2.3


def test_slowed_clip_blends_instead_of_duplicating(tmp_path, settings) -> None:
    clip = make_clip(tmp_path / "c.mp4", 1.0, rate=30)
    cand = Candidate(id="c", source="local", local_path=str(clip), media="video")
    plan = footage_plan(cand, clip, 1.2, FPS)
    _, chain = _video_input(cand, clip, plan, settings, 803)
    assert "framerate=fps=30" in chain and "tpad=stop_mode=clone:stop=-1" in chain


def test_interlaced_source_deinterlaces_per_field(tmp_path, settings) -> None:
    clip = make_clip(tmp_path / "c.mp4", 1.0)
    cand = Candidate(id="c", source="local", local_path=str(clip), media="video")
    _, chain = _video_input(cand, clip, footage_plan(cand, clip, 0.5, FPS), settings, 803)
    assert "bwdif=mode=send_field" in chain


def test_hdr_transfer_gets_tone_mapped() -> None:
    assert tonemap_chain("bt709") == ""
    if fc.has_filter("zscale"):
        assert "tonemap=tonemap=hable" in tonemap_chain("smpte2084")
        assert "tin=arib-std-b67" in tonemap_chain("arib-std-b67")


def test_hdr_clip_renders(tmp_path, settings) -> None:
    if not fc.has_filter("zscale"):
        return
    clip = tmp_path / "hdr.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=1",
                    "-pix_fmt", "yuv420p10le", "-c:v", "libx264", "-color_trc", "smpte2084", "-color_primaries",
                    "bt2020", "-colorspace", "bt2020nc", str(clip)], check=True)
    shot = _shot(clip, "video", 0.4)
    out = render_segment(shot, StyleProfile(id="t"), [], settings, tmp_path, (0, 12))
    assert count_frames(out) == 12
