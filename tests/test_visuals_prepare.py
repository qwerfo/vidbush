"""Derived crops, repeated clip in points, AI image candidates, prepare_visuals glue."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vidbush.models import Candidate, Shot, ShotList, ShotType, SourceKind
from vidbush.visuals import ai_images, depthflow
from vidbush.visuals.derive import derive_crops, framing_crop, iou, ranked_crops, saliency_map, vary_repeated_video
from vidbush.visuals.prepare import prepare_visuals


@pytest.fixture
def salient_image(tmp_path: Path) -> Path:
    """Smooth dark gradient with one busy textured patch in the upper right third."""
    rng = np.random.default_rng(3)
    h, w = 1350, 2400
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 60 + 40 * (xx / w) + 20 * (yy / h)
    img = np.stack([base, base * 0.9, base * 0.8], axis=-1)
    y0, x0 = 300, 1650
    patch = rng.integers(0, 255, (360, 480, 1)).astype(np.float32)
    checker = ((np.indices((360, 480)).sum(0) // 24) % 2 * 200)[..., None]
    img[y0:y0 + 360, x0:x0 + 480] = np.clip(0.5 * patch + checker, 0, 255)
    p = tmp_path / "salient.jpg"
    Image.fromarray(img.astype(np.uint8)).save(p, quality=92)
    return p


def still_shot(i: int, path: Path, framing: str = "full") -> Shot:
    c = Candidate(id=f"shot_{i:03d}_local_1", source=SourceKind.local, local_path=str(path), media="image",
                  width=2400, height=1350, keyframes=[str(path)])
    return Shot(id=f"shot_{i:03d}", index=i, beat_id="b", start=i * 5.0, end=i * 5.0 + 5, duration=5.0,
                type=ShotType.image, intent="detail", candidates=[c], selected=c.id, framing=framing)


def test_saliency_finds_the_busy_patch(salient_image):
    sal = saliency_map(salient_image)
    rows, cols = sal.shape
    ys, xs = np.unravel_index(np.argmax(sal), sal.shape)
    assert xs / cols > 0.6 and ys / rows < 0.6


def test_detail_crop_lands_on_patch_and_stays_inside(salient_image):
    crop = framing_crop(str(salient_image), 2400, 1350, "detail")
    x, y, w, h = crop
    assert 0 <= x and 0 <= y and x + w <= 1 + 1e-6 and y + h <= 1 + 1e-6
    assert w * 2400 / (h * 1350) == pytest.approx(16 / 9, rel=0.02)
    cx, cy = x + w / 2, y + h / 2
    assert 0.62 < cx and cy < 0.62  # patch center is at (0.79, 0.36)


def test_ranked_crops_are_distinct(salient_image):
    crops = ranked_crops(saliency_map(salient_image), 0.45, 0.45, n=3)
    assert len(crops) == 3
    for i in range(3):
        for j in range(i + 1, 3):
            assert iou(crops[i], crops[j]) <= 0.35


def test_reused_still_gets_full_tight_detail(make_ctx, salient_image):
    ctx = make_ctx()
    shots = [still_shot(0, salient_image), still_shot(2, salient_image), still_shot(4, salient_image),
             still_shot(9, salient_image)]
    sl = ShotList(project_id="t1", duration=50, shots=shots)
    derive_crops(sl, ctx)
    framings = [s.framing for s in sl.shots]
    assert framings[:3] == ["full", "tight", "detail"] and framings[3] == "full"
    crops = [s.selected_candidate().crop for s in sl.shots]
    assert crops[0] is None and crops[3] is None
    assert crops[1] != crops[2] and crops[1][2] > crops[2][2]
    for c in crops[1:3]:
        assert c[0] >= 0 and c[1] >= 0 and c[0] + c[2] <= 1 + 1e-6 and c[1] + c[3] <= 1 + 1e-6


def test_single_tight_shot_gets_crop(make_ctx, salient_image):
    ctx = make_ctx()
    sl = ShotList(project_id="t1", duration=5, shots=[still_shot(0, salient_image, "tight")])
    derive_crops(sl, ctx)
    assert sl.shots[0].selected_candidate().crop is not None


def test_repeated_video_gets_different_in_points(make_ctx):
    ctx = make_ctx()
    clip = Path(ctx.settings.repo_root) / "fixtures" / "sample_assets" / "sample_footage_01.mp4"
    shots = []
    for i in (0, 1, 3):
        c = Candidate(id=f"shot_{i:03d}_local_1", source=SourceKind.local, local_path=str(clip), media="video",
                      in_point=0.0, out_point=4.0, width=1920, height=1080, keyframes=[])
        shots.append(Shot(id=f"shot_{i:03d}", index=i, beat_id="b", start=i * 4.0, end=i * 4.0 + 4, duration=4.0,
                          type=ShotType.footage, intent="x", candidates=[c], selected=c.id))
    sl = ShotList(project_id="t1", duration=16, shots=shots)
    assert vary_repeated_video(sl, ctx) == 2
    ins = [s.selected_candidate().in_point for s in sl.shots]
    assert len(set(ins)) == 3
    for s in sl.shots:
        c = s.selected_candidate()
        assert c.out_point - c.in_point == pytest.approx(4.0) and c.out_point <= 12.0 + 1e-6


def test_ai_candidate_prompt_and_files(make_ctx):
    ctx = make_ctx()
    s = Shot(id="shot_005", beat_id="b", start=0, end=5, duration=5, type=ShotType.ai_image,
             intent="Archival footage of workers in the Culebra Cut", must_show=["steam shovel"],
             must_avoid=["modern trucks"])
    prompt = ai_images.build_prompt(s)
    assert prompt.startswith("Workers in the Culebra Cut") and "steam shovel" in prompt
    assert "no watermark" in prompt and "Do not include modern trucks" in prompt
    assert chr(0x2014) not in prompt
    c = ai_images.generate_ai_candidate(s, ctx)
    assert c.id == "shot_005_ai_image_1" and Path(c.local_path) == ctx.work / "ai" / "shot_005_1.png"
    assert c.keyframes == [c.local_path] and c.flags == ["mock"] and c.source == SourceKind.ai_image
    assert (c.width, c.height) == (1920, 1080)
    s.candidates.append(c)
    assert ai_images.generate_ai_candidate(s, ctx).id == "shot_005_ai_image_2"


def test_prepare_visuals_depthflow_missing_keeps_still(make_ctx, salient_image, monkeypatch):
    ctx = make_ctx()
    ctx.profile.motion.depthflow = True
    monkeypatch.setattr(depthflow, "available", lambda: False)
    sl = ShotList(project_id="t1", duration=10, shots=[still_shot(0, salient_image), still_shot(1, salient_image)])
    out = prepare_visuals(sl, ctx)
    assert all(s.selected_candidate().media == "image" for s in out.shots)
    assert out.shots[1].selected_candidate().crop is not None


def test_prepare_visuals_depthflow_swaps_to_video(make_ctx, salient_image, monkeypatch):
    ctx = make_ctx()
    ctx.profile.motion.depthflow = True
    monkeypatch.setattr(depthflow, "available", lambda: True)

    def fake_render(image: Path, out: Path, duration: float, w: int, h: int, fps: int) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake")
        return out

    monkeypatch.setattr(depthflow, "render_parallax", fake_render)
    monkeypatch.setattr(depthflow, "keyframes", lambda *a: [])
    sl = ShotList(project_id="t1", duration=5, shots=[still_shot(0, salient_image)])
    c = prepare_visuals(sl, ctx).shots[0].selected_candidate()
    assert c.media == "video" and "depthflow" in c.flags and c.local_path.endswith(".mp4")


def test_ai_video_shot_without_real_seat_becomes_ai_image(make_ctx, salient_image):
    ctx = make_ctx()
    ctx.profile.visual_mix["ai_video"] = 5
    s = still_shot(0, salient_image)
    s.type = ShotType.ai_video
    out = prepare_visuals(ShotList(project_id="t1", duration=5, shots=[s]), ctx)
    assert out.shots[0].type == ShotType.ai_image
