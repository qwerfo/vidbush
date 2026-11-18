"""Picture layout kinds: nothing before its cue, split sides fill on their own word, name card fallback."""
import numpy as np
import pytest
from PIL import Image

from vidbush.mg.layouts import comp_class
from vidbush.mg.layouts.base import FrameProfile, LayoutJob, PictureRef
from vidbush.mg.layouts.pictures import NameCard, NameLowerThird
from vidbush.mg.layouts.timing import Cue
from vidbush.models import LayoutElement, LayoutKind, LayoutSpec

W, H = 480, 270
PROFILES = [FrameProfile(letterbox=True, captions=True), FrameProfile(letterbox=False, captions=False)]


def _pic(tmp_path, name, color):
    p = tmp_path / f"{name}.png"
    img = np.zeros((300, 450, 3), np.uint8)
    img[:] = color
    img[100:200, 150:300] = 255 - np.array(color, np.uint8)
    Image.fromarray(img).save(p)
    return PictureRef(str(p), "image", source="wikimedia")


def _job(kind, elements, appears, pictures, frame, dur=3.0):
    spec = LayoutSpec(kind=kind, elements=[LayoutElement(**e) for e in elements])
    cues = [Cue(i, a, a + 0.25, a + 0.6, True) for i, a in enumerate(appears)]
    return LayoutJob(spec=spec, cues=cues, duration=dur, width=W, height=H, fps=10, frame=frame, pictures=pictures)


def _cases(tmp_path):
    a, b, c = _pic(tmp_path, "a", (200, 40, 40)), _pic(tmp_path, "b", (40, 200, 40)), _pic(tmp_path, "c", (40, 40, 200))
    return {
        LayoutKind.split_comparison: ([dict(role="left", text="Alaska"), dict(role="right", text="Fort Worth")],
                                      [1.0, 2.0], {0: a, 1: b}),
        LayoutKind.name_card: ([dict(role="portrait"), dict(role="name", text="Leonhard Seppala"),
                                dict(role="role", text="Musher, Nome")], [1.0, 1.0, 1.4], {0: a}),
        LayoutKind.list_slideshow: ([dict(role="item", text="Bush plane"), dict(role="item", text="Snowmobile"),
                                     dict(role="item", text="Dog sled")], [1.0, 1.5, 2.0], {0: a, 1: b, 2: c}),
        LayoutKind.date_title: ([dict(role="background"), dict(role="date", text="January 2020"),
                                 dict(role="label", text="Toksook Bay")], [1.0, 1.0, 1.8], {0: a}),
        LayoutKind.equals_card: ([dict(role="left", text="240 miles"), dict(role="right", text="about 1 inch")],
                                 [1.0, 2.0], {}),
    }


@pytest.mark.parametrize("frame", PROFILES)
def test_nothing_drawn_before_first_cue(tmp_path, frame):
    for kind, (els, appears, pics) in _cases(tmp_path).items():
        comp = comp_class(kind)(_job(kind, els, appears, pics, frame), W, H, False)
        before = comp.render(9)  # t = 0.9 s, first cue appears at 1.0 s
        assert np.array_equal(before, comp.canvas(9)), kind
        assert not np.array_equal(comp.render(25), comp.canvas(25)), kind  # and something shows later
        comp.close()


@pytest.mark.parametrize("frame", PROFILES)
def test_split_right_half_empty_until_its_cue(tmp_path, frame):
    els, appears, pics = _cases(tmp_path)[LayoutKind.split_comparison]
    comp = comp_class(LayoutKind.split_comparison)(_job(LayoutKind.split_comparison, els, appears, pics, frame), W, H)
    right = slice(W // 2 + 4, W)
    f = comp.render(19)  # t = 1.9: left shown, right not yet (appears at 2.0)
    assert not np.array_equal(f[:, : W // 2], comp.canvas(19)[:, : W // 2])
    assert np.array_equal(f[:, right], comp.canvas(19)[:, right])
    assert not np.array_equal(comp.render(28)[:, right], comp.canvas(28)[:, right])


def test_name_card_falls_back_without_real_portrait(tmp_path):
    els = [dict(role="portrait", picture="x"), dict(role="name", text="Lizzie"), dict(role="role", text="Elder")]
    real = _pic(tmp_path, "p", (90, 90, 90))
    generated = PictureRef(real.path, "image", source="ai_image")
    frame = PROFILES[0]
    assert NameCard.fallback_needed(_job(LayoutKind.name_card, els, [1, 1, 1], {}, frame))
    assert NameCard.fallback_needed(_job(LayoutKind.name_card, els, [1, 1, 1], {0: generated}, frame))
    assert not NameCard.fallback_needed(_job(LayoutKind.name_card, els, [1, 1, 1], {0: real}, frame))


@pytest.mark.parametrize("frame", PROFILES)
def test_lower_third_is_transparent_before_cue_and_inside_visible_area(tmp_path, frame):
    els = [dict(role="portrait"), dict(role="name", text="Lizzie"), dict(role="role", text="Elder, Toksook Bay")]
    job = _job(LayoutKind.name_card, els, [1.0, 1.0, 1.3], {}, frame)
    lt = NameLowerThird(job, W, H, False)
    assert not lt.render_rgba(9).any()
    f = lt.render_rgba(25)
    ys, xs = np.nonzero(f[..., 3] > 8)
    assert len(ys)
    x0, y0, x1, y1 = frame.visible(W, H)
    assert xs.max() < W / 2 and ys.min() > H / 2  # lower left
    assert ys.max() <= y1 + 4  # above the bars or the caption band (shadow allowance)
