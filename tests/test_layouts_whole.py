"""A graphic is shown whole or not at all, and a crop never cuts through a face."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from vidbush.mg.layouts.base import LayoutJob, cover
from vidbush.mg.layouts.placement import cover_window, face_origin
from vidbush.mg.layouts.stage import _panel_window, render_shot_layouts
from vidbush.mg.layouts.timing import Cue
from vidbush.models import Candidate, LayoutElement, LayoutSpec, Narration, Shot, ShotList, Word

SENT = "the sea route from Alaska to Fort Worth was long and the town was far"


def _words(start: float) -> list[Word]:
    return [Word(text=t, start=start + i * 0.3, end=start + i * 0.3 + 0.25) for i, t in enumerate(SENT.split())]


def _ctx(tmp_path, nar: Narration) -> SimpleNamespace:
    warned: list[str] = []
    ctx = SimpleNamespace(
        maybe_load=lambda key, model: nar, brand=None, work=tmp_path,
        profile=SimpleNamespace(letterbox=SimpleNamespace(enabled=False, ratio=2.39),
                                captions=SimpleNamespace(enabled=False)),
        settings=SimpleNamespace(render=SimpleNamespace(width=320, height=180, fps=10)),
        log=SimpleNamespace(warn=lambda m, **k: warned.append(m), info=lambda m, **k: None))
    ctx.warned = warned
    return ctx


def _still(path, color=(90, 110, 130)) -> str:
    Image.new("RGB", (480, 270), color).save(path)
    return str(path)


def test_split_with_a_missing_picture_is_dropped_and_the_shot_keeps_its_picture(tmp_path):
    left = Candidate(id="s_left", source="wikimedia", local_path=_still(tmp_path / "l.png"))
    spec = LayoutSpec(kind="split_comparison", elements=[
        LayoutElement(role="left", text="Alaska", anchor="Alaska", picture="s_left"),
        LayoutElement(role="right", text="Fort Worth", anchor="Fort Worth", picture="s_gone")])
    own = Candidate(id="s_fit", source="wikimedia", local_path=_still(tmp_path / "own.png"), flags=["alt:fitting"])
    shot = Shot(id="s", beat_id="b", start=0, end=6, duration=6.0, type="motion_graphic", intent="x",
                layout=spec, candidates=[left, own])
    ctx = _ctx(tmp_path, Narration(audio_path="", duration=10, words=_words(0.2)))
    render_shot_layouts(ShotList(project_id="p", duration=6, shots=[shot]), ctx)
    assert shot.selected == "s_fit" and shot.overlay_path == ""  # no layout clip, the alternate shows
    assert not any(c.id == "s_layout_1" for c in shot.candidates)
    assert any("not shown" in n and "s_gone not found" in n for n in shot.layout_notes)
    assert not list(tmp_path.glob("mg/layout_*"))


def test_full_layout_with_late_words_enters_on_them_over_the_picture(tmp_path):
    own = Candidate(id="s_pic", source="wikimedia", local_path=_still(tmp_path / "own.png"))
    spec = LayoutSpec(kind="equals_card", elements=[
        LayoutElement(role="left", text="the sea route", anchor="sea route"),
        LayoutElement(role="right", text="long", anchor="long")])
    shot = Shot(id="s", beat_id="b", start=0, end=7, duration=7.0, type="image", intent="x", layout=spec,
                candidates=[own], selected="s_pic")
    ctx = _ctx(tmp_path, Narration(audio_path="", duration=10, words=_words(2.7)))  # "sea" at 3.0 s
    render_shot_layouts(ShotList(project_id="p", duration=7, shots=[shot]), ctx)
    assert shot.selected == "s_pic"  # the picture shows until the words
    assert shot.overlay_path.endswith(".mov") and 2.5 <= shot.layout_at <= 3.0
    assert any("enters on its first words" in n for n in shot.layout_notes)


def test_panel_with_under_3_s_before_the_cut_is_dropped_not_clipped():
    spec = LayoutSpec(kind="table", elements=[LayoutElement(role="row", text="a", anchor="a")])
    job = LayoutJob(spec=spec, cues=[Cue(0, 3.75, 4.0, 4.4, True)], duration=6.5)
    assert _panel_window(job, "panel", 6.5) is None  # 2.5 s from the first word to the cut
    start, end = _panel_window(job, "panel", 7.5)
    assert start == 3.75 and end == 7.0  # 3 s from the first word, not to the cut


def test_crop_keeps_a_face_near_the_edge_whole():
    """Haar finds no face in a drawn box, so the face box is given: the window math is what is tested."""
    img = np.full((1000, 2000, 3), 60, np.uint8)
    face = (0.86, 0.10, 0.97, 0.35)  # a face near the right edge of a wide picture
    img[100:350, 1720:1940] = (255, 0, 0)
    out = cover(img, 400, 500, faces=[face])  # a 0.8 portrait frame: a centred crop would miss the face
    assert out is not None and (out[..., 0] > 200).sum() == (img[..., 0] > 200).sum() * (500 / 1000) ** 2
    assert cover(img, 400, 500) is not None and (cover(img, 400, 500)[..., 0] > 200).sum() == 0  # centred
    assert cover(img, 400, 500, faces=[face, (0.02, 0.1, 0.1, 0.3)]) is None  # two faces too far apart
    win = cover_window([face], 2.0, 0.8)
    assert win is not None and win[0] <= face[0] and win[0] + win[2] >= face[2]
    assert face_origin([face], 0.05, 0.5) is None  # narrower than the face itself


def test_still_move_is_widened_so_every_face_stays_whole():
    from vidbush.render.stillmotion import keep_faces
    kb = {"zoom0": 2.0, "zoom1": 1.0, "x0": 0.3, "y0": 0.5, "x1": 0.5, "y1": 0.5}
    face = (0.7, 0.2, 0.9, 0.5)
    held, changed = keep_faces(kb, [face])
    side = 1 / held["zoom0"]
    assert changed and held["x0"] - side / 2 <= face[0] and held["x0"] + side / 2 >= face[2]
    assert keep_faces(kb, []) == (kb, False)
