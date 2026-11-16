"""V2 layouts core: anchor timing, placement from the picture, panel timing, still motion, matched look."""
from __future__ import annotations

import numpy as np
import pytest

from vidbush.mg.house import MAX_LEAD_S, MIN_LABEL_PX, type_px
from vidbush.mg.layouts import placement as P
from vidbush.mg.layouts.base import FrameProfile, LayoutJob
from vidbush.mg.layouts.panel import NEVER, change_time
from vidbush.mg.layouts.timing import Cue, anchor_span, cue_times, tokens
from vidbush.models import LayoutElement, LayoutSpec

WORDS = [{"text": t, "start": i * 0.5, "end": i * 0.5 + 0.4} for i, t in enumerate(
    "If Manhattan were as empty as Alaska, the island would hold about twenty-nine people. Alaska again".split())]


def test_tokens_split_hyphens_and_punctuation():
    assert tokens("Twenty-nine people.") == ["twenty", "nine", "people"]


def test_anchor_found_and_nearest_occurrence_wins():
    assert anchor_span("twenty-nine people", WORDS) == (6.0, 6.9)
    assert anchor_span("Alaska", WORDS, near=7.0)[0] == 7.0
    assert anchor_span("Alaska", WORDS, near=1.0)[0] == 3.0
    assert anchor_span("not said", WORDS) is None


def test_cue_never_earlier_than_lead_before_word_and_relative_to_clip():
    els = [LayoutElement(role="number", text="29 people", anchor="twenty-nine people"),
           LayoutElement(role="label", text="x", anchor="nowhere")]
    cues = cue_times(els, WORDS, 5.0, 9.0)
    assert cues[0].found and cues[0].word == pytest.approx(1.0)
    assert cues[0].word - cues[0].appear <= MAX_LEAD_S + 1e-9
    assert not cues[1].found and cues[1].appear == cues[0].appear  # unspoken: with the first element


def test_type_floor_scales_with_frame():
    assert type_px(10, MIN_LABEL_PX, 1080) == MIN_LABEL_PX
    assert type_px(10, MIN_LABEL_PX, 2160) == 2 * MIN_LABEL_PX


def _analysis(face=None, text=None) -> P.Analysis:
    detail = np.full((270, 480), 0.2, np.float32)
    sal = np.zeros((270, 480), np.float32)
    sal[120:150, 220:260] = 1.0  # subject in the centre
    return P.Analysis([face] if face else [], [text] if text else [], detail, sal)


def test_panel_avoids_faces_and_text_and_varies_from_previous():
    vis = FrameProfile(letterbox=False, captions=False).visible(1920, 1080)
    size = (700, 400)
    face_ul = (0.05, 0.05, 0.25, 0.3)
    pl = P.choose([_analysis(face=face_ul)], size, vis, 1920, 1080)
    assert pl.position != "upper_left" and "covers a face" in pl.reason
    text_ur = (0.7, 0.05, 0.95, 0.2)
    pl2 = P.choose([_analysis(text=text_ur)], size, vis, 1920, 1080, previous=pl.position)
    assert pl2.position not in ("upper_right", pl.position)


def test_no_clear_position_returns_none():
    vis = (0.0, 0.0, 1920.0, 1080.0)
    a = _analysis()
    a.faces = [(0.0, 0.0, 1.0, 1.0)]  # a face filling the frame
    assert P.choose([a], (700, 400), vis, 1920, 1080).position is None


def _job(stages: list[int], words: list[float], dur: float = 10.0) -> LayoutJob:
    els = [LayoutElement(role="row", text=f"r{i}", stage=s) for i, s in enumerate(stages)]
    cues = [Cue(i, w - 0.25, w, w + 0.6, True, s) for i, (w, s) in enumerate(zip(words, stages))]
    return LayoutJob(spec=LayoutSpec(kind="table", elements=els), cues=cues, duration=dur)


def test_dock_after_stage1_read_and_expand_before_stage2():
    j = _job([1, 1, 2], [0.5, 2.0, 6.0])
    dock = change_time(j, "full_then_dock")
    assert dock >= 2.6 and dock <= 10.0 - 0.8 - 2.0
    exp = change_time(j, "panel_then_expand")
    assert exp + 0.8 <= 6.0 - 0.25 + 1e-9
    assert change_time(_job([1], [1.0]), "panel_then_expand") == NEVER


def test_panel_window_ends_with_content_not_at_cut():
    from vidbush.mg.layouts.stage import PANEL_HOLD_S, _panel_window
    j = _job([1, 1], [1.0, 2.0], dur=12.0)
    start, end = _panel_window(j, "panel", 12.0)
    assert start == pytest.approx(0.75) and end == pytest.approx(2.6 + PANEL_HOLD_S)


def test_matched_look_moves_toward_target_and_leaves_bw_colour_alone():
    from vidbush.render.look import correction
    dark_blue = (0.25, 0.20, 0.25, 0.40, 0.2)
    target = (0.45, 0.45, 0.44, 0.42, 0.18)
    f = correction(dark_blue, target, 0.6)
    assert "gamma=" in f and "colorchannelmixer" in f
    gam = float(f.split("gamma=")[1].split(",")[0])
    assert gam > 1.0  # brightens a dark source
    bw = correction((0.3, 0.3, 0.3, 0.3, 0.01), target, 0.6)
    assert "colorchannelmixer" not in bw and "saturation" not in bw


def test_still_motion_short_holds_sequence_and_narrow(tmp_path):
    from PIL import Image

    from vidbush.models import Candidate, Shot, StillSequence
    from vidbush.render import stillmotion as sm
    im = Image.new("RGB", (3000, 2000), (60, 70, 80))
    im.paste((240, 240, 240), (2200, 300, 2500, 600))  # one bright detail, upper right
    p = tmp_path / "s.png"
    im.save(p)
    c = Candidate(id="c", source="wikimedia", local_path=str(p), width=3000, height=2000)
    shot = Shot(id="s", beat_id="b", start=0, end=2, duration=2.0, type="image", intent="x")
    kb, why = sm.plan(shot, c, 1920, 1080, 0.12)
    assert "holds" in why and kb["zoom1"] - kb["zoom0"] < 0.03
    shot = shot.model_copy(update={"duration": 6.0, "still_sequence": StillSequence(id="q", step="detail")})
    kb, why = sm.plan(shot, c, 1920, 1080, 0.12)
    assert "detail" in why and kb["x0"] > 0.6 and kb["zoom1"] <= sm.zoom_cap(3000, 2000, 1920, 1080) + 1e-9
    shot = shot.model_copy(update={"still_sequence": None})
    kb, why = sm.plan(shot, c, 1920, 1080, 0.12)
    assert "detail" in why and (kb["x0"] > 0.6 or kb["x1"] > 0.6)
    assert sm.is_narrow(600, 900, 1920, 804) and not sm.is_narrow(1600, 1000, 1920, 804)
