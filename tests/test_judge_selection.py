"""Run 3 selection rules: spoken words first for real candidates, real over AI, the floor, type follows pick,
small archival stills accepted and framed."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from test_judge_director import cand, ctx, fake_vision, img, registry, shot, shotlist  # noqa: F401  fixtures
from test_render_helpers import make_clip, make_still
from vidbush.judge import director_loop as dl
from vidbush.judge import judge as J
from vidbush.models import MotionKind, MotionSpec, Narration, ShotType, SourceKind, Word
from vidbush.render.segment import MAX_STILL_UPSCALE, _prepare_still


def ai_cand(shot_id: str, n: int, image: int) -> Any:
    return cand(shot_id, n, image, id=f"{shot_id}_ai_image_{n}", source=SourceKind.ai_image, licence="AI generated")


def vision(scores: dict[str, float], reasons: Optional[dict[str, str]] = None, ai_score: float = 85.0):
    """fake_vision with a per-file reason; records each prompt."""
    prompts: list[str] = []

    def vision_json(role: str, task: str, system: str, prompt: str, images: list[str], stage: str,
                    intent: str = "", details: Optional[list[str]] = None) -> dict[str, Any]:
        prompts.append(prompt)
        name = Path(images[0]).name
        score = ai_score if Path(images[0]).parent.name == "ai" else scores.get(name, 50.0)
        return {"score": score, "reason": (reasons or {}).get(name, f"scored {name}"), "rejects": [], "_mock": False}

    vision_json.prompts = prompts  # type: ignore[attr-defined]
    return vision_json


# ---------------------------------------------------------------- 1. spoken words

def test_real_candidate_prompt_leads_with_spoken_words_ai_keeps_strict_brief():
    s = shot(0)
    real = cand("shot_000", 1, 1)
    p = J.judge_prompt(s, real, "The whole beat.", "words over the shot")
    assert p.startswith("Spoken during this shot: words over the shot")
    assert "Beat narration (context): The whole beat." in p and "Director's intent (a hint, not a checklist)" in p
    assert "Beat intent:" not in p and "Must show:" not in p
    ai = ai_cand("shot_000", 1, 1)
    q = J.judge_prompt(s, ai, "The whole beat.", "words over the shot")
    assert "Spoken during" not in q and q.startswith("Narration: The whole beat.") and "Must show: steam shovel" in q
    assert J.judge_prompt(s, real, "The whole beat.").startswith("Narration: The whole beat.")  # no timings
    assert "the words spoken during the shot come first" in J.JUDGE_SYSTEM


def test_score_many_gives_the_judge_words_spoken_in_the_shot_window(ctx, monkeypatch):
    ctx.save("narration", Narration(audio_path="n.wav", duration=4.0, words=[
        Word(text="before", start=0.0, end=0.4), Word(text="edge", start=0.6, end=0.9),
        Word(text="inside", start=1.2, end=1.6), Word(text="after", start=2.6, end=3.0)]))
    s = shot(1, cands=[cand("shot_001", 1, 1)])  # window 1.0 to 2.0 s, margin 0.5 s
    fake = vision({})
    monkeypatch.setattr(ctx.seats, "vision_json", fake)
    J.score_many([(s, s.candidates[0])], ctx)
    assert fake.prompts[0].startswith("Spoken during this shot: edge inside\n")


# ---------------------------------------------------------------- 2. real over AI

def test_real_at_55_is_not_converted_to_ai_but_still_requeried(ctx, monkeypatch, registry):
    s = shot(0, ShotType.image, [cand("shot_000", 1, 1)])
    monkeypatch.setattr(ctx.seats, "vision_json", fake_vision({"sample_image_01.jpg": 58}, ai_score=95))
    out = dl.judge_and_select(shotlist([s]), ctx).shots[0]
    assert len(registry["calls"]) == 1, "below 70 still requeries"
    assert not any(c.source == SourceKind.ai_image for c in out.candidates), "no AI still generated"
    assert out.selected == "shot_000_local_1" and out.type == ShotType.image
    assert "no AI still" in out.judge_notes


def test_real_at_55_beats_a_better_scored_ai_still_in_selection(ctx, monkeypatch, registry):
    s = shot(0, ShotType.footage, [cand("shot_000", 1, 1), ai_cand("shot_000", 1, 2)])
    monkeypatch.setattr(ctx.seats, "vision_json", fake_vision({"sample_image_01.jpg": 55, "sample_image_02.jpg": 92}))
    out = dl.judge_and_select(shotlist([s]), ctx).shots[0]
    assert out.selected == "shot_000_local_1" and out.type == ShotType.image  # real still: its media sets the type


def test_picture_kind_is_the_one_classifier(monkeypatch):
    assert J.picture_kind(cand("s", 1, 1, media="video")) == J.REAL_FOOTAGE
    assert J.picture_kind(cand("s", 1, 1)) == J.REAL_STILL
    assert J.picture_kind(ai_cand("s", 1, 1)) == J.AI_PHOTO
    assert J.picture_kind(cand("s", 1, 1, source=SourceKind.motion)) == J.GRAPHIC
    monkeypatch.setattr(J, "DESIGNED_SOURCES", {"pixabay"})
    assert J.picture_kind(cand("s", 1, 1, source=SourceKind.pixabay)) == J.DESIGNED and not J.is_real(
        cand("s", 1, 1, source=SourceKind.pixabay))


# ---------------------------------------------------------------- 3. floor and branding

@pytest.mark.parametrize("text,hit", [
    ("Clip ends on a like and subscribe animation", "subscribe"),
    ("Lighthouse fits, but a station logo watermark sits in the corner.", "logo"),
    ("fits the beat, though a corner archive logo is minor", "logo"),
    ("Ends with an end screen of the channel", "end screen"),
    ("Vintage bar; no modern elements or logos.", ""),
    ("Clear harbour; watermark: none", ""),
    ("Liberty Bell in Philadelphia, free of watermarks", ""),
])
def test_branding_note_reads_reason_text_and_respects_negation(text, hit):
    assert J.branding_note(text) == hit


def test_branding_the_brief_asks_for_is_subject():
    assert J.branding_note("shows the railway logo on the car", "Close detail of the railway logo") == ""


def test_branded_or_low_real_never_ships_ai_takes_the_slot(ctx, monkeypatch, registry):
    branded = shot(0, ShotType.footage, [cand("shot_000", 1, 1, media="video")])
    low = shot(1, ShotType.image, [cand("shot_001", 1, 2)])
    monkeypatch.setattr(ctx.seats, "vision_json", vision(
        {"sample_image_01.jpg": 88, "sample_image_02.jpg": 40},
        {"sample_image_01.jpg": "Right subject but the clip ends on a subscribe animation"}, ai_score=80))
    b, lo = dl.judge_and_select(shotlist([branded, low]), ctx).shots
    for out in (b, lo):
        assert out.selected_candidate().source == SourceKind.ai_image and out.type == ShotType.ai_image
        assert dl.BELOW_FLOOR_FLAG not in out.flags


def test_floor_slot_borrows_the_best_real_candidate_of_the_beat(ctx, monkeypatch, registry):
    good = shot(0, ShotType.image, [cand("shot_000", 1, 1), cand("shot_000", 2, 2)])
    weak = shot(1, ShotType.image_overlay, [cand("shot_001", 1, 3)], overlay=MotionSpec(
        kind=MotionKind.lower_third, duration=1.0, background="transparent", payload={"title": "x"}))
    sl = shotlist([good, weak])
    sl.shots[1].start, sl.shots[1].end = 90.0, 91.0  # outside the recent window, so a still may repeat
    monkeypatch.setattr(ctx.seats, "vision_json", fake_vision(
        {"sample_image_01.jpg": 90, "sample_image_02.jpg": 75, "sample_image_03.jpg": 20}))
    out = dl.judge_and_select(sl, ctx).shots[1]
    pick = out.selected_candidate()
    assert not any(c.source == SourceKind.ai_image for c in out.candidates), "no AI still while the beat has real"
    assert pick.id.startswith("shot_001_local_b") and dl.BORROWED_FLAG in pick.flags and pick.score >= 75
    assert out.type == ShotType.image_overlay and out.overlay is not None


def test_borrow_never_takes_a_later_siblings_best_and_skips_assets_already_held(ctx, monkeypatch, registry):
    # needy shot first; its donor follows. The donor's best (01) stays with it; 02 is already on the needy
    # shot (failed there), so the needy shot borrows the next one (03) instead of giving up.
    needy = shot(0, ShotType.image, [cand("shot_000", 1, 4), cand("shot_000", 2, 2, score=40.0)])
    donor = shot(1, ShotType.image, [cand("shot_001", 1, 1), cand("shot_001", 2, 2), cand("shot_001", 3, 3)])
    monkeypatch.setattr(ctx.seats, "vision_json", fake_vision(
        {"sample_image_01.jpg": 90, "sample_image_02.jpg": 85, "sample_image_03.jpg": 70,
         "sample_image_04.jpg": 20}))
    n, d = dl.judge_and_select(shotlist([needy, donor]), ctx).shots
    assert d.selected == "shot_001_local_1", "the donor keeps its best"
    assert n.selected_candidate().local_path == img(3) and dl.BORROWED_FLAG in n.selected_candidate().flags
    assert not any(c.source == SourceKind.ai_image for c in n.candidates + d.candidates)


def test_borrow_is_rejudged_when_the_donor_brief_missed_a_must_avoid_and_refused_falls_to_ai(
        ctx, monkeypatch, registry):
    needy = shot(0, ShotType.image, [cand("shot_000", 1, 4)], must_avoid=["people"])
    donor = shot(1, ShotType.image, [cand("shot_001", 1, 1), cand("shot_001", 2, 3)])
    base = fake_vision({"sample_image_01.jpg": 90, "sample_image_03.jpg": 70, "sample_image_04.jpg": 20})

    def vision_json(*a: Any, **k: Any) -> dict[str, Any]:
        out = base(*a, **k)
        if "Must avoid: people" in a[3] and Path(a[4][0]).name == "sample_image_03.jpg":
            out = {**out, "score": 15.0, "reason": "a crowd of people fills the dock"}
        return out

    monkeypatch.setattr(ctx.seats, "vision_json", vision_json)
    n, d = dl.judge_and_select(shotlist([needy, donor]), ctx).shots
    copy = next(c for c in n.candidates if dl.BORROWED_FLAG in c.flags)
    assert copy.score == 15.0, "judged against the needy shot's own brief"
    assert n.selected_candidate().source == SourceKind.ai_image and n.type == ShotType.ai_image
    assert d.selected == "shot_001_local_1"


def test_nothing_meets_floor_and_generation_fails_keeps_least_bad_flagged(ctx, monkeypatch, registry):
    s = shot(0, ShotType.image, [cand("shot_000", 1, 1), cand("shot_000", 2, 2), cand("shot_000", 3, 3)])
    monkeypatch.setattr(ctx.seats, "vision_json", vision(
        {"sample_image_01.jpg": 45, "sample_image_02.jpg": 30, "sample_image_03.jpg": 49},
        {"sample_image_03.jpg": "has a channel watermark"}))

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("image seat down")

    monkeypatch.setattr("vidbush.visuals.ai_images.generate_ai_candidate", boom)
    out = dl.judge_and_select(shotlist([s]), ctx).shots[0]
    assert out.selected == "shot_000_local_1", "branded 49 loses to clean 45"
    assert dl.BELOW_FLOOR_FLAG in out.flags and "no_match_accepted" in out.flags


# ---------------------------------------------------------------- 4. type follows pick

def test_sync_type_per_pick_kind(monkeypatch):
    overlay = MotionSpec(kind=MotionKind.lower_third, duration=1.0, background="transparent", payload={"title": "x"})
    s = shot(0, ShotType.footage_overlay, [cand("shot_000", 1, 1), ai_cand("shot_000", 1, 2),
                                           cand("shot_000", 2, 3, media="video")], overlay=overlay)
    s.selected = "shot_000_ai_image_1"
    dl.sync_type(s)
    assert s.type == ShotType.ai_image and s.overlay is overlay
    s.selected = "shot_000_local_1"
    dl.sync_type(s)
    assert s.type == ShotType.image_overlay
    s.selected = "shot_000_local_2"
    dl.sync_type(s)
    assert s.type == ShotType.footage_overlay
    plain = shot(1, ShotType.ai_image, [cand("shot_001", 1, 1, media="video")], selected="shot_001_local_1")
    dl.sync_type(plain)
    assert plain.type == ShotType.footage
    planned = shot(2, ShotType.ai_video, [ai_cand("shot_002", 1, 1)], selected="shot_002_ai_image_1")
    dl.sync_type(planned)
    assert planned.type == ShotType.ai_video, "a planned animation waits for _finish_ai_video"
    monkeypatch.setattr(J, "DESIGNED_SOURCES", {"pixabay"})
    art = shot(3, ShotType.image, [cand("shot_003", 1, 1, source=SourceKind.pixabay)], selected="shot_003_local_1")
    dl.sync_type(art)
    assert art.type == ShotType.ai_image and dl.DESIGNED_FLAG in art.flags


def test_ensure_usable_and_segment_repeat_ai_paths_set_ai_type(ctx, monkeypatch, registry):
    empty = shot(0, ShotType.footage)
    clip = cand("shot_001", 1, 1, media="video", url="https://x/v", in_point=0.0, out_point=4.0)
    first = shot(1, ShotType.footage, [clip])
    again = shot(2, ShotType.footage, [clip.model_copy(update={"id": "shot_002_local_1"})])
    monkeypatch.setattr(ctx.seats, "vision_json", fake_vision({"sample_image_01.jpg": 90}, ai_score=80))
    a, f, g = dl.judge_and_select(shotlist([empty, first, again]), ctx).shots
    assert a.selected_candidate().source == SourceKind.ai_image and a.type == ShotType.ai_image
    assert f.type == ShotType.footage
    assert g.selected_candidate().source == SourceKind.ai_image and g.type == ShotType.ai_image


# ---------------------------------------------------------------- 5. small archival stills

def test_small_public_domain_archival_still_is_accepted_stock_is_not():
    pd = cand("s", 1, 1, source=SourceKind.loc, licence="No known restrictions on publication", width=600, height=450)
    assert J.local_reject(pd) is None
    nara = pd.model_copy(update={"source": "nara", "licence": "Public domain"}, deep=True)
    assert J.archival_still(nara)
    assert J.archival_still(pd.model_copy(update={"source": SourceKind.wikimedia, "licence": "No restrictions"}))
    assert "too small" in J.local_reject(pd.model_copy(update={"width": 400}))
    assert "too small" in J.local_reject(cand("s", 1, 1, source=SourceKind.pexels, licence="Pexels", width=600))
    assert "too small" in J.local_reject(pd.model_copy(update={"licence": "CC BY-SA 4.0"}))


def test_small_still_renders_whole_on_a_blurred_fill_never_over_1_5x(tmp_path, settings):
    p = make_still(tmp_path / "small.png", size=(600, 400))
    kb = {"zoom0": 1.0, "zoom1": 1.0, "x0": 0.5, "y0": 0.5, "x1": 0.5, "y1": 0.5}
    img = _prepare_still(p, None, kb, settings, tmp_path / "cache", 1080).astype(int)
    assert img.shape[:2] == (1080, 1920)
    fw, fh = int(600 * MAX_STILL_UPSCALE), int(400 * MAX_STILL_UPSCALE)  # 900 x 600, centred
    x0, y0 = (1920 - fw) // 2, (1080 - fh) // 2
    inside, outside = img[y0 + 10:y0 + fh - 10, x0 + 10:x0 + fw - 10], img[y0:y0 + fh, 20:x0 - 20]
    assert np.abs(np.diff(inside, axis=1)).mean() > 2 * np.abs(np.diff(outside, axis=1)).mean(), "sharp centre"
    assert outside.mean() < inside.mean(), "darkened fill"
    big = _prepare_still(make_still(tmp_path / "big.png", size=(1600, 1000)), None, kb, settings,
                         tmp_path / "cache", 1080).astype(int)
    assert np.abs(np.diff(big[100:900, 20:300], axis=1)).mean() > 2, "a large still still covers the frame"
    # a derived detail crop (half of a 1920 still) is a zoom: it still covers the frame, no blurred border
    full = make_still(tmp_path / "full.png", size=(1920, 1080))
    zoom = _prepare_still(full, [0.25, 0.25, 0.5, 0.5], kb, settings, tmp_path / "cache", 1080).astype(int)
    assert np.abs(np.diff(zoom[100:900, 20:300], axis=1)).mean() > 2, "a crop of a large still covers the frame"


# ---------------------------------------------------------------- 6. short clip penalty

def test_short_window_in_a_long_file_is_not_penalised(tmp_path):
    long = make_clip(tmp_path / "long.mp4", 6.0, rate=30)
    s = shot(0, ShotType.footage)
    s.duration = 5.0
    c = cand("shot_000", 1, 1, media="video", local_path=str(long), in_point=0.5, out_point=2.0)
    assert J._short_clip_penalty(s, c, 80.0, "ok") == (80.0, "ok")
    short = make_clip(tmp_path / "short.mp4", 2.0, rate=30)
    assert J._short_clip_penalty(s, c.model_copy(update={"local_path": str(short)}), 80.0, "ok")[0] == 65.0


def test_finalize_flags_small_archival_instead_of_low_res(ctx, monkeypatch):
    from vidbush.source import keyframes, registry as reg
    monkeypatch.setattr(keyframes, "extract_keyframes", lambda c, d: None)
    pd = cand("x", 1, 1, source=SourceKind.wikimedia, licence="Public domain", width=700, height=500)
    stock = cand("x", 2, 2, source=SourceKind.pexels, licence="Pexels", width=1100, height=700)
    out = reg.finalize(shot(0), [pd, stock], ctx)
    assert "small_archival" in out[0].flags and "low_res" not in out[0].flags
    assert "low_res" in out[1].flags
