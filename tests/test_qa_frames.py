"""Rendered-frame QA: per-shot frames from the shot's own window, what is really on screen, every on-screen
text item, checks decided without the model, the fail rule, the five-line summary and the calibration
label parser."""
from __future__ import annotations

import importlib.util
import json
import subprocess

import pytest

from vidbush.config import REPO_ROOT
from vidbush.judge import qa
from vidbush.models import (Candidate, MotionKind, MotionSpec, Narration, QAReport, Shot, ShotList, ShotType,
                            SourceKind, Word)
from vidbush.util import read_model

IMG = str(REPO_ROOT / "fixtures" / "sample_assets" / "sample_image_07.jpg")


def _shot(i: int, start: float, end: float, typ: ShotType = ShotType.footage, src: SourceKind = SourceKind.youtube,
          media: str = "video", overlay: MotionSpec | None = None, motion: MotionSpec | None = None,
          intent: str = "x", **cand) -> Shot:
    c = Candidate(id=f"c{i}", source=src, media=media, local_path=IMG, keyframes=[IMG], **cand)
    return Shot(id=f"shot_{i:03d}", index=i, beat_id="b1", start=start, end=end, duration=end - start, type=typ,
                intent=intent, overlay=overlay, motion=motion, candidates=[c], selected=c.id)


def _words(text: str, start: float = 0.0, step: float = 0.5) -> list[Word]:
    return [Word(text=w, start=start + k * step, end=start + k * step + 0.4) for k, w in enumerate(text.split())]


def _spec(kind: MotionKind, start_at: float = 0.0, anchor: str = "", **payload) -> MotionSpec:
    return MotionSpec(kind=kind, duration=5.0, payload=payload, start_at=start_at, anchor=anchor,
                      background="transparent")


# ---------------------------------------------------------------- frames from the shot's own window

def test_short_shot_is_sampled_in_its_own_middle_never_in_the_next_shot():
    assert qa.shot_times(_shot(1, 10.0, 11.0)) == [10.5]
    long = qa.shot_times(_shot(2, 0.0, 8.0))
    assert long == [2.0, 4.0, 6.0]


def test_late_overlay_text_gets_a_frame_where_it_is_on_screen():
    s = _shot(1, 0.0, 3.0, ShotType.footage_overlay, overlay=_spec(MotionKind.lower_third, 2.0, title="Nome"))
    times = qa.shot_times(s)
    assert times[0] == 1.5 and any(t >= 2.0 for t in times) and all(0.0 < t < 3.0 for t in times)


def test_sample_frames_grabs_every_shot_at_its_middle(tmp_path):
    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=30:duration=3", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                    "-preset", "ultrafast", str(video)], check=True)
    sl = ShotList(project_id="p", duration=3.0, shots=[_shot(1, 0.0, 0.6), _shot(2, 0.6, 3.0)])
    frames = qa.sample_frames(video, sl, tmp_path / "f")
    assert set(frames) == {"shot_001", "shot_002"} and len(frames["shot_001"]) == 1


# ---------------------------------------------------------------- what is on screen

def test_on_screen_follows_the_pick_not_the_planned_type():
    converted = _shot(1, 0, 4, ShotType.footage, SourceKind.ai_image, "image")
    real = _shot(2, 4, 6, ShotType.image, SourceKind.wikimedia, "image")
    card = _shot(3, 6, 10, ShotType.motion_graphic, SourceKind.motion)
    clip = _shot(4, 10, 20, ShotType.footage_overlay, SourceKind.youtube, "video")
    assert [qa.on_screen(s) for s in (converted, real, card, clip)] == [
        "generated_still", "real_still", "graphic", "real_footage"]
    assert qa.effective_type(converted) == "ai_image" and qa.effective_type(clip) == "footage_overlay"
    pct, secs = qa.screen_time(ShotList(project_id="p", duration=20, shots=[converted, real, card, clip]))
    assert secs == {"real_footage": 10.0, "real_still": 2.0, "generated_still": 4.0, "generated_video": 0.0,
                    "graphic": 4.0}
    assert pct["real_footage"] == 50.0
    mix, _pacing = qa.mix_and_pacing(ShotList(project_id="p", duration=20, shots=[converted, real, card, clip]))
    assert mix["ai_image"] == 25.0 and mix["footage"] == 0.0


# ---------------------------------------------------------------- every piece of on-screen text

def test_spec_texts_reads_every_kind():
    m = _spec(MotionKind.map, title="Bethel to Red Devil", markers=[{"lat": 1, "lon": 2, "label": "Bethel"}],
              labels=[{"lat": 1, "lon": 2, "text": "Kuskokwim"}], routes=[{"label": "Ice road"}], center=[0, 0])
    assert qa.spec_texts(m) == [("title", "Bethel to Red Devil"), ("place", "Bethel"), ("place", "Kuskokwim"),
                                ("place", "Ice road")]
    assert qa.spec_texts(_spec(MotionKind.number_card, number="260", label="miles")) == [("number", "260 miles")]
    assert qa.spec_texts(_spec(MotionKind.highlight_box, label="Town")) == [("label", "Town")]
    assert qa.spec_texts(_spec(MotionKind.bar_chart, title="T", labels=["a", "b"], series=[])) == [
        ("title", "T"), ("label", "a"), ("label", "b")]
    assert qa.spec_texts(_spec(MotionKind.lower_third, title="Nome", subtitle="1925")) == [
        ("title", "Nome"), ("subtitle", "1925")]


def test_text_metric_counts_map_titles_and_cards_and_finds_an_early_map():
    words = _words("we drove north for hours and then reached Red Devil at last", start=0.0, step=1.0)
    early_map = _shot(1, 0.0, 5.0, ShotType.footage_overlay,
                      overlay=_spec(MotionKind.map, title="Red Devil", center=[0, 0]))
    card = _shot(2, 5.0, 9.0, ShotType.motion_graphic, SourceKind.motion,
                 motion=_spec(MotionKind.title_card, title="We drove north for hours"))
    stats = qa.overlay_text_stats(ShotList(project_id="p", duration=9, shots=[early_map, card]), words)
    assert stats["all_items"] == 2 and stats["text_items"] == 0  # neither is a text overlay of the old metric
    assert stats["early"] == 1  # the map shows at 0 s, Red Devil is spoken at 8 s
    assert stats["late"] == 1 and stats["all_verbatim_pct"] == 50.0  # the card repeats a line spoken 5 s before


def test_a_graphic_may_lead_its_word_by_the_card_cue_window_but_a_label_may_not():
    words = _words("it weighed 800 pounds in all", start=0.0, step=0.5)  # 800 at 1.0 s
    number = _shot(1, 0.0, 4.0, ShotType.motion_graphic, SourceKind.motion,
                   motion=_spec(MotionKind.number_card, number="800", label="pounds", anchor="800 pounds"))
    label = _shot(2, 0.0, 4.0, ShotType.footage_overlay,
                  overlay=_spec(MotionKind.lower_third, 0.0, anchor="800 pounds", title="800 pounds"))
    assert not qa.text_item_report(number, words)["early"]
    assert qa.text_item_report(label, words)["early"] is False  # 1.0 s lead is the overlay limit itself
    late_label = label.model_copy(update={"overlay": _spec(MotionKind.lower_third, 0.0, anchor="in all",
                                                           title="800 pounds")})
    assert qa.text_item_report(late_label, words)["early"]  # 'in all' is spoken at 2.5 s


def test_text_that_leaves_before_it_can_be_read_is_short():
    words = _words("the Coast Guard icebreaker Healy arrived", start=0.0, step=0.4)
    s = _shot(1, 0.0, 2.6, ShotType.footage_overlay,
              overlay=_spec(MotionKind.lower_third, 0.6, anchor="Healy", title="USCGC Healy"))
    assert qa.text_item_report(s, words)["short"]  # on screen 2.0 s, below SHORT_S


# ---------------------------------------------------------------- checks the model cannot make

def test_local_checks_find_a_repeated_label_a_filler_subtitle_and_a_repeated_still():
    words = _words("Anchorage grew fast " * 10, step=0.5)
    lt = lambda title, sub="": _spec(MotionKind.lower_third, 0.2, anchor="Anchorage", title=title, subtitle=sub)
    a = _shot(1, 0.0, 4.0, ShotType.footage_overlay, overlay=lt("Anchorage", "United States"))
    b = _shot(2, 4.0, 8.0, ShotType.footage_overlay, overlay=lt("Anchorage", "1950s"))
    still1 = _shot(3, 8.0, 12.0, ShotType.ai_image, SourceKind.ai_image, "image", intent="Ships on frozen sea",
                   prompt="One ship on a frozen sea at dusk")
    still2 = _shot(4, 12.0, 15.0, ShotType.ai_image, SourceKind.ai_image, "image", intent="Ships on frozen sea",
                   prompt="One ship on a frozen sea at dusk")
    found = qa.local_checks(ShotList(project_id="p", duration=15, shots=[a, b, still1, still2]), words, {})
    assert "text_filler" in found["shot_001"] and "text_filler" in found["shot_002"]
    assert "text_repeat" in found["shot_002"] and "text_repeat" not in found.get("shot_001", {})
    assert found["shot_004"]["picture_repeat"] == "same picture as shot_003"


def test_frame_hash_ignores_flat_frames(tmp_path):
    from PIL import Image
    flat = tmp_path / "black.jpg"
    Image.new("RGB", (64, 36)).save(flat)
    assert qa._frame_hash(str(flat)) is None
    assert qa._frame_hash(IMG) == qa._frame_hash(IMG) is not None


# ---------------------------------------------------------------- the fail rule

def test_parse_checks_reads_false_and_reasons_and_never_fails_an_unanswered_check():
    failed, why = qa.parse_checks({"checks": {"matches_words": False, "text_true": "no", "clean": True,
                                              "quality": {"ok": False, "reason": "head cut"}},
                                   "fails": {"matches_words": "shows the next line"}})
    assert failed == ["matches_words", "text_true", "quality"]
    assert why == {"matches_words": "shows the next line", "text_true": "", "quality": "head cut"}
    assert qa.parse_checks({"score": 80}) == ([], {})


def test_judge_rendered_fails_on_a_check_or_a_very_low_score_but_not_on_a_middling_score(make_ctx, monkeypatch):
    ctx = make_ctx()
    ctx.save("narration", Narration(audio_path="x.wav", duration=12, words=_words("dogs pulled the sled " * 6)))
    replies = {"shot_001": {"score": 68, "reason": "fine", "rejects": [], "checks": {"matches_words": True}},
               "shot_002": {"score": 80, "reason": "next line", "rejects": [], "checks": {"matches_words": False},
                            "fails": {"matches_words": "illustrates the next line"}},
               "shot_003": {"score": 20, "reason": "bad", "rejects": [], "checks": {}}}
    prompts: list[str] = []

    def fake(role, task, system, prompt, images, stage, **k):
        prompts.append(prompt)
        return replies[f"shot_00{len(prompts)}"]

    monkeypatch.setattr(ctx.seats, "vision_json", fake)
    monkeypatch.setattr(qa, "JUDGE_WORKERS", 1)
    shots = [_shot(1, 0, 4), _shot(2, 4, 8), _shot(3, 8, 12)]
    out = qa.judge_rendered(ShotList(project_id="p", duration=12, shots=shots), {s.id: [IMG] for s in shots}, ctx)
    assert [out[s.id]["fail"] for s in shots] == [False, True, True]
    assert "matches_words: illustrates the next line" in out["shot_002"]["reason"]
    assert prompts[1].startswith("Narration during this shot (4.0 s): ") and "Just before this shot:" in prompts[1]


# ---------------------------------------------------------------- the report

def test_report_opens_with_five_trustworthy_lines(make_ctx):
    ctx = make_ctx()
    gen = _shot(1, 0, 6, ShotType.footage, SourceKind.ai_image, "image")
    real = _shot(2, 6, 10, ShotType.footage, SourceKind.youtube, "video")
    sl = ShotList(project_id="p", duration=10, shots=[gen, real])
    post = {"shot_001": {"score": 30, "reason": "men carry the serum", "rejects": [], "failed": ["matches_words"],
                         "window": "the team carried it", "fail": True},
            "shot_002": {"score": 85, "reason": "ok", "rejects": [], "failed": [], "window": "", "fail": False}}
    r = qa.make_report(sl, ctx, 70.0, post, 1.0, 1.0, [])
    assert len(r.summary) == 5 and r.summary[0].startswith("Screen time: real 40%, generated 60%")
    assert "1 of 2 (50%)" in r.summary[1] and "picture does not match the words" in r.summary[1]
    assert r.worst == ["00:00 shot_001 (AI still): men carry the serum"]
    md = qa.render_markdown(r, offset=2.0).splitlines()
    assert md[2:7] == qa.summary_lines(r, 2.0) and "00:02 shot_001" in md[4]
    q = {s.shot_id: s for s in r.shots}
    assert q["shot_001"].type == "ai_image" and q["shot_001"].on_screen == "generated_still"
    assert q["shot_001"].failed and "rendered_fail" in q["shot_001"].flags and not q["shot_002"].failed
    assert QAReport.model_validate_json(r.model_dump_json()).screen_time_s["generated_still"] == 6.0


def test_older_score_pairs_still_make_a_report(make_ctx):
    r = qa.make_report(ShotList(project_id="p", duration=4, shots=[_shot(1, 0, 4)]), make_ctx(), 70.0,
                       {"shot_001": (50.0, "weak")}, 1.0, 1.0, [])
    assert r.shots[0].failed and r.shots[0].post_render_score == 50.0


# ---------------------------------------------------------------- calibration labels

def _calibration():
    spec = importlib.util.spec_from_file_location("qa_calibration", REPO_ROOT / "scripts" / "qa_calibration.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_review_labels_read_defects_skip_fine_and_corrected_shots():
    cal = _calibration()
    review = {"reviews": [{"area": "maps (x)", "defects": [
        {"shot": "shot_034 (footage_overlay, map inset 4.1 s)", "problem": "early"},
        {"shot": "shot_038 (image_overlay) OK", "problem": "FINE. right"},
        {"shot": "shot_063 / shot_065", "problem": "Swapped pair."},
        {"shot": "callouts 064, 083", "problem": "Every callout is centred on the frame"},
        {"shot": "shot_005", "problem": "wrong map"}]}],
        "synth": {"notes": "- 00:14 (shot_005): one correction. So it is a loose fit, not a clear mismatch."}}
    defects, style = cal.review_labels(review)
    assert {"shot_034", "shot_063", "shot_065"} <= set(defects) and "shot_038" not in defects
    assert "shot_005" not in defects and style == {"shot_064", "shot_083"}
    res = cal.compare({"shot_034", "shot_001"}, {"shot_034", "shot_063"}, {"shot_001", "shot_034", "shot_063",
                                                                          "shot_002"})
    assert res["recall_pct"] == 50.0 and res["false_alarm_pct"] == 50.0 and res["missed"] == ["shot_063"]
    assert json.dumps(res)
