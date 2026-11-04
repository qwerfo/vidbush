"""Style learner on the synthetic reference video, plus analysis edge cases."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from vidbush.config import REPO_ROOT, load_settings
from vidbush.learn import analyze
from vidbush.learn.style_learner import (LearnError, build_profile, caption_colour, classify_shots, fetch_source,
                                         learn_style)
from vidbush.models import SHOT_TYPES, StyleProfile

FIXTURE = REPO_ROOT / "fixtures" / "reference_video.mp4"


def _settings(root: Path):
    s = load_settings(REPO_ROOT, use_dotenv=False)
    s.offline = True
    s.output_folder, s.data_folder, s.source_cache_folder = str(root / "out"), str(root / "data"), str(root / "cache")
    shutil.copytree(Path(s.config_folder), root / "config")
    s.config_folder = str(root / "config")
    return s


@pytest.fixture(scope="module")
def learned(tmp_path_factory):
    """Run learn_style once on the fixture (mock mode: no classification)."""
    s = _settings(tmp_path_factory.mktemp("learn"))
    return learn_style(str(FIXTURE), "Reference Test", s), s


def test_fixture_is_small_and_long_enough():
    assert FIXTURE.exists() and FIXTURE.stat().st_size < 6 * 1024 * 1024
    w, h, fps, dur = analyze.probe(FIXTURE)
    assert (w, h, round(fps)) == (1280, 720, 30) and 40 <= dur <= 50


def test_profile_written_and_valid(learned):
    path, s = learned
    assert path == Path(s.config_folder) / "style_profiles" / "reference_test.yaml" and path.exists()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    prof = StyleProfile(**data)
    assert prof.id == "reference_test" and prof.name == "Reference Test"
    from vidbush.config import load_style_profile
    assert load_style_profile(s, "reference_test").letterbox.enabled


def test_letterbox_and_captions_detected(learned):
    prof = StyleProfile(**yaml.safe_load(learned[0].read_text(encoding="utf-8")))
    assert prof.letterbox.enabled and abs(prof.letterbox.ratio - 2.39) < 0.03
    assert prof.captions.enabled
    assert 0.3 <= prof.captions.band_height <= 0.7
    assert 42 <= prof.captions.size <= 60  # burned at 34 px on 720p = 51 px at 1080p
    assert prof.captions.color == "#FFFFFF"


def test_provenance_and_pacing(learned):
    prof = StyleProfile(**yaml.safe_load(learned[0].read_text(encoding="utf-8")))
    p = prof.provenance
    for key in ("source", "date", "shot_count", "duration_s", "mix", "letterbox_detected", "captions_detected"):
        assert key in p
    assert 10 <= p["shot_count"] <= 14 and abs(p["duration_s"] - 45.0) < 1.0
    assert p["mix"] == "unknown" and p["letterbox_detected"] and p["captions_detected"]
    assert abs(p["durations"]["mean"] - 3.75) < 0.3 and 2.0 <= p["durations"]["min"] <= 2.5
    from vidbush.config import load_style_profile
    base = load_style_profile(learned[1], "documentary_default")
    assert prof.visual_mix == base.visual_mix
    assert prof.pacing.footage.mean < base.pacing.footage.mean  # reference cuts faster than the default
    for t in SHOT_TYPES:
        tp = prof.pacing.for_type(t)
        assert 1.5 <= tp.min <= tp.mean <= tp.max


def test_classified_mix(tmp_path):
    shots = [(0.0, 3.0), (3.0, 7.0), (7.0, 9.0), (9.0, 15.0)]
    labels = ["footage", "talking_head", "motion_graphic", "image"]
    lb = analyze.LetterboxResult(detected=False)
    cap = analyze.CaptionResult(detected=False)
    prof = build_profile("C", StyleProfile(id="b"), [3, 4, 2, 6], [s for s, _ in shots], labels, lb, cap,
                         {"mix": "classified"})
    assert prof.visual_mix["footage"] == pytest.approx(50.0, abs=0.1)  # share of shot count, talking head as footage
    assert prof.visual_mix["full_screen_cards"] == pytest.approx(25.0, abs=0.1)  # a classified card is full screen
    assert prof.visual_mix["animated_overlays"] == 0.0 and prof.shot_mix()["motion_graphic"] == pytest.approx(25.0)
    assert not prof.letterbox.enabled and not prof.captions.enabled


def test_classify_shots_uses_cheap_classifier():
    calls = []

    class FakeSeats:
        def vision_json(self, role, task, system, prompt, images, stage, intent=""):
            calls.append((role, task, len(images)))
            return {"type": "Image Overlay" if "a" in images[0] else "spaceship", "score": 80}

    assert classify_shots([["a1.jpg", "a2.jpg"], ["b1.jpg", "b2.jpg"], []], FakeSeats()) == [
        "image_overlay", "unknown", "unknown"]
    assert calls == [("cheap_classifier", "classify", 2), ("cheap_classifier", "classify", 2)]


def test_classify_path_in_learn_style(tmp_path, monkeypatch):
    from vidbush.providers.seats import Seats
    s = _settings(tmp_path)
    monkeypatch.setattr(Seats, "is_real", lambda self, role: role == "cheap_classifier")
    monkeypatch.setattr(Seats, "vision_json", lambda self, *a, **k: {"type": "footage", "score": 90})
    monkeypatch.setattr(analyze, "detect_shots", lambda p, **k: [(0.0, 4.0), (4.0, 7.0), (7.0, 12.0)])
    prof = StyleProfile(**yaml.safe_load(learn_style(str(FIXTURE), "classified", s).read_text(encoding="utf-8")))
    assert prof.provenance["mix"] == "classified" and prof.visual_mix["footage"] == 100.0
    assert prof.pacing.footage.mean == 4.0


def test_letterbox_negative_and_caption_negative():
    rng = np.random.default_rng(1)
    frames = [rng.integers(60, 200, (360, 640, 3), dtype=np.uint8) for _ in range(6)]
    lb = analyze.detect_letterbox(frames)
    assert not lb.detected
    assert not analyze.detect_captions([np.full((360, 640, 3), 90, np.uint8)] * 6, lb).detected


def test_snap_ratio_and_colour():
    assert analyze.snap_ratio(2.37) == 2.39 and analyze.snap_ratio(1.6) == 1.6
    assert caption_colour("#FAFBF9") == "#FFFFFF" and caption_colour("#ffd84a") == "#FFD84A"


def test_source_errors(tmp_path):
    s = _settings(tmp_path)
    with pytest.raises(LearnError, match="not found"):
        fetch_source(str(tmp_path / "missing.mp4"), s)
    with pytest.raises(LearnError, match="offline"):
        fetch_source("https://www.youtube.com/watch?v=abc", s)


def test_download_error_is_clear(tmp_path, monkeypatch):
    import yt_dlp
    s = _settings(tmp_path)
    s.offline = False

    class Boom:
        def __init__(self, opts):
            assert "720" in opts["format"]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            raise yt_dlp.utils.DownloadError("network unreachable")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", Boom)
    with pytest.raises(LearnError, match="could not download"):
        fetch_source("https://example.com/v.mp4", s)


def test_mix_is_count_share_with_screen_time_secondary():
    from vidbush.learn.style_learner import screen_time_mix, visual_mix
    labels = ["footage", "motion_graphic", "footage", "motion_graphic"]
    durs = [3.0, 9.0, 3.0, 9.0]
    assert visual_mix(durs, labels, {})["full_screen_cards"] == pytest.approx(50.0)
    assert screen_time_mix(durs, labels)["motion_graphic"] == pytest.approx(75.0)
