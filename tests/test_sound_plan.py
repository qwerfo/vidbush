"""Planned sound: music per scene, quiet, ambience, effects, the mixer's curves and the sound QA."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from vidbush.audio import mixer, music, sound_plan as sp, sound_qa
from vidbush.audio.loudness import SR, read_audio, write_audio
from vidbush.models import Beat, Brand, Narration, Scene, Script, Shot, ShotList, StyleProfile, Word
from vidbush.util import NullLogger


def _shot(i: int, start: float, end: float, intent: str, kind: str = "footage", **kw: Any) -> Shot:
    return Shot(id=f"shot_{i:03d}", beat_id="b1", start=start, end=end, duration=end - start, type=kind,
                intent=intent, **kw)


def _tracks() -> list[dict[str, Any]]:
    return [{"file": "calm.ogg", "mood": "calm", "energy": 0.3, "phrase_s": 8.0, "bpm_confidence": 0.9,
             "first_downbeat_s": 0.0},
            {"file": "tense.ogg", "mood": "tense", "energy": 0.6, "phrase_s": 8.0, "bpm_confidence": 0.9,
             "first_downbeat_s": 0.0},
            {"file": "bright.ogg", "mood": "uplifting", "energy": 0.65, "phrase_s": 0.0, "bpm_confidence": 0.0,
             "first_downbeat_s": 0.0}]


def test_mood_from_stated_purpose_and_turn() -> None:
    assert sp.mood_of("Make the viewer feel the desperation of Nome in 1925")[0] == "tense"
    assert sp.mood_of("Pose the viewer a guess about the answer")[0] == "mysterious"
    assert sp.mood_of("", "calm")[0] == "calm"
    assert sp.turn_text("Music drops for the held breath, then lifts into arrival.") == "arrival"
    assert sp.turn_text("Music drops before the answer lands") == "the answer lands"
    assert sp.turn_text("A plain note") == ""


def test_place_of_reads_only_what_is_shown() -> None:
    assert sp.place_of("Close view of the Talkeetna railroad depot platform")[0] == "interior_public"  # not road
    assert sp.place_of("Wide shot of a dog team pulling a sled across sea ice")[0] == "snow_wind"
    assert sp.place_of("Archival black and white photo of a 1960s Bethel main street")[0] is None  # no modern bed
    assert sp.place_of("Close view of a 49-star American flag on a pole")[0] is None
    assert sp.place_of("Close view of an old pull-down classroom wall map")[0] is None
    assert sp.place_of("Wide aerial over lake-studded tundra")[0] == "open_ground"


def test_ambience_holds_across_one_place_and_skips_short_places() -> None:
    shots = [_shot(1, 0, 5, "dog team on sea ice"), _shot(2, 5, 8, "map", "motion_graphic"),
             _shot(3, 8, 14, "a sled crossing sea ice at dusk"), _shot(4, 14, 17, "a river"),
             _shot(5, 17, 30, "a calm harbor with boats")]
    beds = sp.plan_ambience(ShotList(project_id="t", duration=30, shots=shots), [])
    assert [(b.place, b.start, b.end) for b in beds] == [("snow_wind", 0.0, 14.0), ("coast", 17.0, 30.0)]
    assert beds[0].shots == ["shot_001", "shot_003"]  # held through the full frame map


def test_quiet_fades_before_the_clause_and_returns_after_it_lands() -> None:
    q = sp.QuietSpan(10.0, 12.0, "s1")
    pts = sp.quiet_gain_points([q], 30.0)
    assert (10.0 - sp.QUIET_FADE_S, 1.0) in pts and (10.0, 0.0) in pts
    assert (12.0 + sp.LAND_S, 0.0) in pts and (12.0 + sp.LAND_S + sp.RETURN_FADE_S, 1.0) in pts


def test_music_holds_one_level_under_speech() -> None:
    nar = Narration(audio_path="x", duration=40, words=[Word(text="w", start=s, end=s + 2) for s in (3, 6, 9, 25)])
    spans = sp.speech_spans(nar)
    assert spans == [(3, 11), (25, 27)]  # 1 s pauses merge; the 14 s pause does not
    pts = sp.music_gain_points(spans, 40.0, -18.0, -10.0)
    under = [t for t, v in pts if v == -18.0]
    assert min(under) <= 3.0 and not any(3.0 < t < 11.0 for t, v in pts if v != -18.0)
    assert sound_qa.shortest_ramp(pts) >= 1.0


def _scenes() -> list[sp.SceneInfo]:
    return [sp.SceneInfo("s1", 0.0, 40.0, "everyday life at home", "setup", "calm", 0.3),
            sp.SceneInfo("s2", 41.0, 90.0, "the desperation of a deadly storm", "turn", "tense", 0.6),
            sp.SceneInfo("s3", 91.0, 140.0, "a deadly race", "act", "tense", 0.6)]


def test_music_changes_at_scene_moods_on_a_phrase() -> None:
    sections, changes = sp.plan_music(_scenes(), [], _tracks(), 150.0)
    assert [s.track for s in sections] == ["calm.ogg", "tense.ogg"]  # s3 keeps the tense music
    c = changes[0]
    assert c.kind == "crossfade" and c.xfade_s == sp.XFADE_BY_MOOD["tense"]
    assert c.snapped and abs((c.t % 8.0)) < 1e-6  # on the calm track's 8 s phrase grid


def test_scene_change_inside_a_quiet_moves_to_its_return_and_a_turn_changes_inside_a_scene() -> None:
    quiets = [sp.QuietSpan(39.0, 41.5, "s1"),
              sp.QuietSpan(70.0, 72.0, "s2", note="Music drops for the held breath, then lifts into arrival")]
    sections, changes = sp.plan_music(_scenes(), quiets, _tracks(), 150.0)
    assert changes[0].kind == "quiet_return" and changes[0].t == round(41.5 + sp.LAND_S, 3)
    assert changes[1].kind == "quiet_return" and sections[2].mood == "uplifting"
    assert all(c.xfade_s == 0.0 for c in changes[:2])


def test_effects_follow_graphics_and_stay_off_in_quiet_moments() -> None:
    from vidbush.models import MotionSpec
    callout = lambda at: MotionSpec(kind="callout", payload={"title": "A"}, start_at=at, duration=2.0)  # noqa: E731
    shots = [_shot(1, 0, 5, "x", "footage_overlay", overlay=callout(1.0)),
             _shot(2, 3, 6, "x", "footage_overlay", overlay=callout(0.1)),
             _shot(3, 20, 25, "x", "footage_overlay", overlay=callout(1.0))]
    nar = Narration(audio_path="x", duration=30, words=[])
    events, skipped = sp.plan_effects(ShotList(project_id="t", duration=30, shots=shots), nar,
                                      [sp.QuietSpan(21.0, 23.0, "s1")])
    assert [(e.t, e.kind) for e in events] == [(1.3, "card_settle")]
    assert {s["why"] for s in skipped} == {"its moment is quiet", f"within {sp.EFFECT_SPACING_S} s of the previous effect"}


def test_fade_curve_moves_at_an_even_db_rate() -> None:
    n = int(4 * SR)
    g = mixer.curve([(0.0, 1.0), (1.0, 1.0), (2.5, 0.0)], n, fade=True)
    db = 20 * np.log10(np.maximum(g[::int(0.2 * SR)], 1e-9))
    steps = np.abs(np.diff(db[(db > mixer.FADE_FLOOR_DB) & (db < 0)]))
    assert steps.max() < 7.0 and g[-1] == 0.0


def test_leveller_holds_a_bed_with_swells() -> None:
    t = np.arange(int(20 * SR)) / SR
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(len(t)).astype(np.float32) * 0.01
    swell = np.where((t % 5) < 1.0, 6.0, 1.0).astype(np.float32)  # a 15 dB wave every 5 s
    out = mixer.levelled(np.stack([noise * swell] * 2, 1))
    db = sound_qa.window_db(out[:, 0], int(0.4 * SR))
    assert np.percentile(db, 99) - np.median(db) < 8.0


def test_qa_finds_a_gap_and_a_cut() -> None:
    x = np.full(int(6 * SR), 0.05, dtype=np.float32)
    x[int(2 * SR): int(2.5 * SR)] = 0.0
    assert sound_qa.gaps(x) == [(2.0, 0.5)]
    y = np.full(int(6 * SR), 0.05, dtype=np.float32)
    y[int(3 * SR):] *= 0.01  # a hard 40 dB drop
    assert sound_qa.cuts(y) and sound_qa.cuts(np.full(int(6 * SR), 0.05, dtype=np.float32)) == []


def _speech(path: Path, spans: list[tuple[float, float]], total: float) -> Path:
    t = np.arange(int(total * SR)) / SR
    sig = np.zeros_like(t)
    for s, e in spans:
        m = (t >= s) & (t < e)
        sig[m] = 0.3 * np.sin(2 * np.pi * 200 * t[m]) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t[m]))
    return write_audio(path, sig.astype(np.float32))


def test_planned_mix_writes_plan_stems_and_exact_length(tmp_path: Path, settings: Any) -> None:
    rng = np.random.default_rng(0)
    folder = tmp_path / "music"
    folder.mkdir()
    for name in ("calm.wav", "tense.wav"):
        write_audio(folder / name, (rng.standard_normal((int(30 * SR), 2)) * 0.05).astype(np.float32))
    (folder / "tracks.json").write_text(json.dumps({"calm.wav": {"mood": "calm", "checked": ["mood"]},
                                                    "tense.wav": {"mood": "tense", "checked": ["mood"]}}),
                                        encoding="utf-8")
    total = 30.0
    spans = [(0.5, 9.0), (10.0, 14.0), (17.0, 29.0)]
    audio = _speech(tmp_path / "narration.wav", spans, total)
    words = [Word(text="w", start=s, end=e, beat_id="b1" if s < 15 else "b2") for s, e in spans]
    nar = Narration(audio_path=str(audio), duration=total, words=words, beat_spans={"b1": [0.0, 14.0],
                                                                                    "b2": [15.0, total]})
    script = Script(scenes=[Scene(id="s1", title="a", purpose="setup", beats=[Beat(id="b1", text="w")]),
                            Scene(id="s2", title="b", purpose="turn", beats=[Beat(id="b2", text="w")])])
    shots = [_shot(1, 0, 15, "a dog team on sea ice"), _shot(2, 15, 30, "a sled on sea ice", flags=["quiet"])]
    sl = ShotList(project_id="t", duration=total, shots=shots)
    work = tmp_path / "work"
    out = music.build_mix(nar, script, sl, Brand(id="b", music_folder=str(folder)), StyleProfile(id="p"), settings,
                          work, NullLogger(), random.Random(1))
    assert len(read_audio(out, SR, 2)) == int(total * SR)
    plan = json.loads((work / "sound_plan.json").read_text(encoding="utf-8"))
    assert plan["music"] and plan["quiets"] and "qa" in plan
    assert plan["qa"]["longest_gap_s"] == 0.0
    assert abs(plan["qa"]["lufs"] - settings.render.target_lufs) < 1.0
    assert all((work / "stems" / f"{k}.wav").exists() for k in ("narration", "music", "ambience", "effects"))
    assert json.loads((work / "mix.json").read_text(encoding="utf-8"))["tracks"]
    assert "## Sound" in sound_qa.qa_section(work)
