"""Visual mix semantics: full screen cards versus animated overlays, the card spacing cap, map and chart insets."""
from __future__ import annotations

import random

import numpy as np
import pytest

from test_shots_helpers import brand, doc_script, narrate, profile, project, top10_script
from vidbush.config import REPO_ROOT, load_settings, load_style_profile
from vidbush.judge.qa import animation_share
from vidbush.mg import builders as b
from vidbush.models import MotionKind, Shot, ShotList, ShotType, StyleProfile, VisualIdea
from vidbush.shots.director import build_shotlist, card_exempt_ranges, structural_anchors
from vidbush.shots.pacing import card_too_close, mix_targets, plan_slots, type_shares
from vidbush.shots.rules import space_cards


def _cards(sl: ShotList) -> list[Shot]:
    return [s for s in sl.shots if s.type == ShotType.motion_graphic]


def test_default_split_is_twelve_cards_and_thirty_overlays() -> None:
    prof = StyleProfile(id="x")
    assert prof.visual_mix["full_screen_cards"] == 12 and prof.visual_mix["animated_overlays"] == 30
    mix = prof.shot_mix()
    assert sum(mix.values()) == pytest.approx(100.0, abs=0.05)
    assert mix["motion_graphic"] == 12 and mix["footage"] == 20 and mix["image"] == 7
    # 30 animated overlays decorate footage and stills in the 41 to 17 ratio of the picture mix
    assert mix["footage_overlay"] == pytest.approx(21 + 30 * 41 / 58, abs=0.05)
    assert mix["image_overlay"] == pytest.approx(10 + 30 * 17 / 58, abs=0.05)
    assert prof.animation_target() == {"full_screen_cards": 12.0, "overlay_shots": pytest.approx(61.0)}


def test_shipped_profiles_load_the_split() -> None:
    settings = load_settings(REPO_ROOT)
    doc = load_style_profile(settings, "documentary_default")
    top = load_style_profile(settings, "top10_default")
    assert doc.shot_mix()["motion_graphic"] == 12 and doc.pacing.card_spacing_s == 45
    assert top.top10 and top.shot_mix()["motion_graphic"] == 20  # number cards are full screen by design


def test_legacy_motion_graphic_key_reads_as_full_screen_cards() -> None:
    prof = StyleProfile(id="old", visual_mix={"footage": 50, "motion_graphic": 50})
    assert "motion_graphic" not in prof.visual_mix and prof.visual_mix["full_screen_cards"] == 50
    assert prof.shot_mix() == {"footage": 50.0, "motion_graphic": 50.0}


def test_card_too_close_ignores_exempt_ranges() -> None:
    exempt = [(0.0, 60.0)]
    assert card_too_close(100.0, [70.0], 45.0, exempt)
    assert not card_too_close(100.0, [50.0], 45.0, exempt)  # the other card is in the hook
    assert not card_too_close(30.0, [20.0], 45.0, exempt)  # this card is in the hook
    assert not card_too_close(100.0, [70.0], 0.0, exempt)  # 0 turns the cap off


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_planner_keeps_card_spacing_and_mix(seed: int) -> None:
    script = doc_script(n_scenes=8, beats_per_scene=7)
    nar = narrate(script)
    prof = profile()
    slots = plan_slots(nar.duration + 1.0, prof, random.Random(seed), nar.words)
    cards = [s.start for s in slots if s.type == ShotType.motion_graphic]
    outside = [c for c in cards if c >= prof.pacing.hook_seconds]
    assert all(b2 - a >= prof.pacing.card_spacing_s - 1e-6 for a, b2 in zip(outside, outside[1:])), outside
    shares = type_shares([s.type for s in slots])
    for t, share in mix_targets(prof).items():
        assert abs(shares.get(t.value, 0.0) - share * 100) <= 5.0, (t, shares)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_built_documentary_keeps_one_card_per_45_seconds(seed: int) -> None:
    script = doc_script(n_scenes=8, beats_per_scene=7)
    nar = narrate(script)
    prof = profile()
    sl, _plan = build_shotlist(project(), script, nar, None, prof, brand(), None, random.Random(seed))
    from vidbush.shots.director import beat_spans
    spans = beat_spans(script, nar)
    exempt = card_exempt_ranges(script, spans, prof, structural_anchors(script, spans, prof, False, sl.duration, "x"))
    starts = [s.start for s in _cards(sl)]
    for k, t in enumerate(starts):
        assert not card_too_close(t, starts[:k] + starts[k + 1:], prof.pacing.card_spacing_s, exempt), starts
    anim = animation_share(sl)
    assert anim["full_screen_cards"] <= 17 and anim["overlay_shots"] >= 50  # animation moves over the picture
    assert anim["inset_overlays"] > 0


def test_top10_keeps_every_number_card() -> None:
    script = top10_script()
    nar = narrate(script)
    prof = profile(top10=True)
    sl, _plan = build_shotlist(project(), script, nar, None, prof, brand(), None, random.Random(1))
    numbers = [s for s in _cards(sl) if s.motion and s.motion.kind == MotionKind.number_card]
    assert len(numbers) == 10


def test_space_cards_moves_a_close_map_onto_footage_as_an_inset() -> None:
    prof = profile()
    map_spec = b.route_map_spec([[("Colon", 9.36, -79.9), ("Panama City", 8.97, -79.54)]], title="Across")
    card = b.title_card_spec("Chapter")

    def shot(i: int, start: float, typ: ShotType, motion=None) -> Shot:
        return Shot(id=f"shot_{i:03d}", index=i, beat_id="b1", start=start, end=start + 8, duration=8.0, type=typ,
                    intent="x", motion=motion)

    shots = [shot(1, 100.0, ShotType.motion_graphic, card), shot(2, 108.0, ShotType.footage),
             shot(3, 116.0, ShotType.motion_graphic, map_spec)]
    ideas = [VisualIdea(beat_id="b1", family="footage", intent="Ships in the locks", queries=["ships locks"])]

    def rebind(s: Shot, idea: VisualIdea, idx=None) -> None:
        s.intent, s.motion = idea.intent, None

    demoted = space_cards(shots, prof, [(0.0, 60.0)], ideas, rebind, lambda s: 0, lambda s: None)
    assert demoted == 1 and shots[0].type == ShotType.motion_graphic
    assert shots[2].type == ShotType.footage_overlay and shots[2].overlay.kind == MotionKind.map
    assert shots[2].overlay.background == "transparent" and shots[2].intent == "Ships in the locks"


def test_map_and_chart_overlays_render_as_an_inset_panel() -> None:
    from vidbush.mg.inset import InsetComp
    from vidbush.render.mg_python import build_comp
    spec = b.bar_chart_spec(["A", "B"], [13000, 5200], units="miles", title="Distance", duration=3.0)
    full = build_comp(spec, False)
    assert not isinstance(full, InsetComp)
    comp = build_comp(spec.model_copy(update={"background": "transparent"}), True)
    assert isinstance(comp, InsetComp)
    frame = comp.render(45)
    assert frame.shape == (1080, 1920, 4)
    assert frame[540, 100, 3] == 0  # the picture shows through on the left
    cx, cy = int(comp.x0 + comp.pw / 2), int(comp.y0 + comp.ph / 2)
    assert frame[cy, cx, 3] == 255  # the panel is opaque
    assert np.count_nonzero(frame[..., 3]) < 0.3 * 1920 * 1080


def test_qa_animation_share_counts_overlays_on_top_of_their_shot() -> None:
    def shot(i: int, typ: ShotType, overlay=None) -> Shot:
        return Shot(id=f"s{i}", index=i, beat_id="b", start=i, end=i + 1, duration=1, type=typ, intent="x",
                     overlay=overlay)
    inset = b.route_map_spec([[("Colon", 9.36, -79.9), ("Panama City", 8.97, -79.54)]]).model_copy(
        update={"background": "transparent"})
    sl = ShotList(project_id="p", duration=4, shots=[
        shot(0, ShotType.footage), shot(1, ShotType.footage_overlay, inset),
        shot(2, ShotType.image_overlay, b.lower_third_spec("Colon")), shot(3, ShotType.motion_graphic)])
    assert animation_share(sl) == {"full_screen_cards": 25.0, "overlay_shots": 50.0, "inset_overlays": 25.0}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_map_and_chart_insets_get_room(seed: int) -> None:
    from vidbush.shots.assembler import INSET_GAP_S
    from vidbush.shots.assembler import motion_signature
    script = doc_script(n_scenes=8, beats_per_scene=7)
    sl, _plan = build_shotlist(project(), script, narrate(script), None, profile(), brand(), None,
                               random.Random(seed))
    insets = [s for s in sl.shots if s.overlay is not None and s.overlay.kind.value in ("map", "bar_chart", "line_chart")]
    assert insets
    starts = [s.start for s in insets]
    assert all(b2 - a >= INSET_GAP_S - 1.0 for a, b2 in zip(starts, starts[1:])), starts
    sigs = [motion_signature(s.overlay) for s in insets]
    for i, sig in enumerate(sigs):
        later = [insets[j].start - insets[i].start for j in range(i + 1, len(sigs)) if sigs[j] == sig]
        assert all(d >= 180.0 for d in later)
