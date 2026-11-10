"""Director: full shot lists from synthetic scripts, with and without the mock director seat."""
from __future__ import annotations

import json
import random
from collections import Counter

import pytest

from test_shots_helpers import brand, doc_script, narrate, profile, project, research, six_minute_script, top10_script
from vidbush.config import REPO_ROOT
from vidbush.mg import geo
from vidbush.models import (OVERLAY_TYPES, STATIC_CARD_KINDS, ChartPayload, MapPayload, MotionKind, Research, Script,
                            ShotType, VideoFormat)
from vidbush.providers.base import CostTracker
from vidbush.providers.seats import Seats
from vidbush.shots import director
from vidbush.shots.director import build_shotlist, validate_idea
from vidbush.shots.pacing import bounds, mix_targets, type_shares

FIX = REPO_ROOT / "fixtures" / "llm" / "the_panama_canal"


def _build(script: Script, seed: int = 1, seats=None, res=None, prof=None):
    nar = narrate(script)
    sl, plan = build_shotlist(project(), script, nar, res, prof or profile(), brand(), seats, random.Random(seed))
    return sl, plan, nar


def _check_timeline(sl, nar) -> None:
    assert sl.duration == pytest.approx(nar.duration + 1.0)
    assert sl.shots[0].start == 0.0
    assert sl.shots[-1].end == pytest.approx(sl.duration)
    for i, (a, b) in enumerate(zip(sl.shots, sl.shots[1:])):
        assert a.end == pytest.approx(b.start, abs=1e-3)
        assert b.index == i + 1
    assert [s.id for s in sl.shots] == [f"shot_{i + 1:03d}" for i in range(len(sl.shots))]
    assert all(s.duration >= 1.5 for s in sl.shots)


def _check_specs(sl, palette: str) -> None:
    for s in sl.shots:
        assert s.intent
        if s.type == ShotType.motion_graphic:
            assert s.motion is not None and s.overlay is None
            s.motion.typed_payload()
            assert s.motion.duration == pytest.approx(s.duration, abs=1e-3)
            assert s.motion.background != "transparent"
            assert s.motion.style.palette == palette
            if s.motion.kind in (MotionKind.map, MotionKind.bar_chart, MotionKind.line_chart):
                assert s.motion.source_note
        elif s.type in OVERLAY_TYPES:
            assert s.overlay is not None and s.motion is None
            s.overlay.typed_payload()
            assert s.overlay.background == "transparent"
            assert s.overlay.duration == pytest.approx(s.duration, abs=1e-3)
        else:
            assert s.overlay is None and s.motion is None
            assert s.queries and s.ai_prompt


def test_heuristic_documentary_shotlist() -> None:
    script = six_minute_script()
    sl, plan, nar = _build(script, res=research())
    _check_timeline(sl, nar)
    _check_specs(sl, "documentary_warm")
    shares = type_shares([s.type for s in sl.shots])
    for t, share in mix_targets(profile()).items():
        assert abs(shares.get(t.value, 0) - share * 100) <= 5.0
    assert sl.shots[0].type in (ShotType.footage, ShotType.footage_overlay)
    families = Counter((i.beat_id, i.family) for i in plan.ideas)
    for beat in script.beats():
        for fam in ("footage", "still", "motion"):
            assert families[(beat.id, fam)] >= 1


def test_documentary_title_card_at_setup_start() -> None:
    script = doc_script(n_scenes=4, beats_per_scene=6)
    sl, _plan, nar = _build(script)
    setup_start = nar.beat_spans[script.scenes[1].beats[0].id][0]
    cards = [s for s in sl.shots if s.motion and s.motion.kind == MotionKind.title_card
             and abs(s.start - setup_start) < 0.01]
    assert cards and cards[0].motion.payload["title"] == "The Panama Canal"
    assert cards[0].transition_in == "fade"
    assert sl.shots[0].type != ShotType.motion_graphic


def test_no_consecutive_same_idea() -> None:
    sl, _plan, _nar = _build(six_minute_script(), seed=4)
    for a, b in zip(sl.shots, sl.shots[1:]):
        if a.type != ShotType.motion_graphic and b.type != ShotType.motion_graphic:
            assert not (a.intent == b.intent and a.framing == b.framing)


def test_motion_graphics_respect_min() -> None:
    prof = profile()
    sl, _plan, _nar = _build(six_minute_script(), seed=2, prof=prof)
    mg_min = prof.pacing.motion_graphic.min
    short = [s for s in sl.shots if s.type == ShotType.motion_graphic and s.duration < mg_min - 1e-6
             and s.motion is not None and s.motion.kind not in STATIC_CARD_KINDS]  # text and number cards are brief
    assert len(short) <= 1
    for s in sl.shots:
        if s.type != ShotType.motion_graphic:
            assert s.duration <= bounds(prof, s.type, s.start)[1] * 1.25 + 1e-6


def test_seed_determinism_and_variation() -> None:
    script = six_minute_script()
    a, _, _ = _build(script, seed=9)
    b, _, _ = _build(script, seed=9)
    c, _, _ = _build(script, seed=10)
    assert a.model_dump() == b.model_dump()
    assert [s.duration for s in a.shots] != [s.duration for s in c.shots]


def test_top10_structure() -> None:
    script = top10_script()
    prof = profile(top10=True, palette="top10_bold")
    sl, _plan, nar = _build(script, prof=prof)
    _check_timeline(sl, nar)
    _check_specs(sl, "top10_bold")
    items = [sc for sc in script.scenes if sc.item_number is not None]
    titles = [director.item_title(sc.title) for sc in items]
    for sc, title in zip(items, titles):
        start = nar.beat_spans[sc.beats[0].id][0]
        k = next(i for i, s in enumerate(sl.shots) if abs(s.start - start) < 0.01)
        card = sl.shots[k]
        assert card.type == ShotType.motion_graphic and card.motion.kind == MotionKind.number_card
        assert card.motion.payload["number"] == f"#{sc.item_number}"
        assert card.motion.payload["label"] == title
        nxt = sl.shots[k + 1]
        assert nxt.type == ShotType.footage_overlay
        assert nxt.overlay.kind == MotionKind.lower_third and nxt.overlay.payload["title"] == title
    recap = [s for s in sl.shots if s.motion and s.motion.kind == MotionKind.recap_grid]
    assert len(recap) == 1 and recap[0].motion.payload["items"] == titles
    assert recap[0].scene_id == "s99"


def test_top10_from_script_format_without_profile_flag() -> None:
    script = top10_script(items=5)
    assert script.format == VideoFormat.top10
    sl, _plan, _nar = _build(script)
    cards = [s for s in sl.shots if s.motion and s.motion.kind == MotionKind.number_card
             and str(s.motion.payload["number"]).startswith("#")]
    assert len(cards) == 5


# ---------------------------------------------------------------- with the mock director seat

@pytest.fixture
def panama(settings):
    script = Script(**json.loads((FIX / "script.json").read_text()))
    res = Research(**json.loads((FIX / "research.json").read_text()))
    seats = Seats(settings, CostTracker("p1"), fixture_key="the_panama_canal", force_mock=True)
    return script, res, seats


def test_fixture_plan_maps_and_charts_valid(panama) -> None:
    script, res, seats = panama
    sl, plan, nar = _build(script, seats=seats, res=res)
    _check_timeline(sl, nar)
    _check_specs(sl, "documentary_warm")
    assert len(plan.ideas) > 126
    for s in sl.shots:
        if s.motion is None:
            continue
        p = s.motion.typed_payload()
        if isinstance(p, MapPayload):
            assert -90 <= p.center[0] <= 90 and -180 <= p.center[1] <= 180 and 1 <= p.zoom <= 12
            for m in p.markers:
                hit = geo.lookup(m.label) if m.label else None
                if hit:
                    assert geo.distance_deg((m.lat, m.lon), (hit.lat, hit.lon)) <= 1.0
                assert 0 <= m.appear_at <= s.duration
            for r in p.routes:
                assert 0 <= r.reveal_start <= r.reveal_end <= s.duration
        if isinstance(p, ChartPayload):
            for v in p.series[0].values:
                assert director.I.number_is_cited(v, res.facts)


def test_validate_idea_drops_and_fixes(panama) -> None:
    script, res, _seats = panama
    beats = {b.id for b in script.beats()}
    good = {"beat_id": "b001", "family": "footage", "intent": "Ships in a storm", "queries": ["storm ship"]}
    assert validate_idea(good, beats, res) is not None
    assert validate_idea({**good, "family": "drone"}, beats, res) is None
    assert validate_idea({**good, "beat_id": "nope"}, beats, res) is None
    assert validate_idea({"beat_id": "b001", "family": "motion", "intent": "x"}, beats, res) is None
    chart = {"beat_id": "b001", "family": "motion", "intent": "chart", "motion": {
        "kind": "bar_chart", "duration": 8, "payload": {"labels": ["a", "b"], "series": [{"values": [123457, 99]}]}}}
    assert validate_idea(chart, beats, res) is None
    cited = json.loads(json.dumps(chart))
    cited["motion"]["payload"]["series"][0]["values"] = [13000, 5200]
    ok = validate_idea(cited, beats, res)
    assert ok is not None and ok.motion.source_note
    bad_map = {"beat_id": "b001", "family": "motion", "intent": "map", "motion": {
        "kind": "map", "duration": 8, "payload": {"center": [0, 0], "zoom": 3, "markers": [
            {"lat": 0.0, "lon": 0.0, "label": "Colon"}, {"lat": 8.97, "lon": -79.53, "label": "Panama City"}],
            "routes": [{"points": [[0.0, 0.0], [8.97, -79.53]]}]}}}
    fixed = validate_idea(bad_map, beats, res).motion.typed_payload()
    assert fixed.markers[0].lat == pytest.approx(9.359, abs=0.05)
    assert fixed.routes[0].points[0][0] == pytest.approx(9.359, abs=0.05)
    assert 8 < fixed.center[0] < 10
    overlay_bad = {**good, "overlay": {"kind": "map", "duration": 5, "payload": {"center": [0, 0]}}}
    assert validate_idea(overlay_bad, beats, res).overlay is None


def test_seat_failure_falls_back_to_heuristics() -> None:
    class Broken:
        log = None

        def llm_json(self, *a, **k):
            raise RuntimeError("down")

    sl, plan, _nar = _build(doc_script(3, 4), seats=Broken())
    assert sl.shots and plan.ideas
