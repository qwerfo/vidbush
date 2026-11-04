"""Shot list review fixes: per scene director calls, beat-local motion graphics, card runs, overlays, recap."""
from __future__ import annotations

import json
import random
import threading

import pytest

from test_shots_helpers import brand, doc_script, narrate, profile, project, six_minute_script, top10_script
from vidbush.models import Beat, MotionKind, Scene, ScenePurpose, Script, ShotType, VisualIdea
from vidbush.providers.base import ProviderError
from vidbush.shots import ideas as I
from vidbush.shots import seat
from vidbush.shots.assembler import ai_prompt, motion_signature, prompt_subject
from vidbush.shots.director import build_shotlist, structural_anchors
from vidbush.shots.pacing import MAX_MG_RUN, MAX_MG_RUN_S, Anchor, plan_slots, type_shares
from vidbush.mg import builders as b


def _build(script: Script, seed: int = 1, seats=None, prof=None):
    nar = narrate(script)
    sl, plan = build_shotlist(project(), script, nar, None, prof or profile(), brand(), seats, random.Random(seed))
    return sl, plan, nar


def _bc(text: str, title: str = "Chapter", purpose: ScenePurpose = ScenePurpose.act, era: int = 1931):
    beat = Beat(id="b1", text=text)
    scene = Scene(id="s1", title=title, purpose=purpose, beats=[beat])
    return I.BeatContext(beat=beat, scene=scene, era=era, topic="Hoover Dam")


# ---------------------------------------------------------------- director seat

class FakeSeat:
    """llm_json stand in: records the beats each call asked about and answers for them."""

    log = None

    def __init__(self, fail: Exception | None = None) -> None:
        self.calls: list[list[str]] = []
        self.timeouts: list[float] = []
        self.fail = fail
        self.lock = threading.Lock()

    def llm_json(self, role, task, system, user, stage, temperature=0.6, timeout=180.0):
        brief = json.loads(user)
        ids = [bt["id"] for bt in brief["beats"]]
        with self.lock:
            self.calls.append(ids)
            self.timeouts.append(timeout)
        if self.fail is not None:
            raise self.fail
        out = []
        for bid in ids + ["b999"]:
            out.append({"beat_id": bid, "family": "footage", "intent": f"Ships for {bid}", "queries": ["ships at sea"],
                        "overlay": {"kind": "lower_third", "payload": {"title": "Colon"}}})
            out.append({"beat_id": bid, "family": "motion", "intent": "Card",
                        "motion": {"kind": "number_card", "payload": {"number": "1914", "label": "opened"}}})
        return {"ideas": out}


def test_director_called_per_scene_with_schema_and_long_timeout() -> None:
    script = doc_script(n_scenes=3, beats_per_scene=10)
    fake = FakeSeat()
    nar = narrate(script)
    spans = {k: tuple(v) for k, v in nar.beat_spans.items()}
    ideas = seat.director_ideas(script, spans, None, profile(), brand(), fake, _Log())
    assert len(fake.calls) == 6 and all(len(c) <= seat.CHUNK_BEATS for c in fake.calls)
    assert sorted(b for c in fake.calls for b in c) == sorted(bt.id for bt in script.beats())
    assert set(fake.timeouts) == {seat.SEAT_TIMEOUT_S}
    assert {i.beat_id for i in ideas} == {bt.id for bt in script.beats()}  # b999 ignored
    foot = [i for i in ideas if i.family == "footage"]
    assert all(i.overlay is not None and i.overlay.duration > 0 for i in foot)  # duration injected
    assert all(i.motion is not None for i in ideas if i.family == "motion")
    brief = json.loads(seat.director_prompt(script, spans, None, profile(), brand(), seat.chunk_beats(script)[0]))
    kinds = brief["motion_spec_schema"]["kinds"]
    assert {"map", "bar_chart", "number_card", "lower_third"} <= set(kinds) and "markers" in kinds["map"]
    assert "exactly 1 footage idea" in " ".join(brief["rules"])


class _Log:
    def info(self, *a, **k) -> None:
        pass

    def warn(self, *a, **k) -> None:
        pass


def test_bad_overlay_only_clears_overlay() -> None:
    item = {"beat_id": "b001", "family": "footage", "intent": "Ships", "queries": ["ships"],
            "overlay": {"kind": "map", "payload": {"center": [0, 0]}}}
    idea = seat.validate_idea(item, {"b001"}, None, {"b001": 5.0})
    assert idea is not None and idea.overlay is None
    broken = {**item, "overlay": {"kind": "lower_third", "payload": {"subtitle": "no title"}}}
    assert seat.validate_idea(broken, {"b001"}, None).overlay is None


def test_every_chunk_provider_error_raises_but_other_errors_fall_back() -> None:
    script = doc_script(n_scenes=2, beats_per_scene=3)
    spans = {k: tuple(v) for k, v in narrate(script).beat_spans.items()}
    with pytest.raises(ProviderError):
        seat.director_ideas(script, spans, None, profile(), brand(), FakeSeat(ProviderError("down")), _Log())
    assert seat.director_ideas(script, spans, None, profile(), brand(), FakeSeat(RuntimeError("x")), _Log()) == []


# ---------------------------------------------------------------- motion graphics stay on their beat

def _leak_script() -> Script:
    beats = [Beat(id="b001", text="Workers dug through the jungle day and night with shovels and picks."),
             Beat(id="b002", text="The crews kept digging while the rain fell on the camps and the mud rose."),
             Beat(id="b003", text="Ships sailed from New York to San Francisco around the tip of South America."),
             Beat(id="b004", text="The trains carried rock and dirt away from the cut every single hour.")]
    return Script(title="Test", scenes=[Scene(id="s1", title="Digging", purpose=ScenePurpose.act, beats=beats)])


def test_motion_slot_never_takes_the_next_beats_map() -> None:
    script = _leak_script()
    nar = narrate(script)
    for seed in range(6):
        sl, plan = build_shotlist(project(), script, nar, None, profile(), brand(), None, random.Random(seed))
        b3 = nar.beat_spans["b003"]
        for s in sl.shots:
            if s.type == ShotType.motion_graphic and s.motion and s.motion.kind == MotionKind.map:
                shared = max(0.0, min(s.end, b3[1]) - max(s.start, b3[0]))
                assert shared >= min(2.0, 0.3 * s.duration), (seed, s.start, s.end, b3)


# ---------------------------------------------------------------- full screen card runs

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 5, 8])
def test_card_runs_are_short(seed: int) -> None:
    for script, prof in ((six_minute_script(), profile()), (top10_script(), profile(top10=True))):
        sl, _p, _n = _build(script, seed=seed, prof=prof)
        run_n, run_s = 0, 0.0
        for s in sl.shots:
            if s.type == ShotType.motion_graphic:
                run_n, run_s = run_n + 1, run_s + s.duration
                assert run_n <= MAX_MG_RUN and run_s <= MAX_MG_RUN_S + 0.5, (seed, s.id)
            else:
                run_n, run_s = 0, 0.0
        mg = profile().pacing.motion_graphic
        for s in sl.shots:
            if s.motion and s.motion.kind in (MotionKind.title_card, MotionKind.text_overlay, MotionKind.callout):
                assert s.duration <= mg.min + 1.5


def test_planner_mix_still_within_five_points_by_count() -> None:
    nar = narrate(six_minute_script())
    for seed in (0, 4, 9):
        slots = plan_slots(nar.duration + 1.0, profile(), random.Random(seed), nar.words)
        shares = type_shares([s.type for s in slots])
        for t, target in profile().shot_mix().items():
            assert abs(shares.get(t, 0) - target) <= 5, (seed, t, shares)


# ---------------------------------------------------------------- overlays

def test_no_overlay_text_repeats_within_30_seconds() -> None:
    for seed in (0, 1, 2):
        sl, _p, _n = _build(six_minute_script(), seed=seed)
        seen: dict[str, float] = {}
        cards = [(s.motion.payload.get("title", "").lower(), s.start, s.end) for s in sl.shots
                 if s.motion and s.motion.kind in (MotionKind.text_overlay, MotionKind.callout, MotionKind.title_card)]
        for s in sl.shots:
            if s.overlay is None:
                continue
            key = (s.overlay.payload.get("title") or "").lower()
            assert not (key in seen and s.start - seen[key] < 30.0), (seed, s.id, key)
            assert not any(k == key and a - s.end < 30 and s.start - e < 30 for k, a, e in cards), (seed, s.id)
            seen[key] = s.end


def test_cta_beat_has_no_overlay_map_or_number_card() -> None:
    bc = _bc("Subscribe for more of the history that built America, from New York to California.")
    made = I.ideas_for_beat(bc, None, "documentary_warm", random.Random(1))
    assert all(i.overlay is None for i in made)
    assert not any(i.motion and i.motion.kind in (MotionKind.map, MotionKind.number_card) for i in made)
    assert {i.family for i in made} == {"footage", "still", "motion"}
    assert I.overlay_choices(bc, "documentary_warm", ["New York"]) == []


def test_place_lower_thirds_only_name_points_the_idea_shows() -> None:
    bc = _bc("Workers from New York and all over the United States poured into Europe and Boulder City.")
    made = I.ideas_for_beat(bc, None, "documentary_warm", random.Random(1))
    for idea in made:
        ov = idea.overlay
        if ov is not None and ov.kind == MotionKind.lower_third:
            assert ov.payload["title"] in idea.must_show
            assert ov.payload["title"] not in ("United States", "Europe")
    overlays = [id(i.overlay) for i in made if i.family == "footage" and i.overlay is not None]
    assert len(overlays) == len(set(overlays))  # one overlay per footage idea


@pytest.mark.parametrize("text,name,role", [
    ("Frank Crowe, a blunt, tireless engineer, ran the job.", "Frank Crowe", "Engineer"),
    ("Ferdinand de Lesseps, the celebrated builder of the Suez Canal, arrived.", "Ferdinand de Lesseps",
     "Builder of the Suez Canal"),
    ("President Theodore Roosevelt rode a steam shovel.", "Theodore Roosevelt", "President"),
    ("Frank Crowe arrived in 1931.", "Frank Crowe", ""),
])
def test_person_role(text: str, name: str, role: str) -> None:
    assert I.person_role(name, text) == role


# ---------------------------------------------------------------- maps and routes

@pytest.mark.parametrize("text,route", [
    ("Ships sailing from New York to San Francisco once had to round Cape Horn.", True),
    ("The railroad ran between Boulder City and Las Vegas.", True),
    ("Enough concrete to pave a highway from San Francisco to New York.", False),
    ("Power went to homes in Nevada, Arizona and California.", False),
    ("Visitors from Chicago to Boston admired the dam.", False),
])
def test_routes_only_for_real_journeys(text: str, route: bool) -> None:
    assert (I.travel_route(text) is not None) == route


def test_states_are_labelled_not_routed() -> None:
    bc = _bc("Sending electricity to homes in Nevada, Arizona and California.")
    idea = I.map_idea(bc, I.beat_entities(bc), "documentary_warm")
    p = idea.motion.typed_payload()
    assert p.routes == [] and p.markers == [] and {"NEVADA", "ARIZONA", "CALIFORNIA"} <= {lb.text for lb in p.labels}


def test_comparison_places_make_no_map() -> None:
    bc = _bc("The dam held enough concrete to pave a highway from San Francisco to New York.")
    assert I.map_idea(bc, I.beat_entities(bc), "documentary_warm") is None


def test_map_signature_ignores_wording() -> None:
    a = b.map_spec([("Colon", 9.36, -79.9), ("Panama City", 8.97, -79.54)], route=True, title="A")
    c = b.map_spec([("Panama City", 8.97, -79.54), ("Colon", 9.36, -79.9)], route=True, title="B")
    assert motion_signature(a) == motion_signature(c)
    assert motion_signature(a) != motion_signature(b.map_spec([("Las Vegas", 36.2, -115.2)]))


def test_repeated_map_content_is_penalised() -> None:
    beats = [Beat(id=f"b{i:03d}", text="Ships crossed the Panama Canal near Colon and Panama City every day.")
             for i in range(1, 25)]
    script = Script(title="Canal", scenes=[Scene(id="s1", title="Canal", purpose=ScenePurpose.act, beats=beats)])
    sl, _p, _n = _build(script)
    motion = [s for s in sl.shots if s.type == ShotType.motion_graphic]
    maps = [s for s in motion if s.motion.kind == MotionKind.map]
    assert len(maps) <= max(2, round(0.5 * len(motion)))


# ---------------------------------------------------------------- number cards and phrases

def test_number_label_and_sublabel() -> None:
    t = "Boulder City sat about 30 miles from a small railroad stop."
    m = next(x for x in I.extract_numbers(t) if not x.is_year)
    assert I.number_label(m, t) == "miles" and I.number_sublabel(m, t) == ""
    t = "In 1935 the dam took 3.25 million cubic yards of concrete."
    m = next(x for x in I.extract_numbers(t) if not x.is_year)
    assert I.number_label(m, t) == "cubic yards of concrete" and I.number_sublabel(m, t) == "1935"
    assert b.format_number(m.value, m.unit) == "3.25 million"
    t = "Boulder City was 30 miles from Las Vegas."
    m = I.extract_numbers(t)[0]
    assert I.number_label(m, t) == "miles from Las Vegas"


def test_number_card_has_no_era_filler() -> None:
    bc = _bc("Boulder City sat about 30 miles from a small railroad stop.")
    idea = I.number_idea(bc, I.beat_entities(bc), None, "documentary_warm")
    assert idea.motion.payload["sublabel"] == "" and idea.motion.payload["label"] == "miles"


@pytest.mark.parametrize("text", ["In November 1932, the river was turned aside.",
                                  "In the summer of 1931, the camp baked in heat."])
def test_key_phrase_skips_bare_dates(text: str) -> None:
    phrase = I.key_phrase(text)
    assert not I.is_time_only(phrase) and "1932" not in phrase and "1931" not in phrase
    assert I.is_time_only("In November 1932") and I.is_time_only("In the summer of 1931")


# ---------------------------------------------------------------- AI prompts

def test_ai_prompt_drops_purpose_clause_and_negations() -> None:
    idea = VisualIdea(beat_id="b1", family="footage", intent="A steamship ploughing through heavy seas, to show how "
                      "dangerous the old voyage around South America was.")
    prompt = ai_prompt(idea, _bc("x", era=1914))
    assert prompt.startswith("Black and white archival documentary photograph of a steamship ploughing through "
                             "heavy seas, 1910s.")
    assert "to show" not in prompt and " no " not in prompt.lower()
    old_aerial = prompt_subject("Aerial view of green jungle hills", 1885)
    assert old_aerial == "an elevated view from a hillside of jungle hills"
    assert prompt_subject("Ships on a calm sea", 1950) == "ships on a calm sea"


# ---------------------------------------------------------------- top 10 anchors

def _top10_with_recap(items: int = 5, beats: int = 4) -> Script:
    s = top10_script(items=items, beats_per_item=beats)
    recap = Scene(id="s98", title="Recap", purpose=ScenePurpose.resolution,
                  beats=[Beat(id="b900", text="Let us run through the whole list one more time from the top."),
                         Beat(id="b901", text="From the dams to the bridges, every one of them changed the country.")])
    s.scenes.insert(len(s.scenes) - 1, recap)
    return s


def test_recap_grid_on_recap_scene_with_numbers() -> None:
    script = _top10_with_recap()
    sl, _p, nar = _build(script, prof=profile(top10=True, palette="top10_bold"))
    recap = [s for s in sl.shots if s.motion and s.motion.kind == MotionKind.recap_grid]
    assert len(recap) == 1 and recap[0].scene_id == "s98"
    assert recap[0].motion.payload["numbers"] == [5, 4, 3, 2, 1]
    span = (nar.beat_spans["b900"][0], nar.beat_spans["b901"][1])
    assert recap[0].start == pytest.approx(span[0], abs=0.6)
    assert recap[0].end <= sl.duration


def test_short_items_keep_number_cards_at_item_start() -> None:
    script = top10_script(items=6, beats_per_item=1)
    for sc in script.scenes:
        for bt in sc.beats:
            if sc.item_number is not None:
                bt.text = f"Number {sc.item_number} is a short one."
    nar = narrate(script)
    spans = {k: tuple(v) for k, v in nar.beat_spans.items()}
    anchors = structural_anchors(script, spans, profile(top10=True), True, nar.duration + 1.0, "t")
    from vidbush.shots.pacing import _place_anchors
    placed = [a for a in _place_anchors(anchors, nar.duration + 1.0) if a.tag == "number_card"]
    items = [sc for sc in script.scenes if sc.item_number is not None]
    assert len(placed) == len(items)
    for a, sc in zip(placed, items):
        assert a.start == pytest.approx(spans[sc.beats[0].id][0], abs=1e-6)


def test_pinned_anchor_trims_the_one_before() -> None:
    from vidbush.shots.pacing import _place_anchors
    a = Anchor(start=0.0, duration=8.0, type=ShotType.footage_overlay, tag="item_lower_third")
    c = Anchor(start=5.0, duration=6.0, type=ShotType.motion_graphic, tag="number_card", pinned=True)
    out = _place_anchors([a, c], 60.0)
    assert out[0].duration == pytest.approx(5.0) and out[1].start == pytest.approx(5.0)
