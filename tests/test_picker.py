"""The picker (describe once, pick by comparison): time ranges, blocked records, identity, document words,
labels, the description cache, and the ladder for clauses where nothing fits."""
import json

import pytest

from vidbush.judge import describe as DS
from vidbush.judge import ladder as L
from vidbush.models import Candidate, Narration, SourceKind, Word
from vidbush.plan import edit as E

from test_plan_edit import jpg, picks, pool, recs, shots  # noqa: F401  (fixtures)
from test_plan_helpers import FakeSeats

FILLED = {"pick", "resource", "restate", "hold", "ai_still", "none"}


def ids(out):
    return [p["_cand"].id if p.get("_cand") else None for p in out]


def named(sh, name="Juneau Harbour", kind="place"):
    for s in sh:
        s["plan"]["real_thing"] = {"name": name, "kind": kind}
    return sh


# ---------------------------------------------------------------- video time ranges


def test_video_start_plus_shot_past_the_clip_is_an_error(pool, recs):
    sh = shots([5.0])
    p, errors = E.check(picks({"pick": 6, "start": 27.0}), sh, pool, recs, {}, 0)
    assert any("outside the clip" in e for e in errors)


def test_resolve_removes_a_video_pick_whose_start_is_still_outside_the_clip(pool, recs):
    sh = shots([5.0])
    p, _ = E.check(picks({"pick": 6, "start": 27.0}), sh, pool, recs, {}, 0)
    out = E.resolve(p, sh, 0, recs, {})
    assert out[0]["_cand"] is None and "outside the clip" in out[0]["resolved"]


def test_video_clip_shorter_than_the_shot_is_an_error(pool, recs):
    _, errors = E.check(picks(6), shots([25.0]), pool, recs, {}, 0)
    assert any("the clip is 20.0 s and the shot needs 25.0 s" in e for e in errors)


def test_video_without_a_start_plays_from_the_first_usable_range(pool, recs):
    sh = shots([5.0])
    p, errors = E.check(picks(6), sh, pool, recs, {}, 0)
    assert errors == [] and p[0]["start"] == 12.0 and p[0]["frame"] == 12.0
    assert E.resolve(p, sh, 0, recs, {})[0]["_cand"].id == "vid6"


def test_video_without_usable_ranges_plays_from_the_in_point(pool, recs):
    recs["vid6"]["visual"]["usable"] = []
    p, errors = E.check(picks(6), shots([5.0]), pool, recs, {}, 0)
    assert errors == [] and p[0]["start"] == 10.0


# ---------------------------------------------------------------- blocked records


@pytest.mark.parametrize("mark", ["watermark", "stock_preview_mark", "burned_in_subtitles", "channel_logo"])
def test_blocked_record_is_an_error_and_resolve_removes_it(pool, recs, mark):
    recs["real1"]["visual"]["quality"] = [mark, "low_resolution"]
    sh = shots([3.0])
    p, errors = E.check(picks(1), sh, pool, recs, {}, 0)
    assert any(f"asset 1 has {mark} in its record" in e for e in errors)
    out = E.resolve(p, sh, 0, recs, {})
    assert out[0]["_cand"] is None and mark in out[0]["resolved"] and out[0]["filled_by"] == ""


def test_low_resolution_alone_does_not_block(pool, recs):
    recs["real1"]["visual"]["quality"] = ["low_resolution", "presenter_to_camera"]
    assert DS.blocked(recs["real1"]) == []
    _, errors = E.check(picks(1), shots([3.0]), pool, recs, {}, 0)
    assert errors == []


# ---------------------------------------------------------------- identity of a named real thing


@pytest.fixture
def harbour(pool, recs):
    pool[0].title = "Juneau Harbour 1931"
    recs["real1"]["source_facts"]["title"] = "Juneau Harbour 1931"
    recs["real1"]["source_facts"]["description"] = "a photograph of boats"
    recs["real1"]["visual"]["shows"] = "Juneau Harbour with boats"
    return pool


def test_identity_words_in_the_named_field_are_accepted(harbour, recs):
    sh = named(shots([3.0]))
    p, errors = E.check(picks({"pick": 1, "identity": "Juneau Harbour", "identity_field": "title"}),
                        sh, harbour, recs, {}, 0)
    assert errors == []
    assert E.resolve(p, sh, 0, recs, {})[0]["_cand"].id == "real1"


@pytest.mark.parametrize("identity, field, problem", [
    ("", "title", "not in the record's title"),
    ("Juneau Harbour", "description", "not in the record's description"),
    ("1931", "title", "do not name Juneau Harbour"),
    ("Juneau Harbour", "shows", "identity_field must be one of"),
])
def test_identity_not_supported_is_refused(harbour, recs, identity, field, problem):
    sh = named(shots([3.0]))
    p, errors = E.check(picks({"pick": 1, "identity": identity, "identity_field": field}), sh, harbour, recs, {}, 0)
    assert any(problem in e and "is not shown to be Juneau Harbour" in e for e in errors)
    out = E.resolve(p, sh, 0, recs, {})
    assert out[0]["_cand"] is None and "does not show it is Juneau Harbour" in out[0]["resolved"]


def test_identity_by_the_real_things_date(harbour, recs):
    sh = shots([3.0])
    sh[0]["plan"]["real_thing"] = {"name": "the opening", "kind": "place", "date": "1931"}
    _, errors = E.check(picks({"pick": 1, "identity": "1931", "identity_field": "title"}), sh, harbour, recs, {}, 0)
    assert errors == []


# ---------------------------------------------------------------- document words


def test_passages_found_reads_ocr_text_and_text_in_frame():
    rec = {"ocr_text": "Alaska was admitted to the Union", "visual": {"text_in_frame": "the forty-ninth state"}}
    assert E.passages_found(rec, ["admitted to the Union", "as the forty-ninth state", "statehood vote"]) == \
        [True, True, False]
    assert E.passages_found(rec, ["a b"]) == [False]  # no content words: never counts as found


def test_document_evidence_pick_refused_unless_the_passage_is_on_the_page(pool, recs):
    sh = shots([3.0])
    sh[0]["plan"].update(treatment="document_focus", presentation="full",
                         layout={"purpose": "evidence", "elements": [{"text": "statehood vote passed"}]})
    recs["real1"]["visual"]["shows"] = "a page about the statehood vote passed"  # shows is not page text
    p, errors = E.check(picks({"pick": None, "document_pick": 1}), sh, pool, recs, {}, 0)
    assert any("did not find" in e for e in errors)
    assert E.resolve(p, sh, 0, recs, {})[0]["_doc"] is None
    recs["real1"]["ocr_text"] = "THE STATEHOOD VOTE PASSED"
    p, errors = E.check(picks({"pick": None, "document_pick": 1}), sh, pool, recs, {}, 0)
    assert errors == [] and E.resolve(p, sh, 0, recs, {})[0]["_doc"].id == "real1"


# ---------------------------------------------------------------- labels


def test_unsupported_label_is_removed_and_the_pick_stays(pool, recs):
    sh = shots([3.0])
    p, errors = E.check(picks({"pick": 1, "label": "Juneau 1931"}), sh, pool, recs, {}, 0)
    assert any("label 'Juneau 1931' is not supported" in e for e in errors)
    out = E.resolve(p, sh, 0, recs, {})
    assert out[0]["_cand"].id == "real1" and out[0]["label"] == "" and "label 'Juneau 1931' removed" in out[0]["resolved"]
    assert out[0]["filled_by"] == "pick"


def test_supported_label_stays(pool, recs):
    recs["real1"]["source_facts"]["date"] = "1931"
    sh = shots([3.0])
    p, errors = E.check(picks({"pick": 1, "label": "Thing 1931"}), sh, pool, recs, {}, 0)
    assert errors == [] and E.resolve(p, sh, 0, recs, {})[0]["label"] == "Thing 1931"


# ---------------------------------------------------------------- describe


class CountingSeats:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def providers_for(self, role):
        return []

    def vision_json(self, role, task, system, prompt, images, stage, **kw):
        self.calls += 1
        assert role == DS.FALLBACK_ROLE  # no describer provider configured
        return dict(self.reply)


@pytest.fixture
def still(jpg):
    return Candidate(id="st", source=SourceKind.wikimedia, url="https://x/st.jpg", local_path=jpg("still"),
                     title="Juneau Harbour")


@pytest.fixture(autouse=True)
def no_ocr(monkeypatch):
    monkeypatch.setattr(DS, "ocr_words", lambda path: "")


def test_record_key_differs_for_two_clip_ranges_of_one_file(pool):
    v = pool[5]
    other = v.model_copy(update={"in_point": 30.0, "out_point": 40.0})
    assert DS.record_key(v) != DS.record_key(other)
    assert DS.record_key(v).startswith(DS.content_hash(v.local_path))


def test_describe_one_caches_by_content_hash(still, tmp_path):
    seats = CountingSeats({"shows": "boats in a harbour", "quality": ["watermark", "not_a_quality"],
                           "look": "colour_film"})
    root = tmp_path / "describe"
    r1 = DS.describe_one(seats, still, root)
    assert seats.calls == 1 and r1["cached"] is False
    assert r1["visual"]["shows"] == "boats in a harbour" and r1["visual"]["quality"] == ["watermark"]
    assert r1["source_facts"]["title"] == "Juneau Harbour" and r1["ai"] is False
    saved = json.loads((root / f"{DS.record_key(still)}.json").read_text(encoding="utf-8"))
    assert "source_facts" not in saved  # only visible fields are cached
    copy = still.model_copy(update={"id": "st2", "title": "Another title"})  # same file, other id
    r2 = DS.describe_one(seats, copy, root)
    assert seats.calls == 1 and r2["cached"] is True and r2["visual"] == r1["visual"]
    assert r2["source_facts"]["title"] == "Another title"


def test_describe_one_does_not_cache_a_mock_reply(still, tmp_path):
    seats = CountingSeats({"_mock": True, "shows": "x"})
    root = tmp_path / "describe"
    assert DS.describe_one(seats, still, root)["mock"] is True
    assert not (root / f"{DS.record_key(still)}.json").exists()
    DS.describe_one(seats, still, root)
    assert seats.calls == 2


def test_describe_one_without_a_file_is_none(tmp_path):
    c = Candidate(id="x", source=SourceKind.wikimedia, url="https://x", local_path=str(tmp_path / "nope.jpg"))
    assert DS.describe_one(CountingSeats({}), c, tmp_path) is None


def test_blocked_lists_only_the_blocking_marks():
    rec = {"visual": {"quality": ["low_resolution", "channel_logo", "burned_in_subtitles"]}}
    assert DS.blocked(rec) == ["channel_logo", "burned_in_subtitles"]
    assert DS.blocked({}) == []


# ---------------------------------------------------------------- ladder


def narration(sh, word="river"):
    ws = [Word(text=word, start=s["clauses"][0].start, end=s["clauses"][0].start + 0.4, beat_id="b001") for s in sh]
    return Narration(audio_path="n.wav", duration=sh[-1]["clauses"][0].end, words=ws)


def run_ladder(sh, pool, recs, raw, llm=(), used_before=None, ai_tail=0):
    used_before = used_before or {}
    p, _ = E.check(raw, sh, pool, recs, used_before, ai_tail)
    e = {"scene": "s01", "pool": pool, "picks": E.resolve(p, sh, ai_tail, recs, used_before), "shots": sh,
         "records": recs, "attempts": []}
    seats = FakeSeats(llm=list(llm))
    lad = L.Ladder(seats, None, None)
    lad(e, {"scene": "s01", "scene_purpose": "a river"}, narration(sh), used_before, ai_tail, [])
    return e, seats, lad


def restated(beneath, clause="c001", kind="text_over_footage", text="river", anchor="river"):
    return {"restated": [{"clause": clause, "kind": kind, "text": text, "anchor": anchor, "beneath": beneath,
                          "reason": "no picture of it"}]}


def test_restate_as_text_over_footage(pool, recs):
    e, seats, _ = run_ladder(shots([3.0]), pool, recs, picks(None), llm=[restated(6)])
    p = e["picks"][0]
    assert len(seats.llm_calls) == 1 and seats.llm_calls[0]["role"] == L.RESTATE_SEAT
    assert p["filled_by"] == "restate" and p["fallback"] == "text_over_footage"
    assert p["_fallback_cand"].id == "vid6" and p["overlay_text"] == "river" and p["frame"] == 12.0


@pytest.mark.parametrize("beneath, problem", [(3, "beneath must be a real asset"), (1, "already used at c009")])
def test_restate_naming_an_ai_or_used_asset_is_rejected(pool, recs, beneath, problem):
    used = {E.asset_key(pool[0]): "c009"}
    e, seats, _ = run_ladder(shots([3.0]), pool, recs, picks(None), llm=[restated(beneath)] * 2, used_before=used)
    assert len(seats.llm_calls) == 2  # one retry with the problems
    assert any(problem in x for x in e["restate_attempts"][0]["errors"])
    p = e["picks"][0]
    assert p["filled_by"] != "restate" and p.get("_fallback_cand") is None


def test_short_open_clause_after_a_real_pick_holds_it(pool, recs):
    e, _, _ = run_ladder(shots([3.0, 2.0]), pool, recs, picks(1, None), llm=[{"restated": []}])
    p = e["picks"][1]
    assert p["filled_by"] == "hold" and p["fallback"] == "hold_previous"


def test_hold_never_passes_the_hold_limit(pool, recs):
    e, _, _ = run_ladder(shots([8.0, 2.0]), pool, recs, picks(1, None), llm=[{"restated": []}])
    assert e["picks"][1]["filled_by"] == "ai_still"  # 10 s of one picture is over 9 s; an AI still may fill 2 s


def test_open_clause_naming_a_real_event_never_gets_an_ai_still(pool, recs):
    e, _, _ = run_ladder(shots([3.0], ["event"]), pool, recs, picks(None),
                         llm=[{"restated": [{"clause": "c001", "kind": None, "reason": "nothing honest"}]}])
    p = e["picks"][0]
    assert p.get("fallback") != "ai_still" and p["filled_by"] == "none"


def test_open_clause_without_a_real_thing_gets_an_ai_still(pool, recs):
    e, _, _ = run_ladder(shots([3.5]), pool, recs, picks(None), llm=[{"restated": []}])
    p = e["picks"][0]
    assert p["filled_by"] == "ai_still" and p["fallback"] == "ai_still"


def test_no_ai_still_over_4_s_or_next_to_two_ai_stills(pool, recs):
    e, _, _ = run_ladder(shots([4.5]), pool, recs, picks(None), llm=[{"restated": []}])
    assert e["picks"][0]["filled_by"] == "none"
    e, _, _ = run_ladder(shots([3.0, 3.0, 3.0]), pool, recs, picks(3, 4, None), llm=[{"restated": []}])
    assert e["picks"][2]["filled_by"] == "none"
    e, _, _ = run_ladder(shots([3.0, 3.0]), pool, recs, picks(3, None), llm=[{"restated": []}], ai_tail=1)
    assert e["picks"][1]["filled_by"] == "none"  # the previous scene ended on an AI still


def test_every_clause_ends_with_a_filled_by(pool, recs):
    sh = shots([3.0, 2.0, 3.0, 3.0, 5.0], [None, None, None, "person", None])
    sh[2]["plan"].update(treatment="chart", presentation="full")
    e, _, lad = run_ladder(sh, pool, recs, picks(1, None, None, None, None), llm=[{"restated": []}])
    assert [p["filled_by"] for p in e["picks"]] == ["pick", "hold", "pick", "none", "none"]
    assert all(p["filled_by"] in FILLED for p in e["picks"])
    assert sum(lad.stats.values()) == 5


def test_unfound_evidence_document_is_open_for_the_ladder():
    from vidbush.judge import ladder as L
    plan = {"treatment": "document_focus", "presentation": "full",
            "layout": {"purpose": "evidence", "elements": [{"text": "words on the page"}]}}
    assert L.is_open({"filled_by": "pick", "_doc": None, "_cand": None}, plan)
    assert not L.is_open({"filled_by": "restate", "_doc": None, "_cand": None, "fallback": "text_over_footage"}, plan)
    assert not L.is_open({"filled_by": "pick", "_doc": object(), "_cand": None}, plan)
    period = dict(plan, layout={"purpose": "period", "elements": []})
    assert not L.is_open({"filled_by": "pick", "_doc": None, "_cand": None}, period)
