"""Regression tests for the independent review of the clause-first planner."""
from vidbush.models import Word
from vidbush.plan import anchors as A
from vidbush.plan import direct as D
from vidbush.plan import edit as E
from vidbush.shots import ideas as I

from test_plan_direct import run, scene  # noqa: F401  (fixture)
from test_plan_edit import jpg, picks, pool, recs, shots  # noqa: F401  (fixtures)


def words(text, t0=0.0, step=0.5):
    return [Word(text=w, start=t0 + i * step, end=t0 + i * step + 0.4, beat_id="b001")
            for i, w in enumerate(text.split())]


def test_malformed_director_reply_still_plans_every_clause(scene):  # noqa: F811
    script, nar, cl = scene
    _, out = run(script, nar, cl, ["not json", None, "still not json"])
    assert [p["id"] for p in out["clauses"]] == [c.id for c in cl]
    assert out["clauses"][0]["treatment"] == "picture"


def test_digit_anchor_finds_number_said_over_several_words():
    w = words("nineteen fifty-nine was the year")
    assert A.find(w, "1959 was the year") == 0
    w = words("about two hundred thousand people lived there")
    assert A.find(w, "200,000 people") == 1


def test_old_matcher_still_matches_number_phrases_written_in_words():
    w = words("seven point six percent of the land")
    assert I.phrase_times("seven point six", w) == [0.0]
    assert I.phrase_times("7.6", w) == [0.0]


def test_ai_element_pick_on_a_person_layout_is_refused(pool, recs):  # noqa: F811
    sh = shots([3.0], ["person"])
    sh[0]["plan"].update(treatment="name_card", layout={"elements": [{"text": "A Name", "picture_subject": "face"}]})
    p, errors = E.check(picks({"pick": None, "element_picks": [{"element": 0, "pick": 3}]}), sh, pool, recs, {}, 0)
    assert any("cannot be an AI still" in e for e in errors)
    out = E.resolve(p, sh, 0, recs, {})
    assert out[0]["element_picks"] == []


def test_ai_element_pick_on_a_layout_without_a_real_thing_stays(pool, recs):  # noqa: F811
    sh = shots([3.0])
    sh[0]["plan"].update(treatment="split_comparison", layout={"elements": [{"text": "x", "picture_subject": "y"}]})
    p, errors = E.check(picks({"pick": None, "element_picks": [{"element": 0, "pick": 3}]}), sh, pool, recs, {}, 0)
    assert errors == []
    assert [ep["_cand"].id for ep in E.resolve(p, sh, 0, recs, {})[0]["element_picks"]] == ["ai3"]


def test_element_pick_repeat_needs_callback(pool, recs):  # noqa: F811
    sh = shots([3.0, 3.0])
    sh[1]["plan"].update(treatment="split_comparison", layout={"elements": [{"text": "x"}, {"text": "y"}]})
    eps = [{"element": 0, "pick": 1}, {"element": 1, "pick": 2}]
    p, errors = E.check(picks(1, {"pick": None, "element_picks": eps}), sh, pool, recs, {}, 0)
    assert any("element 0 asset 1 is already used at c001" in e for e in errors)
    out = E.resolve(p, sh, 0, recs, {})
    assert [ep["_cand"].id for ep in out[1]["element_picks"]] == ["real2"]
    p, errors = E.check(picks(1, {"pick": None, "element_picks": eps, "callback": True}), sh, pool, recs, {}, 0)
    assert errors == [] and len(E.resolve(p, sh, 0, recs, {})[1]["element_picks"]) == 2


def test_element_pick_with_a_blocked_record_is_refused(pool, recs):  # noqa: F811
    sh = shots([3.0])
    sh[0]["plan"].update(treatment="split_comparison", layout={"elements": [{"text": "x"}]})
    recs["real2"]["visual"]["quality"] = ["watermark"]
    p, errors = E.check(picks({"pick": None, "element_picks": [{"element": 0, "pick": 2}]}), sh, pool, recs, {}, 0)
    assert any("has a mark or subtitles" in e for e in errors)
    assert E.resolve(p, sh, 0, recs, {})[0]["element_picks"] == []


def test_element_pick_out_of_range_is_an_error(pool, recs):  # noqa: F811
    sh = shots([3.0])
    sh[0]["plan"].update(treatment="split_comparison", layout={"elements": [{"text": "x"}]})
    _, errors = E.check(picks({"pick": None, "element_picks": [{"element": 5, "pick": 1}]}), sh, pool, recs, {}, 0)
    assert any("names no element" in e for e in errors)


def test_same_ai_still_on_adjacent_shots_counts_total_time(pool, recs):  # noqa: F811
    sh = shots([3.0, 3.0])
    p, errors = E.check(picks(3, 3), sh, pool, recs, {}, 0)
    assert any("6.0 s" in e for e in errors)
    out = E.resolve(p, sh, 0, recs, {})
    assert out[0]["_cand"].id == "ai3" and out[1]["_cand"] is None


def test_ai_run_resets_after_a_layout_in_check_and_resolve(pool, recs):  # noqa: F811
    sh = shots([3.0, 3.0, 3.0, 3.0])
    sh[1]["plan"].update(treatment="chart", presentation="full")
    p, errors = E.check(picks(3, {"pick": None}, 4, 5), sh, pool, recs, {}, 0)
    assert not any("in a row" in e for e in errors)
    out = E.resolve(p, sh, 0, recs, {})
    assert [x["_cand"].id if x["_cand"] else None for x in out] == ["ai3", None, "ai4", "ai5"]
    assert out[1]["filled_by"] == "pick"  # a full chart needs no picture
