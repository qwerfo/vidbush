"""Part 3 of the clause-first planner: picture picks checked against the rulings, resolved, and contact sheets."""
import math

import pytest
from PIL import Image

from vidbush.models import Candidate, SourceKind
from vidbush.plan import edit as E
from vidbush.plan import sheets as SH
from vidbush.plan.clauses import Clause


@pytest.fixture
def jpg(tmp_path):
    def make(name: str) -> str:
        p = tmp_path / f"{name}.jpg"
        Image.new("RGB", (64, 36), (120, 80, 40)).save(p)
        return str(p)
    return make


def rec(c, quality=(), usable=(), shows="", text_in_frame="", ocr_text="", **facts):
    """A description record for candidate c, as vidbush/judge/describe.py returns it."""
    facts = {"title": c.title, **facts}
    return {"media": c.media, "frames": [{"t": c.in_point, "path": c.local_path}],
            "visual": {"shows": shows, "text_in_frame": text_in_frame, "quality": list(quality),
                       "usable": list(usable)},
            "ocr_text": ocr_text, "source_facts": facts, "ai": E.is_ai(c)}


@pytest.fixture
def pool(jpg, tmp_path):
    """#1, #2 real photos; #3, #4, #5 AI stills; #6 a real video (clip 10 s to 30 s)."""
    out = [Candidate(id=f"real{i}", source=SourceKind.wikimedia, url=f"https://x/real{i}.jpg",
                     local_path=jpg(f"real{i}"), title=f"Thing photograph {i}", keyframes=[jpg(f"k_real{i}")])
           for i in (1, 2)]
    out += [Candidate(id=f"ai{i}", source=SourceKind.ai_image, local_path=jpg(f"ai{i}"),
                      keyframes=[jpg(f"k_ai{i}")]) for i in (3, 4, 5)]
    v = tmp_path / "v6.mp4"
    v.write_bytes(b"not really a video")
    out.append(Candidate(id="vid6", source=SourceKind.youtube, url="https://x/v6", media="video", in_point=10.0,
                         out_point=30.0, local_path=str(v), title="River footage",
                         keyframes=[jpg("v6a"), jpg("v6b"), jpg("v6c")]))
    return out


@pytest.fixture
def recs(pool):
    out = {c.id: rec(c) for c in pool}
    out["vid6"] = rec(pool[5], usable=[{"from": 12.0, "to": 28.0, "note": "river"}])
    return out


AFTER_OPENING = 20.0  # a start past vocab.AI_OPENING_S, for tests about AI runs rather than the opening


def shots(durs, kinds=None, t0=0.0):
    """Consecutive one-clause picture shots lasting durs seconds from t0; kinds give each a real_thing kind."""
    out, t = [], t0
    kinds = kinds or [None] * len(durs)
    for i, (d, k) in enumerate(zip(durs, kinds)):
        c = Clause(id=f"c{i + 1:03d}", beat="b001", scene="s01", text="words", start=t, end=t + d, words=[i, i])
        plan = {"id": c.id, "treatment": "picture", "subject": "a thing",
                "real_thing": {"name": "Thing", "kind": k} if k else None}
        out.append({"plan": plan, "clauses": [c]})
        t += d
    return out


def picks(*specs):
    out = []
    for i, s in enumerate(specs):
        p = {"clause": f"c{i + 1:03d}"}
        p.update(s if isinstance(s, dict) else {"pick": s})
        out.append(p)
    return {"picks": out}


def ids(out):
    return [p["_cand"].id if p.get("_cand") else None for p in out]


def move(sh, i, start):
    """Shift clause i (and every later one) so clause i starts at start, keeping durations."""
    d = start - sh[i]["clauses"][0].start
    for s in sh[i:]:
        for c in s["clauses"]:
            c.start, c.end = c.start + d, c.end + d
    return sh


@pytest.mark.parametrize("kind", ["person", "document"])
def test_ai_still_for_real_person_or_document_is_an_error(pool, recs, kind):
    sh = shots([3.0], [kind], AFTER_OPENING)
    p, errors = E.check(picks(3), sh, pool, recs, {}, 0)
    assert any("never allowed for a real " + kind in e for e in errors)
    out = E.resolve(p, sh, 0, recs, {})
    assert out[0]["_cand"] is None and "removed by the AI rulings" in out[0]["resolved"]
    assert out[0]["filled_by"] == ""  # left for the ladder


@pytest.mark.parametrize("kind", ["place", "event"])
def test_ai_still_for_a_place_or_an_event_is_allowed(pool, recs, kind):
    # Adam: anonymous figures in a real event may be an AI illustration
    sh = shots([3.0], [kind], AFTER_OPENING)
    p, errors = E.check(picks(3), sh, pool, recs, {}, 0)
    assert errors == []
    assert ids(E.resolve(p, sh, 0, recs, {})) == ["ai3"]


def test_ai_still_in_the_first_15_s_is_an_error_and_removed(pool, recs):
    sh = shots([3.0], None, 12.0)
    p, errors = E.check(picks(3), sh, pool, recs, {}, 0)
    assert any("first 15 s of the film" in e for e in errors)
    out = E.resolve(p, sh, 0, recs, {})
    assert out[0]["_cand"] is None and "removed by the AI rulings" in out[0]["resolved"]
    _, errors = E.check(picks(3), shots([3.0], None, 15.0), pool, recs, {}, 0)
    assert errors == []


def test_third_ai_still_in_a_row_is_an_error(pool, recs):
    _, errors = E.check(picks(3, 4, 5), shots([3.0, 3.0, 3.0], None, AFTER_OPENING), pool, recs, {}, 0)
    assert len(errors) == 1 and errors[0].startswith("c003") and "number 3 in a row" in errors[0]
    _, errors = E.check(picks(3, 4), shots([3.0, 3.0], None, AFTER_OPENING), pool, recs, {}, 0)
    assert errors == []


def test_ai_tail_from_previous_scene_counts(pool, recs):
    sh = shots([3.0, 3.0], None, AFTER_OPENING)
    p, errors = E.check(picks(3, 4), sh, pool, recs, {}, 1)
    assert len(errors) == 1 and errors[0].startswith("c002")
    assert ids(E.resolve(p, sh, 1, recs, {})) == ["ai3", None]


def test_ai_still_over_hold_limit_is_an_error(pool, recs):
    sh = shots([4.5], None, AFTER_OPENING)
    p, errors = E.check(picks(3), sh, pool, recs, {}, 0)
    assert any("limit is 4 s" in e for e in errors)
    assert ids(E.resolve(p, sh, 0, recs, {})) == [None]
    _, errors = E.check(picks(3), shots([4.0], None, AFTER_OPENING), pool, recs, {}, 0)
    assert errors == []


def test_reuse_non_adjacent_needs_callback(pool, recs):
    _, errors = E.check(picks(1, 2, 1), shots([3.0] * 3), pool, recs, {}, 0)
    assert len(errors) == 1 and "already used at c001" in errors[0]
    _, errors = E.check(picks(1, 1), shots([3.0] * 2), pool, recs, {}, 0)
    assert errors == []


def test_a_callback_within_60_s_is_refused(pool, recs):
    sh = shots([3.0] * 3)
    p, errors = E.check(picks(1, 2, {"pick": 1, "callback": True}), sh, pool, recs, {}, 0)
    assert len(errors) == 1 and errors[0].startswith("c003") and "never shown again within 60 s" in errors[0]
    out = E.resolve(p, sh, 0, recs, {})
    assert ids(out) == ["real1", "real2", None] and "within a minute" in out[2]["resolved"]
    sh = move(shots([3.0] * 3), 2, 3.0 + 60.0)  # real1 left the screen at 3 s; back at 63 s
    p, errors = E.check(picks(1, 2, {"pick": 1, "callback": True}), sh, pool, recs, {}, 0)
    assert errors == [] and ids(E.resolve(p, sh, 0, recs, {})) == ["real1", "real2", "real1"]


def test_a_callback_within_60_s_of_an_earlier_scene_is_refused(pool, recs):
    sh = shots([3.0], None, 100.0)
    shown_end = {E.asset_key(pool[0]): 50.0}
    used = {E.asset_key(pool[0]): "c009"}
    _, errors = E.check(picks({"pick": 1, "callback": True}), sh, pool, recs, used, 0, shown_end)
    assert any("50 s ago" in e for e in errors)
    _, errors = E.check(picks({"pick": 1, "callback": True}), sh, pool, recs, used, 0, {E.asset_key(pool[0]): 40.0})
    assert errors == []


def test_reuse_of_asset_from_earlier_scene_is_an_error(pool, recs):
    sh = shots([3.0])
    used = {E.asset_key(pool[0]): "c009"}
    p, errors = E.check(picks(1), sh, pool, recs, used, 0)
    assert any("already used at c009" in e for e in errors)
    out = E.resolve(p, sh, 0, recs, used)
    assert out[0]["_cand"] is None and "already used at c009" in out[0]["resolved"]


def test_label_on_ai_pick_is_an_error(pool, recs):
    _, errors = E.check(picks({"pick": 3, "label": "Thing"}), shots([3.0]), pool, recs, {}, 0)
    assert any("label 'Thing' is not supported" in e for e in errors)
    _, errors = E.check(picks({"pick": 1, "label": "Thing"}), shots([3.0]), pool, recs, {}, 0)
    assert errors == []  # the word is in real1's title


def test_resolve_removes_bad_ai_picks_and_leaves_the_clause_open(pool, recs):
    sh = shots([3.0, 3.0, 5.0, 3.0], ["person", None, None, None], AFTER_OPENING)
    p, errors = E.check(picks(3, 1, 4, {"pick": 5, "label": "x"}), sh, pool, recs, {}, 0)
    assert errors
    out = E.resolve(p, sh, 0, recs, {})
    assert out[0]["pick"] is None and out[0]["_cand"] is None and out[0]["filled_by"] == ""
    assert "removed by the AI rulings" in out[0]["resolved"]
    assert out[1]["_cand"].id == "real1" and out[1]["filled_by"] == "pick"
    assert out[2]["pick"] is None and "over the limit" in out[2]["resolved"] and out[2]["filled_by"] == ""
    # a legal AI still stays, but loses its label
    assert out[3]["_cand"].id == "ai5" and out[3]["label"] == ""
    assert not any("fallback" in x for x in out)  # the model no longer chooses fallbacks


def test_resolve_third_ai_in_a_row(pool, recs):
    sh = shots([3.0, 3.0, 3.0], None, AFTER_OPENING)
    p, _ = E.check(picks(3, 4, 5), sh, pool, recs, {}, 0)
    out = E.resolve(p, sh, 0, recs, {})
    assert ids(out) == ["ai3", "ai4", None]
    assert out[2]["filled_by"] == ""


def test_hold_limit_counts_the_adjacent_same_asset(pool, recs):
    sh = shots([5.0, 5.0])  # 10 s of one real picture, limit 9 s
    p, errors = E.check(picks(1, 1), sh, pool, recs, {}, 0)
    assert len(errors) == 1 and errors[0].startswith("c002") and "10.0 s" in errors[0]
    assert ids(E.resolve(p, sh, 0, recs, {})) == ["real1", None]


def test_third_consecutive_shot_of_one_asset_is_an_error_and_dropped(pool, recs):
    sh = shots([2.0, 2.0, 2.0])  # 6 s: inside the hold limit, so only the run rule speaks
    p, errors = E.check(picks(1, 1, 1), sh, pool, recs, {}, 0)
    assert len(errors) == 1 and errors[0].startswith("c003") and "shot number 3 in a row" in errors[0]
    out = E.resolve(p, sh, 0, recs, {})
    assert ids(out) == ["real1", "real1", None] and "third shot in a row" in out[2]["resolved"]
    assert out[2]["filled_by"] == ""  # left for the ladder


def test_in_point_from_a_start_time_or_a_legacy_letter(pool):
    v = pool[5]  # 10 s to 30 s
    assert E.in_point(v, 12.0, 5.0) == (12.0, 17.0)
    assert E.in_point(v, 28.0, 5.0) == (25.0, 30.0)  # would run past the clip end
    assert E.in_point(v, "a", 5.0) == (14.0, 19.0)
    assert E.in_point(v, "b", 5.0) == (20.0, 25.0)
    assert E.in_point(v, "c", 5.0) == (25.0, 30.0)
    assert E.in_point(v, "b", 25.0) == (10.0, 30.0)  # shot longer than the clip: whole clip
    assert E.in_point(pool[0], 3.0, 5.0) == (0.0, 0.0)  # a photo keeps its points


def test_scene_pool_needs_a_local_file(pool, tmp_path):
    from vidbush.models import Shot, ShotList, ShotType
    gone = Candidate(id="gone", source=SourceKind.wikimedia, url="https://x/gone.jpg",
                     local_path=str(tmp_path / "missing.jpg"))
    shot = Shot(id="s1", beat_id="b001", start=0.0, end=5.0, duration=5.0, intent="a thing", type=ShotType.image,
                candidates=[pool[0], gone, pool[0]])
    old = ShotList(project_id="p", duration=5.0, shots=[shot])
    assert [c.id for c in E.scene_pool(old, 0.0, 5.0)] == ["real1"]


def test_sheets_build_ceil_n_over_rows_640_wide(pool, tmp_path):
    entries = [{"n": n, "keyframes": c.keyframes, "ai": E.is_ai(c), "media": c.media}
               for n, c in enumerate(pool + pool[:3], 1)]
    paths = SH.build(entries, tmp_path / "sheets", "s01")
    assert len(paths) == math.ceil(len(entries) / SH.ROWS) == 3
    for p in paths:
        assert p.endswith(".jpg")
        with Image.open(p) as im:
            assert im.width == 640


def test_resolve_removes_accidental_repeat_but_keeps_adjacent_and_callback(pool, recs):
    sh = move(shots([3.0, 3.0, 3.0, 3.0, 3.0]), 4, 80.0)  # the callback comes over a minute later
    p, errors = E.check(picks(1, 1, 6, 1, {"pick": 1, "callback": True}), sh, pool, recs, {}, 0)
    assert errors
    out = E.resolve(p, sh, 0, recs, {})
    assert out[1]["_cand"] is not None  # adjacent: the same picture continues
    assert out[2]["_cand"].id == "vid6" and out[2]["start"] == 12.0
    assert out[3]["_cand"] is None and out[3]["filled_by"] == ""
    assert "already used" in out[3]["resolved"]
    assert out[4]["_cand"] is not None  # labelled callback stays


def test_open_clause_no_ai_can_fill_must_name_a_loose_picture(pool, recs):
    """A clause nothing else can fill (too long for an AI still, or in the opening) needs a loose real picture."""
    sh = shots([6.0], None, AFTER_OPENING)
    _, errors = E.check(picks({"pick": None}), sh, pool, recs, {}, 0)
    assert any("name the loose" in e for e in errors)
    _, errors = E.check(picks({"pick": None, "loose": 1}), sh, pool, recs, {}, 0)
    assert not any("name the loose" in e for e in errors)
    _, errors = E.check(picks({"pick": None}), shots([3.0], None, AFTER_OPENING), pool, recs, {}, 0)
    assert not any("name the loose" in e for e in errors), "a short clause may still take an AI still"


def test_a_slightly_short_clip_is_slowed_not_refused(pool):
    v = pool[5]  # clip 10 s to 30 s: 20 s
    assert E.time_problem(v, {}, 10.0, 22.0) == (10.0, None)
    assert E.time_problem(v, {}, 10.0, 24.0)[1] is not None


def test_an_alternate_used_elsewhere_is_still_offered_and_layout_clauses_hold_it_longer(pool, recs):
    sh = shots([10.0], None, AFTER_OPENING)
    p = E.resolve(picks({"pick": None, "loose": 1})["picks"], sh, 0, recs, {"https://x/real1.jpg": "c050"}, pool)
    assert p[0]["_loose"] is None, "a still over 9 s does not stand in for a picture clause"
    sh[0]["plan"]["treatment"] = "document_focus"
    p = E.resolve(picks({"pick": None, "loose": 1})["picks"], sh, 0, recs, {"https://x/real1.jpg": "c050"}, pool)
    assert p[0]["_loose"][0].id == "real1"
