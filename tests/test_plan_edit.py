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


@pytest.fixture
def pool(jpg):
    """#1, #2 real photos; #3, #4, #5 AI stills; #6 a real video."""
    out = [Candidate(id=f"real{i}", source=SourceKind.wikimedia, url=f"https://x/real{i}.jpg",
                     keyframes=[jpg(f"real{i}")]) for i in (1, 2)]
    out += [Candidate(id=f"ai{i}", source=SourceKind.ai_image, local_path=f"/ai/{i}.png",
                      keyframes=[jpg(f"ai{i}")]) for i in (3, 4, 5)]
    out.append(Candidate(id="vid6", source=SourceKind.youtube, url="https://x/v6", media="video", in_point=10.0,
                         out_point=30.0, keyframes=[jpg("v6a"), jpg("v6b"), jpg("v6c")]))
    return out


def shots(durs, kinds=None):
    """Consecutive one-clause picture shots lasting durs seconds; kinds give each a real_thing kind."""
    out, t = [], 0.0
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


@pytest.mark.parametrize("kind", ["person", "event", "document"])
def test_ai_still_for_real_person_event_document_is_an_error(pool, kind):
    _, errors = E.check(picks(3), shots([3.0], [kind]), pool, {}, 0)
    assert any("never allowed for a real " + kind in e for e in errors)


def test_ai_still_for_a_place_is_allowed(pool):
    _, errors = E.check(picks(3), shots([3.0], ["place"]), pool, {}, 0)
    assert errors == []


def test_third_ai_still_in_a_row_is_an_error(pool):
    _, errors = E.check(picks(3, 4, 5), shots([3.0, 3.0, 3.0]), pool, {}, 0)
    assert len(errors) == 1 and errors[0].startswith("c003") and "number 3 in a row" in errors[0]
    _, errors = E.check(picks(3, 4), shots([3.0, 3.0]), pool, {}, 0)
    assert errors == []


def test_ai_tail_from_previous_scene_counts(pool):
    _, errors = E.check(picks(3, 4), shots([3.0, 3.0]), pool, {}, 1)
    assert len(errors) == 1 and errors[0].startswith("c002")


def test_ai_still_over_hold_limit_is_an_error(pool):
    _, errors = E.check(picks(3), shots([4.5]), pool, {}, 0)
    assert any("at most 4 s" in e for e in errors)
    _, errors = E.check(picks(3), shots([4.0]), pool, {}, 0)
    assert errors == []


def test_reuse_non_adjacent_needs_callback(pool):
    _, errors = E.check(picks(1, 2, 1), shots([3.0] * 3), pool, {}, 0)
    assert len(errors) == 1 and "already used at c001" in errors[0]
    _, errors = E.check(picks(1, 2, {"pick": 1, "callback": True}), shots([3.0] * 3), pool, {}, 0)
    assert errors == []
    _, errors = E.check(picks(1, 1), shots([3.0] * 2), pool, {}, 0)
    assert errors == []


def test_reuse_of_asset_from_earlier_scene_is_an_error(pool):
    _, errors = E.check(picks(1), shots([3.0]), pool, {E.asset_key(pool[0]): "c009"}, 0)
    assert any("already used at c009" in e for e in errors)


def test_label_on_ai_pick_is_an_error(pool):
    _, errors = E.check(picks({"pick": 3, "label": "Thing"}), shots([3.0]), pool, {}, 0)
    assert any("label is only for a real picture" in e for e in errors)
    _, errors = E.check(picks({"pick": 1, "label": "Thing"}), shots([3.0]), pool, {}, 0)
    assert errors == []


def test_resolve_removes_bad_ai_picks_with_fallbacks(pool):
    sh = shots([3.0, 3.0, 5.0, 3.0], ["person", None, None, None])
    p, errors = E.check(picks(3, 1, 4, {"pick": 5, "label": "x"}), sh, pool, {}, 0)
    assert errors
    out = E.resolve(p, sh, 0)
    # no earlier real pick: text over footage
    assert out[0]["pick"] is None and out[0]["_cand"] is None and out[0]["fallback"] == "text_over_footage"
    assert "removed by the AI rulings" in out[0]["resolved"]
    assert out[1]["_cand"].id == "real1"
    # an earlier real pick exists: hold it
    assert out[2]["pick"] is None and out[2]["fallback"] == "hold_previous"
    # a legal AI still stays, but loses its label
    assert out[3]["_cand"].id == "ai5" and out[3]["label"] == ""


def test_resolve_third_ai_in_a_row(pool):
    sh = shots([3.0, 3.0, 3.0])
    p, _ = E.check(picks(3, 4, 5), sh, pool, {}, 0)
    out = E.resolve(p, sh, 0)
    assert [x["_cand"].id if x["_cand"] else None for x in out] == ["ai3", "ai4", None]
    assert out[2]["fallback"] == "text_over_footage"


def test_in_point_starts_at_keyframe_and_stays_inside_clip(pool):
    v = pool[5]  # 10 s to 30 s
    assert E.in_point(v, "a", 5.0) == (14.0, 19.0)
    assert E.in_point(v, "b", 5.0) == (20.0, 25.0)
    assert E.in_point(v, "c", 5.0) == (25.0, 30.0)  # 26 s would run past the clip end
    assert E.in_point(v, "b", 25.0) == (10.0, 30.0)  # shot longer than the clip: whole clip
    assert E.in_point(pool[0], "b", 5.0) == (0.0, 0.0)  # a photo keeps its points


def test_sheets_build_ceil_n_over_rows_640_wide(pool, tmp_path):
    entries = [{"n": n, "keyframes": c.keyframes, "ai": E.is_ai(c), "media": c.media}
               for n, c in enumerate(pool + pool[:3], 1)]
    paths = SH.build(entries, tmp_path / "sheets", "s01")
    assert len(paths) == math.ceil(len(entries) / SH.ROWS) == 3
    for p in paths:
        assert p.endswith(".jpg")
        with Image.open(p) as im:
            assert im.width == 640
