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


def ids(out):
    return [p["_cand"].id if p.get("_cand") else None for p in out]


@pytest.mark.parametrize("kind", ["person", "event", "document"])
def test_ai_still_for_real_person_event_document_is_an_error(pool, recs, kind):
    sh = shots([3.0], [kind])
    p, errors = E.check(picks(3), sh, pool, recs, {}, 0)
    assert any("never allowed for a real " + kind in e for e in errors)
    out = E.resolve(p, sh, 0, recs, {})
    assert out[0]["_cand"] is None and "removed by the AI rulings" in out[0]["resolved"]
    assert out[0]["filled_by"] == ""  # left for the ladder


def test_ai_still_for_a_place_is_allowed(pool, recs):
    _, errors = E.check(picks(3), shots([3.0], ["place"]), pool, recs, {}, 0)
    assert errors == []


def test_third_ai_still_in_a_row_is_an_error(pool, recs):
    _, errors = E.check(picks(3, 4, 5), shots([3.0, 3.0, 3.0]), pool, recs, {}, 0)
    assert len(errors) == 1 and errors[0].startswith("c003") and "number 3 in a row" in errors[0]
    _, errors = E.check(picks(3, 4), shots([3.0, 3.0]), pool, recs, {}, 0)
    assert errors == []


def test_ai_tail_from_previous_scene_counts(pool, recs):
    sh = shots([3.0, 3.0])
    p, errors = E.check(picks(3, 4), sh, pool, recs, {}, 1)
    assert len(errors) == 1 and errors[0].startswith("c002")
    assert ids(E.resolve(p, sh, 1, recs, {})) == ["ai3", None]


def test_ai_still_over_hold_limit_is_an_error(pool, recs):
    sh = shots([4.5])
    p, errors = E.check(picks(3), sh, pool, recs, {}, 0)
    assert any("limit is 4 s" in e for e in errors)
    assert ids(E.resolve(p, sh, 0, recs, {})) == [None]
    _, errors = E.check(picks(3), shots([4.0]), pool, recs, {}, 0)
    assert errors == []


def test_reuse_non_adjacent_needs_callback(pool, recs):
    _, errors = E.check(picks(1, 2, 1), shots([3.0] * 3), pool, recs, {}, 0)
    assert len(errors) == 1 and "already used at c001" in errors[0]
    _, errors = E.check(picks(1, 2, {"pick": 1, "callback": True}), shots([3.0] * 3), pool, recs, {}, 0)
    assert errors == []
    _, errors = E.check(picks(1, 1), shots([3.0] * 2), pool, recs, {}, 0)
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
    sh = shots([3.0, 3.0, 5.0, 3.0], ["person", None, None, None])
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
    sh = shots([3.0, 3.0, 3.0])
    p, _ = E.check(picks(3, 4, 5), sh, pool, recs, {}, 0)
    out = E.resolve(p, sh, 0, recs, {})
    assert ids(out) == ["ai3", "ai4", None]
    assert out[2]["filled_by"] == ""


def test_hold_limit_counts_the_adjacent_same_asset(pool, recs):
    sh = shots([4.0, 4.0, 4.0])  # 12 s of one real picture, limit 9 s
    p, errors = E.check(picks(1, 1, 1), sh, pool, recs, {}, 0)
    assert len(errors) == 1 and errors[0].startswith("c003") and "12.0 s" in errors[0]
    assert ids(E.resolve(p, sh, 0, recs, {})) == ["real1", "real1", None]


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
    sh = shots([3.0, 3.0, 3.0, 3.0, 3.0])
    p, errors = E.check(picks(1, 1, 6, 1, {"pick": 1, "callback": True}), sh, pool, recs, {}, 0)
    assert errors
    out = E.resolve(p, sh, 0, recs, {})
    assert out[1]["_cand"] is not None  # adjacent: the same picture continues
    assert out[2]["_cand"].id == "vid6" and out[2]["start"] == 12.0
    assert out[3]["_cand"] is None and out[3]["filled_by"] == ""
    assert "already used" in out[3]["resolved"]
    assert out[4]["_cand"] is not None  # labelled callback stays
