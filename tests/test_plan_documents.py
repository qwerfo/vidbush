"""Adam's ruling on documents: presentation follows purpose; a document is never a static full page."""
from vidbush.models import ShotType
from vidbush.plan import direct as D
from vidbush.plan import edit as E
from vidbush.plan import emit as M

from test_plan_direct import by_id, scene, valid  # noqa: F401  (fixture)
from test_plan_edit import jpg, picks, pool, shots  # noqa: F401  (fixtures)
from test_plan_emit import cands, setup  # noqa: F401  (fixtures)


def _doc(r, cid="c003", **kw):
    p = by_id(r, cid)
    p.update(treatment="document_focus", **kw)
    return r


def _errors(r, cl, nar):
    return D.validate(r, cl, nar)[1]


def test_evidence_document_marks_its_passage_in_close(scene):  # noqa: F811
    _, nar, cl = scene
    r = _doc(valid(), presentation="full", presentation_reason="the quoted words prove it",
             layout={"purpose": "evidence",
                     "elements": [{"text": "admitted to the Union as the forty-ninth state",
                                   "anchor": "Alaska became a state"}]})
    assert _errors(r, cl, nar) == []  # a page passage may be longer than 6 words


def test_static_full_page_is_refused(scene):  # noqa: F811
    _, nar, cl = scene
    r = _doc(valid(), presentation="full", layout={"purpose": "period", "elements": []})
    assert any("never a static full page" in e for e in _errors(r, cl, nar))


def test_period_document_rides_as_a_panel_without_passages(scene):  # noqa: F811
    _, nar, cl = scene
    r = _doc(valid(), presentation="panel", presentation_reason="atmosphere, the story goes on",
             layout={"purpose": "period", "elements": []})
    assert _errors(r, cl, nar) == []


def test_evidence_without_passages_and_missing_purpose_are_errors(scene):  # noqa: F811
    _, nar, cl = scene
    errs = _errors(_doc(valid(), presentation="panel", layout={"purpose": "evidence", "elements": []}), cl, nar)
    assert any("passages to mark" in e for e in errs)
    errs = _errors(_doc(valid(), presentation="panel", layout={"elements": []}), cl, nar)
    assert any("layout.purpose" in e for e in errs)
    errs = _errors(_doc(valid(), layout={"purpose": "period", "elements": []}), cl, nar)
    assert any("needs presentation" in e for e in errs)


def test_repair_never_leaves_a_static_full_page(scene):  # noqa: F811
    _, nar, cl = scene
    plans, _ = D.validate(_doc(valid(), presentation="full", layout={"elements": []}), cl, nar)
    out = D.repair(plans, cl, nar)
    assert out[2]["treatment"] == "document_focus"
    assert out[2]["presentation"] == "panel" and out[2]["layout"]["purpose"] == "period"


def test_document_pick_must_be_real(pool):  # noqa: F811
    sh = shots([3.0])
    sh[0]["plan"].update(treatment="document_focus", presentation="panel", layout={"purpose": "period"})
    p, errors = E.check(picks({"pick": 6, "document_pick": 3}), sh, pool, {}, 0)
    assert any("document_pick 3 must be a real candidate" in e for e in errors)
    p, errors = E.check(picks({"pick": 6, "document_pick": 1}), sh, pool, {}, 0)
    assert not errors and p[0]["_doc"].id == "real1"


def test_emit_document_full_is_the_picture_and_panel_sits_over_footage(setup, cands):  # noqa: F811
    script, nar, cl, sp = setup
    sp[0]["clauses"][4].update(treatment="document_focus", presentation="full",
                               layout={"purpose": "subject", "elements": [{"text": "Juneau", "anchor": "Juneau"}]})
    sp[0]["clauses"][5].update(treatment="document_focus", presentation="panel", layout={"purpose": "period"})
    picks_ = [{"_cand": cands["video"], "frame": "b"}, {}, {"_cand": cands["image"]},
              {"_cand": None, "_doc": cands["image2"]}, {"_cand": cands["video"], "_doc": cands["image2"],
                                                         "callback": True}]
    _, sl = M.emit("p1", script, nar, cl, sp, [{"scene": "s01", "picks": picks_}])
    full, panel = sl.shots[3], sl.shots[4]
    assert full.type == ShotType.image and full.selected == "img2" and full.layout.purpose == "subject"
    assert full.layout.document == "img2"
    assert panel.type == ShotType.footage and panel.selected == "vid" and panel.layout.document == "img2"
    assert panel.layout.presentation == "panel"


def test_evidence_document_needs_its_passages_visible(pool):  # noqa: F811
    sh = shots([3.0])
    sh[0]["plan"].update(treatment="document_focus", presentation="full",
                         layout={"purpose": "evidence", "elements": [{"text": "a line"}, {"text": "b line"}]})
    p, errors = E.check(picks({"pick": None, "document_pick": 1, "passages_visible": [True, False]}),
                        sh, pool, {}, 0)
    assert any("passages_visible all true" in e for e in errors)
    out = E.resolve(p, sh, 0)
    assert out[0]["_doc"] is None and "passages not visible" in out[0]["resolved"]
    p, errors = E.check(picks({"pick": None, "document_pick": 1, "passages_visible": [True, True]}),
                        sh, pool, {}, 0)
    assert not errors and E.resolve(p, sh, 0)[0]["_doc"].id == "real1"


def test_full_route_map_with_a_pick_stays_a_layout(setup, cands):  # noqa: F811
    script, nar, cl, sp = setup
    picks_ = [{"_cand": cands["video"]}, {"_cand": cands["image"]}, {}, {}, {}]
    _, sl = M.emit("p1", script, nar, cl, sp, [{"scene": "s01", "picks": picks_}])
    assert sl.shots[1].type == ShotType.motion_graphic and sl.shots[1].layout.kind == "route_map"
