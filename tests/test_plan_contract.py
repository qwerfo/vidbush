"""The planner emits exactly the layouts lane's contract (models.LayoutSpec, LAYOUT_ROLES)."""
from vidbush.models import LAYOUT_ROLES, Candidate, LayoutKind, LayoutSpec, Shot, ShotType, SourceKind
from vidbush.plan import emit as E


def _cand(cid, src=SourceKind.wikimedia, media="image"):
    return Candidate(id=cid, source=src, media=media)


def test_unknown_roles_become_renderer_roles():
    plan = {"treatment": "timeline", "presentation": "panel",
            "layout": {"timeline_id": "t1", "elements": [{"role": "date", "text": "1925: Serum run", "anchor": "x"}]}}
    lay = E._layout(plan, {}, None, None)
    assert lay.kind == LayoutKind.timeline and lay.id == "t1" and lay.presentation == "panel"
    assert [e.role for e in lay.elements] == ["event"]
    plan = {"treatment": "number_over_footage", "presentation": "panel",
            "layout": {"elements": [{"role": "value", "text": "40 KM", "anchor": "forty"},
                                    {"role": "caption", "text": "of ice", "anchor": "ice"}]}}
    lay = E._layout(plan, {}, _cand("bg", SourceKind.youtube, "video"), None)
    assert [e.role for e in lay.elements] == ["background", "number", "label"]
    assert lay.elements[0].picture == "bg" and lay.presentation is None  # not a presented kind
    for e in lay.elements:
        assert e.role in LAYOUT_ROLES[lay.kind]


def test_name_card_pick_becomes_portrait_and_document_is_an_element():
    face = _cand("face")
    plan = {"treatment": "name_card", "layout": {"elements": [{"text": "A Name", "anchor": "a name"}]}}
    lay = E._layout(plan, {"element_picks": [{"element": 0, "_cand": face}]}, None, None)
    assert [(e.role, e.picture) for e in lay.elements] == [("portrait", "face"), ("name", None)]
    doc = _cand("page")
    plan = {"treatment": "document_focus", "presentation": "full", "real_thing": {"name": "The Gazette", "date": "1958"},
            "layout": {"purpose": "evidence", "elements": [{"text": "bridge condemned", "anchor": "condemned"}]}}
    lay = E._layout(plan, {}, doc, doc)
    assert [e.role for e in lay.elements] == ["document", "mark", "source"]
    assert lay.elements[0].picture == "page" and lay.purpose == "evidence"
    assert lay.elements[2].text == "The Gazette, 1958"


def _shot(sid, start, end, typ=ShotType.footage, layout=None):
    return Shot(id=sid, beat_id="b", scene_id="s01", start=start, end=end, duration=end - start, type=typ, intent="x",
                layout=layout)


def test_full_layout_cut_moves_to_its_first_word_within_hold_limits():
    lay = LayoutSpec(kind=LayoutKind.route_map, presentation="full")
    shots = [_shot("a", 0.0, 4.0), _shot("b", 4.0, 10.0, ShotType.motion_graphic, lay)]
    E.start_full_layouts_on_words(shots, {"b": 6.25}, {})
    assert shots[0].end == shots[1].start == 6.0 and shots[1].duration == 4.0
    assert "cut_on_layout_words" in shots[1].flags
    # an AI still never holds past 4 s, and a layout keeps at least 3 s
    shots = [_shot("a", 0.0, 3.0, ShotType.ai_image), _shot("b", 3.0, 10.0, ShotType.motion_graphic, lay)]
    E.start_full_layouts_on_words(shots, {"b": 6.25}, {})
    assert shots[1].start == 3.0
    shots = [_shot("a", 0.0, 3.0), _shot("b", 3.0, 7.0, ShotType.motion_graphic, lay)]
    E.start_full_layouts_on_words(shots, {"b": 5.0}, {})
    assert shots[1].start == 3.0
    # a panel is not moved: it enters on its words over the shot's own picture
    panel = LayoutSpec(kind=LayoutKind.route_map, presentation="panel")
    shots = [_shot("a", 0.0, 4.0), _shot("b", 4.0, 10.0, ShotType.footage, panel)]
    E.start_full_layouts_on_words(shots, {"b": 6.25}, {})
    assert shots[1].start == 4.0
