"""route_map and document_focus: phrase finding on a page, what makes them incomplete, route cue timing."""
from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from vidbush.mg.layouts.base import LayoutJob, PictureRef
from vidbush.mg.layouts.document import DocumentFocus, find_phrase, mark_form, ocr_lines, read_progress
from vidbush.mg.layouts.routemap import RouteMap, parse_route, route_window, split_place
from vidbush.mg.layouts.timing import Cue
from vidbush.mg.textdraw import font
from vidbush.models import LayoutElement, LayoutSpec

PAGE_LINES = ["THE DAILY RECORD", "Congress voted today that the", "territory shall become a state",
              "after years of patient waiting.", "Dated August 2nd, 1939"]


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    img = Image.new("RGB", (1400, 900), (236, 230, 214))
    d = ImageDraw.Draw(img)
    f = font("Medium", 54)
    for k, line in enumerate(PAGE_LINES):
        d.text((90, 80 + k * 140), line, fill=(30, 26, 22), font=f)
    path = tmp_path_factory.mktemp("doc") / "page.png"
    img.save(path)
    try:
        if not ocr_lines(str(path)):
            pytest.skip("OCR engine read nothing")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"OCR engine unavailable: {e}")
    return str(path)


def test_phrase_found_across_two_lines(page):
    hit = find_phrase(ocr_lines(page), "voted today that the territory shall become a state")
    assert hit is not None and hit["ratio"] > 0.85
    assert len(hit["lines"]) == 2
    assert len(hit["words"]) >= 8
    xs = [b[0] for b in hit["words"]]
    assert xs[0] < xs[3]  # words run left to right on the first line


def test_phrase_absent_is_none(page):
    assert find_phrase(ocr_lines(page), "Texas no longer largest") is None


def _doc_job(path: str, source: str, quote: str) -> LayoutJob:
    spec = LayoutSpec(kind="document_focus", elements=[
        LayoutElement(role="document", picture="p"), LayoutElement(role="quote", text=quote, anchor=quote),
        LayoutElement(role="source", text="The Daily Record")])
    cues = [Cue(0, 0.0, 0.0, 0.0, False), Cue(1, 1.75, 2.0, 3.5, True), Cue(2, 0.0, 0.0, 0.0, False)]
    job = LayoutJob(spec=spec, cues=cues, duration=5.0, width=640, height=360)
    job.pictures[0] = PictureRef(path, "image", source=source)
    return job


def test_quote_not_on_the_page_is_incomplete(page):
    job = _doc_job(page, "wikimedia", "Texas no longer largest")
    DocumentFocus(job, 640, 360)
    assert any("not legible" in m for m in job.missing)  # dropped, never the quotation typed beside the page


def test_focus_when_phrase_present(page):
    job = _doc_job(page, "wikimedia", "territory shall become a state")
    comp = DocumentFocus(job, 1280, 720)
    assert job.missing == []
    assert [t.form for t in comp.targets] == ["highlight"]
    comp.render(10), comp.render(100)


def test_no_document_is_incomplete(page):
    job = _doc_job(page, "wikimedia", "territory shall become a state")
    job.pictures.clear()  # a generated picture never reaches job.pictures (make_job refuses it)
    DocumentFocus(job, 640, 360)
    assert any("no real document" in m for m in job.missing)  # never the quotation alone


def test_read_progress_follows_words():
    times = [(1.0, 1.4), (1.5, 1.9), (2.0, 2.4), (2.5, 2.9)]
    assert read_progress(0.5, times) == 0
    assert read_progress(1.95, times) == pytest.approx(0.5)
    assert read_progress(2.2, times) == pytest.approx(0.625)
    assert read_progress(9.0, times) == 1


def test_parse_route_and_place():
    assert parse_route("Nenana to Nome by trail") == ("Nenana", "Nome", "trail")
    assert parse_route("Santa Monica to Jacksonville") == ("Santa Monica", "Jacksonville", "")
    assert parse_route("Nome") is None
    assert split_place("Santa Monica, California") == ("Santa Monica", "California")


def test_route_window_between_place_words():
    a, b = Cue(0, 0.75, 1.0, 1.4, True), Cue(1, 3.75, 4.0, 4.4, True)
    own = Cue(2, 0.75, 1.0, 4.4, True)
    assert route_window(a, b, own) == (1.0, 4.0)
    assert route_window(None, None, own) == (1.0, 4.4)  # places not separate elements: the route's own span
    assert route_window(a, Cue(1, 0.8, 1.05, 1.2, True), own) == (1.0, 1.5)  # at least 0.5 s


def test_route_map_cues_and_trail_fallback():
    spec = LayoutSpec(kind="route_map", elements=[
        LayoutElement(role="place", text="Nenana", anchor="Nenana"),
        LayoutElement(role="place", text="Nome", anchor="Nome"),
        LayoutElement(role="route", text="Nenana to Nome by trail", anchor="Nenana"),
        LayoutElement(role="distance", text="674 miles", anchor="six hundred seventy-four miles")])
    cues = [Cue(0, 1.75, 2.0, 2.5, True), Cue(1, 5.75, 6.0, 6.3, True), Cue(2, 1.75, 2.0, 2.5, True),
            Cue(3, 4.0, 4.25, 5.5, True)]
    job = LayoutJob(spec=spec, cues=cues, duration=7.0)
    comp = RouteMap(job, 652, 432, True)
    m = {mk.label: mk for mk in comp.map.p.markers}
    assert m["Nenana"].appear_at == 1.75 and m["Nome"].appear_at == 5.75
    r = comp.map.p.routes[0]
    assert (r.reveal_start, r.reveal_end) == (2.0, 6.0)
    assert r.dashed and r.label == "674 miles"
    assert any("great circle" in n for n in job.notes)
    assert comp.route_chip_cues[0] is cues[3]
    assert RouteMap.panel_share(spec) == (0.34, 0.40)
    # chips are at the type floor even in a small panel
    assert all(s.h >= 34 for s in comp.map.marker_pills if s is not None)
    frame = comp.render(0)
    assert frame.shape == (432, 652, 3)


def test_unresolved_place_dropped():
    spec = LayoutSpec(kind="route_map", elements=[
        LayoutElement(role="place", text="Nome", anchor="Nome"),
        LayoutElement(role="place", text="Qwxzyville", anchor="Qwxzyville")])
    job = LayoutJob(spec=spec, cues=[Cue(0, 0.5, 0.75, 1.0, True), Cue(1, 1.5, 1.75, 2.0, True)], duration=3.0)
    comp = RouteMap(job, 640, 360, True)
    assert [mk.label for mk in comp.map.p.markers] == ["Nome"]
    assert any("Qwxzyville" in m and "did not resolve" in m for m in job.missing)  # the stage drops the map


def _marks_job(path: str, purpose=None, presentation=None) -> LayoutJob:
    spec = LayoutSpec(kind="document_focus", purpose=purpose, presentation=presentation, elements=[
        LayoutElement(role="document", picture="p"),
        LayoutElement(role="mark", text="August 2nd, 1939", anchor="in August"),
        LayoutElement(role="mark", text="territory shall become a state", anchor="become a state"),
        LayoutElement(role="source", text="The Daily Record, 1939")])
    cues = [Cue(0, 0.0, 0.0, 0.0, False), Cue(1, 1.0, 1.25, 1.7, True), Cue(2, 3.25, 3.5, 4.4, True),
            Cue(3, 1.0, 1.0, 1.0, False)]
    job = LayoutJob(spec=spec, cues=cues, duration=6.0)
    job.pictures[0] = PictureRef(path, "image", source="wikimedia")
    return job


def test_mark_forms():
    assert mark_form("August 2nd, 1939", 3, 1) == "ring"
    assert mark_form("signature Albert Einstein", 2, 1) == "ring"
    assert mark_form("64-20", 2, 1) == "ring"
    assert mark_form("Texas", 1, 1) == "underline"
    assert mark_form("extremely powerful bombs of a new type", 7, 1) == "highlight"
    assert mark_form("a long passage over three lines of type", 8, 3) == "block"


def test_two_marks_in_spoken_order(page):
    comp = DocumentFocus(_marks_job(page), 1920, 1080)
    a, b = comp.targets
    assert (a.form, b.form) == ("ring", "highlight")
    assert a.start == 1.25 and b.start == 3.5  # each mark starts on its own first word
    assert a.t1 <= a.start and b.t0 >= a.t1 and b.t1 <= b.start  # the camera is there as each word comes
    drawn = []
    for f in ("_ring", "_highlight", "_underline", "_block"):
        setattr(comp, f, lambda *args, _f=f: drawn.append(_f))
    for t, want in ((1.2, []), (2.0, ["_ring"]), (3.45, ["_ring"]), (3.8, ["_ring", "_highlight"])):
        drawn.clear()
        comp.render(int(round(t * comp.fps)))
        assert drawn == want, (t, drawn)


def test_camera_never_still(page):
    comp = DocumentFocus(_marks_job(page), 1920, 1080)
    views = [comp.camera(t) for t in (0.0, 0.2, 2.5, 2.7, 5.5, 5.9)]
    for (s1, c1), (s2, c2) in zip(views[::2], views[1::2]):
        assert s1 != s2 or (c1 != c2).any()


def test_period_has_no_marks(page):
    job = _marks_job(page, purpose="period")
    comp = DocumentFocus(job, 1920, 1080)
    assert comp.targets == [] and job.missing == []
    assert any("period" in n for n in job.notes)


def test_small_panel_is_look_only(page):
    job = _marks_job(page, presentation="panel")
    comp = DocumentFocus(job, 240, 160, True)
    assert comp.targets == [] and not comp.fits()
    assert any("look only" in n for n in job.notes)
    w, h = DocumentFocus.panel_share(LayoutSpec(kind="document_focus", elements=[
        LayoutElement(role="document", picture=page), LayoutElement(role="mark", text="territory shall become a state")]))
    assert 0.29 <= w <= 0.5 and 0.35 <= h <= 0.6
    big = DocumentFocus(_marks_job(page, presentation="panel"), int(w * 1920), int(h * 1080), True)
    assert big.fits() and len(big.targets) == 2
    assert all(t.lh * t.s >= 34 for t in big.targets)  # marked type at the floor on screen
