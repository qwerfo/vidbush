"""Data layouts (chart, table, timeline): timing, panels, history, emphasis. Small frames, no ffmpeg."""
import numpy as np

from vidbush.mg.layouts import comp_class
from vidbush.mg.layouts.base import LayoutJob
from vidbush.mg.layouts.data import Table, chart_emphasis, parse_event, parse_value, timeline_history
from vidbush.mg.layouts.panel import PanelComp, PanelPlan, change_time, large_rect
from vidbush.mg.layouts.timing import Cue
from vidbush.models import LayoutElement as E, LayoutKind, LayoutSpec

W, H, FPS = 480, 270, 10


def _job(spec, appears, dur=6.0, extra=None):
    cues = [Cue(i, a, a + 0.25, a + 0.8, True, e.stage) for i, (e, a) in enumerate(zip(spec.elements, appears))]
    return LayoutJob(spec=spec, cues=cues, duration=dur, width=W, height=H, fps=FPS, extra=extra or {})


CHART = LayoutSpec(kind="chart", title="Communities", elements=[
    E(role="bar", text="No road: 82%"), E(role="bar", text="Road: 18%")])
LINE = LayoutSpec(kind="chart", title="Fuel", elements=[E(role="point", text=f"{y}: {v}") for y, v in
                                                        ((2000, 1), (2001, 2), (2002, 3), (2003, 9), (2004, 10))])
TABLE = LayoutSpec(kind="table", title="Coast", elements=[
    E(role="header", text="Measure | Miles"), E(role="row", text="Coastline, point to point | 6,640 mi"),
    E(role="row", text="Shoreline, every inlet | 47,300 mi", stage=2),
    E(role="row", text="Earth, around | 24,900 mi", stage=2)])
TL = LayoutSpec(kind="timeline", id="nome", title="Nome", elements=[E(role="event", text="2011: Fuel barge iced out")])


def _blank(spec, appears, extra=None):
    job = _job(spec, appears, extra=extra)
    c = comp_class(spec.kind)(job, W, H, False)
    return c, c.render(0), c._ground


def test_nothing_drawn_before_first_cue():
    for spec, ap in ((CHART, [1.0, 2.0]), (LINE, [1, 1.5, 2, 2.5, 3]), (TABLE, [1.0, 1.5, 2.0, 2.5]), (TL, [1.0])):
        c, f0, g = _blank(spec, ap)
        assert np.array_equal(f0, g), spec.kind
        assert not np.array_equal(c.render(int(4 * FPS)), g), spec.kind


def test_timeline_history_shown_from_frame_zero():
    _, f0, g = _blank(TL, [3.0], extra={"history": ["1925: Diphtheria in Nome"]})
    assert not np.array_equal(f0, g)


def test_timeline_history_helper():
    a = LayoutSpec(kind="timeline", id="nome", elements=[E(role="event", text="1925: Serum run")])
    other = LayoutSpec(kind="timeline", id="x", elements=[E(role="event", text="1900: Gold")])
    b = LayoutSpec(kind="timeline", id="nome", elements=[E(role="event", text="2011: Barge"),
                                                         E(role="event", text="1925: Serum run")])
    c = LayoutSpec(kind="timeline", id="nome", elements=[E(role="event", text="2012: Tanker")])
    assert timeline_history([a, other, b], c) == ["1925: Serum run", "2011: Barge"]
    assert timeline_history([a], b) == []  # its own events are not history
    assert timeline_history([a], LayoutSpec(kind="timeline", elements=[])) == []  # no id, no history


def test_chart_emphasis_index():
    assert chart_emphasis([82, 18], False) == 0
    assert chart_emphasis([1, 2, 3, 9, 10], True) == 2  # the knee: the slope changes most at 3
    assert chart_emphasis([1, 2, 3], True) == 2  # straight line: the peak
    assert chart_emphasis([], False) == -1
    assert parse_value("Communities with no road: 82%") == ("Communities with no road", "82%", 82.0)
    assert parse_value("Shoreline: 47,300 mi")[2] == 47300.0
    assert parse_event("1925: Diphtheria in Nome") == ("1925", "Diphtheria in Nome", 1925)


def test_stage2_hidden_while_panel_then_shown_after_expand():
    spec = TABLE.model_copy(update={"presentation": "panel_then_expand"})
    job = _job(spec, [0.2, 0.4, 3.0, 4.0], dur=6.0)
    small = (20.0, 20.0, 200.0, 120.0)
    plan = PanelPlan("panel_then_expand", small, large_rect(job, True), change_time(job, "panel_then_expand"))
    pc = PanelComp(job, lambda j, w, h, p: Table(j, w, h, p), plan)
    assert [r[1].appear for r in pc.small.rows] == [0.4]  # the small panel never lays out stage 2 rows
    assert plan.change_at < 3.0
    assert len(pc.large.rows) == 3
    assert all(c.appear >= plan.change_at + 0.8 for c in (r[1] for r in pc.large.rows[1:]))


def test_panel_share_fits_at_floors():
    for spec in (CHART, LINE, TABLE, TL):
        cls = comp_class(spec.kind)
        w, h = cls.panel_share(spec)
        assert 0.2 < w <= 0.62 and 0.2 < h <= 0.62
        job = _job(spec, [0.0] * len(spec.elements))
        job.width, job.height = 1920, 1080
        c = cls(job, int(1920 * w) // 2 * 2, int(1080 * h) // 2 * 2, True)
        assert c.fits(), (spec.kind, c.need, c.bw, c.bh)
    wide = TL.model_copy(update={"elements": [E(role="event", text=f"{1900 + 10 * i}: Harbour closed by the ice again {i}")
                                              for i in range(5)]})
    assert comp_class(LayoutKind.timeline).panel_share(wide)[0] > comp_class(LayoutKind.timeline).panel_share(TL)[0]
