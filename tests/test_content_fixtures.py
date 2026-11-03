"""Validate every content fixture against the pydantic contracts and the house rules."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from vidbush.cli import parse_brief_file
from vidbush.config import REPO_ROOT
from vidbush.models import (DirectorPlan, MapPayload, MotionKind, Outline, Research, ScenePurpose, Script,
                            VideoFormat)
from vidbush.script.parser import parse_script, split_sentences
from vidbush.util import EM_DASH, read_model

ROOT = Path(REPO_ROOT)
PANAMA = ROOT / "fixtures" / "llm" / "the_panama_canal"
OWN_FILES = (list((ROOT / "fixtures" / "llm").glob("*/*.json"))
             + [ROOT / "fixtures" / f for f in ("demo_brief.md", "demo_script.md", "demo_voiceover.md")]
             + list((ROOT / "vidbush" / "research").glob("*.py")) + list((ROOT / "vidbush" / "script").glob("*.py"))
             + list((ROOT / "tests").glob("test_content_*.py")))


def _load(name: str) -> dict:
    return json.loads((PANAMA / name).read_text(encoding="utf-8"))


def test_no_em_dashes_anywhere():
    for p in OWN_FILES:
        text = p.read_text(encoding="utf-8")
        assert EM_DASH not in text, p
        assert " " + "-" * 2 + " " not in text, p


def test_demo_brief():
    b = parse_brief_file(ROOT / "fixtures" / "demo_brief.md")
    assert b is not None and b.topic == "The Panama Canal" and b.target_minutes == 4
    assert b.format == VideoFormat.documentary and b.audience and b.angle


def test_research_fixture():
    r = read_model(PANAMA / "research.json", Research)
    assert 30 <= len(r.facts) <= 60
    assert [f.id for f in r.facts] == [f"f{i + 1:02d}" for i in range(len(r.facts))]
    assert all(f.source_url.startswith("https://en.wikipedia.org/wiki/") for f in r.facts)
    assert 5 <= len(r.anchors) <= 10
    ids = {f.id for f in r.facts}
    assert all(set(a.fact_ids) <= ids for a in r.anchors)
    places = [a for a in r.anchors if a.kind == "place"]
    assert places and all(a.lat is not None and a.lon is not None for a in places)
    text = " ".join(f.claim for f in r.facts)
    for must in ("1881", "22,000", "November 3, 1903", "Gorgas", "Goethals", "Culebra", "Gatun Lake", "110 feet",
                 "August 15, 1914", "375 million", "5,600", "13,000", "5,200", "1977", "December 31, 1999", "2016"):
        assert must in text, must
    verdicts = _load("factcheck.json")["verdicts"]
    assert {v["id"] for v in verdicts} == ids
    assert {v["status"] for v in verdicts} <= {"verified", "unverified", "contradicted"}


def test_outline_and_script_fixture():
    o = read_model(PANAMA / "outline.json", Outline)
    s = read_model(PANAMA / "script.json", Script)
    fact_ids = {f["id"] for f in _load("research.json")["facts"]}
    assert len(o.sections) == len(s.scenes) == 7
    assert [sc.purpose for sc in s.scenes] == [ScenePurpose.hook, ScenePurpose.setup, ScenePurpose.act,
                                               ScenePurpose.turn, ScenePurpose.act, ScenePurpose.climax,
                                               ScenePurpose.outro]
    beats = s.beats()
    assert 30 <= len(beats) <= 40
    assert 560 <= s.word_count() <= 680
    assert [b.id for b in beats] == [f"b{i + 1:03d}" for i in range(len(beats))]
    assert [sc.id for sc in s.scenes] == [f"s{i + 1:02d}" for i in range(7)]
    for b in beats:
        assert 1 <= len(split_sentences(b.text)) <= 3 and len(b.text.split()) <= 45
        assert set(b.facts_used) <= fact_ids
    assert all(set(sec.fact_ids) <= fact_ids for sec in o.sections)


def test_director_fixture_covers_every_beat():
    plan = read_model(PANAMA / "director.json", DirectorPlan)
    beats = [b.id for b in read_model(PANAMA / "script.json", Script).beats()]
    by_beat: dict[str, list] = {b: [] for b in beats}
    for idea in plan.ideas:
        assert idea.beat_id in by_beat
        by_beat[idea.beat_id].append(idea)
    for bid, ideas in by_beat.items():
        assert {i.family for i in ideas} == {"footage", "still", "motion"}, bid
    assert sum(1 for ideas in by_beat.values() if sum(i.family == "motion" for i in ideas) >= 2) >= 20
    for idea in plan.ideas:
        assert idea.intent.endswith(".")
        if idea.family == "motion":
            assert idea.motion is not None and idea.motion.background == "solid"
            idea.motion.typed_payload()
            if idea.motion.kind in (MotionKind.map, MotionKind.bar_chart, MotionKind.line_chart):
                assert idea.motion.source_note
        else:
            assert idea.must_show and idea.must_avoid and len(idea.queries) >= 2
            if idea.overlay is not None:
                assert idea.overlay.background == "transparent"
                assert idea.overlay.kind in (MotionKind.lower_third, MotionKind.text_overlay, MotionKind.callout,
                                             MotionKind.highlight_box)
                idea.overlay.typed_payload()


def test_director_maps_use_real_coordinates():
    plan = read_model(PANAMA / "director.json", DirectorPlan)
    maps = [i.motion.typed_payload() for i in plan.ideas if i.motion and i.motion.kind == MotionKind.map]
    assert all(isinstance(m, MapPayload) for m in maps)
    routes = [r for m in maps for r in m.routes]
    starts_ends = {(tuple(r.points[0]), tuple(r.points[-1])) for r in routes}
    assert ((40.713, -74.006), (37.775, -122.419)) in starts_ends
    assert ((9.359, -79.901), (8.983, -79.517)) in starts_ends
    horn = [r for r in routes if [-55.98, -67.27] in r.points]
    assert horn and horn[0].dashed
    assert any("Panama" in m.shaded_regions and m.zoom >= 7 for m in maps)
    for m in maps:
        for r in m.routes:
            assert all(-90 <= la <= 90 and -180 <= lo <= 180 for la, lo in r.points)
    charts = [i.motion.typed_payload() for i in plan.ideas if i.motion and i.motion.kind == MotionKind.bar_chart]
    values = [c.series[0].values for c in charts]
    assert [13000, 5200] in values and [22000, 5600] in values


@pytest.mark.parametrize("key", ["the_panama_canal", "demo_script", "demo_voiceover"])
def test_packaging_fixtures(key: str):
    d = ROOT / "fixtures" / "llm" / key
    titles = json.loads((d / "titles.json").read_text())
    assert len(titles["titles"]) == 3 and all(len(t) <= 100 for t in titles["titles"]) and titles["tags"]
    desc = json.loads((d / "description.json").read_text())["description"]
    assert 150 <= len(desc.split()) <= 250
    texts = json.loads((d / "thumbnail_text.json").read_text())["texts"]
    assert len(texts) == 3 and all(len(t.split()) <= 5 for t in texts)


def test_demo_script_parses():
    s = parse_script((ROOT / "fixtures" / "demo_script.md").read_text(encoding="utf-8"))
    assert s.title.startswith("Hoover Dam")
    assert len(s.scenes) == 7 and 590 <= s.word_count() <= 680
    assert s.scenes[0].purpose == ScenePurpose.hook and s.scenes[-1].purpose == ScenePurpose.outro
    text = " ".join(b.text for b in s.beats())
    for must in ("Black Canyon", "Boulder City", "Las Vegas", "Lake Mead", "1931", "21,000", "726 feet",
                 "3.25 million cubic yards", "112", "two years ahead", "Six Companies", "Frank Crowe", "high scalers"):
        assert must in text, must


def test_demo_voiceover_parses_without_headings():
    raw = (ROOT / "fixtures" / "demo_voiceover.md").read_text(encoding="utf-8")
    assert not re.search(r"^##", raw, re.M)
    s = parse_script(raw)
    assert s.title.startswith("The Erie Canal") and len(s.scenes) >= 3
    assert s.word_count() >= 560
    text = " ".join(b.text for b in s.beats())
    for must in ("Albany", "Buffalo", "363 miles", "1825", "DeWitt Clinton", "Clinton's Ditch", "83",
                 "seven million dollars", "ninety percent", "leading port"):
        assert must in text, must
