"""Packaging: chapters, titles, description, tags, credits and thumbnails."""
from __future__ import annotations

import re
import sys

from PIL import Image

from test_render_helpers import build_shotlist
from vidbush.models import Beat, Narration, Scene, Script, Word
from vidbush.render.package import build_chapters, fit_title, fmt_ts, heuristic_titles, package
from vidbush.render.thumbnails import heuristic_text, thumbnail_texts
from vidbush.util import EM_DASH, write_json

SCENES = [  # (title, purpose, start seconds, sentences)
    ("Ocean to Ocean", "hook", 0.0, ["In 1914 a ship crossed a continent in a single day.",
                                     "The shortcut cost a fortune and thousands of lives.",
                                     "This is the story of how it was built and who paid the price."]),
    ("The French Dream", "setup", 3.0, ["Ferdinand de Lesseps arrived in Panama in 1881 with a plan.",
                                        "He had built the Suez Canal and expected another triumph.",
                                        "His engineers planned a sea level cut through solid rock and swamp."]),
    ("Death in the Jungle", "act", 40.0, ["Yellow fever and malaria swept through the work camps.",
                                          "By 1889 the French company had collapsed.",
                                          "An estimated twenty two thousand workers never went home."]),
    ("The Big Ditch", "climax", 95.0, ["George Goethals drove the Culebra Cut through the mountains.",
                                       "The Gatun Dam created the largest artificial lake on Earth.",
                                       "Locks a thousand feet long lifted ships eighty five feet above the sea."]),
    ("The Price and the Legacy", "outro", 150.0, ["The canal opened on August 15, 1914.",
                                                  "In 1999 it finally passed to Panama."]),
]
DURATION = 175.0


def _script_and_narration() -> tuple[Script, Narration]:
    scenes, words = [], []
    for si, (title, purpose, start, sentences) in enumerate(SCENES):
        beats = [Beat(id=f"b{si}_{k}", text=s) for k, s in enumerate(sentences)]
        scenes.append(Scene(id=f"s{si}", title=title, purpose=purpose, beats=beats))
        t = start + 0.3
        for b in beats:
            for w in b.text.split():
                words.append(Word(text=w, start=t, end=t + 0.3, beat_id=b.id))
                t += 0.4
    script = Script(title="The Panama Canal: The Shortcut That Cost a Fortune", scenes=scenes)
    return script, Narration(audio_path="n.wav", duration=DURATION - 1.0, words=words)


def _prepare(ctx, intro: float = 0.0):
    script, narration = _script_and_narration()
    ctx.save("script", script)
    ctx.save("narration", narration)
    write_json(ctx.work / "assembly.json", {"intro_duration": intro, "outro_duration": 0.0})
    shotlist = build_shotlist(ctx.work / "media")
    shotlist.duration = DURATION
    return shotlist


def _chapters(ctx) -> list[tuple[str, str]]:
    lines = (ctx.root / "chapters.txt").read_text(encoding="utf-8").strip().splitlines()
    out = []
    for ln in lines:
        m = re.fullmatch(r"(\d{2}:\d{2}) (.+)", ln)
        assert m, f"bad chapter line {ln!r}"
        out.append((m.group(1), m.group(2)))
    return out


def _secs(ts: str) -> int:
    m, s = ts.split(":")
    return int(m) * 60 + int(s)


def test_package_with_fixtures(make_ctx) -> None:
    ctx = make_ctx(mode="brief")
    shotlist = _prepare(ctx, intro=5.0)
    paths = package(shotlist, ctx)
    names = [p.name for p in paths]
    assert names[:2] == ["chapters.txt", "description.md"]
    assert names[2:] == ["thumb_1.png", "thumb_2.png", "thumb_3.png"]
    chapters = _chapters(ctx)
    assert chapters[0] == ("00:00", "Ocean to Ocean")
    assert "The French Dream" not in [t for _, t in chapters], "short hook merged with the next scene"
    starts = [_secs(ts) for ts, _ in chapters]
    assert len(chapters) >= 3 and all(b - a >= 10 for a, b in zip(starts, starts[1:]))
    assert starts[1] == int(40.3 + 5.0), "intro offset added"
    md = (ctx.root / "description.md").read_text(encoding="utf-8")
    assert EM_DASH not in md
    assert "The Panama Canal: The Shortcut That Cost a Fortune" in md
    assert "Canal photo by Photographer" in md and "Footage and image credits" in md
    assert "00:00 Ocean to Ocean" in md and "panama canal history" in md
    desc = md.split("## Description\n\n")[1].split("\n\n## Chapters")[0]
    assert 150 <= len(desc.split()) <= 250
    for p in paths[2:]:
        assert Image.open(p).size == (1280, 720)


def test_package_heuristic_fallbacks(make_ctx, monkeypatch) -> None:
    ctx = make_ctx(mode="script")
    shotlist = _prepare(ctx)
    monkeypatch.setitem(sys.modules, "vidbush.source.attribution", None)
    package(shotlist, ctx)
    md = (ctx.root / "description.md").read_text(encoding="utf-8")
    opts = re.findall(r"^\d\. (.+)$", md, flags=re.M)
    assert len(opts) == 3 and len(set(opts)) == 3 and all(len(t) < 70 for t in opts)
    desc = md.split("## Description\n\n")[1].split("\n\n## Chapters")[0]
    assert 150 <= len(desc.split()) <= 250
    assert "Canal photo by Photographer, CC BY-SA 4.0" in md, "local credits fallback"
    assert EM_DASH not in md
    assert len(list((ctx.root / "thumbnails").glob("thumb_*.png"))) == 3


def test_chapters_split_when_too_few_scenes() -> None:
    script, narration = _script_and_narration()
    script.scenes = script.scenes[:2]
    chapters = build_chapters(script, narration, 60.0, 0.0, script.title)
    assert len(chapters) >= 3 and chapters[0].start == 0.0
    assert all(b.start - a.start >= 10 for a, b in zip(chapters, chapters[1:]))


def test_last_short_chapter_dropped_and_hours_format() -> None:
    script, narration = _script_and_narration()
    chapters = build_chapters(script, narration, 155.0, 0.0, script.title)
    assert chapters[-1].title != "The Price and the Legacy"
    assert fmt_ts(3725.4, 4000) == "1:02:05" and fmt_ts(65, 100) == "01:05"


def test_titles_and_texts_heuristics() -> None:
    script, _ = _script_and_narration()
    titles = heuristic_titles(script, script.title)
    assert len(titles) >= 3 and all(len(t) <= 70 for t in titles)
    assert fit_title("A" * 30 + ": " + "b " * 40) == "A" * 30
    assert heuristic_text("22,000 Lost Before the Americans Arrived") == "22,000 LOST"
    assert heuristic_text("In 1914 the World Changed Forever") == "WORLD CHANGED FOREVER"
    assert heuristic_text("The Panama Canal: The Shortcut That Cost a Fortune") == "PANAMA CANAL"


def test_thumbnail_texts_from_fixture(make_ctx) -> None:
    ctx = make_ctx(mode="brief")
    assert thumbnail_texts(["x"], ctx) == ["22,000 DEAD", "THE BIG DITCH", "13,000 TO 5,200 MILES"]
