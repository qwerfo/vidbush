"""Caption grouping, geometry, ASS and SRT output, windowing and a real burn."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from vidbush.models import CaptionStyle, Narration, StyleProfile, Word
from vidbush.render import captions as cap
from vidbush.render.ffmpeg_cmds import subtitles
from vidbush.util import run

TEXT = ("The canal opened in 1914. Ships no longer had to sail around Cape Horn, "
        "a voyage of many weeks. The French had tried first, and failed at a terrible cost.")


def _words(text: str = TEXT, step: float = 0.32, beat: str = "b1") -> list[Word]:
    out, t = [], 0.0
    for tok in text.split():
        out.append(Word(text=tok, start=round(t, 3), end=round(t + step - 0.04, 3), beat_id=beat))
        t += step + (0.35 if tok.endswith(".") else 0.0)
    return out


def test_group_words_sizes_and_order() -> None:
    words = _words()
    lines = cap.group_words(words, CaptionStyle())
    assert all(3 <= len(ln.words) <= 5 for ln in lines)
    assert [w.text for ln in lines for w in ln.words] == [w.text for w in words]
    texts = [ln.text for ln in lines]
    assert any(t.endswith("1914") for t in texts)  # breaks at the sentence end, trailing period dropped
    assert all(not t.endswith((",", ".")) for t in texts)


def test_group_words_no_orphans_for_short_sentences() -> None:
    for n in range(3, 14):
        words = _words(" ".join(["word"] * (n - 2) + ["end.", "Yes."]))
        lines = cap.group_words(words, CaptionStyle())
        assert all(3 <= len(ln.words) <= 5 for ln in lines), n
    tiny = cap.group_words(_words("Hello there."), CaptionStyle())
    assert len(tiny) == 1 and tiny[0].text == "Hello there"


def test_group_words_relaxes_impossible_sizes() -> None:
    style = CaptionStyle(min_words=4, max_words=5)
    lines = cap.group_words(_words(" ".join(["w"] * 7)), style)
    assert sum(len(ln.words) for ln in lines) == 7 and all(len(ln.words) >= 3 for ln in lines)


def test_line_hold_rule() -> None:
    words = [Word(text=t, start=s, end=e) for t, s, e in [
        ("one", 0.0, 0.2), ("two", 0.25, 0.4), ("three.", 0.45, 0.6),
        ("four", 0.8, 1.0), ("five", 1.05, 1.2), ("six.", 1.25, 1.4),
        ("seven", 3.0, 3.2), ("eight", 3.25, 3.4), ("nine.", 3.45, 3.6)]]
    lines = cap.group_words(words, CaptionStyle())
    assert [ln.start for ln in lines] == [0.0, 0.8, 3.0]
    assert lines[0].end == 0.8  # gap 0.2 s: held until the next line
    assert lines[1].end == 1.4  # gap 1.6 s: not held


def test_uppercase_style() -> None:
    lines = cap.group_words(_words("one two three."), CaptionStyle(uppercase=True))
    assert lines[0].text == "ONE TWO THREE"


def test_caption_geometry_letterbox_on_and_off() -> None:
    on = StyleProfile(id="p")
    g = cap.caption_geometry(on, 1920, 1080)
    assert g["band_px"] == 138 and g["band_top"] == 942 and g["y"] == 1011 and g["x"] == 960
    assert g["font_px"] + 2 * on.captions.outline <= g["band_px"]
    off = StyleProfile(id="p", letterbox={"enabled": False, "ratio": 2.39})
    assert cap.caption_geometry(off, 1920, 1080)["y"] == 1011
    high = StyleProfile(id="p", captions={"band_height": 0.25})
    assert cap.caption_geometry(high, 1920, 1080)["y"] == 942 + round(138 * 0.25)
    huge = StyleProfile(id="p", captions={"size": 200})
    gh = cap.caption_geometry(huge, 1920, 1080)
    assert gh["font_px"] + 2 * huge.captions.outline <= gh["band_px"]


def test_long_lines_shrink_instead_of_wrapping() -> None:
    prof = StyleProfile(id="p")
    geo = cap.caption_geometry(prof)
    short = "The canal opened"
    long = "Extraordinarily uncharacteristically incomprehensible internationalization counterrevolutionaries"
    assert cap.line_font_px(short, prof, geo) == geo["font_px"]
    size = cap.line_font_px(long, prof, geo)
    assert size < geo["font_px"]
    assert cap.text_width(long, prof.captions, size) <= geo["max_width"]


def test_ass_and_srt_output(tmp_path: Path, settings: Any) -> None:
    prof = StyleProfile(id="p")
    nar = Narration(audio_path="x.wav", duration=20.0, words=_words())
    paths = cap.build_captions(nar, prof, tmp_path, settings)
    assert paths == [tmp_path / "captions.srt", tmp_path / "captions.ass", tmp_path / "work" / "captions.json"]
    ass = paths[1].read_text()
    assert "PlayResX: 1920" in ass and "PlayResY: 1080" in ass and "WrapStyle: 2" in ass
    assert "Style: Default,Inter ExtraBold,64,&H00FFFFFF" in ass
    events = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    assert events and all("{\\an5\\pos(960,1011)" in e for e in events)
    srt = paths[0].read_text()
    assert srt.startswith("1\n00:00:00,000 --> ")
    lines = cap.load_caption_lines(paths[2])
    assert [ln.text for ln in lines] == [e.split("}", 1)[1] for e in events]


def test_ass_font_names() -> None:
    assert cap._ass_font(CaptionStyle(weight="Bold")) == ("Inter", -1)
    assert cap._ass_font(CaptionStyle(weight="Regular")) == ("Inter", 0)
    assert cap._ass_font(CaptionStyle(weight="Black")) == ("Inter Black", 0)
    assert cap._ass_colour("#FFD84A") == "&H004AD8FF"


def test_write_ass_window_shifts_and_clips(tmp_path: Path) -> None:
    lines = [cap.CaptionLine(text="first line here", start=0.0, end=1.5),
             cap.CaptionLine(text="second line here", start=1.5, end=3.2),
             cap.CaptionLine(text="third line here", start=3.2, end=5.0)]
    path = cap.write_ass_window(lines, StyleProfile(id="p"), 2.0, 4.0, tmp_path / "w.ass")
    events = [ln for ln in path.read_text().splitlines() if ln.startswith("Dialogue:")]
    assert len(events) == 2
    assert events[0].startswith("Dialogue: 0,0:00:00.00,0:00:01.20,") and events[0].endswith("second line here")
    assert events[1].startswith("Dialogue: 0,0:00:01.20,0:00:02.00,") and events[1].endswith("third line here")
    empty = cap.write_ass_window(lines, StyleProfile(id="p"), 6.0, 8.0, tmp_path / "e.ass")
    assert "Dialogue:" not in empty.read_text()


@pytest.mark.parametrize("letterbox", [True, False])
def test_burned_caption_pixels_in_band(tmp_path: Path, settings: Any, letterbox: bool) -> None:
    prof = StyleProfile(id="p", letterbox={"enabled": letterbox, "ratio": 2.39})
    lines = [cap.CaptionLine(text="The Panama Canal opened", start=0.0, end=3.0)]
    ass = cap.write_ass_window(lines, prof, 0.5, 2.5, tmp_path / "w.ass")
    png = tmp_path / "f.png"
    fonts = settings.path("assets/fonts")
    run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=1920x1080:d=2:r=30", "-vf",
         subtitles(ass, fonts), "-ss", "1.0", "-frames:v", "1", str(png)])
    img = np.asarray(Image.open(png).convert("L"))
    ys, xs = np.nonzero(img > 128)
    assert len(ys) > 2000
    assert ys.min() >= 942 and ys.max() < 1080
    assert abs((ys.min() + ys.max()) / 2 - 1011) < 12
    assert abs((xs.min() + xs.max()) / 2 - 960) < 12
    predicted = cap.text_width("The Panama Canal opened", prof.captions, 64) - 2 * prof.captions.outline
    assert (xs.max() - xs.min()) == pytest.approx(predicted, rel=0.08)
