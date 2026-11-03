"""Parser tests: headings, top10 markers, paragraph grouping, directions, sentences, long input."""
from __future__ import annotations

import random
import re
import time

from vidbush.models import ScenePurpose, VideoFormat
from vidbush.script.parser import infer_purposes, make_beat_texts, parse_script, split_sentences

SENT = ("The crews worked through the heat of the day while the river rose behind the cofferdam. "
        "Engineers watched the gauges every hour. ")


def _para(n: int) -> str:
    return " ".join([SENT] * n).strip()


def test_markdown_title_and_headings():
    text = f"# The Big Dam\n\n## The River\n\n{_para(3)}\n\n## The Men\n\n{_para(3)}\n\n## Legacy\n\n{_para(2)}\n"
    s = parse_script(text)
    assert s.title == "The Big Dam"
    assert [sc.title for sc in s.scenes] == ["The River", "The Men", "Legacy"]
    assert s.scenes[0].purpose == ScenePurpose.hook and s.scenes[-1].purpose == ScenePurpose.outro
    assert s.format == VideoFormat.documentary


def test_caps_scene_and_part_headings():
    text = (f"COLD OPEN\n{_para(2)}\n\nScene 2: The Dig\n{_para(2)}\n\nPart III\n{_para(2)}\n\n"
            f"CHAPTER FOUR: THE FLOOD\n{_para(2)}\n")
    s = parse_script(text)
    assert [sc.title for sc in s.scenes] == ["Cold Open", "The Dig", "Part III", "The Flood"]


def test_top10_markers_set_format_and_items():
    text = ("Here are the ten greatest dams ever built, counted down to the most impressive of all.\n\n"
            "#10: Grand Coulee\nIt is huge and it powers much of the Pacific Northwest to this day.\n\n"
            "Number 9: Aswan High Dam\nIt tamed the Nile and changed the lives of millions of farmers.\n\n"
            "8. Three Gorges\nIt is the biggest power station on Earth by installed capacity.\n\n"
            "No. 7 Itaipu, on the border of Brazil and Paraguay, which is a very long marker line that keeps going.\n\n"
            "## Number six: Hoover Dam\nBuilt in the Great Depression, two years ahead of schedule.\n\n"
            "That is our list. Thanks for watching, and subscribe for more.\n")
    s = parse_script(text)
    assert s.format == VideoFormat.top10
    items = [sc for sc in s.scenes if sc.purpose == ScenePurpose.item]
    assert [sc.item_number for sc in items] == [10, 9, 8, 7, 6]
    assert items[0].title == "Grand Coulee" and items[2].title == "Three Gorges" and items[3].title == "Itaipu"
    assert "Itaipu" in items[3].beats[0].text
    assert s.scenes[0].purpose == ScenePurpose.hook
    assert s.scenes[-1].purpose == ScenePurpose.outro and "subscribe" in s.scenes[-1].beats[-1].text


def test_numbered_prose_is_not_top10():
    text = "# Notes\n\n10. This is a list item in an ordinary script with plenty of words in it.\n\nMore text here."
    assert parse_script(text).format == VideoFormat.documentary


def test_forced_top10_without_markers():
    text = "\n\n".join(f"## Section {k}\n{_para(1)}" for k in range(1, 6))
    s = parse_script(text, fmt="top10")
    assert s.format == VideoFormat.top10
    assert [sc.item_number for sc in s.scenes if sc.item_number] == [3, 2, 1]


def test_plain_paragraphs_grouped_into_scenes():
    paras = [_para(3) for _ in range(12)]  # about 55 words each
    s = parse_script("\n\n".join(paras))
    assert len(s.scenes) >= 3
    for sc in s.scenes[:-1]:
        words = sum(len(b.text.split()) for b in sc.beats)
        assert 120 <= words <= 250
    assert s.title.startswith("The crews worked")


def test_stage_directions_and_markup_stripped():
    text = ("---\ntitle: Front Matter Title\nformat: documentary\n---\n"
            "## One\n[B-roll: jungle at dawn] The jungle was thick and wet. (music swells)\n"
            "VISUAL: map of the isthmus\n"
            "00:12 The **work** went on for years, through rain and mud (as it always did).\n"
            "SFX: explosion\n"
            "NARRATOR: Men kept digging. (Cut to black) <!-- note --> [00:45]\n"
            "> At 10:30 in the morning the whistle blew across the camp.\n")
    s = parse_script(text)
    joined = " ".join(b.text for b in s.beats())
    assert s.title == "Front Matter Title"
    for bad in ("B-roll", "music", "VISUAL", "map of", "00:12", "SFX", "explosion", "NARRATOR", "Cut to", "**",
                "note", "00:45", ">"):
        assert bad not in joined, bad
    assert "(as it always did)" in joined
    assert "At 10:30 in the morning" in joined
    assert "The work went on for years" in joined


def test_sentence_split_respects_abbreviations_and_decimals():
    text = ("Mr. Smith met Dr. Jones at the U.S. Army post in St. Louis. It opened in 1914. "
            "They poured 3.25 million yards. No. 10 is next! Was it John D. Rockefeller? Yes, etc. Then more.")
    assert split_sentences(text) == [
        "Mr. Smith met Dr. Jones at the U.S. Army post in St. Louis.", "It opened in 1914.",
        "They poured 3.25 million yards.", "No. 10 is next!", "Was it John D. Rockefeller?", "Yes, etc.",
        "Then more."]


def test_beats_have_one_to_three_sentences_and_sane_length():
    s = parse_script(f"# T\n\n{_para(10)}\n\nShort one.\n\n{_para(4)}")
    for b in s.beats():
        assert 1 <= len(split_sentences(b.text)) <= 3
        assert len(b.text.split()) <= 45
    ids = [b.id for b in s.beats()]
    assert ids == [f"b{i + 1:03d}" for i in range(len(ids))]


def test_long_sentence_is_split():
    long = ("The engineers, who had come from every corner of the country, worked through the night, " * 4).strip()
    beats = make_beat_texts([long + "."])
    assert len(beats) >= 2 and all(len(b.split()) <= 45 for b in beats)


def test_purpose_inference_positions():
    p = infer_purposes([100] * 10)
    assert p[0] == ScenePurpose.hook and p[1] == ScenePurpose.setup and p[-1] == ScenePurpose.outro
    assert p.index(ScenePurpose.turn) == 5  # midpoint 0.55
    assert p.index(ScenePurpose.climax) == 7  # midpoint 0.75, nearest to 0.8
    assert p[8] == ScenePurpose.resolution
    assert infer_purposes([50]) == [ScenePurpose.hook]
    assert infer_purposes([50, 50]) == [ScenePurpose.hook, ScenePurpose.outro]


def test_very_long_input_9000_words():
    rng = random.Random(3)
    words = ["canal", "river", "workers", "steam", "shovel", "rock", "jungle", "engineer", "lock", "water"]
    paras = []
    for _ in range(150):
        sents = [" ".join(rng.choice(words) for _ in range(rng.randint(8, 16))).capitalize() + "."
                 for _ in range(5)]
        paras.append(" ".join(sents))
    text = "# Long One\n\n" + "\n\n".join(paras)
    n_words = len(re.findall(r"\S+", text)) - 3
    assert n_words >= 9000
    t0 = time.time()
    s = parse_script(text)
    assert time.time() - t0 < 5
    assert s.word_count() == n_words
    assert len(s.scenes) >= 36
    assert s.scenes[0].purpose == ScenePurpose.hook and s.scenes[-1].purpose == ScenePurpose.outro
    assert len({sc.id for sc in s.scenes}) == len(s.scenes)


def test_empty_input_still_returns_a_scene():
    s = parse_script("")
    assert len(s.scenes) == 1 and s.beats()
