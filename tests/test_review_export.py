"""review.md export: exact structure, comments kept whole, escaping, clock() and summary_text."""
from __future__ import annotations

import pytest

from review_fixture import PID, build_review_project
from vidbush.review import export, store

EM = chr(0x2014)


@pytest.fixture
def root(settings):
    return build_review_project(settings)


LONG = "\n".join(f"Paragraph {i}: the river shots look too modern, use 1930s film stock." for i in range(300))


def _fill(settings) -> store.Review:
    store.update_shot(settings, PID, 1, verdict="good")
    store.update_shot(settings, PID, 2, verdict="quality_problem", rejects=["watermark_logo"],
                      comment="Channel | logo in the corner.\n\nSecond paragraph after a blank line.")
    store.update_shot(settings, PID, 3, verdict="wrong_picture", comment=LONG)
    store.update_shot(settings, PID, 4, verdict="wrong_timing")  # not good, no comment
    store.update_shot(settings, PID, 5, comment=f"Nice light {EM} keep it")  # comment, no verdict
    store.update_shot(settings, PID, 7, verdict="quality_problem", rejects=["anachronism", "repeated_clip"])
    store.update_shot(settings, PID, 8, verdict="other", comment="   ")  # blank comment counts as none
    store.set_project_comment(settings, PID, "Overall good.\nMusic too loud | at the end.")
    return store.load(settings, PID)


def test_markdown_structure(settings, root):
    md = export.markdown(_fill(settings), "Hoover Dam: Taming the Colorado", 70)
    lines = md.splitlines()
    assert lines[0] == "# Review: Hoover Dam: Taming the Colorado"
    assert lines[2] == f"Project id `{PID}`. Reviewed 6 of 8 shots. Judge threshold 70."
    assert lines[4].startswith("How to read this:")
    heads = [ln for ln in lines if ln.startswith("## ")]
    assert heads == ["## Summary", "## Hard rejects", "## Project comment", "## Shots with a comment (3)",
                     "## Shots marked not good without a comment (3)"]
    i = lines.index("## Summary")
    assert lines[i + 2:i + 10] == ["| Verdict | Shots |", "|---|---|", "| good | 1 |", "| wrong picture | 1 |",
                                   "| wrong timing | 1 |", "| quality problem | 2 |", "| other | 1 |",
                                   "| not reviewed | 2 |"]
    i = lines.index("## Hard rejects")
    assert lines[i + 2:i + 6] == ["| Shot | Time | Hard rejects | Verdict |", "|---|---|---|---|",
                                  "| 2 (shot_002) | 0:04.0 | watermark or logo | quality problem |",
                                  "| 7 (shot_007) | 0:28.0 | anachronism, repeated clip | quality problem |"]
    assert lines[i + 7] == "Totals: watermark or logo 1, anachronism 1, repeated clip 1."
    i = lines.index("## Project comment")
    assert lines[i + 2:i + 4] == ["> Overall good.", "> Music too loud | at the end."]
    assert md.endswith("\n") and not md.endswith("\n\n")


def test_commented_shots_in_order_and_whole(settings, root):
    md = export.markdown(_fill(settings))
    blocks = [ln for ln in md.splitlines() if ln.startswith("### ")]
    assert blocks == ["### Shot 2 (shot_002) at 0:04.0: quality problem",
                      "### Shot 3 (shot_003) at 0:09.0: wrong picture",
                      "### Shot 5 (shot_005) at 0:18.5: not reviewed"]
    assert "### Shot 1 " not in md and "### Shot 4 " not in md and "### Shot 8 " not in md
    for i in range(300):  # every line of the long comment, nothing cut
        assert f"> Paragraph {i}: the river shots look too modern, use 1930s film stock." in md
    assert "> Channel | logo in the corner.\n>\n> Second paragraph after a blank line." in md
    s2 = md.split("### Shot 2 ")[1].split("### Shot 3 ")[0]
    assert "- Type: footage_overlay, 5.0 s (0:04.0 to 0:09.0)" in s2
    assert "- Intent: The Colorado River flooding farms" in s2
    assert ("- Selected: shot_002_archive_0 from archive.org, licence Public domain" in s2)
    assert "- Judge score: 72, reason: flood water over fields" in s2
    assert "- Hard rejects ticked: watermark or logo" in s2
    assert "Comment:" in s2


def test_no_em_dash_in_output(settings, root):
    md = export.markdown(_fill(settings))
    assert EM not in md
    assert "> Nice light, keep it" in md


def test_pipes_escaped_in_cells(settings, root):
    review = _fill(settings)
    review.shots[1].snapshot.intent = "River | farms\nflooded"
    review.shots[1].snapshot.licence = "CC BY | SA"
    md = export.markdown(review, title="Dams | Rivers")
    assert md.splitlines()[0] == "# Review: Dams / Rivers"
    assert "- Intent: River / farms flooded" in md
    assert "licence CC BY / SA" in md
    for ln in md.splitlines():
        if ln.startswith("| 2 (shot_002)"):
            assert ln.count("|") == 5


def test_not_good_without_comment_table(settings, root):
    md = export.markdown(_fill(settings))
    lines = md.splitlines()
    i = lines.index("## Shots marked not good without a comment (3)")
    assert lines[i + 2:i + 7] == ["| Shot | Time | Verdict | Type | Judge score | Hard rejects |",
                                  "|---|---|---|---|---|---|",
                                  "| 4 (shot_004) | 0:12.5 | wrong timing | image_overlay | 77 | - |",
                                  "| 7 (shot_007) | 0:28.0 | quality problem | footage | 58 | anachronism, repeated clip |",
                                  "| 8 (shot_008) | 0:31.0 | other | image | 91 | - |"]


def test_empty_review(settings, root):
    md = export.markdown(store.load(settings, PID))
    assert "Reviewed 0 of 8 shots." in md and "| not reviewed | 8 |" in md
    assert "None ticked." in md and "## Shots with a comment (0)" in md
    assert "## Project comment\n\n(none)" in md
    assert "not good without a comment" not in md


def test_write_export(settings, root):
    _fill(settings)
    path = export.write_export(settings, PID)
    assert path == root / "review.md"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Review: ") and "Judge threshold 70." in text


@pytest.mark.parametrize("sec,text", [(None, "-"), (0, "0:00.0"), (4.0, "0:04.0"), (9.96, "0:10.0"),
                                      (65.25, "1:05.2"), (59.96, "1:00.0"), (119.97, "2:00.0"), (-3, "0:00.0"),
                                      (600.4, "10:00.4")])
def test_clock(sec, text):
    assert export.clock(sec) == text


def test_summary_and_text(settings, root):
    review = _fill(settings)
    s = export.summary(review)
    assert s["reviewed"] == 6 and s["unreviewed"] == 2 and s["comments"] == 3 and s["has_project_comment"]
    assert s["verdicts"] == {"good": 1, "wrong_picture": 1, "wrong_timing": 1, "quality_problem": 2, "other": 1}
    assert s["rejects"] == {"burned_in_text": 0, "watermark_logo": 1, "talking_head": 0, "anachronism": 1,
                            "repeated_clip": 1}
    text = export.summary_text(review).splitlines()
    assert text[0] == f"Review of {PID}: reviewed 6 of 8 shots"
    assert text[1] == ("Verdicts: good 1, wrong picture 1, wrong timing 1, quality problem 2, other 1, "
                       "not reviewed 2")
    assert text[2] == "Hard rejects on 2 shot(s):"
    assert text[3] == "  shot 2 at 0:04.0: watermark or logo"
    assert text[4] == "  shot 7 at 0:28.0: anachronism, repeated clip"
    assert text[5] == "Shots with a comment: 3; project comment: yes"
    empty = export.summary_text(store.load(settings, "demo").model_copy(update={"shots": []}))
    assert "Hard rejects: none ticked" in empty
