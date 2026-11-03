"""Caption line breaks keep names, numbers with their nouns, and avoid weak line endings."""
from __future__ import annotations

from test_captions_lines import _words
from vidbush.models import CaptionStyle
from vidbush.render import captions as cap


def _texts(text: str) -> list[str]:
    return [ln.text for ln in cap.group_words(_words(text), CaptionStyle())]


def _joined_breaks(text: str) -> list[tuple[str, str]]:
    lines = cap.group_words(_words(text), CaptionStyle())
    return [(a.words[-1].text, b.words[0].text) for a, b in zip(lines, lines[1:])]


def test_does_not_split_capitalised_names() -> None:
    breaks = _joined_breaks("The contract went to Six Companies, a group of builders from the west coast.")
    assert ("Six", "Companies,") not in breaks


def test_keeps_numbers_with_their_nouns() -> None:
    for text in ("By the end about 22,000 workers died during the French attempt at the canal.",
                 "It cost two nations a fortune and took more than thirty years to finish."):
        for last, nxt in _joined_breaks(text):
            assert not (last[:1].isdigit() or last in cap.NUMBER_WORDS) or not nxt[:1].islower(), (last, nxt)
            assert last.lower() not in {"about", "two"}, (last, nxt)


def test_weak_endings_are_rare() -> None:
    text = ("And it cost tens of thousands of workers their lives. The dam created the largest artificial lake "
            "in the world at the time, and the locks lifted ships some twenty six metres above the sea.")
    ends = [t.split()[-1].lower() for t in _texts(text)[:-1]]
    assert sum(e in cap.WEAK_END for e in ends) <= 1, _texts(text)


def test_break_cost_rules() -> None:
    assert cap._break_cost("Six", "Companies") > 0.9
    assert cap._break_cost("22,000", "workers") > 0.9
    assert cap._break_cost("1914", "the") == 0.0
    assert cap._break_cost("canal.", "The") == 0.0
    assert cap._break_cost("of", "thousands") >= cap.WEAK_END_COST
