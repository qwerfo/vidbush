"""Spoken numbers in narration words match the digits labels and anchors use."""
import pytest

from vidbush.models import Word
from vidbush.plan.numbers import normalise_number, spoken_numbers
from vidbush.shots.ideas import anchor_time, phrase_times


def said(text: str) -> list[tuple[int, int, str]]:
    return spoken_numbers(text.split())


@pytest.mark.parametrize("text,digits", [
    ("nineteen fifty-nine", "1959"),
    ("nineteen twenty-five", "1925"),
    ("eighteen sixty-seven", "1867"),
    ("nineteen oh five", "1905"),
    ("two thousand seven", "2007"),
    ("two thousand and eleven", "2011"),
    ("twenty eleven", "2011"),
    ("seven hundred thirty-three thousand, three hundred ninety-one", "733391"),
    ("seven point six", "7.6"),
    ("twenty-nine", "29"),
])
def test_whole_run(text, digits):
    assert said(text) == [(0, len(text.split()) - 1, digits)]


def test_run_inside_sentence_with_punctuation():
    toks = ["When", "Alaska", "became", "a", "state", "in", "nineteen", "fifty-nine,", "it"]
    assert spoken_numbers(toks) == [(6, 7, "1959")]


def test_twenty_five_is_a_count_not_a_year():
    assert said("twenty five dogs") == [(0, 1, "25")]


def test_digit_tokens_normalised():
    assert spoken_numbers(["In", "1959,", "about", "2,000", "people", "paid", "$77."]) == [
        (1, 1, "1959"), (3, 3, "2000"), (6, 6, "77")]


def test_no_merge_across_sentence_end():
    assert said("it was nineteen. Fifty-nine more came") == [(2, 2, "19"), (3, 3, "59")]


def test_one_is_returned():
    assert said("one of the dogs") == [(0, 0, "1")]


def test_separate_small_numbers_do_not_add():
    assert said("seven, eight") == [(0, 0, "7"), (1, 1, "8")]


@pytest.mark.parametrize("label,digits", [("1,959", "1959"), ("7.6", "7.6"), ("$77", "77"), ("NOME, 1925", "1925"),
                                          ("no number", "")])
def test_normalise_number(label, digits):
    assert normalise_number(label) == digits


def _words(texts: list[str]) -> list[Word]:
    return [Word(text=t, start=float(i), end=i + 0.9, beat_id="b001") for i, t in enumerate(texts)]


STATE = ["When", "Alaska", "became", "a", "state", "in", "nineteen", "fifty-nine,", "it"]


def test_year_label_finds_spoken_year():
    words = _words(STATE)
    assert phrase_times("1959", words) == [6.0]
    assert anchor_time("1959", words, 0.0, 10.0) == 6.0


def test_place_and_year_label():
    words = _words(["The", "serum", "reached", "Nome", "in", "nineteen", "twenty-five."])
    assert phrase_times("NOME, 1925", words) == [3.0]


def test_non_number_phrase_unchanged():
    words = _words(STATE)
    assert phrase_times("became a state", words) == [2.0]
    assert phrase_times("Alaska", words) == [1.0]
