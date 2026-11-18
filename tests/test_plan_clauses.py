"""Part 1 of the clause-first planner: clauses from punctuation and word timings."""
from vidbush.plan import clauses as C
from vidbush.plan.clauses import segment

from test_plan_helpers import build, seq


def texts(script, nar):
    return [c.text for c in segment(script, nar)]


def test_splits_at_sentence_ends_and_beat_changes():
    script, nar = build([[seq("The ship sailed north today.", 0) + seq("It reached Alaska in May.", 2.5),
                          seq("Then winter came very fast", 5.0)]])
    cl = segment(script, nar)
    assert [c.text for c in cl] == ["The ship sailed north today.", "It reached Alaska in May.",
                                    "Then winter came very fast"]
    assert [c.beat for c in cl] == ["b001", "b001", "b002"]
    assert all(c.scene == "s01" for c in cl)


def test_comma_split_when_both_sides_long():
    s, n = build([[seq("On the long road north, the crew found nothing at all.", 0)]])
    assert texts(s, n) == ["On the long road north,", "the crew found nothing at all."]


def test_no_comma_split_when_one_side_short():
    s, n = build([[seq("Later, the crew found nothing at all to eat.", 0)]])
    assert len(texts(s, n)) == 1
    s, n = build([[seq("The crew found nothing at all to eat, sadly.", 0)]])
    assert len(texts(s, n)) == 1


def test_split_before_conjunction_when_both_sides_long():
    s, n = build([[seq("The crew built a long road and the town grew up around it.", 0)]])
    assert texts(s, n) == ["The crew built a long road", "and the town grew up around it."]


def test_no_conjunction_split_when_one_side_short():
    s, n = build([[seq("Boats and the town grew up around the harbor.", 0)]])
    assert len(texts(s, n)) == 1


def test_spoken_list_one_clause_per_item():
    words = [("They", 0.0, 0.4), ("carried", 0.5, 0.9), ("flour,", 1.0, 1.9), ("salt,", 2.0, 3.1),
             ("and", 3.2, 3.5), ("rifles.", 3.6, 4.5)]
    s, n = build([[words]])
    cl = segment(s, n)
    assert [c.text for c in cl] == ["They carried flour,", "salt,", "and rifles."]
    assert all(c.list_item for c in cl)


def test_short_sentence_merges_backward():
    s, n = build([[seq("The ship sailed into the harbor at dawn.", 0) + seq("It sank.", 4.0)]])
    assert texts(s, n) == ["The ship sailed into the harbor at dawn. It sank."]


def test_lead_in_with_comma_merges_forward():
    # a beat change cuts the sentence after "Meanwhile,": the lead-in still belongs to what follows
    s, n = build([[seq("The winter was long and very cold.", 0), [("Meanwhile,", 3.5, 4.2)],
                   seq("the town grew up around the harbor.", 4.5)]])
    cl = segment(s, n)
    assert [c.text for c in cl] == ["The winter was long and very cold.",
                                    "Meanwhile, the town grew up around the harbor."]
    assert cl[1].beat == "b002"


def test_short_piece_never_merges_across_scenes():
    s, n = build([[seq("The winter was very long that year.", 0)],
                  [seq("Spring came.", 3.6) + seq("The ice broke up on the river.", 4.6)],
                  [seq("The end.", 8.5)]])
    cl = segment(s, n)
    assert [(c.scene, c.text) for c in cl] == [
        ("s01", "The winter was very long that year."),
        ("s02", "Spring came. The ice broke up on the river."),
        ("s03", "The end."),
    ]


def test_clauses_tile_the_timeline_and_cut_before_first_word():
    s, n = build([[seq("On the long road north, the crew found nothing at all.", 0),
                   seq("The crew built a long road and the town grew up around it.", 6.0)],
                  [seq("The winter was very long that year.", 13.0)]])
    cl = segment(s, n)
    assert len(cl) == 5
    assert cl[0].start == 0.0
    for a, b in zip(cl, cl[1:]):
        assert a.end == b.start
    assert cl[-1].end == round(n.duration + 1.0, 3)
    for c in cl[1:]:
        assert c.start == round(n.words[c.words[0]].start - 0.1, 3)
        assert c.speech_start == n.words[c.words[0]].start


def test_save_load_round_trip(tmp_path):
    s, n = build([[seq("On the long road north, the crew found nothing at all.", 0)]])
    cl = segment(s, n)
    p = C.save(cl, tmp_path / "work" / "clauses.json")
    assert C.load(p) == cl
