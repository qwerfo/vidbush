"""Fallback queries carry the shot's own words, never the video topic appended to them."""
from __future__ import annotations

from vidbush.models import Beat, Scene, ScenePurpose
from vidbush.shots import ideas as I


def _bc(text: str, topic: str = "Zebra Topic"):
    beat = Beat(id="b1", text=text)
    scene = Scene(id="s1", title="Chapter", purpose=ScenePurpose.act, beats=[beat])
    return I.BeatContext(beat=beat, scene=scene, era=1931, topic=topic)


def test_fallback_queries_never_append_the_topic():
    bc = _bc("The steam locomotive crossed the wooden trestle above the frozen river at dawn.")
    ent = I.beat_entities(bc)
    ideas = I.footage_ideas(bc, ent) + I.still_ideas(bc, ent)
    queries = [q for idea in ideas for q in idea.queries]
    assert queries
    assert not any("zebra" in q.lower() for q in queries), queries


def test_topic_still_used_when_the_beat_names_no_subject():
    bc = _bc("And then it all changed.")
    ent = I.beat_entities(bc)
    if not ent.nouns and not ent.people:
        queries = [q for idea in I.footage_ideas(bc, ent) for q in idea.queries]
        assert any("zebra" in q.lower() for q in queries)
