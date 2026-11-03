"""Synthetic Script and Narration builders shared by the shots tests (no other engineers' code)."""
from __future__ import annotations

import random

from vidbush.models import (Beat, Brand, FactCard, Narration, Project, ProjectPaths, Research, Scene, ScenePurpose,
                            Script, StyleProfile, VideoFormat, Word)

DOC_SENTENCES = [
    "In 1904, thousands of workers arrived at Colon to dig through the jungle.",
    "Steam shovels loaded rock onto trains in the Culebra Cut, day and night.",
    "Ships sailing from New York to San Francisco once had to round Cape Horn.",
    "The voyage of 13,000 miles shrank to roughly 5,200 miles.",
    "Doctors fought yellow fever and malaria in the crowded camps near Panama City.",
    "The Gatun Locks lifted ships 85 feet up to Gatun Lake.",
    "Engineers poured concrete for the giant lock walls, block by block.",
    "By 1914, the first ships crossed from the Atlantic to the Pacific.",
    "The project cost the United States about $375 million.",
    "President Theodore Roosevelt visited the works in 1906 and rode a steam shovel.",
    "Mudslides buried the cut again and again, and the crews kept digging.",
    "Today tankers and container ships pass through every single day.",
]


def doc_script(n_scenes: int = 6, beats_per_scene: int = 6, seed: int = 0, sentences: int = 2) -> Script:
    """Documentary script: hook, setup, acts, outro; beats drawn from DOC_SENTENCES."""
    rng = random.Random(seed)
    purposes = [ScenePurpose.hook, ScenePurpose.setup] + [ScenePurpose.act] * max(0, n_scenes - 3) + [ScenePurpose.outro]
    scenes, k = [], 0
    for si, purpose in enumerate(purposes[:n_scenes]):
        beats = []
        for _ in range(beats_per_scene):
            k += 1
            text = " ".join(rng.sample(DOC_SENTENCES, sentences))
            beats.append(Beat(id=f"b{k:03d}", text=text, facts_used=["f1", "f2"] if "miles" in text else []))
        scenes.append(Scene(id=f"s{si + 1:02d}", title=f"Chapter {si + 1}", purpose=purpose, beats=beats))
    return Script(title="The Panama Canal: Ocean to Ocean", format=VideoFormat.documentary, scenes=scenes)


def top10_script(items: int = 10, beats_per_item: int = 4) -> Script:
    """Top 10 script: hook, ten item scenes counting down, outro."""
    names = ["Hoover Dam", "Erie Canal", "Golden Gate Bridge", "Brooklyn Bridge", "Panama Canal", "Empire State Building",
             "Mount Rushmore", "Transcontinental Railroad", "Grand Coulee Dam", "Suez Canal", "Kitty Hawk", "Alamo"]
    scenes = [Scene(id="s00", title="Intro", purpose=ScenePurpose.hook,
                    beats=[Beat(id="b000", text="These ten projects changed America forever. Workers risked their lives."),
                           Beat(id="b001", text="Ships, trains and cranes moved mountains of earth across the country.")])]
    k = 2
    for i in range(items):
        n = items - i
        beats = []
        for j in range(beats_per_item):
            beats.append(Beat(id=f"b{k:03d}", text=f"The {names[i]} was built by thousands of workers in {1900 + 3 * i}. "
                                                   f"It took {3 + j} years and cost {10 + i} million dollars."))
            k += 1
        scenes.append(Scene(id=f"s{i + 1:02d}", title=f"Number {n}: {names[i]}", purpose=ScenePurpose.item,
                            item_number=n, beats=beats))
    scenes.append(Scene(id="s99", title="Outro", purpose=ScenePurpose.outro,
                        beats=[Beat(id=f"b{k:03d}", text="Those were the greatest engineering feats in history."),
                               Beat(id=f"b{k + 1:03d}", text="Which one surprised you the most? Thanks for watching.")]))
    return Script(title="Top 10 Engineering Feats", format=VideoFormat.top10, scenes=scenes)


def narrate(script: Script, wpm: float = 150.0, seed: int = 0) -> Narration:
    """Word timings at about wpm with small gaps between words and pauses between beats."""
    rng = random.Random(seed)
    words, spans, t = [], {}, 0.35
    per_word = 60.0 / wpm
    for beat in script.beats():
        start = t
        for w in beat.text.split():
            d = per_word * rng.uniform(0.7, 0.9)
            words.append(Word(text=w, start=round(t, 3), end=round(t + d, 3), beat_id=beat.id))
            t += per_word * rng.uniform(0.95, 1.1)
        spans[beat.id] = [round(start, 3), round(t, 3)]
        t += 0.4
    return Narration(audio_path="narration.wav", duration=round(t + 0.2, 3), words=words, beat_spans=spans)


def research() -> Research:
    """Two fact cards with comparable numbers."""
    return Research(topic="The Panama Canal", facts=[
        FactCard(id="f1", claim="The sea voyage from New York to San Francisco was about 13,000 miles around Cape Horn.",
                 source_title="Panama Canal", numbers=[{"value": 13000, "unit": "miles", "label": "around Cape Horn"}]),
        FactCard(id="f2", claim="Through the canal the voyage is about 5,200 miles.", source_title="Panama Canal",
                 numbers=[{"value": 5200, "unit": "miles", "label": "through the canal"}]),
        FactCard(id="f3", claim="The United States spent about $375 million on the canal.", source_title="Panama Canal",
                 numbers=[{"value": 375000000, "unit": "dollars", "label": "US construction cost"}]),
    ])


def project(pid: str = "p1") -> Project:
    return Project(id=pid, input_mode="script", paths=ProjectPaths(root="/tmp/p", work="/tmp/p/work", sources="/tmp/p/s",
                                                                   segments="/tmp/p/g", thumbnails="/tmp/p/t"))


def profile(**kw) -> StyleProfile:
    return StyleProfile(id="test", **kw)


def brand() -> Brand:
    return Brand(id="default", niche_notes="Calm documentary for an older US audience.")


def six_minute_script() -> Script:
    """About 900 words: roughly six minutes of narration at 150 wpm."""
    return doc_script(n_scenes=7, beats_per_scene=10, seed=3, sentences=1)


def test_helpers_build_six_minutes() -> None:
    nar = narrate(six_minute_script())
    assert 330 <= nar.duration <= 420
