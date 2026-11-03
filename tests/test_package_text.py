"""Packaging text: brand language in prompts, natural heuristic description and fact-led titles."""
from __future__ import annotations

from types import SimpleNamespace

from test_package_outputs import _script_and_narration
from vidbush.models import Beat, Brand, Scene, Script
from vidbush.render import package as pk
from vidbush.render import thumbnails as th


class RecordingSeats:
    def __init__(self) -> None:
        self.systems: list[str] = []

    def llm_json(self, role, task, system, user, stage):  # noqa: ANN001
        self.systems.append(system)
        return None


def test_prompts_honour_brand_language() -> None:
    seats = RecordingSeats()
    ctx = SimpleNamespace(seats=seats, brand=Brand(id="b", language="de"), project=SimpleNamespace(title="x"))
    script, _ = _script_and_narration()
    pk.make_titles(script, script.title, ctx)
    pk.make_description(script, script.title, ctx)
    th.thumbnail_texts(["Der Kanal"], ctx)
    assert len(seats.systems) == 3 and all(s.endswith("Write every text in German.") for s in seats.systems)
    assert pk.language_name("pt-BR") == "Portuguese" and pk.language_name("xx") == "xx"


def test_heuristic_description_reads_naturally() -> None:
    script = Script(title="The Golden Gate Bridge", scenes=[
        Scene(id="s0", title="Hook", purpose="hook", beats=[Beat(id="b0", text="Fog rolled in over San Francisco.")]),
        Scene(id="s1", title="The build", purpose="act", beats=[Beat(id="b1", text="Work began in 1933.")]),
        Scene(id="s2", title="Opening day", purpose="outro", beats=[Beat(id="b2", text="It opened in May.")])])
    desc = pk.heuristic_description(script, script.title)
    assert "We follow the build, then opening day." in desc
    assert "The build" not in desc.split("\n\n")[1]


def test_titles_are_fact_led_without_mid_title_the() -> None:
    script = Script(title="The Golden Gate Bridge", scenes=[Scene(id="s", title="a", purpose="act", beats=[
        Beat(id="b", text="Crews strung 80,000 miles of wire in 1935 across the strait.")])])
    titles = pk.heuristic_titles(script, script.title)
    assert "80,000 Miles: The Story of the Golden Gate Bridge" in titles
    assert "Why the Golden Gate Bridge Still Matters" in titles
    assert not any("Untold" in t or " of The " in t for t in titles)
    assert len(titles) >= 3
