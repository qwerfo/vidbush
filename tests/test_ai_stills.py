"""AI generated stills: spoken years set the era, prompts keep every must_show item, the rewrite keeps content,
every still is judged against the narration and regenerated with the judge's reason, text subjects are flagged."""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from vidbush.judge import director_loop as dl
from vidbush.judge import qa
from vidbush.judge.judge import judge_prompt
from vidbush.models import Beat, Candidate, Scene, ScenePurpose, Script, Shot, ShotList, ShotType, SourceKind, \
    VisualIdea
from vidbush.shots import ideas as I
from vidbush.shots.assembler import ai_prompt
from vidbush.visuals import ai_images

SERUM = "Twenty mushers and their dog teams pulling sleds carried the serum to Nome."


def shot(i: int = 0, typ: ShotType = ShotType.footage, **kw: Any) -> Shot:
    base = dict(id=f"shot_{i:03d}", index=i, beat_id="b1", scene_id="s1", start=float(i), end=float(i + 1),
                duration=1.0, type=typ, intent="Dog team pulling a sled with a musher riding behind",
                must_show=["team of sled dogs pulling a sled", "musher riding the sled"])
    base.update(kw)
    return Shot(**base)


def script(*texts: str) -> Script:
    beats = [Beat(id=f"b{i + 1}", text=t) for i, t in enumerate(texts)]
    return Script(title="Serum run", scenes=[Scene(id="s1", title="Serum run", purpose=ScenePurpose.act, beats=beats)])


# ---------------------------------------------------------------- spoken years

@pytest.mark.parametrize("text, years", [
    ("In January of nineteen twenty-five the town was cut off.", [1925]),
    ("Alaska was bought in eighteen sixty-seven.", [1867]),
    ("By two thousand eight the road was paved.", [2008]),
    ("In two thousand and twelve it closed.", [2012]),
    ("The town grew in the nineteen twenties.", [1920]),
    ("Built in nineteen oh five.", [1905]),
    ("Twenty-five dogs ran.", []),
    ("They left at ten thirty that night.", []),
    ("fifteen twenty minutes later", []),
    ("about sixteen seventeen year olds", []),
    ("twenty thirty miles a day", []),
    ("In nineteen twenty-five dogs carried the serum.", [1925]),
])
def test_spoken_years_are_years(text, years):
    assert I.extract_years(text) == years


def test_spelled_to_digits_reads_a_spoken_year_as_one_year():
    assert I.spelled_to_digits("nineteen twenty-five") == "1925"
    assert "2030" not in I.spelled_to_digits("twenty thirty miles a day")
    assert I.spelled_to_digits("twenty-five dogs") == "25 dogs"


def test_spoken_year_sets_the_era_and_the_archival_prompt_style():
    bcs = I.beat_contexts(script("In January of nineteen twenty-five, diphtheria reached Nome.", SERUM).scenes,
                          None, "serum run")
    assert [bc.era for bc in bcs] == [1925, 1925]
    idea = VisualIdea(beat_id="b2", family="still", intent="Dog team on the trail")
    text = ai_prompt(idea, bcs[1])
    assert text.startswith("Black and white archival documentary photograph") and "Period-accurate" in text


# ---------------------------------------------------------------- prompts keep content

def test_prompt_builders_keep_every_must_show_item():
    items = ["Gunnar Kaasen", "Balto", "sled dogs", "heavy snow", "dark sky"]
    assert "dark sky" in ai_prompt(VisualIdea(beat_id="b1", family="still", intent="Musher at night",
                                              must_show=items), None)
    assert "dark sky" in ai_images.build_prompt(shot(must_show=items, ai_prompt=""))


def _ctx_with_rewrite(make_ctx, monkeypatch, answer: str):
    ctx = make_ctx()
    seen: dict[str, str] = {}

    def fake_llm(role, task, system, user, stage, temperature=0.6, timeout=0):
        seen["user"] = user
        return {"prompt": answer}

    monkeypatch.setattr(ctx.seats, "llm_json", fake_llm)
    return ctx, seen


def test_rewrite_that_drops_a_keep_item_returns_the_original(make_ctx, monkeypatch):
    original = "Leonhard Seppala with a team of dogs pulling a sled across sea ice."
    ctx, seen = _ctx_with_rewrite(make_ctx, monkeypatch, "An anonymous musher running beside one dog on sea ice.")
    out = ai_images.safe_prompt(original, ctx, keep=["dog team pulling a sled"], narration=SERUM)
    assert out == original
    assert "Must keep: dog team pulling a sled" in seen["user"] and f"Narration: {SERUM}" in seen["user"]


def test_rewrite_that_only_removes_names_is_accepted(make_ctx, monkeypatch):
    original = "Leonhard Seppala with a team of dogs pulling a sled across sea ice."
    rewrite = "An anonymous musher with a team of dogs pulling a sled across sea ice."
    ctx, _ = _ctx_with_rewrite(make_ctx, monkeypatch, rewrite)
    assert ai_images.safe_prompt(original, ctx, keep=["Leonhard Seppala", "team of dogs pulling a sled"]) == rewrite


def test_rewrite_keeps_counts_and_may_drop_text_objects():
    lost = ai_images.lost_keep_words("20 dogs pulling a sled.", "A dog pulling a sled.", ["20 dogs pulling a sled"])
    assert lost == ["20"]
    assert ai_images.lost_keep_words("One husky beside a sled.", "A husky beside a sled.", ["one husky"]) == ["one"]
    assert ai_images.lost_keep_words("A man holding a sign reading HELP.", "A man holding a blank board.",
                                     ["a man holding a sign reading HELP"]) == []
    # a capitalised first word is content unless the next word is capitalised too
    lost = ai_images.lost_keep_words("Sled dogs resting.", "Huskies resting.", ["Sled dogs resting"])
    assert lost == ["Sled", "dogs"]


def test_names_possessives_and_acronyms_are_never_lost_content():
    old = "Leonhard Seppala's lead dog Togo beside him on the US trail near I-80."
    new = "An anonymous musher's lead dog beside a man on a trail near a highway."
    keep = ["Seppala's lead dog", "Togo beside him", "US trail", "I-80"]
    assert ai_images.lost_keep_words(old, new, keep, narration="Seppala trusted Togo.") == []
    assert ai_images.lost_keep_words("Kaasen and Balto.", "A musher and a dog.", ["Kaasen", "Balto"],
                                     narration="Then Kaasen drove Balto on.") == []


def test_count_synonyms_pronouns_and_auxiliaries_are_not_lost():
    assert ai_images.lost_keep_words("A single loaded tanker truck.", "One loaded tanker truck.",
                                     ["single loaded tanker truck"]) == []
    assert ai_images.lost_keep_words("Notes being taken by a crowd around him at sun setting.",
                                     "A crowd taking notes around a man at dusk.",
                                     ["notes being taken", "crowd around him", "sun setting"]) == ["sun"]
    assert ai_images.lost_keep_words("A sled flipping, hands digging, waves hitting tiled walls.",
                                     "A sled flips, hands dig, waves hit tile walls.",
                                     ["sled flipping", "hands digging", "waves hitting tiled walls"]) == []
    assert ai_images.lost_keep_words("The musher's hands.", "The musher’s gloved hands.", ["musher's hands"]) == []


def test_a_dog_team_rewritten_to_one_dog_is_still_lost():
    assert ai_images.lost_keep_words("A dog team pulling a sled.", "One dog pulling a sled.",
                                     ["dog team pulling a sled"]) == ["team"]


def test_lost_words_get_one_named_retry(make_ctx, monkeypatch):
    ctx = make_ctx()
    answers = ["Musher beside one dog on sea ice.", "Musher with a dog team pulling a sled on sea ice."]
    asked: list[str] = []

    def fake_llm(role, task, system, user, stage, temperature=0.6, timeout=0):
        asked.append(user)
        return {"prompt": answers[len(asked) - 1]}

    monkeypatch.setattr(ctx.seats, "llm_json", fake_llm)
    assert ai_images.safe_prompt("A dog team pulling a sled on sea ice.", ctx,
                                 keep=["dog team pulling a sled"]) == answers[1]
    assert len(asked) == 2 and "Must keep these words the last rewrite lost: team" in asked[1]


@pytest.mark.parametrize("answer", ["An anonymous musher running beside one dog on sea ice.", None, "boom"])
def test_no_fallback_prompt_sends_a_keep_item_name(make_ctx, monkeypatch, answer):
    ctx = make_ctx()
    keep = ["Leonhard Seppala", "Togo leading a team of dogs pulling a sled", "Seppala's sled"]

    def fake_llm(*a, **k):
        if answer == "boom":
            raise RuntimeError("editor down")
        return {"prompt": answer} if answer else None

    monkeypatch.setattr(ctx.seats, "llm_json", fake_llm)
    original = "Leonhard Seppala with Togo leading a team of dogs pulling Seppala's sled across sea ice."
    out = ai_images.safe_prompt(original, ctx, keep=keep, narration="Seppala ran Togo across the ice.")
    assert not {"Leonhard", "Seppala", "Togo"} & set(re.findall(r"[A-Za-z]+", out))
    assert "team of dogs pulling" in out and "sled across sea ice" in out


@pytest.mark.parametrize("text, is_text", [
    ("A soldier writing a letter home by candlelight", False),
    ("Photographers document the war", False),
    ("A record label office", False),
    ("Prospectors map the creek", False),
    ("A statement of grief, a widow at a grave", False),
    ("Map-making tools on a drafting table", False),
    ("Dalton Highway map showing Coldfoot", True),
    ("A map of the Iditarod trail", True),
    ("Newspaper headline announcing statehood", True),
    ("The text of the telegram", True),
    ("Texas Our Texas sheet music", True),
    ("A bar chart of gold output", True),
])
def test_readable_text_means_a_text_artifact_not_a_word(text, is_text):
    assert ai_images.needs_readable_text(shot(intent=text, must_show=[])) is is_text


# ---------------------------------------------------------------- the judge hears the narration

def test_judge_and_qa_prompts_carry_the_narration(make_ctx, monkeypatch):
    s = shot()
    c = Candidate(id="c1", source=SourceKind.ai_image, media="image", title="AI generated still")
    assert f"Narration: {SERUM}" in judge_prompt(s, c, SERUM)
    ctx = make_ctx()
    ctx.save("script", script(SERUM))
    seen: list[str] = []
    monkeypatch.setattr(ctx.seats, "vision_json", lambda role, task, system, prompt, images, stage, **k: (
        seen.append(prompt) or {"score": 80, "reason": "ok", "rejects": []}))
    qa.score_post_render(ShotList(project_id="t1", duration=1.0, shots=[s]), {s.id: ["f.jpg"]}, ctx)
    assert f"Narration: {SERUM}" in seen[0]


def test_ai_candidate_title_is_not_its_prompt(make_ctx):
    ctx = make_ctx()
    cand = ai_images.generate_ai_candidate(shot(ai_prompt="A dog team pulling a sled. Natural light."), ctx)
    assert cand.title == "AI generated still"


# ---------------------------------------------------------------- judged and regenerated

def ai_vision(scores: list[float], prompts: Optional[list[str]] = None):
    """vision_json stand-in: each AI still gets the next score in the list (the last one repeats)."""
    left = list(scores)

    def vision_json(role, task, system, prompt, images, stage, intent="", details=None) -> dict[str, Any]:
        if prompts is not None:
            prompts.append(prompt)
        score = left.pop(0) if len(left) > 1 else left[0]
        return {"score": score, "reason": f"no sled in take scored {score}", "rejects": [], "_mock": False}

    return vision_json


def image_prompts(ctx, monkeypatch) -> list[str]:
    sent: list[str] = []
    real = ctx.seats.image

    def image(prompt, out_path, stage, negative_prompt="", note="image"):
        sent.append(prompt)
        return real(prompt, out_path, stage, negative_prompt=negative_prompt, note=note)

    monkeypatch.setattr(ctx.seats, "image", image)
    return sent


def test_failed_still_is_regenerated_with_the_judge_reason_and_the_passing_take_wins(make_ctx, monkeypatch):
    ctx = make_ctx()
    ctx.stage = "judge"
    sent = image_prompts(ctx, monkeypatch)
    monkeypatch.setattr(ctx.seats, "vision_json", ai_vision([50, 80]))
    out = dl.judge_and_select(ShotList(project_id="t1", duration=1.0, shots=[shot(typ=ShotType.ai_image)]), ctx)
    s = out.shots[0]
    assert [c.score for c in s.candidates] == [50, 80]
    assert s.selected == s.candidates[1].id and "no_match_accepted" not in s.flags
    assert "Must clearly show: team of sled dogs pulling a sled" in sent[1] and "no sled" not in sent[1]
    notes = [e.note for e in ctx.tracker.report.entries if e.role == "image"]
    assert notes == ["image", "image retry"]


def test_judge_reason_is_rewrite_context_and_never_a_negative(make_ctx, monkeypatch):
    """The judge describes the wrong picture: its 'no sled' must not reach the negative prompt or the prompt."""
    ctx = make_ctx()
    ctx.stage = "judge"
    s = shot(typ=ShotType.ai_image, must_show=["a team of dogs pulling a sled with a musher riding behind"],
             ai_prompt="A dog team on sea ice, no modern buildings. Natural light.")
    negatives: list[str] = []
    real = ctx.seats.image
    monkeypatch.setattr(ctx.seats, "image", lambda prompt, out_path, stage, negative_prompt="", note="image": (
        negatives.append(negative_prompt) or real(prompt, out_path, stage, negative_prompt=negative_prompt)))
    users: list[str] = []
    monkeypatch.setattr(ctx.seats, "llm_json", lambda role, task, system, user, stage, **k: users.append(user))
    reason = "Shows a man holding one husky on a leash, no sled and no team of dogs"
    ai_images.generate_ai_candidate(s, ctx, problem=reason)
    assert f"Problem: {reason}" in users[0] and "husky on a leash" not in users[0].splitlines()[0]
    for word in ("sled", "team", "dogs"):
        assert word not in negatives[0]
    assert "modern buildings" in negatives[0]  # the base prompt's own negations still apply


def test_a_mock_judge_score_buys_no_retry(make_ctx, monkeypatch):
    ctx = make_ctx()
    ctx.stage = "judge"
    monkeypatch.setattr(ctx.seats, "vision_json", lambda role, task, system, prompt, images, stage, **k: {
        "score": 40, "reason": "mock score image statistics", "rejects": [], "_mock": True})
    s = dl.judge_and_select(ShotList(project_id="t1", duration=1.0, shots=[shot(typ=ShotType.ai_image)]),
                            ctx).shots[0]
    assert len([c for c in s.candidates if c.source == SourceKind.ai_image]) == 1


def test_retries_stop_at_the_limit_and_the_best_take_is_accepted(make_ctx, monkeypatch):
    ctx = make_ctx()
    ctx.stage = "judge"
    monkeypatch.setattr(ctx.seats, "vision_json", ai_vision([55, 62, 58]))
    s = dl.judge_and_select(ShotList(project_id="t1", duration=1.0, shots=[shot(typ=ShotType.ai_image)]),
                            ctx).shots[0]
    assert len(s.candidates) == 1 + ctx.settings.judge.ai_retries == 3
    assert s.judge_score == 62 and "no_match_accepted" in s.flags


def test_text_subject_is_never_converted_to_an_ai_still(make_ctx, monkeypatch):
    ctx = make_ctx()
    ctx.stage = "judge"
    still = str(Path(__file__).resolve().parents[1] / "fixtures" / "sample_assets" / "sample_image_01.jpg")
    c = Candidate(id="shot_000_local_1", source=SourceKind.local, local_path=still, media="image", width=2400,
                  height=1350, keyframes=[still])
    s = shot(intent="Dalton Highway map showing Coldfoot", must_show=["map"], candidates=[c], type=ShotType.image)
    monkeypatch.setattr(ctx.seats, "vision_json", ai_vision([40]))
    monkeypatch.setattr(dl, "_requery", lambda shot, ctx: "requery found 0 new")
    out = dl.judge_and_select(ShotList(project_id="t1", duration=1.0, shots=[s]), ctx).shots[0]
    assert all(x.source != SourceKind.ai_image for x in out.candidates)
    assert out.selected == c.id and dl.NEEDS_MOTION_FLAG in out.flags and out.type == ShotType.image


def test_packaging_writes_the_qa_report_before_the_thumbnails(monkeypatch):
    from vidbush.jobs import stages
    order: list[str] = []
    monkeypatch.setattr("vidbush.judge.qa.build_qa", lambda sl, ctx: order.append("qa") or [])

    def package(sl, ctx):
        order.append("package")
        raise RuntimeError("stopped during thumbnails")

    monkeypatch.setattr("vidbush.render.package.package", package)
    ctx = SimpleNamespace(load=lambda key, model: ShotList(project_id="t1", duration=1.0, shots=[]))
    with pytest.raises(RuntimeError):
        stages.stage_packaging(ctx)
    assert order == ["qa", "package"]


def test_mock_roles_from_packaging_reach_the_qa_report(make_ctx):
    from vidbush.util import read_json, write_json
    ctx = make_ctx()
    report = qa.make_report(ShotList(project_id="t1", duration=1.0, shots=[]), ctx, 70.0, {}, 1.0, 1.0, [])
    report.mock_used = []
    write_json(ctx.root / "qa_report.json", report)
    ctx.seats.mock_roles.add("image")  # a thumbnail fell back to mock after build_qa
    qa.refresh_mock_used(ctx)
    used = read_json(ctx.root / "qa_report.json")["mock_used"]
    assert "image" in used and used == sorted(used)
    assert "Mock seats: " in (ctx.root / "qa_report.md").read_text(encoding="utf-8")
