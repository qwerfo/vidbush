"""AI generated stills: spoken years set the era, prompts keep every must_show item, the rewrite keeps content,
every still is judged against the narration and regenerated with the judge's reason, text subjects are flagged."""
from __future__ import annotations

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
])
def test_spoken_years_are_years(text, years):
    assert I.extract_years(text) == years


def test_spelled_to_digits_reads_a_spoken_year_as_one_year():
    assert I.spelled_to_digits("nineteen twenty-five") == "1925"
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
    assert "Must clearly show: team of sled dogs pulling a sled" in sent[1] and "no sled in take scored 50" in sent[1]
    notes = [e.note for e in ctx.tracker.report.entries if e.role == "image"]
    assert notes == ["image", "image retry"]


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
