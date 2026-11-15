"""AI stills: the scene brief grounds the prompt in the narration spoken during the shot, stub and multi-moment
intents never become prompts, named people never reach the image model, and a fixed-question vision check
fails a still that shows the wrong subject, count, setting, a collage, text, a named face or a cut head."""
from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from vidbush.judge import ai_checks
from vidbush.judge import director_loop as dl
from vidbush.models import Beat, Candidate, Narration, Scene, ScenePurpose, Script, Shot, ShotType, SourceKind, Word
from vidbush.visuals import ai_images as A

NARRATION = "In the dark the serum crossed Norton Sound by dog team while the wind blew"


def shot(**kw: Any) -> Shot:
    base = dict(id="shot_000", index=0, beat_id="b1", scene_id="s1", start=2.0, end=4.0, duration=2.0,
                type=ShotType.ai_image, intent="Archival footage of a sea, 1920s", must_show=["sea"])
    base.update(kw)
    return Shot(**base)


def with_story(ctx, title: str = "The Serum Run to Nome", gap: float = 0.5):
    """Script with one beat and narration words gap seconds apart starting at 0 s."""
    ctx.save("script", Script(title=title, scenes=[Scene(id="s1", title=title, purpose=ScenePurpose.act,
                                                         beats=[Beat(id="b1", text=NARRATION)])]))
    words = [Word(text=w, start=i * gap, end=i * gap + gap * 0.8, beat_id="b1") for i, w in enumerate(NARRATION.split())]
    ctx.save("narration", Narration(audio_path="n.wav", duration=len(words) * gap, words=words))
    return ctx


def fake_llm(ctx, monkeypatch, brief: Any):
    """Editor stand-in: the scene brief answer, and a safe prompt rewrite that echoes the prompt."""
    calls: list[tuple[str, str]] = []

    def llm_json(role, task, system, user, stage, temperature=0.6, timeout=0):
        calls.append((task, user))
        if task == "ai_scene_brief":
            return brief
        return {"prompt": user.splitlines()[0].removeprefix("Prompt: "), "removed": []}

    monkeypatch.setattr(ctx.seats, "llm_json", llm_json)
    return calls


def image_prompts(ctx, monkeypatch) -> list[str]:
    sent: list[str] = []
    real = ctx.seats.image

    def image(prompt, out_path, stage, negative_prompt="", note="image"):
        sent.append(prompt)
        return real(prompt, out_path, stage, negative_prompt=negative_prompt, note=note)

    monkeypatch.setattr(ctx.seats, "image", image)
    return sent


BRIEF = {"agent": "one musher and a team of sled dogs", "action": "crossing", "object": "frozen sea ice",
         "place": "Norton Sound, Alaska: flat treeless sea ice under a dark sky", "era": "1925", "people": [],
         "moment": "A musher and his dog team crossing frozen Norton Sound in the dark, then arriving in Nome"}


# ---------------------------------------------------------------- stubs, one moment, the narration window

@pytest.mark.parametrize("intent, title, stub", [
    ("Archival footage of a sea, 1920s", "", True),
    ("The Serum Run to Nome", "The Serum Run to Nome", True),
    ("Photograph of the serum run, 1925", "The Serum Run", True),
    ("Musher driving a dog team across sea ice at night", "The Serum Run", False),
])
def test_stub_intents(intent, title, stub):
    assert A.is_stub(shot(intent=intent), title) is stub


@pytest.mark.parametrize("text", [
    "A dog team on the ice, then a crowd in Nome",
    "A dog team on the ice; then a crowd in Nome",
    "A dog team on the ice, followed by a crowd in Nome",
    "A dog team on the ice before cutting to a crowd in Nome",
    "A dog team on the ice and then a crowd in Nome",
    "First a dog team on the ice, later a crowd in Nome",
])
def test_one_moment_keeps_the_first_moment(text):
    assert A.one_moment(text) == "a dog team on the ice" or A.one_moment(text) == "A dog team on the ice"


def test_one_moment_leaves_a_single_moment_alone():
    assert A.one_moment("A musher resting his dogs at a roadhouse.") == "A musher resting his dogs at a roadhouse."


def test_window_text_takes_the_words_of_the_shot_and_widens_when_short(make_ctx):
    ctx = with_story(make_ctx())
    words = NARRATION.split()
    assert A.shot_window_text(shot(start=0.0, end=3.0), ctx) == " ".join(words[:6])
    short = A.shot_window_text(shot(start=2.0, end=2.4), ctx)  # one word spoken, widened to at least six
    assert words[4] in short and len(short.split()) >= 6
    assert A.shot_window_text(shot(start=2.0, end=2.4), make_ctx(project_id="t2")) == ""


def test_window_widening_stops_at_three_seconds(make_ctx):
    ctx = with_story(make_ctx(), gap=4.0)  # words 4 s apart: 3 s each side still finds few
    got = A.shot_window_text(shot(start=8.0, end=8.5), ctx).split()
    assert got == NARRATION.split()[1:3]


# ---------------------------------------------------------------- scene brief

@pytest.mark.parametrize("answer", [None, "junk", {"moment": ["a", "list"]}, {"moment": "   "},
                                    RuntimeError("editor down")])
def test_brief_falls_back_to_must_show_and_the_narration(make_ctx, monkeypatch, answer):
    ctx = with_story(make_ctx())

    def llm_json(*a, **k):
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(ctx.seats, "llm_json", llm_json)
    b = A.scene_brief(shot(), ctx)
    assert b["stub"] is True and b["place"] == "" and b["era"] == "" and b["people"] == []
    assert "Norton Sound" in b["moment"] and "sea, 1920s" not in b["moment"]
    real = A.scene_brief(shot(intent="Musher driving a dog team across sea ice, then a crowd cheering",
                              must_show=["dog team", "musher"]), ctx)
    assert real["agent"] == "dog team" and real["object"] == "musher"
    assert real["moment"] == "Musher driving a dog team across sea ice" and real["stub"] is False


def test_brief_user_text_puts_the_shot_narration_first_and_drops_a_stub_intent(make_ctx, monkeypatch):
    ctx = with_story(make_ctx())
    calls = fake_llm(ctx, monkeypatch, BRIEF)
    b = A.scene_brief(shot(), ctx)
    user = calls[0][1]
    assert user.splitlines()[0].startswith("Narration during this shot (primary): ")
    assert "Video title (context for the place): The Serum Run to Nome" in user
    assert "Archival footage" not in user and "Must show" not in user
    assert b["moment"] == "A musher and his dog team crossing frozen Norton Sound in the dark"


# ---------------------------------------------------------------- prompt

def test_prompt_has_place_period_and_the_letterbox_band_and_no_stub_text(make_ctx, monkeypatch):
    ctx = with_story(make_ctx())
    ctx.stage = "judge"
    fake_llm(ctx, monkeypatch, BRIEF)
    sent = image_prompts(ctx, monkeypatch)
    cand = A.generate_ai_candidate(shot(), ctx)
    p = sent[0]
    assert "Place: Norton Sound, Alaska" in p and "Period: 1925." in p
    assert "2.39:1 frame" in p and "top or bottom quarter" in p and "natural light" in p
    assert "a sea" not in p and "1920s" not in p and "Clearly showing" not in p and "arriving in Nome" not in p
    assert cand.prompt == p and cand.brief["place"].startswith("Norton Sound") and cand.brief["stub"] is True
    for neg in ("split panels", "collage", "diptych", "grid", "multiple frames", "border"):
        assert neg in A.BASE_NEGATIVE


def test_letterbox_off_composes_for_the_full_frame(make_ctx, monkeypatch):
    ctx = with_story(make_ctx())
    ctx.profile.letterbox.enabled = False
    assert A.letterbox_ratio(ctx) is None and "16:9" in A.house_style(None)
    ctx.profile.letterbox.enabled, ctx.profile.letterbox.ratio = True, 2.0
    assert "2:1 frame" in A.house_style(A.letterbox_ratio(ctx))


def test_an_operator_prompt_is_kept_whole_with_the_house_style_and_the_place_lock(make_ctx, monkeypatch):
    ctx = with_story(make_ctx())
    fake_llm(ctx, monkeypatch, BRIEF)
    sent = image_prompts(ctx, monkeypatch)
    # the operator's text carries their prompt delta, so it is never cut
    A.generate_ai_candidate(shot(), ctx, prompt="A lighthouse on a rocky coast. Stormier sky")
    assert sent[0].startswith("A lighthouse on a rocky coast. Stormier sky") and "2.39:1" in sent[0]
    assert "Place: Norton Sound" in sent[0] and "Period: 1925." in sent[0]


@pytest.mark.parametrize("text", ["Miners back then lived in canvas tents on the beach",
                                  "The town, by then a boomtown of tents, filled the beach",
                                  "A dog team followed by a crowd of children runs down Front Street",
                                  "The serum arrived, then 21 children were treated"])
def test_one_moment_leaves_a_single_moment_whole(text):
    assert A.one_moment(text) == text


def test_brief_reads_a_numeric_era_and_people_given_as_objects(make_ctx, monkeypatch):
    ctx = with_story(make_ctx())
    fake_llm(ctx, monkeypatch, dict(BRIEF, era=1925, people=[{"name": "Leonhard Seppala"}, 7]))
    b = A.scene_brief(shot(), ctx)
    assert b["era"] == "1925" and b["people"] == ["Leonhard Seppala"]


def test_the_place_lock_survives_a_silent_editor(make_ctx, monkeypatch):
    ctx = with_story(make_ctx())
    sent = image_prompts(ctx, monkeypatch)

    def llm_json(role, task, system, user, stage, **k):
        return BRIEF if task == "ai_scene_brief" else None  # the name removal gives no answer

    monkeypatch.setattr(ctx.seats, "llm_json", llm_json)
    A.generate_ai_candidate(shot(), ctx)
    assert sent[0].endswith("Place: Norton Sound, Alaska: flat treeless sea ice under a dark sky. Period: 1925.")


def test_named_people_flag_the_shot_and_never_reach_the_image_model(make_ctx, monkeypatch):
    ctx = with_story(make_ctx())
    brief = dict(BRIEF, people=["Larry Csonka"],
                 moment="Larry Csonka's hunting rifle leaning on a cabin wall on the tundra")
    fake_llm(ctx, monkeypatch, brief)
    sent = image_prompts(ctx, monkeypatch)
    s = shot(intent="Larry Csonka at his hunting camp on the tundra", must_show=["hunting camp"])
    A.generate_ai_candidate(s, ctx)
    A.generate_ai_candidate(s, ctx)
    assert s.flags.count("real_person_photo_wanted") == 1
    assert all("Csonka" not in p and "Larry" not in p for p in sent) and "hunting rifle" in sent[0].lower()


def test_safe_prompt_names_seed_the_names_to_remove(make_ctx, monkeypatch):
    ctx = make_ctx()
    monkeypatch.setattr(ctx.seats, "llm_json", lambda *a, **k: None)
    out = A.safe_prompt("Csonka beside a cabin on the tundra at dusk.", ctx, names=["Csonka"])
    assert "Csonka" not in out and "cabin on the tundra" in out


# ---------------------------------------------------------------- fixed-question check

def still(tmp_path, w: int = 1600, h: int = 900) -> str:
    p = tmp_path / "shot_000_1.png"
    Image.new("RGB", (w, h), (90, 100, 110)).save(p)
    return str(p)


def ai_cand(path: str, **kw: Any) -> Candidate:
    return Candidate(id="shot_000_ai_image_1", source=SourceKind.ai_image, local_path=path, media="image",
                     keyframes=[path], brief=dict(BRIEF, stub=True), **kw)


def answers(**fails: str) -> dict[str, Any]:
    return {"answers": {q: {"ok": q not in fails, "reason": fails.get(q, "fine")} for q in ai_checks.QUESTIONS},
            "_mock": False}


def vision(ctx, monkeypatch, replies: list[Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def vision_json(role, task, system, prompt, images, stage, intent="", details=None):
        calls.append(dict(task=task, prompt=prompt, images=images, details=details))
        r = replies[min(len(calls), len(replies)) - 1]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(ctx.seats, "vision_json", vision_json)
    return calls


def test_all_ok_passes_and_stores_the_answers(make_ctx, monkeypatch, tmp_path):
    ctx = with_story(make_ctx())
    calls = vision(ctx, monkeypatch, [answers()])
    c = ai_cand(still(tmp_path))
    out = ai_checks.check_still(shot(), c, ctx)
    assert out["failed"] == [] and set(out["answers"]) == set(ai_checks.QUESTIONS)
    assert c.reject_reason is None and ai_checks.FAIL_FLAG not in c.flags and c.checks == out
    assert calls[0]["task"] == "ai_check" and calls[0]["details"] == ["high"]
    assert "Norton Sound" in calls[0]["prompt"] and "Brief era: 1925" in calls[0]["prompt"]


def test_one_not_ok_answer_fails_the_still_with_its_reason(make_ctx, monkeypatch, tmp_path):
    ctx = with_story(make_ctx())
    vision(ctx, monkeypatch, [answers(subject="four men carrying a stretcher, no dog team")])
    c = ai_cand(still(tmp_path), reject_reason="anachronism")
    ai_checks.check_still(shot(), c, ctx)
    assert c.reject_reason == "anachronism; ai_check: subject: four men carrying a stretcher, no dog team"
    assert ai_checks.FAIL_FLAG in c.flags and c.checks["failed"] == ["subject"]


@pytest.mark.parametrize("reply", [{"score": 40, "_mock": True}, None])
def test_mock_reply_and_other_sources_are_skipped(make_ctx, monkeypatch, tmp_path, reply):
    ctx = with_story(make_ctx())
    vision(ctx, monkeypatch, [dict(answers(subject="bad"), _mock=True) if reply else answers(subject="bad")])
    c = ai_cand(still(tmp_path))
    if reply is None:
        c.source = SourceKind.local
    assert ai_checks.check_still(shot(), c, ctx) == {} and c.reject_reason is None


@pytest.mark.parametrize("reply", [{"score": 80, "_mock": False}, {"answers": "yes", "_mock": False},
                                   {"answers": {"subject": {"reason": "no ok key"}}, "_mock": False},
                                   ["not", "a", "dict"], RuntimeError("vision down")])
def test_unusable_reply_never_fails_the_still(make_ctx, monkeypatch, tmp_path, reply):
    ctx = with_story(make_ctx())
    vision(ctx, monkeypatch, [reply])
    c = ai_cand(still(tmp_path))
    out = ai_checks.check_still(shot(), c, ctx)
    assert c.reject_reason is None and ai_checks.FAIL_FLAG not in c.flags
    if out:
        assert out["failed"] == [] and set(out["unanswered"]) == set(ai_checks.QUESTIONS)


def test_band_crop_has_the_letterbox_aspect(make_ctx, monkeypatch, tmp_path):
    ctx = with_story(make_ctx())
    calls = vision(ctx, monkeypatch, [answers()])
    ai_checks.check_still(shot(), ai_cand(still(tmp_path)), ctx)
    band = calls[0]["images"][0]
    assert band.endswith("shot_000_1_band.jpg")
    with Image.open(band) as im:
        assert im.size[0] / im.size[1] == pytest.approx(2.39, abs=0.01)
    ctx.profile.letterbox.enabled = False
    ai_checks.check_still(shot(), ai_cand(still(tmp_path)), ctx)
    assert calls[1]["images"][0].endswith("shot_000_1.png")


def test_add_ai_regenerates_after_a_check_failure_with_the_reason_as_context_only(make_ctx, monkeypatch):
    ctx = with_story(make_ctx())
    ctx.stage = "judge"
    calls = fake_llm(ctx, monkeypatch, BRIEF)
    sent = image_prompts(ctx, monkeypatch)
    checks = iter([answers(subject="four men carrying a stretcher"), answers()])
    monkeypatch.setattr(ctx.seats, "vision_json", lambda role, task, system, prompt, images, stage, **k: (
        next(checks) if task == "ai_check" else {"score": 90, "reason": "fits", "rejects": [], "_mock": False}))
    s = shot(must_show=["dog team pulling a sled"])
    best = dl.add_ai(s, ctx)
    takes = [c for c in s.candidates if c.source == SourceKind.ai_image]
    assert len(takes) == 2 and best is takes[1] and ai_checks.FAIL_FLAG in takes[0].flags
    assert "ai_check: subject: four men carrying a stretcher" in takes[0].reject_reason
    retry_user = [u for t, u in calls if t == "ai_prompt_safety"][-1]
    hits = [line for line in retry_user.splitlines() if "stretcher" in line]
    assert len(hits) == 1 and hits[0].startswith("Problem: ")
    assert "stretcher" not in sent[1]
