"""Art layer Part C: illustration picks, prompt, variants and selection (no network)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from vidbush.art import common, illustrate as I
from vidbush.models import ArtProfile, Candidate, Narration, Shot, ShotList, ShotType, SourceKind, Word


def _shot(i: int, start: float, dur: float = 5.0, typ: ShotType = ShotType.ai_image, intent: str = "x",
          sel: Candidate | None = None) -> Shot:
    s = Shot(id=f"s{i}", beat_id="b", scene_id=f"sc{i}", start=start, end=start + dur, duration=dur, type=typ,
             intent=intent)
    if sel is not None:
        s.candidates.append(sel)
        s.selected = sel.id
    return s


class Seats:
    def __init__(self, reply=None):
        self.reply, self.fallbacks = reply, []
        self.client = SimpleNamespace(call=lambda role, stage, specs, fn, note: fn(specs[0]))

    def llm_json(self, *a, **k):
        return self.reply

    def note_fallback(self, role, stage, scene_id, reason):
        self.fallbacks.append(scene_id)


def _ctx(reply=None, words=(), tmp: Path | None = None, variants: int = 2):
    narr = Narration(audio_path="", duration=100, words=[Word(text=t, start=s, end=s + 0.5) for t, s in words])
    return SimpleNamespace(seats=Seats(reply), stage="visuals", maybe_load=lambda k, m: narr,
                           profile=SimpleNamespace(art=ArtProfile(illustration_variants=variants)),
                           brand=SimpleNamespace(illustration_style="Gouache style line."),
                           settings=SimpleNamespace(judge=SimpleNamespace(threshold=70.0)),
                           log=SimpleNamespace(warn=lambda *a, **k: None), work=tmp)


def test_choose_validates_and_spaces():
    real = Candidate(id="r", source=SourceKind.youtube, media="video", score=85)
    shots = [_shot(0, 0), _shot(1, 10), _shot(2, 40, sel=real), _shot(3, 60, dur=2.0), _shot(4, 80),
             _shot(5, 120, typ=ShotType.motion_graphic)]
    sl = ShotList(project_id="p", duration=200, shots=shots)
    reply = {"picks": [{"shot_id": "nope", "brief": "a"}, {"shot_id": "s0", "category": "scale", "brief": "a"},
                       {"shot_id": "s0", "brief": "dup"}, {"shot_id": "s1", "category": "bogus", "brief": "b"},
                       {"shot_id": "s2", "brief": "real"}, {"shot_id": "s3", "brief": "short"},
                       {"shot_id": "s4", "brief": "c"}, {"shot_id": "s5", "brief": "mg"}]}
    picks = I.choose(sl, _ctx(reply), budget=2)
    assert [p["shot_id"] for p in picks] == ["s0", "s4"]  # s1 is 10 s from s0, spaced out
    assert picks[0]["category"] == "scale"
    # budget forces the spaced-out one back in
    assert [p["shot_id"] for p in I.choose(sl, _ctx(reply), budget=3)] == ["s0", "s4", "s1"]
    assert I.choose(sl, _ctx(reply), budget=0) == []


def test_low_scoring_real_material_may_be_replaced():
    weak = Candidate(id="w", source=SourceKind.wikimedia, score=40)
    sl = ShotList(project_id="p", duration=50, shots=[_shot(0, 0, sel=weak)])
    assert I.choose(sl, _ctx({"picks": [{"shot_id": "s0", "brief": "x"}]}), 1)[0]["shot_id"] == "s0"


def test_heuristic_fallback_notes_fallback(monkeypatch):
    shots = [_shot(0, 0, dur=4, intent="Glacier carving at Mendenhall Glacier slowly"),
             _shot(1, 30, dur=6, intent="Ice age land bridge"), _shot(2, 60, dur=6)]
    sl = ShotList(project_id="p", duration=80, shots=shots)
    ctx = _ctx(None, words=[("how", 1.0), ("years", 31.0), ("ago", 31.5)])
    monkeypatch.setattr(ctx.seats, "note_fallback", lambda *a: ctx.seats.fallbacks.append(a[2]))
    picks = I.choose(sl, ctx, budget=5)
    assert [p["shot_id"] for p in picks] == ["s1", "s0"]  # longest first; s2 has no matching words
    assert picks[1]["brief"] == "Glacier carving at slowly"
    assert ctx.seats.fallbacks == ["sc1", "sc0"]


def test_prompt_for_style_first_names_stripped_rules():
    p = I.prompt_for("A trader named John Smith crossing the Bering land bridge", "My style.")
    assert p.startswith("My style.")
    assert "John Smith" not in p and "Illustration:" in p
    assert "No text" in p and "not a photograph" in p and chr(0x2014) not in p
    assert I.prompt_for("a river", "").startswith(I.DEFAULT_STYLE)
    assert chr(0x2014) not in I.prompt_for("x", "") + I.ILLUSTRATE_SYSTEM


def _fake(monkeypatch, tmp_path, scores, texts=None, spec=True):
    calls = []

    def gen(spec, prompt, out, model="", seed=None, **k):
        calls.append((out.name, seed, model))
        Image.new("RGB", (64, 36), (10, 20, 30)).save(out)

    import vidbush.art.elevate as E
    import vidbush.judge.judge as J
    monkeypatch.setattr(E, "generate", gen)
    monkeypatch.setattr(common, "elevate_spec", lambda ctx: SimpleNamespace(name="e") if spec else None)
    monkeypatch.setattr(common, "find_text", lambda p: (texts or {}).get(Path(p).name.split("_16x9")[0], []))
    it = iter(scores)
    monkeypatch.setattr(J, "score_candidate", lambda shot, cand, ctx, narration="": next(it))
    return calls


def test_illustrate_picks_best_passing(monkeypatch, tmp_path):
    calls = _fake(monkeypatch, tmp_path, [(75.0, "ok", None), (90.0, "bad", "off_topic"), (80.0, "good", None)],
                  texts={"s0_ill_1": ["SIGN"]}, )
    ctx = _ctx(variants=4)
    shot = _shot(0, 0)
    rep = I.illustrate(shot, {"shot_id": "s0", "brief": "a glacier"}, ctx, tmp_path, "narr")
    assert [c[0] for c in calls] == ["s0_ill_0.png", "s0_ill_1.png", "s0_ill_2.png", "s0_ill_3.png"]
    assert len({c[1] for c in calls}) == 4 and calls[0][2] == "nano_banana_pro"
    assert [v["reject"] for v in rep["variants"]] == [None, "text in picture", "off_topic", None]
    assert rep["chosen"] == "s0_ill_3" and rep["candidate"].source == SourceKind.illustration
    assert Image.open(rep["variants"][0]["path"]).size == (1920, 1080)
    # resume: existing raw files are not generated again
    calls = _fake(monkeypatch, tmp_path, [(60.0, "", None)] * 4)
    rep = I.illustrate(shot, {"shot_id": "s0", "brief": "a glacier"}, ctx, tmp_path, "narr")
    assert calls == [] and rep["chosen"] is None  # nothing paid twice; 60 is under the threshold


def test_illustrate_failed_variant_and_no_seat(monkeypatch, tmp_path):
    _fake(monkeypatch, tmp_path, [(80.0, "", None)])
    import vidbush.art.elevate as E

    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(E, "generate", boom)
    rep = I.illustrate(_shot(0, 0), {"shot_id": "s0", "brief": "b"}, _ctx(variants=1), tmp_path, "")
    assert rep["chosen"] is None and rep["variants"][0]["reject"].startswith("failed")
    _fake(monkeypatch, tmp_path, [], spec=False)
    assert I.illustrate(_shot(0, 0), {"brief": "b"}, _ctx(), tmp_path, "") == {"skipped": "no Elevate image seat"}


def test_apply_to_shot_types():
    c = Candidate(id="ill", source=SourceKind.illustration, score=81, judge_reason="fits")
    for before, after in ((ShotType.footage, ShotType.ai_image), (ShotType.footage_overlay, ShotType.image_overlay),
                          (ShotType.image, ShotType.image), (ShotType.ai_image, ShotType.ai_image)):
        s = _shot(0, 0, typ=before)
        I.apply_to_shot(s, c)
        assert s.type == after and s.selected == "ill" and s.judge_score == 81 and s.judge_notes == "fits"


def test_lettered_artifact_briefs_are_dropped():
    from vidbush.art.illustrate import TEXT_ARTIFACT
    assert TEXT_ARTIFACT.search("A printed map of the United States with Alaska as an inset")
    assert TEXT_ARTIFACT.search("a newspaper headline")
    assert not TEXT_ARTIFACT.search("A dog team crossing cracked sea ice at dusk")
