"""render_all_motion: candidates, overlays, cache reuse, stale overlay cleanup."""
from __future__ import annotations

from pathlib import Path

from vidbush.mg import builders as b
from vidbush.models import Candidate, Shot, ShotList, ShotType, SourceKind


def tiny_shotlist() -> ShotList:
    shots = [
        Shot(id="s1", beat_id="b1", start=0, end=1, duration=1.0, type=ShotType.motion_graphic, intent="cost",
             motion=b.number_card_spec("$375M", "Cost", count_up=True, duration=6.0),
             candidates=[Candidate(id="s1_motion_1", source=SourceKind.motion, local_path="old.mp4")]),
        Shot(id="s2", beat_id="b1", start=1, end=2, duration=1.0, type=ShotType.footage_overlay, intent="who",
             overlay=b.lower_third_spec("Theodore Roosevelt", "President")),
        Shot(id="s3", beat_id="b2", start=2, end=3, duration=1.0, type=ShotType.footage, intent="ships",
             overlay=b.callout_spec("Stale"), overlay_path="work/mg/stale.mov"),
        Shot(id="s4", beat_id="b2", start=3, end=4, duration=1.0, type=ShotType.motion_graphic, intent="Broken chart",
             motion=b.bar_chart_spec(["a"], [1]).model_copy(update={"payload": {"labels": "x"}})),
    ]
    return ShotList(project_id="t1", width=320, height=180, duration=4.0, shots=shots)


def test_render_all_motion_and_cache(make_ctx) -> None:
    from vidbush.mg.renderer import render_all_motion
    ctx = make_ctx()
    ctx.settings.render.workers = 2
    sl, outs = render_all_motion(tiny_shotlist(), ctx)
    s1, s2, s3, s4 = sl.shots
    motion = [c for c in s1.candidates if c.source == SourceKind.motion]
    assert len(motion) == 1 and s1.selected == "s1_motion_1"
    cand = motion[0]
    assert cand.media == "video" and cand.licence == "generated" and cand.score is None
    assert Path(cand.local_path).exists() and Path(cand.keyframes[0]).exists()
    assert Path(cand.local_path).parent == ctx.work / "mg"
    assert s2.overlay_path.endswith(".mov") and Path(s2.overlay_path).exists()
    assert s3.overlay_path == ""
    assert s4.selected == "s4_motion_1" and Path(s4.selected_candidate().local_path).exists()
    assert len(outs) == 3
    mtimes = {p: p.stat().st_mtime_ns for p in outs}
    sl2, outs2 = render_all_motion(tiny_shotlist(), ctx)
    assert outs2 == outs
    assert all(p.stat().st_mtime_ns == mtimes[p] for p in outs2)
    log = ctx.log.path.read_text()
    assert '"cached": 3' in log and "invalid bar_chart payload" in log


def test_cache_key_includes_renderer_and_version(make_ctx) -> None:
    from vidbush.mg import renderer as r
    ctx = make_ctx()
    spec = b.number_card_spec("10", "Ten", duration=2.0)
    base = r.mg_key(spec, ctx.settings)
    other = ctx.settings.model_copy(deep=True)
    other.render.renderer = "remotion"
    assert r.mg_key(spec, other) != base
    assert r.mg_key(spec, ctx.settings) == base


def test_failed_render_is_cached_as_fallback_and_retried(make_ctx, monkeypatch) -> None:
    """A text-card fallback never lands under the real key; the next run retries and then replaces it."""
    from vidbush.mg import renderer as r
    from vidbush.render import mg_python
    from vidbush.models import MotionKind
    ctx = make_ctx()
    ctx.settings.render.workers = 1
    real = mg_python.render_spec_python
    broken = {"on": True}

    def flaky(spec, out, alpha, settings):  # noqa: ANN001
        if broken["on"] and spec.kind != MotionKind.text_overlay:
            raise RuntimeError("drawing bug")
        return real(spec, out, alpha, settings)

    monkeypatch.setattr(mg_python, "render_spec_python", flaky)
    sl = ShotList(project_id="t1", width=320, height=180, duration=1.0, shots=[
        Shot(id="s1", beat_id="b1", start=0, end=1, duration=1.0, type=ShotType.motion_graphic, intent="cost",
             motion=b.number_card_spec("$375M", "Cost", duration=1.0))])
    sl, outs = r.render_all_motion(sl, ctx)
    assert outs[0].name.endswith(".fallback.mp4") and outs[0].exists()
    planned = outs[0].with_name(outs[0].name.replace(".fallback", ""))
    assert not planned.exists()
    mtime = outs[0].stat().st_mtime_ns
    sl, outs2 = r.render_all_motion(sl, ctx)
    assert outs2 == outs and outs2[0].stat().st_mtime_ns == mtime, "old fallback reused while still failing"
    broken["on"] = False
    sl, outs3 = r.render_all_motion(sl, ctx)
    assert outs3 == [planned] and planned.exists() and not outs[0].exists()
    assert sl.shots[0].selected_candidate().local_path == str(planned)
