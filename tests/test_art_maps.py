"""Art layer maps: reference, registration, paint decisions, renderer hooks, planner (no network)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vidbush.art import hooks, mapref, paint, plan
from vidbush.models import (ArtProfile, Brand, MapMarker, MapPayload, MapRoute, MotionKind, MotionSpec, Shot,
                            ShotList, ShotType, StyleProfile)

COAST = MapPayload(center=[58.5, -135.0], zoom=5.0)  # Inside Passage: land and sea in view


def _big(p: MapPayload) -> np.ndarray:
    return cv2.resize(mapref.reference(p, 768, 432), (1376, 768))


def test_reference_is_text_free_and_two_colour_families():
    ref = mapref.reference(COAST, 384, 216)
    assert ref.shape == (216, 384, 3)
    land = (np.abs(ref.astype(int) - mapref.REF_LAND).sum(-1) < 30).mean()
    water = (np.abs(ref.astype(int) - mapref.REF_WATER).sum(-1) < 30).mean()
    assert land > 0.2 and water > 0.2


def test_view_key_uses_widest_zoom():
    p = MapPayload(center=[60, -150], zoom=4.0, zoom_end=4.6)
    assert mapref.view_key(p)["zoom"] == 4.0
    p2 = MapPayload(center=[60, -150], zoom=4.6, zoom_end=4.0)
    assert mapref.view_key(p2) == mapref.view_key(p)


def test_checkable_needs_land_and_water():
    assert mapref.checkable(mapref.land_mask(COAST, 192, 108))
    sea = MapPayload(center=[0.0, -140.0], zoom=6.0)  # open Pacific
    assert not mapref.checkable(mapref.land_mask(sea, 192, 108))
    r = mapref.register(_big(sea), sea)
    assert not r["ok"] and "no coastline" in r["reason"]


def test_register_passes_the_reference_and_small_canvas_refits():
    big = _big(COAST)
    assert mapref.register(big, COAST)["ok"]
    shifted = cv2.warpAffine(big, np.float32([[1.02, 0, 15], [0, 1.0, -8]]), (1376, 768),
                             borderMode=cv2.BORDER_REFLECT)
    r = mapref.register(shifted, COAST)
    assert r["ok"], r
    assert r["warp"] is not None


def test_register_rejects_a_moved_coast():
    big = _big(COAST)
    bad = big.copy()
    h, w = bad.shape[:2]
    # paint a wide block of sea over land that touches the coast: a peninsula vanished
    m = cv2.resize(mapref.land_mask(COAST, 768, 432), (w, h)) > 127
    ys, xs = np.nonzero(m)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    bad[cy - 90:cy + 90, cx - 90:cx + 90] = mapref.REF_WATER
    r = mapref.register(bad, COAST)
    assert not r["ok"]


def test_register_rejects_a_different_view():
    other = MapPayload(center=[61.0, -150.0], zoom=5.0)
    r = mapref.register(_big(other), COAST)
    assert not r["ok"]


def test_inland_water_is_not_a_coast():
    ref = np.zeros((100, 200), np.uint8)
    ref[:, 60:] = 255
    pred = ref.copy()
    pred[40:60, 150:170] = 0  # a lake deep inside the land
    out = mapref.inland_as_land(pred, ref)
    assert out[50, 160] == 255
    assert mapref.measure(out, ref)["worst_patch"] == 0.0


# ---------------------------------------------------------------- paint decisions

def _spec(**kw) -> MotionSpec:
    p = MapPayload(**{"center": [58.5, -135.0], "zoom": 5.0, **kw})
    return MotionSpec(kind=MotionKind.map, duration=6.0, payload=p.model_dump())


def test_paint_reason_keeps_precise_maps_plain():
    assert paint.paint_reason(_spec(markers=[MapMarker(lat=58.3, lon=-134.4, label="Juneau")]), "Juneau sits on the coast.") is None
    assert paint.paint_reason(_spec(routes=[MapRoute(points=[[58, -135], [59, -136]], via="sea")]), "") == "route"
    assert "shaded" in paint.paint_reason(_spec(shaded_regions=["Alaska"]), "")
    assert paint.paint_reason(_spec(), "It is 900 miles from here.") == "narration makes a precise point"
    assert paint.paint_reason(_spec(center=[0.0, -140.0], zoom=6.0), "Open ocean.") is not None


class _Seats:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def llm_json(self, role, task, system, user, stage, temperature=0.6):
        self.calls.append((role, task, user))
        return self.reply


def _ctx(reply=None, tmp=None, profile=None):
    log = SimpleNamespace(warn=lambda *a, **k: None, info=lambda *a, **k: None)
    return SimpleNamespace(seats=_Seats(reply), log=log, stage="visuals", profile=profile or StyleProfile(id="t"),
                           brand=Brand(id="b"), work=tmp)


def test_map_prompt_from_narration_strips_names_and_digits():
    p = MapPayload(center=[58.5, -135], zoom=5, markers=[MapMarker(lat=58.3, lon=-134.4, label="Juneau")])
    reply = {"land": "Steep rainforest mountains around Juneau with 3 glaciers", "water": "Dark fjords",
             "touches": ["humpback whales near Juneau", "fishing boats"], "light": "misty"}
    ctx = _ctx(reply)
    prompt, ok = paint.map_prompt(p, "Juneau can only be reached by sea or air.", ctx)
    assert ok
    assert "Juneau" not in prompt and "3" not in prompt
    assert "rainforest" in prompt and "whales" in prompt
    assert "No text" in prompt and "no borders" in prompt
    assert chr(0x2014) not in prompt
    assert "Juneau" in ctx.seats.calls[0][2]  # the art director sees the places; the painter never does


def test_map_prompt_fallback_when_director_silent():
    prompt, ok = paint.map_prompt(COAST, "", _ctx(None))
    assert not ok and "No text" in prompt and "Land:" in prompt


# ---------------------------------------------------------------- hooks

def _painted(tmp_path: Path, p: MapPayload, ok: bool = True, view=None) -> Path:
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[...] = (10, 200, 30)
    path = tmp_path / "base.png"
    cv2.imwrite(str(path), img)
    hooks.sidecar(path).write_text(json.dumps({"ok": ok, "view": view or mapref.view_key(p)}), encoding="utf-8")
    return path


def _render_frame(payload: MapPayload, i: int = 30):
    from vidbush.mg.mapdraw import MapComp
    from vidbush.mg.palettes import get_palette
    spec = MotionSpec(kind=MotionKind.map, duration=2.0, width=480, height=270, payload=payload.model_dump())
    comp = MapComp(spec, payload, get_palette("documentary_warm"), False)
    return comp, comp.render(i)


def test_painted_base_replaces_land_and_water(tmp_path):
    p = MapPayload(center=[58.5, -135.0], zoom=5.0)
    path = _painted(tmp_path, p)
    _, plain = _render_frame(p)
    comp, painted = _render_frame(p.model_copy(update={"art_base": str(path)}))
    assert getattr(comp.base, "art_overlay", None) is not None
    green = lambda f: float((f[..., 1].astype(int) - f[..., 0].astype(int) > 80).mean())  # noqa: E731
    assert green(painted) > 0.5 > green(plain)


@pytest.mark.parametrize("ok,view", [(False, None), (True, {"lat": 1.0, "lon": 2.0, "zoom": 3.0})])
def test_painted_base_ignored_when_sidecar_does_not_vouch_for_this_view(tmp_path, ok, view):
    p = MapPayload(center=[58.5, -135.0], zoom=5.0)
    path = _painted(tmp_path, p, ok=ok, view=view)
    comp, _ = _render_frame(p.model_copy(update={"art_base": str(path)}))
    assert getattr(comp.base, "art_overlay", None) is None


def test_moving_base_plays_clip_frames(tmp_path):
    p = MapPayload(center=[58.5, -135.0], zoom=5.0)
    base = _painted(tmp_path, p)
    clip = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 180))
    for k in range(48):
        f = np.zeros((180, 320, 3), np.uint8)
        f[...] = (200, 20, 20) if k > 20 else (20, 200, 20)  # BGR: blue after frame 20
        vw.write(f)
    vw.release()
    q = p.model_copy(update={"art_base": str(base), "art_clip": str(clip)})
    comp, early = _render_frame(q, 2)
    late = comp.render(50)
    assert late[..., 2].mean() > early[..., 2].mean() + 40  # RGB frame: blue rose


def test_moving_base_skipped_for_shaded_maps(tmp_path):
    p = MapPayload(center=[58.5, -135.0], zoom=5.0, shaded_regions=["Alaska"])
    base = _painted(tmp_path, p)
    comp, _ = _render_frame(p.model_copy(update={"art_base": str(base), "art_clip": str(tmp_path / "x.mp4")}))
    assert not hooks._clip_usable(comp)


# ---------------------------------------------------------------- planner

def test_art_profile_defaults_are_off():
    a = ArtProfile()
    assert a.illustrated_maps == 0 and a.illustrations == 0 and a.clip_every_s == 0
    assert StyleProfile(id="x").art.clip_every_s == 0


def test_apply_art_does_nothing_when_disabled(tmp_path):
    sl = ShotList(project_id="p", duration=5, shots=[Shot(id="s1", beat_id="b", start=0, end=5, duration=5,
                                                             type=ShotType.footage, intent="x")])
    ctx = _ctx(tmp=tmp_path)
    ctx.seats.force_mock = False
    ctx.seats.providers_for = lambda role: []
    assert plan.apply_art(sl, ctx) == {}


def _map_shot(sid: str, start: float, payload: MapPayload, full: bool = True) -> Shot:
    spec = MotionSpec(kind=MotionKind.map, duration=6.0, payload=payload.model_dump(),
                      background="solid" if full else "transparent")
    kw = {"motion": spec} if full else {"overlay": spec}
    return Shot(id=sid, beat_id="b", start=start, end=start + 6, duration=6,
                type=ShotType.motion_graphic if full else ShotType.footage_overlay, intent="map", **kw)


def test_paint_maps_share_repeat_window_and_reuse(tmp_path, monkeypatch):
    p = MapPayload(center=[58.5, -135.0], zoom=5.0, markers=[MapMarker(lat=58.3, lon=-134.4, label="Juneau")])
    shots = [_map_shot("a", 0, p), _map_shot("b", 60, p), _map_shot("c", 400, p),
             _map_shot("d", 100, p.model_copy(update={"routes": [MapRoute(points=[[58, -135], [59, -136]], via="sea")]}))]
    sl = ShotList(project_id="p", duration=500, shots=shots)
    calls = []

    def fake_paint(payload, window, ctx, out, tag, variants=0):
        calls.append(tag)
        return {"base": str(tmp_path / f"{tag}.png")}

    monkeypatch.setattr(plan.P, "paint_map", fake_paint)
    monkeypatch.setattr(plan.common, "elevate_spec", lambda ctx: None)
    prof = StyleProfile(id="t", art=ArtProfile(illustrated_maps=1.0, map_repeat_s=180))
    ctx = _ctx(tmp=tmp_path, profile=prof)
    report = {"maps": [], "illustrations": [], "clips": []}
    n = plan.paint_maps(sl, ctx, None, report)
    assert calls == ["a"]  # one painting for the view
    assert shots[0].motion.payload["art_base"].endswith("a.png")
    assert "art_base" not in shots[1].motion.payload or not shots[1].motion.payload["art_base"]  # within 180 s
    assert shots[2].motion.payload["art_base"].endswith("a.png")  # reused outside the window, no cost
    assert not shots[3].motion.payload.get("art_base")  # a route stays plain
    assert n == 2


def test_clip_kind_never_animates_real_photographs(tmp_path):
    from vidbush.models import Candidate, SourceKind
    still = tmp_path / "s.png"
    cv2.imwrite(str(still), np.zeros((10, 10, 3), np.uint8))
    s = Shot(id="s", beat_id="b", start=0, end=5, duration=5, type=ShotType.image, intent="x")
    for src, want in ((SourceKind.wikimedia, None), (SourceKind.archive, None), (SourceKind.ai_image, "ai_image"),
                      (SourceKind.illustration, "illustration")):
        s.candidates = [Candidate(id="c", source=src, local_path=str(still), media="image")]
        s.selected = "c"
        assert plan._clip_kind(s) == want


def test_repeat_window_checks_every_showing(tmp_path, monkeypatch):
    p = MapPayload(center=[58.5, -135.0], zoom=5.0, markers=[MapMarker(lat=58.3, lon=-134.4, label="Juneau")])
    shots = [_map_shot("a", 0, p), _map_shot("c", 400, p), _map_shot("i", 150, p, full=False)]
    sl = ShotList(project_id="p", duration=600, shots=shots)
    monkeypatch.setattr(plan.P, "paint_map", lambda payload, w, ctx, out, tag, variants=0: {"base": str(tmp_path / f"{tag}.png")})
    monkeypatch.setattr(plan.common, "elevate_spec", lambda ctx: None)
    ctx = _ctx(tmp=tmp_path, profile=StyleProfile(id="t", art=ArtProfile(illustrated_maps=1.0, map_repeat_s=180)))
    plan.paint_maps(sl, ctx, None, {"maps": [], "illustrations": [], "clips": []})
    assert shots[1].motion.payload["art_base"]  # 400 s after the first showing
    assert not shots[2].overlay.payload.get("art_base")  # 150 s after t=0, although 250 s from t=400


def test_rerun_clears_stale_art_and_keeps_existing_illustrations(tmp_path, monkeypatch):
    from vidbush.models import Candidate, SourceKind
    p = MapPayload(center=[58.5, -135.0], zoom=5.0, routes=[MapRoute(points=[[58, -135], [59, -136]], via="sea")],
                   art_base="old.png", art_clip="old.mp4")
    m = _map_shot("m", 0, p)
    s = Shot(id="s", beat_id="b", start=10, end=15, duration=5, type=ShotType.ai_image, intent="x",
             candidates=[Candidate(id="ill", source=SourceKind.illustration)], selected="ill")
    sl = ShotList(project_id="p", duration=20, shots=[m, s])
    monkeypatch.setattr(plan.common, "elevate_spec", lambda ctx: None)
    monkeypatch.setattr(plan.IL, "choose", lambda *a, **k: pytest.fail("no new picks when the budget is met"))
    ctx = _ctx(tmp=tmp_path, profile=StyleProfile(id="t", art=ArtProfile(illustrated_maps=1.0, illustrations=0.5)))
    rep = {"maps": [], "illustrations": [], "clips": []}
    plan.paint_maps(sl, ctx, None, rep)
    assert m.motion.payload["art_base"] == "" and m.motion.payload["art_clip"] == ""  # a route map is plain now
    assert plan.add_illustrations(sl, ctx, None, rep) == 1


def test_view_key_names_coast_detail():
    a = MapPayload(center=[60, -150], zoom=2.6)
    b = MapPayload(center=[60, -150], zoom=2.6, zoom_end=3.4)
    assert mapref.view_key(a)["zoom"] == mapref.view_key(b)["zoom"]
    assert mapref.view_key(a) != mapref.view_key(b)  # 110 m vs 50 m coast: never the same painting


def test_rerun_keeps_clip_slots_without_paying(tmp_path, monkeypatch):
    from vidbush.models import Candidate, SourceKind
    still = tmp_path / "s.png"
    cv2.imwrite(str(still), np.zeros((10, 10, 3), np.uint8))
    a = Shot(id="a", beat_id="b", start=0, end=5, duration=5, type=ShotType.ai_video, intent="x",
             candidates=[Candidate(id="a_art_clip", source=SourceKind.ai_video, media="video")], selected="a_art_clip")
    y = Shot(id="y", beat_id="b", start=5, end=25, duration=20, type=ShotType.ai_image, intent="x",
             candidates=[Candidate(id="c", source=SourceKind.ai_image, local_path=str(still))], selected="c")
    sl = ShotList(project_id="p", duration=60, shots=[a, y])
    ctx = _ctx(tmp=tmp_path, profile=StyleProfile(id="t", art=ArtProfile(clip_every_s=60)))
    ctx.seats.is_real = lambda role: True
    monkeypatch.setattr(plan.C, "make_clip", lambda *a, **k: pytest.fail("a kept clip slot must not be paid again"))
    assert plan.add_clips(sl, ctx, None, {"maps": [], "illustrations": [], "clips": []}) == 1
