"""Ken Burns parameters: bounds, determinism, framing, overrides."""
from __future__ import annotations

import pytest

from vidbush.models import Shot, ShotType
from vidbush.visuals.kenburns import center_bound, kenburns_params


def mk(sid: str, framing: str = "full", kb: dict | None = None) -> Shot:
    return Shot(id=sid, beat_id="b", start=0, end=5, duration=5, type=ShotType.image, intent="x",
                framing=framing, kenburns=kb)


def _inside(p: dict[str, float]) -> bool:
    for c, z in (("x0", "zoom0"), ("y0", "zoom0"), ("x1", "zoom1"), ("y1", "zoom1")):
        if abs(p[c] - 0.5) > center_bound(p[z]) + 1e-4:
            return False
    return True


@pytest.mark.parametrize("framing", ["full", "tight", "detail"])
def test_bounds_hold_for_many_shots(framing):
    for i in range(300):
        p = kenburns_params(mk(f"shot_{i:03d}", framing), 0.12)
        assert set(p) == {"zoom0", "zoom1", "x0", "y0", "x1", "y1"}
        assert 1.0 <= p["zoom0"] <= 1.12 + 1e-9 and 1.0 <= p["zoom1"] <= 1.12 + 1e-9
        assert p["zoom0"] != p["zoom1"]
        assert _inside(p)
        for t in (0.25, 0.5, 0.75):  # every interpolated frame stays inside too
            mid = {k: p[k + "0"] * (1 - t) + p[k + "1"] * t for k in ("zoom", "x", "y")}
            assert abs(mid["x"] - 0.5) <= center_bound(mid["zoom"]) + 1e-4
            assert abs(mid["y"] - 0.5) <= center_bound(mid["zoom"]) + 1e-4


def test_deterministic_and_varied():
    a = kenburns_params(mk("shot_007"), 0.12)
    assert a == kenburns_params(mk("shot_007"), 0.12)
    others = {tuple(kenburns_params(mk(f"shot_{i:03d}"), 0.12).values()) for i in range(20)}
    assert len(others) == 20
    params = [kenburns_params(mk(f"shot_{i:03d}"), 0.12) for i in range(40)]
    directions = {p["zoom1"] > p["zoom0"] for p in params}
    assert directions == {True, False}  # both push in and pull out occur


def test_tight_and_detail_start_higher():
    for i in range(50):
        full = kenburns_params(mk(f"s{i}", "full"), 0.2)
        tight = kenburns_params(mk(f"s{i}", "tight"), 0.2)
        detail = kenburns_params(mk(f"s{i}", "detail"), 0.2)
        assert min(tight["zoom0"], tight["zoom1"]) >= 1.0 + 0.2 * 0.5 - 1e-9
        assert min(detail["zoom0"], detail["zoom1"]) >= 1.0 + 0.2 * 0.7 - 1e-9
        assert tight["zoom0"] >= 1.1 - 1e-9 and detail["zoom0"] >= 1.14 - 1e-9
        assert min(full["zoom0"], full["zoom1"]) == 1.0


def test_zero_intensity_is_static_and_centered():
    p = kenburns_params(mk("shot_001"), 0.0)
    assert p == {"zoom0": 1.0, "zoom1": 1.0, "x0": 0.5, "y0": 0.5, "x1": 0.5, "y1": 0.5}


def test_override_wins_but_is_clamped():
    p = kenburns_params(mk("shot_001", kb={"zoom0": 1.3, "zoom1": 1.0, "x0": 0.9, "x1": 0.8}), 0.12)
    assert p["zoom0"] == 1.3 and p["zoom1"] == 1.0
    assert p["x0"] == pytest.approx(0.5 + center_bound(1.3), abs=1e-4)
    assert p["x1"] == 0.5  # zoom 1.0 allows no pan
