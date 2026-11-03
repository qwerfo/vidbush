"""Motion graphics renderer: every kind renders, exact frame counts, alpha, fallbacks, helpers."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from vidbush.mg import builders as b
from vidbush.mg.chartdraw import format_value, nice_ticks
from vidbush.mg.palettes import PALETTES, get_palette
from vidbush.mg.renderer import render_spec
from vidbush.mg.textdraw import coloured, fit_text, text_width, unpremultiply
from vidbush.models import MotionKind, MotionSpec
from vidbush.render.mg_python import build_comp, count_start, format_count, parse_number, recap_numbers

W, H = 640, 360


def small(spec: MotionSpec, duration: float = 1.0) -> MotionSpec:
    return spec.model_copy(update={"width": W, "height": H, "duration": duration})


def specs() -> dict[str, MotionSpec]:
    return {
        "map": b.map_spec([("Colon", 9.35, -79.9), ("Panama City", 8.98, -79.52)], route=True, shaded=["Panama"],
                          title="The Panama Canal"),
        "bar_chart": b.bar_chart_spec(["1914", "1950", "2020"], [1000, 6000, 13300], units="ships",
                                      title="Transits", highlight_index=2),
        "line_chart": b.line_chart_spec(["1914", "1950", "1980", "2020"], [1.1, 5.4, 13.2, 13.3], units="%",
                                        title="Share", highlight_index=2),
        "number_card": b.number_card_spec("$375M", "Construction cost", "In 1914 dollars", count_up=True),
        "title_card": b.title_card_spec("A Path Between the Seas", "Chapter one"),
        "lower_third": b.lower_third_spec("Theodore Roosevelt", "26th President"),
        "callout": b.callout_spec("The French lost 22,000 workers", "Yellow fever and malaria"),
        "highlight_box": b.highlight_box_spec(0.3, 0.3, 0.3, 0.3, label="Culebra Cut"),
        "text_overlay": b.text_overlay_spec("August 15, 1914", position="lower_left"),
        "arrow": b.arrow_spec(0.2, 0.3, 0.6, 0.6, label="Gatun Locks"),
        "recap_grid": b.recap_grid_spec(["One", "Two", "Three", "Four", "Five"], title="Top 5"),
    }


def probe_frames(path: Path) -> tuple[int, int, int, str]:
    cp = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                         "stream=nb_read_frames,width,height,pix_fmt", "-of", "json", str(path)],
                        capture_output=True, check=True)
    s = json.loads(cp.stdout)["streams"][0]
    return int(s["nb_read_frames"]), int(s["width"]), int(s["height"]), s["pix_fmt"]


def decode_rgba(path: Path, t: float, w: int = W, h: int = H) -> np.ndarray:
    cp = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t}", "-i", str(path), "-frames:v", "1", "-f", "rawvideo",
                         "-pix_fmt", "rgba", "-"], capture_output=True, check=True)
    return np.frombuffer(cp.stdout, np.uint8).reshape(h, w, 4)


def test_every_kind_has_a_spec() -> None:
    assert {s.kind for s in specs().values()} == set(MotionKind)


@pytest.mark.parametrize("name", list(specs().keys()))
def test_kind_renders_exact_frames(name: str, settings, tmp_path: Path) -> None:
    spec = small(specs()[name])
    alpha = spec.background == "transparent"
    out = render_spec(spec, tmp_path / (name + (".mov" if alpha else ".mp4")), alpha, settings)
    n, w, h, pix = probe_frames(out)
    assert (n, w, h) == (30, W, H)
    assert pix == ("argb" if alpha else "yuv420p")
    if not alpha:
        mid = decode_rgba(out, 0.5)
        assert mid[..., :3].std() > 4, "frame should not be flat"


def test_alpha_mov_is_transparent(settings, tmp_path: Path) -> None:
    spec = small(b.lower_third_spec("Theodore Roosevelt", "President"), 1.5)
    out = render_spec(spec, tmp_path / "lt.mov", True, settings)
    a = decode_rgba(out, 0.9)[..., 3]
    assert (a == 0).mean() > 0.5 and (a == 255).any()


def test_straight_alpha_composites_like_premultiplied() -> None:
    """Straight output composited by the usual formula matches the premultiplied frame (no dark halos)."""
    spec = small(b.callout_spec("Edges stay clean", "Soft shadow"), 1.5)
    prem = build_comp(spec, True).render(30).astype(np.float32)
    straight = unpremultiply(prem.astype(np.uint8)).astype(np.float32)
    bg = np.full(prem.shape[:2] + (3,), 255.0, np.float32)
    a = straight[..., 3:4] / 255.0
    via_straight = straight[..., :3] * a + bg * (1 - a)
    via_prem = prem[..., :3] + bg * (1 - prem[..., 3:4] / 255.0)
    assert np.abs(via_straight - via_prem).max() <= 3.0


def test_unpremultiply_restores_straight_colour() -> None:
    mask = np.linspace(0, 255, 256, dtype=np.uint8)[None, :].repeat(4, 0)
    straight = unpremultiply(coloured(mask, (250, 250, 250)))
    lit = straight[mask > 16]
    assert lit[:, :3].min() >= 238
    assert np.array_equal(straight[..., 3], mask)


def test_full_hd_frame(settings, tmp_path: Path) -> None:
    spec = b.title_card_spec("Full Frame", "Check").model_copy(update={"duration": 2 / 30})
    out = render_spec(spec, tmp_path / "hd.mp4", False, settings)
    n, w, h, _ = probe_frames(out)
    assert (n, w, h) == (2, 1920, 1080)


def test_invalid_payload_falls_back_to_text(settings, tmp_path: Path) -> None:
    spec = MotionSpec(kind=MotionKind.bar_chart, duration=1.0, width=W, height=H, payload={"title": "Broken"})
    comp = build_comp(spec, False)
    assert type(comp).__name__ == "TextOverlay"
    out = render_spec(spec, tmp_path / "bad.mp4", False, settings)
    assert probe_frames(out)[0] == 30


def test_remotion_failure_falls_back_to_python(settings, tmp_path: Path, monkeypatch) -> None:
    import vidbush.mg.renderer as r

    def boom(*a, **k):
        raise RuntimeError("no chromium")
    monkeypatch.setattr(r, "render_remotion", boom)
    settings.render.renderer = "remotion"
    notes: list[str] = []
    out = r.render_spec(small(specs()["callout"], 0.5), tmp_path / "c.mov", True, settings, notes)
    assert out.exists() and "remotion" in notes[0]


def test_palettes_complete() -> None:
    keys = ["bg", "bg2", "land", "land_border", "water", "text", "muted", "accent", "accent2", "route", "shade"]
    assert set(PALETTES) == {"documentary_warm", "documentary_cold", "top10_bold"}
    for p in PALETTES.values():
        for k in keys:
            assert len(p.rgb(k)) == 3
    assert get_palette("nope").name == "documentary_warm"


def test_number_parsing_and_count() -> None:
    assert parse_number("$375M") == ("$", 375.0, "M", 0, False)
    assert parse_number("22,000")[1:] == (22000.0, "", 0, True)
    assert parse_number("#3")[0] == "#"
    assert parse_number("Many") is None
    assert format_count(12345.0, 0, True) == "12,345"
    assert count_start(1914, "", "", False, 0) == 1854
    assert count_start(22000, "", "", True, 0) == 0


def test_value_formatting_and_ticks() -> None:
    assert format_value(22000, "ships") == "22,000 ships"
    assert format_value(375e6, "$") == "$375M"
    assert format_value(40, "percent") == "40%"
    t = nice_ticks(0, 13.3)
    assert t[0] <= 0 and t[-1] >= 13.3 and len(t) <= 7


def test_fit_text_never_overflows() -> None:
    long = "Supercalifragilistic " * 12
    f, lines = fit_text(long, "Bold", 120, 500, 2, 30)
    assert len(lines) <= 2
    assert all(text_width(ln, f) <= 500 for ln in lines)


def test_recap_numbering() -> None:
    assert recap_numbers(["a", "b", "c"], True) == [(3, "a"), (2, "b"), (1, "c")]
    assert recap_numbers(["#7 Gatun", "2. Colon"], False) == [(7, "Gatun"), (2, "Colon")]


def test_map_fps_benchmark_1080p() -> None:
    spec = b.map_spec([("Colon", 9.35, -79.9), ("Panama City", 8.98, -79.52)], duration=2.0, route=True,
                      shaded=["Panama"], title="Benchmark")
    t0 = time.perf_counter()
    comp = build_comp(spec, False)
    t1 = time.perf_counter()
    n = 30
    for i in range(n):
        comp.render(i)
    fps = n / (time.perf_counter() - t1)
    print(f"\nmap 1080p: base {t1 - t0:.2f}s, {fps:.1f} frames/s (frame drawing, one process)")
    assert fps > 5  # loose floor: the machine is shared; target is >= 20 fps on a free core


@pytest.mark.slow
def test_remotion_title_card_when_installed(settings, tmp_path: Path) -> None:
    """Real Remotion render (skipped unless renderer/node_modules exists)."""
    if not settings.path("renderer/node_modules").exists():
        pytest.skip("remotion not installed")
    from vidbush.mg.renderer import render_remotion
    out = render_remotion(small(specs()["title_card"]), tmp_path / "rm.mp4", False, settings)
    assert probe_frames(out)[:3] == (30, W, H)
