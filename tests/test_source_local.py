"""Local footage library: keyword ranking, offline rotation, clip spans, bundled sample assets."""
from __future__ import annotations

import json
from pathlib import Path

from vidbush.config import REPO_ROOT
from vidbush.models import Shot, ShotType
from vidbush.source.local import LocalAdapter, clip_span, library_index, rotation, stem, tokens

SAMPLES = REPO_ROOT / "fixtures" / "sample_assets"


def _shot(sid: str = "s001", dur: float = 5.0, typ: ShotType = ShotType.footage) -> Shot:
    return Shot(id=sid, beat_id="b1", start=0, end=dur, duration=dur, type=typ, intent="x")


def _lib(tmp_path: Path) -> Path:
    lib = tmp_path / "lib"
    (lib / "sub").mkdir(parents=True)
    items = {
        "ships.jpg": {"title": "Steamship at sea", "tags": ["ships", "ocean"], "description": "an old steamer"},
        "sub/workers.jpg": {"title": "Workers digging", "tags": ["construction", "labourers"]},
        "crane_photo.png": None,  # no sidecar: filename fallback
        "clouds.jpg": {"title": "Clouds", "tags": ["sky"], "era": "1910s"},
    }
    from PIL import Image
    for name, meta in items.items():
        p = lib / name
        Image.new("RGB", (1920, 1080), (100, 100, 100)).save(p)
        if meta:
            p.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    return lib


def test_stem_and_tokens() -> None:
    assert stem("ships") == "ship" and stem("digging") == "digg" and stem("cities") == "city"
    assert tokens("The old steamships of the 1910s") == {"steamship", "1910"}


def test_keyword_ranking(settings, tmp_path) -> None:
    settings.sources.local_library = str(_lib(tmp_path))
    settings.offline = False
    res = LocalAdapter().search("steamship ships ocean", "image", 4, settings, shot=_shot())
    assert res[0].title == "Steamship at sea" and res[0].score > 0.5 and res[0].note == ""
    assert len(res) == 1  # online with matches: no rotation filler
    res = LocalAdapter().search("construction workers", "image", 4, settings)
    assert res[0].title == "Workers digging"
    res = LocalAdapter().search("crane", "image", 4, settings)
    assert res[0].url.endswith("crane_photo.png") and res[0].score > 0
    res = LocalAdapter().search("1910s weather", "image", 4, settings)
    assert res[0].title == "Clouds"


def test_rotation_when_nothing_matches_or_offline(settings, tmp_path) -> None:
    settings.sources.local_library = str(_lib(tmp_path))
    settings.offline = False
    res = LocalAdapter().search("zebra savanna", "image", 3, settings, shot=_shot("s007"))
    assert len(res) == 3 and all(r.note == "fixture rotation" and r.score == 0 for r in res)
    again = LocalAdapter().search("zebra savanna", "image", 3, settings, shot=_shot("s007"))
    assert [r.url for r in res] == [r.url for r in again]  # deterministic
    settings.offline = True
    off = LocalAdapter().search("steamship", "image", 4, settings, shot=_shot("s007"))
    assert off[0].title == "Steamship at sea" and len(off) == 4
    assert {r.note for r in off[1:]} == {"fixture rotation"}


def test_rotation_differs_by_shot(tmp_path) -> None:
    items = library_index(_lib(tmp_path))
    starts = {rotation(items, f"s{i:03d}", 1)[0].path.name for i in range(20)}
    assert len(starts) >= 3
    full = rotation(items, "abc", 10)
    assert len(full) == len(items) and len({i.path for i in full}) == len(items)


def test_clip_span_covers_shot_and_varies() -> None:
    a = clip_span(12.0, _shot("s001", 5.0), "clip.mp4")
    b = clip_span(12.0, _shot("s002", 5.0), "clip.mp4")
    assert abs((a[1] - a[0]) - 5.0) < 1e-6 and 0 <= a[0] and a[1] <= 12.0
    assert a != b
    assert clip_span(3.0, _shot("s003", 5.0), "clip.mp4") == (0.0, 3.0)


def test_bundled_sample_library() -> None:
    items = library_index(SAMPLES)
    videos = [i for i in items if i.media == "video"]
    stills = [i for i in items if i.media == "image"]
    assert len(videos) == 24 and len(stills) == 16
    for it in items:
        assert it.meta["licence"] == "CC0 (Vidbush generated fixture)" and it.meta["author"] == "Vidbush fixtures"
        assert it.meta["tags"] and it.meta["title"]
    total = sum(p.stat().st_size for p in SAMPLES.rglob("*") if p.is_file())
    assert total < 45_000_000


def test_bundled_library_keyword_search(settings) -> None:
    res = LocalAdapter().search("steamship rough seas archival film", "video", 4, settings, shot=_shot())
    assert "ship" in res[0].title.lower() or "steam" in res[0].title.lower()
    res = LocalAdapter().search("workers digging construction", "image", 4, settings, shot=_shot(typ=ShotType.image))
    assert any("construction" in r.extra["tags"] for r in res[:2])


def test_fetch_links_and_spans(settings, tmp_path) -> None:
    src = sorted(SAMPLES.glob("sample_footage_*.mp4"))[0]
    settings.sources.local_library = str(SAMPLES)
    raw = LocalAdapter()._raw(next(i for i in library_index(SAMPLES) if i.path == src), 0.5, "")
    cands = LocalAdapter().fetch(raw, tmp_path / "out", _shot("s010", 4.0), settings)
    c = cands[0]
    assert Path(c.local_path).exists() and Path(c.local_path).parent == tmp_path / "out"
    assert abs((c.out_point - c.in_point) - 4.0) < 1e-6 and c.width in (1280, 1920)
    side = json.loads(Path(c.local_path + ".json").read_text())
    assert side["source"] == "local" and side["licence"] == "CC0 (Vidbush generated fixture)"
