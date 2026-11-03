"""source_all / source_shot: candidates, keyframes, sidecars, cap, diversity, blocklist, fallbacks."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import pytest
from PIL import Image

from vidbush.models import Candidate, Shot, ShotList, ShotType, SourceKind
from vidbush.source import registry
from vidbush.source.base import RawResult


@pytest.fixture(scope="module")
def tiny_library(tmp_path_factory) -> Path:
    """12 short clips and 12 stills, all tagged 'ships' so every shot matches the same pool."""
    lib = tmp_path_factory.mktemp("lib")
    for i in range(12):
        p = lib / f"clip_{i:02d}.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
                        "-i", f"testsrc2=s=320x180:r=30:d=3,hue=h={i * 30}", "-c:v", "libx264", "-preset", "ultrafast",
                        "-g", "15", "-pix_fmt", "yuv420p", str(p)], check=True)
        p.with_suffix(".json").write_text(json.dumps({"title": f"Ship clip {i}", "tags": ["ships", "water"],
                                                      "licence": "CC0", "author": "Tester"}))
    for i in range(12):
        p = lib / f"still_{i:02d}.jpg"
        Image.new("RGB", (1920, 1080), (20 * i, 80, 120)).save(p)
        p.with_suffix(".json").write_text(json.dumps({"title": f"Ship still {i}", "tags": ["ships"], "licence": "CC0"}))
    return lib


def _shot(i: int, typ: ShotType, queries: Optional[list[str]] = None) -> Shot:
    return Shot(id=f"s{i:03d}", index=i, beat_id="b001", start=i * 2.0, end=i * 2.0 + 2.0, duration=2.0, type=typ,
                intent="a steamship at sea", queries=queries if queries is not None else ["ships at sea", "steamship"])


@pytest.fixture
def ctx(make_ctx, settings, tiny_library):
    settings.sources.local_library = str(tiny_library)
    return make_ctx(mode="script")


def _shotlist(shots: list[Shot]) -> ShotList:
    return ShotList(project_id="t1", duration=sum(s.duration for s in shots), shots=shots)


def test_source_all_fills_candidates(ctx) -> None:
    mg = _shot(3, ShotType.motion_graphic)
    sl = _shotlist([_shot(0, ShotType.footage), _shot(1, ShotType.image), _shot(2, ShotType.footage_overlay), mg])
    out = registry.source_all(sl, ctx)
    for shot in out.shots[:3]:
        assert 1 <= len(shot.candidates) <= ctx.settings.sources.max_candidates
        assert len(shot.candidates) == 8
        ids = [c.id for c in shot.candidates]
        assert len(set(ids)) == len(ids) and all(i.startswith(f"{shot.id}_local_") for i in ids)
        for c in shot.candidates:
            assert Path(c.local_path).exists() and Path(c.local_path + ".json").exists()
            assert Path(c.local_path).is_relative_to(Path(ctx.project.paths.sources) / shot.id)
            assert c.width > 0 and c.height > 0 and c.licence == "CC0 1.0"
            assert all(Path(k).exists() for k in c.keyframes)
    assert all(c.media == "video" and len(c.keyframes) == 3 for c in out.shots[0].candidates)
    assert all(c.media == "image" and len(c.keyframes) == 1 for c in out.shots[1].candidates)
    assert all("low_res" in c.flags for c in out.shots[0].candidates)  # 320x180 test clips
    assert out.shots[3].candidates == []  # motion shots are not sourced
    side = json.loads(Path(out.shots[0].candidates[0].local_path + ".json").read_text())
    assert {"source", "source_url", "page_url", "title", "author", "licence", "attribution", "downloaded_at",
            "flags"} <= set(side)


def test_adjacent_shots_prefer_different_assets(ctx) -> None:
    ctx.settings.sources.max_candidates = 3
    sl = _shotlist([_shot(i, ShotType.footage) for i in range(4)])
    out = registry.source_all(sl, ctx)
    sets = [{c.url for c in s.candidates} for s in out.shots]
    assert all(len(s) == 3 for s in sets)
    for i in range(1, 4):
        recent = set().union(*sets[max(0, i - 3):i])
        assert not (sets[i] & recent), f"shot {i} reuses an asset of the previous shots"


def test_skip_already_sourced_unless_forced(ctx) -> None:
    shot = _shot(0, ShotType.footage)
    existing = Candidate(id="s000_local_0", source=SourceKind.local, local_path="/x.mp4", media="video")
    motion = Candidate(id="s000_motion_0", source=SourceKind.motion, media="video")
    shot.candidates = [existing, motion]
    out = registry.source_all(_shotlist([shot]), ctx)
    assert out.shots[0].candidates == [existing, motion]
    out = registry.source_all(out, ctx, force=True)
    cands = out.shots[0].candidates
    assert motion in cands and existing not in cands and len(cands) == 9


def test_source_shot_excludes_existing(ctx) -> None:
    sl = registry.source_all(_shotlist([_shot(0, ShotType.image)]), ctx)
    shot = sl.shots[0]
    old_ids = {c.id for c in shot.candidates}
    old_urls = {c.url for c in shot.candidates}
    new = registry.source_shot(shot, ctx, queries=["steamship"], exclude_ids=list(old_ids))
    assert new and not ({c.id for c in new} & old_ids) and not ({c.url for c in new} & old_urls)
    assert len(new) == 4  # 12 stills minus the 8 already on the shot


class FailingAdapter:
    name = "nasa"
    media = {"video", "image"}

    def available(self, settings) -> bool:
        return True

    def search(self, query, media, n, settings, shot=None):
        raise ConnectionError("network down")

    def fetch(self, raw, out_dir, shot, settings):
        raise AssertionError("never fetched")


class BlockedAdapter(FailingAdapter):
    name = "pexels"

    def search(self, query, media, n, settings, shot=None):
        return [RawResult(source="pexels", url="https://cdn.badstock.net/a.jpg", page_url="https://badstock.net/a",
                          download_url="https://cdn.badstock.net/a.jpg", media=media, score=0.9)]


def test_failures_and_blocklist_are_not_fatal(ctx, monkeypatch) -> None:
    ctx.brand.blocklist = ["badstock.net"]
    real = registry.active_adapters(ctx)
    monkeypatch.setattr(registry, "active_adapters", lambda c: real + [FailingAdapter(), BlockedAdapter()])
    out = registry.source_all(_shotlist([_shot(0, ShotType.image)]), ctx)
    assert len(out.shots[0].candidates) == 8
    assert all(c.source == SourceKind.local for c in out.shots[0].candidates)


def test_interleave_mixes_sources_and_puts_fillers_last() -> None:
    def r(src: str, n: int, score: float) -> RawResult:
        return RawResult(source=src, url=f"https://{src}/{n}", score=score)
    per = {"local": [r("local", 1, 0.9), r("local", 2, 0.0), r("local", 3, 0.0)],
           "wikimedia": [r("wikimedia", 1, 0.8), r("wikimedia", 2, 0.8)], "nasa": [r("nasa", 1, 0.7)]}
    out = registry.interleave(per, ["local", "wikimedia", "nasa"])
    assert [x.url for x in out] == ["https://local/1", "https://wikimedia/1", "https://nasa/1", "https://wikimedia/2",
                                    "https://local/2", "https://local/3"]


def test_still_shot_falls_back_to_video_frames(ctx, monkeypatch, tiny_library) -> None:
    from vidbush.source import local

    class VideoOnly(local.LocalAdapter):
        def search(self, query, media, n, settings, shot=None):
            return [] if media == "image" else super().search(query, media, n, settings, shot)

    monkeypatch.setattr(registry, "active_adapters", lambda c: [VideoOnly()])
    out = registry.source_all(_shotlist([_shot(0, ShotType.image)]), ctx)
    cands = out.shots[0].candidates
    assert cands and all(c.media == "image" and "frame_grab" in c.flags for c in cands)
    assert all(c.local_path.endswith(".jpg") and len(c.keyframes) == 1 for c in cands)


def test_no_queries_uses_intent(ctx) -> None:
    shot = _shot(0, ShotType.footage, queries=[])
    assert registry.shot_queries(shot) == ["a steamship at sea"]
    assert registry.source_all(_shotlist([shot]), ctx).shots[0].candidates
