"""Offline sourcing replay: the stub keeps the adapter's keyword form and expired cache entries are served."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from vidbush.source import registry
from vidbush.source.base import RawResult
from vidbush.source.replay import DROP_MESSAGES, OfflineAdapter, ignore_ttl


class _ThreeWords:
    name, media, keyword_words = "nasa", {"image"}, 3


def test_replay_reads_the_keyword_cache_key_and_ignores_the_ttl(tmp_path):
    settings = SimpleNamespace(source_cache_folder=str(tmp_path), path=lambda p: Path(p))
    off = OfflineAdapter(_ThreeWords())
    query = "Apollo eleven lunar module landing 1969 archival photograph"
    sent = registry.sent_query(off, query)
    assert off.keyword_words == 3 and sent == registry.sent_query(_ThreeWords(), query) and sent != query
    hit = [RawResult(url="https://x/1.jpg", title="Apollo lunar module", media="image", source="nasa")]
    registry.cache_put("nasa", sent, "image", 4, hit, settings)
    path = registry._cache_path("nasa", sent, "image", 4, settings)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["ts"] = 0  # written in 1970: far past any TTL
    path.write_text(json.dumps(data), encoding="utf-8")
    assert registry.cache_get("nasa", sent, "image", 4, settings) is None  # live: expired
    with ignore_ttl():
        got = registry.cache_get("nasa", sent, "image", 4, settings)
    assert got and got[0].url == "https://x/1.jpg"
    assert registry.cache_get("nasa", sent, "image", 4, settings) is None  # TTL restored after replay


def test_replay_drops_reads_the_current_log_message():
    assert "off-topic results demoted to fillers" in DROP_MESSAGES
