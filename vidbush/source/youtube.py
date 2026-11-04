"""YouTube adapter via the yt-dlp Python API.

Search "ytsearch10:<query>" (flat), score uploads: archival, documentary,
newsreel, public domain and Creative Commons uploads rank up; reaction,
vlog, podcast, interview, review, explained, top 10 and shorts rank down
or are dropped. Fetch reads the full info dict, skips videos without a
720p (settings.sources.youtube_min_height) rendition, downloads only the
needed section plus padding at a capped height, then cuts 3 to 12 s clips
on PySceneDetect scene boundaries. Licence is "Creative Commons
Attribution" when yt-dlp reports it, else "Standard YouTube License".

YouTube needs a JavaScript runtime for its player challenges: yt-dlp uses
the yt-dlp-ejs package (installed by the yt-dlp[default] extra) and deno
(preferred), node or bun. js_runtimes() finds one on PATH and passes it to
yt-dlp; runtime_warning() explains what is missing so the registry can log
it once per run. Without a runtime most formats disappear and every video
looks like it has no 720p rendition. All yt-dlp calls are serialised with a
small gap (throttle host "youtube") to avoid bot checks.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..models import Candidate, Shot
from ..util import clamp
from . import base, throttle
from .base import BaseAdapter, Media, RawResult, link_into, make_candidates

BOOST = {"archival": 0.25, "archive": 0.2, "documentary": 0.2, "newsreel": 0.3, "public domain": 0.3,
         "creative commons": 0.25, "historical footage": 0.2, "film": 0.05, "footage": 0.1, "1080p": 0.05,
         "restored": 0.1, "4k": 0.05}
PENALTY = {"reaction": 0.5, "reacts": 0.5, "vlog": 0.4, "podcast": 0.5, "interview": 0.35, "review": 0.35,
           "explained": 0.3, "top 10": 0.4, "top10": 0.4, "#shorts": 0.8, "shorts": 0.5, "unboxing": 0.5,
           "livestream": 0.4, "tutorial": 0.3, "gameplay": 0.6, "trailer": 0.2, "lyrics": 0.5}
PADDING_S = 10.0
RUNTIMES = ("deno", "node", "bun")
YDL_MAX_WAIT_S = 120.0


def js_runtimes() -> dict[str, dict[str, str]]:
    """yt-dlp js_runtimes option for the first runtime found on PATH (deno preferred); {} when none."""
    for name in RUNTIMES:
        path = shutil.which(name)
        if path:
            return {name: {"path": path}}
    return {}


def ejs_installed() -> bool:
    try:
        import yt_dlp_ejs  # noqa: F401
    except ImportError:
        return False
    return True


def runtime_warning() -> str:
    """Why YouTube extraction will be degraded, or "" when the runtime and solver are present."""
    missing = []
    if not js_runtimes():
        missing.append("no JavaScript runtime on PATH (install deno, see docs)")
    if not ejs_installed():
        missing.append('yt-dlp-ejs is not installed (pip install "yt-dlp[default]")')
    if not missing:
        return ""
    return "YouTube sourcing degraded: " + "; ".join(missing) + ". Most formats will be missing."


def _base_opts() -> dict[str, Any]:
    opts: dict[str, Any] = {"quiet": True, "no_warnings": True, "noplaylist": True}
    runtimes = js_runtimes()
    if runtimes:
        opts["js_runtimes"] = runtimes
    return opts


def score_entry(entry: dict[str, Any]) -> Optional[float]:
    """Relevance multiplier in 0..1 for a flat search entry, None when it must be skipped."""
    url = str(entry.get("url") or entry.get("webpage_url") or "")
    dur = float(entry.get("duration") or 0)
    if "/shorts/" in url or (0 < dur < 45) or entry.get("live_status") in {"is_live", "is_upcoming"}:
        return None
    text = " ".join(str(entry.get(k) or "") for k in ("title", "channel", "uploader", "description")).lower()
    score = 0.5 + sum(v for k, v in BOOST.items() if k in text) - sum(v for k, v in PENALTY.items() if k in text)
    if dur > 3 * 3600:
        score -= 0.2
    return None if score <= 0.05 else round(clamp(score, 0.05, 1.0), 3)


def ydl_extract(target: str, opts: dict[str, Any]) -> dict[str, Any]:
    """yt-dlp extract_info without download (network seam for tests)."""
    import yt_dlp
    with throttle.slot("youtube", max_wait=YDL_MAX_WAIT_S):
        with yt_dlp.YoutubeDL({**_base_opts(), **opts}) as ydl:
            return ydl.sanitize_info(ydl.extract_info(target, download=False))


def ydl_download(url: str, opts: dict[str, Any], dest_stem: Path) -> Path:
    """Download with yt-dlp into dest_stem.<ext>; returns the file path (network seam)."""
    import yt_dlp
    full = {**_base_opts(), "outtmpl": f"{dest_stem}.%(ext)s", **opts}
    with throttle.slot("youtube", max_wait=YDL_MAX_WAIT_S):
        with yt_dlp.YoutubeDL(full) as ydl:
            ydl.download([url])
    files = sorted(p for p in dest_stem.parent.glob(dest_stem.name + ".*") if not p.name.endswith((".part", ".ytdl")))
    if not files:
        raise base.provider_http.TransportError("yt-dlp produced no file")
    return files[0]


def parse_search(info: dict[str, Any]) -> list[RawResult]:
    out = []
    for e in info.get("entries") or []:
        if not e or not e.get("id"):
            continue
        s = score_entry(e)
        if s is None:
            continue
        page = e.get("webpage_url") or f"https://www.youtube.com/watch?v={e['id']}"
        out.append(RawResult(source="youtube", url=page, page_url=page, title=clean_title(str(e.get("title") or "")),
                             author=str(e.get("channel") or e.get("uploader") or ""), licence="Standard YouTube License",
                             media="video", duration=float(e.get("duration") or 0), score=s,
                             extra={"id": e["id"]}))
    return sorted(out, key=lambda r: -r.score)


def youtube_licence(info: dict[str, Any]) -> str:
    lic = str(info.get("license") or "").lower()
    return "Creative Commons Attribution" if "creative commons" in lic else "Standard YouTube License"


def max_height(info: dict[str, Any]) -> int:
    heights = [int(f.get("height") or 0) for f in info.get("formats") or [] if f.get("vcodec") not in (None, "none")]
    return max(heights or [int(info.get("height") or 0)])


def section_for(duration: float, shot: Shot) -> tuple[float, float]:
    """Window to download: skips the intro, long enough for several clips, plus padding."""
    want = clamp(shot.duration * 4, 30.0, 75.0)
    if duration <= want + 2 * PADDING_S:
        return 0.0, duration
    start = clamp(duration * 0.15, PADDING_S, duration - want - PADDING_S)
    return start - PADDING_S, start + want + PADDING_S


class YoutubeAdapter(BaseAdapter):
    name = "youtube"
    media = {"video"}

    def available(self, settings: Settings) -> bool:
        if settings.offline or not settings.sources.allow_youtube:
            return False
        try:
            import yt_dlp  # noqa: F401
        except ImportError:
            return False
        return True

    def search(self, query: str, media: Media, n: int, settings: Settings,
               shot: Optional[Shot] = None) -> list[RawResult]:
        if media != "video":
            return []
        info = ydl_extract(f"ytsearch10:{query}", {"extract_flat": "in_playlist", "skip_download": True})
        return parse_search(info)[:n]

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Settings) -> list[Candidate]:
        cfg = settings.sources
        info = ydl_extract(raw.url, {"skip_download": True})
        if max_height(info) < cfg.youtube_min_height:
            return []
        a, b = section_for(float(info.get("duration") or raw.duration), shot)
        stem = base.cache_dir(settings) / base.url_hash(f"{raw.url}#{a:.0f}-{b:.0f}-{cfg.youtube_max_height}")
        existing = [p for p in stem.parent.glob(stem.name + ".*") if p.suffix in {".mp4", ".webm", ".mkv"}]
        src = existing[0] if existing else ydl_download(raw.url, self._download_opts(a, b, settings), stem)
        dst = link_into(src, out_dir, f"youtube_{raw.extra.get('id', base.url_hash(raw.url))}_{int(a)}{src.suffix}")
        w, h = int(info.get("width") or 0), int(info.get("height") or 0)
        full = raw.model_copy(update={"licence": youtube_licence(info), "width": w, "height": h,
                                      "author": str(info.get("channel") or raw.author), "download_url": raw.url,
                                      "title": str(info.get("title") or raw.title)})
        spans = base.cut_clips(base.detect_scenes(dst), shot.duration, max_clips=4)
        return make_candidates(full, dst, shot, settings, spans=spans)

    @staticmethod
    def _download_opts(a: float, b: float, settings: Settings) -> dict[str, Any]:
        from yt_dlp.utils import download_range_func
        lo, hi = settings.sources.youtube_min_height, settings.sources.youtube_max_height
        fmt = (f"bv*[height<={hi}][height>={lo}][ext=mp4]/bv*[height<={hi}][height>={lo}]/"
               f"b[height<={hi}][height>={lo}]")
        # h264 after res: same height, but OpenCV scene detection decodes it about 10x faster than AV1
        return {"format": fmt, "format_sort": ["hdr:SDR", "res", "vcodec:h264", "fps"], "download_ranges": download_range_func(None, [(a, b)]),
                "force_keyframes_at_cuts": False, "retries": 3, "socket_timeout": settings.sources.timeout_s}


def clean_title(title: str) -> str:
    """Drop bracketed noise like [HD] or (1080p) from a YouTube title."""
    return re.sub(r"\s*[\[(](?:hd|4k|1080p|720p|full hd|remastered)[\])]\s*", " ", title, flags=re.I).strip()
