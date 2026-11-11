"""YouTube adapter via the yt-dlp Python API.

Search "ytsearch10:<query>" (flat), score uploads: archival, documentary,
newsreel, public domain and Creative Commons uploads rank up; reaction,
vlog, podcast, interview, review, explained, top 10 and shorts rank down
or are dropped. When the brand's licence policy prefers Creative Commons
(the registry passes prefer_cc=True), a first pass runs the YouTube search
URL with the Creative Commons filter (sp=EgIwAQ%3D%3D, read by yt-dlp's
youtube:search_url extractor) and the normal pass follows; results are
merged with the Creative Commons pass first and deduped by video id. Each
pass counts against the run's search budget (the registry takes it).

Fetch reads the full info dict, skips videos without a 720p
(settings.sources.youtube_min_height) rendition and picks the windows to
download: the transcript window (subtitles or automatic captions, fetched
through ydl_fetch_text) or chapter whose text best matches the shot's
intent and must_show (keyword overlap, stopwords removed, must_show terms
weigh double), else up to settings.sources.youtube_windows short windows
spread across the video, away from its first and last few percent. Each
window gets PADDING_S on both sides (clamped to the video), is downloaded
as its own cached file at a capped height preferring h264 (avc1, which
PySceneDetect and OpenCV decode) over av1 and vp9, and is cut into 3 to
12 s clips on its own scene boundaries, so in and out points refer to the
local file; about MAX_CLIPS_PER_VIDEO clips per video in total. Windows are
stream copied, so a file starts at the keyframe before the requested cut;
window_start() measures that lead from the file's length and the file is
named youtube_<id>_<true start s>, the offset the director loop adds to in
points to compare windows of one video. If a Creative Commons pass fails the
normal pass still runs. Licence is
"Creative Commons Attribution" when yt-dlp reports it, else "Standard
YouTube License" (flagged standard_youtube_licence, see source/licence.py).

YouTube needs a JavaScript runtime for its player challenges: yt-dlp uses
the yt-dlp-ejs package (installed by the yt-dlp[default] extra) and deno
(preferred), node or bun. js_runtimes() finds one on PATH and passes it to
yt-dlp; runtime_warning() explains what is missing so the registry can log
it once per run. Without a runtime most formats disappear and every video
looks like it has no 720p rendition. Cookies (settings.sources
youtube_cookies_from_browser or youtube_cookie_file) go to every yt-dlp
call; each YoutubeDL gets a private copy of the cookie file (yt-dlp rewrites
it in place on close), so parallel calls never corrupt the operator's file. Searches, info extraction and caption fetches are serialised with a
small gap (throttle host "youtube"); downloads of different videos run in
parallel up to settings.sources.youtube_download_workers on their own
semaphore, still paced by the same rate, and one video is never
downloaded twice at the same time.
"""
from __future__ import annotations

import contextlib
import json
import logging
import threading
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import quote_plus

from ..config import Settings
from ..models import Candidate, Shot
from ..util import clamp, media_duration
from . import base, throttle
from .base import BaseAdapter, Media, RawResult, link_into, make_candidates
from .local import tokens as keyword_tokens

BOOST = {"archival": 0.25, "archive": 0.2, "documentary": 0.2, "newsreel": 0.3, "public domain": 0.3,
         "creative commons": 0.25, "historical footage": 0.2, "film": 0.05, "footage": 0.1, "1080p": 0.05,
         "restored": 0.1, "4k": 0.05}
PENALTY = {"reaction": 0.5, "reacts": 0.5, "vlog": 0.4, "podcast": 0.5, "interview": 0.35, "review": 0.35,
           "explained": 0.3, "top 10": 0.4, "top10": 0.4, "#shorts": 0.8, "shorts": 0.5, "unboxing": 0.5,
           "livestream": 0.4, "tutorial": 0.3, "gameplay": 0.6, "trailer": 0.2, "lyrics": 0.5}
PADDING_S = 10.0
RUNTIMES = ("deno", "node", "bun")
YDL_MAX_WAIT_S = 120.0
DOWNLOAD_WAIT_S = 900.0  # a download waits this long for one of the parallel download slots
SEARCH_N = 10
CC_FILTER_SP = "EgIwAQ%3D%3D"  # URL encoded EgIwAQ==, YouTube's Creative Commons search filter
CC_LICENCE = "Creative Commons Attribution"
STANDARD_LICENCE = "Standard YouTube License"
EDGE_FRACTION = 0.05  # spread windows stay out of the first and last 5 percent (intros, end cards)
MAX_CLIPS_PER_VIDEO = 4
MUST_SHOW_WEIGHT = 2.0
QUERY_WEIGHT = 0.5
CHAPTER_WEIGHT = 1.0
CAPTION_EXTS = ("json3", "vtt")
VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv"}
MAX_KEYFRAME_LEAD_S = 15.0  # a stream copied window never starts more than this before the requested cut
log = logging.getLogger(__name__)
_BROWSER_SPEC = re.compile(r"(?P<name>[^+:]+)(?:\s*\+\s*(?P<keyring>[^:]+))?(?:\s*:\s*(?!:)(?P<profile>.+?))?"
                           r"(?:\s*::\s*(?P<container>.+))?")


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


def browser_cookies(spec: str) -> Optional[tuple[str, Optional[str], Optional[str], Optional[str]]]:
    """cookiesfrombrowser tuple (browser, profile, keyring, container) from BROWSER[+KEYRING][:PROFILE][::CONTAINER].

    Same grammar as yt-dlp's --cookies-from-browser; None when spec is empty or malformed.
    """
    spec = (spec or "").strip()
    m = _BROWSER_SPEC.fullmatch(spec) if spec else None
    if m is None:
        return None
    name, keyring, profile, container = m.group("name", "keyring", "profile", "container")
    return name.strip().lower(), profile, keyring.upper() if keyring else None, container


def cookie_opts(settings: Optional[Settings]) -> dict[str, Any]:
    """cookiesfrombrowser and cookiefile yt-dlp options from settings.sources; {} when unset."""
    if settings is None:
        return {}
    cfg = settings.sources
    opts: dict[str, Any] = {}
    browser = browser_cookies(getattr(cfg, "youtube_cookies_from_browser", ""))
    if browser:
        opts["cookiesfrombrowser"] = browser
    cookie_file = str(getattr(cfg, "youtube_cookie_file", "") or "").strip()
    if cookie_file:
        opts["cookiefile"] = str(settings.path(cookie_file))
    return opts


def _base_opts(settings: Optional[Settings] = None) -> dict[str, Any]:
    opts: dict[str, Any] = {"quiet": True, "no_warnings": True, "noplaylist": True}
    runtimes = js_runtimes()
    if runtimes:
        opts["js_runtimes"] = runtimes
    if base.HTTP_PROXY:
        opts["proxy"] = base.HTTP_PROXY  # also handed to ffmpeg, which ignores proxy env vars
    return {**opts, **cookie_opts(settings)}


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


# ---------------------------------------------------------------- network seams

@contextlib.contextmanager
def private_cookies(params: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """params with cookiefile pointing at a private temporary copy, removed afterwards.

    yt-dlp rewrites its cookiefile in place when a YoutubeDL closes; parallel
    instances sharing one file can truncate it under each other. Each instance
    gets its own copy instead, so the operator's file is only ever read.
    """
    src = params.get("cookiefile")
    if not src or not isinstance(src, (str, Path)):
        yield params
        return
    tmp = Path(tempfile.mkdtemp(prefix="vidbush_cookies_"))
    try:
        copy = tmp / "cookies.txt"
        with contextlib.suppress(OSError):
            shutil.copyfile(src, copy)  # a missing file stays missing, as yt-dlp expects
        yield {**params, "cookiefile": str(copy)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@contextlib.contextmanager
def youtube_dl(params: dict[str, Any]) -> Iterator[Any]:
    """yt_dlp.YoutubeDL(params) with a private cookie file copy."""
    import yt_dlp
    with private_cookies(params) as private:
        with yt_dlp.YoutubeDL(private) as ydl:
            yield ydl


def ydl_extract(target: str, opts: dict[str, Any]) -> dict[str, Any]:
    """yt-dlp extract_info without download, serial on the "youtube" slot (network seam for tests)."""
    with throttle.slot("youtube", max_wait=YDL_MAX_WAIT_S):
        with youtube_dl({**_base_opts(), **opts}) as ydl:
            return ydl.sanitize_info(ydl.extract_info(target, download=False))


def ydl_download(url: str, opts: dict[str, Any], dest_stem: Path) -> Path:
    """Download with yt-dlp into dest_stem.<ext>; returns the file path (network seam).

    Paced by the "youtube" rate but not serial: the caller holds the download semaphore.
    """
    full = {**_base_opts(), "outtmpl": f"{dest_stem}.%(ext)s", **opts}
    with throttle.paced("youtube", max_wait=YDL_MAX_WAIT_S):
        with youtube_dl(full) as ydl:
            ydl.download([url])
    files = sorted(p for p in dest_stem.parent.glob(dest_stem.name + ".*") if not p.name.endswith((".part", ".ytdl")))
    if not files:
        raise base.provider_http.TransportError("yt-dlp produced no file")
    return files[0]


def probe_duration(path: Path) -> float:
    """Duration of a local media file in seconds, 0 when it cannot be read (seam for tests)."""
    try:
        return media_duration(path)
    except Exception:  # noqa: BLE001 - an unreadable file only loses the start correction
        return 0.0


def ydl_fetch_text(url: str, opts: dict[str, Any]) -> str:
    """Text of a caption track URL through yt-dlp's opener (cookies, proxy), serial (network seam)."""
    with throttle.slot("youtube", max_wait=YDL_MAX_WAIT_S):
        with youtube_dl({**_base_opts(), **opts}) as ydl:
            return ydl.urlopen(url).read().decode("utf-8", "replace")


# ---------------------------------------------------------------- search

def cc_search_url(query: str) -> str:
    """YouTube search URL with the Creative Commons filter (yt-dlp youtube:search_url)."""
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}&sp={CC_FILTER_SP}"


def parse_search(info: dict[str, Any], cc: bool = False) -> list[RawResult]:
    """Scored results of one flat search; cc marks the Creative Commons filtered pass."""
    out = []
    for e in info.get("entries") or []:
        if not e or not e.get("id"):
            continue
        s = score_entry(e)
        if s is None:
            continue
        page = e.get("webpage_url") or f"https://www.youtube.com/watch?v={e['id']}"
        extra: dict[str, Any] = {"id": e["id"], "cc_search": True} if cc else {"id": e["id"]}
        out.append(RawResult(source="youtube", url=page, page_url=page, title=clean_title(str(e.get("title") or "")),
                             author=str(e.get("channel") or e.get("uploader") or ""),
                             licence=CC_LICENCE if cc else STANDARD_LICENCE, media="video",
                             duration=float(e.get("duration") or 0), score=s, extra=extra))
    return sorted(out, key=lambda r: -r.score)


def merge_passes(passes: list[list[RawResult]]) -> list[RawResult]:
    """Concatenate search passes in order (Creative Commons first), deduped by video id."""
    seen: set[str] = set()
    out: list[RawResult] = []
    for results in passes:
        for r in results:
            key = str(r.extra.get("id") or r.url)
            if key not in seen:
                seen.add(key)
                out.append(r)
    return out


def youtube_licence(info: dict[str, Any]) -> str:
    lic = str(info.get("license") or "").lower()
    return CC_LICENCE if "creative commons" in lic else STANDARD_LICENCE


def max_height(info: dict[str, Any]) -> int:
    heights = [int(f.get("height") or 0) for f in info.get("formats") or [] if f.get("vcodec") not in (None, "none")]
    return max(heights or [int(info.get("height") or 0)])


# ---------------------------------------------------------------- transcripts and chapters

Cue = tuple[float, float, str]


def caption_track(info: dict[str, Any], language: str = "") -> Optional[tuple[str, str]]:
    """(url, ext) of the best caption track: manual before automatic, wanted language, json3 before vtt."""
    lang = (language or str(info.get("language") or "") or "en").lower().split("-")[0]
    for key in ("subtitles", "automatic_captions"):
        tracks = info.get(key) or {}
        names = sorted((k for k in tracks if k.lower() == lang or k.lower().startswith(lang + "-")),
                       key=lambda k: (k.lower() != lang, k))
        for name in names:
            for ext in CAPTION_EXTS:
                for t in tracks.get(name) or []:
                    if t.get("ext") == ext and t.get("url"):
                        return str(t["url"]), ext
    return None


def parse_json3(text: str) -> list[Cue]:
    """Cues from a YouTube json3 caption track."""
    try:
        events = json.loads(text).get("events") or []
    except (ValueError, AttributeError):
        return []
    out: list[Cue] = []
    for ev in events:
        words = "".join(str(s.get("utf8") or "") for s in ev.get("segs") or []).strip()
        if words:
            start = float(ev.get("tStartMs") or 0) / 1000
            out.append((start, start + float(ev.get("dDurationMs") or 0) / 1000, words))
    return out


def _vtt_time(stamp: str) -> float:
    parts = [float(p) for p in stamp.strip().replace(",", ".").split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def parse_vtt(text: str) -> list[Cue]:
    """Cues from a WebVTT caption track (inline tags dropped)."""
    out: list[Cue] = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        for i, ln in enumerate(lines):
            if "-->" not in ln:
                continue
            a, b = ln.split("-->")[:2]
            words = re.sub(r"<[^>]+>", "", " ".join(lines[i + 1:])).strip()
            try:
                cue = (_vtt_time(a), _vtt_time(b.split()[0]), words)
            except (ValueError, IndexError):
                break
            if words:
                out.append(cue)
            break
    return out


def transcript_cues(info: dict[str, Any], settings: Settings, language: str = "") -> list[Cue]:
    """Timed transcript lines of a video, [] when there is no usable track (best effort)."""
    track = caption_track(info, language)
    if track is None:
        return []
    url, ext = track
    try:
        text = ydl_fetch_text(url, cookie_opts(settings))
    except Exception:  # noqa: BLE001 - a missing transcript only loses the window hint
        return []
    return parse_json3(text) if ext == "json3" else parse_vtt(text)


def chapters_of(info: dict[str, Any]) -> list[Cue]:
    out: list[Cue] = []
    for c in info.get("chapters") or []:
        try:
            out.append((float(c.get("start_time") or 0), float(c.get("end_time") or 0), str(c.get("title") or "")))
        except (TypeError, ValueError):
            continue
    return [c for c in out if c[1] > c[0]]


# ---------------------------------------------------------------- window selection

def shot_weights(shot: Shot) -> dict[str, float]:
    """Keyword weights: must_show terms count double, intent once, queries half."""
    weights: dict[str, float] = {}
    for texts, w in ((shot.queries, QUERY_WEIGHT), ([shot.intent], 1.0), (shot.must_show, MUST_SHOW_WEIGHT)):
        for tok in keyword_tokens(" ".join(texts)):
            weights[tok] = max(weights.get(tok, 0.0), w)
    return weights


def text_score(text: str, weights: dict[str, float]) -> float:
    """Sum of the weights of the distinct shot keywords present in text."""
    return sum(weights.get(t, 0.0) for t in keyword_tokens(text))


def window_length(shot: Shot) -> float:
    """Length of a matched window: long enough for several clips."""
    return clamp(shot.duration * 4, 30.0, 75.0)


def _centred(mid: float, length: float, duration: float) -> tuple[float, float]:
    a = clamp(mid - length / 2, 0.0, max(0.0, duration - length))
    return a, min(duration, a + length)


def best_text_window(cues: list[Cue], chapters: list[Cue], weights: dict[str, float], length: float,
                     duration: float) -> Optional[tuple[float, float]]:
    """Window whose transcript (plus its chapter title) or chapter title best matches; None without a match."""
    def chapter_bonus(t: float) -> float:
        return sum(CHAPTER_WEIGHT * text_score(title, weights) for a, b, title in chapters if a <= t < b)

    scored: list[tuple[float, tuple[float, float]]] = []
    if cues:
        step = max(length / 2, 1.0)
        a = 0.0
        while a < duration:
            b = min(duration, a + length)
            text = " ".join(t for s, e, t in cues if e > a and s < b)
            scored.append((text_score(text, weights) + chapter_bonus((a + b) / 2), (a, b)))
            if b >= duration:
                break
            a += step
    else:
        for a, b, title in chapters:
            win = (a, min(b, a + length)) if b - a >= length else _centred((a + b) / 2, length, duration)
            scored.append((text_score(title, weights), win))
    best = max(scored, key=lambda p: p[0], default=None)
    return best[1] if best and best[0] > 0 else None


def spread_windows(duration: float, n: int, shot: Shot) -> list[tuple[float, float]]:
    """Up to n short windows centred at even fractions of the video, away from its first and last few percent."""
    n = max(1, int(n))
    length = clamp(shot.duration * 2.5, 15.0, 30.0)
    lo, hi = duration * EDGE_FRACTION, duration * (1 - EDGE_FRACTION)
    out = []
    for i in range(n):
        mid = lo + (hi - lo) * (i + 0.5) / n
        a = clamp(mid - length / 2, lo, max(lo, hi - length))
        out.append((a, min(hi, a + length)))
    return out


def pad_windows(windows: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    """PADDING_S around each window, clamped to the video; overlapping windows merge."""
    padded = sorted((max(0.0, a - PADDING_S), min(duration, b + PADDING_S)) for a, b in windows)
    out: list[tuple[float, float]] = []
    for a, b in padded:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def choose_windows(info: dict[str, Any], shot: Shot, settings: Settings,
                   cues: Optional[list[Cue]] = None) -> list[tuple[float, float]]:
    """Padded download windows: whole short videos, the best matching section, else spread windows."""
    duration = float(info.get("duration") or 0)
    length = window_length(shot)
    if duration <= length + 2 * PADDING_S:
        return [(0.0, duration)]
    best = best_text_window(cues or [], chapters_of(info), shot_weights(shot), length, duration)
    windows = [best] if best else spread_windows(duration, settings.sources.youtube_windows, shot)
    return pad_windows(windows, duration)


def window_start(a: float, b: float, path: Path) -> float:
    """True source time of the first frame of a window file requested as (a, b).

    Windows are stream copied (no re-encode), so the file starts at the
    keyframe at or before a. Its end is cut at b, so any extra length is that
    lead. Clamped to the plausible range; a when the file cannot be probed.
    """
    lead = probe_duration(path) - (b - a)
    return a - clamp(lead, 0.0, min(a, MAX_KEYFRAME_LEAD_S))


def window_name(vid: str, start: float, suffix: str) -> str:
    """File name youtube_<id>_<start s>: the start the director loop reads as the window's source offset."""
    return f"youtube_{vid}_{round(start)}{suffix}"


def clip_quota(n_windows: int, total: int = MAX_CLIPS_PER_VIDEO) -> list[int]:
    """Clips per window so a video gives about total clips (at least one per window)."""
    n = max(1, n_windows)
    return [max(1, total // n + (1 if i < total % n else 0)) for i in range(n)]


# ---------------------------------------------------------------- adapter

BOT_BLOCK = threading.Event()  # set when YouTube demands sign-in; YouTube is skipped for the rest of the process
BOT_MARKERS = ("confirm you", "not a bot", "sign in to confirm")


def note_bot_block(err: Exception) -> None:
    """Remember a bot check so the run stops wasting requests on YouTube."""
    if any(m in str(err).lower() for m in BOT_MARKERS) and not BOT_BLOCK.is_set():
        BOT_BLOCK.set()
        logging.getLogger("vidbush.source").warning(
            "YouTube asked to sign in (bot check). YouTube is skipped for this run. "
            "Set YOUTUBE_COOKIES or YOUTUBE_COOKIES_FROM_BROWSER in .env to fix it.")


class YoutubeAdapter(BaseAdapter):
    name = "youtube"
    media = {"video"}
    search_hints = ("prefer_cc", "language")  # keyword arguments the registry may pass to search()

    def available(self, settings: Settings) -> bool:
        if settings.offline or not settings.sources.allow_youtube:
            return False
        try:
            import yt_dlp  # noqa: F401
        except ImportError:
            return False
        return True

    def search(self, query: str, media: Media, n: int, settings: Settings, shot: Optional[Shot] = None,
               prefer_cc: bool = False, language: str = "") -> list[RawResult]:
        """Normal search; with prefer_cc a Creative Commons filtered pass first, merged ahead of it."""
        if media != "video" or BOT_BLOCK.is_set():
            return []
        opts = {"extract_flat": "in_playlist", "skip_download": True, **cookie_opts(settings)}
        passes: list[list[RawResult]] = []
        if prefer_cc:
            passes.append(self._cc_pass(query, opts))
        try:
            passes.append(parse_search(ydl_extract(f"ytsearch{SEARCH_N}:{query}", opts)))
        except throttle.Throttled as e:
            e.sent = len(passes)  # the Creative Commons pass went out
            raise
        results = merge_passes(passes)[:n]
        if language:
            for r in results:
                r.extra["lang"] = language
        return results

    @staticmethod
    def _cc_pass(query: str, opts: dict[str, Any]) -> list[RawResult]:
        """Creative Commons filtered results; [] when that request fails, so the normal pass still runs."""
        try:
            info = ydl_extract(cc_search_url(query), {**opts, "playlist_items": f"1:{SEARCH_N}"})
        except throttle.Throttled:
            raise  # nothing was sent, and the normal pass would wait on the same host
        except Exception as e:  # noqa: BLE001 - the CC pass is optional, the normal pass decides
            log.warning("YouTube Creative Commons search failed, normal search only: %s", str(e)[:200])
            return []
        return parse_search(info, cc=True)

    def fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Settings) -> list[Candidate]:
        if BOT_BLOCK.is_set():
            return []
        try:
            return self._fetch(raw, out_dir, shot, settings)
        except Exception as e:
            note_bot_block(e)
            raise

    def _fetch(self, raw: RawResult, out_dir: Path, shot: Shot, settings: Settings) -> list[Candidate]:
        cfg = settings.sources
        info = ydl_extract(raw.url, {"skip_download": True, **cookie_opts(settings)})
        if max_height(info) < cfg.youtube_min_height:
            raise base.Rejected(f"{max_height(info)}p is below the {cfg.youtube_min_height}p floor")
        info = {**info, "duration": float(info.get("duration") or raw.duration)}
        cues = transcript_cues(info, settings, str(raw.extra.get("lang") or ""))
        windows = choose_windows(info, shot, settings, cues)
        files = self._download_windows(raw.url, windows, settings)
        full = raw.model_copy(update={"licence": youtube_licence(info), "width": int(info.get("width") or 0),
                                      "height": int(info.get("height") or 0), "download_url": raw.url,
                                      "author": str(info.get("channel") or raw.author),
                                      "title": str(info.get("title") or raw.title)})
        vid = raw.extra.get("id", base.url_hash(raw.url))
        out: list[Candidate] = []
        for (a, b, src), quota in zip(files, clip_quota(len(files))):
            dst = link_into(src, out_dir, window_name(vid, window_start(a, b, src), src.suffix))
            spans = base.cut_clips(base.detect_scenes(dst), shot.duration, max_clips=quota)
            part = full.model_copy(update={"note": f"section {a:.0f}-{b:.0f} s of the source video"})
            out.extend(make_candidates(part, dst, shot, settings, spans=spans))
        return out

    def _download_windows(self, url: str, windows: list[tuple[float, float]],
                          settings: Settings) -> list[tuple[float, float, Path]]:
        """Each window as its own cached file; one video at a time per URL, downloads capped in parallel."""
        cfg = settings.sources
        out: list[tuple[float, float, Path]] = []
        error: Optional[Exception] = None
        with throttle.exclusive("youtube", url):
            for a, b in windows:
                stem = base.cache_dir(settings) / base.url_hash(f"{url}#{a:.0f}-{b:.0f}-{cfg.youtube_max_height}")
                existing = sorted(p for p in stem.parent.glob(stem.name + ".*") if p.suffix in VIDEO_SUFFIXES)
                if existing:
                    out.append((a, b, existing[0]))
                    continue
                try:
                    with throttle.download_slot("youtube", cfg.youtube_download_workers, max_wait=DOWNLOAD_WAIT_S):
                        out.append((a, b, ydl_download(url, self._download_opts(a, b, settings), stem)))
                except Exception as e:  # noqa: BLE001 - other windows may still arrive
                    error = e
        if not out and error is not None:
            raise error
        return out

    @staticmethod
    def _download_opts(a: float, b: float, settings: Settings) -> dict[str, Any]:
        from yt_dlp.utils import download_range_func
        return {"format": format_selector(settings), "format_sort": FORMAT_SORT,
                "download_ranges": download_range_func(None, [(a, b)]), "force_keyframes_at_cuts": False,
                "retries": 3, "socket_timeout": settings.sources.timeout_s, **cookie_opts(settings)}


FORMAT_SORT = ["hdr:SDR", "res", "vcodec:h264", "fps"]  # resolution is never traded; h264 breaks ties (OpenCV decodes it about 10x faster than AV1)


def format_selector(settings: Settings) -> str:
    """Video format within the height limits; format_sort then picks the highest resolution, h264 on ties."""
    lo, hi = settings.sources.youtube_min_height, settings.sources.youtube_max_height
    h = f"[height<={hi}][height>={lo}]"
    return f"bv*{h}[ext=mp4]/bv*{h}/b{h}"


def clean_title(title: str) -> str:
    """Drop bracketed noise like [HD] or (1080p) from a YouTube title."""
    return re.sub(r"\s*[\[(](?:hd|4k|1080p|720p|full hd|remastered)[\])]\s*", " ", title, flags=re.I).strip()
