"""Second YouTube download route: video-download-api.com (host p.savenow.to), a paid download service.

The free route (yt-dlp, youtube.py) is always tried first. When YouTube answers a download with a bot check,
youtube.py switches the rest of the process's YouTube downloads to this route instead of pausing YouTube.
The service downloads whole videos only (no time sections), so the smallest file that keeps the height floor
(MP4 720p) is fetched once per video, cached, and each window is cut from it locally by stream copy, exactly
as the free route's section downloads are (window_start corrects for the keyframe lead the same way).

API (as used by Adam's ai-avatar-gen and ULF projects, docs https://video-download-api.com/api/docs):
  GET /ajax/download.php?url=&format=720&apikey=&add_info=1  -> {success, id, info{title, ...}}
  GET /ajax/progress.php?id=                                  -> {progress (0..1000), success, download_url, text}
  GET /ajax/api.php?function=i&api=&u=                        -> {title, duration, thumbnail} (no job, not billed)
  GET /ajax/subtitles?url=&api=&lang=&format=vtt              -> {id, progress_url}, then as progress
Charged when a job is accepted (it returns an id): MP4 720p 0.0002 USD, VTT 0.0001 USD (pricing page,
2026-11-18). Key: VIDEO_DOWNLOAD_API_KEY (CaseFile and ULF name), else VDA_API_KEY. The key goes only into
query strings sent to the service and is never logged: error texts are scrubbed of it.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import requests

log = logging.getLogger("vidbush.source")

HOST = "https://p.savenow.to"
KEY_NAMES = ("VIDEO_DOWNLOAD_API_KEY", "VDA_API_KEY")
FORMAT = "720"  # the smallest MP4 the service offers at the 720p floor
PRICE_USD = {"720": 0.0002, "1080": 0.0002, "vtt": 0.0001}
POLL_S = 4.0
JOB_TIMEOUT_S = 900.0
HTTP_TIMEOUT_S = 60.0


class VdaError(RuntimeError):
    pass


def api_key(settings: Any) -> str:
    env = getattr(settings, "env", {}) or {}
    return next((env[k] for k in KEY_NAMES if env.get(k)), "")


class Ledger:
    """This process's paid jobs: spend against the per-run budget, and entries the sourcing stage writes to the
    cost report (registry.record_usage drains them)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.spent = 0.0
        self.jobs: dict[str, int] = {}  # kind ("720", "vtt") -> accepted jobs
        self.pending: list[dict[str, Any]] = []

    def can_spend(self, kind: str, budget: float) -> bool:
        with self.lock:
            return self.spent + PRICE_USD.get(kind, 0.0) <= budget + 1e-9

    def charge(self, kind: str, note: str) -> None:
        with self.lock:
            cost = PRICE_USD.get(kind, 0.0)
            self.spent = round(self.spent + cost, 6)
            self.jobs[kind] = self.jobs.get(kind, 0) + 1
            self.pending.append({"kind": kind, "cost": cost, "note": note})

    def drain(self) -> list[dict[str, Any]]:
        with self.lock:
            out, self.pending = self.pending, []
            return out


LEDGER = Ledger()


def flush_costs(ctx: Any) -> None:
    """Write this process's new paid YouTube jobs into the project's cost report."""
    from ..models import CostEntry
    if getattr(ctx, "tracker", None) is None:
        return  # kept for a context that has a cost report
    for job in LEDGER.drain():
        unit = "videos" if job["kind"] != "vtt" else "subtitles"
        ctx.tracker.add(CostEntry(role="source", provider="video_download_api", model=f"mp4_{job['kind']}"
                                  if job["kind"] != "vtt" else "vtt", stage="sourcing", units=1, unit_kind=unit,
                                  cost_usd=job["cost"], note=job["note"][:200]))


class VdaClient:
    def __init__(self, key: str, session: Optional[requests.Session] = None) -> None:
        if not key:
            raise VdaError("no VIDEO_DOWNLOAD_API_KEY")
        self.key = key
        self.http = session or requests.Session()

    def _scrub(self, text: Any) -> str:
        return str(text).replace(self.key, "***")[:300]

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            r = self.http.get(f"{HOST}{path}", params=params, timeout=HTTP_TIMEOUT_S)
        except requests.RequestException as e:
            raise VdaError(f"{path}: {self._scrub(e)}") from None
        if r.status_code >= 400:
            raise VdaError(f"{path}: http {r.status_code}: {self._scrub(r.text)}")
        try:
            return r.json()
        except ValueError:
            raise VdaError(f"{path}: not JSON: {self._scrub(r.text)}") from None

    def info(self, url: str) -> dict[str, Any]:
        """Title, duration, thumbnail when the service knows them; {} when it does not answer."""
        try:
            got = self._get("/ajax/api.php", {"function": "i", "api": self.key, "u": url})
        except VdaError as e:
            log.info("video download API info failed: %s", e)
            return {}
        return got if isinstance(got, dict) else {}

    def _wait(self, job_id: str, progress_url: str = "") -> str:
        deadline = time.monotonic() + JOB_TIMEOUT_S
        while time.monotonic() < deadline:
            if progress_url:
                try:
                    r = self.http.get(progress_url, timeout=HTTP_TIMEOUT_S)
                    p = r.json() if r.status_code < 400 else {}
                except (requests.RequestException, ValueError):
                    p = {}
            else:
                try:
                    p = self._get("/ajax/progress.php", {"id": job_id})
                except VdaError:
                    p = {}
            prog = int(float(p.get("progress") or 0))
            if prog >= 1000:
                url = str(p.get("download_url") or "").strip()
                if p.get("success") in (1, True, "1") and url:
                    return url
                raise VdaError(f"job failed: {self._scrub(p.get('text') or p.get('message') or 'no download url')}")
            time.sleep(POLL_S)
        raise VdaError("job did not finish in time")

    def _save(self, url: str, dest: Path) -> Path:
        part = dest.with_suffix(dest.suffix + ".part")
        try:
            with self.http.get(url, stream=True, timeout=HTTP_TIMEOUT_S) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
        except requests.RequestException as e:
            part.unlink(missing_ok=True)
            raise VdaError(f"file download: {self._scrub(e)}") from None
        part.replace(dest)
        return dest

    def download(self, url: str, dest: Path, fmt: str = FORMAT) -> tuple[Path, dict[str, Any]]:
        """Whole video as MP4 at fmt into dest; returns (path, the job's info). Charges one job."""
        got = self._get("/ajax/download.php", {"url": url, "format": fmt, "apikey": self.key, "add_info": 1,
                                               "allow_extended_duration": 0})
        if not got.get("success") or not got.get("id"):
            raise VdaError(f"not accepted: {self._scrub(got.get('text') or got.get('message') or got)}")
        LEDGER.charge(fmt, url)
        link = self._wait(str(got["id"]))
        try:
            path = self._save(link, dest)
        except VdaError:  # a link cut off mid file is fetched once more (the job is already paid)
            path = self._save(link, dest)
        return path, dict(got.get("info") or {})

    def subtitles(self, url: str, lang: str = "en") -> str:
        """VTT text, or "" when the video has none. Charges one job when accepted."""
        try:
            got = self._get("/ajax/subtitles", {"url": url, "api": self.key, "lang": lang or "en", "format": "vtt"})
            if not got.get("id"):
                return ""
            LEDGER.charge("vtt", url)
            link = self._wait(str(got["id"]), str(got.get("progress_url") or ""))
            r = self.http.get(link, timeout=HTTP_TIMEOUT_S)
            return r.text if r.status_code < 400 else ""
        except (VdaError, requests.RequestException) as e:
            log.info("video download API subtitles failed: %s", self._scrub(e))
            return ""


def cut_window(src: Path, a: float, b: float, dest: Path) -> Path:
    """Window (a, b) of a local video by stream copy: like a yt-dlp section download, the file starts at the
    keyframe at or before a and ends at b (youtube.window_start reads the lead back from its length)."""
    from ..util import run
    part = dest.with_name(dest.stem + ".part" + dest.suffix)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", str(src),
           "-map", "0:v:0", "-map", "0:a?", "-c", "copy", "-avoid_negative_ts", "make_zero", str(part)]
    run(cmd, timeout=600)
    part.replace(dest)
    return dest
