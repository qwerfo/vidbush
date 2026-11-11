"""Elevate TTS adapter (ported from RadioPlay), its registry line, probe, and the narration alignment route.
All HTTP is mocked: no live network call."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from vidbush.audio import narration as nar
from vidbush.audio import transcribe
from vidbush.models import Beat, Scene, Script, VoiceConfig, Word
from vidbush.providers import mock, probe as probe_mod, seats as seats_mod, tts_elevate
from vidbush.providers.base import (CostTracker, ProviderSpec, RateLimitError, RefusalError, SeatUnavailable,
                                    TransportError)

MP3 = b"ID3" + b"\x00" * 2048
BASE = "https://public-api.elevate.uno"


class Resp:
    def __init__(self, body: Any = None, status: int = 200, raw: bytes = b"", headers: Optional[dict] = None) -> None:
        self.status_code = status
        self.headers = headers or {}
        self._body = body
        self.content = raw or json.dumps(body).encode()
        self.text = self.content.decode("utf-8", "replace")

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no json")
        return self._body


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    tts_elevate.reset()
    monkeypatch.setattr(tts_elevate.time, "sleep", lambda s: None)
    yield
    tts_elevate.reset()


def _spec(**opts) -> ProviderSpec:
    return ProviderSpec("elevate_tts", "elevate_tts", "eleven_v4", api_key="k", options=opts)


def _wire(monkeypatch, replies: list, then: Any = None) -> list[dict]:
    """Serve replies in order, then `then` for ever; each request is recorded."""
    sent: list[dict] = []

    def request(method, url, json=None, headers=None, timeout=None):
        sent.append({"method": method, "url": url, "json": json, "headers": headers or {}, "timeout": timeout})
        r = replies.pop(0) if replies or then is None else then
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(tts_elevate.requests, "request", request)
    return sent


def _ok(data: dict) -> Resp:
    return Resp({"success": True, "data": data})


def test_submit_poll_download_returns_no_words_and_the_model(tmp_path: Path, monkeypatch) -> None:
    sent = _wire(monkeypatch, [_ok({"id": "t1", "status": "pending"}), _ok({"id": "t1", "status": "processing"}),
                               _ok({"id": "t1", "status": "completed", "result_url": "https://cdn/a.mp3"}),
                               Resp(raw=MP3)])
    text = "The canal opened in 1914. " * 4
    r = tts_elevate.synthesize(_spec(), text, "george", "auto", tmp_path / "c.mp3", {"stability": 0.4},
                               previous_text="before", next_text="after")
    path, words, model = r.value
    assert path.read_bytes() == MP3 and words is None and model == "eleven_v4"
    post = sent[0]
    assert (post["method"], post["url"]) == ("POST", f"{BASE}/v2/media")
    assert post["json"] == {"type": "tts", "prompt": text, "voice_id": "george", "model_id": "eleven_v4"}
    assert post["headers"]["Authorization"] == "Bearer k" and "User-Agent" in post["headers"]
    assert post["timeout"] == pytest.approx(400 + len(text) * 0.002)
    assert [s["url"] for s in sent[1:3]] == [f"{BASE}/v2/media/t1?type=tts"] * 2
    assert sent[3]["url"] == "https://cdn/a.mp3" and "Authorization" not in sent[3]["headers"]
    assert r.cost_usd == 0.0 and r.units == len(text) and r.unit_kind == "characters"


def test_model_and_timeout_rules() -> None:
    assert tts_elevate.resolve_model(_spec(), "eleven_v3") == "eleven_v3"
    assert tts_elevate.resolve_model(ProviderSpec("e", "elevate_tts", ""), "auto") == "eleven_v4"
    assert tts_elevate.request_timeout(_spec(), 100_000) == 600.0  # capped
    assert tts_elevate.request_timeout(_spec(timeout_base_s=10, timeout_per_char_s=0.1), 50) == 15.0


@pytest.mark.parametrize("code,message,exc", [
    ("INVALID_REQUEST", "bad voice", RefusalError), ("UNAUTHORIZED", "bad key", SeatUnavailable),
    ("CONTENT_POLICY", "blocked by safety filters", RefusalError), ("RATE_LIMIT_EXCEEDED", "slow down", RateLimitError),
    ("SERVICE_ERROR", "please retry", TransportError)])
def test_envelope_errors_are_classified(code: str, message: str, exc: type) -> None:
    with pytest.raises(exc) as e:
        tts_elevate.unwrap({"success": False, "error": {"code": code, "message": message}}, "create tts")
    if exc is RefusalError:
        assert not isinstance(e.value, SeatUnavailable)


def test_failed_task_naming_the_voice_is_a_refusal_else_transient(tmp_path: Path, monkeypatch) -> None:
    _wire(monkeypatch, [_ok({"id": "t"}), _ok({"status": "failed", "error": "Voice not available"})])
    with pytest.raises(RefusalError):
        tts_elevate.synthesize(_spec(), "x" * 50, "v", "auto", tmp_path / "a.mp3")
    _wire(monkeypatch, [_ok({"id": "t"}), _ok({"status": "failed", "error": "worker crashed"})])
    with pytest.raises(TransportError):
        tts_elevate.synthesize(_spec(), "x" * 50, "v", "auto", tmp_path / "a.mp3")
    with pytest.raises(RefusalError):
        tts_elevate.synthesize(_spec(), "x", "", "auto", tmp_path / "a.mp3")


def test_capacity_429_is_waited_out_and_plain_429_retries_then_raises(monkeypatch) -> None:
    cap = Resp({"success": False, "error": {"code": "RATE_LIMIT_EXCEEDED", "message": "TTS concurrent limit reached (3/3)",
                                            "details": {"used": 3, "limit": 3}}}, status=429)
    waits: list[float] = []
    _wire(monkeypatch, [cap] * 6 + [_ok({"id": "t"})])
    r = tts_elevate._call(_spec(), "POST", f"{BASE}/v2/media", 600, sleep=waits.append, clock=lambda: 0.0)
    assert r.json()["data"]["id"] == "t" and waits == [18.0] * 6  # more than rate_limit_retries, never counted
    plain = Resp({"success": False, "error": {"code": "RATE_LIMIT_EXCEEDED", "message": "slow down"}}, status=429,
                 headers={"Retry-After": "7"})
    waits.clear()
    _wire(monkeypatch, [plain] * 5)
    with pytest.raises(RateLimitError):
        tts_elevate._call(_spec(), "POST", f"{BASE}/v2/media", 600, sleep=waits.append, clock=lambda: 0.0)
    assert waits == [7.0] * 4
    ticks = iter([0.0, 0.0, 700.0])
    _wire(monkeypatch, [cap] * 3)
    with pytest.raises(TransportError, match="still at capacity"):
        tts_elevate._call(_spec(), "POST", f"{BASE}/v2/media", 600, sleep=lambda s: None, clock=lambda: next(ticks))


def test_http_errors_map_like_every_provider(monkeypatch) -> None:
    _wire(monkeypatch, [Resp({"error": "nope"}, status=401)])
    with pytest.raises(SeatUnavailable):
        tts_elevate._call(_spec(), "POST", f"{BASE}/v2/media", 60)
    _wire(monkeypatch, [Resp({"error": "down"}, status=503)])
    with pytest.raises(TransportError):
        tts_elevate._call(_spec(), "POST", f"{BASE}/v2/media", 60)


def test_task_running_at_the_deadline_is_resumed_not_paid_twice(tmp_path: Path, monkeypatch) -> None:
    spec = _spec(timeout_s=0.01)
    sent = _wire(monkeypatch, [_ok({"id": "t9"})], then=_ok({"status": "processing"}))
    with pytest.raises(TransportError, match="did not finish"):
        tts_elevate.synthesize(spec, "y" * 40, "v", "auto", tmp_path / "a.mp3")
    assert [s["method"] for s in sent].count("POST") == 1
    sent = _wire(monkeypatch, [_ok({"status": "completed", "result_url": "https://cdn/b.mp3"}), Resp(raw=MP3)])
    r = tts_elevate.synthesize(spec, "y" * 40, "v", "auto", tmp_path / "a.mp3")
    assert [s["method"] for s in sent] == ["GET", "GET"] and sent[0]["url"].endswith("/v2/media/t9?type=tts")
    assert r.value[0].read_bytes() == MP3


def test_probe_checks_the_key_then_needs_the_tts_lane_operational(monkeypatch) -> None:
    usage = {"success": True, "data": {"used": 1}}
    replies = [usage, {"success": True, "data": {"tts": {"status": "operational"}}},
               usage, {"success": True, "data": {"tts": {"status": "degraded", "success_rate": 60.0}}},
               {"success": False, "error": {"code": "UNAUTHORIZED", "message": "invalid api key"}}]
    seen: list[tuple[str, Any]] = []
    monkeypatch.setattr(tts_elevate, "get_json",
                        lambda url, headers=None, timeout=0: seen.append((url, headers)) or replies.pop(0))
    assert "auth ok; tts lane operational" in probe_mod.probe_spec("tts", _spec())
    with pytest.raises(TransportError, match="degraded"):
        tts_elevate.probe(_spec())
    with pytest.raises(SeatUnavailable):  # a bad key fails the probe even while the lane is healthy
        tts_elevate.probe(_spec())
    assert [u for u, _ in seen] == [f"{BASE}/v2/media/usage", f"{BASE}/v2/media/health"] * 2 + [
        f"{BASE}/v2/media/usage"]
    assert all(h["Authorization"].startswith("Bearer ") for _, h in seen)


def test_registered_and_seats_log_the_model_once(settings, tmp_path: Path, monkeypatch) -> None:
    assert seats_mod.TTS_TYPES["elevate_tts"] is tts_elevate
    settings.offline = False
    settings.seats = {"roles": {"tts": [{"provider": "elevate_tts", "model": "eleven_v4"}]},
                      "providers": {"elevate_tts": {"type": "elevate_tts", "key_env": "ELEVATE_API_KEY"}}}
    settings.env = {"ELEVATE_API_KEY": "k"}
    infos: list = []

    class Log:
        def warn(self, msg, **kw): pass
        def info(self, msg, **kw): infos.append((msg, kw))

    got: list = []
    monkeypatch.setattr(tts_elevate, "synthesize",
                        lambda spec, text, vid, model, out, *a, **k: got.append((vid, k)) or tts_elevate.CallResult(
                            value=(out, None, "eleven_v4"), provider=spec, mock=False, units=len(text),
                            unit_kind="characters"))
    s = seats_mod.Seats(settings, CostTracker("p"), Log())
    voice = VoiceConfig(voice_id="brian", voice_ids={"elevate_tts": "george"})
    for i in range(2):
        path, words, is_mock, engine = s.tts("hello", voice, tmp_path / f"{i}.wav", "narration",
                                             previous_text="a", next_text="b")
    assert words is None and not is_mock and engine == "elevate_tts"
    assert got[0][0] == "george" and got[0][1]["previous_text"] == "a" and got[0][1]["next_text"] == "b"
    assert [kw for msg, kw in infos if msg == "tts model"] == [{"provider": "elevate_tts", "model": "eleven_v4"}]


# ---------------------------------------------------------------- narration route

def _script() -> Script:
    return Script(title="t", scenes=[
        Scene(id="s1", title="Hook", purpose="hook", beats=[Beat(id="b1", text="The Panama Canal opened in 1914.")]),
        Scene(id="s2", title="Setup", purpose="setup", beats=[Beat(id="b2", text="Ships no longer sailed around Cape Horn.")]),
        Scene(id="s3", title="Turn", purpose="setup", beats=[Beat(id="b3", text="But the story begins with a French dream.")]),
    ])


class ElevateLikeSeats:
    """A real tts seat that returns audio only, as elevate_tts does; records the kwargs of every call."""

    def __init__(self) -> None:
        self.calls: dict[str, dict] = {}

    def is_real(self, role: str) -> bool:
        return role == "tts"

    def tts(self, text: str, voice: VoiceConfig, out_path: Path, stage: str, **kw: Any):
        self.calls[text] = kw
        return Path(mock.mock_tts(text, out_path).value[0]), None, False, "elevate_tts"

    def tts_option(self, provider: str, key: str, default: Any = None) -> Any:
        return default


def test_elevate_audio_without_timings_is_aligned_and_gets_neighbour_text(make_ctx: Any, monkeypatch) -> None:
    from vidbush.audio import align

    def fake_transcribe(audio: Path, settings: Any, log: Any) -> list[Word]:
        regions, dur = align.speech_regions(audio)
        return [Word(text=w.text.lower().strip(".,"), start=w.start, end=w.end)
                for w in align.silence_align(align.script_tokens(_script()), regions, 0.0, dur)]
    monkeypatch.setattr(transcribe, "transcribe_words", fake_transcribe)
    ctx = make_ctx(mode="script", project_id="elev1")
    seats = ElevateLikeSeats()
    n = nar.narrate(_script(), ctx.brand, seats, ctx.settings, ctx.work, ctx.log)
    assert n.aligned_by == "whisper" and n.provider == "elevate_tts"
    chunks = nar.plan_chunks(_script(), per_beat=False)
    assert len(chunks) == 3
    for i, c in enumerate(chunks):
        kw = seats.calls[c.text]
        assert kw["previous_text"] == (chunks[i - 1].text if i else "")
        assert kw["next_text"] == (chunks[i + 1].text if i + 1 < len(chunks) else "")
