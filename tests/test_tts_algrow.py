"""Algrow TTS adapter and the tts seat registry, offline with canned replies shaped like RadioPlay's."""
from __future__ import annotations

from pathlib import Path

import pytest

from vidbush.models import CostEntry, VoiceConfig
from vidbush.providers import seats as seats_mod
from vidbush.providers import tts_algrow
from vidbush.providers.base import CostTracker, ProviderSpec, RefusalError, TransportError

MP3 = b"ID3" + b"\x00" * 2048


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    tts_algrow.reset_models()
    monkeypatch.setattr(tts_algrow.time, "sleep", lambda s: None)
    yield
    tts_algrow.reset_models()


def _spec(**opts) -> ProviderSpec:
    return ProviderSpec("algrow_tts", "algrow_tts", "eleven_v4", api_key="k", options=opts)


def _wire(monkeypatch, submits: list, jobs: list, sent: list) -> None:
    def post(url, fields, headers, timeout):
        sent.append({"url": url, **fields, "auth": headers["Authorization"]})
        r = submits.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(tts_algrow, "_post_form", post)
    monkeypatch.setattr(tts_algrow, "get_json", lambda url, headers, timeout: jobs.pop(0))
    monkeypatch.setattr(tts_algrow, "get_bytes", lambda url, timeout: MP3)


def test_submit_poll_download_returns_no_words_and_records_credits(tmp_path: Path, monkeypatch) -> None:
    sent: list = []
    _wire(monkeypatch, [{"success": True, "job_id": "j1", "credits_used": 2.5}],
          [{"status": "pending"}, {"status": "processing"}, {"status": "completed", "audio_url": "https://cdn/a.mp3"}],
          sent)
    text = "The canal opened in 1914. " * 10
    r = tts_algrow.synthesize(_spec(), text, "voice1", "auto", tmp_path / "c.mp3")
    path, words, model = r.value
    assert path.read_bytes() == MP3 and words is None and model == "eleven_v4"
    assert sent[0]["url"].endswith("/api/generate-simple") and sent[0]["model_id"] == "eleven_v4"
    assert sent[0]["voice_id"] == "voice1" and sent[0]["auth"] == "Bearer k" and sent[0]["script"] == text
    assert r.cost_usd == 0.0 and r.credits == 2.5 and r.units == len(text) and r.unit_kind == "characters"


def test_short_script_is_padded_to_the_floor(tmp_path: Path, monkeypatch) -> None:
    sent: list = []
    _wire(monkeypatch, [{"success": True, "job_id": "j"}], [{"status": "completed", "audio_url": "u"}], sent)
    tts_algrow.synthesize(_spec(), "Short line.", "v", "auto", tmp_path / "c.mp3")
    script = sent[0]["script"]
    assert script.startswith("Short line.") and len(script) >= tts_algrow.MIN_SCRIPT_CHARS
    assert tts_algrow.PAD_TAG in script


def test_refused_model_moves_the_whole_run_to_the_fallback(tmp_path: Path, monkeypatch) -> None:
    sent: list = []
    _wire(monkeypatch, [RefusalError("http 400: model eleven_v4 is not available"),
                        {"success": True, "job_id": "a"}, {"success": True, "job_id": "b"}],
          [{"status": "completed", "audio_url": "u"}, {"status": "completed", "audio_url": "u"}], sent)
    tts_algrow.synthesize(_spec(), "x" * 300, "v", "auto", tmp_path / "1.mp3")
    r2 = tts_algrow.synthesize(_spec(), "y" * 300, "v", "auto", tmp_path / "2.mp3")
    assert [s["model_id"] for s in sent] == ["eleven_v4", "eleven_multilingual_v2", "eleven_multilingual_v2"]
    assert r2.value[2] == "eleven_multilingual_v2"


def test_quota_errors_and_explicit_models_never_switch_model(tmp_path: Path, monkeypatch) -> None:
    sent: list = []
    _wire(monkeypatch, [RefusalError("http 403: model quota exceeded, insufficient credits")], [], sent)
    with pytest.raises(RefusalError):
        tts_algrow.synthesize(_spec(), "x" * 300, "v", "auto", tmp_path / "1.mp3")
    _wire(monkeypatch, [RefusalError("http 400: model eleven_v3 refused")], [], sent)
    with pytest.raises(RefusalError):
        tts_algrow.synthesize(_spec(), "x" * 300, "v", "eleven_v3", tmp_path / "1.mp3")
    assert tts_algrow.resolve_model(_spec(), "auto") == "eleven_v4"


def test_failed_job_timeout_and_envelope_errors(tmp_path: Path, monkeypatch) -> None:
    _wire(monkeypatch, [{"success": True, "job_id": "j"}], [{"status": "failed", "error": "worker crashed"}], [])
    with pytest.raises(TransportError):
        tts_algrow.synthesize(_spec(), "x" * 300, "v", "auto", tmp_path / "1.mp3")
    _wire(monkeypatch, [{"success": False, "error": {"message": "Invalid voice id"}}], [], [])
    with pytest.raises(RefusalError):
        tts_algrow.synthesize(_spec(), "x" * 300, "v", "auto", tmp_path / "1.mp3")
    ticks = iter(range(0, 10000, 400))
    monkeypatch.setattr(tts_algrow, "get_json", lambda *a, **k: {"status": "processing"})
    with pytest.raises(TransportError, match="did not finish"):
        tts_algrow._poll(_spec(timeout_s=600), "j", sleep=lambda s: None, clock=lambda: next(ticks))


def test_missing_voice_is_a_refusal(tmp_path: Path) -> None:
    with pytest.raises(RefusalError):
        tts_algrow.synthesize(_spec(), "x" * 300, "", "auto", tmp_path / "1.mp3")


def test_registry_voice_map_and_unknown_type_skip(settings, tmp_path: Path, monkeypatch) -> None:
    settings.offline = False
    settings.seats = {
        "roles": {"tts": [{"provider": "elevate_tts", "model": "v4"}, {"provider": "algrow_tts", "model": "eleven_v4"}]},
        "providers": {"elevate_tts": {"type": "not_yet_registered", "key_env": "X"},
                      "algrow_tts": {"type": "algrow_tts", "key_env": "ALGROW_API_KEY", "options": {"force_align": True}}},
    }
    settings.env = {"ALGROW_API_KEY": "k"}
    warned: list = []

    class Log:
        def warn(self, msg, **kw): warned.append((msg, kw))
        def info(self, *a, **k): pass

    seats_mod._UNKNOWN_LOGGED.clear()
    s = seats_mod.Seats(settings, CostTracker("p"), Log())
    assert [p.name for p in s.providers_for("tts")] == ["algrow_tts"]
    s.providers_for("tts")
    assert len([w for w in warned if "not registered" in w[0]]) == 1
    assert s.tts_option("algrow_tts", "force_align") is True
    voice = VoiceConfig(voice_id="default_v", voice_ids={"algrow_tts": "algrow_v"})
    spec = s.providers_for("tts")[0]
    assert s.voice_id(voice, spec) == "algrow_v"
    assert s.voice_id(VoiceConfig(voice_id="default_v"), spec) == "default_v"
    got: dict = {}
    monkeypatch.setitem(seats_mod.TTS_TYPES, "algrow_tts",
                        lambda spec, text, vid, model, out, *a, **k: got.update(vid=vid) or tts_algrow.CallResult(
                            value=(out, None, "eleven_v4"), provider=spec, mock=False, credits=1.0,
                            units=len(text), unit_kind="characters"))
    path, words, is_mock, engine = s.tts("hello", voice, tmp_path / "a.wav", "narration")
    assert got["vid"] == "algrow_v" and words is None and not is_mock and engine == "algrow_tts"
    entry = [e for e in s.tracker.report.entries if e.ok][-1]
    assert isinstance(entry, CostEntry) and entry.credits == 1.0 and entry.cost_usd == 0.0
