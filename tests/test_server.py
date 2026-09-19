"""
Server tests — no model download, no network: the transcriber is stubbed.

The rule under test: the API never returns text it did not transcribe.
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server.app as app_module


class StubTranscriber:
    provider = "CPUExecutionProvider"

    def transcribe(self, audio):
        return {"text": "hello world", "seconds": round(len(audio) / 16_000, 2), "chunks": 1,
                "encoder_provider": self.provider, "encoder_ms": 1.0, "decoder_ms": 2.0}

    def normalize(self, text):
        return text.lower()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module.engine, "load", lambda: None)   # skip real model load
    with TestClient(app_module.app) as c:
        yield c


def _wav_bytes(seconds=1.0):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(16_000 * seconds), np.float32), 16_000, format="WAV")
    return buf.getvalue()


def test_no_model_means_503_not_fake_text(client, monkeypatch):
    monkeypatch.setattr(app_module.engine, "transcriber", None)
    monkeypatch.setattr(app_module.engine, "load_error", "No encoder at models/...")
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", _wav_bytes())})
    assert r.status_code == 503
    assert "No encoder" in r.json()["detail"]


def test_garbage_audio_is_a_400(client, monkeypatch):
    monkeypatch.setattr(app_module.engine, "transcriber", StubTranscriber())
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"dummy audio bytes")})
    assert r.status_code == 400


@pytest.mark.parametrize("fmt,check", [
    ("json", lambda r: r.json() == {"text": "hello world"}),
    ("text", lambda r: r.text == "hello world"),
    ("verbose_json", lambda r: r.json()["duration"] == 1.0
        and r.json()["x_hexagon_bridge"]["encoder_provider"] == "CPUExecutionProvider"),
])
def test_openai_response_formats(client, monkeypatch, fmt, check):
    monkeypatch.setattr(app_module.engine, "transcriber", StubTranscriber())
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", _wav_bytes())},
                    data={"response_format": fmt})
    assert r.status_code == 200 and check(r)


def test_no_hardcoded_transcripts_in_server_source():
    source = open(app_module.__file__, encoding="utf-8").read().lower()
    assert "simulated transcription" not in source
    assert "mock transcription" not in source
