"""
Hexagon Bridge — Inference Server
==================================
OpenAI-compatible transcription API. Whisper's encoder runs as the quantized
ONNX model — on the Hexagon NPU (QNN EP) when this machine has one, CPU EP
otherwise — and the decoder runs on CPU. See scripts/transcriber.py.

Every response and telemetry event reports the provider the encoder ACTUALLY
ran on, read back from the session — never the one that was requested.

Endpoints:
  POST /v1/audio/transcriptions   OpenAI-compatible; WAV/FLAC/OGG/MP3 upload
  GET  /api/samples               held-out LibriSpeech clips (python -m scripts.fetch_speech)
  POST /api/transcribe-sample     transcribe one clip; returns reference + WER
  GET  /api/status                model/provider/coverage status
  WS   /ws/telemetry              live events for the dashboard

Run from the project root:  python -m server.app
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("hexagon-bridge.server")

ENCODER_PATH = Path("models/whisper-tiny-qdq/encoder_model.onnx")
SAMPLES_DIR = Path("data/speech/eval")
COVERAGE_REPORT = Path("models/reports/coverage_report.json")

app = FastAPI(title="Hexagon Bridge API", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# --- Telemetry ---------------------------------------------------------------

class TelemetryState:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.total_requests = 0
        self.last_latency_ms = 0.0
        self.current_ep = "not loaded"
        self.coverage_data = self._load_coverage()

    @staticmethod
    def _load_coverage() -> Dict[str, Any]:
        if COVERAGE_REPORT.exists():
            try:
                return json.loads(COVERAGE_REPORT.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Failed to load coverage report: {e}")
        return {"error": "Coverage report not found — run python -m scripts.run_pipeline"}

    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        message = json.dumps({"type": event_type, "data": data})
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                self.active_connections.remove(connection)


telemetry = TelemetryState()


# --- Model -------------------------------------------------------------------

class InferenceEngine:
    def __init__(self, encoder_path: Path = ENCODER_PATH):
        self.encoder_path = encoder_path
        self.transcriber = None
        self.load_error: Optional[str] = None
        self.lock = asyncio.Lock()   # one inference at a time: the NPU is a single device

    def load(self):
        if not self.encoder_path.exists():
            self.load_error = (
                f"No encoder at {self.encoder_path}. Build it: python -m scripts.run_pipeline "
                "--model openai/whisper-tiny --quantized-dir ./models/whisper-tiny-qdq"
            )
            logger.error(self.load_error)
            return
        try:
            from scripts.transcriber import WhisperTranscriber
            self.transcriber = WhisperTranscriber(self.encoder_path)
            telemetry.current_ep = self.transcriber.provider.replace("ExecutionProvider", "")
            logger.info(f"Encoder loaded on {self.transcriber.provider}")
        except Exception as e:
            self.load_error = f"Model failed to load: {e}"
            logger.exception(self.load_error)

    def require(self):
        if self.transcriber is None:
            raise HTTPException(status_code=503, detail=self.load_error or "Model not loaded")

    async def transcribe(self, audio) -> dict:
        self.require()
        async with self.lock:
            await telemetry.broadcast("inference_start", {"timestamp": time.time(),
                                                          "provider": telemetry.current_ep})
            started = time.perf_counter()
            result = await asyncio.to_thread(self.transcriber.transcribe, audio)
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)

        telemetry.total_requests += 1
        telemetry.last_latency_ms = result["latency_ms"]
        await telemetry.broadcast("inference_end", {
            "timestamp": time.time(),
            "provider": result["encoder_provider"].replace("ExecutionProvider", ""),
            "latency_ms": result["latency_ms"],
            "encoder_ms": result["encoder_ms"],
            "decoder_ms": result["decoder_ms"],
            "audio_seconds": result["seconds"],
            "transcript": result["text"],
        })
        return result


engine = InferenceEngine()


@app.on_event("startup")
async def startup_event():
    await asyncio.to_thread(engine.load)
    await telemetry.broadcast("status", {"provider": telemetry.current_ep,
                                         "is_ready": engine.transcriber is not None,
                                         "coverage": telemetry.coverage_data})


# --- Endpoints ---------------------------------------------------------------

@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
):
    """OpenAI-compatible transcription. English only (the model is run with language='en')."""
    from scripts.transcriber import load_audio

    engine.require()
    data = await file.read()
    try:
        audio = load_audio(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode audio ({e}). Send WAV, FLAC, OGG or MP3.")

    result = await engine.transcribe(audio)
    if response_format == "text":
        return PlainTextResponse(result["text"])
    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": "english",
            "duration": result["seconds"],
            "text": result["text"],
            "segments": [{"id": 0, "start": 0.0, "end": result["seconds"], "text": result["text"]}],
            "x_hexagon_bridge": {k: result[k] for k in ("encoder_provider", "encoder_ms", "decoder_ms", "latency_ms")},
        }
    return {"text": result["text"]}


def _sample_manifest() -> list[dict]:
    manifest = SAMPLES_DIR / "transcripts.jsonl"
    if not manifest.exists():
        return []
    return [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]


@app.get("/api/samples")
async def list_samples():
    rows = _sample_manifest()
    if not rows:
        raise HTTPException(status_code=404, detail="No samples. Run: python -m scripts.fetch_speech")
    return [{"file": r["file"], "seconds": r["seconds"]} for r in rows]


@app.post("/api/transcribe-sample")
async def transcribe_sample(name: Optional[str] = None):
    """Transcribe a held-out LibriSpeech clip and score it against its reference."""
    from scripts.transcriber import load_audio, word_error_rate

    rows = _sample_manifest()
    if not rows:
        raise HTTPException(status_code=404, detail="No samples. Run: python -m scripts.fetch_speech")
    if name is None:
        row = rows[telemetry.total_requests % len(rows)]
    else:
        row = next((r for r in rows if r["file"] == name), None)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Unknown sample {name}")

    result = await engine.transcribe(load_audio((SAMPLES_DIR / row["file"]).read_bytes()))
    t = engine.transcriber
    wer = word_error_rate([t.normalize(row["text"])], [t.normalize(result["text"])])
    return {**result, "file": row["file"], "reference": row["text"], **wer}


@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "model_ready": engine.transcriber is not None,
        "load_error": engine.load_error,
        "encoder_provider": telemetry.current_ep,
        "total_requests": telemetry.total_requests,
        "last_latency_ms": telemetry.last_latency_ms,
        "coverage_report": telemetry.coverage_data,
    }


@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    telemetry.active_connections.append(websocket)
    try:
        await websocket.send_json({"type": "init", "data": {
            "provider": telemetry.current_ep,
            "is_ready": engine.transcriber is not None,
            "load_error": engine.load_error,
            "coverage": telemetry.coverage_data,
        }})
        while True:
            data = await websocket.receive_text()
            try:
                if json.loads(data).get("action") == "ping":
                    await websocket.send_json({"type": "pong"})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        if websocket in telemetry.active_connections:
            telemetry.active_connections.remove(websocket)


# Static dashboard last, so it acts as the fallback route.
app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="127.0.0.1", port=8000)
