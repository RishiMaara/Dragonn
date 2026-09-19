"""
Hexagon Bridge — Whisper Transcription, Encoder on the NPU
=========================================================
Real speech-to-text:

  audio -> log-mel features (WhisperFeatureExtractor)
        -> encoder: the quantized ONNX model — QNN EP on the Hexagon NPU when
           this machine has one, CPU EP otherwise
        -> decoder: Whisper's original PyTorch decoder, on CPU
        -> text

Why the split: the encoder is one fixed-shape pass over a 30 s window — the
static graph the HTP compiles well. The autoregressive decoder has growing
shapes and a KV cache; moving it to the NPU is future work, not something to
fake. Every result reports which provider the encoder actually ran on.
"""

import io
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("hexagon-bridge.transcriber")

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 30          # Whisper's fixed input window


def load_audio(data: bytes) -> np.ndarray:
    """Decode WAV/FLAC/OGG/MP3 bytes to mono float32 at 16 kHz."""
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def _model_id_for(encoder_path: Path) -> str:
    config = encoder_path.parent / "config.json"
    if config.exists():
        return json.loads(config.read_text()).get("model_id", "openai/whisper-tiny")
    return "openai/whisper-tiny"


class WhisperTranscriber:
    """
    Args:
        encoder_path: ONNX encoder. None = run the whole model in PyTorch (the
                      reference the ONNX paths are measured against).
        model_id:     HF checkpoint for the processor and decoder (default: from
                      the encoder directory's config.json).
        use_npu:      None = use the NPU if this machine has one.
    """

    def __init__(self, encoder_path=None, model_id=None, use_npu=None):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.encoder_path = Path(encoder_path) if encoder_path else None
        self.model_id = model_id or (_model_id_for(self.encoder_path) if self.encoder_path else "openai/whisper-tiny")
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id).eval()

        self.session = None
        self.provider = "PyTorch (reference)"
        if self.encoder_path:
            import onnxruntime as ort
            from scripts.qnn_ep import create_session, npu_available

            if use_npu is None:
                use_npu = npu_available()
            if use_npu:
                # Cache the compiled HTP graph next to the model: later starts
                # load it instead of recompiling (5.28 s -> 0.53 s on X Elite).
                self.session = create_session(
                    self.encoder_path, {"htp_performance_mode": "burst"},
                    cache_dir=self.encoder_path.parent / ".qnn_cache",
                )
            else:
                self.session = ort.InferenceSession(
                    str(self.encoder_path), providers=["CPUExecutionProvider"]
                )
            self.provider = self.session.get_providers()[0]
            self.input_name = self.session.get_inputs()[0].name

    def features(self, audio: np.ndarray) -> np.ndarray:
        return self.processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="np"
        ).input_features.astype(np.float32)

    def decode_hidden(self, hidden: np.ndarray) -> str:
        """Run the decoder on encoder output (e.g. produced on a remote NPU)."""
        import torch
        from transformers.modeling_outputs import BaseModelOutput

        with torch.no_grad():
            ids = self.model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=torch.from_numpy(hidden)),
                language="en",
                task="transcribe",
            )
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def transcribe(self, audio: np.ndarray) -> dict:
        """Transcribe mono 16 kHz audio of any length, in 30 s windows."""
        import torch

        texts, encoder_ms, decoder_ms = [], 0.0, 0.0
        window = CHUNK_SECONDS * SAMPLE_RATE
        chunks = range(0, max(len(audio), 1), window)

        for start in chunks:
            feats = self.features(audio[start:start + window])
            t0 = time.perf_counter()
            if self.session is not None:
                hidden = self.session.run(None, {self.input_name: feats})[0]
                t1 = time.perf_counter()
                texts.append(self.decode_hidden(hidden))
            else:
                t1 = t0  # reference path: encoder and decoder are one generate call
                with torch.no_grad():
                    ids = self.model.generate(
                        torch.from_numpy(feats), language="en", task="transcribe",
                    )
                texts.append(self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
            t2 = time.perf_counter()
            encoder_ms += (t1 - t0) * 1000
            decoder_ms += (t2 - t1) * 1000

        return {
            "text": " ".join(t for t in texts if t),
            "seconds": round(len(audio) / SAMPLE_RATE, 2),
            "chunks": len(chunks),
            "encoder_provider": self.provider,
            "encoder_ms": round(encoder_ms, 1),
            "decoder_ms": round(decoder_ms, 1),
        }

    def normalize(self, text: str) -> str:
        """Whisper's English normalizer — applied to references and hypotheses alike for WER."""
        return self.processor.tokenizer.normalize(text)


def word_error_rate(references: list[str], hypotheses: list[str]) -> dict:
    """Corpus-level WER: total word edits / total reference words."""
    edits = words = 0
    for ref, hyp in zip(references, hypotheses):
        r, h = ref.split(), hyp.split()
        row = list(range(len(h) + 1))
        for i in range(1, len(r) + 1):
            prev, row[0] = row[0], i
            for j in range(1, len(h) + 1):
                cur = min(row[j] + 1, row[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
                prev, row[j] = row[j], cur
        edits += row[len(h)]
        words += len(r)
    return {"wer_percent": round(edits / words * 100, 2) if words else 0.0, "edits": edits, "ref_words": words}
