"""
Hexagon Bridge — Word Error Rate on Real Speech
===============================================
End-to-end accuracy: transcribe held-out LibriSpeech clips and score them
against reference transcripts, for each encoder configuration:

  PyTorch reference   whisper-tiny exactly as released — the bar to match
  ONNX FP32           checks the export is faithful
  ONNX a16w8 (CPU)    what quantization costs
  ONNX a16w8 on NPU   --aihub: the encoder runs on a real Snapdragon X Elite via
                      Qualcomm AI Hub; its outputs are decoded here. This is the
                      accuracy a user actually gets.

Cosine similarity on encoder outputs says the numbers are close; WER says
whether the words are still right.

Usage:
    python -m scripts.fetch_speech                 # once
    python -m scripts.eval_wer                     # local configurations
    python -m scripts.eval_wer --aihub             # + encoder on a real X Elite
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from scripts.transcriber import CHUNK_SECONDS, SAMPLE_RATE, WhisperTranscriber, load_audio, word_error_rate

logger = logging.getLogger("hexagon-bridge.wer")


def load_rows(audio_dir: Path, limit: int | None) -> list[dict]:
    manifest = audio_dir / "transcripts.jsonl"
    if not manifest.exists():
        logger.error(f"No {manifest}. Run: python -m scripts.fetch_speech")
        sys.exit(1)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def score(label: str, refs: list[str], hyps: list[str], extra: dict) -> dict:
    result = {"config": label, **word_error_rate(refs, hyps), **extra}
    result["examples"] = [
        {"ref": r, "hyp": h} for r, h in zip(refs, hyps) if r != h
    ][:5]
    return result


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def fp32_hidden_states(fp32: WhisperTranscriber, rows, audio_dir) -> dict:
    """FP32 encoder output per clip — the reference each quantized encoder's cosine is taken against."""
    out = {}
    for row in rows:
        feats = fp32.features(load_audio((audio_dir / row["file"]).read_bytes()))
        out[row["file"]] = fp32.session.run(None, {fp32.input_name: feats})[0]
    return out


def evaluate_local(label: str, transcriber: WhisperTranscriber, rows, audio_dir,
                   fp32_hidden: dict | None = None) -> dict:
    refs, hyps, enc_ms, cosines = [], [], [], []
    for row in rows:
        audio = load_audio((audio_dir / row["file"]).read_bytes())
        out = transcriber.transcribe(audio)
        refs.append(transcriber.normalize(row["text"]))
        hyps.append(transcriber.normalize(out["text"]))
        enc_ms.append(out["encoder_ms"])
        # Cosine next to WER, per config: the two can disagree, and WER is the one users feel.
        if fp32_hidden is not None and transcriber.session is not None:
            hidden = transcriber.session.run(None, {transcriber.input_name: transcriber.features(audio)})[0]
            cosines.append(_cosine(fp32_hidden[row["file"]], hidden))
    logger.info(f"{label}: done")
    return score(label, refs, hyps, {
        "encoder_provider": transcriber.provider,
        "encoder_cosine_vs_fp32": round(float(np.mean(cosines)), 5) if cosines else None,
        "mean_encoder_ms_here": round(float(np.mean(enc_ms)), 1) if transcriber.session else None,
    })


def evaluate_on_aihub(encoder_path: Path, device_name: str, reference: WhisperTranscriber,
                      rows, audio_dir) -> dict:
    """Encoder on a real device's NPU; features and decoding stay local."""
    from scripts.aihub_validate import (
        QNN_OPTIONS, _upload, _wait, _with_network_retry, check_auth, import_hub, resolve_device,
    )

    hub = import_hub()
    if not check_auth(hub):
        sys.exit(1)
    device = resolve_device(hub, device_name)

    # Every 30 s window becomes one inference sample; remember which clip it came from.
    window = CHUNK_SECONDS * SAMPLE_RATE
    features, owner = [], []
    for i, row in enumerate(rows):
        audio = load_audio((audio_dir / row["file"]).read_bytes())
        for start in range(0, max(len(audio), 1), window):
            features.append(reference.features(audio[start:start + window]))
            owner.append(i)

    logger.info(f"Uploading encoder + {len(features)} feature windows to AI Hub...")
    model = _upload(hub, encoder_path, "hexbridge-wer-encoder")
    job = hub.submit_inference_job(
        model=model, device=device, name="hexbridge-wer-npu",
        inputs={"input_features": features}, options=QNN_OPTIONS,
    )
    _wait(job, "Inference job")
    outputs = _with_network_retry(job.download_output_data, job, "Inference job")
    hidden = next(iter(outputs.values()))

    pieces = [[] for _ in rows]
    for idx, h in zip(owner, hidden):
        pieces[idx].append(reference.decode_hidden(np.asarray(h, dtype=np.float32)))
    hyps = [reference.normalize(" ".join(p)) for p in pieces]
    refs = [reference.normalize(r["text"]) for r in rows]
    return score(f"ONNX a16w8 on {device.name} NPU", refs, hyps, {
        "encoder_provider": "QNNExecutionProvider (on device)",
        "aihub_job": job.url,
    })


def main():
    parser = argparse.ArgumentParser(description="Word error rate on held-out real speech")
    parser.add_argument("--audio-dir", default="./data/speech/eval")
    parser.add_argument("--fp32", default="./models/whisper-tiny-onnx/encoder_model.onnx")
    parser.add_argument("--qdq", default="./models/whisper-tiny-qdq/encoder_model.onnx")
    parser.add_argument("--variant", action="append", default=[], metavar="LABEL=PATH",
                        help="Extra quantized encoder to evaluate on CPU (repeatable)")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N clips")
    parser.add_argument("--aihub", action="store_true", help="Also run the QDQ encoder on a real device NPU")
    parser.add_argument("--device", default="Snapdragon X Elite CRD")
    parser.add_argument("--json", default="./models/reports/wer_report.json")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    audio_dir = Path(args.audio_dir)
    rows = load_rows(audio_dir, args.limit)
    logger.info(f"Evaluating {len(rows)} held-out clips "
                f"({sum(r['seconds'] for r in rows):.0f} s of speech)")

    started = time.time()
    reference = WhisperTranscriber(None)
    fp32 = WhisperTranscriber(args.fp32, use_npu=False)
    fp32_hidden = fp32_hidden_states(fp32, rows, audio_dir)
    results = [
        evaluate_local("PyTorch reference (FP32)", reference, rows, audio_dir),
        evaluate_local("ONNX FP32 encoder (CPU)", fp32, rows, audio_dir, fp32_hidden),
        evaluate_local("ONNX a16w8 encoder (CPU)", WhisperTranscriber(args.qdq, use_npu=False),
                       rows, audio_dir, fp32_hidden),
    ]
    for variant in args.variant:
        label, _, path = variant.partition("=")
        results.append(evaluate_local(label, WhisperTranscriber(path, use_npu=False),
                                      rows, audio_dir, fp32_hidden))
    if args.aihub:
        results.append(evaluate_on_aihub(Path(args.qdq), args.device, reference, rows, audio_dir))

    base = results[0]["wer_percent"]
    for r in results:
        r["delta_vs_reference"] = round(r["wer_percent"] - base, 2)

    report = {"clips": len(rows), "speech_seconds": round(sum(r["seconds"] for r in rows), 1),
              "dataset": "hf-internal-testing/librispeech_asr_dummy (held-out split)",
              "results": results, "wall_time_s": round(time.time() - started, 1)}
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    bar = "=" * 78
    print(f"\n{bar}\n  WORD ERROR RATE — {len(rows)} held-out LibriSpeech clips, "
          f"{report['speech_seconds']:.0f} s of speech\n{bar}")
    print(f"  {'encoder':44} {'WER':>7} {'Δ vs ref':>9} {'errors':>11} {'cosine':>8}")
    print(f"  {'-' * 83}")
    for r in results:
        cos = f"{r['encoder_cosine_vs_fp32']:.4f}" if r.get("encoder_cosine_vs_fp32") else "—"
        print(f"  {r['config']:44} {r['wer_percent']:>6.2f}% {r['delta_vs_reference']:>+8.2f} "
              f"{r['edits']:>5}/{r['ref_words']:<5} {cos:>8}")
        if r.get("aihub_job"):
            print(f"  {'':44} {r['aihub_job']}")
    print(f"\n  Report: {out}\n{bar}")


if __name__ == "__main__":
    main()
