"""
Hexagon Bridge — Real Speech for Calibration and Evaluation
===========================================================
Downloads LibriSpeech's small test set (hf-internal-testing/librispeech_asr_dummy:
73 clips of read English with reference transcripts) and writes it as WAV files
in two DISJOINT splits:

    data/speech/calib/   used by the quantizer to set activation ranges
    data/speech/eval/    held out; used only to measure word error rate

Keeping them disjoint matters: calibrating on the clips you evaluate on would
flatter the accuracy number.

Usage:
    python -m scripts.fetch_speech
    python -m converter.quantize ... --audio-dir data/speech/calib
    python -m scripts.eval_wer --audio-dir data/speech/eval
"""

import argparse
import io
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("hexagon-bridge.speech")

DATASET = "hf-internal-testing/librispeech_asr_dummy"
SAMPLE_RATE = 16_000


def fetch(out_dir: Path, n_calib: int) -> dict:
    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset(DATASET, "clean", split="validation")
    # Decode ourselves with soundfile: newer `datasets` releases need extra
    # codec packages for their built-in audio decoding.
    ds = ds.cast_column("audio", Audio(decode=False))

    counts = {"calib": 0, "eval": 0}
    manifests = {"calib": [], "eval": []}
    for i, row in enumerate(ds):
        split = "calib" if i < n_calib else "eval"
        audio_bytes = row["audio"]["bytes"]
        if audio_bytes is None:
            audio_bytes = Path(row["audio"]["path"]).read_bytes()
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        name = f"{row['id']}.wav"
        sf.write(str(split_dir / name), audio.astype(np.float32), SAMPLE_RATE)
        manifests[split].append({"file": name, "text": row["text"], "seconds": round(len(audio) / SAMPLE_RATE, 2)})
        counts[split] += 1

    for split, rows in manifests.items():
        with open(out_dir / split / "transcripts.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    return counts


def main():
    parser = argparse.ArgumentParser(description="Fetch real speech for calibration and WER evaluation")
    parser.add_argument("--out", default="./data/speech")
    parser.add_argument("--calib", type=int, default=16, help="Clips reserved for calibration (rest are eval)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    counts = fetch(Path(args.out), args.calib)
    print(f"Wrote {counts['calib']} calibration clips and {counts['eval']} held-out eval clips to {args.out}")


if __name__ == "__main__":
    main()
