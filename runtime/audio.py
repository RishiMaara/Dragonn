"""
Audio decode and resample — the bottom of the stack.

Kept here, not in the transcriber, because the calibration reader in
converter/ needs the same 16 kHz mono float32 as the app does, and a converter
importing speech code to get it is how import cycles start.
"""

import io

import numpy as np

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 30          # Whisper's fixed input window


def resample(audio: np.ndarray, source_rate: int, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Resample without librosa, which pulls in numba/llvmlite — no reliable
    win-arm64 wheels, so it cannot be a dependency of the Snapdragon app, and
    on this development PC its DLL is blocked by an Application Control policy.
    soxr is librosa's own resampling backend; scipy is the fallback.
    """
    if source_rate == target_rate:
        return audio.astype(np.float32)
    try:
        import soxr
        return soxr.resample(audio, source_rate, target_rate).astype(np.float32)
    except ImportError:
        from math import gcd

        from scipy.signal import resample_poly
        divisor = gcd(int(source_rate), int(target_rate))
        return resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(np.float32)


def load_audio(data: bytes) -> np.ndarray:
    """Decode WAV/FLAC/OGG/MP3 bytes to mono float32 at 16 kHz."""
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample(audio, sr, SAMPLE_RATE)
